# Precompute Performance Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Context:** `precompute_all_games()` is the bottleneck in the overnight pipeline. Each of ~1000+ games runs `_precompute_one()` in a 14-thread pool. Currently, every game creates a fresh in-memory SQLite copy via `sqlite3.backup()`, and dozens of per-game queries fetch data that could be pre-built once. This plan eliminates the per-game backup overhead and replaces repeated DB queries with dict lookups.

**Architecture:** Three independent optimizations, ordered by impact. Each is independently committable.

**Tech Stack:** Python, SQLite, pandas

---

## Task 1: Move thread_local_db from per-game to per-thread

**Why:** `_precompute_one()` wraps every game in `with thread_local_db():`, which calls `sqlite3.backup()` to copy the entire in-memory DB. With 14 threads and 1000 games, that's 1000 backups. Moving to per-thread means 14 backups total.

**Files:**
- Modify: `src/database/db.py` (add `ensure_thread_local_db()`)
- Modify: `src/analytics/prediction.py` (lines 1231, 1283, 1518)

### Step 1: Add ensure_thread_local_db() to db.py

Add after the `thread_local_db` class (after line 294):

```python
def ensure_thread_local_db():
    """Set up a thread-local read-only DB for the current thread (idempotent).

    Unlike the ``thread_local_db`` context manager this does NOT auto-close.
    Call ``close_thread_local_db()`` when done, or let GC handle it.
    Intended for ``ThreadPoolExecutor(initializer=...)``.
    """
    if getattr(_thread_local_db, "conn", None) is not None:
        return
    mem = _get_mem_conn()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _rwlock.read_acquire()
    try:
        mem.backup(conn)
    finally:
        _rwlock.read_release()
    _thread_local_db.conn = conn


def close_thread_local_db():
    """Close the thread-local DB if one exists (idempotent)."""
    conn = getattr(_thread_local_db, "conn", None)
    _thread_local_db.conn = None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
```

### Step 2: Update precompute_all_games() in prediction.py

```python
# BEFORE (line 1231):
    from src.database.db import thread_local_db

# AFTER:
    from src.database.db import ensure_thread_local_db

# BEFORE (line 1283):
    def _precompute_one(g):
        """Process one game with a thread-local DB."""
        with thread_local_db():
            htid = g["home_team_id"]
            ...

# AFTER:
    def _precompute_one(g):
        """Process one game (thread-local DB set up by pool initializer)."""
        htid = g["home_team_id"]
        ...
        # (remove the `with thread_local_db():` and dedent body)

# BEFORE (line 1518):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

# AFTER:
    with ThreadPoolExecutor(max_workers=max_workers, initializer=ensure_thread_local_db) as executor:
```

### Step 3: Verify

```bash
python -c "from src.database.db import ensure_thread_local_db, close_thread_local_db; print('OK')"
python -c "from src.analytics.prediction import precompute_all_games; print('OK')"
```

### Step 4: Commit

```bash
git commit -m "perf: move thread_local_db from per-game to per-thread in precompute"
```

---

## Task 2: Pre-load memory store before precompute

**Why:** `InMemoryDataStore` provides pandas DataFrames for `player_splits()` and `_get_team_metrics()`. When loaded, these functions skip their DB queries and filter in-memory DataFrames instead. `player_splits()` is called 12-15× per game (once per roster player) — that's 12,000-15,000 DB queries eliminated.

**Files:**
- Modify: `src/analytics/prediction.py` (add 4 lines before thread pool)

### Step 1: Add store.load() call

In `precompute_all_games()`, after `ctx = _build_precompute_context(all_games)` (line 1274) and before the ThreadPoolExecutor:

```python
    # Pre-load memory store so player_splits() / _get_team_metrics() use
    # pandas DataFrames instead of per-player DB queries.
    from src.analytics.memory_store import InMemoryDataStore
    _store = InMemoryDataStore()
    if not _store.is_loaded:
        if callback:
            callback("Loading memory store for fast lookups...")
        _store.load()
```

### Step 2: Verify

```bash
python -c "from src.analytics.prediction import precompute_all_games; print('OK')"
```

### Step 3: Commit

```bash
git commit -m "perf: pre-load memory store before precompute for DataFrame-backed lookups"
```

---

## Task 3: Pre-build invariant lookup caches (Elo, on/off impact, odds)

**Why:** Three queries in `_precompute_one()` fetch data that is identical across many games:
- `get_team_elo()`: 2 queries/game × 1000 games = 2000 queries, **zero caching** currently
- On/off impact: 2 queries/game, same (team_id, season) across all games in a season
- Odds: 1 query/game, but easily batch-fetched

Replace all three with dict lookups built from one-time bulk queries.

**Files:**
- Modify: `src/analytics/prediction.py` (build caches + use in `_precompute_one`)

### Step 1: Build Elo lookup table

In `precompute_all_games()`, after memory store loading, before `_precompute_one` definition:

```python
    # ── Pre-build Elo lookup: team_id -> sorted [(game_date, elo), ...] ──
    import bisect
    _elo_rows = db.fetch_all(
        "SELECT team_id, game_date, elo FROM elo_ratings ORDER BY team_id, game_date"
    )
    _elo_by_team: dict[int, list[tuple[str, float]]] = {}
    for r in _elo_rows:
        _elo_by_team.setdefault(r["team_id"], []).append((r["game_date"], r["elo"]))

    def _cached_elo(team_id: int, game_date: str) -> float:
        entries = _elo_by_team.get(team_id)
        if not entries:
            return 1500.0
        idx = bisect.bisect_left(entries, (game_date,)) - 1
        return entries[idx][1] if idx >= 0 else 1500.0
```

### Step 2: Build on/off impact cache

```python
    # ── Pre-build on/off impact: (team_id, season) -> weighted_sum ──
    _impact_rows = db.fetch_all(
        "SELECT team_id, season, net_rating_diff, on_court_minutes "
        "FROM player_impact WHERE on_court_minutes > 0"
    )
    _onoff_cache: dict[tuple[int, str], float] = {}
    for r in _impact_rows:
        key = (r["team_id"], r["season"])
        nrd = r["net_rating_diff"]
        if nrd is not None:
            _onoff_cache[key] = _onoff_cache.get(key, 0.0) + nrd * min(r["on_court_minutes"], 30) / 30.0
```

### Step 3: Build odds cache

```python
    # ── Pre-build odds: (game_date, home_team_id, away_team_id) -> dict ──
    _odds_rows = db.fetch_all(
        "SELECT game_date, home_team_id, away_team_id, spread_home_money, spread_home_public "
        "FROM game_odds"
    )
    _odds_cache: dict[tuple[str, int, int], dict] = {}
    for r in _odds_rows:
        _odds_cache[(r["game_date"], r["home_team_id"], r["away_team_id"])] = r
```

### Step 4: Update _precompute_one to use caches

**Elo (replace lines 1477-1478):**
```python
# BEFORE:
                home_elo=get_team_elo(htid, gdate, game_season),
                away_elo=get_team_elo(atid, gdate, game_season),

# AFTER:
                home_elo=_cached_elo(htid, gdate),
                away_elo=_cached_elo(atid, gdate),
```

**On/Off impact (replace lines 1398-1417):**
```python
# BEFORE:
            _home_onoff = 0.0
            _away_onoff = 0.0
            for _side, _tid, _target in [("home", htid, "_home_onoff"), ("away", atid, "_away_onoff")]:
                from src.database import db as _db
                _impact_rows = _db.fetch_all(
                    "SELECT pi.net_rating_diff, pi.on_court_minutes "
                    "FROM player_impact pi "
                    "WHERE pi.season = ? AND pi.team_id = ? AND pi.on_court_minutes > 0",
                    (game_season, _tid),
                )
                _total_impact = sum(
                    r["net_rating_diff"] * min(r["on_court_minutes"], 30) / 30.0
                    for r in _impact_rows
                    if r.get("net_rating_diff") is not None
                ) if _impact_rows else 0.0
                if _side == "home":
                    _home_onoff = _total_impact
                else:
                    _away_onoff = _total_impact

# AFTER:
            _home_onoff = _onoff_cache.get((htid, game_season), 0.0)
            _away_onoff = _onoff_cache.get((atid, game_season), 0.0)
```

**Odds (replace lines 1419-1430):**
```python
# BEFORE:
            _spread_sharp = 0.0
            from src.database import db as _db
            _odds_row = _db.fetch_all(
                "SELECT spread_home_money, spread_home_public FROM game_odds "
                "WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
                (gdate, htid, atid),
            )
            if _odds_row:
                _sm = _odds_row[0].get("spread_home_money", 0) or 0
                _sp = _odds_row[0].get("spread_home_public", 0) or 0
                _spread_sharp = float(_sm - _sp)

# AFTER:
            _spread_sharp = 0.0
            _odds_entry = _odds_cache.get((gdate, htid, atid))
            if _odds_entry:
                _sm = _odds_entry.get("spread_home_money", 0) or 0
                _sp = _odds_entry.get("spread_home_public", 0) or 0
                _spread_sharp = float(_sm - _sp)
```

### Step 5: Remove unused import

Remove `get_team_elo` from the imports at the top of `_precompute_one` if it was imported there, or from the GameInput construction area. The `from src.analytics.elo import get_team_elo` at the module level can stay (used by `predict_matchup`).

### Step 6: Verify

```bash
python -c "from src.analytics.prediction import precompute_all_games; print('OK')"
```

### Step 7: Commit

```bash
git commit -m "perf: pre-build Elo/on-off/odds lookup caches for precompute"
```

---

## Verification Checklist (Post-Fix)

1. `python -c "from src.database.db import ensure_thread_local_db, close_thread_local_db; print('OK')"` — imports work
2. `python -c "from src.analytics.prediction import precompute_all_games; print('OK')"` — module loads
3. `grep -n "with thread_local_db" src/analytics/prediction.py` — should show only the `predict_matchup` usage, NOT the precompute path
4. `grep -n "ensure_thread_local_db" src/analytics/prediction.py` — should show the import and ThreadPoolExecutor initializer
5. Run a short overnight to measure improvement: `python overnight.py --plain --hours 0.5`
   - Compare precompute step timing against previous run
   - Verify game counts and predictions are unchanged

---

## Future Optimization (Not In This Plan)

**Pre-build team schedule context for momentum/schedule_spots/fg3_luck:**
These functions run 6-7, 4+, and 3 DB queries per call respectively (30+ per game, ~30,000 total). Building a sorted schedule map per team from one bulk query and using bisect lookups would eliminate all of these. This is a larger refactor and should be a follow-up after measuring the gains from Tasks 1-3.
