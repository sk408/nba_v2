# Static Audit Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 8 confirmed findings from the static error audit, plus document 6 deferred findings with rationale.

**Architecture:** Each fix is independent. Ordered by impact: correctness bugs first (season leakage corrupts ~85% of training features), then data pipeline bugs, then UI/cleanup. All changes are backwards-compatible — new `season` parameters default to `None` which falls back to `get_season()`.

**Tech Stack:** Python, SQLite, PySide6

---

## Task 1: Fix season leakage in feature functions

**Why first:** `compute_momentum`, `compute_fg3_luck`, and `compute_schedule_spots` all call `get_season()` which returns the *current* season. When `precompute_all_games()` processes historical games (e.g. 2022-23), these functions query against 2025-26 data — returning zeros for everything. The optimizer trains on dead features. `get_home_court_advantage` has the same bug plus a cache keyed only by `team_id` (ignoring season).

**Files:**
- Modify: `src/analytics/stats_engine.py` (lines 697, 772, 839, 395)
- Modify: `src/analytics/prediction.py` (lines 1299, 1391-1394, 1508-1509)

**Step 1: Add `season` parameter to `compute_momentum()`**

In `src/analytics/stats_engine.py`, line 697:

```python
# BEFORE:
def compute_momentum(team_id: int, game_date: str) -> Dict[str, Any]:

# AFTER:
def compute_momentum(team_id: int, game_date: str, season: Optional[str] = None) -> Dict[str, Any]:
```

Replace lines 704-709:
```python
# BEFORE:
    from src.config import get_season
    result = {"streak": 0, "mov_trend": 0.0}
    try:
        season = get_season()

# AFTER:
    result = {"streak": 0, "mov_trend": 0.0}
    try:
        if season is None:
            from src.config import get_season
            season = get_season()
```

**Step 2: Add `season` parameter to `compute_fg3_luck()`**

Same file, line 772:

```python
# BEFORE:
def compute_fg3_luck(team_id: int, game_date: str) -> float:

# AFTER:
def compute_fg3_luck(team_id: int, game_date: str, season: Optional[str] = None) -> float:
```

Replace lines 781-784:
```python
# BEFORE:
    from src.config import get_season
    try:
        season = get_season()

# AFTER:
    try:
        if season is None:
            from src.config import get_season
            season = get_season()
```

**Step 3: Add `season` parameter to `compute_schedule_spots()`**

Same file, line 839:

```python
# BEFORE:
def compute_schedule_spots(team_id: int, game_date: str,
                           opponent_team_id: int) -> Dict[str, Any]:

# AFTER:
def compute_schedule_spots(team_id: int, game_date: str,
                           opponent_team_id: int, season: Optional[str] = None) -> Dict[str, Any]:
```

Replace lines 848-853:
```python
# BEFORE:
    from src.config import get_season
    result = {"lookahead": False, "letdown": False, "road_trip_game": 0}
    try:
        season = get_season()

# AFTER:
    result = {"lookahead": False, "letdown": False, "road_trip_game": 0}
    try:
        if season is None:
            from src.config import get_season
            season = get_season()
```

**Step 4: Fix HCA cache key to include season**

In `get_home_court_advantage()` (line 395-424), the cache at line 402-404 is keyed by `team_id` only. Change to `(team_id, season)`:

```python
# BEFORE (lines 402-404):
    with _hca_cache_lock:
        if team_id in _hca_cache:
            return _hca_cache[team_id]

# AFTER:
    cache_key = (team_id, season)
    with _hca_cache_lock:
        if cache_key in _hca_cache:
            return _hca_cache[cache_key]
```

Also update lines 414-415 and 422-423 to use `cache_key`:
```python
# Line 414-415 BEFORE:
        _hca_cache[team_id] = _HOME_COURT_FALLBACK

# AFTER:
        _hca_cache[cache_key] = _HOME_COURT_FALLBACK

# Line 422-423 BEFORE:
        _hca_cache[team_id] = result

# AFTER:
        _hca_cache[cache_key] = result
```

**Step 5: Pass `game_season` from precompute callers**

In `src/analytics/prediction.py`, `_precompute_one()` already derives `game_season` on line 1302 via `_game_date_to_season(gdate)`. Pass it through:

```python
# Line 1299 BEFORE:
home_court = get_home_court_advantage(htid)
# AFTER:
home_court = get_home_court_advantage(htid, season=game_season)

# Lines 1391-1394 BEFORE:
_home_momentum = compute_momentum(htid, gdate)
_away_momentum = compute_momentum(atid, gdate)
_home_sched = compute_schedule_spots(htid, gdate, atid)
_away_sched = compute_schedule_spots(atid, gdate, htid)
# AFTER:
_home_momentum = compute_momentum(htid, gdate, season=game_season)
_away_momentum = compute_momentum(atid, gdate, season=game_season)
_home_sched = compute_schedule_spots(htid, gdate, atid, season=game_season)
_away_sched = compute_schedule_spots(atid, gdate, htid, season=game_season)

# Lines 1508-1509 BEFORE:
home_fg3_luck=compute_fg3_luck(htid, gdate),
away_fg3_luck=compute_fg3_luck(atid, gdate),
# AFTER:
home_fg3_luck=compute_fg3_luck(htid, gdate, season=game_season),
away_fg3_luck=compute_fg3_luck(atid, gdate, season=game_season),
```

**Step 6: Verify**

Run: `python -c "from src.analytics.stats_engine import compute_momentum, compute_fg3_luck, compute_schedule_spots; print('OK')"`

**Step 7: Commit**

```bash
git add src/analytics/stats_engine.py src/analytics/prediction.py
git commit -m "fix: thread season param through feature functions to prevent historical leakage"
```

---

## Task 2: Fix odds sync — propagate force flag and fix date-level skip

**Why:** Two independent bugs: (1) `sync_historical_odds` accepts `force=True` but never passes it to `backfill_odds`, so force-refresh is silently ignored. (2) The non-force query skips an entire *date* if any game on that date has odds, missing games on partially-populated dates.

**Files:**
- Modify: `src/data/sync_service.py` (line 638)
- Modify: `src/data/odds_sync.py` (lines 165-172)

**Step 1: Pass `force` through to `backfill_odds()`**

In `src/data/sync_service.py`, line 638:

```python
# BEFORE:
    count = backfill_odds(callback=callback)

# AFTER:
    count = backfill_odds(callback=callback, force=force)
```

**Step 2: Fix date-level skip to game-level check**

The `game_odds` table has PK `(game_date, home_team_id, away_team_id)` — no `game_id` column. The fix must use the available keys. For `player_stats` rows where `is_home = 1`, the player's own team is the home team, obtainable via `JOIN players`. But since Finding #4 (team attribution) notes `players.team_id` is mutable, use a simpler count-based approach:

In `src/data/odds_sync.py`, replace lines 165-172:

```python
# BEFORE:
        rows = db.fetch_all("""
            SELECT DISTINCT game_date
            FROM player_stats
            WHERE game_date NOT IN (
                SELECT game_date FROM game_odds WHERE spread_home_public IS NOT NULL
            )
            ORDER BY game_date DESC
        """)

# AFTER:
        # Game-level completeness: find dates where the number of games
        # with complete odds is fewer than the number of games played.
        # Each game has exactly one game_id; game_odds has one row per game.
        rows = db.fetch_all("""
            SELECT game_date
            FROM (
                SELECT ps.game_date,
                       COUNT(DISTINCT ps.game_id) as games_played,
                       COUNT(DISTINCT go.home_team_id) as games_with_odds
                FROM player_stats ps
                LEFT JOIN game_odds go
                    ON go.game_date = ps.game_date
                    AND go.spread_home_public IS NOT NULL
                WHERE ps.game_id IS NOT NULL
                GROUP BY ps.game_date
            )
            WHERE games_with_odds < games_played
            ORDER BY game_date DESC
        """)
```

Note: `COUNT(DISTINCT ps.game_id)` gives the number of actual games played on that date. `COUNT(DISTINCT go.home_team_id)` gives games with complete odds on that date. When the latter is less, we re-fetch odds for that date.

**Step 3: Verify**

Run: `python -c "from src.data.odds_sync import backfill_odds; print('OK')"`

**Step 4: Commit**

```bash
git add src/data/sync_service.py src/data/odds_sync.py
git commit -m "fix: propagate force flag and use game-level odds completeness check"
```

---

## Task 3: Sync pipeline step definitions between core and UI

**Why:** Core pipeline (`pipeline.py:261-274`) has 10 steps but the UI (`pipeline_view.py:40-47`) only tracks 6. The 4 V2.1 steps (seed_arenas, bbref_sync, referee_sync, elo_compute) are invisible in the UI — progress bars, timing, and step indicators are incomplete.

**Files:**
- Modify: `src/ui/views/pipeline_view.py` (lines 1-2, 39-47)

**Step 1: Update STEP_LABELS to match pipeline's 10 steps**

```python
# BEFORE (lines 39-47):
# Pipeline steps (name, display label)
STEP_LABELS = [
    ("backup", "Backup"),
    ("sync", "Data Sync"),
    ("precompute", "Precompute"),
    ("optimize_fundamentals", "Optimize Fund."),
    ("optimize_sharp", "Optimize Sharp"),
    ("backtest", "Backtest"),
]

# AFTER:
# Pipeline steps (name, display label) — must match PIPELINE_STEPS in pipeline.py
STEP_LABELS = [
    ("backup", "Backup"),
    ("sync", "Data Sync"),
    ("seed_arenas", "Arenas"),
    ("bbref_sync", "BBRef"),
    ("referee_sync", "Referees"),
    ("elo_compute", "Elo"),
    ("precompute", "Precompute"),
    ("optimize_fundamentals", "Optimize Fund."),
    ("optimize_sharp", "Optimize Sharp"),
    ("backtest", "Backtest"),
]
```

**Step 2: Update the module docstring**

```python
# BEFORE (lines 1-2):
"""Pipeline tab -- run/monitor pipeline, view step state, manage snapshots.

V2 design: 6-step pipeline (backup, sync, precompute, optimize x2, backtest)

# AFTER:
"""Pipeline tab -- run/monitor pipeline, view step state, manage snapshots.

V2.1 design: 10-step pipeline (backup, sync, arenas, bbref, referees, elo,
precompute, optimize x2, backtest)
```

**Step 3: Verify**

Run: `python -c "from src.ui.views.pipeline_view import STEP_LABELS; print(f'{len(STEP_LABELS)} steps'); assert len(STEP_LABELS) == 10"`

**Step 4: Commit**

```bash
git add src/ui/views/pipeline_view.py
git commit -m "fix: sync UI step labels with 10-step pipeline definition"
```

---

## Task 4: Align optimizer winner threshold with live prediction

**Why:** The optimizer evaluates accuracy using `game_score > 0.5` / `< -0.5` (lines 376-377), creating a dead zone where scores between -0.5 and 0.5 count as "no pick." But live predictions use `> 0` / `< 0` — every game gets a pick. The optimizer therefore optimizes for a different decision boundary than what's used in production.

The fix: change the optimizer to use `> 0` / `< 0`. The `actual_spread` thresholds (lines 378-380) stay at 0.5 because actual pushes are real in spread betting.

**Files:**
- Modify: `src/analytics/optimizer.py` (lines 376-377, 920-921)

**Step 1: Fix prediction threshold in metrics block**

```python
# BEFORE (lines 376-377):
        pred_home_win = game_score > 0.5
        pred_away_win = game_score < -0.5

# AFTER:
        pred_home_win = game_score > 0
        pred_away_win = game_score < 0
```

**Step 2: Fix threshold in sharp-flip accuracy block**

```python
# BEFORE (lines 920-921):
    actual_home_win = vg.actual_spread > 0.5
    actual_away_win = vg.actual_spread < -0.5

# NOTE: These are ACTUAL spread thresholds, not prediction thresholds.
# Keep at 0.5 — a 0.3-point actual margin is legitimately a push.
# No change needed here.
```

**Step 3: Verify**

Run: `python -c "from src.analytics.optimizer import optimize_weights; print('OK')"`

**Step 4: Commit**

```bash
git add src/analytics/optimizer.py
git commit -m "fix: align optimizer prediction threshold with live path (>0 not >0.5)"
```

---

## Task 5: Fix sync metadata to read counts from DB instead of unloaded store

**Why:** `_mark_step_done` creates a fresh `InMemoryDataStore()` and immediately calls `get_game_count_and_last_date()` without calling `load()`. Since `player_stats` is `None` on a fresh instance, this always writes `(0, "")` into `sync_meta` — making freshness checks unreliable.

**Files:**
- Modify: `src/analytics/pipeline.py` (lines 109-114)

**Step 1: Replace memory store read with direct DB query**

```python
# BEFORE (lines 109-114):
def _mark_step_done(step_name: str):
    """Mark a step as completed in sync_meta."""
    from src.analytics.memory_store import InMemoryDataStore

    store = InMemoryDataStore()
    count, last_date = store.get_game_count_and_last_date()

# AFTER:
def _mark_step_done(step_name: str):
    """Mark a step as completed in sync_meta."""
    row = db.fetch_one(
        "SELECT COUNT(*) as cnt, MAX(game_date) as last_date FROM player_stats"
    )
    count = row["cnt"] if row else 0
    last_date = row["last_date"] or "" if row else ""
```

**Step 2: Verify**

Run: `python -c "from src.analytics.pipeline import _mark_step_done; print('OK')"`

**Step 3: Commit**

```bash
git add src/analytics/pipeline.py
git commit -m "fix: read game counts from DB directly instead of unloaded memory store"
```

---

## Task 6: Remove dead worker classes

**Why:** Multiple worker classes in `workers.py` import functions that don't exist (`precompute_game_data`, `compute_feature_importance`, `per_team_refinement`, `run_full_pipeline`, `run_retune`, `get_shap_importance`, `train_models` from nonexistent `ml_model` module). These are lazy imports inside `run()` methods so they don't fail at module load, but they crash with `ImportError` if any of these workers are ever triggered from the UI.

`OverviewWorker` additionally passes invalid kwargs (`skip_ml=True, skip_espn=True`) to `predict_matchup` which doesn't accept them. `start_overview_worker` is never imported or called outside `workers.py`.

**Files:**
- Modify: `src/ui/workers.py`

**Step 1: Identify which workers are dead**

Search the codebase for imports/references to each worker class and its `start_*` function outside of `workers.py`. Workers that are truly unreferenced should be deleted. Workers that ARE referenced from UI views need their imports fixed to use current APIs.

For each dead worker: delete the class and its `start_*` function.
For each referenced worker with wrong imports: update to use `precompute_all_games` (from `prediction.py`), `optimize_weights` (from `optimizer.py`), `run_pipeline` (from `pipeline.py`).

**Step 2: Specifically delete OverviewWorker and start_overview_worker**

These are confirmed unreferenced and have broken `predict_matchup` kwargs.

**Step 3: Verify**

Run: `python -c "from src.ui.workers import *; print('OK')"`

**Step 4: Commit**

```bash
git add src/ui/workers.py
git commit -m "fix: remove dead worker classes with broken imports, fix referenced workers"
```

---

## Task 7: Change player_stats INSERT OR IGNORE to upsert

**Why:** `INSERT OR IGNORE` silently drops rows that conflict on the unique index `(player_id, game_id)`. If the NBA API corrects a stat line after initial fetch, the correction is never applied. Using `ON CONFLICT ... DO UPDATE` fixes this.

**Files:**
- Modify: `src/data/nba_fetcher.py` (lines 480-488)

**Step 1: Replace INSERT OR IGNORE with upsert**

```python
# BEFORE (lines 480-488):
            db.execute_many(
                """INSERT OR IGNORE INTO player_stats
                   (player_id, opponent_team_id, is_home, game_date, game_id, season,
                    points, rebounds, assists, minutes, steals, blocks, turnovers,
                    fg_made, fg_attempted, fg3_made, fg3_attempted, ft_made, ft_attempted,
                    oreb, dreb, plus_minus, win_loss, personal_fouls)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                batch,
            )

# AFTER:
            db.execute_many(
                """INSERT INTO player_stats
                   (player_id, opponent_team_id, is_home, game_date, game_id, season,
                    points, rebounds, assists, minutes, steals, blocks, turnovers,
                    fg_made, fg_attempted, fg3_made, fg3_attempted, ft_made, ft_attempted,
                    oreb, dreb, plus_minus, win_loss, personal_fouls)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(player_id, game_id) DO UPDATE SET
                    points=excluded.points, rebounds=excluded.rebounds,
                    assists=excluded.assists, minutes=excluded.minutes,
                    steals=excluded.steals, blocks=excluded.blocks,
                    turnovers=excluded.turnovers, fg_made=excluded.fg_made,
                    fg_attempted=excluded.fg_attempted, fg3_made=excluded.fg3_made,
                    fg3_attempted=excluded.fg3_attempted, ft_made=excluded.ft_made,
                    ft_attempted=excluded.ft_attempted, oreb=excluded.oreb,
                    dreb=excluded.dreb, plus_minus=excluded.plus_minus,
                    win_loss=excluded.win_loss, personal_fouls=excluded.personal_fouls""",
                batch,
            )
```

**Step 2: Verify**

Run: `python -c "from src.data.nba_fetcher import save_game_logs; print('OK')"`

**Step 3: Commit**

```bash
git add src/data/nba_fetcher.py
git commit -m "fix: use upsert for player_stats to accept upstream stat corrections"
```

---

## Task 8: Remove dead prediction_quality import guard

**Why:** `src/analytics/prediction_quality.py` doesn't exist. Two files import from it inside `try/except ImportError: pass` blocks, silently disabling odds cache invalidation. Since the module doesn't exist and hasn't for a while, remove the dead import blocks and add a `# TODO` for re-implementing invalidation if needed.

**Files:**
- Modify: `src/data/odds_sync.py` (lines 138-141)
- Modify: `src/data/sync_service.py` (lines 82-85 and line 152, if applicable)

**Step 1: Remove dead import blocks**

In each file, find the pattern:
```python
try:
    from src.analytics.prediction_quality import invalidate_odds_cache
    invalidate_odds_cache()
except ImportError:
    pass
```

Replace with nothing (just delete the block). If other code depends on it being there, add a comment:
```python
# TODO: re-implement odds cache invalidation when prediction_quality module is restored
```

**Step 2: Verify**

Run: `python -c "from src.data.odds_sync import backfill_odds; from src.data.sync_service import full_sync; print('OK')"`

**Step 3: Commit**

```bash
git add src/data/odds_sync.py src/data/sync_service.py
git commit -m "chore: remove dead prediction_quality import guards"
```

---

## Deferred Findings

These findings are confirmed real but deferred due to high risk, large scope, or low frequency.

### Deferred #1: Stale-read architecture in write-heavy workflows (Audit Finding #1)

**Status:** Confirmed. `thread_local_db()` creates a frozen in-memory snapshot. Wrapping full pipeline/overnight runs in it means reads don't see writes made during the same run.

**Why deferred:** The fix (removing `with thread_local_db():` from `overnight.py` lines 611/644 and orchestration workers) changes the concurrency model for the entire pipeline. The current execution order partially masks the issue because `precompute_all_games` creates per-game snapshots *after* sync completes (line 1283 in prediction.py). Needs careful concurrency testing.

**Future fix:** Remove `thread_local_db()` wrapper from `overnight.py` and pipeline-running workers. Keep it for read-heavy analytics (per-game precompute, sensitivity analysis, diagnostics).

### Deferred #2: Historical team attribution instability (Audit Finding #4)

**Status:** Confirmed. `player_stats` lacks a `team_id` column. Queries infer team via `JOIN players` whose `team_id` is overwritten by each roster sync. Historical aggregates attribute stats to the player's *current* team, not their team at game time.

**Why deferred:** Requires a schema migration (add `team_id` to `player_stats`), a data backfill (derive team from game context for all historical rows), and updating every team-aggregate query in `stats_engine.py`. Highest-risk change in the codebase. Should be its own dedicated plan.

**Future fix:** Add `team_id` column to `player_stats`, populate from game context during fetcher sync, migrate all `JOIN players` team lookups to use `player_stats.team_id` directly.

### Deferred #3: Nuke flow can leave partial data (Audit Finding #9)

**Status:** Confirmed. Table deletes in `sync_service.py:114-119` continue on exception. No FK ordering. Reports success regardless.

**Why deferred:** Nuke is an infrequent manual operation. The risk of partial cleanup is real but low-frequency, and the fix (transaction wrapping, FK ordering) needs careful testing with the full table dependency graph.

**Future fix:** Wrap deletes in a single transaction with explicit FK-safe ordering. Fail loudly on any error.

### Deferred #4: Full sync reports success after partial failures (Audit Finding #12)

**Status:** Confirmed. `full_sync()` at line 741-750 catches exceptions per step, logs them, but always emits "Full data sync complete!" at the end.

**Why deferred:** This is a reporting/UX issue, not a data correctness issue. The sync steps themselves handle their own data integrity.

**Future fix:** Track per-step pass/fail, return structured result, change final message to reflect actual outcome (e.g., "Sync completed with 2 errors").

### Deferred #5: OverviewWorker invalid kwargs (Audit Finding #11)

**Status:** Confirmed. `predict_matchup(..., skip_ml=True, skip_espn=True)` passes kwargs that don't exist in the function signature. Python silently ignores them — the intended behavior (skipping ML/ESPN) is not implemented.

**Why deferred:** Covered by Task 6 (dead worker removal). If OverviewWorker is instead kept and fixed, update the call to use valid parameters.

### Deferred #6: Optimizer picks vs evaluation threshold inconsistency (Audit Finding #10, partial)

**Status:** The optimizer pick logic at lines 909-913 uses `> 0` while the metrics block at 376-377 uses `> 0.5`. Task 4 fixes the metrics block to match. However, the `actual_spread` thresholds (lines 378-380, 920-921) intentionally use `> 0.5` for pushes. No change needed on actual-spread lines — documented here for future reference.

---

## Verification Checklist (Post-Fix)

After all tasks:

1. `python -c "from src.analytics.stats_engine import compute_momentum; print(compute_momentum.__code__.co_varnames[:4])"` — should include `season`
2. `python -c "from src.ui.views.pipeline_view import STEP_LABELS; assert len(STEP_LABELS) == 10"` — 10 steps
3. `python -c "from src.data.odds_sync import backfill_odds; print('OK')"` — imports clean
4. `python -c "from src.analytics.pipeline import _mark_step_done; print('OK')"` — no memory store import
5. `grep -rn 'OverviewWorker' src/ui/workers.py` — should return nothing (or show fixed version)
6. `grep -n 'INSERT OR IGNORE' src/data/nba_fetcher.py` — should return nothing
7. `grep -rn 'prediction_quality' src/` — should return nothing
8. Delete precompute cache and `optuna_studies.db`, then run overnight to rebuild with correct season-scoped features
