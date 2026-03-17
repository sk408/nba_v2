# Odds Monitor & Tomorrow Sync Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `num_bets` tracking, sync tomorrow's odds alongside today's, and piggyback periodic odds refresh on the injury monitor every 3rd cycle (~15 min).

**Architecture:** Extend `game_odds` table with `num_bets`. Add `nba_tomorrow()` helper and `sync_upcoming_odds()` convenience function. Piggyback odds refresh on the existing `InjuryMonitor._check_changes()` loop with a mod-3 cycle counter. Update the web odds sync button and pipeline step 8 to include tomorrow.

**Tech Stack:** Python 3, SQLite, threading (existing InjuryMonitor pattern), Action Network API

---

## Chunk 1: Database & Core Sync Changes

### Task 1: Add `num_bets` column to `game_odds`

**Files:**
- Modify: `src/database/migrations.py:345-368` (SCHEMA_SQL game_odds table)
- Modify: `src/database/migrations.py:457-496` (_run_column_migrations)
- Test: `tests/test_odds_monitor.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_odds_monitor.py`:

```python
"""Tests for odds monitor: num_bets column, upcoming odds sync, monitor integration."""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from src.database.migrations import init_db
from src.database import db


def test_game_odds_has_num_bets_column():
    """num_bets column exists in game_odds after init_db."""
    cols = db.fetch_all("PRAGMA table_info(game_odds)")
    col_names = [c["name"] for c in cols]
    assert "num_bets" in col_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds_monitor.py::test_game_odds_has_num_bets_column -v`
Expected: FAIL — `num_bets` not in column list

- [ ] **Step 3: Add `num_bets` to schema and migration**

In `src/database/migrations.py`, add `num_bets INTEGER` to the `game_odds` CREATE TABLE in SCHEMA_SQL (after `ml_away_money INTEGER`):

```python
    ml_away_money INTEGER,
    num_bets INTEGER,
```

In `_run_column_migrations()`, add after the `spread_movement` migration line:

```python
    _add_column_if_missing("game_odds", "num_bets", "INTEGER")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds_monitor.py::test_game_odds_has_num_bets_column -v`
Expected: PASS

---

### Task 2: Capture `num_bets` in `sync_odds_for_date()`

**Files:**
- Modify: `src/data/odds_sync.py:256-393` (sync_odds_for_date function)
- Test: `tests/test_odds_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_odds_monitor.py`:

```python
@patch("src.data.odds_sync.fetch_action_odds")
@patch("src.data.odds_sync.sync_betting_splits", return_value=0)
def test_sync_odds_saves_num_bets(mock_sbd, mock_fetch):
    """sync_odds_for_date saves num_bets from consensus odds."""
    from src.data.odds_sync import sync_odds_for_date
    from src.analytics.stats_engine import get_team_abbreviations

    id_to_abbr = get_team_abbreviations()
    abbr_to_id = {v: k for k, v in id_to_abbr.items()}

    # Pick two real teams from the DB
    abbrs = list(abbr_to_id.keys())[:2]
    home_abbr, away_abbr = abbrs[0], abbrs[1]
    home_id, away_id = abbr_to_id[home_abbr], abbr_to_id[away_abbr]

    # Map abbreviations back through the Action Network mapper
    from src.utils.team_mapper import normalize_action_abbr
    # Build reverse map: find AN abbreviation that maps to each standard abbr
    # For simplicity, use standard abbreviations (the mapper handles edge cases)

    mock_fetch.return_value = [{
        "id": 999,
        "home_team_id": 100,
        "away_team_id": 200,
        "teams": [
            {"id": 100, "abbr": home_abbr},
            {"id": 200, "abbr": away_abbr},
        ],
        "odds": [{
            "type": "game",
            "book_id": 15,
            "spread_home": -5.5,
            "total": 220.0,
            "ml_home": -220,
            "ml_away": 180,
            "spread_home_public": 60,
            "spread_away_public": 40,
            "spread_home_money": 55,
            "spread_away_money": 45,
            "ml_home_public": 70,
            "ml_away_public": 30,
            "ml_home_money": 65,
            "ml_away_money": 35,
            "num_bets": 12345,
        }],
    }]

    saved = sync_odds_for_date("2026-03-16", invalidate_cache=False)
    assert saved == 1

    row = db.fetch_one(
        "SELECT num_bets FROM game_odds WHERE game_date = ? AND home_team_id = ?",
        ("2026-03-16", home_id),
    )
    assert row is not None
    assert row["num_bets"] == 12345
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds_monitor.py::test_sync_odds_saves_num_bets -v`
Expected: FAIL — `num_bets` not in INSERT statement

- [ ] **Step 3: Update `sync_odds_for_date` to capture `num_bets`**

In `src/data/odds_sync.py`, in the `sync_odds_for_date()` function, after the sharp money extraction (line ~335):

```python
            # Bet count (updates as more bets come in)
            num_bets = game_odds.get("num_bets")
```

Update the INSERT statement to include `num_bets` in both the column list and VALUES, and add `num_bets=excluded.num_bets` to the ON CONFLICT UPDATE clause:

```python
            db.execute("""
                INSERT INTO game_odds (
                    game_date, home_team_id, away_team_id, spread, over_under,
                    home_moneyline, away_moneyline, fetched_at, provider,
                    spread_home_public, spread_away_public, spread_home_money, spread_away_money,
                    ml_home_public, ml_away_public, ml_home_money, ml_away_money,
                    num_bets
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'actionnetwork', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_date, home_team_id, away_team_id) DO UPDATE SET
                    spread=excluded.spread,
                    over_under=excluded.over_under,
                    home_moneyline=excluded.home_moneyline,
                    away_moneyline=excluded.away_moneyline,
                    spread_home_public=excluded.spread_home_public,
                    spread_away_public=excluded.spread_away_public,
                    spread_home_money=excluded.spread_home_money,
                    spread_away_money=excluded.spread_away_money,
                    ml_home_public=excluded.ml_home_public,
                    ml_away_public=excluded.ml_away_public,
                    ml_home_money=excluded.ml_home_money,
                    ml_away_money=excluded.ml_away_money,
                    num_bets=excluded.num_bets,
                    fetched_at=excluded.fetched_at,
                    provider=excluded.provider
            """, (game_date, home_id, away_id, spread, ou, home_ml, away_ml, now,
                  spread_home_public, spread_away_public, spread_home_money, spread_away_money,
                  ml_home_public, ml_away_public, ml_home_money, ml_away_money,
                  num_bets))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds_monitor.py::test_sync_odds_saves_num_bets -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/database/migrations.py src/data/odds_sync.py tests/test_odds_monitor.py
git commit -m "feat: add num_bets column to game_odds and capture from Action Network"
```

---

### Task 3: Add `nba_tomorrow()` helper

**Files:**
- Modify: `src/utils/timezone_utils.py`
- Test: `tests/test_odds_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_odds_monitor.py`:

```python
def test_nba_tomorrow_returns_next_day():
    """nba_tomorrow() returns the day after nba_today()."""
    from src.utils.timezone_utils import nba_today, nba_tomorrow
    from datetime import datetime, timedelta

    today = datetime.strptime(nba_today(), "%Y-%m-%d")
    tomorrow = datetime.strptime(nba_tomorrow(), "%Y-%m-%d")
    assert tomorrow - today == timedelta(days=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds_monitor.py::test_nba_tomorrow_returns_next_day -v`
Expected: FAIL — `nba_tomorrow` cannot be imported

- [ ] **Step 3: Add `nba_tomorrow()` to timezone_utils**

In `src/utils/timezone_utils.py`, after `nba_today()`:

```python
def nba_tomorrow() -> str:
    """Return the NBA "next game date" as YYYY-MM-DD.

    Uses the same 6 AM ET rollover logic as ``nba_today`` but adds one day.
    """
    return (datetime.now(tz=_ET) - timedelta(hours=6) + timedelta(days=1)).strftime("%Y-%m-%d")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds_monitor.py::test_nba_tomorrow_returns_next_day -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/timezone_utils.py tests/test_odds_monitor.py
git commit -m "feat: add nba_tomorrow() helper for next-day odds syncing"
```

---

### Task 4: Add `sync_upcoming_odds()` convenience function

**Files:**
- Modify: `src/data/odds_sync.py` (add new function at end)
- Test: `tests/test_odds_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_odds_monitor.py`:

```python
@patch("src.data.odds_sync.sync_odds_for_date")
def test_sync_upcoming_odds_calls_today_and_tomorrow(mock_sync):
    """sync_upcoming_odds calls sync_odds_for_date for both today and tomorrow."""
    from src.data.odds_sync import sync_upcoming_odds

    mock_sync.return_value = 3
    result = sync_upcoming_odds()

    assert mock_sync.call_count == 2
    dates_called = [call.args[0] for call in mock_sync.call_args_list]
    # Should contain two different dates (today and tomorrow)
    assert len(set(dates_called)) == 2
    assert result == 6  # 3 + 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_odds_monitor.py::test_sync_upcoming_odds_calls_today_and_tomorrow -v`
Expected: FAIL — `sync_upcoming_odds` cannot be imported

- [ ] **Step 3: Implement `sync_upcoming_odds()`**

Add at the end of `src/data/odds_sync.py`:

```python
def sync_upcoming_odds(callback: Optional[Callable] = None) -> int:
    """Sync odds for today and tomorrow (Action Network primary, ESPN/SBD fallback).

    Returns total games saved/updated across both dates.
    """
    from src.utils.timezone_utils import nba_today, nba_tomorrow

    total = 0
    for date in (nba_today(), nba_tomorrow()):
        try:
            saved = sync_odds_for_date(date, callback=callback)
            total += saved
        except Exception as e:
            logger.warning("sync_upcoming_odds failed for %s: %s", date, e)
    return total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_odds_monitor.py::test_sync_upcoming_odds_calls_today_and_tomorrow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/odds_sync.py tests/test_odds_monitor.py
git commit -m "feat: add sync_upcoming_odds() for today + tomorrow odds refresh"
```

---

## Chunk 2: Monitor Integration, Web Endpoint, Pipeline

### Task 5: Add ODDS notification category

**Files:**
- Modify: `src/notifications/models.py`

- [ ] **Step 1: Add ODDS to NotificationCategory**

In `src/notifications/models.py`:

```python
class NotificationCategory(str, Enum):
    INJURY = "injury"
    UNDERDOG = "underdog"
    ODDS = "odds"
```

- [ ] **Step 2: Commit**

```bash
git add src/notifications/models.py
git commit -m "feat: add ODDS notification category for line movement alerts"
```

---

### Task 6: Piggyback odds refresh on InjuryMonitor

**Files:**
- Modify: `src/notifications/injury_monitor.py`
- Test: `tests/test_odds_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_odds_monitor.py`:

```python
@patch("src.notifications.injury_monitor.sync_upcoming_odds")
@patch("src.notifications.injury_monitor.sync_injuries", return_value=0)
def test_injury_monitor_refreshes_odds_every_3rd_cycle(mock_inj, mock_odds):
    """InjuryMonitor calls sync_upcoming_odds every 3rd _check_changes cycle."""
    from src.notifications.injury_monitor import InjuryMonitor

    monitor = InjuryMonitor()
    monitor._previous_state = {}

    # Run 6 cycles
    for _ in range(6):
        monitor._check_changes()

    # Should have called odds sync exactly 2 times (cycles 3 and 6)
    assert mock_odds.call_count == 2


@patch("src.notifications.injury_monitor.sync_upcoming_odds")
@patch("src.notifications.injury_monitor.sync_injuries", return_value=0)
def test_injury_monitor_skips_odds_on_non_3rd_cycles(mock_inj, mock_odds):
    """InjuryMonitor does NOT call odds sync on cycles 1, 2."""
    from src.notifications.injury_monitor import InjuryMonitor

    monitor = InjuryMonitor()
    monitor._previous_state = {}

    # Run 2 cycles
    for _ in range(2):
        monitor._check_changes()

    assert mock_odds.call_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_odds_monitor.py::test_injury_monitor_refreshes_odds_every_3rd_cycle tests/test_odds_monitor.py::test_injury_monitor_skips_odds_on_non_3rd_cycles -v`
Expected: FAIL — `sync_upcoming_odds` not imported in injury_monitor

- [ ] **Step 3: Add odds refresh to InjuryMonitor**

In `src/notifications/injury_monitor.py`:

Add import at top (after existing imports):
```python
from src.data.odds_sync import sync_upcoming_odds
```

Add cycle counter to `__init__`:
```python
    def __init__(self):
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._previous_state: Dict[int, Dict] = {}
        self._lock = threading.Lock()
        self._odds_cycle = 0  # odds refresh every 3rd cycle (~15 min)
```

At the end of `_check_changes()`, after updating state (after `self._previous_state = current_state`), add odds refresh:

```python
        # Odds refresh — piggyback every 3rd cycle (~15 min)
        self._odds_cycle += 1
        if self._odds_cycle % 3 == 0:
            try:
                updated = sync_upcoming_odds()
                if updated:
                    logger.info("Odds monitor refreshed %d game(s)", updated)
            except Exception as e:
                logger.debug("Odds refresh failed: %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_odds_monitor.py::test_injury_monitor_refreshes_odds_every_3rd_cycle tests/test_odds_monitor.py::test_injury_monitor_skips_odds_on_non_3rd_cycles -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/notifications/injury_monitor.py tests/test_odds_monitor.py
git commit -m "feat: piggyback odds refresh on injury monitor every 3rd cycle (~15 min)"
```

---

### Task 7: Update web odds sync endpoint for tomorrow

**Files:**
- Modify: `src/web/app.py:1884-1963` (api_sync_odds_today route)
- Test: `tests/test_odds_monitor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_odds_monitor.py`:

```python
@patch("src.data.odds_sync.sync_odds_for_date", return_value=4)
def test_sync_upcoming_odds_is_importable_and_works(mock_sync):
    """sync_upcoming_odds is importable from odds_sync and calls two dates."""
    from src.data.odds_sync import sync_upcoming_odds
    result = sync_upcoming_odds()
    assert mock_sync.call_count == 2
    assert result == 8
```

- [ ] **Step 2: Run test to verify it passes** (already implemented in Task 4)

Run: `python -m pytest tests/test_odds_monitor.py::test_sync_upcoming_odds_is_importable_and_works -v`
Expected: PASS

- [ ] **Step 3: Update web endpoint to sync today + tomorrow**

In `src/web/app.py`, modify the `_run_odds_sync()` inner function (line ~1904):

Replace the single-date sync with `sync_upcoming_odds`:

```python
    def _run_odds_sync():
        global _sync_running, _sync_status
        today = nba_today()

        try:
            from src.data.odds_sync import sync_upcoming_odds, sync_odds_for_date
            from src.utils.timezone_utils import nba_tomorrow

            tomorrow = nba_tomorrow()

            missing_before = _count_missing_odds_for_today(today)
            if missing_before is not None:
                _sync_status = (
                    f"Odds sync running for {today} + {tomorrow} "
                    f"({missing_before} missing matchup(s) for today)..."
                )
            else:
                _sync_status = f"Odds sync running for {today} + {tomorrow}..."

            saved_count = sync_upcoming_odds(
                callback=lambda msg: _update_sync_status(msg),
            )
            missing_after = _count_missing_odds_for_today(today)

            if missing_before is not None and missing_after is not None:
                filled = max(0, missing_before - missing_after)
                if filled > 0:
                    _sync_status = (
                        f"Odds sync complete. Filled {filled} missing matchup(s). "
                        f"Updated {saved_count} game(s) for {today} + {tomorrow}."
                    )
                elif missing_after == 0:
                    _sync_status = (
                        f"Odds sync complete. {saved_count} game(s) updated for {today} + {tomorrow}."
                    )
                elif saved_count > 0:
                    _sync_status = (
                        f"Odds sync complete. Updated {saved_count} game(s); "
                        f"{missing_after} matchup(s) still missing for today."
                    )
                else:
                    _sync_status = (
                        f"Odds sync complete. No new odds returned; "
                        f"{missing_after} matchup(s) still missing for today."
                    )
            elif saved_count > 0:
                _sync_status = f"Odds sync complete. Updated {saved_count} game(s) for {today} + {tomorrow}."
            else:
                _sync_status = f"Odds sync complete. No new odds returned."
        except Exception as e:
            logger.error("Background odds sync error: %s", e, exc_info=True)
            _sync_status = "Odds sync failed. See server logs."
        finally:
            with _sync_lock:
                _sync_running = False
```

Also update the return message (line ~1960):

```python
    return jsonify({
        "status": "started",
        "message": "Odds sync for today + tomorrow started in background",
    })
```

- [ ] **Step 4: Commit**

```bash
git add src/web/app.py
git commit -m "feat: web odds sync button now fetches today + tomorrow odds"
```

---

### Task 8: Update sync pipeline step 8 for tomorrow

**Files:**
- Modify: `src/data/sync_service.py:774-793` (sync_historical_odds function)

- [ ] **Step 1: Update `sync_historical_odds` to also sync tomorrow**

In `src/data/sync_service.py`, modify `sync_historical_odds()`:

```python
def sync_historical_odds(callback: Optional[Callable] = None, force: bool = False):
    """Step 7: Sync Vegas odds for recent games + upcoming (today/tomorrow)."""
    from src.data.odds_sync import backfill_odds, sync_upcoming_odds

    meta = _get_sync_meta("odds_sync")
    current_gc = _get_game_count()
    if not force and _is_fresh("odds_sync", 24) and meta.get("game_count_at_sync", 0) == current_gc:
        if callback:
            callback("Odds sync is fresh, skipping...")
        return

    if callback:
        callback("Syncing historical Vegas odds...")

    count = backfill_odds(callback=callback, force=force)

    # Also sync upcoming games (today + tomorrow) for fresh sharp money
    if callback:
        callback("Syncing upcoming odds (today + tomorrow)...")
    upcoming = sync_upcoming_odds(callback=callback)
    count += upcoming

    _set_sync_meta("odds_sync", current_gc, _get_last_game_date())

    if callback:
        callback(f"Odds sync complete: {count} games updated.")
```

- [ ] **Step 2: Commit**

```bash
git add src/data/sync_service.py
git commit -m "feat: sync pipeline step 8 now includes tomorrow's odds"
```

---

### Task 9: Run full test suite and verify

- [ ] **Step 1: Run all odds monitor tests**

Run: `python -m pytest tests/test_odds_monitor.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing odds-related tests to check for regressions**

Run: `python -m pytest tests/ -k "odds or sync" -v`
Expected: All PASS (no regressions)

- [ ] **Step 3: Manual verification**

Run a quick check that tomorrow's odds sync works end-to-end:

```python
python -c "
from src.database.migrations import init_db
init_db()
from src.data.odds_sync import sync_upcoming_odds
result = sync_upcoming_odds(callback=print)
print(f'Total updated: {result}')

from src.database import db
rows = db.fetch_all('SELECT game_date, COUNT(*) as games, AVG(num_bets) as avg_bets FROM game_odds WHERE game_date >= date(\"now\") GROUP BY game_date')
for r in rows:
    print(dict(r))
"
```

Expected: Rows for both today and tomorrow with non-null `num_bets`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: odds monitor with tomorrow sync, num_bets tracking, 15-min refresh"
```
