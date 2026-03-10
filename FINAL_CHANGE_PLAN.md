# NBA V2 — Final Change Plan
### Synthesized from Opus Review, Codex Review, and Both Unified Plans | March 9, 2026

---

## What This Is

Two AIs independently reviewed the full codebase, then read each other's work, then each produced a unified plan. This document is the final reconciliation — one authoritative list of changes to make, in order.

**Key insight from the process:** Codex found deeper runtime bugs (injury monitor, notification ID race, gamecast stale responses). Opus found broader systemic issues (security, infrastructure, performance). Both agreed on architecture concerns (season leakage, stale reads, cache coherency). The P0-V concept from Codex's unified plan (verify before patching) is adopted here for claims that need quick reproduction.

---

## Rules of Engagement

1. **Backtest before and after every phase.** Compare winner%, upset accuracy, ML ROI, spread MAE. If metrics regress unexpectedly, stop and investigate.
2. **Do NOT touch optimizer formula/weights in Phase 0.** Only fix data correctness and runtime behavior.
3. **Commit after each numbered item**, not after each phase. Small reviewable diffs.
4. **Add a regression test alongside every fix** — not after, alongside.
5. **Preserve old cache files** during cache-key changes. One-click rollback.

---

## Phase 0: Confirmed Runtime Bugs (Days 1-2)

*High-confidence bugs both reviewers agree are real. Fix first.*

### 0.1 Injury Monitor Never Persists Scraped Data
**Files:** `src/notifications/injury_monitor.py`, `src/data/injury_scraper.py`

The monitor calls `scrape_all_injuries()` then diffs against DB, but scrape doesn't write to DB. It's comparing stale DB vs stale DB every poll.

**Change:** Replace scrape-only call with `sync_injuries()` (or explicit persist step) before diffing. Add log line: `injury_monitor_db_updates=N`.

**Accept:** Status change in source feed updates DB row and emits exactly one notification.

---

### 0.2 Notification Insert ID Race
**Files:** `src/notifications/service.py`, `src/database/db.py`

`create_notification()` inserts then reads `last_insert_rowid()` as a separate call. Under concurrent writers, returns wrong row's ID.

**Change:** Add `execute_returning_id(sql, params)` to `db.py` that returns `cursor.lastrowid` from the same write operation. Update notification service to use it.

**Accept:** Concurrent insert test — each notification ID matches its inserted row.

---

### 0.3 Web Gamecast Stale-Response Race
**Files:** `src/web/templates/gamecast.html`

Rapid game switching: async fetches resolve out of order; stale response overwrites newer selection. Polling doesn't guard against overlapping requests.

**Change:**
- Add `requestSeq` counter, increment on each game selection, ignore responses with stale seq.
- Add `isFetching` guard, skip interval tick when prior fetch is running.
- Reset overlays/timers on game switch.

**Accept:** Rapidly switch games in browser; UI never shows previously-selected game.

---

### 0.4 Desktop Headshot Pending Set Never Clears
**Files:** `src/ui/views/gamecast_view.py`

`_pending_headshots` grows unbounded. URLs added but never removed on completion or failure. Failed fetches never retry.

**Change:** Remove URL from pending set in `finally` block. Add retry budget with backoff. Optionally TTL failed URLs.

**Accept:** Forced failure then success → image eventually appears. Set size stays bounded.

---

### 0.5 Cache Invalidation References Dead Functions
**Files:** `src/data/sync_service.py`, `src/data/odds_sync.py`, `src/analytics/prediction.py`, `src/analytics/backtester.py`

Invalidation imports reference missing functions (`invalidate_residual_cache`, `invalidate_tuning_cache`, etc.). Single try block means one failure skips all invalidation.

**Change:**
- Replace dead imports with live ones: `invalidate_precompute_cache()`, `invalidate_results_cache()`, `invalidate_backtest_cache()`, `invalidate_stats_caches()`, `invalidate_weight_cache()`.
- Split into independent try blocks.
- Add `invalidate_actionnetwork_cache()` in `src/data/gamecast.py`.

**Accept:** Sync/nuke reliably cold-starts all relevant caches. Smoke test that all invalidation function names are importable.

---

### 0.6 Schedule-Spot Query Missing Season Filter
**Files:** `src/analytics/stats_engine.py` (`compute_schedule_spots`)

Next-game query: `game_date > ? AND team_id = ?` without `season = ?`. Leaks across season boundaries in historical evaluation.

**Change:** Add `AND season = ?` to the query and related schedule lookups.

**Accept:** With multi-season data loaded, final games of a season don't pull next-season context.

---

### 0.7 Flask Debug Reloader Double-Bootstrap
**Files:** `web.py`, `src/bootstrap.py`

Flask debug mode runs bootstrap in both parent and reloader child process → duplicate injury monitors, doubled network load.

**Change:** Gate bootstrap on `os.environ.get('WERKZEUG_RUN_MAIN') == 'true'` or move under `if __name__ == "__main__"` with reloader-safe check.

**Accept:** Flask debug mode starts exactly one injury monitor instance.

---

## Phase 0-V: Verify Quickly, Then Patch (Day 2-3)

*High-impact claims that need 15-30 min reproduction before committing to fix. If confirmed, fix immediately.*

### V.1 Score Calibration Crash — Undefined Functions
**Files:** `src/analytics/score_calibration.py` (~lines 431, 441)

**Claim (Opus):** `_fit_mode_payload()` calls `_blend_near_spread()` and `_apply_team_point_ranges_arrays()` which don't exist. Also references undefined settings and variables.

**Verify:** Search codebase for these function names. Try to reach the code path in a test or manual run.

**If confirmed:** Gate behind `score_calibration_near_spread_enabled` (default `False`), or implement the missing functions. Do NOT leave crashing code reachable.

---

### V.2 Pipeline seed_arenas Parameter Mismatch
**Files:** `src/analytics/pipeline.py`

**Claim (Opus):** `run_seed_arenas()` doesn't accept `is_cancelled` but pipeline passes it.

**Verify:** Check `run_seed_arenas` signature. If it uses `**kwargs`, it silently absorbs extras (no bug). If not, it's a TypeError.

**If confirmed:** Add `is_cancelled=None` parameter or remove from call site.

---

### V.3 Main Window CloseEvent Thread Contract Mismatch
**Files:** `src/ui/main_window.py`, `src/ui/views/pipeline_view.py`

**Claim (Codex):** `closeEvent()` calls `stop()` / `_thread_ref` on QThread objects that don't have those methods.

**Verify:** Read `closeEvent()` and check what methods it actually calls on the worker.

**If confirmed:** Expose `request_stop()` on `PipelineView` with bounded worker teardown. Call it from `closeEvent()`.

---

### V.4 Pipeline Step Count Drift (Core vs UI)
**Files:** `src/analytics/pipeline.py`, `src/ui/views/pipeline_view.py`

**Claim (both):** Core has 10 steps, UI tracks 6.

**Verify:** Count steps in both files.

**If confirmed:** Create shared `PIPELINE_STEPS` constant imported by both. Also sync `overnight.py` if needed.

---

### V.5 Odds Force Flag Propagation
**Files:** `src/data/sync_service.py`, `src/data/odds_sync.py`

**Claim (from STATIC_ERROR_AUDIT.md):** `sync_historical_odds(..., force=True)` doesn't pass force to `backfill_odds()`.

**Verify:** Trace the `force` parameter through the call chain.

**If confirmed:** Thread `force` argument end-to-end. Add regression test.

---

## Phase 1: Security Baseline + Install/Test Foundation (Days 3-5)

*These can be worked in parallel: security hardening on the web layer, infrastructure on the project layer.*

### Security Track

**1.1** Add security response headers (`X-Frame-Options`, `X-Content-Type-Options`, CSP baseline) via `@app.after_request` in `src/web/app.py`.

**1.2** Stop exposing raw exception messages to users. Replace `error = str(e)` with generic messages at route boundaries. Log full exception server-side.

**1.3** Add CSRF protection to POST endpoints (`/api/predict`, `/api/sync`, `/api/shutdown`). Use `flask-wtf` or manual token validation.

**1.4** Remove `force=True` from `request.get_json()`. Require proper `Content-Type` headers.

**1.5** Add custom 404/500 error page templates. Register `@app.errorhandler()` decorators.

**1.6** Stabilize secret key: `app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(24)`.

**1.7** Add rate limiting on `/api/sync` (one at a time) and protect/gate `/api/shutdown`.

**1.8** Validate route parameters — date format regex on `/matchup/<home_abbr>/<away_abbr>/<date>`.

**1.9** Replace inline `onclick="GC.select('${g.espn_id}')"` with `data-` attributes + event listeners in `gamecast.html`.

### Infrastructure Track

**1.10** Complete `requirements.txt` with all actual dependencies: PySide6, optuna, rich, nba_api, cloudscraper, websocket-client, scipy. Consider `pyproject.toml` with optional dependency groups.

**1.11** Create `tests/conftest.py` with in-memory SQLite fixture. Write initial tests for:
- `prediction.py` core logic
- `db.py` read/write/lock + the new `execute_returning_id()`
- `elo.py` mathematical correctness
- `injury_monitor.py` diff logic
- `notification service` concurrent inserts
- Cache invalidation import validity

**1.12** Add `pyproject.toml` with ruff config (`line-length = 120`, `target-version = "py310"`).

**1.13** Add GitHub Actions CI workflow: checkout → setup Python → install deps → `ruff check src/` → `pytest tests/ -v`.

**Accept (Phase 1):**
- All security headers present in responses.
- No raw Python exceptions visible to web users.
- `pip install -r requirements.txt` installs everything on a clean env.
- `pytest tests/` passes with >0 tests.
- CI pipeline runs on push.

---

## Phase 2: Data/Model Correctness + Cache Architecture (Week 2)

### 2.1 Fix Stale-Read Architecture
**Files:** `src/database/db.py`, `src/ui/workers.py`, `overnight.py`

`thread_local_db()` creates isolated snapshots that don't see concurrent writes. Use the shared DB instance for write-heavy flows, or refresh thread-local connections after write operations.

### 2.2 Fix Historical Season Leakage
**Files:** `src/analytics/stats_engine.py`, `src/analytics/prediction.py`, `src/analytics/elo.py`

Feature builders call `get_season()` (current) instead of the game's actual season. Thread explicit season through all feature-building functions. Add assertion checks.

### 2.3 Unify Optimizer vs Prediction Thresholds
**Files:** `src/analytics/optimizer.py`, `src/analytics/prediction.py`

Optimizer: `> 0.5 / < -0.5`. Live prediction: `> 0 / < 0`. Create shared constant, use in both.

### 2.4 Build Central Cache Invalidation Registry
**Files:** `src/analytics/cache_registry.py` (NEW)

Named invalidators triggered by events: `post_sync`, `post_odds_sync`, `post_nuke`, `post_weight_save`. Debug endpoint to print cache states and last invalidation timestamps.

### 2.5 Add Source Fingerprinting to Cache Keys
**Files:** `src/analytics/prediction.py`, `src/analytics/backtester.py`

Context cache uses only `game_count` — data corrections with unchanged count stay stale. Add max-update-timestamp + checksum to cache metadata. Rebuild on mismatch. Preserve old files for rollback.

### 2.6 Add WHERE Clause to Bulk player_stats Load
**Files:** `src/analytics/prediction.py`

Precompute loads ALL player stats for ALL seasons with no filter. Add `WHERE season = ?` or date range filter.

### 2.7 Bound Module-Level Caches
**Files:** `src/analytics/prediction.py`, `src/analytics/stats_engine.py`, `src/ui/views/gamecast_view.py`

Add maxsize/TTL to: `_mem_pc_cache`, `_mem_ctx_cache`, `_splits_cache`, `_streak_cache`, `_fatigue_cache`, `_espn_headshot_data`, `_game_data_cache`.

### 2.8 Add Deterministic Optimizer Mode
**Files:** `src/analytics/optimizer.py`

Optional fixed seed for reproducibility. Set `seed` in CMA-ES sampler. Persist run metadata: seed, data fingerprint, config snapshot.

### 2.9 Verify team_id Backfill in player_stats
**Files:** `src/database/migrations.py`, queries throughout

Column was added in commit `7b3e58b`. Verify backfill is complete for all historical data. Ensure all queries use `player_stats.team_id` not `JOIN players`.

**Accept (Phase 2):**
- Backtest metrics stable after stale-read and season-leakage fixes.
- Cache invalidation verifiably triggers on every sync/nuke/save event.
- `player_stats` precompute load time measurably reduced.

---

## Phase 3: Code Quality + Reliability (Week 2-3)

### 3.1 Consolidate Duplicated Utilities
Move `_safe_float_setting()`, `_safe_int_setting()`, `_safe_bool_setting()` from 3+ files to `src/utils/settings_helpers.py`.

### 3.2 Normalize HTTP Client / Backoff
Create shared HTTP wrapper with: timeout, retry policy, jittered backoff, typed error classes. Replace per-file retry implementations in `nba_fetcher.py`, `gamecast.py`, `odds_sync.py`, `injury_scraper.py`.

### 3.3 Replace Broad Exception Handlers
Rules: broad catch only at process boundaries. Inside domain logic: catch specific exceptions. At minimum: `logger.debug("suppressed", exc_info=True)` on every catch.

### 3.4 Fix Prediction Cache Race Condition
**File:** `src/analytics/prediction.py`

Return a copy under the lock, or hold lock through the caller's usage scope.

### 3.5 Thread Lifecycle Hygiene
- Explicit `shutdown()`/`closeEvent()` handlers for views with workers/timers.
- Join/quit all active threads on teardown.
- WebSocket worker: bounded stop wait.
- Tab fade animation: reuse effects instead of creating new ones.
- Notification panel: `setParent(None)` before `deleteLater()`.

### 3.6 Fix Odds Backfill Aggressiveness
**File:** `src/data/odds_sync.py`

Change from date-level `NOT IN` to game-level `(game_date, home_team_id, away_team_id)` completeness check.

### 3.7 Remove Orphaned TODOs and Dead Code
Three identical TODO comments reference `prediction_quality` module that doesn't exist. Remove them and any associated dead code paths.

### 3.8 Add `.env` Support
Add `python-dotenv` to `src/config.py` with env var overrides for sensitive settings.

### 3.9 Save-Gate Observability
Persist save-gate reason/details per optimization pass. Expose in pipeline UI and overnight console. Alert on repeated no-save passes.

---

## Phase 4: UX + Features (Week 3+)

*Only after Phases 0-3 are stable and backtests are green.*

### UX Improvements
| Item | File Area | Effort |
|------|-----------|--------|
| Keyboard navigation (Alt+1-5 for tabs, Ctrl+R refresh) | `src/ui/` | 3h |
| Colorblind accessibility (shapes/icons alongside colors) | `src/ui/widgets/` | 2h |
| Responsive window sizing (use QScreen instead of 1200x800) | `src/ui/main_window.py` | 2h |
| User-friendly error messages + retry buttons | `src/ui/views/` | 2h |
| High-DPI font scaling | `src/ui/theme.py` | 2h |
| Static file caching headers | `src/web/app.py` | 30m |

### Feature Enhancements (by value/effort)
| # | Feature | Effort |
|---|---------|--------|
| 1 | CSV export for predictions/backtest results | 2h |
| 2 | Auto-backtest after optimization (wire existing pipeline steps) | 2h |
| 3 | Referee impact overlay in matchup view (data already scraped) | 3h |
| 4 | Prediction confidence intervals (+/- range on spreads) | 4h |
| 5 | Historical trend sparklines (last 10 games in matchup view) | 4h |
| 6 | Head-to-head matchup history (last 5-10 meetings) | 3h |
| 7 | Backtest confidence intervals + segmentation by line/rest/injury | 4h |
| 8 | Model explanation waterfall chart (feature weight * value) | 6h |
| 9 | Virtual bet tracker / P&L chart | 4h |
| 10 | Prediction leaderboard / calendar history | 4h |
| 11 | Push notifications for injury alerts (desktop tray or webhook) | 3h |
| 12 | Typed settings schema with pydantic validation | 4h |

---

## Stability Gates (Do Not Skip)

| Gate | When | How |
|------|------|-----|
| Backtest comparison | After every phase | Compare winner%, upset accuracy, ROI, spread MAE |
| Concurrent insert test | After 0.2 | Multiple threads creating notifications simultaneously |
| Rapid interaction test | After 0.3 | Fast game switching in web gamecast |
| Cache freshness test | After 0.5 | Sync/nuke then verify all caches are cold |
| Clean install test | After 1.10 | `pip install -r requirements.txt` on fresh venv |
| CI green gate | After 1.13 | All tests pass, ruff clean |
| Season boundary test | After 2.2 | Historical features at season edges use correct season |

---

## Complete File Change Manifest

| File | Phase | What Changes |
|------|-------|-------------|
| `src/notifications/injury_monitor.py` | 0.1 | Use sync_injuries() before diff |
| `src/notifications/service.py` | 0.2 | Consume insert ID from write call |
| `src/database/db.py` | 0.2 | Add execute_returning_id() helper |
| `src/web/templates/gamecast.html` | 0.3, 1.9 | Request sequencing + replace inline onclick |
| `src/ui/views/gamecast_view.py` | 0.4 | Clear _pending_headshots in finally |
| `src/data/sync_service.py` | 0.5 | Fix dead invalidation imports; independent try blocks |
| `src/data/odds_sync.py` | 0.5, V.5, 3.6 | Fix invalidation; thread force flag; game-level backfill |
| `src/analytics/prediction.py` | 0.5, 2.2, 2.5, 2.6, 3.4 | Fix invalidation; season threading; cache fingerprint; WHERE clause; race fix |
| `src/analytics/backtester.py` | 0.5, 2.5 | Fix invalidation; cache fingerprint |
| `src/analytics/stats_engine.py` | 0.6, 2.2 | Season filter on schedule spots; season threading |
| `web.py` | 0.7 | Reloader-safe bootstrap |
| `src/bootstrap.py` | 0.7 | Support reloader-safe gating |
| `src/analytics/score_calibration.py` | V.1 | Gate or implement missing functions |
| `src/analytics/pipeline.py` | V.2, V.4, 2.3 | Fix param mismatch; shared step constant; threshold |
| `src/ui/main_window.py` | V.3, 3.5 | Fix closeEvent; thread lifecycle |
| `src/ui/views/pipeline_view.py` | V.3, V.4 | Expose request_stop(); shared step constant |
| `src/web/app.py` | 1.1-1.8 | Security headers, CSRF, error handling, validation, rate limit, secret key, error pages |
| `requirements.txt` | 1.10 | Add all missing dependencies |
| `pyproject.toml` | 1.12 | NEW — ruff config + optional dep groups |
| `tests/conftest.py` | 1.11 | NEW — test fixtures |
| `tests/test_*.py` | 1.11 | NEW — initial test suite |
| `.github/workflows/ci.yml` | 1.13 | NEW — CI pipeline |
| `src/analytics/elo.py` | 2.2 | Season-aware Elo lookups |
| `src/analytics/cache_registry.py` | 2.4 | NEW — central invalidation registry |
| `src/analytics/optimizer.py` | 2.3, 2.8, 3.1 | Shared threshold; seed mode; consolidate helpers |
| `src/data/gamecast.py` | 0.5 | Add invalidate_actionnetwork_cache() |
| `src/utils/settings_helpers.py` | 3.1 | NEW — consolidated _safe_*_setting() |
| `src/data/injury_scraper.py` | 2.7 | Add lock to cache |
| `src/ui/notification_widget.py` | 3.5 | Fix deleteLater() leak |
| `src/config.py` | 3.8 | Add .env support |

**Total: ~30 files modified, ~6 files created, across 4 phases.**

---

*Final synthesis from two independent AI reviews + two unified plans. Optimized for shipping correct software fast.*
