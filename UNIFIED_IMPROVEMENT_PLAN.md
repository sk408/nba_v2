# NBA V2 - Unified Improvement Plan
### Merged from Claude Opus 4.6 + Codex Reviews | March 9, 2026

---

## How This Plan Was Built

Two independent AI reviewers audited the full codebase. Their strengths were complementary:

- **Codex** excelled at finding **confirmed runtime bugs** — subtle concurrency races, lifecycle management issues, and broken function references that cause silent failures in production. Their P0 list is surgical.
- **Opus** excelled at **breadth** — security hardening, missing infrastructure, performance bottlenecks, UI/UX accessibility, and feature ideas. Their coverage touched every layer of the stack.

This document merges both into a single actionable plan, resolving overlaps, and ordering work by actual risk.

**Guiding principle from Codex that I agree with:** *"Keep optimizer formula/weights unchanged in early phases; only fix data correctness and runtime behavior. Run backtest before/after each phase to verify metric stability."*

---

## Priority Rubric

- **P0**: Confirmed runtime bug or crash — changes outputs or breaks workflows *now*
- **P1**: Security, reliability, or data integrity risk — will bite you under real conditions
- **P2**: Performance, code quality, infrastructure — makes everything else easier
- **P3**: Polish, features, nice-to-haves — value-add work after the foundation is solid

---

## Phase 0: Confirmed Runtime Bugs (Days 1-2)

These are verified broken behaviors. Fix first, add regression tests for each.

### 0.1 — Score Calibration Crashes on Undefined Functions
**Source:** Opus BUG-01 | **Files:** `src/analytics/score_calibration.py`

`_fit_mode_payload()` calls `_blend_near_spread()` and `_apply_team_point_ranges_arrays()` — neither function exists anywhere in the codebase. Also references undefined settings (`near_spread_identity_band`, `near_spread_deadband`, `near_spread_raw_weight`) and undefined variable `team_point_ranges`.

**Action:** Either implement the missing functions or gate the code paths behind `score_calibration_near_spread_enabled` (default `False`) so the feature is disabled until ready. Do NOT leave crashing code reachable.

**Test:** Call `_fit_mode_payload()` with representative data; verify it completes without error.

---

### 0.2 — Injury Monitor Never Persists Scraped Changes to DB
**Source:** Codex P0-1 | **Files:** `src/notifications/injury_monitor.py`, `src/data/injury_scraper.py`

`InjuryMonitor._check_changes()` calls `scrape_all_injuries()` then diffs against DB rows, but the scrape function does *not* persist updates. The monitor compares stale DB against the same stale DB on every poll — notifications are silently broken unless another sync path updates injuries first.

**Action:**
- Replace the scrape-only call with `sync_injuries()` (or an explicit DB-apply step) *before* reading current state for diffing.
- Add a log line: `injury_monitor_db_updates=N` per poll cycle.
- Keep the fast-path where no changes skip notification emission.

**Test:** Mock a scrape payload with a changed player status. Verify the DB row mutates and a notification is emitted.

---

### 0.3 — Cache Invalidation Imports Reference Removed Functions
**Source:** Codex P0-2 | **Files:** `src/data/sync_service.py`, `src/analytics/prediction.py`, `src/analytics/backtester.py`

Invalidation imports reference functions that no longer exist: `invalidate_residual_cache`, `invalidate_tuning_cache`, `invalidate_elo_cache`, `invalidate_actual_results_cache`. Because they're in a single try block, one import failure prevents *all* other valid invalidations from running.

**Action:**
- Replace dead imports with the functions that actually exist:
  - `invalidate_precompute_cache()`
  - `invalidate_results_cache()`
  - `invalidate_backtest_cache()`
  - `invalidate_stats_caches()`
  - `invalidate_weight_cache()`
- Split invalidation into independent try blocks so one failure doesn't skip all.
- Add `invalidate_actionnetwork_cache()` in `src/data/gamecast.py` and call it from sync/odds paths.

**Test:** Smoke test that all invalidation function references are importable. Run sync/nuke and confirm all relevant caches are cold afterward.

---

### 0.4 — `last_insert_rowid()` Race in Notification Creation
**Source:** Codex P0-4 | **Files:** `src/notifications/service.py`, `src/database/db.py`

`create_notification()` inserts a row then separately reads `last_insert_rowid()` through the shared connection flow. Under concurrent writers, this can return the wrong row's ID.

**Action:**
- Modify `db.execute()` to return `cursor.lastrowid` from the same write call.
- Or add `execute_returning_id(sql, params)` helper that returns the ID atomically.
- Update notification service to consume the returned ID directly.

**Test:** Concurrent insert test with multiple threads; each emitted notification ID must match its inserted row.

---

### 0.5 — Main Window CloseEvent Calls Non-Existent Thread APIs
**Source:** Codex P0-3 | **Files:** `src/ui/main_window.py`, `src/ui/views/pipeline_view.py`

`MainWindow.closeEvent()` grabs `pipeline._current_worker` (a QThread) and calls `stop()` / `_thread_ref`, which are not valid QThread methods. The close path logs warnings and may not wait for graceful pipeline cancellation.

**Action:**
- Expose `request_stop()` on `PipelineView` that handles worker cancel + `quit()/wait()` on active threads.
- In `MainWindow.closeEvent()`, call `pipeline_view.request_stop()` with a bounded wait.
- Add optional stop hooks for `GamecastView` / `MatchupView` workers too.

**Test:** Close app while pipeline is running; verify no thread warnings and no hung process.

---

### 0.6 — Web Gamecast Stale-Response Race on Rapid Game Switch
**Source:** Codex P0-5 | **Files:** `src/web/templates/gamecast.html`

Async fetches can resolve out of order when the user switches games quickly. A stale response overwrites the newer selection. Interval polling also doesn't guard against overlapping in-flight requests.

**Action:**
- Add a `requestSeq` counter. Increment on each game selection. Ignore responses whose seq doesn't match current.
- Add `isFetching` guard — skip interval tick when a prior fetch is still running.
- Reset overlays/timers defensively on game switch.

**Test:** Rapidly switch games in browser; verify UI never shows a previously-selected game after the latest selection.

---

### 0.7 — Desktop Gamecast Headshot Pending Set Never Clears
**Source:** Codex P0-6 | **Files:** `src/ui/views/gamecast_view.py`

`_pending_headshots` is added to but never removed on completion or failure. Failed headshots never retry. Memory grows. Stale visual state.

**Action:**
- Remove URL from `_pending_headshots` in a `finally` block in the runnable.
- Add retry budget with backoff for transient failures.
- Optionally TTL failed fetches to avoid hammering the CDN.

**Test:** Force one failure then success; image should eventually appear.

---

### 0.8 — Pipeline Parameter Mismatch (seed_arenas)
**Source:** Opus BUG-02 | **Files:** `src/analytics/pipeline.py`

`run_seed_arenas()` doesn't accept `is_cancelled` but the pipeline passes it. Could cause `TypeError` at runtime.

**Action:** Add `is_cancelled=None` parameter to `run_seed_arenas()`, or remove it from the pipeline call site.

**Test:** Run the pipeline through the seed_arenas step; verify no TypeError.

---

### 0.9 — Schedule-Spot Query Lacks Season Filter
**Source:** Codex P0-7 | **Files:** `src/analytics/stats_engine.py` (`compute_schedule_spots`)

Next-game query uses `game_date > ? AND team_id = ?` without `season = ?`. Causes cross-season leakage near season boundaries in historical evaluation.

**Action:** Add season filter to the next-game query and related schedule lookups.

**Test:** With multi-season data present, verify final games of a season don't leak next-season data.

---

### 0.10 — Flask Debug Reloader Bootstraps Twice
**Source:** Codex P0-8 | **Files:** `web.py`, `src/bootstrap.py`

Flask debug/reloader executes module-level bootstrap in both parent and child process. Creates duplicate injury monitors and doubled network load in dev.

**Action:** Move bootstrap under `if __name__ == "__main__"` with reloader-safe check (`os.environ.get('WERKZEUG_RUN_MAIN') == 'true'`).

**Test:** Start Flask in debug mode; verify only one injury monitor instance runs.

---

## Phase 1: Security Hardening (Days 3-4)

*Note: Severity depends on deployment context. If this only runs on localhost, these are MEDIUM. If exposed to a network, they're CRITICAL. Implementing them regardless is cheap insurance.*

### 1.1 — Add Security Headers to Flask
**Source:** Opus SEC-02 | **File:** `src/web/app.py`

```python
@app.after_request
def set_security_headers(response):
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'"
    return response
```

---

### 1.2 — Stop Exposing Raw Exception Messages to Users
**Source:** Opus SEC-04 | **File:** `src/web/app.py`

Replace `error = str(e)` with generic user-friendly messages at every route boundary. Log the full exception server-side with `exc_info=True`.

---

### 1.3 — Add CSRF Protection to POST Endpoints
**Source:** Opus SEC-01 | **File:** `src/web/app.py`

Add `flask-wtf` CSRF protection, or at minimum validate a custom token for `/api/predict`, `/api/sync`, and especially `/api/shutdown`.

---

### 1.4 — Remove `force=True` from JSON Parsing
**Source:** Opus SEC-03 | **File:** `src/web/app.py`

Change `request.get_json(force=True, silent=True)` to `request.get_json(silent=True)`. Require proper `Content-Type: application/json`.

---

### 1.5 — Add Custom 404/500 Error Pages
**Source:** Opus SEC-05 | **File:** `src/web/app.py`

Register `@app.errorhandler(404)` and `@app.errorhandler(500)` with custom templates. Prevents Flask version fingerprinting.

---

### 1.6 — Stabilize Flask Secret Key
**Source:** Opus SEC-06 | **File:** `src/web/app.py`

```python
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.urandom(24)
```

---

### 1.7 — Add Rate Limiting to Dangerous Endpoints
**Source:** Opus SEC-07 | **File:** `src/web/app.py`

At minimum, add an in-memory guard on `/api/sync` (only one sync at a time) and `/api/shutdown` (require confirmation or auth). Consider `flask-limiter` for broader protection.

---

### 1.8 — Validate Route Parameters
**Source:** Opus SEC-08 | **File:** `src/web/app.py`

Validate `date` parameter format in `/matchup/<home_abbr>/<away_abbr>/<date>` with regex and `datetime.strptime`. Return 400 on invalid input.

---

### 1.9 — Fix Unescaped ESPN ID in onclick Handler
**Source:** Opus SEC-09 | **File:** `src/web/templates/gamecast.html`

Replace inline `onclick="GC.select('${g.espn_id}')"` with `data-game-id` attribute + event listener.

---

## Phase 2: Infrastructure & Data Integrity (Days 5-10)

### 2.1 — Complete `requirements.txt`
**Source:** Opus INFRA-02

Add ALL actual dependencies. Current file is missing PySide6, optuna, rich, nba_api, cloudscraper, websocket-client, scipy. Consider migrating to `pyproject.toml` with optional dependency groups (desktop, optimizer, cli, dev).

---

### 2.2 — Set Up pytest + Initial Test Suite
**Source:** Opus INFRA-01, Codex P2-4

Priority test targets (both reviewers agree):
1. `prediction.py` — core prediction logic, feature building
2. `db.py` — read/write/lock behavior, `execute_returning_id`
3. `elo.py` — mathematical correctness
4. `backtester.py` — walk-forward correctness
5. `injury_monitor.py` — diffing logic (Codex-specific)
6. `notification service` — concurrent insert IDs (Codex-specific)
7. `cache invalidation` — all import references valid (Codex-specific)
8. `sync_service.py` — freshness checks, incremental sync

Create `tests/conftest.py` with an in-memory SQLite fixture DB.

---

### 2.3 — Fix Stale-Read Architecture in Write-Heavy Flows
**Source:** Opus ARCH-01, also noted in `STATIC_ERROR_AUDIT.md`

`thread_local_db()` creates isolated snapshots that don't see concurrent writes. Use the shared DB instance for pipeline/sync/overnight flows, or refresh thread-local connections after write operations.

---

### 2.4 — Fix Historical Season Leakage in Feature Generation
**Source:** Opus ARCH-02, Codex P0-7 (related)

Feature builders call `get_season()` (current season) instead of the game's actual season. Elo lookup also ignores the season parameter. Thread the game's season through all feature-building functions. Add assertion checks.

---

### 2.5 — Unify Optimizer vs Prediction Thresholds
**Source:** Opus ARCH-04

Optimizer classifies winners at `> 0.5 / < -0.5` but live prediction uses `> 0 / < 0`. Create a shared constant and use it in both places.

---

### 2.6 — Sync Pipeline Step Count (Core vs UI)
**Source:** Opus ARCH-03

Core pipeline has 10 steps, UI tracks 6. Create a shared `PIPELINE_STEPS` constant that both `pipeline.py` and `pipeline_view.py` import.

---

### 2.7 — Build Central Cache Invalidation Registry
**Source:** Codex P1-B

Caches are scattered and easy to miss during sync. Create `src/analytics/cache_registry.py` with named invalidators triggered by events: `post_sync`, `post_odds_sync`, `post_nuke`, `post_weight_save`. Add debug endpoint/CLI to print cache states and last invalidation timestamps.

---

### 2.8 — Strengthen Cache Key Correctness
**Source:** Codex P1-C

Context cache keys use only `game_count` — data corrections with unchanged count stay stale. Add source fingerprint (max update timestamp + checksum) to cache metadata. Rebuild on fingerprint mismatch. Preserve old cache files for rollback during validation.

---

### 2.9 — Set Up GitHub Actions CI
**Source:** Opus INFRA-03

Minimal workflow: checkout → setup Python → install deps → `ruff check src/` → `pytest tests/ -v`. Gate merges on green CI.

---

## Phase 3: Performance & Code Quality (Week 2)

### 3.1 — Add WHERE Clause to Bulk player_stats Load
**Source:** Opus PERF-01 | **File:** `src/analytics/prediction.py`

The precompute query loads ALL player stats for ALL seasons with no filter. Add `WHERE season = ?` or `WHERE game_date >= ?`. This could be 100K+ rows.

---

### 3.2 — Bound All Module-Level Caches
**Source:** Opus PERF-02

Add `maxsize` or TTL eviction to: `_mem_pc_cache`, `_mem_ctx_cache`, `_splits_cache`, `_streak_cache`, `_fatigue_cache`, `_espn_headshot_data`, `_game_data_cache`. Use `functools.lru_cache` with bounded sizes where applicable.

---

### 3.3 — Normalize HTTP Client / Backoff Patterns
**Source:** Codex P1-E

Create a shared HTTP client wrapper with: timeout, retry policy, jittered backoff, typed error classes. Replace per-file retry implementations in `nba_fetcher.py`, `gamecast.py`, `odds_sync.py`, `injury_scraper.py`. Add centralized rate-limit metrics and failure counters.

---

### 3.4 — Consolidate Duplicated Utility Functions
**Source:** Opus QUAL-01

`_safe_float_setting()`, `_safe_int_setting()`, `_safe_bool_setting()` are copy-pasted across 3+ files. Move to `src/utils/settings_helpers.py`.

---

### 3.5 — Replace Broad Exception Handlers
**Source:** Opus QUAL-03, Codex P1-A

8+ locations use `except Exception: pass` without logging. Rules:
- Broad catch only at process boundaries (UI thread handler, route top-level)
- Inside domain logic: catch specific exceptions (`requests.Timeout`, `sqlite3.Error`, `ValueError`)
- At minimum: `logger.debug("suppressed", exc_info=True)` on every except

---

### 3.6 — Fix Prediction Cache Race Condition
**Source:** Opus BUG-03 | **File:** `src/analytics/prediction.py`

Lock is released before caller uses the returned cache reference. Return a copy under the lock, or use a context manager that holds the lock for the caller's scope.

---

### 3.7 — Add Linter Configuration
**Source:** Opus QUAL-06

```toml
# pyproject.toml
[tool.ruff]
line-length = 120
target-version = "py310"
```

---

### 3.8 — Thread Lifecycle Hygiene
**Source:** Codex P1-D, Opus BUG-04/BUG-05

- Add explicit `shutdown()`/`closeEvent()` handlers for all views with worker pools and timers.
- Join/quit all active worker threads on teardown.
- Ensure WebSocket worker has bounded stop wait.
- Fix tab fade animation to reuse effects instead of creating new ones.
- Fix notification panel `deleteLater()` leak with `setParent(None)` before deletion.

---

### 3.9 — Add `.env` Support for Secrets
**Source:** Opus INFRA-04

Add `python-dotenv` support in `src/config.py` with env var overrides for sensitive settings. Future-proofs for API keys, premium data sources, etc.

---

### 3.10 — Add Deterministic Optimizer Mode
**Source:** Codex P1-A (Prediction/Optimization), Opus QUAL-07

Optional fixed seed mode for reproducibility. Persist run metadata: seed, data fingerprint, config snapshot. Set `seed` parameter in CMA-ES sampler.

---

## Phase 4: Architecture & Maintainability (Week 3)

### 4.1 — Shared Gamecast Domain Service
**Source:** Codex P2-1

Parsing and transformation logic is duplicated between `src/ui/views/gamecast_view.py` and `src/web/app.py` / template JS. Introduce `src/gamecast/service.py` returning a normalized DTO. Desktop/web render same schema.

---

### 4.2 — Typed Settings Schema + Validation
**Source:** Codex P2-2, Opus QUAL-02

Add a pydantic or dataclass-based schema for all 95+ settings. Validate on load (not just selective keys on set). Emit warnings for unknown keys. This also absorbs the magic-numbers cleanup — move analytical constants to validated config fields.

---

### 4.3 — Fix Odds Backfill Aggressiveness
**Source:** Opus ARCH-07

Date-level `NOT IN` query skips ALL games on a date if ANY game has odds. Change to game-level `(game_date, home_team_id, away_team_id)` completeness check.

---

### 4.4 — Verify team_id Backfill in player_stats
**Source:** Opus ARCH-06

The `team_id` column was recently added (commit `7b3e58b`). Verify the backfill is complete for all historical data and that all queries now use `player_stats.team_id` instead of joining through `players.team_id`.

---

### 4.5 — Remove Orphaned TODOs and Dead References
**Source:** Opus QUAL-04

Three identical TODO comments reference `prediction_quality` module that doesn't exist. Either implement the module or remove the TODOs and dead code paths.

---

### 4.6 — Save-Gate Observability
**Source:** Codex P1-B (Prediction/Optimization)

Persist save-gate reason/details per optimization pass to a structured artifact. Expose in pipeline UI and overnight console table. Add alert when repeated no-save passes exceed threshold.

---

## Phase 5: UI/UX Polish (Week 4+)

### 5.1 — Keyboard Navigation
**Source:** Opus UX-02

Add `QShortcut` bindings: `Alt+1` through `Alt+5` for tabs, `Ctrl+R` for refresh/sync.

### 5.2 — Colorblind Accessibility
**Source:** Opus UX-03

Add shape/icon/text indicators alongside color for: dog pick badges, direction indicators, court widget shot chart.

### 5.3 — Responsive Window Sizing
**Source:** Opus UX-01

Calculate minimum from `QScreen.availableGeometry()` instead of hardcoding 1200x800.

### 5.4 — User-Friendly Error Messages
**Source:** Opus UX-06

Map common exceptions to human-readable messages in the desktop UI status bar. Add a "Retry" action.

### 5.5 — Static File Caching Headers
**Source:** Opus PERF-04

`app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000`

### 5.6 — High-DPI Font Scaling
**Source:** Opus UX-04

Scale fonts based on `QScreen` logical DPI instead of hardcoded pixel sizes.

---

## Phase 6: Feature Enhancements (Ongoing)

Ordered by value-to-effort ratio:

| # | Feature | Effort | Why |
|---|---------|--------|-----|
| 1 | **Prediction Confidence Intervals** | 4h | Optimizer already has variance data; show "+/- X" on spreads |
| 2 | **CSV Export** for predictions/backtest results | 2h | Most-requested "table stakes" feature |
| 3 | **Referee Impact Overlay** in matchup view | 3h | Data already scraped; just needs surfacing |
| 4 | **Auto-Backtest After Optimization** | 2h | Pipeline already has both steps; wire them together |
| 5 | **Historical Trend Sparklines** in matchup view | 4h | Last-10-game W/L + scoring trend |
| 6 | **Head-to-Head Matchup History** | 3h | Last 5-10 meetings with model accuracy |
| 7 | **Model Explanation Waterfall** | 6h | Show feature weight * value breakdown per game |
| 8 | **Virtual Bet Tracker / P&L Chart** | 4h | Surface backtester ML ROI as running chart |
| 9 | **Push Notifications for Injuries** | 3h | `QSystemTrayIcon.showMessage()` or webhook |
| 10 | **Backtest Confidence Intervals + Segmentation** | 4h | CI for winner%/ROI, segment by line/rest/injury (Codex) |
| 11 | **Dark/Light Theme Toggle** | 4h | Proper light theme, not just OLED variant |
| 12 | **Prediction Leaderboard / Calendar History** | 4h | Daily picks with hit rates, streaks |

---

## Rollout Guardrails (Both Reviewers Agree)

1. **Run backtest before AND after each phase.** Compare: winner%, upset accuracy/rate, comp-dog and long-dog diagnostics, ROI, spread MAE.
2. **Do NOT change optimizer formula/weights in Phase 0.** Only fix data correctness and runtime behavior.
3. **Preserve old cache files** until the new cache key logic is validated. One-click rollback.
4. **Add regression tests alongside every fix** — not after, alongside.
5. **Commit after each numbered item**, not after each phase. Small, reviewable diffs.

---

## Success Criteria

Phase 0:
- [ ] Score calibration completes without crash (or is safely disabled)
- [ ] Injury notifications reflect real scrape changes within one poll interval
- [ ] Sync/nuke reliably invalidates all relevant caches
- [ ] App close during pipeline is graceful and warning-free
- [ ] Notification IDs are deterministic under concurrent writes
- [ ] Web gamecast never renders stale game after rapid switching
- [ ] Headshot pending set is cleaned up on completion/failure
- [ ] Pipeline seed_arenas step runs without TypeError
- [ ] No cross-season leakage in schedule-spot features
- [ ] Flask debug mode starts only one injury monitor

Phase 1:
- [ ] All security headers present in Flask responses
- [ ] No raw Python exceptions visible to web users
- [ ] CSRF tokens validated on all POST endpoints
- [ ] Route parameters validated before processing

Phase 2:
- [ ] `pip install -r requirements.txt` installs everything needed
- [ ] `pytest tests/` passes with >0 tests
- [ ] CI pipeline runs on every push
- [ ] Backtest metrics remain stable after stale-read and season-leakage fixes

---

## File-Level Change Summary (All Phases)

| File | Changes |
|------|---------|
| `src/analytics/score_calibration.py` | Gate undefined function calls behind feature flag |
| `src/analytics/pipeline.py` | Fix seed_arenas parameter; shared PIPELINE_STEPS constant |
| `src/analytics/prediction.py` | Fix cache race; add WHERE to bulk load; fix season threading |
| `src/analytics/stats_engine.py` | Add season filter to schedule-spot query |
| `src/analytics/optimizer.py` | Add deterministic seed mode; consolidate _safe_* helpers |
| `src/analytics/backtester.py` | Fix dead invalidation imports; consolidate _safe_* helpers |
| `src/analytics/cache_registry.py` | **NEW** — central invalidation registry |
| `src/notifications/injury_monitor.py` | Use sync_injuries() instead of scrape-only call |
| `src/notifications/service.py` | Consume insert ID from write call directly |
| `src/database/db.py` | Add execute_returning_id() helper |
| `src/data/sync_service.py` | Fix dead invalidation imports; independent try blocks |
| `src/data/gamecast.py` | Add invalidate_actionnetwork_cache() |
| `src/ui/main_window.py` | Fix closeEvent to use view-level stop API |
| `src/ui/views/pipeline_view.py` | Expose request_stop() with bounded teardown |
| `src/ui/views/gamecast_view.py` | Clear _pending_headshots in finally; add retry |
| `src/ui/notification_widget.py` | Fix deleteLater() leak |
| `src/web/app.py` | Security headers, CSRF, error handling, rate limiting, input validation, secret key, error pages |
| `src/web/templates/gamecast.html` | Request sequencing; replace inline onclick |
| `src/config.py` | Add .env support; typed schema validation |
| `src/utils/settings_helpers.py` | **NEW** — consolidated _safe_*_setting() functions |
| `web.py` | Reloader-safe bootstrap |
| `requirements.txt` | Add all missing dependencies |
| `pyproject.toml` | **NEW** — ruff config, optional dep groups |
| `tests/conftest.py` | **NEW** — test fixtures |
| `tests/test_prediction.py` | **NEW** |
| `tests/test_db.py` | **NEW** |
| `tests/test_elo.py` | **NEW** |
| `tests/test_injury_monitor.py` | **NEW** |
| `tests/test_notifications.py` | **NEW** |
| `.github/workflows/ci.yml` | **NEW** — CI pipeline |

---

*Synthesized from independent reviews by Claude Opus 4.6 and Codex. 42 action items across 6 phases, ordered by confirmed risk and implementation dependency.*
