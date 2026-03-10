# NBA V2 Deep Review and Improvement Plan (Codex)

Date: 2026-03-09  
Scope: `src/` plus entrypoints (`desktop.py`, `web.py`, `overnight.py`) and operational flow (data sync, prediction, optimization, web/desktop UI).

---

## What This Document Is

This is an implementation-ready review plan with:
- High-confidence bug fixes (first priority)
- Reliability, performance, and modeling improvements
- A phased roadmap so changes can ship safely
- Test strategy tied directly to the risky areas

The goal is not generic cleanup; it is to reduce bad predictions, stale data, UI/runtime regressions, and hard-to-debug production failures.

---

## Priority Rubric

- **P0**: correctness or data integrity risk; can change outputs or break core workflows now
- **P1**: reliability/performance risk; likely to degrade UX, freshness, or throughput
- **P2**: maintainability/quality; improves speed of future work and lowers regression risk

---

## P0: High-Confidence Bug Fixes

### 1) Injury monitor is not applying scraped changes to DB
**Files:** `src/notifications/injury_monitor.py`, `src/data/injury_scraper.py`  
**Issue:** `InjuryMonitor._check_changes()` calls `scrape_all_injuries()` then diffs against DB rows, but `scrape_all_injuries()` does not persist updates. The monitor compares old DB vs same DB in many runs.  
**Impact:** injury notifications can silently miss real-world status changes unless another sync path updates injuries first.  
**Plan:**
- Replace monitor scrape call with `sync_injuries()` (or explicit apply step) before reading current state.
- Add a metric/log line: `injury_monitor_db_updates` per poll.
- Keep a fast-path where no changes skip notification emission.
**Validation:**
- Unit test: mock scrape payload changed status; verify DB row mutates and notification emitted.
- Integration test: run monitor loop once with seeded injuries and changed feed.

---

### 2) Cache invalidation in sync/nuke paths references removed functions
**Files:** `src/data/sync_service.py`, `src/analytics/prediction.py`, `src/analytics/backtester.py`  
**Issue:** invalidation imports reference missing functions (`invalidate_residual_cache`, `invalidate_tuning_cache`, `invalidate_elo_cache`, `invalidate_actual_results_cache`). In one block, import failure prevents other valid invalidations from running.  
**Impact:** stale caches after sync/nuke; model/backtest can read outdated derived state.  
**Plan:**
- Replace invalidation imports with existing APIs:
  - `invalidate_precompute_cache()`
  - `invalidate_results_cache()`
  - `invalidate_backtest_cache()`
  - `invalidate_stats_caches()`
  - `invalidate_weight_cache()`
- Split invalidation into independent try blocks per module so one failure does not skip all.
- Add `invalidate_actionnetwork_cache()` in `src/data/gamecast.py` and call it from sync/odds paths.
**Validation:**
- Test sync/nuke command updates data then confirms all relevant caches are cold.
- Add smoke test asserting invalidation function references are importable.

---

### 3) Main-window shutdown logic is calling thread APIs that do not exist
**Files:** `src/ui/main_window.py`, `src/ui/views/pipeline_view.py`  
**Issue:** `MainWindow.closeEvent()` grabs `pipeline._current_worker` (currently a `QThread`) and calls `stop()` / `_thread_ref`, which are not valid for this object.  
**Impact:** close path logs warnings and may not wait for graceful pipeline cancel properly.  
**Plan:**
- Normalize pipeline view shutdown contract:
  - expose `request_stop()` on `PipelineView`
  - `request_stop()` calls worker cancel + `request_cancel()`, then `quit()/wait()` on active threads
- In `MainWindow.closeEvent()`, call `pipeline.request_stop()` and bounded wait.
- Also add optional stop hooks for `GamecastView`/`MatchupView` workers.
**Validation:**
- UI test: close app while pipeline is active; verify no thread warnings and no hung process.

---

### 4) `last_insert_rowid()` usage is racy in notification creation
**Files:** `src/notifications/service.py`, `src/database/db.py`  
**Issue:** `create_notification()` inserts then separately reads `last_insert_rowid()` through shared connection flow. In concurrent writers, this can return wrong row id.  
**Impact:** wrong notification ID sent to listeners/push payloads; subtle inconsistencies in read/mark operations.  
**Plan:**
- Change DB write API to return deterministic insert id from the same write call:
  - either return `cursor.lastrowid` from `db.execute(...)`
  - or add `execute_returning_id(...)` helper
- Update notification service to consume returned ID directly.
**Validation:**
- Concurrent insert test with multiple threads: each emitted notification id matches inserted row.

---

### 5) Web gamecast has request-order race and possible polling overlap
**Files:** `src/web/templates/gamecast.html`  
**Issue:** async fetches can resolve out of order when user switches games quickly; stale response can overwrite newer selection. Interval polling also does not guard against overlapping in-flight requests.  
**Impact:** intermittent wrong game rendered, flicker, and excess API load.  
**Plan:**
- Add request token/versioning (`requestSeq`) and ignore stale responses.
- Add in-flight guard (`isFetching`) and skip interval tick when prior fetch is still running.
- Reset overlays/timers on game switch defensively.
**Validation:**
- Browser test: rapidly switch games; verify UI never displays previous game after latest selection.

---

### 6) Desktop gamecast headshot queue never clears pending URL set
**Files:** `src/ui/views/gamecast_view.py`  
**Issue:** `_pending_headshots` is added to but not removed on completion/failure.  
**Impact:** failed headshots may never retry; memory grows; stale visual state.  
**Plan:**
- Ensure URL removed from pending set in runnable `finally`.
- Add retry budget/backoff for transient failures.
- Optionally TTL failed fetches to avoid hammering.
**Validation:**
- Test with one forced failure then success; image should eventually appear.

---

### 7) Schedule-spot lookahead query lacks season filter
**Files:** `src/analytics/stats_engine.py` (`compute_schedule_spots`)  
**Issue:** next-game query uses `game_date > ? AND team_id = ?` without `season = ?`.  
**Impact:** cross-season leakage near season boundaries in historical evaluation/features.  
**Plan:**
- Add season filter to next-game query and related schedule lookups.
- Add regression tests around final games of a season with next-season data present.

---

### 8) Debug web startup can bootstrap background services twice
**Files:** `web.py`, `src/bootstrap.py`  
**Issue:** with Flask debug/reloader, module-level bootstrap can execute in both parent and reloader child process.  
**Impact:** duplicate injury monitor/network load in dev, confusing logs.  
**Plan:**
- Move bootstrap under explicit main guard with reloader-safe check.
- For development, use app factory pattern and gate monitor startup by environment flag.

---

## P1: Reliability and Data Freshness

### A) Replace broad exception swallowing in critical paths
**Targets:** `src/data/*`, `src/web/app.py`, `src/ui/views/*`, `src/analytics/*`  
**Plan:**
- Keep broad catch only at process boundaries (UI thread handler, route top-level).
- Inside domain logic: catch specific exceptions (`requests.Timeout`, `sqlite3.Error`, `ValueError`) and emit structured context.
- Create helper `log_exception_context(event, **fields)` for consistent logging.

### B) Introduce a central cache invalidation registry
**Problem:** cache responsibilities are spread and easy to miss during sync.  
**Plan:**
- Create `src/analytics/cache_registry.py` with named invalidators.
- Trigger by events: `post_sync`, `post_odds_sync`, `post_nuke`, `post_weight_save`.
- Add debug endpoint/CLI to print cache states and last invalidation timestamp.

### C) Strengthen cache key correctness for historical recomputation
**Files:** `src/analytics/prediction.py`, `src/analytics/backtester.py`  
**Observations:**
- Context cache uses only `game_count`; data corrections with unchanged count can stay stale.
- Precompute game key (`home_away_date`) cannot detect corrected source stats for same game tuple.
**Plan:**
- Add source fingerprint to cache metadata:
  - max update timestamp and checksum over relevant tables
  - include schema version + source fingerprint in cache files
- Rebuild on fingerprint mismatch.

### D) Thread lifecycle hygiene
**Plan:**
- Add explicit `shutdown()`/`closeEvent()` handlers for views with worker pools and timers.
- Join/quit all active worker threads on teardown.
- Ensure websocket worker has bounded stop wait.

### E) API backoff and resilience normalization
**Files:** `src/data/nba_fetcher.py`, `src/data/gamecast.py`, `src/data/odds_sync.py`, `src/data/injury_scraper.py`  
**Plan:**
- Shared HTTP client wrapper: timeout, retry policy, jittered backoff, error classes.
- Centralized rate-limit metrics and failure counters.
- Add per-source cooldown to avoid repeated failures thrashing logs.

---

## P1: Prediction/Optimization Integrity

### A) Deterministic optimization mode
**Files:** `src/analytics/optimizer.py`, `src/analytics/pipeline.py`  
**Plan:**
- Optional fixed seed mode for reproducibility during validation/benchmarking.
- Persist run metadata: seed, data fingerprint, config snapshot.

### B) Save-gate observability
**Plan:**
- Persist save-gate reason/details per pass to structured artifact.
- Expose in pipeline UI and overnight console table.
- Add alert when repeated no-save passes exceed threshold.

### C) Backtest confidence and ROI diagnostics
**Files:** `src/analytics/backtester.py`, `src/analytics/optimizer.py`  
**Plan:**
- Add confidence intervals for winner%/ROI.
- Segment metrics by line buckets, rest disadvantage, injury load, and underdog classes.
- Track calibration drift over time windows.

---

## P2: Architecture and Maintainability

### 1) Shared gamecast domain service for desktop + web
**Problem:** parsing and transformation logic duplicated between `src/ui/views/gamecast_view.py` and `src/web/app.py`/template JS.  
**Plan:**
- Introduce `src/gamecast/service.py` returning a normalized DTO.
- Desktop/web layers render same schema.
- Keep UI-only formatting in client/view layer.

### 2) Tighten settings schema and validation
**Files:** `src/config.py`  
**Plan:**
- Add typed schema (pydantic/dataclass validator).
- Validate all values on load, not only selective keys on set.
- Emit warning for unknown keys and preserve migrations.

### 3) Error budgets and SLO-ish telemetry
**Plan:**
- Track key rates: sync success, API error %, prediction latency, UI worker failures.
- Add lightweight local dashboard/log summary every N minutes.

### 4) Test harness expansion
**Plan:**
- Unit tests: feature builders, cache invalidation, injury monitor diffing, odds mapping.
- Integration tests: one full sync + one prediction + one backtest with fixture DB.
- Concurrency tests: notification insert IDs, worker shutdown.
- Browser/UI smoke tests for web gamecast and desktop tab transitions.

---

## Suggested Execution Roadmap

## Phase 0 (Day 1-2): Safety Patches
- Fix P0 items 1-5 first (injury monitor, invalidation, closeEvent contract, notification ID race, web gamecast request token).
- Add targeted regression tests for each.
- Ship behind minimal risk toggles where needed.

## Phase 1 (Day 3-5): Freshness and Threading
- Implement cache registry + source fingerprint keys.
- Patch gamecast headshot pending lifecycle.
- Add season filter fix in schedule spots.
- Harden thread shutdown paths.

## Phase 2 (Week 2): Reliability Platform
- Introduce shared HTTP wrapper/backoff.
- Normalize exception taxonomy and structured logs.
- Add run metadata artifacts for optimizer/pipeline reproducibility.

## Phase 3 (Week 3+): De-duplication and Quality
- Build shared gamecast domain service.
- Expand automated test suite and smoke pipelines.
- Add telemetry rollups and operational dashboards.

---

## Concrete File-Level Change List (First Pass)

- `src/notifications/injury_monitor.py`
  - replace scrape-only polling with persisted sync step
  - preserve diff logic and lock semantics
- `src/data/sync_service.py`
  - remove dead invalidation imports
  - call real invalidators independently
- `src/data/gamecast.py`
  - add odds cache invalidation function
- `src/notifications/service.py`
  - consume insert id from write call (no `last_insert_rowid()` read race)
- `src/database/db.py`
  - return stable insert id helper
- `src/ui/main_window.py`
  - close-event calls view-level stop API, not raw thread assumptions
- `src/ui/views/pipeline_view.py`
  - expose `request_stop()` and bounded worker teardown
- `src/ui/views/gamecast_view.py`
  - clear `_pending_headshots` in `finally`, add retry handling
- `src/web/templates/gamecast.html`
  - request sequencing + in-flight polling guard
- `src/analytics/stats_engine.py`
  - add season filter in next-game schedule query
- `web.py`
  - reloader-safe bootstrap pattern

---

## Risk Notes and Rollout Guardrails

- Keep optimizer formula/weights unchanged in Phase 0; only fix data correctness and runtime behavior.
- Run backtest before/after each phase; compare:
  - winner%
  - upset accuracy/rate
  - comp-dog and long-dog diagnostics
  - ROI and spread MAE
- Add one-click rollback for new cache key logic by preserving old cache files until validation passes.

---

## Success Criteria

- Injury notifications reflect real scrape changes within one poll interval.
- Sync/nuke reliably invalidate all relevant in-memory/disk caches.
- App close during pipeline is graceful and warning-free.
- Notification IDs are deterministic under concurrent writes.
- Web gamecast never renders stale game after rapid switching.
- No cross-season leakage in schedule-spot features.
- Backtest metrics remain stable unless intended by data freshness corrections.

---

## Optional Stretch Work (after core fixes)

- Add small “health panel” in pipeline view: cache state + last invalidation + monitor status.
- Add feature flags in settings for:
  - live sharp-money source preference
  - injury monitor poll interval
  - web gamecast polling cadence per state.

