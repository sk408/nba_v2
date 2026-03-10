# NBA V2 Unified Review Change Plan

Date: 2026-03-09  
Scope: Reconciled action plan based on both independent review documents and their cross-feedback.

---

## Goal

Ship the fastest path to a more correct, secure, and maintainable system by:

1. Fixing confirmed runtime/data-integrity bugs first.
2. Closing security and install/test infrastructure gaps in parallel.
3. Hardening model/data correctness and cache reliability.
4. Deferring UX/features until core correctness and operability are stable.

---

## Priority Framework

- **P0 (Do now):** Confirmed correctness/runtime bugs and data integrity risks.
- **P0-V (Verify quickly, then patch):** High-impact claims requiring quick reproduction or code confirmation.
- **P1 (Same sprint):** Security baseline + install/test/CI foundations.
- **P2 (Next sprint):** Model integrity, cache architecture, and performance.
- **P3 (After stabilization):** UX improvements and feature expansions.

---

## P0 - Confirmed Immediate Fixes

### 1) Injury monitor persistence gap
- **Change:** Ensure monitor poll path persists scraped injuries before diff/notify.
- **Files:** `src/notifications/injury_monitor.py`, `src/data/injury_scraper.py`
- **Acceptance:** Status change in source feed updates DB and emits one notification.

### 2) Notification insert ID race
- **Change:** Return insert ID from the same DB write operation (`cursor.lastrowid` pattern), stop separate `last_insert_rowid()` read.
- **Files:** `src/notifications/service.py`, `src/database/db.py`
- **Acceptance:** Concurrent inserts return correct IDs per emitted notification.

### 3) Web gamecast stale-response race
- **Change:** Add request sequencing and ignore out-of-order responses; prevent overlapping poll fetches.
- **Files:** `src/web/templates/gamecast.html`
- **Acceptance:** Rapid game switching never renders stale game data.

### 4) Desktop headshot pending set leak
- **Change:** Remove entries from `_pending_headshots` in success/failure `finally`; allow bounded retry.
- **Files:** `src/ui/views/gamecast_view.py`
- **Acceptance:** Failed headshot can retry and pending set size remains bounded.

### 5) Cache invalidation dead paths
- **Change:** Replace missing/dead invalidation imports with live invalidators; make invalidation calls independent so one failure does not skip others.
- **Files:** `src/data/sync_service.py`, `src/data/odds_sync.py`, `src/analytics/prediction.py`, `src/analytics/backtester.py`, `src/analytics/stats_engine.py`, `src/analytics/weight_config.py`
- **Acceptance:** Sync/nuke/odds-refresh consistently cold-start relevant caches.

### 6) Schedule-spot season leakage
- **Change:** Add missing season filter(s) in schedule spot historical lookups.
- **Files:** `src/analytics/stats_engine.py`
- **Acceptance:** Cross-season test fixtures no longer leak future season context.

### 7) Flask debug double-bootstrap
- **Change:** Make startup reloader-safe so bootstrap/background services run once in debug mode.
- **Files:** `web.py`, `src/bootstrap.py`
- **Acceptance:** One injury monitor/service instance in debug runs.

---

## P0-V - Verify in 2-4 Hours, Patch Immediately If Confirmed

### A) Score calibration hard crash candidates
- **Claim:** Missing function references/settings in score calibration path.
- **Files:** `src/analytics/score_calibration.py`
- **Action:** Reproduce path; if confirmed, gate feature path or implement missing functions/settings.

### B) Pipeline worker/shutdown contract mismatch
- **Claim:** Main window shutdown calls thread APIs not guaranteed by current worker object.
- **Files:** `src/ui/main_window.py`, `src/ui/views/pipeline_view.py`
- **Action:** Standardize `request_stop()` API on view; remove assumptions about private thread attrs.

### C) Pipeline/API drift in worker layer
- **Claim:** `src/ui/workers.py` imports/calls legacy symbols not present in current analytics modules.
- **Files:** `src/ui/workers.py`, `src/analytics/*`
- **Action:** Remove dead workers or migrate to current APIs.

### D) Pipeline step count drift (core vs UI/CLI)
- **Claim:** Step definitions differ, producing incorrect progress/status reporting.
- **Files:** `src/analytics/pipeline.py`, `src/ui/views/pipeline_view.py`, `overnight.py`
- **Action:** Share one canonical step definition.

### E) Odds force propagation bug
- **Claim:** `force=True` is dropped before reaching historical backfill call.
- **Files:** `src/data/sync_service.py`, `src/data/odds_sync.py`
- **Action:** Thread force argument end-to-end and add regression test.

---

## P1 - Security and Foundation (Same Sprint)

### Security baseline
- Add CSRF or authenticated action tokens for sensitive POST routes.
- Add security headers (`X-Frame-Options`, `X-Content-Type-Options`, CSP baseline, HSTS when HTTPS is enforced).
- Remove unsafe `force=True` JSON parsing and enforce content type.
- Stop returning raw exception internals to clients.
- Protect and/or disable remote shutdown endpoint in non-local contexts.
- Add endpoint-level rate limiting for sync/predict-heavy routes.
- **Files:** `src/web/app.py`, `src/web/templates/*.html`

### Installability and reproducibility
- Complete dependency declarations (`PySide6`, `optuna`, `rich`, `nba_api`, `cloudscraper`, `websocket-client`, etc.).
- Add deterministic optimizer mode (fixed seed option) for repeatable validation runs.
- **Files:** `requirements.txt` or `pyproject.toml`, `src/analytics/optimizer.py`, `src/analytics/pipeline.py`

### Test and CI bootstrap
- Create pytest scaffold with fixture DB and smoke data.
- Add CI workflow for lint + tests.
- Add lint/format configuration (ruff baseline).
- **Files:** `tests/`, `.github/workflows/`, `pyproject.toml` (or equivalent config files)

---

## P2 - Data/Model Correctness and Performance

### Historical correctness
- Thread explicit season through feature builders and Elo lookups.
- Ensure historical team attribution uses immutable-at-game-time team identity.
- Align optimizer winner threshold semantics with live prediction semantics (or clearly separate and document them).
- **Files:** `src/analytics/stats_engine.py`, `src/analytics/prediction.py`, `src/analytics/elo.py`, `src/database/migrations.py`, `src/data/nba_fetcher.py`

### Sync integrity and observability
- Return structured per-step sync status (success/partial/fail), not just best-effort completion.
- Ensure metadata counts come from loaded/canonical sources.
- **Files:** `src/data/sync_service.py`, `src/analytics/pipeline.py`, `src/analytics/memory_store.py`

### Cache architecture and invalidation
- Introduce cache invalidation registry keyed by events (`post_sync`, `post_nuke`, `post_odds_sync`, `post_weight_save`).
- Add source fingerprinting to cache keys/artifacts to avoid stale hits after data corrections.
- **Files:** `src/analytics/cache_registry.py` (new), `src/analytics/prediction.py`, `src/analytics/backtester.py`, `src/data/sync_service.py`

### Performance hardening
- Add filters/pagination to heavy `player_stats` loads.
- Bound long-lived in-memory caches with max size and/or TTL.
- Add lock for injury scraper cache access if shared across threads.
- Add static asset cache headers for web assets.
- **Files:** `src/analytics/prediction.py`, `src/analytics/stats_engine.py`, `src/data/injury_scraper.py`, `src/web/app.py`

---

## P3 - UX and Feature Layer (After Stabilization)

- Keyboard-first navigation in desktop UI.
- Accessibility improvements beyond color-only signals.
- Responsive/high-DPI scaling cleanup.
- Export pathways (CSV/report snapshots).
- Confidence intervals, explanation/waterfall view, trend and history surfaces.
- **Files:** `src/ui/*`, `src/web/*`, reporting/export modules

---

## Recommended Execution Sequence

### Phase 0 (Day 1-2): Correctness Hotfixes
- Complete all P0 items.
- Run targeted regression checks for each bug.

### Phase 1 (Day 3-4): Verify-and-patch P0-V + security baseline start
- Close all verified P0-V findings.
- Implement web security baseline controls.

### Phase 2 (Day 5-7): Foundation
- Complete dependencies, tests scaffold, lint config, and CI.

### Phase 3 (Week 2): Data/model integrity + cache/perf
- Implement P2 correctness and cache architecture updates.

### Phase 4 (Week 3+): UX/features
- Execute P3 improvements after core stability KPIs pass.

---

## Stability Gates (Do Not Skip)

- **Before/after backtest comparison** for every phase touching model/data path.
- **Concurrency checks** for notification creation and thread shutdown.
- **Rapid interaction web test** for gamecast switching/poll overlap.
- **Freshness tests** proving cache invalidation after sync/nuke/odds refresh.
- **Install test on clean environment** after dependency file updates.

---

## Success Criteria

- No known P0 bug remains reproducible.
- Sensitive web routes have baseline protections and safer error handling.
- Project installs cleanly from dependency manifest.
- Minimal test/CI pipeline is green.
- Historical feature generation is season-correct and free of known leakage.
- Sync and cache behavior is observable, deterministic, and verifiably fresh.

