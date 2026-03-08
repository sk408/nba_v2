# Project-Wide Static Error Audit

Date: 2026-03-07

## Scope

- Included all Python code under `src/` plus top-level scripts `desktop.py`, `web.py`, and `overnight.py`.
- Excluded non-code runtime artifacts in `data/` (db/log/cache/journal/wal files).
- Static analysis only (no runtime test/lint/type-check execution).

## Executive Summary

- Highest-risk issues are concentrated in orchestration boundaries: pipeline execution, worker wiring, and data/feature consistency.
- Critical defects include stale-read behavior in write-heavy flows, broken/legacy worker imports, and season leakage in historical feature generation.
- Several high-impact consistency issues can silently degrade model correctness (odds backfill coverage, pipeline step drift, sync metadata inaccuracies).

## Critical Findings

### 1) Stale-read architecture used in write-heavy workflows

- **Affected files:** `src/database/db.py`, `src/ui/views/pipeline_view.py`, `src/ui/workers.py`, `overnight.py`
- **Failure mode:** Code wraps full pipeline/sync/overnight routines in `thread_local_db()`, which uses a private snapshot where later reads do not see writes made during the same run.
- **Why it occurs:** `thread_local_db` intentionally isolates reads from concurrent writes, but current orchestration uses it for end-to-end write/read workflows that expect fresh state.
- **Recommendation:** Do not run write-heavy orchestration flows inside `thread_local_db()`. Reserve it for read-mostly analytics tasks. Run pipeline/sync against shared DB path.

### 2) Worker API drift and dead imports

- **Affected file:** `src/ui/workers.py`
- **Failure mode:** Workers import modules/functions that do not exist in current codebase (`weight_optimizer`, `ml_model`, `precompute_game_data`, `run_full_pipeline`, `run_retune`).
- **Why it occurs:** Worker layer appears partially legacy while analytics modules were refactored.
- **Recommendation:** Either remove stale workers or migrate them to current APIs (`precompute_all_games`, `optimize_weights` in `optimizer.py`, `run_pipeline`, `run_overnight`).

### 3) Historical season leakage in feature generation

- **Affected files:** `src/analytics/stats_engine.py`, `src/analytics/prediction.py`, `src/analytics/elo.py`
- **Failure mode:** Historical precompute calls feature builders that use `get_season()` (current season) instead of game season; Elo lookup signature accepts `season` but ignores it.
- **Why it occurs:** Feature helpers are season-implicit while precompute is season-explicit.
- **Recommendation:** Thread explicit `season` through feature functions and enforce season filters in all historical reads (including Elo lookup SQL).

### 4) Historical team attribution instability

- **Affected files:** `src/database/migrations.py`, `src/data/nba_fetcher.py`, `src/analytics/stats_engine.py`, `src/analytics/elo.py`
- **Failure mode:** `player_stats` does not store `team_id`; many queries infer team via `JOIN players`, but `players.team_id` is mutable and overwritten by roster sync.
- **Why it occurs:** Current schema relies on current roster mapping for historical logs.
- **Recommendation:** Add `team_id` to `player_stats` and backfill from game context; migrate analytics queries to use `player_stats.team_id` directly.

## High Findings

### 5) Odds backfill skips dates too aggressively

- **Affected file:** `src/data/odds_sync.py`
- **Failure mode:** Date-level `NOT IN` query can skip missing games when any game on that date already has sharp fields.
- **Recommendation:** Backfill at game-level `(game_date, home_team_id, away_team_id)` completeness, not date-only completeness.

### 6) Force flag dropped in odds sync path

- **Affected file:** `src/data/sync_service.py`
- **Failure mode:** `sync_historical_odds(..., force=True)` does not pass `force` into `backfill_odds`, so full refresh intent is lost.
- **Recommendation:** Propagate `force` parameter through to `backfill_odds(force=force)`.

### 7) Pipeline step model drift between core and UI

- **Affected files:** `src/analytics/pipeline.py`, `src/ui/views/pipeline_view.py`, `overnight.py`
- **Failure mode:** Core pipeline has 10 steps but `PipelineView` still tracks 6; UI progress, timing, and status are incomplete/misaligned.
- **Recommendation:** Use a single canonical step definition imported by both pipeline and UI.

### 8) Sync metadata uses unloaded memory store

- **Affected files:** `src/analytics/pipeline.py`, `src/analytics/memory_store.py`
- **Failure mode:** `_mark_step_done` reads counts from `InMemoryDataStore` without guaranteeing it is loaded, often writing zeros/blank dates.
- **Recommendation:** Read counts directly from DB (`player_stats`) for sync metadata, or ensure store load is explicit and validated before read.

### 9) Nuke flow can leave partial data

- **Affected file:** `src/data/sync_service.py`
- **Failure mode:** Table deletes are best-effort and continue on failure; FK order is not guaranteed. Function can report completion with leftover data.
- **Recommendation:** Use explicit deletion order inside one guarded transaction (or temporary FK disable with strict re-enable), and fail loudly when critical tables fail.

### 10) Optimizer winner thresholds differ from live prediction semantics

- **Affected files:** `src/analytics/optimizer.py`, `src/analytics/prediction.py`
- **Failure mode:** Optimizer classification uses `> 0.5 / < -0.5`, while live pick boundary is `> 0 / < 0`.
- **Recommendation:** Align decision thresholds with live prediction path or clearly separate calibration logic and document purpose.

### 11) Overview worker calls prediction with invalid parameters

- **Affected files:** `src/ui/workers.py`, `src/analytics/prediction.py`
- **Failure mode:** `predict_matchup(..., skip_ml=True, skip_espn=True)` is called with unsupported kwargs, causing runtime errors.
- **Recommendation:** Update call site to valid signature and remove unsupported flags.

## Medium Findings

### 12) Full sync reports success after partial failures

- **Affected file:** `src/data/sync_service.py`
- **Failure mode:** Step exceptions are logged but sync continues; terminal message still says complete.
- **Recommendation:** Return structured per-step status with failed/partial state; bubble failure to UI and pipeline state.

### 13) Incremental game log sync ignores source corrections

- **Affected file:** `src/data/nba_fetcher.py`
- **Failure mode:** `INSERT OR IGNORE` for `player_stats` prevents updates to existing rows when upstream values change.
- **Recommendation:** Use upsert (`ON CONFLICT ... DO UPDATE`) for mutable stat fields.

### 14) Optional missing module silently disables invalidation

- **Affected files:** `src/data/sync_service.py`, `src/data/odds_sync.py`
- **Failure mode:** `prediction_quality` import is guarded and absent; odds cache invalidation silently no-ops.
- **Recommendation:** Remove dead import path or reintroduce module; surface explicit warning in logs when invalidation path is unavailable.

## Suggested Remediation Order

1. Fix critical orchestration correctness (`thread_local_db` misuse, worker API drift).
2. Fix historical feature correctness (season propagation, team attribution schema).
3. Fix data completeness paths (odds backfill/game-level criteria, force propagation).
4. Unify pipeline step definitions across core/UI/CLI.
5. Harden sync/error reporting and metadata accuracy.

## Verification Checklist (Post-Fix)

- Pipeline run shows all defined steps consistently in UI and CLI.
- Historical precompute for at least two prior seasons uses correct season-scoped features.
- Odds backfill fills missing games on partially populated dates.
- Worker actions used by UI import and execute without `ImportError`/`TypeError`.
- Sync result reports explicit partial-failure status when any sub-step fails.
