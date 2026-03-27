# LightGBM Interaction Model — Design Spec

**Date**: 2026-03-26
**Status**: Approved

## Problem

The current prediction engine (`predict()` in `prediction.py`) computes `game_score` as a purely additive sum of ~30 independent feature edges, each scaled by a static weight from `WeightConfig`. This misses two classes of signal:

1. **Feature interactions** — when a team dominates one dimension (e.g., rebounding), it amplifies the predictive power of related dimensions (e.g., OREB four factor). The linear model treats these independently.
2. **Non-linear scaling** — a 10-rebound edge matters disproportionately more than a 2-rebound edge. The linear model scales both proportionally.

## Solution: Hybrid Linear + LightGBM Residual Layer

Keep the existing linear model as the interpretable backbone. Add a LightGBM gradient-boosted tree model that learns *residual corrections* — the systematic errors the linear model makes due to interactions and non-linearity it can't capture.

```
GameInput --> Linear Model (predict()) --> game_score_linear
                                               |
GameInput features + game_score_linear --> LightGBM --> residual_correction
                                               |
                             game_score_final = game_score_linear + residual_correction
```

### Why hybrid over pure ML or pure optimizer interactions

- **vs. pure ML replacement**: The linear model is interpretable — you can see exactly why a pick was made via `pred.adjustments`. A pure ML model is a black box.
- **vs. optimizer interaction terms**: Adding pairwise interaction weights to Optuna would create identifiability issues (interaction terms are correlated with their base features) and overfitting risk. The current ~50-parameter CMA-ES space is already near practical limits.
- **Hybrid advantage**: Each layer learns what it's best at. The linear model captures main effects; the tree model captures residual patterns. The tree's feature importances reveal *which* interactions matter, providing new basketball insight.

## Architecture

### Feature Engineering (ML Layer Inputs)

Three categories of features, ~150 total:

**Base edges (~30 features):**
The differential values already computed during `predict()` — rebound diff, four factor edges, rating matchup, fatigue adjustment, elo diff, injury VORP diff, travel diff, etc. These are the same values stored in `pred.adjustments`. No new data ingestion required.

**Interaction features (~105 features):**
Pairwise products of the top ~15 base edges (ranked by feature importance from an initial LightGBM training run). Examples: `reb_diff * ff_oreb_edge`, `fatigue_adj * travel_diff`, `elo_diff * injury_vorp_diff`. LightGBM naturally ignores irrelevant interactions via its split-based feature selection.

**Magnitude context features (~15 features):**
Absolute values of key edges (e.g., `abs(reb_diff)`), allowing the model to learn non-linear self-scaling independent of direction. "A rebound gap above 8 matters disproportionately" is learned through tree splits on these features.

**Excluded**: Raw team IDs, player names, or any feature that would cause the model to memorize teams rather than learn generalizable patterns.

### Training Pipeline

**Training target**: `residual = actual_margin - game_score_linear`

**Data**: All games from the current season (and optionally historical seasons) with precomputed linear predictions. Walk-forward split: train on first 80% chronologically, validate on last 20% — matching the existing optimizer's `WALK_FORWARD_SPLIT`.

**Minimum sample guard**: Training requires at least **200 games** in the training split (i.e., ~250 total games after walk-forward split). Before this threshold (roughly early December of each season), the step exits early with an info log and the system falls back to linear-only. This prevents overfitting on thin early-season data.

**Training process**:
1. Run linear model on all historical games to get `game_score_linear`
2. Compute residuals: `actual_margin - game_score_linear`
3. Build feature matrix (base edges + interactions + magnitudes)
4. Train LightGBM regressor:
   - `max_depth=4` (shallow trees prevent overfitting)
   - `n_estimators=200` with early stopping on validation set
   - `learning_rate=0.05`
   - `min_child_samples=20`
5. Evaluate on validation fold: RMSE, mean absolute correction, feature importances
6. Cap predictions to configurable max (default +/- 3.0 points)

**Correction cap rationale**: The cap prevents the ML layer from overwhelming the linear core. With a +/- 3.0 cap, the linear model remains the primary driver while the ML layer provides meaningful refinement.

### Model Persistence

- **Model file**: `data/interaction_model.lgb` (LightGBM native binary, ~100KB)
- **Metadata sidecar**: `data/interaction_model_meta.json` containing:
  - Training timestamp
  - Game count
  - Validation RMSE
  - Feature importance rankings (ordered list)
  - Correction cap value
  - Linear weights hash (for staleness detection — see hashing spec below)
  - Top discovered interactions with human-readable labels
- **Staleness guard**: If model is >48 hours old or linear weights hash mismatches, prediction engine logs a warning but still uses the stale model.

**Weights hash computation**: Canonical hash is computed as:
```python
hashlib.sha256(
    json.dumps(sorted(w.to_dict().items()), separators=(',', ':')).encode()
).hexdigest()[:16]
```
Sorting by key name ensures deterministic ordering. Compact JSON separators eliminate whitespace variance. Truncated to 16 hex chars for readability in metadata.

### Prediction Integration

**In `predict()` (prediction.py)**:

After all linear adjustments are applied to `game_score`, but **before** `apply_score_calibration()`:
1. Load LightGBM model (cached in memory after first load, like `WeightConfig`)
2. Build feature vector from edges already computed during this prediction
3. Run single-row inference (sub-millisecond)
4. Cap correction to +/- max
5. Add to `game_score`
6. Store: `pred.adjustments["interaction_correction"] = correction_value`
7. Store detail in `pred.interaction_detail` (new `Optional[Dict]` field on `Prediction` dataclass) using LightGBM's `predict(pred_contrib=True)` for per-prediction feature contributions

Score calibration then receives the post-interaction `game_score`, which is correct because the interaction layer is part of the model's output. However, **score calibration must be retrained** after the interaction model is first deployed (and after any significant interaction model change). The `train_interaction_model` pipeline step will invalidate the calibration freshness flag so it retrains on the next pipeline run.

**`Prediction` dataclass change**: Add field:
```python
interaction_detail: Optional[Dict[str, Any]] = None
```
Follows the same pattern as existing nullable fields on `Prediction` (e.g., `sharp_agrees: Optional[bool] = None`, `calibrated_spread: Optional[float] = None`). Visible to `asdict()`, serialization, and `fields(pred)` iteration.

**NOT in `VectorizedGames.evaluate()` during optimization**:

The interaction layer is **excluded** from the optimizer's `evaluate()` method. The residuals were learned against a specific set of linear weights (the weights at training time). During optimization, `evaluate()` is called with candidate weight configurations that differ from the training weights — applying the interaction correction would corrupt the optimizer's loss signal and produce meaningless trial scores.

The interaction layer is applied only in:
- **Live `predict()`** — real-time predictions for the web/desktop
- **Backtesting** — post-optimization evaluation where final weights are locked

This means `optimizer.py` does **not** need modification for the interaction layer. Removed from the Modified Files table.

**Graceful fallback**: If no model file exists, the interaction layer is skipped entirely. `game_score` is pure linear — identical to current behavior. No errors, no code path changes.

**Toggle**: Config key `interaction_model_enabled` (default `true`). Disables ML layer without deleting the model file. Useful for A/B comparison in backtests.

## Pipeline Integration

**Updated 12-step overnight pipeline**:
1. backup
2. sync
3. settle_recommendations
4. seed_arenas
5. bbref_sync
6. referee_sync
7. elo_compute
8. precompute
9. optimize_fundamentals
10. optimize_sharp
11. **train_interaction_model** (NEW)
12. backtest

The interaction model trains after both optimizers so it learns residuals from the final tuned linear weights. Backtest runs last to evaluate the full stack including the interaction layer.

**Manual trigger**: Exposed as a callable function `train_interaction_model()` in `src/analytics/interaction_model.py`. Invocable from:
- **Desktop app**: Button in the pipeline/settings panel that calls the function directly
- **CLI**: `python -m src.analytics.interaction_model` (module-level `__main__` block that runs bootstrap + training)

The existing `overnight.py` does not have a `--step` mechanism and adding one is out of scope for this feature. The module-level CLI entry point achieves the same goal without modifying the overnight argparser.

**Failure handling**: If training fails (insufficient games, LightGBM error), the step logs the error and the pipeline continues. Backtest falls back to linear-only. Non-blocking.

## UI Surfacing

### Desktop App (Settings Panel)
- Toggle switch: "Interaction Model" on/off (maps to `interaction_model_enabled`)
- Status line below toggle: model age ("Trained 6h ago, 1,247 games"), validation RMSE, staleness indicator
- "Retrain Now" button to manually trigger training

### Web Dashboard (Prediction Detail)
- New adjustment row in prediction breakdown: `Interaction Correction: +2.3`
- Expandable detail showing top 3-5 interaction drivers with human-readable labels:
  - "Rebound dominance x OREB four factor: +1.4"
  - "Fatigue x travel load: +0.6"
  - "Elo gap x injury VORP: +0.3"
- Label mapping from feature names (e.g., `reb_diff__x__ff_oreb_edge` -> "Rebound dominance x OREB four factor")

### Desktop Gamecast View
- Same adjustment row in prediction panel, consistent with web

### Overnight Pipeline TUI
- `train_interaction_model` step in Rich progress panel with status and duration
- Add entry to `STEP_LABELS` dict: `"train_interaction_model": "Interaction Model"`
- Update TUI step transition logic: add `"train_interaction_model"` to the `in_pipeline = False` group alongside `optimize_fundamentals`, `optimize_sharp`, and `backtest` (lines 304-308). This keeps the display in the detailed view after optimization finishes, preventing a visual flash back to the pipeline steps view before backtest starts
- Completion summary: training games, validation RMSE, top 5 discovered interactions, correction cap, accuracy lift vs linear-only

### Overnight Control Center (`overnight_control_center.py`)
The standalone PySide6 control center UI needs updates across three areas:

**Settings panel** — new settings group:
- Add a new `_group("Interaction Model")` section (after the existing "ML Underdog Gate" group)
- Toggle: `interaction_model_enabled` — bool checkbox, "Enable Interaction Model"
- Status display: read `data/interaction_model_meta.json` and show model age, game count, validation RMSE, staleness indicator (similar to how other status info is displayed)
- "Retrain Now" button that calls `train_interaction_model()` directly, with a progress spinner and result toast

**Step progress table** — the `_steps_tbl` QTableWidget:
- Row count auto-derives from `RichOvernightConsole.PIPELINE_STEPS` (which imports from `pipeline.py`), so adding the step to `pipeline.py`'s `PIPELINE_STEPS` automatically adds the table row
- **`STEP_LABELS` is NOT auto-derived** — it is a hardcoded dict in `RichOvernightConsole`. The `"train_interaction_model": "Interaction Model"` entry must be manually added there (same edit as the `overnight.py` TUI section). Without this, the fallback label would display "Train Interaction Model" from the `step.replace("_", " ").title()` fallback

**TUI log parsing** — the progress parser and log feed:
- Add parsing for `train_interaction_model` step completion messages (training games, RMSE, top interactions)
- The step sits between `optimize_sharp` and `backtest`, so any display mode transitions that check for specific step names (e.g., switching from optimization view to pipeline view) need to handle `train_interaction_model` as a non-optimization step that precedes backtest

### Backtest Reports
- New metric: "Interaction Lift" — accuracy delta between linear-only and linear+interaction

## New Files

| File | Purpose |
|------|---------|
| `src/analytics/interaction_model.py` | Training, inference, feature engineering, model loading/caching, `__main__` CLI entry point |
| `data/interaction_model.lgb` | Trained LightGBM model binary (gitignored) |
| `data/interaction_model_meta.json` | Training metadata and feature importances (gitignored) |

## Modified Files

| File | Changes |
|------|---------|
| `src/analytics/prediction.py` | Add `interaction_detail` field to `Prediction` dataclass; add interaction layer call after linear game_score (before calibration), store adjustment + detail |
| `src/analytics/pipeline.py` | Add `train_interaction_model` step (step 11); update step count in docstring from 11 to 12 |
| `overnight.py` | Register new step in Rich TUI: add to `STEP_LABELS`, update step transition logic for the new step between optimize_sharp and backtest |
| `overnight_control_center.py` | New "Interaction Model" settings group with toggle + status + retrain button; TUI log parsing for new step; step table auto-updates from PIPELINE_STEPS |
| `src/web/app.py` | Surface interaction correction + detail in prediction breakdown |
| `src/ui/views/settings_view.py` | Add interaction model toggle + status + retrain button |
| `src/ui/views/gamecast_view.py` | Show interaction correction in prediction panel |
| `src/config.py` | Register `interaction_model_enabled` default |
| `requirements.txt` | Add `lightgbm` dependency |
| `.gitignore` | Add `data/interaction_model.lgb` and `data/interaction_model_meta.json` |

## Dependencies

- `lightgbm` (pip install) — the only new dependency. No PyTorch/TensorFlow required.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting on small game sample | Shallow trees (depth 4), min 20 samples per leaf, early stopping, walk-forward validation, **hard minimum of 200 training games** — step exits early below threshold |
| ML layer overwhelms linear model | Correction cap (+/- 3.0 default), configurable |
| Stale model after weight changes | Canonical weights hash in metadata triggers staleness warning; pipeline ordering ensures retraining after optimizer |
| LightGBM import failure / missing model | Graceful fallback to linear-only, no errors |
| Feature importance instability across retrains | Log top interactions to metadata for tracking drift over time |
| Score calibration distribution shift | Interaction model training invalidates calibration freshness flag, forcing recalibration on next pipeline run |
| Double-counting / layer drift | The pipeline always retrains the interaction model *after* optimizer updates, so the ML layer always learns residuals against the latest linear weights. Over successive nights, the linear optimizer tunes against raw outcomes (no interaction layer in `evaluate()`), then the interaction model re-learns the fresh residuals. This prevents the two layers from absorbing the same variance. Accepted minor risk: if the interaction model captures signal that the linear optimizer could theoretically learn, the linear weights may shift slightly on the next run — but the nightly retrain cycle self-corrects this within one pipeline iteration |
| Interaction layer applied during optimization | Explicitly excluded from `VectorizedGames.evaluate()` — interaction model only runs in live `predict()` and backtesting |
