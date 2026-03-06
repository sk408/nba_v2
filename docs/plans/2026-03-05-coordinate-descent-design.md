# Coordinate Descent Refinement — Design Document

## Purpose

Add coordinate descent (CD) as an on-demand weight refinement tool to the V2 desktop app. CD grid-searches each parameter individually after Optuna TPE finds the right neighborhood, squeezing out incremental gains that stochastic sampling misses.

## Key Decisions

- **Loss function**: V2's existing `-(winner_pct + upset_accuracy * upset_rate / 100 * upset_bonus_mult)`. Same as Optuna — no dog-focused variants.
- **Pipeline integration**: On-demand only. Not part of the automatic pipeline. Triggered manually from Settings view.
- **Parameter ranges**: `CD_RANGES` — wider than `OPTIMIZER_RANGES` (~20-30% wider bounds) for broader exploration.
- **Steps**: User-configurable, 50–2000 (default 100).

## Architecture

### 1. `coordinate_descent()` in `optimizer.py`

```python
def coordinate_descent(
    games: List[GameInput],
    params: Optional[List[str]] = None,  # None = all CD_RANGES keys
    steps: int = 100,                     # grid resolution (up to 2000)
    max_rounds: int = 10,
    convergence_threshold: float = 0.005,
    include_sharp: bool = False,
    callback: Optional[Callable] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    save: bool = True,
) -> Dict[str, Any]
```

**Algorithm:**
1. Load current weights, split games 80/20 train/val (chronological).
2. Create `VectorizedGames(train)` and `VectorizedGames(val)`.
3. Evaluate baseline on both sets, record initial loss and winner_pct.
4. For each round (up to `max_rounds`):
   - For each parameter in `params`:
     - Generate grid: `np.linspace(lo, hi, steps)` from `CD_RANGES`.
     - Evaluate each grid value on **training set only**.
     - Accept if training loss < current training loss.
   - End-of-round: evaluate full config on validation.
   - Check convergence: `accepted_count == 0` or `improvement < threshold`.
5. **Save gate**: validation `winner_pct > max(favorites_pct, previous_best_winner_pct)`.
6. If improved and `save=True`: `save_weight_config()`, `invalidate_weight_cache()`.

**Per-parameter acceptance uses training loss only** (not validation metrics). This prevents overfitting to the smaller validation set.

**Returns:**
```python
{
    "weights": dict,           # Final weight values
    "history": list,           # Per-round metrics
    "changes": dict,           # {param: {"before": x, "after": y}}
    "rounds": int,             # Rounds executed
    "improved": bool,          # Save gate passed
    "initial_winner_pct": float,
    "final_winner_pct": float,
    "initial_loss": float,
    "final_loss": float,
    "favorites_pct": float,
}
```

### 2. `CD_RANGES` in `weight_config.py`

Same keys as `OPTIMIZER_RANGES` but with wider bounds. Example:
```python
CD_RANGES = {
    "def_factor_dampening": (0.05, 12.0),    # OPTIMIZER: (0.1, 10.0)
    "turnover_margin_mult": (0.0, 12.0),     # OPTIMIZER: (0.0, 10.0)
    "four_factors_scale":   (0.5, 120.0),     # OPTIMIZER: (1.0, 100.0)
    ...
}
```

### 3. Settings View — CD Section

**Controls:**
- Mode selector: radio buttons "Fundamentals Only" / "Fundamentals + Sharp"
- Steps slider: 50–2000, default 100, with numeric value label
- Max rounds spinner: 1–20, default 10
- Run CD button (green), Cancel button (red, visible only while running)

**Progress display:**
- Terminal-styled monospace log (same pattern as Pipeline view)
- Round indicator: "Round 2/10 — 15/26 params"
- Current best winner_pct and improvement delta

**Results display (after completion):**
- Summary card: rounds executed, params changed, winner_pct improvement
- Changes table: Parameter | Before | After | Delta (sorted by |delta|)

### 4. Data Flow

```
Settings View → [Run CD clicked]
  → Spawn _CDWorker(QObject) on QThread
  → Worker: thread_local_db(), precompute_all_games(), coordinate_descent()
  → Progress signals → log updates
  → save_weight_config() on improvement
  → finished signal → results summary + changes table
```

**Cancellation**: `is_cancelled` checks `threading.Event` between each parameter.

## Files Modified

| File | Change |
|------|--------|
| `weight_config.py` | Add `CD_RANGES` dict |
| `optimizer.py` | Add `coordinate_descent()` function (~200 lines) |
| `settings_view.py` | Add CD section with controls + progress + results (~200 lines) |

## Files NOT Modified

- `pipeline.py` — CD is on-demand only
- `prediction.py` — CD saves weights via existing `save_weight_config()`
- `main_window.py` — Settings tab already exists

## Sync Guarantee

CD uses the same `VectorizedGames.evaluate()` as Optuna. The formula is identical to `predict()` in `prediction.py`. Two code paths, already in sync by design.
