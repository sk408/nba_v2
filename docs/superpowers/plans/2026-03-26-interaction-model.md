# LightGBM Interaction Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LightGBM residual correction layer that learns feature interactions and non-linear scaling the linear prediction model misses.

**Architecture:** The existing linear `predict()` stays as the interpretable backbone. A LightGBM model trains on the linear model's residuals (actual_margin - game_score_linear) using ~150 features: base edges, pairwise interaction products, and magnitude context features. The correction is capped at +/- 3.0 and applied after the linear score but before score calibration. The ML layer is excluded from the optimizer's evaluate() — it only runs in live predict() and backtesting.

**Tech Stack:** Python 3.10+, LightGBM, NumPy, existing SQLite/Flask/PySide6 stack

**Spec:** `docs/superpowers/specs/2026-03-26-interaction-model-design.md`

---

## File Structure

| File | Role |
|------|------|
| `src/analytics/interaction_model.py` | **NEW** — Core module: feature engineering, training, inference, model caching, `__main__` CLI |
| `tests/test_interaction_model.py` | **NEW** — Unit tests for feature engineering, training, inference, fallback |
| `src/analytics/prediction.py` | **MODIFY** — Add `interaction_detail` field to `Prediction`, call interaction layer in `predict()` |
| `src/analytics/pipeline.py` | **MODIFY** — Register `train_interaction_model` as pipeline step 11 |
| `src/config.py` | **MODIFY** — Add `interaction_model_enabled` and `interaction_model_correction_cap` defaults |
| `overnight.py` | **MODIFY** — Add step label and TUI transition for new step |
| `overnight_control_center.py` | **MODIFY** — Add "Interaction Model" settings group |
| `src/analytics/backtester.py` | **MODIFY** — Pass interaction toggle through to predict() |
| `src/web/app.py` | **MODIFY** — Surface interaction correction + detail in prediction breakdown |
| `requirements.txt` | **MODIFY** — Add `lightgbm` |
| `.gitignore` | **MODIFY** — Add model/metadata files |

---

### Task 1: Add LightGBM dependency and gitignore entries

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Add lightgbm to requirements.txt**

Add `lightgbm>=4.0.0` to `requirements.txt`, after the existing ML/data dependencies.

- [ ] **Step 2: Add model files to .gitignore**

Add these lines to `.gitignore`:

```
# Interaction model (retrained nightly, machine-specific)
data/interaction_model.lgb
data/interaction_model_meta.json
```

- [ ] **Step 3: Install lightgbm**

Run: `pip install lightgbm>=4.0.0`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "feat: add lightgbm dependency and gitignore interaction model files"
```

---

### Task 2: Add config defaults

**Files:**
- Modify: `src/config.py:215` (after ML underdog scorer settings)

- [ ] **Step 1: Add interaction model settings to _DEFAULTS**

Insert after line 215 (after `"optimizer_ml_underdog_scorer_min_brier_lift": 0.0025,`):

```python
    # Interaction model (LightGBM residual correction layer)
    "interaction_model_enabled": True,
    "interaction_model_correction_cap": 3.0,
    "interaction_model_min_train_games": 200,
```

- [ ] **Step 2: Verify settings load**

Run: `python -c "from src.config import get; print(get('interaction_model_enabled', None))"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add interaction_model config defaults"
```

---

### Task 3: Add interaction_detail field to Prediction dataclass

**Files:**
- Modify: `src/analytics/prediction.py:195-196`
- Test: `tests/test_interaction_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_interaction_model.py`:

```python
"""Tests for the LightGBM interaction model layer."""

from dataclasses import asdict, fields

from src.analytics.prediction import Prediction


def test_prediction_has_interaction_detail_field():
    """Prediction dataclass has interaction_detail field, defaults to None."""
    pred = Prediction()
    assert pred.interaction_detail is None
    # Verify it's a real dataclass field (visible to asdict, fields)
    field_names = {f.name for f in fields(pred)}
    assert "interaction_detail" in field_names
    d = asdict(pred)
    assert "interaction_detail" in d
    assert d["interaction_detail"] is None


def test_prediction_interaction_detail_settable():
    """interaction_detail can hold driver info dict."""
    pred = Prediction()
    pred.interaction_detail = {
        "top_drivers": [
            {"feature": "reb_diff__x__ff_oreb_edge", "contribution": 1.4},
        ]
    }
    assert len(pred.interaction_detail["top_drivers"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interaction_model.py::test_prediction_has_interaction_detail_field -v`
Expected: FAIL — `interaction_detail` not found in fields

- [ ] **Step 3: Add the field to Prediction dataclass**

In `src/analytics/prediction.py`, add after line 195 (`score_calibration_mode: str = ""`):

```python
    # Interaction model correction detail (populated when model is active)
    interaction_detail: Optional[Dict[str, Any]] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_interaction_model.py -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analytics/prediction.py tests/test_interaction_model.py
git commit -m "feat: add interaction_detail field to Prediction dataclass"
```

---

### Task 4: Build the interaction model core module — feature engineering

**Files:**
- Create: `src/analytics/interaction_model.py`
- Test: `tests/test_interaction_model.py` (append)

- [ ] **Step 1: Write the failing test for feature extraction**

Append to `tests/test_interaction_model.py`:

```python
import numpy as np


def test_build_feature_vector_from_adjustments():
    """Feature vector built from prediction adjustments dict."""
    from src.analytics.interaction_model import build_feature_vector

    adjustments = {
        "fatigue": -2.0,
        "turnover": 1.5,
        "rebound": 3.0,
        "rating_matchup": 4.0,
        "four_factors": 2.5,
        "opp_four_factors": -1.0,
        "hustle": 0.5,
        "rest_advantage": 1.0,
        "elo": 0.8,
        "travel": -0.3,
        "momentum": 0.2,
        "injury_vorp": 1.2,
        "pythag": 0.6,
        "srs": 0.4,
        "onoff_impact": 0.3,
    }
    names, vector = build_feature_vector(adjustments)

    # Should have base edges + interactions + magnitudes
    assert len(names) == len(vector)
    assert len(vector) > len(adjustments)  # more features than just base edges

    # Base edges present
    assert "rebound" in names
    assert "four_factors" in names

    # At least some interaction features present
    interaction_names = [n for n in names if "__x__" in n]
    assert len(interaction_names) > 0

    # Magnitude features present
    magnitude_names = [n for n in names if n.startswith("abs_")]
    assert len(magnitude_names) > 0

    # Interaction value is product of base edges
    reb_idx = names.index("rebound")
    ff_idx = names.index("four_factors")
    interaction_name = "rebound__x__four_factors"
    if interaction_name in names:
        ix_idx = names.index(interaction_name)
        assert vector[ix_idx] == vector[reb_idx] * vector[ff_idx]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interaction_model.py::test_build_feature_vector_from_adjustments -v`
Expected: FAIL — `interaction_model` module not found

- [ ] **Step 3: Implement feature engineering**

Create `src/analytics/interaction_model.py`:

```python
"""LightGBM interaction model — residual correction layer.

Learns feature interactions and non-linear scaling that the linear
prediction model misses. Trains on residuals (actual_margin - game_score_linear).
Correction is capped and applied after the linear score.

The model is excluded from the optimizer's evaluate() loop. It only runs
in live predict() and backtesting.
"""

import hashlib
import json
import logging
import os
import threading
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
_MODEL_PATH = os.path.join(_MODEL_DIR, "interaction_model.lgb")
_META_PATH = os.path.join(_MODEL_DIR, "interaction_model_meta.json")

# Base edge keys used for interaction feature generation.
# These are the adjustment keys from predict() that represent meaningful
# matchup edges. Order matters — first N are used for pairwise interactions.
BASE_EDGE_KEYS = [
    "fatigue", "turnover", "rebound", "rating_matchup",
    "four_factors", "opp_four_factors", "hustle", "rest_advantage",
    "elo", "travel", "timezone", "cum_travel_7d",
    "momentum", "mov_trend", "injury_vorp", "ref_home_bias",
    "schedule_spots", "road_trip_game", "season_progress",
    "roster_shock", "srs", "pythag", "onoff_impact",
    "fg3_luck", "process_stats", "sharp_ml", "sharp_spread",
    "clutch", "altitude_b2b",
]

# Top edges used for pairwise interaction generation (subset of BASE_EDGE_KEYS).
# These are the features most likely to have meaningful interactions.
INTERACTION_EDGE_KEYS = [
    "rebound", "four_factors", "opp_four_factors", "rating_matchup",
    "fatigue", "turnover", "elo", "injury_vorp",
    "travel", "hustle", "momentum", "pythag",
    "srs", "onoff_impact", "fg3_luck",
]

# Edges that get magnitude (abs) features for non-linear scaling.
MAGNITUDE_EDGE_KEYS = [
    "rebound", "four_factors", "opp_four_factors", "rating_matchup",
    "fatigue", "turnover", "elo", "injury_vorp",
    "travel", "hustle", "momentum", "pythag",
    "srs", "onoff_impact", "fg3_luck",
]


def build_feature_vector(
    adjustments: Dict[str, float],
) -> Tuple[List[str], np.ndarray]:
    """Build the full feature vector from prediction adjustments.

    Returns (feature_names, feature_values) where feature_values is a 1-D
    NumPy array. Three categories of features:
      1. Base edges — raw adjustment values
      2. Interactions — pairwise products of top edges
      3. Magnitudes — abs() of key edges for non-linear scaling

    Args:
        adjustments: The pred.adjustments dict from predict().

    Returns:
        Tuple of (list of feature name strings, numpy array of values).
    """
    names: List[str] = []
    values: List[float] = []

    # 1. Base edges
    for key in BASE_EDGE_KEYS:
        names.append(key)
        values.append(adjustments.get(key, 0.0))

    # 2. Pairwise interaction features
    for a, b in combinations(INTERACTION_EDGE_KEYS, 2):
        name = f"{a}__x__{b}"
        val_a = adjustments.get(a, 0.0)
        val_b = adjustments.get(b, 0.0)
        names.append(name)
        values.append(val_a * val_b)

    # 3. Magnitude context features
    for key in MAGNITUDE_EDGE_KEYS:
        names.append(f"abs_{key}")
        values.append(abs(adjustments.get(key, 0.0)))

    return names, np.array(values, dtype=np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_interaction_model.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analytics/interaction_model.py tests/test_interaction_model.py
git commit -m "feat: interaction model feature engineering (base edges, interactions, magnitudes)"
```

---

### Task 5: Build the interaction model core — training and inference

**Files:**
- Modify: `src/analytics/interaction_model.py`
- Test: `tests/test_interaction_model.py` (append)

- [ ] **Step 1: Write failing tests for training and inference**

Append to `tests/test_interaction_model.py`:

```python
import tempfile
import os


def test_train_and_predict_roundtrip():
    """Train on synthetic residuals, predict corrections."""
    from src.analytics.interaction_model import (
        train_model,
        predict_correction,
        build_feature_vector,
        BASE_EDGE_KEYS,
    )

    rng = np.random.RandomState(42)
    n_games = 300

    # Generate synthetic adjustments and residuals.
    # Create a signal: when rebound AND four_factors are both positive,
    # the linear model under-predicts (positive residual).
    all_adjustments = []
    residuals = []
    for _ in range(n_games):
        adj = {k: rng.randn() * 2.0 for k in BASE_EDGE_KEYS}
        all_adjustments.append(adj)
        # Residual has an interaction signal + noise
        residual = 0.5 * adj["rebound"] * adj["four_factors"] + rng.randn() * 1.0
        residuals.append(residual)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.lgb")
        meta_path = os.path.join(tmpdir, "test_meta.json")

        result = train_model(
            all_adjustments=all_adjustments,
            residuals=residuals,
            model_path=model_path,
            meta_path=meta_path,
            correction_cap=3.0,
        )

        assert result["status"] == "trained"
        assert result["n_train"] > 0
        assert result["val_rmse"] > 0
        assert os.path.exists(model_path)
        assert os.path.exists(meta_path)

        # Predict correction for a game with strong interaction signal
        strong_adj = {k: 0.0 for k in BASE_EDGE_KEYS}
        strong_adj["rebound"] = 5.0
        strong_adj["four_factors"] = 5.0
        correction, detail = predict_correction(
            adjustments=strong_adj,
            model_path=model_path,
            correction_cap=3.0,
        )

        # Should predict a positive correction (interaction signal)
        assert correction > 0.0
        # Should be capped
        assert abs(correction) <= 3.0
        # Detail should have top drivers
        assert "top_drivers" in detail
        assert len(detail["top_drivers"]) > 0


def test_predict_correction_no_model_returns_zero():
    """When model file doesn't exist, correction is 0.0."""
    from src.analytics.interaction_model import predict_correction, BASE_EDGE_KEYS

    adj = {k: 1.0 for k in BASE_EDGE_KEYS}
    correction, detail = predict_correction(
        adjustments=adj,
        model_path="/nonexistent/model.lgb",
        correction_cap=3.0,
    )
    assert correction == 0.0
    assert detail == {}


def test_train_model_insufficient_games():
    """Training with too few games returns early."""
    from src.analytics.interaction_model import train_model, BASE_EDGE_KEYS

    rng = np.random.RandomState(42)
    # Only 50 games — below 200 minimum
    all_adjustments = [{k: rng.randn() for k in BASE_EDGE_KEYS} for _ in range(50)]
    residuals = [rng.randn() for _ in range(50)]

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_model(
            all_adjustments=all_adjustments,
            residuals=residuals,
            model_path=os.path.join(tmpdir, "model.lgb"),
            meta_path=os.path.join(tmpdir, "meta.json"),
            correction_cap=3.0,
            min_train_games=200,
        )
        assert result["status"] == "skipped"
        assert "insufficient" in result.get("reason", "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_interaction_model.py::test_train_and_predict_roundtrip -v`
Expected: FAIL — `train_model` not found

- [ ] **Step 3: Implement training and inference functions**

Add to `src/analytics/interaction_model.py`:

```python
# ──────────────────────────────────────────────────────────────
# Model cache (thread-safe singleton)
# ──────────────────────────────────────────────────────────────

_model_cache_lock = threading.Lock()
_cached_model: Optional[Any] = None  # lightgbm.Booster
_cached_model_path: Optional[str] = None
_cached_feature_names: Optional[List[str]] = None


def _load_model(model_path: str):
    """Load LightGBM model from disk, with in-memory caching."""
    global _cached_model, _cached_model_path, _cached_feature_names
    with _model_cache_lock:
        if _cached_model is not None and _cached_model_path == model_path:
            mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0
            cache_mtime = getattr(_load_model, "_mtime", 0)
            if mtime == cache_mtime:
                return _cached_model, _cached_feature_names

        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed — interaction model disabled")
            return None, None

        if not os.path.exists(model_path):
            return None, None

        model = lgb.Booster(model_file=model_path)
        feature_names = model.feature_name()
        _cached_model = model
        _cached_model_path = model_path
        _cached_feature_names = feature_names
        _load_model._mtime = os.path.getmtime(model_path)
        return model, feature_names


def invalidate_model_cache():
    """Force reload on next prediction."""
    global _cached_model, _cached_model_path, _cached_feature_names
    with _model_cache_lock:
        _cached_model = None
        _cached_model_path = None
        _cached_feature_names = None


def weights_hash(w) -> str:
    """Canonical hash of a WeightConfig for staleness detection."""
    d = w.to_dict() if hasattr(w, "to_dict") else dict(w)
    return hashlib.sha256(
        json.dumps(sorted(d.items()), separators=(",", ":")).encode()
    ).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_model(
    all_adjustments: List[Dict[str, float]],
    residuals: List[float],
    model_path: str = _MODEL_PATH,
    meta_path: str = _META_PATH,
    correction_cap: float = 3.0,
    min_train_games: int = 200,
    weight_config=None,
) -> Dict[str, Any]:
    """Train a LightGBM model on linear model residuals.

    Args:
        all_adjustments: List of pred.adjustments dicts, one per game.
        residuals: List of (actual_margin - game_score_linear), one per game.
        model_path: Where to save the trained model.
        meta_path: Where to save training metadata.
        correction_cap: Max absolute correction value.
        min_train_games: Minimum games in training split.
        weight_config: Optional WeightConfig for staleness hash.

    Returns:
        Dict with training results: status, n_train, val_rmse, top_interactions.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return {"status": "error", "reason": "lightgbm not installed"}

    n = len(all_adjustments)
    if n != len(residuals):
        return {"status": "error", "reason": f"length mismatch: {n} vs {len(residuals)}"}

    # Walk-forward split: 80% train, 20% validation
    split_idx = int(n * 0.80)
    if split_idx < min_train_games:
        return {
            "status": "skipped",
            "reason": f"Insufficient training games: {split_idx} < {min_train_games}",
        }

    # Build feature matrices
    feature_names = None
    train_features = []
    val_features = []

    for i, adj in enumerate(all_adjustments):
        names, vec = build_feature_vector(adj)
        if feature_names is None:
            feature_names = names
        if i < split_idx:
            train_features.append(vec)
        else:
            val_features.append(vec)

    X_train = np.array(train_features)
    X_val = np.array(val_features)
    y_train = np.array(residuals[:split_idx])
    y_val = np.array(residuals[split_idx:])

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Evaluate
    val_preds = np.clip(model.predict(X_val), -correction_cap, correction_cap)
    val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
    mean_abs_correction = float(np.mean(np.abs(val_preds)))

    # Feature importances (gain-based)
    importance = model.feature_importance(importance_type="gain")
    importance_pairs = sorted(
        zip(feature_names, importance), key=lambda x: x[1], reverse=True
    )
    top_interactions = [
        {"feature": name, "importance": float(imp)}
        for name, imp in importance_pairs[:20]
    ]

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    # Save metadata
    meta = {
        "trained_at": __import__("datetime").datetime.now().isoformat(),
        "n_games": n,
        "n_train": split_idx,
        "n_val": n - split_idx,
        "val_rmse": round(val_rmse, 4),
        "mean_abs_correction": round(mean_abs_correction, 4),
        "correction_cap": correction_cap,
        "weights_hash": weights_hash(weight_config) if weight_config else "",
        "top_interactions": top_interactions,
        "feature_count": len(feature_names),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    invalidate_model_cache()

    return {
        "status": "trained",
        "n_train": split_idx,
        "n_val": n - split_idx,
        "val_rmse": val_rmse,
        "mean_abs_correction": mean_abs_correction,
        "top_interactions": top_interactions[:5],
    }


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

# Human-readable labels for feature names in the UI.
_FEATURE_LABELS = {
    "rebound": "Rebound edge",
    "four_factors": "Four factors",
    "opp_four_factors": "Opp four factors",
    "rating_matchup": "Rating matchup",
    "fatigue": "Fatigue",
    "turnover": "Turnover edge",
    "elo": "Elo rating",
    "injury_vorp": "Injury impact",
    "travel": "Travel fatigue",
    "hustle": "Hustle effort",
    "momentum": "Momentum",
    "pythag": "Pythagorean",
    "srs": "SRS",
    "onoff_impact": "On/off impact",
    "fg3_luck": "3PT luck regression",
}


def _human_label(feature_name: str) -> str:
    """Convert feature name to human-readable label."""
    if feature_name in _FEATURE_LABELS:
        return _FEATURE_LABELS[feature_name]
    if feature_name.startswith("abs_"):
        base = feature_name[4:]
        base_label = _FEATURE_LABELS.get(base, base.replace("_", " ").title())
        return f"|{base_label}|"
    if "__x__" in feature_name:
        a, b = feature_name.split("__x__", 1)
        label_a = _FEATURE_LABELS.get(a, a.replace("_", " ").title())
        label_b = _FEATURE_LABELS.get(b, b.replace("_", " ").title())
        return f"{label_a} x {label_b}"
    return feature_name.replace("_", " ").title()


def predict_correction(
    adjustments: Dict[str, float],
    model_path: str = _MODEL_PATH,
    correction_cap: float = 3.0,
) -> Tuple[float, Dict[str, Any]]:
    """Predict the residual correction for a single game.

    Args:
        adjustments: The pred.adjustments dict from predict().
        model_path: Path to the trained model file.
        correction_cap: Max absolute correction.

    Returns:
        (correction_value, detail_dict). If no model, returns (0.0, {}).
    """
    model, feature_names = _load_model(model_path)
    if model is None:
        return 0.0, {}

    names, vector = build_feature_vector(adjustments)
    X = vector.reshape(1, -1)

    # Raw prediction
    raw = float(model.predict(X)[0])
    correction = max(-correction_cap, min(correction_cap, raw))

    # Per-feature contributions (SHAP-like) for interpretability
    detail: Dict[str, Any] = {}
    try:
        contribs = model.predict(X, pred_contrib=True)[0]
        # contribs has len(features)+1 entries; last is bias
        feature_contribs = contribs[:-1]
        # Top 5 contributors by absolute contribution
        pairs = sorted(
            zip(names, feature_contribs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        detail["top_drivers"] = [
            {
                "feature": name,
                "label": _human_label(name),
                "contribution": round(float(c), 3),
            }
            for name, c in pairs[:5]
            if abs(c) > 0.01
        ]
    except Exception:
        logger.debug("pred_contrib failed, skipping detail", exc_info=True)
        detail["top_drivers"] = []

    return correction, detail


# ──────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────

def get_model_metadata() -> Optional[Dict[str, Any]]:
    """Load and return model metadata, or None if unavailable."""
    if not os.path.exists(_META_PATH):
        return None
    try:
        with open(_META_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def is_model_stale(current_weights=None) -> bool:
    """Check if the trained model is stale.

    Stale means: model is >48 hours old, or weights hash mismatches.
    """
    meta = get_model_metadata()
    if meta is None:
        return True

    from datetime import datetime, timedelta

    try:
        trained_at = datetime.fromisoformat(meta["trained_at"])
        if datetime.now() - trained_at > timedelta(hours=48):
            return True
    except (KeyError, ValueError):
        return True

    if current_weights is not None:
        stored_hash = meta.get("weights_hash", "")
        if stored_hash and stored_hash != weights_hash(current_weights):
            return True

    return False
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_interaction_model.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analytics/interaction_model.py tests/test_interaction_model.py
git commit -m "feat: interaction model training, inference, and caching"
```

---

### Task 6: Integrate interaction layer into predict()

**Files:**
- Modify: `src/analytics/prediction.py:503-530`
- Test: `tests/test_interaction_model.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interaction_model.py`:

```python
def test_predict_applies_interaction_correction(tmp_path, monkeypatch):
    """predict() applies interaction correction when model exists and is enabled."""
    from src.analytics.prediction import predict, GameInput, Prediction
    from src.analytics.interaction_model import _MODEL_PATH

    # We need a trained model for this test — train one on synthetic data
    from src.analytics.interaction_model import train_model, BASE_EDGE_KEYS, invalidate_model_cache

    rng = np.random.RandomState(42)
    n_games = 300
    all_adj = [{k: rng.randn() * 2.0 for k in BASE_EDGE_KEYS} for _ in range(n_games)]
    residuals = [rng.randn() * 2.0 for _ in range(n_games)]

    model_path = str(tmp_path / "model.lgb")
    meta_path = str(tmp_path / "meta.json")
    train_model(all_adj, residuals, model_path=model_path, meta_path=meta_path)

    # Patch the model path and enable setting
    monkeypatch.setattr("src.analytics.interaction_model._MODEL_PATH", model_path)
    monkeypatch.setattr("src.analytics.interaction_model._META_PATH", meta_path)
    invalidate_model_cache()

    # Enable interaction model — use targeted patching that passes through
    # to the real config.get for all other keys
    from src import config as _config_mod
    _real_get = _config_mod.get
    def _patched_get(key, default=None):
        overrides = {
            "interaction_model_enabled": True,
            "interaction_model_correction_cap": 3.0,
        }
        if key in overrides:
            return overrides[key]
        return _real_get(key, default)
    monkeypatch.setattr("src.config.get", _patched_get)

    game = GameInput()
    from src.analytics.weight_config import WeightConfig
    w = WeightConfig()
    pred = predict(game, w)

    # The interaction correction should be in adjustments
    assert "interaction_correction" in pred.adjustments
    assert abs(pred.adjustments["interaction_correction"]) <= 3.0
    # interaction_detail should be populated
    assert pred.interaction_detail is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interaction_model.py::test_predict_applies_interaction_correction -v`
Expected: FAIL — `interaction_correction` not in adjustments

- [ ] **Step 3: Add interaction layer call to predict()**

In `src/analytics/prediction.py`, add an import at the top (near other analytics imports):

```python
from src.analytics.interaction_model import predict_correction as _interaction_correction
```

Then insert after the process stats block (after `pred.adjustments["process_stats"] = process_adj`, around line 503) and **before** the "Derive projected scores" comment (line 505):

```python
    # ── Interaction model correction (LightGBM residual layer) ──
    try:
        from src.config import get as _get_config
        if _get_config("interaction_model_enabled", True):
            cap = _get_config("interaction_model_correction_cap", 3.0)
            correction, detail = _interaction_correction(
                adjustments=pred.adjustments,
                correction_cap=float(cap),
            )
            if correction != 0.0:
                game_score += correction
                pred.adjustments["interaction_correction"] = correction
                pred.interaction_detail = detail
    except Exception:
        logger.debug("Interaction model unavailable", exc_info=True)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_interaction_model.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing prediction tests to verify no regression**

Run: `pytest tests/test_prediction_core.py -v`
Expected: All existing tests PASS (interaction layer falls back gracefully when no model)

- [ ] **Step 6: Commit**

```bash
git add src/analytics/prediction.py tests/test_interaction_model.py
git commit -m "feat: integrate interaction model correction into predict()"
```

---

### Task 7: Build the pipeline training step

**Files:**
- Create: pipeline training function in `src/analytics/interaction_model.py`
- Modify: `src/analytics/pipeline.py:467-469`
- Test: `tests/test_interaction_model.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_interaction_model.py`:

```python
def test_run_train_interaction_model_step():
    """Pipeline step function has correct signature and returns dict."""
    from src.analytics.interaction_model import run_train_interaction_model

    import inspect
    sig = inspect.signature(run_train_interaction_model)
    assert "callback" in sig.parameters
    assert "is_cancelled" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interaction_model.py::test_run_train_interaction_model_step -v`
Expected: FAIL — `run_train_interaction_model` not found

- [ ] **Step 3: Implement the pipeline step function**

Add to `src/analytics/interaction_model.py`:

```python
# ──────────────────────────────────────────────────────────────
# Pipeline step
# ──────────────────────────────────────────────────────────────

def run_train_interaction_model(
    callback=None, is_cancelled=None
) -> Dict[str, Any]:
    """Pipeline step: train the interaction model on linear model residuals.

    Loads all precomputed games, runs the linear model to get game_score,
    computes residuals vs actual margins, and trains LightGBM.

    Args:
        callback: Optional progress message function.
        is_cancelled: Optional cancellation check function.

    Returns:
        Dict with training results.
    """
    from src.analytics.prediction import predict, GameInput, precompute_all_games
    from src.analytics.weight_config import get_weight_config
    from src.config import get as get_setting

    def emit(msg: str):
        if callback:
            callback(msg)
        logger.info(msg)

    if not get_setting("interaction_model_enabled", True):
        emit("Interaction model disabled in settings — skipping.")
        return {"status": "disabled"}

    w = get_weight_config()
    cap = float(get_setting("interaction_model_correction_cap", 3.0))
    min_games = int(get_setting("interaction_model_min_train_games", 200))

    # Load precomputed games (same function used by optimizer and backtester)
    games = precompute_all_games()
    if not games:
        emit("No precomputed games available — skipping interaction model training.")
        return {"status": "skipped", "reason": "no precomputed games"}

    emit(f"Building residuals from {len(games)} precomputed games...")

    # Run linear model (without interaction layer) to get residuals
    all_adjustments = []
    residuals = []
    for game in games:
        if is_cancelled and is_cancelled():
            return {"status": "cancelled"}

        actual_margin = game.actual_home_score - game.actual_away_score
        if actual_margin == 0.0 and game.actual_home_score == 0.0:
            continue  # Skip games without actual results

        # Temporarily disable interaction model for pure linear score
        pred = predict(game, w, include_sharp=False)
        linear_score = pred.game_score
        # Remove interaction correction if it snuck in
        if "interaction_correction" in pred.adjustments:
            linear_score -= pred.adjustments["interaction_correction"]
            del pred.adjustments["interaction_correction"]

        all_adjustments.append(dict(pred.adjustments))
        residuals.append(actual_margin - linear_score)

    emit(f"Computed {len(residuals)} residuals. Training LightGBM...")

    result = train_model(
        all_adjustments=all_adjustments,
        residuals=residuals,
        correction_cap=cap,
        min_train_games=min_games,
        weight_config=w,
    )

    if result["status"] == "trained":
        top = result.get("top_interactions", [])
        top_str = ", ".join(f"{t['feature']}({t['importance']:.0f})" for t in top[:3])
        emit(
            f"Interaction model trained: {result['n_train']} train / "
            f"{result['n_val']} val, RMSE={result['val_rmse']:.3f}, "
            f"avg |correction|={result['mean_abs_correction']:.2f}"
        )
        if top_str:
            emit(f"Top interactions: {top_str}")
    else:
        emit(f"Interaction model training: {result.get('status')} — {result.get('reason', '')}")

    return result
```

- [ ] **Step 4: Register the step in pipeline.py**

In `src/analytics/pipeline.py`, add import at the top:

```python
from src.analytics.interaction_model import run_train_interaction_model
```

Then modify `PIPELINE_STEPS` (around line 467) to insert the new step before backtest:

```python
    ("optimize_sharp", run_optimize_sharp),
    ("train_interaction_model", run_train_interaction_model),
    ("backtest", run_backtest_and_compare),
```

Also update the docstring at the top of `pipeline.py` to say "12-step" instead of "11-step".

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_interaction_model.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/analytics/interaction_model.py src/analytics/pipeline.py tests/test_interaction_model.py
git commit -m "feat: add train_interaction_model pipeline step"
```

---

### Task 8: Update overnight.py TUI for new step

**Files:**
- Modify: `overnight.py:93-105` (STEP_LABELS)
- Modify: `overnight.py:304-310` (step transitions)

- [ ] **Step 1: Add step label**

In `overnight.py`, add to `STEP_LABELS` dict (after `"optimize_sharp": "Optimize Sharp",`):

```python
        "train_interaction_model": "Interaction Model",
```

- [ ] **Step 2: Update step transition logic**

In `overnight.py`, find the step transition block (around lines 304-310) where specific step names set `self.in_pipeline = False`. Add `"train_interaction_model"` to that group:

Where you see checks like:
```python
if step_name in ("optimize_fundamentals", "optimize_sharp", "backtest"):
    self.in_pipeline = False
```

Change to:
```python
if step_name in ("optimize_fundamentals", "optimize_sharp", "train_interaction_model", "backtest"):
    self.in_pipeline = False
```

- [ ] **Step 3: Verify overnight.py imports still work**

Run: `python -c "from overnight import RichOvernightConsole; print(len(RichOvernightConsole.PIPELINE_STEPS), 'steps'); print(RichOvernightConsole.STEP_LABELS.get('train_interaction_model'))"`
Expected: `12 steps` and `Interaction Model`

- [ ] **Step 4: Commit**

```bash
git add overnight.py
git commit -m "feat: add interaction model step to overnight TUI"
```

---

### Task 9: Update overnight_control_center.py

**Files:**
- Modify: `overnight_control_center.py` (after ML Underdog Gate group, around line 908)

- [ ] **Step 1: Add the Interaction Model settings group**

Find the `_build_ml_gate_card` method (around line 838). After it, add a new method:

```python
    def _build_interaction_model_card(self) -> QWidget:
        box = self._group("Interaction Model")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._add_bool(
            form,
            "interaction_model_enabled",
            "Enable Interaction Model",
            "Apply LightGBM residual correction layer to predictions.\n"
            "Learns feature interactions the linear model misses.",
        )
        self._add_float(
            form,
            "interaction_model_correction_cap",
            "Correction Cap",
            0.5, 10.0, 0.5, 1,
            "Max absolute correction in game_score points (default 3.0).",
        )
        self._add_int(
            form,
            "interaction_model_min_train_games",
            "Min Training Games",
            50, 1000,
            "Minimum games in training split before model trains (default 200).",
        )

        # Model status display
        status_label = QLabel("Model status: checking...")
        status_label.setStyleSheet("font-size: 11px; color: #94a3b8;")
        form.addRow(status_label)
        self._interaction_status_label = status_label

        # Retrain button
        retrain_btn = QPushButton("Retrain Now")
        retrain_btn.setToolTip("Manually trigger interaction model training")
        retrain_btn.setFixedWidth(160)
        retrain_btn.clicked.connect(self._on_retrain_interaction_model)
        form.addRow(self._label(""), retrain_btn)

        box.layout().addLayout(form)

        # Load initial status
        QTimer.singleShot(500, self._refresh_interaction_model_status)

        return box

    def _refresh_interaction_model_status(self):
        """Update interaction model status display."""
        try:
            from src.analytics.interaction_model import get_model_metadata, is_model_stale
            meta = get_model_metadata()
            if meta is None:
                self._interaction_status_label.setText("Model status: not trained yet")
                self._interaction_status_label.setStyleSheet("font-size: 11px; color: #FFB300;")
                return

            from datetime import datetime
            trained_at = datetime.fromisoformat(meta["trained_at"])
            age = datetime.now() - trained_at
            hours = age.total_seconds() / 3600
            age_str = f"{hours:.0f}h ago" if hours < 48 else f"{hours / 24:.0f}d ago"
            n_games = meta.get("n_games", "?")
            rmse = meta.get("val_rmse", "?")
            stale = is_model_stale()

            color = RED if stale else GREEN
            stale_tag = " (STALE)" if stale else ""
            self._interaction_status_label.setText(
                f"Trained {age_str}, {n_games} games, RMSE={rmse}{stale_tag}"
            )
            self._interaction_status_label.setStyleSheet(f"font-size: 11px; color: {color};")
        except Exception:
            self._interaction_status_label.setText("Model status: error reading metadata")
            self._interaction_status_label.setStyleSheet(f"font-size: 11px; color: {RED};")

    def _on_retrain_interaction_model(self):
        """Manually trigger interaction model retraining (threaded to avoid UI freeze)."""
        from src.analytics.interaction_model import run_train_interaction_model

        # Disable button and show spinner text while training
        sender = self.sender()
        if sender:
            sender.setEnabled(False)
            sender.setText("Training...")
        self._interaction_status_label.setText("Training in progress...")
        self._interaction_status_label.setStyleSheet(f"font-size: 11px; color: {AMBER};")

        class _TrainWorker(QThread):
            finished = Signal(dict)

            def run(self):
                try:
                    result = run_train_interaction_model(
                        callback=lambda msg: logger.info("[interaction] %s", msg)
                    )
                    self.finished.emit(result)
                except Exception as e:
                    self.finished.emit({"status": "error", "reason": str(e)})

        def _on_train_done(result):
            if sender:
                sender.setEnabled(True)
                sender.setText("Retrain Now")
            status = result.get("status", "unknown")
            if status == "trained":
                QMessageBox.information(
                    self, "Interaction Model",
                    f"Model trained successfully.\n"
                    f"RMSE: {result.get('val_rmse', '?'):.4f}\n"
                    f"Games: {result.get('n_train', '?')} train / {result.get('n_val', '?')} val"
                )
            else:
                QMessageBox.warning(
                    self, "Interaction Model",
                    f"Training {status}: {result.get('reason', 'see logs')}"
                )
            self._refresh_interaction_model_status()

        self._train_worker = _TrainWorker()
        self._train_worker.finished.connect(_on_train_done)
        self._train_worker.start()
```

- [ ] **Step 2: Add the card to the settings layout**

Find where `_build_ml_gate_card()` is called and added to the layout. Add the interaction model card right after it:

```python
        layout.addWidget(self._build_interaction_model_card())
```

- [ ] **Step 3: Verify the control center launches**

Run: `python -c "from overnight_control_center import OvernightControlCenter; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add overnight_control_center.py
git commit -m "feat: add Interaction Model settings group to overnight control center"
```

---

### Task 10: Add __main__ CLI entry point

**Files:**
- Modify: `src/analytics/interaction_model.py` (append at bottom)

- [ ] **Step 1: Add __main__ block**

Append to `src/analytics/interaction_model.py`:

```python
# ──────────────────────────────────────────────────────────────
# CLI entry point: python -m src.analytics.interaction_model
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from src.bootstrap import bootstrap
        bootstrap()
    except Exception as e:
        print(f"Bootstrap failed: {e}", file=sys.stderr)
        sys.exit(1)

    result = run_train_interaction_model(
        callback=lambda msg: print(msg),
    )
    status = result.get("status", "unknown")
    if status == "trained":
        print(f"\nTraining complete. RMSE: {result['val_rmse']:.4f}")
        sys.exit(0)
    elif status == "skipped":
        print(f"\nSkipped: {result.get('reason', '')}")
        sys.exit(0)
    else:
        print(f"\nFailed: {result.get('reason', status)}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 2: Verify module is runnable**

Run: `python -c "import src.analytics.interaction_model; print('module imports OK')"`
Expected: `module imports OK`

- [ ] **Step 3: Commit**

```bash
git add src/analytics/interaction_model.py
git commit -m "feat: add CLI entry point for manual interaction model training"
```

---

### Task 11: Update backtester to support interaction model toggle

**Files:**
- Modify: `src/analytics/backtester.py:475-507`

- [ ] **Step 1: Read backtester predict() calls**

Read `src/analytics/backtester.py` around lines 475-507 to see current predict() calls.

- [ ] **Step 2: Add interaction model awareness to backtester**

The interaction model is controlled by the config toggle `interaction_model_enabled`, which `predict()` already checks. No changes needed to the backtester's predict() calls — the interaction layer auto-activates based on the config setting.

However, for the "Interaction Lift" metric, the backtester needs to know whether the interaction model affected predictions. The `_build_per_game_result` function does not currently store individual adjustments — it stores the derived pick/confidence/score. To implement Interaction Lift without invasive changes to the backtester's result structure:

1. In `_build_per_game_result`, add `interaction_correction` to the per-game result dict by reading it from `pred.adjustments`:

```python
    # In _build_per_game_result, after building the result dict:
    result["interaction_correction"] = pred.adjustments.get("interaction_correction", 0.0)
```

2. In the summary section after both loops complete, compute interaction lift:

```python
    # Interaction model lift metric
    interaction_corrections = [r.get("interaction_correction", 0.0) for r in fund_per_game]
    games_with_correction = sum(1 for c in interaction_corrections if c != 0.0)
    if games_with_correction > 0:
        summary["interaction_model"] = {
            "games_with_correction": games_with_correction,
            "avg_abs_correction": round(
                sum(abs(c) for c in interaction_corrections) / max(games_with_correction, 1), 3
            ),
        }
```

**Note:** Full A/B accuracy comparison (running predict() twice per game — with and without the model, then comparing correct pick rates) is deferred to a follow-up task. This initial implementation records the correction magnitude, which is sufficient to verify the model is active and to gauge its influence. The full lift metric requires a second predict() pass per game, which would roughly double backtest runtime.

- [ ] **Step 3: Commit**

```bash
git add src/analytics/backtester.py
git commit -m "feat: add interaction model awareness to backtester"
```

---

### Task 12: Surface interaction correction in web dashboard

**Files:**
- Modify: `src/web/app.py` (prediction detail template rendering)
- Modify: `src/web/templates/` (if prediction detail is in a template)

- [ ] **Step 1: Find where prediction adjustments are rendered**

Search `src/web/app.py` and `src/web/templates/` for where `adjustments` or the prediction breakdown is displayed to users. The interaction correction will appear as `adjustments["interaction_correction"]` which may auto-render if the template iterates all adjustments.

- [ ] **Step 2: Add interaction detail rendering**

If the template iterates `adjustments`, the correction value will auto-appear. For the expandable detail (top drivers), add rendering for `pred.interaction_detail`:

In the prediction detail section of the template/route, after rendering adjustments, add:

```python
    # In the route that builds prediction context:
    interaction_detail = None
    if hasattr(pred, 'interaction_detail') and pred.interaction_detail:
        interaction_detail = pred.interaction_detail.get("top_drivers", [])
```

Pass `interaction_detail` to the template. In the template, render as a collapsible section:

```html
{% if interaction_detail %}
<div class="interaction-drivers">
    <small>Top interaction drivers:</small>
    <ul>
    {% for driver in interaction_detail %}
        <li>{{ driver.label }}: {{ "%+.2f"|format(driver.contribution) }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}
```

Exact template locations depend on how the prediction breakdown is currently rendered — this task requires reading the specific template files first.

- [ ] **Step 3: Verify web app renders without errors**

Run: `python -c "from src.web.app import app; print('web app imports OK')"`
Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add src/web/app.py src/web/templates/
git commit -m "feat: surface interaction correction detail in web dashboard"
```

---

### Task 13: Update desktop gamecast view

**Files:**
- Modify: `src/ui/views/gamecast_view.py`

- [ ] **Step 1: Find where prediction adjustments are displayed**

Read `src/ui/views/gamecast_view.py` and search for where `adjustments` or prediction breakdown is shown. The interaction correction appears as `adjustments["interaction_correction"]` and may auto-render.

- [ ] **Step 2: Add interaction detail to gamecast prediction panel**

If the view iterates adjustments, the value auto-appears. For the driver detail, add a tooltip or expandable section similar to the web dashboard:

```python
    # When building adjustment display items:
    if pred.interaction_detail and pred.interaction_detail.get("top_drivers"):
        drivers = pred.interaction_detail["top_drivers"]
        tooltip = "\n".join(
            f"{d['label']}: {d['contribution']:+.2f}" for d in drivers
        )
        # Set tooltip on the interaction_correction row
```

Exact implementation depends on the current gamecast view structure — read the file first.

- [ ] **Step 3: Commit**

```bash
git add src/ui/views/gamecast_view.py
git commit -m "feat: show interaction correction in desktop gamecast"
```

---

### Task 14: Run full test suite and verify

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `ruff check src/`
Expected: No errors (or only pre-existing ones)

- [ ] **Step 3: Verify web app starts**

Run: `python -c "from src.web.app import app; print('OK')"`

- [ ] **Step 4: Verify overnight imports**

Run: `python -c "from overnight import RichOvernightConsole; print(len(RichOvernightConsole.PIPELINE_STEPS), 'steps')"`
Expected: `12 steps`

- [ ] **Step 5: Final commit if any lint fixes were needed**

```bash
git add -A
git commit -m "fix: lint and test fixes for interaction model"
```
