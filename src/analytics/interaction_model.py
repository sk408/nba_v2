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
    from datetime import datetime

    meta = {
        "trained_at": datetime.now().isoformat(),
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
    model_path: Optional[str] = None,
    correction_cap: float = 3.0,
) -> Tuple[float, Dict[str, Any]]:
    """Predict the residual correction for a single game.

    Args:
        adjustments: The pred.adjustments dict from predict().
        model_path: Path to the trained model file (defaults to _MODEL_PATH).
        correction_cap: Max absolute correction.

    Returns:
        (correction_value, detail_dict). If no model, returns (0.0, {}).
    """
    if model_path is None:
        model_path = _MODEL_PATH
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
    from src.analytics.prediction import predict, precompute_all_games  # noqa: I001
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

        # Skip games without actual results
        if game.actual_home_score == 0 and game.actual_away_score == 0:
            continue

        actual_margin = game.actual_home_score - game.actual_away_score

        pred = predict(game, w, include_sharp=False)
        linear_score = pred.game_score
        # Remove interaction correction if it was applied, to get pure linear score
        if "interaction_correction" in pred.adjustments:
            linear_score -= pred.adjustments["interaction_correction"]
            del pred.adjustments["interaction_correction"]

        all_adjustments.append(dict(pred.adjustments))
        residuals.append(actual_margin - linear_score)

    if not residuals:
        emit("No games with actual results — skipping interaction model training.")
        return {"status": "skipped", "reason": "no games with results"}

    emit(f"Computed {len(residuals)} residuals. Training LightGBM...")

    try:
        result = train_model(
            all_adjustments=all_adjustments,
            residuals=residuals,
            correction_cap=cap,
            min_train_games=min_games,
            weight_config=w,
        )
    except Exception as e:
        logger.exception("Interaction model training failed")
        emit(f"Interaction model training failed: {e}")
        return {"status": "error", "reason": str(e)}

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


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.bootstrap import bootstrap, setup_logging

    setup_logging()
    bootstrap(enable_daily_automation=False)

    result = run_train_interaction_model(callback=lambda msg: print(msg))
    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
    print(f"\nDone — status: {status}")
    sys.exit(0 if status in ("trained", "skipped", "disabled") else 1)
