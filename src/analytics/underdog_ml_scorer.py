"""Optional walk-forward ML ranker for underdog upset picks.

This module trains a lightweight logistic stack (NumPy only) to estimate
P(upset pick is correct). It is designed for promotion-time comparison:
baseline weights vs candidate weights on the same walk-forward split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.analytics.prediction import GameInput, predict
from src.analytics.thresholds import ACTUAL_WIN_THRESHOLD, MODEL_PICK_EDGE_THRESHOLD
from src.analytics.weight_config import WeightConfig


def _ml_payout_multiplier(ml_line: int) -> float:
    """Convert American odds to decimal payout multiplier."""
    if ml_line == 0:
        return 0.0
    if ml_line > 0:
        return 1.0 + float(ml_line) / 100.0
    return 1.0 + 100.0 / max(1.0, abs(float(ml_line)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    p = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _build_upset_feature_rows(
    games: Sequence[GameInput],
    weights: WeightConfig,
    include_sharp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix/labels for model-picked upset opportunities."""
    rows: List[List[float]] = []
    labels: List[float] = []

    for game in games:
        pred = predict(game, weights, include_sharp=include_sharp)
        vegas_fav_home = float(game.vegas_spread) < 0.0
        model_picks_home = float(pred.game_score) > MODEL_PICK_EDGE_THRESHOLD
        is_upset_pick = model_picks_home != vegas_fav_home
        if not is_upset_pick:
            continue

        actual_spread = float(game.actual_home_score) - float(game.actual_away_score)
        actual_home_win = actual_spread > ACTUAL_WIN_THRESHOLD
        actual_away_win = actual_spread < -ACTUAL_WIN_THRESHOLD
        actual_push = abs(actual_spread) <= ACTUAL_WIN_THRESHOLD
        model_correct = (
            (model_picks_home and actual_home_win)
            or ((not model_picks_home) and actual_away_win)
            or (actual_push and abs(float(pred.game_score)) <= 3.0)
        )

        ml_line = int(game.vegas_home_ml if model_picks_home else game.vegas_away_ml)
        payout = _ml_payout_multiplier(ml_line)
        edge_abs = abs(float(pred.game_score))
        spread_abs = abs(float(game.vegas_spread))
        edge_vs_spread = edge_abs / max(0.5, spread_abs)
        confidence = max(0.0, min(100.0, float(pred.confidence))) / 100.0
        payout_is_long_dog = 1.0 if payout >= 3.0 else 0.0

        rows.append(
            [
                edge_abs,
                spread_abs,
                payout,
                edge_vs_spread,
                confidence,
                payout_is_long_dog,
            ]
        )
        labels.append(1.0 if model_correct else 0.0)

    if not rows:
        return np.zeros((0, 6), dtype=float), np.zeros((0,), dtype=float)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=float)


def _fit_logistic_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    l2: float,
    max_iter: int,
) -> Dict[str, Any]:
    """Fit a simple L2-regularized logistic model with gradient descent."""
    n_samples = int(labels.shape[0])
    base_rate = float(np.clip(np.mean(labels) if n_samples else 0.5, 1e-4, 1.0 - 1e-4))

    if n_samples == 0:
        return {
            "kind": "constant",
            "prob": base_rate,
        }

    positives = int(np.sum(labels >= 0.5))
    negatives = n_samples - positives
    if positives == 0 or negatives == 0:
        return {
            "kind": "constant",
            "prob": base_rate,
        }

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = (features - mean) / std

    lr = max(1e-4, float(learning_rate))
    reg = max(0.0, float(l2)) / max(1.0, float(n_samples))
    steps = max(50, int(max_iter))

    weights = np.zeros((x.shape[1],), dtype=float)
    bias = float(np.log(base_rate / (1.0 - base_rate)))

    for _ in range(steps):
        logits = x @ weights + bias
        probs = _sigmoid(logits)
        err = probs - labels
        grad_w = (x.T @ err) / float(n_samples) + reg * weights
        grad_b = float(np.mean(err))
        weights -= lr * grad_w
        bias -= lr * grad_b

    return {
        "kind": "logistic",
        "weights": weights,
        "bias": bias,
        "mean": mean,
        "std": std,
        "prob": base_rate,
    }


def _predict_ranker_probs(model: Dict[str, Any], features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=float)

    if str(model.get("kind")) == "constant":
        prob = float(np.clip(model.get("prob", 0.5), 1e-6, 1.0 - 1e-6))
        return np.full((features.shape[0],), prob, dtype=float)

    mean = np.asarray(model.get("mean"), dtype=float)
    std = np.asarray(model.get("std"), dtype=float)
    std = np.where(std < 1e-6, 1.0, std)
    x = (features - mean) / std
    weights = np.asarray(model.get("weights"), dtype=float)
    bias = float(model.get("bias", 0.0))
    probs = _sigmoid(x @ weights + bias)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def evaluate_walk_forward_underdog_scorer(
    train_games: Sequence[GameInput],
    val_games: Sequence[GameInput],
    weights: WeightConfig,
    *,
    include_sharp: bool,
    min_train_samples: int,
    min_val_samples: int,
    learning_rate: float,
    l2: float,
    max_iter: int,
) -> Dict[str, Any]:
    """Train on walk-forward train slice, score on validation slice."""
    x_train, y_train = _build_upset_feature_rows(train_games, weights, include_sharp)
    x_val, y_val = _build_upset_feature_rows(val_games, weights, include_sharp)

    train_samples = int(y_train.shape[0])
    val_samples = int(y_val.shape[0])
    train_pos = int(np.sum(y_train >= 0.5))
    train_neg = int(train_samples - train_pos)

    if train_samples < max(1, int(min_train_samples)):
        return {
            "valid": False,
            "reason": f"insufficient train upset samples ({train_samples} < {min_train_samples})",
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_positive_rate": float(np.mean(y_train) * 100.0) if train_samples else 0.0,
            "val_positive_rate": float(np.mean(y_val) * 100.0) if val_samples else 0.0,
            "brier": None,
            "logloss": None,
            "accuracy": None,
        }

    if train_pos == 0 or train_neg == 0:
        return {
            "valid": False,
            "reason": (
                "insufficient train class diversity "
                f"(positives={train_pos}, negatives={train_neg})"
            ),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_positive_rate": float(np.mean(y_train) * 100.0),
            "val_positive_rate": float(np.mean(y_val) * 100.0) if val_samples else 0.0,
            "brier": None,
            "logloss": None,
            "accuracy": None,
        }

    if val_samples < max(1, int(min_val_samples)):
        return {
            "valid": False,
            "reason": f"insufficient validation upset samples ({val_samples} < {min_val_samples})",
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_positive_rate": float(np.mean(y_train) * 100.0),
            "val_positive_rate": float(np.mean(y_val) * 100.0) if val_samples else 0.0,
            "brier": None,
            "logloss": None,
            "accuracy": None,
        }

    model = _fit_logistic_ranker(
        x_train,
        y_train,
        learning_rate=learning_rate,
        l2=l2,
        max_iter=max_iter,
    )
    probs = _predict_ranker_probs(model, x_val)
    brier = float(np.mean((probs - y_val) ** 2))
    logloss = _safe_logloss(y_val, probs)
    accuracy = float(np.mean((probs >= 0.5) == (y_val >= 0.5)) * 100.0)

    return {
        "valid": True,
        "reason": "ok",
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_positive_rate": float(np.mean(y_train) * 100.0),
        "val_positive_rate": float(np.mean(y_val) * 100.0),
        "brier": brier,
        "logloss": logloss,
        "accuracy": accuracy,
    }


def compare_walk_forward_underdog_scorer(
    *,
    train_games: Sequence[GameInput],
    val_games: Sequence[GameInput],
    baseline_weights: WeightConfig,
    candidate_weights: WeightConfig,
    include_sharp: bool,
    min_train_samples: int,
    min_val_samples: int,
    learning_rate: float,
    l2: float,
    max_iter: int,
    min_brier_improvement: float,
) -> Dict[str, Any]:
    """Compare baseline vs candidate underdog scorer quality on walk-forward."""
    baseline = evaluate_walk_forward_underdog_scorer(
        train_games,
        val_games,
        baseline_weights,
        include_sharp=include_sharp,
        min_train_samples=min_train_samples,
        min_val_samples=min_val_samples,
        learning_rate=learning_rate,
        l2=l2,
        max_iter=max_iter,
    )
    candidate = evaluate_walk_forward_underdog_scorer(
        train_games,
        val_games,
        candidate_weights,
        include_sharp=include_sharp,
        min_train_samples=min_train_samples,
        min_val_samples=min_val_samples,
        learning_rate=learning_rate,
        l2=l2,
        max_iter=max_iter,
    )

    if not baseline.get("valid", False) or not candidate.get("valid", False):
        reason = (
            "ml underdog gate skipped: "
            f"baseline={baseline.get('reason', 'n/a')}; "
            f"candidate={candidate.get('reason', 'n/a')}"
        )
        return {
            "enabled": True,
            "applied": False,
            "passed": True,
            "reason": reason,
            "min_brier_improvement": float(min_brier_improvement),
            "brier_lift": None,
            "logloss_lift": None,
            "baseline": baseline,
            "candidate": candidate,
        }

    baseline_brier = float(baseline.get("brier", 0.0))
    candidate_brier = float(candidate.get("brier", 0.0))
    baseline_logloss = float(baseline.get("logloss", 0.0))
    candidate_logloss = float(candidate.get("logloss", 0.0))
    brier_lift = baseline_brier - candidate_brier
    logloss_lift = baseline_logloss - candidate_logloss

    min_gain = max(0.0, float(min_brier_improvement))
    passed = brier_lift >= min_gain
    reason = (
        "pass"
        if passed
        else (
            "ml underdog gate failed "
            f"(brier {baseline_brier:.4f}->{candidate_brier:.4f}, "
            f"lift {brier_lift:+.4f} < +{min_gain:.4f})"
        )
    )

    return {
        "enabled": True,
        "applied": True,
        "passed": passed,
        "reason": reason,
        "min_brier_improvement": min_gain,
        "brier_lift": float(brier_lift),
        "logloss_lift": float(logloss_lift),
        "baseline": baseline,
        "candidate": candidate,
    }
