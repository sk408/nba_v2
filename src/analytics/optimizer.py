"""Optuna CMA-ES optimizer for NBA Fundamentals V2.

Loss function:
-(winner_pct + upset_accuracy * upset_rate / 100 * upset_bonus_mult - objective_penalty)
where objective_penalty increases when upset coverage or Tier-A hit rate
falls below configured targets.

Studies are persisted to data/optuna_studies.db so trials accumulate across
runs and survive crashes.  CMA-ES sampler learns parameter correlations for
efficient 49-dimensional exploration.

VectorizedGames converts List[GameInput] into flat NumPy arrays for fast evaluation.
optimize_weights() runs walk-forward Optuna optimization.
compare_modes() A/B tests fundamentals-only vs fundamentals+sharp.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
from typing import List, Dict, Any, Optional, Callable, Tuple

import numpy as np

from src.config import get as get_setting
from src.analytics.prediction import GameInput
from src.analytics.weight_config import (
    FOUR_FACTORS_FIXED_SCALE,
    WeightConfig,
    get_weight_config,
    save_weight_config,
    OPTIMIZER_RANGES, SHARP_MODE_RANGES, invalidate_weight_cache,
    CD_RANGES, CD_SHARP_RANGES,
)
from src.analytics.thresholds import MODEL_PICK_EDGE_THRESHOLD, ACTUAL_WIN_THRESHOLD
from src.analytics.underdog_metrics import (
    DEFAULT_TIER_A_MIN_CONFIDENCE,
    DEFAULT_TIER_B_MIN_CONFIDENCE,
    summarize_underdog_quality,
)
from src.analytics.underdog_ml_scorer import compare_walk_forward_underdog_scorer
from src.utils.settings_helpers import (
    safe_bool_setting,
    safe_float_setting,
    safe_int_setting,
)

logger = logging.getLogger(__name__)

# Module-level study cache: keeps in-memory studies alive across optimization
# passes so we don't reload 18K+ trials from disk every time.
_study_cache: Dict[str, Any] = {}        # study_name -> optuna.Study
_save_threads: Dict[str, Any] = {}       # study_name -> threading.Thread
_stage_champion_bank_lock = threading.Lock()
_stage_champion_bank_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "stage_champion_bank.json",
)
_stage_champion_modes = ("fundamentals", "sharp")
_stage_champion_stages = ("core", "ff", "onoff", "joint_refine")

# Walk-forward: train on first N% of games, validate on last (1-N)%.
WALK_FORWARD_SPLIT = 0.80

# Compression penalty: prevents optimizer from collapsing all predictions to 0.
COMPRESSION_RATIO_FLOOR = 0.55
COMPRESSION_PENALTY_MULT = 80.0

# Default minimum payout multiplier for ML ROI calculations.
# Can be overridden by config key: optimizer_min_ml_payout.
MIN_ML_PAYOUT = 1.50

# One-possession underdog threshold (diagnostic only).
ONE_POSSESSION_DOG_MARGIN = 3.0

# Long-dog payout filter for one-possession diagnostics/tiebreaks.
LONG_DOG_MIN_PAYOUT = 3.0


def _safe_float_setting(key: str, default: float) -> float:
    """Read a float setting with defensive fallback."""
    return safe_float_setting(key, default)


def _safe_int_setting(key: str, default: int) -> int:
    """Read an int setting with defensive fallback."""
    return max(0, safe_int_setting(key, default))


def _safe_bool_setting(key: str, default: bool) -> bool:
    """Read a bool setting with tolerant parsing."""
    return safe_bool_setting(key, default)


def _objective_signature_blob() -> str:
    """Return objective settings fingerprint for Optuna study names."""
    return "|".join(
        [
            f"upset_bonus:{_safe_float_setting('upset_bonus_mult', 0.5):.4f}",
            f"onepos_credit_enabled:{int(_safe_bool_setting('optimizer_onepos_credit_enabled', True))}",
            f"onepos_credit_margin:{_safe_float_setting('optimizer_onepos_credit_margin', ONE_POSSESSION_DOG_MARGIN):.4f}",
            f"onepos_credit_all_w:{_safe_float_setting('optimizer_onepos_credit_all_dogs_weight', 0.5):.4f}",
            f"onepos_credit_long_w:{_safe_float_setting('optimizer_onepos_credit_long_dogs_weight', 1.0):.4f}",
            f"onepos_credit_affects_winner:{int(_safe_bool_setting('optimizer_onepos_credit_affects_winner_pct', True))}",
            f"save_onepos_credit_bump_mult:{_safe_float_setting('optimizer_save_onepos_credit_bump_mult', 0.15):.4f}",
            f"target_cov:{_safe_float_setting('optimizer_objective_target_upset_coverage_pct', 20.0):.4f}",
            f"target_tier_a:{_safe_float_setting('optimizer_objective_target_tier_a_hit_rate', 56.0):.4f}",
            f"cov_short_mult:{_safe_float_setting('optimizer_objective_coverage_shortfall_mult', 0.20):.4f}",
            f"tier_a_short_mult:{_safe_float_setting('optimizer_objective_tier_a_shortfall_mult', 0.12):.4f}",
            f"tier_a_prior_w:{_safe_float_setting('optimizer_objective_tier_a_prior_weight', 12.0):.4f}",
            f"tier_a_min:{_safe_float_setting('optimizer_objective_tier_a_min_confidence', DEFAULT_TIER_A_MIN_CONFIDENCE):.4f}",
            f"tier_b_min:{_safe_float_setting('optimizer_objective_tier_b_min_confidence', DEFAULT_TIER_B_MIN_CONFIDENCE):.4f}",
            f"val_probe:{int(_safe_bool_setting('optimizer_objective_val_probe_enabled', True))}",
            f"val_probe_n:{_safe_int_setting('optimizer_objective_val_probe_sample_size', 480)}",
            f"val_probe_slices:{_safe_int_setting('optimizer_objective_val_probe_slices', 3)}",
            f"val_probe_loss_mult:{_safe_float_setting('optimizer_objective_val_probe_loss_mult', 1.0):.4f}",
            f"val_probe_wdrop_mult:{_safe_float_setting('optimizer_objective_val_probe_winner_drop_mult', 0.35):.4f}",
            f"udog_conf_edge:{int(_safe_bool_setting('underdog_confidence_use_edge_logistic', True))}",
            f"udog_conf_scale:{_safe_float_setting('underdog_confidence_edge_scale', 70.0):.4f}",
            f"rolling_cv:{int(_safe_bool_setting('optimizer_rolling_cv_enabled', True))}",
            f"rolling_cv_folds:{_safe_int_setting('optimizer_rolling_cv_folds', 4)}",
            f"rolling_cv_min_train:{_safe_int_setting('optimizer_rolling_cv_min_train_games', 320)}",
            f"rolling_cv_val_games:{_safe_int_setting('optimizer_rolling_cv_val_games', 160)}",
            f"rolling_cv_worst_mult:{_safe_float_setting('optimizer_rolling_cv_worst_fold_mult', 0.40):.4f}",
            f"objective_track:{str(get_setting('optimizer_objective_primary_track', 'dual_track'))}",
            f"objective_dual_live_w:{_safe_float_setting('optimizer_objective_dual_live_weight', 0.70):.4f}",
            f"tank_mode:{str(get_setting('optimizer_tanking_feature_mode', 'both'))}",
            f"wide_ranges:{int(_safe_bool_setting('optimizer_use_wide_ranges', False))}",
            f"tuning_mode:{str(get_setting('optimizer_tuning_mode', 'classic'))}",
            f"family_pen:{int(_safe_bool_setting('optimizer_objective_use_family_dominance_penalty', True))}",
            f"family_pen_mult:{_safe_float_setting('optimizer_objective_family_penalty_mult', 0.05):.4f}",
            f"ff_p95_cap:{_safe_float_setting('optimizer_objective_ff_p95_cap', 95.0):.4f}",
            f"onoff_p95_cap:{_safe_float_setting('optimizer_objective_onoff_p95_cap', 25.0):.4f}",
            f"l2_prior_mult:{_safe_float_setting('optimizer_objective_l2_prior_mult', 0.02):.4f}",
            f"onoff_player_smooth:{_safe_float_setting('optimizer_onoff_player_minutes_smoothing', 800.0):.4f}",
            f"onoff_team_slots:{_safe_float_setting('optimizer_onoff_team_reliability_slots', 12.0):.4f}",
        ]
    )


def _build_rolling_time_folds(
    n_games: int,
    fold_count: int,
    min_train_games: int,
    val_games: int,
) -> List[Tuple[int, int]]:
    """Return expanding-train rolling folds as (train_end, val_end) indices."""
    total = max(0, int(n_games))
    if total <= 0:
        return []

    min_train = max(40, int(min_train_games))
    val_len = max(20, int(val_games))
    if min_train + val_len > total:
        return []

    max_train_end = total - val_len
    if max_train_end < min_train:
        return []

    folds = max(1, int(fold_count))
    if folds == 1:
        return [(max_train_end, max_train_end + val_len)]

    start = min_train
    end = max_train_end
    if end <= start:
        return [(max_train_end, max_train_end + val_len)]

    train_ends: List[int] = []
    for i in range(folds):
        frac = i / float(max(1, folds - 1))
        train_end = int(round(start + (end - start) * frac))
        train_end = max(min_train, min(max_train_end, train_end))
        if not train_ends or train_end != train_ends[-1]:
            train_ends.append(train_end)

    out: List[Tuple[int, int]] = []
    for train_end in train_ends:
        val_end = train_end + val_len
        if train_end >= min_train and val_end <= total:
            out.append((train_end, val_end))

    if not out:
        out.append((max_train_end, max_train_end + val_len))
    return out


def _rolling_cv_objective_loss(
    fold_losses: List[float],
    worst_fold_mult: float,
) -> Tuple[float, float, float]:
    """Return (robust_score, mean_loss, worst_loss)."""
    if not fold_losses:
        return float("inf"), float("inf"), float("inf")
    mean_loss = float(np.mean(fold_losses))
    worst_loss = float(np.max(fold_losses))
    worst_excess = max(0.0, worst_loss - mean_loss)
    robust_score = mean_loss + worst_excess * max(0.0, float(worst_fold_mult))
    return robust_score, mean_loss, worst_loss


def _normalize_objective_track(raw: Any) -> str:
    """Normalize objective track mode to live/oracle/dual_track."""
    mode = str(raw or "dual_track").strip().lower()
    if mode in {"live", "oracle", "dual_track"}:
        return mode
    if mode in {"dual", "both"}:
        return "dual_track"
    return "dual_track"


def _compose_objective_loss(
    live_loss: float,
    oracle_loss: float,
    objective_track: str,
    dual_live_weight: float,
) -> float:
    """Combine live/oracle objective losses based on configured track mode."""
    if objective_track == "live":
        return float(live_loss)
    if objective_track == "oracle":
        return float(oracle_loss)
    live_w = float(np.clip(dual_live_weight, 0.0, 1.0))
    oracle_w = 1.0 - live_w
    return float(live_w * float(live_loss) + oracle_w * float(oracle_loss))


def _max_weight_delta(a: WeightConfig, b: WeightConfig) -> float:
    """Return max absolute per-parameter delta between two weight configs."""
    da = a.to_dict()
    db = b.to_dict()
    keys = set(da.keys()) & set(db.keys())
    if not keys:
        return float("inf")
    return max(abs(float(da[k]) - float(db[k])) for k in keys)


def _shrunk_rate_pct(
    successes: int,
    attempts: int,
    prior_pct: float,
    prior_weight: float,
) -> float:
    """Return empirical rate shrunk toward prior_pct by prior_weight."""
    if attempts <= 0:
        return float(np.clip(prior_pct, 0.0, 100.0))
    prior_prob = float(np.clip(prior_pct, 0.0, 100.0)) / 100.0
    weight = max(0.0, float(prior_weight))
    shrunk = (float(successes) + prior_prob * weight) / (float(attempts) + weight)
    return float(np.clip(shrunk * 100.0, 0.0, 100.0))


def _weighted_onepos_credit_count(
    all_dog_near_miss_count: int,
    long_dog_near_miss_count: int,
    all_dogs_weight: float,
    long_dogs_weight: float,
) -> float:
    """Return weighted near-miss credit count for upset picks."""
    all_count = max(0, int(all_dog_near_miss_count))
    long_count = max(0, min(all_count, int(long_dog_near_miss_count)))
    non_long_count = max(0, all_count - long_count)
    all_weight = max(0.0, float(all_dogs_weight))
    long_weight = max(0.0, float(long_dogs_weight))
    return float(non_long_count) * all_weight + float(long_count) * long_weight


def _passes_robust_save_gate(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    n_validation_games: int,
    baseline_all: Optional[Dict[str, float]] = None,
    candidate_all: Optional[Dict[str, float]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Gate saves using robust loss + upset quality (+ optional ROI)."""
    loss_margin = _safe_float_setting("optimizer_save_loss_margin", 0.01)
    winner_drop_tol = _safe_float_setting("optimizer_save_max_winner_drop", 0.35)
    favorites_slack = _safe_float_setting("optimizer_save_favorites_slack", 0.25)
    compression_floor = _safe_float_setting(
        "optimizer_save_compression_floor",
        COMPRESSION_RATIO_FLOOR,
    )

    min_upset_count_default = max(20, int(n_validation_games * 0.06))
    min_upset_count_cfg = _safe_int_setting(
        "optimizer_save_min_upset_count",
        min_upset_count_default,
    )
    min_upset_count = (
        min_upset_count_default if min_upset_count_cfg <= 0 else min_upset_count_cfg
    )
    min_upset_rate = _safe_float_setting("optimizer_save_min_upset_rate", 8.0)
    max_upset_rate = _safe_float_setting("optimizer_save_max_upset_rate", 55.0)
    upset_prior_weight = _safe_float_setting("optimizer_save_upset_prior_weight", 25.0)
    min_shrunk_upset_lift = _safe_float_setting(
        "optimizer_save_min_shrunk_upset_lift",
        0.40,
    )

    min_ml_bets_default = max(40, int(n_validation_games * 0.12))
    min_ml_bets_cfg = _safe_int_setting(
        "optimizer_save_min_ml_bets",
        min_ml_bets_default,
    )
    min_ml_bets = min_ml_bets_default if min_ml_bets_cfg <= 0 else min_ml_bets_cfg
    min_roi_lift = _safe_float_setting("optimizer_save_min_roi_lift", 0.15)
    roi_lb95_slack = _safe_float_setting("optimizer_save_roi_lb95_slack", 0.35)
    use_roi_gate = _safe_bool_setting("optimizer_save_use_roi_gate", False)
    use_hybrid_loss_gate = _safe_bool_setting("optimizer_save_use_hybrid_loss_gate", True)
    hybrid_val_weight = float(
        np.clip(_safe_float_setting("optimizer_save_hybrid_val_weight", 0.70), 0.0, 1.0)
    )
    hybrid_margin = _safe_float_setting("optimizer_save_hybrid_margin", loss_margin)
    max_val_loss_regress = _safe_float_setting("optimizer_save_max_val_loss_regress", 0.02)
    use_long_dog_tiebreak_gate = _safe_bool_setting(
        "optimizer_save_use_long_dog_tiebreak_gate",
        True,
    )
    long_dog_min_count_default = max(20, int(n_validation_games * 0.03))
    long_dog_min_count_cfg = _safe_int_setting(
        "optimizer_save_long_dog_min_count",
        long_dog_min_count_default,
    )
    long_dog_min_count = (
        long_dog_min_count_default
        if long_dog_min_count_cfg <= 0
        else long_dog_min_count_cfg
    )
    long_dog_prior_weight = _safe_float_setting(
        "optimizer_save_long_dog_prior_weight",
        25.0,
    )
    min_long_dog_onepos_lift = _safe_float_setting(
        "optimizer_save_min_long_dog_onepos_lift",
        0.75,
    )
    long_dog_tiebreak_loss_window = _safe_float_setting(
        "optimizer_save_long_dog_tiebreak_loss_window",
        0.01,
    )
    onepos_credit_bump_mult = max(
        0.0,
        _safe_float_setting("optimizer_save_onepos_credit_bump_mult", 0.15),
    )

    baseline_loss = float(baseline.get("loss", float("inf")))
    candidate_loss = float(candidate.get("loss", float("inf")))
    val_loss_improved = candidate_loss <= (baseline_loss - loss_margin)
    val_not_much_worse = candidate_loss <= (baseline_loss + max_val_loss_regress)

    has_all_loss = (
        baseline_all is not None
        and candidate_all is not None
        and "loss" in baseline_all
        and "loss" in candidate_all
    )
    baseline_all_loss = float(baseline_all.get("loss")) if has_all_loss else baseline_loss
    candidate_all_loss = float(candidate_all.get("loss")) if has_all_loss else candidate_loss
    baseline_hybrid_loss = (
        hybrid_val_weight * baseline_loss + (1.0 - hybrid_val_weight) * baseline_all_loss
    )
    candidate_hybrid_loss = (
        hybrid_val_weight * candidate_loss + (1.0 - hybrid_val_weight) * candidate_all_loss
    )
    hybrid_loss_improved = candidate_hybrid_loss <= (baseline_hybrid_loss - hybrid_margin)

    if use_hybrid_loss_gate and has_all_loss:
        loss_improved = (val_loss_improved or hybrid_loss_improved) and val_not_much_worse
    else:
        loss_improved = val_loss_improved

    baseline_winner_pct = float(baseline.get("winner_pct", 0.0))
    candidate_winner_pct = float(candidate.get("winner_pct", 0.0))
    baseline_winner_pct_raw = float(baseline.get("winner_pct_raw", baseline_winner_pct))
    candidate_winner_pct_raw = float(
        candidate.get("winner_pct_raw", candidate_winner_pct)
    )
    baseline_winner_pct_credit = float(
        baseline.get("winner_pct_credit", baseline_winner_pct_raw)
    )
    candidate_winner_pct_credit = float(
        candidate.get("winner_pct_credit", candidate_winner_pct_raw)
    )
    baseline_winner_credit_delta = float(
        baseline.get(
            "winner_pct_credit_delta",
            baseline_winner_pct_credit - baseline_winner_pct_raw,
        )
    )
    candidate_winner_credit_delta = float(
        candidate.get(
            "winner_pct_credit_delta",
            candidate_winner_pct_credit - candidate_winner_pct_raw,
        )
    )
    favorites_pct = float(candidate.get("favorites_pct", 0.0))

    compression_ratio = float(candidate.get("compression_ratio", 0.0))
    compression_ok = compression_ratio >= compression_floor

    upset_count = int(candidate.get("upset_count", 0))
    upset_rate = float(candidate.get("upset_rate", 0.0))
    upset_correct_count = int(candidate.get("upset_correct_count", 0))
    underdog_base_pct = float(np.clip(100.0 - favorites_pct, 0.0, 100.0))
    shrunk_upset_accuracy = _shrunk_rate_pct(
        upset_correct_count,
        upset_count,
        underdog_base_pct,
        upset_prior_weight,
    )
    shrunk_upset_lift = shrunk_upset_accuracy - underdog_base_pct
    upset_sample_ok = upset_count >= min_upset_count
    upset_rate_ok = min_upset_rate <= upset_rate <= max_upset_rate
    # When upset signal is stable, allow more room below baseline/favorites.
    # This supports underdog-oriented strategies that trade raw hit-rate
    # for higher-value upset selection.
    if upset_sample_ok and upset_rate_ok:
        relax_lift = max(0.0, shrunk_upset_lift - min_shrunk_upset_lift)
    else:
        relax_lift = 0.0
    onepos_credit_lift = candidate_winner_credit_delta - baseline_winner_credit_delta
    onepos_credit_bump_enabled = onepos_credit_bump_mult > 0.0
    onepos_credit_bump_raw = (
        max(0.0, onepos_credit_lift) * onepos_credit_bump_mult
        if onepos_credit_bump_enabled
        else 0.0
    )
    onepos_credit_bump_cap = max(0.0, min(6.0, winner_drop_tol * 4.0))
    onepos_credit_bump = min(onepos_credit_bump_raw, onepos_credit_bump_cap)
    adaptive_winner_drop_tol = min(
        winner_drop_tol + relax_lift * 0.35 + onepos_credit_bump,
        max(4.0, winner_drop_tol) + onepos_credit_bump_cap,
    )
    adaptive_favorites_slack = min(
        favorites_slack + relax_lift * 1.25 + onepos_credit_bump * 1.25,
        max(12.0, favorites_slack) + onepos_credit_bump_cap * 1.25,
    )
    winner_guard = candidate_winner_pct >= (baseline_winner_pct - adaptive_winner_drop_tol)
    favorites_guard = candidate_winner_pct >= (favorites_pct - adaptive_favorites_slack)
    upset_edge_ok = upset_sample_ok and upset_rate_ok and (
        shrunk_upset_lift >= min_shrunk_upset_lift
    )

    ml_bet_count = int(candidate.get("ml_bet_count", 0))
    baseline_ml_roi = float(baseline.get("ml_roi", 0.0))
    candidate_ml_roi = float(candidate.get("ml_roi", 0.0))
    candidate_ml_roi_lb95 = float(candidate.get("ml_roi_lb95", candidate_ml_roi))
    roi_sample_ok = ml_bet_count >= min_ml_bets
    roi_lift = candidate_ml_roi - baseline_ml_roi
    roi_edge_ok = roi_sample_ok and roi_lift >= min_roi_lift and (
        candidate_ml_roi_lb95 >= (baseline_ml_roi - roi_lb95_slack)
    )

    baseline_long_dog_count = int(baseline.get("long_dog_count", 0))
    candidate_long_dog_count = int(candidate.get("long_dog_count", 0))
    baseline_long_dog_onepos_count = int(baseline.get("long_dog_onepos_count", 0))
    candidate_long_dog_onepos_count = int(candidate.get("long_dog_onepos_count", 0))
    baseline_long_dog_onepos_rate = float(baseline.get("long_dog_onepos_rate", 0.0))
    candidate_long_dog_onepos_rate = float(candidate.get("long_dog_onepos_rate", 0.0))
    candidate_long_dog_onepos_shrunk = _shrunk_rate_pct(
        candidate_long_dog_onepos_count,
        candidate_long_dog_count,
        baseline_long_dog_onepos_rate,
        long_dog_prior_weight,
    )
    long_dog_onepos_lift = (
        candidate_long_dog_onepos_shrunk - baseline_long_dog_onepos_rate
    )
    long_dog_sample_ok = (
        baseline_long_dog_count >= long_dog_min_count
        and candidate_long_dog_count >= long_dog_min_count
    )
    long_dog_edge_ok = (
        long_dog_sample_ok and long_dog_onepos_lift >= min_long_dog_onepos_lift
    )

    # Upset is the primary signal. ROI can be optionally enabled as a hard gate.
    # When ROI gate is off, keep ROI diagnostics but do not block saves on ROI.
    if use_roi_gate and roi_sample_ok and upset_sample_ok:
        edge_ok = roi_edge_ok and upset_edge_ok
    elif use_roi_gate:
        edge_ok = roi_edge_ok or upset_edge_ok
    elif upset_sample_ok:
        edge_ok = upset_edge_ok
    else:
        # If upset sample is still too small, don't block solely on edge gate.
        edge_ok = True

    core_guards_ok = winner_guard and favorites_guard and compression_ok and edge_ok
    near_val_loss_ok = candidate_loss <= (baseline_loss + long_dog_tiebreak_loss_window)
    near_hybrid_loss_ok = (
        candidate_hybrid_loss <= (baseline_hybrid_loss + long_dog_tiebreak_loss_window)
        if (use_hybrid_loss_gate and has_all_loss)
        else True
    )
    long_dog_tiebreak_loss_ok = near_val_loss_ok and near_hybrid_loss_ok
    long_dog_tiebreak_ok = (
        use_long_dog_tiebreak_gate
        and not loss_improved
        and core_guards_ok
        and long_dog_tiebreak_loss_ok
        and long_dog_edge_ok
    )
    save_ok = (loss_improved and core_guards_ok) or long_dog_tiebreak_ok

    reasons: List[str] = []
    if not loss_improved:
        if use_hybrid_loss_gate and has_all_loss:
            reasons.append(
                f"loss gate failed (val {candidate_loss:.3f} vs {baseline_loss:.3f}, "
                f"delta {candidate_loss - baseline_loss:+.3f}; "
                f"hybrid {candidate_hybrid_loss:.3f} vs {baseline_hybrid_loss:.3f}, "
                f"delta {candidate_hybrid_loss - baseline_hybrid_loss:+.3f}; "
                f"lower loss is better)"
            )
            if not (val_loss_improved or hybrid_loss_improved):
                reasons.append(
                    f"needs val improve >= {loss_margin:.4f} "
                    f"or hybrid improve >= {hybrid_margin:.4f}"
                )
            if not val_not_much_worse:
                reasons.append(
                    f"validation loss regression too large "
                    f"({candidate_loss - baseline_loss:+.3f} > +{max_val_loss_regress:.4f})"
                )
        else:
            reasons.append(
                f"loss {candidate_loss:.3f} must improve baseline {baseline_loss:.3f} "
                f"by >= {loss_margin:.4f}"
            )
    if not winner_guard:
        reasons.append(
            f"winner% dropped too far ({candidate_winner_pct:.1f}% vs "
            f"baseline {baseline_winner_pct:.1f}%, tol {adaptive_winner_drop_tol:.2f})"
        )
    if not favorites_guard:
        reasons.append(
            f"winner% below favorites guard ({candidate_winner_pct:.1f}% < "
            f"{favorites_pct - adaptive_favorites_slack:.1f}%)"
        )
    if not compression_ok:
        reasons.append(
            f"compression {compression_ratio:.3f} < floor {compression_floor:.3f}"
        )
    if not edge_ok:
        if use_roi_gate:
            reasons.append(
                "edge gate failed "
                f"(ROI lift {roi_lift:+.2f}pp, ROI lb95 {candidate_ml_roi_lb95:+.2f}%, "
                f"shrunk upset lift {shrunk_upset_lift:+.2f}pp)"
            )
            if not roi_sample_ok:
                reasons.append(f"ML sample too small ({ml_bet_count} < {min_ml_bets})")
            if not upset_sample_ok:
                reasons.append(f"upset sample too small ({upset_count} < {min_upset_count})")
            elif not upset_rate_ok:
                reasons.append(
                    f"upset rate {upset_rate:.1f}% outside "
                    f"[{min_upset_rate:.1f}, {max_upset_rate:.1f}]"
                )
        else:
            reasons.append(
                "upset edge gate failed "
                f"(shrunk upset lift {shrunk_upset_lift:+.2f}pp < "
                f"min {min_shrunk_upset_lift:+.2f}pp)"
            )
            if not upset_rate_ok:
                reasons.append(
                    f"upset rate {upset_rate:.1f}% outside "
                    f"[{min_upset_rate:.1f}, {max_upset_rate:.1f}]"
                )
    if not save_ok and not loss_improved and use_long_dog_tiebreak_gate:
        if not long_dog_tiebreak_loss_ok:
            if not near_val_loss_ok:
                reasons.append(
                    f"long-dog tiebreak loss window failed "
                    f"({candidate_loss - baseline_loss:+.3f} > +{long_dog_tiebreak_loss_window:.3f})"
                )
            if use_hybrid_loss_gate and has_all_loss and not near_hybrid_loss_ok:
                reasons.append(
                    "long-dog tiebreak hybrid window failed "
                    f"({candidate_hybrid_loss - baseline_hybrid_loss:+.3f} > "
                    f"+{long_dog_tiebreak_loss_window:.3f})"
                )
        if not long_dog_sample_ok:
            reasons.append(
                f"long-dog sample too small "
                f"(baseline {baseline_long_dog_count}, candidate {candidate_long_dog_count}, "
                f"need >= {long_dog_min_count})"
            )
        elif not long_dog_edge_ok:
            reasons.append(
                "long-dog one-possession lift too small "
                f"({long_dog_onepos_lift:+.2f}pp < +{min_long_dog_onepos_lift:.2f}pp)"
            )

    details: Dict[str, Any] = {
        "ml_min_payout": float(candidate.get("ml_min_payout", MIN_ML_PAYOUT)),
        "loss_margin": loss_margin,
        "use_hybrid_loss_gate": use_hybrid_loss_gate,
        "hybrid_val_weight": hybrid_val_weight,
        "hybrid_margin": hybrid_margin,
        "max_val_loss_regress": max_val_loss_regress,
        "has_all_loss": has_all_loss,
        "baseline_val_loss": baseline_loss,
        "candidate_val_loss": candidate_loss,
        "baseline_all_loss": baseline_all_loss if has_all_loss else None,
        "candidate_all_loss": candidate_all_loss if has_all_loss else None,
        "baseline_hybrid_loss": baseline_hybrid_loss if has_all_loss else None,
        "candidate_hybrid_loss": candidate_hybrid_loss if has_all_loss else None,
        "val_loss_improved": val_loss_improved,
        "hybrid_loss_improved": hybrid_loss_improved if has_all_loss else False,
        "val_not_much_worse": val_not_much_worse,
        "winner_drop_tol": winner_drop_tol,
        "favorites_slack": favorites_slack,
        "adaptive_winner_drop_tol": adaptive_winner_drop_tol,
        "adaptive_favorites_slack": adaptive_favorites_slack,
        "onepos_credit_bump_mult": onepos_credit_bump_mult,
        "onepos_credit_bump_enabled": onepos_credit_bump_enabled,
        "onepos_credit_bump_cap": onepos_credit_bump_cap,
        "onepos_credit_bump": onepos_credit_bump,
        "baseline_winner_pct_raw": baseline_winner_pct_raw,
        "candidate_winner_pct_raw": candidate_winner_pct_raw,
        "baseline_winner_pct_credit": baseline_winner_pct_credit,
        "candidate_winner_pct_credit": candidate_winner_pct_credit,
        "baseline_winner_credit_delta": baseline_winner_credit_delta,
        "candidate_winner_credit_delta": candidate_winner_credit_delta,
        "onepos_credit_lift": onepos_credit_lift,
        "upset_relax_lift": relax_lift,
        "compression_floor": compression_floor,
        "min_upset_count": min_upset_count,
        "min_upset_rate": min_upset_rate,
        "max_upset_rate": max_upset_rate,
        "upset_prior_weight": upset_prior_weight,
        "min_shrunk_upset_lift": min_shrunk_upset_lift,
        "min_ml_bets": min_ml_bets,
        "min_roi_lift": min_roi_lift,
        "roi_lb95_slack": roi_lb95_slack,
        "use_long_dog_tiebreak_gate": use_long_dog_tiebreak_gate,
        "long_dog_min_count": long_dog_min_count,
        "long_dog_prior_weight": long_dog_prior_weight,
        "min_long_dog_onepos_lift": min_long_dog_onepos_lift,
        "long_dog_tiebreak_loss_window": long_dog_tiebreak_loss_window,
        "baseline_long_dog_count": baseline_long_dog_count,
        "candidate_long_dog_count": candidate_long_dog_count,
        "baseline_long_dog_onepos_count": baseline_long_dog_onepos_count,
        "candidate_long_dog_onepos_count": candidate_long_dog_onepos_count,
        "baseline_long_dog_onepos_rate": baseline_long_dog_onepos_rate,
        "candidate_long_dog_onepos_rate": candidate_long_dog_onepos_rate,
        "candidate_long_dog_onepos_shrunk": candidate_long_dog_onepos_shrunk,
        "long_dog_onepos_lift": long_dog_onepos_lift,
        "long_dog_sample_ok": long_dog_sample_ok,
        "long_dog_edge_ok": long_dog_edge_ok,
        "near_val_loss_ok": near_val_loss_ok,
        "near_hybrid_loss_ok": near_hybrid_loss_ok,
        "long_dog_tiebreak_loss_ok": long_dog_tiebreak_loss_ok,
        "long_dog_tiebreak_ok": long_dog_tiebreak_ok,
        "long_dog_min_payout": float(candidate.get("long_dog_min_payout", LONG_DOG_MIN_PAYOUT)),
        "long_dog_onepos_margin": float(
            candidate.get("long_dog_onepos_margin", ONE_POSSESSION_DOG_MARGIN)
        ),
        "loss_improved": loss_improved,
        "core_guards_ok": core_guards_ok,
        "winner_guard": winner_guard,
        "favorites_guard": favorites_guard,
        "compression_ok": compression_ok,
        "roi_sample_ok": roi_sample_ok,
        "roi_edge_ok": roi_edge_ok,
        "use_roi_gate": use_roi_gate,
        "upset_sample_ok": upset_sample_ok,
        "upset_rate_ok": upset_rate_ok,
        "upset_edge_ok": upset_edge_ok,
        "edge_ok": edge_ok,
        "roi_lift": roi_lift,
        "candidate_ml_roi_lb95": candidate_ml_roi_lb95,
        "shrunk_upset_accuracy": shrunk_upset_accuracy,
        "shrunk_upset_lift": shrunk_upset_lift,
    }
    if save_ok and long_dog_tiebreak_ok and not loss_improved:
        reason = "pass (long-dog one-possession tiebreak)"
    else:
        reason = "pass" if save_ok else "; ".join(reasons)
    return save_ok, reason, details


class VectorizedGames:
    """Converts List[GameInput] into flat NumPy arrays for fast loss evaluation."""

    def __init__(self, games: List[GameInput]):
        n = len(games)
        self.n = n

        # Projected points (from player projections)
        self.home_pts_raw = np.array([g.home_proj.get("points", 0) for g in games])
        self.away_pts_raw = np.array([g.away_proj.get("points", 0) for g in games])

        # Defensive factors
        self.home_def_factor_raw = np.array([g.home_def_factor_raw for g in games])
        self.away_def_factor_raw = np.array([g.away_def_factor_raw for g in games])

        # Fatigue flags (bool -> 1.0/0.0)
        self.home_b2b_flag = np.array([1.0 if g.home_b2b else 0.0 for g in games])
        self.away_b2b_flag = np.array([1.0 if g.away_b2b else 0.0 for g in games])
        self.home_3in4 = np.array([1.0 if g.home_3in4 else 0.0 for g in games])
        self.away_3in4 = np.array([1.0 if g.away_3in4 else 0.0 for g in games])
        self.home_4in6 = np.array([1.0 if g.home_4in6 else 0.0 for g in games])
        self.away_4in6 = np.array([1.0 if g.away_4in6 else 0.0 for g in games])
        self.home_same_day = np.array([1.0 if g.home_same_day else 0.0 for g in games])
        self.away_same_day = np.array([1.0 if g.away_same_day else 0.0 for g in games])

        # Rest bonus tiers: rest_days >= 4 -> 1.5, >= 3 -> 1.0, else 0
        self.home_rest_tier = np.array([
            1.5 if g.home_rest_days >= 4 else 1.0 if g.home_rest_days >= 3 else 0.0
            for g in games
        ])
        self.away_rest_tier = np.array([
            1.5 if g.away_rest_days >= 4 else 1.0 if g.away_rest_days >= 3 else 0.0
            for g in games
        ])

        # Home court advantage
        self.home_court = np.array([g.home_court for g in games])

        # Turnover differential: away_to - home_to (positive = home advantage)
        self.to_diff = np.array([
            g.away_proj.get("turnovers", 0) - g.home_proj.get("turnovers", 0)
            for g in games
        ])

        # Rebound differential: home_reb - away_reb (positive = home advantage)
        self.reb_diff = np.array([
            g.home_proj.get("rebounds", 0) - g.away_proj.get("rebounds", 0)
            for g in games
        ])

        # Team ratings
        self.home_off = np.array([g.home_off for g in games])
        self.away_off = np.array([g.away_off for g in games])
        self.home_def = np.array([g.home_def for g in games])
        self.away_def = np.array([g.away_def for g in games])

        # Four Factors edges (offensive)
        self.ff_efg_edge = np.array([
            g.home_ff.get("efg", 0) - g.away_ff.get("efg", 0) for g in games
        ])
        self.ff_tov_edge = np.array([
            g.away_ff.get("tov", 0) - g.home_ff.get("tov", 0) for g in games
        ])
        self.ff_oreb_edge = np.array([
            g.home_ff.get("oreb", 0) - g.away_ff.get("oreb", 0) for g in games
        ])
        self.ff_fta_edge = np.array([
            g.home_ff.get("fta", 0) - g.away_ff.get("fta", 0) for g in games
        ])

        # Opponent Four Factors edges (defensive matchup)
        self.opp_ff_efg_edge = np.array([
            g.away_ff.get("opp_efg", 0) - g.home_ff.get("opp_efg", 0) for g in games
        ])
        self.opp_ff_tov_edge = np.array([
            g.home_ff.get("opp_tov", 0) - g.away_ff.get("opp_tov", 0) for g in games
        ])
        self.opp_ff_oreb_edge = np.array([
            g.away_ff.get("opp_oreb", 0) - g.home_ff.get("opp_oreb", 0) for g in games
        ])
        self.opp_ff_fta_edge = np.array([
            g.away_ff.get("opp_fta", 0) - g.home_ff.get("opp_fta", 0) for g in games
        ])

        # Clutch differential
        self.clutch_diff = np.array([
            g.home_clutch.get("net_rating", 0) - g.away_clutch.get("net_rating", 0)
            for g in games
        ])

        # Hustle — raw components (evaluate() combines with hustle_contested_wt)
        self.home_defl = np.array([g.home_hustle.get("deflections", 0) for g in games])
        self.away_defl = np.array([g.away_hustle.get("deflections", 0) for g in games])
        self.home_contested = np.array([g.home_hustle.get("contested", 0) for g in games])
        self.away_contested = np.array([g.away_hustle.get("contested", 0) for g in games])

        # Combined stats for total adjustment (diagnostic)
        self.combined_steals = np.array([
            g.home_proj.get("steals", 0) + g.away_proj.get("steals", 0)
            for g in games
        ])
        self.combined_blocks = np.array([
            g.home_proj.get("blocks", 0) + g.away_proj.get("blocks", 0)
            for g in games
        ])
        self.combined_deflections = np.array([
            g.home_hustle.get("deflections", 0) + g.away_hustle.get("deflections", 0)
            for g in games
        ])

        # Rest / altitude
        self.net_rest = np.array([
            g.home_rest_days - g.away_rest_days for g in games
        ], dtype=float)
        self.away_b2b_at_altitude = np.array([
            1.0 if g.away_b2b and g.home_team_id in (1610612743, 1610612762) else 0.0
            for g in games
        ])

        # Actual results
        self.actual_spread = np.array([
            g.actual_home_score - g.actual_away_score for g in games
        ])
        self.actual_total = np.array([
            g.actual_home_score + g.actual_away_score for g in games
        ])

        # Vegas lines (for upset identification, NOT used in prediction formula)
        self.vegas_spread = np.array([g.vegas_spread for g in games])
        self.vegas_home_ml = np.array([g.vegas_home_ml for g in games])
        self.vegas_away_ml = np.array([g.vegas_away_ml for g in games])

        # Sharp ML edge: (ml_home_money - ml_home_public) / 100.0
        # Zeroed when either value is missing
        has_ml = np.array([
            bool(g.ml_home_public) and bool(g.ml_home_money) for g in games
        ])
        ml_home_money = np.array([g.ml_home_money for g in games], dtype=float)
        ml_home_public = np.array([g.ml_home_public for g in games], dtype=float)
        self.sharp_ml_edge = np.where(
            has_ml,
            (ml_home_money - ml_home_public) / 100.0,
            0.0,
        )

        # ── V2.1 feature arrays ──
        self.home_elo = np.array([g.home_elo for g in games])
        self.away_elo = np.array([g.away_elo for g in games])
        self.home_travel_miles = np.array([g.home_travel_miles for g in games])
        self.away_travel_miles = np.array([g.away_travel_miles for g in games])
        self.home_cum_travel_7d = np.array([g.home_cum_travel_7d for g in games])
        self.away_cum_travel_7d = np.array([g.away_cum_travel_7d for g in games])
        self.home_tz_crossings = np.array([g.home_tz_crossings for g in games], dtype=float)
        self.away_tz_crossings = np.array([g.away_tz_crossings for g in games], dtype=float)
        self.home_streak = np.array([g.home_streak for g in games], dtype=float)
        self.away_streak = np.array([g.away_streak for g in games], dtype=float)
        self.home_mov_trend = np.array([g.home_mov_trend for g in games])
        self.away_mov_trend = np.array([g.away_mov_trend for g in games])
        self.home_injury_vorp = np.array([g.home_injury_vorp_lost for g in games])
        self.away_injury_vorp = np.array([g.away_injury_vorp_lost for g in games])
        self.ref_crew_fouls_pg = np.array([g.ref_crew_fouls_pg for g in games])
        self.ref_crew_home_bias = np.array([g.ref_crew_home_bias for g in games])
        self.spread_sharp_edge = np.array([g.spread_sharp_edge for g in games])
        self.home_lookahead = np.array([float(g.home_lookahead) for g in games])
        self.away_lookahead = np.array([float(g.away_lookahead) for g in games])
        self.home_letdown = np.array([float(g.home_letdown) for g in games])
        self.away_letdown = np.array([float(g.away_letdown) for g in games])
        self.home_road_trip_game = np.array([g.home_road_trip_game for g in games], dtype=float)
        self.away_road_trip_game = np.array([g.away_road_trip_game for g in games], dtype=float)
        self.home_season_progress = np.array([g.home_season_progress for g in games], dtype=float)
        self.away_season_progress = np.array([g.away_season_progress for g in games], dtype=float)
        self.home_tank_signal_live = np.array([g.home_tank_signal_live for g in games], dtype=float)
        self.away_tank_signal_live = np.array([g.away_tank_signal_live for g in games], dtype=float)
        self.home_tank_signal_oracle = np.array([g.home_tank_signal_oracle for g in games], dtype=float)
        self.away_tank_signal_oracle = np.array([g.away_tank_signal_oracle for g in games], dtype=float)
        self.home_roster_shock = np.array([g.home_roster_shock for g in games], dtype=float)
        self.away_roster_shock = np.array([g.away_roster_shock for g in games], dtype=float)
        self.home_srs = np.array([g.home_srs for g in games])
        self.away_srs = np.array([g.away_srs for g in games])
        self.home_pythag_wpct = np.array([g.home_pythag_wpct for g in games], dtype=float)
        self.away_pythag_wpct = np.array([g.away_pythag_wpct for g in games], dtype=float)
        self.home_onoff = np.array([g.home_onoff_impact for g in games])
        self.away_onoff = np.array([g.away_onoff_impact for g in games])
        self.home_onoff_reliability = np.array(
            [g.home_onoff_reliability for g in games],
            dtype=float,
        )
        self.away_onoff_reliability = np.array(
            [g.away_onoff_reliability for g in games],
            dtype=float,
        )
        self.pace_diff = np.array([abs(g.home_pace - g.away_pace) for g in games])
        self.home_fg3_luck = np.array([g.home_fg3_luck for g in games])
        self.away_fg3_luck = np.array([g.away_fg3_luck for g in games])

        # Process stats matchup edges
        # paint_edge = (home_paint - away_opp_paint) - (away_paint - home_opp_paint)
        self._process_paint_edge = np.array([
            (g.home_process.get("paint", 0) - g.away_process.get("opp_paint", 0))
            - (g.away_process.get("paint", 0) - g.home_process.get("opp_paint", 0))
            for g in games
        ])
        self._process_fb_edge = np.array([
            (g.home_process.get("fb", 0) - g.away_process.get("opp_fb", 0))
            - (g.away_process.get("fb", 0) - g.home_process.get("opp_fb", 0))
            for g in games
        ])
        self._process_sec_edge = np.array([
            (g.home_process.get("sec", 0) - g.away_process.get("opp_sec", 0))
            - (g.away_process.get("sec", 0) - g.home_process.get("opp_sec", 0))
            for g in games
        ])
        self._process_tov_edge = np.array([
            (g.home_process.get("off_tov", 0) - g.away_process.get("opp_off_tov", 0))
            - (g.away_process.get("off_tov", 0) - g.home_process.get("opp_off_tov", 0))
            for g in games
        ])

        # Pre-compute constants used every evaluate() call
        self._actual_std = float(np.std(self.actual_spread))
        self._ref_fouls_centered = self.ref_crew_fouls_pg - 38.0
        self._ref_bias_centered = (self.ref_crew_home_bias - 50.0) / 50.0
        self._spread_sharp_scaled = self.spread_sharp_edge / 100.0
        self._elo_diff_scaled = (self.home_elo - self.away_elo) / 400.0
        self._travel_diff_scaled = (self.away_travel_miles - self.home_travel_miles) / 1000.0
        self._cum_travel_diff_scaled = (self.away_cum_travel_7d - self.home_cum_travel_7d) / 1000.0
        self._tz_diff = self.away_tz_crossings - self.home_tz_crossings
        self._streak_diff = (self.home_streak - self.away_streak).astype(float)
        self._mov_diff = self.home_mov_trend - self.away_mov_trend
        self._injury_diff = self.away_injury_vorp - self.home_injury_vorp
        self._road_trip_diff = self.away_road_trip_game - self.home_road_trip_game
        self._season_progress_diff = self.home_season_progress - self.away_season_progress
        self._tank_live_diff = self.away_tank_signal_live - self.home_tank_signal_live
        self._tank_oracle_diff = self.away_tank_signal_oracle - self.home_tank_signal_oracle
        self._roster_shock_diff = self.away_roster_shock - self.home_roster_shock
        self._srs_diff = self.home_srs - self.away_srs
        self._pythag_diff = self.home_pythag_wpct - self.away_pythag_wpct
        self._home_onoff_signal = self.home_onoff
        self._away_onoff_signal = self.away_onoff
        self._home_onoff_reliability = np.clip(self.home_onoff_reliability, 0.0, 1.0)
        self._away_onoff_reliability = np.clip(self.away_onoff_reliability, 0.0, 1.0)
        self._fg3_luck_diff = self.home_fg3_luck - self.away_fg3_luck
        self._process_total_edge = (self._process_paint_edge + self._process_fb_edge
                                    + self._process_sec_edge + self._process_tov_edge)

        upset_bonus_max = max(0.0, _safe_float_setting("upset_bonus_mult_max", 5.0))
        self._upset_bonus_mult = float(
            np.clip(_safe_float_setting("upset_bonus_mult", 0.5), 0.0, upset_bonus_max)
        )
        self._objective_target_upset_coverage_pct = max(
            0.0,
            _safe_float_setting("optimizer_objective_target_upset_coverage_pct", 20.0),
        )
        self._objective_target_tier_a_hit_rate = float(
            np.clip(
                _safe_float_setting("optimizer_objective_target_tier_a_hit_rate", 56.0),
                0.0,
                100.0,
            )
        )
        self._objective_coverage_shortfall_mult = max(
            0.0,
            _safe_float_setting("optimizer_objective_coverage_shortfall_mult", 0.20),
        )
        self._objective_tier_a_shortfall_mult = max(
            0.0,
            _safe_float_setting("optimizer_objective_tier_a_shortfall_mult", 0.12),
        )
        self._objective_tier_a_prior_weight = max(
            0.0,
            _safe_float_setting("optimizer_objective_tier_a_prior_weight", 12.0),
        )
        self._objective_tier_a_min_confidence = float(
            np.clip(
                _safe_float_setting(
                    "optimizer_objective_tier_a_min_confidence",
                    DEFAULT_TIER_A_MIN_CONFIDENCE,
                ),
                0.0,
                100.0,
            )
        )
        self._objective_tier_b_min_confidence = float(
            np.clip(
                _safe_float_setting(
                    "optimizer_objective_tier_b_min_confidence",
                    DEFAULT_TIER_B_MIN_CONFIDENCE,
                ),
                0.0,
                self._objective_tier_a_min_confidence,
            )
        )
        self._underdog_confidence_use_edge_logistic = _safe_bool_setting(
            "underdog_confidence_use_edge_logistic",
            True,
        )
        self._underdog_confidence_edge_scale = max(
            1.0,
            _safe_float_setting("underdog_confidence_edge_scale", 70.0),
        )
        self._competitive_dog_margin = max(
            0.0,
            _safe_float_setting("optimizer_competitive_dog_margin", 7.5),
        )
        self._onepos_credit_enabled = _safe_bool_setting(
            "optimizer_onepos_credit_enabled",
            True,
        )
        self._onepos_credit_margin = max(
            0.0,
            _safe_float_setting(
                "optimizer_onepos_credit_margin",
                ONE_POSSESSION_DOG_MARGIN,
            ),
        )
        self._onepos_credit_all_dogs_weight = max(
            0.0,
            _safe_float_setting("optimizer_onepos_credit_all_dogs_weight", 0.5),
        )
        self._onepos_credit_long_dogs_weight = max(
            0.0,
            _safe_float_setting("optimizer_onepos_credit_long_dogs_weight", 1.0),
        )
        self._onepos_credit_affects_winner_pct = _safe_bool_setting(
            "optimizer_onepos_credit_affects_winner_pct",
            True,
        )
        self._long_dog_min_payout = max(
            1.0,
            _safe_float_setting("optimizer_save_long_dog_min_payout", LONG_DOG_MIN_PAYOUT),
        )
        self._long_dog_onepos_margin = max(
            0.0,
            _safe_float_setting(
                "optimizer_save_long_dog_onepos_margin",
                ONE_POSSESSION_DOG_MARGIN,
            ),
        )
        self._min_ml_payout = max(
            1.0,
            _safe_float_setting("optimizer_min_ml_payout", MIN_ML_PAYOUT),
        )
        self._objective_use_family_dominance_penalty = _safe_bool_setting(
            "optimizer_objective_use_family_dominance_penalty",
            True,
        )
        self._objective_family_penalty_mult = max(
            0.0,
            _safe_float_setting("optimizer_objective_family_penalty_mult", 0.05),
        )
        self._objective_ff_p95_cap = max(
            0.0,
            _safe_float_setting("optimizer_objective_ff_p95_cap", 95.0),
        )
        self._objective_onoff_p95_cap = max(
            0.0,
            _safe_float_setting("optimizer_objective_onoff_p95_cap", 25.0),
        )
        self._tanking_feature_mode = str(
            get_setting("optimizer_tanking_feature_mode", "both")
        ).strip().lower()
        if self._tanking_feature_mode not in {"live", "oracle", "both"}:
            self._tanking_feature_mode = "both"
        self._tank_use_live = self._tanking_feature_mode in {"live", "both"}
        self._tank_use_oracle = self._tanking_feature_mode in {"oracle", "both"}

    def evaluate(self, w: WeightConfig, include_sharp: bool = False,
                 fast: bool = False) -> Dict[str, float]:
        """Vectorized evaluation. Returns metrics dict including loss.

        Formula mirrors predict() in prediction.py exactly, but operates on
        entire arrays at once for speed during optimization.

        When ``fast=True``, skip diagnostics not used by the loss function
        (ML ROI, spread MAE, favorites_pct, total score).  This is ~30%
        faster and intended for inner-loop Optuna trials.
        """
        # 1. Defensive adjustment (dampened)
        away_def_f = 1.0 + (self.away_def_factor_raw - 1.0) * w.def_factor_dampening
        home_def_f = 1.0 + (self.home_def_factor_raw - 1.0) * w.def_factor_dampening

        home_base = self.home_pts_raw * away_def_f
        away_base = self.away_pts_raw * home_def_f

        # 2. Base: home_base_pts - away_base_pts + HCA
        game_score = (home_base - away_base) + self.home_court

        # 3. Fatigue (decomposed — each component tunable)
        home_fat = (self.home_b2b_flag * w.fatigue_b2b
                    + self.home_3in4 * w.fatigue_3in4
                    + self.home_4in6 * w.fatigue_4in6
                    + self.home_same_day * w.fatigue_same_day
                    - self.home_rest_tier * w.fatigue_rest_bonus)
        away_fat = (self.away_b2b_flag * w.fatigue_b2b
                    + self.away_3in4 * w.fatigue_3in4
                    + self.away_4in6 * w.fatigue_4in6
                    + self.away_same_day * w.fatigue_same_day
                    - self.away_rest_tier * w.fatigue_rest_bonus)
        game_score -= (home_fat - away_fat)

        # 4. Turnover differential
        game_score += self.to_diff * w.turnover_margin_mult

        # 5. Rebound differential
        game_score += self.reb_diff * w.rebound_diff_mult

        # 6. Rating matchup
        home_me = self.home_off - self.away_def
        away_me = self.away_off - self.home_def
        game_score += (home_me - away_me) * w.rating_matchup_mult

        # 7. Four Factors (offensive)
        ff = (self.ff_efg_edge * w.ff_efg_weight
              + self.ff_tov_edge * w.ff_tov_weight
              + self.ff_oreb_edge * w.ff_oreb_weight
              + self.ff_fta_edge * w.ff_fta_weight) * FOUR_FACTORS_FIXED_SCALE
        game_score += ff

        # 8. Opponent Four Factors (defensive matchup)
        opp_ff = (self.opp_ff_efg_edge * w.opp_ff_efg_weight
                  + self.opp_ff_tov_edge * w.opp_ff_tov_weight
                  + self.opp_ff_oreb_edge * w.opp_ff_oreb_weight
                  + self.opp_ff_fta_edge * w.opp_ff_fta_weight) * FOUR_FACTORS_FIXED_SCALE
        game_score += opp_ff

        # 9. Clutch (masked: only applied when |game_score| < threshold)
        clutch_mask = np.abs(game_score) < w.clutch_threshold
        clutch_adj = np.clip(self.clutch_diff * w.clutch_scale, -w.clutch_cap, w.clutch_cap)
        game_score += clutch_adj * clutch_mask

        # 10. Hustle
        home_eff = self.home_defl + self.home_contested * w.hustle_contested_wt
        away_eff = self.away_defl + self.away_contested * w.hustle_contested_wt
        game_score += (home_eff - away_eff) * w.hustle_effort_mult

        # 11. Rest advantage (continuous)
        game_score += self.net_rest * w.rest_advantage_mult

        # 12. Altitude B2B penalty
        game_score -= self.away_b2b_at_altitude * w.altitude_b2b_penalty

        # 13. Sharp ML (optional toggle layer)
        if include_sharp:
            game_score += self.sharp_ml_edge * w.sharp_ml_weight

        # ── V2.1 vectorized adjustments (using pre-computed diffs) ──
        game_score += self._elo_diff_scaled * w.elo_diff_mult
        game_score -= self._travel_diff_scaled * w.travel_dist_mult
        game_score -= self._tz_diff * w.timezone_crossing_mult
        game_score += self._cum_travel_diff_scaled * w.cum_travel_7d_mult
        game_score += self._streak_diff * w.momentum_streak_mult
        game_score += self._mov_diff * w.mov_trend_mult
        game_score += self._injury_diff * w.injury_vorp_mult
        game_score += self._ref_bias_centered * w.ref_home_bias_mult
        game_score += self._spread_sharp_scaled * w.sharp_spread_weight
        game_score += (-(self.home_lookahead * w.lookahead_penalty + self.home_letdown * w.letdown_penalty)
                       + (self.away_lookahead * w.lookahead_penalty + self.away_letdown * w.letdown_penalty))
        game_score += self._road_trip_diff * w.road_trip_game_mult
        game_score += self._season_progress_diff * w.season_progress_mult
        game_score += self._roster_shock_diff * w.roster_shock_mult
        if self._tank_use_live:
            game_score += self._tank_live_diff * w.tank_live_mult
        if self._tank_use_oracle:
            game_score += self._tank_oracle_diff * w.tank_oracle_mult
        game_score += self._srs_diff * w.srs_diff_mult
        game_score += self._pythag_diff * w.pythag_diff_mult
        onoff_lambda = max(0.0, float(getattr(w, "onoff_reliability_lambda", 0.0)))
        home_onoff = self._home_onoff_signal * (
            self._home_onoff_reliability / (self._home_onoff_reliability + onoff_lambda + 1e-9)
        )
        away_onoff = self._away_onoff_signal * (
            self._away_onoff_reliability / (self._away_onoff_reliability + onoff_lambda + 1e-9)
        )
        onoff_adj = (home_onoff - away_onoff) * w.onoff_impact_mult
        game_score += onoff_adj
        game_score -= self._fg3_luck_diff * w.fg3_luck_mult  # negative: hot team regresses down
        game_score += self._process_total_edge * w.process_edge_mult

        # ──────────────────────────────────────────────────────────
        # TOTAL (projected combined score — diagnostic, skip in fast mode)
        # ──────────────────────────────────────────────────────────
        if not fast:
            total = home_base + away_base
            # Defensive disruption total adjustment
            total -= (np.maximum(0, self.combined_steals - w.steals_threshold) * w.steals_penalty +
                      np.maximum(0, self.combined_blocks - w.blocks_threshold) * w.blocks_penalty)
            # Hustle deflection total adjustment
            defl_over = np.maximum(0, self.combined_deflections - w.hustle_defl_baseline)
            total -= defl_over * w.hustle_defl_penalty
            # Fatigue total
            total -= (home_fat + away_fat) * w.fatigue_total_mult
            # V2.1 total adjustments
            total += self.pace_diff * w.pace_mismatch_mult
            total += self._ref_fouls_centered * w.ref_fouls_mult
            # Clamp total
            total = np.clip(total, w.total_min, w.total_max)

        # ──────────────────────────────────────────────────────────
        # METRICS
        # ──────────────────────────────────────────────────────────

        # Winner accuracy
        pred_home_win = game_score > MODEL_PICK_EDGE_THRESHOLD
        pred_away_win = game_score < -MODEL_PICK_EDGE_THRESHOLD
        actual_home_win = self.actual_spread > ACTUAL_WIN_THRESHOLD
        actual_away_win = self.actual_spread < -ACTUAL_WIN_THRESHOLD
        actual_push = np.abs(self.actual_spread) <= ACTUAL_WIN_THRESHOLD

        correct = ((pred_home_win & actual_home_win)
                    | (pred_away_win & actual_away_win)
                    | (actual_push & (np.abs(game_score) <= 3.0)))
        winner_pct_raw = float(np.mean(correct)) * 100.0

        # Vegas favorite direction (needed by both favorites_pct and upset detection)
        vegas_fav_home = self.vegas_spread < 0  # negative spread = home favored

        # Favorites baseline — how often does the Vegas favorite win?
        if not fast:
            actual_fav_won = ((vegas_fav_home & actual_home_win)
                              | (~vegas_fav_home & actual_away_win))
            non_push = ~actual_push
            n_non_push = int(np.sum(non_push))
            favorites_pct = (float(np.sum(actual_fav_won & non_push))
                             / max(1, n_non_push) * 100.0)
        else:
            favorites_pct = 0.0

        # Upset detection
        # Model picks home when game_score exceeds threshold.
        model_picks_home = game_score > MODEL_PICK_EDGE_THRESHOLD
        # Upset = model disagrees with Vegas on who wins
        model_picks_upset = model_picks_home != vegas_fav_home
        upset_correct = (model_picks_upset
                         & ((model_picks_home & actual_home_win)
                            | (~model_picks_home & actual_away_win)))
        upset_count = int(np.sum(model_picks_upset))
        upset_rate = float(upset_count) / max(1, self.n) * 100.0
        upset_correct_count = int(np.sum(upset_correct))
        upset_accuracy_raw = (
            float(upset_correct_count) / max(1, upset_count) * 100.0
        )
        dog_actual_margin = np.where(vegas_fav_home, -self.actual_spread, self.actual_spread)
        competitive_dog = model_picks_upset & (
            dog_actual_margin >= -self._competitive_dog_margin
        )
        competitive_dog_count = int(np.sum(competitive_dog))
        competitive_dog_rate = (
            float(competitive_dog_count) / max(1, upset_count) * 100.0
        )
        one_possession_dog = model_picks_upset & (
            dog_actual_margin >= -ONE_POSSESSION_DOG_MARGIN
        )
        one_possession_dog_count = int(np.sum(one_possession_dog))
        one_possession_dog_rate = (
            float(one_possession_dog_count) / max(1, upset_count) * 100.0
        )
        dog_ml_line = np.where(vegas_fav_home, self.vegas_away_ml, self.vegas_home_ml)
        dog_payout_mult = np.where(
            dog_ml_line < 0,
            1.0 + 100.0 / np.maximum(np.abs(dog_ml_line), 1),
            np.where(dog_ml_line > 0, 1.0 + dog_ml_line / 100.0, 0.0),
        )
        long_dog_pick = model_picks_upset & (dog_payout_mult >= self._long_dog_min_payout)
        long_dog_onepos = long_dog_pick & (
            dog_actual_margin >= -self._long_dog_onepos_margin
        )
        long_dog_count = int(np.sum(long_dog_pick))
        long_dog_onepos_count = int(np.sum(long_dog_onepos))
        long_dog_onepos_rate = (
            float(long_dog_onepos_count) / max(1, long_dog_count) * 100.0
        )
        onepos_credit_near_miss = (
            model_picks_upset
            & (~upset_correct)
            & (dog_actual_margin >= -self._onepos_credit_margin)
        )
        onepos_credit_near_miss_count = int(np.sum(onepos_credit_near_miss))
        onepos_credit_near_miss_rate = (
            float(onepos_credit_near_miss_count) / max(1, upset_count) * 100.0
        )
        long_dog_onepos_credit_near_miss = (
            long_dog_pick
            & (~upset_correct)
            & (dog_actual_margin >= -self._onepos_credit_margin)
        )
        long_dog_onepos_credit_near_miss_count = int(
            np.sum(long_dog_onepos_credit_near_miss)
        )
        long_dog_onepos_credit_near_miss_rate = (
            float(long_dog_onepos_credit_near_miss_count) / max(1, long_dog_count) * 100.0
        )
        onepos_credit_weighted_count = _weighted_onepos_credit_count(
            onepos_credit_near_miss_count,
            long_dog_onepos_credit_near_miss_count,
            self._onepos_credit_all_dogs_weight,
            self._onepos_credit_long_dogs_weight,
        )
        onepos_credit_weighted_rate = (
            float(onepos_credit_weighted_count) / max(1, upset_count) * 100.0
        )
        upset_accuracy_credit = float(
            np.clip(
                upset_accuracy_raw + onepos_credit_weighted_rate,
                0.0,
                100.0,
            )
        )
        upset_accuracy_credit_delta = upset_accuracy_credit - upset_accuracy_raw
        upset_accuracy = (
            upset_accuracy_credit
            if self._onepos_credit_enabled
            else upset_accuracy_raw
        )
        winner_pct_credit_delta = (
            float(onepos_credit_weighted_count) / max(1, self.n) * 100.0
        )
        winner_pct_credit = float(
            np.clip(
                winner_pct_raw + winner_pct_credit_delta,
                0.0,
                100.0,
            )
        )
        winner_pct = (
            winner_pct_credit
            if (self._onepos_credit_enabled and self._onepos_credit_affects_winner_pct)
            else winner_pct_raw
        )

        abs_game_score = np.abs(game_score)
        if self._underdog_confidence_use_edge_logistic:
            confidence = 100.0 * (
                1.0 - np.exp(-abs_game_score / self._underdog_confidence_edge_scale)
            )
        else:
            confidence = np.clip(abs_game_score / 15.0 * 100.0, 0.0, 100.0)
        tier_a_mask = model_picks_upset & (
            confidence >= self._objective_tier_a_min_confidence
        )
        tier_a_count_fast = int(np.sum(tier_a_mask))
        tier_a_correct_count_fast = int(np.sum(tier_a_mask & upset_correct))
        tier_a_hit_rate_fast = (
            float(tier_a_correct_count_fast) / max(1, tier_a_count_fast) * 100.0
        )
        tier_a_hit_rate_for_loss = _shrunk_rate_pct(
            tier_a_correct_count_fast,
            tier_a_count_fast,
            upset_accuracy if upset_count > 0 else 50.0,
            self._objective_tier_a_prior_weight,
        )
        coverage_shortfall = max(
            0.0,
            self._objective_target_upset_coverage_pct - upset_rate,
        )
        tier_a_shortfall = max(
            0.0,
            self._objective_target_tier_a_hit_rate - tier_a_hit_rate_for_loss,
        )
        objective_penalty = (
            coverage_shortfall * self._objective_coverage_shortfall_mult
            + tier_a_shortfall * self._objective_tier_a_shortfall_mult
        )

        quality_summary = summarize_underdog_quality(
            [],
            total_games=self.n,
            tier_a_min_confidence=self._objective_tier_a_min_confidence,
            tier_b_min_confidence=self._objective_tier_b_min_confidence,
            confidence_use_edge_logistic=self._underdog_confidence_use_edge_logistic,
            confidence_edge_scale=self._underdog_confidence_edge_scale,
        )
        if not fast and upset_count > 0:
            upset_idx = np.flatnonzero(model_picks_upset)
            upset_samples = []
            for idx in upset_idx:
                is_correct = bool(upset_correct[idx])
                payout = float(dog_payout_mult[idx])
                upset_samples.append(
                    {
                        "confidence": float(confidence[idx]),
                        "edge_abs": float(abs_game_score[idx]),
                        "upset_correct": is_correct,
                        "ml_profit": (payout - 1.0) if is_correct else -1.0,
                        "ml_payout": payout,
                    }
                )
            quality_summary = summarize_underdog_quality(
                upset_samples,
                total_games=self.n,
                tier_a_min_confidence=self._objective_tier_a_min_confidence,
                tier_b_min_confidence=self._objective_tier_b_min_confidence,
                confidence_use_edge_logistic=self._underdog_confidence_use_edge_logistic,
                confidence_edge_scale=self._underdog_confidence_edge_scale,
            )

        # ML ROI (diagnostic only — not in loss function, skip in fast mode)
        ml_roi = -4.54
        ml_win_rate = winner_pct_raw
        ml_bet_count = 0
        ml_roi_lb95 = ml_roi
        spread_mae = 0.0
        if not fast:
            ml_mask = (self.vegas_home_ml != 0) & (self.vegas_away_ml != 0)
            if np.any(ml_mask):
                h_ml = self.vegas_home_ml[ml_mask]
                a_ml = self.vegas_away_ml[ml_mask]
                p_score_ml = game_score[ml_mask]
                a_spread_ml = self.actual_spread[ml_mask]

                pick_home_ml = p_score_ml > 0
                actual_home_win_ml = a_spread_ml > 0

                # ML payout multipliers
                h_mult = np.where(h_ml < 0,
                                  1.0 + 100.0 / np.maximum(np.abs(h_ml), 1),
                                  1.0 + h_ml / 100.0)
                a_mult = np.where(a_ml < 0,
                                  1.0 + 100.0 / np.maximum(np.abs(a_ml), 1),
                                  1.0 + a_ml / 100.0)

                picked_mult = np.where(pick_home_ml, h_mult, a_mult)
                payout_ok = picked_mult >= self._min_ml_payout

                # Profit calculation
                h_profit = np.where(pick_home_ml & actual_home_win_ml, h_mult - 1.0,
                           np.where(pick_home_ml & ~actual_home_win_ml, -1.0, 0.0))
                a_profit = np.where(~pick_home_ml & ~actual_home_win_ml, a_mult - 1.0,
                           np.where(~pick_home_ml & actual_home_win_ml, -1.0, 0.0))
                total_profit = h_profit + a_profit

                if np.any(payout_ok):
                    ml_bet_count = int(np.sum(payout_ok))
                    bettable_correct = (((pick_home_ml & actual_home_win_ml)
                                         | (~pick_home_ml & ~actual_home_win_ml))
                                        & payout_ok)
                    ml_win_rate = float(np.sum(bettable_correct)) / max(1.0, float(ml_bet_count)) * 100.0
                    roi_samples = total_profit[payout_ok]
                    ml_roi = float(np.mean(roi_samples)) * 100.0
                    if ml_bet_count > 1:
                        roi_std = float(np.std(roi_samples, ddof=1)) * 100.0
                        roi_sem = roi_std / np.sqrt(float(ml_bet_count))
                        ml_roi_lb95 = ml_roi - (1.96 * roi_sem)
                    else:
                        ml_roi_lb95 = ml_roi

            # Spread MAE (diagnostic)
            spread_mae = float(np.mean(np.abs(game_score - self.actual_spread)))

        # Compression ratio
        pred_std = float(np.std(game_score))
        compression_ratio = pred_std / max(0.01, self._actual_std)

        # ──────────────────────────────────────────────────────────
        # LOSS FUNCTION
        # ──────────────────────────────────────────────────────────

        objective_base_score = (
            winner_pct + upset_accuracy * upset_rate / 100.0 * self._upset_bonus_mult
        )
        objective_score = objective_base_score - objective_penalty
        loss = -objective_score

        ff_total_adj = ff + opp_ff
        ff_p95_abs = float(np.percentile(np.abs(ff_total_adj), 95))
        onoff_p95_abs = float(np.percentile(np.abs(onoff_adj), 95))
        ff_p95_excess = 0.0
        onoff_p95_excess = 0.0
        family_penalty = 0.0
        if self._objective_use_family_dominance_penalty:
            ff_p95_excess = max(0.0, ff_p95_abs - self._objective_ff_p95_cap)
            onoff_p95_excess = max(0.0, onoff_p95_abs - self._objective_onoff_p95_cap)
            family_penalty = (
                ff_p95_excess + onoff_p95_excess
            ) * self._objective_family_penalty_mult
            loss += family_penalty

        # Compression penalty — prevent degenerate narrow-band predictions
        if compression_ratio < COMPRESSION_RATIO_FLOOR:
            loss += (COMPRESSION_RATIO_FLOOR - compression_ratio) * COMPRESSION_PENALTY_MULT

        return {
            "winner_pct": winner_pct,
            "winner_pct_raw": winner_pct_raw,
            "winner_pct_credit": winner_pct_credit,
            "winner_pct_credit_delta": winner_pct_credit_delta,
            "favorites_pct": favorites_pct,
            "upset_rate": upset_rate,
            "upset_accuracy": upset_accuracy,
            "upset_accuracy_raw": upset_accuracy_raw,
            "upset_accuracy_credit": upset_accuracy_credit,
            "upset_accuracy_credit_delta": upset_accuracy_credit_delta,
            "upset_correct_count": upset_correct_count,
            "upset_count": upset_count,
            "competitive_dog_rate": competitive_dog_rate,
            "competitive_dog_count": competitive_dog_count,
            "competitive_dog_margin": self._competitive_dog_margin,
            "one_possession_dog_rate": one_possession_dog_rate,
            "one_possession_dog_count": one_possession_dog_count,
            "one_possession_dog_margin": ONE_POSSESSION_DOG_MARGIN,
            "long_dog_count": long_dog_count,
            "long_dog_onepos_count": long_dog_onepos_count,
            "long_dog_onepos_rate": long_dog_onepos_rate,
            "long_dog_min_payout": self._long_dog_min_payout,
            "long_dog_onepos_margin": self._long_dog_onepos_margin,
            "onepos_credit_enabled": self._onepos_credit_enabled,
            "onepos_credit_affects_winner_pct": self._onepos_credit_affects_winner_pct,
            "onepos_credit_margin": self._onepos_credit_margin,
            "onepos_credit_all_dogs_weight": self._onepos_credit_all_dogs_weight,
            "onepos_credit_long_dogs_weight": self._onepos_credit_long_dogs_weight,
            "onepos_credit_all_dogs_near_miss_count": onepos_credit_near_miss_count,
            "onepos_credit_all_dogs_near_miss_rate": onepos_credit_near_miss_rate,
            "onepos_credit_long_dogs_near_miss_count": long_dog_onepos_credit_near_miss_count,
            "onepos_credit_long_dogs_near_miss_rate": long_dog_onepos_credit_near_miss_rate,
            "onepos_credit_weighted_count": onepos_credit_weighted_count,
            "onepos_credit_weighted_rate": onepos_credit_weighted_rate,
            "upset_coverage_pct": quality_summary.get("coverage_pct", upset_rate),
            "upset_tier_metrics": quality_summary.get("tier_metrics", {}),
            "upset_roi_by_odds_band": quality_summary.get("roi_by_odds_band", {}),
            "upset_quality_frontier": quality_summary.get("quality_frontier", []),
            "hit_rate_quality_observation": quality_summary.get(
                "hit_rate_quality_observation",
                "",
            ),
            "tier_a_hit_rate": quality_summary.get(
                "tier_a_hit_rate",
                tier_a_hit_rate_fast,
            ),
            "tier_b_hit_rate": quality_summary.get("tier_b_hit_rate", 0.0),
            "tier_c_hit_rate": quality_summary.get("tier_c_hit_rate", 0.0),
            "tier_a_coverage_pct": quality_summary.get("tier_a_coverage_pct", 0.0),
            "tier_b_coverage_pct": quality_summary.get("tier_b_coverage_pct", 0.0),
            "tier_c_coverage_pct": quality_summary.get("tier_c_coverage_pct", 0.0),
            "ml_roi": ml_roi,
            "ml_win_rate": ml_win_rate,
            "ml_bet_count": ml_bet_count,
            "ml_roi_lb95": ml_roi_lb95,
            "ml_min_payout": self._min_ml_payout,
            "spread_mae": spread_mae,
            "compression_ratio": compression_ratio,
            "objective_base_score": objective_base_score,
            "objective_penalty": objective_penalty,
            "objective_family_penalty": family_penalty,
            "objective_ff_p95_abs": ff_p95_abs,
            "objective_onoff_p95_abs": onoff_p95_abs,
            "objective_ff_p95_excess": ff_p95_excess,
            "objective_onoff_p95_excess": onoff_p95_excess,
            "objective_score": objective_score,
            "objective_coverage_shortfall": coverage_shortfall,
            "objective_tier_a_shortfall": tier_a_shortfall,
            "objective_tier_a_hit_rate_for_loss": tier_a_hit_rate_for_loss,
            "objective_target_upset_coverage_pct": self._objective_target_upset_coverage_pct,
            "objective_target_tier_a_hit_rate": self._objective_target_tier_a_hit_rate,
            "loss": loss,
        }


_FF_FAMILY_KEYS = {
    "ff_efg_weight",
    "ff_tov_weight",
    "ff_oreb_weight",
    "ff_fta_weight",
    "opp_ff_efg_weight",
    "opp_ff_tov_weight",
    "opp_ff_oreb_weight",
    "opp_ff_fta_weight",
}

_ONOFF_FAMILY_KEYS = {
    "onoff_impact_mult",
    "onoff_reliability_lambda",
}


def _select_optimizer_ranges(
    include_sharp: bool,
) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """Return active optimizer ranges and profile label."""
    wide_ranges_enabled = _safe_bool_setting("optimizer_use_wide_ranges", False)
    if include_sharp:
        ranges = CD_SHARP_RANGES if wide_ranges_enabled else SHARP_MODE_RANGES
    else:
        ranges = CD_RANGES if wide_ranges_enabled else OPTIMIZER_RANGES
    range_profile = "wide (CD)" if wide_ranges_enabled else "default"
    return dict(ranges), range_profile


def _allocate_blocked_stage_trials(total_trials: int) -> Dict[str, int]:
    """Allocate total trials across blocked pathway stages."""
    total = max(1, int(total_trials))
    min_stage = max(25, _safe_int_setting("optimizer_blocked_min_stage_trials", 250))
    core_frac = max(0.0, _safe_float_setting("optimizer_blocked_core_fraction", 0.35))
    ff_frac = max(0.0, _safe_float_setting("optimizer_blocked_ff_fraction", 0.25))
    onoff_frac = max(0.0, _safe_float_setting("optimizer_blocked_onoff_fraction", 0.15))
    joint_frac = max(0.0, _safe_float_setting("optimizer_blocked_joint_fraction", 0.25))
    frac_map = {
        "core": core_frac,
        "ff": ff_frac,
        "onoff": onoff_frac,
        "joint_refine": joint_frac,
    }
    frac_sum = sum(frac_map.values())
    if frac_sum <= 0.0:
        frac_map = {"core": 0.35, "ff": 0.25, "onoff": 0.15, "joint_refine": 0.25}
        frac_sum = 1.0

    allocations: Dict[str, int] = {}
    for stage, frac in frac_map.items():
        allocations[stage] = int(round(total * (frac / frac_sum)))

    deficit = total - sum(allocations.values())
    if deficit != 0:
        allocations["joint_refine"] = max(1, allocations["joint_refine"] + deficit)

    if total >= (4 * min_stage):
        for stage in ("core", "ff", "onoff", "joint_refine"):
            if allocations[stage] < min_stage:
                delta = min_stage - allocations[stage]
                allocations[stage] = min_stage
                allocations["joint_refine"] = max(1, allocations["joint_refine"] - delta)
        # Re-balance again if needed.
        deficit = total - sum(allocations.values())
        if deficit != 0:
            allocations["joint_refine"] = max(1, allocations["joint_refine"] + deficit)
    return allocations


def _build_trust_region_ranges(
    center_weights: WeightConfig,
    base_ranges: Dict[str, Tuple[float, float]],
    radius_fraction: float,
) -> Dict[str, Tuple[float, float]]:
    """Build a full-parameter trust region around center_weights."""
    out: Dict[str, Tuple[float, float]] = {}
    center_dict = center_weights.to_dict()
    radius = float(np.clip(radius_fraction, 0.01, 1.0))
    for key, (base_lo, base_hi) in base_ranges.items():
        span = float(base_hi - base_lo)
        center = float(center_dict.get(key, base_lo))
        half_width = span * radius
        lo = max(float(base_lo), center - half_width)
        hi = min(float(base_hi), center + half_width)
        if hi - lo < 1e-8:
            eps = max(1e-6, span * 1e-3)
            lo = max(float(base_lo), center - eps)
            hi = min(float(base_hi), center + eps)
        if hi < lo:
            lo, hi = hi, lo
        out[key] = (float(lo), float(hi))
    return out


def _ranges_signature_blob(ranges: Dict[str, Tuple[float, float]]) -> str:
    """Return a stable signature that includes keys and numeric bounds."""
    parts: List[str] = []
    for key in sorted(ranges.keys()):
        lo, hi = ranges[key]
        parts.append(f"{key}:{float(lo):.12g}:{float(hi):.12g}")
    return "|".join(parts)


def _blocked_cycle_tag() -> str:
    """Return rotating cycle tag used to reduce long-run Optuna anchoring."""
    explicit = str(get_setting("optimizer_blocked_cycle_tag", "") or "").strip()
    if explicit:
        return explicit

    base_tag = str(get_setting("optimizer_study_tag", "") or "").strip()
    auto_rotate = _safe_bool_setting("optimizer_blocked_auto_cycle_tag", True)
    if not auto_rotate:
        return base_tag

    cycle_hours = max(1, _safe_int_setting("optimizer_blocked_cycle_hours", 24))
    bucket = int(time.time() // (cycle_hours * 3600))
    auto_tag = f"cycle_{bucket}"
    if base_tag:
        return f"{base_tag}_{auto_tag}"
    return auto_tag


def _empty_stage_champion_bank() -> Dict[str, Any]:
    """Return empty persisted stage-champion structure."""
    items = {
        mode: {stage: [] for stage in _stage_champion_stages}
        for mode in _stage_champion_modes
    }
    return {
        "version": 1,
        "updated_at": "",
        "items": items,
    }


def _normalize_stage_champion_bank(data: Any) -> Dict[str, Any]:
    """Normalize persisted champion-bank payload into expected shape."""
    bank = _empty_stage_champion_bank()
    if not isinstance(data, dict):
        return bank
    items = data.get("items", {})
    if not isinstance(items, dict):
        items = {}
    for mode in _stage_champion_modes:
        mode_items = items.get(mode, {})
        if not isinstance(mode_items, dict):
            mode_items = {}
        for stage in _stage_champion_stages:
            entries = mode_items.get(stage, [])
            if not isinstance(entries, list):
                entries = []
            normalized_entries = []
            for entry in entries:
                if isinstance(entry, dict):
                    weights = entry.get("best_weights")
                    if isinstance(weights, dict):
                        normalized_entries.append(dict(entry))
            bank["items"][mode][stage] = normalized_entries
    return bank


def _load_stage_champion_bank() -> Dict[str, Any]:
    """Load persistent stage champions from disk."""
    with _stage_champion_bank_lock:
        if not os.path.exists(_stage_champion_bank_path):
            return _empty_stage_champion_bank()
        try:
            with open(_stage_champion_bank_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.debug("Failed loading stage champion bank", exc_info=True)
            return _empty_stage_champion_bank()
        return _normalize_stage_champion_bank(payload)


def _save_stage_champion_bank(bank: Dict[str, Any]) -> None:
    """Persist stage champions to disk atomically."""
    with _stage_champion_bank_lock:
        normalized = _normalize_stage_champion_bank(bank)
        normalized["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(_stage_champion_bank_path) or ".", exist_ok=True)
        tmp_path = _stage_champion_bank_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
        os.replace(tmp_path, _stage_champion_bank_path)


def _stage_champion_weight_hash(stage_name: str, weights: Dict[str, Any]) -> str:
    """Create stable hash for champion dedupe."""
    parts = [stage_name]
    for key in sorted(weights.keys()):
        try:
            value = float(weights[key])
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        parts.append(f"{key}:{value:.12g}")
    blob = "|".join(parts)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _sorted_stage_champion_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort champion entries by objective then val-loss then winner%."""
    ranked = [dict(entry) for entry in entries if isinstance(entry, dict)]
    ranked.sort(
        key=lambda e: (
            float(e.get("objective_selected_loss", 1.0e12)),
            float(e.get("best_loss", 1.0e12)),
            -float(e.get("best_winner_pct", 0.0)),
        )
    )
    return ranked


def _persist_stage_champion_candidate(
    mode_key: str,
    stage_name: str,
    candidate: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    """Upsert stage champion candidate and keep only top_k entries."""
    mode = str(mode_key or "").strip().lower()
    stage = str(stage_name or "").strip().lower()
    if mode not in _stage_champion_modes or stage not in _stage_champion_stages:
        return {"updated": False, "rank": None, "count": 0}
    if not isinstance(candidate, dict):
        return {"updated": False, "rank": None, "count": 0}
    weights = candidate.get("best_weights")
    if not isinstance(weights, dict):
        return {"updated": False, "rank": None, "count": 0}

    try:
        objective = float(candidate.get("objective_selected_loss", 1.0e12))
    except Exception:
        objective = 1.0e12
    if not np.isfinite(objective):
        objective = 1.0e12
    try:
        val_loss = float(candidate.get("best_loss", 1.0e12))
    except Exception:
        val_loss = 1.0e12
    if not np.isfinite(val_loss):
        val_loss = 1.0e12
    try:
        winner_pct = float(candidate.get("best_winner_pct", 0.0))
    except Exception:
        winner_pct = 0.0
    if not np.isfinite(winner_pct):
        winner_pct = 0.0

    now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    weight_hash = _stage_champion_weight_hash(stage, weights)
    incoming_entry = {
        "stage": stage,
        "stage_id": str(candidate.get("stage_id", "")),
        "trials": int(max(0, int(candidate.get("trials", 0) or 0))),
        "objective_selected_loss": objective,
        "best_loss": val_loss,
        "best_winner_pct": winner_pct,
        "best_weights": dict(weights),
        "weight_hash": weight_hash,
        "updated_at": now_ts,
        "created_at": now_ts,
    }

    bank = _load_stage_champion_bank()
    entries = list(bank["items"][mode][stage])
    existing_idx = None
    for idx, entry in enumerate(entries):
        if str(entry.get("weight_hash", "")) == weight_hash:
            existing_idx = idx
            break

    updated = False
    if existing_idx is None:
        entries.append(incoming_entry)
        updated = True
    else:
        existing = dict(entries[existing_idx])
        existing_created = str(existing.get("created_at", "")) or now_ts
        old_tuple = (
            float(existing.get("objective_selected_loss", 1.0e12)),
            float(existing.get("best_loss", 1.0e12)),
            -float(existing.get("best_winner_pct", 0.0)),
        )
        new_tuple = (objective, val_loss, -winner_pct)
        if new_tuple < old_tuple:
            existing.update(incoming_entry)
            existing["created_at"] = existing_created
            entries[existing_idx] = existing
            updated = True
        else:
            existing["updated_at"] = now_ts
            existing["stage_id"] = incoming_entry["stage_id"]
            existing["trials"] = incoming_entry["trials"]
            entries[existing_idx] = existing
            updated = True

    limit = max(1, int(top_k))
    entries = _sorted_stage_champion_entries(entries)[:limit]
    bank["items"][mode][stage] = entries
    if updated:
        _save_stage_champion_bank(bank)

    rank = None
    for idx, entry in enumerate(entries, start=1):
        if str(entry.get("weight_hash", "")) == weight_hash:
            rank = idx
            break
    return {
        "updated": bool(updated),
        "rank": rank,
        "count": len(entries),
    }


def _get_stage_champion_entries(
    mode_key: str,
    stage_name: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Return top-N persistent champions for mode/stage."""
    mode = str(mode_key or "").strip().lower()
    stage = str(stage_name or "").strip().lower()
    if mode not in _stage_champion_modes or stage not in _stage_champion_stages:
        return []
    bank = _load_stage_champion_bank()
    entries = bank["items"][mode][stage]
    if not isinstance(entries, list):
        return []
    sorted_entries = _sorted_stage_champion_entries(entries)
    max_items = max(0, int(limit))
    if max_items <= 0:
        return []
    return sorted_entries[:max_items]


def _build_stage_champion_seed_trials(
    optuna_module: Any,
    *,
    mode_key: str,
    stage_name: str,
    ranges: Dict[str, Tuple[float, float]],
    baseline_weights: WeightConfig,
    max_entries: int,
) -> List[Any]:
    """Build synthetic completed trials from persistent stage champions."""
    entries = _get_stage_champion_entries(mode_key, stage_name, max_entries)
    if not entries:
        return []

    distributions = {
        key: optuna_module.distributions.FloatDistribution(float(lo), float(hi))
        for key, (lo, hi) in ranges.items()
    }
    baseline_dict = baseline_weights.to_dict()
    out = []
    for entry in entries:
        weights = entry.get("best_weights", {})
        if not isinstance(weights, dict):
            continue
        params = {}
        for key, (lo, hi) in ranges.items():
            base_v = float(baseline_dict.get(key, lo))
            try:
                raw_v = float(weights.get(key, base_v))
            except Exception:
                raw_v = base_v
            if not np.isfinite(raw_v):
                raw_v = base_v
            params[key] = float(np.clip(raw_v, float(lo), float(hi)))
        try:
            trial_value = float(entry.get("objective_selected_loss", 1.0e12))
        except Exception:
            trial_value = 1.0e12
        if not np.isfinite(trial_value):
            trial_value = 1.0e12
        try:
            out.append(
                optuna_module.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    value=trial_value,
                    user_attrs={
                        "seed_source": "stage_champion_bank",
                        "seed_mode": mode_key,
                        "seed_stage": stage_name,
                        "seed_weight_hash": str(entry.get("weight_hash", "")),
                    },
                )
            )
        except Exception:
            logger.debug("Failed creating champion seed trial", exc_info=True)
    return out


def clear_stage_champion_bank(reason: str = "") -> bool:
    """Clear persisted stage champion bank."""
    try:
        _save_stage_champion_bank(_empty_stage_champion_bank())
        if reason:
            logger.info("Cleared stage champion bank (%s)", reason)
        else:
            logger.info("Cleared stage champion bank")
        return True
    except Exception:
        logger.exception("Failed to clear stage champion bank")
        return False


def _blocked_candidate_snapshot(
    stage_name: str,
    stage_id: str,
    trials: int,
    stage_result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Normalize a blocked-stage winner into a comparable candidate snapshot."""
    weights = stage_result.get("best_weights")
    if not isinstance(weights, dict):
        return None
    missing_loss = 1.0e12

    def _safe_metric(value: Any, fallback: float) -> float:
        try:
            metric = float(value)
        except Exception:
            return float(fallback)
        if not np.isfinite(metric):
            return float(fallback)
        return metric

    selected_objective = _safe_metric(
        stage_result.get("objective_selected_loss", stage_result.get("objective_loss")),
        missing_loss,
    )
    val_loss = _safe_metric(stage_result.get("best_loss"), missing_loss)
    winner_pct = _safe_metric(stage_result.get("best_winner_pct"), 0.0)
    return {
        "stage": stage_name,
        "stage_id": stage_id,
        "trials": int(max(0, trials)),
        "objective_selected_loss": selected_objective,
        "best_loss": val_loss,
        "best_winner_pct": winner_pct,
        "best_weights": dict(weights),
    }


def _rank_blocked_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank blocked candidates by objective, then val loss, then winner%."""
    ranked = [dict(c) for c in candidates if isinstance(c, dict)]
    ranked.sort(
        key=lambda c: (
            float(c.get("objective_selected_loss", 1.0e12)),
            float(c.get("best_loss", 1.0e12)),
            -float(c.get("best_winner_pct", 0.0)),
        )
    )
    for idx, candidate in enumerate(ranked, start=1):
        candidate["rank"] = idx
    return ranked


def optimize_weights(
    games: List[GameInput],
    n_trials: int = 3000,
    include_sharp: bool = False,
    callback: Optional[Callable] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    *,
    baseline_override: Optional[WeightConfig] = None,
    param_ranges_override: Optional[Dict[str, Tuple[float, float]]] = None,
    candidate_override: Optional[WeightConfig] = None,
    champion_bank_stage: Optional[str] = None,
    skip_save: bool = False,
    _internal_blocked: bool = False,
    stage_label: str = "main",
) -> Dict[str, Any]:
    """Run Optuna TPE optimization with walk-forward validation.

    Games are split chronologically: first WALK_FORWARD_SPLIT for training,
    remainder for validation. Optuna optimises on the training set; weights
    are only saved when they also improve on the held-out validation set.

    Save gate: anti-gaming validation gate blends loss, ROI, and upset lift.
    candidate_override: optional pre-selected weights to evaluate/promote
    through the same walk-forward + gate stack without new trial selection.
    champion_bank_stage: blocked stage key used for persistent champion seeding.
    """
    # Walk-forward split
    sorted_games = sorted(games, key=lambda g: g.game_date)
    split_idx = int(len(sorted_games) * WALK_FORWARD_SPLIT)
    train_games = sorted_games[:split_idx]
    val_games = sorted_games[split_idx:]

    if not train_games or not val_games:
        if callback:
            callback("Not enough games for walk-forward split")
        return {"improved": False}

    tuning_mode = str(get_setting("optimizer_tuning_mode", "classic") or "classic").strip().lower()
    blocked_mode_enabled = tuning_mode == "blocked"
    if (
        blocked_mode_enabled
        and not _internal_blocked
        and param_ranges_override is None
        and not skip_save
    ):
        stage_champion_bank_enabled = _safe_bool_setting(
            "optimizer_stage_champion_bank_enabled",
            True,
        )
        stage_champion_bank_top_k = max(
            1,
            _safe_int_setting("optimizer_stage_champion_bank_top_k", 100),
        )
        preserve_stage_champions = _safe_bool_setting(
            "optimizer_blocked_preserve_stage_champions",
            True,
        )
        champion_log_top_k = max(
            1,
            _safe_int_setting("optimizer_blocked_champion_log_top_k", 4),
        )
        baseline_root = baseline_override if baseline_override is not None else get_weight_config()
        active_ranges, active_range_profile = _select_optimizer_ranges(include_sharp)
        ff_keys = {k for k in _FF_FAMILY_KEYS if k in active_ranges}
        onoff_keys = {k for k in _ONOFF_FAMILY_KEYS if k in active_ranges}
        core_keys = sorted(set(active_ranges.keys()) - ff_keys - onoff_keys)
        stage_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}
        if core_keys:
            stage_ranges["core"] = {k: active_ranges[k] for k in core_keys}
        if ff_keys:
            stage_ranges["ff"] = {k: active_ranges[k] for k in sorted(ff_keys)}
        if onoff_keys:
            stage_ranges["onoff"] = {k: active_ranges[k] for k in sorted(onoff_keys)}
        trial_alloc = _allocate_blocked_stage_trials(n_trials)
        cycle_tag = _blocked_cycle_tag()
        mode_prefix = "sharp" if include_sharp else "fundamentals"

        if callback:
            callback(
                "Blocked optimizer pathways: "
                f"mode={tuning_mode}, range_profile={active_range_profile}, "
                f"cycle_tag={cycle_tag or 'none'}"
            )
            callback(
                "Pathway trial allocation: "
                f"core={trial_alloc.get('core', 0)}, "
                f"ff={trial_alloc.get('ff', 0)}, "
                f"onoff={trial_alloc.get('onoff', 0)}, "
                f"joint_refine={trial_alloc.get('joint_refine', 0)}"
            )
            callback(
                "Stage champion bank: "
                f"{'enabled' if stage_champion_bank_enabled else 'disabled'} "
                f"(top_k={stage_champion_bank_top_k})"
            )

        stage_current = baseline_root
        stage_summaries: List[Dict[str, Any]] = []
        stage_candidates: List[Dict[str, Any]] = []
        for stage_name in ("core", "ff", "onoff"):
            this_ranges = stage_ranges.get(stage_name)
            this_trials = int(trial_alloc.get(stage_name, 0))
            if not this_ranges or this_trials <= 0:
                continue
            stage_id = f"{mode_prefix}_{cycle_tag}_{stage_name}" if cycle_tag else f"{mode_prefix}_{stage_name}"
            if callback:
                callback(
                    f"[blocked:{stage_name}] tuning {len(this_ranges)} params "
                    f"for {this_trials} trials"
                )
            stage_result = optimize_weights(
                games,
                n_trials=this_trials,
                include_sharp=include_sharp,
                callback=callback,
                is_cancelled=is_cancelled,
                baseline_override=stage_current,
                param_ranges_override=this_ranges,
                skip_save=True,
                _internal_blocked=True,
                stage_label=stage_id,
                champion_bank_stage=stage_name,
            )
            stage_best = stage_result.get("best_weights")
            if isinstance(stage_best, dict):
                stage_current = WeightConfig.from_dict(stage_best)
            stage_summaries.append(
                {
                    "stage": stage_name,
                    "trials": this_trials,
                    "objective_loss": stage_result.get("objective_loss"),
                    "best_loss": stage_result.get("best_loss"),
                    "best_winner_pct": stage_result.get("best_winner_pct"),
                }
            )
            stage_candidate = _blocked_candidate_snapshot(
                stage_name=stage_name,
                stage_id=stage_id,
                trials=this_trials,
                stage_result=stage_result,
            )
            if stage_candidate is not None:
                stage_candidates.append(stage_candidate)
                if stage_champion_bank_enabled:
                    persist_info = _persist_stage_champion_candidate(
                        mode_key=mode_prefix,
                        stage_name=stage_name,
                        candidate=stage_candidate,
                        top_k=stage_champion_bank_top_k,
                    )
                    if callback and bool(persist_info.get("updated", False)):
                        callback(
                            f"[blocked] bank update {stage_name}: "
                            f"rank #{persist_info.get('rank', '-')}, "
                            f"kept {persist_info.get('count', 0)}/{stage_champion_bank_top_k}"
                        )

        radius_fraction = float(
            np.clip(
                _safe_float_setting("optimizer_blocked_joint_radius_fraction", 0.18),
                0.01,
                1.0,
            )
        )
        joint_ranges = _build_trust_region_ranges(stage_current, active_ranges, radius_fraction)
        joint_trials = int(trial_alloc.get("joint_refine", max(1, n_trials // 4)))
        joint_id = (
            f"{mode_prefix}_{cycle_tag}_joint_refine"
            if cycle_tag
            else f"{mode_prefix}_joint_refine"
        )
        if callback:
            callback(
                f"[blocked:joint_refine] trust-region radius={radius_fraction:.2f}, "
                f"params={len(joint_ranges)}, trials={joint_trials}"
            )
        joint_skip_save = bool(preserve_stage_champions)
        joint_result = optimize_weights(
            games,
            n_trials=joint_trials,
            include_sharp=include_sharp,
            callback=callback,
            is_cancelled=is_cancelled,
            baseline_override=baseline_root,
            param_ranges_override=joint_ranges,
            skip_save=joint_skip_save,
            _internal_blocked=True,
            stage_label=joint_id,
            champion_bank_stage="joint_refine",
        )
        joint_candidate = _blocked_candidate_snapshot(
            stage_name="joint_refine",
            stage_id=joint_id,
            trials=joint_trials,
            stage_result=joint_result,
        )
        if joint_candidate is not None:
            stage_candidates.append(joint_candidate)
            if stage_champion_bank_enabled:
                persist_info = _persist_stage_champion_candidate(
                    mode_key=mode_prefix,
                    stage_name="joint_refine",
                    candidate=joint_candidate,
                    top_k=stage_champion_bank_top_k,
                )
                if callback and bool(persist_info.get("updated", False)):
                    callback(
                        "[blocked] bank update joint_refine: "
                        f"rank #{persist_info.get('rank', '-')}, "
                        f"kept {persist_info.get('count', 0)}/{stage_champion_bank_top_k}"
                    )

        ranked_candidates = _rank_blocked_candidates(stage_candidates)
        promoted_stage = "joint_refine"
        final_result = joint_result

        if callback and ranked_candidates:
            callback(
                f"[blocked] candidate playoff: {len(ranked_candidates)} stage champions retained"
            )
            for candidate in ranked_candidates[:champion_log_top_k]:
                callback(
                    f"  [blocked] #{int(candidate.get('rank', 0))} "
                    f"{candidate.get('stage', 'n/a')} "
                    f"objective={float(candidate.get('objective_selected_loss', 1.0e12)):.3f}, "
                    f"val_loss={float(candidate.get('best_loss', 1.0e12)):.3f}, "
                    f"winner={float(candidate.get('best_winner_pct', 0.0)):.1f}%"
                )

        if preserve_stage_champions and ranked_candidates:
            champion = ranked_candidates[0]
            champion_weights = champion.get("best_weights")
            if isinstance(champion_weights, dict):
                promoted_stage = str(champion.get("stage", "joint_refine"))
                if callback:
                    callback(
                        "[blocked] promoting champion from "
                        f"{promoted_stage} pathway through save gate"
                    )
                final_result = optimize_weights(
                    games,
                    n_trials=0,
                    include_sharp=include_sharp,
                    callback=callback,
                    is_cancelled=is_cancelled,
                    baseline_override=baseline_root,
                    candidate_override=WeightConfig.from_dict(champion_weights),
                    skip_save=False,
                    _internal_blocked=True,
                    stage_label=(
                        f"{mode_prefix}_{cycle_tag}_champion"
                        if cycle_tag
                        else f"{mode_prefix}_champion"
                    ),
                    champion_bank_stage=promoted_stage,
                )
            elif callback:
                callback(
                    "[blocked] champion candidate missing weights; falling back to joint_refine result"
                )
        elif callback and preserve_stage_champions:
            callback("[blocked] no valid stage champions found; using joint_refine outcome")
        elif callback and not preserve_stage_champions:
            callback("[blocked] cross-path playoff disabled; joint_refine controls promotion")

        ranked_candidate_summary = [
            {
                "rank": int(candidate.get("rank", idx + 1)),
                "stage": candidate.get("stage"),
                "stage_id": candidate.get("stage_id"),
                "trials": int(candidate.get("trials", 0)),
                "objective_selected_loss": float(
                    candidate.get("objective_selected_loss", 1.0e12)
                ),
                "best_loss": float(candidate.get("best_loss", 1.0e12)),
                "best_winner_pct": float(candidate.get("best_winner_pct", 0.0)),
            }
            for idx, candidate in enumerate(ranked_candidates)
        ]
        final_result["blocked_pathways"] = {
            "enabled": True,
            "mode": tuning_mode,
            "cycle_tag": cycle_tag,
            "range_profile": active_range_profile,
            "stage_trials": trial_alloc,
            "stages": stage_summaries,
            "joint_radius_fraction": radius_fraction,
            "ff_params": sorted(ff_keys),
            "onoff_params": sorted(onoff_keys),
            "core_param_count": len(core_keys),
            "cross_path_playoff_enabled": bool(preserve_stage_champions),
            "champion_log_top_k": champion_log_top_k,
            "joint_skip_save": joint_skip_save,
            "stage_champion_bank_enabled": bool(stage_champion_bank_enabled),
            "stage_champion_bank_top_k": stage_champion_bank_top_k,
            "promotion_source_stage": promoted_stage,
            "ranked_candidates": ranked_candidate_summary,
            "ranked_candidates_with_weights": ranked_candidates,
        }
        return final_result

    vg_train = VectorizedGames(train_games)
    vg_val = VectorizedGames(val_games)
    vg_all = VectorizedGames(sorted_games)

    if callback:
        callback(f"Walk-forward: {len(train_games)} train "
                 f"({train_games[0].game_date} to {train_games[-1].game_date}), "
                 f"{len(val_games)} validation "
                 f"({val_games[0].game_date} to {val_games[-1].game_date})")

    # Baseline evaluation on both sets
    baseline_w = baseline_override if baseline_override is not None else get_weight_config()
    baseline_train = vg_train.evaluate(baseline_w, include_sharp=include_sharp)
    baseline_val = vg_val.evaluate(baseline_w, include_sharp=include_sharp)
    baseline_all = vg_all.evaluate(baseline_w, include_sharp=include_sharp)
    val_probe_enabled = _safe_bool_setting("optimizer_objective_val_probe_enabled", True)
    val_probe_sample_size = _safe_int_setting("optimizer_objective_val_probe_sample_size", 480)
    val_probe_slices = max(1, _safe_int_setting("optimizer_objective_val_probe_slices", 3))
    val_probe_loss_mult = max(
        0.0,
        _safe_float_setting("optimizer_objective_val_probe_loss_mult", 1.0),
    )
    val_probe_winner_drop_mult = max(
        0.0,
        _safe_float_setting("optimizer_objective_val_probe_winner_drop_mult", 0.35),
    )
    target_probe_games = max(50, val_probe_sample_size)
    vg_val_probes: List[VectorizedGames] = []
    baseline_val_probes: List[Dict[str, float]] = []
    val_probe_games_counts: List[int] = []
    if val_probe_enabled:
        if target_probe_games >= len(val_games):
            probe_windows = [list(val_games)]
        else:
            max_start = max(0, len(val_games) - target_probe_games)
            if val_probe_slices <= 1:
                starts = [max_start // 2]
            else:
                starts = [
                    int(round(max_start * i / float(val_probe_slices - 1)))
                    for i in range(val_probe_slices)
                ]
            seen_starts = set()
            ordered_starts: List[int] = []
            for start in starts:
                if start not in seen_starts:
                    seen_starts.add(start)
                    ordered_starts.append(start)
            probe_windows = [
                val_games[start:start + target_probe_games]
                for start in ordered_starts
                if val_games[start:start + target_probe_games]
            ]
            if not probe_windows:
                probe_windows = [list(val_games[:target_probe_games])]
        for probe_games in probe_windows:
            vg_probe = VectorizedGames(list(probe_games))
            baseline_probe = vg_probe.evaluate(
                baseline_w,
                include_sharp=include_sharp,
                fast=True,
            )
            vg_val_probes.append(vg_probe)
            baseline_val_probes.append(baseline_probe)
            val_probe_games_counts.append(len(probe_games))

    rolling_cv_enabled = _safe_bool_setting("optimizer_rolling_cv_enabled", True)
    rolling_cv_folds = max(1, _safe_int_setting("optimizer_rolling_cv_folds", 4))
    rolling_cv_min_train_games = max(
        120,
        _safe_int_setting(
            "optimizer_rolling_cv_min_train_games",
            max(240, int(len(train_games) * 0.45)),
        ),
    )
    rolling_cv_val_games = max(
        40,
        _safe_int_setting(
            "optimizer_rolling_cv_val_games",
            max(120, int(len(train_games) * 0.12)),
        ),
    )
    rolling_cv_worst_fold_mult = max(
        0.0,
        _safe_float_setting("optimizer_rolling_cv_worst_fold_mult", 0.40),
    )
    rolling_fold_ranges = (
        _build_rolling_time_folds(
            len(train_games),
            rolling_cv_folds,
            rolling_cv_min_train_games,
            rolling_cv_val_games,
        )
        if rolling_cv_enabled
        else []
    )
    rolling_cv_val_windows: List[VectorizedGames] = []
    baseline_cv_fold_losses: List[float] = []
    baseline_cv_fold_winners: List[float] = []
    if rolling_fold_ranges:
        for train_end, val_end in rolling_fold_ranges:
            fold_val_games = train_games[train_end:val_end]
            vg_fold_val = VectorizedGames(fold_val_games)
            baseline_fold = vg_fold_val.evaluate(
                baseline_w,
                include_sharp=include_sharp,
                fast=True,
            )
            rolling_cv_val_windows.append(vg_fold_val)
            baseline_cv_fold_losses.append(float(baseline_fold.get("loss", 0.0)))
            baseline_cv_fold_winners.append(
                float(baseline_fold.get("winner_pct", 0.0))
            )
    else:
        rolling_cv_enabled = False
    if rolling_cv_enabled:
        baseline_cv_score, baseline_cv_mean, baseline_cv_worst = _rolling_cv_objective_loss(
            baseline_cv_fold_losses,
            rolling_cv_worst_fold_mult,
        )
    else:
        baseline_cv_score = float(baseline_train["loss"])
        baseline_cv_mean = baseline_cv_score
        baseline_cv_worst = baseline_cv_score
    objective_track = _normalize_objective_track(
        get_setting("optimizer_objective_primary_track", "dual_track")
    )
    objective_dual_live_weight = float(
        np.clip(
            _safe_float_setting("optimizer_objective_dual_live_weight", 0.70),
            0.0,
            1.0,
        )
    )
    baseline_live_objective = float(baseline_cv_score)
    baseline_oracle_objective = float(baseline_all.get("loss", baseline_live_objective))
    baseline_selected_objective = _compose_objective_loss(
        baseline_live_objective,
        baseline_oracle_objective,
        objective_track,
        objective_dual_live_weight,
    )

    ml_underdog_gate_enabled = (
        (not skip_save)
        and _safe_bool_setting(
            "optimizer_ml_underdog_scorer_enabled",
            False,
        )
    )
    ml_underdog_lr = max(
        0.0001,
        _safe_float_setting("optimizer_ml_underdog_scorer_lr", 0.05),
    )
    ml_underdog_l2 = max(
        0.0,
        _safe_float_setting("optimizer_ml_underdog_scorer_l2", 1.0),
    )
    ml_underdog_max_iter = max(
        50,
        _safe_int_setting("optimizer_ml_underdog_scorer_max_iter", 220),
    )
    ml_underdog_min_train_samples = max(
        20,
        _safe_int_setting("optimizer_ml_underdog_scorer_min_train_samples", 140),
    )
    ml_underdog_min_val_samples = max(
        20,
        _safe_int_setting("optimizer_ml_underdog_scorer_min_val_samples", 60),
    )
    ml_underdog_min_brier_lift = max(
        0.0,
        _safe_float_setting("optimizer_ml_underdog_scorer_min_brier_lift", 0.0025),
    )
    blocked_stage_verbose = _safe_bool_setting(
        "optimizer_blocked_stage_verbose",
        False,
    )
    emit_stage_details = (not skip_save) or blocked_stage_verbose

    if callback and emit_stage_details:
        callback(f"Baseline (train): Winner={baseline_train['winner_pct']:.1f}%, "
                 f"Upset={baseline_train['upset_accuracy']:.1f}% @ {baseline_train['upset_rate']:.1f}% rate, "
                 f"CompDog={baseline_train.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={baseline_train.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={baseline_train.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={baseline_train['loss']:.3f}")
        callback(f"  {baseline_train.get('hit_rate_quality_observation', '')}")
        callback(f"Baseline (valid): Winner={baseline_val['winner_pct']:.1f}%, "
                 f"Upset={baseline_val['upset_accuracy']:.1f}% @ {baseline_val['upset_rate']:.1f}% rate, "
                 f"CompDog={baseline_val.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={baseline_val.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={baseline_val.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Favorites={baseline_val['favorites_pct']:.1f}%, "
                 f"Loss={baseline_val['loss']:.3f}")
        callback(f"  {baseline_val.get('hit_rate_quality_observation', '')}")
        callback(f"Baseline (all):   Winner={baseline_all['winner_pct']:.1f}%, "
                 f"Upset={baseline_all['upset_accuracy']:.1f}% @ {baseline_all['upset_rate']:.1f}% rate, "
                 f"CompDog={baseline_all.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={baseline_all.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={baseline_all.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={baseline_all['loss']:.3f}")
        callback(f"  {baseline_all.get('hit_rate_quality_observation', '')}")
        if vg_val_probes:
            baseline_probe_losses = [float(p.get("loss", 0.0)) for p in baseline_val_probes]
            baseline_probe_winners = [float(p.get("winner_pct", 0.0)) for p in baseline_val_probes]
            callback(
                "Objective val-probe: "
                f"{len(vg_val_probes)} slices x {target_probe_games} games "
                f"(actual sizes: {val_probe_games_counts}), "
                f"baseline loss range={min(baseline_probe_losses):.3f}..{max(baseline_probe_losses):.3f}, "
                f"baseline winner range={min(baseline_probe_winners):.1f}%..{max(baseline_probe_winners):.1f}%, "
                f"loss_mult={val_probe_loss_mult:.2f}, "
                f"winner_drop_mult={val_probe_winner_drop_mult:.2f}"
            )
        if rolling_cv_enabled:
            fold_sizes = [val_end - train_end for train_end, val_end in rolling_fold_ranges]
            callback(
                "Objective rolling-CV: "
                f"{len(rolling_fold_ranges)} folds, "
                f"val sizes={fold_sizes}, "
                f"worst_mult={rolling_cv_worst_fold_mult:.2f}, "
                f"baseline score={baseline_cv_score:.3f} "
                f"(mean={baseline_cv_mean:.3f}, worst={baseline_cv_worst:.3f})"
            )
        else:
            callback("Objective rolling-CV: disabled (falling back to train loss objective)")
        callback(
            "Objective track mode: "
            f"{objective_track} "
            f"(baseline selected={baseline_selected_objective:.3f}, "
            f"live={baseline_live_objective:.3f}, "
            f"oracle={baseline_oracle_objective:.3f}, "
            f"dual_live_weight={objective_dual_live_weight:.2f})"
        )
        if ml_underdog_gate_enabled:
            callback(
                "ML underdog promotion gate: enabled "
                f"(min brier lift +{ml_underdog_min_brier_lift:.4f}, "
                f"min train/val upset samples "
                f"{ml_underdog_min_train_samples}/{ml_underdog_min_val_samples})"
            )
        else:
            callback("ML underdog promotion gate: disabled")
    elif callback and skip_save:
        callback(
            f"[stage:{stage_label}] baseline "
            f"(train={baseline_train['loss']:.3f}, valid={baseline_val['loss']:.3f}, "
            f"all={baseline_all['loss']:.3f})"
        )

    provided_candidate = candidate_override
    best_w = provided_candidate if provided_candidate is not None else baseline_w
    best_objective_loss = baseline_selected_objective
    best_live_objective_loss = baseline_live_objective
    best_oracle_objective_loss = baseline_oracle_objective
    best_train_loss = baseline_train["loss"]
    last_saved_w = [baseline_w]
    min_weight_delta = _safe_float_setting("optimizer_save_min_weight_delta", 1e-4)

    # Select parameter ranges
    if param_ranges_override is not None:
        ranges = dict(param_ranges_override)
        range_profile = "override"
    else:
        ranges, range_profile = _select_optimizer_ranges(include_sharp)
    if callback:
        callback(
            f"Optimizer search ranges: {range_profile} ({len(ranges)} params)"
            f" [stage={stage_label}]"
        )
    champion_bank_mode_key = "sharp" if include_sharp else "fundamentals"
    champion_bank_stage_key = str(champion_bank_stage or "").strip().lower()
    stage_champion_seed_enabled = (
        champion_bank_stage_key in _stage_champion_stages
        and _safe_bool_setting("optimizer_stage_champion_bank_enabled", True)
        and _safe_bool_setting("optimizer_stage_champion_bank_seed_enabled", True)
    )
    stage_champion_seed_max = max(
        0,
        _safe_int_setting("optimizer_stage_champion_bank_seed_max", 100),
    )
    deterministic = _safe_bool_setting("optimizer_deterministic", False)
    deterministic_seed = _safe_int_setting("optimizer_deterministic_seed", 42)
    objective_l2_prior_mult = max(
        0.0,
        _safe_float_setting("optimizer_objective_l2_prior_mult", 0.02),
    )
    if deterministic:
        random.seed(deterministic_seed)
        np.random.seed(deterministic_seed)
        if callback:
            callback(f"Deterministic optimizer mode enabled (seed={deterministic_seed})")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for key, (lo, hi) in ranges.items():
                params[key] = trial.suggest_float(key, lo, hi)

            w = WeightConfig.from_dict({**baseline_w.to_dict(), **params})
            train_result = vg_train.evaluate(w, include_sharp=include_sharp, fast=True)
            live_objective_loss = float(train_result["loss"])
            rolling_cv_score = live_objective_loss
            rolling_cv_mean_loss = live_objective_loss
            rolling_cv_worst_loss = live_objective_loss
            rolling_cv_mean_winner = float(train_result.get("winner_pct", 0.0))
            rolling_cv_worst_winner = rolling_cv_mean_winner
            if rolling_cv_enabled and rolling_cv_val_windows:
                fold_losses: List[float] = []
                fold_winners: List[float] = []
                for vg_fold_val in rolling_cv_val_windows:
                    fold_result = vg_fold_val.evaluate(
                        w,
                        include_sharp=include_sharp,
                        fast=True,
                    )
                    fold_losses.append(float(fold_result.get("loss", 0.0)))
                    fold_winners.append(float(fold_result.get("winner_pct", 0.0)))
                rolling_cv_score, rolling_cv_mean_loss, rolling_cv_worst_loss = (
                    _rolling_cv_objective_loss(
                        fold_losses,
                        rolling_cv_worst_fold_mult,
                    )
                )
                rolling_cv_mean_winner = float(np.mean(fold_winners)) if fold_winners else 0.0
                rolling_cv_worst_winner = float(np.min(fold_winners)) if fold_winners else 0.0
                live_objective_loss = rolling_cv_score
            overfit_penalty = 0.0
            l2_prior_penalty = 0.0
            probe_loss_delta_mean = 0.0
            probe_loss_delta_max = 0.0
            probe_winner_drop_mean = 0.0
            if vg_val_probes:
                probe_loss_deltas: List[float] = []
                probe_winner_drops: List[float] = []
                for probe_idx, vg_probe in enumerate(vg_val_probes):
                    val_probe_result = vg_probe.evaluate(
                        w,
                        include_sharp=include_sharp,
                        fast=True,
                    )
                    baseline_probe = baseline_val_probes[probe_idx]
                    probe_loss = float(val_probe_result.get("loss", 0.0))
                    probe_winner = float(val_probe_result.get("winner_pct", 0.0))
                    probe_loss_deltas.append(
                        max(
                            0.0,
                            probe_loss - float(baseline_probe.get("loss", 0.0)),
                        )
                    )
                    probe_winner_drops.append(
                        max(
                            0.0,
                            float(baseline_probe.get("winner_pct", 0.0)) - probe_winner,
                        )
                    )

                probe_loss_delta_mean = float(np.mean(probe_loss_deltas))
                probe_loss_delta_max = float(np.max(probe_loss_deltas))
                probe_winner_drop_mean = float(np.mean(probe_winner_drops))
                overfit_penalty = (
                    (probe_loss_delta_mean + 0.5 * probe_loss_delta_max) * val_probe_loss_mult
                    + probe_winner_drop_mean * val_probe_winner_drop_mult
                )
                live_objective_loss += overfit_penalty
            if objective_l2_prior_mult > 0.0 and ranges:
                sq_sum = 0.0
                count = 0
                for key, (lo, hi) in ranges.items():
                    span = max(1e-9, float(hi) - float(lo))
                    base_v = float(getattr(baseline_w, key, 0.0))
                    cur_v = float(getattr(w, key, base_v))
                    sq_sum += ((cur_v - base_v) / span) ** 2
                    count += 1
                if count > 0:
                    l2_prior_penalty = objective_l2_prior_mult * (sq_sum / float(count))
                    live_objective_loss += l2_prior_penalty
            oracle_result = vg_all.evaluate(w, include_sharp=include_sharp, fast=True)
            oracle_objective_loss = float(oracle_result.get("loss", 0.0))
            objective_loss = _compose_objective_loss(
                live_objective_loss,
                oracle_objective_loss,
                objective_track,
                objective_dual_live_weight,
            )
            trial.set_user_attr(
                "result",
                {
                    "winner_pct": float(train_result.get("winner_pct", 0.0)),
                    "upset_accuracy": float(train_result.get("upset_accuracy", 0.0)),
                    "upset_rate": float(train_result.get("upset_rate", 0.0)),
                    "loss": float(train_result.get("loss", 0.0)),
                },
            )
            trial.set_user_attr("objective_overfit_penalty", overfit_penalty)
            trial.set_user_attr("objective_l2_prior_penalty", l2_prior_penalty)
            trial.set_user_attr(
                "rolling_cv",
                {
                    "enabled": bool(rolling_cv_enabled and rolling_cv_val_windows),
                    "fold_count": len(rolling_cv_val_windows),
                    "score": rolling_cv_score,
                    "mean_loss": rolling_cv_mean_loss,
                    "worst_loss": rolling_cv_worst_loss,
                    "mean_winner_pct": rolling_cv_mean_winner,
                    "worst_winner_pct": rolling_cv_worst_winner,
                    "worst_mult": rolling_cv_worst_fold_mult,
                },
            )
            trial.set_user_attr(
                "objective_tracks",
                {
                    "track": objective_track,
                    "dual_live_weight": objective_dual_live_weight,
                    "selected_loss": objective_loss,
                    "live_loss": live_objective_loss,
                    "oracle_loss": oracle_objective_loss,
                },
            )
            if vg_val_probes:
                trial.set_user_attr(
                    "val_probe",
                    {
                        "loss_delta_mean": probe_loss_delta_mean,
                        "loss_delta_max": probe_loss_delta_max,
                        "winner_drop_mean": probe_winner_drop_mean,
                        "slices": len(vg_val_probes),
                    },
                )
            return objective_loss

        # ── Persistent study with CMA-ES (in-memory for speed) ──
        # Version hash: changes when parameter space or training window changes.
        # A new hash creates a new study (old trials become irrelevant).
        range_blob = _ranges_signature_blob(ranges)
        objective_blob = _objective_signature_blob()
        study_tag = str(get_setting("optimizer_study_tag", "") or "").strip()
        version_blob = (
            range_blob
            + "|"
            + train_games[-1].game_date
            + "|"
            + objective_blob
        )
        if stage_label:
            version_blob += f"|stage:{stage_label}"
        if study_tag:
            version_blob += f"|tag:{study_tag}"
        if deterministic:
            version_blob += f"|deterministic:{deterministic_seed}"
        version_hash = hashlib.md5(version_blob.encode()).hexdigest()[:8]
        study_name = f"{'sharp' if include_sharp else 'fundamentals'}_{version_hash}"

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "optuna_studies.db")
        storage_url = f"sqlite:///{db_path}"

        reuse_study_cache = not deterministic

        # Reuse cached in-memory study across passes (avoids reloading 18K+ trials)
        if reuse_study_cache and study_name in _study_cache:
            study = _study_cache[study_name]
            prior_trials = len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE])
            loaded_champs = 0
            if (
                stage_champion_seed_enabled
                and stage_champion_seed_max > 0
                and n_trials > 0
            ):
                champion_seed_trials = _build_stage_champion_seed_trials(
                    optuna_module=optuna,
                    mode_key=champion_bank_mode_key,
                    stage_name=champion_bank_stage_key,
                    ranges=ranges,
                    baseline_weights=baseline_w,
                    max_entries=stage_champion_seed_max,
                )
                if champion_seed_trials:
                    existing_seed_hashes = {
                        str((t.user_attrs or {}).get("seed_weight_hash", ""))
                        for t in study.trials
                        if isinstance(getattr(t, "user_attrs", None), dict)
                        and str((t.user_attrs or {}).get("seed_source", "")) == "stage_champion_bank"
                        and str((t.user_attrs or {}).get("seed_mode", "")) == champion_bank_mode_key
                        and str((t.user_attrs or {}).get("seed_stage", "")) == champion_bank_stage_key
                    }
                    unique_seed_trials = [
                        t
                        for t in champion_seed_trials
                        if str((t.user_attrs or {}).get("seed_weight_hash", "")) not in existing_seed_hashes
                    ]
                    if unique_seed_trials:
                        study.add_trials(unique_seed_trials)
                        loaded_champs = len(unique_seed_trials)
            if callback:
                callback(f"Study '{study_name}': {prior_trials} trials in memory, "
                         f"loaded {loaded_champs} persistent stage-champion seeds, "
                         f"adding {n_trials} more")
        else:
            # First call: load from disk, then cache in memory.
            # NOTE: Optuna 4.4+ deprecates restart_strategy on CmaEsSampler.
            # Keep sampler args minimal to avoid runtime warning spam.
            suppress_independent_sampling_warn = _safe_bool_setting(
                "optimizer_cmaes_suppress_independent_sampling_warn",
                True,
            )
            sampler_kwargs = {
                "seed": deterministic_seed if deterministic else None,
            }
            if suppress_independent_sampling_warn:
                try:
                    sampler = optuna.samplers.CmaEsSampler(
                        warn_independent_sampling=False,
                        **sampler_kwargs,
                    )
                except TypeError:
                    sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
            else:
                sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)

            # Load best N trials from disk to warm-start CMA-ES.
            # Too many trials locks CMA-ES into a converged state;
            # too few loses the benefit of prior exploration.
            seed_disk_trials_enabled = _safe_bool_setting(
                "optimizer_seed_disk_trials_enabled",
                True,
            )
            max_seed_trials = _safe_int_setting(
                "optimizer_seed_disk_trials_max",
                600,
            )
            seed_top_fraction = float(
                np.clip(
                    _safe_float_setting("optimizer_seed_disk_trials_top_fraction", 0.6),
                    0.0,
                    1.0,
                )
            )
            prior_trials = 0
            seed_trials = []
            if seed_disk_trials_enabled and max_seed_trials > 0 and not deterministic:
                try:
                    disk_study = optuna.load_study(
                        study_name=study_name,
                        storage=storage_url,
                    )
                    disk_completed = [t for t in disk_study.trials
                                      if t.state == optuna.trial.TrialState.COMPLETE]
                    prior_trials = len(disk_completed)
                    if prior_trials > max_seed_trials:
                        # Use a blend of best + sampled tail to avoid
                        # over-anchoring CMA-ES to one stale local basin.
                        disk_completed.sort(key=lambda t: t.value)
                        top_count = max(
                            1,
                            min(max_seed_trials, int(round(max_seed_trials * seed_top_fraction))),
                        )
                        tail_count = max(0, max_seed_trials - top_count)
                        top_trials = disk_completed[:top_count]
                        sampled_tail = []
                        if tail_count > 0:
                            tail_pool_end = min(len(disk_completed), max_seed_trials * 8)
                            tail_pool = disk_completed[top_count:tail_pool_end]
                            if tail_pool:
                                sampled_tail = random.sample(
                                    tail_pool,
                                    min(tail_count, len(tail_pool)),
                                )
                        seed_trials = top_trials + sampled_tail
                    else:
                        seed_trials = disk_completed
                except KeyError:
                    pass

            champion_seed_trials = []
            if (
                stage_champion_seed_enabled
                and stage_champion_seed_max > 0
                and n_trials > 0
            ):
                champion_seed_trials = _build_stage_champion_seed_trials(
                    optuna_module=optuna,
                    mode_key=champion_bank_mode_key,
                    stage_name=champion_bank_stage_key,
                    ranges=ranges,
                    baseline_weights=baseline_w,
                    max_entries=stage_champion_seed_max,
                )

            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=sampler,
            )
            all_seed_trials = list(seed_trials) + list(champion_seed_trials)
            if all_seed_trials:
                study.add_trials(all_seed_trials)

            if reuse_study_cache:
                _study_cache[study_name] = study
            if callback:
                loaded = len(seed_trials)
                loaded_champs = len(champion_seed_trials)
                callback(f"Study '{study_name}': {prior_trials} prior trials on disk, "
                         f"loaded {loaded} disk seed trials + {loaded_champs} "
                         "persistent stage-champion seeds for CMA-ES (IPOP), "
                         f"adding {n_trials} more")

        # Log interval from config
        log_interval = int(get_setting("optimizer_log_interval", 300))
        _best_logged_loss = best_objective_loss
        _stagnation_counter = [0]
        _stagnation_threshold = int(get_setting("optuna_stagnation_threshold", 500))
        _early_stop_trials = int(get_setting("optuna_early_stop_trials", 2000))
        _min_trials_before_stop = int(get_setting("optuna_min_trials_before_stop", 500))

        # Track best weights for checkpoint saving
        _checkpoint_best_loss = [best_objective_loss]
        if ml_underdog_gate_enabled and callback:
            callback(
                "Checkpoint saves disabled while ML underdog gate is enabled."
            )

        def trial_callback(study, trial):
            nonlocal _best_logged_loss
            if is_cancelled and is_cancelled():
                if callback:
                    callback("Optimization cancelled by user. Stopping gracefully...")
                study.stop()
                return

            is_new_best = trial.value < _best_logged_loss
            if is_new_best:
                _best_logged_loss = trial.value
                _stagnation_counter[0] = 0

                # Checkpoint: save best weights immediately so they survive crashes.
                # Only checkpoint if meaningfully better than last checkpoint.
                if (
                    not skip_save
                    and not ml_underdog_gate_enabled
                    and trial.value < _checkpoint_best_loss[0] - 0.01
                ):
                    _checkpoint_best_loss[0] = trial.value
                    try:
                        ckpt_w = WeightConfig.from_dict(
                            {**baseline_w.to_dict(), **trial.params})
                        ckpt_val = vg_val.evaluate(ckpt_w, include_sharp=include_sharp)
                        ckpt_all = vg_all.evaluate(ckpt_w, include_sharp=include_sharp)
                        ckpt_ok, _, _ = _passes_robust_save_gate(
                            baseline=baseline_val,
                            candidate=ckpt_val,
                            n_validation_games=len(val_games),
                            baseline_all=baseline_all,
                            candidate_all=ckpt_all,
                        )
                        if ckpt_ok:
                            ckpt_delta = _max_weight_delta(last_saved_w[0], ckpt_w)
                            if ckpt_delta >= min_weight_delta:
                                save_weight_config(ckpt_w)
                                invalidate_weight_cache()
                                last_saved_w[0] = ckpt_w
                                logger.info("Checkpoint saved: train loss=%.3f, "
                                            "val winner=%.1f%%, dW=%.6f",
                                            trial.value, ckpt_val["winner_pct"], ckpt_delta)
                            else:
                                logger.info(
                                    "Checkpoint skipped (no-op): dW=%.6f < %.6f",
                                    ckpt_delta, min_weight_delta
                                )
                    except Exception:
                        logger.debug("checkpoint save skipped", exc_info=True)
            else:
                _stagnation_counter[0] += 1

            # Stagnation: warn at threshold intervals, early-stop at multiplied threshold
            if _stagnation_counter[0] > 0 and _stagnation_counter[0] % _stagnation_threshold == 0:
                if callback:
                    callback(f"  Stagnation: {_stagnation_counter[0]} trials "
                             f"without improvement (best loss={_best_logged_loss:.3f})")
                # Early stop after extended stagnation (once enough trials have run)
                if (_stagnation_counter[0] >= _early_stop_trials
                        and trial.number >= _min_trials_before_stop):
                    if callback:
                        callback(f"  Early stopping: {_stagnation_counter[0]} trials "
                                 f"without improvement after {trial.number} total trials")
                    study.stop()

            if trial.number % log_interval == 0 or (is_new_best and emit_stage_details):
                if callback:
                    res = trial.user_attrs.get("result", {})
                    win = res.get("winner_pct", 0)
                    upset_acc = res.get("upset_accuracy", 0)
                    upset_r = res.get("upset_rate", 0)
                    rolling_cv_meta = trial.user_attrs.get("rolling_cv", {})
                    cv_enabled = False
                    cv_score = float(res.get("loss", 0.0))
                    cv_mean = cv_score
                    cv_worst = cv_score
                    if isinstance(rolling_cv_meta, dict):
                        cv_enabled = bool(rolling_cv_meta.get("enabled", False))
                        cv_score = float(rolling_cv_meta.get("score", cv_score) or cv_score)
                        cv_mean = float(rolling_cv_meta.get("mean_loss", cv_score) or cv_score)
                        cv_worst = float(rolling_cv_meta.get("worst_loss", cv_score) or cv_score)
                    overfit_penalty = float(
                        trial.user_attrs.get("objective_overfit_penalty", 0.0)
                    )
                    l2_prior_penalty = float(
                        trial.user_attrs.get("objective_l2_prior_penalty", 0.0)
                    )
                    val_probe_meta = trial.user_attrs.get("val_probe", {})
                    probe_delta_mean = 0.0
                    probe_delta_max = 0.0
                    if isinstance(val_probe_meta, dict):
                        probe_delta_mean = float(
                            val_probe_meta.get("loss_delta_mean", 0.0) or 0.0
                        )
                        probe_delta_max = float(
                            val_probe_meta.get("loss_delta_max", 0.0) or 0.0
                        )
                    objective_tracks = trial.user_attrs.get("objective_tracks", {})
                    track_name = objective_track
                    live_loss = float(res.get("loss", 0.0))
                    oracle_loss = live_loss
                    if isinstance(objective_tracks, dict):
                        track_name = str(objective_tracks.get("track", track_name) or track_name)
                        live_loss = float(objective_tracks.get("live_loss", live_loss) or live_loss)
                        oracle_loss = float(
                            objective_tracks.get("oracle_loss", oracle_loss) or oracle_loss
                        )
                    callback(
                        f"Trial {trial.number}/{n_trials}: objective={trial.value:.3f} "
                        f"(train_loss={res.get('loss', 0.0):.3f}, "
                        f"track={track_name}, "
                        f"live_obj={live_loss:.3f}, "
                        f"oracle_obj={oracle_loss:.3f}, "
                        f"cv={'on' if cv_enabled else 'off'}, "
                        f"cv_score={cv_score:.3f}, "
                        f"cv_mean={cv_mean:.3f}, "
                        f"cv_worst={cv_worst:.3f}, "
                        f"overfit_penalty={overfit_penalty:.3f}, "
                        f"l2_prior_penalty={l2_prior_penalty:.3f}, "
                        f"probe_loss_delta_mean={probe_delta_mean:.3f}, "
                        f"probe_loss_delta_max={probe_delta_max:.3f}, "
                        f"Winner={win:.1f}%, "
                        f"Upset={upset_acc:.1f}% @ {upset_r:.1f}% rate)"
                    )
                    if is_new_best and emit_stage_details:
                        merged_params = {**baseline_w.to_dict(), **trial.params}
                        callback(
                            "  Best multipliers: "
                            f"pythag={float(merged_params.get('pythag_diff_mult', baseline_w.pythag_diff_mult)):.3f}, "
                            f"road_trip={float(merged_params.get('road_trip_game_mult', baseline_w.road_trip_game_mult)):.3f}, "
                            f"cum_travel_7d={float(merged_params.get('cum_travel_7d_mult', baseline_w.cum_travel_7d_mult)):.3f}"
                        )

        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

        # Cache trial list once (avoid repeated iteration over thousands of objects)
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        total_trials = len(completed)

        # Persist new trials to disk in background (truly non-blocking)
        new_trials = completed[prior_trials:]
        if new_trials:
            import threading

            # Snapshot the trials to save (list copy so background thread is safe)
            trials_to_save = list(new_trials)

            def _save_to_disk():
                # Wait for any previous save to finish (avoid SQLite lock contention)
                prev = _save_threads.get(study_name)
                if prev is not None and prev is not threading.current_thread() and prev.is_alive():
                    prev.join()
                try:
                    disk_save = optuna.create_study(
                        study_name=study_name,
                        storage=storage_url,
                        direction="minimize",
                        load_if_exists=True,
                    )
                    disk_save.add_trials(trials_to_save)
                    logger.info("Saved %d new trials to disk (%d total)",
                                len(trials_to_save), total_trials)
                except Exception as e:
                    logger.warning("Failed to save trials to disk: %s", e)

            save_thread = threading.Thread(target=_save_to_disk, daemon=True)
            save_thread.start()
            _save_threads[study_name] = save_thread

        if callback:
            callback(f"Study has {total_trials} total trials "
                     f"(saving {len(new_trials)} to disk in background)")

        # Evaluate top-N training trials on validation to find best generalizer
        top_n_raw = int(get_setting("optuna_top_n_validation", 10))
        top_n = max(30, top_n_raw)
        if completed:
            top_n = min(top_n, len(completed))
        if callback and top_n != top_n_raw:
            callback(
                f"optuna_top_n_validation={top_n_raw} adjusted to {top_n} "
                "(minimum 30 for validation generalization)"
            )
        completed.sort(key=lambda t: t.value)
        candidates = completed[:top_n]

        if (
            provided_candidate is None
            and candidates
            and candidates[0].value < best_objective_loss
        ):
            best_candidate_objective = float("inf")
            best_candidate_val_loss = float("inf")
            chosen_rank = 0
            if callback and emit_stage_details:
                callback(f"Validating top {len(candidates)} training trials...")

            for rank, trial in enumerate(candidates):
                cand_w = WeightConfig.from_dict(
                    {**baseline_w.to_dict(), **trial.params})
                cand_val = vg_val.evaluate(cand_w, include_sharp=include_sharp)
                cand_val_loss = float(cand_val["loss"])
                cand_live_objective = float(trial.user_attrs.get("result", {}).get("loss", trial.value))
                cand_cv_mean = cand_live_objective
                cand_cv_worst = cand_live_objective
                if rolling_cv_enabled and rolling_cv_val_windows:
                    cand_fold_losses = []
                    for vg_fold_val in rolling_cv_val_windows:
                        fold_result = vg_fold_val.evaluate(
                            cand_w,
                            include_sharp=include_sharp,
                            fast=True,
                        )
                        cand_fold_losses.append(float(fold_result.get("loss", 0.0)))
                    cand_live_objective, cand_cv_mean, cand_cv_worst = _rolling_cv_objective_loss(
                        cand_fold_losses,
                        rolling_cv_worst_fold_mult,
                    )
                cand_oracle_objective = float(
                    vg_all.evaluate(cand_w, include_sharp=include_sharp, fast=True).get("loss", 0.0)
                )
                cand_objective = _compose_objective_loss(
                    cand_live_objective,
                    cand_oracle_objective,
                    objective_track,
                    objective_dual_live_weight,
                )

                if callback and emit_stage_details and rank < 5:
                    tr = trial.user_attrs.get("result", {})
                    callback(
                        f"  #{rank + 1} objective={cand_objective:.3f} "
                        f"(track={objective_track}, live={cand_live_objective:.3f}, "
                        f"oracle={cand_oracle_objective:.3f}, blend_w={objective_dual_live_weight:.2f}, "
                        f"cv mean={cand_cv_mean:.3f}, cv worst={cand_cv_worst:.3f}) "
                        f"train loss={trial.value:.3f} "
                        f"(Winner={tr.get('winner_pct', 0):.1f}%) "
                        f"-> val loss={cand_val_loss:.3f} "
                        f"(Winner={cand_val.get('winner_pct', 0):.1f}%)")

                if (
                    cand_objective < best_candidate_objective
                    or (
                        abs(cand_objective - best_candidate_objective) <= 1e-9
                        and cand_val_loss < best_candidate_val_loss
                    )
                ):
                    best_candidate_objective = cand_objective
                    best_candidate_val_loss = cand_val_loss
                    best_w = cand_w
                    best_objective_loss = cand_objective
                    best_live_objective_loss = cand_live_objective
                    best_oracle_objective_loss = cand_oracle_objective
                    best_train_loss = float(
                        trial.user_attrs.get("result", {}).get("loss", trial.value)
                    )
                    chosen_rank = rank

            if callback and emit_stage_details and chosen_rank > 0:
                callback(
                    f"Selected trial #{chosen_rank + 1} "
                    f"(not #1) -- better robust objective")

    except ImportError:
        if callback:
            callback("Optuna not installed, using random search...")
        for i in range(n_trials):
            if is_cancelled and is_cancelled():
                if callback:
                    callback("Optimization cancelled by user.")
                break
            params = {}
            for key, (lo, hi) in ranges.items():
                params[key] = random.uniform(lo, hi)
            w = WeightConfig.from_dict({**baseline_w.to_dict(), **params})
            result = vg_train.evaluate(w, include_sharp=include_sharp, fast=True)
            live_objective = float(result.get("loss", 0.0))
            oracle_objective = float(
                vg_all.evaluate(w, include_sharp=include_sharp, fast=True).get("loss", 0.0)
            )
            selected_objective = _compose_objective_loss(
                live_objective,
                oracle_objective,
                objective_track,
                objective_dual_live_weight,
            )
            if selected_objective < best_objective_loss:
                best_w = w
                best_objective_loss = selected_objective
                best_live_objective_loss = live_objective
                best_oracle_objective_loss = oracle_objective
                best_train_loss = live_objective
            if callback and (i + 1) % 300 == 0:
                callback(f"Random trial {i + 1}/{n_trials}: "
                         f"best_objective={best_objective_loss:.3f}")

    if provided_candidate is not None:
        best_w = provided_candidate
        if callback and emit_stage_details:
            callback(
                f"[stage:{stage_label}] candidate override supplied "
                "(search results bypassed)"
            )

    # Walk-forward validation
    best_train_full = vg_train.evaluate(best_w, include_sharp=include_sharp)
    best_val = vg_val.evaluate(best_w, include_sharp=include_sharp)
    best_all = vg_all.evaluate(best_w, include_sharp=include_sharp)
    best_train_loss = float(best_train_full.get("loss", best_train_loss))
    candidate_cv_score: Optional[float] = None
    candidate_cv_mean: Optional[float] = None
    candidate_cv_worst: Optional[float] = None
    if rolling_cv_enabled and rolling_cv_val_windows:
        candidate_cv_fold_losses = []
        for vg_fold_val in rolling_cv_val_windows:
            fold_result = vg_fold_val.evaluate(
                best_w,
                include_sharp=include_sharp,
                fast=True,
            )
            candidate_cv_fold_losses.append(float(fold_result.get("loss", 0.0)))
        candidate_cv_score, candidate_cv_mean, candidate_cv_worst = _rolling_cv_objective_loss(
            candidate_cv_fold_losses,
            rolling_cv_worst_fold_mult,
        )
    candidate_live_objective = float(
        candidate_cv_score
        if rolling_cv_enabled and candidate_cv_score is not None
        else best_train_full["loss"]
    )
    candidate_oracle_objective = float(best_all.get("loss", candidate_live_objective))
    candidate_selected_objective = _compose_objective_loss(
        candidate_live_objective,
        candidate_oracle_objective,
        objective_track,
        objective_dual_live_weight,
    )
    best_live_objective_loss = candidate_live_objective
    best_oracle_objective_loss = candidate_oracle_objective
    best_objective_loss = candidate_selected_objective

    if callback and emit_stage_details:
        callback("-- Walk-forward results --")
        callback(f"  Train:  Winner={best_train_full['winner_pct']:.1f}%, "
                 f"Upset={best_train_full['upset_accuracy']:.1f}% "
                 f"@ {best_train_full['upset_rate']:.1f}% rate, "
                 f"CompDog={best_train_full.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={best_train_full.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={best_train_full.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={best_train_full['loss']:.3f}")
        callback(f"    {best_train_full.get('hit_rate_quality_observation', '')}")
        callback(f"  Valid:  Winner={best_val['winner_pct']:.1f}%, "
                 f"Upset={best_val['upset_accuracy']:.1f}% "
                 f"@ {best_val['upset_rate']:.1f}% rate, "
                 f"CompDog={best_val.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={best_val.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={best_val.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Favorites={best_val['favorites_pct']:.1f}%, "
                 f"Loss={best_val['loss']:.3f}")
        callback(f"    {best_val.get('hit_rate_quality_observation', '')}")
        callback(f"  All:    Winner={best_all['winner_pct']:.1f}%, "
                 f"Upset={best_all['upset_accuracy']:.1f}% "
                 f"@ {best_all['upset_rate']:.1f}% rate, "
                 f"CompDog={best_all.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={best_all.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={best_all.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={best_all['loss']:.3f}")
        callback(f"    {best_all.get('hit_rate_quality_observation', '')}")
        callback(
            "Objective tracks (candidate): "
            f"selected={best_objective_loss:.3f} "
            f"(mode={objective_track}, live={best_live_objective_loss:.3f}, "
            f"oracle={best_oracle_objective_loss:.3f}, "
            f"dual_live_weight={objective_dual_live_weight:.2f})"
        )

    # Save gate: robust anti-gaming guardrails on validation
    baseline_winner_pct = baseline_val.get("winner_pct", 0)
    favorites_pct = best_val.get("favorites_pct", 0)
    best_winner_pct = best_val.get("winner_pct", 0)
    if skip_save:
        save_ok = False
        save_reason = f"stage search only ({stage_label})"
        save_details = {
            "stage_label": stage_label,
            "skip_save": True,
            "rolling_cv_enabled": rolling_cv_enabled,
            "rolling_cv_fold_count": len(rolling_cv_val_windows),
            "rolling_cv_worst_fold_mult": rolling_cv_worst_fold_mult,
            "baseline_cv_score": baseline_cv_score if rolling_cv_enabled else None,
            "baseline_cv_mean_loss": baseline_cv_mean if rolling_cv_enabled else None,
            "baseline_cv_worst_loss": baseline_cv_worst if rolling_cv_enabled else None,
            "candidate_cv_score": candidate_cv_score if rolling_cv_enabled else None,
            "candidate_cv_mean_loss": candidate_cv_mean if rolling_cv_enabled else None,
            "candidate_cv_worst_loss": candidate_cv_worst if rolling_cv_enabled else None,
            "objective_track": objective_track,
            "objective_dual_live_weight": objective_dual_live_weight,
            "baseline_live_objective_loss": baseline_live_objective,
            "baseline_oracle_objective_loss": baseline_oracle_objective,
            "baseline_selected_objective_loss": baseline_selected_objective,
            "candidate_live_objective_loss": candidate_live_objective,
            "candidate_oracle_objective_loss": candidate_oracle_objective,
            "candidate_selected_objective_loss": candidate_selected_objective,
            "candidate_only": True,
        }
    else:
        save_ok, save_reason, save_details = _passes_robust_save_gate(
            baseline=baseline_val,
            candidate=best_val,
            n_validation_games=len(val_games),
            baseline_all=baseline_all,
            candidate_all=best_all,
        )
        save_details["rolling_cv_enabled"] = rolling_cv_enabled
        save_details["rolling_cv_fold_count"] = len(rolling_cv_val_windows)
        save_details["rolling_cv_worst_fold_mult"] = rolling_cv_worst_fold_mult
        save_details["baseline_cv_score"] = baseline_cv_score if rolling_cv_enabled else None
        save_details["baseline_cv_mean_loss"] = baseline_cv_mean if rolling_cv_enabled else None
        save_details["baseline_cv_worst_loss"] = baseline_cv_worst if rolling_cv_enabled else None
        save_details["candidate_cv_score"] = candidate_cv_score if rolling_cv_enabled else None
        save_details["candidate_cv_mean_loss"] = candidate_cv_mean if rolling_cv_enabled else None
        save_details["candidate_cv_worst_loss"] = candidate_cv_worst if rolling_cv_enabled else None
        save_details["objective_track"] = objective_track
        save_details["objective_dual_live_weight"] = objective_dual_live_weight
        save_details["baseline_live_objective_loss"] = baseline_live_objective
        save_details["baseline_oracle_objective_loss"] = baseline_oracle_objective
        save_details["baseline_selected_objective_loss"] = baseline_selected_objective
        save_details["candidate_live_objective_loss"] = candidate_live_objective
        save_details["candidate_oracle_objective_loss"] = candidate_oracle_objective
        save_details["candidate_selected_objective_loss"] = candidate_selected_objective

    ml_gate_result: Dict[str, Any] = {
        "enabled": bool(ml_underdog_gate_enabled),
        "applied": False,
        "passed": True,
        "reason": "disabled",
    }
    if ml_underdog_gate_enabled:
        ml_gate_result = compare_walk_forward_underdog_scorer(
            train_games=train_games,
            val_games=val_games,
            baseline_weights=baseline_w,
            candidate_weights=best_w,
            include_sharp=include_sharp,
            min_train_samples=ml_underdog_min_train_samples,
            min_val_samples=ml_underdog_min_val_samples,
            learning_rate=ml_underdog_lr,
            l2=ml_underdog_l2,
            max_iter=ml_underdog_max_iter,
            min_brier_improvement=ml_underdog_min_brier_lift,
        )
        if callback:
            if ml_gate_result.get("applied"):
                baseline_ml = ml_gate_result.get("baseline", {})
                candidate_ml = ml_gate_result.get("candidate", {})
                baseline_brier = float(
                    (baseline_ml or {}).get("brier", 0.0) or 0.0
                )
                candidate_brier = float(
                    (candidate_ml or {}).get("brier", 0.0) or 0.0
                )
                baseline_logloss = float(
                    (baseline_ml or {}).get("logloss", 0.0) or 0.0
                )
                candidate_logloss = float(
                    (candidate_ml or {}).get("logloss", 0.0) or 0.0
                )
                callback(
                    "ML underdog diagnostics: "
                    f"brier {baseline_brier:.4f}->{candidate_brier:.4f} "
                    f"(lift {float(ml_gate_result.get('brier_lift', 0.0)):+.4f}), "
                    f"logloss {baseline_logloss:.4f}->{candidate_logloss:.4f} "
                    f"(lift {float(ml_gate_result.get('logloss_lift', 0.0)):+.4f}), "
                    f"gate {'PASS' if ml_gate_result.get('passed') else 'FAIL'} "
                    f"(min +{ml_underdog_min_brier_lift:.4f})"
                )
            else:
                callback(
                    f"ML underdog diagnostics: skipped ({ml_gate_result.get('reason', 'n/a')})"
                )
        if not bool(ml_gate_result.get("passed", True)):
            save_ok = False
            ml_reason = str(ml_gate_result.get("reason", "")).strip()
            if save_reason == "pass":
                save_reason = ml_reason or "ml underdog gate failed"
            elif ml_reason:
                save_reason = f"{save_reason}; {ml_reason}"
    save_details["ml_underdog_gate_enabled"] = bool(ml_gate_result.get("enabled", False))
    save_details["ml_underdog_gate_applied"] = bool(ml_gate_result.get("applied", False))
    save_details["ml_underdog_gate_passed"] = bool(ml_gate_result.get("passed", True))
    save_details["ml_underdog_gate_reason"] = str(ml_gate_result.get("reason", ""))
    save_details["ml_underdog_gate_min_brier_lift"] = ml_underdog_min_brier_lift
    save_details["ml_underdog_gate_brier_lift"] = ml_gate_result.get("brier_lift")
    save_details["ml_underdog_gate_logloss_lift"] = ml_gate_result.get("logloss_lift")
    ml_gate_baseline = ml_gate_result.get("baseline", {})
    ml_gate_candidate = ml_gate_result.get("candidate", {})
    save_details["ml_underdog_gate_baseline_brier"] = (
        (ml_gate_baseline or {}).get("brier")
        if isinstance(ml_gate_baseline, dict)
        else None
    )
    save_details["ml_underdog_gate_candidate_brier"] = (
        (ml_gate_candidate or {}).get("brier")
        if isinstance(ml_gate_candidate, dict)
        else None
    )
    save_details["ml_underdog_gate_baseline_logloss"] = (
        (ml_gate_baseline or {}).get("logloss")
        if isinstance(ml_gate_baseline, dict)
        else None
    )
    save_details["ml_underdog_gate_candidate_logloss"] = (
        (ml_gate_candidate or {}).get("logloss")
        if isinstance(ml_gate_candidate, dict)
        else None
    )
    save_details["ml_underdog_gate"] = ml_gate_result

    if callback and emit_stage_details:
        callback(
            "Save gate diagnostics: "
            f"loss(val) {baseline_val.get('loss', 0.0):.3f}->{best_val.get('loss', 0.0):.3f} "
            f"(delta {best_val.get('loss', 0.0) - baseline_val.get('loss', 0.0):+.3f}; lower is better), "
            f"loss(all) {baseline_all.get('loss', 0.0):.3f}->{best_all.get('loss', 0.0):.3f} "
            f"(delta {best_all.get('loss', 0.0) - baseline_all.get('loss', 0.0):+.3f}), "
            f"compDog(val) {baseline_val.get('competitive_dog_rate', 0.0):.1f}%"
            f"->{best_val.get('competitive_dog_rate', 0.0):.1f}%, "
            f"onePosDog(val) {baseline_val.get('one_possession_dog_rate', 0.0):.1f}%"
            f"->{best_val.get('one_possession_dog_rate', 0.0):.1f}%, "
            f"longDog1P(val) {baseline_val.get('long_dog_onepos_rate', 0.0):.1f}%"
            f"->{best_val.get('long_dog_onepos_rate', 0.0):.1f}%, "
            f"hybrid {float(save_details.get('baseline_hybrid_loss', baseline_val.get('loss', 0.0))):.3f}"
            f"->{float(save_details.get('candidate_hybrid_loss', best_val.get('loss', 0.0))):.3f} "
            f"(delta {float(save_details.get('candidate_hybrid_loss', best_val.get('loss', 0.0))) - float(save_details.get('baseline_hybrid_loss', baseline_val.get('loss', 0.0))):+.3f}), "
            f"ROI {baseline_val.get('ml_roi', 0.0):+.2f}%->{best_val.get('ml_roi', 0.0):+.2f}% "
            f"(lb95 {best_val.get('ml_roi_lb95', 0.0):+.2f}%), "
            f"min ML payout {best_val.get('ml_min_payout', MIN_ML_PAYOUT):.2f}x, "
            f"ROI gate {'ON' if bool(save_details.get('use_roi_gate', False)) else 'OFF'}, "
            f"long-dog tiebreak {'ON' if bool(save_details.get('use_long_dog_tiebreak_gate', False)) else 'OFF'}, "
            f"hybrid loss gate {'ON' if bool(save_details.get('use_hybrid_loss_gate', False)) else 'OFF'}, "
            f"shrunk upset lift {float(save_details.get('shrunk_upset_lift', 0.0)):+.2f}pp"
        )

    weight_delta = _max_weight_delta(last_saved_w[0], best_w)
    weight_change_ok = weight_delta >= min_weight_delta
    save_details["weight_delta"] = weight_delta
    save_details["min_weight_delta"] = min_weight_delta
    save_details["weight_change_ok"] = weight_change_ok
    did_save = False

    if save_ok and weight_change_ok:
        save_weight_config(best_w)
        invalidate_weight_cache()
        last_saved_w[0] = best_w
        did_save = True
        if callback:
            gate_note = "" if save_reason == "pass" else f", {save_reason}"
            callback(f"Saved optimized weights "
                     f"(val Winner: {baseline_winner_pct:.1f}% -> {best_winner_pct:.1f}%, "
                     f"vs Favorites baseline: {favorites_pct:.1f}%, "
                     f"dW={weight_delta:.6f}{gate_note})")
    elif save_ok:
        save_ok = False
        save_reason = (f"no-op weight update (dW {weight_delta:.6f} "
                       f"< {min_weight_delta:.6f})")
        if callback:
            callback(f"Validation save gate rejected: {save_reason} "
                     f"- keeping current weights")
    else:
        if callback and not skip_save:
            callback(f"Validation save gate rejected: {save_reason} "
                     f"- keeping current weights")
        elif callback and skip_save:
            callback(f"[stage:{stage_label}] Candidate selected (save skipped)")

    # Checkpoint saves can happen during Optuna callbacks before final selection.
    # If final selection is a no-op versus last checkpoint, report improved=True
    # so overnight no-save logic does not treat this pass as a false negative.
    checkpoint_delta = _max_weight_delta(baseline_w, last_saved_w[0])
    checkpoint_saved = (
        checkpoint_delta >= min_weight_delta and not ml_underdog_gate_enabled and not skip_save
    )
    save_details["checkpoint_weight_delta"] = checkpoint_delta
    save_details["checkpoint_saved"] = checkpoint_saved
    save_details["checkpoint_disabled_by_ml_gate"] = ml_underdog_gate_enabled
    if checkpoint_saved and not did_save:
        did_save = True
        if save_reason.startswith("no-op weight update"):
            save_reason = "pass (checkpoint already saved optimized weights)"

    return {
        "baseline_loss": baseline_val["loss"],
        "best_loss": best_val["loss"],
        "baseline_all_loss": baseline_all["loss"],
        "best_all_loss": best_all["loss"],
        "baseline_winner_pct": baseline_winner_pct,
        "best_winner_pct": best_winner_pct,
        "favorites_pct": favorites_pct,
        "improved": did_save,
        "save_gate_reason": save_reason,
        "save_gate_details": save_details,
        "train_loss": best_train_full["loss"],
        "trial_train_loss": best_train_loss,
        "objective_loss": best_objective_loss,
        "objective_mode": "rolling_cv" if rolling_cv_enabled else "train_loss",
        "objective_track": objective_track,
        "objective_dual_live_weight": objective_dual_live_weight,
        "objective_live_loss": best_live_objective_loss,
        "objective_oracle_loss": best_oracle_objective_loss,
        "objective_selected_loss": best_objective_loss,
        "rolling_cv_enabled": rolling_cv_enabled,
        "rolling_cv_fold_count": len(rolling_cv_val_windows),
        "ml_underdog_gate": ml_gate_result,
        "deterministic": deterministic,
        "deterministic_seed": deterministic_seed if deterministic else None,
        "best_weights": best_w.to_dict(),
        "stage_label": stage_label,
        "skip_save": bool(skip_save),
        **best_val,
    }


def compare_modes(
    games: List[GameInput],
    callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run both fundamentals-only and fundamentals+sharp on the same validation data.

    Returns comparison including which picks were flipped by sharp money
    and whether those flips were accurate.
    """
    # Use the validation portion (same split as optimizer)
    sorted_games = sorted(games, key=lambda g: g.game_date)
    split_idx = int(len(sorted_games) * WALK_FORWARD_SPLIT)
    val_games = sorted_games[split_idx:]

    if not val_games:
        if callback:
            callback("No validation games for comparison")
        return {}

    vg = VectorizedGames(val_games)
    w = get_weight_config()

    # Evaluate both modes
    fund_result = vg.evaluate(w, include_sharp=False)
    sharp_result = vg.evaluate(w, include_sharp=True)

    if callback:
        callback(f"Fundamentals only:  Winner={fund_result['winner_pct']:.1f}%, "
                 f"Upset={fund_result['upset_accuracy']:.1f}% "
                 f"@ {fund_result['upset_rate']:.1f}% rate, "
                 f"CompDog={fund_result.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={fund_result.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={fund_result.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"ML ROI={fund_result['ml_roi']:+.1f}%")
        callback(f"Fundamentals+Sharp: Winner={sharp_result['winner_pct']:.1f}%, "
                 f"Upset={sharp_result['upset_accuracy']:.1f}% "
                 f"@ {sharp_result['upset_rate']:.1f}% rate, "
                 f"CompDog={sharp_result.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={sharp_result.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={sharp_result.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"ML ROI={sharp_result['ml_roi']:+.1f}%")

    # Compute which picks were flipped by sharp money
    # Reconstruct game_score for both modes to identify flips
    away_def_f = 1.0 + (vg.away_def_factor_raw - 1.0) * w.def_factor_dampening
    home_def_f = 1.0 + (vg.home_def_factor_raw - 1.0) * w.def_factor_dampening
    home_base = vg.home_pts_raw * away_def_f
    away_base = vg.away_pts_raw * home_def_f
    gs = (home_base - away_base) + vg.home_court

    home_fat = (vg.home_b2b_flag * w.fatigue_b2b + vg.home_3in4 * w.fatigue_3in4
                + vg.home_4in6 * w.fatigue_4in6 + vg.home_same_day * w.fatigue_same_day
                - vg.home_rest_tier * w.fatigue_rest_bonus)
    away_fat = (vg.away_b2b_flag * w.fatigue_b2b + vg.away_3in4 * w.fatigue_3in4
                + vg.away_4in6 * w.fatigue_4in6 + vg.away_same_day * w.fatigue_same_day
                - vg.away_rest_tier * w.fatigue_rest_bonus)
    gs -= (home_fat - away_fat)
    gs += vg.to_diff * w.turnover_margin_mult
    gs += vg.reb_diff * w.rebound_diff_mult
    home_me = vg.home_off - vg.away_def
    away_me = vg.away_off - vg.home_def
    gs += (home_me - away_me) * w.rating_matchup_mult
    ff = (vg.ff_efg_edge * w.ff_efg_weight + vg.ff_tov_edge * w.ff_tov_weight
          + vg.ff_oreb_edge * w.ff_oreb_weight + vg.ff_fta_edge * w.ff_fta_weight
          ) * FOUR_FACTORS_FIXED_SCALE
    gs += ff
    opp_ff = (vg.opp_ff_efg_edge * w.opp_ff_efg_weight
              + vg.opp_ff_tov_edge * w.opp_ff_tov_weight
              + vg.opp_ff_oreb_edge * w.opp_ff_oreb_weight
              + vg.opp_ff_fta_edge * w.opp_ff_fta_weight
              ) * FOUR_FACTORS_FIXED_SCALE
    gs += opp_ff
    clutch_mask = np.abs(gs) < w.clutch_threshold
    clutch_adj = np.clip(vg.clutch_diff * w.clutch_scale, -w.clutch_cap, w.clutch_cap)
    gs += clutch_adj * clutch_mask
    home_eff = vg.home_defl + vg.home_contested * w.hustle_contested_wt
    away_eff = vg.away_defl + vg.away_contested * w.hustle_contested_wt
    gs += (home_eff - away_eff) * w.hustle_effort_mult
    gs += vg.net_rest * w.rest_advantage_mult
    gs -= vg.away_b2b_at_altitude * w.altitude_b2b_penalty
    onoff_lambda = max(0.0, float(getattr(w, "onoff_reliability_lambda", 0.0)))
    home_onoff = vg._home_onoff_signal * (
        vg._home_onoff_reliability / (vg._home_onoff_reliability + onoff_lambda + 1e-9)
    )
    away_onoff = vg._away_onoff_signal * (
        vg._away_onoff_reliability / (vg._away_onoff_reliability + onoff_lambda + 1e-9)
    )
    gs += (home_onoff - away_onoff) * w.onoff_impact_mult

    # Fundamentals-only picks
    fund_picks_home = gs > MODEL_PICK_EDGE_THRESHOLD

    # With sharp
    gs_sharp = gs + vg.sharp_ml_edge * w.sharp_ml_weight
    sharp_picks_home = gs_sharp > MODEL_PICK_EDGE_THRESHOLD

    # Flipped picks
    flipped = fund_picks_home != sharp_picks_home
    n_flipped = int(np.sum(flipped))

    # Accuracy of flipped picks
    actual_home_win = vg.actual_spread > ACTUAL_WIN_THRESHOLD
    actual_away_win = vg.actual_spread < -ACTUAL_WIN_THRESHOLD
    if n_flipped > 0:
        flipped_correct = (
            (sharp_picks_home[flipped] & actual_home_win[flipped])
            | (~sharp_picks_home[flipped] & actual_away_win[flipped])
        )
        flipped_accuracy = float(np.mean(flipped_correct)) * 100.0
    else:
        flipped_accuracy = 0.0

    # Net contribution: sharp winner% minus fundamentals winner%
    net_contribution = sharp_result["winner_pct"] - fund_result["winner_pct"]

    comparison = {
        "fundamentals": fund_result,
        "sharp": sharp_result,
        "picks_flipped": n_flipped,
        "flipped_accuracy": flipped_accuracy,
        "net_contribution": net_contribution,
        "validation_games": len(val_games),
    }

    if callback:
        callback(f"Sharp flipped {n_flipped} picks "
                 f"({flipped_accuracy:.1f}% accurate), "
                 f"net contribution: {net_contribution:+.2f}% winner")

    return comparison


def coordinate_descent(
    games: List[GameInput],
    params: Optional[List[str]] = None,
    steps: int = 100,
    max_rounds: int = 10,
    convergence_threshold: float = 0.005,
    include_sharp: bool = False,
    callback: Optional[Callable] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Grid-search refinement of individual parameters after Optuna TPE.

    For each parameter, evaluates `steps` equally-spaced values across the
    CD_RANGES bounds.  Accepts a new value only when it improves training loss.
    Repeats for up to `max_rounds` until convergence.

    Save gate: anti-gaming validation gate blends loss, ROI, and upset lift
    (same gate as optimize_weights).
    """
    # Walk-forward split (same as Optuna)
    sorted_games = sorted(games, key=lambda g: g.game_date)
    split_idx = int(len(sorted_games) * WALK_FORWARD_SPLIT)
    train_games = sorted_games[:split_idx]
    val_games = sorted_games[split_idx:]

    if not train_games or not val_games:
        if callback:
            callback("Not enough games for walk-forward split")
        return {"improved": False, "rounds": 0}

    vg_train = VectorizedGames(train_games)
    vg_val = VectorizedGames(val_games)
    vg_all = VectorizedGames(sorted_games)

    if callback:
        callback(f"CD: {len(train_games)} train, {len(val_games)} validation")

    # Select ranges and parameters
    ranges = CD_SHARP_RANGES if include_sharp else CD_RANGES
    if params is None:
        params = list(ranges.keys())

    if callback:
        callback(f"CD: {len(params)} parameters, {steps} steps/param, "
                 f"max {max_rounds} rounds")

    # Load current weights as starting point
    w = get_weight_config()
    w_dict = w.to_dict()

    # Baseline evaluation
    baseline_train = vg_train.evaluate(w, include_sharp=include_sharp)
    baseline_val = vg_val.evaluate(w, include_sharp=include_sharp)
    baseline_all = vg_all.evaluate(w, include_sharp=include_sharp)
    current_train_loss = baseline_train["loss"]
    min_weight_delta = _safe_float_setting("optimizer_save_min_weight_delta", 1e-4)

    if callback:
        callback(f"CD baseline (train): Winner={baseline_train['winner_pct']:.1f}%, "
                 f"CompDog={baseline_train.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={baseline_train.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={baseline_train.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={baseline_train['loss']:.3f}")
        callback(f"CD baseline (valid): Winner={baseline_val['winner_pct']:.1f}%, "
                 f"CompDog={baseline_val.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={baseline_val.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={baseline_val.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Favorites={baseline_val['favorites_pct']:.1f}%")
        callback(f"CD baseline (all):   Winner={baseline_all['winner_pct']:.1f}%, "
                 f"CompDog={baseline_all.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={baseline_all.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={baseline_all.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={baseline_all['loss']:.3f}")

    best_w_dict = w_dict.copy()
    history = []
    all_changes = {}
    prev_val_loss = baseline_val["loss"]

    for round_num in range(1, max_rounds + 1):
        if is_cancelled and is_cancelled():
            if callback:
                callback("CD cancelled by user.")
            break

        accepted_count = 0

        if callback:
            callback(f"--- Round {round_num}/{max_rounds} ---")

        for p_idx, param_name in enumerate(params):
            if is_cancelled and is_cancelled():
                break

            lo, hi = ranges.get(param_name, (0, 1))
            grid = np.linspace(lo, hi, steps)

            best_param_loss = current_train_loss
            best_param_val = w_dict[param_name]

            for val in grid:
                test_dict = {**w_dict, param_name: float(val)}
                test_w = WeightConfig.from_dict(test_dict)
                result = vg_train.evaluate(test_w, include_sharp=include_sharp, fast=True)
                if result["loss"] < best_param_loss:
                    best_param_loss = result["loss"]
                    best_param_val = float(val)

            # Accept if training loss improved
            if best_param_loss < current_train_loss:
                old_val = w_dict[param_name]
                w_dict[param_name] = best_param_val
                current_train_loss = best_param_loss
                best_w_dict = w_dict.copy()
                accepted_count += 1
                all_changes[param_name] = {
                    "before": old_val,
                    "after": best_param_val,
                }
                if callback:
                    callback(f"  {param_name}: {old_val:.4f} -> "
                             f"{best_param_val:.4f} "
                             f"(loss {current_train_loss:.4f}) KEPT")
            else:
                if callback and p_idx % 5 == 0:
                    callback(f"  [{p_idx + 1}/{len(params)}] {param_name}: no improvement")

        # End-of-round: evaluate on validation
        round_w = WeightConfig.from_dict(best_w_dict)
        round_val = vg_val.evaluate(round_w, include_sharp=include_sharp)
        round_train = vg_train.evaluate(round_w, include_sharp=include_sharp)

        round_info = {
            "round": round_num,
            "accepted": accepted_count,
            "train_loss": round_train["loss"],
            "train_winner_pct": round_train["winner_pct"],
            "val_loss": round_val["loss"],
            "val_winner_pct": round_val["winner_pct"],
            "val_favorites_pct": round_val["favorites_pct"],
        }
        history.append(round_info)

        if callback:
            callback(f"Round {round_num} summary: "
                     f"{accepted_count}/{len(params)} params accepted, "
                     f"val Winner={round_val['winner_pct']:.1f}% "
                     f"(fav={round_val['favorites_pct']:.1f}%), "
                     f"train_loss={round_train['loss']:.3f}")

        # Convergence checks
        if accepted_count == 0:
            if callback:
                callback("No parameters improved this round. Stopping.")
            break

        val_improvement = abs(prev_val_loss - round_val["loss"])
        if round_num >= 2 and val_improvement < convergence_threshold:
            if callback:
                callback(f"Converged (improvement {val_improvement:.4f} "
                         f"< threshold {convergence_threshold}). Stopping.")
            break

        prev_val_loss = round_val["loss"]

    # Final evaluation
    final_w = WeightConfig.from_dict(best_w_dict)
    final_val = vg_val.evaluate(final_w, include_sharp=include_sharp)
    final_train = vg_train.evaluate(final_w, include_sharp=include_sharp)
    final_all = vg_all.evaluate(final_w, include_sharp=include_sharp)

    # Save gate: same robust anti-gaming gate as optimize_weights
    baseline_winner_pct = baseline_val.get("winner_pct", 0)
    favorites_pct = final_val.get("favorites_pct", 0)
    best_winner_pct = final_val.get("winner_pct", 0)
    save_ok, save_reason, save_details = _passes_robust_save_gate(
        baseline=baseline_val,
        candidate=final_val,
        n_validation_games=len(val_games),
        baseline_all=baseline_all,
        candidate_all=final_all,
    )

    if callback:
        callback("--- CD Final ---")
        callback(f"  Train:  Winner={final_train['winner_pct']:.1f}%, "
                 f"CompDog={final_train.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={final_train.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={final_train.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={final_train['loss']:.3f}")
        callback(f"  Valid:  Winner={best_winner_pct:.1f}% "
                 f"CompDog={final_val.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={final_val.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={final_val.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"(was {baseline_winner_pct:.1f}%), "
                 f"Favorites={favorites_pct:.1f}%")
        callback(f"  All:    Winner={final_all['winner_pct']:.1f}% "
                 f"CompDog={final_all.get('competitive_dog_rate', 0.0):.1f}%, "
                 f"OnePosDog={final_all.get('one_possession_dog_rate', 0.0):.1f}%, "
                 f"LongDog1P={final_all.get('long_dog_onepos_rate', 0.0):.1f}%, "
                 f"Loss={final_all['loss']:.3f}")
        callback(
            "  Save gate diagnostics: "
            f"loss(val) {baseline_val.get('loss', 0.0):.3f}->{final_val.get('loss', 0.0):.3f}, "
            f"loss(all) {baseline_all.get('loss', 0.0):.3f}->{final_all.get('loss', 0.0):.3f}, "
            f"compDog(val) {baseline_val.get('competitive_dog_rate', 0.0):.1f}%"
            f"->{final_val.get('competitive_dog_rate', 0.0):.1f}%, "
            f"onePosDog(val) {baseline_val.get('one_possession_dog_rate', 0.0):.1f}%"
            f"->{final_val.get('one_possession_dog_rate', 0.0):.1f}%, "
            f"longDog1P(val) {baseline_val.get('long_dog_onepos_rate', 0.0):.1f}%"
            f"->{final_val.get('long_dog_onepos_rate', 0.0):.1f}%, "
            f"hybrid {float(save_details.get('baseline_hybrid_loss', baseline_val.get('loss', 0.0))):.3f}"
            f"->{float(save_details.get('candidate_hybrid_loss', final_val.get('loss', 0.0))):.3f}, "
            f"ROI {baseline_val.get('ml_roi', 0.0):+.2f}%->{final_val.get('ml_roi', 0.0):+.2f}% "
            f"(lb95 {final_val.get('ml_roi_lb95', 0.0):+.2f}%), "
            f"min ML payout {final_val.get('ml_min_payout', MIN_ML_PAYOUT):.2f}x, "
            f"ROI gate {'ON' if bool(save_details.get('use_roi_gate', False)) else 'OFF'}, "
            f"long-dog tiebreak {'ON' if bool(save_details.get('use_long_dog_tiebreak_gate', False)) else 'OFF'}, "
            f"hybrid loss gate {'ON' if bool(save_details.get('use_hybrid_loss_gate', False)) else 'OFF'}, "
            f"shrunk upset lift {float(save_details.get('shrunk_upset_lift', 0.0)):+.2f}pp"
        )

    weight_delta = _max_weight_delta(w, final_w)
    weight_change_ok = weight_delta >= min_weight_delta
    save_details["weight_delta"] = weight_delta
    save_details["min_weight_delta"] = min_weight_delta
    save_details["weight_change_ok"] = weight_change_ok
    did_save = False

    if save and save_ok and weight_change_ok:
        save_weight_config(final_w)
        invalidate_weight_cache()
        did_save = True
        if callback:
            gate_note = "" if save_reason == "pass" else f", {save_reason}"
            callback(f"CD saved improved weights "
                     f"({baseline_winner_pct:.1f}% -> {best_winner_pct:.1f}%, "
                     f"dW={weight_delta:.6f}{gate_note})")
    elif save and save_ok:
        save_ok = False
        save_reason = (f"no-op weight update (dW {weight_delta:.6f} "
                       f"< {min_weight_delta:.6f})")
        if callback:
            callback(f"CD: validation save gate rejected: {save_reason} "
                     f"- keeping current weights")
    elif save:
        if callback:
            callback(f"CD: validation save gate rejected: {save_reason} "
                     f"- keeping current weights")

    return {
        "weights": best_w_dict,
        "history": history,
        "changes": all_changes,
        "rounds": len(history),
        "improved": did_save,
        "save_gate_reason": save_reason,
        "save_gate_details": save_details,
        "initial_winner_pct": baseline_winner_pct,
        "final_winner_pct": best_winner_pct,
        "initial_loss": baseline_val["loss"],
        "final_loss": final_val["loss"],
        "initial_all_loss": baseline_all["loss"],
        "final_all_loss": final_all["loss"],
        "favorites_pct": favorites_pct,
        **final_val,
    }
