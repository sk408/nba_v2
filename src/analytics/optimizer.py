"""Optuna CMA-ES optimizer for NBA Fundamentals V2.

Loss function: -(winner_pct + upset_accuracy * upset_rate / 100 * upset_bonus_mult)

Studies are persisted to data/optuna_studies.db so trials accumulate across
runs and survive crashes.  CMA-ES sampler learns parameter correlations for
efficient 49-dimensional exploration.

VectorizedGames converts List[GameInput] into flat NumPy arrays for fast evaluation.
optimize_weights() runs walk-forward Optuna optimization.
compare_modes() A/B tests fundamentals-only vs fundamentals+sharp.
"""

import hashlib
import logging
import os
import random
from typing import List, Dict, Any, Optional, Callable, Tuple

import numpy as np

from src.analytics.prediction import GameInput
from src.analytics.weight_config import (
    WeightConfig, get_weight_config, save_weight_config,
    OPTIMIZER_RANGES, SHARP_MODE_RANGES, invalidate_weight_cache,
    CD_RANGES, CD_SHARP_RANGES,
)

logger = logging.getLogger(__name__)

# Module-level study cache: keeps in-memory studies alive across optimization
# passes so we don't reload 18K+ trials from disk every time.
_study_cache: Dict[str, Any] = {}        # study_name -> optuna.Study
_save_threads: Dict[str, Any] = {}       # study_name -> threading.Thread

# Walk-forward: train on first N% of games, validate on last (1-N)%.
WALK_FORWARD_SPLIT = 0.80

# Compression penalty: prevents optimizer from collapsing all predictions to 0.
COMPRESSION_RATIO_FLOOR = 0.55
COMPRESSION_PENALTY_MULT = 80.0

# Default minimum payout multiplier for ML ROI calculations.
# Can be overridden by config key: optimizer_min_ml_payout.
MIN_ML_PAYOUT = 1.50


def _safe_float_setting(key: str, default: float) -> float:
    """Read a float setting with defensive fallback."""
    from src.config import get as get_setting
    try:
        return float(get_setting(key, default))
    except (TypeError, ValueError):
        return float(default)


def _safe_int_setting(key: str, default: int) -> int:
    """Read an int setting with defensive fallback."""
    from src.config import get as get_setting
    try:
        return max(0, int(get_setting(key, default)))
    except (TypeError, ValueError):
        return max(0, int(default))


def _safe_bool_setting(key: str, default: bool) -> bool:
    """Read a bool setting with tolerant parsing."""
    from src.config import get as get_setting
    raw = get_setting(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "on", "y"):
            return True
        if v in ("0", "false", "no", "off", "n"):
            return False
    return bool(default)


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
    adaptive_winner_drop_tol = min(
        winner_drop_tol + relax_lift * 0.35,
        max(4.0, winner_drop_tol),
    )
    adaptive_favorites_slack = min(
        favorites_slack + relax_lift * 1.25,
        max(12.0, favorites_slack),
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

    save_ok = loss_improved and winner_guard and favorites_guard and compression_ok and edge_ok

    reasons: List[str] = []
    if not loss_improved:
        if use_hybrid_loss_gate and has_all_loss:
            reasons.append(
                f"loss gate failed (val {candidate_loss:.3f} vs {baseline_loss:.3f}; "
                f"hybrid {candidate_hybrid_loss:.3f} vs {baseline_hybrid_loss:.3f})"
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
        "loss_improved": loss_improved,
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
        self.home_srs = np.array([g.home_srs for g in games])
        self.away_srs = np.array([g.away_srs for g in games])
        self.home_pythag_wpct = np.array([g.home_pythag_wpct for g in games], dtype=float)
        self.away_pythag_wpct = np.array([g.away_pythag_wpct for g in games], dtype=float)
        self.home_onoff = np.array([g.home_onoff_impact for g in games])
        self.away_onoff = np.array([g.away_onoff_impact for g in games])
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
        self._srs_diff = self.home_srs - self.away_srs
        self._pythag_diff = self.home_pythag_wpct - self.away_pythag_wpct
        self._onoff_diff = self.home_onoff - self.away_onoff
        self._fg3_luck_diff = self.home_fg3_luck - self.away_fg3_luck
        self._process_total_edge = (self._process_paint_edge + self._process_fb_edge
                                    + self._process_sec_edge + self._process_tov_edge)

        from src.config import get as get_setting
        upset_bonus_max = max(0.0, _safe_float_setting("upset_bonus_mult_max", 5.0))
        self._upset_bonus_mult = float(
            np.clip(_safe_float_setting("upset_bonus_mult", 0.5), 0.0, upset_bonus_max)
        )
        self._min_ml_payout = max(
            1.0,
            _safe_float_setting("optimizer_min_ml_payout", MIN_ML_PAYOUT),
        )

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
              + self.ff_fta_edge * w.ff_fta_weight) * w.four_factors_scale
        game_score += ff

        # 8. Opponent Four Factors (defensive matchup)
        opp_ff = (self.opp_ff_efg_edge * w.opp_ff_efg_weight
                  + self.opp_ff_tov_edge * w.opp_ff_tov_weight
                  + self.opp_ff_oreb_edge * w.opp_ff_oreb_weight
                  + self.opp_ff_fta_edge * w.opp_ff_fta_weight) * w.four_factors_scale
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
        game_score += self._srs_diff * w.srs_diff_mult
        game_score += self._pythag_diff * w.pythag_diff_mult
        game_score += self._onoff_diff * w.onoff_impact_mult
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
        pred_home_win = game_score > 0
        pred_away_win = game_score < 0
        actual_home_win = self.actual_spread > 0.5
        actual_away_win = self.actual_spread < -0.5
        actual_push = np.abs(self.actual_spread) <= 0.5

        correct = ((pred_home_win & actual_home_win)
                    | (pred_away_win & actual_away_win)
                    | (actual_push & (np.abs(game_score) <= 3.0)))
        winner_pct = float(np.mean(correct)) * 100.0

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
        # Model picks home when game_score > 0, away when game_score < 0
        model_picks_home = game_score > 0
        # Upset = model disagrees with Vegas on who wins
        model_picks_upset = model_picks_home != vegas_fav_home
        upset_correct = (model_picks_upset
                         & ((model_picks_home & actual_home_win)
                            | (~model_picks_home & actual_away_win)))
        upset_count = int(np.sum(model_picks_upset))
        upset_rate = float(upset_count) / max(1, self.n) * 100.0
        upset_accuracy = float(np.sum(upset_correct)) / max(1, upset_count) * 100.0
        upset_correct_count = int(np.sum(upset_correct))

        # ML ROI (diagnostic only — not in loss function, skip in fast mode)
        ml_roi = -4.54
        ml_win_rate = winner_pct
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

        loss = -(winner_pct + upset_accuracy * upset_rate / 100.0 * self._upset_bonus_mult)

        # Compression penalty — prevent degenerate narrow-band predictions
        if compression_ratio < COMPRESSION_RATIO_FLOOR:
            loss += (COMPRESSION_RATIO_FLOOR - compression_ratio) * COMPRESSION_PENALTY_MULT

        return {
            "winner_pct": winner_pct,
            "favorites_pct": favorites_pct,
            "upset_rate": upset_rate,
            "upset_accuracy": upset_accuracy,
            "upset_correct_count": upset_correct_count,
            "upset_count": upset_count,
            "ml_roi": ml_roi,
            "ml_win_rate": ml_win_rate,
            "ml_bet_count": ml_bet_count,
            "ml_roi_lb95": ml_roi_lb95,
            "ml_min_payout": self._min_ml_payout,
            "spread_mae": spread_mae,
            "compression_ratio": compression_ratio,
            "loss": loss,
        }


def optimize_weights(
    games: List[GameInput],
    n_trials: int = 3000,
    include_sharp: bool = False,
    callback: Optional[Callable] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run Optuna TPE optimization with walk-forward validation.

    Games are split chronologically: first WALK_FORWARD_SPLIT for training,
    remainder for validation. Optuna optimises on the training set; weights
    are only saved when they also improve on the held-out validation set.

    Save gate: anti-gaming validation gate blends loss, ROI, and upset lift.
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

    vg_train = VectorizedGames(train_games)
    vg_val = VectorizedGames(val_games)
    vg_all = VectorizedGames(sorted_games)

    if callback:
        callback(f"Walk-forward: {len(train_games)} train "
                 f"({train_games[0].game_date} to {train_games[-1].game_date}), "
                 f"{len(val_games)} validation "
                 f"({val_games[0].game_date} to {val_games[-1].game_date})")

    # Baseline evaluation on both sets
    baseline_w = get_weight_config()
    baseline_train = vg_train.evaluate(baseline_w, include_sharp=include_sharp)
    baseline_val = vg_val.evaluate(baseline_w, include_sharp=include_sharp)
    baseline_all = vg_all.evaluate(baseline_w, include_sharp=include_sharp)

    if callback:
        callback(f"Baseline (train): Winner={baseline_train['winner_pct']:.1f}%, "
                 f"Upset={baseline_train['upset_accuracy']:.1f}% @ {baseline_train['upset_rate']:.1f}% rate, "
                 f"Loss={baseline_train['loss']:.3f}")
        callback(f"Baseline (valid): Winner={baseline_val['winner_pct']:.1f}%, "
                 f"Upset={baseline_val['upset_accuracy']:.1f}% @ {baseline_val['upset_rate']:.1f}% rate, "
                 f"Favorites={baseline_val['favorites_pct']:.1f}%, "
                 f"Loss={baseline_val['loss']:.3f}")
        callback(f"Baseline (all):   Winner={baseline_all['winner_pct']:.1f}%, "
                 f"Upset={baseline_all['upset_accuracy']:.1f}% @ {baseline_all['upset_rate']:.1f}% rate, "
                 f"Loss={baseline_all['loss']:.3f}")

    best_w = baseline_w
    best_train_loss = baseline_train["loss"]
    best_train_result = baseline_train
    last_saved_w = [baseline_w]
    min_weight_delta = _safe_float_setting("optimizer_save_min_weight_delta", 1e-4)

    # Select parameter ranges
    ranges = SHARP_MODE_RANGES if include_sharp else OPTIMIZER_RANGES

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {}
            for key, (lo, hi) in ranges.items():
                params[key] = trial.suggest_float(key, lo, hi)

            w = WeightConfig.from_dict({**baseline_w.to_dict(), **params})
            result = vg_train.evaluate(w, include_sharp=include_sharp, fast=True)
            trial.set_user_attr("result", result)
            return result["loss"]

        # ── Persistent study with CMA-ES (in-memory for speed) ──
        # Version hash: changes when parameter space or training window changes.
        # A new hash creates a new study (old trials become irrelevant).
        range_keys = sorted(ranges.keys())
        version_blob = ",".join(range_keys) + "|" + train_games[-1].game_date
        version_hash = hashlib.md5(version_blob.encode()).hexdigest()[:8]
        study_name = f"{'sharp' if include_sharp else 'fundamentals'}_{version_hash}"

        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "optuna_studies.db")
        storage_url = f"sqlite:///{db_path}"

        # Reuse cached in-memory study across passes (avoids reloading 18K+ trials)
        if study_name in _study_cache:
            study = _study_cache[study_name]
            prior_trials = len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.COMPLETE])
            if callback:
                callback(f"Study '{study_name}': {prior_trials} trials in memory, "
                         f"adding {n_trials} more")
        else:
            # First call: load from disk, then cache in memory.
            # IPOP restart strategy: when CMA-ES converges (step size shrinks),
            # it restarts with doubled population size to escape local minima.
            sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop")

            # Load best N trials from disk to warm-start CMA-ES.
            # Too many trials locks CMA-ES into a converged state;
            # too few loses the benefit of prior exploration.
            MAX_SEED_TRIALS = 3000
            prior_trials = 0
            seed_trials = []
            try:
                disk_study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                )
                disk_completed = [t for t in disk_study.trials
                                  if t.state == optuna.trial.TrialState.COMPLETE]
                prior_trials = len(disk_completed)
                if prior_trials > MAX_SEED_TRIALS:
                    # Keep the best trials for warm-starting
                    disk_completed.sort(key=lambda t: t.value)
                    seed_trials = disk_completed[:MAX_SEED_TRIALS]
                else:
                    seed_trials = disk_completed
            except KeyError:
                pass

            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=sampler,
            )
            if seed_trials:
                study.add_trials(seed_trials)

            _study_cache[study_name] = study
            if callback:
                loaded = len(seed_trials)
                callback(f"Study '{study_name}': {prior_trials} prior trials on disk, "
                         f"loaded best {loaded} to seed CMA-ES (IPOP), "
                         f"adding {n_trials} more")

        # Log interval from config
        from src.config import get as get_setting
        log_interval = int(get_setting("optimizer_log_interval", 300))
        _best_logged_loss = best_train_loss
        _stagnation_counter = [0]
        _stagnation_threshold = int(get_setting("optuna_stagnation_threshold", 500))
        _early_stop_trials = int(get_setting("optuna_early_stop_trials", 2000))
        _min_trials_before_stop = int(get_setting("optuna_min_trials_before_stop", 500))

        # Track best weights for checkpoint saving
        _checkpoint_best_loss = [best_train_loss]

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
                if trial.value < _checkpoint_best_loss[0] - 0.01:
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
                        pass  # checkpoint is best-effort
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

            if trial.number % log_interval == 0 or is_new_best:
                if callback:
                    res = trial.user_attrs.get("result", {})
                    win = res.get("winner_pct", 0)
                    upset_acc = res.get("upset_accuracy", 0)
                    upset_r = res.get("upset_rate", 0)
                    callback(f"Trial {trial.number}/{n_trials}: loss={trial.value:.3f} "
                             f"(Winner={win:.1f}%, "
                             f"Upset={upset_acc:.1f}% @ {upset_r:.1f}% rate)")
                    if is_new_best:
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
        from src.config import get as _get_setting
        top_n = int(_get_setting("optuna_top_n_validation", 10))
        completed.sort(key=lambda t: t.value)
        candidates = completed[:top_n]

        if candidates and candidates[0].value < best_train_loss:
            best_val_loss = float("inf")
            chosen_rank = 0
            if callback:
                callback(f"Validating top {len(candidates)} training trials...")

            for rank, trial in enumerate(candidates):
                cand_w = WeightConfig.from_dict(
                    {**baseline_w.to_dict(), **trial.params})
                cand_val = vg_val.evaluate(cand_w, include_sharp=include_sharp)
                cand_val_loss = cand_val["loss"]

                if callback and rank < 5:
                    tr = trial.user_attrs.get("result", {})
                    callback(
                        f"  #{rank + 1} train loss={trial.value:.3f} "
                        f"(Winner={tr.get('winner_pct', 0):.1f}%) "
                        f"-> val loss={cand_val_loss:.3f} "
                        f"(Winner={cand_val.get('winner_pct', 0):.1f}%)")

                if cand_val_loss < best_val_loss:
                    best_val_loss = cand_val_loss
                    best_w = cand_w
                    best_train_loss = trial.value
                    best_train_result = trial.user_attrs.get(
                        "result",
                        vg_train.evaluate(cand_w, include_sharp=include_sharp))
                    chosen_rank = rank

            if callback and chosen_rank > 0:
                callback(
                    f"Selected trial #{chosen_rank + 1} "
                    f"(not #1) -- better validation performance")

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
            if result["loss"] < best_train_loss:
                best_w = w
                best_train_loss = result["loss"]
                best_train_result = result
            if callback and (i + 1) % 300 == 0:
                callback(f"Random trial {i + 1}/{n_trials}: "
                         f"best_loss={best_train_loss:.3f}")

    # Walk-forward validation
    best_val = vg_val.evaluate(best_w, include_sharp=include_sharp)
    best_all = vg_all.evaluate(best_w, include_sharp=include_sharp)

    if callback:
        callback("-- Walk-forward results --")
        callback(f"  Train:  Winner={best_train_result['winner_pct']:.1f}%, "
                 f"Upset={best_train_result['upset_accuracy']:.1f}% "
                 f"@ {best_train_result['upset_rate']:.1f}% rate, "
                 f"Loss={best_train_loss:.3f}")
        callback(f"  Valid:  Winner={best_val['winner_pct']:.1f}%, "
                 f"Upset={best_val['upset_accuracy']:.1f}% "
                 f"@ {best_val['upset_rate']:.1f}% rate, "
                 f"Favorites={best_val['favorites_pct']:.1f}%, "
                 f"Loss={best_val['loss']:.3f}")
        callback(f"  All:    Winner={best_all['winner_pct']:.1f}%, "
                 f"Upset={best_all['upset_accuracy']:.1f}% "
                 f"@ {best_all['upset_rate']:.1f}% rate, "
                 f"Loss={best_all['loss']:.3f}")

    # Save gate: robust anti-gaming guardrails on validation
    baseline_winner_pct = baseline_val.get("winner_pct", 0)
    favorites_pct = best_val.get("favorites_pct", 0)
    best_winner_pct = best_val.get("winner_pct", 0)
    save_ok, save_reason, save_details = _passes_robust_save_gate(
        baseline=baseline_val,
        candidate=best_val,
        n_validation_games=len(val_games),
        baseline_all=baseline_all,
        candidate_all=best_all,
    )

    if callback:
        callback(
            "Save gate diagnostics: "
            f"loss(val) {baseline_val.get('loss', 0.0):.3f}->{best_val.get('loss', 0.0):.3f}, "
            f"loss(all) {baseline_all.get('loss', 0.0):.3f}->{best_all.get('loss', 0.0):.3f}, "
            f"hybrid {float(save_details.get('baseline_hybrid_loss', baseline_val.get('loss', 0.0))):.3f}"
            f"->{float(save_details.get('candidate_hybrid_loss', best_val.get('loss', 0.0))):.3f}, "
            f"ROI {baseline_val.get('ml_roi', 0.0):+.2f}%->{best_val.get('ml_roi', 0.0):+.2f}% "
            f"(lb95 {best_val.get('ml_roi_lb95', 0.0):+.2f}%), "
            f"min ML payout {best_val.get('ml_min_payout', MIN_ML_PAYOUT):.2f}x, "
            f"ROI gate {'ON' if bool(save_details.get('use_roi_gate', False)) else 'OFF'}, "
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
            callback(f"Saved optimized weights "
                     f"(val Winner: {baseline_winner_pct:.1f}% -> {best_winner_pct:.1f}%, "
                     f"vs Favorites baseline: {favorites_pct:.1f}%, "
                     f"dW={weight_delta:.6f})")
    elif save_ok:
        save_ok = False
        save_reason = (f"no-op weight update (dW {weight_delta:.6f} "
                       f"< {min_weight_delta:.6f})")
        if callback:
            callback(f"Validation save gate rejected: {save_reason} "
                     f"- keeping current weights")
    else:
        if callback:
            callback(f"Validation save gate rejected: {save_reason} "
                     f"- keeping current weights")

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
        "train_loss": best_train_loss,
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
                 f"ML ROI={fund_result['ml_roi']:+.1f}%")
        callback(f"Fundamentals+Sharp: Winner={sharp_result['winner_pct']:.1f}%, "
                 f"Upset={sharp_result['upset_accuracy']:.1f}% "
                 f"@ {sharp_result['upset_rate']:.1f}% rate, "
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
          ) * w.four_factors_scale
    gs += ff
    opp_ff = (vg.opp_ff_efg_edge * w.opp_ff_efg_weight
              + vg.opp_ff_tov_edge * w.opp_ff_tov_weight
              + vg.opp_ff_oreb_edge * w.opp_ff_oreb_weight
              + vg.opp_ff_fta_edge * w.opp_ff_fta_weight
              ) * w.four_factors_scale
    gs += opp_ff
    clutch_mask = np.abs(gs) < w.clutch_threshold
    clutch_adj = np.clip(vg.clutch_diff * w.clutch_scale, -w.clutch_cap, w.clutch_cap)
    gs += clutch_adj * clutch_mask
    home_eff = vg.home_defl + vg.home_contested * w.hustle_contested_wt
    away_eff = vg.away_defl + vg.away_contested * w.hustle_contested_wt
    gs += (home_eff - away_eff) * w.hustle_effort_mult
    gs += vg.net_rest * w.rest_advantage_mult
    gs -= vg.away_b2b_at_altitude * w.altitude_b2b_penalty

    # Fundamentals-only picks
    fund_picks_home = gs > 0

    # With sharp
    gs_sharp = gs + vg.sharp_ml_edge * w.sharp_ml_weight
    sharp_picks_home = gs_sharp > 0

    # Flipped picks
    flipped = fund_picks_home != sharp_picks_home
    n_flipped = int(np.sum(flipped))

    # Accuracy of flipped picks
    actual_home_win = vg.actual_spread > 0.5
    actual_away_win = vg.actual_spread < -0.5
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
                 f"Loss={baseline_train['loss']:.3f}")
        callback(f"CD baseline (valid): Winner={baseline_val['winner_pct']:.1f}%, "
                 f"Favorites={baseline_val['favorites_pct']:.1f}%")
        callback(f"CD baseline (all):   Winner={baseline_all['winner_pct']:.1f}%, "
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
                 f"Loss={final_train['loss']:.3f}")
        callback(f"  Valid:  Winner={best_winner_pct:.1f}% "
                 f"(was {baseline_winner_pct:.1f}%), "
                 f"Favorites={favorites_pct:.1f}%")
        callback(f"  All:    Winner={final_all['winner_pct']:.1f}% "
                 f"Loss={final_all['loss']:.3f}")
        callback(
            "  Save gate diagnostics: "
            f"loss(val) {baseline_val.get('loss', 0.0):.3f}->{final_val.get('loss', 0.0):.3f}, "
            f"loss(all) {baseline_all.get('loss', 0.0):.3f}->{final_all.get('loss', 0.0):.3f}, "
            f"hybrid {float(save_details.get('baseline_hybrid_loss', baseline_val.get('loss', 0.0))):.3f}"
            f"->{float(save_details.get('candidate_hybrid_loss', final_val.get('loss', 0.0))):.3f}, "
            f"ROI {baseline_val.get('ml_roi', 0.0):+.2f}%->{final_val.get('ml_roi', 0.0):+.2f}% "
            f"(lb95 {final_val.get('ml_roi_lb95', 0.0):+.2f}%), "
            f"min ML payout {final_val.get('ml_min_payout', MIN_ML_PAYOUT):.2f}x, "
            f"ROI gate {'ON' if bool(save_details.get('use_roi_gate', False)) else 'OFF'}, "
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
            callback(f"CD saved improved weights "
                     f"({baseline_winner_pct:.1f}% -> {best_winner_pct:.1f}%, "
                     f"dW={weight_delta:.6f})")
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
