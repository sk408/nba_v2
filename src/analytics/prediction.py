"""Single-path prediction engine for NBA Fundamentals V2.

One predict() function used by live predictions, backtesting, and optimization.
No three-path sync problem. No Elo, no ESPN blend, no ML ensemble, no opening
spread, no autotune, no per-team weights.

Optional score realism calibration is post-prediction only and does not change
game_score/pick (winner logic remains raw model output).

game_score (not "spread") -- the model predicts a strength edge, not a point spread.
Sign = pick (positive = home), magnitude = confidence.
Vegas lines are consulted AFTER prediction for upset identification only.
"""

import bisect
import hashlib
import logging
import os
import pickle
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields as dc_fields
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.database import db
from src.config import get_season, get_config, get_historical_seasons
from src.analytics.weight_config import (
    FOUR_FACTORS_FIXED_SCALE,
    WeightConfig,
    get_weight_config,
)
from src.analytics.thresholds import MODEL_PICK_EDGE_THRESHOLD, ACTUAL_WIN_THRESHOLD
from src.analytics.stats_engine import (
    aggregate_projection, get_home_court_advantage, compute_fatigue,
    get_team_abbreviations, get_team_names,
    _LEAGUE_AVG_PPG, _PACE_FALLBACK, _RATING_FALLBACK,
)
from src.analytics.cache import team_cache

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class GameInput:
    """Raw inputs for a single game prediction. Stored in precompute cache."""
    game_date: str = ""
    season: str = ""
    home_team_id: int = 0
    away_team_id: int = 0
    actual_home_score: float = 0.0
    actual_away_score: float = 0.0
    # Player projections
    home_proj: Dict[str, float] = field(default_factory=dict)
    away_proj: Dict[str, float] = field(default_factory=dict)
    # Defensive factors
    home_def_factor_raw: float = 1.0
    away_def_factor_raw: float = 1.0
    # Home court
    home_court: float = 3.0
    # Fatigue
    home_rest_days: int = 3
    away_rest_days: int = 3
    home_b2b: bool = False
    away_b2b: bool = False
    home_3in4: bool = False
    away_3in4: bool = False
    home_4in6: bool = False
    away_4in6: bool = False
    home_same_day: bool = False
    away_same_day: bool = False
    # Ratings
    home_off: float = 110.0
    away_off: float = 110.0
    home_def: float = 110.0
    away_def: float = 110.0
    home_pace: float = 98.0
    away_pace: float = 98.0
    # Four Factors
    home_ff: Dict[str, float] = field(default_factory=dict)
    away_ff: Dict[str, float] = field(default_factory=dict)
    # Clutch
    home_clutch: Dict[str, float] = field(default_factory=dict)
    away_clutch: Dict[str, float] = field(default_factory=dict)
    # Hustle (per-game normalized)
    home_hustle: Dict[str, float] = field(default_factory=dict)
    away_hustle: Dict[str, float] = field(default_factory=dict)
    # Sharp money (ML only) -- populated but only used in sharp mode
    ml_home_public: int = 0
    ml_home_money: int = 0
    # Vegas reference (NOT used in prediction, only for upset identification)
    vegas_spread: float = 0.0
    vegas_home_ml: int = 0
    vegas_away_ml: int = 0
    # ── V2.1 features ──
    # Elo ratings
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    # Travel & Geography
    home_travel_miles: float = 0.0
    away_travel_miles: float = 0.0
    home_tz_crossings: int = 0
    away_tz_crossings: int = 0
    home_cum_travel_7d: float = 0.0
    away_cum_travel_7d: float = 0.0
    # Momentum
    home_streak: int = 0
    away_streak: int = 0
    home_mov_trend: float = 0.0
    away_mov_trend: float = 0.0
    # Injury VORP
    home_injury_vorp_lost: float = 0.0
    away_injury_vorp_lost: float = 0.0
    # Referee
    ref_crew_fouls_pg: float = 38.0
    ref_crew_home_bias: float = 50.0
    # Spread sharp money
    spread_sharp_edge: float = 0.0
    # Schedule spots
    home_lookahead: bool = False
    away_lookahead: bool = False
    home_letdown: bool = False
    away_letdown: bool = False
    home_road_trip_game: int = 0
    away_road_trip_game: int = 0
    # Season phase + regime features
    home_season_progress: float = 0.0
    away_season_progress: float = 0.0
    home_tank_signal_live: float = 0.0
    away_tank_signal_live: float = 0.0
    home_tank_signal_oracle: float = 0.0
    away_tank_signal_oracle: float = 0.0
    home_roster_shock: float = 0.0
    away_roster_shock: float = 0.0
    # SRS / Pythagorean
    home_srs: float = 0.0
    away_srs: float = 0.0
    home_pythag_wpct: float = 0.5
    away_pythag_wpct: float = 0.5
    # Player On/Off impact
    home_onoff_impact: float = 0.0
    away_onoff_impact: float = 0.0
    home_onoff_reliability: float = 0.0
    away_onoff_reliability: float = 0.0
    # Pace differential
    pace_diff: float = 0.0
    # 3PT luck (regression to mean)
    home_fg3_luck: float = 0.0
    away_fg3_luck: float = 0.0
    # Process stats (per-game)
    home_process: Dict[str, float] = field(default_factory=dict)
    away_process: Dict[str, float] = field(default_factory=dict)


@dataclass
class Prediction:
    """Output of predict()."""
    home_team: str = ""
    away_team: str = ""
    home_team_id: int = 0
    away_team_id: int = 0
    game_date: str = ""
    game_score: float = 0.0          # internal strength edge (positive = home)
    pick: str = ""                    # "HOME" or "AWAY"
    confidence: float = 0.0           # |game_score| normalized to 0-100%
    projected_home_pts: float = 0.0
    projected_away_pts: float = 0.0
    adjustments: Dict[str, float] = field(default_factory=dict)
    # Sharp money overlay (populated regardless of mode, for display)
    ml_sharp_home_public: int = 0
    ml_sharp_home_money: int = 0
    sharp_agrees: Optional[bool] = None  # True if sharp money agrees with pick
    # Vegas reference (carried through for upset identification)
    vegas_spread: float = 0.0
    vegas_home_ml: int = 0
    vegas_away_ml: int = 0
    is_dog_pick: bool = False
    is_value_zone: bool = False
    dog_payout: float = 0.0
    # Odds freshness (ISO timestamp from game_odds.fetched_at)
    odds_fetched_at: Optional[str] = None
    # Post-prediction score calibration (display-only by default)
    calibrated_spread: Optional[float] = None
    calibrated_total: Optional[float] = None
    calibrated_home_pts: Optional[float] = None
    calibrated_away_pts: Optional[float] = None
    score_calibrated: bool = False
    score_calibration_mode: str = ""
    # Interaction model correction detail (populated when model is active)
    interaction_detail: Optional[Dict[str, Any]] = None


def _moneyline_payout_multiplier(ml_line: int) -> float:
    """Return decimal payout multiplier from American odds line."""
    if ml_line == 0:
        return 0.0
    if ml_line < 0:
        return 1.0 + 100.0 / abs(ml_line)
    return 1.0 + ml_line / 100.0


def _classify_dog_pick(
    game_score: float,
    vegas_spread: float,
    vegas_home_ml: int,
    vegas_away_ml: int,
) -> tuple[bool, bool, float]:
    """Classify underdog/value signals from spread and moneyline context.

    Moneyline is the primary source of favorite/underdog direction so upset
    highlighting aligns with moneyline-first workflows.
    Spread is only used as a fallback when moneyline is unavailable.
    """
    if abs(game_score) <= MODEL_PICK_EDGE_THRESHOLD:
        return False, False, 0.0

    model_favors_home = game_score > MODEL_PICK_EDGE_THRESHOLD
    has_moneyline = vegas_home_ml != 0 and vegas_away_ml != 0
    has_spread = vegas_spread != 0
    is_value_zone = has_spread and 4.0 <= abs(vegas_spread) <= 12.0

    favorite_home: Optional[bool]
    if has_moneyline and vegas_home_ml != vegas_away_ml:
        # Lower American odds line implies the favorite (e.g. -150 < +130).
        favorite_home = vegas_home_ml < vegas_away_ml
    elif has_spread:
        # Home spread is from home-team perspective (negative => home favorite).
        favorite_home = vegas_spread < 0
    else:
        favorite_home = None

    if favorite_home is None:
        return False, is_value_zone, 0.0

    underdog_home = not favorite_home
    is_dog_pick = model_favors_home == underdog_home
    if not is_dog_pick or not has_moneyline:
        return is_dog_pick, is_value_zone, 0.0

    dog_ml = vegas_home_ml if model_favors_home else vegas_away_ml
    return is_dog_pick, is_value_zone, _moneyline_payout_multiplier(dog_ml)


def _resolve_tanking_feature_mode() -> str:
    """Return validated tanking feature mode: live, oracle, or both."""
    try:
        cfg = get_config()
        mode = str(cfg.get("optimizer_tanking_feature_mode", "both") or "both")
    except Exception:
        mode = "both"
    mode = mode.strip().lower()
    return mode if mode in {"live", "oracle", "both"} else "both"


# ──────────────────────────────────────────────────────────────
# predict() -- THE formula (single path)
# ──────────────────────────────────────────────────────────────

def predict(game: GameInput, w: WeightConfig, include_sharp: bool = False) -> Prediction:
    """Single prediction path. Used by live predictions, backtesting, and optimization."""
    pred = Prediction(
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
        game_date=game.game_date,
        ml_sharp_home_public=game.ml_home_public,
        ml_sharp_home_money=game.ml_home_money,
        vegas_spread=game.vegas_spread,
        vegas_home_ml=game.vegas_home_ml,
        vegas_away_ml=game.vegas_away_ml,
    )

    # Defensive adjustment (dampened)
    away_def_f = 1.0 + (game.away_def_factor_raw - 1.0) * w.def_factor_dampening
    home_def_f = 1.0 + (game.home_def_factor_raw - 1.0) * w.def_factor_dampening

    home_base = game.home_proj.get("points", 0) * away_def_f
    away_base = game.away_proj.get("points", 0) * home_def_f

    # BASE: home_base_pts - away_base_pts + HCA
    game_score = (home_base - away_base) + game.home_court

    # Fatigue (decomposed -- each component tunable)
    home_rest_bonus = (1.5 if game.home_rest_days >= 4 else 1.0 if game.home_rest_days >= 3 else 0.0) * w.fatigue_rest_bonus
    away_rest_bonus = (1.5 if game.away_rest_days >= 4 else 1.0 if game.away_rest_days >= 3 else 0.0) * w.fatigue_rest_bonus
    home_fat = (game.home_b2b * w.fatigue_b2b + game.home_3in4 * w.fatigue_3in4
                + game.home_4in6 * w.fatigue_4in6
                + game.home_same_day * w.fatigue_same_day - home_rest_bonus)
    away_fat = (game.away_b2b * w.fatigue_b2b + game.away_3in4 * w.fatigue_3in4
                + game.away_4in6 * w.fatigue_4in6
                + game.away_same_day * w.fatigue_same_day - away_rest_bonus)
    fatigue_adj = home_fat - away_fat
    game_score -= fatigue_adj
    pred.adjustments["fatigue"] = -fatigue_adj

    # Turnover differential
    home_to = game.home_proj.get("turnovers", 0)
    away_to = game.away_proj.get("turnovers", 0)
    to_adj = (away_to - home_to) * w.turnover_margin_mult
    game_score += to_adj
    pred.adjustments["turnover"] = to_adj

    # Rebound differential
    home_reb = game.home_proj.get("rebounds", 0)
    away_reb = game.away_proj.get("rebounds", 0)
    reb_adj = (home_reb - away_reb) * w.rebound_diff_mult
    game_score += reb_adj
    pred.adjustments["rebound"] = reb_adj

    # Rating matchup -- cross-team matchup
    home_me = game.home_off - game.away_def
    away_me = game.away_off - game.home_def
    rating_adj = (home_me - away_me) * w.rating_matchup_mult
    game_score += rating_adj
    pred.adjustments["rating_matchup"] = rating_adj

    # Four Factors -- offensive
    hff, aff = game.home_ff, game.away_ff
    efg_e = hff.get("efg", 0) - aff.get("efg", 0)
    tov_e = aff.get("tov", 0) - hff.get("tov", 0)
    oreb_e = hff.get("oreb", 0) - aff.get("oreb", 0)
    fta_e = hff.get("fta", 0) - aff.get("fta", 0)
    ff_adj = (efg_e * w.ff_efg_weight + tov_e * w.ff_tov_weight +
              oreb_e * w.ff_oreb_weight + fta_e * w.ff_fta_weight) * FOUR_FACTORS_FIXED_SCALE
    game_score += ff_adj
    pred.adjustments["four_factors"] = ff_adj

    # Opponent Four Factors (defensive matchup)
    opp_efg_e = aff.get("opp_efg", 0) - hff.get("opp_efg", 0)
    opp_tov_e = hff.get("opp_tov", 0) - aff.get("opp_tov", 0)
    opp_oreb_e = aff.get("opp_oreb", 0) - hff.get("opp_oreb", 0)
    opp_fta_e = aff.get("opp_fta", 0) - hff.get("opp_fta", 0)
    opp_ff_adj = (opp_efg_e * w.opp_ff_efg_weight + opp_tov_e * w.opp_ff_tov_weight +
                  opp_oreb_e * w.opp_ff_oreb_weight + opp_fta_e * w.opp_ff_fta_weight) * FOUR_FACTORS_FIXED_SCALE
    game_score += opp_ff_adj
    pred.adjustments["opp_four_factors"] = opp_ff_adj

    # Clutch (only if close game)
    if abs(game_score) < w.clutch_threshold:
        h_clutch = game.home_clutch.get("net_rating", 0)
        a_clutch = game.away_clutch.get("net_rating", 0)
        clutch_diff = (h_clutch - a_clutch) * w.clutch_scale
        clutch_adj = max(-w.clutch_cap, min(w.clutch_cap, clutch_diff))
        game_score += clutch_adj
        pred.adjustments["clutch"] = clutch_adj

    # Hustle
    h_eff = game.home_hustle.get("deflections", 0) + game.home_hustle.get("contested", 0) * w.hustle_contested_wt
    a_eff = game.away_hustle.get("deflections", 0) + game.away_hustle.get("contested", 0) * w.hustle_contested_wt
    hustle_adj = (h_eff - a_eff) * w.hustle_effort_mult
    game_score += hustle_adj
    pred.adjustments["hustle"] = hustle_adj

    # Rest advantage (continuous)
    rest_adj = (game.home_rest_days - game.away_rest_days) * w.rest_advantage_mult
    game_score += rest_adj
    pred.adjustments["rest_advantage"] = rest_adj

    # Altitude B2B -- away team on B2B at DEN(1610612743) or UTA(1610612762)
    if game.away_b2b and game.home_team_id in (1610612743, 1610612762):
        game_score -= w.altitude_b2b_penalty
        pred.adjustments["altitude_b2b"] = -w.altitude_b2b_penalty

    # Sharp money ML (only when include_sharp=True)
    if include_sharp:
        has_ml = bool(game.ml_home_public) and bool(game.ml_home_money)
        if has_ml:
            ml_sharp_edge = (game.ml_home_money - game.ml_home_public) / 100.0
            sharp_adj = ml_sharp_edge * w.sharp_ml_weight
            game_score += sharp_adj
            pred.adjustments["sharp_ml"] = sharp_adj

    # ── V2.1 adjustment layers ──

    # 14. Elo differential
    elo_adj = (game.home_elo - game.away_elo) / 400.0 * w.elo_diff_mult
    game_score += elo_adj
    pred.adjustments["elo"] = elo_adj

    # 15. Travel fatigue
    travel_adj = -((game.away_travel_miles / 1000.0) - (game.home_travel_miles / 1000.0)) * w.travel_dist_mult
    game_score += travel_adj
    pred.adjustments["travel"] = travel_adj

    # 16. Timezone crossing
    tz_adj = -(game.away_tz_crossings - game.home_tz_crossings) * w.timezone_crossing_mult
    game_score += tz_adj
    pred.adjustments["timezone"] = tz_adj

    # 17. Cumulative 7-day travel load
    cum_travel_adj = ((game.away_cum_travel_7d - game.home_cum_travel_7d) / 1000.0) * w.cum_travel_7d_mult
    game_score += cum_travel_adj
    pred.adjustments["cum_travel_7d"] = cum_travel_adj

    # 18. Momentum (win/loss streak)
    momentum_adj = (game.home_streak - game.away_streak) * w.momentum_streak_mult
    game_score += momentum_adj
    pred.adjustments["momentum"] = momentum_adj

    # 19. Margin-of-victory trend
    mov_adj = (game.home_mov_trend - game.away_mov_trend) * w.mov_trend_mult
    game_score += mov_adj
    pred.adjustments["mov_trend"] = mov_adj

    # 20. Injury VORP impact (more VORP lost = weaker team)
    injury_adj = (game.away_injury_vorp_lost - game.home_injury_vorp_lost) * w.injury_vorp_mult
    game_score += injury_adj
    pred.adjustments["injury_vorp"] = injury_adj

    # 21. Referee home bias (50% = neutral, >50% = favors home)
    ref_bias_adj = (game.ref_crew_home_bias - 50.0) / 50.0 * w.ref_home_bias_mult
    game_score += ref_bias_adj
    pred.adjustments["ref_home_bias"] = ref_bias_adj

    # 22. Spread sharp money
    if game.spread_sharp_edge:
        spread_sharp_adj = game.spread_sharp_edge / 100.0 * w.sharp_spread_weight
        game_score += spread_sharp_adj
        pred.adjustments["sharp_spread"] = spread_sharp_adj

    # 23. Schedule spots
    sched_adj = (-(game.home_lookahead * w.lookahead_penalty +
                   game.home_letdown * w.letdown_penalty)
                 + (game.away_lookahead * w.lookahead_penalty +
                    game.away_letdown * w.letdown_penalty))
    game_score += sched_adj
    pred.adjustments["schedule_spots"] = sched_adj

    # 24. Road-trip depth (away longer trip should favor home)
    road_trip_adj = (game.away_road_trip_game - game.home_road_trip_game) * w.road_trip_game_mult
    game_score += road_trip_adj
    pred.adjustments["road_trip_game"] = road_trip_adj

    # 25. Season progress differential (late-season asymmetry and cadence)
    season_progress_adj = (
        game.home_season_progress - game.away_season_progress
    ) * w.season_progress_mult
    game_score += season_progress_adj
    pred.adjustments["season_progress"] = season_progress_adj

    # 26. Roster/trade shock (higher churn is penalized)
    roster_shock_adj = (
        game.away_roster_shock - game.home_roster_shock
    ) * w.roster_shock_mult
    game_score += roster_shock_adj
    pred.adjustments["roster_shock"] = roster_shock_adj

    # 27. Tanking pressure (configurable live/oracle/both)
    tank_mode = _resolve_tanking_feature_mode()
    if tank_mode in {"live", "both"}:
        tank_live_adj = (
            game.away_tank_signal_live - game.home_tank_signal_live
        ) * w.tank_live_mult
        game_score += tank_live_adj
        pred.adjustments["tank_live"] = tank_live_adj
    if tank_mode in {"oracle", "both"}:
        tank_oracle_adj = (
            game.away_tank_signal_oracle - game.home_tank_signal_oracle
        ) * w.tank_oracle_mult
        game_score += tank_oracle_adj
        pred.adjustments["tank_oracle"] = tank_oracle_adj

    # 28. SRS differential
    srs_adj = (game.home_srs - game.away_srs) * w.srs_diff_mult
    game_score += srs_adj
    pred.adjustments["srs"] = srs_adj

    # 29. Pythagorean differential
    pythag_adj = (game.home_pythag_wpct - game.away_pythag_wpct) * w.pythag_diff_mult
    game_score += pythag_adj
    pred.adjustments["pythag"] = pythag_adj

    # 30. Player On/Off impact (reliability-aware z-scored team signal)
    onoff_lambda = max(0.0, float(getattr(w, "onoff_reliability_lambda", 0.0)))
    home_rel = max(0.0, float(game.home_onoff_reliability))
    away_rel = max(0.0, float(game.away_onoff_reliability))
    home_onoff = game.home_onoff_impact * (home_rel / (home_rel + onoff_lambda + 1e-9))
    away_onoff = game.away_onoff_impact * (away_rel / (away_rel + onoff_lambda + 1e-9))
    onoff_adj = (home_onoff - away_onoff) * w.onoff_impact_mult
    game_score += onoff_adj
    pred.adjustments["onoff_impact"] = onoff_adj

    # 31. 3PT luck regression (hot team regresses down, cold team regresses up)
    # Negative because positive luck = overperforming = expect regression down
    fg3_luck_adj = -(game.home_fg3_luck - game.away_fg3_luck) * w.fg3_luck_mult
    game_score += fg3_luck_adj
    pred.adjustments["fg3_luck"] = fg3_luck_adj

    # 32. Process stats matchup (paint scoring vs paint defense, etc.)
    # Each component: (team's scoring - opponent's allowed) normalized by league avg
    hp, ap = game.home_process, game.away_process
    if hp and ap:
        paint_edge = (hp.get("paint", 0) - ap.get("opp_paint", 0)) - (ap.get("paint", 0) - hp.get("opp_paint", 0))
        fb_edge = (hp.get("fb", 0) - ap.get("opp_fb", 0)) - (ap.get("fb", 0) - hp.get("opp_fb", 0))
        sec_edge = (hp.get("sec", 0) - ap.get("opp_sec", 0)) - (ap.get("sec", 0) - hp.get("opp_sec", 0))
        tov_edge = (hp.get("off_tov", 0) - ap.get("opp_off_tov", 0)) - (ap.get("off_tov", 0) - hp.get("opp_off_tov", 0))
        process_adj = (paint_edge + fb_edge + sec_edge + tov_edge) * w.process_edge_mult
        game_score += process_adj
        pred.adjustments["process_stats"] = process_adj

    # Derive projected scores (diagnostic only)
    total = home_base + away_base
    # Defensive disruption total adjustment
    comb_steals = game.home_proj.get("steals", 0) + game.away_proj.get("steals", 0)
    comb_blocks = game.home_proj.get("blocks", 0) + game.away_proj.get("blocks", 0)
    total -= (max(0, comb_steals - w.steals_threshold) * w.steals_penalty +
              max(0, comb_blocks - w.blocks_threshold) * w.blocks_penalty)
    # Hustle deflection total adjustment
    comb_defl = game.home_hustle.get("deflections", 0) + game.away_hustle.get("deflections", 0)
    if comb_defl > w.hustle_defl_baseline:
        total -= (comb_defl - w.hustle_defl_baseline) * w.hustle_defl_penalty
    # Fatigue total
    total -= (home_fat + away_fat) * w.fatigue_total_mult
    # Pace mismatch total adjustment (higher pace diff = more scoring variance)
    total += abs(game.home_pace - game.away_pace) * w.pace_mismatch_mult
    # Referee fouls total adjustment (more fouls = more FTs = higher scoring)
    total += (game.ref_crew_fouls_pg - 38.0) * w.ref_fouls_mult
    # Clamp total
    total = max(w.total_min, min(w.total_max, total))

    pred.projected_home_pts = (total + game_score) / 2.0
    pred.projected_away_pts = (total - game_score) / 2.0

    # Pick and confidence
    pred.game_score = game_score
    pred.pick = "HOME" if game_score > MODEL_PICK_EDGE_THRESHOLD else "AWAY"
    # Confidence: |game_score| normalized -- ~15 pts edge = 100% confidence
    pred.confidence = min(100.0, abs(game_score) / 15.0 * 100.0)

    # Sharp money agreement (for display, regardless of mode)
    if game.ml_home_public and game.ml_home_money:
        sharp_favors_home = game.ml_home_money > game.ml_home_public
        model_favors_home = game_score > MODEL_PICK_EDGE_THRESHOLD
        pred.sharp_agrees = sharp_favors_home == model_favors_home

    # Optional post-prediction score calibration.
    # This does not alter game_score/pick winner logic.
    try:
        from src.analytics.score_calibration import apply_score_calibration

        apply_score_calibration(pred, game, include_sharp=include_sharp)
    except Exception as e:
        logger.debug("Score calibration unavailable: %s", e)
        if pred.calibrated_spread is None:
            pred.calibrated_spread = pred.game_score
            pred.calibrated_total = pred.projected_home_pts + pred.projected_away_pts
            pred.calibrated_home_pts = pred.projected_home_pts
            pred.calibrated_away_pts = pred.projected_away_pts
            pred.score_calibrated = False

    return pred


# ──────────────────────────────────────────────────────────────
# Injury loading (live)
# ──────────────────────────────────────────────────────────────

def _load_current_injuries(*team_ids: int) -> Dict[int, float]:
    """Load current injuries from DB and compute play probabilities.

    Returns dict mapping player_id -> play_probability (0.0 to 1.0).

    When a player has a minutes_cap set, the effective play probability is
    capped at (minutes_cap / avg_minutes) so that projection weighting and
    injury VORP reflect the restricted contribution.
    """
    if not team_ids:
        return {}

    placeholders = ",".join("?" for _ in team_ids)
    rows = db.fetch_all(
        f"SELECT player_id, status, reason, minutes_cap "
        f"FROM injuries WHERE team_id IN ({placeholders})",
        tuple(team_ids),
    )
    injured = {}
    for r in rows:
        pid = r.get("player_id")
        if pid is None:
            continue
        status = r.get("status", "Out")
        status_lower = status.lower() if status else "out"
        if status_lower == "out":
            play_prob = 0.0
        elif status_lower in ("doubtful",):
            play_prob = 0.15
        elif status_lower in ("questionable",):
            play_prob = 0.5
        elif status_lower in ("probable", "available"):
            play_prob = 0.85
        else:
            play_prob = 0.0

        # Minutes restriction: cap effective play_prob at (cap / avg_minutes)
        cap = r.get("minutes_cap")
        if cap is not None and play_prob > 0.0:
            try:
                cap = int(cap)
                avg_row = db.fetch_one(
                    "SELECT AVG(minutes) AS avg_min FROM player_stats "
                    "WHERE player_id = ? AND minutes > 0 "
                    "ORDER BY game_date DESC LIMIT 20",
                    (pid,),
                )
                avg_min = float((avg_row or {}).get("avg_min") or 0.0)
                if avg_min > 0:
                    play_prob = min(play_prob, cap / avg_min)
            except (TypeError, ValueError):
                pass

        injured[pid] = play_prob
    return injured


# ──────────────────────────────────────────────────────────────
# On/Off team signal helpers (reliability-weighted + z-scored)
# ──────────────────────────────────────────────────────────────

_onoff_team_signal_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
_onoff_team_signal_lock = threading.Lock()


def _onoff_player_minutes_smoothing() -> float:
    """Minutes smoothing term for player on/off reliability weighting."""
    try:
        cfg = get_config()
        return max(
            1.0,
            float(cfg.get("optimizer_onoff_player_minutes_smoothing", 800.0) or 800.0),
        )
    except Exception:
        return 800.0


def _onoff_team_reliability_slots() -> float:
    """Equivalent 30-minute slots needed for full team on/off reliability."""
    try:
        cfg = get_config()
        return max(
            1.0,
            float(cfg.get("optimizer_onoff_team_reliability_slots", 12.0) or 12.0),
        )
    except Exception:
        return 12.0


def _compute_onoff_team_signals_for_season(season: str) -> Dict[int, Dict[str, float]]:
    """Compute per-team on/off z-signal and reliability for one season."""
    rows = db.fetch_all(
        "SELECT team_id, net_rating_diff, on_court_minutes "
        "FROM player_impact "
        "WHERE season = ? AND on_court_minutes > 0",
        (season,),
    )
    if not rows:
        return {}

    smooth = _onoff_player_minutes_smoothing()
    reliability_slots = _onoff_team_reliability_slots()
    raw_signal_by_team: Dict[int, float] = defaultdict(float)
    capped_slots_by_team: Dict[int, float] = defaultdict(float)

    for row in rows:
        team_id = int(row.get("team_id") or 0)
        if team_id <= 0:
            continue
        net_diff = float(row.get("net_rating_diff", 0.0) or 0.0)
        minutes = float(row.get("on_court_minutes", 0.0) or 0.0)
        if minutes <= 0.0:
            continue
        minute_share = min(minutes, 30.0) / 30.0
        reliability = minutes / (minutes + smooth)
        raw_signal_by_team[team_id] += net_diff * minute_share * reliability
        capped_slots_by_team[team_id] += minute_share

    if not raw_signal_by_team:
        return {}

    raw_values = np.array(list(raw_signal_by_team.values()), dtype=float)
    season_mean = float(np.mean(raw_values))
    season_std = float(np.std(raw_values))

    out: Dict[int, Dict[str, float]] = {}
    for team_id, raw_signal in raw_signal_by_team.items():
        if season_std > 1e-8:
            signal_z = (raw_signal - season_mean) / season_std
        else:
            signal_z = 0.0
        reliability = min(1.0, capped_slots_by_team.get(team_id, 0.0) / reliability_slots)
        out[team_id] = {
            "signal": float(signal_z),
            "reliability": float(reliability),
        }
    return out


def _get_onoff_team_signals_for_season(season: str) -> Dict[int, Dict[str, float]]:
    """Cached accessor for season-level on/off team signal map."""
    smooth = _onoff_player_minutes_smoothing()
    reliability_slots = _onoff_team_reliability_slots()
    cache_key = f"{season}|{smooth:.3f}|{reliability_slots:.3f}"
    with _onoff_team_signal_lock:
        cached = _onoff_team_signal_cache.get(cache_key)
    if cached is not None:
        return cached

    computed = _compute_onoff_team_signals_for_season(season)
    with _onoff_team_signal_lock:
        _onoff_team_signal_cache[cache_key] = computed
    return computed


def _get_onoff_team_signals_for_seasons(
    seasons: List[str],
) -> Dict[tuple[int, str], Dict[str, float]]:
    """Build (team_id, season) -> signal/reliability map for many seasons."""
    out: Dict[tuple[int, str], Dict[str, float]] = {}
    for season in sorted({str(s) for s in seasons if s}):
        per_season = _get_onoff_team_signals_for_season(season)
        for team_id, payload in per_season.items():
            out[(team_id, season)] = payload
    return out


# ──────────────────────────────────────────────────────────────
# Team metrics helper
# ──────────────────────────────────────────────────────────────

def _get_team_metrics(team_id: int, season: Optional[str] = None) -> Dict[str, float]:
    """Fetch team metrics as a flat dict, using cache + memory store."""
    if season is None:
        season = get_season()
    cache_key = f"metrics_{season}"
    cached = team_cache.get(team_id, cache_key)
    if cached is not None:
        return cached

    try:
        from src.analytics.memory_store import get_store
        store = get_store()
        if store.team_metrics is not None and not store.team_metrics.empty:
            rows = store.team_metrics[
                (store.team_metrics["team_id"] == team_id) &
                (store.team_metrics["season"] == season)
            ]
            if not rows.empty:
                result = {str(k): (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in rows.iloc[0].to_dict().items()}
                team_cache.set(team_id, cache_key, result)
                return result
    except Exception:
        logger.debug("memory_store team metrics fallback", exc_info=True)

    row = db.fetch_one(
        "SELECT * FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season)
    )
    result = dict(row) if row else {}
    team_cache.set(team_id, cache_key, result)
    return result


def _derive_pythag_wpct(metrics: Dict[str, float]) -> float:
    """Return Pythagorean win% from team metrics with safe fallbacks."""
    gp = metrics.get("gp", 0) or 0
    pythag_wins = metrics.get("pythag_wins")

    try:
        gp_f = float(gp)
    except (TypeError, ValueError):
        gp_f = 0.0

    if gp_f > 0 and pythag_wins is not None:
        try:
            wpct = float(pythag_wins) / gp_f
            return max(0.0, min(1.0, wpct))
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    try:
        wpct = float(metrics.get("w_pct", 0.5) or 0.5)
        return max(0.0, min(1.0, wpct))
    except (TypeError, ValueError):
        return 0.5


# ──────────────────────────────────────────────────────────────
# predict_matchup() -- convenience wrapper
# ──────────────────────────────────────────────────────────────

def _game_date_to_season(game_date: str) -> str:
    """Map a game date (YYYY-MM-DD) to NBA season string (e.g. '2024-25')."""
    try:
        year, month = int(game_date[:4]), int(game_date[5:7])
        if month >= 7:
            return f"{year}-{str(year + 1)[2:]}"
        else:
            return f"{year - 1}-{str(year)[2:]}"
    except (ValueError, IndexError):
        return get_season()


def predict_matchup(home_team_id: int, away_team_id: int, game_date: str,
                    as_of_date: Optional[str] = None,
                    injured_players: Optional[Dict[int, float]] = None,
                    include_sharp: bool = False) -> Prediction:
    """Full prediction: gather data from DB -> GameInput -> predict().

    This is the convenience wrapper for live / single-game predictions.
    """
    # Team abbreviations (cached singleton)
    abbr_map = get_team_abbreviations()
    name_map = get_team_names()
    home_abbr = abbr_map.get(home_team_id, str(home_team_id))
    away_abbr = abbr_map.get(away_team_id, str(away_team_id))
    home_name = name_map.get(home_team_id, "")
    away_name = name_map.get(away_team_id, "")

    # Injuries
    if injured_players is None:
        injured_players = _load_current_injuries(home_team_id, away_team_id)

    # Global weight config (no per-team blending in V2)
    w = get_weight_config()

    # Player projections
    home_proj = aggregate_projection(home_team_id, away_team_id, is_home=1,
                                     as_of_date=as_of_date,
                                     injured_players=injured_players)
    away_proj = aggregate_projection(away_team_id, home_team_id, is_home=0,
                                     as_of_date=as_of_date,
                                     injured_players=injured_players)

    # Season for this game (needed before HCA + team metrics)
    game_season = _game_date_to_season(game_date)

    # Home court advantage
    home_court = get_home_court_advantage(home_team_id, season=game_season)

    # Team metrics
    hm = _get_team_metrics(home_team_id, season=game_season)
    am = _get_team_metrics(away_team_id, season=game_season)

    # Defensive factors
    league_avg = _LEAGUE_AVG_PPG
    away_opp_pts = am.get("opp_pts", league_avg) or league_avg
    home_opp_pts = hm.get("opp_pts", league_avg) or league_avg
    away_def_raw = away_opp_pts / league_avg if league_avg > 0 else 1.0
    home_def_raw = home_opp_pts / league_avg if league_avg > 0 else 1.0

    # Fatigue
    hfat = compute_fatigue(home_team_id, game_date, w=w)
    afat = compute_fatigue(away_team_id, game_date, w=w)

    # Ratings
    home_off = hm.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
    away_off = am.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
    home_def = hm.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK
    away_def = am.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK

    # Pace
    home_pace = hm.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK
    away_pace = am.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK

    # Four Factors (offensive + defensive opponent)
    h_efg = hm.get("ff_efg_pct", 0) or 0
    a_efg = am.get("ff_efg_pct", 0) or 0
    h_tov = hm.get("ff_tm_tov_pct", 0) or 0
    a_tov = am.get("ff_tm_tov_pct", 0) or 0
    h_oreb = hm.get("ff_oreb_pct", 0) or 0
    a_oreb = am.get("ff_oreb_pct", 0) or 0
    h_fta = hm.get("ff_fta_rate", 0) or 0
    a_fta = am.get("ff_fta_rate", 0) or 0

    h_opp_efg = hm.get("opp_efg_pct", 0) or 0
    a_opp_efg = am.get("opp_efg_pct", 0) or 0
    h_opp_tov = hm.get("opp_tm_tov_pct", 0) or 0
    a_opp_tov = am.get("opp_tm_tov_pct", 0) or 0
    h_opp_oreb = hm.get("opp_oreb_pct", 0) or 0
    a_opp_oreb = am.get("opp_oreb_pct", 0) or 0
    h_opp_fta = hm.get("opp_fta_rate", 0) or 0
    a_opp_fta = am.get("opp_fta_rate", 0) or 0

    # Clutch
    h_clutch = {"net_rating": hm.get("clutch_net_rating", 0) or 0,
                "efg_pct": hm.get("clutch_efg_pct", 0) or 0}
    a_clutch = {"net_rating": am.get("clutch_net_rating", 0) or 0,
                "efg_pct": am.get("clutch_efg_pct", 0) or 0}

    # Hustle (normalize season totals to per-game)
    h_gp = max(1, hm.get("gp", 1) or 1)
    a_gp = max(1, am.get("gp", 1) or 1)
    h_hustle = {"deflections": (hm.get("deflections", 0) or 0) / h_gp,
                "contested": (hm.get("contested_shots", 0) or 0) / h_gp,
                "loose_balls": (hm.get("loose_balls_recovered", 0) or 0) / h_gp}
    a_hustle = {"deflections": (am.get("deflections", 0) or 0) / a_gp,
                "contested": (am.get("contested_shots", 0) or 0) / a_gp,
                "loose_balls": (am.get("loose_balls_recovered", 0) or 0) / a_gp}

    # Process stats (per-game from Misc API)
    h_process = {
        "paint": hm.get("points_in_paint", 0) or 0,
        "fb": hm.get("fast_break_pts", 0) or 0,
        "sec": hm.get("second_chance_pts", 0) or 0,
        "off_tov": hm.get("pts_off_tov", 0) or 0,
        "opp_paint": hm.get("opp_pts_paint", 0) or 0,
        "opp_fb": hm.get("opp_pts_fb", 0) or 0,
        "opp_sec": hm.get("opp_pts_2nd_chance", 0) or 0,
        "opp_off_tov": hm.get("opp_pts_off_tov", 0) or 0,
    }
    a_process = {
        "paint": am.get("points_in_paint", 0) or 0,
        "fb": am.get("fast_break_pts", 0) or 0,
        "sec": am.get("second_chance_pts", 0) or 0,
        "off_tov": am.get("pts_off_tov", 0) or 0,
        "opp_paint": am.get("opp_pts_paint", 0) or 0,
        "opp_fb": am.get("opp_pts_fb", 0) or 0,
        "opp_sec": am.get("opp_pts_2nd_chance", 0) or 0,
        "opp_off_tov": am.get("opp_pts_off_tov", 0) or 0,
    }

    # Sharp money (ML only)
    ml_pub = 0
    ml_mon = 0
    from src.utils.timezone_utils import nba_today
    today_str = nba_today()
    if game_date >= today_str:
        # Live fetch from Action Network
        try:
            from src.data.gamecast import get_actionnetwork_odds
            live_odds = get_actionnetwork_odds(home_abbr, away_abbr)
            if live_odds and live_odds.get("ml_home_public") is not None:
                ml_pub = live_odds.get("ml_home_public", 0) or 0
                ml_mon = live_odds.get("ml_home_money", 0) or 0
        except Exception:
            logger.debug("live ActionNetwork odds unavailable", exc_info=True)
    if not ml_pub:
        # Fallback to DB (historical or if live fetch failed)
        odds_row = db.fetch_one(
            "SELECT ml_home_public, ml_home_money "
            "FROM game_odds WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
            (game_date, home_team_id, away_team_id))
        if odds_row:
            ml_pub = odds_row.get("ml_home_public") or 0
            ml_mon = odds_row.get("ml_home_money") or 0

    # Vegas lines (for upset identification only, NOT in formula)
    vegas_spread = 0.0
    vegas_home_ml = 0
    vegas_away_ml = 0
    odds_fetched_at = None
    try:
        odds = db.fetch_one(
            "SELECT spread, home_moneyline, away_moneyline, fetched_at FROM game_odds "
            "WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
            (game_date, home_team_id, away_team_id))
        if odds and odds["spread"] is not None:
            vegas_spread = odds["spread"]
            vegas_home_ml = odds.get("home_moneyline") or 0
            vegas_away_ml = odds.get("away_moneyline") or 0
            odds_fetched_at = odds.get("fetched_at")
    except Exception:
        logger.debug("historical Vegas lines unavailable", exc_info=True)

    # ── V2.1 features ──
    from src.analytics.elo import get_team_elo
    from src.analytics.stats_engine import (
        compute_travel,
        compute_momentum,
        compute_schedule_spots,
        compute_fg3_luck,
        compute_season_progress,
        compute_tanking_signal,
        compute_roster_shock,
    )

    _home_travel = compute_travel(home_team_id, game_date, away_team_id, is_home=True)
    _away_travel = compute_travel(away_team_id, game_date, home_team_id, is_home=False)
    _home_momentum = compute_momentum(home_team_id, game_date, season=game_season)
    _away_momentum = compute_momentum(away_team_id, game_date, season=game_season)
    _home_sched = compute_schedule_spots(home_team_id, game_date, away_team_id, season=game_season)
    _away_sched = compute_schedule_spots(away_team_id, game_date, home_team_id, season=game_season)
    feature_date = as_of_date or game_date
    _home_season_progress = compute_season_progress(
        home_team_id,
        feature_date,
        season=game_season,
    )
    _away_season_progress = compute_season_progress(
        away_team_id,
        feature_date,
        season=game_season,
    )
    _home_tank_live = compute_tanking_signal(
        home_team_id,
        feature_date,
        season=game_season,
        mode="live",
    )
    _away_tank_live = compute_tanking_signal(
        away_team_id,
        feature_date,
        season=game_season,
        mode="live",
    )
    _home_tank_oracle = compute_tanking_signal(
        home_team_id,
        feature_date,
        season=game_season,
        mode="oracle",
    )
    _away_tank_oracle = compute_tanking_signal(
        away_team_id,
        feature_date,
        season=game_season,
        mode="oracle",
    )
    _home_roster_shock = compute_roster_shock(
        home_team_id,
        feature_date,
        season=game_season,
    )
    _away_roster_shock = compute_roster_shock(
        away_team_id,
        feature_date,
        season=game_season,
    )

    # On/Off impact (season z-scored + reliability payloads)
    _onoff_signal_map = _get_onoff_team_signals_for_season(game_season)
    _home_onoff_payload = _onoff_signal_map.get(
        home_team_id,
        {"signal": 0.0, "reliability": 0.0},
    )
    _away_onoff_payload = _onoff_signal_map.get(
        away_team_id,
        {"signal": 0.0, "reliability": 0.0},
    )
    _home_onoff = float(_home_onoff_payload.get("signal", 0.0) or 0.0)
    _away_onoff = float(_away_onoff_payload.get("signal", 0.0) or 0.0)
    _home_onoff_rel = float(_home_onoff_payload.get("reliability", 0.0) or 0.0)
    _away_onoff_rel = float(_away_onoff_payload.get("reliability", 0.0) or 0.0)

    # Injury VORP lost
    _home_injury_vorp = 0.0
    _away_injury_vorp = 0.0
    if injured_players:
        for _pid, _play_prob in injured_players.items():
            _vorp_row = db.fetch_one(
                "SELECT pi.vorp, pi.team_id FROM player_impact pi "
                "WHERE pi.player_id = ? AND pi.season = ?",
                (_pid, game_season),
            )
            if _vorp_row and _vorp_row.get("vorp"):
                _lost = _vorp_row["vorp"] * (1.0 - _play_prob)
                if _vorp_row["team_id"] == home_team_id:
                    _home_injury_vorp += _lost
                elif _vorp_row["team_id"] == away_team_id:
                    _away_injury_vorp += _lost

    # Referee crew stats
    _ref_fouls_pg = 38.0
    _ref_home_bias = 50.0
    try:
        _ref_rows = db.fetch_all(
            "SELECT r.fouls_per_game, r.home_win_pct "
            "FROM game_referees gr "
            "JOIN referees r ON gr.referee_name = r.referee_name "
            "WHERE gr.game_date = ? AND gr.home_team_id = ? AND gr.away_team_id = ?",
            (game_date, home_team_id, away_team_id),
        )
        if _ref_rows:
            _ref_fouls_pg = sum(r.get("fouls_per_game", 38.0) or 38.0 for r in _ref_rows) / len(_ref_rows)
            _ref_home_bias = sum(r.get("home_win_pct", 50.0) or 50.0 for r in _ref_rows) / len(_ref_rows)
    except Exception:
        logger.debug("ref crew stats unavailable", exc_info=True)

    # Build GameInput
    game = GameInput(
        game_date=game_date,
        season=game_season,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_proj={k: v for k, v in home_proj.items() if not k.startswith("_")},
        away_proj={k: v for k, v in away_proj.items() if not k.startswith("_")},
        home_def_factor_raw=home_def_raw,
        away_def_factor_raw=away_def_raw,
        home_court=home_court,
        home_rest_days=hfat["rest_days"],
        away_rest_days=afat["rest_days"],
        home_b2b=hfat["b2b"],
        away_b2b=afat["b2b"],
        home_3in4=hfat["three_in_four"],
        away_3in4=afat["three_in_four"],
        home_4in6=hfat["four_in_six"],
        away_4in6=afat["four_in_six"],
        home_same_day=hfat["rest_days"] == 0,
        away_same_day=afat["rest_days"] == 0,
        home_off=home_off,
        away_off=away_off,
        home_def=home_def,
        away_def=away_def,
        home_pace=home_pace,
        away_pace=away_pace,
        home_ff={"efg": h_efg, "tov": h_tov, "oreb": h_oreb, "fta": h_fta,
                 "opp_efg": h_opp_efg, "opp_tov": h_opp_tov,
                 "opp_oreb": h_opp_oreb, "opp_fta": h_opp_fta},
        away_ff={"efg": a_efg, "tov": a_tov, "oreb": a_oreb, "fta": a_fta,
                 "opp_efg": a_opp_efg, "opp_tov": a_opp_tov,
                 "opp_oreb": a_opp_oreb, "opp_fta": a_opp_fta},
        home_clutch=h_clutch,
        away_clutch=a_clutch,
        home_hustle=h_hustle,
        away_hustle=a_hustle,
        ml_home_public=ml_pub,
        ml_home_money=ml_mon,
        vegas_spread=vegas_spread,
        vegas_home_ml=vegas_home_ml,
        vegas_away_ml=vegas_away_ml,
        # ── V2.1 fields ──
        # Elo
        home_elo=get_team_elo(home_team_id, game_date, game_season),
        away_elo=get_team_elo(away_team_id, game_date, game_season),
        # Travel
        home_travel_miles=_home_travel["travel_miles"],
        away_travel_miles=_away_travel["travel_miles"],
        home_tz_crossings=_home_travel["tz_crossings"],
        away_tz_crossings=_away_travel["tz_crossings"],
        home_cum_travel_7d=_home_travel["cum_travel_7d"],
        away_cum_travel_7d=_away_travel["cum_travel_7d"],
        # Momentum
        home_streak=_home_momentum["streak"],
        away_streak=_away_momentum["streak"],
        home_mov_trend=_home_momentum["mov_trend"],
        away_mov_trend=_away_momentum["mov_trend"],
        # Injury VORP
        home_injury_vorp_lost=_home_injury_vorp,
        away_injury_vorp_lost=_away_injury_vorp,
        # Referee
        ref_crew_fouls_pg=_ref_fouls_pg,
        ref_crew_home_bias=_ref_home_bias,
        # Schedule
        home_lookahead=_home_sched["lookahead"],
        away_lookahead=_away_sched["lookahead"],
        home_letdown=_home_sched["letdown"],
        away_letdown=_away_sched["letdown"],
        home_road_trip_game=_home_sched["road_trip_game"],
        away_road_trip_game=_away_sched["road_trip_game"],
        home_season_progress=_home_season_progress,
        away_season_progress=_away_season_progress,
        home_tank_signal_live=_home_tank_live,
        away_tank_signal_live=_away_tank_live,
        home_tank_signal_oracle=_home_tank_oracle,
        away_tank_signal_oracle=_away_tank_oracle,
        home_roster_shock=_home_roster_shock,
        away_roster_shock=_away_roster_shock,
        # SRS
        home_srs=hm.get("srs", 0.0) or 0.0,
        away_srs=am.get("srs", 0.0) or 0.0,
        # Pythagorean
        home_pythag_wpct=_derive_pythag_wpct(hm),
        away_pythag_wpct=_derive_pythag_wpct(am),
        # On/Off
        home_onoff_impact=_home_onoff,
        away_onoff_impact=_away_onoff,
        home_onoff_reliability=_home_onoff_rel,
        away_onoff_reliability=_away_onoff_rel,
        # Pace diff
        pace_diff=abs(home_pace - away_pace),
        # 3PT luck
        home_fg3_luck=compute_fg3_luck(home_team_id, game_date, season=game_season),
        away_fg3_luck=compute_fg3_luck(away_team_id, game_date, season=game_season),
        # Process stats
        home_process=h_process,
        away_process=a_process,
    )

    # Determine include_sharp from settings if not explicitly passed
    if not include_sharp:
        cfg = get_config()
        include_sharp = cfg.get("prediction_mode", "fundamentals") == "fundamentals_sharp"

    # Call THE prediction function
    pred = predict(game, w, include_sharp=include_sharp)

    # Fill in team names
    pred.home_team = home_abbr
    pred.away_team = away_abbr

    # Dog pick detection (moneyline-first, spread as fallback only).
    pred.is_dog_pick, pred.is_value_zone, pred.dog_payout = _classify_dog_pick(
        game_score=pred.game_score,
        vegas_spread=pred.vegas_spread,
        vegas_home_ml=pred.vegas_home_ml,
        vegas_away_ml=pred.vegas_away_ml,
    )

    pred.odds_fetched_at = odds_fetched_at
    return pred


# ──────────────────────────────────────────────────────────────
# Precompute disk + memory cache
# ──────────────────────────────────────────────────────────────

_PRECOMPUTE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")
_PRECOMPUTE_CACHE_FILE = os.path.join(_PRECOMPUTE_CACHE_DIR, "precomputed_games.pkl")
_CONTEXT_CACHE_FILE = os.path.join(_PRECOMPUTE_CACHE_DIR, "precompute_context.pkl")

# In-memory caches
_mem_pc_cache: Optional[Dict[str, GameInput]] = None
_mem_pc_schema: Optional[str] = None
_mem_ctx_cache: Optional[Dict[str, Any]] = None
_mem_ctx_fingerprint: Optional[str] = None
_mem_pc_lock = threading.Lock()
_MEM_PC_CACHE_MAX_GAMES = 15000


def _precompute_schema_version() -> str:
    """Hash of GameInput field names -- auto-invalidates when fields change."""
    names = tuple(f.name for f in dc_fields(GameInput))
    return hashlib.md5(str(names).encode()).hexdigest()[:12]


def _source_table_fingerprint(sql: str, params: tuple = ()) -> str:
    row = db.fetch_one(sql, params) or {}
    # Deterministic, compact, and cheap to compute.
    payload = "|".join(str(row.get(k, "")) for k in sorted(row.keys()))
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _precompute_source_fingerprint() -> str:
    """Fingerprint source tables used by precompute/cache pipelines."""
    parts = [
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date, "
            "COALESCE(SUM(points), 0.0) AS points_sum FROM player_stats"
        ),
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date, "
            "COALESCE(SUM(COALESCE(spread, 0.0)), 0.0) AS spread_sum FROM game_odds"
        ),
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(last_synced_at), '') AS max_sync "
            "FROM team_metrics"
        ),
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(last_synced_at), '') AS max_sync "
            "FROM player_impact"
        ),
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date FROM elo_ratings"
        ),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _compute_actual_results_fingerprint() -> str:
    """Fingerprint inputs used by get_actual_game_results()."""
    parts = [
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date, "
            "COALESCE(SUM(points), 0.0) AS points_sum FROM player_stats"
        ),
        _source_table_fingerprint(
            "SELECT COUNT(*) AS c, COALESCE(MAX(game_date), '') AS max_date, "
            "COALESCE(SUM(COALESCE(spread, 0.0)), 0.0) AS spread_sum FROM game_odds"
        ),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _backup_cache_file(path: str) -> None:
    """Keep a single .bak copy before overwriting cache files."""
    if not os.path.exists(path):
        return
    bak_path = f"{path}.bak"
    try:
        if os.path.exists(bak_path):
            os.remove(bak_path)
        os.replace(path, bak_path)
    except Exception:
        logger.debug("cache backup skipped for %s", path, exc_info=True)


def _game_cache_key(home_team_id: int, away_team_id: int, game_date: str) -> str:
    """Unique key for a game."""
    return f"{home_team_id}_{away_team_id}_{game_date}"


def _load_pc_cache(source_fingerprint: str) -> Dict[str, GameInput]:
    """Load precompute cache from memory or disk.

    Cache reuse is schema-gated (GameInput field layout). Source fingerprints are
    advisory metadata for diagnostics/context-cache decisions, but do not force a
    full precompute rebuild.
    """
    global _mem_pc_cache, _mem_pc_schema
    schema = _precompute_schema_version()

    with _mem_pc_lock:
        if _mem_pc_cache is not None and _mem_pc_schema == schema:
            return dict(_mem_pc_cache)

        try:
            if os.path.exists(_PRECOMPUTE_CACHE_FILE):
                with open(_PRECOMPUTE_CACHE_FILE, "rb") as f:
                    data = pickle.load(f)

                if not isinstance(data, dict):
                    logger.info("Precompute cache format mismatch -- will rebuild")
                    return {}

                if data.get("schema") != schema:
                    logger.info("Precompute cache schema mismatch -- will rebuild")
                    return {}

                loaded_games_raw = data.get("games")
                if not isinstance(loaded_games_raw, dict):
                    logger.info("Precompute cache payload mismatch -- will rebuild")
                    return {}

                loaded_games = dict(loaded_games_raw)
                cached_source_fingerprint = str(data.get("source_fingerprint") or "")
                if (
                    cached_source_fingerprint
                    and cached_source_fingerprint != source_fingerprint
                ):
                    logger.info(
                        "Precompute source fingerprint changed "
                        "(cache=%s, current=%s) -- reusing cached games and "
                        "computing only missing keys",
                        cached_source_fingerprint,
                        source_fingerprint,
                    )

                if len(loaded_games) <= _MEM_PC_CACHE_MAX_GAMES:
                    _mem_pc_cache = dict(loaded_games)
                else:
                    _mem_pc_cache = None
                _mem_pc_schema = schema
                logger.info(
                    "Loaded precompute cache from disk (%d games)", len(loaded_games)
                )
                return dict(loaded_games)
        except Exception as e:
            logger.warning("Failed to load precompute cache: %s", e)

    return {}


def _save_pc_cache(cache: Dict[str, GameInput], source_fingerprint: str):
    """Persist precompute cache to disk and update in-memory copy."""
    global _mem_pc_cache, _mem_pc_schema
    schema = _precompute_schema_version()
    snapshot = dict(cache)
    os.makedirs(_PRECOMPUTE_CACHE_DIR, exist_ok=True)
    with _mem_pc_lock:
        try:
            _backup_cache_file(_PRECOMPUTE_CACHE_FILE)
            with open(_PRECOMPUTE_CACHE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "schema": schema,
                        "source_fingerprint": source_fingerprint,
                        "games": snapshot,
                    },
                    f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved precompute cache to disk (%d games)", len(snapshot))
        except Exception as e:
            logger.warning("Failed to save precompute cache: %s", e)
        _mem_pc_cache = dict(snapshot) if len(snapshot) <= _MEM_PC_CACHE_MAX_GAMES else None
        _mem_pc_schema = schema


def invalidate_precompute_cache():
    """Clear all precompute caches (games + context, memory + disk)."""
    global _mem_pc_cache, _mem_pc_schema, _mem_ctx_cache, _mem_ctx_fingerprint
    _mem_pc_cache = None
    _mem_pc_schema = None
    _mem_ctx_cache = None
    _mem_ctx_fingerprint = None
    for path in (_PRECOMPUTE_CACHE_FILE, _CONTEXT_CACHE_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.debug("cache delete skipped for %s", path, exc_info=True)
    logger.info("Invalidated all precompute caches")


# ──────────────────────────────────────────────────────────────
# Precompute context: historical rosters + inferred injuries
# ──────────────────────────────────────────────────────────────

def _load_ctx_cache(source_fingerprint: str) -> Optional[Dict[str, Any]]:
    """Load precompute context from memory or disk if fingerprint matches."""
    global _mem_ctx_cache, _mem_ctx_fingerprint
    if _mem_ctx_cache is not None and _mem_ctx_fingerprint == source_fingerprint:
        return _mem_ctx_cache
    try:
        if os.path.exists(_CONTEXT_CACHE_FILE):
            with open(_CONTEXT_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("source_fingerprint") == source_fingerprint:
                _mem_ctx_cache = data["ctx"]
                _mem_ctx_fingerprint = source_fingerprint
                logger.info("Loaded precompute context from disk")
                return _mem_ctx_cache
    except Exception as e:
        logger.warning("Failed to load context cache: %s", e)
    return None


def _save_ctx_cache(ctx: Dict[str, Any], source_fingerprint: str):
    """Persist precompute context to disk."""
    global _mem_ctx_cache, _mem_ctx_fingerprint
    os.makedirs(_PRECOMPUTE_CACHE_DIR, exist_ok=True)
    try:
        _backup_cache_file(_CONTEXT_CACHE_FILE)
        with open(_CONTEXT_CACHE_FILE, "wb") as f:
            pickle.dump({"source_fingerprint": source_fingerprint, "ctx": ctx}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved precompute context to disk")
    except Exception as e:
        logger.warning("Failed to save context cache: %s", e)
    _mem_ctx_cache = ctx
    _mem_ctx_fingerprint = source_fingerprint


def _build_precompute_context(
    games: List[Dict], source_fingerprint: str, force: bool = False
) -> Dict[str, Any]:
    """Build lookup tables for historical roster + injury inference.

    One bulk SQL query determines which team each player was on for every
    game. For each game we can infer injuries by comparing the recent active
    roster vs who actually played. Cached to disk.
    """
    if not force:
        cached = _load_ctx_cache(source_fingerprint)
        if cached is not None:
            return cached

    # game_id -> (home_team_id, away_team_id)
    game_team_map: Dict[str, tuple] = {}
    for g in games:
        gid = g.get("game_id")
        if gid:
            game_team_map[gid] = (g["home_team_id"], g["away_team_id"])

    # Player info
    player_info: Dict[int, Dict] = {}
    for r in db.fetch_all("SELECT player_id, name, position FROM players"):
        player_info[r["player_id"]] = {
            "player_id": r["player_id"],
            "name": r["name"],
            "position": r["position"],
        }

    # All player game appearances (scoped to seasons present in these games).
    game_seasons = sorted({_game_date_to_season(g["game_date"]) for g in games if g.get("game_date")})
    if game_seasons:
        placeholders = ",".join("?" for _ in game_seasons)
        rows = db.fetch_all(
            f"""
            SELECT player_id, game_id, game_date, is_home,
                   points, minutes
            FROM player_stats
            WHERE season IN ({placeholders})
            ORDER BY game_date
            """,
            tuple(game_seasons),
        )
    else:
        rows = db.fetch_all(
            """
            SELECT player_id, game_id, game_date, is_home,
                   points, minutes
            FROM player_stats
            ORDER BY game_date
            """
        )

    team_game_players: Dict[tuple, set] = defaultdict(set)
    player_season_stats: Dict[int, List[Dict]] = defaultdict(list)

    for r in rows:
        gid = r["game_id"]
        if gid not in game_team_map:
            continue
        htid, atid = game_team_map[gid]
        team_id = htid if r["is_home"] else atid
        pid = r["player_id"]
        gdate = r["game_date"]

        team_game_players[(team_id, gdate)].add(pid)
        player_season_stats[pid].append({
            "pts": r["points"] or 0,
            "mins": r["minutes"] or 0,
        })

    # Team game dates (sorted) for binary search
    team_dates: Dict[int, List[str]] = defaultdict(list)
    for (tid, gdate) in team_game_players:
        team_dates[tid].append(gdate)
    for tid in team_dates:
        team_dates[tid] = sorted(set(team_dates[tid]))

    # Player average stats (full season)
    player_avg: Dict[int, Dict[str, float]] = {}
    for pid, stats in player_season_stats.items():
        n = len(stats)
        if n > 0:
            player_avg[pid] = {
                "ppg": sum(s["pts"] for s in stats) / n,
                "mpg": sum(s["mins"] for s in stats) / n,
            }

    result = {
        "game_team_map": dict(game_team_map),
        "team_game_players": dict(team_game_players),
        "team_dates": dict(team_dates),
        "player_info": player_info,
        "player_avg": player_avg,
    }
    _save_ctx_cache(result, source_fingerprint)
    return result


def _get_historical_roster(team_id: int, game_date: str,
                           ctx: Dict[str, Any]) -> List[Dict]:
    """Get the roster for a team on a given date from precompute context."""
    pids = ctx["team_game_players"].get((team_id, game_date), set())
    info = ctx["player_info"]
    return [info.get(pid, {"player_id": pid, "name": "Unknown", "position": "F"})
            for pid in pids]


def _infer_historical_injuries(team_id: int, game_date: str,
                               ctx: Dict[str, Any]) -> Dict[str, float]:
    """Infer injuries for a historical game by comparing recent roster vs actual.

    If a rotation player (10+ min/game) was playing in the last 5 games
    but not in THIS game, they were effectively injured/unavailable.
    """
    team_dates = ctx["team_dates"].get(team_id, [])
    tgp = ctx["team_game_players"]
    pavg = ctx["player_avg"]

    idx = bisect.bisect_left(team_dates, game_date)
    recent_dates = team_dates[max(0, idx - 5):idx]

    if not recent_dates:
        return {"injured_count": 0, "injury_ppg_lost": 0.0, "injury_minutes_lost": 0.0}

    expected = set()
    for d in recent_dates:
        expected |= tgp.get((team_id, d), set())

    actual = tgp.get((team_id, game_date), set())
    missing = expected - actual

    count = 0
    ppg_lost = 0.0
    min_lost = 0.0
    for pid in missing:
        avg = pavg.get(pid, {})
        mpg = avg.get("mpg", 0)
        if mpg >= 10:
            count += 1
            ppg_lost += avg.get("ppg", 0)
            min_lost += mpg

    return {
        "injured_count": count,
        "injury_ppg_lost": ppg_lost,
        "injury_minutes_lost": min_lost,
    }


# ──────────────────────────────────────────────────────────────
# get_actual_game_results() -- reconstructs games from player_stats
# ──────────────────────────────────────────────────────────────

_actual_results_cache: Optional[List[Dict]] = None
_actual_results_fingerprint: Optional[str] = None


def get_actual_game_results(team_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Aggregate player_stats by (game_id, is_home) to reconstruct game scores.

    Cached at module level and invalidated by source-data fingerprint.
    """
    global _actual_results_cache, _actual_results_fingerprint
    source_fingerprint = _compute_actual_results_fingerprint()

    if _actual_results_cache is not None and source_fingerprint == _actual_results_fingerprint:
        if team_id:
            return [
                dict(g)
                for g in _actual_results_cache
                if g["home_team_id"] == team_id or g["away_team_id"] == team_id
            ]
        return [dict(g) for g in _actual_results_cache]

    query = """
        SELECT ps.game_id, ps.game_date, ps.is_home,
               ps.opponent_team_id,
               SUM(ps.points) as total_pts, COUNT(*) as player_count
        FROM player_stats ps
        GROUP BY ps.game_id, ps.is_home
        ORDER BY ps.game_date ASC
    """
    rows = db.fetch_all(query)

    games_dict: Dict[str, Dict] = {}
    for r in rows:
        gid = r["game_id"]
        if gid not in games_dict:
            games_dict[gid] = {"game_id": gid, "game_date": r["game_date"]}

        if r["is_home"]:
            games_dict[gid]["home_score"] = r["total_pts"]
            games_dict[gid]["away_team_id"] = r["opponent_team_id"]
            games_dict[gid]["home_count"] = r["player_count"]
        else:
            games_dict[gid]["away_score"] = r["total_pts"]
            games_dict[gid]["home_team_id"] = r["opponent_team_id"]
            games_dict[gid]["away_count"] = r["player_count"]

    results = []
    for gid, g in games_dict.items():
        home_score = g.get("home_score", 0)
        away_score = g.get("away_score", 0)
        if home_score < 40 or away_score < 40:
            continue
        if g.get("home_count", 0) < 4 or g.get("away_count", 0) < 4:
            continue
        if "home_team_id" not in g or "away_team_id" not in g:
            continue

        spread = home_score - away_score
        if spread > ACTUAL_WIN_THRESHOLD:
            winner = "HOME"
        elif spread < -ACTUAL_WIN_THRESHOLD:
            winner = "AWAY"
        else:
            winner = "PUSH"

        g["home_score"] = home_score
        g["away_score"] = away_score
        g["actual_spread"] = spread
        g["actual_total"] = home_score + away_score
        g["winner"] = winner
        results.append(g)

    # Attach vegas odds
    try:
        odds_rows = db.fetch_all(
            "SELECT game_date, home_team_id, away_team_id, spread, over_under, "
            "home_moneyline, away_moneyline, ml_home_public, ml_home_money "
            "FROM game_odds"
        )
        odds_map = {(r["game_date"], r["home_team_id"], r["away_team_id"]): r
                    for r in odds_rows}
        for g in results:
            key = (g["game_date"], g.get("home_team_id"), g.get("away_team_id"))
            if key in odds_map:
                o = odds_map[key]
                g["vegas_spread"] = o["spread"]
                g["vegas_total"] = o["over_under"]
                g["vegas_home_ml"] = o["home_moneyline"]
                g["vegas_away_ml"] = o["away_moneyline"]
                g["ml_home_public"] = o["ml_home_public"]
                g["ml_home_money"] = o["ml_home_money"]
            else:
                g["vegas_spread"] = None
                g["vegas_total"] = None
                g["vegas_home_ml"] = None
                g["vegas_away_ml"] = None
                g["ml_home_public"] = None
                g["ml_home_money"] = None
    except Exception as e:
        logger.warning("Could not load game_odds: %s", e)
        for g in results:
            g["vegas_spread"] = None
            g["vegas_total"] = None
            g["vegas_home_ml"] = None
            g["vegas_away_ml"] = None
            g["ml_home_public"] = None
            g["ml_home_money"] = None

    _actual_results_cache = results
    _actual_results_fingerprint = source_fingerprint

    if team_id:
        return [
            dict(g)
            for g in results
            if g["home_team_id"] == team_id or g["away_team_id"] == team_id
        ]
    return [dict(g) for g in results]


def invalidate_results_cache():
    """Clear cached game results (call when new data is synced)."""
    global _actual_results_cache, _actual_results_fingerprint
    _actual_results_cache = None
    _actual_results_fingerprint = None


# ──────────────────────────────────────────────────────────────
# precompute_all_games() -- builds GameInput for all historical games
# ──────────────────────────────────────────────────────────────

def precompute_all_games(callback=None, force=False) -> List[GameInput]:
    """Build a list of GameInput objects for optimization/backtest.

    Uses a persistent disk + memory cache so that already-computed historical
    games are never reprocessed. Only truly new games go through the expensive
    per-game projection pipeline. Cache invalidates automatically when the
    GameInput schema changes (or when ``force=True`` is used).

    Pass ``force=True`` to discard the cache and recompute everything.
    """
    from src.database.db import ensure_thread_local_db
    source_fingerprint = _precompute_source_fingerprint()

    # Load cache
    cache = {} if force else _load_pc_cache(source_fingerprint)

    all_games = get_actual_game_results()
    if not all_games:
        if callback:
            callback("No game results found")
        return []

    # Filter teams with < 5 games
    team_games = Counter()
    for g in all_games:
        team_games[g.get("home_team_id", 0)] += 1
        team_games[g.get("away_team_id", 0)] += 1
    valid_teams = {tid for tid, cnt in team_games.items() if cnt >= 5}
    games = [g for g in all_games
             if g.get("home_team_id") in valid_teams and g.get("away_team_id") in valid_teams]
    game_seasons = sorted({_game_date_to_season(g["game_date"]) for g in games if g.get("game_date")})

    # Determine which games still need computing
    valid_keys = set()
    new_games = []
    for g in games:
        key = _game_cache_key(g["home_team_id"], g["away_team_id"], g["game_date"])
        valid_keys.add(key)
        if key not in cache:
            new_games.append(g)

    if not new_games:
        result = [cache[k] for k in valid_keys if k in cache]
        result.sort(key=lambda gi: gi.game_date)
        if callback:
            callback(f"Loaded {len(result)} precomputed games from cache (0 new)")
        return result

    cached_count = len(valid_keys) - len(new_games)
    if callback:
        callback(f"Precomputing {len(new_games)} new games ({cached_count} cached)...")

    # Build historical context ONCE
    if callback:
        callback("Building historical roster & injury context...")
    ctx = _build_precompute_context(games, source_fingerprint)

    # Pre-load memory store so player_splits() / _get_team_metrics() use
    # pandas DataFrames instead of per-player DB queries.
    from src.analytics.memory_store import InMemoryDataStore
    _store = InMemoryDataStore()
    if not _store.is_loaded:
        if callback:
            callback("Loading memory store for fast lookups...")
        _store.load()

    # ── Pre-build invariant lookup caches (read once, used per-game) ──
    import bisect

    # Elo: (team_id, season) -> sorted [(game_date, elo), ...]
    if game_seasons:
        placeholders = ",".join("?" for _ in game_seasons)
        _elo_rows = db.fetch_all(
            f"SELECT team_id, game_date, season, elo FROM elo_ratings "
            f"WHERE season IN ({placeholders}) ORDER BY team_id, season, game_date",
            tuple(game_seasons),
        )
    else:
        _elo_rows = db.fetch_all(
            "SELECT team_id, game_date, season, elo FROM elo_ratings "
            "ORDER BY team_id, season, game_date"
        )
    _elo_by_team: dict = {}
    for _r in (_elo_rows or []):
        _key = (_r["team_id"], _r.get("season") or "")
        _elo_by_team.setdefault(_key, []).append((_r["game_date"], _r["elo"]))

    def _cached_elo(team_id: int, game_date: str, season: str) -> float:
        entries = _elo_by_team.get((team_id, season)) or _elo_by_team.get((team_id, ""))
        if not entries:
            return 1500.0
        idx = bisect.bisect_left(entries, (game_date,)) - 1
        return entries[idx][1] if idx >= 0 else 1500.0

    # On/Off impact: (team_id, season) -> {"signal", "reliability"}
    _onoff_cache = _get_onoff_team_signals_for_seasons(game_seasons)

    # Odds: (game_date, home_team_id, away_team_id) -> row dict
    _odds_rows = db.fetch_all(
        "SELECT game_date, home_team_id, away_team_id, spread_home_money, spread_home_public "
        "FROM game_odds"
    )
    _odds_cache: dict = {}
    for _r in (_odds_rows or []):
        _odds_cache[(_r["game_date"], _r["home_team_id"], _r["away_team_id"])] = _r

    if callback:
        callback(f"Cached {len(_elo_by_team)} team Elo histories, "
                 f"{len(_onoff_cache)} on/off entries, {len(_odds_cache)} odds rows")

    cfg = get_config()
    max_workers = cfg.get("worker_threads", 4)
    precompute_log_every = max(
        25,
        int(cfg.get("precompute_progress_log_every", 200) or 200),
    )
    _pc_lock = threading.Lock()
    completed_count = [0]

    def _precompute_one(g):
        """Process one game (thread-local DB set up by pool initializer)."""
        htid = g["home_team_id"]
        atid = g["away_team_id"]
        gdate = g["game_date"]

        # Historical roster from context
        home_roster = _get_historical_roster(htid, gdate, ctx)
        away_roster = _get_historical_roster(atid, gdate, ctx)

        # Projections
        home_proj = aggregate_projection(htid, atid, is_home=1, as_of_date=gdate,
                                         roster=home_roster)
        away_proj = aggregate_projection(atid, htid, is_home=0, as_of_date=gdate,
                                         roster=away_roster)

        # Season for this game
        game_season = _game_date_to_season(gdate)

        # Home court
        home_court = get_home_court_advantage(htid, season=game_season)

        # Metrics
        hm = _get_team_metrics(htid, season=game_season)
        am = _get_team_metrics(atid, season=game_season)

        league_avg = _LEAGUE_AVG_PPG
        away_opp_pts = am.get("opp_pts", league_avg) or league_avg
        home_opp_pts = hm.get("opp_pts", league_avg) or league_avg
        away_def_raw = away_opp_pts / league_avg if league_avg > 0 else 1.0
        home_def_raw = home_opp_pts / league_avg if league_avg > 0 else 1.0

        # Fatigue
        from src.analytics.weight_config import get_weight_config as _gwc
        _w = _gwc()
        hfat = compute_fatigue(htid, gdate, w=_w)
        afat = compute_fatigue(atid, gdate, w=_w)

        # Ratings
        home_off = hm.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
        away_off = am.get("off_rating", _RATING_FALLBACK) or _RATING_FALLBACK
        home_def = hm.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK
        away_def = am.get("def_rating", _RATING_FALLBACK) or _RATING_FALLBACK

        # Pace
        home_pace = hm.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK
        away_pace = am.get("pace", _PACE_FALLBACK) or _PACE_FALLBACK

        # Four Factors
        h_efg = hm.get("ff_efg_pct", 0) or 0
        a_efg = am.get("ff_efg_pct", 0) or 0
        h_tov = hm.get("ff_tm_tov_pct", 0) or 0
        a_tov = am.get("ff_tm_tov_pct", 0) or 0
        h_oreb = hm.get("ff_oreb_pct", 0) or 0
        a_oreb = am.get("ff_oreb_pct", 0) or 0
        h_fta = hm.get("ff_fta_rate", 0) or 0
        a_fta = am.get("ff_fta_rate", 0) or 0

        h_opp_efg = hm.get("opp_efg_pct", 0) or 0
        a_opp_efg = am.get("opp_efg_pct", 0) or 0
        h_opp_tov = hm.get("opp_tm_tov_pct", 0) or 0
        a_opp_tov = am.get("opp_tm_tov_pct", 0) or 0
        h_opp_oreb = hm.get("opp_oreb_pct", 0) or 0
        a_opp_oreb = am.get("opp_oreb_pct", 0) or 0
        h_opp_fta = hm.get("opp_fta_rate", 0) or 0
        a_opp_fta = am.get("opp_fta_rate", 0) or 0

        # Clutch
        h_clutch = {"net_rating": hm.get("clutch_net_rating", 0) or 0,
                    "efg_pct": hm.get("clutch_efg_pct", 0) or 0}
        a_clutch = {"net_rating": am.get("clutch_net_rating", 0) or 0,
                    "efg_pct": am.get("clutch_efg_pct", 0) or 0}

        # Hustle (normalize to per-game)
        h_gp = max(1, hm.get("gp", 1) or 1)
        a_gp = max(1, am.get("gp", 1) or 1)
        h_hustle = {"deflections": (hm.get("deflections", 0) or 0) / h_gp,
                    "contested": (hm.get("contested_shots", 0) or 0) / h_gp,
                    "loose_balls": (hm.get("loose_balls_recovered", 0) or 0) / h_gp}
        a_hustle = {"deflections": (am.get("deflections", 0) or 0) / a_gp,
                    "contested": (am.get("contested_shots", 0) or 0) / a_gp,
                    "loose_balls": (am.get("loose_balls_recovered", 0) or 0) / a_gp}

        # Process stats (already per-game from Misc API)
        h_process = {
            "paint": hm.get("points_in_paint", 0) or 0,
            "fb": hm.get("fast_break_pts", 0) or 0,
            "sec": hm.get("second_chance_pts", 0) or 0,
            "off_tov": hm.get("pts_off_tov", 0) or 0,
            "opp_paint": hm.get("opp_pts_paint", 0) or 0,
            "opp_fb": hm.get("opp_pts_fb", 0) or 0,
            "opp_sec": hm.get("opp_pts_2nd_chance", 0) or 0,
            "opp_off_tov": hm.get("opp_pts_off_tov", 0) or 0,
        }
        a_process = {
            "paint": am.get("points_in_paint", 0) or 0,
            "fb": am.get("fast_break_pts", 0) or 0,
            "sec": am.get("second_chance_pts", 0) or 0,
            "off_tov": am.get("pts_off_tov", 0) or 0,
            "opp_paint": am.get("opp_pts_paint", 0) or 0,
            "opp_fb": am.get("opp_pts_fb", 0) or 0,
            "opp_sec": am.get("opp_pts_2nd_chance", 0) or 0,
            "opp_off_tov": am.get("opp_pts_off_tov", 0) or 0,
        }

        # ── V2.1 features ──
        from src.analytics.stats_engine import (
            compute_travel,
            compute_momentum,
            compute_schedule_spots,
            compute_fg3_luck,
            compute_season_progress,
            compute_tanking_signal,
            compute_roster_shock,
        )

        _home_travel = compute_travel(htid, gdate, atid, is_home=True)
        _away_travel = compute_travel(atid, gdate, htid, is_home=False)
        _home_momentum = compute_momentum(htid, gdate, season=game_season)
        _away_momentum = compute_momentum(atid, gdate, season=game_season)
        _home_sched = compute_schedule_spots(htid, gdate, atid, season=game_season)
        _away_sched = compute_schedule_spots(atid, gdate, htid, season=game_season)
        _home_season_progress = compute_season_progress(
            htid,
            gdate,
            season=game_season,
        )
        _away_season_progress = compute_season_progress(
            atid,
            gdate,
            season=game_season,
        )
        _home_tank_live = compute_tanking_signal(
            htid,
            gdate,
            season=game_season,
            mode="live",
        )
        _away_tank_live = compute_tanking_signal(
            atid,
            gdate,
            season=game_season,
            mode="live",
        )
        _home_tank_oracle = compute_tanking_signal(
            htid,
            gdate,
            season=game_season,
            mode="oracle",
        )
        _away_tank_oracle = compute_tanking_signal(
            atid,
            gdate,
            season=game_season,
            mode="oracle",
        )
        _home_roster_shock = compute_roster_shock(
            htid,
            gdate,
            season=game_season,
        )
        _away_roster_shock = compute_roster_shock(
            atid,
            gdate,
            season=game_season,
        )

        # On/Off impact (from pre-built cache)
        _home_onoff_payload = _onoff_cache.get(
            (htid, game_season),
            {"signal": 0.0, "reliability": 0.0},
        )
        _away_onoff_payload = _onoff_cache.get(
            (atid, game_season),
            {"signal": 0.0, "reliability": 0.0},
        )
        _home_onoff = float(_home_onoff_payload.get("signal", 0.0) or 0.0)
        _away_onoff = float(_away_onoff_payload.get("signal", 0.0) or 0.0)
        _home_onoff_rel = float(_home_onoff_payload.get("reliability", 0.0) or 0.0)
        _away_onoff_rel = float(_away_onoff_payload.get("reliability", 0.0) or 0.0)

        # Spread sharp edge (from pre-built cache)
        _spread_sharp = 0.0
        _odds_entry = _odds_cache.get((gdate, htid, atid))
        if _odds_entry:
            _sm = _odds_entry.get("spread_home_money", 0) or 0
            _sp = _odds_entry.get("spread_home_public", 0) or 0
            _spread_sharp = float(_sm - _sp)

        return GameInput(
            game_date=gdate,
            season=game_season,
            home_team_id=htid,
            away_team_id=atid,
            actual_home_score=g.get("home_score", 0),
            actual_away_score=g.get("away_score", 0),
            home_proj={k: v for k, v in home_proj.items() if not k.startswith("_")},
            away_proj={k: v for k, v in away_proj.items() if not k.startswith("_")},
            home_def_factor_raw=home_def_raw,
            away_def_factor_raw=away_def_raw,
            home_court=home_court,
            home_rest_days=hfat["rest_days"],
            away_rest_days=afat["rest_days"],
            home_b2b=hfat["b2b"],
            away_b2b=afat["b2b"],
            home_3in4=hfat["three_in_four"],
            away_3in4=afat["three_in_four"],
            home_4in6=hfat["four_in_six"],
            away_4in6=afat["four_in_six"],
            home_same_day=hfat["rest_days"] == 0,
            away_same_day=afat["rest_days"] == 0,
            home_off=home_off,
            away_off=away_off,
            home_def=home_def,
            away_def=away_def,
            home_pace=home_pace,
            away_pace=away_pace,
            home_ff={"efg": h_efg, "tov": h_tov, "oreb": h_oreb, "fta": h_fta,
                     "opp_efg": h_opp_efg, "opp_tov": h_opp_tov,
                     "opp_oreb": h_opp_oreb, "opp_fta": h_opp_fta},
            away_ff={"efg": a_efg, "tov": a_tov, "oreb": a_oreb, "fta": a_fta,
                     "opp_efg": a_opp_efg, "opp_tov": a_opp_tov,
                     "opp_oreb": a_opp_oreb, "opp_fta": a_opp_fta},
            home_clutch=h_clutch,
            away_clutch=a_clutch,
            home_hustle=h_hustle,
            away_hustle=a_hustle,
            ml_home_public=g.get("ml_home_public") or 0,
            ml_home_money=g.get("ml_home_money") or 0,
            vegas_spread=g.get("vegas_spread") or 0.0,
            vegas_home_ml=g.get("vegas_home_ml") or 0,
            vegas_away_ml=g.get("vegas_away_ml") or 0,
            # ── V2.1 fields ──
            # Elo (from pre-built cache)
            home_elo=_cached_elo(htid, gdate, game_season),
            away_elo=_cached_elo(atid, gdate, game_season),
            # Travel
            home_travel_miles=_home_travel["travel_miles"],
            away_travel_miles=_away_travel["travel_miles"],
            home_tz_crossings=_home_travel["tz_crossings"],
            away_tz_crossings=_away_travel["tz_crossings"],
            home_cum_travel_7d=_home_travel["cum_travel_7d"],
            away_cum_travel_7d=_away_travel["cum_travel_7d"],
            # Momentum
            home_streak=_home_momentum["streak"],
            away_streak=_away_momentum["streak"],
            home_mov_trend=_home_momentum["mov_trend"],
            away_mov_trend=_away_momentum["mov_trend"],
            # Schedule
            home_lookahead=_home_sched["lookahead"],
            away_lookahead=_away_sched["lookahead"],
            home_letdown=_home_sched["letdown"],
            away_letdown=_away_sched["letdown"],
            home_road_trip_game=_home_sched["road_trip_game"],
            away_road_trip_game=_away_sched["road_trip_game"],
            home_season_progress=_home_season_progress,
            away_season_progress=_away_season_progress,
            home_tank_signal_live=_home_tank_live,
            away_tank_signal_live=_away_tank_live,
            home_tank_signal_oracle=_home_tank_oracle,
            away_tank_signal_oracle=_away_tank_oracle,
            home_roster_shock=_home_roster_shock,
            away_roster_shock=_away_roster_shock,
            # SRS
            home_srs=hm.get("srs", 0.0) or 0.0,
            away_srs=am.get("srs", 0.0) or 0.0,
            # Pythagorean
            home_pythag_wpct=_derive_pythag_wpct(hm),
            away_pythag_wpct=_derive_pythag_wpct(am),
            # On/Off
            home_onoff_impact=_home_onoff,
            away_onoff_impact=_away_onoff,
            home_onoff_reliability=_home_onoff_rel,
            away_onoff_reliability=_away_onoff_rel,
            # Pace diff
            pace_diff=abs(home_pace - away_pace),
            # Spread sharp edge
            spread_sharp_edge=_spread_sharp,
            # 3PT luck
            home_fg3_luck=compute_fg3_luck(htid, gdate, season=game_season),
            away_fg3_luck=compute_fg3_luck(atid, gdate, season=game_season),
            # Process stats
            home_process=h_process,
            away_process=a_process,
        )

    # Compute new games in parallel
    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers, initializer=ensure_thread_local_db) as executor:
        futures = {executor.submit(_precompute_one, g): g for g in new_games}
        for future in as_completed(futures):
            game = futures[future]
            try:
                gi = future.result()
                if gi is not None:
                    new_results.append(gi)
            except Exception as e:
                logger.warning("Skipping game %s: %s", game.get("game_id"), e)

            with _pc_lock:
                completed_count[0] += 1
                c = completed_count[0]
                if callback and c % precompute_log_every == 0:
                    callback(f"Precomputed {c}/{len(new_games)} games")

    # Merge new results into cache and persist
    for gi in new_results:
        key = _game_cache_key(gi.home_team_id, gi.away_team_id, gi.game_date)
        cache[key] = gi

    if new_results:
        _save_pc_cache(cache, source_fingerprint)

    # Return only valid games
    result = [cache[k] for k in valid_keys if k in cache]
    result.sort(key=lambda gi: gi.game_date)

    if callback:
        callback(f"Precomputed {len(result)} games total ({len(new_results)} new, {cached_count} cached)")
    return result
