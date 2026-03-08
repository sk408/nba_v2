"""Single-path prediction engine for NBA Fundamentals V2.

One predict() function used by live predictions, backtesting, and optimization.
No three-path sync problem. No Elo, no ESPN blend, no ML ensemble, no opening
spread, no residual calibration, no autotune, no per-team weights.

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

from src.database import db
from src.config import get_season, get_config, get_historical_seasons
from src.analytics.weight_config import WeightConfig, get_weight_config
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
    # SRS / Pythagorean
    home_srs: float = 0.0
    away_srs: float = 0.0
    home_pythag_wpct: float = 0.5
    away_pythag_wpct: float = 0.5
    # Player On/Off impact
    home_onoff_impact: float = 0.0
    away_onoff_impact: float = 0.0
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
              oreb_e * w.ff_oreb_weight + fta_e * w.ff_fta_weight) * w.four_factors_scale
    game_score += ff_adj
    pred.adjustments["four_factors"] = ff_adj

    # Opponent Four Factors (defensive matchup)
    opp_efg_e = aff.get("opp_efg", 0) - hff.get("opp_efg", 0)
    opp_tov_e = hff.get("opp_tov", 0) - aff.get("opp_tov", 0)
    opp_oreb_e = aff.get("opp_oreb", 0) - hff.get("opp_oreb", 0)
    opp_fta_e = aff.get("opp_fta", 0) - hff.get("opp_fta", 0)
    opp_ff_adj = (opp_efg_e * w.opp_ff_efg_weight + opp_tov_e * w.opp_ff_tov_weight +
                  opp_oreb_e * w.opp_ff_oreb_weight + opp_fta_e * w.opp_ff_fta_weight) * w.four_factors_scale
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

    # 17. Momentum (win/loss streak)
    momentum_adj = (game.home_streak - game.away_streak) * w.momentum_streak_mult
    game_score += momentum_adj
    pred.adjustments["momentum"] = momentum_adj

    # 18. Margin-of-victory trend
    mov_adj = (game.home_mov_trend - game.away_mov_trend) * w.mov_trend_mult
    game_score += mov_adj
    pred.adjustments["mov_trend"] = mov_adj

    # 19. Injury VORP impact (more VORP lost = weaker team)
    injury_adj = (game.away_injury_vorp_lost - game.home_injury_vorp_lost) * w.injury_vorp_mult
    game_score += injury_adj
    pred.adjustments["injury_vorp"] = injury_adj

    # 20. Referee home bias (50% = neutral, >50% = favors home)
    ref_bias_adj = (game.ref_crew_home_bias - 50.0) / 50.0 * w.ref_home_bias_mult
    game_score += ref_bias_adj
    pred.adjustments["ref_home_bias"] = ref_bias_adj

    # 21. Spread sharp money
    if game.spread_sharp_edge:
        spread_sharp_adj = game.spread_sharp_edge / 100.0 * w.sharp_spread_weight
        game_score += spread_sharp_adj
        pred.adjustments["sharp_spread"] = spread_sharp_adj

    # 22. Schedule spots
    sched_adj = (-(game.home_lookahead * w.lookahead_penalty +
                   game.home_letdown * w.letdown_penalty)
                 + (game.away_lookahead * w.lookahead_penalty +
                    game.away_letdown * w.letdown_penalty))
    game_score += sched_adj
    pred.adjustments["schedule_spots"] = sched_adj

    # 23. SRS differential
    srs_adj = (game.home_srs - game.away_srs) * w.srs_diff_mult
    game_score += srs_adj
    pred.adjustments["srs"] = srs_adj

    # 24. Player On/Off impact
    onoff_adj = (game.home_onoff_impact - game.away_onoff_impact) * w.onoff_impact_mult
    game_score += onoff_adj
    pred.adjustments["onoff_impact"] = onoff_adj

    # 25. 3PT luck regression (hot team regresses down, cold team regresses up)
    # Negative because positive luck = overperforming = expect regression down
    fg3_luck_adj = -(game.home_fg3_luck - game.away_fg3_luck) * w.fg3_luck_mult
    game_score += fg3_luck_adj
    pred.adjustments["fg3_luck"] = fg3_luck_adj

    # 26. Process stats matchup (paint scoring vs paint defense, etc.)
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
    pred.pick = "HOME" if game_score > 0 else "AWAY"
    # Confidence: |game_score| normalized -- ~15 pts edge = 100% confidence
    pred.confidence = min(100.0, abs(game_score) / 15.0 * 100.0)

    # Sharp money agreement (for display, regardless of mode)
    if game.ml_home_public and game.ml_home_money:
        sharp_favors_home = game.ml_home_money > game.ml_home_public
        model_favors_home = game_score > 0
        pred.sharp_agrees = sharp_favors_home == model_favors_home

    return pred


# ──────────────────────────────────────────────────────────────
# Injury loading (live)
# ──────────────────────────────────────────────────────────────

def _load_current_injuries(*team_ids: int) -> Dict[int, float]:
    """Load current injuries from DB and compute play probabilities.

    Returns dict mapping player_id -> play_probability (0.0 to 1.0).
    """
    if not team_ids:
        return {}

    placeholders = ",".join("?" for _ in team_ids)
    rows = db.fetch_all(
        f"SELECT player_id, status, reason FROM injuries WHERE team_id IN ({placeholders})",
        tuple(team_ids),
    )
    injured = {}
    for r in rows:
        pid = r.get("player_id")
        if pid is None:
            continue
        status = r.get("status", "Out")
        # Simple probability mapping (no injury_intelligence in V2)
        status_lower = status.lower() if status else "out"
        if status_lower == "out":
            injured[pid] = 0.0
        elif status_lower in ("doubtful",):
            injured[pid] = 0.15
        elif status_lower in ("questionable",):
            injured[pid] = 0.5
        elif status_lower in ("probable", "available"):
            injured[pid] = 0.85
        else:
            injured[pid] = 0.0
    return injured


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
        pass

    row = db.fetch_one(
        "SELECT * FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season)
    )
    result = dict(row) if row else {}
    team_cache.set(team_id, cache_key, result)
    return result


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

    # Home court advantage
    home_court = get_home_court_advantage(home_team_id)

    # Team metrics
    game_season = _game_date_to_season(game_date)
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
    today_str = datetime.now().strftime("%Y-%m-%d")
    if game_date >= today_str:
        # Live fetch from Action Network
        try:
            from src.data.gamecast import get_actionnetwork_odds
            live_odds = get_actionnetwork_odds(home_abbr, away_abbr)
            if live_odds and live_odds.get("ml_home_public") is not None:
                ml_pub = live_odds.get("ml_home_public", 0) or 0
                ml_mon = live_odds.get("ml_home_money", 0) or 0
        except Exception:
            pass
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
    try:
        odds = db.fetch_one(
            "SELECT spread, home_moneyline, away_moneyline FROM game_odds "
            "WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
            (game_date, home_team_id, away_team_id))
        if odds and odds["spread"] is not None:
            vegas_spread = odds["spread"]
            vegas_home_ml = odds.get("home_moneyline") or 0
            vegas_away_ml = odds.get("away_moneyline") or 0
    except Exception:
        pass

    # ── V2.1 features ──
    from src.analytics.elo import get_team_elo
    from src.analytics.stats_engine import compute_travel, compute_momentum, compute_schedule_spots, compute_fg3_luck

    _home_travel = compute_travel(home_team_id, game_date, away_team_id, is_home=True)
    _away_travel = compute_travel(away_team_id, game_date, home_team_id, is_home=False)
    _home_momentum = compute_momentum(home_team_id, game_date)
    _away_momentum = compute_momentum(away_team_id, game_date)
    _home_sched = compute_schedule_spots(home_team_id, game_date, away_team_id)
    _away_sched = compute_schedule_spots(away_team_id, game_date, home_team_id)

    # On/Off impact
    _home_onoff = 0.0
    _away_onoff = 0.0
    for _side, _tid in [("home", home_team_id), ("away", away_team_id)]:
        _impact_rows = db.fetch_all(
            "SELECT pi.net_rating_diff, pi.on_court_minutes "
            "FROM player_impact pi "
            "JOIN players p ON pi.player_id = p.player_id "
            "WHERE pi.season = ? AND p.team_id = ? AND pi.on_court_minutes > 0",
            (game_season, _tid),
        )
        _total_impact = sum(
            r["net_rating_diff"] * min(r["on_court_minutes"], 30) / 30.0
            for r in _impact_rows
            if r.get("net_rating_diff") is not None
        ) if _impact_rows else 0.0
        if _side == "home":
            _home_onoff = _total_impact
        else:
            _away_onoff = _total_impact

    # Injury VORP lost
    _home_injury_vorp = 0.0
    _away_injury_vorp = 0.0
    if injured_players:
        for _pid, _play_prob in injured_players.items():
            _vorp_row = db.fetch_one(
                "SELECT pi.vorp, p.team_id FROM player_impact pi "
                "JOIN players p ON pi.player_id = p.player_id "
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
        pass

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
        # SRS
        home_srs=hm.get("srs", 0.0) or 0.0,
        away_srs=am.get("srs", 0.0) or 0.0,
        # On/Off
        home_onoff_impact=_home_onoff,
        away_onoff_impact=_away_onoff,
        # Pace diff
        pace_diff=abs(home_pace - away_pace),
        # 3PT luck
        home_fg3_luck=compute_fg3_luck(home_team_id, game_date),
        away_fg3_luck=compute_fg3_luck(away_team_id, game_date),
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

    # Dog pick detection (using Vegas lines from GameInput)
    vs = game.vegas_spread
    if vs != 0 and abs(pred.game_score) > 0.5:
        pred.is_dog_pick = (vs * pred.game_score > 0)
        pred.is_value_zone = 4.0 <= abs(vs) <= 12.0
        if pred.is_dog_pick and pred.vegas_home_ml != 0 and pred.vegas_away_ml != 0:
            dog_ml = pred.vegas_away_ml if vs < 0 else pred.vegas_home_ml
            if dog_ml < 0:
                pred.dog_payout = 1.0 + 100.0 / abs(dog_ml)
            else:
                pred.dog_payout = 1.0 + dog_ml / 100.0

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
_mem_ctx_game_count: Optional[int] = None
_mem_pc_lock = threading.Lock()


def _precompute_schema_version() -> str:
    """Hash of GameInput field names -- auto-invalidates when fields change."""
    names = tuple(f.name for f in dc_fields(GameInput))
    return hashlib.md5(str(names).encode()).hexdigest()[:12]


def _game_cache_key(home_team_id: int, away_team_id: int, game_date: str) -> str:
    """Unique key for a game."""
    return f"{home_team_id}_{away_team_id}_{game_date}"


def _load_pc_cache() -> Dict[str, GameInput]:
    """Load precompute cache from memory or disk. Returns empty dict on miss."""
    global _mem_pc_cache, _mem_pc_schema
    schema = _precompute_schema_version()

    with _mem_pc_lock:
        if _mem_pc_cache is not None and _mem_pc_schema == schema:
            return _mem_pc_cache

        try:
            if os.path.exists(_PRECOMPUTE_CACHE_FILE):
                with open(_PRECOMPUTE_CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                if data.get("schema") == schema:
                    _mem_pc_cache = data["games"]
                    _mem_pc_schema = schema
                    logger.info("Loaded precompute cache from disk (%d games)", len(_mem_pc_cache))
                    return _mem_pc_cache
                else:
                    logger.info("Precompute cache schema mismatch -- will rebuild")
        except Exception as e:
            logger.warning("Failed to load precompute cache: %s", e)

    return {}


def _save_pc_cache(cache: Dict[str, GameInput]):
    """Persist precompute cache to disk and update in-memory copy."""
    global _mem_pc_cache, _mem_pc_schema
    schema = _precompute_schema_version()
    os.makedirs(_PRECOMPUTE_CACHE_DIR, exist_ok=True)
    with _mem_pc_lock:
        try:
            with open(_PRECOMPUTE_CACHE_FILE, "wb") as f:
                pickle.dump({"schema": schema, "games": cache}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved precompute cache to disk (%d games)", len(cache))
        except Exception as e:
            logger.warning("Failed to save precompute cache: %s", e)
        _mem_pc_cache = cache
        _mem_pc_schema = schema


def invalidate_precompute_cache():
    """Clear all precompute caches (games + context, memory + disk)."""
    global _mem_pc_cache, _mem_pc_schema, _mem_ctx_cache, _mem_ctx_game_count
    _mem_pc_cache = None
    _mem_pc_schema = None
    _mem_ctx_cache = None
    _mem_ctx_game_count = None
    for path in (_PRECOMPUTE_CACHE_FILE, _CONTEXT_CACHE_FILE):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    logger.info("Invalidated all precompute caches")


# ──────────────────────────────────────────────────────────────
# Precompute context: historical rosters + inferred injuries
# ──────────────────────────────────────────────────────────────

def _load_ctx_cache(game_count: int) -> Optional[Dict[str, Any]]:
    """Load precompute context from memory or disk if game count matches."""
    global _mem_ctx_cache, _mem_ctx_game_count
    if _mem_ctx_cache is not None and _mem_ctx_game_count == game_count:
        return _mem_ctx_cache
    try:
        if os.path.exists(_CONTEXT_CACHE_FILE):
            with open(_CONTEXT_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("game_count") == game_count:
                _mem_ctx_cache = data["ctx"]
                _mem_ctx_game_count = game_count
                logger.info("Loaded precompute context from disk (%d games)", game_count)
                return _mem_ctx_cache
    except Exception as e:
        logger.warning("Failed to load context cache: %s", e)
    return None


def _save_ctx_cache(ctx: Dict[str, Any], game_count: int):
    """Persist precompute context to disk."""
    global _mem_ctx_cache, _mem_ctx_game_count
    os.makedirs(_PRECOMPUTE_CACHE_DIR, exist_ok=True)
    try:
        with open(_CONTEXT_CACHE_FILE, "wb") as f:
            pickle.dump({"game_count": game_count, "ctx": ctx}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved precompute context to disk (%d games)", game_count)
    except Exception as e:
        logger.warning("Failed to save context cache: %s", e)
    _mem_ctx_cache = ctx
    _mem_ctx_game_count = game_count


def _build_precompute_context(games: List[Dict], force: bool = False) -> Dict[str, Any]:
    """Build lookup tables for historical roster + injury inference.

    One bulk SQL query determines which team each player was on for every
    game. For each game we can infer injuries by comparing the recent active
    roster vs who actually played. Cached to disk.
    """
    game_count = len(games)

    if not force:
        cached = _load_ctx_cache(game_count)
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

    # All player game appearances
    rows = db.fetch_all("""
        SELECT player_id, game_id, game_date, is_home,
               points, minutes
        FROM player_stats
        ORDER BY game_date
    """)

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
    _save_ctx_cache(result, game_count)
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
_actual_results_game_count: Optional[int] = None


def get_actual_game_results(team_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Aggregate player_stats by (game_id, is_home) to reconstruct game scores.

    Cached at module level -- only rebuilt when game count changes.
    """
    global _actual_results_cache, _actual_results_game_count

    current_count_row = db.fetch_one("SELECT COUNT(*) as c FROM player_stats")
    current_count = current_count_row["c"] if current_count_row else 0

    if _actual_results_cache is not None and current_count == _actual_results_game_count:
        if team_id:
            return [g for g in _actual_results_cache
                    if g["home_team_id"] == team_id or g["away_team_id"] == team_id]
        return list(_actual_results_cache)

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
        if spread > 0.5:
            winner = "HOME"
        elif spread < -0.5:
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
    _actual_results_game_count = current_count

    if team_id:
        return [g for g in results
                if g["home_team_id"] == team_id or g["away_team_id"] == team_id]
    return list(results)


def invalidate_results_cache():
    """Clear cached game results (call when new data is synced)."""
    global _actual_results_cache, _actual_results_game_count
    _actual_results_cache = None
    _actual_results_game_count = None


# ──────────────────────────────────────────────────────────────
# precompute_all_games() -- builds GameInput for all historical games
# ──────────────────────────────────────────────────────────────

def precompute_all_games(callback=None, force=False) -> List[GameInput]:
    """Build a list of GameInput objects for optimization/backtest.

    Uses a persistent disk + memory cache so that already-computed historical
    games are never reprocessed. Only truly new games go through the expensive
    per-game projection pipeline.

    Pass ``force=True`` to discard the cache and recompute everything.
    """
    from src.database.db import thread_local_db

    # Load cache
    cache = {} if force else _load_pc_cache()

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
    ctx = _build_precompute_context(all_games)

    cfg = get_config()
    max_workers = cfg.get("worker_threads", 4)
    _pc_lock = threading.Lock()
    completed_count = [0]

    def _precompute_one(g):
        """Process one game with a thread-local DB."""
        with thread_local_db():
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
            from src.analytics.elo import get_team_elo
            from src.analytics.stats_engine import compute_travel, compute_momentum, compute_schedule_spots, compute_fg3_luck

            _home_travel = compute_travel(htid, gdate, atid, is_home=True)
            _away_travel = compute_travel(atid, gdate, htid, is_home=False)
            _home_momentum = compute_momentum(htid, gdate, season=game_season)
            _away_momentum = compute_momentum(atid, gdate, season=game_season)
            _home_sched = compute_schedule_spots(htid, gdate, atid, season=game_season)
            _away_sched = compute_schedule_spots(atid, gdate, htid, season=game_season)

            # On/Off impact
            _home_onoff = 0.0
            _away_onoff = 0.0
            for _side, _tid, _target in [("home", htid, "_home_onoff"), ("away", atid, "_away_onoff")]:
                from src.database import db as _db
                _impact_rows = _db.fetch_all(
                    "SELECT pi.net_rating_diff, pi.on_court_minutes "
                    "FROM player_impact pi "
                    "JOIN players p ON pi.player_id = p.player_id "
                    "WHERE pi.season = ? AND p.team_id = ? AND pi.on_court_minutes > 0",
                    (game_season, _tid),
                )
                _total_impact = sum(
                    r["net_rating_diff"] * min(r["on_court_minutes"], 30) / 30.0
                    for r in _impact_rows
                    if r.get("net_rating_diff") is not None
                ) if _impact_rows else 0.0
                if _side == "home":
                    _home_onoff = _total_impact
                else:
                    _away_onoff = _total_impact

            # Spread sharp edge
            _spread_sharp = 0.0
            from src.database import db as _db
            _odds_row = _db.fetch_all(
                "SELECT spread_home_money, spread_home_public FROM game_odds "
                "WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?",
                (gdate, htid, atid),
            )
            if _odds_row:
                _sm = _odds_row[0].get("spread_home_money", 0) or 0
                _sp = _odds_row[0].get("spread_home_public", 0) or 0
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
                # Elo
                home_elo=get_team_elo(htid, gdate, game_season),
                away_elo=get_team_elo(atid, gdate, game_season),
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
                # SRS
                home_srs=hm.get("srs", 0.0) or 0.0,
                away_srs=am.get("srs", 0.0) or 0.0,
                # On/Off
                home_onoff_impact=_home_onoff,
                away_onoff_impact=_away_onoff,
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
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                if callback and c % 25 == 0:
                    callback(f"Precomputed {c}/{len(new_games)} games")

    # Merge new results into cache and persist
    for gi in new_results:
        key = _game_cache_key(gi.home_team_id, gi.away_team_id, gi.game_date)
        cache[key] = gi

    if new_results:
        _save_pc_cache(cache)

    # Return only valid games
    result = [cache[k] for k in valid_keys if k in cache]
    result.sort(key=lambda gi: gi.game_date)

    if callback:
        callback(f"Precomputed {len(result)} games total ({len(new_results)} new, {cached_count} cached)")
    return result
