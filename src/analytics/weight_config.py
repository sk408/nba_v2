"""WeightConfig dataclass — fundamentals-only prediction parameters.

V2 clean rebuild: no ESPN blend, no ML ensemble, no Elo, no opening spread,
no pace_mult, no oreb_mult. Only basketball fundamentals + sharp money toggle.
"""

import json
import logging
import os
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from typing import Optional, Dict, List

from src.database import db

logger = logging.getLogger(__name__)

_SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "snapshots")

# Four-factors contribution normalization constant.
# This replaces the old tunable four_factors_scale pathway to reduce
# identifiability issues between scale and per-factor weights.
FOUR_FACTORS_FIXED_SCALE = 100.0


@dataclass
class WeightConfig:
    # Defensive adjustment
    def_factor_dampening: float = 6.90

    # Core edges
    turnover_margin_mult: float = 2.00
    rebound_diff_mult: float = 0.50
    rating_matchup_mult: float = 4.06

    # Four Factors
    # Legacy compatibility knob (no longer tuned; fixed normalization is used).
    four_factors_scale: float = 1.0
    ff_efg_weight: float = 7.36
    ff_tov_weight: float = 3.40
    ff_oreb_weight: float = 1.97
    ff_fta_weight: float = 0.05

    # Opponent Four Factors (defensive matchup)
    opp_ff_efg_weight: float = 1.0
    opp_ff_tov_weight: float = 4.50
    opp_ff_oreb_weight: float = 1.0
    opp_ff_fta_weight: float = 0.05

    # Clutch
    clutch_scale: float = 0.13
    clutch_cap: float = 3.5
    clutch_threshold: float = 6.0

    # Hustle
    hustle_effort_mult: float = 0.75
    hustle_contested_wt: float = 0.04

    # Pace / Total
    steals_threshold: float = 14.0
    steals_penalty: float = 0.15
    blocks_threshold: float = 10.0
    blocks_penalty: float = 0.12
    hustle_defl_baseline: float = 30.0
    hustle_defl_penalty: float = 0.1

    # Fatigue
    fatigue_total_mult: float = 0.3
    fatigue_b2b: float = 7.68
    fatigue_3in4: float = 4.99
    fatigue_4in6: float = 2.67
    fatigue_same_day: float = 3.0
    fatigue_rest_bonus: float = 1.85

    # Rest advantage
    rest_advantage_mult: float = 0.42

    # Altitude B2B
    altitude_b2b_penalty: float = 0.77

    # Sharp Money (ML only) — toggle layer, not in fundamentals-only mode
    sharp_ml_weight: float = 1.5

    # ── V2.1 weights ──
    elo_diff_mult: float = 1.0
    travel_dist_mult: float = 0.5
    timezone_crossing_mult: float = 1.0
    momentum_streak_mult: float = 0.3
    mov_trend_mult: float = 0.2
    injury_vorp_mult: float = 1.0
    ref_fouls_mult: float = 0.3
    ref_home_bias_mult: float = 0.3
    sharp_spread_weight: float = 1.0
    lookahead_penalty: float = 0.5
    letdown_penalty: float = 0.5
    srs_diff_mult: float = 0.5
    pythag_diff_mult: float = 2.0
    onoff_impact_mult: float = 0.5
    onoff_reliability_lambda: float = 0.35
    road_trip_game_mult: float = 0.25
    season_progress_mult: float = 0.8
    roster_shock_mult: float = 1.2
    tank_live_mult: float = 1.0
    tank_oracle_mult: float = 0.4
    cum_travel_7d_mult: float = 0.75
    pace_mismatch_mult: float = 0.2
    fg3_luck_mult: float = 15.0
    process_edge_mult: float = 0.3

    # Total floor/ceiling
    total_min: float = 140.0
    total_max: float = 280.0

    def blend(self, other: "WeightConfig",
              self_games: int = 0, other_games: int = 0) -> "WeightConfig":
        """Blend two WeightConfigs weighted by games_analyzed.

        If games counts are provided, the team with more analysed games
        gets proportionally higher influence. Falls back to simple average
        when no counts are provided (backward-compatible).
        """
        total = self_games + other_games
        if total > 0:
            w_self = self_games / total
            w_other = other_games / total
        else:
            w_self = 0.5
            w_other = 0.5

        new_data = {}
        for f in fields(self):
            v1 = getattr(self, f.name)
            v2 = getattr(other, f.name)
            new_data[f.name] = v1 * w_self + v2 * w_other
        return WeightConfig(**new_data)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "WeightConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ──────────────────────────────────────────────────────────────
# Optimizer Ranges — fundamentals parameters only
# ──────────────────────────────────────────────────────────────

OPTIMIZER_RANGES = {
    "def_factor_dampening": (0.1, 10.0),
    "turnover_margin_mult": (0.0, 10.0),
    "rebound_diff_mult": (0.0, 4.0),
    "rating_matchup_mult": (0.0, 5.0),
    "ff_efg_weight": (0.0, 15.0),
    "ff_tov_weight": (0.0, 15.0),
    "ff_oreb_weight": (0.0, 15.0),
    "ff_fta_weight": (0.0, 15.0),
    "opp_ff_efg_weight": (0.0, 15.0),
    "opp_ff_tov_weight": (0.0, 15.0),
    "opp_ff_oreb_weight": (0.0, 15.0),
    "opp_ff_fta_weight": (0.0, 15.0),
    "clutch_scale": (0.0, 2.0),
    "hustle_effort_mult": (0.0, 2.0),
    "hustle_contested_wt": (0.0, 1.5),
    "steals_penalty": (0.0, 4.0),
    "blocks_penalty": (0.0, 4.0),
    "rest_advantage_mult": (0.0, 3.0),
    "altitude_b2b_penalty": (0.0, 10.0),
    "fatigue_b2b": (0.0, 10.0),
    "fatigue_3in4": (0.0, 10.0),
    "fatigue_4in6": (0.0, 5.0),
    "fatigue_same_day": (0.0, 6.0),
    "fatigue_rest_bonus": (0.0, 3.0),
    "fatigue_total_mult": (0.0, 2.0),
    "elo_diff_mult": (0.0, 5.0),
    "travel_dist_mult": (0.0, 3.0),
    "timezone_crossing_mult": (0.0, 5.0),
    "momentum_streak_mult": (0.0, 3.0),
    "mov_trend_mult": (0.0, 2.0),
    "injury_vorp_mult": (0.0, 5.0),
    "ref_fouls_mult": (0.0, 3.0),
    "ref_home_bias_mult": (0.0, 3.0),
    "sharp_spread_weight": (0.0, 15.0),
    "lookahead_penalty": (0.0, 3.0),
    "letdown_penalty": (0.0, 3.0),
    "srs_diff_mult": (0.0, 3.0),
    "pythag_diff_mult": (0.0, 8.0),
    "onoff_impact_mult": (0.0, 3.0),
    "onoff_reliability_lambda": (0.0, 2.5),
    "road_trip_game_mult": (0.0, 2.0),
    "season_progress_mult": (0.0, 3.0),
    "roster_shock_mult": (0.0, 4.0),
    "tank_live_mult": (0.0, 5.0),
    "tank_oracle_mult": (0.0, 5.0),
    "cum_travel_7d_mult": (0.0, 5.0),
    "pace_mismatch_mult": (0.0, 2.0),
    "fg3_luck_mult": (0.0, 50.0),
    "process_edge_mult": (0.0, 2.0),
}

# Sharp money range — separate toggle layer
SHARP_RANGES = {
    "sharp_ml_weight": (0.0, 15.0),
    "sharp_spread_weight": (0.0, 15.0),
}

# Combined ranges for sharp-enabled optimization
SHARP_MODE_RANGES = {**OPTIMIZER_RANGES, **SHARP_RANGES}

# Coordinate Descent ranges — wider than OPTIMIZER_RANGES for broader exploration.
# CD grid-searches every value so it can safely explore beyond Optuna's TPE bounds.
CD_RANGES = {
    "def_factor_dampening": (0.05, 12.0),
    "turnover_margin_mult": (0.0, 12.0),
    "rebound_diff_mult": (0.0, 5.0),
    "rating_matchup_mult": (0.0, 6.0),
    "ff_efg_weight": (0.0, 18.0),
    "ff_tov_weight": (0.0, 18.0),
    "ff_oreb_weight": (0.0, 18.0),
    "ff_fta_weight": (0.0, 18.0),
    "opp_ff_efg_weight": (0.0, 18.0),
    "opp_ff_tov_weight": (0.0, 18.0),
    "opp_ff_oreb_weight": (0.0, 18.0),
    "opp_ff_fta_weight": (0.0, 18.0),
    "clutch_scale": (0.0, 3.0),
    "hustle_effort_mult": (0.0, 3.0),
    "hustle_contested_wt": (0.0, 2.0),
    "steals_penalty": (0.0, 5.0),
    "blocks_penalty": (0.0, 5.0),
    "rest_advantage_mult": (0.0, 5.0),
    "altitude_b2b_penalty": (0.0, 12.0),
    "fatigue_b2b": (0.0, 12.0),
    "fatigue_3in4": (0.0, 12.0),
    "fatigue_4in6": (0.0, 6.0),
    "fatigue_same_day": (0.0, 8.0),
    "fatigue_rest_bonus": (0.0, 5.0),
    "fatigue_total_mult": (0.0, 3.0),
    "elo_diff_mult": (0.0, 7.0),
    "travel_dist_mult": (0.0, 5.0),
    "timezone_crossing_mult": (0.0, 7.0),
    "momentum_streak_mult": (0.0, 5.0),
    "mov_trend_mult": (0.0, 3.0),
    "injury_vorp_mult": (0.0, 7.0),
    "ref_fouls_mult": (0.0, 5.0),
    "ref_home_bias_mult": (0.0, 5.0),
    "sharp_spread_weight": (0.0, 20.0),
    "lookahead_penalty": (0.0, 5.0),
    "letdown_penalty": (0.0, 5.0),
    "srs_diff_mult": (0.0, 5.0),
    "pythag_diff_mult": (0.0, 10.0),
    "onoff_impact_mult": (0.0, 5.0),
    "onoff_reliability_lambda": (0.0, 4.0),
    "road_trip_game_mult": (0.0, 3.0),
    "season_progress_mult": (0.0, 5.0),
    "roster_shock_mult": (0.0, 6.0),
    "tank_live_mult": (0.0, 7.0),
    "tank_oracle_mult": (0.0, 7.0),
    "cum_travel_7d_mult": (0.0, 7.0),
    "pace_mismatch_mult": (0.0, 3.0),
    "fg3_luck_mult": (0.0, 75.0),
    "process_edge_mult": (0.0, 3.0),
}

# CD ranges with sharp money parameter included
CD_SHARP_RANGES = {**CD_RANGES, "sharp_ml_weight": (0.0, 20.0), "sharp_spread_weight": (0.0, 20.0)}


# ──────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────

_cached_global: Optional[WeightConfig] = None


def get_weight_config() -> WeightConfig:
    """Lazy-loaded, cached singleton for global weights."""
    global _cached_global
    if _cached_global is not None:
        return _cached_global
    rows = db.fetch_all("SELECT key, value FROM model_weights")
    if rows:
        d = {r["key"]: r["value"] for r in rows}
        _cached_global = WeightConfig.from_dict(d)
    else:
        _cached_global = WeightConfig()
    return _cached_global


def _enforce_ranges(w: WeightConfig) -> WeightConfig:
    """Clamp tunable parameters to active optimizer bounds.

    Prevents degenerate configurations (negative weights, inverted signs)
    from being persisted. When optimizer wide ranges are enabled, use the
    corresponding wider CD bounds so persisted weights match what Optuna scored.
    """
    use_wide_ranges = False
    try:
        from src.config import get as get_setting

        use_wide_ranges = bool(get_setting("optimizer_use_wide_ranges", False))
    except Exception:
        logger.debug("Failed reading optimizer_use_wide_ranges; using defaults", exc_info=True)

    all_ranges = (
        CD_SHARP_RANGES if use_wide_ranges else {**OPTIMIZER_RANGES, **SHARP_RANGES}
    )
    d = w.to_dict()
    for k, (lo, hi) in all_ranges.items():
        if k in d:
            old = d[k]
            d[k] = max(lo, min(hi, d[k]))
            if d[k] != old:
                logger.warning(
                    "Weight %s clamped from %.4f to %.4f (range %.4f-%.4f)",
                    k, old, d[k], lo, hi,
                )
    return WeightConfig.from_dict(d)


def save_weight_config(w: WeightConfig):
    """Save global weights to DB and refresh cache.

    All tunable parameters are clamped to valid ranges before saving
    to prevent degenerate configurations.
    """
    global _cached_global
    w = _enforce_ranges(w)
    for k, v in w.to_dict().items():
        db.execute(
            "INSERT INTO model_weights (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, v)
        )
    _cached_global = w
    try:
        from src.analytics.cache_registry import invalidate_for_event
        invalidate_for_event("post_weight_save")
    except Exception:
        logger.debug("post_weight_save invalidation failed", exc_info=True)


def invalidate_weight_cache():
    """Force reload of cached weights on next access."""
    global _cached_global
    _cached_global = None


def clear_all_weights():
    """Clear global and per-team weights, reset cache and freshness."""
    global _cached_global
    db.execute("DELETE FROM model_weights")
    db.execute("DELETE FROM team_weight_overrides")
    _cached_global = None
    # Invalidate freshness so the pipeline re-runs weight-related steps
    for step in ("weight_optimize", "team_refine", "residual_cal"):
        db.execute("DELETE FROM sync_meta WHERE step_name = ?", (step,))
    try:
        from src.analytics.cache_registry import invalidate_for_event
        invalidate_for_event("post_weight_save")
    except Exception:
        logger.debug("post_weight_save invalidation failed", exc_info=True)


# ──────────────────────────────────────────────────────────────
# Pipeline Snapshots
# ──────────────────────────────────────────────────────────────

def save_snapshot(name: str, notes: str = "", metrics: Optional[Dict] = None) -> str:
    """Save the current weight config + optimizer ranges to a named snapshot.

    Returns the path to the snapshot file.
    """
    os.makedirs(_SNAPSHOTS_DIR, exist_ok=True)
    w = get_weight_config()

    snapshot = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "notes": notes,
        "weights": w.to_dict(),
        "optimizer_ranges": {k: list(v) for k, v in OPTIMIZER_RANGES.items()},
        "sharp_ranges": {k: list(v) for k, v in SHARP_RANGES.items()},
        "metrics": metrics or {},
    }

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_name}.json"
    path = os.path.join(_SNAPSHOTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return path


def load_snapshot(path: str) -> Dict:
    """Load a snapshot from a JSON file. Returns the parsed dict."""
    with open(path) as f:
        return json.load(f)


def restore_snapshot(path: str):
    """Restore weights from a snapshot file."""
    snap = load_snapshot(path)

    # Restore global weights
    w = WeightConfig.from_dict(snap["weights"])
    save_weight_config(w)

    invalidate_weight_cache()


def list_snapshots() -> List[Dict]:
    """List all available snapshots, newest first."""
    if not os.path.isdir(_SNAPSHOTS_DIR):
        return []
    snaps = []
    for f in sorted(os.listdir(_SNAPSHOTS_DIR), reverse=True):
        if f.endswith(".json"):
            try:
                path = os.path.join(_SNAPSHOTS_DIR, f)
                with open(path) as fh:
                    data = json.load(fh)
                snaps.append({
                    "filename": f,
                    "path": path,
                    "name": data.get("name", f),
                    "created_at": data.get("created_at", ""),
                    "notes": data.get("notes", ""),
                    "metrics": data.get("metrics", {}),
                })
            except Exception:
                logger.debug("snapshot parse failed for %s", f, exc_info=True)
                continue
    return snaps
