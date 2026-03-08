"""Player splits, aggregate_projection(), 240-min budget, fatigue detection."""

import logging
import os
import pickle
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import numpy as np

from src.database import db
from src.analytics.memory_store import get_store

logger = logging.getLogger(__name__)

# ──────────────────── Caches ────────────────────
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")

# player_splits cache: (player_id, opponent_team_id, is_home, as_of_date) → result dict
_splits_cache: Dict[tuple, Dict[str, float]] = {}
_splits_cache_lock = threading.Lock()

# get_games_missed_streak cache: (player_id, as_of_date) → int
_streak_cache: Dict[tuple, int] = {}
_streak_cache_lock = threading.Lock()

# fatigue cache: (team_id, game_date) → result dict
_fatigue_cache: Dict[tuple, Dict[str, Any]] = {}
_fatigue_cache_lock = threading.Lock()

# team abbreviation singleton
_team_abbr_cache: Optional[Dict[int, str]] = None
_team_name_cache: Optional[Dict[int, str]] = None
_team_cache_lock = threading.Lock()

# home court advantage cache: team_id → float
_hca_cache: Dict[int, float] = {}
_hca_cache_lock = threading.Lock()


def invalidate_stats_caches():
    """Clear all stats engine caches. Call after data sync or optimizer run."""
    global _splits_cache, _streak_cache, _fatigue_cache, _hca_cache
    global _team_abbr_cache, _team_name_cache
    with _splits_cache_lock:
        _splits_cache.clear()
    with _streak_cache_lock:
        _streak_cache.clear()
    with _fatigue_cache_lock:
        _fatigue_cache.clear()
    with _hca_cache_lock:
        _hca_cache.clear()
    with _team_cache_lock:
        _team_abbr_cache = None
        _team_name_cache = None
    logger.info("All stats engine caches invalidated")


def get_team_abbreviations() -> Dict[int, str]:
    """Return {team_id: abbreviation} singleton (loaded once, never changes)."""
    global _team_abbr_cache
    with _team_cache_lock:
        if _team_abbr_cache is not None:
            return _team_abbr_cache
        store = get_store()
        if store.teams is not None and not store.teams.empty:
            _team_abbr_cache = dict(zip(store.teams["team_id"], store.teams["abbreviation"]))
        else:
            rows = db.fetch_all("SELECT team_id, abbreviation FROM teams")
            _team_abbr_cache = {r["team_id"]: r["abbreviation"] for r in rows}
        return _team_abbr_cache


def get_team_names() -> Dict[int, str]:
    """Return {team_id: name} singleton (loaded once, never changes)."""
    global _team_name_cache
    with _team_cache_lock:
        if _team_name_cache is not None:
            return _team_name_cache
        store = get_store()
        if store.teams is not None and not store.teams.empty:
            _team_name_cache = dict(zip(store.teams["team_id"], store.teams["name"]))
        else:
            rows = db.fetch_all("SELECT team_id, name FROM teams")
            _team_name_cache = {r["team_id"]: r["name"] for r in rows}
        return _team_name_cache

# ──────────────────── Constants ────────────────────
TEAM_MINUTES_PER_GAME = 240.0
_DECAY = 0.9
_HOME_COURT_FALLBACK = 2.0              # modern NBA HCA is ~2-3 pts
_HOME_COURT_CLAMP = (-1.0, 5.0)         # reduced from (-2,8) to fix +76 home bias
_PACE_FALLBACK = 98.0
_RATING_FALLBACK = 110.0
_LEAGUE_AVG_PPG = 112.0
_HIGH_SCORER_THRESHOLD = 15.0
_ACTIVE_HIGH_SCORER_THRESHOLD = 12.0
_PLAYMAKER_THRESHOLD = 6.0
_PLAYMAKER_PENALTY_MULT = 0.5
_REBOUNDER_THRESHOLD = 8.0
_REBOUNDER_PENALTY_MULT = 0.2
_FT_VOLUME_THRESHOLD = 2.0
_DEFAULT_REPLACEMENT_FT_PCT = 78.0
_RETURN_DISCOUNT_PER_GAME = 0.03
_RETURN_DISCOUNT_FLOOR = 0.85
_ROSTER_CHANGE_HIGH_IMPACT_MPG = 20.0

STAT_COLS = [
    "points", "rebounds", "assists", "minutes", "steals", "blocks",
    "turnovers", "oreb", "dreb", "fg_made", "fg_attempted",
    "fg3_made", "fg3_attempted", "ft_made", "ft_attempted",
    "plus_minus", "personal_fouls",
]


def _exponential_decay_weights(n: int, decay: float = _DECAY) -> np.ndarray:
    """Generate normalized exponential decay weights. Row 0 = most recent = weight 1.0."""
    if n <= 0:
        return np.array([])
    raw = np.array([decay ** i for i in range(n)])
    total = raw.sum()
    return raw / total if total > 0 else raw


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean, handling edge cases."""
    if len(values) == 0 or len(weights) == 0:
        return 0.0
    return float(np.dot(values, weights[:len(values)]))


def player_splits(player_id: int, opponent_team_id: int, is_home: int,
                  recent_games: int = 20, as_of_date: Optional[str] = None) -> Dict[str, float]:
    """Compute blended stat projection for a player.

    Uses a larger window (20 games) with exponential decay so recent form
    dominates while still capturing enough signal from location/opponent splits.

    Results are cached in memory — same args always return the same result.
    Returns dict of stat averages.
    """
    # In-memory cache lookup
    cache_key = (player_id, opponent_team_id, is_home, recent_games, as_of_date)
    with _splits_cache_lock:
        if cache_key in _splits_cache:
            return _splits_cache[cache_key]

    _empty = {c: 0.0 for c in STAT_COLS}

    store = get_store()
    if store.player_stats is not None and not store.player_stats.empty:
        ps = store.player_stats[store.player_stats["player_id"] == player_id].copy()
    else:
        rows = db.fetch_all(
            "SELECT * FROM player_stats WHERE player_id = ? ORDER BY game_date DESC",
            (player_id,)
        )
        if not rows:
            with _splits_cache_lock:
                _splits_cache[cache_key] = _empty
            return _empty
        import pandas as pd
        ps = pd.DataFrame([dict(r) for r in rows])

    if ps.empty:
        with _splits_cache_lock:
            _splits_cache[cache_key] = _empty
        return _empty

    # as_of_date filter (prevent lookahead)
    if as_of_date:
        ps = ps[ps["game_date"] < as_of_date]
    if ps.empty:
        with _splits_cache_lock:
            _splits_cache[cache_key] = _empty
        return _empty

    ps = ps.sort_values("game_date", ascending=False)

    # Base: overall recent
    base = ps.head(recent_games)

    # Location split
    loc = ps[ps["is_home"] == is_home].head(recent_games)

    # vs-opponent split
    opp = ps[ps["opponent_team_id"] == opponent_team_id].head(recent_games)

    # Blend weights — graduated for both opponent and location splits
    n_opp = len(opp)
    if n_opp >= 3:
        w_opp = 0.25
    elif n_opp == 2:
        w_opp = 0.15
    elif n_opp == 1:
        w_opp = 0.10
    else:
        w_opp = 0.0

    n_loc = len(loc)
    if n_loc >= 5:
        w_loc = 0.25
    elif n_loc >= 3:
        w_loc = 0.20
    elif n_loc >= 2:
        w_loc = 0.15
    elif n_loc == 1:
        w_loc = 0.10
    else:
        w_loc = 0.0

    w_base = 1.0 - w_loc - w_opp

    result = {}
    for col in STAT_COLS:
        if col not in ps.columns:
            result[col] = 0.0
            continue

        base_vals = base[col].values.astype(float)
        base_weights = _exponential_decay_weights(len(base_vals))

        loc_vals = loc[col].values.astype(float) if w_loc > 0 else np.array([])
        loc_weights = _exponential_decay_weights(len(loc_vals)) if w_loc > 0 else np.array([])

        opp_vals = opp[col].values.astype(float) if w_opp > 0 else np.array([])
        opp_weights = _exponential_decay_weights(len(opp_vals)) if w_opp > 0 else np.array([])

        val = (w_base * _weighted_mean(base_vals, base_weights) +
               w_loc * _weighted_mean(loc_vals, loc_weights) +
               w_opp * _weighted_mean(opp_vals, opp_weights))
        result[col] = val

    # Store in cache
    with _splits_cache_lock:
        _splits_cache[cache_key] = result
    return result


def get_games_missed_streak(player_id: int, as_of_date: Optional[str] = None) -> int:
    """Count consecutive games missed before as_of_date (or now)."""
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")

    # In-memory cache lookup
    streak_key = (player_id, as_of_date)
    with _streak_cache_lock:
        if streak_key in _streak_cache:
            return _streak_cache[streak_key]

    rows = db.fetch_all(
        """SELECT game_date FROM injury_history
           WHERE player_id = ? AND was_out = 1 AND game_date <= ?
           ORDER BY game_date DESC LIMIT 20""",
        (player_id, as_of_date)
    )
    if not rows:
        with _streak_cache_lock:
            _streak_cache[streak_key] = 0
        return 0

    last_played = db.fetch_one(
        """SELECT MAX(game_date) as d FROM player_stats
           WHERE player_id = ? AND game_date <= ?""",
        (player_id, as_of_date)
    )
    if not last_played or not last_played["d"]:
        result = len(rows)
        with _streak_cache_lock:
            _streak_cache[streak_key] = result
        return result

    # Count missed games after last played
    count = 0
    for r in rows:
        if r["game_date"] is None or last_played["d"] is None:
            continue
        if r["game_date"] > last_played["d"]:
            count += 1
        else:
            break

    with _streak_cache_lock:
        _streak_cache[streak_key] = count
    return count


def aggregate_projection(team_id: int, opponent_team_id: int, is_home: int,
                         as_of_date: Optional[str] = None,
                         player_weights: Optional[Dict[int, float]] = None,
                         injured_players: Optional[Dict[int, float]] = None,
                         roster: Optional[List[Dict]] = None) -> Dict[str, float]:
    """Aggregate team projection from player splits with 240-min budget.

    Args:
        injured_players: {player_id: play_probability} for injured players
        roster: Optional list of {player_id, name, position} dicts.
                When provided (historical precompute), uses this instead of
                querying the current players table — avoids traded-player
                lookahead bias.
    """
    if player_weights is None:
        player_weights = {}
    if injured_players is None:
        injured_players = {}

    # Get active players — historical roster or current
    if roster is not None:
        rows = roster
    else:
        rows = db.fetch_all(
            "SELECT player_id, name, position FROM players WHERE team_id = ?",
            (team_id,)
        )

    totals = {c: 0.0 for c in STAT_COLS}
    total_projected_minutes = 0.0
    active_players = []

    for p in rows:
        pid = p["player_id"]

        # Skip players who are definitely out
        play_prob = injured_players.get(pid, 1.0)
        if play_prob < 0.3:
            continue

        splits = player_splits(pid, opponent_team_id, is_home, recent_games=20,
                               as_of_date=as_of_date)
        if splits["minutes"] < 1.0:
            if play_prob >= 0.3:
                # Try full-season averages for returning players
                splits = player_splits(pid, opponent_team_id, is_home,
                                       recent_games=9999, as_of_date=as_of_date)
                if splits["minutes"] < 1.0:
                    continue
                # Return-from-long-absence discount
                play_prob *= 0.5
            else:
                continue

        weight = player_weights.get(pid, 1.0) * play_prob

        # Return-from-injury discount
        missed = get_games_missed_streak(pid, as_of_date)
        if missed > 0:
            discount = max(_RETURN_DISCOUNT_FLOOR, 1.0 - _RETURN_DISCOUNT_PER_GAME * missed)
            weight *= discount

        for k in STAT_COLS:
            totals[k] += splits[k] * weight

        total_projected_minutes += splits["minutes"] * weight
        active_players.append({
            "player_id": pid,
            "name": p["name"],
            "position": p["position"],
            "splits": splits,
            "weight": weight,
        })

    # 240-minute budget normalization (scale down OR up)
    scaling_cols = ["points", "rebounds", "assists", "minutes", "steals",
                    "blocks", "turnovers", "oreb", "dreb",
                    "fg_made", "fg_attempted", "fg3_made", "fg3_attempted",
                    "ft_made", "ft_attempted"]
    if total_projected_minutes > TEAM_MINUTES_PER_GAME:
        scale = TEAM_MINUTES_PER_GAME / total_projected_minutes
        for c in scaling_cols:
            totals[c] *= scale
    elif 0 < total_projected_minutes < TEAM_MINUTES_PER_GAME:
        # Short-handed teams: remaining players absorb extra minutes but
        # replacement-level production is lower — use a dampened scale-up.
        # Full scale would assume bench players are equally productive;
        # 70% accounts for drop-off to replacement level.
        gap_ratio = TEAM_MINUTES_PER_GAME / total_projected_minutes
        dampened_scale = 1.0 + (gap_ratio - 1.0) * 0.70
        for c in scaling_cols:
            totals[c] *= dampened_scale

    totals["_active_players"] = active_players
    totals["_total_projected_minutes"] = min(total_projected_minutes, TEAM_MINUTES_PER_GAME)

    # Fallback: if projection produced zero points (e.g. first game of season
    # with no prior data), use league-average estimate so downstream calcs get
    # a reasonable baseline instead of 0.
    if totals["points"] < 1.0:
        totals["points"] = _LEAGUE_AVG_PPG
        logger.debug("aggregate_projection for team %d produced 0 pts — using league avg fallback", team_id)

    return totals


def get_home_court_advantage(team_id: int, season: Optional[str] = None) -> float:
    """Calculate home court advantage from home/road point differentials.

    NOTE: home_pts / road_pts are stored as **per-game** averages (from
    LeagueDashTeamStats with per_mode=PerGame), so we must NOT divide by GP.
    Results cached in memory per (team_id, season).
    """
    from src.config import get_season
    if season is None:
        season = get_season()

    cache_key = (team_id, season)
    with _hca_cache_lock:
        if cache_key in _hca_cache:
            return _hca_cache[cache_key]
    row = db.fetch_one(
        "SELECT home_pts, road_pts, home_gp, road_gp FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season)
    )
    if not row or not row["home_gp"] or not row["road_gp"]:
        with _hca_cache_lock:
            _hca_cache[cache_key] = _HOME_COURT_FALLBACK
        return _HOME_COURT_FALLBACK

    home_ppg = row["home_pts"]   # already per-game
    road_ppg = row["road_pts"]   # already per-game
    hca = home_ppg - road_ppg
    result = max(_HOME_COURT_CLAMP[0], min(_HOME_COURT_CLAMP[1], hca))
    with _hca_cache_lock:
        _hca_cache[cache_key] = result
    return result


def compute_fatigue(team_id: int, game_date: str, w=None) -> Dict[str, Any]:
    """Detect back-to-back, 3-in-4, 4-in-6 fatigue situations.

    Args:
        w: Optional WeightConfig to read b2b/3in4/4in6 penalties from.
           Falls back to defaults if None.
    The schedule-based detection (dates, b2b, 3in4, 4in6) is cached per
    (team_id, game_date). Penalty values that depend on weights are applied
    on top of the cached schedule data.
    """
    _default_result = {"penalty": 0.0, "rest_days": 3, "b2b": False, "three_in_four": False, "four_in_six": False}
    try:
        gd = datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return _default_result

    # Cache the schedule detection (weight-independent)
    fatigue_key = (team_id, game_date)
    with _fatigue_cache_lock:
        cached_sched = _fatigue_cache.get(fatigue_key)

    if cached_sched is not None:
        rest_days = cached_sched["rest_days"]
        is_b2b = cached_sched["b2b"]
        is_same_day = cached_sched["is_same_day"]
        is_3in4 = cached_sched["three_in_four"]
        is_4in6 = cached_sched["four_in_six"]
    else:
        # Find team's recent games
        rows = db.fetch_all(
            """SELECT DISTINCT game_date FROM player_stats ps
               JOIN players p ON ps.player_id = p.player_id
               WHERE p.team_id = ? AND ps.game_date < ?
               ORDER BY ps.game_date DESC LIMIT 10""",
            (team_id, game_date)
        )
        if not rows:
            return _default_result

        dates = []
        for r in rows:
            try:
                dates.append(datetime.strptime(r["game_date"], "%Y-%m-%d"))
            except (ValueError, TypeError):
                pass
        if not dates:
            return _default_result

        rest_days = (gd - dates[0]).days
        is_b2b = rest_days == 1
        is_same_day = rest_days == 0

        # 3-in-4 and 4-in-6
        window_4 = gd - timedelta(days=4)
        window_6 = gd - timedelta(days=6)
        games_in_4 = sum(1 for d in dates if d >= window_4) + 1  # +1 for current game
        games_in_6 = sum(1 for d in dates if d >= window_6) + 1

        is_3in4 = games_in_4 >= 3
        is_4in6 = games_in_6 >= 4

        # Store schedule facts in cache
        with _fatigue_cache_lock:
            _fatigue_cache[fatigue_key] = {
                "rest_days": rest_days, "b2b": is_b2b, "is_same_day": is_same_day,
                "three_in_four": is_3in4, "four_in_six": is_4in6,
            }

    # Apply weight-dependent penalties on top of cached schedule
    b2b_pen = w.fatigue_b2b if w else 2.0
    pen_3in4 = w.fatigue_3in4 if w else 1.0
    pen_4in6 = w.fatigue_4in6 if w else 1.5

    penalty = 0.0
    if is_same_day:
        penalty += 3.0
    if is_b2b:
        penalty += b2b_pen
    if is_3in4:
        penalty += pen_3in4
    if is_4in6:
        penalty += pen_4in6

    # Rest bonus
    rest_bonus = 0.0
    if rest_days >= 4:
        rest_bonus = -1.5
    elif rest_days >= 3:
        rest_bonus = -1.0

    return {
        "penalty": penalty + rest_bonus,
        "rest_days": rest_days,
        "b2b": is_b2b,
        "three_in_four": is_3in4,
        "four_in_six": is_4in6,
    }


def compute_shooting_efficiency(proj: Dict[str, float]) -> Dict[str, float]:
    """Compute TS%, eFG%, FG3 rate, FT rate from projection."""
    fga = proj.get("fg_attempted", 0)
    fta = proj.get("ft_attempted", 0)
    fgm = proj.get("fg_made", 0)
    fg3m = proj.get("fg3_made", 0)
    pts = proj.get("points", 0)

    ts_pct = (pts / (2 * (fga + 0.44 * fta)) * 100) if (fga + 0.44 * fta) > 0 else 0
    efg_pct = ((fgm + 0.5 * fg3m) / fga * 100) if fga > 0 else 0
    fg3_rate = (proj.get("fg3_attempted", 0) / fga * 100) if fga > 0 else 0
    ft_rate = (fta / fga * 100) if fga > 0 else 0

    return {"ts_pct": ts_pct, "efg_pct": efg_pct, "fg3_rate": fg3_rate, "ft_rate": ft_rate}


def detect_roster_change(team_id: int) -> Dict[str, Any]:
    """Compare current roster vs players who appeared in last 5 game dates."""
    current = set()
    rows = db.fetch_all("SELECT player_id FROM players WHERE team_id = ?", (team_id,))
    for r in rows:
        current.add(r["player_id"])

    # Players who appeared in recent games
    recent = set()
    game_rows = db.fetch_all(
        """SELECT DISTINCT ps.player_id FROM player_stats ps
           JOIN players p ON ps.player_id = p.player_id
           WHERE p.team_id = ?
           AND ps.game_date IN (
             SELECT game_date FROM (
               SELECT DISTINCT game_date FROM player_stats ps2
               JOIN players p2 ON ps2.player_id = p2.player_id
               WHERE p2.team_id = ?
               ORDER BY game_date DESC LIMIT 5
             )
           )""",
        (team_id, team_id)
    )
    for r in game_rows:
        recent.add(r["player_id"])

    added = current - recent
    removed = recent - current

    # Check if any are high impact
    high_impact = False
    for pid in added | removed:
        avg_row = db.fetch_one(
            "SELECT AVG(minutes) as avg_min FROM (SELECT minutes FROM player_stats WHERE player_id = ? ORDER BY game_date DESC LIMIT 10)",
            (pid,)
        )
        if avg_row and avg_row["avg_min"] and avg_row["avg_min"] >= _ROSTER_CHANGE_HIGH_IMPACT_MPG:
            high_impact = True
            break

    return {
        "changed": bool(added or removed),
        "players_added": list(added),
        "players_removed": list(removed),
        "high_impact": high_impact,
    }


def get_team_matchup_stats(team_id: int, opponent_team_id: int,
                           is_home: int, as_of_date: Optional[str] = None) -> List[Dict]:
    """Get per-player contributions for matchup display.

    contribution = ppg * 0.4 + location_ppg * 0.3 + vs_opp_ppg * 0.3
    """
    rows = db.fetch_all(
        "SELECT player_id, name, position FROM players WHERE team_id = ?",
        (team_id,)
    )
    result = []
    for p in rows:
        pid = p["player_id"]
        splits = player_splits(pid, opponent_team_id, is_home, recent_games=20,
                               as_of_date=as_of_date)
        if splits["minutes"] < 1.0:
            continue

        # Location-specific ppg
        loc_rows = db.fetch_all(
            """SELECT AVG(points) as ppg FROM (
               SELECT points FROM player_stats
               WHERE player_id = ? AND is_home = ?
               ORDER BY game_date DESC LIMIT 10)""",
            (pid, is_home)
        )
        loc_ppg = loc_rows[0]["ppg"] if loc_rows and loc_rows[0]["ppg"] else splits["points"]

        # vs-opponent ppg
        vs_rows = db.fetch_all(
            """SELECT AVG(points) as ppg FROM (
               SELECT points FROM player_stats
               WHERE player_id = ? AND opponent_team_id = ?
               ORDER BY game_date DESC LIMIT 10)""",
            (pid, opponent_team_id)
        )
        vs_ppg = vs_rows[0]["ppg"] if vs_rows and vs_rows[0]["ppg"] else splits["points"]

        contribution = splits["points"] * 0.4 + loc_ppg * 0.3 + vs_ppg * 0.3

        result.append({
            "player_id": pid,
            "name": p["name"],
            "position": p["position"],
            "ppg": round(splits["points"], 1),
            "rpg": round(splits["rebounds"], 1),
            "apg": round(splits["assists"], 1),
            "mpg": round(splits["minutes"], 1),
            "contribution": round(contribution, 1),
        })

    result.sort(key=lambda x: x["contribution"], reverse=True)
    return result


# ──────────────────── V2.1 Feature Computation ────────────────────

def compute_travel(team_id: int, game_date: str, opponent_team_id: int,
                   is_home: int) -> Dict[str, Any]:
    """Compute travel-related features for a team in a specific game.

    Returns {"travel_miles": float, "tz_crossings": int, "cum_travel_7d": float}
    """
    from src.data.arenas import travel_distance, timezone_crossings
    from src.config import get_season

    result = {"travel_miles": 0.0, "tz_crossings": 0, "cum_travel_7d": 0.0}

    try:
        if is_home:
            # Home team: check if last game was away (they traveled back)
            rows = db.fetch_all(
                """SELECT DISTINCT ps.game_date, ps.is_home, ps.opponent_team_id
                   FROM player_stats ps
                   JOIN players p ON ps.player_id = p.player_id
                   WHERE p.team_id = ? AND ps.game_date < ?
                   ORDER BY ps.game_date DESC LIMIT 1""",
                (team_id, game_date)
            )
            if rows and rows[0]["is_home"] == 0:
                # Last game was away — traveled back from that opponent's arena
                last_away_opp = rows[0]["opponent_team_id"]
                try:
                    result["travel_miles"] = travel_distance(last_away_opp, team_id)
                except KeyError:
                    pass
            result["tz_crossings"] = 0
        else:
            # Away team: distance from home arena to opponent's arena
            try:
                result["travel_miles"] = travel_distance(team_id, opponent_team_id)
            except KeyError:
                pass
            try:
                result["tz_crossings"] = timezone_crossings(team_id, opponent_team_id)
            except KeyError:
                pass

        # cum_travel_7d: approximate with current game distance
        result["cum_travel_7d"] = result["travel_miles"]

    except Exception as e:
        logger.debug("compute_travel error for team %d on %s: %s", team_id, game_date, e)

    return result


def compute_momentum(team_id: int, game_date: str, season: Optional[str] = None) -> Dict[str, Any]:
    """Compute momentum features: win/loss streak and margin-of-victory trend.

    Returns {"streak": int, "mov_trend": float}
    - streak: positive = consecutive wins, negative = consecutive losses (capped +-10)
    - mov_trend: average margin of victory over last 5 games (positive = winning by more)
    """
    result = {"streak": 0, "mov_trend": 0.0}

    try:
        if season is None:
            from src.config import get_season
            season = get_season()

        # Get recent game results for this team
        rows = db.fetch_all(
            """SELECT DISTINCT game_date, win_loss FROM player_stats
               WHERE season = ? AND game_date < ?
               AND player_id IN (SELECT player_id FROM players WHERE team_id = ?)
               ORDER BY game_date DESC LIMIT 10""",
            (season, game_date, team_id)
        )

        if not rows:
            return result

        # Compute streak
        streak = 0
        first_wl = rows[0]["win_loss"]
        for r in rows:
            if r["win_loss"] == first_wl:
                streak += 1
            else:
                break

        if first_wl == "L":
            streak = -streak

        # Clamp to +-10
        streak = max(-10, min(10, streak))
        result["streak"] = streak

        # mov_trend: average margin of victory from last 5 games
        # Compute from player_stats: team points vs opponent points on each game date
        recent_dates = [r["game_date"] for r in rows[:5]]
        if recent_dates:
            total_margin = 0.0
            counted = 0
            for gd in recent_dates:
                scores = db.fetch_one(
                    """SELECT
                        (SELECT COALESCE(SUM(ps.points), 0)
                         FROM player_stats ps
                         JOIN players p ON ps.player_id = p.player_id
                         WHERE p.team_id = ? AND ps.game_date = ? AND ps.season = ?) as team_pts,
                        (SELECT COALESCE(SUM(ps.points), 0)
                         FROM player_stats ps
                         WHERE ps.opponent_team_id = ? AND ps.game_date = ? AND ps.season = ?) as opp_pts
                    """,
                    (team_id, gd, season, team_id, gd, season)
                )
                if scores and scores["team_pts"] > 0:
                    margin = scores["team_pts"] - scores["opp_pts"]
                    total_margin += margin
                    counted += 1

            if counted > 0:
                result["mov_trend"] = total_margin / counted

    except Exception as e:
        logger.debug("compute_momentum error for team %d on %s: %s", team_id, game_date, e)

    return result


def compute_fg3_luck(team_id: int, game_date: str, season: Optional[str] = None) -> float:
    """Compute 3-point shooting luck signal for regression-to-mean adjustment.

    Returns the difference between recent 3PT% (last 5 games) and season-average
    3PT%. Positive = team has been shooting "hot" and should regress down.
    Negative = team has been shooting "cold" and should regress up.

    Uses player_stats fg3_made / fg3_attempted aggregated to team level.
    """
    try:
        if season is None:
            from src.config import get_season
            season = get_season()

        # Season-average 3PT% for this team (all games before game_date)
        season_row = db.fetch_one(
            """SELECT COALESCE(SUM(ps.fg3_made), 0) as fg3m,
                      COALESCE(SUM(ps.fg3_attempted), 0) as fg3a
               FROM player_stats ps
               JOIN players p ON ps.player_id = p.player_id
               WHERE p.team_id = ? AND ps.game_date < ? AND ps.season = ?""",
            (team_id, game_date, season)
        )

        if not season_row or season_row["fg3a"] < 50:  # need minimum sample
            return 0.0

        season_fg3_pct = season_row["fg3m"] / season_row["fg3a"]

        # Recent 3PT% (last 5 game dates)
        recent_dates = db.fetch_all(
            """SELECT DISTINCT game_date FROM player_stats
               WHERE season = ? AND game_date < ?
               AND player_id IN (SELECT player_id FROM players WHERE team_id = ?)
               ORDER BY game_date DESC LIMIT 5""",
            (season, game_date, team_id)
        )

        if not recent_dates or len(recent_dates) < 3:  # need at least 3 games
            return 0.0

        dates = [r["game_date"] for r in recent_dates]
        placeholders = ",".join("?" * len(dates))
        recent_row = db.fetch_one(
            f"""SELECT COALESCE(SUM(ps.fg3_made), 0) as fg3m,
                       COALESCE(SUM(ps.fg3_attempted), 0) as fg3a
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.player_id
                WHERE p.team_id = ? AND ps.game_date IN ({placeholders})
                AND ps.season = ?""",
            (team_id, *dates, season)
        )

        if not recent_row or recent_row["fg3a"] < 20:
            return 0.0

        recent_fg3_pct = recent_row["fg3m"] / recent_row["fg3a"]

        # Luck signal: positive = shooting above average (will regress down)
        # Scale to percentage points (e.g., 0.05 = 5 percentage points above avg)
        return recent_fg3_pct - season_fg3_pct

    except Exception as e:
        logger.debug("compute_fg3_luck error for team %d on %s: %s", team_id, game_date, e)
        return 0.0


def compute_schedule_spots(team_id: int, game_date: str,
                           opponent_team_id: int, season: Optional[str] = None) -> Dict[str, Any]:
    """Compute schedule-spot features: lookahead, letdown, road trip game number.

    Returns {"lookahead": bool, "letdown": bool, "road_trip_game": int}
    - lookahead: next opponent is top-8 by w_pct AND current opponent is NOT top-8
    - letdown: previous opponent was top-8 AND we won that game AND current opponent is NOT top-8
    - road_trip_game: count of consecutive away games up to and including current (0 if home)
    """
    result = {"lookahead": False, "letdown": False, "road_trip_game": 0}

    try:
        if season is None:
            from src.config import get_season
            season = get_season()

        # Get top-8 teams by w_pct
        top8_rows = db.fetch_all(
            "SELECT team_id FROM team_metrics WHERE season = ? ORDER BY w_pct DESC LIMIT 8",
            (season,)
        )
        top8 = {r["team_id"] for r in top8_rows} if top8_rows else set()

        current_is_top8 = opponent_team_id in top8

        # Lookahead: check next game's opponent
        next_rows = db.fetch_all(
            """SELECT DISTINCT ps.opponent_team_id FROM player_stats ps
               WHERE ps.game_date > ?
               AND ps.player_id IN (SELECT player_id FROM players WHERE team_id = ?)
               ORDER BY ps.game_date ASC LIMIT 1""",
            (game_date, team_id)
        )
        if next_rows and not current_is_top8:
            next_opp = next_rows[0]["opponent_team_id"]
            if next_opp in top8:
                result["lookahead"] = True

        # Letdown: check previous game
        prev_rows = db.fetch_all(
            """SELECT DISTINCT ps.game_date, ps.opponent_team_id, ps.win_loss
               FROM player_stats ps
               WHERE ps.game_date < ? AND ps.season = ?
               AND ps.player_id IN (SELECT player_id FROM players WHERE team_id = ?)
               ORDER BY ps.game_date DESC LIMIT 1""",
            (game_date, season, team_id)
        )
        if prev_rows and not current_is_top8:
            prev_opp = prev_rows[0]["opponent_team_id"]
            prev_wl = prev_rows[0]["win_loss"]
            if prev_opp in top8 and prev_wl == "W":
                result["letdown"] = True

        # Road trip game: count consecutive away games including current
        recent_rows = db.fetch_all(
            """SELECT DISTINCT ps.game_date, ps.is_home FROM player_stats ps
               WHERE ps.game_date <= ? AND ps.season = ?
               AND ps.player_id IN (SELECT player_id FROM players WHERE team_id = ?)
               ORDER BY ps.game_date DESC LIMIT 15""",
            (game_date, season, team_id)
        )
        if recent_rows:
            road_count = 0
            for r in recent_rows:
                if r["is_home"] == 0:
                    road_count += 1
                else:
                    break
            result["road_trip_game"] = road_count

    except Exception as e:
        logger.debug("compute_schedule_spots error for team %d on %s: %s",
                     team_id, game_date, e)

    return result
