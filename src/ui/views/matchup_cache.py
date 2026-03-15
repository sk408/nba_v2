"""Cache helpers for Matchup predictions."""

from typing import Any, Dict, Tuple

from src.utils.timezone_utils import nba_today


def prediction_cache_key(home_team_id: int, away_team_id: int, game_date: str) -> Tuple[int, int, str]:
    """Return a stable cache key scoped to a specific game date."""
    return (int(home_team_id), int(away_team_id), str(game_date or ""))


def is_live_or_future_game(game_date: str, today: str = "") -> bool:
    """Return True for games on/after ``today`` (usually today's slate)."""
    today_str = today or nba_today()
    return str(game_date or "") >= today_str


def prediction_has_sharp_money(pred: Dict[str, Any]) -> bool:
    """Return True when prediction payload includes usable ML sharp percentages."""
    if not isinstance(pred, dict):
        return False

    ml_pub = pred.get("ml_sharp_home_public")
    ml_mon = pred.get("ml_sharp_home_money")
    if ml_pub is None or ml_mon is None:
        return False

    try:
        ml_pub_f = float(ml_pub)
        ml_mon_f = float(ml_mon)
    except (TypeError, ValueError):
        return False

    return ml_pub_f > 0.0 or ml_mon_f > 0.0


def should_refresh_sharp_panel(pred: Dict[str, Any], game_date: str, today: str = "") -> bool:
    """Return True when sharp panel should be refreshed at display time."""
    if not isinstance(pred, dict):
        return False
    if not is_live_or_future_game(game_date, today=today):
        return False
    return bool(pred.get("sharp_needs_refresh"))


def sanitize_prediction_for_cache(pred: Dict[str, Any], game_date: str, today: str = "") -> Dict[str, Any]:
    """Prepare a prediction payload for cache storage.

    Today's/future games intentionally avoid persisting sharp values so each
    display pass can re-resolve from live/API/DB data and stay fresh.
    """
    out = dict(pred or {})
    if is_live_or_future_game(game_date, today=today):
        out["ml_sharp_home_public"] = 0
        out["ml_sharp_home_money"] = 0
        out.pop("sharp_agrees", None)
        out["sharp_resolved"] = False
        out["sharp_needs_refresh"] = True
    else:
        out.pop("sharp_needs_refresh", None)
    return out


def should_use_cached_prediction(pred: Dict[str, Any], game_date: str, today: str = "") -> bool:
    """Decide whether cached prediction is fresh enough to display.

    For today's/future games we invalidate cache entries that have never been
    through the sharp-data resolution chain (``sharp_resolved`` absent) *and*
    lack sharp-money values.  Once ``resolve_sharp_panel_values`` has run and
    set the flag, the cache entry is accepted even if sharp data stayed at zero
    -- this prevents an infinite re-prediction loop when ActionNetwork simply
    has no ML data for a game.

    ``sharp_needs_refresh`` is a special-case flag for live/future cache
    entries where sharp values were intentionally stripped at write-time; these
    entries are accepted and re-hydrated at display-time.
    """
    if not isinstance(pred, dict) or not pred:
        return False

    if is_live_or_future_game(game_date, today=today):
        if pred.get("sharp_needs_refresh"):
            return True
        if not prediction_has_sharp_money(pred) and not pred.get("sharp_resolved"):
            return False

    return True
