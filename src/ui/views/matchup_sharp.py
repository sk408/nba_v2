"""Helpers for resolving sharp-money values displayed in Matchup view."""

from typing import Any, Callable, Dict, Optional, Tuple


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _has_sharp_data(ml_pub: int, ml_mon: int) -> bool:
    return ml_pub > 0 or ml_mon > 0


def _derive_sharp_agreement(pick: str, ml_pub: int, ml_mon: int) -> Optional[bool]:
    if not _has_sharp_data(ml_pub, ml_mon):
        return None
    pick_str = str(pick or "").upper()
    if pick_str not in ("HOME", "AWAY"):
        return None
    sharp_favors_home = ml_mon > ml_pub
    model_favors_home = pick_str == "HOME"
    return sharp_favors_home == model_favors_home


def resolve_sharp_panel_values(
    result: Dict[str, Any],
    game_data: Dict[str, Any],
    fetch_live_odds: Callable[[str, str], Dict[str, Any]],
    fetch_db_odds: Callable[[str, int, int], Dict[str, Any]],
) -> Tuple[int, int, Optional[bool]]:
    """Resolve ML sharp percentages from prediction, then live/API fallbacks.

    The order is:
    1) prediction payload
    2) live ActionNetwork odds
    3) DB `game_odds` row
    """
    result = result if isinstance(result, dict) else {}
    game_data = game_data if isinstance(game_data, dict) else {}

    ml_pub = _to_int(result.get("ml_sharp_home_public"))
    ml_mon = _to_int(result.get("ml_sharp_home_money"))
    sharp_agrees = result.get("sharp_agrees")

    if not _has_sharp_data(ml_pub, ml_mon):
        home_abbr = str(game_data.get("home_team", "") or "").strip().upper()
        away_abbr = str(game_data.get("away_team", "") or "").strip().upper()
        if home_abbr and away_abbr:
            live_odds = fetch_live_odds(home_abbr, away_abbr) or {}
            ml_pub = _to_int(live_odds.get("ml_home_public"))
            ml_mon = _to_int(live_odds.get("ml_home_money"))

    if not _has_sharp_data(ml_pub, ml_mon):
        game_date = str(game_data.get("game_date", "") or "").strip()
        home_id = _to_int(game_data.get("home_team_id"))
        away_id = _to_int(game_data.get("away_team_id"))
        if game_date and home_id and away_id:
            row = fetch_db_odds(game_date, home_id, away_id) or {}
            ml_pub = _to_int(row.get("ml_home_public"))
            ml_mon = _to_int(row.get("ml_home_money"))

    if sharp_agrees is None:
        sharp_agrees = _derive_sharp_agreement(
            result.get("pick", ""),
            ml_pub,
            ml_mon,
        )

    return ml_pub, ml_mon, sharp_agrees


def hydrate_scan_sharp_data(
    pred_fund: Dict[str, Any],
    pred_sharp: Dict[str, Any],
    game_data: Dict[str, Any],
    fetch_live_odds: Callable[[str, str], Dict[str, Any]],
    fetch_db_odds: Callable[[str, int, int], Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Hydrate sharp values for both scan modes with shared fallbacks.

    Live/DB lookups are memoized per game so fundamentals+sharp payloads can both
    be enriched without duplicating network/database calls.
    """
    pred_fund = dict(pred_fund or {})
    pred_sharp = dict(pred_sharp or {})
    game_data = game_data if isinstance(game_data, dict) else {}

    live_cache: Dict[str, Any] = {"loaded": False, "value": {}}
    db_cache: Dict[str, Any] = {"loaded": False, "value": {}}

    def _fetch_live_cached(home_abbr: str, away_abbr: str) -> Dict[str, Any]:
        if not live_cache["loaded"]:
            live_cache["value"] = fetch_live_odds(home_abbr, away_abbr) or {}
            live_cache["loaded"] = True
        return live_cache["value"]

    def _fetch_db_cached(game_date: str, home_id: int, away_id: int) -> Dict[str, Any]:
        if not db_cache["loaded"]:
            db_cache["value"] = fetch_db_odds(game_date, home_id, away_id) or {}
            db_cache["loaded"] = True
        return db_cache["value"]

    for pred in (pred_fund, pred_sharp):
        ml_pub, ml_mon, sharp_agrees = resolve_sharp_panel_values(
            result=pred,
            game_data=game_data,
            fetch_live_odds=_fetch_live_cached,
            fetch_db_odds=_fetch_db_cached,
        )
        # Always persist resolved values (even zeros) so the cache knows
        # the fallback chain has been exhausted.
        pred["ml_sharp_home_public"] = ml_pub
        pred["ml_sharp_home_money"] = ml_mon
        if sharp_agrees is not None:
            pred["sharp_agrees"] = sharp_agrees
        pred["sharp_resolved"] = True

    return pred_fund, pred_sharp
