"""Team context helpers used by web and desktop UIs.

This module keeps "display-only" team metadata (record, streak, rest, and
projected starters out) out of the core prediction math.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.analytics.stats_engine import compute_momentum
from src.config import get_season
from src.database import db

_OUT_STATUS_TOKENS = (
    "out",
    "doubtful",
    "suspend",
    "inactive",
    "not with team",
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_game_date(game_date: Optional[str]) -> str:
    """Return a valid YYYY-MM-DD date string."""
    if isinstance(game_date, str) and len(game_date) >= 10:
        candidate = game_date[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            pass
    from src.utils.timezone_utils import nba_today
    return nba_today()


def _season_for_game_date(game_date: str) -> str:
    """Map a game date to NBA season string (e.g. 2025-26)."""
    try:
        year = int(game_date[:4])
        month = int(game_date[5:7])
        if month >= 7:
            return f"{year}-{str(year + 1)[2:]}"
        return f"{year - 1}-{str(year)[2:]}"
    except (TypeError, ValueError, IndexError):
        return get_season()


def format_streak(streak_value: int) -> str:
    """Return a compact streak label."""
    if streak_value > 0:
        return f"W{streak_value}"
    if streak_value < 0:
        return f"L{abs(streak_value)}"
    return "EVEN"


def _is_out_status(status: str) -> bool:
    low = (status or "").strip().lower()
    if not low:
        return False
    if any(token in low for token in ("questionable", "probable", "day-to-day", "gtd", "available")):
        return False
    return any(token in low for token in _OUT_STATUS_TOKENS)


def _fetch_record(team_id: int, season: str, game_date: str) -> str:
    """Fetch season record, preferring team_metrics with player_stats fallback."""
    row = db.fetch_one(
        "SELECT w, l, gp FROM team_metrics WHERE team_id = ? AND season = ?",
        (team_id, season),
    )
    if row and _safe_int(row.get("gp"), 0) > 0:
        return f"{_safe_int(row.get('w'))}-{_safe_int(row.get('l'))}"

    fallback = db.fetch_one(
        """
        SELECT
            SUM(CASE WHEN gl.win_loss = 'W' THEN 1 ELSE 0 END) AS w,
            SUM(CASE WHEN gl.win_loss = 'L' THEN 1 ELSE 0 END) AS l
        FROM (
            SELECT game_id, MAX(win_loss) AS win_loss
            FROM player_stats
            WHERE team_id = ? AND season = ? AND game_date < ?
            GROUP BY game_id
        ) gl
        """,
        (team_id, season, game_date),
    )
    w = _safe_int((fallback or {}).get("w"), 0)
    l = _safe_int((fallback or {}).get("l"), 0)
    if w or l:
        return f"{w}-{l}"
    return ""


def _fetch_last_game_info(team_id: int, game_date: str) -> Tuple[Optional[int], str, str]:
    """Return (days_since_last_game, long_text, short_text)."""
    row = db.fetch_one(
        "SELECT MAX(game_date) AS game_date FROM player_stats WHERE team_id = ? AND game_date < ?",
        (team_id, game_date),
    )
    last_game = (row or {}).get("game_date")
    if not last_game:
        return None, "", ""

    try:
        current_dt = datetime.strptime(game_date, "%Y-%m-%d")
        last_dt = datetime.strptime(str(last_game)[:10], "%Y-%m-%d")
        days = (current_dt - last_dt).days
    except ValueError:
        return None, "", ""

    if days < 0:
        return None, "", ""
    if days == 1:
        return days, "Last game 1 day ago", "1d rest"
    return days, f"Last game {days} days ago", f"{days}d rest"


def _confirmed_starters(team_id: int, game_date: str) -> List[Dict[str, Any]]:
    """Look up confirmed first-quarter starters from the confirmed_lineups table."""
    rows = db.fetch_all(
        """
        SELECT cl.player_id, cl.player_name AS name
        FROM confirmed_lineups cl
        WHERE cl.team_id = ? AND cl.game_date = ?
        ORDER BY cl.player_id
        """,
        (team_id, game_date),
    )
    return [
        {
            "player_id": _safe_int(r.get("player_id"), 0),
            "name": str(r.get("name") or "").strip(),
        }
        for r in (rows or [])
        if _safe_int(r.get("player_id"), 0) > 0
    ]


def _mpg_projected_starters(team_id: int, season: str, game_date: str) -> List[Dict[str, Any]]:
    """Approximate likely starters using current roster + pregame MPG ranking."""
    rows = db.fetch_all(
        """
        SELECT
            p.player_id,
            p.name,
            COALESCE(AVG(ps.minutes), 0) AS mpg,
            COUNT(DISTINCT ps.game_id) AS gp
        FROM players p
        LEFT JOIN player_stats ps
            ON ps.player_id = p.player_id
           AND ps.team_id = ?
           AND ps.season = ?
           AND ps.game_date < ?
        WHERE p.team_id = ?
        GROUP BY p.player_id, p.name
        ORDER BY mpg DESC, gp DESC, p.name ASC
        LIMIT 12
        """,
        (team_id, season, game_date, team_id),
    )
    if not rows:
        return []

    starters: List[Dict[str, Any]] = [
        r for r in rows if float(r.get("mpg") or 0.0) >= 16.0
    ][:5]

    if len(starters) < 5:
        for r in rows:
            if r in starters:
                continue
            if float(r.get("mpg") or 0.0) > 0:
                starters.append(r)
            if len(starters) >= 5:
                break

    if len(starters) < 5:
        for r in rows:
            if r in starters:
                continue
            starters.append(r)
            if len(starters) >= 5:
                break

    return [
        {
            "player_id": _safe_int(r.get("player_id"), 0),
            "name": str(r.get("name") or "").strip(),
        }
        for r in starters
        if _safe_int(r.get("player_id"), 0) > 0
    ]


def _projected_starters(
    team_id: int, season: str, game_date: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Return starters and whether they are confirmed or projected.

    Prefers confirmed first-quarter starters when available; falls back
    to the MPG-based heuristic otherwise.
    """
    confirmed = _confirmed_starters(team_id, game_date)
    if confirmed:
        return confirmed, True
    return _mpg_projected_starters(team_id, season, game_date), False


def _fetch_starters_out(team_id: int, starters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not starters:
        return []

    starter_ids = [s["player_id"] for s in starters if s.get("player_id")]
    if not starter_ids:
        return []

    placeholders = ",".join("?" for _ in starter_ids)
    rows = db.fetch_all(
        f"""
        SELECT player_id, player_name, status, reason, expected_return
        FROM injuries
        WHERE team_id = ? AND player_id IN ({placeholders})
        """,
        (team_id, *starter_ids),
    )
    by_id = {
        _safe_int(r.get("player_id"), 0): r
        for r in rows
        if _safe_int(r.get("player_id"), 0) > 0
    }

    out_list: List[Dict[str, Any]] = []
    for starter in starters:
        sid = _safe_int(starter.get("player_id"), 0)
        injury = by_id.get(sid)
        if not injury:
            continue
        status = str(injury.get("status") or "").strip()
        if not _is_out_status(status):
            continue
        out_list.append(
            {
                "player_id": sid,
                "name": str(injury.get("player_name") or starter.get("name") or "").strip(),
                "status": status or "Out",
                "reason": str(injury.get("reason") or "").strip(),
                "expected_return": str(injury.get("expected_return") or "").strip(),
            }
        )

    return out_list


def _fetch_minutes_restricted(team_id: int) -> List[Dict[str, Any]]:
    """Find players on minutes restrictions for a team."""
    rows = db.fetch_all(
        "SELECT player_id, player_name, status, minutes_cap "
        "FROM injuries WHERE team_id = ? AND minutes_cap IS NOT NULL",
        (team_id,),
    )
    return [
        {
            "player_id": _safe_int(r.get("player_id"), 0),
            "name": str(r.get("player_name") or "").strip(),
            "status": str(r.get("status") or "").strip(),
            "minutes_cap": _safe_int(r.get("minutes_cap"), 0),
        }
        for r in (rows or [])
        if _safe_int(r.get("player_id"), 0) > 0
        and _safe_int(r.get("minutes_cap"), 0) > 0
    ]


def get_team_display_context(
    team_id: int,
    game_date: Optional[str] = None,
    include_starters_out: bool = False,
) -> Dict[str, Any]:
    """Build display metadata for one team on a specific game date."""
    if not team_id:
        return {
            "record": "",
            "streak": "",
            "streak_value": 0,
            "days_since_last_game": None,
            "last_game_text": "",
            "last_game_short": "",
            "starters_out": [],
        }

    game_date = _coerce_game_date(game_date)
    season = _season_for_game_date(game_date)

    record = _fetch_record(team_id, season, game_date)

    streak_value = 0
    try:
        momentum = compute_momentum(team_id, game_date, season=season)
        streak_value = _safe_int((momentum or {}).get("streak"), 0)
    except Exception:
        streak_value = 0

    days_since_last_game, last_game_text, last_game_short = _fetch_last_game_info(
        team_id, game_date
    )

    starters_out: List[Dict[str, Any]] = []
    starters_confirmed = False
    minutes_restricted: List[Dict[str, Any]] = []
    if include_starters_out:
        starters, starters_confirmed = _projected_starters(team_id, season, game_date)
        starters_out = _fetch_starters_out(team_id, starters)
        minutes_restricted = _fetch_minutes_restricted(team_id)

    return {
        "record": record,
        "streak": format_streak(streak_value),
        "streak_value": streak_value,
        "days_since_last_game": days_since_last_game,
        "last_game_text": last_game_text,
        "last_game_short": last_game_short,
        "starters_out": starters_out,
        "starters_confirmed": starters_confirmed,
        "minutes_restricted": minutes_restricted,
    }
