"""NBA API wrappers – teams, rosters, game logs, metrics."""

import logging
from datetime import datetime
import threading
import time
from typing import List, Dict, Any, Optional

from src.config import get as get_setting, get_season, get_season_year
from src.database import db
from src.data.http_client import get_json, retry_call, HttpClientError
from src.utils.timezone_utils import nba_today, to_display_tz

logger = logging.getLogger(__name__)


def _normalize_game_date(raw: str) -> str:
    """Convert any NBA API date format to YYYY-MM-DD.

    Handles:
      - 'Oct 31, 2025' (PlayerGameLog GAME_DATE — full 4-digit year)
      - '2025-10-31'   (already correct)
      - '2025-10-31T00:00:00' (ISO with time)
    """
    s = str(raw).strip()
    if not s:
        return ""
    # Already in YYYY-MM-DD?
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        return s[:10]
    # Try common NBA API formats (full 4-digit year first)
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning("Could not parse game date: %r — returning raw[:10]", s)
    return s[:10]

_API_SLEEP = 0.8
_nba_api_rate_lock = threading.Lock()
_next_nba_api_request_at = 0.0


def _nba_api_min_interval_seconds() -> float:
    """Resolve NBA API pacing from settings with safe fallback."""
    try:
        interval = float(get_setting("nba_api_min_interval_seconds", _API_SLEEP))
    except (TypeError, ValueError):
        interval = _API_SLEEP
    return max(0.0, interval)


def _on_off_timeout_seconds() -> float:
    """Timeout override for TeamPlayerOnOffSummary requests."""
    try:
        timeout_seconds = float(get_setting("nba_api_on_off_timeout_seconds", 12.0))
    except (TypeError, ValueError):
        timeout_seconds = 12.0
    return max(3.0, min(30.0, timeout_seconds))


def _on_off_retries() -> int:
    """Retry override for TeamPlayerOnOffSummary requests."""
    try:
        retries = int(get_setting("nba_api_on_off_retries", 1))
    except (TypeError, ValueError):
        retries = 1
    return max(1, min(5, retries))


def _looks_like_rate_limit_timeout(exc: BaseException) -> bool:
    """Heuristic: stats.nba.com timeout patterns often indicate throttling."""
    text = str(exc).lower()
    return (
        "timed out" in text
        or "read timed out" in text
        or "readtimeout" in text
    )


def _pace_nba_api_request() -> None:
    """Apply a global spacing guard before each NBA API request."""
    global _next_nba_api_request_at
    min_interval = _nba_api_min_interval_seconds()
    if min_interval <= 0.0:
        return

    now = time.monotonic()
    with _nba_api_rate_lock:
        scheduled_at = max(now, _next_nba_api_request_at)
        _next_nba_api_request_at = scheduled_at + min_interval
    wait_seconds = scheduled_at - now
    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _row_to_game_log(row, player_id: int) -> dict:
    """Convert a single NBA API row to our game log dict format."""
    matchup = str(row.get("MATCHUP", ""))
    is_home = 1 if "vs." in matchup else 0
    opp_abbr = matchup.split(" ")[-1] if matchup else ""
    player_team_abbr = matchup.split(" ")[0] if matchup else ""
    return {
        "player_id": player_id,
        "player_team_abbr": player_team_abbr,
        "game_id": str(row.get("GAME_ID", row.get("Game_ID", ""))),
        "game_date": _normalize_game_date(row.get("GAME_DATE", "")),
        "matchup": matchup,
        "is_home": is_home,
        "opponent_abbr": opp_abbr,
        "win_loss": str(row.get("WL", "")),
        "minutes": float(row.get("MIN", 0) or 0),
        "points": float(row.get("PTS", 0) or 0),
        "rebounds": float(row.get("REB", 0) or 0),
        "assists": float(row.get("AST", 0) or 0),
        "steals": float(row.get("STL", 0) or 0),
        "blocks": float(row.get("BLK", 0) or 0),
        "turnovers": float(row.get("TOV", 0) or 0),
        "fg_made": int(row.get("FGM", 0) or 0),
        "fg_attempted": int(row.get("FGA", 0) or 0),
        "fg3_made": int(row.get("FG3M", 0) or 0),
        "fg3_attempted": int(row.get("FG3A", 0) or 0),
        "ft_made": int(row.get("FTM", 0) or 0),
        "ft_attempted": int(row.get("FTA", 0) or 0),
        "oreb": float(row.get("OREB", 0) or 0),
        "dreb": float(row.get("DREB", 0) or 0),
        "plus_minus": float(row.get("PLUS_MINUS", 0) or 0),
        "personal_fouls": float(row.get("PF", 0) or 0),
    }


def _safe_get(func, *args, retries=3, log_label: Optional[str] = None, **kwargs):
    """Call an nba_api function with rate limiting, retries, and error handling."""
    label = log_label or getattr(func, "__name__", "callable")

    def _on_retry(attempt: int, total: int, exc: BaseException):
        if _looks_like_rate_limit_timeout(exc):
            logger.warning(
                "NBA API retry %d/%d for %s after timeout (likely upstream rate limiting): %s",
                attempt,
                total,
                label,
                exc,
            )
        else:
            logger.warning(
                "NBA API retry %d/%d for %s: %s",
                attempt,
                total,
                label,
                exc,
            )

    try:
        def _paced_call(*paced_args, **paced_kwargs):
            _pace_nba_api_request()
            return func(*paced_args, **paced_kwargs)

        return retry_call(
            _paced_call,
            *args,
            retries=retries,
            backoff_base=_API_SLEEP,
            backoff_max=8.0,
            jitter_ratio=0.2,
            on_retry=_on_retry,
            **kwargs,
        )
    except Exception as e:
        if _looks_like_rate_limit_timeout(e):
            logger.warning(
                "NBA API gave repeated timeouts for %s after %d attempt(s); treating as rate-limit pressure and skipping this call.",
                label,
                retries,
            )
        else:
            logger.error(
                "NBA API error in %s after %d attempts: %s",
                label,
                retries,
                e,
            )
        logger.debug("NBA API call stacktrace", exc_info=True)
        return None


def fetch_teams() -> List[Dict[str, Any]]:
    """Fetch all NBA teams from nba_api static data."""
    try:
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        return all_teams
    except ImportError:
        logger.warning("nba_api not installed")
        return []
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return []


def fetch_players(team_id: int) -> List[Dict[str, Any]]:
    """Fetch roster for a single team using CommonTeamRoster."""
    try:
        from nba_api.stats.endpoints import CommonTeamRoster
        season = get_season()
        result = _safe_get(CommonTeamRoster, team_id=team_id, season=season)
        if result is None:
            return []
        df = result.get_data_frames()[0]
        players = []
        for _, row in df.iterrows():
            players.append({
                "player_id": int(row.get("PLAYER_ID", 0)),
                "name": row.get("PLAYER", ""),
                "team_id": team_id,
                "position": row.get("POSITION", ""),
                "height": row.get("HEIGHT", ""),
                "weight": row.get("WEIGHT", ""),
                "age": int(row.get("AGE", 0)) if row.get("AGE") else 0,
                "experience": int(row.get("EXP", 0)) if row.get("EXP") and str(row.get("EXP")).strip() not in ("R", "") else 0,
            })
        return players
    except Exception as e:
        logger.error(f"Error fetching players for team {team_id}: {e}")
        return []


def fetch_player_game_logs(player_id: int, season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch game logs for a single player using PlayerGameLog."""
    try:
        from nba_api.stats.endpoints import PlayerGameLog
        if season is None:
            season = get_season()
        result = _safe_get(PlayerGameLog, player_id=player_id, season=season)
        if result is None:
            return []
        df = result.get_data_frames()[0]
        return [_row_to_game_log(row, player_id) for _, row in df.iterrows()]
    except Exception as e:
        logger.error(f"Error fetching game logs for player {player_id}: {e}")
        return []


def fetch_bulk_game_logs(date_from: Optional[str] = None,
                         date_to: Optional[str] = None,
                         season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch ALL player game logs in one API call using LeagueGameLog.

    Args:
        date_from: Start date in MM/DD/YYYY format (inclusive). None = season start.
        date_to: End date in MM/DD/YYYY format (inclusive). None = today.
        season: NBA season string (e.g. '2025-26'). None = current season.

    Returns:
        List of game log dicts in the same format as fetch_player_game_logs().
    """
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        if season is None:
            season = get_season()
        kwargs = {
            "season": season,
            "player_or_team_abbreviation": "P",
            "season_type_all_star": "Regular Season",
        }
        if date_from:
            kwargs["date_from_nullable"] = date_from
        if date_to:
            kwargs["date_to_nullable"] = date_to

        result = _safe_get(LeagueGameLog, **kwargs)
        if result is None:
            return []
        df = result.get_data_frames()[0]
        if df.empty:
            return []

        return [_row_to_game_log(row, int(row["PLAYER_ID"])) for _, row in df.iterrows()]
    except Exception as e:
        logger.error(f"Error fetching bulk game logs: {e}")
        return []


def fetch_schedule_played() -> List[Dict[str, Any]]:
    """Fetch played games using LeagueGameFinder."""
    try:
        from nba_api.stats.endpoints import LeagueGameFinder
        season = get_season()
        result = _safe_get(
            LeagueGameFinder,
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        if result is None:
            return []
        df = result.get_data_frames()[0]
        games = []
        for _, row in df.iterrows():
            games.append({
                "team_id": int(row.get("TEAM_ID", 0)),
                "game_date": _normalize_game_date(row.get("GAME_DATE", "")),
                "matchup": str(row.get("MATCHUP", "")),
                "game_id": str(row.get("GAME_ID", "")),
            })
        return games
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        return []


def fetch_team_estimated_metrics(season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch TeamEstimatedMetrics."""
    try:
        from nba_api.stats.endpoints import TeamEstimatedMetrics
        if season is None:
            season = get_season()
        result = _safe_get(TeamEstimatedMetrics, season=season, league_id="00")
        if result is None:
            return []
        df = result.get_data_frames()[0]
        metrics = []
        for _, row in df.iterrows():
            metrics.append({
                "team_id": int(row.get("TEAM_ID", 0)),
                "gp": int(row.get("GP", 0) or 0),
                "w": int(row.get("W", 0) or 0),
                "l": int(row.get("L", 0) or 0),
                "w_pct": float(row.get("W_PCT", 0) or 0),
                "e_off_rating": float(row.get("E_OFF_RATING", 0) or 0),
                "e_def_rating": float(row.get("E_DEF_RATING", 0) or 0),
                "e_net_rating": float(row.get("E_NET_RATING", 0) or 0),
                "e_pace": float(row.get("E_PACE", 0) or 0),
                "e_ast_ratio": float(row.get("E_AST_RATIO", 0) or 0),
                "e_oreb_pct": float(row.get("E_OREB_PCT", 0) or 0),
                "e_dreb_pct": float(row.get("E_DREB_PCT", 0) or 0),
                "e_reb_pct": float(row.get("E_REB_PCT", 0) or 0),
                "e_tm_tov_pct": float(row.get("E_TM_TOV_PCT", 0) or 0),
            })
        return metrics
    except Exception as e:
        logger.error(f"Error fetching team estimated metrics: {e}")
        return []


def fetch_league_dash_team_stats(measure_type: str = "Advanced",
                                  per_mode: str = "PerGame",
                                  location: str = "",
                                  season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch LeagueDashTeamStats with flexible measure type and location."""
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        if season is None:
            season = get_season()
        kwargs = {
            "season": season,
            "measure_type_detailed_defense": measure_type,
            "per_mode_detailed": per_mode,
            "league_id_nullable": "00",
            "season_type_all_star": "Regular Season",
        }
        if location:
            kwargs["location_nullable"] = location
        result = _safe_get(LeagueDashTeamStats, **kwargs)
        if result is None:
            return []
        df = result.get_data_frames()[0]
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"Error fetching league dash team stats ({measure_type}, {location}): {e}")
        return []


def fetch_team_clutch_stats(season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch LeagueDashTeamClutch (Advanced)."""
    try:
        from nba_api.stats.endpoints import LeagueDashTeamClutch
        if season is None:
            season = get_season()
        result = _safe_get(
            LeagueDashTeamClutch,
            season=season,
            measure_type_detailed_defense="Advanced",
            league_id_nullable="00",
            season_type_all_star="Regular Season",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff="5",
        )
        if result is None:
            return []
        df = result.get_data_frames()[0]
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"Error fetching team clutch stats: {e}")
        return []


def fetch_team_hustle_stats(season: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch LeagueHustleStatsTeam."""
    try:
        from nba_api.stats.endpoints import LeagueHustleStatsTeam
        if season is None:
            season = get_season()
        result = _safe_get(LeagueHustleStatsTeam, season=season, league_id_nullable="00")
        if result is None:
            return []
        df = result.get_data_frames()[0]
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"Error fetching team hustle stats: {e}")
        return []


def fetch_player_on_off(team_id: int) -> Dict[str, Any]:
    """Fetch TeamPlayerOnOffSummary for on/off court ratings."""
    try:
        from nba_api.stats.endpoints import TeamPlayerOnOffSummary
        season = get_season()
        timeout_seconds = _on_off_timeout_seconds()
        retries = _on_off_retries()
        result = _safe_get(
            TeamPlayerOnOffSummary,
            team_id=team_id,
            season=season,
            measure_type_detailed_defense="Advanced",
            timeout=timeout_seconds,
            retries=retries,
            log_label=f"TeamPlayerOnOffSummary(team_id={team_id})",
        )
        if result is None:
            logger.warning(
                "On/off fetch unavailable for team_id=%s (timeout=%.1fs retries=%d)",
                team_id,
                timeout_seconds,
                retries,
            )
            return {"on": [], "off": [], "_ok": False}
        dfs = result.get_data_frames()
        # dfs[0] = team aggregate (no per-player ratings), skip it
        # dfs[1] = per-player ON court (COURT_STATUS='On')
        # dfs[2] = per-player OFF court (COURT_STATUS='Off')
        on_court = dfs[1].to_dict("records") if len(dfs) > 1 else []
        off_court = dfs[2].to_dict("records") if len(dfs) > 2 else []
        return {"on": on_court, "off": off_court, "_ok": True}
    except Exception as e:
        logger.error(f"Error fetching player on/off for team {team_id}: {e}")
        return {"on": [], "off": [], "_ok": False}


def fetch_player_estimated_metrics() -> List[Dict[str, Any]]:
    """Fetch PlayerEstimatedMetrics."""
    try:
        from nba_api.stats.endpoints import PlayerEstimatedMetrics
        season = get_season()
        result = _safe_get(PlayerEstimatedMetrics, season=season, league_id="00")
        if result is None:
            return []
        df = result.get_data_frames()[0]
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"Error fetching player estimated metrics: {e}")
        return []


def fetch_nba_cdn_schedule() -> List[Dict[str, Any]]:
    """Fetch future schedule from NBA CDN."""
    from datetime import datetime, timezone
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    try:
        data = get_json(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
            retries=3,
            backoff_base=0.8,
        )
        games = []
        today = nba_today()
        for game_date_obj in data.get("leagueSchedule", {}).get("gameDates", []):
            for game in game_date_obj.get("games", []):
                gd = str(game.get("gameDateEst", ""))[:10]
                if gd >= today:
                    # Parse UTC time and convert to local
                    utc_str = game.get("gameDateTimeUTC", "")
                    local_time_str = ""
                    if utc_str:
                        try:
                            utc_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                            local_dt = to_display_tz(utc_dt)
                            local_time_str = local_dt.strftime("%I:%M %p").lstrip("0")
                        except ValueError as e:
                            logger.warning("Time parse failed for %s: %s", utc_str, e)
                            local_time_str = ""
                    games.append({
                        "game_date": gd,
                        "home_team": game.get("homeTeam", {}).get("teamTricode", ""),
                        "away_team": game.get("awayTeam", {}).get("teamTricode", ""),
                        "home_team_id": game.get("homeTeam", {}).get("teamId"),
                        "away_team_id": game.get("awayTeam", {}).get("teamId"),
                        "game_time": local_time_str,
                        "game_time_utc": utc_str,
                        "arena": game.get("arenaName", ""),
                        "status_text": game.get("gameStatusText", ""),
                    })
        return games
    except HttpClientError as e:
        logger.error(f"Error fetching NBA CDN schedule: {e}")
        return []


def fetch_daily_lineups(game_date: str) -> List[Dict[str, Any]]:
    """Fetch confirmed first-quarter starters from stats.nba.com for a date.

    Returns list of dicts with game_id, team_id, player_id, player_name.
    Returns empty list if lineups are not yet posted (404) or on error.
    The endpoint only retains ~2 seasons of data.
    """
    date_compact = game_date.replace("-", "")
    url = f"https://stats.nba.com/js/data/leaders/00_daily_lineups_{date_compact}.json"
    try:
        data = get_json(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.nba.com/",
                "Origin": "https://www.nba.com",
            },
            timeout=12,
            retries=2,
            backoff_base=1.0,
        )
    except HttpClientError as e:
        if "404" in str(e) or "Not Found" in str(e):
            logger.debug("Daily lineups not available for %s (404)", game_date)
            return []
        logger.warning("Failed to fetch daily lineups for %s: %s", game_date, e)
        return []

    rows: List[Dict[str, Any]] = []
    try:
        games = data.get("gs", {}).get("g", [])
        for game in games:
            game_id = str(game.get("gid", ""))
            for side_key in ("hls", "vls"):
                side = game.get(side_key, {})
                team_id = int(side.get("tid", 0))
                for player in side.get("pstsg", []):
                    rows.append({
                        "game_id": game_id,
                        "team_id": team_id,
                        "player_id": int(player.get("pid", 0)),
                        "player_name": str(player.get("fn", "")) + " "
                                       + str(player.get("ln", "")),
                    })
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("Error parsing daily lineups for %s: %s", game_date, e)
        return []

    logger.info("Fetched %d confirmed starters for %s", len(rows), game_date)
    return rows


def resolve_opponent_team_id(opponent_abbr: str) -> int:
    """Look up team_id from abbreviation."""
    row = db.fetch_one(
        "SELECT team_id FROM teams WHERE abbreviation = ?",
        (opponent_abbr.upper(),)
    )
    return row["team_id"] if row else 0


def save_teams(teams: List[Dict[str, Any]]):
    """Upsert teams into the database (batched)."""
    batch = [
        (t.get("id", t.get("team_id")),
         t.get("full_name", t.get("name", "")),
         t.get("abbreviation", ""),
         t.get("conference", t.get("state", "")))
        for t in teams
    ]
    if batch:
        db.execute_many(
            """INSERT INTO teams (team_id, name, abbreviation, conference)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(team_id) DO UPDATE SET
                 name=excluded.name,
                 abbreviation=excluded.abbreviation,
                 conference=excluded.conference""",
            batch,
        )


def save_players(players: List[Dict[str, Any]]):
    """Upsert players into the database (batched)."""
    batch = [
        (p["player_id"], p["name"], p["team_id"], p.get("position", ""),
         p.get("height", ""), p.get("weight", ""),
         p.get("age", 0), p.get("experience", 0))
        for p in players
    ]
    if batch:
        db.execute_many(
            """INSERT INTO players (player_id, name, team_id, position, height, weight, age, experience)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id) DO UPDATE SET
                 name=excluded.name,
                 team_id=excluded.team_id,
                 position=excluded.position,
                 height=excluded.height,
                 weight=excluded.weight,
                 age=excluded.age,
                 experience=excluded.experience""",
            batch,
        )


def save_game_logs(logs: List[Dict[str, Any]], season: Optional[str] = None):
    """Insert game logs into player_stats with conflict ignore (batched)."""
    if season is None:
        season = get_season()
    batch = []
    for log in logs:
        opp_id = log.get("opponent_team_id", 0)
        if opp_id == 0:
            opp_id = resolve_opponent_team_id(log.get("opponent_abbr", ""))
        if opp_id == 0:
            continue
        player_team_id = resolve_opponent_team_id(log.get("player_team_abbr", "")) or None
        batch.append((
            log["player_id"], opp_id, log["is_home"], log["game_date"],
            log.get("game_id", ""), season,
            log["points"], log["rebounds"], log["assists"], log["minutes"],
            log.get("steals", 0), log.get("blocks", 0), log.get("turnovers", 0),
            log.get("fg_made", 0), log.get("fg_attempted", 0),
            log.get("fg3_made", 0), log.get("fg3_attempted", 0),
            log.get("ft_made", 0), log.get("ft_attempted", 0),
            log.get("oreb", 0), log.get("dreb", 0),
            log.get("plus_minus", 0), log.get("win_loss", ""),
            log.get("personal_fouls", 0),
            player_team_id,
        ))
    if batch:
        # Disable FK checks — game logs reference players who may not be in
        # the players table (traded/cut players, historical seasons).
        db.execute("PRAGMA foreign_keys=OFF")
        try:
            db.execute_many(
                """INSERT INTO player_stats
                   (player_id, opponent_team_id, is_home, game_date, game_id, season,
                    points, rebounds, assists, minutes, steals, blocks, turnovers,
                    fg_made, fg_attempted, fg3_made, fg3_attempted, ft_made, ft_attempted,
                    oreb, dreb, plus_minus, win_loss, personal_fouls, team_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(player_id, game_id) DO UPDATE SET
                    points=excluded.points, rebounds=excluded.rebounds,
                    assists=excluded.assists, minutes=excluded.minutes,
                    steals=excluded.steals, blocks=excluded.blocks,
                    turnovers=excluded.turnovers, fg_made=excluded.fg_made,
                    fg_attempted=excluded.fg_attempted, fg3_made=excluded.fg3_made,
                    fg3_attempted=excluded.fg3_attempted, ft_made=excluded.ft_made,
                    ft_attempted=excluded.ft_attempted, oreb=excluded.oreb,
                    dreb=excluded.dreb, plus_minus=excluded.plus_minus,
                    win_loss=excluded.win_loss, personal_fouls=excluded.personal_fouls,
                    team_id=COALESCE(excluded.team_id, player_stats.team_id)""",
                batch,
            )
        finally:
            db.execute("PRAGMA foreign_keys=ON")
