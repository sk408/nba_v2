import logging
import time
from datetime import datetime
from typing import Optional, Callable
from src.database import db
from src.data.http_client import get_json, HttpClientError

logger = logging.getLogger(__name__)

_ACTIONNETWORK_BLOCK_COOLDOWN_SEC = 5 * 60
_actionnetwork_blocked_until = 0.0
_actionnetwork_last_block_log = 0.0


def _is_actionnetwork_403(exc: Exception) -> bool:
    text = str(exc)
    return "HTTP 403" in text and "api.actionnetwork.com" in text

def fetch_action_odds(date_str: str) -> list:
    """Fetch games and odds from Action Network for a specific date (YYYYMMDD)."""
    global _actionnetwork_blocked_until, _actionnetwork_last_block_log

    now = time.time()
    if now < _actionnetwork_blocked_until:
        if now - _actionnetwork_last_block_log >= 60:
            remaining = int(max(0.0, _actionnetwork_blocked_until - now))
            logger.info(
                "ActionNetwork cooldown active (%ss remaining); skipping odds fetch for %s.",
                remaining,
                date_str,
            )
            _actionnetwork_last_block_log = now
        return []

    url = f"https://api.actionnetwork.com/web/v1/scoreboard/nba?date={date_str}"
    try:
        # Action Network API often blocks default requests User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        data = get_json(
            url,
            headers=headers,
            timeout=10,
            retries=3,
            backoff_base=0.8,
        )
        _actionnetwork_blocked_until = 0.0
        return data.get("games", [])
    except HttpClientError as e:
        if _is_actionnetwork_403(e):
            _actionnetwork_blocked_until = now + _ACTIONNETWORK_BLOCK_COOLDOWN_SEC
            _actionnetwork_last_block_log = now
            logger.warning(
                "ActionNetwork returned 403 for %s (likely VPN/IP block); cooling down for %d minutes.",
                date_str,
                _ACTIONNETWORK_BLOCK_COOLDOWN_SEC // 60,
            )
        else:
            logger.error("Error fetching Action Network odds for %s: %s", date_str, e)
            logger.debug("ActionNetwork fetch stacktrace", exc_info=True)
        return []

from src.utils.team_mapper import normalize_action_abbr as _map_action_abbrev

_SBD_API_URL = "https://www.sportsbettingdime.com/wp-json/adpt/v1/nba-odds"
_SBD_BOOKS = "sr:book:17324,sr:book:18149,sr:book:18186"


def sync_betting_splits(game_date: str, callback: Optional[Callable] = None) -> int:
    """Fetch betting splits (bet% / money%) from SportsBettingDime and update game_odds.

    Only fills in NULL split columns — never overwrites existing ActionNetwork data.
    Uses standard NBA abbreviations (no mapping needed).
    """
    from src.analytics.stats_engine import get_team_abbreviations

    try:
        data = get_json(
            _SBD_API_URL,
            params={"books": _SBD_BOOKS, "format": "us", "date": game_date},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
            },
            timeout=15,
            retries=2,
            backoff_base=1.0,
        )
    except HttpClientError as e:
        logger.warning("SportsBettingDime splits fetch failed for %s: %s", game_date, e)
        return 0

    games = data.get("data", [])
    if not games:
        return 0

    id_to_abbr = get_team_abbreviations()
    abbr_to_id = {v: k for k, v in id_to_abbr.items()}
    now = datetime.now().isoformat()
    updated = 0

    for game in games:
        splits = game.get("bettingSplits")
        if not splits:
            continue

        comp = game.get("competitors", {})
        home_abbr = comp.get("home", {}).get("abbreviation", "")
        away_abbr = comp.get("away", {}).get("abbreviation", "")
        home_id = abbr_to_id.get(home_abbr)
        away_id = abbr_to_id.get(away_abbr)
        if not home_id or not away_id:
            continue

        sp = splits.get("spread", {})
        ml = splits.get("moneyline", {})

        spread_home_public = (sp.get("home") or {}).get("betsPercentage")
        spread_away_public = (sp.get("away") or {}).get("betsPercentage")
        spread_home_money = (sp.get("home") or {}).get("stakePercentage")
        spread_away_money = (sp.get("away") or {}).get("stakePercentage")
        ml_home_public = (ml.get("home") or {}).get("betsPercentage")
        ml_away_public = (ml.get("away") or {}).get("betsPercentage")
        ml_home_money = (ml.get("home") or {}).get("stakePercentage")
        ml_away_money = (ml.get("away") or {}).get("stakePercentage")

        if spread_home_public is None and ml_home_public is None:
            continue

        try:
            cur = db.execute("""
                UPDATE game_odds SET
                    spread_home_public = COALESCE(spread_home_public, ?),
                    spread_away_public = COALESCE(spread_away_public, ?),
                    spread_home_money  = COALESCE(spread_home_money,  ?),
                    spread_away_money  = COALESCE(spread_away_money,  ?),
                    ml_home_public     = COALESCE(ml_home_public,     ?),
                    ml_away_public     = COALESCE(ml_away_public,     ?),
                    ml_home_money      = COALESCE(ml_home_money,      ?),
                    ml_away_money      = COALESCE(ml_away_money,      ?)
                WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?
                  AND (spread_home_public IS NULL OR ml_home_public IS NULL)
            """, (spread_home_public, spread_away_public,
                  spread_home_money, spread_away_money,
                  ml_home_public, ml_away_public,
                  ml_home_money, ml_away_money,
                  game_date, home_id, away_id))
            if cur.rowcount > 0:
                updated += 1
        except Exception as e:
            logger.warning("SBD splits update failed for %s vs %s: %s", home_abbr, away_abbr, e)

    if updated > 0:
        logger.info("SportsBettingDime: filled betting splits for %d games on %s", updated, game_date)
        if callback:
            callback(f"SBD: filled betting splits for {updated} games on {game_date}")

    return updated


def _sync_espn_odds_fallback(game_date: str, callback: Optional[Callable] = None) -> int:
    """Fetch odds from ESPN game summaries as a fallback when ActionNetwork is blocked.

    ESPN provides spread, moneylines, and over/under but NOT sharp money
    (public/money bet percentages).  Those columns are left NULL.
    """
    import time
    from src.data.gamecast import fetch_espn_scoreboard, fetch_espn_game_summary
    from src.analytics.stats_engine import get_team_abbreviations
    from src.utils.team_mapper import normalize_espn_abbr

    scoreboard = fetch_espn_scoreboard(game_date)
    if not scoreboard:
        return 0

    id_to_abbr = get_team_abbreviations()
    abbr_to_id = {v: k for k, v in id_to_abbr.items()}
    now = datetime.now().isoformat()
    saved = 0

    for game in scoreboard:
        espn_id = game.get("espn_id")
        if not espn_id:
            continue

        home_abbr = normalize_espn_abbr(game.get("home_team", ""))
        away_abbr = normalize_espn_abbr(game.get("away_team", ""))
        home_id = abbr_to_id.get(home_abbr)
        away_id = abbr_to_id.get(away_abbr)
        if not home_id or not away_id:
            continue

        try:
            summary = fetch_espn_game_summary(espn_id)
            pickcenter = summary.get("pickcenter", [])
            if not pickcenter:
                continue
            pc = pickcenter[0]

            # Home spread (signed, from the home team's perspective)
            ps = pc.get("pointSpread", {})
            home_line_str = (ps.get("home") or {}).get("close", {}).get("line", "")
            if home_line_str:
                spread = float(home_line_str)
            else:
                spread_mag = pc.get("spread")
                if spread_mag is not None:
                    home_fav = (pc.get("homeTeamOdds") or {}).get("favorite", False)
                    spread = -float(spread_mag) if home_fav else float(spread_mag)
                else:
                    spread = None

            ou = pc.get("overUnder")
            home_ml = (pc.get("homeTeamOdds") or {}).get("moneyLine")
            away_ml = (pc.get("awayTeamOdds") or {}).get("moneyLine")

            if spread is None and ou is None:
                continue

            # Opening spread for spread_movement tracking
            open_line_str = (ps.get("home") or {}).get("open", {}).get("line", "")
            opening_spread = float(open_line_str) if open_line_str else None

            db.execute("""
                INSERT INTO game_odds (
                    game_date, home_team_id, away_team_id, spread, over_under,
                    home_moneyline, away_moneyline, opening_spread,
                    fetched_at, provider
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'espn')
                ON CONFLICT(game_date, home_team_id, away_team_id) DO UPDATE SET
                    spread=COALESCE(excluded.spread, spread),
                    over_under=COALESCE(excluded.over_under, over_under),
                    home_moneyline=COALESCE(excluded.home_moneyline, home_moneyline),
                    away_moneyline=COALESCE(excluded.away_moneyline, away_moneyline),
                    opening_spread=COALESCE(excluded.opening_spread, opening_spread),
                    fetched_at=excluded.fetched_at,
                    provider=CASE WHEN provider='actionnetwork' THEN provider ELSE excluded.provider END
            """, (game_date, home_id, away_id, spread, ou, home_ml, away_ml,
                  opening_spread, now))
            saved += 1
        except Exception as e:
            logger.warning("ESPN odds fallback failed for game %s: %s", espn_id, e)

        time.sleep(0.3)

    if saved > 0:
        logger.info("ESPN fallback: saved odds for %d games on %s", saved, game_date)
        if callback:
            callback(f"ESPN fallback: saved odds for {saved} games on {game_date}")
    return saved


def sync_odds_for_date(
    game_date: str,
    callback: Optional[Callable] = None,
    invalidate_cache: bool = True,
) -> int:
    """Fetch and store odds for all games on a date (YYYY-MM-DD)."""
    action_date = game_date.replace("-", "")
    games = fetch_action_odds(action_date)
    if not games:
        # ActionNetwork returned nothing (403 block or no data) → try ESPN
        saved = _sync_espn_odds_fallback(game_date, callback=callback)
        # Supplement with SBD betting splits (fills NULL split columns)
        sync_betting_splits(game_date, callback=callback)
        if saved > 0 and invalidate_cache:
            try:
                from src.analytics.cache_registry import invalidate_for_event
                invalidate_for_event("post_odds_sync")
            except Exception as e:
                logger.debug("ESPN fallback cache invalidation failed: %s", e)
        return saved

    # Get team mapping (cached singleton)
    from src.analytics.stats_engine import get_team_abbreviations
    id_to_abbrev = get_team_abbreviations()
    abbrev_to_id = {v: k for k, v in id_to_abbrev.items()}
    
    saved_count = 0
    now = datetime.now().isoformat()
    
    # We will log if any games were found but had no odds
    games_with_no_odds = 0

    for game in games:
        try:
            home_abbr = None
            away_abbr = None
            for team in game.get("teams", []):
                if team.get("id") == game.get("home_team_id"):
                    home_abbr = team.get("abbr")
                elif team.get("id") == game.get("away_team_id"):
                    away_abbr = team.get("abbr")
            
            if not home_abbr or not away_abbr:
                continue
                
            home_id = abbrev_to_id.get(_map_action_abbrev(home_abbr))
            away_id = abbrev_to_id.get(_map_action_abbrev(away_abbr))
            
            if not home_id or not away_id:
                continue

            odds_list = game.get("odds", [])
            if not odds_list: 
                games_with_no_odds += 1
                continue
            
            # Find consensus odds (book_id == 15 and type == "game")
            # Fallback to any game odds if book 15 isn't found
            game_odds = next((o for o in odds_list if o.get("type") == "game" and o.get("book_id") == 15), None)
            if not game_odds:
                game_odds = next((o for o in odds_list if o.get("type") == "game"), None)
                
            if not game_odds:
                games_with_no_odds += 1
                continue
            
            spread = game_odds.get("spread_home")
            ou = game_odds.get("total")
            home_ml = game_odds.get("ml_home")
            away_ml = game_odds.get("ml_away")
            
            # Sharp money metrics
            spread_home_public = game_odds.get("spread_home_public")
            spread_away_public = game_odds.get("spread_away_public")
            spread_home_money = game_odds.get("spread_home_money")
            spread_away_money = game_odds.get("spread_away_money")
            ml_home_public = game_odds.get("ml_home_public")
            ml_away_public = game_odds.get("ml_away_public")
            ml_home_money = game_odds.get("ml_home_money")
            ml_away_money = game_odds.get("ml_away_money")

            # Bet count (updates as more bets come in)
            num_bets = game_odds.get("num_bets")

            if spread is None and ou is None:
                continue
                
            # If the spread is 0, we can save it as 0.0
            if spread == "EVEN":
                spread = 0.0
            elif spread is not None:
                spread = float(spread)

            db.execute("""
                INSERT INTO game_odds (
                    game_date, home_team_id, away_team_id, spread, over_under,
                    home_moneyline, away_moneyline, fetched_at, provider,
                    spread_home_public, spread_away_public, spread_home_money, spread_away_money,
                    ml_home_public, ml_away_public, ml_home_money, ml_away_money,
                    num_bets
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'actionnetwork', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_date, home_team_id, away_team_id) DO UPDATE SET
                    spread=excluded.spread,
                    over_under=excluded.over_under,
                    home_moneyline=excluded.home_moneyline,
                    away_moneyline=excluded.away_moneyline,
                    spread_home_public=excluded.spread_home_public,
                    spread_away_public=excluded.spread_away_public,
                    spread_home_money=excluded.spread_home_money,
                    spread_away_money=excluded.spread_away_money,
                    ml_home_public=excluded.ml_home_public,
                    ml_away_public=excluded.ml_away_public,
                    ml_home_money=excluded.ml_home_money,
                    ml_away_money=excluded.ml_away_money,
                    num_bets=excluded.num_bets,
                    fetched_at=excluded.fetched_at,
                    provider=excluded.provider
            """, (game_date, home_id, away_id, spread, ou, home_ml, away_ml, now,
                  spread_home_public, spread_away_public, spread_home_money, spread_away_money,
                  ml_home_public, ml_away_public, ml_home_money, ml_away_money,
                  num_bets))
            
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to parse odds for game {game.get('id')}: {e}")
            
    if saved_count > 0 and invalidate_cache:
        try:
            from src.analytics.cache_registry import invalidate_for_event
            invalidate_for_event("post_odds_sync")
        except Exception as e:
            logger.debug("Odds cache invalidation failed: %s", e)

    # Supplement with SBD betting splits (fills NULL split columns)
    sync_betting_splits(game_date, callback=callback)

    if callback:
        if saved_count > 0:
            callback(f"Saved odds for {saved_count} games on {game_date}")
        elif games_with_no_odds > 0:
            logger.info(f"Skipped {game_date} - Action Network had no odds for these {games_with_no_odds} historical games.")

    return saved_count

def backfill_odds(callback: Optional[Callable] = None, force: bool = False) -> int:
    """Backfill odds for all historical games that have player_stats but no odds.

    Args:
        force: If True, re-fetch ALL dates (even those with existing sharp money data).
    """
    if force:
        rows = db.fetch_all("""
            SELECT DISTINCT game_date
            FROM player_stats
            ORDER BY game_date DESC
        """)
    else:
        # Game-level completeness: find dates where the number of games
        # with complete odds is fewer than the number of games played.
        # Match at game granularity (date + home team + away team).
        rows = db.fetch_all("""
            SELECT game_date
            FROM (
                SELECT p.game_date, p.home_team_id, p.away_team_id
                FROM (
                    SELECT DISTINCT ps.game_date, ps.team_id AS home_team_id, ps.opponent_team_id AS away_team_id
                    FROM player_stats ps
                    WHERE ps.game_id IS NOT NULL
                      AND ps.is_home = 1
                ) p
                LEFT JOIN (
                    SELECT DISTINCT go.game_date, go.home_team_id, go.away_team_id
                    FROM game_odds go
                    WHERE go.spread_home_public IS NOT NULL
                ) o
                  ON o.game_date = p.game_date
                 AND o.home_team_id = p.home_team_id
                 AND o.away_team_id = p.away_team_id
                WHERE o.home_team_id IS NULL
            )
            GROUP BY game_date
            ORDER BY game_date DESC
        """)
    
    dates = [r["game_date"] for r in rows]
    total_dates = len(dates)
    total_saved = 0
    
    if callback:
        callback(f"Found {total_dates} dates needing odds/sharp money backfill. Starting...")
        
    import time
    for i, game_date in enumerate(dates):
        if not game_date or len(game_date) != 10:
            continue
        try:
            saved = sync_odds_for_date(game_date, invalidate_cache=False)
            total_saved += saved
            # Sleep to avoid rate limiting from Action Network
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error backfilling odds for {game_date}: {e}")
            
        if callback and (i + 1) % 10 == 0:
            callback(f"Odds backfill progress: {i + 1}/{total_dates} dates processed.")
            
    if callback:
        callback(f"Odds backfill complete! Saved/updated odds for {total_saved} games.")

    if total_saved > 0:
        try:
            from src.analytics.cache_registry import invalidate_for_event
            invalidate_for_event("post_odds_sync")
        except Exception as e:
            logger.debug("Backfill cache invalidation failed: %s", e)
        
    return total_saved


def sync_upcoming_odds(callback: Optional[Callable] = None) -> int:
    """Sync odds for today and tomorrow (Action Network primary, ESPN/SBD fallback).

    Returns total games saved/updated across both dates.
    """
    from src.utils.timezone_utils import nba_today, nba_tomorrow

    total = 0
    for date in (nba_today(), nba_tomorrow()):
        try:
            saved = sync_odds_for_date(date, callback=callback)
            total += saved
        except Exception as e:
            logger.warning("sync_upcoming_odds failed for %s: %s", date, e)
    return total
