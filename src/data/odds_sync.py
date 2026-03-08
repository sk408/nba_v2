import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Callable
from src.database import db

logger = logging.getLogger(__name__)

def fetch_action_odds(date_str: str) -> list:
    """Fetch games and odds from Action Network for a specific date (YYYYMMDD)."""
    url = f"https://api.actionnetwork.com/web/v1/scoreboard/nba?date={date_str}"
    try:
        # Action Network API often blocks default requests User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("games", [])
    except Exception as e:
        logger.error(f"Error fetching Action Network odds for {date_str}: {e}")
        return []

from src.utils.team_mapper import normalize_action_abbr as _map_action_abbrev

def sync_odds_for_date(game_date: str, callback: Optional[Callable] = None) -> int:
    """Fetch and store odds for all games on a date (YYYY-MM-DD)."""
    action_date = game_date.replace("-", "")
    games = fetch_action_odds(action_date)
    if not games:
        return 0

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
                    ml_home_public, ml_away_public, ml_home_money, ml_away_money
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'actionnetwork', ?, ?, ?, ?, ?, ?, ?, ?)
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
                    fetched_at=excluded.fetched_at,
                    provider=excluded.provider
            """, (game_date, home_id, away_id, spread, ou, home_ml, away_ml, now,
                  spread_home_public, spread_away_public, spread_home_money, spread_away_money,
                  ml_home_public, ml_away_public, ml_home_money, ml_away_money))
            
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to parse odds for game {game.get('id')}: {e}")
            
    # TODO: re-implement odds cache invalidation when prediction_quality module is restored

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
        # Each game has exactly one game_id; game_odds has one row per game.
        rows = db.fetch_all("""
            SELECT game_date
            FROM (
                SELECT ps.game_date,
                       COUNT(DISTINCT ps.game_id) as games_played,
                       COUNT(DISTINCT go.home_team_id) as games_with_odds
                FROM player_stats ps
                LEFT JOIN game_odds go
                    ON go.game_date = ps.game_date
                    AND go.spread_home_public IS NOT NULL
                WHERE ps.game_id IS NOT NULL
                GROUP BY ps.game_date
            )
            WHERE games_with_odds < games_played
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
            saved = sync_odds_for_date(game_date)
            total_saved += saved
            # Sleep to avoid rate limiting from Action Network
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error backfilling odds for {game_date}: {e}")
            
        if callback and (i + 1) % 10 == 0:
            callback(f"Odds backfill progress: {i + 1}/{total_dates} dates processed.")
            
    if callback:
        callback(f"Odds backfill complete! Saved/updated odds for {total_saved} games.")
        
    return total_saved
