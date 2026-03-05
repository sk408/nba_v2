"""Disk cache for player headshot and team logo images."""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PLAYER_PHOTOS_DIR = Path("data") / "cache" / "player_photos"
TEAM_LOGOS_DIR = Path("data") / "cache" / "team_logos"
_RATE_LIMIT = 0.4  # seconds between image fetches


def _ensure_dirs():
    PLAYER_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    TEAM_LOGOS_DIR.mkdir(parents=True, exist_ok=True)


def get_player_photo_path(player_id: int) -> Optional[str]:
    """Get cached player photo path, or download if missing."""
    _ensure_dirs()
    path = PLAYER_PHOTOS_DIR / f"{player_id}.png"
    if path.exists():
        return str(path)
    url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                f.write(resp.content)
            return str(path)
    except Exception as e:
        logger.debug(f"Failed to fetch photo for {player_id}: {e}")
    return None


def get_team_logo_path(team_id: int) -> Optional[str]:
    """Get cached team logo path, or download if missing."""
    _ensure_dirs()
    path = TEAM_LOGOS_DIR / f"{team_id}.svg"
    if path.exists():
        return str(path)
    url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    try:
        time.sleep(_RATE_LIMIT)
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                f.write(resp.content)
            return str(path)
    except Exception as e:
        logger.debug(f"Failed to fetch logo for {team_id}: {e}")
    return None


def preload_images(player_ids=None, team_ids=None, callback=None):
    """Preload images for given player and team IDs.

    If player_ids or team_ids are None, auto-fetches all from the database.
    """
    _ensure_dirs()

    if player_ids is None or team_ids is None:
        try:
            from src.database import db
            if team_ids is None:
                rows = db.fetch_all("SELECT team_id FROM teams")
                team_ids = [r["team_id"] for r in rows] if rows else []
            if player_ids is None:
                # We want a very comprehensive list of player IDs since gamecasts can include anyone.
                # Try getting from standard tables.
                rows = db.fetch_all("""
                    SELECT player_id FROM players
                    UNION 
                    SELECT player_id FROM player_sync_cache
                    UNION
                    SELECT player_id FROM injuries
                    UNION
                    SELECT player_id FROM player_impact
                    UNION
                    SELECT player_id FROM player_stats
                    UNION
                    SELECT player_id FROM injury_history
                """)
                player_ids = [r["player_id"] for r in rows] if rows else []
                
                # Fallback to some generic IDs just to get started if the DB is somehow empty
                if not player_ids:
                    player_ids = [2544, 201939]
        except Exception as e:
            logger.warning(f"Failed to auto-fetch IDs: {e}")
            if player_ids is None:
                player_ids = []
            if team_ids is None:
                team_ids = []

    total = len(player_ids) + len(team_ids)
    done = 0
    
    if callback:
        callback(f"Images: {done}/{total}")
        
    for tid in team_ids:
        get_team_logo_path(tid)
        done += 1
        if callback:
            callback(f"Images: {done}/{total}")
    for pid in player_ids:
        get_player_photo_path(pid)
        done += 1
        if callback and done % 10 == 0:
            callback(f"Images: {done}/{total}")
            
    if callback:
        callback(f"Images: {total}/{total}")
        callback("Done")
