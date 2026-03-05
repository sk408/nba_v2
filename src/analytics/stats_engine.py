"""Minimal stats_engine stub for first-boot data sync.

Only provides get_team_abbreviations() which is needed by odds_sync.
Task 7 will replace this with the full implementation.
"""

import threading
from typing import Dict
from src.database import db

_team_abbr_cache = None
_team_cache_lock = threading.Lock()


def get_team_abbreviations() -> Dict[int, str]:
    """Return {team_id: abbreviation} singleton (loaded once, never changes)."""
    global _team_abbr_cache
    with _team_cache_lock:
        if _team_abbr_cache is not None:
            return _team_abbr_cache
        rows = db.fetch_all("SELECT team_id, abbreviation FROM teams")
        _team_abbr_cache = {r["team_id"]: r["abbreviation"] for r in rows}
        return _team_abbr_cache


def invalidate_stats_caches():
    """Clear cached data. Full implementation in Task 7."""
    global _team_abbr_cache
    _team_abbr_cache = None
