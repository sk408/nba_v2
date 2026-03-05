"""Singleton InMemoryDataStore — loads 7 tables as pandas DataFrames."""

import threading
import logging
from typing import Optional, Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


class InMemoryDataStore:
    """Double-checked locking singleton for in-memory data."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.player_stats = None
        self.teams = None
        self.players = None
        self.team_metrics = None
        self.player_impact = None
        self.team_tuning = None
        self.injury_history = None
        self.precomputed_games: Optional[list] = None
        self._loaded = False
        self._initialized = True

    def load(self):
        """Load all 7 tables as DataFrames."""
        if not HAS_PANDAS:
            logger.warning("pandas not available, memory store disabled")
            return

        import sqlite3
        from src.config import get_db_path

        db_path = get_db_path()
        try:
            conn = sqlite3.connect(db_path)
            self.player_stats = pd.read_sql("SELECT * FROM player_stats", conn)
            self.teams = pd.read_sql("SELECT * FROM teams", conn)
            self.players = pd.read_sql("SELECT * FROM players", conn)
            self.team_metrics = pd.read_sql("SELECT * FROM team_metrics", conn)
            self.player_impact = pd.read_sql("SELECT * FROM player_impact", conn)
            self.injury_history = pd.read_sql("SELECT * FROM injury_history", conn)
            conn.close()
            self._loaded = True
            logger.info("Memory store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load memory store: {e}")

    _reload_lock = threading.Lock()

    def reload(self):
        """Reload all data (thread-safe: concurrent readers see old or new, never partial)."""
        with self._reload_lock:
            self._loaded = False
            self.precomputed_games = None
            self.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_team_abbrs(self) -> Dict[int, str]:
        """Return {team_id: abbreviation} mapping."""
        if self.teams is None:
            return {}
        return dict(zip(self.teams["team_id"], self.teams["abbreviation"]))

    def get_team_list(self) -> List[Dict[str, Any]]:
        """Return list of team dicts."""
        if self.teams is None:
            return []
        return self.teams.to_dict("records")

    def get_player_stats_for_team(self, team_id: int):
        """Get player stats filtered by team."""
        if self.player_stats is None or self.players is None:
            return None
        team_players = self.players[self.players["team_id"] == team_id]["player_id"].tolist()
        return self.player_stats[self.player_stats["player_id"].isin(team_players)]

    def get_team_metrics_dict(self, team_id: int) -> Dict[str, Any]:
        """Get metrics for a specific team as dict."""
        if self.team_metrics is None:
            return {}
        rows = self.team_metrics[self.team_metrics["team_id"] == team_id]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    def get_game_count_and_last_date(self) -> Tuple[int, str]:
        """Return (game_count, last_game_date) from player_stats."""
        if self.player_stats is None or self.player_stats.empty:
            return 0, ""
        count = len(self.player_stats)
        last = self.player_stats["game_date"].max()
        return count, str(last)


# Module-level convenience
def get_store() -> InMemoryDataStore:
    return InMemoryDataStore()
