"""TTL decorator, TeamCache, SessionCache utilities."""

import time
import functools
import threading
from typing import Any, Dict, Optional


def ttl_cache(seconds: int = 300):
    """Decorator that caches function results with a TTL."""
    def decorator(func):
        cache = {}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            with lock:
                if key in cache:
                    result, ts = cache[key]
                    if time.time() - ts < seconds:
                        return result
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, time.time())
            return result

        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator


class TeamCache:
    """Per-team data cache with TTL."""

    def __init__(self, ttl: int = 3600):
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._timestamps: Dict[int, float] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, team_id: int, key: str) -> Optional[Any]:
        with self._lock:
            if team_id in self._cache:
                if time.time() - self._timestamps.get(team_id, 0) < self._ttl:
                    return self._cache[team_id].get(key)
                else:
                    del self._cache[team_id]
        return None

    def set(self, team_id: int, key: str, value: Any):
        with self._lock:
            if team_id not in self._cache:
                self._cache[team_id] = {}
            self._cache[team_id][key] = value
            self._timestamps[team_id] = time.time()

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


class SessionCache:
    """Generic session cache, cleared on demand."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value

    def clear(self):
        with self._lock:
            self._data.clear()

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._data


# Global instances
team_cache = TeamCache()
session_cache = SessionCache()


def start_session_caches():
    """Clear and initialize session caches for a fresh run (e.g. backtest).

    NOTE: We only clear team_cache and session_cache here. Stats engine
    caches (player_splits, streak, fatigue, hca) are NOT cleared between
    backtest sessions because they hold immutable historical data that
    is valid across sessions. They are only cleared on data sync.
    """
    team_cache.clear()
    session_cache.clear()


def stop_session_caches():
    """Clear session caches after a run completes."""
    team_cache.clear()
    session_cache.clear()
