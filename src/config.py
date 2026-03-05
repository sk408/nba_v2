"""JSON-backed application settings at data/app_settings.json."""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

_SETTINGS_PATH = Path("data") / "app_settings.json"
_settings_lock = threading.Lock()

_DEFAULTS: Dict[str, Any] = {
    "db_path": "data/nba_analytics.db",
    "season": "2025-26",
    "season_year": "2025",
    "historical_seasons": ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"],
    "theme": "dark",
    "auto_sync_interval_minutes": 60,
    "notification_webhook_url": "",
    "notification_ntfy_topic": "",
    "enable_toast_notifications": True,
    "log_level": "INFO",
    "worker_threads": max(1, (os.cpu_count() or 4) - 2),
    "oled_mode": False,
    "sync_freshness_hours": 4,
    "optimizer_log_interval": 300,
    "prediction_mode": "fundamentals",  # "fundamentals" or "fundamentals_sharp"
    "upset_bonus_mult": 0.5,  # optimizer upset reward multiplier
}

_cache: Dict[str, Any] | None = None


def _ensure_dir():
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    """Load settings from disk, merging with defaults."""
    global _cache
    with _settings_lock:
        if _cache is not None:
            return _cache
        _ensure_dir()
        if _SETTINGS_PATH.exists():
            try:
                with open(_SETTINGS_PATH, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        else:
            data = {}
        merged = {**_DEFAULTS, **data}
        _cache = merged
        return merged


def save_settings(settings: Dict[str, Any] | None = None):
    """Persist current settings to disk."""
    global _cache
    with _settings_lock:
        if settings is not None:
            _cache = settings
        if _cache is None:
            _cache = dict(_DEFAULTS)
        _ensure_dir()
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(_cache, f, indent=2)


def get(key: str, default: Any = None) -> Any:
    s = load_settings()
    return s.get(key, default)


def set_value(key: str, value: Any):
    s = load_settings()
    s[key] = value
    save_settings(s)


def get_db_path() -> str:
    return get("db_path", _DEFAULTS["db_path"])


def get_season() -> str:
    return get("season", _DEFAULTS["season"])


def get_season_year() -> str:
    return get("season_year", _DEFAULTS["season_year"])


def invalidate_cache():
    global _cache
    with _settings_lock:
        _cache = None


def get_historical_seasons() -> list:
    return get("historical_seasons", [])


def get_config() -> Dict[str, Any]:
    """Return the full settings dict (alias for load_settings)."""
    return load_settings()
