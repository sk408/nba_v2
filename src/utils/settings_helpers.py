"""Shared typed setting readers."""

from src.config import get as get_setting


def safe_float_setting(key: str, default: float) -> float:
    try:
        return float(get_setting(key, default))
    except (TypeError, ValueError):
        return float(default)


def safe_int_setting(key: str, default: int) -> int:
    try:
        return int(get_setting(key, default))
    except (TypeError, ValueError):
        return int(default)


def safe_bool_setting(key: str, default: bool) -> bool:
    raw = get_setting(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("1", "true", "yes", "on", "y"):
            return True
        if v in ("0", "false", "no", "off", "n"):
            return False
    return bool(default)
