"""Central timezone utilities for NBA game-date matching and display.

The NBA API stores game dates in Eastern Time (ET).  The helpers here
ensure that "today" comparisons always use the ET date, and that
user-facing times are converted to the configured display timezone.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src import config

_ET = ZoneInfo("America/New_York")

# US timezone choices for the settings dropdown: (display_label, IANA key)
TIMEZONE_CHOICES = [
    ("Eastern (ET)", "America/New_York"),
    ("Central (CT)", "America/Chicago"),
    ("Mountain (MT)", "America/Denver"),
    ("Pacific (PT)", "America/Los_Angeles"),
    ("Alaska (AKT)", "America/Anchorage"),
    ("Hawaii (HT)", "Pacific/Honolulu"),
]


def nba_today() -> str:
    """Return the current NBA "game date" as YYYY-MM-DD.

    The NBA day doesn't roll over at midnight — late games on the West Coast
    can run past 1 AM ET.  We keep showing the current slate until 6 AM ET
    so the dashboard / gamecast never flip to tomorrow's empty card while
    tonight's games are still finishing.

    Use this instead of ``datetime.now().strftime("%Y-%m-%d")`` whenever
    comparing against the DB ``game_date`` column.
    """
    return (datetime.now(tz=_ET) - timedelta(hours=6)).strftime("%Y-%m-%d")


def nba_tomorrow() -> str:
    """Return the NBA "next game date" as YYYY-MM-DD.

    Uses the same 6 AM ET rollover logic as ``nba_today`` but adds one day.
    """
    return (datetime.now(tz=_ET) - timedelta(hours=6) + timedelta(days=1)).strftime("%Y-%m-%d")


def nba_game_date_from_utc_iso(raw_iso: str) -> str:
    """Convert an ESPN-style UTC timestamp to the NBA game date.

    ESPN competition timestamps are UTC (often next-day for late West games).
    We convert to ET and apply the same 6 AM rollover rule used by nba_today().
    """
    text = str(raw_iso or "").strip()
    if not text:
        return nba_today()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        # Fallback for unexpected formats that still include YYYY-MM-DD.
        if len(text) >= 10:
            return text[:10]
        return nba_today()
    return (dt.astimezone(_ET) - timedelta(hours=6)).strftime("%Y-%m-%d")


def nba_now() -> datetime:
    """Current datetime in Eastern Time (timezone-aware).

    Useful for date-range calculations (e.g. "next 15 days from ET today").
    """
    return datetime.now(tz=_ET)


def display_tz() -> ZoneInfo:
    """Return the user's configured display timezone as a ``ZoneInfo``."""
    tz_key = config.get("timezone", "US/Pacific")
    try:
        return ZoneInfo(tz_key)
    except (KeyError, Exception):
        return ZoneInfo("America/Los_Angeles")


def to_display_tz(dt: datetime) -> datetime:
    """Convert a UTC-aware datetime to the user's display timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(display_tz())


def et_to_display(dt: datetime) -> datetime:
    """Convert an ET-aware datetime to the user's display timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_ET)
    return dt.astimezone(display_tz())
