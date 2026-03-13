"""Central timezone utilities for NBA game-date matching and display.

The NBA API stores game dates in Eastern Time (ET).  The helpers here
ensure that "today" comparisons always use the ET date, and that
user-facing times are converted to the configured display timezone.
"""

from datetime import datetime, timezone
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
    """Today's date in Eastern Time as YYYY-MM-DD.

    Use this instead of ``datetime.now().strftime("%Y-%m-%d")`` whenever
    comparing against the DB ``game_date`` column.
    """
    return datetime.now(tz=_ET).strftime("%Y-%m-%d")


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
