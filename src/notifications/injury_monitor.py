"""Injury monitoring — background diff engine, polls every 5 minutes."""

import logging
import threading
import time
from typing import Dict, Set, Optional

from src.database import db
from src.data.injury_scraper import scrape_all_injuries
from src.notifications.service import create_notification
from src.notifications.models import NotificationCategory, NotificationSeverity

logger = logging.getLogger(__name__)

HIGH_IMPACT_MPG = 20.0
POLL_INTERVAL = 300  # 5 minutes


class InjuryMonitor:
    """Background thread that monitors injury changes."""

    def __init__(self):
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._previous_state: Dict[int, Dict] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start the monitoring loop."""
        if self._running:
            return
        self._running = True
        self._load_initial_state()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Injury monitor started")

    def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        self._stop_event.set()  # wake from sleep immediately
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Injury monitor stopped")

    def _load_initial_state(self):
        """Load current injury state from DB."""
        rows = db.fetch_all("""
            SELECT i.player_id, i.player_name, i.team_id, i.status, i.reason,
                   COALESCE((SELECT AVG(minutes) FROM (
                       SELECT minutes FROM player_stats ps
                       WHERE ps.player_id = i.player_id
                       ORDER BY ps.game_date DESC LIMIT 20)), 0) as mpg,
                   COALESCE((SELECT AVG(points) FROM (
                       SELECT points FROM player_stats ps
                       WHERE ps.player_id = i.player_id
                       ORDER BY ps.game_date DESC LIMIT 20)), 0) as ppg
            FROM injuries i
        """)
        with self._lock:
            self._previous_state = {
                r["player_id"]: dict(r) for r in rows
            }

    def _loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Interruptible sleep — wakes immediately when stop_event is set
                if self._stop_event.wait(timeout=POLL_INTERVAL):
                    break  # stop requested
                if not self._running:
                    break
                self._check_changes()
            except Exception as e:
                logger.warning(f"Injury monitor error: {e}")

    def _check_changes(self):
        """Check for injury changes and create notifications."""
        # Scrape current injuries
        try:
            scrape_all_injuries()
        except Exception as e:
            logger.debug(f"Scrape failed in monitor: {e}")
            return

        # Get current state
        rows = db.fetch_all("""
            SELECT i.player_id, i.player_name, i.team_id, i.status, i.reason,
                   COALESCE((SELECT AVG(minutes) FROM (
                       SELECT minutes FROM player_stats ps
                       WHERE ps.player_id = i.player_id
                       ORDER BY ps.game_date DESC LIMIT 20)), 0) as mpg,
                   COALESCE((SELECT AVG(points) FROM (
                       SELECT points FROM player_stats ps
                       WHERE ps.player_id = i.player_id
                       ORDER BY ps.game_date DESC LIMIT 20)), 0) as ppg,
                   t.abbreviation
            FROM injuries i
            LEFT JOIN teams t ON i.team_id = t.team_id
        """)

        current_state = {r["player_id"]: dict(r) for r in rows}

        with self._lock:
            prev = self._previous_state.copy()

        # Detect changes
        # New injuries
        for pid, current in current_state.items():
            if pid not in prev:
                self._notify_new_injury(current)
            elif prev[pid].get("status") != current.get("status"):
                self._notify_status_change(prev[pid], current)

        # Removed from injury report (recovered)
        for pid, previous in prev.items():
            if pid not in current_state:
                self._notify_recovered(previous)

        # Update state
        with self._lock:
            self._previous_state = current_state

    def _get_severity(self, player: Dict) -> str:
        """Determine notification severity based on player impact."""
        mpg = player.get("mpg") or 0
        if mpg >= HIGH_IMPACT_MPG:
            return NotificationSeverity.CRITICAL
        elif mpg >= 15:
            return NotificationSeverity.WARNING
        return NotificationSeverity.INFO

    def _notify_new_injury(self, player: Dict):
        """Create notification for new injury."""
        severity = self._get_severity(player)
        abbr = player.get("abbreviation") or "???"
        name = player.get("player_name", "Unknown")
        status = player.get("status", "Unknown")
        reason = player.get("reason", "")
        mpg = player.get("mpg", 0) or 0

        impact = "HIGH IMPACT" if mpg >= HIGH_IMPACT_MPG else ""
        title = f"New Injury: {name} ({abbr})"
        message = f"{status} - {reason}. {mpg:.0f} MPG. {impact}".strip()

        create_notification(
            category=NotificationCategory.INJURY,
            severity=severity,
            title=title,
            message=message,
            data={"player_id": player["player_id"], "team_id": player["team_id"]},
        )

    def _notify_status_change(self, previous: Dict, current: Dict):
        """Create notification for status change."""
        severity = self._get_severity(current)
        name = current.get("player_name", "Unknown")
        abbr = current.get("abbreviation") or "???"

        old_status = previous.get("status", "?")
        new_status = current.get("status", "?")

        title = f"Status Change: {name} ({abbr})"
        message = f"{old_status} → {new_status}"

        create_notification(
            category=NotificationCategory.INJURY,
            severity=severity,
            title=title,
            message=message,
            data={"player_id": current["player_id"], "team_id": current["team_id"]},
        )

    def _notify_recovered(self, player: Dict):
        """Create notification for injury removal (recovery)."""
        name = player.get("player_name", "Unknown")
        abbr = player.get("abbreviation") or "???"

        title = f"Cleared: {name} ({abbr})"
        message = f"Removed from injury report (was {player.get('status', '?')})"

        create_notification(
            category=NotificationCategory.INJURY,
            severity=NotificationSeverity.INFO,
            title=title,
            message=message,
            data={"player_id": player["player_id"], "team_id": player["team_id"]},
        )


# Global singleton
_monitor: Optional[InjuryMonitor] = None


def get_injury_monitor() -> InjuryMonitor:
    global _monitor
    if _monitor is None:
        _monitor = InjuryMonitor()
    return _monitor
