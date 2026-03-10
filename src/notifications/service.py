"""Notification service — DB-persisted, listener pattern, push channels."""

import json
import logging
import subprocess
import sys
import threading
from typing import List, Dict, Any, Optional, Callable

from src.database import db
from src.notifications.models import NotificationCategory, NotificationSeverity

logger = logging.getLogger(__name__)

# Listeners for real-time UI updates
_listeners: List[Callable] = []
_listeners_lock = threading.Lock()


def add_listener(fn: Callable):
    """Register a notification listener."""
    with _listeners_lock:
        if fn not in _listeners:
            _listeners.append(fn)


def remove_listener(fn: Callable):
    """Remove a notification listener."""
    with _listeners_lock:
        if fn in _listeners:
            _listeners.remove(fn)


def _notify_listeners(notification: Dict):
    """Notify all registered listeners."""
    with _listeners_lock:
        snapshot = list(_listeners)
    for fn in snapshot:
        try:
            fn(notification)
        except Exception as e:
            logger.debug(f"Listener error: {e}")


def create_notification(category: str, severity: str, title: str,
                         message: str, data: Optional[Dict] = None) -> int:
    """Create and persist a notification."""
    data_json = json.dumps(data) if data else ""

    nid = db.execute_returning_id("""
        INSERT INTO notifications (category, severity, title, message, created_at, read, data)
        VALUES (?, ?, ?, ?, datetime('now'), 0, ?)
    """, (category, severity, title, message, data_json))

    notif = {
        "id": nid,
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "data": data,
    }

    _notify_listeners(notif)
    _push_notification(notif)

    return nid


def get_recent(limit: int = 30) -> List[Dict]:
    """Get recent notifications."""
    rows = db.fetch_all("""
        SELECT * FROM notifications
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    return [dict(r) for r in rows]


def get_unread_count() -> int:
    """Get count of unread notifications."""
    row = db.fetch_one("SELECT COUNT(*) as cnt FROM notifications WHERE read = 0")
    return row["cnt"] if row else 0


def mark_read(notification_id: int):
    """Mark a notification as read."""
    db.execute("UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,))


def mark_all_read():
    """Mark all notifications as read."""
    db.execute("UPDATE notifications SET read = 1 WHERE read = 0")


def delete_old(days: int = 30):
    """Delete notifications older than N days."""
    db.execute("""
        DELETE FROM notifications 
        WHERE created_at < datetime('now', ?)
    """, (f"-{days} days",))


def _push_notification(notif: Dict):
    """Send push notifications through configured channels."""
    from src.config import get_config
    config = get_config()

    # Webhook
    webhook_url = config.get("notification_webhook_url")
    if webhook_url:
        _push_webhook(webhook_url, notif)

    # ntfy
    ntfy_topic = config.get("notification_ntfy_topic")
    if ntfy_topic:
        _push_ntfy(ntfy_topic, notif)

    # Windows toast (BurntToast)
    if sys.platform == "win32" and config.get("notification_toast", False):
        _push_toast(notif)

    # plyer (cross-platform)
    if config.get("notification_plyer", False):
        _push_plyer(notif)


def _push_webhook(url: str, notif: Dict):
    """Send notification via webhook."""
    try:
        import requests
        requests.post(url, json=notif, timeout=5)
    except Exception as e:
        logger.debug(f"Webhook push failed: {e}")


def _push_ntfy(topic: str, notif: Dict):
    """Send notification via ntfy.sh."""
    try:
        import requests
        priority = "high" if notif.get("severity") == "critical" else "default"
        requests.post(
            f"https://ntfy.sh/{topic}",
            data=notif.get("message", ""),
            headers={
                "Title": notif.get("title", "NBA Alert"),
                "Priority": priority,
                "Tags": notif.get("category", "info"),
            },
            timeout=5,
        )
    except Exception as e:
        logger.debug(f"ntfy push failed: {e}")


def _push_toast(notif: Dict):
    """Send Windows toast via BurntToast PowerShell module."""
    try:
        title = notif.get("title", "NBA Alert")
        message = notif.get("message", "")
        # Escape PowerShell special characters to prevent injection
        for char in ('`', '"', '$', ';', '|', '&'):
            esc = '`' + char
            title = title.replace(char, esc)
            message = message.replace(char, esc)
        cmd = f'New-BurntToastNotification -Text "{title}", "{message}"'
        subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True, timeout=5,
        )
    except Exception as e:
        logger.debug(f"Toast push failed: {e}")


def _push_plyer(notif: Dict):
    """Send notification via plyer."""
    try:
        from plyer import notification as plyer_notif
        plyer_notif.notify(
            title=notif.get("title", "NBA Alert"),
            message=notif.get("message", ""),
            app_name="NBA Predictor",
            timeout=10,
        )
    except Exception as e:
        logger.debug(f"plyer push failed: {e}")
