"""Notification models and constants."""

from enum import Enum


class NotificationCategory(str, Enum):
    INJURY = "injury"
    UNDERDOG = "underdog"


class NotificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
