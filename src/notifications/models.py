"""Notification models and constants."""

from enum import Enum


class NotificationCategory(str, Enum):
    INJURY = "injury"
    UNDERDOG = "underdog"
    ODDS = "odds"


class NotificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
