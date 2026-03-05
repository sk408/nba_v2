"""Notification bell and panel for the desktop UI.

NotificationBell — custom painted badge with unread count.
NotificationPanel — popup showing 30 recent, severity colors, mark-all-read.
"""

import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, QTimer, QPoint, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QPen

logger = logging.getLogger(__name__)

_SEVERITY_COLORS = {
    "critical": QColor(239, 68, 68),   # Red
    "warning": QColor(251, 191, 36),    # Amber
    "info": QColor(59, 130, 246),       # Blue
}


class NotificationBell(QPushButton):
    """Notification bell button with painted badge count."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(40, 40))
        self.setText("🔔")
        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 18px; }"
        )
        self._count = 0
        self._panel = NotificationPanel(parent, bell=self)
        self._panel.hide()

        self.clicked.connect(self._toggle_panel)

        # Poll for new notifications every 30s
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(30_000)
        self._poll()

    def _poll(self):
        """Check unread count."""
        try:
            from src.notifications.service import get_unread_count
            self._count = get_unread_count()
            self.update()
        except Exception:
            pass

    def _toggle_panel(self):
        if self._panel.isVisible():
            self._panel.hide()
        else:
            # Position below the bell
            pos = self.mapToGlobal(QPoint(0, self.height()))
            self._panel.move(pos.x() - 280, pos.y())
            self._panel.refresh()
            self._panel.show()
            self._panel.raise_()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._count > 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Red badge circle
            painter.setBrush(QColor(239, 68, 68))
            painter.setPen(QPen(Qt.PenStyle.NoPen))
            badge_size = 16
            x = self.width() - badge_size - 2
            y = 2
            painter.drawEllipse(x, y, badge_size, badge_size)

            # Count text
            painter.setPen(QColor(255, 255, 255))
            font = QFont()
            font.setPixelSize(10)
            font.setBold(True)
            painter.setFont(font)
            text = str(min(self._count, 99))
            painter.drawText(x, y, badge_size, badge_size, Qt.AlignmentFlag.AlignCenter, text)
            painter.end()


class NotificationPanel(QFrame):
    """Popup panel showing recent notifications."""

    def __init__(self, parent=None, bell=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.bell = bell
        self.setFixedSize(320, 400)
        self.setStyleSheet(
            "NotificationPanel { background: #1e293b; border: 1px solid #334155; "
            "border-radius: 8px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Notifications"))
        mark_btn = QPushButton("Mark All Read")
        mark_btn.setStyleSheet("font-size: 11px;")
        mark_btn.clicked.connect(self._mark_all_read)
        h_layout.addWidget(mark_btn)
        layout.addLayout(h_layout)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(4)
        scroll.setWidget(self.content)
        layout.addWidget(scroll)

    def refresh(self):
        """Load recent notifications."""
        # Clear existing
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        try:
            from src.notifications.service import get_recent
            notifications = get_recent(30)
            for n in notifications:
                card = self._make_card(n)
                self.content_layout.addWidget(card)

            if not notifications:
                empty = QLabel("No notifications")
                empty.setStyleSheet("color: #64748b; padding: 20px;")
                empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.content_layout.addWidget(empty)
        except Exception as e:
            err = QLabel(f"Error: {e}")
            err.setStyleSheet("color: #ef4444;")
            self.content_layout.addWidget(err)

    def _make_card(self, notification) -> QFrame:
        """Create a notification card."""
        frame = QFrame()
        is_read = bool(notification.get("read", 0))
        bg_color = "#0f172a" if not is_read else "#0b0f19"
        border_color = "#334155" if not is_read else "#1e293b"
        
        frame.setStyleSheet(
            f"QFrame {{ background: {bg_color}; border: 1px solid {border_color}; "
            f"border-radius: 4px; padding: 6px; }}"
        )
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(6, 4, 6, 4)
        fl.setSpacing(2)

        # Severity indicator + title
        severity = notification.get("severity", "info")
        if isinstance(severity, str):
            sev_name = severity
        else:
            sev_name = severity.value if hasattr(severity, "value") else str(severity)

        color = _SEVERITY_COLORS.get(sev_name, _SEVERITY_COLORS["info"])
        # Dim the color if read
        if is_read:
            color = QColor(color)
            color.setAlpha(100)

        title = notification.get("title", "")
        title_label = QLabel(f"● {title}")
        title_label.setStyleSheet(f"color: {color.name(QColor.NameFormat.HexArgb)}; font-weight: 600; font-size: 12px;")
        fl.addWidget(title_label)

        # Message
        message = notification.get("message", "")
        if message:
            msg_label = QLabel(message)
            msg_color = "#94a3b8" if not is_read else "#475569"
            msg_label.setStyleSheet(f"color: {msg_color}; font-size: 11px;")
            msg_label.setWordWrap(True)
            fl.addWidget(msg_label)

        # Timestamp
        ts = notification.get("created_at", "")
        if ts:
            ts_label = QLabel(str(ts))
            ts_label.setStyleSheet("color: #475569; font-size: 10px;")
            fl.addWidget(ts_label)

        return frame

    def _mark_all_read(self):
        try:
            from src.notifications.service import mark_all_read
            mark_all_read()
            self.refresh()
            if self.bell and hasattr(self.bell, "_poll"):
                self.bell._poll()
            elif hasattr(self.parent(), "_poll"):
                self.parent()._poll()
        except Exception as e:
            logger.error(f"Mark all read error: {e}")
