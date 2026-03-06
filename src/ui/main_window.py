"""NBA Fundamentals V2 — Main window with tabbed layout."""

import logging
import os

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QApplication, QWidget,
    QVBoxLayout, QLabel, QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFontDatabase, QFont

from src.ui.theme import setup_theme

logger = logging.getLogger(__name__)


def _placeholder(label: str) -> QWidget:
    """Return a placeholder widget for tabs whose views aren't built yet."""
    w = QWidget()
    layout = QVBoxLayout(w)
    layout.setAlignment(Qt.AlignCenter)
    lbl = QLabel(f"{label}\n(coming soon)")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color: #94a3b8; font-size: 20px;")
    layout.addWidget(lbl)
    return w


class MainWindow(QMainWindow):
    """Main application window with 5 tabs."""

    def __init__(self):
        super().__init__()

        # Load bundled Oswald font
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        oswald_path = os.path.join(font_dir, "Oswald.ttf")
        if os.path.exists(oswald_path):
            font_id = QFontDatabase.addApplicationFont(oswald_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    app_font = QFont(families[0], 10)
                    QApplication.setFont(app_font)

        self.setWindowTitle("NBA Fundamentals V2")
        self.setMinimumSize(1200, 800)
        setup_theme(self)

        # Central widget with tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Initialize tabs (lazy imports to avoid circular deps)
        self._init_tabs()

        # Tab crossfade transition
        self._tab_fade_effect = None
        self._tab_fade_anim = None
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Notification bell
        self._init_notifications()

    def _init_tabs(self):
        """Create all five tabs with try/except fallbacks."""
        # Gamecast (live games — first tab)
        try:
            from src.ui.views.gamecast_view import GamecastView
            self.gamecast = GamecastView(self)
        except ImportError:
            logger.debug("GamecastView not available, using placeholder")
            self.gamecast = _placeholder("Gamecast")

        # Matchups
        try:
            from src.ui.views.matchup_view import MatchupView
            self.matchup = MatchupView(self)
        except ImportError:
            logger.debug("MatchupView not available, using placeholder")
            self.matchup = _placeholder("Matchups")

        # Accuracy
        try:
            from src.ui.views.accuracy_view import AccuracyView
            self.accuracy = AccuracyView(self)
        except ImportError:
            logger.debug("AccuracyView not available, using placeholder")
            self.accuracy = _placeholder("Accuracy")

        # Pipeline
        try:
            from src.ui.views.pipeline_view import PipelineView
            self.pipeline = PipelineView(self)
        except ImportError:
            logger.debug("PipelineView not available, using placeholder")
            self.pipeline = _placeholder("Pipeline")

        # Settings
        try:
            from src.ui.views.settings_view import SettingsView
            self.settings = SettingsView(self)
        except ImportError:
            logger.debug("SettingsView not available, using placeholder")
            self.settings = _placeholder("Settings")

        self.tabs.addTab(self.gamecast, "Gamecast")
        self.tabs.addTab(self.matchup, "Matchups")
        self.tabs.addTab(self.accuracy, "Accuracy")
        self.tabs.addTab(self.pipeline, "Pipeline")
        self.tabs.addTab(self.settings, "Settings")

    def _init_notifications(self):
        """Set up notification bell in the tab bar corner."""
        try:
            from src.ui.notification_widget import NotificationBell
            self.notif_bell = NotificationBell(self)
            self.tabs.setCornerWidget(self.notif_bell, Qt.Corner.TopRightCorner)
        except Exception as e:
            logger.warning("Failed to initialize notification bell: %s", e)

    def _on_tab_changed(self, index: int):
        """Apply a quick fade-in when switching tabs."""
        widget = self.tabs.widget(index)
        if not widget:
            return
        # Stop any running tab animation before starting a new one
        if self._tab_fade_anim is not None:
            self._tab_fade_anim.stop()
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.3)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(250)
        anim.setStartValue(0.3)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        # Store refs so they aren't garbage collected mid-animation
        self._tab_fade_effect = effect
        self._tab_fade_anim = anim
        anim.finished.connect(lambda w=widget: w.setGraphicsEffect(None))
        anim.start()

    def set_status(self, msg: str):
        """Update status bar message."""
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        """Clean up on close."""
        logger.info("Application closing")

        # If a pipeline worker is running, signal cancellation so it can
        # finish its current trial and run the save gate.
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, '_current_worker'):
            worker = getattr(self.pipeline, '_current_worker', None)
            if worker and hasattr(worker, 'isRunning') and worker.isRunning():
                try:
                    from src.analytics.pipeline import request_cancel
                    logger.info("Pipeline running -- requesting graceful stop...")
                    self.status_bar.showMessage("Saving optimization results...")
                    request_cancel()
                    worker.stop()
                    if hasattr(worker, '_thread_ref') and worker._thread_ref is not None:
                        worker._thread_ref.wait(10000)
                except Exception as e:
                    logger.warning("Error during graceful pipeline stop: %s", e)

        event.accept()
