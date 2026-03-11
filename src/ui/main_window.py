"""NBA Fundamentals V2 — Main window with tabbed layout."""

import logging
import os

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QApplication, QWidget,
    QVBoxLayout, QLabel, QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFontDatabase, QFont, QShortcut, QKeySequence

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
        self.setMinimumSize(960, 640)
        self._apply_initial_window_size()
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
        self._tab_fade_widget = None
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Notification bell
        self._init_notifications()

        # Keyboard shortcuts (Alt+1..5 tabs, Ctrl+R refresh active tab)
        self._shortcuts = []
        self._init_shortcuts()

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

    def _apply_initial_window_size(self):
        """Choose a responsive startup size from primary screen geometry."""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 800)
            return
        geometry = screen.availableGeometry()
        target_width = min(geometry.width(), max(960, int(geometry.width() * 0.84)))
        target_height = min(geometry.height(), max(640, int(geometry.height() * 0.88)))
        self.resize(target_width, target_height)

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
        # Stop any active transition and aggressively clear old effects so
        # tabs cannot get stuck partially transparent when switching rapidly.
        if self._tab_fade_anim is not None:
            self._tab_fade_anim.stop()
        if self._tab_fade_widget is not None and self._tab_fade_widget is not widget:
            try:
                self._tab_fade_widget.setGraphicsEffect(None)
            except RuntimeError:
                logger.debug("Previous tab widget deleted during fade cleanup", exc_info=True)

        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.3)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(220)
        anim.setStartValue(0.3)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _cleanup(w=widget, eff=effect):
            try:
                if w.graphicsEffect() is eff:
                    w.setGraphicsEffect(None)
            except RuntimeError:
                logger.debug("Tab fade cleanup skipped for deleted widget", exc_info=True)

        anim.finished.connect(_cleanup)
        self._tab_fade_effect = effect
        self._tab_fade_anim = anim
        self._tab_fade_widget = widget
        anim.start()

    def _init_shortcuts(self):
        """Register tab navigation and refresh shortcuts."""
        for idx in range(min(5, self.tabs.count())):
            shortcut = QShortcut(QKeySequence(f"Alt+{idx + 1}"), self)
            shortcut.activated.connect(lambda i=idx: self.tabs.setCurrentIndex(i))
            self._shortcuts.append(shortcut)

        refresh_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        refresh_shortcut.activated.connect(self._refresh_active_tab)
        self._shortcuts.append(refresh_shortcut)

    def _refresh_active_tab(self):
        """Best-effort refresh action for current tab."""
        widget = self.tabs.currentWidget()
        if widget is None:
            return
        for method_name in (
            "_on_refresh",
            "_load_games",
            "_load_schedule",
            "_load_pipeline_state",
            "_update_weight_summary",
        ):
            method = getattr(widget, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    logger.debug("Shortcut refresh failed via %s", method_name, exc_info=True)
                return

    def set_status(self, msg: str):
        """Update status bar message."""
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        """Clean up on close."""
        logger.info("Application closing")

        # If a pipeline worker is running, request graceful cancellation and
        # wait briefly so save-gate logic can complete.
        pipeline_view = getattr(self, "pipeline", None)
        if pipeline_view and hasattr(pipeline_view, "request_stop"):
            try:
                from src.analytics.pipeline import request_cancel

                logger.info("Pipeline running -- requesting graceful stop...")
                self.status_bar.showMessage("Saving optimization results...")
                request_cancel()
                stopped = bool(pipeline_view.request_stop(timeout_ms=10000))
                if not stopped:
                    logger.warning("Pipeline threads did not stop within timeout")
            except Exception as e:
                logger.warning("Error during graceful pipeline stop: %s", e)

        gamecast_view = getattr(self, "gamecast", None)
        if gamecast_view and hasattr(gamecast_view, "request_stop"):
            try:
                gamecast_view.request_stop(timeout_ms=5000)
            except Exception as e:
                logger.warning("Error during gamecast shutdown: %s", e)

        for view_attr in ("matchup", "accuracy", "settings"):
            view = getattr(self, view_attr, None)
            if view and hasattr(view, "request_stop"):
                try:
                    view.request_stop(timeout_ms=5000)
                except Exception as e:
                    logger.warning("Error during %s shutdown: %s", view_attr, e)

        event.accept()
