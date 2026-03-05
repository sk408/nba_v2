"""Accuracy tab -- backtest dashboard with A/B comparison.

Shows winner accuracy, upset metrics, fundamentals vs fundamentals+sharp
comparison, ML ROI, and sharp money impact analysis.
"""

import logging
from typing import Any, Dict, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QSizePolicy,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QTimer,
    QPropertyAnimation, QEasingCurve,
)

from src.ui.theme import apply_card_shadow

logger = logging.getLogger(__name__)

# ─── Theme colors ──────────────────────────────────────────
CYAN = "#00E5FF"
GREEN = "#00E676"
AMBER = "#FFB300"
RED = "#FF5252"
GRAY = "#78909C"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"


# ─────────────────────────────────────────────────────────────
# MetricCard -- broadcast-styled value card (reuses PredictionCard pattern)
# ─────────────────────────────────────────────────────────────

class _MetricCard(QFrame):
    """Broadcast-styled card showing a single metric value."""

    def __init__(self, title: str, value: str = "\u2014", accent: str = CYAN,
                 min_width: int = 140, min_height: int = 90):
        super().__init__()
        self.setProperty("class", "broadcast-card")
        self.setMinimumHeight(min_height)
        self.setMinimumWidth(min_width)
        self._accent = accent

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 8, 12, 8)

        self.title_label = QLabel(title.upper())
        self.title_label.setProperty("class", "stat-label")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(
            f"color: {accent}; font-size: 26px; font-weight: 700; "
            f"font-family: 'Oswald'; background: transparent;"
        )
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

        self._opacity_effect = None
        apply_card_shadow(self)

    def animate_in(self, delay_ms: int = 0):
        """Fade in with optional delay for stagger effect."""
        if self._opacity_effect is not None:
            self.setGraphicsEffect(None)
            self._opacity_effect = None
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._anim.setDuration(400)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(lambda: self.setGraphicsEffect(None))
        if delay_ms > 0:
            QTimer.singleShot(delay_ms, self._anim.start)
        else:
            self._anim.start()

    def set_value(self, val: str, color: str = None):
        if color:
            self.value_label.setStyleSheet(
                f"color: {color}; font-size: 26px; font-weight: 700; "
                f"font-family: 'Oswald'; background: transparent;"
            )
        self.value_label.setText(val)


# ─────────────────────────────────────────────────────────────
# A/B Column Card -- stacked metrics in a single broadcast card
# ─────────────────────────────────────────────────────────────

class _ABColumnCard(QFrame):
    """Broadcast-styled card for one side of the A/B comparison."""

    def __init__(self, title: str, accent: str = CYAN):
        super().__init__()
        self.setProperty("class", "broadcast-card")
        self.setMinimumWidth(240)
        self._accent = accent

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 12, 16, 12)

        # Title
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet(
            f"color: {accent}; font-size: 14px; font-weight: 700; "
            f"font-family: 'Oswald'; letter-spacing: 1px; background: transparent;"
        )
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        # Separator line
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {accent}; opacity: 0.4;")
        layout.addWidget(sep)

        # Metric rows (label: value pairs)
        self._rows: Dict[str, QLabel] = {}
        self._grid = QGridLayout()
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(0, 4, 0, 4)
        layout.addLayout(self._grid)
        self._row_count = 0

        apply_card_shadow(self)

    def add_row(self, key: str, label_text: str, value: str = "\u2014",
                value_color: str = TEXT_PRIMARY) -> QLabel:
        """Add a label-value row and return the value QLabel for updates."""
        lbl = QLabel(label_text)
        lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px; font-weight: 600; "
            f"text-transform: uppercase; background: transparent;"
        )
        val = QLabel(value)
        val.setStyleSheet(
            f"color: {value_color}; font-size: 18px; font-weight: 700; "
            f"font-family: 'Oswald'; background: transparent;"
        )
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._grid.addWidget(lbl, self._row_count, 0)
        self._grid.addWidget(val, self._row_count, 1)
        self._rows[key] = val
        self._row_count += 1
        return val

    def set_row_value(self, key: str, value: str, color: str = None):
        """Update the value of an existing row."""
        lbl = self._rows.get(key)
        if lbl:
            lbl.setText(value)
            if color:
                lbl.setStyleSheet(
                    f"color: {color}; font-size: 18px; font-weight: 700; "
                    f"font-family: 'Oswald'; background: transparent;"
                )


# ─────────────────────────────────────────────────────────────
# Backtest Worker
# ─────────────────────────────────────────────────────────────

class _BacktestWorker(QObject):
    """Background worker to run full backtest."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def run(self):
        try:
            from src.analytics.backtester import run_backtest
            from src.database.db import thread_local_db
            with thread_local_db():
                result = run_backtest(callback=lambda msg: self.progress.emit(msg))
            self.finished.emit(result)
        except Exception as e:
            logger.error("BacktestWorker error: %s", e, exc_info=True)
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────
# AccuracyView -- main widget
# ─────────────────────────────────────────────────────────────

class AccuracyView(QWidget):
    """Backtest accuracy dashboard with A/B comparison."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[_BacktestWorker] = None
        self._last_results: Optional[Dict[str, Any]] = None

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # Wrap everything in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(12)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Header
        header = QLabel("Backtest Accuracy")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # ── Section 1: Winner Accuracy ──
        self._build_winner_section(layout)

        # ── Section 2: Upset Analysis ──
        self._build_upset_section(layout)

        # ── Section 3: A/B Comparison ──
        self._build_ab_section(layout)

        # ── Section 4: Sharp Impact Summary ──
        self._build_sharp_impact_section(layout)

        # ── Section 5: Refresh button + status ──
        self._build_controls(layout)

        layout.addStretch()

        # Try to load cached results on init (non-blocking)
        QTimer.singleShot(300, self._try_load_cached)

    # ──────────────────────────────────────────────────────────
    # UI builders
    # ──────────────────────────────────────────────────────────

    def _build_winner_section(self, parent_layout: QVBoxLayout):
        """Build the primary winner accuracy metrics panel."""
        section_lbl = QLabel("Winner Accuracy")
        section_lbl.setProperty("class", "section-title")
        parent_layout.addWidget(section_lbl)

        row = QHBoxLayout()
        row.setSpacing(10)

        self._winner_card = _MetricCard("Winner %", accent=CYAN)
        self._winner_card.setToolTip("Model's straight-up winner accuracy")
        row.addWidget(self._winner_card)

        self._favorites_card = _MetricCard("Favorites %", accent=GRAY)
        self._favorites_card.setToolTip("Always-pick-favorites baseline accuracy")
        row.addWidget(self._favorites_card)

        self._beats_fav_card = _MetricCard("vs Favorites", accent=GREEN)
        self._beats_fav_card.setToolTip("Does the model beat the favorites baseline?")
        row.addWidget(self._beats_fav_card)

        self._total_games_card = _MetricCard("Total Games", accent=CYAN)
        self._total_games_card.setToolTip("Total games evaluated in backtest")
        row.addWidget(self._total_games_card)

        parent_layout.addLayout(row)

    def _build_upset_section(self, parent_layout: QVBoxLayout):
        """Build the upset analysis metrics panel."""
        section_lbl = QLabel("Upset Analysis")
        section_lbl.setProperty("class", "section-title")
        parent_layout.addWidget(section_lbl)

        row = QHBoxLayout()
        row.setSpacing(10)

        self._upset_rate_card = _MetricCard("Upset Rate", accent=AMBER)
        self._upset_rate_card.setToolTip(
            "What % of games does the model pick against the favorite")
        row.addWidget(self._upset_rate_card)

        self._upset_acc_card = _MetricCard("Upset Accuracy", accent=AMBER)
        self._upset_acc_card.setToolTip(
            "When it picks an upset, how often is it correct")
        row.addWidget(self._upset_acc_card)

        self._upset_correct_card = _MetricCard("Correct Upsets", accent=AMBER)
        self._upset_correct_card.setToolTip("Count of upset picks that were right")
        row.addWidget(self._upset_correct_card)

        parent_layout.addLayout(row)

    def _build_ab_section(self, parent_layout: QVBoxLayout):
        """Build the A/B comparison panel with side-by-side cards."""
        section_lbl = QLabel("A/B Comparison")
        section_lbl.setProperty("class", "section-title")
        parent_layout.addWidget(section_lbl)

        row = QHBoxLayout()
        row.setSpacing(12)

        # Fundamentals-only column
        self._fund_col = _ABColumnCard("Fundamentals Only", accent=CYAN)
        self._fund_col.add_row("winner", "Winner %", value_color=CYAN)
        self._fund_col.add_row("upset_rate", "Upset Rate", value_color=AMBER)
        self._fund_col.add_row("upset_acc", "Upset Accuracy", value_color=AMBER)
        self._fund_col.add_row("ml_roi", "ML ROI", value_color=GREEN)
        row.addWidget(self._fund_col)

        # Fundamentals + Sharp column
        self._sharp_col = _ABColumnCard("Fundamentals + Sharp", accent="#2196F3")
        self._sharp_col.add_row("winner", "Winner %", value_color=CYAN)
        self._sharp_col.add_row("upset_rate", "Upset Rate", value_color=AMBER)
        self._sharp_col.add_row("upset_acc", "Upset Accuracy", value_color=AMBER)
        self._sharp_col.add_row("ml_roi", "ML ROI", value_color=GREEN)
        row.addWidget(self._sharp_col)

        parent_layout.addLayout(row)

    def _build_sharp_impact_section(self, parent_layout: QVBoxLayout):
        """Build the sharp money impact summary row."""
        self._sharp_impact_frame = QFrame()
        self._sharp_impact_frame.setProperty("class", "broadcast-card")
        impact_layout = QVBoxLayout(self._sharp_impact_frame)
        impact_layout.setContentsMargins(16, 10, 16, 10)
        impact_layout.setSpacing(6)

        impact_title = QLabel("SHARP MONEY IMPACT")
        impact_title.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px; font-weight: 700; "
            f"text-transform: uppercase; letter-spacing: 1px; background: transparent;"
        )
        impact_layout.addWidget(impact_title)

        self._sharp_summary_lbl = QLabel(
            "No backtest data \u2014 click Refresh to run backtest")
        self._sharp_summary_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 14px; background: transparent;"
        )
        self._sharp_summary_lbl.setWordWrap(True)
        impact_layout.addWidget(self._sharp_summary_lbl)

        self._sharp_net_lbl = QLabel("")
        self._sharp_net_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 13px; background: transparent;"
        )
        self._sharp_net_lbl.setVisible(False)
        impact_layout.addWidget(self._sharp_net_lbl)

        apply_card_shadow(self._sharp_impact_frame)
        parent_layout.addWidget(self._sharp_impact_frame)

    def _build_controls(self, parent_layout: QVBoxLayout):
        """Build the refresh button and progress label."""
        controls = QHBoxLayout()
        controls.setSpacing(12)

        self._refresh_btn = QPushButton("REFRESH BACKTEST")
        self._refresh_btn.setProperty("class", "primary")
        self._refresh_btn.setFixedHeight(44)
        self._refresh_btn.setMinimumWidth(200)
        self._refresh_btn.clicked.connect(self._on_refresh)
        controls.addWidget(self._refresh_btn)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px; "
            f"font-family: 'Segoe UI', sans-serif; background: transparent;"
        )
        self._progress_lbl.setWordWrap(True)
        self._progress_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        controls.addWidget(self._progress_lbl)

        parent_layout.addLayout(controls)

    # ──────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────

    def _try_load_cached(self):
        """Attempt to load backtest results from the backtester's cache.

        This is quick (disk/memory) and avoids a full backtest run on init.
        """
        try:
            from src.analytics.backtester import (
                _mem_cache, _mem_cache_lock, _load_disk_cache,
                _cache_hash,
            )
            from src.analytics.prediction import precompute_all_games
            from src.analytics.weight_config import get_weight_config

            # Check memory cache first
            with _mem_cache_lock:
                if _mem_cache is not None:
                    self._last_results = _mem_cache
                    self._display_results(_mem_cache)
                    self._progress_lbl.setText("Loaded from memory cache")
                    return

            # Try disk cache -- we need to compute the hash
            # Just estimate game count from disk cache files
            import os
            import glob as glob_mod
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "data", "backtest_cache"
            )
            cache_files = glob_mod.glob(os.path.join(cache_dir, "backtest_*.pkl"))
            if cache_files:
                # Load the most recent cache file directly
                import pickle
                import time as time_mod
                newest = max(cache_files, key=os.path.getmtime)
                age = time_mod.time() - os.path.getmtime(newest)
                if age < 3600:  # 1hr TTL
                    with open(newest, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and "fundamentals" in data:
                        self._last_results = data
                        self._display_results(data)
                        self._progress_lbl.setText(
                            f"Loaded from disk cache ({age:.0f}s old)")
                        return

            self._progress_lbl.setText(
                "No backtest data \u2014 click Refresh to run backtest")
        except Exception as e:
            logger.debug("No cached backtest data available: %s", e)
            self._progress_lbl.setText(
                "No backtest data \u2014 click Refresh to run backtest")

    # ──────────────────────────────────────────────────────────
    # Refresh backtest
    # ──────────────────────────────────────────────────────────

    def _on_refresh(self):
        """Start a backtest run in a background thread."""
        # Busy guard
        try:
            if self._worker_thread is not None and self._worker_thread.isRunning():
                return
        except RuntimeError:
            self._worker_thread = None

        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("Running...")
        self._progress_lbl.setText("Starting backtest...")

        if self.main_window:
            self.main_window.set_status("Running backtest...")

        self._worker = _BacktestWorker()
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)

        _QC = Qt.ConnectionType.QueuedConnection
        self._worker.finished.connect(self._on_backtest_done, _QC)
        self._worker.error.connect(self._on_backtest_error, _QC)
        self._worker.progress.connect(self._on_backtest_progress, _QC)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _on_backtest_progress(self, msg: str):
        """Update progress label from worker thread."""
        self._progress_lbl.setText(msg)

    def _on_backtest_done(self, results: dict):
        """Handle completed backtest results."""
        self._refresh_btn.setEnabled(True)
        self._refresh_btn.setText("REFRESH BACKTEST")
        self._last_results = results
        self._display_results(results)

        total = results.get("fundamentals", {}).get("total_games", 0)
        self._progress_lbl.setText(f"Backtest complete: {total} games evaluated")

        if self.main_window:
            self.main_window.set_status(
                f"Backtest complete: {total} games evaluated")

    def _on_backtest_error(self, msg: str):
        """Handle backtest error."""
        self._refresh_btn.setEnabled(True)
        self._refresh_btn.setText("REFRESH BACKTEST")
        self._progress_lbl.setText(f"Error: {msg}")
        logger.error("Backtest error: %s", msg)

        if self.main_window:
            self.main_window.set_status(f"Backtest error: {msg}")

    def _cleanup_worker(self):
        """Clean up worker and thread references."""
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker_thread = None
        self._worker = None

    # ──────────────────────────────────────────────────────────
    # Display results
    # ──────────────────────────────────────────────────────────

    def _display_results(self, results: Dict[str, Any]):
        """Populate all panels from backtest results dict."""
        fund = results.get("fundamentals", {})
        sharp = results.get("sharp", {})
        comp = results.get("comparison", {})

        if not fund:
            return

        # ── Section 1: Winner Accuracy ──
        winner_pct = fund.get("winner_pct", 0)
        fav_pct = fund.get("favorites_pct", 0)
        beats_fav = fund.get("beats_favorites", False)
        total_games = fund.get("total_games", 0)

        self._winner_card.set_value(f"{winner_pct:.1f}%", color=CYAN)
        self._favorites_card.set_value(f"{fav_pct:.1f}%", color=GRAY)
        self._total_games_card.set_value(f"{total_games:,}")

        if beats_fav:
            diff = winner_pct - fav_pct
            self._beats_fav_card.set_value(f"+{diff:.1f}%", color=GREEN)
            self._beats_fav_card.title_label.setText("BEATS FAVORITES")
        else:
            diff = winner_pct - fav_pct
            self._beats_fav_card.set_value(f"{diff:+.1f}%", color=RED)
            self._beats_fav_card.title_label.setText("VS FAVORITES")

        # Stagger animations
        for i, card in enumerate([
            self._winner_card, self._favorites_card,
            self._beats_fav_card, self._total_games_card,
        ]):
            card.animate_in(delay_ms=i * 80)

        # ── Section 2: Upset Analysis ──
        upset_rate = fund.get("upset_rate", 0)
        upset_acc = fund.get("upset_accuracy", 0)
        upset_correct = fund.get("upset_correct", 0)

        self._upset_rate_card.set_value(f"{upset_rate:.1f}%", color=AMBER)
        self._upset_acc_card.set_value(f"{upset_acc:.1f}%", color=AMBER)
        self._upset_correct_card.set_value(f"{upset_correct:,}", color=AMBER)

        for i, card in enumerate([
            self._upset_rate_card, self._upset_acc_card,
            self._upset_correct_card,
        ]):
            card.animate_in(delay_ms=(i + 4) * 80)

        # ── Section 3: A/B Comparison columns ──
        self._update_ab_column(self._fund_col, fund)
        self._update_ab_column(self._sharp_col, sharp)

        # ── Section 4: Sharp impact summary ──
        flipped = comp.get("sharp_flipped_picks", 0)
        flipped_correct = comp.get("sharp_flipped_correct", 0)
        flipped_acc = comp.get("sharp_flipped_accuracy", 0)
        net_value = comp.get("sharp_net_value", 0)

        if flipped > 0:
            self._sharp_summary_lbl.setText(
                f"Sharp money flipped {flipped} picks, "
                f"{flipped_correct} correct ({flipped_acc:.1f}%)"
            )
            self._sharp_summary_lbl.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 14px; background: transparent;"
            )
        else:
            self._sharp_summary_lbl.setText(
                "Sharp money did not flip any picks"
            )
            self._sharp_summary_lbl.setStyleSheet(
                f"color: {TEXT_MUTED}; font-size: 14px; background: transparent;"
            )

        # Net contribution
        if net_value > 0.05:
            net_color = GREEN
            sign = "+"
        elif net_value < -0.05:
            net_color = RED
            sign = ""
        else:
            net_color = TEXT_MUTED
            sign = ""
        self._sharp_net_lbl.setText(
            f"Net contribution: {sign}{net_value:+.1f}% winner accuracy"
        )
        self._sharp_net_lbl.setStyleSheet(
            f"color: {net_color}; font-size: 13px; font-weight: 600; "
            f"background: transparent;"
        )
        self._sharp_net_lbl.setVisible(True)

    def _update_ab_column(self, col: _ABColumnCard, metrics: Dict[str, Any]):
        """Update an A/B comparison column with metrics."""
        if not metrics:
            return

        winner_pct = metrics.get("winner_pct", 0)
        upset_rate = metrics.get("upset_rate", 0)
        upset_acc = metrics.get("upset_accuracy", 0)
        ml_roi = metrics.get("ml_roi", 0)

        col.set_row_value("winner", f"{winner_pct:.1f}%", color=CYAN)
        col.set_row_value("upset_rate", f"{upset_rate:.1f}%", color=AMBER)
        col.set_row_value("upset_acc", f"{upset_acc:.1f}%", color=AMBER)

        # ML ROI color: green if positive, red if negative
        if ml_roi > 0:
            roi_color = GREEN
            roi_text = f"+{ml_roi:.1f}%"
        elif ml_roi < 0:
            roi_color = RED
            roi_text = f"{ml_roi:.1f}%"
        else:
            roi_color = TEXT_MUTED
            roi_text = "0.0%"
        col.set_row_value("ml_roi", roi_text, color=roi_color)
