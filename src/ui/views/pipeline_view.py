"""Pipeline tab -- run/monitor pipeline, view step state, manage snapshots.

V2 design: 10-step pipeline (backup, sync, seed_arenas, bbref_sync,
referee_sync, elo_compute, precompute, optimize x2, backtest) with real-time
progress log, step indicators, elapsed timers, and snapshot management for
weight configs.
"""

import json
import logging
import os
import time
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QTextEdit, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QGraphicsOpacityEffect, QMessageBox, QSpinBox, QCheckBox,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QTimer,
    QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import QColor

from src.ui.theme import apply_card_shadow

logger = logging.getLogger(__name__)

# ---- Theme colors ----
CYAN = "#00E5FF"
GREEN = "#00E676"
AMBER = "#FFB300"
RED = "#FF5252"
GRAY = "#78909C"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"

# Pipeline steps (name, display label) — must match PIPELINE_STEPS in pipeline.py
STEP_LABELS = [
    ("backup", "Backup"),
    ("sync", "Data Sync"),
    ("seed_arenas", "Arenas"),
    ("bbref_sync", "BBRef"),
    ("referee_sync", "Referees"),
    ("elo_compute", "Elo"),
    ("precompute", "Precompute"),
    ("optimize_fundamentals", "Optimize Fund."),
    ("optimize_sharp", "Optimize Sharp"),
    ("backtest", "Backtest"),
]


# ----------------------------------------------------------------
# Pipeline Worker
# ----------------------------------------------------------------

class _PipelineWorker(QObject):
    """Background worker that runs the full pipeline."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    step_changed = Signal(str)  # current step name

    def __init__(self, overnight: bool = False, max_hours: float = 8.0,
                 reset_weights: bool = False):
        super().__init__()
        self._cancelled = False
        self._overnight = overnight
        self._max_hours = max_hours
        self._reset_weights = reset_weights

    def _check_cancel(self) -> bool:
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        try:
            from src.analytics.pipeline import request_cancel
            request_cancel()
        except Exception:
            pass

    def run(self):
        try:
            if self._overnight:
                from src.analytics.pipeline import run_overnight
                result = run_overnight(
                    max_hours=self._max_hours,
                    reset_weights=self._reset_weights,
                    callback=self._on_progress,
                )
            else:
                from src.analytics.pipeline import run_pipeline
                result = run_pipeline(
                    callback=self._on_progress,
                    is_cancelled_fn=self._check_cancel,
                )
            self.finished.emit(result)
        except Exception as e:
            logger.error("PipelineWorker error: %s", e, exc_info=True)
            self.error.emit(str(e))

    def _on_progress(self, msg: str):
        self.progress.emit(msg)
        # Detect step changes from the "[Step N/M] step_name..." format
        if msg.startswith("[Step ") and "]" in msg:
            try:
                after_bracket = msg.split("] ", 1)[1]
                step_name = after_bracket.split("...")[0].strip()
                self.step_changed.emit(step_name)
            except Exception:
                pass


# ----------------------------------------------------------------
# Sync-Only Worker
# ----------------------------------------------------------------

class _SyncWorker(QObject):
    """Background worker that runs data sync only."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, force: bool = False):
        super().__init__()
        self._force = force

    def run(self):
        try:
            from src.data.sync_service import full_sync
            mode = "force" if self._force else "incremental"
            self.progress.emit(f"Starting {mode} data sync...")
            result = full_sync(
                force=self._force,
                callback=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit({"sync_result": result})
        except Exception as e:
            logger.error("SyncWorker error: %s", e, exc_info=True)
            self.error.emit(str(e))


class _OddsSyncWorker(QObject):
    """Background worker that runs odds sync only."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def run(self):
        try:
            from src.data.sync_service import sync_historical_odds
            self.progress.emit("Force-syncing Vegas odds...")
            sync_historical_odds(
                force=True,
                callback=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit({"odds_sync": "complete"})
        except Exception as e:
            logger.error("OddsSyncWorker error: %s", e, exc_info=True)
            self.error.emit(str(e))


# ----------------------------------------------------------------
# Step Indicator Widget
# ----------------------------------------------------------------

class _StepIndicator(QFrame):
    """Small pill-shaped indicator for a single pipeline step."""

    def __init__(self, step_name: str, display_label: str):
        super().__init__()
        self.step_name = step_name
        self.setFixedHeight(36)
        self.setMinimumWidth(100)
        self.setProperty("class", "step-pending")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(6)

        self._icon_lbl = QLabel("\u25CB")  # hollow circle
        self._icon_lbl.setFixedWidth(16)
        layout.addWidget(self._icon_lbl)

        self._name_lbl = QLabel(display_label.upper())
        self._name_lbl.setStyleSheet(
            "font-size: 11px; font-weight: 600; letter-spacing: 1px;"
        )
        layout.addWidget(self._name_lbl)

        self._time_lbl = QLabel("")
        self._time_lbl.setStyleSheet("font-size: 10px;")
        self._time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._time_lbl)

    def set_state(self, state: str, elapsed_text: str = ""):
        """State: 'pending', 'active', 'done', 'error', 'skipped'."""
        cls = f"step-{state}"
        self.setProperty("class", cls)
        self.style().unpolish(self)
        self.style().polish(self)

        icons = {
            "pending": "\u25CB",   # hollow circle
            "active": "\u25CF",    # filled circle (pulsing cyan)
            "done": "\u2713",      # checkmark
            "error": "\u2717",     # cross
            "skipped": "\u2014",   # dash
        }
        self._icon_lbl.setText(icons.get(state, "\u25CB"))
        if elapsed_text:
            self._time_lbl.setText(elapsed_text)


# ----------------------------------------------------------------
# PipelineView -- main widget
# ----------------------------------------------------------------

class PipelineView(QWidget):
    """Pipeline control and monitoring view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker: Optional[_PipelineWorker] = None
        self._worker_thread: Optional[QThread] = None
        self._sync_worker: Optional[_SyncWorker] = None
        self._sync_thread: Optional[QThread] = None
        self._current_worker = None  # ref for MainWindow cleanup
        self._running = False
        self._step_start_time: float = 0.0
        self._pipeline_start_time: float = 0.0

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(12)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Header
        header = QLabel("Pipeline")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # ---- Control bar ----
        self._build_controls(layout)

        # ---- Step indicators ----
        self._build_step_indicators(layout)

        # ---- Progress log ----
        self._build_log(layout)

        # ---- Pipeline state display ----
        self._build_state_display(layout)

        # ---- Snapshot management ----
        self._build_snapshot_section(layout)

        layout.addStretch()

        # Elapsed timer (updates every second while running)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        # Load last run state on init
        QTimer.singleShot(200, self._load_pipeline_state)

    # ---------------------------------------------------------------
    # UI builders
    # ---------------------------------------------------------------

    def _build_controls(self, parent_layout: QVBoxLayout):
        """Build the control panel with pipeline, sync, and overnight sections."""
        control_frame = QFrame()
        control_frame.setProperty("class", "broadcast-card")
        cl = QVBoxLayout(control_frame)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(10)

        # ── Row 1: Pipeline + Sync actions ──
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        # Run Pipeline button
        self._run_btn = QPushButton("RUN PIPELINE")
        self._run_btn.setProperty("class", "success")
        self._run_btn.setFixedHeight(44)
        self._run_btn.setMinimumWidth(180)
        self._run_btn.clicked.connect(self._on_run_toggle)
        row1.addWidget(self._run_btn)

        # Vertical separator
        sep1 = QFrame()
        sep1.setFixedWidth(1)
        sep1.setFixedHeight(36)
        sep1.setStyleSheet("background: rgba(255, 255, 255, 0.12);")
        row1.addWidget(sep1)

        # Sync section
        self._sync_btn = QPushButton("SYNC DATA")
        self._sync_btn.setProperty("class", "primary")
        self._sync_btn.setFixedHeight(44)
        self._sync_btn.setMinimumWidth(140)
        self._sync_btn.clicked.connect(self._on_sync)
        row1.addWidget(self._sync_btn)

        self._force_sync_cb = QCheckBox("Force full")
        self._force_sync_cb.setToolTip(
            "Bypass freshness checks and re-fetch all data from scratch"
        )
        self._force_sync_cb.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px; spacing: 5px;"
        )
        row1.addWidget(self._force_sync_cb)

        # Vertical separator
        sep2 = QFrame()
        sep2.setFixedWidth(1)
        sep2.setFixedHeight(36)
        sep2.setStyleSheet("background: rgba(255, 255, 255, 0.12);")
        row1.addWidget(sep2)

        # Sync Odds button
        self._sync_odds_btn = QPushButton("SYNC ODDS")
        self._sync_odds_btn.setProperty("class", "primary")
        self._sync_odds_btn.setFixedHeight(44)
        self._sync_odds_btn.setMinimumWidth(130)
        self._sync_odds_btn.setToolTip(
            "Force-sync Vegas odds from Action Network"
        )
        self._sync_odds_btn.clicked.connect(self._on_sync_odds)
        row1.addWidget(self._sync_odds_btn)

        row1.addStretch()

        # Current step / elapsed (right-aligned)
        self._current_step_lbl = QLabel("")
        self._current_step_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 13px; font-weight: 700;"
        )
        row1.addWidget(self._current_step_lbl)

        self._elapsed_lbl = QLabel("")
        self._elapsed_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px;"
        )
        row1.addWidget(self._elapsed_lbl)

        cl.addLayout(row1)

        # ── Thin divider ──
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: rgba(255, 255, 255, 0.06);")
        cl.addWidget(divider)

        # ── Row 2: Overnight section ──
        row2 = QHBoxLayout()
        row2.setSpacing(10)

        overnight_lbl = QLabel("OVERNIGHT")
        overnight_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; font-weight: 700; "
            f"letter-spacing: 2px;"
        )
        row2.addWidget(overnight_lbl)

        row2.addSpacing(8)

        hours_lbl = QLabel("Hours:")
        hours_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        row2.addWidget(hours_lbl)

        self._hours_spin = QSpinBox()
        self._hours_spin.setRange(1, 24)
        self._hours_spin.setValue(8)
        self._hours_spin.setFixedWidth(70)
        self._hours_spin.setSuffix("h")
        row2.addWidget(self._hours_spin)

        row2.addSpacing(8)

        self._overnight_btn = QPushButton("RUN OVERNIGHT")
        self._overnight_btn.setProperty("class", "indigo")
        self._overnight_btn.setFixedHeight(36)
        self._overnight_btn.setMinimumWidth(160)
        self._overnight_btn.clicked.connect(self._on_overnight)
        row2.addWidget(self._overnight_btn)

        row2.addStretch()

        cl.addLayout(row2)

        apply_card_shadow(control_frame)
        parent_layout.addWidget(control_frame)

    def _build_step_indicators(self, parent_layout: QVBoxLayout):
        """Build the row of step indicator pills."""
        row = QHBoxLayout()
        row.setSpacing(6)

        self._step_indicators = {}
        for step_name, display_label in STEP_LABELS:
            indicator = _StepIndicator(step_name, display_label)
            self._step_indicators[step_name] = indicator
            row.addWidget(indicator)

        row.addStretch()
        parent_layout.addLayout(row)

    def _build_log(self, parent_layout: QVBoxLayout):
        """Build the scrollable progress log."""
        log_lbl = QLabel("Progress Log")
        log_lbl.setProperty("class", "section-title")
        parent_layout.addWidget(log_lbl)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setProperty("class", "terminal")
        self._log_text.setMinimumHeight(250)
        self._log_text.setMaximumHeight(400)
        parent_layout.addWidget(self._log_text)

    def _build_state_display(self, parent_layout: QVBoxLayout):
        """Build the last-run state display panel."""
        state_lbl = QLabel("Last Run Summary")
        state_lbl.setProperty("class", "section-title")
        parent_layout.addWidget(state_lbl)

        self._state_frame = QFrame()
        self._state_frame.setProperty("class", "broadcast-card")
        state_layout = QVBoxLayout(self._state_frame)
        state_layout.setContentsMargins(16, 10, 16, 10)
        state_layout.setSpacing(6)

        # Summary grid
        grid = QGridLayout()
        grid.setSpacing(8)

        lbl_style = f"color: {TEXT_MUTED}; font-size: 12px; text-transform: uppercase;"
        val_style = f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600;"

        grid.addWidget(self._make_label("Last Run", lbl_style), 0, 0)
        self._state_last_run = self._make_label("\u2014", val_style)
        grid.addWidget(self._state_last_run, 0, 1)

        grid.addWidget(self._make_label("Duration", lbl_style), 0, 2)
        self._state_duration = self._make_label("\u2014", val_style)
        grid.addWidget(self._state_duration, 0, 3)

        grid.addWidget(self._make_label("Steps", lbl_style), 1, 0)
        self._state_steps = self._make_label("\u2014", val_style)
        grid.addWidget(self._state_steps, 1, 1)

        grid.addWidget(self._make_label("Result", lbl_style), 1, 2)
        self._state_result = self._make_label("\u2014", val_style)
        grid.addWidget(self._state_result, 1, 3)

        state_layout.addLayout(grid)

        # Per-step timing
        self._step_timing_lbl = QLabel("")
        self._step_timing_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 12px;"
        )
        self._step_timing_lbl.setWordWrap(True)
        state_layout.addWidget(self._step_timing_lbl)

        apply_card_shadow(self._state_frame)
        parent_layout.addWidget(self._state_frame)

    def _build_snapshot_section(self, parent_layout: QVBoxLayout):
        """Build the snapshot management section."""
        snap_lbl = QLabel("Weight Snapshots")
        snap_lbl.setProperty("class", "section-title")
        parent_layout.addWidget(snap_lbl)

        # Save new snapshot row
        save_row = QHBoxLayout()
        save_row.setSpacing(8)

        self._snap_name_input = QLineEdit()
        self._snap_name_input.setPlaceholderText("Snapshot name...")
        self._snap_name_input.setMinimumWidth(200)
        save_row.addWidget(self._snap_name_input)

        self._snap_notes_input = QLineEdit()
        self._snap_notes_input.setPlaceholderText("Notes (optional)...")
        self._snap_notes_input.setMinimumWidth(250)
        save_row.addWidget(self._snap_notes_input)

        save_btn = QPushButton("SAVE SNAPSHOT")
        save_btn.setProperty("class", "primary")
        save_btn.setFixedHeight(36)
        save_btn.clicked.connect(self._on_save_snapshot)
        save_row.addWidget(save_btn)

        save_row.addStretch()

        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setProperty("class", "outline")
        refresh_btn.setFixedHeight(36)
        refresh_btn.clicked.connect(self._refresh_snapshots)
        save_row.addWidget(refresh_btn)

        parent_layout.addLayout(save_row)

        # Snapshot table
        self._snap_table = QTableWidget()
        self._snap_table.setColumnCount(5)
        self._snap_table.setHorizontalHeaderLabels(
            ["Name", "Date", "Notes", "Metrics", "Actions"]
        )
        hdr = self._snap_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._snap_table.setColumnWidth(0, 160)
        self._snap_table.setColumnWidth(1, 160)
        self._snap_table.setColumnWidth(3, 200)
        self._snap_table.setColumnWidth(4, 100)
        self._snap_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._snap_table.verticalHeader().setVisible(False)
        self._snap_table.setAlternatingRowColors(True)
        self._snap_table.setMaximumHeight(260)
        parent_layout.addWidget(self._snap_table)

        # Load snapshots on init
        QTimer.singleShot(300, self._refresh_snapshots)

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _make_label(text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    @staticmethod
    def _fmt_seconds(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m = int(secs // 60)
        s = int(secs % 60)
        if m < 60:
            return f"{m}m {s}s"
        h = m // 60
        m = m % 60
        return f"{h}h {m}m"

    # ---------------------------------------------------------------
    # Pipeline state (last run)
    # ---------------------------------------------------------------

    def _load_pipeline_state(self):
        """Load pipeline_state.json and display last run summary."""
        state_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "data", "pipeline_state.json"
        )
        state_path = os.path.normpath(state_path)
        if not os.path.exists(state_path):
            return

        try:
            with open(state_path, "r") as f:
                state = json.load(f)
        except Exception as e:
            logger.debug("Failed to load pipeline state: %s", e)
            return

        # Last run timestamp
        last_run = state.get("last_run", "\u2014")
        self._state_last_run.setText(last_run)

        # Duration
        elapsed = state.get("elapsed_seconds", 0)
        if elapsed:
            self._state_duration.setText(self._fmt_seconds(elapsed))

        # Count only known pipeline steps so removed/stale steps do not skew summary.
        known_step_keys = [f"step_{step_name}" for step_name, _ in STEP_LABELS]
        step_count = len(known_step_keys)
        completed = sum(
            1
            for key in known_step_keys
            if isinstance(state.get(key), dict)
            and state[key].get("status") == "completed"
        )
        errors = sum(
            1
            for key in known_step_keys
            if isinstance(state.get(key), dict)
            and state[key].get("status") == "error"
        )
        self._state_steps.setText(f"{completed}/{step_count} completed")

        if errors > 0:
            self._state_result.setText(f"{errors} error(s)")
            self._state_result.setStyleSheet(
                f"color: {RED}; font-size: 14px; font-weight: 600;"
            )
        elif completed == step_count and step_count > 0:
            self._state_result.setText("Success")
            self._state_result.setStyleSheet(
                f"color: {GREEN}; font-size: 14px; font-weight: 700;"
            )
        else:
            self._state_result.setText("\u2014")
            self._state_result.setStyleSheet(
                f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600;"
            )

        # Per-step timings
        timing_parts = []
        for step_name, label in STEP_LABELS:
            key = f"step_{step_name}"
            info = state.get(key, {})
            if isinstance(info, dict):
                secs = info.get("elapsed_seconds", 0)
                status = info.get("status", "")
                if status == "completed":
                    timing_parts.append(f"{label}: {self._fmt_seconds(secs)}")
                elif status == "error":
                    timing_parts.append(f"{label}: ERROR")

        if timing_parts:
            self._step_timing_lbl.setText("  |  ".join(timing_parts))

        # Update step indicators from state
        for step_name, _ in STEP_LABELS:
            key = f"step_{step_name}"
            info = state.get(key, {})
            indicator = self._step_indicators.get(step_name)
            if indicator and isinstance(info, dict):
                status = info.get("status", "pending")
                secs = info.get("elapsed_seconds", 0)
                if status == "completed":
                    indicator.set_state("done", self._fmt_seconds(secs))
                elif status == "error":
                    indicator.set_state("error", "ERR")
                else:
                    indicator.set_state("pending")

    # ---------------------------------------------------------------
    # Run / Cancel pipeline
    # ---------------------------------------------------------------

    def _on_run_toggle(self):
        """Toggle between starting and cancelling the pipeline."""
        if self._running:
            self._cancel_pipeline()
        else:
            self._start_pipeline(overnight=False)

    def _on_overnight(self):
        """Start an overnight optimization run."""
        if self._running:
            return
        self._start_pipeline(overnight=True)

    # ---------------------------------------------------------------
    # Sync-only
    # ---------------------------------------------------------------

    def _on_sync(self):
        """Run data sync only (not the full pipeline)."""
        if self._running:
            return

        force = self._force_sync_cb.isChecked()
        self._running = True
        self._pipeline_start_time = time.time()

        # UI updates
        self._sync_btn.setText("SYNCING...")
        self._sync_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._sync_odds_btn.setEnabled(False)
        self._overnight_btn.setEnabled(False)
        self._hours_spin.setEnabled(False)
        self._log_text.clear()

        mode_str = "force" if force else "incremental"
        self._log_text.append(f"Starting {mode_str} data sync...")
        self._current_step_lbl.setText("DATA SYNC")

        # Mark the sync step as active
        ind = self._step_indicators.get("sync")
        if ind:
            ind.set_state("active")
        self._step_start_time = time.time()

        # Create worker
        self._sync_worker = _SyncWorker(force=force)
        self._sync_thread = QThread()
        self._sync_worker.moveToThread(self._sync_thread)
        self._sync_thread.started.connect(self._sync_worker.run)

        _QC = Qt.ConnectionType.QueuedConnection
        self._sync_worker.progress.connect(self._on_progress, _QC)
        self._sync_worker.finished.connect(self._on_sync_finished, _QC)
        self._sync_worker.error.connect(self._on_sync_error, _QC)
        self._sync_worker.finished.connect(self._sync_thread.quit)
        self._sync_worker.error.connect(self._sync_thread.quit)
        self._sync_thread.finished.connect(self._cleanup_sync_worker)

        self._elapsed_timer.start()
        self._sync_thread.start()

        if self.main_window:
            self.main_window.set_status(f"Data sync ({mode_str}) running...")

    def _on_sync_finished(self, result: dict):
        """Handle sync-only completion."""
        self._running = False
        self._elapsed_timer.stop()

        elapsed = time.time() - self._pipeline_start_time
        self._elapsed_lbl.setText(self._fmt_seconds(elapsed))

        # Mark sync step done
        ind = self._step_indicators.get("sync")
        if ind:
            ind.set_state("done", self._fmt_seconds(elapsed))

        self._log_text.append(f"\nData sync complete in {self._fmt_seconds(elapsed)}")
        self._current_step_lbl.setText("Sync Complete")
        self._current_step_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 13px; font-weight: 700;"
        )

        self._reset_controls()

        if self.main_window:
            self.main_window.set_status(
                f"Data sync complete in {self._fmt_seconds(elapsed)}"
            )

    def _on_sync_error(self, msg: str):
        """Handle sync-only error."""
        self._running = False
        self._elapsed_timer.stop()

        ind = self._step_indicators.get("sync")
        if ind:
            ind.set_state("error", "ERR")

        self._log_text.append(f"\nSYNC ERROR: {msg}")
        self._current_step_lbl.setText("Sync Error")
        self._current_step_lbl.setStyleSheet(
            f"color: {RED}; font-size: 13px; font-weight: 700;"
        )
        self._reset_controls()

        if self.main_window:
            self.main_window.set_status(f"Sync error: {msg}")

    def _cleanup_sync_worker(self):
        """Clean up sync worker and thread."""
        if hasattr(self, '_sync_thread') and self._sync_thread is not None:
            self._sync_thread.deleteLater()
        if hasattr(self, '_sync_worker') and self._sync_worker is not None:
            self._sync_worker.deleteLater()
        self._sync_thread = None
        self._sync_worker = None

    # ---------------------------------------------------------------
    # Odds sync-only
    # ---------------------------------------------------------------

    def _on_sync_odds(self):
        """Run odds sync only (force mode)."""
        if self._running:
            return

        self._running = True
        self._pipeline_start_time = time.time()

        # UI updates
        self._sync_odds_btn.setText("SYNCING...")
        self._sync_odds_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._sync_btn.setEnabled(False)
        self._overnight_btn.setEnabled(False)
        self._hours_spin.setEnabled(False)
        self._log_text.clear()

        self._log_text.append("Starting force odds sync...")
        self._current_step_lbl.setText("ODDS SYNC")

        # Create worker
        self._odds_worker = _OddsSyncWorker()
        self._odds_thread = QThread()
        self._odds_worker.moveToThread(self._odds_thread)
        self._odds_thread.started.connect(self._odds_worker.run)

        _QC = Qt.ConnectionType.QueuedConnection
        self._odds_worker.progress.connect(self._on_progress, _QC)
        self._odds_worker.finished.connect(self._on_odds_sync_finished, _QC)
        self._odds_worker.error.connect(self._on_odds_sync_error, _QC)
        self._odds_worker.finished.connect(self._odds_thread.quit)
        self._odds_worker.error.connect(self._odds_thread.quit)
        self._odds_thread.finished.connect(self._cleanup_odds_worker)

        self._elapsed_timer.start()
        self._odds_thread.start()

        if self.main_window:
            self.main_window.set_status("Odds sync running...")

    def _on_odds_sync_finished(self, result: dict):
        """Handle odds sync completion."""
        self._running = False
        self._elapsed_timer.stop()

        elapsed = time.time() - self._pipeline_start_time
        self._elapsed_lbl.setText(self._fmt_seconds(elapsed))

        self._log_text.append(f"\nOdds sync complete in {self._fmt_seconds(elapsed)}")
        self._current_step_lbl.setText("Odds Sync Complete")
        self._current_step_lbl.setStyleSheet(
            f"color: {GREEN}; font-size: 13px; font-weight: 700;"
        )

        self._reset_controls()

        if self.main_window:
            self.main_window.set_status(
                f"Odds sync complete in {self._fmt_seconds(elapsed)}"
            )

    def _on_odds_sync_error(self, msg: str):
        """Handle odds sync error."""
        self._running = False
        self._elapsed_timer.stop()

        self._log_text.append(f"\nODDS SYNC ERROR: {msg}")
        self._current_step_lbl.setText("Odds Sync Error")
        self._current_step_lbl.setStyleSheet(
            f"color: {RED}; font-size: 13px; font-weight: 700;"
        )
        self._reset_controls()

        if self.main_window:
            self.main_window.set_status(f"Odds sync error: {msg}")

    def _cleanup_odds_worker(self):
        """Clean up odds sync worker and thread."""
        if hasattr(self, '_odds_thread') and self._odds_thread is not None:
            self._odds_thread.deleteLater()
        if hasattr(self, '_odds_worker') and self._odds_worker is not None:
            self._odds_worker.deleteLater()
        self._odds_thread = None
        self._odds_worker = None

    def _start_pipeline(self, overnight: bool = False):
        """Start the pipeline in a background thread."""
        # Busy guard
        try:
            if self._worker_thread is not None and self._worker_thread.isRunning():
                return
        except RuntimeError:
            self._worker_thread = None

        self._running = True
        self._pipeline_start_time = time.time()

        # UI updates
        self._run_btn.setText("CANCEL")
        self._run_btn.setProperty("class", "danger")
        self._run_btn.style().unpolish(self._run_btn)
        self._run_btn.style().polish(self._run_btn)
        self._sync_btn.setEnabled(False)
        self._sync_odds_btn.setEnabled(False)
        self._overnight_btn.setEnabled(False)
        self._hours_spin.setEnabled(False)
        self._log_text.clear()

        # Reset step indicators
        for ind in self._step_indicators.values():
            ind.set_state("pending")

        mode = "overnight" if overnight else "pipeline"
        self._log_text.append(f"Starting {mode}...")

        # Create worker
        self._worker = _PipelineWorker(
            overnight=overnight,
            max_hours=self._hours_spin.value(),
            reset_weights=False,
        )
        self._worker_thread = QThread()
        self._current_worker = self._worker_thread  # for MainWindow cleanup
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)

        _QC = Qt.ConnectionType.QueuedConnection
        self._worker.finished.connect(self._on_finished, _QC)
        self._worker.error.connect(self._on_error, _QC)
        self._worker.progress.connect(self._on_progress, _QC)
        self._worker.step_changed.connect(self._on_step_changed, _QC)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)

        self._elapsed_timer.start()
        self._worker_thread.start()

        if self.main_window:
            self.main_window.set_status(f"Pipeline {mode} running...")

    def _cancel_pipeline(self):
        """Request pipeline cancellation."""
        if self._worker:
            self._worker.cancel()
        self._log_text.append("\nCancellation requested... finishing current step.")
        self._current_step_lbl.setText("Cancelling...")

    def _on_progress(self, msg: str):
        """Append a progress message to the log."""
        self._log_text.append(msg)
        # Auto-scroll to bottom
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_step_changed(self, step_name: str):
        """Update step indicators when a new step starts."""
        self._step_start_time = time.time()
        self._current_step_lbl.setText(step_name.upper())

        # Find step in our list and mark indicators
        found = False
        for sn, _ in STEP_LABELS:
            ind = self._step_indicators.get(sn)
            if not ind:
                continue
            if sn == step_name:
                ind.set_state("active")
                found = True
            elif not found:
                # Steps before current one should be done (unless already error)
                cur_class = ind.property("class")
                if cur_class != "step-error":
                    # Check if already done from previous elapsed time
                    ind.set_state("done")

    def _on_finished(self, result: dict):
        """Handle pipeline completion."""
        self._running = False
        self._elapsed_timer.stop()

        elapsed = time.time() - self._pipeline_start_time
        self._elapsed_lbl.setText(self._fmt_seconds(elapsed))

        # Mark remaining steps as done
        for ind in self._step_indicators.values():
            cur_class = ind.property("class")
            if cur_class in ("step-active", "step-pending"):
                ind.set_state("done")

        cancelled = result.get("cancelled", False)
        if cancelled:
            self._log_text.append(f"\nPipeline cancelled after {self._fmt_seconds(elapsed)}")
            self._current_step_lbl.setText("Cancelled")
        else:
            self._log_text.append(f"\nPipeline complete in {self._fmt_seconds(elapsed)}")
            self._current_step_lbl.setText("Complete")
            self._current_step_lbl.setStyleSheet(
                f"color: {GREEN}; font-size: 13px; font-weight: 700;"
            )

        self._reset_controls()
        self._load_pipeline_state()
        self._refresh_snapshots()

        if self.main_window:
            self.main_window.set_status(
                f"Pipeline {'cancelled' if cancelled else 'complete'} "
                f"in {self._fmt_seconds(elapsed)}"
            )

    def _on_error(self, msg: str):
        """Handle pipeline error."""
        self._running = False
        self._elapsed_timer.stop()

        self._log_text.append(f"\nPIPELINE ERROR: {msg}")
        self._current_step_lbl.setText("Error")
        self._current_step_lbl.setStyleSheet(
            f"color: {RED}; font-size: 13px; font-weight: 700;"
        )
        self._reset_controls()

        if self.main_window:
            self.main_window.set_status(f"Pipeline error: {msg}")

    def _reset_controls(self):
        """Reset control buttons to default state."""
        self._run_btn.setText("RUN PIPELINE")
        self._run_btn.setProperty("class", "success")
        self._run_btn.style().unpolish(self._run_btn)
        self._run_btn.style().polish(self._run_btn)
        self._run_btn.setEnabled(True)
        self._sync_btn.setText("SYNC DATA")
        self._sync_btn.setEnabled(True)
        self._sync_odds_btn.setText("SYNC ODDS")
        self._sync_odds_btn.setEnabled(True)
        self._overnight_btn.setEnabled(True)
        self._hours_spin.setEnabled(True)

    def _update_elapsed(self):
        """Update the elapsed time display (called every second)."""
        if self._running:
            total = time.time() - self._pipeline_start_time
            self._elapsed_lbl.setText(self._fmt_seconds(total))

            # Update active step time
            if self._step_start_time > 0:
                step_elapsed = time.time() - self._step_start_time
                for ind in self._step_indicators.values():
                    if ind.property("class") == "step-active":
                        ind._time_lbl.setText(self._fmt_seconds(step_elapsed))

    def _cleanup_worker(self):
        """Clean up worker and thread references."""
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker_thread = None
        self._worker = None
        self._current_worker = None

    # ---------------------------------------------------------------
    # Snapshot management
    # ---------------------------------------------------------------

    def _refresh_snapshots(self):
        """Reload snapshot list into the table."""
        try:
            from src.analytics.weight_config import list_snapshots
            snaps = list_snapshots()
        except Exception as e:
            logger.debug("Failed to list snapshots: %s", e)
            snaps = []

        self._snap_table.setRowCount(len(snaps))
        self._snap_paths = {}  # row -> path

        for row, snap in enumerate(snaps):
            path = snap.get("path", "")
            self._snap_paths[row] = path

            # Name
            name_item = QTableWidgetItem(snap.get("name", ""))
            self._snap_table.setItem(row, 0, name_item)

            # Date
            created = snap.get("created_at", "")
            if created:
                try:
                    # Trim ISO fractional seconds for display
                    display_date = created[:19].replace("T", " ")
                except Exception:
                    display_date = created
            else:
                display_date = ""
            date_item = QTableWidgetItem(display_date)
            self._snap_table.setItem(row, 1, date_item)

            # Notes
            notes_item = QTableWidgetItem(snap.get("notes", ""))
            self._snap_table.setItem(row, 2, notes_item)

            # Metrics summary
            metrics = snap.get("metrics", {})
            if metrics:
                parts = []
                w_pct = metrics.get("winner_pct")
                if w_pct is not None:
                    parts.append(f"Win: {w_pct:.1f}%")
                dog_hit = metrics.get("dog_hit_rate")
                if dog_hit is not None:
                    parts.append(f"Dog: {dog_hit:.1f}%")
                metrics_text = " | ".join(parts) if parts else ""
            else:
                metrics_text = ""
            metrics_item = QTableWidgetItem(metrics_text)
            self._snap_table.setItem(row, 3, metrics_item)

            # Restore button
            restore_btn = QPushButton("Restore")
            restore_btn.setProperty("class", "outline")
            restore_btn.setFixedHeight(28)
            restore_btn.clicked.connect(
                lambda _checked=False, r=row: self._on_restore_snapshot(r)
            )
            self._snap_table.setCellWidget(row, 4, restore_btn)

    def _on_save_snapshot(self):
        """Save a new snapshot with the given name and notes."""
        name = self._snap_name_input.text().strip()
        if not name:
            name = "manual_snapshot"
        notes = self._snap_notes_input.text().strip()

        try:
            from src.analytics.weight_config import save_snapshot
            path = save_snapshot(name, notes=notes)
            self._log_text.append(f"Snapshot saved: {os.path.basename(path)}")
            self._snap_name_input.clear()
            self._snap_notes_input.clear()
            self._refresh_snapshots()
        except Exception as e:
            logger.error("Save snapshot failed: %s", e)
            self._log_text.append(f"Snapshot save failed: {e}")

    def _on_restore_snapshot(self, row: int):
        """Restore weights from the selected snapshot."""
        path = self._snap_paths.get(row, "")
        if not path or not os.path.exists(path):
            return

        name = self._snap_table.item(row, 0)
        name_text = name.text() if name else os.path.basename(path)

        reply = QMessageBox.question(
            self,
            "Restore Snapshot",
            f"Restore weights from snapshot '{name_text}'?\n\n"
            "This will overwrite the current model weights.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                from src.analytics.weight_config import restore_snapshot
                restore_snapshot(path)
                self._log_text.append(f"Restored snapshot: {name_text}")
                if self.main_window:
                    self.main_window.set_status(f"Weights restored from '{name_text}'")
            except Exception as e:
                logger.error("Restore snapshot failed: %s", e)
                self._log_text.append(f"Restore failed: {e}")
