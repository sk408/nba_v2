"""Standalone Overnight Control Center UI.

Run with:
    python overnight_control_center.py

This tool is intentionally separate from PipelineView and focuses only on
overnight sessions. It offers grouped settings, inline guidance, presets,
and run/cancel controls with live logs.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QThread, QTimer, Qt, Signal, QObject
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from src.analytics.pipeline import request_cancel, run_overnight
from src.bootstrap import bootstrap, setup_logging, shutdown
from src.config import get as get_setting
from src.config import invalidate_cache, load_settings, save_settings
from src.ui.theme import apply_card_shadow, setup_theme
from overnight import RichOvernightConsole, configure_overnight_file_logging

logger = logging.getLogger(__name__)


CYAN = "#00E5FF"
GREEN = "#00E676"
AMBER = "#FFB300"
RED = "#FF5252"
MUTED = "#94a3b8"

_ONOFF_FAILED_TEAM_RE = re.compile(r"On/off failed for team_id=(\d+)", re.IGNORECASE)
_ONOFF_FETCH_START_RE = re.compile(r"Fetching player on/off data", re.IGNORECASE)


def _dedupe_ordered_team_ids(team_ids: list[int]) -> list[int]:
    seen = set()
    ordered: list[int] = []
    for team_id in team_ids:
        if team_id in seen:
            continue
        seen.add(team_id)
        ordered.append(team_id)
    return ordered


def _extract_latest_failed_onoff_team_ids(log_text: str) -> list[int]:
    """Extract failed on/off team ids from the latest player-impact fetch block."""
    lines = log_text.splitlines()
    start_idx = -1
    for idx, line in enumerate(lines):
        if _ONOFF_FETCH_START_RE.search(line):
            start_idx = idx

    def _extract(lines_slice: list[str]) -> list[int]:
        ids: list[int] = []
        for line in lines_slice:
            match = _ONOFF_FAILED_TEAM_RE.search(line)
            if not match:
                continue
            try:
                ids.append(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return _dedupe_ordered_team_ids(ids)

    if start_idx >= 0:
        latest_ids = _extract(lines[start_idx:])
        if latest_ids:
            return latest_ids
    return _extract(lines)


@dataclass
class _Binding:
    key: str
    kind: str
    widget: Any


class _OvernightWorker(QObject):
    """Background overnight runner worker."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, max_hours: float, reset_weights: bool):
        super().__init__()
        self._max_hours = max_hours
        self._reset_weights = reset_weights

    def cancel(self):
        request_cancel()

    def run(self):
        try:
            result = run_overnight(
                max_hours=self._max_hours,
                reset_weights=self._reset_weights,
                callback=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - defensive UI path
            logger.error("Overnight worker failed: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class _OnOffRepairWorker(QObject):
    """Background on/off repair sync worker."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, team_ids: list[int]):
        super().__init__()
        self._team_ids = [int(tid) for tid in team_ids if int(tid) > 0]

    def run(self):
        try:
            from src.data.sync_service import sync_player_impact_for_teams
            from src.database import db

            if not self._team_ids:
                raise ValueError("No valid team IDs provided for on/off repair.")

            self.progress.emit(
                "Repairing player on/off for team IDs: "
                + ", ".join(str(tid) for tid in self._team_ids)
            )
            sync_player_impact_for_teams(
                team_ids=self._team_ids,
                callback=lambda msg: self.progress.emit(msg),
                force=True,
            )

            placeholders = ",".join(["?"] * len(self._team_ids))
            stats = db.fetch_all(
                f"""SELECT
                        team_id,
                        COUNT(*) AS n_players,
                        SUM(
                            CASE
                                WHEN COALESCE(on_court_off_rating, 0) != 0
                                  OR COALESCE(on_court_def_rating, 0) != 0
                                  OR COALESCE(on_court_net_rating, 0) != 0
                                  OR COALESCE(off_court_off_rating, 0) != 0
                                  OR COALESCE(off_court_def_rating, 0) != 0
                                  OR COALESCE(off_court_net_rating, 0) != 0
                                  OR COALESCE(net_rating_diff, 0) != 0
                                  OR COALESCE(on_court_minutes, 0) != 0
                                THEN 1 ELSE 0
                            END
                        ) AS n_nonzero,
                        MAX(last_synced_at) AS last_synced_at
                    FROM player_impact
                    WHERE team_id IN ({placeholders})
                    GROUP BY team_id
                    ORDER BY team_id""",
                tuple(self._team_ids),
            )
            self.finished.emit(
                {
                    "team_ids": list(self._team_ids),
                    "stats": stats,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            logger.error("On/off repair worker failed: %s", exc, exc_info=True)
            self.failed.emit(str(exc))


class OvernightControlCenter(QMainWindow):
    """Dedicated UI for configuring and running overnight sessions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NBA Overnight Control Center")
        self.resize(1520, 980)

        self._bindings: Dict[str, _Binding] = {}
        self._worker: Optional[_OvernightWorker] = None
        self._thread: Optional[QThread] = None
        self._repair_worker: Optional[_OnOffRepairWorker] = None
        self._repair_thread: Optional[QThread] = None
        self._repair_running = False
        self._progress_parser: Optional[RichOvernightConsole] = None
        self._blocked_playoff: Dict[str, Dict[str, Any]] = {}
        self._blocked_playoff_mode_context: str = "fundamentals"
        self._running = False
        self._close_after_cancel = False
        self._run_start_ts = 0.0

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed_label)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = self._build_header()
        root_layout.addWidget(header)

        body_split = QSplitter(Qt.Orientation.Horizontal)
        body_split.setChildrenCollapsible(False)
        root_layout.addWidget(body_split, 1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_container = QWidget()
        controls_scroll.setWidget(controls_container)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(10)

        controls_layout.addWidget(self._build_run_card())
        controls_layout.addWidget(self._build_preset_card())
        controls_layout.addWidget(self._build_search_card())
        controls_layout.addWidget(self._build_blocked_card())
        controls_layout.addWidget(self._build_objective_card())
        controls_layout.addWidget(self._build_ml_gate_card())
        controls_layout.addWidget(self._build_save_gate_card())
        controls_layout.addWidget(self._build_advanced_overrides_card())
        controls_layout.addStretch()

        logs_panel = self._build_logs_panel()

        body_split.addWidget(controls_scroll)
        body_split.addWidget(logs_panel)
        body_split.setSizes([860, 640])

        self._load_current_settings()
        self._clear_progress_dashboard()
        self._append_log("Ready. Configure options, then click RUN OVERNIGHT.")

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setProperty("class", "broadcast-card")
        apply_card_shadow(frame, "md")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title = QLabel("Overnight Control Center")
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 800; color: {CYAN}; letter-spacing: 1px;"
        )
        layout.addWidget(title)

        info = QLabel(
            "This window is dedicated to overnight sessions only. "
            "1) Choose run duration and options. "
            "2) Use presets or tune settings. "
            "3) Run and monitor logs live. "
            "All selected settings are saved to data/app_settings.json before run."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {MUTED}; font-size: 13px;")
        layout.addWidget(info)
        return frame

    def _build_run_card(self) -> QWidget:
        box = self._group("Run Session")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        self._hours_spin = QDoubleSpinBox()
        self._hours_spin.setRange(0.25, 24.0)
        self._hours_spin.setSingleStep(0.25)
        self._hours_spin.setDecimals(2)
        self._hours_spin.setValue(8.0)
        self._hours_spin.setSuffix(" hours")
        self._hours_spin.setToolTip("Total wall-clock budget for run_overnight().")
        form.addRow(self._label("Run Budget"), self._hours_spin)

        self._reset_weights_cb = QCheckBox("Reset model weights to defaults at start")
        self._reset_weights_cb.setChecked(False)
        self._reset_weights_cb.setToolTip(
            "Equivalent to run_overnight(..., reset_weights=True). Use this after major "
            "objective or feature changes."
        )
        form.addRow(self._label("Start State"), self._reset_weights_cb)

        self._repair_team_ids_input = QLineEdit()
        self._repair_team_ids_input.setPlaceholderText(
            "Optional CSV team IDs (blank = parse latest failed IDs from data/overnight.log)"
        )
        self._repair_team_ids_input.setToolTip(
            "Enter comma/space-separated team IDs to repair explicitly. "
            "Leave blank to auto-detect failed team IDs from overnight.log."
        )
        form.addRow(self._label("Repair Team IDs"), self._repair_team_ids_input)

        self._status_lbl = QLabel("Idle")
        self._status_lbl.setStyleSheet(f"color: {GREEN}; font-weight: 700;")
        form.addRow(self._label("Status"), self._status_lbl)

        self._elapsed_lbl = QLabel("00:00:00")
        self._elapsed_lbl.setStyleSheet(f"color: {MUTED};")
        form.addRow(self._label("Elapsed"), self._elapsed_lbl)

        box.layout().addLayout(form)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._run_btn = QPushButton("RUN OVERNIGHT")
        self._run_btn.setProperty("class", "success")
        self._run_btn.clicked.connect(self._on_run_clicked)
        row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("CANCEL")
        self._cancel_btn.setProperty("class", "danger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        row.addWidget(self._cancel_btn)

        self._reload_btn = QPushButton("RELOAD SETTINGS")
        self._reload_btn.setProperty("class", "outline")
        self._reload_btn.clicked.connect(self._load_current_settings)
        row.addWidget(self._reload_btn)

        self._repair_onoff_btn = QPushButton("REPAIR ON/OFF TEAMS")
        self._repair_onoff_btn.setProperty("class", "outline")
        self._repair_onoff_btn.clicked.connect(self._on_repair_onoff_clicked)
        row.addWidget(self._repair_onoff_btn)

        row.addStretch()
        box.layout().addLayout(row)
        return box

    def _build_preset_card(self) -> QWidget:
        box = self._group("Presets")
        help_lbl = QLabel(
            "Presets set the controls below for common overnight strategies. "
            "You can still tweak any field after applying a preset."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        box.layout().addWidget(help_lbl)

        row = QHBoxLayout()
        row.setSpacing(8)
        balanced = QPushButton("Balanced")
        balanced.clicked.connect(lambda: self._apply_preset("balanced"))
        row.addWidget(balanced)

        explore = QPushButton("Exploration")
        explore.clicked.connect(lambda: self._apply_preset("exploration"))
        row.addWidget(explore)

        strict = QPushButton("Promotion Strict")
        strict.clicked.connect(lambda: self._apply_preset("strict"))
        row.addWidget(strict)

        quiet = QPushButton("Quiet Logs")
        quiet.clicked.connect(lambda: self._apply_preset("quiet_logs"))
        row.addWidget(quiet)

        row.addStretch()
        box.layout().addLayout(row)
        return box

    def _build_search_card(self) -> QWidget:
        box = self._group("Search & Runtime")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._add_choice(
            form,
            "optimizer_tuning_mode",
            "Tuning Mode",
            [("blocked", "Blocked (staged pathways)"), ("classic", "Classic (single study)")],
            "Blocked runs staged core/ff/onoff/joint_refine pathways before final save gate.",
        )
        self._add_bool(
            form,
            "optimizer_use_wide_ranges",
            "Use Wide Ranges",
            "Use CD-style broader bounds for optimization.",
        )
        self._add_int(
            form,
            "optuna_top_n_validation",
            "Top-N Validation Candidates",
            10,
            400,
            "How many top objective trials to re-evaluate on validation.",
        )
        self._add_int(
            form,
            "optuna_stagnation_threshold",
            "Stagnation Warning Threshold",
            100,
            5000,
            "Warn every N non-improving trials.",
        )
        self._add_int(
            form,
            "optuna_early_stop_trials",
            "Early Stop Non-Improve Window",
            200,
            10000,
            "Stop study after this many non-improving trials (after min trials).",
        )
        self._add_int(
            form,
            "optuna_min_trials_before_stop",
            "Min Trials Before Early Stop",
            100,
            10000,
            "Minimum trials before early stop logic can trigger.",
        )
        self._add_int(
            form,
            "optimizer_log_interval",
            "Optimizer Trial Log Interval",
            25,
            2000,
            "Emit trial line every N trials (plus new best events).",
        )
        self._add_int(
            form,
            "precompute_progress_log_every",
            "Precompute Log Every N Games",
            25,
            2000,
            "Progress line cadence during precompute.",
        )
        self._add_int(
            form,
            "overnight_max_no_save_passes",
            "Max Consecutive No-Save Passes",
            0,
            20,
            "0 disables early stop; otherwise overnight exits after N no-save passes.",
        )

        box.layout().addLayout(form)
        return box

    def _build_blocked_card(self) -> QWidget:
        box = self._group("Blocked Pathways")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._add_float(
            form,
            "optimizer_blocked_core_fraction",
            "Core Fraction",
            0.0,
            1.0,
            0.01,
            2,
            "Trial budget share for core stage.",
        )
        self._add_float(
            form,
            "optimizer_blocked_ff_fraction",
            "FF Fraction",
            0.0,
            1.0,
            0.01,
            2,
            "Trial budget share for FF stage.",
        )
        self._add_float(
            form,
            "optimizer_blocked_onoff_fraction",
            "On/Off Fraction",
            0.0,
            1.0,
            0.01,
            2,
            "Trial budget share for on/off stage.",
        )
        self._add_float(
            form,
            "optimizer_blocked_joint_fraction",
            "Joint Refine Fraction",
            0.0,
            1.0,
            0.01,
            2,
            "Trial budget share for final joint refinement.",
        )
        self._add_int(
            form,
            "optimizer_blocked_min_stage_trials",
            "Min Stage Trials",
            25,
            5000,
            "Minimum trials per stage when budget allows.",
        )
        self._add_float(
            form,
            "optimizer_blocked_joint_radius_fraction",
            "Joint Trust-Region Radius",
            0.01,
            1.0,
            0.01,
            2,
            "Trust region width around staged winner for joint_refine.",
        )
        self._add_bool(
            form,
            "optimizer_blocked_auto_cycle_tag",
            "Auto Rotate Cycle Tag",
            "Rotate study namespace by hour bucket to reduce stale anchoring.",
        )
        self._add_int(
            form,
            "optimizer_blocked_cycle_hours",
            "Cycle Hours",
            1,
            168,
            "Hours per automatic cycle tag bucket.",
        )
        self._add_text(
            form,
            "optimizer_study_tag",
            "Study Tag",
            "Optional suffix for study namespace (blank = none).",
        )
        self._add_text(
            form,
            "optimizer_blocked_cycle_tag",
            "Blocked Cycle Tag Override",
            "Optional explicit cycle tag; overrides auto rotation when non-empty.",
        )
        self._add_bool(
            form,
            "optimizer_blocked_stage_verbose",
            "Verbose Stage Logging",
            "When off, skip-save blocked stages emit compact logs only.",
        )
        self._add_bool(
            form,
            "optimizer_blocked_preserve_stage_champions",
            "Cross-Path Champion Playoff",
            "Keep stage winners (core/ff/onoff/joint) and promote the best objective champion.",
        )
        self._add_int(
            form,
            "optimizer_blocked_champion_log_top_k",
            "Champion Leaderboard Size",
            1,
            10,
            "How many top blocked candidates to print in playoff logs.",
        )
        self._add_bool(
            form,
            "optimizer_stage_champion_bank_enabled",
            "Persistent Stage Champion Bank",
            "Persist top stage champions across runs (per mode/stage) and reuse them.",
        )
        self._add_int(
            form,
            "optimizer_stage_champion_bank_top_k",
            "Champion Bank Size (Per Stage)",
            10,
            500,
            "Max persistent champions to keep for each stage (core/ff/onoff/joint).",
        )
        self._add_bool(
            form,
            "optimizer_stage_champion_bank_seed_enabled",
            "Seed Trials From Champion Bank",
            "Inject persistent stage champions as CMA-ES warm-start trials.",
        )
        self._add_int(
            form,
            "optimizer_stage_champion_bank_seed_max",
            "Champion Seed Max",
            0,
            500,
            "Maximum persisted champions per stage to inject as trial seeds (0 disables).",
        )
        self._add_bool(
            form,
            "optimizer_stage_champion_bank_clear_on_full_promotion",
            "Clear Bank On Full Promotion",
            "Reset persistent stage champions when both fundamentals and sharp promote in a pass.",
        )

        box.layout().addLayout(form)
        return box

    def _build_objective_card(self) -> QWidget:
        box = self._group("Objective & Robustness")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._add_float(
            form,
            "upset_bonus_mult",
            "Upset Bonus Multiplier",
            0.0,
            5.0,
            0.05,
            2,
            "Objective reward multiplier for upset accuracy x rate.",
        )
        self._add_choice(
            form,
            "optimizer_objective_primary_track",
            "Primary Objective Track",
            [
                ("dual_track", "Dual Track (live + oracle blend)"),
                ("live", "Live Only (walk-forward / rolling-CV)"),
                ("oracle", "Oracle Only (hindsight all-games)"),
            ],
            "Select which objective track drives optimizer trial ranking.",
        )
        self._add_float(
            form,
            "optimizer_objective_dual_live_weight",
            "Dual Track Live Weight",
            0.0,
            1.0,
            0.05,
            2,
            "Blend weight for live track when primary track is dual_track (oracle uses 1-live).",
        )
        self._add_choice(
            form,
            "optimizer_tanking_feature_mode",
            "Tanking Feature Mode",
            [
                ("both", "Both (live + oracle tank signals)"),
                ("live", "Live tank signal only"),
                ("oracle", "Oracle tank signal only"),
            ],
            "Controls which tanking signal channels are active in prediction/optimization scoring.",
        )
        self._add_bool(
            form,
            "optimizer_rolling_cv_enabled",
            "Enable Rolling CV Objective",
            "Use mean + worst-fold robust objective instead of train-only.",
        )
        self._add_int(
            form,
            "optimizer_rolling_cv_folds",
            "Rolling CV Folds",
            1,
            12,
            "Number of expanding-window fold slices.",
        )
        self._add_int(
            form,
            "optimizer_rolling_cv_min_train_games",
            "Rolling CV Min Train Games",
            120,
            5000,
            "Minimum training games in each rolling fold.",
        )
        self._add_int(
            form,
            "optimizer_rolling_cv_val_games",
            "Rolling CV Validation Games",
            40,
            2000,
            "Validation window length per fold.",
        )
        self._add_float(
            form,
            "optimizer_rolling_cv_worst_fold_mult",
            "Worst-Fold Penalty Multiplier",
            0.0,
            2.0,
            0.01,
            2,
            "How strongly worst fold is penalized versus mean fold.",
        )
        self._add_bool(
            form,
            "optimizer_objective_val_probe_enabled",
            "Enable Validation Probe Penalty",
            "Add penalties for train-only gains that do not generalize to probe slices.",
        )
        self._add_int(
            form,
            "optimizer_objective_val_probe_sample_size",
            "Probe Sample Size",
            50,
            5000,
            "Games per validation probe slice.",
        )
        self._add_int(
            form,
            "optimizer_objective_val_probe_slices",
            "Probe Slice Count",
            1,
            12,
            "Number of probe slices from validation period.",
        )
        self._add_float(
            form,
            "optimizer_objective_val_probe_loss_mult",
            "Probe Loss Mult",
            0.0,
            5.0,
            0.05,
            2,
            "Weight on probe loss regression component.",
        )
        self._add_float(
            form,
            "optimizer_objective_val_probe_winner_drop_mult",
            "Probe Winner Drop Mult",
            0.0,
            5.0,
            0.05,
            2,
            "Weight on probe winner% drop component.",
        )
        self._add_bool(
            form,
            "optimizer_objective_use_family_dominance_penalty",
            "Use Feature Family Dominance Penalty",
            "Soft p95 contribution penalties for FF and on/off families.",
        )
        self._add_float(
            form,
            "optimizer_objective_family_penalty_mult",
            "Family Penalty Mult",
            0.0,
            1.0,
            0.005,
            3,
            "Scaling for combined p95 excess penalties.",
        )
        self._add_float(
            form,
            "optimizer_objective_ff_p95_cap",
            "FF p95 Cap",
            0.0,
            500.0,
            1.0,
            1,
            "Cap for absolute p95 FF contribution before penalty.",
        )
        self._add_float(
            form,
            "optimizer_objective_onoff_p95_cap",
            "On/Off p95 Cap",
            0.0,
            200.0,
            1.0,
            1,
            "Cap for absolute p95 on/off contribution before penalty.",
        )
        self._add_float(
            form,
            "optimizer_objective_l2_prior_mult",
            "L2 Prior Penalty Mult",
            0.0,
            1.0,
            0.005,
            3,
            "Soft pull toward baseline champion to reduce gratuitous drift.",
        )
        self._add_float(
            form,
            "optimizer_onoff_player_minutes_smoothing",
            "On/Off Player Minute Smoothing",
            1.0,
            5000.0,
            10.0,
            1,
            "Minutes smoothing used for per-player on/off reliability.",
        )
        self._add_float(
            form,
            "optimizer_onoff_team_reliability_slots",
            "On/Off Team Reliability Slots",
            1.0,
            30.0,
            0.5,
            1,
            "Equivalent 30-minute slots for full team on/off reliability.",
        )

        box.layout().addLayout(form)
        return box

    def _build_ml_gate_card(self) -> QWidget:
        box = self._group("ML Underdog Gate")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._add_bool(
            form,
            "optimizer_ml_underdog_scorer_enabled",
            "Enable ML Promotion Gate",
            "Require candidate to pass ML underdog scorer diagnostics.",
        )
        self._add_float(
            form,
            "optimizer_ml_underdog_scorer_min_brier_lift",
            "Min Brier Lift",
            -0.05,
            0.05,
            0.0005,
            4,
            "Required baseline-candidate Brier lift (higher is better).",
        )
        self._add_int(
            form,
            "optimizer_ml_underdog_scorer_min_train_samples",
            "Min Train Samples",
            20,
            5000,
            "Minimum train upset samples for ML gate to apply.",
        )
        self._add_int(
            form,
            "optimizer_ml_underdog_scorer_min_val_samples",
            "Min Validation Samples",
            20,
            5000,
            "Minimum validation upset samples for ML gate to apply.",
        )
        self._add_float(
            form,
            "optimizer_ml_underdog_scorer_lr",
            "Learning Rate",
            0.0001,
            1.0,
            0.001,
            4,
            "Learning rate for logistic scorer fit.",
        )
        self._add_float(
            form,
            "optimizer_ml_underdog_scorer_l2",
            "L2 Regularization",
            0.0,
            50.0,
            0.05,
            3,
            "L2 regularization for logistic scorer fit.",
        )
        self._add_int(
            form,
            "optimizer_ml_underdog_scorer_max_iter",
            "Max Iterations",
            20,
            5000,
            "Maximum iterations for logistic scorer fit.",
        )

        box.layout().addLayout(form)
        return box

    def _build_save_gate_card(self) -> QWidget:
        box = self._group("Save Gate")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self._add_float(
            form,
            "optimizer_save_loss_margin",
            "Val Loss Improve Margin",
            0.0,
            1.0,
            0.001,
            3,
            "Candidate must improve validation loss by at least this margin.",
        )
        self._add_bool(
            form,
            "optimizer_save_use_hybrid_loss_gate",
            "Use Hybrid Loss Gate",
            "Blend validation and all-period losses for save decisions.",
        )
        self._add_float(
            form,
            "optimizer_save_hybrid_val_weight",
            "Hybrid Validation Weight",
            0.0,
            1.0,
            0.01,
            2,
            "Weight for validation loss in hybrid gate.",
        )
        self._add_float(
            form,
            "optimizer_save_hybrid_margin",
            "Hybrid Improve Margin",
            0.0,
            1.0,
            0.001,
            3,
            "Required hybrid loss improvement margin.",
        )
        self._add_float(
            form,
            "optimizer_save_max_val_loss_regress",
            "Max Val Loss Regression",
            0.0,
            1.0,
            0.001,
            3,
            "Allowed validation loss regression when other gates improve.",
        )
        self._add_bool(
            form,
            "optimizer_onepos_credit_enabled",
            "Enable 1P Near-Miss Credit",
            "Enable weighted one-possession near-miss credit for upset scoring.",
        )
        self._add_bool(
            form,
            "optimizer_onepos_credit_affects_winner_pct",
            "1P Credit Affects Winner%",
            "When disabled, one-possession credit does not modify winner%.",
        )
        self._add_float(
            form,
            "optimizer_onepos_credit_margin",
            "1P Credit Margin (pts)",
            0.0,
            20.0,
            0.5,
            1,
            "Maximum underdog loss margin counted as near-miss credit.",
        )
        self._add_float(
            form,
            "optimizer_onepos_credit_all_dogs_weight",
            "All-Dog Credit Weight",
            0.0,
            2.0,
            0.05,
            2,
            "Near-miss credit weight for non-long underdog picks.",
        )
        self._add_float(
            form,
            "optimizer_onepos_credit_long_dogs_weight",
            "Long-Dog Credit Weight",
            0.0,
            2.0,
            0.05,
            2,
            "Near-miss credit weight for long underdog picks.",
        )
        self._add_float(
            form,
            "optimizer_save_onepos_credit_bump_mult",
            "1P Credit Guard Bump Mult",
            0.0,
            2.0,
            0.01,
            2,
            "Multiplier converting one-possession credit lift into save-guard tolerance bump.",
        )
        self._add_bool(
            form,
            "optimizer_save_use_long_dog_tiebreak_gate",
            "Use Long-Dog Tiebreak Gate",
            "Require long-dog one-possession quality lift near tie scenarios.",
        )
        self._add_float(
            form,
            "optimizer_save_long_dog_tiebreak_loss_window",
            "Long-Dog Tiebreak Loss Window",
            0.0,
            1.0,
            0.001,
            3,
            "Loss window where long-dog tiebreak rule can decide promotion.",
        )

        box.layout().addLayout(form)
        return box

    def _build_advanced_overrides_card(self) -> QWidget:
        box = self._group("Advanced JSON Overrides")
        help_lbl = QLabel(
            "Optional: provide extra app_settings overrides as a JSON object. "
            "These are merged after all visible controls and saved before run."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        box.layout().addWidget(help_lbl)

        self._advanced_json = QPlainTextEdit()
        self._advanced_json.setPlaceholderText(
            "{\n"
            '  "optimizer_seed_disk_trials_enabled": true,\n'
            '  "optimizer_seed_disk_trials_max": 600\n'
            "}\n"
        )
        self._advanced_json.setFixedHeight(120)
        box.layout().addWidget(self._advanced_json)
        return box

    def _build_progress_panel(self) -> QWidget:
        box = self._group("Run Progress")

        stats_row = QHBoxLayout()
        stats_row.setSpacing(14)

        self._progress_pass_lbl = QLabel("Pass: -")
        self._progress_pass_lbl.setStyleSheet("font-weight: 700; color: #e2e8f0;")
        stats_row.addWidget(self._progress_pass_lbl)

        self._progress_phase_lbl = QLabel("Phase: Idle")
        self._progress_phase_lbl.setStyleSheet("font-weight: 700; color: #e2e8f0;")
        stats_row.addWidget(self._progress_phase_lbl)

        self._progress_activity_lbl = QLabel("Activity: -")
        self._progress_activity_lbl.setStyleSheet("color: #cbd5e1;")
        stats_row.addWidget(self._progress_activity_lbl, 1)
        box.layout().addLayout(stats_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p%")
        box.layout().addWidget(self._progress_bar)

        self._progress_meta_lbl = QLabel("No active progress yet.")
        self._progress_meta_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        box.layout().addWidget(self._progress_meta_lbl)

        self._steps_tbl = QTableWidget(len(RichOvernightConsole.PIPELINE_STEPS), 4)
        self._steps_tbl.setHorizontalHeaderLabels(["Step", "Status", "Detail", "Time"])
        self._steps_tbl.verticalHeader().setVisible(False)
        self._steps_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._steps_tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._steps_tbl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._steps_tbl.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self._steps_tbl.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self._steps_tbl.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.Stretch,
        )
        self._steps_tbl.horizontalHeader().setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        for row, step in enumerate(RichOvernightConsole.PIPELINE_STEPS):
            label = RichOvernightConsole.STEP_LABELS.get(step, step.replace("_", " ").title())
            self._set_table_cell(self._steps_tbl, row, 0, label, color=QColor("#cbd5e1"), bold=True)
            self._set_table_cell(self._steps_tbl, row, 1, "pending", color=QColor("#64748b"))
            self._set_table_cell(self._steps_tbl, row, 2, "", color=QColor("#94a3b8"))
            self._set_table_cell(self._steps_tbl, row, 3, "", color=QColor("#94a3b8"))
        self._steps_tbl.setMinimumHeight(260)
        self._steps_tbl.setMaximumHeight(320)
        box.layout().addWidget(self._steps_tbl)

        self._passes_tbl = QTableWidget(0, 11)
        self._passes_tbl.setHorizontalHeaderLabels(
            [
                "Pass",
                "Duration",
                "Winner%",
                "Upset%",
                "Upset Rate",
                "ML ROI",
                "Fund Gate",
                "Fund MLΔ",
                "Sharp Gate",
                "Sharp MLΔ",
                "Status",
            ]
        )
        self._passes_tbl.verticalHeader().setVisible(False)
        self._passes_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._passes_tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._passes_tbl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for col in range(self._passes_tbl.columnCount()):
            mode = QHeaderView.ResizeMode.ResizeToContents
            if col in (10,):
                mode = QHeaderView.ResizeMode.Stretch
            self._passes_tbl.horizontalHeader().setSectionResizeMode(col, mode)
        self._passes_tbl.setMinimumHeight(180)
        self._passes_tbl.setMaximumHeight(260)
        box.layout().addWidget(self._passes_tbl)

        self._playoff_lbl = QLabel("Blocked playoff leaderboard: waiting for stage champions.")
        self._playoff_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        self._playoff_lbl.setWordWrap(True)
        box.layout().addWidget(self._playoff_lbl)

        playoff_controls = QHBoxLayout()
        playoff_controls.setSpacing(8)
        self._playoff_current_mode_only_cb = QCheckBox("Show current mode only")
        self._playoff_current_mode_only_cb.setChecked(False)
        self._playoff_current_mode_only_cb.setToolTip(
            "When enabled, leaderboard shows only the currently optimizing mode "
            "(Fundamentals or Sharp)."
        )
        self._playoff_current_mode_only_cb.toggled.connect(
            lambda _checked: self._refresh_blocked_playoff_table()
        )
        playoff_controls.addWidget(self._playoff_current_mode_only_cb)
        playoff_controls.addStretch()
        box.layout().addLayout(playoff_controls)

        self._playoff_tbl = QTableWidget(0, 7)
        self._playoff_tbl.setHorizontalHeaderLabels(
            ["Mode", "Pass", "Rank", "Stage", "Objective", "Val Loss", "Winner%"]
        )
        self._playoff_tbl.verticalHeader().setVisible(False)
        self._playoff_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._playoff_tbl.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._playoff_tbl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._playoff_tbl.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self._playoff_tbl.setMinimumHeight(120)
        self._playoff_tbl.setMaximumHeight(180)
        box.layout().addWidget(self._playoff_tbl)

        self._best_lbl = QLabel("Best: -")
        self._best_lbl.setStyleSheet("color: #22c55e; font-weight: 700;")
        self._best_lbl.setWordWrap(True)
        box.layout().addWidget(self._best_lbl)

        return box

    def _build_logs_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("class", "broadcast-card")
        apply_card_shadow(panel, "md")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        progress_widget = self._build_progress_panel()
        layout.addWidget(progress_widget)

        title = QLabel("Live Overnight Log")
        title.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {CYAN}; letter-spacing: 0.5px;"
        )
        layout.addWidget(title)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._log, 1)

        row = QHBoxLayout()
        row.addStretch()
        clear_btn = QPushButton("CLEAR LOG")
        clear_btn.setProperty("class", "outline")
        clear_btn.clicked.connect(self._log.clear)
        row.addWidget(clear_btn)
        layout.addLayout(row)
        return panel

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    def _group(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: 700; color: #e2e8f0; }")
        apply_card_shadow(box, "sm")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 12, 12, 12)
        box_layout.setSpacing(8)
        return box

    @staticmethod
    def _label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 12px; color: #cbd5e1; font-weight: 600;")
        return lbl

    def _add_bool(self, form: QFormLayout, key: str, label: str, tooltip: str):
        widget = QCheckBox()
        widget.setToolTip(tooltip)
        form.addRow(self._label(label), widget)
        self._bindings[key] = _Binding(key=key, kind="bool", widget=widget)

    def _add_int(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        min_value: int,
        max_value: int,
        tooltip: str,
    ):
        widget = QSpinBox()
        widget.setRange(min_value, max_value)
        widget.setToolTip(tooltip)
        widget.setFixedWidth(160)
        form.addRow(self._label(label), widget)
        self._bindings[key] = _Binding(key=key, kind="int", widget=widget)

    def _add_float(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        min_value: float,
        max_value: float,
        step: float,
        decimals: int,
        tooltip: str,
    ):
        widget = QDoubleSpinBox()
        widget.setRange(min_value, max_value)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.setToolTip(tooltip)
        widget.setFixedWidth(160)
        form.addRow(self._label(label), widget)
        self._bindings[key] = _Binding(key=key, kind="float", widget=widget)

    def _add_choice(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        choices: list[tuple[str, str]],
        tooltip: str,
    ):
        widget = QComboBox()
        for value, text in choices:
            widget.addItem(text, value)
        widget.setToolTip(tooltip)
        widget.setMinimumWidth(240)
        form.addRow(self._label(label), widget)
        self._bindings[key] = _Binding(key=key, kind="choice", widget=widget)

    def _add_text(self, form: QFormLayout, key: str, label: str, tooltip: str):
        widget = QLineEdit()
        widget.setToolTip(tooltip)
        form.addRow(self._label(label), widget)
        self._bindings[key] = _Binding(key=key, kind="text", widget=widget)

    @staticmethod
    def _set_table_cell(
        table: QTableWidget,
        row: int,
        col: int,
        text: str,
        color: Optional[QColor] = None,
        bold: bool = False,
    ):
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText(text)
        if color is not None:
            item.setForeground(color)
        font = item.font()
        font.setBold(bool(bold))
        item.setFont(font)

    @staticmethod
    def _mode_short_label(mode: str) -> str:
        return "Fund" if mode == "fundamentals" else "Sharp"

    def _ensure_blocked_playoff_state(self, mode: str) -> Dict[str, Any]:
        existing = self._blocked_playoff.get(mode)
        if isinstance(existing, dict):
            return existing
        state = {
            "pass": 0,
            "retained": 0,
            "rows": {},
            "promoted_stage": "",
            "status": "idle",
        }
        self._blocked_playoff[mode] = state
        return state

    def _parse_blocked_playoff_message(self, msg: str):
        stripped = msg.strip()
        if not stripped:
            return

        parser = self._progress_parser
        active_mode = None
        if parser is not None:
            active_mode = getattr(parser, "_active_opt_target", None)
        lower = stripped.lower()
        if "optimizing fundamentals" in lower:
            active_mode = "fundamentals"
        elif "optimizing sharp" in lower:
            active_mode = "sharp"

        if active_mode in ("fundamentals", "sharp"):
            self._blocked_playoff_mode_context = active_mode
        mode = (
            active_mode
            if active_mode in ("fundamentals", "sharp")
            else self._blocked_playoff_mode_context
        )
        if mode not in ("fundamentals", "sharp"):
            return

        current_pass = int(parser.current_pass) if parser is not None and parser.current_pass > 0 else 0
        leaderboard_match = re.search(
            r"\[blocked\]\s+candidate playoff:\s*(\d+)\s+stage champions retained",
            stripped,
            flags=re.IGNORECASE,
        )
        if leaderboard_match:
            state = self._ensure_blocked_playoff_state(mode)
            state["pass"] = current_pass
            state["retained"] = int(leaderboard_match.group(1))
            state["rows"] = {}
            state["promoted_stage"] = ""
            state["status"] = "leaderboard"
            return

        row_match = re.search(
            r"\[blocked\]\s+#(\d+)\s+([A-Za-z_]+)\s+objective=([+\-]?\d+(?:\.\d+)?),\s*"
            r"val_loss=([+\-]?\d+(?:\.\d+)?),\s*winner=([+\-]?\d+(?:\.\d+)?)%",
            stripped,
            flags=re.IGNORECASE,
        )
        if row_match:
            state = self._ensure_blocked_playoff_state(mode)
            rank = int(row_match.group(1))
            rows = state.get("rows", {})
            if not isinstance(rows, dict):
                rows = {}
            rows[rank] = {
                "rank": rank,
                "stage": row_match.group(2),
                "objective": float(row_match.group(3)),
                "val_loss": float(row_match.group(4)),
                "winner_pct": float(row_match.group(5)),
            }
            state["rows"] = rows
            state["pass"] = current_pass
            state["status"] = "leaderboard"
            return

        promote_match = re.search(
            r"\[blocked\]\s+promoting champion from\s+([A-Za-z_]+)\s+pathway",
            stripped,
            flags=re.IGNORECASE,
        )
        if promote_match:
            state = self._ensure_blocked_playoff_state(mode)
            state["pass"] = current_pass
            state["promoted_stage"] = str(promote_match.group(1))
            state["status"] = "promoting"
            return

        if "[blocked] no valid stage champions found" in lower:
            state = self._ensure_blocked_playoff_state(mode)
            state["pass"] = current_pass
            state["status"] = "fallback"
            return

        if "[blocked] cross-path playoff disabled" in lower:
            state = self._ensure_blocked_playoff_state(mode)
            state["pass"] = current_pass
            state["status"] = "disabled"
            state["promoted_stage"] = "joint_refine"
            return

    def _refresh_blocked_playoff_table(self):
        parser = self._progress_parser
        if parser is not None:
            parser_mode = getattr(parser, "_active_opt_target", None)
            if parser_mode in ("fundamentals", "sharp"):
                self._blocked_playoff_mode_context = parser_mode

        show_current_only = bool(
            getattr(self, "_playoff_current_mode_only_cb", None)
            and self._playoff_current_mode_only_cb.isChecked()
        )
        selected_modes = ["fundamentals", "sharp"]
        if (
            show_current_only
            and self._blocked_playoff_mode_context in ("fundamentals", "sharp")
        ):
            selected_modes = [self._blocked_playoff_mode_context]

        entries: list[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        mode_states: list[tuple[str, Dict[str, Any]]] = []
        for mode in selected_modes:
            state = self._blocked_playoff.get(mode)
            if not isinstance(state, dict):
                continue
            mode_states.append((mode, state))
            rows = state.get("rows", {})
            if not isinstance(rows, dict):
                continue
            for rank in sorted(rows.keys()):
                row = rows.get(rank)
                if isinstance(row, dict):
                    entries.append((mode, state, row))

        if not mode_states:
            if (
                show_current_only
                and self._blocked_playoff_mode_context in ("fundamentals", "sharp")
            ):
                self._playoff_lbl.setText(
                    "Blocked playoff leaderboard "
                    f"({self._mode_short_label(self._blocked_playoff_mode_context)} only): "
                    "waiting for stage champions."
                )
            else:
                self._playoff_lbl.setText(
                    "Blocked playoff leaderboard: waiting for stage champions."
                )
            self._playoff_tbl.setRowCount(0)
            return

        status_parts = []
        for mode, state in mode_states:
            label = self._mode_short_label(mode)
            status = str(state.get("status", "idle"))
            retained = int(state.get("retained", 0) or 0)
            promoted = str(state.get("promoted_stage", "") or "").strip()
            if status == "disabled":
                status_parts.append(f"{label}: playoff disabled")
            elif promoted:
                status_parts.append(f"{label}: promoting {promoted}")
            elif retained > 0:
                status_parts.append(f"{label}: {retained} champions ranked")
            elif status == "fallback":
                status_parts.append(f"{label}: fallback to joint_refine")
            else:
                status_parts.append(f"{label}: awaiting leaderboard")
        title_prefix = "Blocked playoff"
        if (
            show_current_only
            and self._blocked_playoff_mode_context in ("fundamentals", "sharp")
        ):
            title_prefix += f" ({self._mode_short_label(self._blocked_playoff_mode_context)} only)"
        self._playoff_lbl.setText(title_prefix + ": " + " | ".join(status_parts))

        if not entries:
            self._playoff_tbl.setRowCount(0)
            return

        entries = entries[:14]
        self._playoff_tbl.setRowCount(len(entries))
        for idx, (mode, state, row) in enumerate(entries):
            rank = int(row.get("rank", 0) or 0)
            stage = str(row.get("stage", ""))
            objective = float(row.get("objective", 0.0) or 0.0)
            val_loss = float(row.get("val_loss", 0.0) or 0.0)
            winner_pct = float(row.get("winner_pct", 0.0) or 0.0)
            pass_num = int(state.get("pass", 0) or 0)
            promoted_stage = str(state.get("promoted_stage", "") or "").strip()
            is_promoted = bool(promoted_stage) and promoted_stage == stage

            mode_color = QColor("#cbd5e1") if mode == "fundamentals" else QColor("#9ec5ff")
            rank_color = QColor(GREEN if rank == 1 else "#cbd5e1")
            stage_color = QColor(GREEN if is_promoted else "#cbd5e1")

            self._set_table_cell(
                self._playoff_tbl,
                idx,
                0,
                self._mode_short_label(mode),
                color=mode_color,
                bold=(rank == 1),
            )
            self._set_table_cell(
                self._playoff_tbl,
                idx,
                1,
                str(pass_num) if pass_num > 0 else "-",
                color=QColor("#94a3b8"),
            )
            self._set_table_cell(
                self._playoff_tbl,
                idx,
                2,
                f"#{rank}" if rank > 0 else "-",
                color=rank_color,
                bold=(rank == 1),
            )
            self._set_table_cell(
                self._playoff_tbl,
                idx,
                3,
                f"{stage}{' *' if is_promoted else ''}",
                color=stage_color,
                bold=is_promoted,
            )
            self._set_table_cell(
                self._playoff_tbl,
                idx,
                4,
                f"{objective:.3f}",
                color=QColor("#cbd5e1"),
            )
            self._set_table_cell(
                self._playoff_tbl,
                idx,
                5,
                f"{val_loss:.3f}",
                color=QColor("#cbd5e1"),
            )
            self._set_table_cell(
                self._playoff_tbl,
                idx,
                6,
                f"{winner_pct:.1f}%",
                color=QColor("#cbd5e1"),
            )

    def _reset_progress_dashboard(self, max_hours: float):
        self._progress_parser = RichOvernightConsole(max_hours=max_hours)
        self._blocked_playoff = {}
        self._blocked_playoff_mode_context = "fundamentals"
        self._progress_parser.start_time = time.time()
        self._refresh_progress_dashboard()

    def _clear_progress_dashboard(self):
        self._progress_parser = None
        self._blocked_playoff = {}
        self._blocked_playoff_mode_context = "fundamentals"
        self._progress_pass_lbl.setText("Pass: -")
        self._progress_phase_lbl.setText("Phase: Idle")
        self._progress_activity_lbl.setText("Activity: -")
        self._progress_bar.setValue(0)
        self._progress_meta_lbl.setText("No active progress yet.")
        self._playoff_lbl.setText(
            "Blocked playoff leaderboard: waiting for stage champions."
        )
        self._playoff_tbl.setRowCount(0)
        self._passes_tbl.setRowCount(0)
        self._best_lbl.setText("Best: -")
        for row in range(self._steps_tbl.rowCount()):
            self._set_table_cell(self._steps_tbl, row, 1, "pending", color=QColor("#64748b"))
            self._set_table_cell(self._steps_tbl, row, 2, "", color=QColor("#94a3b8"))
            self._set_table_cell(self._steps_tbl, row, 3, "", color=QColor("#94a3b8"))

    def _refresh_progress_dashboard(self):
        parser = self._progress_parser
        if parser is None:
            self._clear_progress_dashboard()
            return

        pass_text = str(parser.current_pass) if parser.current_pass > 0 else "-"
        phase = "Pipeline" if parser.in_pipeline else "Optimization / Loop"
        if not self._running and parser.current_activity == "Initializing...":
            phase = "Idle"
        self._progress_pass_lbl.setText(f"Pass: {pass_text}")
        self._progress_phase_lbl.setText(f"Phase: {phase}")
        self._progress_activity_lbl.setText(f"Activity: {parser.current_activity}")

        elapsed = max(0.0, time.time() - parser.start_time)
        remaining = max(0.0, parser.max_hours * 3600 - elapsed)

        if parser.trial_total > 0:
            done = int(parser.trial_current)
            if parser.opt_start_trial > 0:
                done = max(0, parser.trial_current - parser.opt_start_trial)
            pct = min(100.0, (done / parser.trial_total) * 100.0)
            self._progress_bar.setValue(int(round(pct)))
            eta_fragment = ""
            if done > 5 and parser.opt_start_time > 0:
                elapsed_opt = time.time() - parser.opt_start_time
                rate = done / elapsed_opt if elapsed_opt > 0 else 0.0
                if rate > 0:
                    left = max(0, parser.trial_total - done)
                    eta_fragment = f" | ETA {self._fmt_hms(left / rate)}"
            resume_fragment = (
                f" | resumed from {parser.opt_start_trial:,}"
                if parser.opt_start_trial > 0
                else ""
            )
            self._progress_meta_lbl.setText(
                f"Progress {done:,}/{parser.trial_total:,} ({pct:.0f}%)"
                f"{resume_fragment}{eta_fragment}"
                f" | elapsed {self._fmt_hms(elapsed)}"
                f" | remaining {self._fmt_hms(remaining)}"
            )
        else:
            self._progress_bar.setValue(0)
            self._progress_meta_lbl.setText(
                f"elapsed {self._fmt_hms(elapsed)} | remaining {self._fmt_hms(remaining)}"
            )

        for row, step in enumerate(RichOvernightConsole.PIPELINE_STEPS):
            status = parser.step_status.get(step, "pending")
            detail = parser.step_detail.get(step, "")
            if step == "sync" and not detail and parser.sync_current_label:
                detail = parser.sync_current_label
            elapsed_step = parser.step_times.get(step)
            elapsed_text = f"{elapsed_step:.0f}s" if elapsed_step is not None else ""
            status_text = status
            status_color = QColor("#64748b")
            if status == "done":
                status_text = "done"
                status_color = QColor(GREEN)
            elif status == "running":
                status_text = "running"
                status_color = QColor(CYAN)
            elif status == "failed":
                status_text = "failed"
                status_color = QColor(RED)
            self._set_table_cell(
                self._steps_tbl,
                row,
                1,
                status_text,
                color=status_color,
                bold=(status == "running"),
            )
            self._set_table_cell(self._steps_tbl, row, 2, detail, color=QColor("#cbd5e1"))
            self._set_table_cell(self._steps_tbl, row, 3, elapsed_text, color=QColor("#94a3b8"))

        visible_results = parser.pass_results[-40:]
        self._passes_tbl.setRowCount(len(visible_results))
        for row, result in enumerate(visible_results):
            pass_num = int(result.get("pass", 0) or 0)
            winner_pct = result.get("winner_pct")
            upset_pct = result.get("upset_pct")
            upset_rate = result.get("upset_rate")
            ml_roi = result.get("ml_roi")
            is_best = bool(result.get("is_best", False))

            fund_gate, fund_gate_color = self._gate_ui_text(parser, pass_num, "fundamentals")
            sharp_gate, sharp_gate_color = self._gate_ui_text(parser, pass_num, "sharp")
            fund_ml, fund_ml_color = self._ml_delta_ui_text(parser, pass_num, "fundamentals")
            sharp_ml, sharp_ml_color = self._ml_delta_ui_text(parser, pass_num, "sharp")

            self._set_table_cell(self._passes_tbl, row, 0, str(pass_num), color=QColor("#94a3b8"))
            self._set_table_cell(
                self._passes_tbl,
                row,
                1,
                self._fmt_hms(float(result.get("duration", 0.0) or 0.0)),
                color=QColor("#cbd5e1"),
            )
            self._set_table_cell(
                self._passes_tbl,
                row,
                2,
                f"{float(winner_pct):.1f}%" if isinstance(winner_pct, (int, float)) else "-",
                color=QColor("#cbd5e1"),
                bold=is_best,
            )
            self._set_table_cell(
                self._passes_tbl,
                row,
                3,
                f"{float(upset_pct):.0f}%"
                if isinstance(upset_pct, (int, float)) and float(upset_pct) > 0
                else "-",
                color=QColor("#cbd5e1"),
            )
            self._set_table_cell(
                self._passes_tbl,
                row,
                4,
                f"{float(upset_rate):.0f}%"
                if isinstance(upset_rate, (int, float)) and float(upset_rate) > 0
                else "-",
                color=QColor("#cbd5e1"),
            )
            self._set_table_cell(
                self._passes_tbl,
                row,
                5,
                f"{float(ml_roi):+.1f}%"
                if isinstance(ml_roi, (int, float)) and float(ml_roi) != 0.0
                else "-",
                color=QColor("#cbd5e1"),
            )
            self._set_table_cell(self._passes_tbl, row, 6, fund_gate, color=fund_gate_color)
            self._set_table_cell(self._passes_tbl, row, 7, fund_ml, color=fund_ml_color)
            self._set_table_cell(self._passes_tbl, row, 8, sharp_gate, color=sharp_gate_color)
            self._set_table_cell(self._passes_tbl, row, 9, sharp_ml, color=sharp_ml_color)
            self._set_table_cell(
                self._passes_tbl,
                row,
                10,
                "NEW BEST" if is_best else "no change",
                color=QColor(GREEN if is_best else "#94a3b8"),
                bold=is_best,
            )

        best = parser.best_result
        if isinstance(best, dict) and best:
            self._best_lbl.setText(
                "Best: "
                f"Winner={float(best.get('winner_pct', 0.0)):.1f}% | "
                f"Upset={float(best.get('upset_pct', 0.0)):.0f}% @ "
                f"{float(best.get('upset_rate', 0.0)):.0f}% | "
                f"ML ROI={float(best.get('ml_roi', 0.0)):+.1f}% | "
                f"Pass {int(best.get('pass', 0) or 0)}"
            )
        else:
            self._best_lbl.setText("Best: -")
        self._refresh_blocked_playoff_table()

    @staticmethod
    def _gate_ui_text(
        parser: RichOvernightConsole,
        pass_num: int,
        mode: str,
    ) -> tuple[str, QColor]:
        gate = parser.pass_gate.get(pass_num, {}).get(mode)
        if not gate:
            return "-", QColor("#64748b")
        if bool(gate.get("saved")):
            return "SAVED", QColor(GREEN)
        return "REJECTED", QColor(RED)

    @staticmethod
    def _ml_delta_ui_text(
        parser: RichOvernightConsole,
        pass_num: int,
        mode: str,
    ) -> tuple[str, QColor]:
        entry = parser.pass_ml_gate.get(pass_num, {}).get(mode)
        if not entry:
            return "-", QColor("#64748b")
        status = str(entry.get("status", "off"))
        if status == "off":
            return "OFF", QColor("#64748b")
        if status == "skip":
            return "SKIP", QColor(AMBER)
        lift = entry.get("brier_lift")
        if lift is None:
            return "-", QColor("#64748b")
        passed = bool(entry.get("passed", False))
        return f"{float(lift):+0.4f}", QColor(GREEN if passed else RED)

    # ------------------------------------------------------------------
    # Settings load/save
    # ------------------------------------------------------------------

    def _load_current_settings(self):
        settings = load_settings()
        for key, binding in self._bindings.items():
            value = settings.get(key)
            self._set_binding_value(binding, value)

        self._hours_spin.setValue(float(get_setting("overnight_ui_default_hours", 8.0)))
        self._append_log("Loaded settings from data/app_settings.json")

    def _set_binding_value(self, binding: _Binding, value: Any):
        widget = binding.widget
        if binding.kind == "bool":
            widget.setChecked(bool(value))
        elif binding.kind == "int":
            try:
                widget.setValue(int(value))
            except Exception:
                pass
        elif binding.kind == "float":
            try:
                widget.setValue(float(value))
            except Exception:
                pass
        elif binding.kind == "choice":
            idx = widget.findData(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif binding.kind == "text":
            widget.setText("" if value is None else str(value))

    def _collect_binding_values(self) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for key, binding in self._bindings.items():
            widget = binding.widget
            if binding.kind == "bool":
                values[key] = bool(widget.isChecked())
            elif binding.kind == "int":
                values[key] = int(widget.value())
            elif binding.kind == "float":
                values[key] = float(widget.value())
            elif binding.kind == "choice":
                values[key] = str(widget.currentData())
            elif binding.kind == "text":
                values[key] = str(widget.text()).strip()
        return values

    def _parse_advanced_overrides(self) -> Dict[str, Any]:
        raw = self._advanced_json.toPlainText().strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Advanced JSON overrides: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Advanced JSON overrides must be an object.")
        return parsed

    def _apply_settings(self):
        settings = load_settings()
        settings.update(self._collect_binding_values())
        settings.update(self._parse_advanced_overrides())
        save_settings(settings)
        invalidate_cache()
        self._append_log("Applied settings to data/app_settings.json")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _apply_preset(self, preset_name: str):
        presets: Dict[str, Dict[str, Any]] = {
            "balanced": {
                "optimizer_tuning_mode": "blocked",
                "optimizer_use_wide_ranges": True,
                "optimizer_ml_underdog_scorer_enabled": True,
                "optimizer_ml_underdog_scorer_min_brier_lift": 0.0,
                "optimizer_blocked_stage_verbose": False,
                "optimizer_log_interval": 300,
                "precompute_progress_log_every": 200,
                "overnight_max_no_save_passes": 3,
            },
            "exploration": {
                "optimizer_tuning_mode": "blocked",
                "optimizer_use_wide_ranges": True,
                "optimizer_ml_underdog_scorer_enabled": False,
                "optimizer_blocked_core_fraction": 0.30,
                "optimizer_blocked_ff_fraction": 0.25,
                "optimizer_blocked_onoff_fraction": 0.20,
                "optimizer_blocked_joint_fraction": 0.25,
                "optimizer_blocked_stage_verbose": True,
                "optimizer_log_interval": 150,
                "precompute_progress_log_every": 300,
            },
            "strict": {
                "optimizer_tuning_mode": "blocked",
                "optimizer_ml_underdog_scorer_enabled": True,
                "optimizer_ml_underdog_scorer_min_brier_lift": 0.001,
                "optimizer_save_use_hybrid_loss_gate": True,
                "optimizer_save_use_long_dog_tiebreak_gate": True,
                "optimizer_save_max_val_loss_regress": 0.03,
                "optimizer_save_hybrid_margin": 0.003,
                "optimizer_blocked_stage_verbose": False,
                "overnight_max_no_save_passes": 2,
            },
            "quiet_logs": {
                "optimizer_blocked_stage_verbose": False,
                "optimizer_log_interval": 600,
                "precompute_progress_log_every": 400,
            },
        }
        values = presets.get(preset_name)
        if not values:
            return
        for key, value in values.items():
            binding = self._bindings.get(key)
            if binding is not None:
                self._set_binding_value(binding, value)
        self._append_log(f"Applied preset: {preset_name}")

    # ------------------------------------------------------------------
    # Run/cancel lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_team_ids_csv(raw: str) -> list[int]:
        if not raw:
            return []
        parts = re.split(r"[,\s]+", raw.strip())
        parsed: list[int] = []
        for part in parts:
            if not part:
                continue
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                parsed.append(value)
        return _dedupe_ordered_team_ids(parsed)

    def _get_failed_onoff_team_ids_from_log(self) -> list[int]:
        log_path = Path("data") / "overnight.log"
        if not log_path.exists():
            return []
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.debug("Could not read overnight log for failed on/off IDs: %s", exc)
            return []
        return _extract_latest_failed_onoff_team_ids(log_text)

    def _on_repair_onoff_clicked(self):
        if self._running or self._repair_running:
            return

        explicit_ids = self._parse_team_ids_csv(self._repair_team_ids_input.text())
        team_ids = explicit_ids if explicit_ids else self._get_failed_onoff_team_ids_from_log()

        if not team_ids:
            QMessageBox.information(
                self,
                "Repair On/Off Teams",
                "No team IDs provided and no failed on/off team IDs were found in data/overnight.log.",
            )
            self._append_log(
                "Repair skipped: no explicit team IDs and no failed on/off team IDs found in overnight.log."
            )
            return

        self._set_repair_running(True)
        self._append_log(
            "Starting on/off repair sync for team IDs: "
            + ", ".join(str(tid) for tid in team_ids)
        )

        self._repair_worker = _OnOffRepairWorker(team_ids=team_ids)
        self._repair_thread = QThread(self)
        self._repair_worker.moveToThread(self._repair_thread)

        self._repair_thread.started.connect(self._repair_worker.run)
        self._repair_worker.progress.connect(self._on_repair_progress)
        self._repair_worker.finished.connect(self._on_repair_finished)
        self._repair_worker.failed.connect(self._on_repair_failed)
        self._repair_worker.finished.connect(self._repair_thread.quit)
        self._repair_worker.failed.connect(lambda _msg: self._repair_thread.quit())
        self._repair_worker.finished.connect(self._repair_worker.deleteLater)
        self._repair_thread.finished.connect(self._repair_thread.deleteLater)
        self._repair_thread.start()

    def _on_repair_progress(self, msg: str):
        self._append_log(f"[repair] {msg}")

    def _on_repair_finished(self, result: object):
        self._set_repair_running(False)
        summary = "On/off repair sync complete."
        if isinstance(result, dict):
            stats = result.get("stats", [])
            if isinstance(stats, list) and stats:
                formatted = []
                for row in stats:
                    try:
                        tid = int(row.get("team_id"))
                        n_nonzero = int(row.get("n_nonzero") or 0)
                        n_players = int(row.get("n_players") or 0)
                    except Exception:
                        continue
                    formatted.append(f"{tid}: {n_nonzero}/{n_players} non-zero")
                if formatted:
                    summary += " " + " | ".join(formatted)
        self._append_log(summary)
        self._repair_worker = None
        self._repair_thread = None
        QMessageBox.information(self, "Repair Complete", summary)

    def _on_repair_failed(self, message: str):
        self._set_repair_running(False)
        self._append_log(f"On/off repair failed: {message}")
        self._repair_worker = None
        self._repair_thread = None
        QMessageBox.critical(self, "Repair Failed", message)

    def _set_repair_running(self, running: bool):
        self._repair_running = running
        busy = self._running or self._repair_running
        self._run_btn.setEnabled(not busy)
        self._reload_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(self._running)
        self._hours_spin.setEnabled(not busy)
        self._reset_weights_cb.setEnabled(not busy)
        self._repair_team_ids_input.setEnabled(not busy)
        self._repair_onoff_btn.setEnabled(not busy)
        for binding in self._bindings.values():
            binding.widget.setEnabled(not busy)
        self._advanced_json.setEnabled(not busy)

        if running and not self._running:
            self._status_lbl.setText("Repairing On/Off...")
            self._status_lbl.setStyleSheet(f"color: {AMBER}; font-weight: 700;")
        elif not self._running:
            self._status_lbl.setText("Idle")
            self._status_lbl.setStyleSheet(f"color: {GREEN}; font-weight: 700;")

    def _on_run_clicked(self):
        if self._running or self._repair_running:
            return

        try:
            self._apply_settings()
            # Reuse the exact overnight.py file logger wiring per run.
            configure_overnight_file_logging()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Settings", str(exc))
            return
        except Exception as exc:
            logger.error("Failed to apply settings: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Settings Error", str(exc))
            return

        max_hours = float(self._hours_spin.value())
        reset_weights = bool(self._reset_weights_cb.isChecked())

        self._set_running(True)
        self._run_start_ts = time.time()
        self._elapsed_lbl.setText("00:00:00")
        self._reset_progress_dashboard(max_hours=max_hours)
        self._append_log(
            f"Starting overnight: budget={max_hours:.2f}h, reset_weights={reset_weights}"
        )

        self._worker = _OvernightWorker(max_hours=max_hours, reset_weights=reset_weights)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(lambda _msg: self._thread.quit())
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_cancel_clicked(self):
        if not self._running or self._worker is None:
            return
        self._append_log("Cancellation requested...")
        self._worker.cancel()
        self._status_lbl.setText("Cancelling...")
        self._status_lbl.setStyleSheet(f"color: {AMBER}; font-weight: 700;")

    def _on_worker_progress(self, msg: str):
        if self._progress_parser is not None:
            self._progress_parser.callback(msg)
            self._parse_blocked_playoff_message(msg)
            self._refresh_progress_dashboard()
        self._append_log(msg)

    def _on_worker_finished(self, result: object):
        self._set_running(False)
        elapsed = time.time() - self._run_start_ts if self._run_start_ts > 0 else 0.0
        self._status_lbl.setText("Complete")
        self._status_lbl.setStyleSheet(f"color: {GREEN}; font-weight: 700;")
        self._append_log(f"Overnight finished in {self._fmt_hms(elapsed)}")

        summary = self._format_result_summary(result)
        if summary:
            self._append_log(summary)

        self._refresh_progress_dashboard()
        self._worker = None
        self._thread = None
        if self._close_after_cancel:
            self.close()

    def _on_worker_failed(self, message: str):
        self._set_running(False)
        self._status_lbl.setText("Error")
        self._status_lbl.setStyleSheet(f"color: {RED}; font-weight: 700;")
        self._append_log(f"ERROR: {message}")
        self._refresh_progress_dashboard()
        self._worker = None
        self._thread = None
        if self._close_after_cancel:
            self.close()
        QMessageBox.critical(self, "Overnight Error", message)

    def _set_running(self, running: bool):
        self._running = running
        busy = self._running or self._repair_running
        self._run_btn.setEnabled(not busy)
        self._reload_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(self._running)
        self._hours_spin.setEnabled(not busy)
        self._reset_weights_cb.setEnabled(not busy)
        self._repair_team_ids_input.setEnabled(not busy)
        self._repair_onoff_btn.setEnabled(not busy)

        # Disable all setting controls while running.
        for binding in self._bindings.values():
            binding.widget.setEnabled(not busy)
        self._advanced_json.setEnabled(not busy)

        if running:
            self._status_lbl.setText("Running")
            self._status_lbl.setStyleSheet(f"color: {CYAN}; font-weight: 700;")
            self._elapsed_timer.start()
        else:
            self._elapsed_timer.stop()

    def _update_elapsed_label(self):
        if not self._running or self._run_start_ts <= 0:
            return
        elapsed = time.time() - self._run_start_ts
        self._elapsed_lbl.setText(self._fmt_hms(elapsed))
        self._refresh_progress_dashboard()

    # ------------------------------------------------------------------
    # Formatting and logging
    # ------------------------------------------------------------------

    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{ts}] {msg}")
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    @staticmethod
    def _fmt_hms(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _format_result_summary(result: object) -> str:
        if not isinstance(result, dict):
            return ""
        parts = []
        passes = result.get("passes")
        attempted = result.get("attempted_passes")
        elapsed = result.get("elapsed_seconds")
        if passes is not None:
            parts.append(f"passes={passes}")
        if attempted is not None:
            parts.append(f"attempted={attempted}")
        if elapsed is not None:
            parts.append(f"elapsed={elapsed}s")

        best = result.get("best_backtest", {})
        fundamentals = best.get("fundamentals", {}) if isinstance(best, dict) else {}
        if isinstance(fundamentals, dict) and fundamentals:
            winner = fundamentals.get("winner_pct")
            upset = fundamentals.get("upset_accuracy")
            upset_rate = fundamentals.get("upset_rate")
            roi = fundamentals.get("ml_roi")
            parts.append(
                "best_fund="
                f"winner:{winner:.1f}% upset:{upset:.1f}%@{upset_rate:.1f}% roi:{roi:+.2f}%"
                if all(isinstance(v, (int, float)) for v in (winner, upset, upset_rate, roi))
                else "best_fund=available"
            )
        return "Summary: " + ", ".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 - Qt API
        if self._running:
            answer = QMessageBox.question(
                self,
                "Overnight Running",
                "An overnight run is still active. Cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._on_cancel_clicked()
            self._close_after_cancel = True
            self._append_log("Waiting for cancellation to finish before closing...")
            event.ignore()
            return
        if self._repair_running:
            QMessageBox.information(
                self,
                "Repair Running",
                "On/off repair is still in progress. Please wait for completion before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)


def main():
    setup_logging()
    app = QApplication(sys.argv)
    setup_theme(app)
    bootstrap(enable_daily_automation=False)

    window = OvernightControlCenter()
    window.show()

    code = app.exec()
    shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
