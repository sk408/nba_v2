"""Settings tab -- application configuration and weight management.

Prediction mode, optimizer save gate, upset bonus, sync freshness,
worker threads, theme, and weight reset with confirmation dialog.
"""

import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox,
    QRadioButton, QButtonGroup, QGroupBox, QSizePolicy,
    QMessageBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor

from src.ui.theme import apply_card_shadow

logger = logging.getLogger(__name__)

# ---- Theme colors ----
CYAN = "#00E5FF"
GREEN = "#00E676"
AMBER = "#FFB300"
RED = "#FF5252"
TEXT_PRIMARY = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"


class _CDWorker(QObject):
    """Runs coordinate descent on a background thread."""
    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, include_sharp: bool, steps: int, max_rounds: int):
        super().__init__()
        self.include_sharp = include_sharp
        self.steps = steps
        self.max_rounds = max_rounds
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from src.database.db import thread_local_db
            thread_local_db()
            from src.analytics.prediction import precompute_all_games
            from src.analytics.optimizer import coordinate_descent

            self.progress.emit("Loading precomputed games...")
            games = precompute_all_games(
                callback=lambda msg: self.progress.emit(msg),
            )
            if not games:
                self.error.emit("No games available for CD")
                return

            self.progress.emit(f"Starting CD: {len(games)} games, "
                              f"steps={self.steps}, rounds={self.max_rounds}")

            result = coordinate_descent(
                games=games,
                steps=self.steps,
                max_rounds=self.max_rounds,
                include_sharp=self.include_sharp,
                callback=lambda msg: self.progress.emit(msg),
                is_cancelled=lambda: self._cancelled,
                save=True,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ----------------------------------------------------------------
# SettingsView -- main widget
# ----------------------------------------------------------------

class SettingsView(QWidget):
    """Application settings and configuration view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(16)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Header
        header = QLabel("Settings")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # ---- Section 1: Prediction Mode ----
        self._build_prediction_mode(layout)

        # ---- Section 2: Upset Bonus Multiplier ----
        self._build_upset_bonus(layout)

        # ---- Section 3: Optimizer Save Gate ----
        self._build_optimizer_save_gate(layout)

        # ---- Section 4: Sync & Performance ----
        self._build_sync_performance(layout)

        # ---- Section 5: Theme ----
        self._build_theme_section(layout)

        # ---- Section 6: Coordinate Descent ----
        self._build_cd_section(layout)

        # ---- Section 7: Weight Management ----
        self._build_weight_management(layout)

        layout.addStretch()

        # Load current values
        self._load_current_settings()

    # ---------------------------------------------------------------
    # UI builders
    # ---------------------------------------------------------------

    def _build_prediction_mode(self, parent_layout: QVBoxLayout):
        """Build prediction mode radio group."""
        group = QGroupBox("Prediction Mode")
        gl = QVBoxLayout(group)
        gl.setSpacing(8)

        desc = QLabel(
            "Choose the default prediction mode for matchup analysis. "
            "'Fundamentals Only' uses team stats and situational factors. "
            "'Fundamentals + Sharp' also incorporates sharp money signals."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc)

        self._mode_group = QButtonGroup(self)

        self._mode_fund = QRadioButton("Fundamentals Only")
        self._mode_fund.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        self._mode_group.addButton(self._mode_fund, 0)
        gl.addWidget(self._mode_fund)

        self._mode_sharp = QRadioButton("Fundamentals + Sharp")
        self._mode_sharp.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        self._mode_group.addButton(self._mode_sharp, 1)
        gl.addWidget(self._mode_sharp)

        self._mode_group.idToggled.connect(self._on_mode_changed)

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    def _build_upset_bonus(self, parent_layout: QVBoxLayout):
        """Build upset bonus multiplier slider."""
        group = QGroupBox("Upset Bonus Multiplier")
        gl = QVBoxLayout(group)
        gl.setSpacing(8)

        desc = QLabel(
            "Controls how much the optimizer rewards upset (dog) picks. "
            "Higher values push the model toward more aggressive upset bets."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(12)

        self._upset_slider = QSlider(Qt.Orientation.Horizontal)
        self._upset_slider.setRange(0, 500)  # 0.0 to 5.0 in 0.01 steps
        self._upset_slider.setTickInterval(10)
        self._upset_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._upset_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._upset_slider.valueChanged.connect(self._on_upset_slider_changed)
        row.addWidget(self._upset_slider)

        self._upset_value_lbl = QLabel("0.50")
        self._upset_value_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 18px; font-weight: 700; "
            f"font-family: 'Oswald'; min-width: 50px;"
        )
        self._upset_value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._upset_value_lbl)

        gl.addLayout(row)

        # Min/max labels
        range_row = QHBoxLayout()
        min_lbl = QLabel("0.0")
        min_lbl.setProperty("class", "muted")
        range_row.addWidget(min_lbl)
        range_row.addStretch()
        self._upset_max_lbl = QLabel("5.0")
        self._upset_max_lbl.setProperty("class", "muted")
        range_row.addWidget(self._upset_max_lbl)
        gl.addLayout(range_row)

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    def _build_optimizer_save_gate(self, parent_layout: QVBoxLayout):
        """Build anti-gaming optimizer save gate controls."""
        group = QGroupBox("Optimizer Save Gate (Anti-Gaming)")
        gl = QGridLayout(group)
        gl.setSpacing(10)
        gl.setContentsMargins(16, 20, 16, 16)

        desc = QLabel(
            "Controls when optimized weights are allowed to save. "
            "The gate uses validation loss plus upset-lift quality. "
            "ROI can be optionally enabled as an additional hard gate, "
            "and long-dog one-possession quality can be used as a near-loss tiebreak."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc, 0, 0, 1, 4)

        self._sg_use_roi_gate_chk = QCheckBox("Use ROI as hard save gate")
        self._sg_use_roi_gate_chk.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"
        )
        self._sg_use_roi_gate_chk.setToolTip(
            "When disabled, ROI is diagnostic-only and does not block saves."
        )
        self._sg_use_roi_gate_chk.toggled.connect(self._on_roi_gate_toggled)
        gl.addWidget(self._sg_use_roi_gate_chk, 1, 0, 1, 4)

        self._sg_use_long_dog_tiebreak_chk = QCheckBox(
            "Enable long-dog one-possession tiebreak"
        )
        self._sg_use_long_dog_tiebreak_chk.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"
        )
        self._sg_use_long_dog_tiebreak_chk.setToolTip(
            "Allows near-equal-loss saves only when long-dog one-possession quality improves."
        )
        self._sg_use_long_dog_tiebreak_chk.toggled.connect(
            self._on_long_dog_tiebreak_toggled
        )
        gl.addWidget(self._sg_use_long_dog_tiebreak_chk, 2, 0, 1, 4)

        auto_note = QLabel("Count fields: set to 0 for auto thresholds by validation sample size.")
        auto_note.setProperty("class", "muted")
        auto_note.setWordWrap(True)
        gl.addWidget(auto_note, 3, 0, 1, 4)

        label_style = f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600;"

        def _double_spin(
            min_v: float,
            max_v: float,
            step: float,
            decimals: int,
            key: str,
            suffix: str = "",
            tooltip: str = "",
        ) -> QDoubleSpinBox:
            box = QDoubleSpinBox()
            box.setRange(min_v, max_v)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setKeyboardTracking(False)
            box.setFixedWidth(120)
            if suffix:
                box.setSuffix(suffix)
            if tooltip:
                box.setToolTip(tooltip)
            box.valueChanged.connect(
                lambda value, setting_key=key: self._on_gate_float_changed(setting_key, value)
            )
            return box

        def _int_spin(
            min_v: int,
            max_v: int,
            key: str,
            tooltip: str = "",
            special_zero_text: str = "",
        ) -> QSpinBox:
            box = QSpinBox()
            box.setRange(min_v, max_v)
            box.setKeyboardTracking(False)
            box.setFixedWidth(120)
            if special_zero_text:
                box.setSpecialValueText(special_zero_text)
            if tooltip:
                box.setToolTip(tooltip)
            box.valueChanged.connect(
                lambda value, setting_key=key: self._on_gate_int_changed(setting_key, value)
            )
            return box

        self._sg_loss_margin_spin = _double_spin(
            0.0, 1.0, 0.0001, 4, "optimizer_save_loss_margin",
            tooltip="Required loss improvement before new weights can save.",
        )
        self._sg_min_weight_delta_spin = _double_spin(
            0.0, 0.01, 0.0001, 4, "optimizer_save_min_weight_delta",
            tooltip="Minimum max per-parameter weight delta required to save.",
        )
        self._sg_compression_floor_spin = _double_spin(
            0.1, 2.0, 0.01, 2, "optimizer_save_compression_floor",
            tooltip="Rejects compressed prediction bands below this floor.",
        )
        self._sg_max_winner_drop_spin = _double_spin(
            0.0, 10.0, 0.05, 2, "optimizer_save_max_winner_drop", " pp",
            "Maximum allowed winner%% drop versus baseline (percentage points).",
        )
        self._sg_favorites_slack_spin = _double_spin(
            0.0, 10.0, 0.05, 2, "optimizer_save_favorites_slack", " pp",
            "How far winner%% may sit below favorites baseline.",
        )
        self._sg_min_ml_payout_spin = _double_spin(
            1.01, 5.00, 0.05, 2, "optimizer_min_ml_payout", "x",
            "Minimum total return multiplier for ML bets used in ROI/gate checks. "
            "1.50x means risk 100 to return 150 total.",
        )
        self._sg_min_upset_count_spin = _int_spin(
            0, 5000, "optimizer_save_min_upset_count",
            "Minimum upset sample size. 0 = auto by validation sample.",
            "Auto",
        )
        self._sg_min_ml_bets_spin = _int_spin(
            0, 5000, "optimizer_save_min_ml_bets",
            "Minimum ML bets for ROI checks. 0 = auto by validation sample.",
            "Auto",
        )
        self._sg_min_upset_rate_spin = _double_spin(
            0.0, 100.0, 0.5, 1, "optimizer_save_min_upset_rate", "%",
            "Lower bound for upset pick-rate sanity check.",
        )
        self._sg_max_upset_rate_spin = _double_spin(
            0.0, 100.0, 0.5, 1, "optimizer_save_max_upset_rate", "%",
            "Upper bound for upset pick-rate sanity check.",
        )
        self._sg_upset_prior_weight_spin = _double_spin(
            0.0, 500.0, 1.0, 1, "optimizer_save_upset_prior_weight",
            tooltip="Shrinkage strength toward underdog baseline upset accuracy.",
        )
        self._sg_min_shrunk_upset_lift_spin = _double_spin(
            0.0, 20.0, 0.05, 2, "optimizer_save_min_shrunk_upset_lift", " pp",
            "Required minimum shrunk upset-lift over underdog baseline.",
        )
        self._sg_min_roi_lift_spin = _double_spin(
            0.0, 20.0, 0.05, 2, "optimizer_save_min_roi_lift", " pp",
            "Required minimum ROI lift over baseline.",
        )
        self._sg_roi_lb95_slack_spin = _double_spin(
            0.0, 20.0, 0.05, 2, "optimizer_save_roi_lb95_slack", " pp",
            "How far the ROI lower bound may trail baseline.",
        )
        self._sg_long_dog_min_payout_spin = _double_spin(
            1.01, 10.0, 0.05, 2, "optimizer_save_long_dog_min_payout", "x",
            "Minimum underdog payout multiplier for long-dog one-possession checks.",
        )
        self._sg_long_dog_onepos_margin_spin = _double_spin(
            0.0, 20.0, 0.5, 1, "optimizer_save_long_dog_onepos_margin", " pts",
            "Maximum loss margin (points) for long-dog one-possession success.",
        )
        self._sg_long_dog_min_count_spin = _int_spin(
            0, 5000, "optimizer_save_long_dog_min_count",
            "Minimum long-dog sample size for tiebreak. 0 = auto by validation sample.",
            "Auto",
        )
        self._sg_long_dog_prior_weight_spin = _double_spin(
            0.0, 500.0, 1.0, 1, "optimizer_save_long_dog_prior_weight",
            tooltip="Shrinkage strength for long-dog one-possession rate.",
        )
        self._sg_min_long_dog_onepos_lift_spin = _double_spin(
            0.0, 20.0, 0.05, 2, "optimizer_save_min_long_dog_onepos_lift", " pp",
            "Required shrunk long-dog one-possession lift for tiebreak saves.",
        )
        self._sg_long_dog_tiebreak_loss_window_spin = _double_spin(
            0.0, 1.0, 0.001, 3, "optimizer_save_long_dog_tiebreak_loss_window",
            tooltip="Maximum allowed loss regression when using long-dog tiebreak.",
        )

        pairs = [
            ("Loss Margin", self._sg_loss_margin_spin, "Compression Floor", self._sg_compression_floor_spin),
            ("Max Winner Drop", self._sg_max_winner_drop_spin, "Favorites Slack", self._sg_favorites_slack_spin),
            ("Min ML Payout", self._sg_min_ml_payout_spin, "Min ML Bets", self._sg_min_ml_bets_spin),
            ("Min Upset Count", self._sg_min_upset_count_spin, "Min Upset Rate", self._sg_min_upset_rate_spin),
            ("Max Upset Rate", self._sg_max_upset_rate_spin, "Upset Prior Weight", self._sg_upset_prior_weight_spin),
            ("Min Shrunk Upset Lift", self._sg_min_shrunk_upset_lift_spin, "Min ROI Lift", self._sg_min_roi_lift_spin),
        ]

        row_idx = 4
        for left_name, left_widget, right_name, right_widget in pairs:
            left_lbl = QLabel(left_name)
            left_lbl.setStyleSheet(label_style)
            right_lbl = QLabel(right_name)
            right_lbl.setStyleSheet(label_style)
            gl.addWidget(left_lbl, row_idx, 0)
            gl.addWidget(left_widget, row_idx, 1, alignment=Qt.AlignmentFlag.AlignLeft)
            gl.addWidget(right_lbl, row_idx, 2)
            gl.addWidget(right_widget, row_idx, 3, alignment=Qt.AlignmentFlag.AlignLeft)
            row_idx += 1

        roi_slack_lbl = QLabel("ROI LB95 Slack")
        roi_slack_lbl.setStyleSheet(label_style)
        gl.addWidget(roi_slack_lbl, row_idx, 0)
        gl.addWidget(
            self._sg_roi_lb95_slack_spin,
            row_idx,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        min_weight_delta_lbl = QLabel("Min Weight Delta")
        min_weight_delta_lbl.setStyleSheet(label_style)
        gl.addWidget(min_weight_delta_lbl, row_idx, 2)
        gl.addWidget(
            self._sg_min_weight_delta_spin,
            row_idx,
            3,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        row_idx += 1

        long_dog_pairs = [
            (
                "Long-Dog Min Payout",
                self._sg_long_dog_min_payout_spin,
                "Long-Dog OnePos Margin",
                self._sg_long_dog_onepos_margin_spin,
            ),
            (
                "Long-Dog Min Count",
                self._sg_long_dog_min_count_spin,
                "Long-Dog Prior Weight",
                self._sg_long_dog_prior_weight_spin,
            ),
            (
                "Min Long-Dog 1P Lift",
                self._sg_min_long_dog_onepos_lift_spin,
                "Tiebreak Loss Window",
                self._sg_long_dog_tiebreak_loss_window_spin,
            ),
        ]
        for left_name, left_widget, right_name, right_widget in long_dog_pairs:
            left_lbl = QLabel(left_name)
            left_lbl.setStyleSheet(label_style)
            right_lbl = QLabel(right_name)
            right_lbl.setStyleSheet(label_style)
            gl.addWidget(left_lbl, row_idx, 0)
            gl.addWidget(left_widget, row_idx, 1, alignment=Qt.AlignmentFlag.AlignLeft)
            gl.addWidget(right_lbl, row_idx, 2)
            gl.addWidget(right_widget, row_idx, 3, alignment=Qt.AlignmentFlag.AlignLeft)
            row_idx += 1

        gl.setColumnStretch(0, 1)
        gl.setColumnStretch(1, 0)
        gl.setColumnStretch(2, 1)
        gl.setColumnStretch(3, 0)

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    def _build_sync_performance(self, parent_layout: QVBoxLayout):
        """Build sync freshness and worker threads controls."""
        group = QGroupBox("Sync & Performance")
        gl = QGridLayout(group)
        gl.setSpacing(12)
        gl.setContentsMargins(16, 20, 16, 16)

        # Sync freshness hours
        fresh_lbl = QLabel("Sync Freshness (hours)")
        fresh_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        gl.addWidget(fresh_lbl, 0, 0)

        fresh_desc = QLabel(
            "Data older than this threshold will be re-synced on pipeline run."
        )
        fresh_desc.setProperty("class", "text-secondary")
        fresh_desc.setWordWrap(True)
        gl.addWidget(fresh_desc, 1, 0)

        self._freshness_spin = QSpinBox()
        self._freshness_spin.setRange(1, 168)  # 1 hour to 7 days
        self._freshness_spin.setSuffix(" hrs")
        self._freshness_spin.setFixedWidth(100)
        self._freshness_spin.valueChanged.connect(self._on_freshness_changed)
        gl.addWidget(self._freshness_spin, 0, 1, 2, 1, Qt.AlignmentFlag.AlignTop)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(255, 255, 255, 0.08);")
        gl.addWidget(sep, 2, 0, 1, 2)

        # Worker threads
        threads_lbl = QLabel("Worker Threads")
        threads_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        gl.addWidget(threads_lbl, 3, 0)

        threads_desc = QLabel(
            "Number of parallel worker threads for data sync and precompute. "
            "Increase for faster pipeline runs on multi-core systems."
        )
        threads_desc.setProperty("class", "text-secondary")
        threads_desc.setWordWrap(True)
        gl.addWidget(threads_desc, 4, 0)

        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(1, 32)
        self._threads_spin.setFixedWidth(100)
        self._threads_spin.valueChanged.connect(self._on_threads_changed)
        gl.addWidget(self._threads_spin, 3, 1, 2, 1, Qt.AlignmentFlag.AlignTop)

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    def _build_theme_section(self, parent_layout: QVBoxLayout):
        """Build theme toggle radio group."""
        group = QGroupBox("Theme")
        gl = QVBoxLayout(group)
        gl.setSpacing(8)

        desc = QLabel(
            "Choose your display theme. Changes take effect on next app launch."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc)

        self._theme_group = QButtonGroup(self)

        row = QHBoxLayout()
        row.setSpacing(16)

        self._theme_dark = QRadioButton("Dark")
        self._theme_dark.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        self._theme_group.addButton(self._theme_dark, 0)
        row.addWidget(self._theme_dark)

        self._theme_light = QRadioButton("Light")
        self._theme_light.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        self._theme_group.addButton(self._theme_light, 1)
        row.addWidget(self._theme_light)

        self._theme_oled = QRadioButton("OLED")
        self._theme_oled.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;"
        )
        self._theme_group.addButton(self._theme_oled, 2)
        row.addWidget(self._theme_oled)

        row.addStretch()
        gl.addLayout(row)

        self._theme_group.idToggled.connect(self._on_theme_changed)

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    def _build_cd_section(self, parent_layout: QVBoxLayout):
        """Build coordinate descent refinement section."""
        group = QGroupBox("Coordinate Descent Refinement")
        gl = QVBoxLayout(group)
        gl.setSpacing(10)

        desc = QLabel(
            "Grid-search refinement of individual parameters after Optuna optimization. "
            "CD fine-tunes each weight by testing many values across wider ranges, "
            "accepting only improvements. Run after the pipeline optimizer for best results."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        mode_row.addWidget(mode_lbl)
        self._cd_mode_group = QButtonGroup(self)
        self._cd_mode_fund = QRadioButton("Fundamentals Only")
        self._cd_mode_fund.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        self._cd_mode_fund.setChecked(True)
        self._cd_mode_group.addButton(self._cd_mode_fund, 0)
        mode_row.addWidget(self._cd_mode_fund)
        self._cd_mode_sharp = QRadioButton("Fundamentals + Sharp")
        self._cd_mode_sharp.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        self._cd_mode_group.addButton(self._cd_mode_sharp, 1)
        mode_row.addWidget(self._cd_mode_sharp)
        mode_row.addStretch()
        gl.addLayout(mode_row)

        # Controls row: steps, max rounds, buttons
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        _spin_style = (
            f"QSpinBox {{ color: {TEXT_PRIMARY}; font-size: 12px; "
            "padding: 4px 6px; min-height: 26px; }}"
        )

        steps_lbl = QLabel("Steps:")
        steps_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        ctrl_row.addWidget(steps_lbl)
        self._cd_steps_spin = QSpinBox()
        self._cd_steps_spin.setRange(50, 2000)
        self._cd_steps_spin.setValue(100)
        self._cd_steps_spin.setSingleStep(50)
        self._cd_steps_spin.setFixedWidth(100)
        self._cd_steps_spin.setStyleSheet(_spin_style)
        ctrl_row.addWidget(self._cd_steps_spin)

        ctrl_row.addSpacing(10)

        rounds_lbl = QLabel("Max Rounds:")
        rounds_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px;")
        ctrl_row.addWidget(rounds_lbl)
        self._cd_rounds_spin = QSpinBox()
        self._cd_rounds_spin.setRange(1, 20)
        self._cd_rounds_spin.setValue(10)
        self._cd_rounds_spin.setFixedWidth(80)
        self._cd_rounds_spin.setStyleSheet(_spin_style)
        ctrl_row.addWidget(self._cd_rounds_spin)

        ctrl_row.addSpacing(10)

        self._cd_run_btn = QPushButton("Run CD")
        self._cd_run_btn.setProperty("class", "success")
        self._cd_run_btn.setFixedHeight(36)
        self._cd_run_btn.setMinimumWidth(100)
        self._cd_run_btn.clicked.connect(self._on_cd_run)
        ctrl_row.addWidget(self._cd_run_btn)

        self._cd_cancel_btn = QPushButton("Cancel")
        self._cd_cancel_btn.setProperty("class", "danger")
        self._cd_cancel_btn.setFixedHeight(36)
        self._cd_cancel_btn.setMinimumWidth(80)
        self._cd_cancel_btn.setVisible(False)
        self._cd_cancel_btn.clicked.connect(self._on_cd_cancel)
        ctrl_row.addWidget(self._cd_cancel_btn)

        ctrl_row.addStretch()
        gl.addLayout(ctrl_row)

        # Progress log
        self._cd_log = QTextEdit()
        self._cd_log.setReadOnly(True)
        self._cd_log.setFixedHeight(200)
        self._cd_log.setStyleSheet(
            "QTextEdit { background: #0a0e14; color: #22c55e; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; "
            "border: 1px solid rgba(255,255,255,0.08); border-radius: 4px; "
            "padding: 8px; }"
        )
        gl.addWidget(self._cd_log)

        # Results label
        self._cd_results_lbl = QLabel("")
        self._cd_results_lbl.setStyleSheet(
            f"color: {CYAN}; font-size: 13px; font-weight: 600;"
        )
        self._cd_results_lbl.setWordWrap(True)
        gl.addWidget(self._cd_results_lbl)

        # Changes table (hidden until CD completes)
        self._cd_changes_table = QTableWidget()
        self._cd_changes_table.setColumnCount(4)
        self._cd_changes_table.setHorizontalHeaderLabels(
            ["Parameter", "Before", "After", "Delta"]
        )
        self._cd_changes_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for col in range(1, 4):
            self._cd_changes_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self._cd_changes_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._cd_changes_table.verticalHeader().setVisible(False)
        self._cd_changes_table.setAlternatingRowColors(True)
        self._cd_changes_table.setMaximumHeight(200)
        self._cd_changes_table.setVisible(False)
        gl.addWidget(self._cd_changes_table)

        # Worker state
        self._cd_worker = None
        self._cd_thread = None

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    def _build_weight_management(self, parent_layout: QVBoxLayout):
        """Build weight management section with reset button."""
        group = QGroupBox("Weight Management")
        group.setProperty("class", "card-panel-danger")
        gl = QVBoxLayout(group)
        gl.setSpacing(12)

        desc = QLabel(
            "Reset all optimized model weights to their defaults. "
            "This action cannot be undone (consider saving a snapshot first). "
            "The pipeline will need to re-optimize after reset."
        )
        desc.setProperty("class", "text-secondary")
        desc.setWordWrap(True)
        gl.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(12)

        self._reset_btn = QPushButton("RESET ALL WEIGHTS")
        self._reset_btn.setProperty("class", "danger")
        self._reset_btn.setFixedHeight(44)
        self._reset_btn.setMinimumWidth(200)
        self._reset_btn.clicked.connect(self._on_reset_weights)
        row.addWidget(self._reset_btn)

        self._reset_status_lbl = QLabel("")
        self._reset_status_lbl.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 12px;"
        )
        row.addWidget(self._reset_status_lbl)

        row.addStretch()
        gl.addLayout(row)

        # Current weights summary
        self._weights_summary_lbl = QLabel("")
        self._weights_summary_lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 12px;"
        )
        self._weights_summary_lbl.setWordWrap(True)
        gl.addWidget(self._weights_summary_lbl)

        apply_card_shadow(group)
        parent_layout.addWidget(group)

    # ---------------------------------------------------------------
    # Load current settings
    # ---------------------------------------------------------------

    @staticmethod
    def _to_bool(value, default: bool = False) -> bool:
        """Parse bool-ish config values safely."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on", "y"):
                return True
            if v in ("0", "false", "no", "off", "n"):
                return False
        return bool(default)

    def _set_roi_controls_enabled(self, enabled: bool):
        """Enable/disable ROI-only gate controls."""
        roi_widgets = (
            self._sg_min_ml_payout_spin,
            self._sg_min_ml_bets_spin,
            self._sg_min_roi_lift_spin,
            self._sg_roi_lb95_slack_spin,
        )
        for widget in roi_widgets:
            widget.setEnabled(bool(enabled))

    def _set_long_dog_controls_enabled(self, enabled: bool):
        """Enable/disable long-dog tiebreak controls."""
        long_dog_widgets = (
            self._sg_long_dog_min_payout_spin,
            self._sg_long_dog_onepos_margin_spin,
            self._sg_long_dog_min_count_spin,
            self._sg_long_dog_prior_weight_spin,
            self._sg_min_long_dog_onepos_lift_spin,
            self._sg_long_dog_tiebreak_loss_window_spin,
        )
        for widget in long_dog_widgets:
            widget.setEnabled(bool(enabled))

    def _load_current_settings(self):
        """Read config values and populate all controls."""
        try:
            from src.config import get
        except ImportError:
            logger.warning("Config module not available")
            return

        # Prediction mode
        mode = get("prediction_mode", "fundamentals")
        if mode == "fundamentals_sharp":
            self._mode_sharp.setChecked(True)
        else:
            self._mode_fund.setChecked(True)

        # Upset bonus
        try:
            upset_max = float(get("upset_bonus_mult_max", 5.0))
        except (TypeError, ValueError):
            upset_max = 5.0
        upset_max = max(0.0, upset_max)
        try:
            upset_val = float(get("upset_bonus_mult", 0.5))
        except (TypeError, ValueError):
            upset_val = 0.5
        upset_val = min(max(0.0, upset_val), upset_max)
        upset_slider_max = max(1, int(round(upset_max * 100.0)))
        self._upset_slider.blockSignals(True)
        self._upset_slider.setRange(0, upset_slider_max)
        self._upset_slider.setValue(int(round(upset_val * 100.0)))
        self._upset_slider.blockSignals(False)
        self._upset_max_lbl.setText(f"{upset_max:.1f}")
        self._upset_value_lbl.setText(f"{upset_val:.2f}")

        # Optimizer save gate
        gate_float_bindings = [
            (self._sg_loss_margin_spin, "optimizer_save_loss_margin", 0.01),
            (self._sg_min_weight_delta_spin, "optimizer_save_min_weight_delta", 0.0001),
            (self._sg_max_winner_drop_spin, "optimizer_save_max_winner_drop", 0.35),
            (self._sg_favorites_slack_spin, "optimizer_save_favorites_slack", 0.25),
            (self._sg_min_ml_payout_spin, "optimizer_min_ml_payout", 1.50),
            (self._sg_compression_floor_spin, "optimizer_save_compression_floor", 0.55),
            (self._sg_min_upset_rate_spin, "optimizer_save_min_upset_rate", 8.0),
            (self._sg_max_upset_rate_spin, "optimizer_save_max_upset_rate", 55.0),
            (self._sg_upset_prior_weight_spin, "optimizer_save_upset_prior_weight", 25.0),
            (self._sg_min_shrunk_upset_lift_spin, "optimizer_save_min_shrunk_upset_lift", 0.40),
            (self._sg_min_roi_lift_spin, "optimizer_save_min_roi_lift", 0.15),
            (self._sg_roi_lb95_slack_spin, "optimizer_save_roi_lb95_slack", 0.35),
            (self._sg_long_dog_min_payout_spin, "optimizer_save_long_dog_min_payout", 3.0),
            (self._sg_long_dog_onepos_margin_spin, "optimizer_save_long_dog_onepos_margin", 3.0),
            (self._sg_long_dog_prior_weight_spin, "optimizer_save_long_dog_prior_weight", 25.0),
            (self._sg_min_long_dog_onepos_lift_spin, "optimizer_save_min_long_dog_onepos_lift", 0.75),
            (
                self._sg_long_dog_tiebreak_loss_window_spin,
                "optimizer_save_long_dog_tiebreak_loss_window",
                0.01,
            ),
        ]
        for widget, key, fallback in gate_float_bindings:
            try:
                value = float(get(key, fallback))
            except (TypeError, ValueError):
                value = float(fallback)
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        gate_int_bindings = [
            (self._sg_min_upset_count_spin, "optimizer_save_min_upset_count", 0),
            (self._sg_min_ml_bets_spin, "optimizer_save_min_ml_bets", 0),
            (self._sg_long_dog_min_count_spin, "optimizer_save_long_dog_min_count", 40),
        ]
        for widget, key, fallback in gate_int_bindings:
            try:
                value = int(get(key, fallback))
            except (TypeError, ValueError):
                value = int(fallback)
            widget.blockSignals(True)
            widget.setValue(max(0, value))
            widget.blockSignals(False)

        roi_gate_enabled = self._to_bool(get("optimizer_save_use_roi_gate", False), False)
        self._sg_use_roi_gate_chk.blockSignals(True)
        self._sg_use_roi_gate_chk.setChecked(roi_gate_enabled)
        self._sg_use_roi_gate_chk.blockSignals(False)
        self._set_roi_controls_enabled(roi_gate_enabled)
        long_dog_gate_enabled = self._to_bool(
            get("optimizer_save_use_long_dog_tiebreak_gate", True),
            True,
        )
        self._sg_use_long_dog_tiebreak_chk.blockSignals(True)
        self._sg_use_long_dog_tiebreak_chk.setChecked(long_dog_gate_enabled)
        self._sg_use_long_dog_tiebreak_chk.blockSignals(False)
        self._set_long_dog_controls_enabled(long_dog_gate_enabled)

        # Sync freshness
        freshness = get("sync_freshness_hours", 4)
        self._freshness_spin.blockSignals(True)
        self._freshness_spin.setValue(freshness)
        self._freshness_spin.blockSignals(False)

        # Worker threads
        threads = get("worker_threads", 2)
        self._threads_spin.blockSignals(True)
        self._threads_spin.setValue(threads)
        self._threads_spin.blockSignals(False)

        # Theme
        theme = get("theme", "dark")
        oled = get("oled_mode", False)
        if oled:
            self._theme_oled.setChecked(True)
        elif theme == "light":
            self._theme_light.setChecked(True)
        else:
            self._theme_dark.setChecked(True)

        # Weight summary
        self._update_weight_summary()

    def _update_weight_summary(self):
        """Show a brief summary of current weight state."""
        try:
            from src.analytics.weight_config import get_weight_config
            from src.database.db import thread_local_db
            w = get_weight_config()
            d = w.to_dict()
            non_default = 0
            defaults = {}
            try:
                from src.analytics.weight_config import WeightConfig
                default_w = WeightConfig()
                defaults = default_w.to_dict()
                for k, v in d.items():
                    if k in defaults and abs(v - defaults[k]) > 0.001:
                        non_default += 1
            except Exception:
                pass
            total = len(d)
            self._weights_summary_lbl.setText(
                f"Current weights: {total} parameters, "
                f"{non_default} modified from defaults"
            )
        except Exception as e:
            self._weights_summary_lbl.setText(f"Unable to read weights: {e}")

    # ---------------------------------------------------------------
    # Event handlers
    # ---------------------------------------------------------------

    def _on_mode_changed(self, button_id: int, checked: bool):
        """Handle prediction mode change."""
        if not checked:
            return
        from src.config import set_value
        if button_id == 0:
            set_value("prediction_mode", "fundamentals")
            self._notify("Prediction mode: Fundamentals Only")
        else:
            set_value("prediction_mode", "fundamentals_sharp")
            self._notify("Prediction mode: Fundamentals + Sharp")

    def _on_upset_slider_changed(self, value: int):
        """Handle upset bonus slider change."""
        from src.config import set_value
        float_val = value / 100.0
        self._upset_value_lbl.setText(f"{float_val:.2f}")
        set_value("upset_bonus_mult", float_val)

    def _on_gate_float_changed(self, key: str, value: float):
        """Persist float-based optimizer save gate settings."""
        from src.config import set_value
        set_value(key, float(value))

    def _on_roi_gate_toggled(self, checked: bool):
        """Toggle ROI hard-gate mode and ROI control availability."""
        self._on_gate_bool_changed("optimizer_save_use_roi_gate", checked)
        self._set_roi_controls_enabled(checked)

    def _on_long_dog_tiebreak_toggled(self, checked: bool):
        """Toggle long-dog tiebreak mode and control availability."""
        self._on_gate_bool_changed("optimizer_save_use_long_dog_tiebreak_gate", checked)
        self._set_long_dog_controls_enabled(checked)

    def _on_gate_bool_changed(self, key: str, checked: bool):
        """Persist bool-based optimizer save gate settings."""
        from src.config import set_value
        set_value(key, bool(checked))

    def _on_gate_int_changed(self, key: str, value: int):
        """Persist int-based optimizer save gate settings."""
        from src.config import set_value
        set_value(key, max(0, int(value)))

    def _on_freshness_changed(self, value: int):
        """Handle sync freshness change."""
        from src.config import set_value
        set_value("sync_freshness_hours", value)

    def _on_threads_changed(self, value: int):
        """Handle worker threads change."""
        from src.config import set_value
        set_value("worker_threads", value)

    def _on_theme_changed(self, button_id: int, checked: bool):
        """Handle theme change."""
        if not checked:
            return
        from src.config import set_value
        if button_id == 0:  # Dark
            set_value("theme", "dark")
            set_value("oled_mode", False)
            self._notify("Theme: Dark (restart to apply)")
        elif button_id == 1:  # Light
            set_value("theme", "light")
            set_value("oled_mode", False)
            self._notify("Theme: Light (restart to apply)")
        elif button_id == 2:  # OLED
            set_value("theme", "dark")
            set_value("oled_mode", True)
            self._notify("Theme: OLED (restart to apply)")

    def _on_reset_weights(self):
        """Reset all weights with confirmation."""
        reply = QMessageBox.warning(
            self,
            "Reset All Weights",
            "Are you sure you want to reset ALL model weights to defaults?\n\n"
            "This will clear:\n"
            "  - Global model weights\n"
            "  - Per-team weight overrides\n"
            "  - Freshness tracking for optimization steps\n\n"
            "Consider saving a snapshot in the Pipeline tab first.\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                from src.analytics.weight_config import clear_all_weights
                clear_all_weights()
                self._reset_status_lbl.setText("Weights reset to defaults")
                self._reset_status_lbl.setStyleSheet(
                    f"color: {GREEN}; font-size: 12px; font-weight: 600;"
                )
                self._update_weight_summary()
                self._notify("All model weights reset to defaults")
            except Exception as e:
                logger.error("Weight reset failed: %s", e)
                self._reset_status_lbl.setText(f"Reset failed: {e}")
                self._reset_status_lbl.setStyleSheet(
                    f"color: {RED}; font-size: 12px; font-weight: 600;"
                )

    def _on_cd_run(self):
        """Start coordinate descent on a background thread."""
        include_sharp = self._cd_mode_group.checkedId() == 1
        steps = self._cd_steps_spin.value()
        max_rounds = self._cd_rounds_spin.value()

        self._cd_log.clear()
        self._cd_results_lbl.setText("")
        self._cd_changes_table.setVisible(False)
        self._cd_run_btn.setEnabled(False)
        self._cd_cancel_btn.setVisible(True)

        mode_str = "Fundamentals + Sharp" if include_sharp else "Fundamentals Only"
        self._cd_log.append(f"Starting CD ({mode_str}, {steps} steps, {max_rounds} rounds)...")

        self._cd_worker = _CDWorker(include_sharp, steps, max_rounds)
        self._cd_thread = QThread()
        self._cd_worker.moveToThread(self._cd_thread)
        self._cd_thread.started.connect(self._cd_worker.run)

        _QC = Qt.ConnectionType.QueuedConnection
        self._cd_worker.progress.connect(self._on_cd_progress, _QC)
        self._cd_worker.finished.connect(self._on_cd_finished, _QC)
        self._cd_worker.error.connect(self._on_cd_error, _QC)
        self._cd_worker.finished.connect(self._cd_thread.quit)
        self._cd_worker.error.connect(self._cd_thread.quit)
        self._cd_thread.finished.connect(self._cd_cleanup)

        self._cd_thread.start()

    def _on_cd_cancel(self):
        """Cancel a running CD."""
        if self._cd_worker:
            self._cd_worker.cancel()
            self._cd_log.append("Cancelling...")

    def _on_cd_progress(self, msg: str):
        """Append progress message to the CD log."""
        self._cd_log.append(msg)
        sb = self._cd_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_cd_finished(self, result: dict):
        """CD completed -- show results."""
        self._cd_run_btn.setEnabled(True)
        self._cd_cancel_btn.setVisible(False)

        improved = result.get("improved", False)
        rounds = result.get("rounds", 0)
        changes = result.get("changes", {})
        initial_wp = result.get("initial_winner_pct", 0)
        final_wp = result.get("final_winner_pct", 0)

        if improved:
            self._cd_results_lbl.setText(
                f"CD improved: {initial_wp:.1f}% -> {final_wp:.1f}% winner "
                f"({rounds} rounds, {len(changes)} params changed)"
            )
            self._cd_results_lbl.setStyleSheet(
                f"color: {GREEN}; font-size: 13px; font-weight: 600;"
            )
        else:
            gate_reason = result.get("save_gate_reason", "validation save gate not met")
            self._cd_results_lbl.setText(
                f"CD did not save ({rounds} rounds, {len(changes)} params tested)"
            )
            self._cd_results_lbl.setStyleSheet(
                f"color: {AMBER}; font-size: 13px; font-weight: 600;"
            )
            self._cd_log.append(f"Save gate reason: {gate_reason}")

        # Populate changes table
        if changes:
            sorted_changes = sorted(
                changes.items(),
                key=lambda x: abs(x[1]["after"] - x[1]["before"]),
                reverse=True,
            )
            self._cd_changes_table.setRowCount(len(sorted_changes))
            for r, (param, vals) in enumerate(sorted_changes):
                before = vals["before"]
                after = vals["after"]
                delta = after - before
                self._cd_changes_table.setItem(
                    r, 0, QTableWidgetItem(param)
                )
                self._cd_changes_table.setItem(
                    r, 1, QTableWidgetItem(f"{before:.4f}")
                )
                self._cd_changes_table.setItem(
                    r, 2, QTableWidgetItem(f"{after:.4f}")
                )
                delta_item = QTableWidgetItem(f"{delta:+.4f}")
                delta_item.setForeground(
                    QColor(GREEN if delta > 0 else RED)
                )
                self._cd_changes_table.setItem(r, 3, delta_item)
            self._cd_changes_table.setVisible(True)

        self._update_weight_summary()
        self._notify(
            f"CD complete: {'improved' if improved else 'no improvement'} "
            f"({rounds} rounds)"
        )

    def _on_cd_error(self, msg: str):
        """CD failed -- show error."""
        self._cd_run_btn.setEnabled(True)
        self._cd_cancel_btn.setVisible(False)
        self._cd_log.append(f"ERROR: {msg}")
        self._cd_results_lbl.setText("CD failed")
        self._cd_results_lbl.setStyleSheet(
            f"color: {RED}; font-size: 13px; font-weight: 600;"
        )

    def _cd_cleanup(self):
        """Clean up CD thread resources."""
        if self._cd_thread:
            self._cd_thread.deleteLater()
        if self._cd_worker:
            self._cd_worker.deleteLater()
        self._cd_thread = None
        self._cd_worker = None

    def _notify(self, msg: str):
        """Show a status bar message."""
        if self.main_window:
            self.main_window.set_status(msg)
