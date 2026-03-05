"""Settings tab -- application configuration and weight management.

Prediction mode, upset bonus, sync freshness, worker threads, theme,
and weight reset with confirmation dialog.
"""

import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QSlider, QSpinBox,
    QRadioButton, QButtonGroup, QGroupBox, QSizePolicy,
    QMessageBox,
)
from PySide6.QtCore import Qt

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

        # ---- Section 3: Sync & Performance ----
        self._build_sync_performance(layout)

        # ---- Section 4: Theme ----
        self._build_theme_section(layout)

        # ---- Section 5: Weight Management ----
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
        self._upset_slider.setRange(0, 200)  # 0.0 to 2.0 in 0.01 steps
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
        max_lbl = QLabel("2.0")
        max_lbl.setProperty("class", "muted")
        range_row.addWidget(max_lbl)
        gl.addLayout(range_row)

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
        upset_val = get("upset_bonus_mult", 0.5)
        self._upset_slider.blockSignals(True)
        self._upset_slider.setValue(int(upset_val * 100))
        self._upset_slider.blockSignals(False)
        self._upset_value_lbl.setText(f"{upset_val:.2f}")

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

    def _notify(self, msg: str):
        """Show a status bar message."""
        if self.main_window:
            self.main_window.set_status(msg)
