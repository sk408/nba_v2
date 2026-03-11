"""Matchup tab -- game selector, prediction cards, upset badge,
sharp money panel, fundamentals breakdown, mode toggle.

V2 design: single predict() path, game_score (not spread), upset detection
from Vegas lines, sharp money overlay for display.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import asdict
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QGridLayout, QScrollArea, QProgressBar,
    QGraphicsOpacityEffect, QSizePolicy,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QTimer,
    QPropertyAnimation, QEasingCurve,
)
from PySide6.QtGui import QColor

from src.ui.widgets.image_utils import get_team_logo, make_placeholder_logo
from src.ui.widgets.nba_colors import get_team_colors, ensure_visible
from src.ui.theme import apply_card_shadow

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Workers (QThread background tasks)
# ─────────────────────────────────────────────────────────────

class _PredictWorker(QObject):
    """Background prediction for the selected game."""
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, home_id: int, away_id: int, game_date: str,
                 include_sharp: bool = False):
        super().__init__()
        self.home_id = home_id
        self.away_id = away_id
        self.game_date = game_date
        self.include_sharp = include_sharp

    def run(self):
        try:
            from src.analytics.prediction import predict_matchup
            from src.database.db import thread_local_db
            with thread_local_db():
                result = predict_matchup(
                    self.home_id, self.away_id, self.game_date,
                    include_sharp=self.include_sharp,
                )
            self.finished.emit(asdict(result))
        except Exception as e:
            logger.error("PredictWorker error: %s", e, exc_info=True)
            self.error.emit(str(e))


class _GameScanWorker(QObject):
    """Background scan of all games -- caches predictions for both modes."""
    finished = Signal(object)  # {idx: {"fundamentals": pred_dict, "sharp": pred_dict}}

    def __init__(self, games: list):
        super().__init__()
        # games: [(combo_idx, home_id, away_id, game_date), ...]
        self.games = games

    def run(self):
        from src.analytics.prediction import predict_matchup
        from src.database.db import thread_local_db
        results = {}
        with thread_local_db():
            for idx, home_id, away_id, game_date in self.games:
                try:
                    pred_fund = predict_matchup(
                        home_id, away_id, game_date, include_sharp=False)
                    pred_sharp = predict_matchup(
                        home_id, away_id, game_date, include_sharp=True)
                    results[idx] = {
                        "fundamentals": asdict(pred_fund),
                        "sharp": asdict(pred_sharp),
                        "home_id": home_id,
                        "away_id": away_id,
                    }
                except Exception as e:
                    logger.debug("Game scan skip idx %s: %s", idx, e)
        self.finished.emit(results)


# ─────────────────────────────────────────────────────────────
# PredictionCard -- broadcast-styled value card
# ─────────────────────────────────────────────────────────────

class PredictionCard(QFrame):
    """Broadcast-styled card showing a single prediction value."""

    def __init__(self, title: str, value: str = "\u2014", accent: str = "#00e5ff"):
        super().__init__()
        self.setProperty("class", "broadcast-card")
        self.setMinimumHeight(90)
        self._accent = accent

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(4)

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
# Sharp money collapsible panel
# ─────────────────────────────────────────────────────────────

class _SharpPanel(QFrame):
    """Collapsible sharp money information panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "broadcast-card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(4)

        # Toggle button
        self._toggle_btn = QPushButton("Sharp Money Info")
        self._toggle_btn.setProperty("class", "link")
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.clicked.connect(self._toggle)
        outer.addWidget(self._toggle_btn)

        # Content (hidden by default)
        self._content = QWidget()
        self._content.setVisible(False)
        cl = QGridLayout(self._content)
        cl.setContentsMargins(4, 2, 4, 2)
        cl.setSpacing(6)

        lbl_style = "color: #94a3b8; font-size: 12px; text-transform: uppercase;"
        val_style = "color: #e2e8f0; font-size: 14px; font-weight: 600;"

        cl.addWidget(self._make_label("ML Public", lbl_style), 0, 0)
        self._pub_val = self._make_label("\u2014", val_style)
        cl.addWidget(self._pub_val, 0, 1)

        cl.addWidget(self._make_label("ML Money", lbl_style), 1, 0)
        self._money_val = self._make_label("\u2014", val_style)
        cl.addWidget(self._money_val, 1, 1)

        cl.addWidget(self._make_label("Divergence", lbl_style), 2, 0)
        self._div_val = self._make_label("\u2014", val_style)
        cl.addWidget(self._div_val, 2, 1)

        cl.addWidget(self._make_label("Sharp Agrees", lbl_style), 3, 0)
        self._agrees_val = self._make_label("\u2014", val_style)
        cl.addWidget(self._agrees_val, 3, 1)

        outer.addWidget(self._content)

    def _make_label(self, text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _toggle(self):
        vis = not self._content.isVisible()
        self._content.setVisible(vis)
        arrow = "\u25BC" if vis else "\u25B6"
        self._toggle_btn.setText(f"{arrow} Sharp Money Info")

    def update_data(self, ml_pub: int, ml_mon: int, sharp_agrees):
        """Update sharp money display values."""
        if ml_pub > 0 or ml_mon > 0:
            self._pub_val.setText(f"Home {ml_pub}% / Away {100 - ml_pub}%")
            self._money_val.setText(f"Home {ml_mon}% / Away {100 - ml_mon}%")
            divergence = abs(ml_mon - ml_pub)
            div_color = "#2196F3" if divergence >= 10 else "#e2e8f0"
            self._div_val.setText(f"{divergence}%")
            self._div_val.setStyleSheet(
                f"color: {div_color}; font-size: 14px; font-weight: 600;"
            )
            if sharp_agrees is True:
                self._agrees_val.setText("YES")
                self._agrees_val.setStyleSheet(
                    "color: #22c55e; font-size: 14px; font-weight: 700;")
            elif sharp_agrees is False:
                self._agrees_val.setText("NO")
                self._agrees_val.setStyleSheet(
                    "color: #ef4444; font-size: 14px; font-weight: 700;")
            else:
                self._agrees_val.setText("\u2014")
                self._agrees_val.setStyleSheet(
                    "color: #e2e8f0; font-size: 14px; font-weight: 600;")
        else:
            self._pub_val.setText("No data")
            self._money_val.setText("No data")
            self._div_val.setText("\u2014")
            self._agrees_val.setText("\u2014")

    def clear_data(self):
        for lbl in (self._pub_val, self._money_val, self._div_val, self._agrees_val):
            lbl.setText("\u2014")


# ─────────────────────────────────────────────────────────────
# MatchupView -- main widget
# ─────────────────────────────────────────────────────────────

class MatchupView(QWidget):
    """Full matchup prediction view with game selector, prediction cards,
    upset badge, sharp money panel, and fundamentals breakdown."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker_thread = None
        self._worker = None
        self._scan_thread = None
        self._scan_worker = None
        self._game_lookup = {}      # combo key -> schedule row dict
        self._prediction_cache = {}   # (home_id, away_id): {"fundamentals": dict, "sharp": dict}
        self._include_sharp = False   # current mode toggle state

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(12, 8, 12, 8)

        # Wrap everything in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(10)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Header
        header = QLabel("Matchup Predictions")
        header.setProperty("class", "header")
        layout.addWidget(header)

        # ── Game selector row ──
        sel_frame = QFrame()
        sel_frame.setProperty("class", "broadcast-card")
        sel_inner = QHBoxLayout(sel_frame)
        sel_inner.setContentsMargins(12, 8, 12, 8)

        # Date combo
        date_col = QVBoxLayout()
        date_lbl = QLabel("DATE")
        date_lbl.setProperty("class", "muted")
        date_col.addWidget(date_lbl)
        self._date_combo = QComboBox()
        self._date_combo.setMinimumWidth(130)
        self._date_combo.currentIndexChanged.connect(self._on_date_changed)
        date_col.addWidget(self._date_combo)
        sel_inner.addLayout(date_col)

        sel_inner.addSpacing(8)

        # Game combo
        game_col = QVBoxLayout()
        game_lbl = QLabel("GAME")
        game_lbl.setProperty("class", "muted")
        game_col.addWidget(game_lbl)
        self._game_combo = QComboBox()
        self._game_combo.setMinimumWidth(300)
        self._game_combo.currentIndexChanged.connect(self._on_game_picked)
        game_col.addWidget(self._game_combo)
        sel_inner.addLayout(game_col)

        sel_inner.addSpacing(12)

        # Predict button
        self._predict_btn = QPushButton("PREDICT")
        self._predict_btn.setProperty("class", "success")
        self._predict_btn.setFixedHeight(40)
        self._predict_btn.clicked.connect(self._on_predict)
        sel_inner.addWidget(self._predict_btn)

        sel_inner.addStretch()

        # Mode toggle
        mode_col = QVBoxLayout()
        mode_lbl = QLabel("MODE")
        mode_lbl.setProperty("class", "muted")
        mode_col.addWidget(mode_lbl)
        self._mode_btn = QPushButton("Fundamentals Only")
        self._mode_btn.setProperty("class", "outline")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(False)
        self._mode_btn.clicked.connect(self._on_mode_toggle)
        mode_col.addWidget(self._mode_btn)
        sel_inner.addLayout(mode_col)

        layout.addWidget(sel_frame)

        # ── Team header (logos + names) ──
        team_header = QFrame()
        team_header.setProperty("class", "broadcast-card")
        th_layout = QHBoxLayout(team_header)
        th_layout.setContentsMargins(16, 10, 16, 10)

        # Away side
        self._away_logo = QLabel()
        self._away_logo.setFixedSize(56, 56)
        self._away_logo.setStyleSheet("background: transparent;")
        th_layout.addWidget(self._away_logo)
        away_text = QVBoxLayout()
        away_text.setSpacing(2)
        self._away_name_lbl = QLabel("AWAY")
        self._away_name_lbl.setStyleSheet(
            "color: #e2e8f0; font-size: 18px; font-weight: 700; "
            "font-family: 'Oswald'; text-transform: uppercase; letter-spacing: 1px;"
        )
        away_text.addWidget(self._away_name_lbl)
        self._away_meta_lbl = QLabel("—")
        self._away_meta_lbl.setProperty("class", "text-hint")
        self._away_meta_lbl.setStyleSheet(
            "color: #94a3b8; font-size: 11px; font-family: 'Segoe UI';"
        )
        away_text.addWidget(self._away_meta_lbl)
        th_layout.addLayout(away_text)

        th_layout.addStretch()

        vs = QLabel("VS")
        vs.setProperty("class", "vs-label")
        vs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        th_layout.addWidget(vs)

        th_layout.addStretch()

        home_text = QVBoxLayout()
        home_text.setSpacing(2)
        self._home_name_lbl = QLabel("HOME")
        self._home_name_lbl.setStyleSheet(
            "color: #e2e8f0; font-size: 18px; font-weight: 700; "
            "font-family: 'Oswald'; text-transform: uppercase; letter-spacing: 1px;"
        )
        home_text.addWidget(self._home_name_lbl)
        self._home_meta_lbl = QLabel("—")
        self._home_meta_lbl.setProperty("class", "text-hint")
        self._home_meta_lbl.setStyleSheet(
            "color: #94a3b8; font-size: 11px; font-family: 'Segoe UI';"
        )
        home_text.addWidget(self._home_meta_lbl)
        th_layout.addLayout(home_text)
        self._home_logo = QLabel()
        self._home_logo.setFixedSize(56, 56)
        self._home_logo.setStyleSheet("background: transparent;")
        th_layout.addWidget(self._home_logo)

        layout.addWidget(team_header)

        # ── Availability panel (projected starters out) ──
        out_panel = QFrame()
        out_panel.setProperty("class", "broadcast-card")
        out_layout = QHBoxLayout(out_panel)
        out_layout.setContentsMargins(14, 10, 14, 10)
        out_layout.setSpacing(20)

        away_out_col = QVBoxLayout()
        away_out_col.setSpacing(4)
        away_out_hdr = QLabel("AWAY STARTERS OUT")
        away_out_hdr.setProperty("class", "muted")
        self._away_out_lbl = QLabel("None")
        self._away_out_lbl.setWordWrap(True)
        self._away_out_lbl.setStyleSheet(
            "color: #86efac; font-size: 12px; font-family: 'Segoe UI';"
        )
        away_out_col.addWidget(away_out_hdr)
        away_out_col.addWidget(self._away_out_lbl)
        out_layout.addLayout(away_out_col)

        out_layout.addStretch()

        home_out_col = QVBoxLayout()
        home_out_col.setSpacing(4)
        home_out_hdr = QLabel("HOME STARTERS OUT")
        home_out_hdr.setProperty("class", "muted")
        self._home_out_lbl = QLabel("None")
        self._home_out_lbl.setWordWrap(True)
        self._home_out_lbl.setStyleSheet(
            "color: #86efac; font-size: 12px; font-family: 'Segoe UI';"
        )
        home_out_col.addWidget(home_out_hdr)
        home_out_col.addWidget(self._home_out_lbl)
        out_layout.addLayout(home_out_col)

        layout.addWidget(out_panel)

        # ── Prediction cards ──
        cards_layout = QGridLayout()
        cards_layout.setSpacing(8)
        self._pick_card = PredictionCard("Pick", accent="#3b82f6")
        self._pick_card.setToolTip("Model pick: HOME or AWAY team")
        self._confidence_card = PredictionCard("Confidence", accent="#00e5ff")
        self._confidence_card.setToolTip("Model confidence (0-100%)")
        self._score_card = PredictionCard("Projected Score", accent="#a78bfa")
        self._score_card.setToolTip("Projected final score")
        self._edge_card = PredictionCard("Game Score", accent="#f59e0b")
        self._edge_card.setToolTip("Internal strength edge (positive = home)")

        cards_layout.addWidget(self._pick_card, 0, 0)
        cards_layout.addWidget(self._confidence_card, 0, 1)
        cards_layout.addWidget(self._score_card, 0, 2)
        cards_layout.addWidget(self._edge_card, 0, 3)
        layout.addLayout(cards_layout)

        # Confidence progress bar
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setProperty("class", "confidence")
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_bar.setFormat("%v% Confidence")
        self._confidence_bar.setFixedHeight(22)
        layout.addWidget(self._confidence_bar)

        # ── Upset badge ──
        self._upset_badge = QLabel()
        self._upset_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._upset_badge.setFixedHeight(34)
        self._upset_badge.setVisible(False)
        layout.addWidget(self._upset_badge)

        # ── Sharp money panel (collapsible) ──
        self._sharp_panel = _SharpPanel()
        layout.addWidget(self._sharp_panel)

        # ── Vegas reference ──
        self._vegas_lbl = QLabel()
        self._vegas_lbl.setProperty("class", "text-secondary")
        self._vegas_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._vegas_lbl.setVisible(False)
        layout.addWidget(self._vegas_lbl)

        # ── Breakdown table ──
        breakdown_lbl = QLabel("Adjustment Breakdown")
        breakdown_lbl.setProperty("class", "section-title")
        layout.addWidget(breakdown_lbl)

        self._breakdown_table = QTableWidget()
        self._breakdown_table.setColumnCount(3)
        self._breakdown_table.setHorizontalHeaderLabels(
            ["Factor", "Contribution", "Direction"])
        hdr = self._breakdown_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._breakdown_table.setColumnWidth(1, 120)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._breakdown_table.setColumnWidth(2, 100)
        self._breakdown_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._breakdown_table.setMaximumHeight(320)
        self._breakdown_table.verticalHeader().setVisible(False)
        self._breakdown_table.setAlternatingRowColors(True)
        layout.addWidget(self._breakdown_table)

        layout.addStretch()

        # ── Load data ──
        self._all_schedule = []   # full schedule list from CDN
        self._populate_dates()
        QTimer.singleShot(500, self._load_schedule)

    # ──────────────────────────────────────────────────────────
    # Date / game selectors
    # ──────────────────────────────────────────────────────────

    def _populate_dates(self):
        """Fill date combo with today +14 days."""
        self._date_combo.blockSignals(True)
        self._date_combo.clear()
        today = datetime.now()
        for i in range(15):
            d = today + timedelta(days=i)
            label = d.strftime("%a %m/%d")
            self._date_combo.addItem(label, d.strftime("%Y-%m-%d"))
        self._date_combo.blockSignals(False)

    def _load_schedule(self):
        """Fetch schedule from NBA CDN and populate game combo for today."""
        try:
            from src.data.nba_fetcher import fetch_nba_cdn_schedule
            self._all_schedule = fetch_nba_cdn_schedule()
        except Exception as e:
            logger.warning("Schedule load failed: %s", e)
            self._all_schedule = []

        # Also try ESPN scoreboard as fallback / supplemental
        if not self._all_schedule:
            try:
                from src.data.gamecast import fetch_espn_scoreboard
                espn_games = fetch_espn_scoreboard()
                today_str = datetime.now().strftime("%Y-%m-%d")
                for g in espn_games:
                    self._all_schedule.append({
                        "game_date": today_str,
                        "home_team": g.get("home_team", ""),
                        "away_team": g.get("away_team", ""),
                        "home_team_id": self._resolve_team_id(g.get("home_team", "")),
                        "away_team_id": self._resolve_team_id(g.get("away_team", "")),
                        "game_time": "",
                    })
            except Exception:
                logger.debug("ESPN fallback schedule unavailable", exc_info=True)

        self._filter_games_for_date()
        # Start background scan for today's games
        self._start_game_scan()

    def _resolve_team_id(self, abbr: str) -> int:
        """Look up team_id from abbreviation."""
        if not abbr:
            return 0
        try:
            from src.database import db
            row = db.fetch_one(
                "SELECT team_id FROM teams WHERE abbreviation = ?",
                (abbr.upper(),))
            return row["team_id"] if row else 0
        except Exception:
            logger.debug("team_id resolve failed for %s", abbr, exc_info=True)
            return 0

    def _on_date_changed(self, idx: int):
        """Repopulate game combo for selected date."""
        self._filter_games_for_date()

    def _filter_games_for_date(self):
        """Filter schedule for the selected date and populate game combo."""
        selected_date = self._date_combo.currentData()
        if not selected_date:
            return

        self._game_combo.blockSignals(True)
        self._game_combo.clear()
        self._game_lookup.clear()
        self._game_combo.addItem("\u2014 Select a game \u2014", "")

        game_idx = 0
        for g in self._all_schedule:
            gd = g.get("game_date", "")
            if gd != selected_date:
                continue
            time_str = g.get("game_time", "")
            away = g.get("away_team", "?")
            home = g.get("home_team", "?")
            label = f"{away} @ {home}"
            if time_str:
                label = f"{time_str}  {label}"
            key = f"g{game_idx}"
            game_idx += 1
            self._game_lookup[key] = g
            self._game_combo.addItem(label, key)

        self._game_combo.blockSignals(False)

    def _game_data_for_index(self, idx: int):
        """Resolve combo row to schedule dict without passing dicts via Qt."""
        key = self._game_combo.itemData(idx)
        if not key:
            return None
        return self._game_lookup.get(str(key))

    def _current_game_data(self):
        """Return schedule dict for current combo selection."""
        return self._game_data_for_index(self._game_combo.currentIndex())

    # ──────────────────────────────────────────────────────────
    # Background game scan
    # ──────────────────────────────────────────────────────────

    def _start_game_scan(self):
        """Scan today's games in background to cache predictions."""
        today = datetime.now().strftime("%Y-%m-%d")
        games_to_scan = []
        for i in range(1, self._game_combo.count()):
            data = self._game_data_for_index(i)
            if not data or not isinstance(data, dict):
                continue
            gd = data.get("game_date", "")
            if gd != today:
                continue
            home_id = data.get("home_team_id")
            away_id = data.get("away_team_id")
            if home_id and away_id:
                games_to_scan.append((i, home_id, away_id, gd))

        if not games_to_scan:
            return

        # Don't start if already scanning
        if self._scan_thread is not None:
            try:
                if self._scan_thread.isRunning():
                    return
            except RuntimeError:
                logger.debug("Scan thread check hit deleted QObject", exc_info=True)

        self._scan_worker = _GameScanWorker(games_to_scan)
        self._scan_thread = QThread()
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        _QC = Qt.ConnectionType.QueuedConnection
        self._scan_worker.finished.connect(self._on_scan_done, _QC)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_thread.finished.connect(self._cleanup_scan)
        self._scan_thread.start()

    def _on_scan_done(self, results: dict):
        """Cache all scanned predictions and tag dog picks in dropdown."""
        self._game_combo.blockSignals(True)
        for idx, info in results.items():
            hid = info.get("home_id")
            aid = info.get("away_id")
            if hid and aid:
                self._prediction_cache[(hid, aid)] = {
                    "fundamentals": info.get("fundamentals", {}),
                    "sharp": info.get("sharp", {}),
                }

            # Tag dog picks in dropdown
            fund = info.get("fundamentals", {})
            is_dog = fund.get("is_dog_pick", False)
            if is_dog:
                old_text = self._game_combo.itemText(idx)
                if "DOG" in old_text:
                    continue
                payout = fund.get("dog_payout", 0)
                is_vz = fund.get("is_value_zone", False)
                if is_vz and payout > 0:
                    tag = f"[DOG {payout:.2f}x] "
                else:
                    tag = "[DOG] "
                self._game_combo.setItemText(idx, tag + old_text)

        self._game_combo.blockSignals(False)

    def _cleanup_scan(self):
        if self._scan_thread is not None:
            self._scan_thread.deleteLater()
        if self._scan_worker is not None:
            self._scan_worker.deleteLater()
        self._scan_thread = None
        self._scan_worker = None

    # ──────────────────────────────────────────────────────────
    # Game selection + prediction
    # ──────────────────────────────────────────────────────────

    def _on_game_picked(self, idx: int):
        """When a game is selected from dropdown, auto-predict."""
        data = self._game_data_for_index(idx)
        if not data or not isinstance(data, dict):
            return
        self._update_team_header(data)
        self._on_predict()

    def _on_mode_toggle(self, checked: bool):
        """Toggle between fundamentals-only and fundamentals+sharp."""
        self._include_sharp = checked
        if checked:
            self._mode_btn.setText("Fundamentals + Sharp")
            self._mode_btn.setProperty("class", "primary")
        else:
            self._mode_btn.setText("Fundamentals Only")
            self._mode_btn.setProperty("class", "outline")
        # Re-polish to apply style change
        self._mode_btn.style().unpolish(self._mode_btn)
        self._mode_btn.style().polish(self._mode_btn)

        # Re-display from cache if we have data for the current game
        data = self._current_game_data()
        if data and isinstance(data, dict):
            home_id = data.get("home_team_id")
            away_id = data.get("away_team_id")
            if home_id and away_id:
                cached = self._prediction_cache.get((home_id, away_id))
                if cached:
                    mode_key = "sharp" if self._include_sharp else "fundamentals"
                    pred = cached.get(mode_key)
                    if pred:
                        self._display_result(pred)
                        return
        # If no cache, run fresh prediction
        self._on_predict()

    def _on_predict(self):
        """Run prediction for the selected game."""
        data = self._current_game_data()
        if not data or not isinstance(data, dict):
            return

        home_id = data.get("home_team_id")
        away_id = data.get("away_team_id")
        game_date = data.get("game_date", datetime.now().strftime("%Y-%m-%d"))

        if not home_id or not away_id:
            return

        # Check cache first
        cached = self._prediction_cache.get((home_id, away_id))
        if cached:
            mode_key = "sharp" if self._include_sharp else "fundamentals"
            pred = cached.get(mode_key)
            if pred:
                self._display_result(pred)
                return

        # Busy guard
        try:
            if self._worker_thread is not None and self._worker_thread.isRunning():
                return
        except RuntimeError:
            self._worker_thread = None

        self._predict_btn.setEnabled(False)
        self._predict_btn.setText("Predicting...")

        self._worker = _PredictWorker(
            home_id, away_id, game_date, include_sharp=self._include_sharp)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        _QC = Qt.ConnectionType.QueuedConnection
        self._worker.finished.connect(self._on_result, _QC)
        self._worker.error.connect(self._on_error, _QC)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _cleanup_worker(self):
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker_thread = None
        self._worker = None

    def request_stop(self, timeout_ms: int = 5000) -> bool:
        """Request graceful shutdown of running worker threads."""
        ok = True
        deadline = time.monotonic() + (max(0, int(timeout_ms)) / 1000.0)
        for name in ("_worker_thread", "_scan_thread"):
            thread = getattr(self, name, None)
            if thread is None:
                continue
            try:
                if not thread.isRunning():
                    continue
            except RuntimeError:
                continue
            thread.quit()
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not thread.wait(remaining_ms):
                ok = False
                logger.warning("Matchup %s did not stop in time", name)
        return ok

    def _on_error(self, msg: str):
        self._predict_btn.setEnabled(True)
        self._predict_btn.setText("PREDICT")
        logger.error("Prediction error: %s", msg)
        if self.main_window:
            self.main_window.set_status(f"Prediction error: {msg}")

    def _on_result(self, result: dict):
        """Handle prediction result from worker thread."""
        self._predict_btn.setEnabled(True)
        self._predict_btn.setText("PREDICT")

        # Cache the result
        home_id = result.get("home_team_id")
        away_id = result.get("away_team_id")
        if home_id and away_id:
            mode_key = "sharp" if self._include_sharp else "fundamentals"
            cache_entry = self._prediction_cache.setdefault(
                (home_id, away_id), {})
            cache_entry[mode_key] = result

        self._display_result(result)

    # ──────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────

    def _update_team_header(self, game_data: dict):
        """Update team logos and names in the header."""
        home_abbr = game_data.get("home_team", "HOME")
        away_abbr = game_data.get("away_team", "AWAY")
        home_id = game_data.get("home_team_id")
        away_id = game_data.get("away_team_id")

        # Try to get full team names from DB
        home_name = home_abbr
        away_name = away_abbr
        try:
            from src.analytics.stats_engine import get_team_names
            names = get_team_names()
            home_name = names.get(home_id, home_abbr) if home_id else home_abbr
            away_name = names.get(away_id, away_abbr) if away_id else away_abbr
        except Exception:
            logger.debug("team-name lookup unavailable", exc_info=True)

        self._home_name_lbl.setText(home_name)
        self._away_name_lbl.setText(away_name)

        # Logos
        for tid, logo_lbl, abbr in [
            (home_id, self._home_logo, home_abbr),
            (away_id, self._away_logo, away_abbr),
        ]:
            if tid:
                logo = get_team_logo(tid, 56)
                if logo:
                    logo_lbl.setPixmap(logo)
                else:
                    primary, _ = get_team_colors(tid)
                    logo_lbl.setPixmap(make_placeholder_logo(abbr, 56, primary))
            else:
                logo_lbl.clear()

        # Color accent for team names
        if home_id:
            primary, _ = get_team_colors(home_id)
            primary = ensure_visible(primary)
            self._home_name_lbl.setStyleSheet(
                f"color: {primary}; font-size: 18px; font-weight: 700; "
                f"font-family: 'Oswald'; text-transform: uppercase; letter-spacing: 1px;"
            )
        if away_id:
            primary, _ = get_team_colors(away_id)
            primary = ensure_visible(primary)
            self._away_name_lbl.setStyleSheet(
                f"color: {primary}; font-size: 18px; font-weight: 700; "
                f"font-family: 'Oswald'; text-transform: uppercase; letter-spacing: 1px;"
            )

        self._update_team_context(game_data)

    def _update_team_context(self, game_data: dict):
        """Update record/streak/rest labels and projected starters out."""
        home_id = game_data.get("home_team_id") or 0
        away_id = game_data.get("away_team_id") or 0
        game_date = game_data.get("game_date", datetime.now().strftime("%Y-%m-%d"))

        home_ctx = {}
        away_ctx = {}
        try:
            from src.analytics.team_context import get_team_display_context

            if home_id:
                home_ctx = get_team_display_context(
                    int(home_id), game_date=game_date, include_starters_out=True
                )
            if away_id:
                away_ctx = get_team_display_context(
                    int(away_id), game_date=game_date, include_starters_out=True
                )
        except Exception as e:
            logger.debug("Matchup team context unavailable: %s", e)

        self._home_meta_lbl.setText(self._format_team_meta(home_ctx))
        self._away_meta_lbl.setText(self._format_team_meta(away_ctx))

        home_out = self._format_starters_out(home_ctx)
        away_out = self._format_starters_out(away_ctx)
        self._home_out_lbl.setText(home_out)
        self._away_out_lbl.setText(away_out)

        self._home_out_lbl.setStyleSheet(
            "color: #fca5a5; font-size: 12px; font-family: 'Segoe UI';"
            if "None" not in home_out else
            "color: #86efac; font-size: 12px; font-family: 'Segoe UI';"
        )
        self._away_out_lbl.setStyleSheet(
            "color: #fca5a5; font-size: 12px; font-family: 'Segoe UI';"
            if "None" not in away_out else
            "color: #86efac; font-size: 12px; font-family: 'Segoe UI';"
        )

    def _format_team_meta(self, ctx: dict) -> str:
        if not ctx:
            return "Record — | EVEN | Rest n/a"
        record = ctx.get("record") or "—"
        streak = ctx.get("streak") or "EVEN"
        rest = ctx.get("last_game_short") or "Rest n/a"
        return f"Record {record} | {streak} | {rest}"

    def _format_starters_out(self, ctx: dict) -> str:
        starters_out = (ctx or {}).get("starters_out") or []
        if not starters_out:
            return "None"
        names = []
        for p in starters_out:
            name = str(p.get("name") or "").strip()
            if not name:
                continue
            status = str(p.get("status") or "").strip()
            if status and status.lower() != "out":
                names.append(f"{name} ({status})")
            else:
                names.append(name)
        return ", ".join(names) if names else "None"

    def _display_result(self, result: dict):
        """Populate all UI elements with a prediction result dict."""
        home_team = result.get("home_team", "HOME")
        away_team = result.get("away_team", "AWAY")
        game_score = result.get("game_score", 0)
        pick = result.get("pick", "")
        confidence = result.get("confidence", 0)
        home_pts = result.get("projected_home_pts", 0)
        away_pts = result.get("projected_away_pts", 0)

        # Pick card -- show team abbreviation of the pick
        pick_team = home_team if pick == "HOME" else away_team
        pick_color = "#22c55e" if pick else "#e2e8f0"
        self._pick_card.set_value(pick_team, color=pick_color)

        # Confidence card
        conf_pct = int(round(confidence))
        self._confidence_card.set_value(f"{conf_pct}%")
        self._confidence_bar.setValue(conf_pct)

        # Score card
        self._score_card.set_value(
            f"{away_pts:.0f} - {home_pts:.0f}",
            color="#a78bfa",
        )

        # Edge card
        edge_color = "#22c55e" if game_score > 0 else "#ef4444" if game_score < 0 else "#e2e8f0"
        self._edge_card.set_value(f"{game_score:+.1f}", color=edge_color)

        # Stagger card animations
        for i, card in enumerate([
            self._pick_card, self._confidence_card,
            self._score_card, self._edge_card,
        ]):
            card.animate_in(delay_ms=i * 80)

        # ── Upset badge ──
        is_dog = result.get("is_dog_pick", False)
        is_vz = result.get("is_value_zone", False)
        dog_payout = result.get("dog_payout", 0)
        vegas_sp = result.get("vegas_spread", 0)

        if is_dog and is_vz and dog_payout > 0:
            self._upset_badge.setText(
                f"  UPSET PICK: {pick_team} ({dog_payout:.2f}x)  |  "
                f"Vegas spread: {vegas_sp:+.1f}  |  Model edge: {game_score:+.1f}  "
            )
            self._upset_badge.setProperty("class", "badge-dog")
            self._upset_badge.style().unpolish(self._upset_badge)
            self._upset_badge.style().polish(self._upset_badge)
            self._upset_badge.setVisible(True)
        elif is_dog:
            self._upset_badge.setText(
                f"  UPSET PICK: {pick_team}  |  "
                f"Vegas spread: {vegas_sp:+.1f}  |  Model edge: {game_score:+.1f}  "
                f"(outside value zone)"
            )
            self._upset_badge.setProperty("class", "badge-dog-outside")
            self._upset_badge.style().unpolish(self._upset_badge)
            self._upset_badge.style().polish(self._upset_badge)
            self._upset_badge.setVisible(True)
        else:
            self._upset_badge.setVisible(False)

        # ── Sharp money panel ──
        ml_pub = result.get("ml_sharp_home_public", 0)
        ml_mon = result.get("ml_sharp_home_money", 0)
        sharp_agrees = result.get("sharp_agrees")
        self._sharp_panel.update_data(ml_pub, ml_mon, sharp_agrees)

        # ── Vegas reference ──
        vegas_home_ml = result.get("vegas_home_ml", 0)
        vegas_away_ml = result.get("vegas_away_ml", 0)
        if vegas_sp or vegas_home_ml or vegas_away_ml:
            parts = []
            if vegas_sp:
                parts.append(f"Spread: {vegas_sp:+.1f}")
            if vegas_home_ml:
                parts.append(f"Home ML: {vegas_home_ml:+d}")
            if vegas_away_ml:
                parts.append(f"Away ML: {vegas_away_ml:+d}")
            self._vegas_lbl.setText("Vegas:  " + "  |  ".join(parts))
            self._vegas_lbl.setVisible(True)
        else:
            self._vegas_lbl.setVisible(False)

        # ── Breakdown table ──
        adjustments = result.get("adjustments", {})
        # Pretty names for factors
        _FACTOR_NAMES = {
            "fatigue": "Fatigue",
            "turnover": "Turnover Diff",
            "rebound": "Rebound Diff",
            "rating_matchup": "Rating Matchup",
            "four_factors": "Four Factors",
            "opp_four_factors": "Opp Four Factors",
            "clutch": "Clutch",
            "hustle": "Hustle",
            "rest_advantage": "Rest Advantage",
            "altitude_b2b": "Altitude B2B",
            "sharp_ml": "Sharp ML",
        }

        # Filter to non-zero adjustments and sort by magnitude
        visible = [
            (k, v) for k, v in adjustments.items() if abs(v) > 0.005
        ]
        visible.sort(key=lambda x: abs(x[1]), reverse=True)

        self._breakdown_table.setRowCount(len(visible))
        for row, (factor, val) in enumerate(visible):
            # Factor name
            display_name = _FACTOR_NAMES.get(factor, factor.replace("_", " ").title())
            name_item = QTableWidgetItem(display_name)
            self._breakdown_table.setItem(row, 0, name_item)

            # Contribution value
            val_item = QTableWidgetItem(f"{val:+.2f}")
            if val > 0:
                val_item.setForeground(QColor("#00E676"))  # green
            elif val < 0:
                val_item.setForeground(QColor("#FF5252"))  # red
            self._breakdown_table.setItem(row, 1, val_item)

            # Direction label
            if val > 0:
                direction = "HOME"
                dir_color = QColor("#00E676")
            else:
                direction = "AWAY"
                dir_color = QColor("#FF5252")
            dir_item = QTableWidgetItem(direction)
            dir_item.setForeground(dir_color)
            dir_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            self._breakdown_table.setItem(row, 2, dir_item)

        if self.main_window:
            self.main_window.set_status("Prediction complete")
