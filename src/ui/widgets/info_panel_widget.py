"""Prediction and live-odds info panel for the gamecast sidebar."""

import logging
from typing import Optional, Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QLinearGradient, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout, QProgressBar,
    QGridLayout,
)

logger = logging.getLogger(__name__)


class _InfoCard(QFrame):
    """Styled collapsible card for a prediction or odds section."""

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(parent)
        self.setObjectName("infoCard")
        self._title = title
        self._collapsed = collapsed
        self._rows = []  # list of QHBoxLayout wrappers

        # Styling handled by global theme via #infoCard object name (OLED-aware)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 6, 10, 6)
        self._layout.setSpacing(4)

        # Clickable header row
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        self._arrow = QLabel("\u25BC" if not collapsed else "\u25B6")
        self._arrow.setStyleSheet("color: #475569; font-size: 10px;")
        self._arrow.setFixedWidth(14)
        header_row.addWidget(self._arrow)

        self._header_label = QLabel(title)
        self._header_label.setStyleSheet("""
            color: #00e5ff; font-size: 13px; font-weight: 700;
            letter-spacing: 1px; text-transform: uppercase;
            font-family: 'Oswald', sans-serif;
        """)
        header_row.addWidget(self._header_label)
        header_row.addStretch()

        # Wrap header in a clickable widget
        self._header_widget = QWidget()
        self._header_widget.setLayout(header_row)
        self._header_widget.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._header_widget.mousePressEvent = self._toggle
        self._layout.addWidget(self._header_widget)

        # Content container — holds all rows, toggled on collapse
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(4)
        self._content.setVisible(not collapsed)
        self._layout.addWidget(self._content)

    def _toggle(self, event=None):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        self._arrow.setText("\u25B6" if self._collapsed else "\u25BC")

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        self._arrow.setText("\u25B6" if collapsed else "\u25BC")

    def add_row(self, label: str, value: str, value_color: str = "#e2e8f0",
                bold: bool = False):
        """Add a label-value row to the content area."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 1, 0, 1)
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        row.addWidget(lbl)

        val = QLabel(value)
        weight = "700" if bold else "500"
        val.setStyleSheet(f"color: {value_color}; font-size: 13px; font-weight: {weight};")
        val.setAlignment(Qt.AlignmentFlag.AlignRight)
        row.addWidget(val)
        self._content_layout.addLayout(row)
        return val


class _CollapsibleSection(QWidget):
    """A generic collapsible section with a thin toggle bar."""

    toggled = Signal(bool)  # emits collapsed state

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(parent)
        self._collapsed = collapsed
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle bar
        self._bar = QWidget()
        self._bar.setObjectName("collapsibleBar")
        self._bar.setFixedHeight(20)
        self._bar.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # Styling handled by global theme via #collapsibleBar object name (OLED-aware)
        bar_layout = QHBoxLayout(self._bar)
        bar_layout.setContentsMargins(8, 0, 8, 0)
        bar_layout.setSpacing(4)

        self._arrow = QLabel("\u25BC" if not collapsed else "\u25B6")
        self._arrow.setStyleSheet("color: #475569; font-size: 9px;")
        self._arrow.setFixedWidth(12)
        bar_layout.addWidget(self._arrow)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(
            "color: #64748b; font-size: 10px; font-weight: 600; "
            "letter-spacing: 1px; text-transform: uppercase;"
        )
        bar_layout.addWidget(self._title_label)
        bar_layout.addStretch()

        self._bar.mousePressEvent = self._toggle
        layout.addWidget(self._bar)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content.setVisible(not collapsed)
        layout.addWidget(self._content)

    def add_widget(self, widget, stretch=0):
        self._content_layout.addWidget(widget, stretch)

    def _toggle(self, event=None):
        self._collapsed = not self._collapsed
        self._content.setVisible(not self._collapsed)
        self._arrow.setText("\u25B6" if self._collapsed else "\u25BC")
        self.toggled.emit(self._collapsed)

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        self._arrow.setText("\u25B6" if collapsed else "\u25BC")


class WinProbBar(QWidget):
    """Horizontal bar showing home/away win probability with team colors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self._home_pct = 50.0
        self._home_color = "#3b82f6"
        self._away_color = "#ef4444"
        self._home_abbr = ""
        self._away_abbr = ""

    def set_data(self, home_pct: float, home_color: str, away_color: str,
                 home_abbr: str = "", away_abbr: str = ""):
        self._home_pct = max(1, min(99, home_pct))
        self._home_color = home_color
        self._away_color = away_color
        self._home_abbr = home_abbr
        self._away_abbr = away_abbr
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background Glass Pill
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 20, 30, 200))
        p.drawRoundedRect(0, 0, w, h, h/2.0, h/2.0)

        home_w = w * self._home_pct / 100.0

        # Away bar (right)
        away_grad = QLinearGradient(0, 0, 0, h)
        away_grad.setColorAt(0, QColor(self._away_color).lighter(120))
        away_grad.setColorAt(0.4, QColor(self._away_color))
        away_grad.setColorAt(1, QColor(self._away_color).darker(150))

        p.setBrush(away_grad)
        p.drawRoundedRect(0, 0, w, h, h/2.0, h/2.0)

        # Home bar (left)
        if home_w > 0:
            home_grad = QLinearGradient(0, 0, 0, h)
            home_grad.setColorAt(0, QColor(self._home_color).lighter(120))
            home_grad.setColorAt(0.4, QColor(self._home_color))
            home_grad.setColorAt(1, QColor(self._home_color).darker(150))

            p.setBrush(home_grad)
            if home_w >= w - 1:
                p.drawRoundedRect(0, 0, home_w, h, h/2.0, h/2.0)
            else:
                p.drawRoundedRect(0, 0, int(home_w), h, h/2.0, h/2.0)
                p.drawRect(int(h/2.0), 0, int(home_w - h/2.0), h)

        # Labels
        p.setFont(QFont("Oswald", 10, QFont.Weight.Bold))

        def draw_text(x, y, text):
            p.setPen(QColor(0, 0, 0, 180))
            p.drawText(int(x+1), int(y+1), text)
            p.setPen(QColor("#ffffff"))
            p.drawText(int(x), int(y), text)

        if home_w > 40:
            draw_text(10, h - 6, f"{self._home_abbr} {self._home_pct:.0f}%")
        away_pct = 100 - self._home_pct
        if w - home_w > 40:
            text = f"{away_pct:.0f}% {self._away_abbr}"
            tw = p.fontMetrics().horizontalAdvance(text)
            draw_text(w - tw - 10, h - 6, text)

        p.end()


class InfoPanelWidget(QWidget):
    """Combined prediction + odds panel for gamecast sidebar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid_mode = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Win probability bar
        self.win_prob_bar = WinProbBar()
        layout.addWidget(self.win_prob_bar)

        # Cards container — switches between VBox (1-col) and Grid (2x2)
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(6)

        # Prediction card
        self.pred_card = _InfoCard("Model Prediction")
        self._pred_spread = self.pred_card.add_row("Spread", "\u2014")
        self._pred_total = self.pred_card.add_row("Total", "\u2014")
        self._pred_home = self.pred_card.add_row("Home", "\u2014", bold=True)
        self._pred_away = self.pred_card.add_row("Away", "\u2014", bold=True)
        self._pred_winner = self.pred_card.add_row("Pick", "\u2014", "#22c55e", bold=True)
        self._cards_layout.addWidget(self.pred_card)

        # Odds card
        self.odds_card = _InfoCard("Live Odds")
        self._odds_spread = self.odds_card.add_row("Spread", "\u2014")
        self._odds_ou = self.odds_card.add_row("O/U", "\u2014")
        self._odds_home_ml = self.odds_card.add_row("Home ML", "\u2014")
        self._odds_away_ml = self.odds_card.add_row("Away ML", "\u2014")
        self._odds_provider = self.odds_card.add_row("Source", "\u2014", "#64748b")
        self._cards_layout.addWidget(self.odds_card)

        # Sharp Money card
        self.sharp_card = _InfoCard("Sharp Money")
        self._sharp_spread_pub = self.sharp_card.add_row("Spread Public", "\u2014")
        self._sharp_spread_mon = self.sharp_card.add_row("Spread Money", "\u2014")
        self._sharp_ml_pub = self.sharp_card.add_row("ML Public", "\u2014")
        self._sharp_ml_mon = self.sharp_card.add_row("ML Money", "\u2014")
        self._sharp_signal = self.sharp_card.add_row("Signal", "\u2014", "#eab308", bold=True)
        self._cards_layout.addWidget(self.sharp_card)

        # Game Flow Stats
        self.flow_card = _InfoCard("Game Flow Stats")
        self._flow_drives = self.flow_card.add_row("Drives Scored", "0 - 0")
        self._flow_poss = self.flow_card.add_row("Poss Scored", "0 - 0")
        self._flow_run = self.flow_card.add_row("Current Run", "None", "#eab308", bold=True)
        self._cards_layout.addWidget(self.flow_card)

        layout.addWidget(self._cards_container)
        layout.addStretch()

    def set_grid_mode(self, enabled: bool):
        """Switch cards between single-column (default) and 2x2 grid layout."""
        if enabled == self._grid_mode:
            return
        self._grid_mode = enabled
        cards = [self.pred_card, self.odds_card, self.sharp_card, self.flow_card]

        # Detach cards from current layout (keep parent intact)
        old_layout = self._cards_container.layout()
        while old_layout.count():
            old_layout.takeAt(0)
        # Discard old layout by transferring to a temporary widget
        QWidget().setLayout(old_layout)

        if enabled:
            grid = QGridLayout(self._cards_container)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            grid.addWidget(cards[0], 0, 0)  # Prediction
            grid.addWidget(cards[1], 0, 1)  # Live Odds
            grid.addWidget(cards[2], 1, 0)  # Sharp Money
            grid.addWidget(cards[3], 1, 1)  # Game Flow
            self._cards_layout = grid
        else:
            vbox = QVBoxLayout(self._cards_container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(6)
            for card in cards:
                vbox.addWidget(card)
            self._cards_layout = vbox

    def update_prediction(self, pred: Optional[Dict],
                          home_abbr: str = "", away_abbr: str = ""):
        """Update model prediction display."""
        if not pred:
            self._pred_spread.setText("\u2014")
            self._pred_total.setText("\u2014")
            self._pred_home.setText("\u2014")
            self._pred_away.setText("\u2014")
            self._pred_winner.setText("\u2014")
            return

        spread = pred.get("spread", 0)
        total = pred.get("total", 0)
        home_proj = pred.get("home_projected", 0)
        away_proj = pred.get("away_projected", 0)

        spread_text = f"{home_abbr} {spread:+.1f}" if spread != 0 else "PICK"
        self._pred_spread.setText(spread_text)
        self._pred_total.setText(f"{total:.1f}")
        self._pred_home.setText(f"{home_abbr} {home_proj:.1f}")
        self._pred_away.setText(f"{away_abbr} {away_proj:.1f}")

        winner = home_abbr if spread > 0 else away_abbr
        self._pred_winner.setText(winner)
        self._pred_winner.setStyleSheet(
            f"color: #22c55e; font-size: 13px; font-weight: 700;"
        )

    def update_odds(self, odds: Dict):
        """Update live odds and sharp money display."""
        if not odds:
            return
        self._odds_spread.setText(str(odds.get("spread", "---")))
        ou = odds.get("over_under")
        self._odds_ou.setText(f"{ou}" if ou else "---")
        home_ml = odds.get("home_moneyline")
        away_ml = odds.get("away_moneyline")
        self._odds_home_ml.setText(f"{home_ml:+d}" if home_ml else "---")
        self._odds_away_ml.setText(f"{away_ml:+d}" if away_ml else "---")
        self._odds_provider.setText(odds.get("provider", "---"))

        # Sharp money
        sp_pub = odds.get("spread_home_public")
        sp_mon = odds.get("spread_home_money")
        ml_pub = odds.get("ml_home_public")
        ml_mon = odds.get("ml_home_money")

        if sp_pub is not None and sp_mon is not None:
            self._sharp_spread_pub.setText(f"H {sp_pub}% / A {100 - sp_pub}%")
            self._sharp_spread_mon.setText(f"H {sp_mon}% / A {100 - sp_mon}%")

            # Signal: where money diverges from public
            edge = sp_mon - sp_pub
            if abs(edge) >= 5:
                side = "HOME" if edge > 0 else "AWAY"
                self._sharp_signal.setText(f"Sharps on {side} ({abs(edge)}pp)")
                color = "#22c55e" if abs(edge) >= 10 else "#eab308"
                self._sharp_signal.setStyleSheet(
                    f"color: {color}; font-size: 13px; font-weight: 700;")
            else:
                self._sharp_signal.setText("No strong signal")
                self._sharp_signal.setStyleSheet(
                    "color: #64748b; font-size: 13px; font-weight: 500;")
        else:
            self._sharp_spread_pub.setText("---")
            self._sharp_spread_mon.setText("---")
            self._sharp_signal.setText("No data")

        if ml_pub is not None and ml_mon is not None:
            self._sharp_ml_pub.setText(f"H {ml_pub}% / A {100 - ml_pub}%")
            self._sharp_ml_mon.setText(f"H {ml_mon}% / A {100 - ml_mon}%")
        else:
            self._sharp_ml_pub.setText("---")
            self._sharp_ml_mon.setText("---")

    def update_flow_stats(self, home_drives: str, away_drives: str, current_run: str, home_poss: str, away_poss: str):
        """Update game flow stats display."""
        self._flow_drives.setText(f"{away_drives} - {home_drives}")
        self._flow_poss.setText(f"{away_poss} - {home_poss}")
        if current_run:
            self._flow_run.setText(current_run)
            self._flow_run.setStyleSheet("color: #eab308; font-weight: bold; font-size: 13px;")
        else:
            self._flow_run.setText("None")
            self._flow_run.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 13px;")

    def update_win_probability(self, home_pct: float,
                                home_color: str = "#3b82f6",
                                away_color: str = "#ef4444",
                                home_abbr: str = "", away_abbr: str = ""):
        self.win_prob_bar.set_data(home_pct, home_color, away_color,
                                   home_abbr, away_abbr)
