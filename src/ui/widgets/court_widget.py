"""Animated half-court widget — broadcast-quality hardwood court with shot chart,
team-colored markers, ball animation, hover tooltips, and team filter toggle."""

import math
import logging
from typing import Optional, Dict, List

from PySide6.QtCore import (
    Qt, QPointF, QRectF, QTimer, QPropertyAnimation,
    Property, QEasingCurve, Signal, QSize,
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QRadialGradient,
    QPainterPath, QLinearGradient, QPixmap, QFontMetrics,
)
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QToolTip

from src.ui.widgets.nba_colors import get_team_colors

logger = logging.getLogger(__name__)

# ── NBA half-court dimensions (in feet) ──
_COURT_W = 50.0
_COURT_H = 47.0
_HOOP_X = 25.0
_HOOP_Y = 5.25
_THREE_RADIUS = 23.75
_FT_LINE_Y = 19.0
_KEY_WIDTH = 16.0
_RESTRICTED_RADIUS = 4.0

# Corner three junction: y where 23.75ft arc reaches 3ft from sideline
_CORNER_DX = _HOOP_X - 3.0  # 22ft
_CORNER_JUNCTION_Y = _HOOP_Y + math.sqrt(_THREE_RADIUS**2 - _CORNER_DX**2)  # ~14.2ft

# Wood floor colors
_WOOD_BASE = QColor("#c4893b")
_WOOD_LIGHT = QColor("#d49a48")
_WOOD_DARK = QColor("#9e6a28")
_WOOD_GRAIN = QColor(60, 30, 10)


class _CourtCanvas(QWidget):
    """The actual painting surface for the court."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(280, 200)

        # State
        self._home_team_id: Optional[int] = None
        self._away_team_id: Optional[int] = None
        self._shots: List[Dict] = []
        self._filter = "both"  # "away", "home", "both"
        self._last_play_text = ""
        self._last_play_team_id: Optional[int] = None

        # Ball animation state
        self._ball_pos = QPointF(25, 30)
        self._ball_visible = False
        self._ball_target = QPointF(25, 5.25)
        self._basket_glow = 0.0
        self._score_flash_text = ""

        # Tooltip state
        self._hover_shot: Optional[Dict] = None

        # Animation timers
        self._ball_anim = None
        self._glow_timer = QTimer(self)
        self._glow_timer.setSingleShot(True)
        self._glow_timer.timeout.connect(self._end_glow)
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)

    # ── Properties for animation ──
    def _get_ball_x(self): return self._ball_pos.x()
    def _set_ball_x(self, v):
        self._ball_pos.setX(v)
        self.update()
    ball_x = Property(float, _get_ball_x, _set_ball_x)

    def _get_ball_y(self): return self._ball_pos.y()
    def _set_ball_y(self, v):
        self._ball_pos.setY(v)
        self.update()
    ball_y = Property(float, _get_ball_y, _set_ball_y)

    def _get_glow(self): return self._basket_glow
    def _set_glow(self, v):
        self._basket_glow = v
        self.update()
    glow = Property(float, _get_glow, _set_glow)

    # ── Coordinate mapping ──
    def _court_to_px(self, cx: float, cy: float) -> QPointF:
        w, h = self.width(), self.height()
        margin = 10
        pw = w - 2 * margin
        ph = h - 2 * margin
        px = margin + (cx / _COURT_W) * pw
        py = margin + (cy / _COURT_H) * ph
        return QPointF(px, py)

    def _px_per_foot(self):
        w, h = self.width(), self.height()
        margin = 10
        return (w - 2 * margin) / _COURT_W, (h - 2 * margin) / _COURT_H

    def _filtered_shots(self):
        if self._filter == "both" or not self._home_team_id:
            return self._shots
        tid = self._home_team_id if self._filter == "home" else self._away_team_id
        return [s for s in self._shots if s.get("team_id") == tid]

    def _shot_color(self, shot):
        """Return the team color for a shot."""
        if self._home_team_id and shot.get("team_id") == self._home_team_id:
            return QColor(get_team_colors(self._home_team_id)[0])
        if self._away_team_id and shot.get("team_id") == self._away_team_id:
            return QColor(get_team_colors(self._away_team_id)[0])
        return QColor("#3b82f6")

    def _team_paint_color(self):
        """Return the color for the paint area / center circle based on filter."""
        if self._filter == "away" and self._away_team_id:
            return QColor(get_team_colors(self._away_team_id)[0])
        if self._home_team_id:
            return QColor(get_team_colors(self._home_team_id)[0])
        return QColor("#3b82f6")

    # ── Animation ──
    def _animate_score(self, from_x, from_y, text):
        self._ball_pos = QPointF(from_x, from_y)
        self._ball_visible = True
        self._score_flash_text = text[:30]

        if self._ball_anim and self._ball_anim.state() == QPropertyAnimation.State.Running:
            self._ball_anim.stop()

        self._ball_anim_x = QPropertyAnimation(self, b"ball_x")
        self._ball_anim_x.setDuration(600)
        self._ball_anim_x.setStartValue(from_x)
        self._ball_anim_x.setEndValue(_HOOP_X)
        self._ball_anim_x.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._ball_anim_x.start()

        anim_y = QPropertyAnimation(self, b"ball_y")
        anim_y.setDuration(600)
        anim_y.setStartValue(from_y)
        anim_y.setEndValue(_HOOP_Y)
        anim_y.setEasingCurve(QEasingCurve.Type.OutBounce)
        anim_y.finished.connect(self._on_ball_arrived)
        anim_y.start()
        self._ball_anim = anim_y

    def _on_ball_arrived(self):
        self._ball_visible = False
        self._basket_glow = 1.0
        glow_anim = QPropertyAnimation(self, b"glow")
        glow_anim.setDuration(800)
        glow_anim.setStartValue(1.0)
        glow_anim.setEndValue(0.0)
        glow_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        glow_anim.start()
        self._glow_anim = glow_anim
        self._flash_timer.start(2000)
        self.update()

    def _end_glow(self):
        self._basket_glow = 0.0
        self.update()

    def _end_flash(self):
        self._score_flash_text = ""
        self.update()

    # ── Mouse hover for tooltips ──
    def mouseMoveEvent(self, event):
        pos = event.position() if hasattr(event, 'position') else event.localPos()
        hit = None
        best_dist = 12.0
        shots = self._filtered_shots()
        for s in reversed(shots):
            sp = self._court_to_px(s["x"], s["y"])
            dx = pos.x() - sp.x()
            dy = pos.y() - sp.y()
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                hit = s
                best_dist = d
        if hit != self._hover_shot:
            self._hover_shot = hit
            self.update()
        if hit:
            result = "Made" if hit["made"] else "Missed"
            text = hit.get("text", "")
            clock = hit.get("clock", "")
            period = hit.get("period", "")
            tip_parts = [f"<b>{result}</b>"]
            if text:
                tip_parts.append(text)
            if period and clock:
                tip_parts.append(f"<span style='color:#888;'>Q{period} {clock}</span>")
            tip = "<br>".join(tip_parts)
            global_pos = self.mapToGlobal(pos.toPoint())
            QToolTip.showText(global_pos, tip, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_shot = None
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    # ── Painting ──
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = 10
        ppfx, ppfy = self._px_per_foot()
        team_clr = self._team_paint_color()

        # ── Hardwood floor ──
        self._draw_hardwood(p, w, h)

        # ── Team-colored paint area ──
        key_left = (_COURT_W - _KEY_WIDTH) / 2.0
        kl = self._court_to_px(key_left, 0)
        kr_ft = self._court_to_px(key_left + _KEY_WIDTH, _FT_LINE_Y)
        paint_clr = QColor(team_clr)
        paint_clr.setAlpha(45)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(paint_clr)
        p.drawRect(QRectF(kl, kr_ft))

        # Center circle fill (visible half into our court)
        center_pt = self._court_to_px(25, _COURT_H)
        crx = 6.0 * ppfx
        cry = 6.0 * ppfy
        circle_clr = QColor(team_clr)
        circle_clr.setAlpha(38)
        p.setBrush(circle_clr)
        center_rect = QRectF(center_pt.x() - crx, center_pt.y() - cry, crx * 2, cry * 2)
        p.drawChord(center_rect, 0, 180 * 16)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # ── Court lines ──
        self._draw_court_lines(p, ppfx, ppfy)

        # ── Shot markers ──
        self._draw_shots(p)

        # ── Ball animation ──
        if self._ball_visible:
            bp = self._court_to_px(self._ball_pos.x(), self._ball_pos.y())
            ball_r = max(5, int(1.0 * ppfx))
            ball_grad = QRadialGradient(bp, ball_r)
            ball_grad.setColorAt(0, QColor("#f97316"))
            ball_grad.setColorAt(1, QColor("#c2410c"))
            p.setPen(QPen(QColor("#7c2d12"), 1))
            p.setBrush(ball_grad)
            p.drawEllipse(bp, ball_r, ball_r)
            p.setPen(QPen(QColor("#1c1917"), 0.5))
            p.drawLine(QPointF(bp.x() - ball_r * 0.7, bp.y()),
                       QPointF(bp.x() + ball_r * 0.7, bp.y()))

        # ── Score flash / last play text ──
        if self._score_flash_text:
            p.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            p.setPen(QColor("#22c55e"))
            p.drawText(QRectF(10, h - 35, w - 20, 25),
                       Qt.AlignmentFlag.AlignCenter, self._score_flash_text)
        elif self._last_play_text:
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QColor("#94a3b8"))
            p.drawText(QRectF(10, h - 30, w - 20, 20),
                       Qt.AlignmentFlag.AlignCenter, self._last_play_text[:60])

        # ── FG% stats overlay ──
        self._draw_stats_overlay(p, w, h)

        p.end()

    def _draw_hardwood(self, p: QPainter, w: int, h: int):
        """Render hardwood floor with planks, grain, and vignette."""
        # Base warm wood
        wood_grad = QLinearGradient(0, 0, w, h)
        wood_grad.setColorAt(0, QColor("#c4893b"))
        wood_grad.setColorAt(0.3, QColor("#b87a30"))
        wood_grad.setColorAt(0.6, QColor("#d49a48"))
        wood_grad.setColorAt(1.0, QColor("#b07328"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(wood_grad)
        p.drawRect(0, 0, w, h)

        # Plank lines
        p.setPen(QPen(QColor(107, 66, 38, 30), 1))
        plank_w = w / 14.0
        for i in range(15):
            px = i * plank_w + (i % 3) * 2
            p.drawLine(QPointF(px, 0), QPointF(px, h))

        # Wood grain (subtle horizontal streaks)
        grain_pen = QPen()
        grain_pen.setWidthF(1.0)
        for gy in range(0, h, 3):
            alpha = int(6 + 4 * math.sin(gy * 0.7) + 3 * math.sin(gy * 2.3))
            alpha = max(0, min(alpha, 20))
            grain_pen.setColor(QColor(60, 30, 10, alpha))
            p.setPen(grain_pen)
            p.drawLine(QPointF(0, gy), QPointF(w, gy))

        # Vignette
        p.setPen(Qt.PenStyle.NoPen)
        vig = QRadialGradient(QPointF(w / 2, h / 2), w * 0.7)
        vig.setColorAt(0, QColor(0, 0, 0, 0))
        vig.setColorAt(1, QColor(0, 0, 0, 65))
        p.setBrush(vig)
        p.drawRect(0, 0, w, h)

    def _draw_court_lines(self, p: QPainter, ppfx: float, ppfy: float):
        """Draw all NBA court markings with correct geometry."""
        line_pen = QPen(QColor(255, 255, 255, 216), 2.0)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(line_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Court outline
        tl = self._court_to_px(0, 0)
        br = self._court_to_px(_COURT_W, _COURT_H)
        p.drawRect(QRectF(tl, br))

        # Half-court line
        hl = self._court_to_px(0, _COURT_H)
        hr = self._court_to_px(_COURT_W, _COURT_H)
        p.drawLine(hl, hr)

        # Center circle
        center = self._court_to_px(25, _COURT_H)
        crx = 6.0 * ppfx
        cry = 6.0 * ppfy
        cr = QRectF(center.x() - crx, center.y() - cry, crx * 2, cry * 2)
        p.drawArc(cr, 0, 180 * 16)  # top semicircle (into our half)

        # Key / paint outline
        key_left = (_COURT_W - _KEY_WIDTH) / 2.0
        kl_bl = self._court_to_px(key_left, 0)
        kr_ft = self._court_to_px(key_left + _KEY_WIDTH, _FT_LINE_Y)
        p.drawLine(self._court_to_px(key_left, 0), self._court_to_px(key_left, _FT_LINE_Y))
        p.drawLine(self._court_to_px(key_left + _KEY_WIDTH, 0), self._court_to_px(key_left + _KEY_WIDTH, _FT_LINE_Y))
        p.drawLine(self._court_to_px(key_left, _FT_LINE_Y), self._court_to_px(key_left + _KEY_WIDTH, _FT_LINE_Y))

        # Free throw circle — solid half (inside key, toward baseline)
        ft_center = self._court_to_px(_HOOP_X, _FT_LINE_Y)
        ftrx = 6.0 * ppfx
        ftry = 6.0 * ppfy
        ft_rect = QRectF(ft_center.x() - ftrx, ft_center.y() - ftry, ftrx * 2, ftry * 2)
        p.drawArc(ft_rect, 0, 180 * 16)  # top half (toward baseline)

        # Free throw circle — dashed half (outside key, toward half-court)
        dash_pen = QPen(QColor(255, 255, 255, 216), 2.0)
        dash_pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(dash_pen)
        p.drawArc(ft_rect, 180 * 16, 180 * 16)  # bottom half

        # Restore solid pen
        p.setPen(line_pen)

        # Restricted area arc
        hoop_px = self._court_to_px(_HOOP_X, _HOOP_Y)
        ra_rx = _RESTRICTED_RADIUS * ppfx
        ra_ry = _RESTRICTED_RADIUS * ppfy
        ra_rect = QRectF(hoop_px.x() - ra_rx, hoop_px.y() - ra_ry, ra_rx * 2, ra_ry * 2)
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        p.drawArc(ra_rect, 0, 180 * 16)

        # Three-point line
        p.setPen(line_pen)
        three_rx = _THREE_RADIUS * ppfx
        three_ry = _THREE_RADIUS * ppfy
        arc_rect = QRectF(hoop_px.x() - three_rx, hoop_px.y() - three_ry,
                          three_rx * 2, three_ry * 2)

        # Compute arc angles from junction points
        # Left junction: court (3, junctionY), Right junction: court (47, junctionY)
        # In Qt, angles are in 1/16th degree, measured counterclockwise from 3 o'clock
        # We need the arc that goes downward (toward half-court) from left to right junction
        # atan2 uses (dy_px, dx_px) relative to center, with Qt's y-axis pointing down

        left_junc = self._court_to_px(3, _CORNER_JUNCTION_Y)
        right_junc = self._court_to_px(47, _CORNER_JUNCTION_Y)

        # Angles in Qt coordinate system (y down, angles counterclockwise from +x)
        dx_l = left_junc.x() - hoop_px.x()
        dy_l = left_junc.y() - hoop_px.y()  # positive = below hoop
        dx_r = right_junc.x() - hoop_px.x()
        dy_r = right_junc.y() - hoop_px.y()

        # Qt uses counterclockwise angles from 3 o'clock, with y-axis inverted
        # atan2(-dy, dx) gives the Qt-convention angle
        ang_l_deg = math.degrees(math.atan2(-dy_l, dx_l))  # left junction angle
        ang_r_deg = math.degrees(math.atan2(-dy_r, dx_r))  # right junction angle

        # We want the arc from right junction to left junction going counterclockwise
        # (which sweeps downward through the court in Qt's flipped y)
        start_deg = ang_r_deg
        span_deg = ang_l_deg - ang_r_deg
        if span_deg > 0:
            span_deg -= 360  # ensure we go the correct (negative = clockwise in Qt) direction

        p.drawArc(arc_rect, int(start_deg * 16), int(span_deg * 16))

        # Corner three straight lines
        p.drawLine(self._court_to_px(3, 0), self._court_to_px(3, _CORNER_JUNCTION_Y))
        p.drawLine(self._court_to_px(47, 0), self._court_to_px(47, _CORNER_JUNCTION_Y))

        # Lane tick marks
        tick_pen = QPen(QColor(255, 255, 255, 180), 1.5)
        p.setPen(tick_pen)
        tick_len = 0.5 * ppfx
        key_l_px = self._court_to_px(key_left, 0).x()
        key_r_px = self._court_to_px(key_left + _KEY_WIDTH, 0).x()
        for ty in [7, 8, 11, 14]:
            y_px = self._court_to_px(0, ty).y()
            p.drawLine(QPointF(key_l_px - tick_len, y_px), QPointF(key_l_px + tick_len, y_px))
            p.drawLine(QPointF(key_r_px - tick_len, y_px), QPointF(key_r_px + tick_len, y_px))

        # Backboard
        p.setPen(QPen(QColor(255, 255, 255, 180), 3))
        bb_hw = 1.5 * ppfx  # half-width (3ft total)
        bb_y = self._court_to_px(0, 4).y()
        p.drawLine(QPointF(hoop_px.x() - bb_hw, bb_y), QPointF(hoop_px.x() + bb_hw, bb_y))

        # Hoop (rim)
        hoop_r = max(3, int(0.75 * ppfx))

        # Glow effect
        if self._basket_glow > 0:
            glow_color = QColor("#22c55e")
            glow_color.setAlphaF(self._basket_glow * 0.6)
            glow_grad = QRadialGradient(hoop_px, hoop_r * 6)
            glow_grad.setColorAt(0, glow_color)
            glow_grad.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow_grad)
            p.drawEllipse(hoop_px, hoop_r * 6, hoop_r * 6)

        p.setPen(QPen(QColor("#ff6b2b"), 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(hoop_px, hoop_r, hoop_r)

        # Net suggestion
        p.setPen(QPen(QColor(255, 255, 255, 75), 0.8))
        for ni in range(-2, 3):
            p.drawLine(QPointF(hoop_px.x() + ni * 2.5, hoop_px.y() + 4),
                       QPointF(hoop_px.x() + ni * 1.8, hoop_px.y() + 10))

    def _draw_shots(self, p: QPainter):
        """Draw team-colored shot markers with glow for makes."""
        shots = self._filtered_shots()
        for shot in shots:
            sp = self._court_to_px(shot["x"], shot["y"])
            clr = self._shot_color(shot)
            is_hovered = (shot is self._hover_shot)

            if shot["made"]:
                # Glow ring
                glow = QColor(clr)
                glow.setAlpha(60 if not is_hovered else 100)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(glow)
                p.drawEllipse(sp, 10 if is_hovered else 8, 10 if is_hovered else 8)

                # Filled circle
                p.setBrush(clr)
                p.setPen(QPen(QColor(255, 255, 255, 150), 1.2))
                r = 5 if not is_hovered else 6
                p.drawEllipse(sp, r, r)
            else:
                # Missed — X in team color
                miss_clr = QColor(clr)
                miss_clr.setAlpha(120 if not is_hovered else 200)
                r = 4 if not is_hovered else 5
                p.setPen(QPen(miss_clr, 2.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(QPointF(sp.x() - r, sp.y() - r), QPointF(sp.x() + r, sp.y() + r))
                p.drawLine(QPointF(sp.x() + r, sp.y() - r), QPointF(sp.x() - r, sp.y() + r))

    def _draw_stats_overlay(self, p: QPainter, w: int, h: int):
        """Draw FG% stats pills at bottom-left."""
        shots = self._filtered_shots()
        if not shots:
            return

        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        fm = QFontMetrics(p.font())
        pill_h = 18
        pill_y = h - 28 - pill_h
        pill_x = 14

        if self._filter == "both" and self._home_team_id and self._away_team_id:
            for tid, label in [(self._away_team_id, "AWY"), (self._home_team_id, "HME")]:
                team_shots = [s for s in shots if s.get("team_id") == tid]
                if not team_shots:
                    continue
                made = sum(1 for s in team_shots if s["made"])
                pct = round(made / len(team_shots) * 100) if team_shots else 0
                txt = f"{label} {made}/{len(team_shots)} {pct}%"
                tw = fm.horizontalAdvance(txt) + 20
                tc = QColor(get_team_colors(tid)[0])

                # Pill background
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 150))
                p.drawRoundedRect(QRectF(pill_x, pill_y, tw, pill_h), 4, 4)

                # Team color dot
                p.setBrush(tc)
                p.drawEllipse(QPointF(pill_x + 8, pill_y + pill_h / 2), 3, 3)

                # Text
                p.setPen(QColor(226, 232, 240))
                p.drawText(QRectF(pill_x + 15, pill_y, tw - 15, pill_h),
                           Qt.AlignmentFlag.AlignVCenter, txt)
                pill_x += tw + 6
        else:
            made = sum(1 for s in shots if s["made"])
            pct = round(made / len(shots) * 100) if shots else 0
            txt = f"FG {made}/{len(shots)} {pct}%"
            tw = fm.horizontalAdvance(txt) + 16
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(QRectF(pill_x, pill_y, tw, pill_h), 4, 4)
            p.setPen(QColor(226, 232, 240))
            p.drawText(QRectF(pill_x + 8, pill_y, tw - 8, pill_h),
                       Qt.AlignmentFlag.AlignVCenter, txt)


class CourtWidget(QWidget):
    """Animated NBA half-court with shot chart, team filter, and hover tooltips.

    Public API is identical to the previous version:
      - set_teams(home_id, away_id)
      - add_play(play_dict)
      - clear_shots()
    """

    play_clicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Team filter toggle bar ──
        self._toggle_bar = QWidget()
        self._toggle_bar.setFixedHeight(28)
        self._toggle_bar.setStyleSheet(
            "background: rgba(0,0,0,0.25); border-bottom: 1px solid rgba(255,255,255,0.06);"
        )
        bar_layout = QHBoxLayout(self._toggle_bar)
        bar_layout.setContentsMargins(8, 2, 8, 2)
        bar_layout.setSpacing(4)

        self._btn_away = self._make_toggle_btn("Away", "away")
        self._btn_both = self._make_toggle_btn("Both", "both")
        self._btn_home = self._make_toggle_btn("Home", "home")
        self._btn_both.setProperty("active", True)
        self._btn_both.style().polish(self._btn_both)

        # Team color dots
        self._dot_away = QLabel("●")
        self._dot_away.setFixedWidth(12)
        self._dot_away.setStyleSheet("color: #ef4444; font-size: 8px; background: transparent; border: none;")
        self._dot_home = QLabel("●")
        self._dot_home.setFixedWidth(12)
        self._dot_home.setStyleSheet("color: #3b82f6; font-size: 8px; background: transparent; border: none;")

        bar_layout.addStretch()
        bar_layout.addWidget(self._dot_away)
        bar_layout.addWidget(self._btn_away)
        bar_layout.addWidget(self._btn_both)
        bar_layout.addWidget(self._btn_home)
        bar_layout.addWidget(self._dot_home)
        bar_layout.addStretch()

        layout.addWidget(self._toggle_bar)

        # ── Court canvas ──
        self._canvas = _CourtCanvas()
        layout.addWidget(self._canvas, 1)

    def _make_toggle_btn(self, label: str, filter_val: str) -> QPushButton:
        btn = QPushButton(label)
        btn.setFixedHeight(22)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("filter_val", filter_val)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: rgba(255,255,255,0.45);
                font-size: 11px;
                font-weight: 700;
                padding: 0 10px;
                border-radius: 4px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                color: rgba(255,255,255,0.7);
                background: rgba(255,255,255,0.05);
            }
            QPushButton[active="true"] {
                color: rgba(255,255,255,0.9);
                background: rgba(255,255,255,0.1);
            }
        """)
        btn.clicked.connect(lambda: self._on_filter(filter_val))
        return btn

    def _on_filter(self, val: str):
        self._canvas._filter = val
        for btn in [self._btn_away, self._btn_both, self._btn_home]:
            btn.setProperty("active", btn.property("filter_val") == val)
            btn.style().polish(btn)
        self._canvas.update()

    # ── Public API (same as before) ──
    def set_teams(self, home_team_id: int, away_team_id: int):
        self._canvas._home_team_id = home_team_id
        self._canvas._away_team_id = away_team_id
        # Update toggle labels and dot colors
        hc, _ = get_team_colors(home_team_id)
        ac, _ = get_team_colors(away_team_id)
        self._dot_home.setStyleSheet(f"color: {hc}; font-size: 8px; background: transparent; border: none;")
        self._dot_away.setStyleSheet(f"color: {ac}; font-size: 8px; background: transparent; border: none;")
        self._canvas.update()

    def add_play(self, play: Dict):
        canvas = self._canvas
        is_scoring = play.get("scoringPlay", False)
        is_shooting = play.get("shootingPlay", False)
        team_id = play.get("team_id")
        text = play.get("text", "")
        coord = play.get("coordinate", {})

        canvas._last_play_text = text
        canvas._last_play_team_id = team_id

        cx = coord.get("x", 25) if coord else 25
        cy = coord.get("y", 20) if coord else 20
        if cx < -100 or cx > 200 or cy < -100 or cy > 200:
            cx, cy = 25, 20

        shot_x = max(0, min(cx, _COURT_W))
        shot_y = max(0, min(cy, _COURT_H - 1))

        if is_shooting or is_scoring:
            canvas._shots.append({
                "x": shot_x, "y": shot_y,
                "made": is_scoring,
                "team_id": team_id,
                "text": text[:40],
                "clock": play.get("clock", ""),
                "period": play.get("period", ""),
            })
            if len(canvas._shots) > 60:
                canvas._shots = canvas._shots[-60:]

        if is_scoring:
            canvas._animate_score(shot_x, shot_y, text)

        canvas.update()

    def clear_shots(self):
        self._canvas._shots.clear()
        self._canvas._last_play_text = ""
        self._canvas.update()

    def setFixedHeight(self, h: int):
        super().setFixedHeight(h)
