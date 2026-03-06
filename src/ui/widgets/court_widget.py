"""Full-court landscape widget — broadcast-quality hardwood with shot chart,
team-colored markers, ball animation, hover tooltips, team filter toggle,
and home team logo at center court."""

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

# ── NBA full-court dimensions (in feet) — landscape orientation ──
# Court is 94 x 50 ft.  Origin at top-left.
# X axis: 0 (left baseline) → 94 (right baseline).  Half-court at x=47.
# Y axis: 0 (top sideline) → 50 (bottom sideline).
_FULL_W = 94.0
_FULL_H = 50.0
_HALF_X = 47.0

# Left-side basket is at (5.25, 25).  Right-side basket at (88.75, 25).
_LEFT_HOOP_X = 5.25
_LEFT_HOOP_Y = 25.0
_RIGHT_HOOP_X = 88.75
_RIGHT_HOOP_Y = 25.0

# Key / paint dimensions (symmetric for both sides)
_KEY_WIDTH = 16.0   # lane width (feet)
_FT_LINE_DIST = 19.0  # free-throw line distance from baseline
_THREE_RADIUS = 23.75
_RESTRICTED_RADIUS = 4.0
_CENTER_CIRCLE_R = 6.0

# Corner three: sideline is 3ft in from edge → y=3 and y=47
_CORNER_SIDELINE = 3.0  # feet from court edge to corner-three line
_CORNER_JUNCTION_DIST = math.sqrt(_THREE_RADIUS**2 - (25.0 - _CORNER_SIDELINE)**2)

# Wood floor colors
_WOOD_BASE = QColor("#c4893b")
_WOOD_LIGHT = QColor("#d49a48")
_WOOD_DARK = QColor("#9e6a28")


class _CourtCanvas(QWidget):
    """The actual painting surface for the full court (landscape)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(400, 160)

        # State
        self._home_team_id: Optional[int] = None
        self._away_team_id: Optional[int] = None
        self._shots: List[Dict] = []
        self._filter = "both"  # "away", "home", "both"
        self._last_play_text = ""
        self._last_play_team_id: Optional[int] = None

        # Center-court logo
        self._home_logo: Optional[QPixmap] = None

        # Ball animation state
        self._ball_pos = QPointF(47, 25)
        self._ball_visible = False
        self._basket_glow = 0.0
        self._basket_glow_side = "left"  # which hoop is glowing
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

    # ── Coordinate mapping (full court, landscape) ──
    def _court_to_px(self, cx: float, cy: float) -> QPointF:
        """Map court coords (0..94, 0..50) to pixel coords."""
        w, h = self.width(), self.height()
        margin = 8
        pw = w - 2 * margin
        ph = h - 2 * margin
        px = margin + (cx / _FULL_W) * pw
        py = margin + (cy / _FULL_H) * ph
        return QPointF(px, py)

    def _px_per_foot(self):
        w, h = self.width(), self.height()
        margin = 8
        return (w - 2 * margin) / _FULL_W, (h - 2 * margin) / _FULL_H

    def _filtered_shots(self):
        if self._filter == "both" or not self._home_team_id:
            return self._shots
        tid = self._home_team_id if self._filter == "home" else self._away_team_id
        return [s for s in self._shots if s.get("team_id") == tid]

    def _shot_color(self, shot):
        if self._home_team_id and shot.get("team_id") == self._home_team_id:
            return QColor(get_team_colors(self._home_team_id)[0])
        if self._away_team_id and shot.get("team_id") == self._away_team_id:
            return QColor(get_team_colors(self._away_team_id)[0])
        return QColor("#3b82f6")

    def _team_paint_color(self):
        if self._filter == "away" and self._away_team_id:
            return QColor(get_team_colors(self._away_team_id)[0])
        if self._home_team_id:
            return QColor(get_team_colors(self._home_team_id)[0])
        return QColor("#3b82f6")

    # ── Animation ──
    def _animate_score(self, from_x, from_y, text):
        """Animate a ball from shot position to the nearest hoop."""
        self._ball_pos = QPointF(from_x, from_y)
        self._ball_visible = True
        self._score_flash_text = text[:30]

        # Decide which hoop to target (closest)
        if from_x <= _HALF_X:
            target_x, target_y = _LEFT_HOOP_X, _LEFT_HOOP_Y
            self._basket_glow_side = "left"
        else:
            target_x, target_y = _RIGHT_HOOP_X, _RIGHT_HOOP_Y
            self._basket_glow_side = "right"

        if self._ball_anim and self._ball_anim.state() == QPropertyAnimation.State.Running:
            self._ball_anim.stop()

        self._ball_anim_x = QPropertyAnimation(self, b"ball_x")
        self._ball_anim_x.setDuration(600)
        self._ball_anim_x.setStartValue(from_x)
        self._ball_anim_x.setEndValue(target_x)
        self._ball_anim_x.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._ball_anim_x.start()

        anim_y = QPropertyAnimation(self, b"ball_y")
        anim_y.setDuration(600)
        anim_y.setStartValue(from_y)
        anim_y.setEndValue(target_y)
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
        ppfx, ppfy = self._px_per_foot()
        team_clr = self._team_paint_color()

        # ── Hardwood floor ──
        self._draw_hardwood(p, w, h)

        # ── Paint areas (both sides) ──
        self._draw_paint_fill(p, team_clr)

        # ── Center circle fill ──
        center_pt = self._court_to_px(_HALF_X, 25)
        crx = _CENTER_CIRCLE_R * ppfx
        cry = _CENTER_CIRCLE_R * ppfy
        circle_clr = QColor(team_clr)
        circle_clr.setAlpha(38)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(circle_clr)
        p.drawEllipse(center_pt, crx, cry)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # ── Home team logo at center court ──
        if self._home_logo and not self._home_logo.isNull():
            logo_size = min(int(crx * 1.5), int(cry * 1.5), 80)
            scaled = self._home_logo.scaled(
                logo_size, logo_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lx = center_pt.x() - scaled.width() / 2
            ly = center_pt.y() - scaled.height() / 2
            p.setOpacity(0.35)
            p.drawPixmap(QPointF(lx, ly), scaled)
            p.setOpacity(1.0)

        # ── Court lines ──
        self._draw_court_lines(p, ppfx, ppfy)

        # ── Shot markers ──
        self._draw_shots(p)

        # ── Ball animation ──
        if self._ball_visible:
            bp = self._court_to_px(self._ball_pos.x(), self._ball_pos.y())
            ball_r = max(4, int(0.8 * ppfx))
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
            p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            p.setPen(QColor("#22c55e"))
            p.drawText(QRectF(10, h - 28, w - 20, 22),
                       Qt.AlignmentFlag.AlignCenter, self._score_flash_text)
        elif self._last_play_text:
            p.setFont(QFont("Segoe UI", 8))
            p.setPen(QColor("#94a3b8"))
            p.drawText(QRectF(10, h - 24, w - 20, 18),
                       Qt.AlignmentFlag.AlignCenter, self._last_play_text[:80])

        # ── FG% stats overlay ──
        self._draw_stats_overlay(p, w, h)

        p.end()

    def _draw_hardwood(self, p: QPainter, w: int, h: int):
        """Render hardwood floor with planks, grain, and vignette."""
        wood_grad = QLinearGradient(0, 0, w, h)
        wood_grad.setColorAt(0, QColor("#c4893b"))
        wood_grad.setColorAt(0.25, QColor("#b87a30"))
        wood_grad.setColorAt(0.5, QColor("#d49a48"))
        wood_grad.setColorAt(0.75, QColor("#b87a30"))
        wood_grad.setColorAt(1.0, QColor("#b07328"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(wood_grad)
        p.drawRect(0, 0, w, h)

        # Plank lines (vertical for landscape court)
        p.setPen(QPen(QColor(107, 66, 38, 25), 1))
        plank_w = w / 24.0
        for i in range(25):
            px = i * plank_w + (i % 3) * 2
            p.drawLine(QPointF(px, 0), QPointF(px, h))

        # Wood grain (subtle horizontal streaks)
        grain_pen = QPen()
        grain_pen.setWidthF(1.0)
        for gy in range(0, h, 3):
            alpha = int(5 + 3 * math.sin(gy * 0.7) + 2 * math.sin(gy * 2.3))
            alpha = max(0, min(alpha, 16))
            grain_pen.setColor(QColor(60, 30, 10, alpha))
            p.setPen(grain_pen)
            p.drawLine(QPointF(0, gy), QPointF(w, gy))

        # Vignette
        p.setPen(Qt.PenStyle.NoPen)
        vig = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.6)
        vig.setColorAt(0, QColor(0, 0, 0, 0))
        vig.setColorAt(1, QColor(0, 0, 0, 55))
        p.setBrush(vig)
        p.drawRect(0, 0, w, h)

    def _draw_paint_fill(self, p: QPainter, team_clr: QColor):
        """Fill the paint / key areas on both sides."""
        paint_clr = QColor(team_clr)
        paint_clr.setAlpha(40)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(paint_clr)

        # Left paint: x=0 to x=FT_LINE_DIST, y centered
        key_top = (_FULL_H - _KEY_WIDTH) / 2.0
        kl = self._court_to_px(0, key_top)
        kr = self._court_to_px(_FT_LINE_DIST, key_top + _KEY_WIDTH)
        p.drawRect(QRectF(kl, kr))

        # Right paint: x=(94 - FT_LINE_DIST) to x=94
        rl = self._court_to_px(_FULL_W - _FT_LINE_DIST, key_top)
        rr = self._court_to_px(_FULL_W, key_top + _KEY_WIDTH)
        p.drawRect(QRectF(rl, rr))

    def _draw_court_lines(self, p: QPainter, ppfx: float, ppfy: float):
        """Draw all NBA court markings for full court landscape."""
        line_pen = QPen(QColor(255, 255, 255, 210), 1.8)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(line_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Court outline
        tl = self._court_to_px(0, 0)
        br = self._court_to_px(_FULL_W, _FULL_H)
        p.drawRect(QRectF(tl, br))

        # Half-court line
        hlt = self._court_to_px(_HALF_X, 0)
        hlb = self._court_to_px(_HALF_X, _FULL_H)
        p.drawLine(hlt, hlb)

        # Center circle
        center = self._court_to_px(_HALF_X, 25)
        crx = _CENTER_CIRCLE_R * ppfx
        cry = _CENTER_CIRCLE_R * ppfy
        p.drawEllipse(center, crx, cry)

        # Inner center circle (2ft radius)
        inner_rx = 2.0 * ppfx
        inner_ry = 2.0 * ppfy
        p.drawEllipse(center, inner_rx, inner_ry)

        # ── LEFT SIDE ──
        self._draw_half_court_lines(p, ppfx, ppfy, side="left")

        # ── RIGHT SIDE ──
        self._draw_half_court_lines(p, ppfx, ppfy, side="right")

    def _draw_half_court_lines(self, p: QPainter, ppfx: float, ppfy: float,
                                side: str):
        """Draw key, three-point, hoop, etc. for one side."""
        line_pen = QPen(QColor(255, 255, 255, 210), 1.8)
        line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(line_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        if side == "left":
            baseline_x = 0.0
            hoop_x, hoop_y = _LEFT_HOOP_X, _LEFT_HOOP_Y
            ft_line_x = _FT_LINE_DIST
            sign = 1  # +x direction is into court
        else:
            baseline_x = _FULL_W
            hoop_x, hoop_y = _RIGHT_HOOP_X, _RIGHT_HOOP_Y
            ft_line_x = _FULL_W - _FT_LINE_DIST
            sign = -1

        key_top = (_FULL_H - _KEY_WIDTH) / 2.0
        key_bot = key_top + _KEY_WIDTH

        # Key / paint outline (3 lines: top, bottom, free-throw line)
        p.drawLine(self._court_to_px(baseline_x, key_top),
                   self._court_to_px(ft_line_x, key_top))
        p.drawLine(self._court_to_px(baseline_x, key_bot),
                   self._court_to_px(ft_line_x, key_bot))
        p.drawLine(self._court_to_px(ft_line_x, key_top),
                   self._court_to_px(ft_line_x, key_bot))

        # Free throw circle
        ft_center = self._court_to_px(ft_line_x, 25)
        ftrx = _CENTER_CIRCLE_R * ppfx
        ftry = _CENTER_CIRCLE_R * ppfy
        ft_rect = QRectF(ft_center.x() - ftrx, ft_center.y() - ftry, ftrx * 2, ftry * 2)
        # Solid half (toward baseline)
        if side == "left":
            p.drawArc(ft_rect, 90 * 16, 180 * 16)  # left semicircle
            dash_pen = QPen(QColor(255, 255, 255, 210), 1.8)
            dash_pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(dash_pen)
            p.drawArc(ft_rect, -90 * 16, 180 * 16)  # right semicircle (dashed)
        else:
            p.drawArc(ft_rect, -90 * 16, 180 * 16)  # right semicircle
            dash_pen = QPen(QColor(255, 255, 255, 210), 1.8)
            dash_pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(dash_pen)
            p.drawArc(ft_rect, 90 * 16, 180 * 16)  # left semicircle (dashed)

        # Restore solid pen
        p.setPen(line_pen)

        # Restricted area arc
        hoop_px = self._court_to_px(hoop_x, hoop_y)
        ra_rx = _RESTRICTED_RADIUS * ppfx
        ra_ry = _RESTRICTED_RADIUS * ppfy
        ra_rect = QRectF(hoop_px.x() - ra_rx, hoop_px.y() - ra_ry, ra_rx * 2, ra_ry * 2)
        thin_pen = QPen(QColor(255, 255, 255, 170), 1.4)
        p.setPen(thin_pen)
        if side == "left":
            p.drawArc(ra_rect, -90 * 16, 180 * 16)
        else:
            p.drawArc(ra_rect, 90 * 16, 180 * 16)

        # Three-point line
        p.setPen(line_pen)
        three_rx = _THREE_RADIUS * ppfx
        three_ry = _THREE_RADIUS * ppfy
        arc_rect = QRectF(hoop_px.x() - three_rx, hoop_px.y() - three_ry,
                          three_rx * 2, three_ry * 2)

        # Corner junction points: y = CORNER_SIDELINE and y = (50 - CORNER_SIDELINE)
        top_junc = self._court_to_px(
            hoop_x + sign * _CORNER_JUNCTION_DIST, _CORNER_SIDELINE
        )
        bot_junc = self._court_to_px(
            hoop_x + sign * _CORNER_JUNCTION_DIST, _FULL_H - _CORNER_SIDELINE
        )

        # Arc angles
        dx_t = top_junc.x() - hoop_px.x()
        dy_t = top_junc.y() - hoop_px.y()
        dx_b = bot_junc.x() - hoop_px.x()
        dy_b = bot_junc.y() - hoop_px.y()

        ang_t = math.degrees(math.atan2(-dy_t, dx_t))
        ang_b = math.degrees(math.atan2(-dy_b, dx_b))

        if side == "left":
            # Arc sweeps from bottom junction to top junction (clockwise in screen)
            start = ang_b
            span = ang_t - ang_b
            if span < 0:
                span += 360
        else:
            # Arc sweeps from top junction to bottom junction
            start = ang_t
            span = ang_b - ang_t
            if span < 0:
                span += 360

        p.drawArc(arc_rect, int(start * 16), int(span * 16))

        # Corner three straight lines (along sidelines from baseline to junction)
        junc_x = hoop_x + sign * _CORNER_JUNCTION_DIST
        p.drawLine(self._court_to_px(baseline_x, _CORNER_SIDELINE),
                   self._court_to_px(junc_x, _CORNER_SIDELINE))
        p.drawLine(self._court_to_px(baseline_x, _FULL_H - _CORNER_SIDELINE),
                   self._court_to_px(junc_x, _FULL_H - _CORNER_SIDELINE))

        # Lane tick marks
        tick_pen = QPen(QColor(255, 255, 255, 170), 1.3)
        p.setPen(tick_pen)
        tick_len = 0.5 * ppfy
        key_top_px = self._court_to_px(0, key_top).y()
        key_bot_px = self._court_to_px(0, key_bot).y()
        for dist in [7, 8, 11, 14]:
            tx = baseline_x + sign * dist
            x_px = self._court_to_px(tx, 0).x()
            p.drawLine(QPointF(x_px, key_top_px - tick_len),
                       QPointF(x_px, key_top_px + tick_len))
            p.drawLine(QPointF(x_px, key_bot_px - tick_len),
                       QPointF(x_px, key_bot_px + tick_len))

        # Backboard
        p.setPen(QPen(QColor(255, 255, 255, 170), 2.5))
        bb_hw = 1.5 * ppfy  # half-width (3ft total, drawn vertically)
        bb_x = self._court_to_px(hoop_x - sign * 1.25, 0).x()
        p.drawLine(QPointF(bb_x, hoop_px.y() - bb_hw),
                   QPointF(bb_x, hoop_px.y() + bb_hw))

        # Hoop (rim)
        hoop_r = max(3, int(0.6 * ppfx))

        # Glow effect
        if self._basket_glow > 0 and self._basket_glow_side == side:
            glow_color = QColor("#22c55e")
            glow_color.setAlphaF(self._basket_glow * 0.6)
            glow_grad = QRadialGradient(hoop_px, hoop_r * 5)
            glow_grad.setColorAt(0, glow_color)
            glow_grad.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow_grad)
            p.drawEllipse(hoop_px, hoop_r * 5, hoop_r * 5)

        p.setPen(QPen(QColor("#ff6b2b"), 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(hoop_px, hoop_r, hoop_r)

        # Net suggestion
        p.setPen(QPen(QColor(255, 255, 255, 65), 0.7))
        net_dx = sign * 1
        for ni in range(-2, 3):
            p.drawLine(QPointF(hoop_px.x(), hoop_px.y() + ni * 2.2),
                       QPointF(hoop_px.x() + net_dx * 5, hoop_px.y() + ni * 1.6))

    def _draw_shots(self, p: QPainter):
        """Draw team-colored shot markers with glow for makes."""
        shots = self._filtered_shots()
        for shot in shots:
            sp = self._court_to_px(shot["x"], shot["y"])
            clr = self._shot_color(shot)
            is_hovered = (shot is self._hover_shot)

            if shot["made"]:
                glow = QColor(clr)
                glow.setAlpha(55 if not is_hovered else 95)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(glow)
                p.drawEllipse(sp, 8 if is_hovered else 6, 8 if is_hovered else 6)

                p.setBrush(clr)
                p.setPen(QPen(QColor(255, 255, 255, 150), 1.0))
                r = 4.5 if not is_hovered else 5.5
                p.drawEllipse(sp, r, r)
            else:
                miss_clr = QColor(clr)
                miss_clr.setAlpha(110 if not is_hovered else 190)
                r = 3.5 if not is_hovered else 4.5
                p.setPen(QPen(miss_clr, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(QPointF(sp.x() - r, sp.y() - r),
                           QPointF(sp.x() + r, sp.y() + r))
                p.drawLine(QPointF(sp.x() + r, sp.y() - r),
                           QPointF(sp.x() - r, sp.y() + r))

    def _draw_stats_overlay(self, p: QPainter, w: int, h: int):
        """Draw FG% stats pills at bottom-left."""
        shots = self._filtered_shots()
        if not shots:
            return

        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        fm = QFontMetrics(p.font())
        pill_h = 16
        pill_y = h - 24 - pill_h
        pill_x = 12

        if self._filter == "both" and self._home_team_id and self._away_team_id:
            for tid, label in [(self._away_team_id, "AWY"), (self._home_team_id, "HME")]:
                team_shots = [s for s in shots if s.get("team_id") == tid]
                if not team_shots:
                    continue
                made = sum(1 for s in team_shots if s["made"])
                pct = round(made / len(team_shots) * 100) if team_shots else 0
                txt = f"{label} {made}/{len(team_shots)} {pct}%"
                tw = fm.horizontalAdvance(txt) + 18
                tc = QColor(get_team_colors(tid)[0])

                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, 145))
                p.drawRoundedRect(QRectF(pill_x, pill_y, tw, pill_h), 4, 4)

                p.setBrush(tc)
                p.drawEllipse(QPointF(pill_x + 7, pill_y + pill_h / 2), 2.5, 2.5)

                p.setPen(QColor(226, 232, 240))
                p.drawText(QRectF(pill_x + 14, pill_y, tw - 14, pill_h),
                           Qt.AlignmentFlag.AlignVCenter, txt)
                pill_x += tw + 5
        else:
            made = sum(1 for s in shots if s["made"])
            pct = round(made / len(shots) * 100) if shots else 0
            txt = f"FG {made}/{len(shots)} {pct}%"
            tw = fm.horizontalAdvance(txt) + 14
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 145))
            p.drawRoundedRect(QRectF(pill_x, pill_y, tw, pill_h), 4, 4)
            p.setPen(QColor(226, 232, 240))
            p.drawText(QRectF(pill_x + 7, pill_y, tw - 7, pill_h),
                       Qt.AlignmentFlag.AlignVCenter, txt)


class CourtWidget(QWidget):
    """Full-court landscape widget with shot chart, team filter, and home logo.

    Public API (unchanged):
      - set_teams(home_id, away_id)
      - add_play(play_dict)
      - clear_shots()
    """

    play_clicked = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Team filter toggle bar ──
        self._toggle_bar = QWidget()
        self._toggle_bar.setFixedHeight(26)
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
        self._dot_away = QLabel("\u25CF")
        self._dot_away.setFixedWidth(12)
        self._dot_away.setStyleSheet("color: #ef4444; font-size: 8px; background: transparent; border: none;")
        self._dot_home = QLabel("\u25CF")
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
        btn.setFixedHeight(20)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setProperty("filter_val", filter_val)
        btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: rgba(255,255,255,0.45);
                font-size: 10px;
                font-weight: 700;
                padding: 0 8px;
                border-radius: 3px;
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

    # ── Public API ──
    def set_teams(self, home_team_id: int, away_team_id: int):
        self._canvas._home_team_id = home_team_id
        self._canvas._away_team_id = away_team_id
        # Update toggle dot colors
        hc, _ = get_team_colors(home_team_id)
        ac, _ = get_team_colors(away_team_id)
        self._dot_home.setStyleSheet(f"color: {hc}; font-size: 8px; background: transparent; border: none;")
        self._dot_away.setStyleSheet(f"color: {ac}; font-size: 8px; background: transparent; border: none;")
        # Load home team logo for center court
        try:
            from src.ui.widgets.image_utils import get_team_logo
            logo = get_team_logo(home_team_id, 80)
            self._canvas._home_logo = logo
        except Exception:
            self._canvas._home_logo = None
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

        # ESPN coords come as half-court (x: 0..50, y: 0..47).
        # We need to map them to full-court landscape.
        raw_x = coord.get("x", 25) if coord else 25
        raw_y = coord.get("y", 20) if coord else 20

        # Clamp raw values
        if raw_x < -100 or raw_x > 200 or raw_y < -100 or raw_y > 200:
            raw_x, raw_y = 25, 20

        # ESPN half-court: x=0..50 (sideline to sideline), y=0..47 (baseline to half)
        # Map to full court left side: court_x = y (distance from baseline),
        #                               court_y = x (sideline position)
        # Determine which side based on team (home = right side, away = left side)
        court_x = max(0, min(raw_y, 47.0))   # distance from baseline
        court_y = max(0, min(raw_x, 50.0))   # sideline position

        # Place on right side if home team, left side if away
        if team_id and team_id == canvas._home_team_id:
            court_x = _FULL_W - court_x  # right side
        # else: left side (court_x stays as distance from left baseline)

        shot_x = max(0, min(court_x, _FULL_W))
        shot_y = max(0, min(court_y, _FULL_H))

        if is_shooting or is_scoring:
            canvas._shots.append({
                "x": shot_x, "y": shot_y,
                "made": is_scoring,
                "team_id": team_id,
                "text": text[:40],
                "clock": play.get("clock", ""),
                "period": play.get("period", ""),
            })
            if len(canvas._shots) > 80:
                canvas._shots = canvas._shots[-80:]

        if is_scoring:
            canvas._animate_score(shot_x, shot_y, text)

        canvas.update()

    def clear_shots(self):
        self._canvas._shots.clear()
        self._canvas._last_play_text = ""
        self._canvas.update()

    def setFixedHeight(self, h: int):
        super().setFixedHeight(h)
