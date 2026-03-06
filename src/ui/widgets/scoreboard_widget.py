"""Visual scoreboard widget with team logos, scores, quarter breakdown, and game clock."""

import logging
from typing import Optional, Dict, List

from PySide6.QtCore import Qt, QPointF, Property, QVariantAnimation, QObject, QEasingCurve, QRectF, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QLinearGradient, QPen, QPixmap, QPolygonF, QPainterPath
from PySide6.QtWidgets import QWidget

from src.ui.widgets.nba_colors import get_team_colors, ensure_visible
from src.ui.widgets.image_utils import get_team_logo, make_placeholder_logo

logger = logging.getLogger(__name__)


class ScoreboardWidget(QWidget):
    """TV-style scoreboard with team logos, scores, quarters, and clock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMaximumHeight(200)

        # State
        self._away_abbr = ""
        self._home_abbr = ""
        self._away_team_id: Optional[int] = None
        self._home_team_id: Optional[int] = None
        self._away_score = 0
        self._home_score = 0
        self._away_quarters: List[int] = []
        self._home_quarters: List[int] = []
        self._status_text = ""
        self._status_state = ""
        self._clock = ""
        self._period = 0
        self._away_timeouts = -1
        self._home_timeouts = -1
        self._away_bonus = False
        self._home_bonus = False
        self._away_fouls = 0
        self._home_fouls = 0

        self._timeout_seconds = 0
        self._timeout_timer = QTimer(self)
        self._timeout_timer.timeout.connect(self._on_timeout_tick)

        self._sub_data: Optional[Dict] = None
        self._sub_timer = QTimer(self)
        self._sub_timer.timeout.connect(self._clear_substitution)

        self._sub_refresh_timer = QTimer(self)
        self._sub_refresh_timer.timeout.connect(self._refresh_sub_pixmaps)

        # Animation state
        self._away_flash_alpha = 0.0
        self._home_flash_alpha = 0.0

        # Pulsing live dot — timer starts on demand when game is live
        self._live_pulse_alpha = 1.0
        self._live_pulse_growing = False
        self._live_pulse_timer = QTimer(self)
        self._live_pulse_timer.setInterval(80)
        self._live_pulse_timer.timeout.connect(self._on_live_pulse)

        # Animations
        self._anim_away = QVariantAnimation(self)
        self._anim_away.setDuration(800)
        self._anim_away.setStartValue(0.8)
        self._anim_away.setEndValue(0.0)
        self._anim_away.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._anim_away.valueChanged.connect(self._set_away_flash_alpha)

        self._anim_home = QVariantAnimation(self)
        self._anim_home.setDuration(800)
        self._anim_home.setStartValue(0.8)
        self._anim_home.setEndValue(0.0)
        self._anim_home.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._anim_home.valueChanged.connect(self._set_home_flash_alpha)

    def start_timeout(self, seconds: int = 75):
        self._timeout_seconds = seconds
        self._timeout_timer.start(1000)
        self.update()

    def stop_timeout(self):
        self._timeout_seconds = 0
        self._timeout_timer.stop()
        self.update()

    def _on_timeout_tick(self):
        if self._timeout_seconds > 0:
            self._timeout_seconds -= 1
            self.update()
            if self._timeout_seconds <= 0:
                self._timeout_timer.stop()

    def _clear_substitution(self):
        self._sub_data = None
        self._sub_timer.stop()
        self._sub_refresh_timer.stop()
        self.update()

    def _refresh_sub_pixmaps(self):
        if not self._sub_data:
            return
            
        updated = False
        from src.ui.views.gamecast_view import _espn_headshot_cache, _espn_headshot_data
        from PySide6.QtCore import QThreadPool, QRunnable
        
        class _FetchRunnable(QRunnable):
            def __init__(self, u):
                super().__init__()
                self.url = u
            def run(self):
                try:
                    import requests
                    if self.url not in _espn_headshot_data:
                        r = requests.get(self.url, timeout=5)
                        if r.status_code == 200 and r.content:
                            _espn_headshot_data[self.url] = r.content
                except Exception:
                    pass

        for key, url_key, name_key in [("in_pixmap", "in_url", "in_name"), ("out_pixmap", "out_url", "out_name")]:
            if not self._sub_data.get(key):
                url = self._sub_data.get(url_key)
                name = self._sub_data.get(name_key)
                pix = None
                
                try:
                    # Check local DB first
                    if name:
                        from src.database import db
                        row = db.fetch_one("SELECT player_id FROM players WHERE name = ?", (name,))
                        local_pid = row["player_id"] if row else None
                        if local_pid:
                            from src.ui.widgets.image_utils import get_player_photo
                            pix = get_player_photo(local_pid, 64, circle=True)
                except Exception:
                    pass
                
                if not pix and url:
                    try:
                        # Check cache
                        pix = _espn_headshot_cache.get((url, 64))
                        if not pix and url in _espn_headshot_data:
                            from PySide6.QtGui import QImage, QPixmap
                            from PySide6.QtCore import Qt
                            from src.ui.widgets.image_utils import _make_circle_pixmap
                            img = QImage()
                            img.loadFromData(_espn_headshot_data[url])
                            if not img.isNull():
                                pix = QPixmap.fromImage(img).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                                pix = _make_circle_pixmap(pix)
                                _espn_headshot_cache[(url, 64)] = pix
                        
                        if not pix:
                            pix = _espn_headshot_cache.get((url, 28))
                            if pix:
                                pix = pix.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        
                        if not pix and url not in _espn_headshot_data:
                            # Not in data, not in cache, queue fetch
                            QThreadPool.globalInstance().start(_FetchRunnable(url))
                    except Exception as e:
                        logger.debug(f"Sub pixmap load failed for {url}: {e}")

                if pix:
                    self._sub_data[key] = pix
                    updated = True
                    
        if updated:
            self.update()

    def _match_name(self, play_name: str, box_name: str) -> bool:
        pn = play_name.lower().replace("'", "").replace(".", "")
        bn = box_name.lower().replace("'", "").replace(".", "")
        if pn in bn or bn in pn:
            return True
        p_parts = pn.split()
        b_parts = bn.split()
        if p_parts and b_parts and p_parts[-1] == b_parts[-1]:
            if len(p_parts) > 1 and len(b_parts) > 1:
                return p_parts[0][0] == b_parts[0][0]
            return True
        return False

    def show_substitution(self, text: str, boxscore: dict):
        import re
        match = re.search(r"(.+?)\s+enters the game for\s+(.+)", text, re.IGNORECASE)
        if not match:
            return
            
        p_in_name = match.group(1).strip()
        p_out_name = match.group(2).replace('.', '').strip()
        
        # Search raw ESPN boxscore for stats
        p_in = None
        p_out = None
        
        box_players = boxscore.get("players", [])
        for tb in box_players:
            stats_blocks = tb.get("statistics", [])
            if not stats_blocks:
                continue
            labels = stats_blocks[0].get("labels", [])
            for ath in stats_blocks[0].get("athletes", []):
                ainfo = ath.get("athlete", {})
                p_name = ainfo.get("displayName", "")
                
                # Extract stats
                stats = ath.get("stats", [])
                smap = dict(zip(labels, stats)) if len(labels) == len(stats) else {}
                
                p_dict = {
                    "name": p_name,
                    "pts": smap.get("PTS", "0"),
                    "reb": smap.get("REB", "0"),
                    "ast": smap.get("AST", "0"),
                    "headshot": ainfo.get("headshot", {}).get("href", "")
                }
                
                if self._match_name(p_in_name, p_name):
                    p_in = p_dict
                    p_in["name"] = p_name # Use full display name for DB match
                if self._match_name(p_out_name, p_name):
                    p_out = p_dict
                    p_out["name"] = p_name # Use full display name for DB match
                    
        in_url = p_in.get("headshot", "") if p_in else ""
        out_url = p_out.get("headshot", "") if p_out else ""
                    
        self._sub_data = {
            "in_name": p_in.get("name", p_in_name) if p_in else p_in_name,
            "out_name": p_out.get("name", p_out_name) if p_out else p_out_name,
            "in_stats": f"{p_in.get('pts', 0)} PTS | {p_in.get('reb', 0)} REB | {p_in.get('ast', 0)} AST" if p_in else "",
            "out_stats": f"{p_out.get('pts', 0)} PTS | {p_out.get('reb', 0)} REB | {p_out.get('ast', 0)} AST" if p_out else "",
            "in_url": in_url,
            "out_url": out_url,
            "in_pixmap": None,
            "out_pixmap": None
        }
        
        # Trigger an initial check
        try:
            self._refresh_sub_pixmaps()
        except Exception as e:
            logger.debug(f"Sub pixmap initial check failed: {e}")
            
        self._sub_timer.start(6000)
        self._sub_refresh_timer.start(500)
        self.update()

    def _on_live_pulse(self):
        """Animate the live indicator dot alpha between 0.3 and 1.0."""
        if self._status_state != "in":
            return
        step = 0.04
        if self._live_pulse_growing:
            self._live_pulse_alpha = min(1.0, self._live_pulse_alpha + step)
            if self._live_pulse_alpha >= 1.0:
                self._live_pulse_growing = False
        else:
            self._live_pulse_alpha = max(0.2, self._live_pulse_alpha - step)
            if self._live_pulse_alpha <= 0.2:
                self._live_pulse_growing = True
        self.update()

    def _set_away_flash_alpha(self, val):
        self._away_flash_alpha = val
        self.update()

    def _set_home_flash_alpha(self, val):
        self._home_flash_alpha = val
        self.update()

    def update_data(self, *, away_abbr: str, home_abbr: str,
                    away_team_id: int = None, home_team_id: int = None,
                    away_score: int = 0, home_score: int = 0,
                    away_quarters: list = None, home_quarters: list = None,
                    status_text: str = "", status_state: str = "",
                    clock: str = "", period: int = 0,
                    away_timeouts: int = -1, home_timeouts: int = -1,
                    away_bonus: bool = False, home_bonus: bool = False,
                    away_fouls: int = 0, home_fouls: int = 0):
        """Set scoreboard data and trigger repaint. Triggers animations on score change."""
        if self._away_score > 0 and away_score > self._away_score:
            self._anim_away.start()
        if self._home_score > 0 and home_score > self._home_score:
            self._anim_home.start()

        self._away_abbr = away_abbr
        self._home_abbr = home_abbr
        self._away_team_id = away_team_id
        self._home_team_id = home_team_id
        self._away_score = away_score
        self._home_score = home_score
        self._away_quarters = away_quarters or []
        self._home_quarters = home_quarters or []
        self._status_text = status_text
        self._status_state = status_state
        # Start/stop pulse timer based on live state
        if status_state == "in" and not self._live_pulse_timer.isActive():
            self._live_pulse_timer.start()
        elif status_state != "in" and self._live_pulse_timer.isActive():
            self._live_pulse_timer.stop()
        self._clock = clock
        self._period = period
        self._away_timeouts = away_timeouts
        self._home_timeouts = home_timeouts
        self._away_bonus = away_bonus
        self._home_bonus = home_bonus
        self._away_fouls = away_fouls
        self._home_fouls = home_fouls
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid_x = w / 2.0

        away_clr, _ = get_team_colors(self._away_team_id) if self._away_team_id else ("#3b82f6", "#1e293b")
        home_clr, _ = get_team_colors(self._home_team_id) if self._home_team_id else ("#ef4444", "#1e293b")
        away_clr = ensure_visible(away_clr)
        home_clr = ensure_visible(home_clr)

        # Background - Dark glass panel
        p.fillRect(0, 0, w, h, QColor(15, 20, 30, 240))
        
        # Angled polygons for modern TV look
        skew = 30.0
        
        # Away Polygon
        away_poly = QPolygonF([
            QPointF(0, 0),
            QPointF(mid_x - skew, 0),
            QPointF(mid_x, h),
            QPointF(0, h)
        ])
        
        # Base Away Gradient
        grad_away = QLinearGradient(0, 0, mid_x, h)
        grad_away.setColorAt(0, QColor(away_clr).darker(300))
        grad_away.setColorAt(1, QColor(away_clr).darker(450))
        
        p.setBrush(grad_away)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(away_poly)
        
        # Blend in flash color if animating
        if self._away_flash_alpha > 0:
            flash_clr = QColor(away_clr)
            flash_clr.setAlphaF(self._away_flash_alpha)
            p.setBrush(flash_clr)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(away_poly)

        # Home Polygon
        home_poly = QPolygonF([
            QPointF(mid_x - skew, 0),
            QPointF(w, 0),
            QPointF(w, h),
            QPointF(mid_x, h)
        ])
        
        # Base Home Gradient
        grad_home = QLinearGradient(mid_x, 0, w, h)
        grad_home.setColorAt(0, QColor(home_clr).darker(300))
        grad_home.setColorAt(1, QColor(home_clr).darker(450))
        
        p.setBrush(grad_home)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(home_poly)

        # Blend in flash color if animating
        if self._home_flash_alpha > 0:
            flash_clr = QColor(home_clr)
            flash_clr.setAlphaF(self._home_flash_alpha)
            p.setBrush(flash_clr)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(home_poly)
            
        # Center separator accent
        p.setPen(QPen(QColor(255, 255, 255, 100), 2))
        p.drawLine(QPointF(mid_x - skew, 0), QPointF(mid_x, h))

        # Bottom glowing borders
        p.fillRect(0, h - 4, int(mid_x - (skew/2)), 4, QColor(away_clr))
        p.fillRect(int(mid_x - (skew/2)), h - 4, int(w - (mid_x - (skew/2))), 4, QColor(home_clr))

        # Typography
        score_font = QFont("Oswald", 48, QFont.Weight.Bold)
        name_font = QFont("Oswald", 18, QFont.Weight.DemiBold)
        small_font = QFont("Oswald", 12)
        tiny_font = QFont("Oswald", 10)

        # ── Away side (left) ──
        logo_size = 72
        away_logo = self._get_logo(self._away_team_id, self._away_abbr, logo_size)
        if away_logo:
            lx = mid_x - 300 - logo_size
            ly = (h - logo_size) // 2 - 10
            p.drawPixmap(int(lx), int(ly), away_logo)

        # Text with subtle drop shadows
        def draw_text_with_shadow(painter, x, y, text, font, color):
            painter.setFont(font)
            # Shadow
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(int(x + 2), int(y + 2), text)
            # Text
            painter.setPen(color)
            painter.drawText(int(x), int(y), text)

        fm_name = p.fontMetrics()

        # Away abbreviation
        p.setFont(name_font)
        away_name_w = fm_name.horizontalAdvance(self._away_abbr)
        away_name_x = (mid_x - 300 - logo_size / 2) - away_name_w / 2
        draw_text_with_shadow(p, away_name_x, h - 30, self._away_abbr, name_font, QColor("#e2e8f0"))

        # Away Bonus, Timeouts, and Fouls
        if self._away_bonus:
            draw_text_with_shadow(p, (mid_x - 300 - logo_size / 2) - 15, h - 10, "BONUS", tiny_font, QColor("#fbbf24"))

        if self._away_timeouts >= 0:
            to_start_x = (mid_x - 300 - logo_size / 2) - (7 * 8) / 2
            for t in range(7):
                to_x = to_start_x + (t * 8)
                to_y = h - 20
                if t < self._away_timeouts:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor("#ffffff"))
                else:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor(255, 255, 255, 50))

        # Away score
        p.setFont(score_font)
        fm_score = p.fontMetrics()
        away_score_text = str(self._away_score)
        sw = fm_score.horizontalAdvance(away_score_text)
        draw_text_with_shadow(p, mid_x - 120 - sw, h // 2 + 20, away_score_text, score_font, QColor("#ffffff"))

        # ── Center VS ──
        draw_text_with_shadow(p, mid_x - 20, h // 2 + 10, "VS", small_font, QColor("#94a3b8"))

        # ── Home side (right) ──
        home_logo = self._get_logo(self._home_team_id, self._home_abbr, logo_size)
        if home_logo:
            lx = mid_x + 300
            ly = (h - logo_size) // 2 - 10
            p.drawPixmap(int(lx), int(ly), home_logo)

        # Home abbreviation
        p.setFont(name_font)
        home_name_w = fm_name.horizontalAdvance(self._home_abbr)
        home_name_x = (mid_x + 300 + logo_size / 2) - home_name_w / 2
        draw_text_with_shadow(p, home_name_x, h - 30, self._home_abbr, name_font, QColor("#e2e8f0"))

        # Home Bonus and Timeouts
        if self._home_bonus:
            draw_text_with_shadow(p, (mid_x + 300 + logo_size / 2) - 15, h - 10, "BONUS", tiny_font, QColor("#fbbf24"))

        if self._home_timeouts >= 0:
            to_start_x = (mid_x + 300 + logo_size / 2) - (7 * 8) / 2
            for t in range(7):
                to_x = to_start_x + (t * 8)
                to_y = h - 20
                if t < self._home_timeouts:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor("#ffffff"))
                else:
                    p.fillRect(int(to_x), int(to_y), 5, 2, QColor(255, 255, 255, 50))

        # Home score
        draw_text_with_shadow(p, mid_x + 100, h // 2 + 20, str(self._home_score), score_font, QColor("#ffffff"))

        # ── Quarter scores (bottom center) ──
        max_quarters = max(len(self._away_quarters), len(self._home_quarters), 4)
        q_start_x = mid_x - (max_quarters * 32) // 2
        q_y = h - 45

        # Background for quarter scores
        bg_rect = QRectF(q_start_x - 10, q_y - 12, max_quarters * 32 + 50, 40)
        p.fillRect(bg_rect, QColor(0, 0, 0, 150))

        # Fouls next to quarters
        draw_text_with_shadow(p, q_start_x - 55, q_y + 14, f"FOULS: {self._away_fouls}", tiny_font, QColor("#94a3b8"))
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 45, q_y + 14, f"FOULS: {self._home_fouls}", tiny_font, QColor("#94a3b8"))

        # Quarter headers
        for qi in range(max_quarters):
            qx = q_start_x + qi * 32
            label = f"Q{qi + 1}" if qi < 4 else f"OT{qi - 3}"
            draw_text_with_shadow(p, qx, q_y, label, tiny_font, QColor("#64748b"))

        # Total header
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 10, q_y, "T", tiny_font, QColor("#64748b"))

        # Away quarter scores
        for qi in range(len(self._away_quarters)):
            qx = q_start_x + qi * 32
            draw_text_with_shadow(p, qx, q_y + 14, str(self._away_quarters[qi]), tiny_font, QColor("#94a3b8"))
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 10, q_y + 14, str(self._away_score), tiny_font, QColor(away_clr))

        # Home quarter scores
        for qi in range(len(self._home_quarters)):
            qx = q_start_x + qi * 32
            draw_text_with_shadow(p, qx, q_y + 26, str(self._home_quarters[qi]), tiny_font, QColor("#94a3b8"))
        draw_text_with_shadow(p, q_start_x + max_quarters * 32 + 10, q_y + 26, str(self._home_score), tiny_font, QColor(home_clr))

        # ── Status / Clock (top center) ──
        # Top banner for time
        p.fillRect(int(mid_x - 80), 0, 160, 32, QColor(10, 15, 25, 220))
        p.setPen(QColor(away_clr))
        p.drawLine(int(mid_x - 80), 32, int(mid_x), 32)
        p.setPen(QColor(home_clr))
        p.drawLine(int(mid_x), 32, int(mid_x + 80), 32)

        if self._status_state == "in":
            # Pulsing live indicator dot with glow
            pulse_a = self._live_pulse_alpha
            # Outer glow ring
            glow_clr = QColor(239, 68, 68, int(80 * pulse_a))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow_clr)
            p.drawEllipse(int(mid_x - 68), 9, 14, 14)
            # Inner dot
            dot_clr = QColor(239, 68, 68, int(255 * pulse_a))
            p.setBrush(dot_clr)
            p.drawEllipse(int(mid_x - 65), 12, 8, 8)
            period_text = f"Q{self._period}" if self._period <= 4 else f"OT{self._period - 4}"
            draw_text_with_shadow(p, mid_x - 45, 22, f"LIVE  {period_text}  {self._clock}", small_font, QColor("#ffffff"))
        elif self._status_state == "post":
            draw_text_with_shadow(p, mid_x - 20, 22, "FINAL", small_font, QColor("#94a3b8"))
        else:
            draw_text_with_shadow(p, mid_x - 40, 22, self._status_text[:20], small_font, QColor("#64748b"))

        # Timeout Timer Overlay
        if self._timeout_seconds > 0:
            to_w = 120
            to_h = 36
            to_rect = QRectF(mid_x - to_w/2, h / 2 - to_h/2 + 30, to_w, to_h)
            p.fillRect(to_rect, QColor(239, 68, 68, 230))
            p.setPen(QPen(QColor(252, 165, 165), 2))
            p.drawRect(to_rect)
            draw_text_with_shadow(p, mid_x - to_w/2 + 10, h / 2 - to_h/2 + 54, f"TIMEOUT: {self._timeout_seconds}s", name_font, QColor("#ffffff"))

        # Substitution Overlay
        if self._sub_data:
            sub_w = 400
            sub_h = 100
            sub_rect = QRectF(mid_x - sub_w/2, h/2 - sub_h/2 + 20, sub_w, sub_h)
            
            p.setPen(QPen(QColor("#3b82f6"), 2))
            p.setBrush(QColor(15, 23, 42, 230))
            p.drawRoundedRect(sub_rect, 12, 12)
            
            # Sub In (Left)
            draw_text_with_shadow(p, mid_x - 140, h/2 - sub_h/2 + 35, "IN", tiny_font, QColor("#22c55e"))
            if self._sub_data.get("in_pixmap"):
                p.drawPixmap(int(mid_x - 170), int(h/2 - sub_h/2 + 45), self._sub_data["in_pixmap"])
            draw_text_with_shadow(p, mid_x - 180, h/2 + 45, self._sub_data["in_name"][:15], small_font, QColor("#ffffff"))
            draw_text_with_shadow(p, mid_x - 180, h/2 + 60, self._sub_data["in_stats"], tiny_font, QColor("#94a3b8"))

            # Sub Out (Right)
            draw_text_with_shadow(p, mid_x + 120, h/2 - sub_h/2 + 35, "OUT", tiny_font, QColor("#ef4444"))
            if self._sub_data.get("out_pixmap"):
                p.drawPixmap(int(mid_x + 90), int(h/2 - sub_h/2 + 45), self._sub_data["out_pixmap"])
            draw_text_with_shadow(p, mid_x + 80, h/2 + 45, self._sub_data["out_name"][:15], small_font, QColor("#ffffff"))
            draw_text_with_shadow(p, mid_x + 80, h/2 + 60, self._sub_data["out_stats"], tiny_font, QColor("#94a3b8"))
            
            # Arrows (Center)
            draw_text_with_shadow(p, mid_x - 15, h/2 + 5, ">>", small_font, QColor("#64748b"))
            draw_text_with_shadow(p, mid_x - 15, h/2 + 25, "<<", small_font, QColor("#64748b"))

        p.end()

    def _get_logo(self, team_id: Optional[int], abbr: str, size: int) -> Optional[QPixmap]:
        """Get team logo or fallback to placeholder."""
        if team_id:
            logo = get_team_logo(team_id, size)
            if logo:
                return logo
        primary, _ = get_team_colors(team_id) if team_id else ("#3b82f6", "#1e293b")
        return make_placeholder_logo(abbr, size, primary)
