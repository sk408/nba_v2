"""NBA 2K-style splash screen with drifting team logo constellation."""

import json
import logging
import os
import random
from datetime import date
from pathlib import Path
from typing import List, Optional, Set

from PySide6.QtCore import Qt, QRectF, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QPainter, QPen, QPixmap,
    QLinearGradient, QRadialGradient,
)
from PySide6.QtWidgets import QSplashScreen, QApplication, QGraphicsOpacityEffect

logger = logging.getLogger(__name__)

_SCOREBOARD_CACHE = Path("data") / "cache" / "scoreboard_today.json"
_SPLASH_CACHE_DIR = Path("data") / "cache" / "splash_stars"
_SPLASH_MANIFEST = _SPLASH_CACHE_DIR / "manifest.json"

# Star list indices for readability
_X, _Y, _DX, _DY, _SIZE, _COMPOSITE = 0, 1, 2, 3, 4, 5


def _load_oswald_font() -> str:
    """Load Oswald font once, return family name."""
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    oswald_path = os.path.join(font_dir, "Oswald.ttf")
    if os.path.exists(oswald_path):
        fid = QFontDatabase.addApplicationFont(oswald_path)
        if fid != -1:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                return families[0]
    return "Segoe UI"


def _center_on_screen(widget):
    """Center a widget on the primary screen."""
    screen = QApplication.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - widget.width()) // 2
        y = geo.y() + (geo.height() - widget.height()) // 2
        widget.move(x, y)


class SplashScreen(QSplashScreen):
    """Dark-themed splash with drifting team logo constellation."""

    _BG = QColor("#0b0f19")
    _ACCENT = QColor("#00e5ff")
    _TEXT = QColor("#e2e8f0")
    _MUTED = QColor("#94a3b8")
    _SURFACE = QColor(20, 30, 45, 200)
    _TRACK = QColor(30, 41, 59, 180)

    def __init__(self):
        super().__init__()
        self.setFixedSize(1200, 800)
        self.setWindowFlags(
            Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self._status_text = ""
        self._progress = 0.0
        self._font_family = _load_oswald_font()
        self._ready = False
        self._linger_window = None
        self._linger_timer: Optional[QTimer] = None

        # Pre-cache fonts
        self._title_font = QFont(self._font_family, 44, QFont.Bold)
        self._title_font.setLetterSpacing(QFont.AbsoluteSpacing, 5.0)
        self._sub_font = QFont(self._font_family, 14)
        self._sub_font.setLetterSpacing(QFont.AbsoluteSpacing, 5.0)
        self._status_font = QFont("Segoe UI", 11)

        # Load constellation immediately (fast from PNG cache, first run
        # defers the expensive bake so the window still appears instantly)
        self._stars: List[list] = []
        self._constellation_ready = False
        self._try_load_cached_constellation()

        # Pre-render static background
        self._bg_pixmap = self._render_static_bg()

        # Animation timer (~30fps) — only starts when linger begins
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_tick)

        _center_on_screen(self)

    # ------------------------------------------------------------------
    # Pre-bake helpers
    # ------------------------------------------------------------------

    def _render_static_bg(self, title="NBA PREDICTIONS",
                          subtitle="GAME PREDICTION SYSTEM") -> QPixmap:
        """Render the static background elements once into a pixmap."""
        w, h = self.width(), self.height()
        cy = h // 2
        pm = QPixmap(w, h)
        pm.fill(self._BG)

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        # Top accent bar
        p.fillRect(0, 0, w, 3, self._ACCENT)

        # Inner panel
        margin = 60
        panel_rect = QRectF(margin, margin + 10, w - margin * 2, h - margin * 2 - 10)
        p.setBrush(self._SURFACE)
        p.setPen(QPen(QColor(255, 255, 255, 25)))
        p.drawRoundedRect(panel_rect, 10, 10)

        # Dark vignette behind center text
        vignette_cx, vignette_cy = w / 2, cy - 10
        vignette_rx, vignette_ry = 300, 130
        vignette_grad = QRadialGradient(vignette_cx, vignette_cy, max(vignette_rx, vignette_ry))
        vignette_grad.setColorAt(0, QColor(11, 15, 25, 242))
        vignette_grad.setColorAt(0.6, QColor(11, 15, 25, 200))
        vignette_grad.setColorAt(1, QColor(11, 15, 25, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(vignette_grad)
        p.drawEllipse(QRectF(
            vignette_cx - vignette_rx, vignette_cy - vignette_ry,
            vignette_rx * 2, vignette_ry * 2,
        ))

        # Title
        p.setFont(self._title_font)
        p.setPen(self._TEXT)
        title_y = cy - 70
        p.drawText(QRectF(0, title_y, w, 80), Qt.AlignHCenter | Qt.AlignTop, title)

        # Accent line
        line_y = title_y + 80
        line_w = 280
        p.setPen(QPen(self._ACCENT, 2))
        p.drawLine((w - line_w) // 2, line_y, (w + line_w) // 2, line_y)

        # Subtitle
        p.setFont(self._sub_font)
        p.setPen(self._MUTED)
        p.drawText(QRectF(0, line_y + 14, w, 30), Qt.AlignHCenter | Qt.AlignTop, subtitle)

        # Bottom accent bar
        p.fillRect(0, h - 3, w, 3, self._ACCENT)

        p.end()
        return pm

    @staticmethod
    def _bake_star_composite(logo_pixmap: QPixmap, size: int, opacity: float,
                             highlighted: bool, primary_hex: str) -> QPixmap:
        """Pre-render a star's glow + logo into one pixmap at target opacity."""
        pad = int(size * 0.5) if highlighted else 4
        cw = size + pad * 2
        ch = size + pad * 2
        comp = QPixmap(cw, ch)
        comp.fill(QColor(0, 0, 0, 0))

        p = QPainter(comp)
        p.setRenderHint(QPainter.Antialiasing)

        if highlighted:
            glow_radius = size * 1.2
            cx, cy = cw / 2, ch / 2
            gradient = QRadialGradient(cx, cy, glow_radius)
            glow_color = QColor(primary_hex)
            glow_color.setAlphaF(0.15)
            gradient.setColorAt(0, glow_color)
            gradient.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(gradient)
            p.drawEllipse(QRectF(cx - glow_radius, cy - glow_radius,
                                 glow_radius * 2, glow_radius * 2))

        p.setOpacity(opacity)
        p.drawPixmap(pad, pad, logo_pixmap)
        p.end()
        return comp

    # ------------------------------------------------------------------
    # Constellation setup (with disk cache)
    # ------------------------------------------------------------------

    def _try_load_cached_constellation(self):
        """Fast path: load cached PNGs at init. If cache miss, defer to linger."""
        today_ids = self._get_today_team_ids()
        today_key = sorted(today_ids) if today_ids else []
        self._today_ids = today_ids
        self._today_key = today_key
        if self._load_cached_constellation(today_key):
            self._constellation_ready = True

    def _build_constellation(self):
        """Full bake — only runs on first-ever launch or cache miss."""
        if self._constellation_ready:
            return

        today_ids = self._today_ids if hasattr(self, "_today_ids") else set()
        today_key = self._today_key if hasattr(self, "_today_key") else []

        try:
            from src.ui.widgets.nba_colors import TEAM_COLORS, get_team_colors
            from src.ui.widgets.image_utils import get_team_logo
        except Exception:
            logger.warning("Could not import team logo/color utilities")
            return

        all_team_ids = list(TEAM_COLORS.keys())

        if not today_ids:
            today_ids = set(random.sample(all_team_ids, min(6, len(all_team_ids))))
            today_key = sorted(today_ids)

        w, h = 1200, 800
        manifest_entries = []

        for team_id in all_team_ids:
            highlighted = team_id in today_ids

            if highlighted:
                size = random.randint(80, 100)
                opacity = round(random.uniform(0.30, 0.45), 3)
            else:
                size = random.randint(28, 56)
                opacity = round(random.uniform(0.08, 0.18), 3)

            pixmap = get_team_logo(team_id, size)
            if pixmap is None or pixmap.isNull():
                continue

            primary, _ = get_team_colors(team_id)
            composite = self._bake_star_composite(pixmap, size, opacity, highlighted, primary)

            x = round(random.uniform(0, w - size), 1)
            y = round(random.uniform(0, h - size), 1)
            dx = round(random.uniform(-1.5, 1.5), 2)
            dy = round(random.uniform(-1.5, 1.5), 2)
            if abs(dx) < 0.3:
                dx = 0.3 if dx >= 0 else -0.3
            if abs(dy) < 0.3:
                dy = 0.3 if dy >= 0 else -0.3

            self._stars.append([x, y, dx, dy, size, composite])
            manifest_entries.append({
                "team_id": team_id, "x": x, "y": y,
                "dx": dx, "dy": dy, "size": size,
            })

        self._save_constellation_cache(today_key, manifest_entries)
        self._constellation_ready = True

    def _load_cached_constellation(self, today_key: list) -> bool:
        """Try to load pre-baked composites from disk. Returns True on hit."""
        try:
            if not _SPLASH_MANIFEST.exists():
                return False
            manifest = json.loads(_SPLASH_MANIFEST.read_text())
            if manifest.get("today_key") != today_key:
                return False

            for entry in manifest["stars"]:
                png_path = _SPLASH_CACHE_DIR / f"{entry['team_id']}.png"
                if not png_path.exists():
                    return False
                comp = QPixmap(str(png_path))
                if comp.isNull():
                    return False
                self._stars.append([
                    entry["x"], entry["y"], entry["dx"], entry["dy"],
                    entry["size"], comp,
                ])
            return True
        except Exception:
            return False

    def _save_constellation_cache(self, today_key: list, entries: list):
        """Save baked composites as PNGs + manifest for instant next-launch."""
        try:
            _SPLASH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for i, star in enumerate(self._stars):
                team_id = entries[i]["team_id"]
                png_path = _SPLASH_CACHE_DIR / f"{team_id}.png"
                star[_COMPOSITE].save(str(png_path), "PNG")
            _SPLASH_MANIFEST.write_text(json.dumps({
                "today_key": today_key,
                "stars": entries,
            }))
        except Exception as e:
            logger.debug(f"Failed to cache splash composites: {e}")

    def _get_today_team_ids(self) -> Set[int]:
        """Get team IDs playing today, using disk cache for fast startup."""
        today_str = date.today().isoformat()

        try:
            if _SCOREBOARD_CACHE.exists():
                cached = json.loads(_SCOREBOARD_CACHE.read_text())
                if cached.get("date") == today_str and cached.get("team_ids"):
                    return set(cached["team_ids"])
        except Exception:
            pass

        team_ids: Set[int] = set()
        try:
            from src.data.gamecast import fetch_espn_scoreboard
            from src.database import db

            games = fetch_espn_scoreboard()
            for g in games:
                for key in ("home_team", "away_team"):
                    row = db.fetch_one(
                        "SELECT team_id FROM teams WHERE abbreviation = ?",
                        (g[key],),
                    )
                    if row:
                        team_ids.add(row["team_id"])

            if team_ids:
                try:
                    _SCOREBOARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
                    _SCOREBOARD_CACHE.write_text(json.dumps({
                        "date": today_str,
                        "team_ids": list(team_ids),
                    }))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not fetch today's games for splash: {e}")

        return team_ids

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _on_tick(self):
        """Update star positions and repaint."""
        w, h = self.width(), self.height()
        for s in self._stars:
            s[_X] += s[_DX]
            s[_Y] += s[_DY]
            sz = s[_SIZE]
            if s[_X] > w:
                s[_X] = float(-sz)
            elif s[_X] < -sz:
                s[_X] = float(w)
            if s[_Y] > h:
                s[_Y] = float(-sz)
            elif s[_Y] < -sz:
                s[_Y] = float(h)
        self.update()

    # ------------------------------------------------------------------
    # Status / progress
    # ------------------------------------------------------------------

    def set_status(self, msg: str, progress: float | None = None):
        """Update the status text and optional progress (0.0-1.0)."""
        self._status_text = msg
        if progress is not None:
            self._progress = max(0.0, min(1.0, progress))
        self.repaint()

    # ------------------------------------------------------------------
    # Paint — fast path: pre-baked pixmap blits only
    # ------------------------------------------------------------------

    def drawContents(self, painter: QPainter):
        w, h = self.width(), self.height()

        # 1. Static background (title, vignette, bars, panel)
        painter.drawPixmap(0, 0, self._bg_pixmap)

        # 2. Star composites (static during load, drifting during linger)
        for s in self._stars:
            comp = s[_COMPOSITE]
            pad = (comp.width() - s[_SIZE]) // 2
            painter.drawPixmap(int(s[_X]) - pad, int(s[_Y]) - pad, comp)

        # 3. Progress bar
        bar_margin = 120
        bar_y = h - 100
        bar_h = 4
        bar_w = w - bar_margin * 2
        bar_radius = 2.0

        painter.setPen(Qt.NoPen)
        painter.setBrush(self._TRACK)
        painter.drawRoundedRect(
            QRectF(bar_margin, bar_y, bar_w, bar_h), bar_radius, bar_radius
        )
        if self._progress > 0:
            fill_w = bar_w * self._progress
            grad = QLinearGradient(bar_margin, 0, bar_margin + fill_w, 0)
            grad.setColorAt(0, QColor("#00b8cc"))
            grad.setColorAt(1, self._ACCENT)
            painter.setBrush(grad)
            painter.drawRoundedRect(
                QRectF(bar_margin, bar_y, fill_w, bar_h), bar_radius, bar_radius
            )

        # 4. Status text
        painter.setFont(self._status_font)
        painter.setPen(self._ACCENT)
        painter.drawText(
            QRectF(bar_margin, bar_y + 12, bar_w, 30),
            Qt.AlignHCenter | Qt.AlignTop,
            self._status_text,
        )

    # ------------------------------------------------------------------
    # Linger + dismiss
    # ------------------------------------------------------------------

    def start_linger(self, window=None, duration_ms: int | None = None):
        """Start the linger period — constellation animates, then crossfade."""
        self._linger_window = window
        self._ready = True

        # Ensure constellation is built (instant from cache, slow only first run)
        if not self._constellation_ready:
            self._build_constellation()
        self._anim_timer.start()

        if duration_ms is None:
            try:
                from src.config import get as get_setting
                seconds = int(get_setting("splash_linger_seconds", 8))
            except Exception:
                seconds = 8
            duration_ms = seconds * 1000

        if duration_ms <= 0:
            if window is not None:
                self.crossfade_to(window)
            else:
                self._skip_linger = True
            return

        self._skip_linger = False
        self._linger_timer = QTimer(self)
        self._linger_timer.setSingleShot(True)
        self._linger_timer.setInterval(duration_ms)
        self._linger_timer.timeout.connect(self._end_linger)
        self._linger_timer.start()

    def set_linger_target(self, window):
        """Provide the main window after deferred construction."""
        self._linger_window = window
        if getattr(self, "_skip_linger", False):
            self._skip_linger = False
            self.crossfade_to(window)

    def _end_linger(self):
        """Linger period ended — crossfade to main window."""
        if self._linger_window is not None:
            self.crossfade_to(self._linger_window)
            self._linger_window = None
        else:
            QTimer.singleShot(100, self._end_linger)

    def mousePressEvent(self, event):
        if self._ready and self._linger_window is not None:
            if self._linger_timer and self._linger_timer.isActive():
                self._linger_timer.stop()
            self._end_linger()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if self._ready and self._linger_window is not None:
            if self._linger_timer and self._linger_timer.isActive():
                self._linger_timer.stop()
            self._end_linger()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Shutdown mode — reuse this splash as the closing screen
    # ------------------------------------------------------------------

    def show_shutdown(self):
        """Re-purpose this splash as a shutdown screen and show it."""
        self._anim_timer.stop()
        self._progress = 0.0
        self._status_text = ""
        # Swap background text, keep stars in place
        self._bg_pixmap = self._render_static_bg(
            title="SHUTTING DOWN",
            subtitle="STOPPING BACKGROUND SERVICES...",
        )
        # Clear any leftover graphics effect from the startup crossfade
        self.setGraphicsEffect(None)
        self.show()
        self.repaint()

    # ------------------------------------------------------------------
    # Crossfade
    # ------------------------------------------------------------------

    def crossfade_to(self, window, duration=600):
        """Fade out splash while the main window fades in underneath."""
        self._anim_timer.stop()

        win_effect = QGraphicsOpacityEffect(window)
        win_effect.setOpacity(0.0)
        window.setGraphicsEffect(win_effect)
        window.show()

        self._win_effect = win_effect
        win_anim = QPropertyAnimation(win_effect, b"opacity")
        win_anim.setDuration(duration)
        win_anim.setStartValue(0.0)
        win_anim.setEndValue(1.0)
        win_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._win_anim = win_anim

        splash_effect = QGraphicsOpacityEffect(self)
        splash_effect.setOpacity(1.0)
        self.setGraphicsEffect(splash_effect)
        self._splash_effect = splash_effect
        splash_anim = QPropertyAnimation(splash_effect, b"opacity")
        splash_anim.setDuration(duration)
        splash_anim.setStartValue(1.0)
        splash_anim.setEndValue(0.0)
        splash_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._splash_anim = splash_anim

        def _finish():
            self.close()
            window.setGraphicsEffect(None)

        splash_anim.finished.connect(_finish)
        win_anim.start()
        splash_anim.start()
