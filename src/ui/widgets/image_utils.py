"""Utility helpers to load team logos (SVG) and player photos (PNG) as QPixmaps."""

import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap, QPainter, QImage, QColor
from PySide6.QtSvg import QSvgRenderer

logger = logging.getLogger(__name__)

TEAM_LOGOS_DIR = Path("data") / "cache" / "team_logos"
PLAYER_PHOTOS_DIR = Path("data") / "cache" / "player_photos"


def _render_svg_to_pixmap(svg_path: str, size: int) -> Optional[QPixmap]:
    """Render an SVG file to a QPixmap at the given size."""
    try:
        with open(svg_path, "rb") as f:
            data = f.read()
        renderer = QSvgRenderer(QByteArray(data))
        if not renderer.isValid():
            return None
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap
    except Exception as e:
        logger.debug(f"SVG render error for {svg_path}: {e}")
        return None


def _load_png_to_pixmap(png_path: str, size: int) -> Optional[QPixmap]:
    """Load a PNG file and scale it to the given size."""
    try:
        pixmap = QPixmap(png_path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
    except Exception as e:
        logger.debug(f"PNG load error for {png_path}: {e}")
        return None


def _make_circle_pixmap(pixmap: QPixmap) -> QPixmap:
    """Crop a pixmap into a circle shape with transparent background."""
    size = min(pixmap.width(), pixmap.height())
    result = QPixmap(size, size)
    result.fill(Qt.GlobalColor.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    from PySide6.QtGui import QPainterPath
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    painter.setClipPath(path)
    x = (size - pixmap.width()) // 2
    y = (size - pixmap.height()) // 2
    painter.drawPixmap(x, y, pixmap)
    painter.end()
    return result


@lru_cache(maxsize=64)
def get_team_logo(team_id: int, size: int = 48) -> Optional[QPixmap]:
    """Get team logo as QPixmap. Cached in memory.

    Handles SVG files.
    """
    path = TEAM_LOGOS_DIR / f"{team_id}.svg"
    if not path.exists():
        # Fallback to PNG if it exists
        path = TEAM_LOGOS_DIR / f"{team_id}.png"
        
    if not path.exists():
        # Try to download
        try:
            from src.data.image_cache import get_team_logo_path
            result = get_team_logo_path(team_id)
            if not result:
                return None
            path = Path(result)
        except Exception:
            return None

    # Detect if file is SVG or PNG by reading header
    try:
        with open(path, "rb") as f:
            header = f.read(10)
    except Exception:
        return None

    if header.startswith(b"<?xml") or header.startswith(b"<svg"):
        return _render_svg_to_pixmap(str(path), size)
    elif header.startswith(b"\x89PNG"):
        return _load_png_to_pixmap(str(path), size)
    else:
        return None


@lru_cache(maxsize=256)
def get_player_photo(player_id: int, size: int = 40, circle: bool = False) -> Optional[QPixmap]:
    """Get player headshot as QPixmap. Cached in memory."""
    path = PLAYER_PHOTOS_DIR / f"{player_id}.png"
    if not path.exists():
        try:
            from src.data.image_cache import get_player_photo_path
            result = get_player_photo_path(player_id)
            if not result:
                return None
        except Exception:
            return None

    pixmap = _load_png_to_pixmap(str(path), size)
    if pixmap and circle:
        pixmap = _make_circle_pixmap(pixmap)
    return pixmap


def make_placeholder_logo(text: str, size: int = 48,
                          bg_color: str = "#3b82f6") -> QPixmap:
    """Create a circular placeholder with team abbreviation text."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(bg_color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.setPen(QColor("#ffffff"))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(max(10, size // 3))
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text[:3])
    painter.end()
    return pixmap


def clear_pixmap_caches():
    """Clear all in-memory pixmap caches."""
    get_team_logo.cache_clear()
    get_player_photo.cache_clear()
