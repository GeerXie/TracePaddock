"""Pictures for the image panel: built-in road signs drawn with QPainter, or image files.

A picture is named by a short spec:

``speed:80``          speed limit sign (red ring, number)
``end`` / ``end:80``  end of restriction (diagonal stripes, optionally the old limit in grey)
``cond:60:雨天``      speed limit with a supplementary panel below (condition text)
``warn`` / ``warn:!`` warning triangle with a symbol or short text
``none``              grey placeholder ("no sign")
``<path>``            an image file (PNG, JPG, SVG ...)

The signs are generic drawings (circle, ring, number), not copies of any
artwork.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtGui

__all__ = ["sign_pixmap", "is_builtin", "auto_speed_spec", "BUILTIN_EXAMPLES"]

BUILTIN_EXAMPLES = ("speed:50", "speed:80", "speed:120", "end", "end:80", "cond:60:雨天", "warn", "none")
_cache: Dict[Tuple[str, int], QtGui.QPixmap] = {}

RED = QtGui.QColor(200, 30, 30)
BLACK = QtGui.QColor(20, 20, 20)
GREY = QtGui.QColor(150, 150, 150)
WHITE = QtGui.QColor(255, 255, 255)


def is_builtin(spec: str) -> bool:
    head = str(spec).split(":", 1)[0].strip().lower()
    return head in ("speed", "end", "cond", "warn", "none")


def auto_speed_spec(value: Optional[float]) -> Optional[str]:
    """Sign for a speed value in km/h: a limit sign, or "end of restriction" for 0."""
    if value is None or value != value:
        return None
    v = float(value)
    if v <= 0:
        return "end"
    return "speed:%g" % round(v)


def _font(px: float, bold: bool = True) -> QtGui.QFont:
    font = QtGui.QFont("Arial")
    font.setPixelSize(max(6, int(px)))
    font.setBold(bold)
    return font


def _fit_text(painter: QtGui.QPainter, rect: QtCore.QRectF, text: str, start_px: float, bold: bool = True) -> None:
    px = start_px
    while px > 6:
        font = _font(px, bold)
        if QtGui.QFontMetricsF(font).horizontalAdvance(text) <= rect.width():
            break
        px -= 1
    painter.setFont(_font(px, bold))
    painter.drawText(rect, int(QtCore.Qt.AlignCenter), text)


def _speed(painter: QtGui.QPainter, rect: QtCore.QRectF, number: str, grey: bool = False) -> None:
    d = min(rect.width(), rect.height())
    circle = QtCore.QRectF(rect.center().x() - d / 2, rect.center().y() - d / 2, d, d)
    ring = d * 0.12
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(GREY if grey else RED)
    painter.drawEllipse(circle)
    painter.setBrush(WHITE)
    painter.drawEllipse(circle.adjusted(ring, ring, -ring, -ring))
    painter.setPen(GREY if grey else BLACK)
    inner = circle.adjusted(ring * 1.6, ring * 1.6, -ring * 1.6, -ring * 1.6)
    _fit_text(painter, inner, number, d * (0.42 if len(number) <= 2 else 0.34))


def _end(painter: QtGui.QPainter, rect: QtCore.QRectF, number: Optional[str]) -> None:
    d = min(rect.width(), rect.height())
    circle = QtCore.QRectF(rect.center().x() - d / 2, rect.center().y() - d / 2, d, d)
    painter.setPen(QtGui.QPen(BLACK, max(1.0, d * 0.03)))
    painter.setBrush(WHITE)
    painter.drawEllipse(circle.adjusted(1, 1, -1, -1))
    if number:
        painter.setPen(GREY)
        _fit_text(painter, circle.adjusted(d * 0.2, d * 0.2, -d * 0.2, -d * 0.2), number, d * 0.36)
    painter.save()
    painter.setClipRegion(QtGui.QRegion(circle.adjusted(2, 2, -2, -2).toRect(), QtGui.QRegion.Ellipse))
    painter.setPen(QtGui.QPen(BLACK, max(1.0, d * 0.035)))
    c = circle.center()
    for k in (-2, -1, 0, 1, 2):
        off = k * d * 0.07
        painter.drawLine(QtCore.QPointF(c.x() - d * 0.45 + off, c.y() + d * 0.45 + off),
                         QtCore.QPointF(c.x() + d * 0.45 + off, c.y() - d * 0.45 + off))
    painter.restore()


def _warn(painter: QtGui.QPainter, rect: QtCore.QRectF, symbol: str) -> None:
    d = min(rect.width(), rect.height())
    cx, top = rect.center().x(), rect.center().y() - d * 0.45
    tri = QtGui.QPolygonF([QtCore.QPointF(cx, top), QtCore.QPointF(cx + d * 0.5, top + d * 0.87),
                           QtCore.QPointF(cx - d * 0.5, top + d * 0.87)])
    painter.setPen(QtGui.QPen(RED, d * 0.09, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    painter.setBrush(WHITE)
    painter.drawPolygon(tri)
    painter.setPen(BLACK)
    _fit_text(painter, QtCore.QRectF(cx - d * 0.25, top + d * 0.35, d * 0.5, d * 0.45), symbol or "!", d * 0.38)


def _none(painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
    d = min(rect.width(), rect.height())
    circle = QtCore.QRectF(rect.center().x() - d / 2 + 2, rect.center().y() - d / 2 + 2, d - 4, d - 4)
    painter.setPen(QtGui.QPen(GREY, max(1.0, d * 0.03), QtCore.Qt.DashLine))
    painter.setBrush(QtCore.Qt.NoBrush)
    painter.drawEllipse(circle)
    painter.setPen(GREY)
    _fit_text(painter, circle, "—", d * 0.4)


def sign_pixmap(spec: str, size: int = 96) -> Optional[QtGui.QPixmap]:
    """Picture of ``spec`` at ``size`` pixels (height of the sign); ``None`` if a file cannot be read."""
    key = (str(spec), int(size))
    if key in _cache:
        return _cache[key]
    text = str(spec).strip()
    if not is_builtin(text):
        path = Path(text)
        pix = QtGui.QPixmap(str(path)) if path.exists() else QtGui.QPixmap()
        if pix.isNull():
            return None
        pix = pix.scaled(size * 2, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        _cache[key] = pix
        return pix
    parts = [p.strip() for p in text.split(":")]
    kind = parts[0].lower()
    height = size * (1.45 if kind == "cond" else 1.0)
    pix = QtGui.QPixmap(int(size), int(height))
    pix.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
    box = QtCore.QRectF(1, 1, size - 2, size - 2)
    try:
        if kind == "speed":
            _speed(painter, box, parts[1] if len(parts) > 1 else "?")
        elif kind == "end":
            _end(painter, box, parts[1] if len(parts) > 1 and parts[1] else None)
        elif kind == "cond":
            _speed(painter, box, parts[1] if len(parts) > 1 else "?")
            panel = QtCore.QRectF(size * 0.08, size * 1.02, size * 0.84, size * 0.4)
            painter.setPen(QtGui.QPen(BLACK, max(1.0, size * 0.02)))
            painter.setBrush(WHITE)
            painter.drawRect(panel)
            painter.setPen(BLACK)
            label = parts[2] if len(parts) > 2 else ""
            font = QtGui.QFont("Microsoft YaHei UI")
            px = size * 0.24
            while px > 6:
                font.setPixelSize(int(px))
                if QtGui.QFontMetricsF(font).horizontalAdvance(label) <= panel.width() - 4:
                    break
                px -= 1
            painter.setFont(font)
            painter.drawText(panel, int(QtCore.Qt.AlignCenter), label)
        elif kind == "warn":
            _warn(painter, box, parts[1] if len(parts) > 1 else "!")
        else:
            _none(painter, box)
    finally:
        painter.end()
    _cache[key] = pix
    return pix
