"""Original vector artwork for the TracePaddock application and its classic tool bar."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence

from PySide6 import QtCore, QtGui, QtSvg

__all__ = ["ICONS", "application_icon", "icon"]

_APPLICATION = '''
<defs>
  <linearGradient id="case" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0" stop-color="#293f78"/><stop offset="1" stop-color="#141c3b"/>
  </linearGradient>
</defs>
<rect x="2" y="2" width="60" height="60" rx="14" fill="url(#case)" stroke="#536bd1" stroke-width="2"/>
<rect x="8" y="15" width="48" height="35" rx="8" fill="#0b152d"/>
<path d="M13 33 H51 M22 20 V45 M42 20 V45" stroke="#263b60" stroke-width="1.5"/>
<path d="M13 35 H21 L27 23 L34 42 L41 28 L46 35 H51" fill="none" stroke="#62f1ef"
      stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="49" cy="9" r="2.5" fill="#92ffc5"/>
<path d="M25 55 H39" stroke="#677dcd" stroke-width="2.4" stroke-linecap="round"/>
'''

_D = "#30353b"            # dark outline
_B = "#2f7bd8"            # blue
_LB = "#cfe6fb"           # light blue (lens, paper shade)
_Y = "#f2c14e"            # folder yellow
_YD = "#d69a1f"
_G = "#2e9e4f"            # green
_R = "#d9363e"            # red
_P = "#ffffff"            # paper


def _lens(cx: float, cy: float, r: float = 4.2) -> str:
    return (f'<path d="M{cx + r * 0.7:.2f} {cy + r * 0.7:.2f} L{cx + r * 0.7 + 4.2:.2f} {cy + r * 0.7 + 4.2:.2f}" '
            f'stroke="{_D}" stroke-width="2.2" stroke-linecap="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{_LB}" stroke="{_D}" stroke-width="1.2"/>')


def _doc(x: float = 2.5, y: float = 1.5, w: float = 9, h: float = 12.5) -> str:
    return (f'<path d="M{x} {y} H{x + w - 3} L{x + w} {y + 3} V{y + h} H{x} Z" fill="{_P}" stroke="{_D}" '
            f'stroke-width="1" stroke-linejoin="round"/><path d="M{x + w - 3} {y} V{y + 3} H{x + w}" fill="none" '
            f'stroke="{_D}" stroke-width="1"/>')


def _arrow(x0: float, y0: float, x1: float, y1: float, colour: str = _B, width: float = 1.8) -> str:
    import math
    a = math.atan2(y1 - y0, x1 - x0)
    hx1 = x1 - 3.2 * math.cos(a - 0.55)
    hy1 = y1 - 3.2 * math.sin(a - 0.55)
    hx2 = x1 - 3.2 * math.cos(a + 0.55)
    hy2 = y1 - 3.2 * math.sin(a + 0.55)
    return (f'<path d="M{x0} {y0} L{x1} {y1}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>'
            f'<path d="M{hx1:.2f} {hy1:.2f} L{x1} {y1} L{hx2:.2f} {hy2:.2f}" fill="none" stroke="{colour}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')


def _plus(cx: float, cy: float, colour: str = _G) -> str:
    return (f'<circle cx="{cx}" cy="{cy}" r="3.3" fill="{colour}"/>'
            f'<path d="M{cx - 1.8} {cy} H{cx + 1.8} M{cx} {cy - 1.8} V{cy + 1.8}" stroke="#fff" stroke-width="1.3"/>')


def _wave(colour: str = _B, y: float = 8.0) -> str:
    return (f'<path d="M1.5 {y} C3.5 {y - 4} 5.5 {y - 4} 7.5 {y} S11.5 {y + 4} 14.5 {y}" fill="none" stroke="{colour}" '
            f'stroke-width="1.4" stroke-linecap="round"/>')


ICONS: Dict[str, str] = {
    "open": (f'<path d="M1.5 4 H6 L7.5 5.5 H13 V13 H1.5 Z" fill="{_YD}" stroke="{_D}" stroke-width="0.9" stroke-linejoin="round"/>'
             f'<path d="M3.2 7.2 H15 L13 13 H1.5 Z" fill="{_Y}" stroke="{_D}" stroke-width="0.9" stroke-linejoin="round"/>'),
    "prev_file": _doc(5.5, 1.5) + _arrow(9, 10.5, 1.8, 10.5),
    "next_file": _doc(1.5, 1.5) + _arrow(7, 10.5, 14.2, 10.5),
    "signals": (f'<rect x="1.5" y="1.5" width="13" height="13" rx="1" fill="{_P}" stroke="{_D}" stroke-width="0.9"/>'
                + "".join(f'<circle cx="4" cy="{y}" r="1.2" fill="{c}"/><path d="M6.5 {y} H12.5" stroke="#7a8590" '
                          f'stroke-width="1.2"/>' for y, c in ((4.8, _R), (8, _G), (11.2, _B)))),
    "save_view": (f'<path d="M1.5 1.5 H12 L14.5 4 V14.5 H1.5 Z" fill="{_B}" stroke="{_D}" stroke-width="0.9" stroke-linejoin="round"/>'
                  f'<rect x="4" y="1.5" width="7" height="4.5" fill="{_P}" stroke="{_D}" stroke-width="0.6"/>'
                  f'<rect x="8.4" y="2.3" width="1.6" height="2.8" fill="{_D}"/>'
                  f'<rect x="3.5" y="8.5" width="9" height="6" fill="{_LB}" stroke="{_D}" stroke-width="0.6"/>'
                  f'<path d="M5 10.5 H11 M5 12.5 H11" stroke="#7a8590" stroke-width="0.8"/>'),
    "view_back": _doc(6, 1.5, 8, 11) + _arrow(9.5, 12.5, 1.6, 12.5, _B, 2.0),
    "view_forward": _doc(2, 1.5, 8, 11) + _arrow(6.5, 12.5, 14.4, 12.5, _B, 2.0),
    "zoom": _lens(6.5, 6.5) + f'<path d="M4.4 6.5 H8.6 M6.5 4.4 V8.6" stroke="{_D}" stroke-width="1.2"/>',
    "zoom_x": _lens(6, 5.5, 3.8) + _arrow(4, 14, 1.2, 14, _B, 1.3) + _arrow(9, 14, 14.8, 14, _B, 1.3)
    + f'<path d="M1.2 14 H14.8" stroke="{_B}" stroke-width="1.3"/>',
    "zoom_y": _lens(5.5, 6.5, 3.8) + _arrow(14, 7, 14, 1.2, _B, 1.3) + _arrow(14, 9, 14, 14.8, _B, 1.3),
    "zoom_undo": _lens(6.5, 6.5) + f'<path d="M4.4 6.5 H8.6" stroke="{_D}" stroke-width="1.3"/>',
    "zoom_all": (f'<rect x="4" y="4" width="8" height="8" fill="{_LB}" stroke="{_D}" stroke-width="0.9"/>'
                 + _arrow(5.5, 5.5, 1.5, 1.5, _B, 1.4) + _arrow(10.5, 5.5, 14.5, 1.5, _B, 1.4)
                 + _arrow(5.5, 10.5, 1.5, 14.5, _B, 1.4) + _arrow(10.5, 10.5, 14.5, 14.5, _B, 1.4)),
    "ref": (f'<rect x="1.5" y="1.5" width="13" height="13" fill="#20242a" stroke="{_D}" stroke-width="0.8"/>'
            + _wave("#5ad1e6", 9) + f'<path d="M9 2 V14" stroke="#ffffff" stroke-width="1.1" stroke-dasharray="1.6 1.2"/>'),
    "page_prev": (f'<path d="M8 3 L3 8 L8 13 M13 3 L8 8 L13 13" fill="none" stroke="{_B}" stroke-width="1.9" '
                  'stroke-linecap="round" stroke-linejoin="round"/>'),
    "page_next": (f'<path d="M3 3 L8 8 L3 13 M8 3 L13 8 L8 13" fill="none" stroke="{_B}" stroke-width="1.9" '
                  'stroke-linecap="round" stroke-linejoin="round"/>'),
    "xy": (f'<path d="M2.5 1.5 V13.5 H14.5" fill="none" stroke="{_D}" stroke-width="1.3"/>'
           f'<path d="M3.5 12 C6 4 9 12 13.5 3" fill="none" stroke="{_R}" stroke-width="1.5" stroke-linecap="round"/>'),
    "auto_range": (f'<path d="M2.5 1.5 V14.5" stroke="{_D}" stroke-width="1.2"/>' + _wave(_G, 8.5)
                   + _arrow(5.5, 5, 5.5, 1.3, _B, 1.3) + _arrow(5.5, 11.5, 5.5, 14.7, _B, 1.3)),
    "by_unit": (_wave(_R, 5) + _wave("#e08a1e", 6.5)
                + f'<path d="M1.5 9.5 H14.5" stroke="#9aa3ad" stroke-width="0.8" stroke-dasharray="1.2 1"/>' + _wave(_B, 12.5)),
    "bands": ("".join(f'<rect x="1.5" y="{y}" width="13" height="4" fill="{f}" stroke="{_D}" stroke-width="0.6"/>'
                      for y, f in ((1.5, "#fde2e2"), (6, "#e2f3e5"), (10.5, "#e2ecfb")))
              + "".join(f'<path d="M2.5 {y + 2} C5 {y} 7 {y + 4} 9.5 {y + 2} S13 {y} 13.5 {y + 2}" fill="none" '
                        f'stroke="{c}" stroke-width="1"/>' for y, c in ((1.5, _R), (6, _G), (10.5, _B)))),
    "unit_bands": ("".join(f'<rect x="1.5" y="{y}" width="13" height="6" fill="{f}" stroke="{_D}" stroke-width="0.6"/>'
                           for y, f in ((1.5, "#fde2e2"), (8.5, "#e2ecfb")))
                   + f'<path d="M2.5 5 C5 2.5 7 7 9.5 4.5 S13 3 13.5 4.5" fill="none" stroke="{_R}" stroke-width="1"/>'
                   + f'<path d="M2.5 5.8 C5 4 7 7.5 9.5 5.5 S13 5 13.5 6" fill="none" stroke="#e08a1e" stroke-width="1"/>'
                   + f'<path d="M2.5 12 C5 9.5 7 14 9.5 11.5 S13 10 13.5 11.5" fill="none" stroke="{_B}" stroke-width="1"/>'),
    "add_signal": _wave(_B, 7) + _plus(11.8, 11.8),
    "add_ref": (f'<path d="M1.5 6 H14.5" stroke="{_R}" stroke-width="1.3" stroke-dasharray="2 1.4"/>' + _wave(_B, 7.5)
                + _plus(11.8, 11.8)),
    "mapping": (f'<rect x="1" y="4.5" width="5.5" height="7" rx="1" fill="{_P}" stroke="{_D}" stroke-width="0.9"/>'
                f'<text x="3.75" y="10.3" font-family="Arial" font-size="6" font-weight="bold" text-anchor="middle" '
                f'fill="{_D}">3</text>' + _arrow(7, 8, 9.6, 8, _G, 1.4)
                + f'<rect x="10" y="4.5" width="5.5" height="7" rx="1" fill="#fff4cc" stroke="{_D}" stroke-width="0.9"/>'
                f'<text x="12.75" y="10.3" font-family="Arial" font-size="6" font-weight="bold" text-anchor="middle" '
                f'fill="{_R}">A</text>'),
    "rules": (f'<rect x="2" y="1.5" width="12" height="13" rx="1" fill="{_P}" stroke="{_D}" stroke-width="0.9"/>'
              f'<path d="M4 5 L5.3 6.4 L7.6 3.8" fill="none" stroke="{_G}" stroke-width="1.4" stroke-linecap="round"/>'
              f'<path d="M4.2 9.6 L7.2 12.6 M7.2 9.6 L4.2 12.6" stroke="{_R}" stroke-width="1.4" stroke-linecap="round"/>'
              f'<path d="M9 5 H12.5 M9 11 H12.5" stroke="#7a8590" stroke-width="1.2"/>'),
    "images": (f'<rect x="1.5" y="2.5" width="13" height="11" rx="1" fill="{_LB}" stroke="{_D}" stroke-width="0.9"/>'
               f'<path d="M2.5 12.5 L6 8 L8.5 10.5 L10.5 8.5 L13.5 12.5 Z" fill="{_G}"/>'
               f'<circle cx="11" cy="5.5" r="1.5" fill="{_Y}"/>'),
    "export": _doc(1.5, 1.5, 9, 13) + f'<path d="M4 6 H8 M4 8.5 H8 M4 11 H7" stroke="#7a8590" stroke-width="1"/>'
    + _arrow(8, 9, 15, 9, _G, 2.0),
    "info": (f'<circle cx="8" cy="8" r="6.5" fill="{_B}" stroke="#1f5ea8" stroke-width="0.8"/>'
             '<circle cx="8" cy="4.8" r="1.1" fill="#ffffff"/><path d="M8 7.2 V12" stroke="#ffffff" stroke-width="1.9" '
             'stroke-linecap="round"/>'),
    "layout": (f'<rect x="1.5" y="2" width="13" height="12" rx="1" fill="{_P}" stroke="{_D}" stroke-width="0.9"/>'
               f'<rect x="1.5" y="2" width="13" height="2.4" fill="{_B}"/><path d="M10 4.4 V14 M1.5 10.5 H10" '
               f'stroke="{_D}" stroke-width="0.9"/>'),
}


def _render_icon(body: str, view_size: int, sizes: Sequence[int]) -> QtGui.QIcon:
    result = QtGui.QIcon()
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_size} {view_size}" '
           f'width="{view_size}" height="{view_size}">{body}</svg>')
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    for size in sizes:
        image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        renderer.render(painter)
        painter.end()
        result.addPixmap(QtGui.QPixmap.fromImage(image))
    return result


@lru_cache(maxsize=1)
def application_icon() -> QtGui.QIcon:
    """The oscilloscope icon shared by the window and portable executable."""
    return _render_icon(_APPLICATION, 64, (16, 20, 24, 32, 48, 64, 128, 256))


@lru_cache(maxsize=None)
def icon(name: str) -> QtGui.QIcon:
    """The tool bar icon ``name`` (empty for an unknown name)."""
    body = ICONS.get(name)
    return _render_icon(body, 16, (16, 20, 24, 32, 48)) if body is not None else QtGui.QIcon()
