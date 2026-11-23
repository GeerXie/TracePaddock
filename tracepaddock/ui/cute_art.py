"""Original cartoon art: characters for the mouse pointers, the racing scene behind the curves and the
decoration of the racing menu bar - all drawn here as SVG (no picture files, no third-party characters).
Pictures of the user (own pointer or background picture) are used as they are.

Characters: a flying squirrel with a mint scarf, a black-and-white tabby, a Shiba and a racing helmet.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtGui, QtSvg

__all__ = ["CHARACTERS", "SCENES", "head_svg", "scene_svg", "render_svg", "cursor", "image_cursor",
           "backdrop_image", "bar_pattern_svg", "write_asset", "IMAGE_FILTER"]

INK = "#5a4334"                                   # outline colour of all characters

#: key -> menu text (English, translated in the UI)
CHARACTERS: Dict[str, str] = {"squirrel": "Flying squirrel", "cat": "Kitten", "dog": "Puppy"}
#: background scenes: key -> menu text
SCENES: Dict[str, str] = {}
#: pictures the user may choose (pointer, background)
IMAGE_FILTER = "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.svg *.ico"


def _svg(body: str, w: float, h: float) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %g %g" width="%g" height="%g">%s</svg>'
            % (w, h, w, h, body))


# ---------------------------------------------------------------------- heads (64 x 64)
def _squirrel_head() -> str:
    return '''<g stroke="#52465f" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round">
  <path d="M36 52 Q47 53 46 42 Q44 33 41 25 Q37 12 45 7 Q50 3 59 7
           L56 10 Q62 13 62 23 Q61 42 53 51 Q48 58 36 57 Z" fill="#a496c7"/>
  <path d="M43 18 Q43 10 51 8 Q56 7 59 9 Q56 11 58 13 Q61 18 59 25
           Q52 22 50 16 Q48 19 43 18 Z" fill="#f0e4ee" stroke="none"/>
  <path d="M54 37 Q52 46 48 49" fill="none" stroke="#d6cae3" stroke-width="1.7"/>
  <path d="M19 49 Q29 47 36 49 Q40 48 42 51 L48 55 L42 61 L34 54 Q25 59 17 55 Z" fill="#82c7b2"/>
  <path d="M12 30 Q4 31 3 24 Q2 17 8 16 Q14 15 18 23 M34 22
           Q36 14 43 16 Q49 18 47 24 Q45 30 39 29" fill="#b1a1cc"/>
  <path d="M7 24 Q6 20 10 20 Q13 20 15 25 M38 24 Q40 20 43 21 Q45 22 42 25"
        fill="#eab0bd" stroke="none"/>
  <path d="M7 35 Q9 25 21 23 L21 19 Q26 19 29 23 Q41 22 46 33 L48 37 L45 40
           Q47 49 37 53 Q25 58 14 51 Q5 47 7 40 L4 37 Z" fill="#c1b1d8"/>
  <path d="M9 36 Q15 28 24 31 Q28 32 29 33 Q34 30 39 32 Q47 36 43 45
           Q38 52 28 52 Q14 55 9 44 Q7 39 9 36 Z" fill="#fff2df" stroke="none"/>
  <path d="M12 43 L16 43 M37 42 L41 41" stroke="#ec9da2" stroke-width="2.7"/>
  <path d="M15 32 L25 34 Q26 41 21 42 Q15 43 15 32 Z
           M31 34 L40 30 Q42 40 36 41 Q31 42 31 34 Z" fill="#3e2e4d" stroke="none"/>
  <path d="M15 30 L24 32 M32 31 L39 28" fill="none" stroke-width="1.8"/>
  <circle cx="22" cy="36" r="1.5" fill="#fffaf4" stroke="none"/>
  <circle cx="37" cy="34" r="1.5" fill="#fffaf4" stroke="none"/>
  <path d="M26 43 L30 42.5 L29 45 Z" fill="#a96f91" stroke="none"/>
  <path d="M23 47 Q29 51 35 45 L35.5 47" fill="none" stroke-width="1.7"/>
</g>'''


def _cat_head() -> str:
    return '''<g stroke="#30313a" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round">
  <path d="M9 30 L10 8 Q11 4 14 7 L28 19 L39 19 L51 6 Q54 4 54 9 L55 31
           Q62 42 53 51 Q46 58 34 59 Q17 59 9 50 Q3 43 9 30 Z" fill="#969aa6"/>
  <path d="M13 23 L14 12 L22 20 Z M45 20 L51 12 L51 24 Z" fill="#edb3bf" stroke="none"/>
  <path d="M11 30 Q20 30 24 39 L29 36 L33 23 L37 36 Q45 33 53 31
           Q60 43 50 51 Q39 59 25 56 Q9 54 8 42 Q8 36 11 30 Z" fill="#fff9ef" stroke="none"/>
  <path d="M19 20 L25 26 M27 18 L30 24 M37 19 L36 25 M45 21 L41 26
           M10 32 L16 35 M54 32 L48 35" stroke="#353845" stroke-width="3.0"/>
  <ellipse cx="23" cy="37" rx="4.4" ry="5.2" fill="#2d5c5b" stroke="none"/>
  <ellipse cx="43" cy="37" rx="4.4" ry="5.2" fill="#2d5c5b" stroke="none"/>
  <circle cx="24" cy="35" r="1.6" fill="#fff" stroke="none"/>
  <circle cx="44" cy="35" r="1.6" fill="#fff" stroke="none"/>
  <path d="M29.5 43 L36.5 43 Q34 48 33 47 Q31 47 29.5 43 Z" fill="#de8493" stroke="none"/>
  <path d="M28 48 Q31 51 33 48 Q36 51 39 48" fill="none" stroke-width="1.6"/>
  <path d="M7 42 L15 44 M7 48 L15 47 M58 41 L50 43 M57 48 L49 47" stroke-width="1.5"/>
</g>'''


def _dog_head() -> str:
    return f'''<g stroke="{INK}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round">
  <path d="M18 53 Q31 49 48 54 L38 61 L31 56 Z" fill="#d46b55"/>
  <path d="M9 31 L11 10 Q11 7 15 10 L28 19 Q35 17 41 20 L53 8
           Q56 6 56 12 L57 32 Q63 46 52 54 Q46 59 30 57 Q14 57 8 47 Q4 41 9 31 Z" fill="#dc9a5e"/>
  <path d="M13 24 L15 15 L23 21 Z M46 22 L53 14 L53 27 Z" fill="#f4ccad" stroke="none"/>
  <path d="M9 36 Q16 42 24 42 Q32 33 42 37 Q47 38 55 37 Q61 47 49 53
           Q42 58 29 55 Q12 54 9 45 Q7 40 9 36 Z" fill="#fff0d9" stroke="none"/>
  <path d="M17 29 Q21 24 26 27 M39 27 Q44 24 50 28" fill="none" stroke="#fff1d9" stroke-width="3.0"/>
  <path d="M16 34 Q23 30 30 34 Q24 39 18 37 Z M37 34 Q44 29 51 34 Q47 39 40 37 Z"
        fill="#fffaf2" stroke-width="1.35"/>
  <ellipse cx="26" cy="34.5" rx="2.9" ry="3.0" fill="#382722" stroke="none"/>
  <ellipse cx="47" cy="34.4" rx="2.7" ry="2.9" fill="#382722" stroke="none"/>
  <ellipse cx="41" cy="42.5" rx="4.2" ry="2.8" fill="#392725" stroke="none"/>
  <path d="M41 45 L40 47 Q34 51 29 47" fill="none" stroke-width="1.8"/>
  <path d="M14 44 L19 45" stroke="#eeb5a5" stroke-width="3"/>
</g>'''


def _helmet_head() -> str:
    return f'''<g stroke="{INK}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round">
  <path d="M8 40 C8 18 22 7 34 7 C48 7 58 19 58 34 L58 46 C58 52 54 55 48 55 L16 55 C11 55 8 51 8 46 Z"
        fill="#d7263d"/>
  <path d="M20 9.5 C26 7.4 30 7 34 7 L34 55 L28 55 L28 10 Z" fill="#ffffff" stroke="none"/>
  <path d="M23 10.5 L23 55" stroke="#1f9d55" stroke-width="3"/>
  <path d="M30 8 C40 7.5 54 14 57 30 L57 40 L30 40 C25 40 22 37 22 33 L22 30 C22 25 25 22 30 22 Z"
        fill="#2b3240"/>
  <path d="M34 26 L50 26" stroke="#9fb3d1" stroke-width="2.2"/>
  <path d="M8 46 L58 46" stroke-width="2"/>
  <circle cx="16" cy="48.5" r="2.4" fill="#ffffff" stroke="none"/>
</g>'''


_HEADS = {"squirrel": _squirrel_head, "cat": _cat_head, "dog": _dog_head, "helmet": _helmet_head}
CHARACTERS.update({"helmet": "Racing helmet"})


def head_svg(character: str) -> str:
    return _svg(_HEADS.get(character, _squirrel_head)(), 64, 64)


# ---------------------------------------------------------------------- scenes (1600 x 500, transparent sky)
def _racing() -> str:
    import math
    cx, cy, r = 1250.0, 520.0, 360.0
    ticks = []
    for k in range(0, 41):
        a = math.radians(200 - k * 4.0)                       # 200 deg .. 40 deg
        inner = r - (34 if k % 5 == 0 else 18)
        colour = "#e8484f" if k >= 32 else "#e6e6e6"
        ticks.append('<path d="M%.1f %.1f L%.1f %.1f" stroke="%s" stroke-width="%g" stroke-linecap="round"/>'
                     % (cx + inner * math.cos(a), cy - inner * math.sin(a), cx + r * math.cos(a), cy - r * math.sin(a),
                        colour, 6 if k % 5 == 0 else 3))
    numbers = "".join('<text x="%.1f" y="%.1f" font-family="Segoe UI" font-size="34" font-weight="bold" fill="%s" '
                      'text-anchor="middle">%d</text>'
                      % (cx + (r - 70) * math.cos(math.radians(200 - k * 20)), cy - (r - 70) * math.sin(math.radians(200 - k * 20)) + 12,
                         "#e8484f" if k >= 7 else "#e6e6e6", k) for k in range(0, 9))
    a = math.radians(72)
    needle = ('<path d="M%.1f %.1f L%.1f %.1f" stroke="#ff3b30" stroke-width="9" stroke-linecap="round"/>'
              '<circle cx="%g" cy="%g" r="26" fill="#2b2b2e" stroke="#9a9aa0" stroke-width="4"/>'
              % (cx, cy, cx + (r - 40) * math.cos(a), cy - (r - 40) * math.sin(a), cx, cy))
    arc = ('<path d="M%.1f %.1f A %g %g 0 1 1 %.1f %.1f" fill="none" stroke="#3a3a40" stroke-width="10"/>'
           % (cx + r * math.cos(math.radians(200)), cy - r * math.sin(math.radians(200)), r, r,
              cx + r * math.cos(math.radians(40)), cy - r * math.sin(math.radians(40))))
    stripes = ('<g transform="skewX(-20)" opacity="0.6">'
               '<rect x="760" y="0" width="34" height="500" fill="#1f9d55"/><rect x="800" y="0" width="34" height="500" '
               'fill="#f2f2f2"/><rect x="840" y="0" width="34" height="500" fill="#d7263d"/></g>')
    streaks = "".join('<path d="M%d %d L%d %d" stroke="#e6e6e6" stroke-width="%g" stroke-linecap="round" opacity="%.2f"/>'
                      % (x, y, x + w, y, s, o) for x, y, w, s, o in ((60, 140, 420, 3, 0.5), (140, 200, 520, 2, 0.4),
                                                                    (20, 270, 380, 4, 0.45), (220, 330, 460, 2, 0.35),
                                                                    (80, 400, 300, 3, 0.4)))
    flag = "".join('<rect x="%d" y="%d" width="22" height="22" fill="%s"/>'
                   % (1480 + 22 * i, 30 + 22 * j, "#f2f2f2" if (i + j) % 2 == 0 else "#2b2b2e")
                   for i in range(5) for j in range(4))
    return stripes + streaks + arc + "".join(ticks) + numbers + needle + flag


SCENES.update({"racing": "Racing: rev counter and stripes"})


def scene_svg(key: str) -> str:
    return _svg(_racing(), 1600, 500)


def bar_pattern_svg(kind: str = "racing", height: int = 30) -> Tuple[str, bool]:
    """Decoration of the menu bar: ``(svg, repeat)``; "racing" is one picture at the right end."""
    mid = height / 2.0
    if kind == "racing":
        stripes = ('<g transform="skewX(-25)"><rect x="70" y="0" width="9" height="%d" fill="#1f9d55"/>'
                   '<rect x="82" y="0" width="9" height="%d" fill="#f2f2f2"/><rect x="94" y="0" width="9" height="%d" '
                   'fill="#d7263d"/></g>' % (height, height, height))
        flag = "".join('<rect x="%d" y="%d" width="5" height="5" fill="%s"/>'
                       % (120 + 5 * i, int(mid) - 10 + 5 * j, "#f2f2f2" if (i + j) % 2 == 0 else "#3a3a40")
                       for i in range(6) for j in range(4))
        return _svg(stripes + flag, 230, height), False

    return "", False


# ---------------------------------------------------------------------- rendering
def render_svg(svg: str, width: int, height: int, ratio: float = 1.0) -> QtGui.QImage:
    image = QtGui.QImage(max(1, int(width * ratio)), max(1, int(height * ratio)), QtGui.QImage.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.transparent)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    image.setDevicePixelRatio(ratio)
    return image


def _crosshair(cx: float, cy: float, arm: float) -> str:
    lines = f"M{cx - arm} {cy} L{cx + arm} {cy} M{cx} {cy - arm} L{cx} {cy + arm}"
    return (f'<path d="{lines}" stroke="#ffffff" stroke-width="4" stroke-linecap="round"/>'
            f'<path d="{lines}" stroke="#3a2c25" stroke-width="1.6" stroke-linecap="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="2.2" fill="#ff8fab" stroke="#3a2c25" stroke-width="1"/>')


def _magnifier(x: float, y: float) -> str:
    return (f'<g transform="translate({x},{y})" stroke="{INK}" stroke-width="2" stroke-linecap="round">'
            '<path d="M7 7 L13 13" stroke-width="3.4"/><circle cx="0" cy="0" r="6.5" fill="#e3f4ff"/>'
            '<path d="M-3 -2 Q-2 -4 0 -4" fill="none" stroke="#ffffff" stroke-width="1.6"/></g>')


#: arrows around the lens (centre 14, 14) of the zoom pointer, clear of the lens and of the handle
_ZOOM_ARROWS = {"left": "M0.7 14 L5.0 9.8 L5.0 18.2 Z", "right": "M27.3 14 L23.0 9.8 L23.0 18.2 Z",
                "up": "M14 0.7 L9.8 5.0 L18.2 5.0 Z", "down": "M14 27.3 L9.8 23.0 L18.2 23.0 Z"}
_ZOOM_DIRECTIONS = {"x": ("left", "right"), "y": ("up", "down"), "xy": ("left", "right", "up", "down")}
#: hot spot of the zoom pointer: the centre of the lens
ZOOM_HOT_SPOT = (14, 14)


def zoom_pointer_svg(mode: str) -> str:
    """The plain zoom pointer (32 x 32, hot spot = the lens centre): a magnifier with a plus and arrows for the
    zoom direction - left / right for time (Z), up / down for values (Y), all four for both (U). Black with a white
    rim, so it is seen on dark and on light plots."""
    ink = 'stroke="#000000" stroke-linecap="round" stroke-linejoin="round"'
    rim = 'stroke="#ffffff" stroke-linecap="round" stroke-linejoin="round"'
    body = (f'<path d="M20.2 20.2 L29.4 29.4" {rim} stroke-width="6.2"/>'
            f'<path d="M20.2 20.2 L29.4 29.4" {ink} stroke-width="3.3"/>'
            f'<circle cx="14" cy="14" r="7" fill="#ffffff" fill-opacity="0.8" {rim} stroke-width="3.8"/>'
            f'<circle cx="14" cy="14" r="7" fill="none" {ink} stroke-width="1.8"/>'
            f'<path d="M10.6 14 H17.4 M14 10.6 V17.4" {ink} stroke-width="1.9"/>')
    for name in _ZOOM_DIRECTIONS.get(mode, _ZOOM_DIRECTIONS["x"]):
        arrow = _ZOOM_ARROWS[name]
        body += f'<path d="{arrow}" fill="#ffffff" {rim} stroke-width="2.4"/><path d="{arrow}" fill="#000000"/>'
    return _svg(body, 32, 32)


@lru_cache(maxsize=8)
def zoom_pointer(mode: str, ratio: float = 1.0) -> QtGui.QCursor:
    """Magnifier pointer of a zoom (``mode`` "x", "y" or "xy"), see :func:`zoom_pointer_svg`."""
    image = render_svg(zoom_pointer_svg(mode), 32, 32, ratio)
    return QtGui.QCursor(QtGui.QPixmap.fromImage(image), *ZOOM_HOT_SPOT)


#: where the character (or the user's picture) sits in the 48 x 48 pointer: x, y, size
_PICTURE_BOX = {"zoom": (16.0, 16.0, 32.0), "pointer": (6.0, 6.0, 36.0)}


def _cursor_parts(kind: str) -> Tuple[str, Tuple[int, int]]:
    """Overlay and hot spot for the zoom cursor or the ordinary picture cursor."""
    if kind == "zoom":
        return _magnifier(17, 36) + _crosshair(9, 9, 8), (9, 9)
    return "", (24, 24)


def cursor_svg(kind: str, character: str) -> Tuple[str, Tuple[int, int]]:
    """SVG (48 x 48) and hot spot of a pointer: ``"zoom"`` (precise cross + character with a magnifier)
    or ``"pointer"`` (character centred at the click point)."""
    head = _HEADS.get(character, _squirrel_head)()
    x, y, size = _PICTURE_BOX["zoom" if kind == "zoom" else "pointer"]
    over, hot = _cursor_parts(kind)
    return _svg(f'<g transform="translate({x},{y}) scale({size / 64.0})">{head}</g>' + over, 48, 48), hot


@lru_cache(maxsize=32)
def cursor(kind: str, character: str, ratio: float = 1.0) -> QtGui.QCursor:
    svg, (hx, hy) = cursor_svg(kind, character)
    image = render_svg(svg, 48, 48, ratio)
    return QtGui.QCursor(QtGui.QPixmap.fromImage(image), hx, hy)


def _load_picture(path: str) -> Optional[QtGui.QImage]:
    image = QtGui.QImage(path)
    return None if image.isNull() else image


@lru_cache(maxsize=16)
def image_cursor(kind: str, path: str, stamp: float, ratio: float = 1.0) -> Optional[QtGui.QCursor]:
    """A pointer made of the user's picture (``stamp``: modification time, so a changed file is read again):
    the ordinary pointer is centred on the picture, and the zoom pointer has a precise cross and magnifier.
    ``None`` when the picture cannot be read."""
    picture = _load_picture(path)
    if picture is None:
        return None
    kind = "zoom" if kind == "zoom" else "pointer"
    over, (hx, hy) = _cursor_parts(kind)
    canvas = QtGui.QImage(int(48 * ratio), int(48 * ratio), QtGui.QImage.Format_ARGB32_Premultiplied)
    canvas.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
    x, y, size = _PICTURE_BOX[kind]
    scaled = picture.scaled(int(size * ratio), int(size * ratio), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    if kind == "zoom":
        left, top = (x + size) * ratio - scaled.width(), (y + size) * ratio - scaled.height()
    else:
        left, top = (canvas.width() - scaled.width()) / 2, (canvas.height() - scaled.height()) / 2
    destination = QtCore.QRectF(left, top, scaled.width(), scaled.height())
    painter.drawImage(destination, scaled)
    if over:
        painter.drawImage(QtCore.QRectF(canvas.rect()), render_svg(_svg(over, 48, 48), 48, 48, ratio))
    painter.end()
    canvas.setDevicePixelRatio(ratio)
    return QtGui.QCursor(QtGui.QPixmap.fromImage(canvas), hx, hy)


@lru_cache(maxsize=2)
def _scene_image(key: str, width: int) -> QtGui.QImage:
    return render_svg(scene_svg(key), width, int(width * 500 / 1600))


@lru_cache(maxsize=1)
def _picture_file(path: str, stamp: float) -> Optional[QtGui.QImage]:
    """A picture file of the user, read once (``stamp``: modification time, so a changed file is read again)."""
    return _load_picture(path)


def backdrop_image(key: str, width: int) -> Optional[QtGui.QImage]:
    """The picture behind the curves: a built-in scene key or a picture file of the user ("" = none); ``None``
    when the file cannot be read. Scenes are drawn for the width (rounded to 50 px: few renders), a file is
    read once; only the picture asked for last stays in memory."""
    if key not in SCENES:
        _scene_image.cache_clear()
    if not key or key in SCENES:
        _picture_file.cache_clear()
        return _scene_image(key, max(200, int(width // 50 * 50))) if key else None
    path = Path(key)
    try:
        stamp = path.stat().st_mtime if path.is_file() else None
    except OSError:
        stamp = None
    if stamp is None:
        _picture_file.cache_clear()
        return None
    return _picture_file(str(path), stamp)


def write_asset(folder: Path, name: str, svg: str) -> Path:
    """Write an SVG for the style sheet (it needs a file); returns the path with forward slashes usable in QSS."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    data = svg.encode("utf-8")
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    return path
