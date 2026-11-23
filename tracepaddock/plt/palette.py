"""Colour palette addressed by the colour number of a PLT entry.

The 32-entry table uses base colours in 0..15 and a pastel range in 16..31.
Colour 1 is black;
on a dark background black and white are swapped ("invert black/white") so
that the default colour stays visible.
"""
from __future__ import annotations

from itertools import cycle
from typing import Iterator, List, Sequence, Tuple

RGB = Tuple[int, int, int]

DEFAULT_PALETTE: List[RGB] = [
    (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0),
    (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255),
    (128, 128, 128), (192, 192, 192), (128, 0, 0), (0, 128, 0),
    (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    (240, 240, 111), (196, 247, 111), (146, 226, 114), (121, 218, 136),
    (121, 231, 184), (128, 231, 237), (127, 197, 255), (119, 150, 237),
    (121, 121, 223), (150, 119, 237), (197, 127, 255), (231, 128, 237),
    (231, 121, 184), (218, 121, 136), (226, 146, 114), (247, 196, 111),
]

#: Colour number used when a PLT entry gives none.
DEFAULT_COLOR = 1

#: A signal with this colour number is not drawn and has no legend colour,
#: but it is still loaded and can be referenced from ``exp(x1=..)``.
HIDDEN_COLOR = 0

#: Colour numbers cycled through by "apply cyclic colours" (inclusive range).
CYCLIC_RANGE = (1, 9)
MAX_PALETTE_COLOR = len(DEFAULT_PALETTE) - 1


def cyclic_colors(color_range: Tuple[int, int] = CYCLIC_RANGE) -> Iterator[int]:
    """Repeat the palette indices in the inclusive range, from lowest to highest."""
    first, last = sorted(map(int, color_range))
    return cycle(range(first, last + 1))

#: Colour words accepted in a PLT script instead of a number (case-insensitive).
#: The names describe the colours as seen on the dark screen background, which
#: is why 0 is called BLACK and 1 WHITE although the stored table is the other
#: way round (see :func:`color_rgb`).
COLOR_NAMES = {
    "BLACK": 0, "WHITE": 1, "LIGHTRED": 2, "LIGHTGREEN": 3, "YELLOW": 4,
    "LIGHTBLUE": 5, "LIGHTMAGENTA": 6, "LIGHTCYAN": 7, "DARKGRAY": 8,
    "LIGHTGRAY": 9, "RED": 10, "GREEN": 11, "BROWN": 12, "BLUE": 13,
    "MAGENTA": 14, "CYAN": 15,
}


def color_rgb(index: int, palette: Sequence[RGB] = DEFAULT_PALETTE, *,
              dark_background: bool = True, invert_bw: bool = True) -> RGB:
    """RGB of palette entry ``index`` (wraps around the palette length).

    The table is stored for a white background (entry 0 = white = paper,
    entry 1 = black = ink), which is what printing and clipboard pictures on
    white use. With ``invert_bw`` pure black and pure white swap places on a
    dark background, so that colour 1 is drawn white and colour 0 equals the
    background on the screen.
    """
    if index is None:
        index = DEFAULT_COLOR
    rgb = tuple(palette[int(index) % len(palette)])
    if invert_bw and dark_background:
        if rgb == (0, 0, 0):
            return (255, 255, 255)
        if rgb == (255, 255, 255):
            return (0, 0, 0)
    return rgb  # type: ignore[return-value]


def text_color_on(rgb: RGB) -> RGB:
    """Black or white, whichever reads better on a ``rgb`` background."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)
