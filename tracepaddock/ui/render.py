"""Painting of the measurement view with QPainter.

The same code paints the screen widget and off-screen pictures (clipboard,
previews). Time view layout::

    +--------------------------------------+-----------+
    | DIGITAL window (one lane per signal) |           |
    +--------------------------------------+  legend   |
    | analog window: every signal has its  |  one row  |
    | own Y scale: scmin = bottom edge,    |  per      |
    | scmax = top edge; 10 x 10 grid       |  signal   |
    +--------------------------------------+-----------+

Signals are drawn as step functions (a sample holds until the next one).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6 import QtCore, QtGui

from ..display import DisplaySignal, c_exponent
from ..i18n import tr
from ..plt.palette import DEFAULT_PALETTE, HIDDEN_COLOR, color_rgb, text_color_on

__all__ = ["ViewState", "Layout", "compute_layout", "paint_view", "paint_plot", "paint_legend",
           "step_polyline", "state_segments", "state_fill", "range_text", "signal_color", "legend_value_text",
           "legend_font", "legend_row_shortened", "xy_axis_index", "xy_curve", "xy_pixels", "BACKGROUNDS",
           "STATE_FILLS"]

#: Fill colours of the states in a state band (value 0 is drawn neutral grey).
STATE_FILLS = [(142, 202, 230), (255, 183, 3), (144, 190, 109), (242, 132, 130), (205, 180, 219),
               (249, 199, 79), (67, 170, 139), (248, 150, 30), (181, 131, 141), (163, 196, 243),
               (255, 214, 165), (202, 255, 191)]
_STATE_ZERO = (217, 217, 217)
#: Height in pixels of a state band and of a one-bit lane when state bands are shown.
LANE_HEIGHT = 22.0
BIT_LANE_HEIGHT = 16.0

#: Background presets: key -> (label, RGB).
BACKGROUNDS = {
    "black": ("Black", (0, 0, 0)),
    "dark": ("Dark grey", (40, 44, 52)),
    "green": ("Eye-care green", (199, 237, 204)),
    "paper": ("Warm paper", (250, 243, 224)),
    "white": ("White", (255, 255, 255)),
}


@dataclass
class ViewState:
    """What is shown: time window, Y zoom window and presentation options."""

    x0: float = 0.0
    x1: float = 1.0
    #: Y zoom as fractions of the full height (0 = bottom, 1 = top).
    y0: float = 0.0
    y1: float = 1.0
    background: Tuple[int, int, int] = (0, 0, 0)
    invert_bw: bool = True
    grid: bool = True
    #: Line width in logical pixels.
    line_width: float = 1.0
    step_mode: bool = True
    font_family: str = "Arial"
    #: Largest legend font in pixels; rows shrink when many signals are shown.
    font_size: int = 13
    legend_fraction: float = 0.757
    digital_fraction: float = 0.097
    palette: Sequence[Tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_PALETTE)
    #: Time of the cursor / reference line, ``None`` = none (legend shows the ranges).
    cursor: Optional[float] = None
    #: Index of the selected signal or -1.
    selected: int = -1
    #: Whether the selected row is currently drawn highlighted (it flashes).
    flash: bool = True
    #: Indices of emphasized signals; when not empty all other analog signals are hidden - also a selected one
    #: (a click only selects; H shows it).
    emphasized: frozenset = frozenset()
    #: While signals are emphasized: the signal reached with TAB / the mouse wheel is shown until the selection
    #: moves on (-1 = none); H makes it emphasized.
    temporary: int = -1
    value_format: str = "%g"
    #: Show the time of every vertical grid line at the bottom of the analog window.
    time_labels: bool = True
    #: XY view: the first analog signal of the legend is the X axis, the others are drawn over it.
    xy: bool = False
    #: XY view: colour the curve by time (light = early, full colour = late) and mark start / end.
    xy_time_color: bool = True
    #: XY view: draw the samples as points instead of connecting them.
    xy_points: bool = False
    #: Results of rules shown as lanes in the upper window: (rule name, [(t0, t1 or None, verdict)]).
    rule_lanes: list = field(default_factory=list)
    #: A pinned time (e.g. an event chosen in the event table): drawn as a line, the legend shows its values.
    marker: Optional[float] = None
    #: A signal that stopped being sent (see :mod:`tracepaddock.core.sampling`): after its end the last value is
    #: drawn "dashed" (default), "hold" (drawn like a valid value) or "cut" (not drawn).
    stopped: str = "dashed"
    # ---- theme (see themes.py); None / "" = the classic look
    #: key of the theme whose colours are filled in (themes.apply_plot_colours), "" = none
    theme: str = ""
    #: "block": coloured name field; "chip": a colour mark and the name in text colour
    legend_style: str = "block"
    legend_bg: Optional[Tuple[int, int, int]] = None
    text_color: Optional[Tuple[int, int, int]] = None
    grid_color: Optional[Tuple[int, int, int]] = None
    axis_color: Optional[Tuple[int, int, int]] = None
    cursor_color: Optional[Tuple[int, int, int]] = None
    marker_color: Optional[Tuple[int, int, int]] = None
    #: font of the legend values (fixed-width digits do not jump when the value changes)
    value_font: str = ""
    state_fills: Optional[Sequence[Tuple[int, int, int]]] = None
    #: picture behind the curves, its opacity and fit: "width" (scene, anchored at the bottom) or "cover"
    backdrop: Optional[QtGui.QImage] = None
    backdrop_opacity: float = 0.0
    backdrop_fit: str = "width"
    #: legend rows share the whole legend height (classic look); otherwise rows are as high as the font needs
    legend_fill: bool = False
    #: without cursor: the range of each signal in the value column (off: the column stays empty unless the
    #: mouse is over the legend, see ``legend_hover``)
    legend_idle_ranges: bool = True
    #: the mouse is over the legend: the value column shows each signal's range ("0 - 150 [15/div]")
    legend_hover: bool = False

    @property
    def dark(self) -> bool:
        r, g, b = self.background
        return 0.299 * r + 0.587 * g + 0.114 * b < 128


@dataclass
class Layout:
    digital: QtCore.QRectF
    analog: QtCore.QRectF
    legend: QtCore.QRectF
    row_height: float
    name_width: float
    font_px: int


def legend_font(state: "ViewState", row_height: float) -> QtGui.QFont:
    font = QtGui.QFont(state.font_family)
    font.setPixelSize(int(max(7, min(state.font_size, row_height - 1))))
    return font


#: where the name starts in a row of the chip legend: after the colour mark
_CHIP_NAME_LEFT = 26.0


def legend_row_shortened(sig: DisplaySignal, value: str, lay: "Layout", state: "ViewState") -> bool:
    """Whether the legend draws the name of ``sig`` or its ``value`` text shortened, measured with the room
    and the fonts the legend is painted with."""
    font = legend_font(state, lay.row_height)
    chip = state.legend_style == "chip"
    name_room = lay.name_width - (_CHIP_NAME_LEFT if chip else 0.0) - 4
    value_room = lay.legend.width() - lay.name_width - 5
    names = QtGui.QFontMetricsF(font)
    values = QtGui.QFontMetricsF(_value_font(state, font)) if chip else names
    return (names.elidedText(sig.label, QtCore.Qt.ElideRight, name_room) != sig.label
            or values.elidedText(value, QtCore.Qt.ElideRight, value_room) != value)


def compute_layout(rect: QtCore.QRectF, signals: Sequence[DisplaySignal], state: ViewState) -> Layout:
    """Split ``rect`` into DIGITAL window, analog window and legend."""
    gap = 3.0
    plot_w = max(50.0, rect.width() * state.legend_fraction - gap)
    dig_h = max(0.0, rect.height() * state.digital_fraction)
    lanes = sum(1 for s in signals if s.lane) + len(state.rule_lanes)
    if lanes:
        # State bands need room for their labels: the upper window grows (at most to 60 %).
        bits = sum(1 for s in signals if s.digital)
        needed = lanes * LANE_HEIGHT + bits * BIT_LANE_HEIGHT + gap + 4.0
        dig_h = min(max(dig_h, needed), rect.height() * 0.6)
    digital = QtCore.QRectF(rect.left(), rect.top(), plot_w, max(0.0, dig_h - gap))
    analog = QtCore.QRectF(rect.left(), rect.top() + dig_h, plot_w, rect.height() - dig_h)
    legend = QtCore.QRectF(rect.left() + plot_w + gap, rect.top(), rect.width() - plot_w - gap, rect.height())
    rows = max(1, len(signals))
    row_h = legend.height() / rows if state.legend_fill else min(float(state.font_size + 9), legend.height() / rows)
    font = legend_font(state, row_h)
    metrics = QtGui.QFontMetricsF(font)
    widest = max((metrics.horizontalAdvance(s.label) for s in signals), default=40.0) + 8.0
    if state.legend_style == "chip":
        widest += _CHIP_NAME_LEFT                      # the name starts after the colour mark
    name_w = min(widest, legend.width() * 0.66)
    return Layout(digital, analog, legend, row_h, name_w, font.pixelSize())


def step_polyline(t: np.ndarray, y: np.ndarray, x0: float, x1: float, width_px: float,
                  step: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Points of the curve of a signal inside the time window ``[x0, x1]``.

    One sample before and after the window is included so the curve reaches
    the edges. When there are far more samples than pixels, each pixel column
    is summarized. For a step signal, keep exact transition times when there
    are few enough transitions and mark unresolved changes within a column
    vertically. Horizontal connections between columns retain the held state.
    """
    n = len(t)
    if n == 0:
        return np.empty(0), np.empty(0)
    i0 = max(0, int(np.searchsorted(t, x0, side="right")) - 1)
    i1 = min(n, int(np.searchsorted(t, x1, side="left")) + 1)
    tt = np.asarray(t[i0:i1], dtype=np.float64)
    yy = np.nan_to_num(np.asarray(y[i0:i1], dtype=np.float64))
    m = len(tt)
    columns = max(1, int(width_px))
    if m > 4 * columns:
        if step:
            return _dense_steps(tt, yy, x0, x1, columns)
        col = np.clip(((tt - x0) / (x1 - x0) * columns).astype(np.int64), 0, columns - 1)
        starts = np.flatnonzero(np.diff(col, prepend=col[0] - 1))
        lo = np.minimum.reduceat(yy, starts)
        hi = np.maximum.reduceat(yy, starts)
        first = yy[starts]
        last = yy[np.append(starts[1:], m) - 1]
        xs = np.repeat(tt[starts], 4)
        ys = np.column_stack((first, lo, hi, last)).ravel()
        return xs, ys
    if step and m > 1:
        xs = np.repeat(tt, 2)[1:]
        ys = np.repeat(yy, 2)[:-1]
        if i1 == n and tt[-1] < x1:            # hold the last sample to the right edge
            xs, ys = np.append(xs, x1), np.append(ys, yy[-1])
        return xs, ys
    if m == 1:                                  # a single sample is held across the window
        return np.array([tt[0], max(x1, tt[0])]), np.array([yy[0], yy[0]])
    return tt, yy


def _dense_steps(t: np.ndarray, y: np.ndarray, x0: float, x1: float,
                 columns: int) -> Tuple[np.ndarray, np.ndarray]:
    """Drop redundant samples first. Compress only when changes are also dense."""
    changes = np.flatnonzero(y[1:] != y[:-1]) + 1
    changes = changes[t[changes] <= x1]
    times, values = t[changes], y[changes]
    last_value = values[-1] if len(values) else y[0]
    if len(changes) <= 2 * columns:
        xs = np.repeat(times, 2)
        ys = np.column_stack((y[changes - 1], values)).ravel()
    else:
        col = np.clip(((times - x0) / (x1 - x0) * columns).astype(np.int64), 0, columns - 1)
        starts = np.flatnonzero(np.diff(col, prepend=-1))
        ends = np.append(starts[1:], len(changes)) - 1
        before, after = y[changes[starts] - 1], values[ends]
        low = np.minimum(before, np.minimum.reduceat(values, starts))
        high = np.maximum(before, np.maximum.reduceat(values, starts))
        rising = before <= after
        xs = np.repeat(times[starts], 4)
        ys = np.column_stack((before, np.where(rising, low, high), np.where(rising, high, low), after)).ravel()
    return (np.concatenate(([t[0]], xs, [max(t[0], x1)])),
            np.concatenate(([y[0]], ys, [last_value])))


def state_segments(t: np.ndarray, y: np.ndarray, x0: float, x1: float,
                   width_px: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segments of constant value of a step signal inside ``[x0, x1]``.

    Returns ``(start, end, value, busy)``: times of each segment, its value
    (NaN = no value) and whether the segment stands for a stretch with more
    changes than pixels (drawn as "many changes"). With far more segments
    than pixel columns the signal is sampled per column first, so the number
    of segments never exceeds the window width.
    """
    n = len(t)
    empty = (np.empty(0), np.empty(0), np.empty(0), np.zeros(0, dtype=bool))
    if n == 0 or not x1 > x0:
        return empty
    i0 = max(0, int(np.searchsorted(t, x0, side="right")) - 1)
    i1 = min(n, int(np.searchsorted(t, x1, side="left")) + 1)
    tt = np.asarray(t[i0:i1], dtype=np.float64)
    yy = np.asarray(y[i0:i1], dtype=np.float64)
    if len(tt) == 0:
        return empty
    key = np.where(np.isnan(yy), np.inf, yy)           # NaN never equals itself: give it one key
    change = np.flatnonzero(key[1:] != key[:-1]) + 1
    columns = max(1, int(width_px))
    if len(change) > 2 * columns:
        edges = np.linspace(x0, x1, columns + 1)
        pos = np.clip(np.searchsorted(tt, edges[:-1], side="right") - 1, 0, len(tt) - 1)
        values = key[pos]
        ends = np.clip(np.searchsorted(tt, edges[1:], side="right") - 1, 0, len(tt) - 1)
        # a column is busy when its value changes inside it
        cum = np.concatenate(([0], np.cumsum(key[1:] != key[:-1])))
        busy = cum[ends] != cum[pos]
        token = np.where(busy, np.nan, values)
        diff = np.ones(columns, dtype=bool)
        diff[1:] = ~((token[1:] == token[:-1]) | (np.isnan(token[1:]) & np.isnan(token[:-1])))
        starts = np.flatnonzero(diff)
        seg_start = edges[starts]
        seg_end = np.append(edges[starts[1:]], x1)
        seg_value = values[starts]
        seg_busy = busy[starts]
    else:
        starts = np.concatenate(([0], change))
        seg_start = np.maximum(tt[starts], x0)
        seg_end = np.minimum(np.append(tt[change], max(x1, tt[-1])), x1)
        seg_value = key[starts]
        seg_busy = np.zeros(len(starts), dtype=bool)
    seg_value = np.where(np.isinf(seg_value) & (seg_value > 0), np.nan, seg_value)
    keep = seg_end > seg_start
    return seg_start[keep], seg_end[keep], seg_value[keep], seg_busy[keep]


def state_fill(sig: DisplaySignal, value: float, fills: Optional[Sequence[Tuple[int, int, int]]] = None
               ) -> Tuple[int, int, int]:
    """Fill colour of one state: stable per value (by its place in the text table, if any)."""
    fills = fills or STATE_FILLS
    if value == 0:
        return _STATE_ZERO if fills is STATE_FILLS else _neutral(fills)
    keys = sorted(sig.enum.values) if sig.enum and sig.enum.values else None
    if keys is not None and value in sig.enum.values:
        rank = keys.index(value)
    else:
        rank = int(abs(hash(round(float(value), 9))))
    return tuple(fills[rank % len(fills)])  # type: ignore[return-value]


def _neutral(fills: Sequence[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """The "zero" state of a themed band: a grey of the average lightness of the theme's fills."""
    level = int(sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in fills) / len(fills))
    return (level, level, level)


def _paint_state_lane(painter: QtGui.QPainter, sig: DisplaySignal, rect: QtCore.QRectF, state: "ViewState",
                      font: QtGui.QFont) -> None:
    span = state.x1 - state.x0
    stop = sig.valid_until() if state.stopped != "hold" else float("inf")
    x_end = min(state.x1, stop)
    if x_end > state.x0:
        start, end, value, busy = state_segments(sig.t, sig.y, state.x0, x_end,
                                                 rect.width() * (x_end - state.x0) / span)
    else:
        start = end = value = np.empty(0)
        busy = np.zeros(0, dtype=bool)
    if rect.height() <= 3.0:
        return
    row = rect.adjusted(0.0, 1.0, 0.0, -1.0)
    metrics = QtGui.QFontMetricsF(font)
    painter.setFont(font)
    edge = QtGui.QColor(112, 92, 83, 100) if not state.dark else QtGui.QColor(214, 179, 158, 78)
    changes = QtGui.QColor(167, 130, 113, 90) if not state.dark else QtGui.QColor(206, 159, 134, 80)
    for a, b, v, many in zip(start, end, value, busy):
        xa = rect.left() + (a - state.x0) / span * rect.width()
        xb = rect.left() + (b - state.x0) / span * rect.width()
        if xb - xa <= 0:
            continue
        box = QtCore.QRectF(xa, row.top(), xb - xa, row.height())
        if many:
            painter.fillRect(box, changes)
            continue
        if v != v:                                      # NaN: no value here
            continue
        fill = state_fill(sig, float(v), state.state_fills)
        painter.fillRect(box, QtGui.QColor(*fill))
        painter.setPen(edge)
        if a > state.x0:
            painter.drawLine(box.topLeft(), box.bottomLeft())
        text_box = box.adjusted(4, 0, -4, 0)
        spec = sig.image_for(float(v)) if box.width() > box.height() * 1.6 else None
        alignment = QtCore.Qt.AlignHCenter
        if spec:
            from ..signs import sign_pixmap
            pix = sign_pixmap(spec, max(8, int(box.height() - 2)))
            if pix is not None:
                painter.drawPixmap(QtCore.QPointF(box.left() + 2, box.top() + (box.height() - pix.height()) / 2), pix)
                text_box = text_box.adjusted(pix.width() + 2, 0, 0, 0)
                alignment = QtCore.Qt.AlignLeft
        if text_box.width() > 10 and box.height() >= metrics.height():
            text = metrics.elidedText(sig.state_text(float(v)), QtCore.Qt.ElideRight, text_box.width() - 4)
            if text and text != "…":
                painter.setPen(QtGui.QColor(*text_color_on(fill)))
                painter.drawText(text_box, int(QtCore.Qt.AlignVCenter | alignment), text)
    if state.stopped == "dashed" and stop < state.x1:
        xa = rect.left() + (max(stop, state.x0) - state.x0) / span * rect.width()
        box = QtCore.QRectF(xa, row.top(), row.right() - xa, row.height())
        if box.width() > 1:
            last = float(np.asarray(sig.y, dtype=np.float64)[-1])
            fill = QtGui.QColor(*state_fill(sig, last, state.state_fills)) if last == last else QtGui.QColor(160, 160, 160)
            fill.setAlpha(70)
            painter.fillRect(box, fill)
            painter.setPen(_dash_pen(edge, 1.0))
            painter.drawRect(box.adjusted(0.5, 0.5, -0.5, -0.5))
            if box.width() > 30:
                text = metrics.elidedText(tr("no new data"), QtCore.Qt.ElideRight, box.width() - 8)
                painter.setPen(QtGui.QColor(110, 110, 110) if not state.dark else QtGui.QColor(190, 190, 190))
                painter.drawText(box.adjusted(4, 0, -2, 0), int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft), text)
    strip = QtCore.QRectF(row.left(), row.top(), 2.0, row.height())
    painter.fillRect(strip, QtGui.QColor(*signal_color(sig.color, state)))


#: colour of a verdict in rule lanes and tables
VERDICT_COLORS = {"PASS": (40, 160, 70), "FAIL": (210, 40, 40), "TOO_EARLY": (230, 140, 0),
                  "INCONCLUSIVE": (150, 150, 150), "FOUND": (40, 110, 220)}


def _paint_rule_lane(painter: QtGui.QPainter, rect: QtCore.QRectF, name: str, events, state: "ViewState",
                     font: QtGui.QFont) -> None:
    span = state.x1 - state.x0
    shade = 0.06 if not state.dark else 0.12
    bg = state.background
    tint = tuple(int(c + ((0 if not state.dark else 255) - c) * shade) for c in bg)
    painter.fillRect(rect, QtGui.QColor(*tint))
    painter.setFont(font)
    metrics = QtGui.QFontMetricsF(font)
    label = metrics.elidedText("▶ " + name, QtCore.Qt.ElideRight, rect.width() * 0.3)
    painter.setPen(QtGui.QColor(120, 120, 120))
    painter.drawText(rect.adjusted(4, 0, 0, 0), int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft), label)
    mid = rect.center().y()
    size = max(4.0, rect.height() * 0.32)
    for t0, t1, verdict in events:
        if t0 > state.x1 or (t1 if t1 is not None else t0) < state.x0:
            continue
        color = QtGui.QColor(*VERDICT_COLORS.get(verdict, (100, 100, 100)))
        x0 = rect.left() + (t0 - state.x0) / span * rect.width()
        if t1 is not None and t1 > t0:
            x1 = rect.left() + (t1 - state.x0) / span * rect.width()
            bar = QtGui.QColor(color)
            bar.setAlpha(110)
            painter.fillRect(QtCore.QRectF(x0, mid - size * 0.35, max(1.0, x1 - x0), size * 0.7), bar)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x0 - size, mid - size), QtCore.QPointF(x0 + size, mid - size),
                                             QtCore.QPointF(x0, mid + size)]))
    painter.setBrush(QtCore.Qt.NoBrush)


def xy_axis_index(signals: Sequence[DisplaySignal]) -> int:
    """Index of the signal that forms the X axis of the XY view (first analog signal with data), or -1."""
    return next((i for i, s in enumerate(signals) if s.in_analog and s.has_data), -1)


def xy_curve(xsig: DisplaySignal, ysig: DisplaySignal, t0: float, t1: float,
             max_points: int = 60000) -> Tuple[np.ndarray, np.ndarray]:
    """Values of ``xsig`` and ``ysig`` at the union of their time stamps inside ``[t0, t1]``.

    Both are step functions, so each holds its last sample. Very long curves
    are thinned out evenly to ``max_points``.
    """
    if not (xsig.has_data and ysig.has_data):
        return np.empty(0), np.empty(0)

    def window(sig: DisplaySignal) -> np.ndarray:
        t = np.asarray(sig.t, dtype=np.float64)
        return t[(t >= t0) & (t <= t1)]

    times = np.union1d(window(xsig), window(ysig))
    if len(times) == 0:
        return np.empty(0), np.empty(0)
    if len(times) > max_points:
        times = times[np.linspace(0, len(times) - 1, max_points).astype(np.int64)]

    def held(sig: DisplaySignal) -> np.ndarray:
        idx = np.clip(np.searchsorted(sig.t, times, side="right") - 1, 0, len(sig.t) - 1)
        return np.asarray(sig.y, dtype=np.float64)[idx]

    return held(xsig), held(ysig)


def xy_pixels(lay: Layout, xsig: DisplaySignal, sig: DisplaySignal, state: "ViewState"
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel positions of the XY curve of ``sig`` over ``xsig`` and the time of every point."""
    xs, ys = xy_curve(xsig, sig, state.x0, state.x1)
    if len(xs) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    t = np.asarray(sig.t, dtype=np.float64)
    tx = np.asarray(xsig.t, dtype=np.float64)
    times = np.union1d(tx[(tx >= state.x0) & (tx <= state.x1)], t[(t >= state.x0) & (t <= state.x1)])
    if len(times) != len(xs):                                   # thinned out: same even selection
        times = times[np.linspace(0, len(times) - 1, len(xs)).astype(np.int64)]
    xspan = (xsig.scmax - xsig.scmin) or 1.0
    yspan = max(1e-12, state.y1 - state.y0)
    rng = (sig.scmax - sig.scmin) or 1.0
    px = np.clip(lay.analog.left() + (np.nan_to_num(xs) - xsig.scmin) / xspan * lay.analog.width(), -1e6, 1e6)
    norm = ((np.nan_to_num(ys) - sig.scmin) / rng - state.y0) / yspan
    py = np.clip(lay.analog.bottom() - norm * lay.analog.height(), -1e6, 1e6)
    return px, py, times


def _blend(a: Tuple[int, int, int], b: Tuple[int, int, int], f: float) -> QtGui.QColor:
    return QtGui.QColor(*(int(round(a[i] + (b[i] - a[i]) * f)) for i in range(3)))


def _paint_xy(painter: QtGui.QPainter, lay: Layout, signals: Sequence[DisplaySignal], state: "ViewState") -> None:
    xi = xy_axis_index(signals)
    if xi < 0:
        return
    xsig = signals[xi]
    painter.save()
    painter.setClipRect(lay.analog)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    for index, sig in enumerate(signals):
        if index == xi or not sig.in_analog or not sig.has_data or sig.color == HIDDEN_COLOR:
            continue
        if suppressed(index, sig, state):
            continue
        px, py, _times = xy_pixels(lay, xsig, sig, state)
        if len(px) < 1:
            continue
        rgb = signal_color(sig.color, state)
        width = state.line_width + (1.0 if index in state.emphasized else 0.0)
        # time colouring: chunks from a light tint (early) to the full colour (late)
        chunks = 24 if state.xy_time_color and len(px) > 24 else 1
        edges = np.linspace(0, len(px), chunks + 1).astype(np.int64)
        light = tuple(int(c + (bg - c) * 0.72) for c, bg in zip(rgb, state.background))
        for k in range(chunks):
            a, b = edges[k], min(len(px), edges[k + 1] + (0 if state.xy_points else 1))
            if b - a < 1:
                continue
            color = _blend(light, rgb, (k + 1) / chunks) if chunks > 1 else QtGui.QColor(*rgb)
            if state.xy_points or b - a == 1:
                pen = QtGui.QPen(color, max(2.5, width + 1.5))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                painter.setPen(pen)
                painter.drawPoints(_polygon(px[a:b], py[a:b]))
            else:
                painter.setPen(_pen(color, width))
                painter.drawPolyline(_polygon(px[a:b], py[a:b]))
        if state.xy_time_color and len(px) > 1:                 # start: open ring, end: filled dot
            color = QtGui.QColor(*rgb)
            painter.setPen(QtGui.QPen(color, 2.0))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(px[0], py[0]), 6.0, 6.0)
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QPointF(px[-1], py[-1]), 5.0, 5.0)
            painter.setBrush(QtCore.Qt.NoBrush)
    painter.restore()


def _polygon(xs: np.ndarray, ys: np.ndarray) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in zip(xs, ys)])


def range_text(sig: DisplaySignal) -> str:
    """Legend text without a cursor: view range and value per grid division."""
    return c_exponent("%.3g - %.3g [%.3g/div]" % (sig.scmin, sig.scmax, (sig.scmax - sig.scmin) / 10.0))


def legend_value_text(sig: DisplaySignal, state: "ViewState") -> str:
    """Right column of a legend row: problem, value at the cursor, or the view range."""
    if sig.error:
        return sig.error
    if state.cursor is not None:
        if state.stopped != "hold" and sig.has_data and sig.stopped_at(state.cursor):
            if state.stopped == "cut":
                return tr("no data")
            return sig.format_value(sig.value_at(state.cursor), state.value_format) + "  " + tr("(no new data)")
        return sig.format_value(sig.value_at(state.cursor), state.value_format)
    if sig.lane:
        return tr("state band") + (" - " + sig.map_name if sig.map_name else "")
    return range_text(sig) if (state.legend_idle_ranges or state.legend_hover) else ""


def signal_color(index: int, state: ViewState) -> Tuple[int, int, int]:
    """Screen colour of palette entry ``index`` on the current background.

    On a light background the very bright palette colours (yellow, cyan,
    light green) are darkened so that thin lines stay readable.
    """
    r, g, b = color_rgb(index, state.palette, dark_background=state.dark, invert_bw=state.invert_bw)
    if not state.dark:
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance > 150 and (r, g, b) != tuple(state.background):
            k = 0.55 if luminance > 200 else 0.7
            r, g, b = int(r * k), int(g * k), int(b * k)
    return r, g, b


def _paint_backdrop(painter: QtGui.QPainter, area: QtCore.QRectF, state: "ViewState") -> None:
    """The picture behind the curves, faded to ``backdrop_opacity``."""
    image = state.backdrop
    ratio = image.devicePixelRatio() or 1.0
    iw, ih = image.width() / ratio, image.height() / ratio
    if iw <= 0 or ih <= 0 or area.width() <= 0 or area.height() <= 0:
        return
    if state.backdrop_fit == "cover":
        scale = max(area.width() / iw, area.height() / ih)
        w, h = iw * scale, ih * scale
        target = QtCore.QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)
    else:                                                  # a wide scene: full width, standing on the bottom
        scale = min(area.width() / iw, area.height() / ih * 1.6)
        w, h = iw * scale, ih * scale
        target = QtCore.QRectF(area.right() - w, area.bottom() - h, w, h)
    painter.save()
    painter.setClipRect(area)
    painter.setOpacity(max(0.0, min(1.0, state.backdrop_opacity)))
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
    painter.drawImage(target, image, QtCore.QRectF(image.rect()))
    painter.restore()


def _curve_parts(sig: DisplaySignal, state: "ViewState", width_px: float, step: bool):
    """The curve inside the window and, for a signal that stopped being sent, the part after its end:
    ``((xs, ys), tail)`` with ``tail = (xs, ys)`` of the held last value (drawn dashed) or ``None``."""
    end = sig.valid_until() if state.stopped != "hold" else float("inf")
    if end >= state.x1:
        return step_polyline(sig.t, sig.y, state.x0, state.x1, width_px, step), None
    if end > state.x0:
        part = step_polyline(sig.t, sig.y, state.x0, end, width_px * (end - state.x0) / (state.x1 - state.x0), step)
    else:
        part = (np.empty(0), np.empty(0))
    if state.stopped != "dashed":
        return part, None
    last = float(np.nan_to_num(np.asarray(sig.y, dtype=np.float64)[-1]))
    return part, (np.array([max(end, state.x0), state.x1]), np.array([last, last]))


def _dash_pen(color: QtGui.QColor, width: float) -> QtGui.QPen:
    pen = _pen(color, width, QtCore.Qt.CustomDashLine)
    pen.setDashPattern([4.0, 4.0] if width <= 1.0 else [3.0, 3.0])
    return pen


def _pen(color: QtGui.QColor, width: float, style=QtCore.Qt.SolidLine) -> QtGui.QPen:
    pen = QtGui.QPen(color, width if width > 1.0 else 0.0, style)
    pen.setCosmetic(width <= 1.0)
    pen.setJoinStyle(QtCore.Qt.BevelJoin)
    return pen


def suppressed(index: int, sig: DisplaySignal, state: "ViewState") -> bool:
    """True while signals are emphasized and this curve of the analog window is not one of them: it is not
    drawn and its legend name is struck out. A selected signal is no exception (a click only selects, H shows
    it); the signal reached with TAB / the wheel is shown for the moment. Upper-window rows are always shown."""
    return (bool(state.emphasized) and index not in state.emphasized and index != state.temporary
            and sig.in_analog)


def upper_rows(lay: Layout, signals: Sequence[DisplaySignal], state: "ViewState") -> List[Tuple[int, QtCore.QRectF]]:
    """Rows of the upper (DIGITAL) window from top to bottom: ``(signal index, rect)`` for every one-bit signal
    and state band, then ``(-1, rect)`` for every rule lane. Empty when there is nothing or no room."""
    upper = [i for i, s in enumerate(signals) if s.digital or s.lane]
    lanes = len(state.rule_lanes)
    if not (upper or lanes) or lay.digital.height() <= 4:
        return []
    if lanes or any(signals[i].lane for i in upper):
        weights = [LANE_HEIGHT if signals[i].lane else BIT_LANE_HEIGHT for i in upper] + [LANE_HEIGHT] * lanes
        k = min(1.4, lay.digital.height() / max(1.0, sum(weights)))
        heights = [w * k for w in weights]
    else:
        heights = [min(24.0, lay.digital.height() / len(upper))] * len(upper)
    rows: List[Tuple[int, QtCore.QRectF]] = []
    top = lay.digital.top()
    for index, height in zip(upper + [-1] * lanes, heights):
        rows.append((index, QtCore.QRectF(lay.digital.left(), top, lay.digital.width(), height)))
        top += height
    return rows


def bit_y(rect: QtCore.QRectF, values):
    """Pixel height of one-bit values in a row of the upper window: 0 near the bottom, 1 near the top (other
    values are clipped to that range)."""
    return rect.bottom() - 2.0 - np.clip(values, 0.0, 1.0) * max(2.0, rect.height() - 5.0)


def curve_distance(lay: Layout, sig: DisplaySignal, state: "ViewState", x: float, y: float) -> float:
    """Distance in pixels from the point (x, y) to the curve of ``sig`` as the analog window draws it (with the
    dashed part after the signal stopped); infinity when no part of the curve is near the point."""
    span = state.x1 - state.x0
    if not sig.has_data or span <= 0:
        return float("inf")
    (xs, ys), tail = _curve_parts(sig, state, lay.analog.width(), state.step_mode)
    rng = sig.scmax - sig.scmin or 1.0
    yspan = max(1e-12, state.y1 - state.y0)
    best = float("inf")
    for tx, ty in [(xs, ys)] + ([tail] if tail is not None else []):
        if len(tx) == 0:
            continue
        px = lay.analog.left() + (np.asarray(tx, dtype=np.float64) - state.x0) / span * lay.analog.width()
        py = lay.analog.bottom() - ((np.asarray(ty, dtype=np.float64) - sig.scmin) / rng - state.y0) / yspan \
            * lay.analog.height()
        if len(px) == 1:
            d = np.hypot(px - x, py - y)
        else:
            lo = max(0, int(np.searchsorted(px, x - 8.0)) - 1)            # only the segments near the point
            hi = min(len(px) - 1, int(np.searchsorted(px, x + 8.0, side="right")))
            if hi <= lo:
                continue
            ax, ay, bx, by = px[lo:hi], py[lo:hi], px[lo + 1:hi + 1], py[lo + 1:hi + 1]
            dx, dy = bx - ax, by - ay
            length = dx * dx + dy * dy
            f = np.clip(((x - ax) * dx + (y - ay) * dy) / np.where(length > 0, length, 1.0), 0.0, 1.0)
            d = np.hypot(ax + f * dx - x, ay + f * dy - y)
        d = d[np.isfinite(d)]
        if len(d):
            best = min(best, float(d.min()))
    return best


def paint_plot(painter: QtGui.QPainter, lay: Layout, signals: Sequence[DisplaySignal], state: ViewState) -> None:
    """Background, grid and all curves (the part that only changes with data or view)."""
    bg = QtGui.QColor(*state.background)
    fg = QtGui.QColor(255, 255, 255) if state.dark else QtGui.QColor(40, 40, 40)
    whole = lay.digital.united(lay.analog)
    painter.fillRect(whole, bg)
    span = state.x1 - state.x0
    if span <= 0:
        return
    if state.backdrop is not None and state.backdrop_opacity > 0 and not state.backdrop.isNull():
        _paint_backdrop(painter, lay.analog, state)
    if state.grid_color is not None:
        grid_color = QtGui.QColor(*state.grid_color)
        grid_style = QtCore.Qt.SolidLine
    else:
        grid_color = QtGui.QColor(128, 128, 128) if state.dark else QtGui.QColor(140, 150, 140)
        grid_style = QtCore.Qt.DotLine
    label_color = QtGui.QColor(*state.axis_color) if state.axis_color is not None else grid_color
    if state.grid:
        painter.setPen(_pen(grid_color, 1.0, grid_style))
        for k in range(1, 10):
            x = lay.analog.left() + lay.analog.width() * k / 10.0
            painter.drawLine(QtCore.QPointF(x, lay.digital.top()), QtCore.QPointF(x, lay.analog.bottom()))
            y = lay.analog.top() + lay.analog.height() * k / 10.0
            painter.drawLine(QtCore.QPointF(lay.analog.left(), y), QtCore.QPointF(lay.analog.right(), y))
    if lay.digital.height() > 0:
        painter.setPen(_pen(fg, 1.0))
        painter.drawLine(lay.digital.bottomLeft(), lay.digital.bottomRight())

    def x_of(values: np.ndarray, area: QtCore.QRectF) -> np.ndarray:
        return area.left() + (values - state.x0) / span * area.width()

    if state.xy:
        _paint_xy(painter, lay, signals, state)
        xi = xy_axis_index(signals)
        if state.time_labels and lay.analog.height() > 60 and xi >= 0:
            xsig = signals[xi]
            font = QtGui.QFont(state.font_family)
            font.setPixelSize(10)
            painter.setFont(font)
            painter.setPen(label_color)
            for k in range(0, 10):
                x = lay.analog.left() + lay.analog.width() * k / 10.0
                value = xsig.scmin + (xsig.scmax - xsig.scmin) * k / 10.0
                painter.drawText(QtCore.QPointF(x + 3, lay.analog.bottom() - 3),
                                 c_exponent("%.4g %s" % (value, xsig.unit)).strip())
        return

    painter.save()
    painter.setClipRect(lay.analog)
    yspan = max(1e-12, state.y1 - state.y0)
    order = [i for i in range(len(signals)) if i not in state.emphasized and i != state.selected]
    order += [i for i in range(len(signals)) if i in state.emphasized and i != state.selected]
    if 0 <= state.selected < len(signals):
        order.append(state.selected)                       # the selected signal is drawn on top
    for index in order:
        sig = signals[index]
        if not sig.in_analog or not sig.has_data or sig.color == HIDDEN_COLOR:
            continue
        if suppressed(index, sig, state):
            continue
        (xs, ys), tail = _curve_parts(sig, state, lay.analog.width(), state.step_mode)
        rng = sig.scmax - sig.scmin or 1.0
        color = QtGui.QColor(*signal_color(sig.color, state))
        width = state.line_width + (1.0 if index in state.emphasized else 0.0)

        def py_of(values: np.ndarray) -> np.ndarray:
            norm = ((values - sig.scmin) / rng - state.y0) / yspan
            return np.clip(lay.analog.bottom() - norm * lay.analog.height(), -1e6, 1e6)

        if len(xs) >= 2:
            painter.setPen(_pen(color, width))
            painter.drawPolyline(_polygon(x_of(xs, lay.analog), py_of(ys)))
        if tail is not None:
            painter.setPen(_dash_pen(color, width))
            painter.drawPolyline(_polygon(x_of(tail[0], lay.analog), py_of(tail[1])))
        if sig.ref_lines:
            painter.setPen(_pen(color, 1.0, QtCore.Qt.DashLine))
            for ref in sig.ref_lines:
                yr = lay.analog.bottom() - (((ref - sig.scmin) / rng - state.y0) / yspan) * lay.analog.height()
                painter.drawLine(QtCore.QPointF(lay.analog.left(), yr), QtCore.QPointF(lay.analog.right(), yr))
    painter.restore()

    rows = upper_rows(lay, signals, state)
    if rows:
        painter.save()
        painter.setClipRect(lay.digital)
        lane_font = QtGui.QFont(state.font_family)
        lane_font.setPixelSize(int(max(8, min(12, min(rect.height() for _i, rect in rows) - 6))))
        for (name, events), rect in zip(state.rule_lanes, [rect for index, rect in rows if index < 0]):
            _paint_rule_lane(painter, rect, name, events, state, lane_font)
        for index, row_rect in rows:
            if index < 0:
                continue
            sig = signals[index]
            if not sig.has_data or sig.color == HIDDEN_COLOR:
                continue
            if sig.lane:
                _paint_state_lane(painter, sig, row_rect, state, lane_font)
                continue
            (xs, ys), tail = _curve_parts(sig, state, lay.digital.width(), True)
            color = QtGui.QColor(*signal_color(sig.color, state))
            width = state.line_width
            if len(xs) >= 2:
                painter.setPen(_pen(color, width))
                painter.drawPolyline(_polygon(x_of(xs, lay.digital), bit_y(row_rect, ys)))
            if tail is not None:
                painter.setPen(_dash_pen(color, width))
                painter.drawPolyline(_polygon(x_of(tail[0], lay.digital), bit_y(row_rect, tail[1])))
        painter.restore()

    if state.time_labels and lay.analog.height() > 60:
        font = QtGui.QFont(state.font_family)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(label_color)
        for k in range(0, 10):
            x = lay.analog.left() + lay.analog.width() * k / 10.0
            text = "%.6g s" % (state.x0 + span * k / 10.0)
            painter.drawText(QtCore.QPointF(x + 3, lay.analog.bottom() - 3), text)


def paint_legend(painter: QtGui.QPainter, lay: Layout, signals: Sequence[DisplaySignal], state: ViewState) -> None:
    """Legend rows: coloured name field and value (at the cursor) or view range."""
    if state.legend_style == "chip":
        _paint_legend_chips(painter, lay, signals, state)
        return
    bg = QtGui.QColor(*state.background)
    fg = QtGui.QColor(255, 255, 255) if state.dark else QtGui.QColor(30, 30, 30)
    painter.fillRect(lay.legend, bg)
    font = legend_font(state, lay.row_height)
    metrics = QtGui.QFontMetricsF(font)
    for index, sig in enumerate(signals):
        top = lay.legend.top() + index * lay.row_height
        if top > lay.legend.bottom():
            break
        name_rect = QtCore.QRectF(lay.legend.left(), top, lay.name_width, lay.row_height)
        value_rect = QtCore.QRectF(lay.legend.left() + lay.name_width + 5, top,
                                   lay.legend.width() - lay.name_width - 5, lay.row_height)
        hidden = sig.color == HIDDEN_COLOR
        rgb = color_rgb(sig.color, state.palette, dark_background=state.dark, invert_bw=state.invert_bw)
        selected = index == state.selected
        if selected and not state.flash:
            painter.fillRect(name_rect, bg)
            painter.setPen(fg)
        else:
            painter.fillRect(name_rect, bg if hidden else QtGui.QColor(*rgb))
            painter.setPen(fg if hidden else QtGui.QColor(*text_color_on(rgb)))
        f = QtGui.QFont(font)
        f.setBold(selected or index in state.emphasized)
        f.setStrikeOut(suppressed(index, sig, state))
        painter.setFont(f)
        name_metrics = QtGui.QFontMetricsF(f)
        label = name_metrics.elidedText(sig.label, QtCore.Qt.ElideRight, name_rect.width() - 4)
        painter.drawText(name_rect.adjusted(3, 0, -1, 0), int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft), label)
        painter.setFont(font)
        painter.setPen(fg)
        text = metrics.elidedText(legend_value_text(sig, state), QtCore.Qt.ElideRight, value_rect.width())
        painter.drawText(value_rect, int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft), text)


def _value_font(state: ViewState, font: QtGui.QFont) -> QtGui.QFont:
    if not state.value_font:
        return font
    f = QtGui.QFont(state.value_font)
    f.setPixelSize(max(7, font.pixelSize() - 1))
    f.setStyleHint(QtGui.QFont.Monospace)
    return f


def _paint_legend_chips(painter: QtGui.QPainter, lay: Layout, signals: Sequence[DisplaySignal],
                        state: ViewState) -> None:
    """Themed legend: a colour mark (line sample, or a square for bands and bits), the name in text colour,
    the value in a fixed-width font; the selected row is tinted."""
    bg = QtGui.QColor(*(state.legend_bg or state.background))
    fg = QtGui.QColor(*state.text_color) if state.text_color else \
        (QtGui.QColor(235, 235, 235) if state.dark else QtGui.QColor(30, 30, 30))
    dim = QtGui.QColor(fg)
    dim.setAlpha(150)
    painter.fillRect(lay.legend, bg)
    painter.setPen(QtGui.QPen(QtGui.QColor(fg.red(), fg.green(), fg.blue(), 40), 1.0))
    painter.drawLine(lay.legend.topLeft(), lay.legend.bottomLeft())
    font = legend_font(state, lay.row_height)
    vfont = _value_font(state, font)
    metrics, vmetrics = QtGui.QFontMetricsF(font), QtGui.QFontMetricsF(vfont)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    for index, sig in enumerate(signals):
        top = lay.legend.top() + index * lay.row_height
        if top > lay.legend.bottom():
            break
        row = QtCore.QRectF(lay.legend.left(), top, lay.legend.width(), lay.row_height)
        hidden = sig.color == HIDDEN_COLOR
        color = QtGui.QColor(*signal_color(sig.color, state))
        selected = index == state.selected
        if selected and state.flash:
            tint = QtGui.QColor(color)
            tint.setAlpha(60)
            painter.fillRect(row.adjusted(1, 1, -1, -1), tint)
        h = lay.row_height
        mark = QtCore.QRectF(row.left() + 6, top + h * 0.5 - 1.6, 14, 3.2)
        if sig.lane or sig.digital:
            side = max(4.0, min(10.0, h * 0.42))
            mark = QtCore.QRectF(row.left() + 6 + (14 - side) / 2, top + (h - side) / 2, side, side)
        if hidden:
            painter.setPen(QtGui.QPen(dim, 1.0))
            painter.setBrush(QtCore.Qt.NoBrush)
        else:
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
        painter.drawRoundedRect(mark, 1.6, 1.6)
        painter.setBrush(QtCore.Qt.NoBrush)
        f = QtGui.QFont(font)
        f.setBold(selected or index in state.emphasized)
        f.setStrikeOut(suppressed(index, sig, state))
        painter.setFont(f)
        painter.setPen(dim if hidden else fg)
        name_rect = QtCore.QRectF(row.left() + _CHIP_NAME_LEFT, top, lay.name_width - _CHIP_NAME_LEFT, h)
        label = QtGui.QFontMetricsF(f).elidedText(sig.label, QtCore.Qt.ElideRight, name_rect.width() - 4)
        painter.drawText(name_rect, int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft), label)
        value_rect = QtCore.QRectF(row.left() + lay.name_width + 5, top, row.width() - lay.name_width - 5, h)
        text = legend_value_text(sig, state)
        painter.setFont(vfont)
        painter.setPen(fg if state.cursor is not None else dim)
        painter.drawText(value_rect, int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft),
                         vmetrics.elidedText(text, QtCore.Qt.ElideRight, value_rect.width()))
    painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
    del metrics


def paint_view(painter: QtGui.QPainter, rect: QtCore.QRectF, signals: Sequence[DisplaySignal],
               state: ViewState) -> Layout:
    """Paint plot windows, cursor and legend into ``rect``. Returns the layout used."""
    lay = compute_layout(rect, signals, state)
    painter.fillRect(rect, QtGui.QColor(*state.background))
    paint_plot(painter, lay, signals, state)
    span = state.x1 - state.x0
    if state.marker is not None and span > 0 and state.x0 <= state.marker <= state.x1 and not state.xy:
        xm = lay.analog.left() + (state.marker - state.x0) / span * lay.analog.width()
        painter.setPen(_pen(QtGui.QColor(*(state.marker_color or (240, 140, 0))), 2.0))
        painter.drawLine(QtCore.QPointF(xm, lay.digital.top()), QtCore.QPointF(xm, lay.analog.bottom()))
    if state.cursor is not None and span > 0 and state.x0 <= state.cursor <= state.x1:
        xc = lay.analog.left() + (state.cursor - state.x0) / span * lay.analog.width()
        fg = QtGui.QColor(*state.cursor_color) if state.cursor_color else \
            (QtGui.QColor(255, 255, 255) if state.dark else QtGui.QColor(30, 30, 30))
        painter.setPen(_pen(fg, 1.0, QtCore.Qt.DashLine))
        painter.drawLine(QtCore.QPointF(xc, lay.digital.top()), QtCore.QPointF(xc, lay.analog.bottom()))
    paint_legend(painter, lay, signals, state)
    return lay
