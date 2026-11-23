"""The interactive measurement view: DIGITAL window, analog window and legend.

Mouse and keyboard bindings:

==============  ===============================================================
``X``           reference line on/off (a vertical line follows the mouse and the
                legend shows the values at that time)
``U`` ``Z`` ``Y``  zoom: drag a frame (both directions / time only / values only)
``-``           undo the last zoom          ``SPACE``  show everything again
``N`` ``P``     next / previous page (starting / ending at the mouse position)
``L R T B``     mark left/right/top/bottom limit at the mouse, ``RETURN`` applies
``TAB``, wheel  select the next / previous signal
``LEFT/RIGHT``  previous / next sample of the selected signal in the time view
``H``           emphasize the signal under the mouse (legend) or the selected one;
                without a selection: select the curve under the mouse
``ESC``         leave zoom / measurement / selection / emphasis
click legend    select a signal (its row flashes, a cross hair follows its value, in
                the DIGITAL window too); while signals are emphasized a selected
                signal is shown only after ``H``
click plot      with reference line or selection: start / stop a difference
                measurement (dX, dY and dY/dX appear in the status bar)
right click     plot: cancel; legend row: edit the signal
==============  ===============================================================

Additions: drag in the plot zooms in time; ``Ctrl`` + wheel zooms around the
mouse; ``Shift`` + wheel scrolls in time; the dividers between the windows can
be dragged.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..display import DisplaySignal, format_number
import numpy as np

from ..plt.palette import HIDDEN_COLOR
from .render import (Layout, ViewState, bit_y, compute_layout, curve_distance, legend_row_shortened,
                     legend_value_text, paint_legend, paint_plot, paint_view, suppressed, upper_rows, xy_axis_index,
                     xy_pixels)

__all__ = ["PlotView"]

_MIN_SPAN = 1e-6


def _physical_pointer() -> Optional[QtCore.QPoint]:
    """Read the native screen position where Qt's Windows coordinates can round at high DPI."""
    if sys.platform != "win32" or QtGui.QGuiApplication.platformName() != "windows":
        return None
    import ctypes
    from ctypes import wintypes

    position = wintypes.POINT()
    if ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(position)):
        return QtCore.QPoint(position.x, position.y)
    return None


class PlotView(QtWidgets.QWidget):
    """Widget showing the signals of the current script."""

    #: Texts for the status bar: keys ``x``, ``y``, ``dydx``, ``xrange``, ``yrange``, ``hint``.
    statusChanged = QtCore.Signal(dict)
    selectionChanged = QtCore.Signal(int)
    #: Right click or double click on a legend row (index) - the main window opens the editor on that signal.
    editRequested = QtCore.Signal(int)
    #: A key that the main window handles: ``"edit"``, ``"reload"``, ``"info"``, ``"copy_values"``, ``"statistics"``.
    commandRequested = QtCore.Signal(str)
    viewChanged = QtCore.Signal()
    #: Emitted after a divider was dragged: (legend_fraction, digital_fraction).
    dividersChanged = QtCore.Signal(float, float)
    #: the view range of signals was changed with the keyboard (Up / Down, Shift+Up / Shift+Down)
    rangesChanged = QtCore.Signal()
    #: The time the legend shows values for changed (cursor or marker); ``None`` when there is none.
    cursorTimeChanged = QtCore.Signal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.signals: List[DisplaySignal] = []
        self.state = ViewState()
        self._has_recording = False
        self.full: Tuple[float, float] = (0.0, 1.0)
        self.reference_line = False
        #: the legend shows the values under the mouse whenever it is over the plot (classic look)
        self.hover_values = False
        self._zoom_stack: List[Tuple[float, float, float, float]] = []
        self._mode = ""                       # "", "x", "y", "xy": the next drag zooms
        #: mouse pointers of the theme: "zoom" (zoom drag), "pointer" (normal); empty = system pointers
        self.theme_cursors: Dict[str, QtGui.QCursor] = {}
        self._drag: Optional[Dict[str, object]] = None
        self._mouse: Optional[QtCore.QPointF] = None
        self._sample_time: Optional[float] = None   # exact keyboard time, independent of the pointer pixel
        self._pointer_target: Optional[QtCore.QPoint] = None
        self._warping_pointer = False
        self._measure: Optional[Tuple[float, Optional[float]]] = None
        self._bounds: Dict[str, float] = {}
        self._xy_hover: Optional[Tuple[float, float, float]] = None     # x px, y px, time of the point
        self._xy_points: Optional[Tuple[tuple, np.ndarray, np.ndarray, np.ndarray]] = None
        self._cache: Optional[QtGui.QPixmap] = None
        self._layout: Optional[Layout] = None
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(320, 200)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self._flash = QtCore.QTimer(self)
        self._flash.setInterval(450)
        self._flash.timeout.connect(self._on_flash)

    # ------------------------------------------------------------------ content
    def set_signals(self, signals: Sequence[DisplaySignal], full_range: Optional[Tuple[float, float]], *,
                    keep_view: bool = False) -> None:
        """Show ``signals``. Use ``full_range=None`` before a recording is loaded."""
        self.signals = list(signals)
        self._sample_time = None
        self._has_recording = full_range is not None
        lo, hi = (0.0, 1.0) if full_range is None else (float(full_range[0]), float(full_range[1]))
        if not hi > lo:
            hi = lo + 1.0
        self.full = (lo, hi)
        if not keep_view:
            self.state.x0, self.state.x1, self.state.y0, self.state.y1 = lo, hi, 0.0, 1.0
            self._zoom_stack.clear()
        st = self.state
        if not 0 <= st.selected < len(self.signals):
            self._select(-1)
        # rows of the upper window are never emphasized (a line may have become a state band)
        st.emphasized = frozenset(i for i in st.emphasized if i < len(self.signals) and self.signals[i].in_analog)
        if not (st.emphasized and st.temporary == st.selected and st.temporary not in st.emphasized
                and self.shows_curve(st.temporary)):
            st.temporary = -1
        self._measure = None
        if self.state.xy and not self.xy_possible():
            self.state.xy = False
            self.viewChanged.emit()
        self.refresh()

    def _apply_theme(self, settings: object) -> None:
        """Theme colours, legend style, fonts and the picture behind the curves (see themes.py)."""
        from .themes import apply_plot_colours, get_theme
        st = self.state
        theme = get_theme(settings.get("theme", "classic"))
        # colours for the background chosen (set before); a custom "palette" setting still wins (read afterwards)
        apply_plot_colours(st, theme)
        st.legend_style = theme.legend_style
        st.value_font = theme.font_values
        st.legend_fill = theme.legend_fill
        st.legend_idle_ranges = theme.legend_idle_ranges
        self.hover_values = theme.hover_values
        if theme.font_legend:
            st.font_family = theme.font_legend
        elif theme.font_ui:
            st.font_family = theme.font_ui
        backdrop = settings.get("backdrop", None)
        key = theme.backdrop if backdrop is None else str(backdrop)
        try:
            opacity = float(settings.get("backdrop_opacity", None) if settings.get("backdrop_opacity", None) is not None
                            else theme.backdrop_opacity or 0.2)
        except (TypeError, ValueError):
            opacity = 0.2
        st.backdrop_opacity = min(1.0, max(0.0, opacity))
        self._backdrop_key = key
        self._load_backdrop()

    def _load_backdrop(self) -> None:
        from .cute_art import SCENES, backdrop_image
        key = getattr(self, "_backdrop_key", "")
        width = max(400, int(self.width() * self.devicePixelRatioF()))
        self.state.backdrop = backdrop_image(key, width)           # "" = none: the pictures kept are let go
        self.state.backdrop_fit = "width" if key in SCENES else "cover"

    def _rest_cursor(self, pos: Optional[QtCore.QPointF] = None) -> None:
        """Use the theme's ordinary pointer or the cursor for the current interaction."""
        pointer = self.theme_cursors.get("pointer")
        if pos is not None and self.legend_row_at(pos) >= 0:
            self.setCursor(pointer if pointer is not None else QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        elif self._mode:
            self.setCursor(self._zoom_cursor(self._mode))
        elif pointer is not None:
            self.setCursor(pointer)
        else:
            self.unsetCursor()

    def _zoom_cursor(self, mode: str) -> QtGui.QCursor:
        """The pointer while a zoom waits for its drag and during the drag: the theme's (or own picture's) zoom
        pointer, else a magnifier with arrows for the direction: time (Z), values (Y) or both (U)."""
        pointer = self.theme_cursors.get("zoom")
        if pointer is not None:
            return pointer
        from .cute_art import zoom_pointer
        return zoom_pointer(mode if mode in ("x", "y", "xy") else "x", float(self.devicePixelRatioF() or 1.0))

    def set_theme_cursors(self, cursors: Dict[str, QtGui.QCursor]) -> None:
        self.theme_cursors = dict(cursors)
        self._rest_cursor(self._mouse)

    def refresh(self) -> None:
        """Repaint after signals, ranges or presentation options changed."""
        self._cache = None
        self._emit_status()
        self.update()

    def apply_settings(self, settings: object) -> None:
        """Take presentation options from the application settings.

        A value of the wrong type or outside its range (e.g. a hand edited
        settings file) falls back to the default instead of stopping the program.
        """
        st = self.state

        def number(key: str, default: float, lo: float, hi: float) -> float:
            try:
                value = float(settings.get(key, default))
            except (TypeError, ValueError):
                return default
            return min(hi, max(lo, value)) if value == value else default

        color = QtGui.QColor(str(settings.get("background", "#000000")))
        st.background = (color.red(), color.green(), color.blue()) if color.isValid() else (0, 0, 0)
        st.invert_bw = bool(settings.get("invert_bw", True))
        st.grid = bool(settings.get("grid", True))
        st.line_width = number("line_width", 1.0, 1.0, 8.0)
        st.step_mode = bool(settings.get("step_mode_analog", True))
        stopped = str(settings.get("stopped_signals", "dashed"))
        st.stopped = stopped if stopped in ("dashed", "hold", "cut") else "dashed"
        st.font_family = str(settings.get("font_family", "Arial") or "Arial")
        st.font_size = int(number("legend_font_size", 13, 6, 40))
        st.legend_fraction = number("legend_fraction", 0.757, 0.3, 0.95)
        st.digital_fraction = number("digital_fraction", 0.097, 0.0, 0.6)
        fmt = str(settings.get("legend_value_format", "%g"))
        try:
            fmt % 1.0
        except (TypeError, ValueError):
            fmt = "%g"
        st.value_format = fmt
        st.time_labels = bool(settings.get("time_labels", True))
        st.xy_time_color = bool(settings.get("xy_time_color", True))
        st.xy_points = bool(settings.get("xy_points", False))
        self._apply_theme(settings)
        palette = settings.get("palette", None)
        try:
            colours = [tuple(int(v) for v in c[:3]) for c in palette] if palette else []
        except (TypeError, ValueError):
            colours = []
        if len(colours) >= 16 and all(len(c) == 3 for c in colours):
            st.palette = colours
        self.refresh()

    # ------------------------------------------------------------------ geometry
    def _lay(self) -> Layout:
        if self._layout is None:
            self._layout = compute_layout(QtCore.QRectF(self.rect()), self.signals, self.state)
        return self._layout

    def time_at(self, x: float) -> float:
        lay = self._lay()
        frac = (x - lay.analog.left()) / max(1.0, lay.analog.width())
        return self.state.x0 + frac * (self.state.x1 - self.state.x0)

    def fraction_at(self, y: float) -> float:
        """Position in the analog window as a fraction of the full height (0 bottom, 1 top)."""
        lay = self._lay()
        rel = (lay.analog.bottom() - y) / max(1.0, lay.analog.height())
        return self.state.y0 + rel * (self.state.y1 - self.state.y0)

    def _x_of(self, time: float) -> float:
        lay = self._lay()
        return lay.analog.left() + (time - self.state.x0) / (self.state.x1 - self.state.x0) * lay.analog.width()

    def _y_of(self, sig: DisplaySignal, value: float) -> float:
        lay = self._lay()
        norm = (value - sig.scmin) / ((sig.scmax - sig.scmin) or 1.0)
        return lay.analog.bottom() - (norm - self.state.y0) / (self.state.y1 - self.state.y0) * lay.analog.height()

    def _in_plot(self, pos: Optional[QtCore.QPointF]) -> bool:
        if not self._has_recording or pos is None:
            return False
        lay = self._lay()
        return lay.analog.left() <= pos.x() <= lay.analog.right() and lay.digital.top() <= pos.y() <= lay.analog.bottom()

    def legend_row_at(self, pos: QtCore.QPointF) -> int:
        lay = self._lay()
        if not lay.legend.contains(pos):
            return -1
        row = int((pos.y() - lay.legend.top()) / max(1e-6, lay.row_height))
        return row if 0 <= row < len(self.signals) else -1

    def _divider_at(self, pos: QtCore.QPointF) -> str:
        if not self._has_recording:
            return ""
        lay = self._lay()
        if abs(pos.x() - (lay.analog.right() + 1.5)) <= 4:
            return "v"
        if pos.x() < lay.analog.right() and abs(pos.y() - (lay.analog.top() - 1.5)) <= 4:
            return "h"
        return ""

    # ------------------------------------------------------------------ view range
    def x_range(self) -> Tuple[float, float]:
        return self.state.x0, self.state.x1

    def _set_view(self, x0: float, x1: float, y0: float, y1: float, *, remember: bool = True) -> None:
        if not self._has_recording:
            return
        lo, hi = self.full
        span = min(max(x1 - x0, _MIN_SPAN), hi - lo)
        x0 = min(max(x0, lo), hi - span)
        yspan = min(max(y1 - y0, 1e-6), 1.0)
        y0 = min(max(y0, 0.0), 1.0 - yspan)
        new = (x0, x0 + span, y0, y0 + yspan)
        old = (self.state.x0, self.state.x1, self.state.y0, self.state.y1)
        if new == old:
            return
        if remember:
            self._zoom_stack.append(old)
            del self._zoom_stack[:-100]
        self.state.x0, self.state.x1, self.state.y0, self.state.y1 = new
        if self._sample_time is not None and not self.state.x0 <= self._sample_time <= self.state.x1:
            self._sample_time = None
        self.viewChanged.emit()
        self.refresh()

    def set_x_range(self, x0: float, x1: float) -> None:
        self._set_view(x0, x1, self.state.y0, self.state.y1)

    def start_zoom(self, mode: str) -> None:
        """The next drag in the plot zooms: ``"xy"``, ``"x"`` or ``"y"``."""
        if not self._has_recording or self.state.xy:
            return
        self._mode = mode
        self._sample_time = None
        self._rest_cursor(self._mouse)
        self._emit_status()

    def undo_zoom(self) -> None:
        if self._zoom_stack:
            self._set_view(*self._zoom_stack.pop(), remember=False)

    def reset_zoom(self) -> None:
        self._zoom_stack.clear()
        self._set_view(self.full[0], self.full[1], 0.0, 1.0, remember=False)

    def is_zoomed(self) -> bool:
        return (self.state.x0, self.state.x1, self.state.y0, self.state.y1) != (self.full[0], self.full[1], 0.0, 1.0)

    def page(self, direction: str) -> None:
        """``"right"``/``"left"`` in time, ``"up"``/``"down"`` in the Y zoom window."""
        st = self.state
        span, yspan = st.x1 - st.x0, st.y1 - st.y0
        at_mouse = self._in_plot(self._mouse)
        if direction == "right":
            start = self.time_at(self._mouse.x()) if at_mouse and self.time_at(self._mouse.x()) > st.x0 else st.x1
            self._set_view(start, start + span, st.y0, st.y1)
        elif direction == "left":
            end = self.time_at(self._mouse.x()) if at_mouse and self.time_at(self._mouse.x()) < st.x1 else st.x0
            self._set_view(end - span, end, st.y0, st.y1)
        elif direction == "up":
            self._set_view(st.x0, st.x1, st.y0 + yspan, st.y1 + yspan)
        elif direction == "down":
            self._set_view(st.x0, st.x1, st.y0 - yspan, st.y1 - yspan)

    def zoom_about(self, factor: float, *, axis: str = "x") -> None:
        """Zoom by ``factor`` (<1 = in) keeping the point under the mouse fixed."""
        st = self.state
        if axis == "x":
            centre = self.time_at(self._mouse.x()) if self._in_plot(self._mouse) else 0.5 * (st.x0 + st.x1)
            self._set_view(centre - (centre - st.x0) * factor, centre + (st.x1 - centre) * factor, st.y0, st.y1)
        else:
            centre = self.fraction_at(self._mouse.y()) if self._in_plot(self._mouse) else 0.5 * (st.y0 + st.y1)
            self._set_view(st.x0, st.x1, centre - (centre - st.y0) * factor, centre + (st.y1 - centre) * factor)

    # ------------------------------------------------------------------ selection / cursor
    def _select(self, index: int) -> None:
        if index == self.state.selected:
            return
        self._sample_time = None
        self.state.temporary = -1                  # shown with TAB / the wheel only until the selection moves on
        self.state.selected = index
        self.state.flash = True
        self._measure = None
        if index >= 0:
            self._flash.start()
        else:
            self._flash.stop()
        self.selectionChanged.emit(index)

    def select(self, index: int) -> None:
        self._select(index if 0 <= index < len(self.signals) else -1)
        self.refresh()

    def selected(self) -> int:
        return self.state.selected

    def select_next(self, step: int) -> None:
        """TAB / mouse wheel: select the next / previous signal with data. While signals are emphasized the
        signal reached is shown as well: with one emphasized signal the emphasis moves along (one signal at a
        time), with several the new one is shown until the selection moves on (H keeps it)."""
        n = len(self.signals)
        if not n:
            return
        index = self.state.selected
        for _ in range(n):
            index = (index + step) % n if index >= 0 else (0 if step > 0 else n - 1)
            if self.signals[index].has_data:
                break
        self._select(index)
        st = self.state
        if st.emphasized and index not in st.emphasized and self._drawable(index):
            if len(st.emphasized) == 1:
                st.emphasized = frozenset({index})
            else:
                st.temporary = index
        self.refresh()

    def shows_curve(self, index: int) -> bool:
        """True for a line that has a curve in the analog window: data and a colour (in the XY view it may be
        the X axis). A signal shown for the moment (TAB) keeps that mark through a rebuild only on such a line."""
        if not 0 <= index < len(self.signals):
            return False
        sig = self.signals[index]
        return sig.in_analog and sig.has_data and sig.color != HIDDEN_COLOR

    def _drawable(self, index: int) -> bool:
        """True for a curve the analog window really draws now: :meth:`shows_curve`, and not the X axis of the
        XY view."""
        return self.shows_curve(index) and not (self.state.xy and index == xy_axis_index(self.signals))

    def set_reference_line(self, on: bool) -> None:
        self.reference_line = bool(on)
        self._measure = None
        self.refresh()

    # ------------------------------------------------------------------ XY view
    def xy_possible(self) -> bool:
        """The XY view needs an X signal and at least one more analog signal with data."""
        return sum(1 for s in self.signals if s.in_analog and s.has_data) >= 2

    def set_xy(self, on: bool) -> bool:
        """Switch the XY view on or off. Returns the new state (off if it is not possible)."""
        on = bool(on) and self.xy_possible()
        if on != self.state.xy:
            self._sample_time = None
            self.state.xy = on
            self._mode = ""
            self._drag = None
            self._measure = None
            self._rest_cursor(self._mouse)
            self.viewChanged.emit()
            self.refresh()
        return self.state.xy

    def is_xy(self) -> bool:
        return self.state.xy

    def _xy_nearest(self) -> Optional[Tuple[float, float, float]]:
        """The XY sample nearest to the mouse (selected signal, else the first Y signal that is drawn):
        (x px, y px, time)."""
        xi = xy_axis_index(self.signals)
        if xi < 0 or not self._in_plot(self._mouse):
            return None
        st = self.state
        candidates = [st.selected] if 0 <= st.selected < len(self.signals) and st.selected != xi else []
        candidates += [i for i, s in enumerate(self.signals) if i != xi and s.in_analog and s.has_data]
        candidates = [i for i in candidates if self._drawable(i) and not suppressed(i, self.signals[i], st)]
        if not candidates:
            return None
        index = candidates[0]
        lay = self._lay()
        key = (index, xi, st.x0, st.x1, st.y0, st.y1, lay.analog.width(), lay.analog.height(),
               id(self.signals[index]), self.signals[index].scmin, self.signals[index].scmax,
               self.signals[xi].scmin, self.signals[xi].scmax)
        if self._xy_points is None or self._xy_points[0] != key:
            px, py, times = xy_pixels(lay, self.signals[xi], self.signals[index], st)
            self._xy_points = (key, px, py, times)
        _key, px, py, times = self._xy_points
        if len(px) == 0:
            return None
        d2 = (px - self._mouse.x()) ** 2 + (py - self._mouse.y()) ** 2
        k = int(np.argmin(d2))
        return (float(px[k]), float(py[k]), float(times[k])) if d2[k] <= 30.0 ** 2 else None

    def toggle_emphasis(self, index: int) -> None:
        """Emphasize a curve of the analog window or end its emphasis. Rows of the upper window are always shown
        and cannot be emphasized."""
        if not 0 <= index < len(self.signals) or not self.signals[index].in_analog:
            return
        marked = set(self.state.emphasized)
        marked.symmetric_difference_update({index})
        self.state.emphasized = frozenset(marked)
        if self.state.temporary == index or not marked:
            self.state.temporary = -1
        self.refresh()

    def emphasis_key(self) -> None:
        """H. Over a legend row: emphasize that signal or end its emphasis. Elsewhere the same for the selected
        signal. Without a selected signal (or with one of the upper window), H over a curve or a row of the
        upper window selects that signal, so its legend row can be found (it flashes)."""
        row = self.legend_row_at(self._mouse) if self._mouse is not None else -1
        if row >= 0:
            self.toggle_emphasis(row)
            return
        selected = self.state.selected
        if 0 <= selected < len(self.signals) and self.signals[selected].in_analog:
            self.toggle_emphasis(selected)
            return
        found = self.signal_at(self._mouse)
        if found >= 0:
            self.select(found)

    def signal_at(self, pos: Optional[QtCore.QPointF]) -> int:
        """The signal drawn at ``pos``: the row of the upper window there, or the nearest curve of the analog
        window within 6 pixels; -1 when there is none (and always in the XY view)."""
        st = self.state
        if pos is None or st.xy:
            return -1
        lay = self._lay()
        if lay.digital.contains(pos):
            for index, rect in upper_rows(lay, self.signals, st):
                if index >= 0 and rect.top() <= pos.y() < rect.bottom():
                    sig = self.signals[index]
                    return index if sig.has_data and sig.color != HIDDEN_COLOR else -1
            return -1
        if not lay.analog.contains(pos):
            return -1
        best, near = -1, 6.0
        for index, sig in enumerate(self.signals):
            if not sig.in_analog or not sig.has_data or sig.color == HIDDEN_COLOR or suppressed(index, sig, st):
                continue
            distance = curve_distance(lay, sig, st, pos.x(), pos.y())
            if distance <= near:
                best, near = index, distance
        return best

    def value_line(self, time: Optional[float]):
        """The cross hair's mark of the selected signal at ``time``: ``("line", left, right, y)`` - across the
        analog window at its value, or across the upper window in the row of a one-bit signal -,
        ``("frame", rect)`` around the row of a state band, or ``None`` (no selection, no value, not shown)."""
        st = self.state
        index = st.selected
        if time is None or not 0 <= index < len(self.signals):
            return None
        sig = self.signals[index]
        value = sig.value_at(time)
        if value is None:
            return None
        lay = self._lay()
        if sig.in_analog:
            if suppressed(index, sig, st):
                return None                            # not drawn while other signals are emphasized
            y = self._y_of(sig, value)
            if not lay.analog.top() <= y <= lay.analog.bottom():
                return None                            # outside the visible range (never in the DIGITAL window)
            return "line", lay.analog.left(), lay.analog.right(), y
        rect = next((r for i, r in upper_rows(lay, self.signals, st) if i == index), None)
        if rect is None:
            return None
        if sig.lane:
            return "frame", rect
        return "line", lay.digital.left(), lay.digital.right(), float(bit_y(rect, value))

    def cancel(self) -> None:
        """ESC: leave zoom mode, measurement, emphasis, selection and reference line, in that order of need."""
        self._mode = ""
        self._drag = None
        self._sample_time = None
        self._pointer_target = None
        self._bounds.clear()
        self._rest_cursor(self._mouse)
        self._measure = None
        self.state.emphasized = frozenset()
        self.state.temporary = -1
        self.state.marker = None
        self._select(-1)
        self.refresh()

    def _plot_time(self) -> Optional[float]:
        """Time at the plot cursor before applying legend visibility or a pinned marker."""
        if not self._has_recording or self.state.xy:
            return None
        if self._sample_time is not None:
            return self._sample_time
        if self._in_plot(self._mouse):
            return self.time_at(self._mouse.x())
        return None

    def cursor_time(self) -> Optional[float]:
        """Time shown in the legend, from the plot cursor or the pinned marker."""
        if not self._has_recording or self.state.xy:
            return None
        time = self._plot_time()
        if time is not None and (self.reference_line or self.state.selected >= 0 or self.hover_values):
            return time
        return self.state.marker

    def set_marker(self, time: Optional[float], *, show: bool = True, span: Optional[float] = None) -> None:
        """Pin a time (e.g. an event): a line is drawn there and the legend shows the values at it.
        With ``show`` the view moves so that the time is visible (``span`` seconds wide if given)."""
        self.state.marker = None if time is None else float(time)
        if time is not None and show:
            st = self.state
            width = span if span is not None else (st.x1 - st.x0)
            if span is not None or not st.x0 <= time <= st.x1:
                self._set_view(time - 0.3 * width, time + 0.7 * width, st.y0, st.y1)
        self.refresh()

    def marker(self) -> Optional[float]:
        return self.state.marker

    def values_text(self) -> str:
        """Tab separated ``name, value, unit`` of all signals at the cursor (or their ranges)."""
        time = self.cursor_time()
        lines = []
        if time is not None:
            text = repr(time) if self._sample_time is not None else "%.9g" % time
            lines.append(f"t\t{text}\ts")
        for sig in self.signals:
            value = sig.format_value(sig.value_at(time), self.state.value_format) if time is not None else ""
            lines.append(f"{sig.name}\t{value}\t{sig.unit}")
        return "\r\n".join(lines)

    def grab_image(self) -> QtGui.QImage:
        """The whole view as a picture (for the clipboard)."""
        image = QtGui.QImage(self.size(), QtGui.QImage.Format_RGB32)
        if not self._has_recording:
            image.fill(QtCore.Qt.black)
            return image
        painter = QtGui.QPainter(image)
        state = self.state
        cursor, state.cursor = state.cursor, self.cursor_time()
        paint_view(painter, QtCore.QRectF(self.rect()), self.signals, state)
        state.cursor = cursor
        painter.end()
        return image

    # ------------------------------------------------------------------ status
    def _emit_status(self) -> None:
        st = self.state
        info = {"x": "", "y": "", "dydx": "", "t": "", "xrange": "", "yrange": "", "hint": ""}
        if self._has_recording:
            info["xrange"] = "%.6g ... %.6g (s)" % (st.x0, st.x1)
            info["yrange"] = "%.4g ... %.4g%%" % (st.y0 * 100.0, st.y1 * 100.0)
        if self._mode:
            info["hint"] = {"x": "Zoom X: drag a frame", "y": "Zoom Y: drag a frame",
                            "xy": "Zoom: drag a frame"}[self._mode]
        if st.xy:
            xi = xy_axis_index(self.signals)
            if xi >= 0 and self._in_plot(self._mouse):
                lay = self._lay()
                xsig = self.signals[xi]
                frac = (self._mouse.x() - lay.analog.left()) / max(1.0, lay.analog.width())
                info["x"] = "%s %s" % (format_number(xsig.denormalize(frac), xsig.fmt, st.value_format), xsig.unit)
                sig = self.signals[st.selected] if 0 <= st.selected < len(self.signals) else None
                if sig is not None and sig.in_analog and st.selected != xi:
                    info["y"] = "%s %s" % (format_number(sig.denormalize(self.fraction_at(self._mouse.y())), sig.fmt,
                                                         st.value_format), sig.unit)
            self._xy_hover = self._xy_nearest()
            if self._xy_hover is not None:
                info["t"] = "%.6g s" % self._xy_hover[2]
            info["xrange"] = "XY: %.6g ... %.6g (s)" % (st.x0, st.x1)
            self.statusChanged.emit(info)
            return
        time = self._plot_time()
        if time is not None:
            text = repr(time) if self._sample_time is not None else "%.6g" % time
            info["t"] = text + " s"
            sig = self.signals[st.selected] if 0 <= st.selected < len(self.signals) else None
            value = sig.value_at(time) if sig is not None else None
            if self._measure is not None:
                t0, v0 = self._measure
                dx = time - t0
                info["x"] = "d %.6g" % dx
                if value is not None and v0 is not None:
                    dy = value - v0
                    info["y"] = "d " + format_number(dy, sig.fmt, st.value_format)
                    info["dydx"] = "%.6g" % (dy / dx) if dx else ""
            else:
                info["x"] = text
                if value is not None:
                    info["y"] = sig.format_value(value, st.value_format)
        self.statusChanged.emit(info)
        now = self.cursor_time()
        if now != getattr(self, "_last_cursor_time", "unset"):
            self._last_cursor_time = now
            self.cursorTimeChanged.emit(now)

    def _on_flash(self) -> None:
        self.state.flash = not self.state.flash
        self.update()

    # ------------------------------------------------------------------ painting
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        from .cute_art import SCENES
        self._cache = None
        self._layout = None
        if getattr(self, "_backdrop_key", "") in SCENES:
            self._load_backdrop()                  # scenes are drawn for the width; a picture file stays as read
        super().resizeEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self._has_recording:
            painter = QtGui.QPainter(self)
            painter.fillRect(event.rect(), QtCore.Qt.black)
            painter.end()
            return
        self._layout = compute_layout(QtCore.QRectF(self.rect()), self.signals, self.state)
        lay = self._layout
        st = self.state
        if self._cache is None:
            ratio = self.devicePixelRatioF()
            pixmap = QtGui.QPixmap(int(self.width() * ratio), int(self.height() * ratio))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(QtGui.QColor(*st.background))
            cache_painter = QtGui.QPainter(pixmap)
            paint_plot(cache_painter, lay, self.signals, st)
            cache_painter.end()
            self._cache = pixmap
        painter = QtGui.QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        fg = QtGui.QColor(255, 255, 255) if st.dark else QtGui.QColor(30, 30, 30)
        time = self.cursor_time()
        painter.setClipRect(lay.digital.united(lay.analog))
        if st.xy and self._xy_hover is not None and self._in_plot(self._mouse):
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setPen(QtGui.QPen(fg, 1.5))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(QtCore.QPointF(self._xy_hover[0], self._xy_hover[1]), 7.0, 7.0)
            painter.drawText(QtCore.QPointF(self._xy_hover[0] + 10, self._xy_hover[1] - 10), "T = %.4g s" % self._xy_hover[2])
            painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        if st.marker is not None and st.x0 <= st.marker <= st.x1 and not st.xy:
            xk = self._x_of(st.marker)
            painter.setPen(QtGui.QPen(QtGui.QColor(240, 140, 0), 2.0))
            painter.drawLine(QtCore.QPointF(xk, lay.digital.top()), QtCore.QPointF(xk, lay.analog.bottom()))
        if self._measure is not None:
            xm = self._x_of(self._measure[0])
            painter.setPen(QtGui.QPen(fg, 0, QtCore.Qt.DashDotLine))
            painter.drawLine(QtCore.QPointF(xm, lay.digital.top()), QtCore.QPointF(xm, lay.analog.bottom()))
        mouse_cursor = (time is not None and (self._sample_time is not None or self._in_plot(self._mouse))
                        and (self.reference_line or st.selected >= 0))
        if self._mode and self._drag is None and not mouse_cursor and not st.xy and self._in_plot(self._mouse):
            # a zoom waits for its drag: a dotted line over the whole height shows where the frame will start
            painter.setPen(self._zoom_pen(fg))
            painter.drawLine(QtCore.QPointF(self._mouse.x(), lay.digital.top()),
                             QtCore.QPointF(self._mouse.x(), lay.analog.bottom()))
        if mouse_cursor:
            x = self._x_of(time)
            painter.setPen(QtGui.QPen(fg, 0, QtCore.Qt.DashLine))
            painter.drawLine(QtCore.QPointF(x, lay.digital.top()), QtCore.QPointF(x, lay.analog.bottom()))
            mark = self.value_line(time)
            if mark is not None and mark[0] == "line":
                _kind, left, right, y = mark
                painter.drawLine(QtCore.QPointF(left, y), QtCore.QPointF(right, y))
            elif mark is not None:
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(mark[1].adjusted(0.5, 0.5, -1.0, -1.0))
        for side, value in self._bounds.items():
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 160, 0), 0, QtCore.Qt.DashLine))
            if side in "LR":
                xb = self._x_of(value)
                painter.drawLine(QtCore.QPointF(xb, lay.analog.top()), QtCore.QPointF(xb, lay.analog.bottom()))
            else:
                yb = lay.analog.bottom() - (value - st.y0) / (st.y1 - st.y0) * lay.analog.height()
                painter.drawLine(QtCore.QPointF(lay.analog.left(), yb), QtCore.QPointF(lay.analog.right(), yb))
        if self._drag is not None and self._drag.get("kind") == "zoom":
            # the zoom frame: dotted lines, nothing filled - the time limits over the whole height and the box
            # that is dragged between them (the X zoom of Z / a drag uses the time limits, Y the box's heights)
            (ax, ay), (bx, by) = self._zoom_corners(self._drag)
            painter.setPen(self._zoom_pen(fg))
            painter.setBrush(QtCore.Qt.NoBrush)
            for x in (ax, bx):
                painter.drawLine(QtCore.QPointF(x, lay.digital.top()), QtCore.QPointF(x, lay.analog.bottom()))
            for y in (ay, by):
                painter.drawLine(QtCore.QPointF(ax, y), QtCore.QPointF(bx, y))
        painter.setClipping(False)
        st.cursor = time
        st.legend_hover = self._mouse is not None and lay.legend.contains(self._mouse)
        paint_legend(painter, lay, self.signals, st)
        st.legend_hover = False
        st.cursor = None
        painter.end()

    @staticmethod
    def _zoom_pen(color: QtGui.QColor) -> QtGui.QPen:
        pen = QtGui.QPen(color, 0, QtCore.Qt.CustomDashLine)
        pen.setDashPattern([3.0, 3.0])
        return pen

    def _zoom_corners(self, drag: Dict[str, object]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Start and current point of a zoom drag, kept inside the plot (DIGITAL and analog window)."""
        lay = self._lay()

        def inside(point: QtCore.QPointF) -> Tuple[float, float]:
            return (min(max(point.x(), lay.analog.left()), lay.analog.right()),
                    min(max(point.y(), lay.digital.top()), lay.analog.bottom()))

        return inside(drag["start"]), inside(drag["cur"])

    # ------------------------------------------------------------------ mouse
    def _selected_curve_y(self, time: float) -> Optional[float]:
        """The selected trace's drawn height, when it can be followed at this time."""
        index = self.state.selected
        if not 0 <= index < len(self.signals):
            return None
        sig = self.signals[index]
        if (not sig.has_data or sig.color == HIDDEN_COLOR or time < sig.t[0]
                or (self.state.stopped == "cut" and sig.stopped_at(time))):
            return None
        mark = self.value_line(time)
        if mark is not None and mark[0] == "line" and np.isfinite(mark[3]):
            return float(mark[3])
        return None

    def _warp_pointer(self, point: QtCore.QPointF, *, vertical_only: bool = False) -> None:
        if (QtGui.QGuiApplication.platformName() == "offscreen"
                or not self.isVisible() or not self.window().isActiveWindow()):
            return
        local_target = point.toPoint()
        target = self.mapToGlobal(local_target)
        position = QtGui.QCursor.pos()
        physical = _physical_pointer()
        move_native = vertical_only and physical is not None
        if vertical_only and physical is not None:
            # Moving Qt's rounded global x back through Windows can advance it by a
            # physical pixel on every event. Preserve the exact hardware x.
            delta = local_target.y() - self.mapFromGlobal(position).y()
            shift = round(delta * self.devicePixelRatioF())
            if not shift:
                return
            target = QtCore.QPoint(physical.x(), physical.y() + shift)
        elif target == position:
            return
        self._warping_pointer = True
        try:
            if move_native:
                import ctypes

                ctypes.windll.user32.SetPhysicalCursorPos(target.x(), target.y())
            else:
                QtGui.QCursor.setPos(target)
        finally:
            self._warping_pointer = False
        actual = _physical_pointer() if physical is not None else QtGui.QCursor.pos()
        if actual is not None and actual != (physical if physical is not None else position):
            self._pointer_target = actual

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._warping_pointer:
            return
        pos = QtCore.QPointF(event.pos())
        if self._pointer_target is not None:
            actual = _physical_pointer()
            if actual is None:
                actual = self.mapToGlobal(event.pos())
            if actual == self._pointer_target:
                return                     # the snap can arrive as several rounded move events
            self._pointer_target = None
        self._sample_time = None
        self._mouse = pos
        if self._drag is not None:
            kind = self._drag["kind"]
            if kind == "zoom":
                self._drag["cur"] = pos
            elif kind == "v":
                self.state.legend_fraction = min(0.95, max(0.3, (pos.x() + 1.5) / max(1, self.width())))
                self._cache = self._layout = None
            elif kind == "h":
                self.state.digital_fraction = min(0.6, max(0.0, pos.y() / max(1, self.height())))
                self._cache = self._layout = None
        else:
            divider = "" if self._mode else self._divider_at(pos)
            if divider == "v":
                self.setCursor(QtCore.Qt.SplitHCursor)
            elif divider == "h":
                self.setCursor(QtCore.Qt.SplitVCursor)
            else:
                self._rest_cursor(pos)
                if not self._mode and not self.state.xy and self._in_plot(pos):
                    y = self._selected_curve_y(self.time_at(pos.x()))
                    if y is not None:
                        self._warp_pointer(QtCore.QPointF(pos.x(), y), vertical_only=True)
            # Small or cut legend rows: show the full row under the mouse as a tip.
            row = self.legend_row_at(pos)
            if row >= 0:
                self._legend_tip(row, event.globalPos())
            else:
                QtWidgets.QToolTip.hideText()
        self._emit_status()
        self.update()

    def _legend_tip(self, row: int, global_pos: QtCore.QPoint) -> None:
        lay = self._lay()
        sig = self.signals[row]
        state = self.state
        cursor, state.cursor = state.cursor, self.cursor_time()
        state.legend_hover = True                                  # the mouse is on a legend row
        value = legend_value_text(sig, state)
        state.cursor, state.legend_hover = cursor, False
        if lay.font_px < 11 or legend_row_shortened(sig, value, lay, state):
            QtWidgets.QToolTip.showText(global_pos, "%s\n%s" % (sig.label, value), self)
        else:
            QtWidgets.QToolTip.hideText()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._mouse = None
        self._sample_time = None
        self._pointer_target = None
        self._emit_status()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.setFocus()
        pos = QtCore.QPointF(event.pos())
        self._mouse = pos
        if self._sample_time is not None and (not self._in_plot(pos)
                                              or abs(pos.x() - self._x_of(self._sample_time)) > 0.5):
            self._sample_time = None
        row = self.legend_row_at(pos)
        if event.button() == QtCore.Qt.RightButton:
            if row >= 0:
                self.editRequested.emit(row)            # the view editor opens on this signal
                return
            elif self._mode or self._bounds:
                self._mode = ""
                self._bounds.clear()
                self._rest_cursor(pos)
            elif self._measure is not None:
                self._measure = None
            elif self.state.selected >= 0:
                self._select(-1)
            elif self.reference_line:
                self.reference_line = False
                self.commandRequested.emit("reference_line_off")
            self.refresh()
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if row >= 0:
            self._select(-1 if row == self.state.selected else row)
            self.refresh()
            return
        divider = "" if self._mode else self._divider_at(pos)
        if divider:
            self._drag = {"kind": divider}
            return
        if not self._in_plot(pos) or self.state.xy:
            return                              # XY does not currently support time-based measurement
        if self._mode:
            self._drag = {"kind": "zoom", "mode": self._mode, "start": pos, "cur": pos, "implicit": False}
            self.setCursor(self._zoom_cursor(self._mode))
        elif self.reference_line or self.state.selected >= 0:
            if self._measure is None:
                time = self._plot_time()
                sig = self.signals[self.state.selected] if self.state.selected >= 0 else None
                self._measure = (time, sig.value_at(time) if sig is not None else None)
            else:
                self._measure = None
            self._emit_status()
            self.update()
        else:
            self._drag = {"kind": "zoom", "mode": "x", "start": pos, "cur": pos, "implicit": True}
            self.setCursor(self._zoom_cursor("x"))

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        row = self.legend_row_at(QtCore.QPointF(event.pos()))
        if event.button() == QtCore.Qt.LeftButton and row >= 0:
            self.editRequested.emit(row)
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        drag, self._drag = self._drag, None
        if drag is None:
            return
        if drag["kind"] in ("v", "h"):
            self.dividersChanged.emit(self.state.legend_fraction, self.state.digital_fraction)
            self.refresh()
            return
        drag["cur"] = QtCore.QPointF(event.pos())
        (ax, ay), (bx, by) = self._zoom_corners(drag)
        mode = drag["mode"]
        moved = (drag["cur"] - drag["start"]).manhattanLength()
        if not drag["implicit"]:
            self._mode = ""
        self._rest_cursor(QtCore.QPointF(event.pos()))
        lay = self._lay()
        left, right = sorted((ax, bx))
        # values: the part of the box inside the analog window (the DIGITAL window has no value axis)
        top, bottom = sorted(min(max(y, lay.analog.top()), lay.analog.bottom()) for y in (ay, by))
        wide, high = right - left >= 2, bottom - top >= 2
        if moved < 5 or not {"x": wide, "y": high, "xy": wide and high}.get(mode, False):
            self.refresh()
            return
        st = self.state
        x0, x1 = (self.time_at(left), self.time_at(right)) if mode in ("x", "xy") else (st.x0, st.x1)
        y0, y1 = (self.fraction_at(bottom), self.fraction_at(top)) if mode in ("y", "xy") else (st.y0, st.y1)
        self._set_view(x0, x1, y0, y1)
        self.refresh()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            return
        mods = event.modifiers()
        self._mouse = QtCore.QPointF(event.pos())
        if self.state.xy and mods & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            event.accept()                       # the XY view keeps its time window
            return
        if mods & QtCore.Qt.ControlModifier:
            self.zoom_about(0.8 if delta > 0 else 1.25, axis="y" if mods & QtCore.Qt.ShiftModifier else "x")
        elif mods & QtCore.Qt.ShiftModifier:
            span = self.state.x1 - self.state.x0
            shift = -0.1 * span if delta > 0 else 0.1 * span
            self._set_view(self.state.x0 + shift, self.state.x1 + shift, self.state.y0, self.state.y1, remember=False)
        else:
            self.select_next(1 if delta < 0 else -1)
        event.accept()

    # ------------------------------------------------------------------ keyboard
    def focusNextPrevChild(self, next_child: bool) -> bool:
        return False                           # TAB moves the signal selection instead of the focus

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & QtCore.Qt.ControlModifier)
        text = event.text().upper()
        if key in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab):
            self.select_next(-1 if key == QtCore.Qt.Key_Backtab or mods & QtCore.Qt.ShiftModifier else 1)
        elif key == QtCore.Qt.Key_Escape:
            self.cancel()
            if self.reference_line:
                self.reference_line = False
                self.commandRequested.emit("reference_line_off")
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self._apply_bounds()
        elif key == QtCore.Qt.Key_Space:
            self.reset_zoom()
        elif key == QtCore.Qt.Key_Minus:
            self.undo_zoom()
        elif key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down) and not ctrl and not mods & QtCore.Qt.AltModifier:
            step = 1 if key == QtCore.Qt.Key_Up else -1
            if not self.nudge_range(step, bool(mods & QtCore.Qt.ShiftModifier)):
                self._move_mouse(0, -step)
        elif key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right) and not mods:
            step = -1 if key == QtCore.Qt.Key_Left else 1
            if not self._step_selected_sample(step):
                self._move_mouse(step, 0)
        elif ctrl and key == QtCore.Qt.Key_C:
            self.commandRequested.emit("copy_name")
        elif ctrl or mods & QtCore.Qt.AltModifier:
            super().keyPressEvent(event)
        elif text == "X":
            self.commandRequested.emit("reference_line_toggle")
        elif text in ("U", "Z", "Y"):
            self.start_zoom({"U": "xy", "Z": "x", "Y": "y"}[text])
        elif text == "N":
            self.page("right")
        elif text == "P":
            self.page("left")
        elif text in ("L", "R", "T", "B") and self._in_plot(self._mouse):
            self._bounds[text] = self._plot_time() if text in "LR" else self.fraction_at(self._mouse.y())
            self.update()
        elif text == "H":
            self.emphasis_key()
        elif text == "E":
            self.commandRequested.emit("edit")
        elif text == "Q":
            self.commandRequested.emit("reload")
        elif text == "I":
            self.commandRequested.emit("info")
        elif text == "C":
            self.commandRequested.emit("copy_values")
        elif text == "S":
            self.commandRequested.emit("statistics")
        elif text == "0":
            self.commandRequested.emit("toggle_script")
        else:
            super().keyPressEvent(event)

    def nudge_range(self, step: int, scale: bool) -> bool:
        """Up / Down keys on the selected signal (without a selection: on the emphasized signals).

        ``step`` +1 (Up) or -1 (Down). Without ``scale`` the view range moves by one screen pixel, so the curve
        moves one pixel up / down and keeps its size. With ``scale`` (Shift) both ends of the visible range move
        by one pixel around the middle of the window: Shift+Up makes the curve taller, Shift+Down flatter.
        Holding the key repeats it. Only curves of the analog window change. Returns ``False`` when no signal
        is selected or emphasized (then the keys move the mouse pointer)."""
        st = self.state
        if st.xy:
            return False
        if 0 <= st.selected < len(self.signals):
            targets = [self.signals[st.selected]]
        elif st.emphasized:
            targets = [self.signals[i] for i in sorted(st.emphasized) if 0 <= i < len(self.signals)]
        else:
            return False
        height = max(1.0, self._lay().analog.height())
        shown = st.y1 - st.y0 or 1.0
        changed = False
        for sig in targets:
            span = sig.scmax - sig.scmin
            if not sig.in_analog or not np.isfinite(span) or span == 0:
                continue
            low, high = sig.scmin + st.y0 * span, sig.scmin + st.y1 * span     # the visible value range
            pixel = (high - low) / height
            if scale:
                if step > 0 and abs(high - low) <= 4 * abs(pixel):
                    continue                                   # keep a few pixels: never turn the range over
                low, high = low + step * pixel, high - step * pixel
            else:
                low, high = low - step * pixel, high - step * pixel
            new_span = (high - low) / shown
            sig.scmin = low - st.y0 * new_span
            sig.scmax = sig.scmin + new_span
            sig.entry.scmin, sig.entry.scmax = sig.scmin, sig.scmax
            sig.entry.scansc = False
            changed = True
        if changed:
            self.refresh()
            self.rangesChanged.emit()
        return True

    def _step_selected_sample(self, direction: int) -> bool:
        if self.state.xy or self._mode or self._drag is not None:
            return False
        index = self.state.selected
        if not 0 <= index < len(self.signals):
            return False
        sig = self.signals[index]
        if not sig.has_data or sig.color == HIDDEN_COLOR or suppressed(index, sig, self.state):
            return False
        times = sig.t
        st = self.state
        time = self.cursor_time()
        if direction > 0:
            sample = np.searchsorted(times, st.x0 if time is None else time,
                                     side="left" if time is None else "right")
        else:
            sample = np.searchsorted(times, st.x1 if time is None else time,
                                     side="right" if time is None else "left") - 1
        time = float(times[max(0, min(len(times) - 1, sample))])
        if not st.x0 <= time <= st.x1:
            half = (st.x1 - st.x0) * 0.5
            self._set_view(time - half, time + half, st.y0, st.y1, remember=False)
        self._sample_time = time
        x = self._x_of(time)
        y = self._selected_curve_y(time)
        if y is None:
            y = self._mouse.y() if self._in_plot(self._mouse) else self._lay().analog.center().y()
        self._mouse = QtCore.QPointF(x, y)
        self._rest_cursor(self._mouse)
        self._warp_pointer(self._mouse)
        self._emit_status()
        self.update()
        return True

    def _move_mouse(self, dx: int, dy: int) -> None:
        """Arrow keys without a signal to move: the mouse pointer moves by one pixel (precise placing)."""
        if self.underMouse():
            QtGui.QCursor.setPos(QtGui.QCursor.pos() + QtCore.QPoint(dx, dy))

    def _apply_bounds(self) -> None:
        if not self._bounds:
            return
        st = self.state
        left, right = self._bounds.get("L", st.x0), self._bounds.get("R", st.x1)
        bottom, top = self._bounds.get("B", st.y0), self._bounds.get("T", st.y1)
        self._bounds.clear()
        if right > left and top > bottom:
            self._set_view(left, right, bottom, top)
        self.refresh()
