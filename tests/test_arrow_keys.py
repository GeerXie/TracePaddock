"""Up / Down move the selected signal's range by one pixel, Shift+Up / Shift+Down scale it by one pixel at each
end around the middle of the window (the curve gets taller / flatter)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from PySide6.QtTest import QTest

from tracepaddock.display import DisplaySignal
from tracepaddock.plt.model import PltEntry
from tracepaddock.ui.plotview import PlotView


def _view(qapp):
    t = np.linspace(0.0, 60.0, 601)
    signals = []
    for name, color in (("a", 2), ("b", 3), ("c", 5)):
        entry = PltEntry(name=name)
        entry.scmin, entry.scmax = 0.0, 10.0
        signals.append(DisplaySignal(entry=entry, name=name, t=t, y=np.full(len(t), 2.0), color=color,
                                     scmin=0.0, scmax=10.0))
    view = PlotView()
    view.resize(1000, 700)
    view.set_signals(signals, (0.0, 60.0))
    view.show()
    qapp.processEvents()
    return view, signals


def _y(view, sig, value):
    lay = view._lay()
    st = view.state
    frac = ((value - sig.scmin) / (sig.scmax - sig.scmin) - st.y0) / (st.y1 - st.y0)
    return lay.analog.bottom() - frac * lay.analog.height()


def test_up_and_down_move_the_selected_curve_by_one_pixel(qapp):
    view, signals = _view(qapp)
    view.select(0)
    seen = []
    view.rangesChanged.connect(lambda: seen.append(1))
    a = signals[0]
    y = _y(view, a, 2.0)
    for _ in range(10):
        QTest.keyClick(view, QtCore.Qt.Key_Up)
    assert _y(view, a, 2.0) == pytest.approx(y - 10.0, abs=1e-6)              # 10 pixels up
    assert a.scmax - a.scmin == pytest.approx(10.0)                            # same size
    assert (a.entry.scmin, a.entry.scmax) == (a.scmin, a.scmax) and not a.entry.scansc
    for _ in range(10):
        QTest.keyClick(view, QtCore.Qt.Key_Down)
    assert (a.scmin, a.scmax) == pytest.approx((0.0, 10.0))
    assert signals[1].scmin == 0.0 and len(seen) == 20                        # only the selected one


def test_left_right_follow_the_selected_samples_even_within_one_pixel(qapp, monkeypatch):
    view, signals = _view(qapp)
    signal = signals[0]
    signal.t = np.array([10.0, 10.0001, 10.0004, 11.0])
    signal.y = np.array([1.0, 1.0, 3.0, 4.0])
    view.select(0)
    view._mouse = None
    view.set_marker(10.0)
    moved = []
    statuses = []
    monkeypatch.setattr(view, "_warp_pointer", lambda point: moved.append(point.toPoint()))
    view.statusChanged.connect(statuses.append)
    QTest.keyClick(view, QtCore.Qt.Key_Right)
    assert view.cursor_time() == 10.0001 and statuses[-1]["x"] == "10.0001"
    QTest.keyClick(view, QtCore.Qt.Key_Right)
    assert view.cursor_time() == 10.0004 and statuses[-1]["y"] == "3"
    assert view.values_text().splitlines()[:2] == ["t\t10.0004\ts", "a\t3\t"]
    assert moved[0].x() == moved[1].x() and moved[0].y() != moved[1].y()
    QTest.keyClick(view, QtCore.Qt.Key_Left)
    assert view.cursor_time() == 10.0001
    view.select(1)     # a different time raster uses that signal's recorded times
    view._mouse = None
    QTest.keyClick(view, QtCore.Qt.Key_Right)
    assert view.cursor_time() == pytest.approx(10.1)
    view.close()


def test_sample_stepping_pans_the_view_and_stops_at_the_last_sample(qapp, monkeypatch):
    view, signals = _view(qapp)
    view.select(0)
    view._mouse = None
    monkeypatch.setattr(view, "_warp_pointer", lambda _point: None)
    view.set_x_range(59.5, 59.95)
    view.set_marker(float(signals[0].t[-2]))
    QTest.keyClick(view, QtCore.Qt.Key_Right)
    assert view.cursor_time() == signals[0].t[-1] and view.x_range()[1] == 60.0
    QTest.keyClick(view, QtCore.Qt.Key_Right)
    assert view.cursor_time() == 60.0
    QTest.keyClick(view, QtCore.Qt.Key_Left)
    assert view.cursor_time() == pytest.approx(59.9)
    view.close()


def test_mouse_warps_to_selected_curve_and_skips_hidden_or_cancelled_selection(qapp, monkeypatch):
    view, signals = _view(qapp)
    view.select(0)
    lay = view._lay()
    point = lay.analog.center().toPoint()
    moves = []
    monkeypatch.setattr(view, "_warp_pointer", lambda target, **options: moves.append((target, options)))

    def move():
        view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(point), QtCore.Qt.NoButton,
                                             QtCore.Qt.NoButton, QtCore.Qt.NoModifier))

    move()
    assert len(moves) == 1 and moves[0][0].x() == point.x()
    assert moves[0][0].y() == pytest.approx(_y(view, signals[0], 2.0))
    assert moves[0][1] == {"vertical_only": True}
    signals[0].color = 0
    move()
    assert len(moves) == 1
    signals[0].color = 2
    QTest.keyClick(view, QtCore.Qt.Key_Escape)
    move()
    assert len(moves) == 1
    view.close()


def test_curve_snap_feedback_does_not_move_time_with_rounded_mouse_events(qapp, monkeypatch):
    from tracepaddock.ui import plotview

    view, _signals = _view(qapp)
    view.select(0)
    point = view._lay().analog.center().toPoint()
    actual = QtCore.QPoint(191, 318)
    requested = []
    seen = []
    monkeypatch.setattr(plotview, "_physical_pointer", lambda: actual)
    view.cursorTimeChanged.connect(seen.append)

    def move(pos):
        view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(pos), QtCore.Qt.NoButton,
                                             QtCore.Qt.NoButton, QtCore.Qt.NoModifier))

    def snap(target, *, vertical_only):
        assert vertical_only
        requested.append(target)
        view._pointer_target = QtCore.QPoint(actual)

    monkeypatch.setattr(view, "_warp_pointer", snap)
    move(point)
    initial_time = view.cursor_time()
    # Fractional display scaling can report a generated move one logical pixel away.
    move(point + QtCore.QPoint(1, 12))
    move(point + QtCore.QPoint(1, 12))
    assert len(requested) == 1 and seen == [initial_time]

    actual.setY(actual.y() + 4)
    move(point + QtCore.QPoint(0, 4))
    assert view.cursor_time() == initial_time and len(requested) == 2
    actual.setX(actual.x() + 4)
    move(point + QtCore.QPoint(4, 0))
    assert view.cursor_time() > initial_time and len(requested) == 3
    view.close()


def test_windows_curve_snap_preserves_the_physical_x_at_fractional_scale(qapp, monkeypatch):
    import ctypes
    from types import SimpleNamespace

    from tracepaddock.ui import plotview

    view, _signals = _view(qapp)
    point = view._lay().analog.center().toPoint()
    physical = QtCore.QPoint(-721, 638)
    moved = []

    def set_physical(x, y):
        moved.append((x, y))
        physical.setX(x)
        physical.setY(y)

    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(SetPhysicalCursorPos=set_physical)), raising=False)
    monkeypatch.setattr(QtGui.QGuiApplication, "platformName", lambda: "windows")
    monkeypatch.setattr(plotview, "_physical_pointer", lambda: QtCore.QPoint(physical))
    monkeypatch.setattr(view, "isActiveWindow", lambda: True)
    monkeypatch.setattr(view, "devicePixelRatioF", lambda: 1.5)
    monkeypatch.setattr(QtGui.QCursor, "pos", lambda: view.mapToGlobal(point))

    view._warp_pointer(QtCore.QPointF(point + QtCore.QPoint(1, 14)), vertical_only=True)

    assert moved == [(-721, 659)]
    assert view._pointer_target == QtCore.QPoint(-721, 659)
    view.close()


def test_qt_curve_snap_ignores_synchronous_feedback(qapp, monkeypatch, move_pointer):
    from tracepaddock.ui import plotview

    view, _signals = _view(qapp)
    view.select(0)
    point = view._lay().analog.center().toPoint()
    position = view.mapToGlobal(point - QtCore.QPoint(7, 8))
    requested = []

    def set_cursor(target):
        requested.append(target)
        move_pointer(view, view.mapFromGlobal(target) + QtCore.QPoint(1, 0))

    monkeypatch.setattr(QtGui.QGuiApplication, "platformName", lambda: "xcb")
    monkeypatch.setattr(plotview, "_physical_pointer", lambda: None)
    monkeypatch.setattr(view, "isActiveWindow", lambda: True)
    monkeypatch.setattr(QtGui.QCursor, "pos", lambda: position)
    monkeypatch.setattr(QtGui.QCursor, "setPos", set_cursor)

    move_pointer(view, point)
    assert len(requested) == 1
    assert view.mapFromGlobal(requested[0]).x() == point.x()
    assert view.cursor_time() == view.time_at(point.x())
    view.close()


def test_shift_up_makes_the_curve_taller_around_the_middle_and_shift_down_flatter(qapp):
    view, signals = _view(qapp)
    view.select(0)
    a = signals[0]
    height = view._lay().analog.height()
    middle = (view._lay().analog.top() + view._lay().analog.bottom()) / 2.0
    value_in_middle = 5.0
    for _ in range(20):
        QTest.keyClick(view, QtCore.Qt.Key_Up, QtCore.Qt.ShiftModifier)
    span = a.scmax - a.scmin
    assert span == pytest.approx(10.0 * (1 - 2.0 / height) ** 20, rel=1e-9)  # one pixel at each end per press
    assert (a.scmin + a.scmax) / 2 == pytest.approx(5.0)                       # around the middle
    assert _y(view, a, value_in_middle) == pytest.approx(middle)
    for _ in range(20):
        QTest.keyClick(view, QtCore.Qt.Key_Down, QtCore.Qt.ShiftModifier)
    assert (a.scmin, a.scmax) == pytest.approx((0.0, 10.0), abs=10.0 / height)   # back within one pixel


def test_without_selection_the_emphasized_signals_move_and_otherwise_nothing(qapp):
    view, signals = _view(qapp)
    view.select(-1)
    view.state.selected = -1
    assert not view.nudge_range(1, False)                                      # nothing selected or emphasized
    assert [s.scmin for s in signals] == [0.0, 0.0, 0.0]
    view.state.emphasized = frozenset({1, 2})
    assert view.nudge_range(1, False)
    assert signals[0].scmin == 0.0 and signals[1].scmin < 0.0 and signals[2].scmin < 0.0


def test_a_y_zoom_moves_by_one_screen_pixel_too(qapp):
    view, signals = _view(qapp)
    view.select(0)
    view.state.y0, view.state.y1 = 0.25, 0.75                                  # zoomed in value
    a = signals[0]
    y = _y(view, a, 4.0)
    QTest.keyClick(view, QtCore.Qt.Key_Up)
    assert _y(view, a, 4.0) == pytest.approx(y - 1.0, abs=1e-6)


def test_the_view_is_marked_changed_in_the_window(qapp, tmp_path, demo_pair):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    win = MainWindow(Settings(tmp_path / "settings.json"))
    assert win.open_dataset(*demo_pair)
    win.script.modified = False
    index = next(i for i, s in enumerate(win.signals) if s.name == "RigVelocity")
    win.view.select(index)
    before = win.signals[index].scmax
    QTest.keyClick(win.view, QtCore.Qt.Key_Up, QtCore.Qt.ShiftModifier)
    assert win.signals[index].scmax < before and win.script.modified and win.windowTitle().count("*") >= 1
    win.script.modified = False
    win.close()


def test_the_legend_shows_the_ranges_while_the_mouse_is_over_it(qapp, move_pointer):
    """Classic look: empty value column without mouse, values with the mouse in the plot, ranges with the mouse
    over the legend."""
    from tracepaddock.ui.render import ViewState, legend_value_text
    sig = DisplaySignal(entry=PltEntry(name="v"), name="v", t=np.array([0.0, 1.0]), y=np.array([3.0, 4.0]),
                        scmin=-30000.0, scmax=30000.0)
    idle = ViewState(legend_idle_ranges=False)
    assert legend_value_text(sig, idle) == ""
    assert legend_value_text(sig, ViewState(legend_idle_ranges=False, legend_hover=True)) == \
        "-3e+004 - 3e+004 [6e+003/div]"
    assert legend_value_text(sig, ViewState(legend_idle_ranges=False, legend_hover=True, cursor=0.5)) == "3"
    # in the view: the flag follows the mouse while painting
    view, signals = _view(qapp)
    view.state.legend_idle_ranges = False
    lay = view._lay()
    seen = []
    import tracepaddock.ui.plotview as pv
    original = pv.paint_legend
    pv.paint_legend = lambda painter, lay_, sigs, st: seen.append(legend_value_text(sigs[0], st))
    try:
        move_pointer(view, QtCore.QPoint(int(lay.legend.right() - 5), int(lay.legend.center().y())))
        view.repaint()
        move_pointer(view, QtCore.QPoint(5, 5))
        view._mouse = None
        view.repaint()
    finally:
        pv.paint_legend = original
    assert "0 - 10 [1/div]" in seen and seen[-1] == ""
