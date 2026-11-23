"""Zoom with a frame (Z time, Y values, U both; a plain drag zooms in time), using an unfilled dotted frame:
dotted lines and nothing filled - the time limits over the whole height and the dragged box between them -
and a magnifier pointer with arrows for the zoom direction."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui
from PySide6.QtTest import QTest

from test_legend_interaction import _pixels, _view


def _drag_to(view, pos):
    view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, pos, QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
                                          QtCore.Qt.NoModifier))


def _points(view, fx0, fy0, fx1, fy1):
    lay = view._lay()
    a = lay.analog
    return (QtCore.QPointF(a.left() + fx0 * a.width(), a.top() + fy0 * a.height()),
            QtCore.QPointF(a.left() + fx1 * a.width(), a.top() + fy1 * a.height()))


def test_the_zoom_pointer_is_a_magnifier_with_the_direction(qapp):
    view, _signals = _view(qapp)
    pictures = {}
    for key, mode in (("z", "x"), ("y", "y"), ("u", "xy")):
        QTest.keyClick(view, key)
        view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, view._lay().analog.center(),
                                              QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
        pointer = view.cursor()
        assert pointer.shape() == QtCore.Qt.BitmapCursor and not pointer.pixmap().isNull()
        assert pointer.hotSpot() == QtCore.QPoint(14, 14)                       # the centre of the lens
        pictures[mode] = pointer.pixmap().toImage()
        QTest.keyClick(view, QtCore.Qt.Key_Escape)
        assert view.cursor().shape() == QtCore.Qt.ArrowCursor
    assert pictures["x"] != pictures["y"] and pictures["y"] != pictures["xy"] and pictures["x"] != pictures["xy"]
    view.close()


def test_z_frame_is_the_dragged_box_with_the_time_limits_and_no_fill(qapp):
    view, _signals = _view(qapp)
    lay = view._lay()
    st = view.state
    fg = (255, 255, 255) if st.dark else (30, 30, 30)
    QTest.keyClick(view, "z")
    a, b = _points(view, 0.31, 0.31, 0.62, 0.47)
    QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, a.toPoint())
    assert view.cursor().shape() == QtCore.Qt.BitmapCursor                     # the magnifier while dragging
    _drag_to(view, b)
    px = _pixels(view)
    x0, x1, y0, y1 = a.toPoint().x(), b.toPoint().x(), a.toPoint().y(), b.toPoint().y()
    on_row = lambda row: max(np.mean([tuple(px[r, x]) == fg for x in range(x0 + 2, x1 - 2)]) for r in (row - 1, row, row + 1))
    assert on_row(y0) > 0.3 and on_row(y1) > 0.3                             # the box: dotted top and bottom
    on_column = lambda col: max(np.mean([tuple(px[y, c]) == fg for y in range(int(lay.digital.top()) + 2,
                                                                               int(lay.analog.bottom()) - 2)])
                                for c in (col - 1, col, col + 1))
    assert on_column(x0) > 0.3 and on_column(x1) > 0.3                       # the time limits, whole height
    probe = px[int(lay.analog.top() + 0.15 * lay.analog.height()), (x0 + x1) // 2]
    assert tuple(probe) == tuple(st.background)                                 # above the box: not filled
    QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, b.toPoint())
    assert st.x0 == pytest.approx(0.31 * 60.0, abs=0.2) and st.x1 == pytest.approx(0.62 * 60.0, abs=0.2)
    assert (st.y0, st.y1) == (0.0, 1.0)                                         # Z: time only
    assert view.cursor().shape() == QtCore.Qt.ArrowCursor
    view.close()


def test_y_and_u_zoom_the_values_of_the_box(qapp):
    view, _signals = _view(qapp)
    st = view.state
    QTest.keyClick(view, "y")
    a, b = _points(view, 0.2, 0.25, 0.4, 0.75)
    QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, a.toPoint())
    _drag_to(view, b)
    QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, b.toPoint())
    assert (st.x0, st.x1) == (0.0, 60.0)                                        # Y: values only
    assert st.y0 == pytest.approx(0.25, abs=0.01) and st.y1 == pytest.approx(0.75, abs=0.01)
    QTest.keyClick(view, " ")
    QTest.keyClick(view, "u")
    a, b = _points(view, 0.5, 0.0, 0.75, 0.5)
    QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, a.toPoint())
    _drag_to(view, b)
    QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, b.toPoint())
    assert st.x0 == pytest.approx(30.0, abs=0.2) and st.x1 == pytest.approx(45.0, abs=0.2)
    assert st.y0 == pytest.approx(0.5, abs=0.01) and st.y1 == pytest.approx(1.0, abs=0.01)
    view.close()


def test_a_flat_drag_still_zooms_in_time_and_a_tiny_one_does_nothing(qapp):
    view, _signals = _view(qapp)
    st = view.state
    a, b = _points(view, 0.1, 0.5, 0.3, 0.5)                                   # no height at all
    QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, a.toPoint())   # a plain drag
    _drag_to(view, b)
    QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, b.toPoint())
    assert st.x0 == pytest.approx(6.0, abs=0.2) and st.x1 == pytest.approx(18.0, abs=0.2)
    QTest.keyClick(view, " ")
    QTest.keyClick(view, "y")
    QTest.mousePress(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, a.toPoint())
    _drag_to(view, b)                                                           # Y with no height: no zoom
    QTest.mouseRelease(view, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, b.toPoint())
    assert (st.x0, st.x1, st.y0, st.y1) == (0.0, 60.0, 0.0, 1.0)
    view.close()


def test_while_a_zoom_waits_a_dotted_line_follows_the_mouse(qapp):
    view, _signals = _view(qapp)
    lay = view._lay()
    fg = (255, 255, 255) if view.state.dark else (30, 30, 30)
    QTest.keyClick(view, "z")
    x = lay.analog.left() + 0.43 * lay.analog.width()
    view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(x, lay.analog.center().y()),
                                          QtCore.Qt.NoButton, QtCore.Qt.NoButton, QtCore.Qt.NoModifier))
    px = _pixels(view)
    column = [tuple(px[y, int(x)]) == fg for y in range(int(lay.digital.top()) + 2, int(lay.analog.bottom()) - 2)]
    assert np.mean(column) > 0.3
    view.close()
