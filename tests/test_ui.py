"""Main window, plot view and dialogs, driven without a visible window."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from tracepaddock.core.model import Dataset, SignalData, SignalInfo
from tracepaddock.display import DisplaySignal
from tracepaddock.plt import arrange
from tracepaddock.plt.model import PltEntry
from tracepaddock.plt.parser import load_plt, parse_plt
from tracepaddock.plt.process import build_display_signals
from tracepaddock.settings import Settings
from tracepaddock.ui.dialogs import AppearanceDialog, ComputedSignalDialog, ReferenceLineDialog
from tracepaddock.ui.editor import PltEditorDialog
from tracepaddock.ui.mainwindow import MainWindow
from tracepaddock.ui.plotview import PlotView
from tracepaddock.ui.render import ViewState, signal_color, step_polyline


class DemoDataset(Dataset):
    """Three analog signals in two units and one bit signal."""

    def __init__(self):
        super().__init__("demo.d97")
        t = np.arange(0.0, 10.0, 0.01)
        self._data = {
            "speed_fl": (t, 20 + 5 * np.sin(t)), "speed_fr": (t, 22 + 5 * np.sin(t + 0.3)),
            "pressure": (t[::4], 100 + 50 * np.cos(t[::4])), "brake": (t, (t > 5).astype(float)),
        }
        self.signals = [SignalInfo("speed_fl", unit="m/s"), SignalInfo("speed_fr", unit="m/s"),
                        SignalInfo("pressure", unit="bar"), SignalInfo("brake", kind="D0")]
        self._finish_catalogue()

    def read(self, signal):
        t, y = self._data[signal.name]
        return SignalData(t, y)

    def time_range(self):
        return (0.0, 9.99)


SCRIPT = "speed_fl nobit 2 scmin=0 scmax=50\nspeed_fr nobit 3 scmin=0 scmax=50\npressure nobit 4 scmin=0 scmax=200\nbrake bit 5\n"


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "settings.json")


@pytest.fixture
def window(qapp, settings):
    win = MainWindow(settings)
    win.resize(1200, 700)
    win.dataset = DemoDataset()
    win.script = parse_plt(SCRIPT)
    win._script_snapshot = parse_plt(SCRIPT)
    win.show()
    win.rebuild(keep_view=False)
    qapp.processEvents()
    yield win
    win.script.modified = False
    win.close()


def test_step_polyline_holds_samples_and_decimates():
    t = np.array([0.0, 1.0, 2.0])
    xs, ys = step_polyline(t, np.array([5.0, 7.0, 6.0]), 0.0, 3.0, 300)
    assert xs.tolist() == [0, 1, 1, 2, 2, 3] and ys.tolist() == [5, 5, 7, 7, 6, 6]
    big_t = np.linspace(0, 1, 100000)
    xs, ys = step_polyline(big_t, np.sin(big_t * 50), 0.0, 1.0, 200)
    assert len(xs) <= 802 and ys.max() == pytest.approx(1.0, abs=1e-3) and ys.min() == pytest.approx(-1.0, abs=1e-3)
    assert np.all(np.diff(xs)[np.diff(ys) != 0] == 0)
    assert len(step_polyline(np.empty(0), np.empty(0), 0, 1, 100)[0]) == 0


def test_dense_samples_with_few_changes_keep_the_true_pulse_edges():
    t = np.arange(10000) / 100
    values = np.zeros_like(t)
    values[(t >= 50.51) & (t < 50.56)] = 1
    values[(t >= 50.82) & (t < 50.91)] = 3
    xs, ys = step_polyline(t, values, 0, 100, 100, step=True)
    transitions = np.flatnonzero(np.diff(ys) != 0) + 1
    np.testing.assert_array_equal(xs[transitions], [50.51, 50.56, 50.82, 50.91])
    np.testing.assert_array_equal(xs[transitions - 1], xs[transitions])
    np.testing.assert_array_equal(ys[transitions], [1, 0, 3, 0])


def test_light_background_darkens_bright_colours():
    dark, light = ViewState(), ViewState(background=(199, 237, 204))
    assert signal_color(4, dark) == (255, 255, 0)
    r, g, b = signal_color(4, light)
    assert r < 200 and g < 200
    assert signal_color(1, dark) == (255, 255, 255) and signal_color(1, light) == (0, 0, 0)


def test_window_shows_signals_and_status(window, qapp):
    view = window.view
    assert [s.name for s in window.signals] == ["speed_fl", "speed_fr", "pressure", "brake"]
    assert window.signals[3].digital and view.x_range() == (0.0, 9.99)
    image = view.grab_image()
    assert image.width() == view.width() and not image.isNull()
    assert "demo.d97" in window.windowTitle()


def test_zoom_undo_reset_and_paging(window, qapp):
    view = window.view
    want = (view.time_at(200), view.time_at(400))          # times under the frame before zooming
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Z)
    QtTest.QTest.mousePress(view, QtCore.Qt.LeftButton, pos=QtCore.QPoint(200, 300))
    QtTest.QTest.mouseRelease(view, QtCore.Qt.LeftButton, pos=QtCore.QPoint(400, 320))
    x0, x1 = view.x_range()
    assert view.is_zoomed() and (x0, x1) == pytest.approx(want)
    assert (view.state.y0, view.state.y1) == (0.0, 1.0)
    view._mouse = QtCore.QPointF(300, 300)                 # N: the next page starts at the mouse position
    start = view.time_at(300)
    view.page("right")
    assert view.x_range() == pytest.approx((start, start + (x1 - x0)))
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Minus)
    assert view.x_range() == pytest.approx((x0, x1))
    view._mouse = None                                     # without a mouse position: one full page
    view.page("right")
    assert view.x_range() == pytest.approx((x1, x1 + (x1 - x0)))
    view.undo_zoom()
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Y)
    QtTest.QTest.mousePress(view, QtCore.Qt.LeftButton, pos=QtCore.QPoint(200, 300))
    QtTest.QTest.mouseRelease(view, QtCore.Qt.LeftButton, pos=QtCore.QPoint(260, 500))
    assert view.state.y1 - view.state.y0 < 1.0 and view.x_range() == pytest.approx((x0, x1))
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Space)
    assert not view.is_zoomed()


def test_reference_line_selection_and_measurement(window, qapp, move_pointer):
    view = window.view
    seen = {}
    view.statusChanged.connect(seen.update)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_X)
    assert view.reference_line and window.act_ref.isChecked()
    move_pointer(view, QtCore.QPoint(300, 400))
    qapp.processEvents()
    time = view.cursor_time()
    assert time == pytest.approx(view.time_at(300))
    assert seen["x"] == "%.6g" % time
    assert window.signals[0].name in view.values_text() and "t\t" in view.values_text()
    lay = view._lay()
    QtTest.QTest.mouseClick(view, QtCore.Qt.LeftButton,
                            pos=QtCore.QPoint(int(lay.legend.left() + 10), int(lay.legend.top() + lay.row_height * 0.5)))
    assert view.selected() == 0
    move_pointer(view, QtCore.QPoint(300, 400))
    QtTest.QTest.mouseClick(view, QtCore.Qt.LeftButton, pos=QtCore.QPoint(300, 400))
    move_pointer(view, QtCore.QPoint(500, 400))
    qapp.processEvents()
    sig = window.signals[0]
    dx = view.time_at(500) - view.time_at(300)
    dy = sig.value_at(view.time_at(500)) - sig.value_at(view.time_at(300))
    assert seen["x"] == "d %.6g" % dx and seen["dydx"] == "%.6g" % (dy / dx)
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Tab)
    assert view.selected() == 1
    QtTest.QTest.keyClick(view, QtCore.Qt.Key_Escape)
    assert view.selected() == -1 and not view.reference_line and not window.act_ref.isChecked()


def test_arrangements(window):
    sigs = window.signals
    window._arrange("auto")
    assert sigs[0].scmin <= 15 and sigs[0].scmax >= 25 and (sigs[0].scmax - sigs[0].scmin) / 10 in (1.0, 2.0, 2.5, 5.0)
    assert window.script.modified and sigs[0].entry.scmin == sigs[0].scmin
    window._arrange("unit")
    assert (sigs[0].scmin, sigs[0].scmax) == (sigs[1].scmin, sigs[1].scmax) != (sigs[2].scmin, sigs[2].scmax)
    window._arrange("bands")
    bands = []
    for s in sigs[:3]:
        lo, hi = arrange.data_range(s)
        span = s.scmax - s.scmin
        bands.append(((lo - s.scmin) / span, (hi - s.scmin) / span))
    assert bands[0][0] > bands[1][1] > bands[1][0] > bands[2][1] and bands[2][0] >= 0 and bands[0][1] <= 1
    window._arrange("unit_bands")
    assert (sigs[0].scmin, sigs[0].scmax) == (sigs[1].scmin, sigs[1].scmax)
    window._arrange("restore")
    assert (window.signals[0].scmin, window.signals[0].scmax) == (0.0, 50.0)
    assert (window.signals[3].scmin, window.signals[3].scmax) == (0.0, 1.0)      # digital signals are left alone


def test_appearance_options(window, settings, qapp):
    window._set_option("background", "#c7edcc")
    window._set_option("line_width", 3.0)
    assert window.view.state.background == (199, 237, 204) and window.view.state.line_width == 3.0
    assert not window.view.state.dark
    dialog = AppearanceDialog(settings, lambda: window.view.apply_settings(settings), window)
    dialog.width_spin.setValue(2.5)
    dialog.grid_check.setChecked(False)
    assert window.view.state.line_width == 2.5 and not window.view.state.grid
    dialog.reject()
    assert window.view.state.line_width == 3.0 and window.view.state.grid


def test_computed_and_reference_signals(window, qapp):
    dialog = ComputedSignalDialog(window.signals, 0, window)
    dialog.kind.setCurrentIndex(dialog.kind.findData("gain"))
    dialog.param_edits[0].setValue(3.6)
    entry = dialog.entry()
    assert entry.name == "speed_fl" and entry.key == "speed_fl_scaled" and entry.ops[0].expr == "3.6*x+0"
    assert (entry.scmin, entry.scmax) == (0.0, 50.0)
    reference = ComputedSignalDialog(window.signals, -1, window)
    reference.kind.setCurrentIndex(reference.kind.findData("const"))
    reference.param_edits[0].setValue(-25.0)
    ref_entry = reference.entry()
    assert ref_entry.name == "*" and ref_entry.ops[0].expr == "(-25)"
    custom = ComputedSignalDialog(window.signals, 0, window)
    custom.kind.setCurrentIndex(custom.kind.findData("custom"))
    custom.formula.setText("nonsense(")
    assert custom.entry() is None
    custom.formula.setText("abs(x)-20")
    assert custom.entry().ops[0].expr == "abs(x)-20"
    window.script.items.insert(1, entry)
    window.script.items.append(ref_entry)
    window.rebuild(keep_view=True)
    scaled = next(s for s in window.signals if s.name == "speed_fl_scaled")
    const = next(s for s in window.signals if s.name == ref_entry.key)
    assert np.allclose(scaled.y, window.signals[0].y * 3.6) and scaled.modified
    assert set(const.y) == {-25.0} and const.error is None
    lines = ReferenceLineDialog(window.signals[0], window)
    lines.rows[0][0].setCurrentText("MEAN")
    lines.rows[1][0].setCurrentIndex(1)
    lines.rows[1][1].setValue(30.0)
    assert lines.refs() == {0: "MEAN", 1: 30.0}


def test_editor_filter_add_edit_and_result(window, settings, qapp):
    dialog = PltEditorDialog(window.dataset, window.script, settings, window)
    dialog.filter_edit.setCurrentText("speed*")
    dialog._refresh_filter()
    assert dialog.name_model.rowCount() == 2
    dialog.filter_edit.setCurrentText("press")                 # plain text means "contains"
    dialog._refresh_filter()
    assert [s.name for s in dialog.name_model.rows] == ["pressure"]
    dialog.filter_mode.setCurrentIndex(2)
    dialog.filter_edit.setCurrentText("(unclosed")
    dialog._refresh_filter()
    assert dialog.name_model.rowCount() == 0
    dialog.filter_edit.setCurrentText(".*_f[lr]")
    dialog._refresh_filter()
    assert dialog.name_model.rowCount() == 2
    dialog.entry_list.setCurrentRow(0)
    dialog.key_edit.setText("v front left")
    dialog.max_edit.setText("80")
    dialog.attr_edit.setPlainText("exp=x*3.6 mavr(5)")
    dialog._store_properties()
    dialog.attr_edit.setPlainText("exp=broken(")
    dialog._store_properties()
    assert dialog.attr_error.text() != ""
    dialog.entry_list.setCurrentRow(3)
    dialog._toggle_entries()
    dialog.accept()
    result = dialog.script()
    first = next(result.entries())
    assert first.key == "v front left" and first.scmax == 80.0 and [o.kind for o in first.ops] == ["exp", "mavr"]
    assert not list(result.entries())[3].enabled and result.modified
    assert next(window.script.entries()).key is None            # the original script is untouched until accepted


def test_file_backed_recording_in_the_window(qapp, settings, mf4_file, mf4_plt):
    win = MainWindow(settings)
    win.resize(1400, 900)
    win.show()
    assert win.open_dataset(mf4_file, mf4_plt)
    qapp.processEvents()
    assert len(win.signals) == 3 and not [s.name for s in win.signals if s.error]
    win.copy_statistics()
    text = QtWidgets.QApplication.clipboard().text()
    assert len(text.splitlines()) == 3 and len(text.splitlines()[0].split("\t")) == 17
    win.script.modified = False
    win.close()
