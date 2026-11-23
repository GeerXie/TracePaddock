"""0.4.3: a DIGITAL-window signal can be followed with the cross hair and picked with H, the pointer is a hand
over legend rows, a right click on a legend row opens the view editor (value display moved into it and into
the Signals menu), and while signals are emphasized a click only selects - H shows the signal."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

import tracepaddock.ui.mainwindow as mainwindow_module
from tracepaddock.display import DisplaySignal
from tracepaddock.i18n_zh import ZH
from tracepaddock.plt.model import PltEntry
from tracepaddock.plt.parser import parse_plt
from tracepaddock.settings import Settings
from tracepaddock.ui.editor import PltEditorDialog
from tracepaddock.ui.mainwindow import SHORTCUT_HELP, MainWindow
from tracepaddock.ui.plotview import PlotView
from tracepaddock.ui.render import bit_y, signal_color, suppressed, upper_rows
from tracepaddock.valuemap import ValueMapLibrary

from test_stage3 import TRAINING_VIEW, TrainingTrace, _NOT_TRANSLATED, _library


# ---------------------------------------------------------------------- helpers
def _view(qapp, digital=False, lane=False):
    """Three flat analog curves (2, 5 and 8 of 0...10) and optionally a one-bit signal and a state band."""
    t = np.linspace(0.0, 60.0, 601)
    signals = []
    for name, color, level in (("a", 2, 2.0), ("b", 3, 5.0), ("c", 5, 8.0)):
        signals.append(DisplaySignal(entry=PltEntry(name=name), name=name, t=t, y=np.full(len(t), level),
                                     color=color, scmin=0.0, scmax=10.0))
    if digital:
        bits = (np.arange(len(t)) // 100 % 2).astype(float)          # 0 for 10 s, then 1 for 10 s, ...
        signals.append(DisplaySignal(entry=PltEntry(name="bit"), name="bit", t=t, y=bits, color=4, digital=True,
                                     scmin=0.0, scmax=1.0))
    if lane:
        states = (np.arange(len(t)) // 200).astype(float)            # 0, 1, 2 for 20 s each
        signals.append(DisplaySignal(entry=PltEntry(name="state"), name="state", t=t, y=states, color=6, lane=True,
                                     scmin=0.0, scmax=3.0))
    view = PlotView()
    view.resize(1000, 700)
    view.set_signals(signals, (0.0, 60.0))
    view.show()
    qapp.processEvents()
    return view, signals


def _row_pos(view, row):
    lay = view._lay()
    return QtCore.QPointF(lay.legend.left() + 20.0, lay.legend.top() + (row + 0.5) * lay.row_height)


def _y(view, sig, value):
    lay = view._lay()
    st = view.state
    frac = ((value - sig.scmin) / (sig.scmax - sig.scmin) - st.y0) / (st.y1 - st.y0)
    return lay.analog.bottom() - frac * lay.analog.height()


def _x(view, time):
    lay = view._lay()
    return lay.analog.left() + (time - view.state.x0) / (view.state.x1 - view.state.x0) * lay.analog.width()


def _move(view, pos):
    view.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, pos, QtCore.Qt.NoButton, QtCore.Qt.NoButton,
                                          QtCore.Qt.NoModifier))


def _click(view, pos, button=QtCore.Qt.LeftButton):
    QTest.mouseClick(view, button, QtCore.Qt.NoModifier, pos.toPoint())


def _key(view, text, pos=None):
    if pos is not None:
        view._mouse = QtCore.QPointF(pos)
    QTest.keyClick(view, text)


def _pixels(view):
    image = view.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
    ptr = image.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
    return arr[:, :image.width(), [2, 1, 0]].astype(int)


def _region(pixels, rect):
    return pixels[int(rect.top()):int(rect.bottom()), int(rect.left()):int(rect.right())]


def _count(pixels, rect, rgb, tolerance=30):
    part = _region(pixels, rect)
    return int((np.abs(part - np.array(rgb)).max(axis=2) <= tolerance).sum())


def _drawn(view, signals, index):
    """The curve of signal ``index`` can be seen in the analog window."""
    return _count(_pixels(view), view._lay().analog, signal_color(signals[index].color, view.state)) > 50


# ---------------------------------------------------------------------- the view
def test_right_click_on_a_legend_row_asks_for_the_editor(qapp):
    view, _signals = _view(qapp)
    asked = []
    view.editRequested.connect(asked.append)
    _click(view, _row_pos(view, 1), QtCore.Qt.RightButton)
    assert asked == [1]
    assert view.state.selected == -1                     # no menu and no change of the selection
    view.close()


def test_the_pointer_is_a_hand_over_legend_rows(qapp):
    view, _signals = _view(qapp)
    _move(view, _row_pos(view, 0))
    assert view.cursor().shape() == QtCore.Qt.PointingHandCursor
    _move(view, view._lay().analog.center())
    assert view.cursor().shape() == QtCore.Qt.ArrowCursor
    pixmap = QtGui.QPixmap(16, 16)                       # a theme / own picture pointer wins over the hand
    pixmap.fill(QtCore.Qt.red)
    view.set_theme_cursors({"pointer": QtGui.QCursor(pixmap, 0, 0)})
    _move(view, _row_pos(view, 2))
    assert view.cursor().shape() == QtCore.Qt.BitmapCursor
    _move(view, view._lay().analog.center())
    assert view.cursor().shape() == QtCore.Qt.BitmapCursor
    view.set_theme_cursors({})
    assert view.cursor().shape() == QtCore.Qt.ArrowCursor
    view.close()


@pytest.mark.parametrize("line_width", [1.0, 2.0])
def test_selection_changes_the_legend_without_thickening_curves(qapp, line_width):
    view, signals = _view(qapp, digital=True)
    view.state.line_width = line_width
    view._mouse = None
    view.refresh()
    lay = view._lay()
    ratio = view.devicePixelRatioF()

    def plot_part(rect):
        physical = QtCore.QRectF(rect.topLeft() * ratio, rect.size() * ratio)
        return _region(_pixels(view), physical)

    analog = plot_part(lay.analog)
    digital = plot_part(lay.digital)
    view.select(0)
    assert np.array_equal(plot_part(lay.analog), analog)
    view.select(3)
    assert np.array_equal(plot_part(lay.digital), digital)
    view.select(-1)
    signals[0].y = np.linspace(0.0, 10.0, len(signals[0].y))
    assert view.set_xy(True)
    xy = plot_part(view._lay().analog)
    view.select(1)
    assert np.array_equal(plot_part(view._lay().analog), xy)
    view.close()


def test_a_click_only_selects_and_h_shows_the_signal(qapp):
    view, signals = _view(qapp)
    st = view.state
    assert all(_drawn(view, signals, i) for i in range(3))
    _click(view, _row_pos(view, 0))                      # select a ...
    _key(view, "h", _row_pos(view, 0))                   # ... H over its row: only a is shown
    assert st.emphasized == {0}
    assert _drawn(view, signals, 0) and not _drawn(view, signals, 1) and not _drawn(view, signals, 2)
    _click(view, _row_pos(view, 1))                      # selecting b does not show it yet
    assert st.selected == 1 and suppressed(1, signals[1], st)
    assert not _drawn(view, signals, 1)
    assert view.value_line(30.0) is None                 # no cross hair line on a curve that is not drawn
    _key(view, "h", _row_pos(view, 1))                   # H: now a and b
    assert st.emphasized == {0, 1}
    assert _drawn(view, signals, 0) and _drawn(view, signals, 1) and not _drawn(view, signals, 2)
    _click(view, _row_pos(view, 1))                      # a second click only ends the selection
    assert st.selected == -1 and _drawn(view, signals, 1)
    _key(view, "h", _row_pos(view, 1))                   # H over the row ends the emphasis of b
    assert st.emphasized == {0} and not _drawn(view, signals, 1)
    _click(view, _row_pos(view, 2))                      # H in the plot acts on the selected signal
    _key(view, "h", view._lay().analog.center())
    assert st.emphasized == {0, 2}
    view.close()


def test_h_in_the_plot_without_emphasis_emphasizes_the_selected_signal(qapp):
    view, signals = _view(qapp)
    view.select(1)
    _key(view, "h", view._lay().analog.center())
    assert view.state.emphasized == {1} and _drawn(view, signals, 1) and not _drawn(view, signals, 0)
    QTest.keyClick(view, QtCore.Qt.Key_Escape)           # ESC ends every emphasis
    assert not view.state.emphasized and _drawn(view, signals, 0)
    view.close()


def test_h_over_a_curve_or_a_digital_row_selects_that_signal(qapp):
    view, signals = _view(qapp, digital=True)
    st = view.state
    x = _x(view, 30.0)
    _key(view, "h", QtCore.QPointF(x, _y(view, signals[1], 5.0) + 3.0))       # 3 px beside b's line
    assert st.selected == 1 and not st.emphasized
    view.select(-1)
    _key(view, "h", QtCore.QPointF(x, _y(view, signals[1], 6.5)))             # far from every curve
    assert st.selected == -1
    rect = next(r for i, r in upper_rows(view._lay(), signals, st) if i == 3)
    _key(view, "h", rect.center())                                            # the row of the bit signal
    assert st.selected == 3
    # a signal of the DIGITAL window is not emphasized: H over a curve selects that curve instead
    _key(view, "h", QtCore.QPointF(x, _y(view, signals[2], 8.0)))
    assert st.selected == 2 and not st.emphasized
    # while signals are emphasized only the curves that are drawn can be picked
    view.select(-1)
    view.toggle_emphasis(0)
    _key(view, "h", QtCore.QPointF(x, _y(view, signals[2], 8.0)))
    assert st.selected == -1
    _key(view, "h", QtCore.QPointF(x, _y(view, signals[0], 2.0)))
    assert st.selected == 0
    view.close()


def test_a_selected_digital_signal_is_followed_by_the_cross_hair(qapp):
    view, signals = _view(qapp, digital=True)
    lay = view._lay()
    rect = next(r for i, r in upper_rows(lay, signals, view.state) if i == 3)
    assert view.value_line(15.0) is None                 # nothing selected
    view.select(3)
    low, high = view.value_line(5.0), view.value_line(15.0)
    assert low == ("line", lay.digital.left(), lay.digital.right(), pytest.approx(float(bit_y(rect, 0.0))))
    assert high[3] == pytest.approx(float(bit_y(rect, 1.0)))
    assert rect.top() < high[3] < low[3] < rect.bottom()
    # in the window: the status bar shows the value and the line is drawn at the high level of the row
    status = []
    view.statusChanged.connect(status.append)
    _move(view, QtCore.QPointF(_x(view, 15.0), lay.analog.center().y()))
    assert status[-1]["y"] == "1"
    fg = (255, 255, 255) if view.state.dark else (30, 30, 30)
    band = QtCore.QRectF(lay.digital.left(), high[3] - 1, lay.digital.width(), 3)
    with_line = _count(_pixels(view), band, fg, 10)
    view.select(-1)
    _move(view, QtCore.QPointF(_x(view, 15.0), lay.analog.center().y()))
    assert with_line > _count(_pixels(view), band, fg, 10) + 100
    view.close()


def test_a_selected_state_band_gets_a_frame(qapp):
    view, signals = _view(qapp, lane=True)
    rect = next(r for i, r in upper_rows(view._lay(), signals, view.state) if i == 3)
    view.select(3)
    assert view.value_line(30.0) == ("frame", rect)
    _move(view, QtCore.QPointF(_x(view, 30.0), view._lay().analog.center().y()))
    assert not view.grab().isNull()
    view.close()


def test_upper_rows_match_the_drawing_order(qapp):
    view, signals = _view(qapp, digital=True, lane=True)
    view.state.rule_lanes = [("rule", [])]
    rows = upper_rows(view._lay(), signals, view.state)
    assert [i for i, _r in rows] == [3, 4, -1]
    tops = [r.top() for _i, r in rows]
    assert tops == sorted(tops) and rows[0][1].top() == view._lay().digital.top()
    assert all(abs(a[1].bottom() - b[1].top()) < 1e-9 for a, b in zip(rows, rows[1:]))
    view.close()


def test_tab_with_one_emphasized_signal_moves_the_emphasis(qapp):
    view, signals = _view(qapp)
    view.toggle_emphasis(0)
    view.select(0)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)
    assert view.state.selected == 1 and view.state.emphasized == {1}
    assert _drawn(view, signals, 1) and not _drawn(view, signals, 0)
    view.close()


def test_tab_with_several_emphasized_signals_shows_the_signal_reached(qapp):
    view, signals = _view(qapp)
    st = view.state
    view.toggle_emphasis(0)
    view.toggle_emphasis(2)
    view.select(0)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)              # to b: shown for the moment
    assert st.selected == 1 and st.temporary == 1 and st.emphasized == {0, 2}
    assert _drawn(view, signals, 1)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)              # on to c (emphasized): b is hidden again
    assert st.selected == 2 and st.temporary == -1 and not _drawn(view, signals, 1)
    QTest.keyClick(view, QtCore.Qt.Key_Backtab)          # back to b, and H keeps it
    assert st.temporary == 1
    QTest.keyClick(view, "h")
    assert st.emphasized == {0, 1, 2} and st.temporary == -1
    _click(view, _row_pos(view, 1))                      # a click never shows a signal for the moment
    assert st.temporary == -1
    view.close()


# ---------------------------------------------------------------------- the window
@pytest.fixture
def training_window(qapp, tmp_path):
    win = MainWindow(Settings(tmp_path / "settings.json"))
    win.valuemaps = _library(tmp_path)
    win.resize(1300, 760)
    win.dataset = TrainingTrace()
    win.script = parse_plt(TRAINING_VIEW)
    win._script_snapshot = parse_plt(TRAINING_VIEW)
    win.show()
    win.rebuild(keep_view=False)
    qapp.processEvents()
    yield win
    win.script.modified = False
    win.close()


def test_right_click_opens_the_editor_on_the_signal_with_its_value_display(training_window, qapp, monkeypatch):
    win = training_window
    seen = {}

    def run(dialog):
        seen.update(entry=dialog._current_entry().name, enabled=dialog.mode_box.isEnabled(),
                    mode=dialog.mode_box.currentData(), lane=dialog.lane_check.isChecked())
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        dialog.lane_check.setChecked(True)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    _click(win.view, _row_pos(win.view, 0), QtCore.Qt.RightButton)
    assert seen == {"entry": "RoadType", "enabled": True, "mode": "mapped", "lane": False}
    assert win.signals[0].value_mode == "raw" and win.signals[0].lane
    saved = ValueMapLibrary(win.valuemaps.path).exact("RoadType")              # stored in the library at once
    assert saved.table == "路况" and saved.mode == "raw" and saved.lane


def test_value_display_from_the_editor_makes_a_rule_for_that_signal_only(training_window, qapp, monkeypatch):
    win = training_window

    def run(dialog):
        assert dialog.mode_box.currentData() == "both"                          # the wildcard rule DemoSpeed*
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals(win.signals[2].entry)
    rule = win.valuemaps.exact("DemoSpeedCode")
    assert rule.table == "限速码" and rule.mode == "raw"
    assert win.valuemaps.exact("DemoSpeed*").mode == "both"
    assert win.signals[2].value_mode == "raw"


def test_cancel_in_the_editor_keeps_the_value_display(training_window, qapp, monkeypatch):
    win = training_window

    def run(dialog):
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        dialog.lane_check.setChecked(True)
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals(win.signals[0].entry)
    assert win.valuemaps.exact("RoadType").mode == "mapped" and not win.signals[0].lane


def test_value_display_of_a_line_without_texts_or_not_shown_yet(training_window, qapp, monkeypatch):
    win = training_window
    seen = {}

    def run(dialog):
        dialog.entry_list.setCurrentRow(3)                                       # v_x: numbers only
        seen["v_x"] = (dialog.mode_box.isEnabled(), dialog.lane_check.isEnabled(), dialog.value_hint.text())
        dialog._script.items.append(PltEntry(name="SignType"))                  # a line added in the editor
        dialog._refill_entries(select=len(dialog._script.items) - 1)
        seen["new"] = (dialog.mode_box.isEnabled(), dialog.lane_check.isEnabled(), dialog.mapping_button.isEnabled(),
                       dialog.value_hint.text())
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert seen["v_x"][:2] == (False, True) and "Value mapping" in seen["v_x"][2]
    assert seen["new"][:3] == (False, False, False) and "OK" in seen["new"][3]
    assert win.signals[-1].name == "SignType"


def test_the_mapping_button_of_the_editor(training_window, qapp, monkeypatch):
    win = training_window
    seen = {}

    class FakeMapDialog:
        def __init__(self, library, signals, current, parent):
            seen["current"] = signals[current].name
            self.library = copy.deepcopy(library)
            self.library.assign("v_x", "限速码", "mapped")

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def result_library(self):
            return self.library

        def numeric_request(self):
            return 3, "限速码"

    def run(dialog):
        dialog.entry_list.setCurrentRow(3)                                       # v_x
        before = dialog.mode_box.isEnabled()
        dialog.mapping_button.click()
        seen.update(before=before, lines=[e.display_name for e in dialog.script().entries()])
        dialog.entry_list.setCurrentRow(3)
        seen["after"] = dialog.mode_box.isEnabled()
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(mainwindow_module, "ValueMapDialog", FakeMapDialog)
    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert seen["current"] == "v_x" and seen["before"] is False and seen["after"] is True
    assert seen["lines"][4].startswith("v_x_")                                 # the numeric line, below v_x
    assert win.valuemaps.exact("v_x").table == "限速码"                         # saved although the editor was left
    assert len(win.script.items) == 5                                           # ... with Cancel: no new line


def test_marks_follow_their_lines_through_the_editor(training_window, qapp, monkeypatch):
    win = training_window
    win.view.select(3)                                                          # v_x
    win.view.toggle_emphasis(3)
    win.view.toggle_emphasis(0)

    def run(dialog):
        dialog._script.items.insert(0, PltEntry(name="SignType", key="first"))  # every line moves down
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert [s.name for s in win.signals][:2] == ["first", "RoadType"]
    assert win.view.selected() == 4 and win.view.state.emphasized == {1, 4}


def test_signals_menu_sets_the_value_display_of_the_selected_signal(training_window, qapp):
    win = training_window
    win._fill_value_display_menu()
    actions = [a for a in win.value_display_menu.actions() if not a.isSeparator()]
    assert not any(a.isEnabled() for a in actions[:4])                          # nothing selected
    win.view.select(0)
    win._fill_value_display_menu()
    actions = [a for a in win.value_display_menu.actions() if not a.isSeparator()]
    assert [a.isChecked() for a in actions[:3]] == [True, False, False]
    actions[2].trigger()                                                        # raw value
    assert win.signals[0].value_mode == "raw"
    win._fill_value_display_menu()
    actions = [a for a in win.value_display_menu.actions() if not a.isSeparator()]
    actions[3].trigger()                                                        # state band
    assert win.signals[0].lane and win.view.selected() == 0


def test_e_opens_the_editor_on_the_selected_signal(training_window, qapp, monkeypatch):
    win = training_window
    seen = []
    monkeypatch.setattr(PltEditorDialog, "exec_",
                        lambda dialog: seen.append(dialog._current_entry().name) or QtWidgets.QDialog.Rejected)
    win.view.select(3)
    QTest.keyClick(win.view, "e")
    assert seen == ["v_x"]


def test_shortcut_help_is_translated():
    keys = _NOT_TRANSLATED | {"F5"}
    missing = [text for pair in SHORTCUT_HELP for text in pair if text not in ZH and text not in keys]
    assert missing == []


# ---------------------------------------------------------------------- findings of the review
def test_a_signal_that_becomes_a_state_band_is_no_longer_emphasized(training_window, qapp):
    win = training_window
    view = win.view
    view.select(0)
    view.toggle_emphasis(0)                                                     # only RoadType is drawn
    win.set_state_band(0, True)                                                 # Signals > value display
    assert win.signals[0].lane and not view.state.emphasized
    assert not any(suppressed(i, s, view.state) for i, s in enumerate(win.signals))


def test_the_signal_shown_with_tab_stays_shown_after_a_rebuild(training_window, qapp, monkeypatch):
    win = training_window
    view = win.view
    st = view.state
    view.toggle_emphasis(0)
    view.toggle_emphasis(3)                                                     # RoadType and v_x
    view.select(0)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)                                     # SignType, shown for the moment
    assert st.selected == 1 and st.temporary == 1
    win.set_value_mode(1, "raw")                                                # Signals > value display
    assert st.selected == 1 and st.temporary == 1 and not suppressed(1, win.signals[1], st)
    monkeypatch.setattr(PltEditorDialog, "exec_", lambda dialog: QtWidgets.QDialog.Accepted)
    win.edit_signals()                                                          # editor OK without a change
    assert st.selected == 1 and st.temporary == 1

    def run(dialog):                                                            # a line above: marks move along
        dialog._script.items.insert(0, PltEntry(name="DemoSpeedCode", key="top"))
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert st.selected == 2 and st.temporary == 2 and st.emphasized == {1, 4}


def test_tab_does_not_move_the_emphasis_onto_a_line_that_is_not_drawn(qapp):
    from tracepaddock.plt.palette import HIDDEN_COLOR
    view, signals = _view(qapp)
    signals[1].color = HIDDEN_COLOR                                             # b is not drawn
    view.refresh()
    view.toggle_emphasis(0)
    view.select(0)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)                                     # to b: the emphasis stays on a
    assert view.state.selected == 1 and view.state.emphasized == {0}
    QTest.keyClick(view, QtCore.Qt.Key_Tab)                                     # to c: it moves along
    assert view.state.selected == 2 and view.state.emphasized == {2}
    view.close()


def test_xy_hover_ring_only_on_curves_that_are_drawn(qapp):
    from tracepaddock.plt.palette import HIDDEN_COLOR
    t = np.linspace(0.0, 10.0, 101)
    x = DisplaySignal(entry=PltEntry(name="x"), name="x", t=t, y=t.copy(), color=2, scmin=0.0, scmax=10.0)
    helper = DisplaySignal(entry=PltEntry(name="h"), name="h", t=t, y=np.full(len(t), 8.0), color=HIDDEN_COLOR,
                           scmin=0.0, scmax=10.0)
    y = DisplaySignal(entry=PltEntry(name="y"), name="y", t=t, y=np.full(len(t), 2.0), color=3, scmin=0.0, scmax=10.0)
    view = PlotView()
    view.resize(1000, 700)
    view.set_signals([x, helper, y], (0.0, 10.0))
    view.show()
    qapp.processEvents()
    assert view.set_xy(True)
    lay = view._lay()
    py = lay.analog.bottom() - 0.2 * lay.analog.height()                        # y = 2 of 0...10
    view._mouse = QtCore.QPointF(lay.analog.left() + 0.5 * lay.analog.width(), py)
    hit = view._xy_nearest()
    assert hit is not None and abs(hit[1] - py) < 2.0
    view.close()


def test_editor_applies_only_the_fields_changed_there(training_window, qapp, monkeypatch):
    win = training_window

    class FakeMapDialog:                                                        # raw chosen there, saved at once
        def __init__(self, library, signals, current, parent):
            self.library = copy.deepcopy(library)
            self.library.assign("RoadType", "路况", "raw")

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def result_library(self):
            return self.library

        def numeric_request(self):
            return None

    seen = {}

    def run(dialog):
        dialog.lane_check.setChecked(True)                                      # band ticked on RoadType ...
        dialog.mapping_button.click()                                           # ... then raw in the mapping dialog
        seen["row"] = (dialog.mode_box.currentData(), dialog.lane_check.isChecked())
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(mainwindow_module, "ValueMapDialog", FakeMapDialog)
    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals(win.signals[0].entry)
    rule = win.valuemaps.exact("RoadType")
    assert seen["row"] == ("raw", True)
    assert (rule.table, rule.mode, rule.lane) == ("路况", "raw", True)


def test_changing_only_the_mode_keeps_the_state_band_of_the_rule(training_window, qapp, monkeypatch):
    win = training_window
    win.valuemaps.assign("RoadType", "路况", "mapped", lane=True)
    win.script.items[0].view = "bit"                                            # here RoadType is a DIGITAL line
    win.rebuild(keep_view=True)

    def run(dialog):
        dialog.entry_list.setCurrentRow(0)
        assert dialog.mode_box.isEnabled() and not dialog.lane_check.isEnabled()
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    rule = win.valuemaps.exact("RoadType")
    assert (rule.mode, rule.lane) == ("raw", True)


def test_value_display_goes_to_the_rule_of_the_shown_signal(training_window, qapp, monkeypatch):
    win = training_window

    def run(dialog):
        dialog.entry_list.setCurrentRow(0)                                      # RoadType: band ticked ...
        dialog.lane_check.setChecked(True)
        dialog.attr_edit.setPlainText("mavr(3)")                                # ... and the line is processed now
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    rule = win.valuemaps.exact("RoadType")
    assert rule.table == "路况" and rule.lane                                     # the table is kept
    assert not any(a.pattern == "*" or not a.table for a in win.valuemaps.assignments if a.pattern != "RoadType")


def test_a_computed_line_without_key_has_no_value_display(training_window, qapp, monkeypatch):
    from tracepaddock.plt.parser import parse_attributes
    win = training_window
    win.script.items.append(parse_plt("v_x nobit 8 exp=x*3.6").items[0])
    win.rebuild(keep_view=True)
    rules = [(a.pattern, a.table, a.mode, a.lane) for a in win.valuemaps.assignments]
    seen = {}

    def run(dialog):
        dialog.entry_list.setCurrentRow(5)
        seen["row"] = (dialog.mode_box.isEnabled(), dialog.lane_check.isEnabled(), dialog.mapping_button.isEnabled(),
                       dialog.value_hint.text())
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert seen["row"][:3] == (False, False, False) and "key=" in seen["row"][3]
    win.set_state_band(5, True)                                                 # the menu writes no rule either
    win.set_value_mode(5, "raw")
    assert [(a.pattern, a.table, a.mode, a.lane) for a in win.valuemaps.assignments] == rules
    placeholder = PltEntry(name="*")
    parse_attributes("exp=sin(t)", placeholder)
    assert win._rule_for_entry(placeholder, None) == ("", None)               # never a rule named "*"


def test_two_lines_of_one_signal_share_their_value_display(training_window, qapp, monkeypatch):
    win = training_window
    win.script.items.append(parse_plt("RoadType nobit 7 key=Road2").items[0])
    win.rebuild(keep_view=True)
    seen = {}

    def run(dialog):
        dialog.entry_list.setCurrentRow(0)
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        dialog.entry_list.setCurrentRow(5)                                      # Road2 shows that change
        seen["road2"] = dialog.mode_box.currentData()
        dialog.lane_check.setChecked(True)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    rule = win.valuemaps.exact("RoadType")
    assert seen["road2"] == "raw" and (rule.mode, rule.lane) == ("raw", True)


def test_numeric_line_from_the_editor_goes_below_its_source(training_window, qapp, monkeypatch):
    win = training_window

    class FakeMapDialog:
        def __init__(self, library, signals, current, parent):
            self.library = copy.deepcopy(library)

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def result_library(self):
            return self.library

        def numeric_request(self):
            return 2, "限速码"                                                  # DemoSpeedCode, chosen in the dialog

    seen = {}

    def run(dialog):
        dialog.entry_list.setCurrentRow(3)                                      # the editor is on v_x
        dialog.mapping_button.click()
        seen["lines"] = [e.display_name for e in dialog.script().entries()]
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(mainwindow_module, "ValueMapDialog", FakeMapDialog)
    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert seen["lines"][2] == "DemoSpeedCode" and seen["lines"][3].startswith("DemoSpeedCode_")


def test_a_signal_added_while_others_are_emphasized_is_shown(training_window, qapp):
    win = training_window
    view = win.view
    view.toggle_emphasis(3)                                                     # v_x
    entry = win.add_mapped_signal(2, "限速码")                                  # inserted below DemoSpeedCode
    new = next(i for i, s in enumerate(win.signals) if s.entry is entry)
    v_x = [s.name for s in win.signals].index("v_x")
    assert view.selected() == new and view.state.emphasized == {v_x, new}
    assert not suppressed(new, win.signals[new], view.state)


def test_pointer_in_zoom_mode_and_after_esc(qapp):
    view, _signals = _view(qapp)
    QTest.keyClick(view, "z")
    _move(view, view._lay().analog.center())
    assert view.cursor().shape() == QtCore.Qt.BitmapCursor                     # the magnifier
    _move(view, _row_pos(view, 0))
    assert view.cursor().shape() == QtCore.Qt.PointingHandCursor               # a click there selects, as always
    _move(view, view._lay().analog.center())
    QTest.keyClick(view, QtCore.Qt.Key_Escape)
    assert view.cursor().shape() == QtCore.Qt.ArrowCursor
    QTest.keyClick(view, "z")
    _move(view, _row_pos(view, 1))
    QTest.keyClick(view, QtCore.Qt.Key_Escape)                                  # ESC with the mouse on a row
    assert view.cursor().shape() == QtCore.Qt.PointingHandCursor
    view.close()


# ---------------------------------------------------------------------- findings of the second review round
def test_placeholder_or_wildcard_key_never_makes_a_wildcard_rule(training_window, qapp):
    from tracepaddock.plt.parser import parse_attributes
    win = training_window
    win.script.items.append(PltEntry(name="*"))                                # a bare '*' line: no formula, no key
    odd = PltEntry(name="v_x", key="v?")
    parse_attributes("exp=x*2", odd)
    win.script.items.append(odd)
    win.rebuild(keep_view=True)
    rules = [(a.pattern, a.table, a.mode, a.lane) for a in win.valuemaps.assignments]
    for index in (5, 6):
        assert win._rule_for(win.signals[index]) == ("", None)
        assert not win._band_possible(win.signals[index])
        win.set_state_band(index, True)
        win.set_value_mode(index, "raw")
    assert [(a.pattern, a.table, a.mode, a.lane) for a in win.valuemaps.assignments] == rules
    assert not any(s.lane for s in win.signals)


def test_a_pending_change_is_dropped_when_the_mapping_dialog_removes_its_table(training_window, qapp, monkeypatch):
    win = training_window

    class FakeMapDialog:
        def __init__(self, library, signals, current, parent):
            self.library = copy.deepcopy(library)
            self.library.remove_table("路况")                                    # its rule goes with it

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def result_library(self):
            return self.library

        def numeric_request(self):
            return None

    seen = {}

    def run(dialog):
        dialog.entry_list.setCurrentRow(0)
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        dialog.mapping_button.click()
        seen["row"] = (dialog.mode_box.isEnabled(), dialog.value_text_changes())
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(mainwindow_module, "ValueMapDialog", FakeMapDialog)
    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert seen["row"] == (False, [])
    assert win.valuemaps.exact("RoadType") is None                             # not created again by OK


def test_a_change_stays_when_the_line_it_was_chosen_on_is_deleted(training_window, qapp, monkeypatch):
    win = training_window
    win.script.items.append(parse_plt("RoadType nobit 7 key=Road2").items[0])
    win.rebuild(keep_view=True)

    def run(dialog):
        dialog.entry_list.setCurrentRow(0)
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("raw"))
        dialog.entry_list.setCurrentRow(0)
        dialog._delete_entries()                                                # RoadType goes, Road2 shows raw
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert win.valuemaps.exact("RoadType").mode == "raw"
    assert [s.name for s in win.signals][-1] == "Road2" and win.signals[-1].value_mode == "raw"


def test_lines_of_one_signal_with_different_rules_keep_both_choices(training_window, qapp, monkeypatch):
    win = training_window
    win.valuemaps.assign("*_disp", "", "mapped")                               # matches the second line by its key
    win.script.items.append(parse_plt("SignType nobit 7 key=Sign_disp").items[0])
    win.rebuild(keep_view=True)

    def run(dialog):
        dialog.entry_list.setCurrentRow(1)                                      # SignType: both -> mapped
        dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("mapped"))
        dialog.entry_list.setCurrentRow(5)                                      # Sign_disp: band
        dialog.lane_check.setChecked(True)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    rule = win.valuemaps.exact("SignType")
    assert (rule.mode, rule.lane) == ("mapped", True)


def test_band_from_the_menu_keeps_the_texts_as_shown(training_window, qapp):
    win = training_window
    assert win.signals[1].value_mode is None                                    # SignType: its own text table
    win.set_state_band(1, True)
    assert win.signals[1].lane and win.signals[1].value_mode == "both"


def test_a_line_that_is_no_longer_drawn_is_not_shown_for_the_moment(training_window, qapp, monkeypatch):
    win = training_window
    view = win.view
    st = view.state
    view.toggle_emphasis(0)
    view.toggle_emphasis(3)
    view.select(0)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)                                     # SignType, shown for the moment
    assert st.selected == 1 and st.temporary == 1

    def run(dialog):
        dialog.entry_list.setCurrentRow(1)
        dialog.color_box.setCurrentIndex(dialog.color_box.findData(0))          # colour 0: not drawn
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert win.signals[1].color == 0 and st.selected == 1 and st.temporary == -1


def test_numeric_line_from_the_editor_is_selected_and_shown_while_emphasized(training_window, qapp, monkeypatch):
    win = training_window
    view = win.view
    view.toggle_emphasis(3)                                                     # v_x

    class FakeMapDialog:
        def __init__(self, library, signals, current, parent):
            self.library = copy.deepcopy(library)

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def result_library(self):
            return self.library

        def numeric_request(self):
            return 2, "限速码"

    def run(dialog):
        dialog.entry_list.setCurrentRow(2)
        dialog.mapping_button.click()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(mainwindow_module, "ValueMapDialog", FakeMapDialog)
    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    names = [s.name for s in win.signals]
    new = next(i for i, name in enumerate(names) if name.startswith("DemoSpeedCode_"))
    assert view.selected() == new and view.state.emphasized == {names.index("v_x"), new}
    assert not suppressed(new, win.signals[new], view.state)


def test_marks_after_the_editor_removed_every_emphasized_line(training_window, qapp, monkeypatch):
    win = training_window
    view = win.view
    st = view.state
    view.toggle_emphasis(2)
    view.toggle_emphasis(3)                                                     # DemoSpeedCode and v_x
    view.select(0)
    QTest.keyClick(view, QtCore.Qt.Key_Tab)                                     # SignType, shown for the moment
    assert st.selected == 1 and st.temporary == 1

    def run(dialog):
        for row in (3, 2):                                                      # both emphasized lines go
            dialog.entry_list.setCurrentRow(row)
            dialog._delete_entries()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PltEditorDialog, "exec_", run)
    win.edit_signals()
    assert st.selected == 1 and not st.emphasized and st.temporary == -1
    view._mouse = _row_pos(view, 0)
    view.emphasis_key()                                                         # H on RoadType: SignType hidden
    assert st.emphasized == {0} and suppressed(1, win.signals[1], st)


def test_the_x_axis_of_the_xy_view_keeps_the_mark_of_the_signal_shown_for_the_moment(training_window, qapp):
    win = training_window
    view = win.view
    st = view.state
    view.toggle_emphasis(2)
    view.toggle_emphasis(3)
    view.select(4)                                                              # LimitTimes2
    QTest.keyClick(view, QtCore.Qt.Key_Tab)                                     # wraps to RoadType
    assert st.selected == 0 and st.temporary == 0
    assert view.set_xy(True)                                                    # RoadType is now the X axis
    win.set_value_mode(1, "raw")                                                # any rebuild
    assert st.temporary == 0
    view.set_xy(False)
    assert not suppressed(0, win.signals[0], st)


def test_the_cross_hair_line_stays_inside_the_analog_window(qapp):
    view, signals = _view(qapp)
    signals[0].y = np.full(len(signals[0].t), 10.4)                             # above the range 0 ... 10
    view.select(0)
    assert view.value_line(30.0) is None
    view.select(1)
    assert view.value_line(30.0)[0] == "line"
    view.close()
