"""Regression checks for cycle colours and the Python vocabulary pane."""
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from tracepaddock.i18n import language, set_language
from tracepaddock.plt.parser import dump_plt, parse_plt
from tracepaddock.settings import Settings
from tracepaddock.ui.color_dialog import CyclicColorsDialog
from tracepaddock.ui.editor import PltEditorDialog
from tracepaddock.ui.render import ViewState, signal_color
from tracepaddock.ui.script_dialog import ScriptDialog
from tracepaddock.ui.themes import apply_plot_colours, get_theme


def test_cycle_all_signals_includes_31_and_round_trips(qapp, tmp_path, monkeypatch):
    original = parse_plt("// a comment\n" + "".join(f"sig{k} 0\n" for k in range(40)))
    settings = Settings(tmp_path / "settings.json")
    state = ViewState(background=(255, 255, 255))
    apply_plot_colours(state, get_theme("daylight"))
    dialog = PltEditorDialog(None, original, settings, view_state=state)

    def apply(chooser):
        chooser.last.setValue(31)
        assert chooser.color_range() == (1, 31)
        for index, swatch in enumerate(chooser._swatches, 1):
            assert swatch.toolTip() == QtGui.QColor(*signal_color(index, state)).name()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(CyclicColorsDialog, "exec_", apply)
    dialog.cyclic_color_button.click()
    saved = parse_plt(dump_plt(dialog.script()))
    colors = [entry.color for entry in saved.entries()]
    assert colors == list(range(1, 32)) + list(range(1, 10))
    assert saved.items[0].text == "// a comment"
    assert all(entry.color == 0 for entry in original.entries())
    assert settings.color_range() == (1, 9)
    dialog.accept()
    assert settings.color_range() == (1, 31)


def test_cycle_selected_preserves_unselected_and_cancel_discards_settings(qapp, tmp_path, monkeypatch):
    original = parse_plt("a 0\n//comment\nb 2\nc 3\nd 4\ne 5\n")
    settings = Settings(tmp_path / "settings.json")
    dialog = PltEditorDialog(None, original, settings)
    dialog.entry_list.clearSelection()
    for row in (1, 2, 4, 5):
        dialog.entry_list.item(row).setSelected(True)

    def apply(chooser):
        chooser.first.setValue(19)
        chooser.last.setValue(20)
        return chooser.Selected

    monkeypatch.setattr(CyclicColorsDialog, "exec_", apply)
    dialog.cyclic_color_button.click()
    assert [entry.color for entry in dialog.script().entries()] == [0, 19, 3, 20, 19]
    dialog.reject()
    assert [entry.color for entry in original.entries()] == [0, 2, 3, 4, 5]
    assert settings.color_range() == (1, 9)


def test_both_range_thumbs_move_on_the_same_track(qapp):
    chooser = CyclicColorsDialog((1, 31), ViewState(), 1, 1)
    chooser.show()
    qapp.processEvents()
    slider = chooser.slider

    def drag(old, new):
        y = slider.rect().center().y()
        QtTest.QTest.mousePress(slider, QtCore.Qt.LeftButton, pos=QtCore.QPoint(slider._thumb_center(old), y))
        QtTest.QTest.mouseRelease(slider, QtCore.Qt.LeftButton, pos=QtCore.QPoint(slider._thumb_center(new), y))

    drag(1, 5)
    drag(31, 20)
    assert chooser.color_range() == (5, 20)
    assert (chooser.first.value(), chooser.last.value()) == (5, 20)
    chooser.close()


@pytest.mark.parametrize("code", ["en", "zh"])
def test_vocabulary_wraps_and_scrolls_in_its_own_resizable_pane(qapp, code):
    previous = language()
    set_language(code)
    try:
        dialog = ScriptDialog("test", "x = 1", ["speed", "limit"], lambda *_: {})
        dialog.show()
        qapp.processEvents()
        help_view = dialog.findChild(QtWidgets.QTextBrowser)
        side = help_view.parentWidget()
        assert isinstance(side, QtWidgets.QSplitter) and side.orientation() == QtCore.Qt.Vertical
        split = side.parentWidget()
        assert isinstance(split, QtWidgets.QSplitter) and split.orientation() == QtCore.Qt.Horizontal
        assert help_view.width() >= 430 and help_view.height() >= 300
        assert "rule(\n" in help_view.toPlainText()
        assert "expect_hold" in help_view.toPlainText()
        split.setSizes([1000, 180])
        qapp.processEvents()
        assert help_view.horizontalScrollBar().maximum() == 0
        assert help_view.verticalScrollBar().maximum() > 0
        dialog.filter.setText("speed")
        assert not dialog.signals.item(0).isHidden() and dialog.signals.item(1).isHidden()
        dialog.signals.itemDoubleClicked.emit(dialog.signals.item(0))
        assert 'S("speed")' in dialog.code()
        dialog.close()
    finally:
        set_language(previous)
