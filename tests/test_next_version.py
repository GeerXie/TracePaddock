from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets

from tracepaddock.core.registry import open_dataset
from tracepaddock.plt.parser import dump_plt, load_plt, parse_plt
from tracepaddock.plt.process import build_display_signals
from tracepaddock.settings import Settings
from tracepaddock.ui.editor import PltEditorDialog
from tracepaddock.ui.mainwindow import MainWindow
from tracepaddock.ui.open_view import OpenViewDialog, recording_file_filters


def _duplicate_channels(path, groups):
    from asammdf import MDF, Signal

    with MDF(version="4.10") as data:
        for group, ident in groups:
            data.append(
                [Signal(samples=np.full(3, ident), timestamps=np.array([0., 1., 2.]), name="CAN_DataFrame.ID"),
                 Signal(samples=np.zeros(3), timestamps=np.array([0., 1., 2.]), name="CAN_DataFrame.IDE")],
                acq_name=group,
            )
        data.save(path)


def test_duplicate_mdf_sources_roundtrip_and_reorder(qapp, tmp_path):
    original = tmp_path / "a.mf4"
    swapped = tmp_path / "b.mf4"
    unavailable = tmp_path / "c.mf4"
    a, b = ("MessageA", 643), ("MessageB", 736)
    _duplicate_channels(original, [a, b])
    _duplicate_channels(swapped, [b, a])
    _duplicate_channels(unavailable, [a])
    cfg = Settings(tmp_path / "settings.json")
    with open_dataset(original) as dataset:
        dialog = PltEditorDialog(dataset, parse_plt(""), cfg)
        rows = [(i, sig) for i, sig in enumerate(dialog.name_model.rows)
                if sig.name == "CAN_DataFrame.ID" and "MessageB" in sig.origin_label]
        row, info = rows[0]
        index = dialog.name_model.index(row, 0)
        assert "MessageB" in index.data()
        dialog.name_view.selectionModel().select(index, QtCore.QItemSelectionModel.Select)
        dialog._add_selected()
        entry = next(dialog.script().entries())
        assert entry.channel_origin == info.channel_origin
        plt = parse_plt(dump_plt(dialog.script()))
        assert build_display_signals(dataset, plt)[0].y.tolist() == [736, 736, 736]
        dialog.reject()
    with open_dataset(swapped) as dataset:
        assert build_display_signals(dataset, plt)[0].y.tolist() == [736, 736, 736]
    with open_dataset(unavailable) as dataset:
        assert build_display_signals(dataset, plt)[0].error


def test_recording_picker_keeps_the_all_files_option(qapp, d97_file):
    filters = recording_file_filters()
    assert d97_file.suffix not in filters.lower()
    assert filters.split(";;")[-1].endswith(" (*)")
    with open_dataset(d97_file) as dataset:
        assert dataset.signals


def test_open_pair_preserves_previous_view_on_error_and_recording_only(qapp, tmp_path, monkeypatch, demo_pair):
    cfg = Settings(tmp_path / "settings.json")
    window = MainWindow(cfg)
    warnings = []
    replacement = tmp_path / "another.mf4"
    replacement.write_bytes((demo_pair[0]).read_bytes())
    monkeypatch.setattr(window, "_error", warnings.append)
    assert window.open_dataset(*demo_pair)
    previous = window.dataset
    original = window.script
    entry = next(original.entries())
    entry.key = "Unsaved label"
    original.modified = True
    assert not window.open_dataset(replacement, tmp_path / "missing.plt")
    assert window.dataset is previous and window.script is original
    assert str(replacement) not in cfg.text_list("recent_datasets")
    assert warnings
    assert window.open_dataset(replacement)
    assert window.script is original and window.script.modified
    assert window.signals[0].name == "Unsaved label"
    window.script.modified = False
    window.close()


def test_open_dialog_prefills_each_path_independently(qapp, tmp_path, demo_pair):
    cfg = Settings(tmp_path / "settings.json")
    cfg.set("default_plt", str(demo_pair[1]))
    cfg.set("recent_datasets", [str(tmp_path / "absent.mf4"), str(demo_pair[0])])
    dlg = OpenViewDialog(cfg, "", "")
    assert dlg.paths() == (str(demo_pair[0]), str(demo_pair[1]))
    dlg.reject()


def test_custom_cursor_replaces_arrows_within_owned_dialog(qapp, tmp_path):
    cfg = Settings(tmp_path / "settings.json")
    window = MainWindow(cfg)
    from tracepaddock.ui.cute_art import CHARACTERS

    cfg.set("cursor_character", next(iter(CHARACTERS)))
    window.apply_theme()
    dialog = QtWidgets.QDialog(window)
    edit = QtWidgets.QLineEdit(dialog)
    dialog.show()
    qapp.processEvents()
    assert window.cursor().shape() == QtCore.Qt.BitmapCursor
    assert dialog.cursor().shape() == QtCore.Qt.BitmapCursor
    assert edit.cursor().shape() == QtCore.Qt.IBeamCursor
    cfg.set("cursor_character", "")
    window.apply_theme()
    assert dialog.cursor().shape() == QtCore.Qt.ArrowCursor
    dialog.close()
    window.close()


def _click_entry(dialog, row, button=QtCore.Qt.LeftButton, modifiers=QtCore.Qt.NoModifier):
    listing = dialog.entry_list
    center = listing.visualItemRect(listing.item(row)).center()
    QtTest.QTest.mouseClick(listing.viewport(), button, modifiers, center)


def test_right_click_copies_selected_properties_in_editor(qapp, tmp_path):
    cfg = Settings(tmp_path / "settings.json")
    original = parse_plt(
        "FrontLeft nobit 2 scmin=0 scmax=100 unit=km/h\n"
        "// Wheels\n"
        "FrontRight nobit 3 scmin=5 scmax=80 unit=m/s fmt=.2f ref0=15 scansc\n"
        "RearRight nobit 4 scmin=10 scmax=70 unit=m/s\n"
    )
    dialog = PltEditorDialog(None, original, cfg)
    dialog.show()
    qapp.processEvents()
    assert dialog.copy_range.isChecked() and not dialog.copy_unit.isChecked()
    _click_entry(dialog, 0)
    _click_entry(dialog, 1, QtCore.Qt.RightButton)
    assert not dialog.script().modified
    dialog.min_edit.setText("-10")
    _click_entry(dialog, 2, QtCore.Qt.RightButton)
    source, target, last = dialog.script().entries()
    assert dialog.entry_list.currentRow() == 2
    assert (target.scmin, target.scmax, target.scansc) == (-10, 100, False)
    assert (target.unit, target.color, target.fmt, target.refs) == ("m/s", 3, ".2f", {0: 15})
    assert (source.unit, source.scmin, source.scmax) == ("km/h", -10, 100)

    dialog.copy_range.setChecked(False)
    dialog.copy_unit.setChecked(True)
    _click_entry(dialog, 0, modifiers=QtCore.Qt.ControlModifier)
    _click_entry(dialog, 3, QtCore.Qt.RightButton)
    assert last.unit == "m/s"
    _click_entry(dialog, 0)
    _click_entry(dialog, 3, QtCore.Qt.RightButton)
    assert (last.unit, last.scmin, last.scmax) == ("km/h", 10, 70)
    assert next(original.entries()).scmin == 0

    dialog.accept()
    reopened = list(parse_plt(dump_plt(dialog.script())).entries())
    assert (reopened[1].scmin, reopened[2].unit) == (-10, "km/h")
    assert cfg.get("editor/copy_unit") and not cfg.get("editor/copy_range")


def test_right_click_uses_effective_autoscaled_range_and_inherited_unit(qapp, tmp_path, demo_pair):
    cfg = Settings(tmp_path / "settings.json")
    script = parse_plt("RigVelocity nobit scansc\nRig_MarkerValue nobit 4 scansc unit=m/s\n")
    with open_dataset(demo_pair[0]) as dataset:
        expected = build_display_signals(dataset, script, cfg)[0]
        dialog = PltEditorDialog(dataset, script, cfg)
        dialog.copy_unit.setChecked(True)
        dialog.show()
        qapp.processEvents()
        _click_entry(dialog, 0)
        _click_entry(dialog, 1, QtCore.Qt.RightButton)
        source, target = dialog.script().entries()
        assert source.scansc and not target.scansc
        assert (target.scmin, target.scmax, target.unit) == (expected.scmin, expected.scmax, expected.unit)
        dialog.reject()
        assert list(script.entries())[1].scansc
        assert not cfg.get("editor/copy_unit", False)
