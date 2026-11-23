"""Stage 2: conversion, automatic ranges, editor details, robustness of the window."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from tracepaddock.core.convert import convert_recording
from tracepaddock.core.d97 import D97Dataset, write_d97
from tracepaddock.core.model import DatasetError
from tracepaddock.core.registry import open_dataset
from tracepaddock.display import DisplaySignal, c_exponent, format_number
from tracepaddock.plt import arrange
from tracepaddock.plt.model import PltEntry
from tracepaddock.plt.parser import PltError, dump_plt, parse_plt
from tracepaddock.settings import Settings
from tracepaddock.ui.dialogs import ComputedSignalDialog
from tracepaddock.ui.editor import PltEditorDialog
from tracepaddock.ui.mainwindow import MainWindow

from test_ui import SCRIPT, DemoDataset


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ conversion
def _same_signals(a, b, names=None):
    """Names whose values, time stamps and units are identical in both recordings."""
    equal, different = [], []
    for sig in a.signals:
        if names is not None and sig.source_name not in names:
            continue
        other = b.find(sig.source_name)
        if other is None:
            different.append((sig.source_name, "missing"))
            continue
        x, y = a.read(sig), b.read(other)
        if (len(x) == len(y) and np.array_equal(x.t, y.t) and np.array_equal(x.y, y.y, equal_nan=True)
                and sig.unit == other.unit):
            equal.append(sig.source_name)
        else:
            different.append(sig.source_name)
    return equal, different


def test_dat_to_mf4_keeps_every_signal(dat_file, tmp_path):
    before = _digest(dat_file)
    target = tmp_path / "converted.mf4"
    convert_recording(dat_file, target)
    with open_dataset(dat_file) as a, open_dataset(target) as b:
        assert b.format_name == "MF4 4.10"
        equal, different = _same_signals(a, b)
        assert not different and len(equal) == len(a.signals) == 2
    assert _digest(dat_file) == before                      # the source is not touched


def test_mf4_to_mdf3_keeps_the_numeric_signals(mf4_file, tmp_path):
    target = tmp_path / "converted.mdf"
    count = convert_recording(mf4_file, target)
    with open_dataset(mf4_file) as a, open_dataset(target) as b:
        assert b.format_name.startswith("MDF 3")
        names = [s.source_name for s in a.signals]
        unique = {n for n in names if names.count(n) == 1}
        with_data = {s.source_name for s in a.signals if s.source_name in unique and s.sample_count}
        equal, different = _same_signals(a, b, with_data)
        assert not different and len(equal) == len(with_data) == 3 and count >= len(equal)


def test_d97_to_mf4(tmp_path):
    class Sig:
        def __init__(self, name, t, y, **kw):
            self.name, self.t, self.y = name, np.asarray(t, float), np.asarray(y, float)
            self.__dict__.update(kw)

    t1, t2 = np.arange(0, 1, 0.01), np.arange(0, 1, 0.1)
    source = tmp_path / "in.d97"
    write_d97(source, [Sig("speed", t1, t1 * 3, unit="m/s"), Sig("flag", t2, (t2 > 0.5) * 1.0, digital=True),
                       Sig("temp", t2, t2 + 20, unit="°C")])
    target = tmp_path / "out.mf4"
    assert convert_recording(source, target) == 3
    with D97Dataset(source) as a, open_dataset(target) as b:
        equal, different = _same_signals(a, b)
        assert not different and sorted(equal) == ["flag", "speed", "temp"]


def test_conversion_refusals(dat_file, tmp_path):
    with pytest.raises(DatasetError, match="bus frames"):
        convert_recording(dat_file, tmp_path / "x.blf")
    with pytest.raises(DatasetError, match="Cannot convert"):
        convert_recording(dat_file, tmp_path / "x.xyz")
    with pytest.raises(DatasetError, match="same file"):
        convert_recording(dat_file, dat_file)
    nested = tmp_path / "new folder" / "x.mf4"               # a missing target folder is created
    convert_recording(dat_file, nested)
    assert nested.exists()


# ------------------------------------------------------------------ automatic ranges
@pytest.mark.parametrize("lo, hi", [(0.0, 99.0), (-3.2, 7.9), (12.82, 19.375), (-1e6, 2e6), (1e-6, 3e-6),
                                    (5.0, 5.0), (-250.0, -240.0)])
def test_nice_range_contains_the_data_with_round_divisions(lo, hi):
    a, b = arrange.nice_range(lo, hi)
    assert a <= lo and b >= hi and b > a
    step = (b - a) / 10.0
    mantissa = step / 10 ** np.floor(np.log10(step))
    assert any(abs(mantissa - m) < 1e-9 for m in (1.0, 2.0, 2.5, 5.0, 10.0))
    if hi > lo:
        assert b - a <= 2.5 * (hi - lo) + 1e-9                # not wastefully wide


def test_place_in_band_puts_the_data_exactly_inside_the_band():
    lo, hi = arrange.place_in_band(10.0, 20.0, 0.25, 0.5, margin=0.1)
    span = hi - lo
    assert (10.0 - lo) / span == pytest.approx(0.275) and (20.0 - lo) / span == pytest.approx(0.475)


# ------------------------------------------------------------------ numeric formatting
def test_three_digit_exponents():
    assert c_exponent("8.13971e-05") == "8.13971e-005" and c_exponent(" 8.13971e-05") == "8.13971e-005"
    assert format_number(123456789.0) == "1.23457e+008" and format_number(1.5e-7, ".2e") == "1.50e-007"
    assert format_number(42.0, "x") == "2a" and format_number(5.0, "8b") == "00000101"


# ------------------------------------------------------------------ editor
@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "settings.json")


def test_editor_ok_without_changes_changes_nothing(qapp, settings):
    text = "fast  nobit 2 unit=m/s scmin=0 scmax=50\r\nbx bit 3\r\nslow 4 exp=x*2 bit\r\n"
    script = parse_plt(text)
    for row in range(3):
        dialog = PltEditorDialog(None, script, settings)
        dialog.entry_list.setCurrentRow(row)
        dialog.accept()
        result = dialog.script()
        assert dump_plt(result) == text and not result.modified


def test_editor_refuses_values_that_would_read_back_differently(qapp, settings):
    script = parse_plt("fast  nobit 2 unit=m/s scmin=0 scmax=50\r\n")
    dialog = PltEditorDialog(None, script, settings)
    dialog.entry_list.setCurrentRow(0)
    dialog.key_edit.setText("Wheel speed max")
    dialog.min_edit.setText("abc")
    dialog.ref_edit.setText("0; middle")
    dialog._store_properties()
    entry = next(dialog.script().entries())
    assert entry.key is None and entry.scmin == 0.0 and entry.refs == {}
    assert "max" in dialog.key_edit.toolTip() and dialog.min_edit.toolTip() and dialog.ref_edit.toolTip()
    dialog.key_edit.setText("Wheel_speed_max")
    dialog.min_edit.setText("-5")
    dialog.ref_edit.setText("0; mean")
    dialog._store_properties()
    assert (entry.key, entry.scmin, entry.refs) == ("Wheel_speed_max", -5.0, {0: 0.0, 1: "MEAN"})


def test_writer_refuses_an_ambiguous_display_name():
    entry = PltEntry(name="v", key="speed max")
    script = parse_plt("")
    script.items.append(entry)
    with pytest.raises(PltError, match="max"):
        dump_plt(script)


def test_editor_writes_mdf_names_without_device_suffix_when_unambiguous(qapp, settings, dat_file):
    with open_dataset(dat_file) as ds:
        dialog = PltEditorDialog(ds, parse_plt(""), settings)
        dialog.filter_edit.setCurrentText("BenchTorque*")
        dialog._refresh_filter()
        assert dialog.name_model.rowCount() == 1
        dialog.name_view.selectAll()
        dialog._add_selected()
        entry = next(dialog.script().entries())
        assert entry.name == "BenchTorque" and entry.key is None
        assert ds.find(entry.name).source_name == "BenchTorque\\Sensor:1"
        dialog._refresh_filter()
        index = dialog.name_model.index(0)
        assert dialog.name_model.data(index, QtCore.Qt.ForegroundRole) is not None     # shown as used


# ------------------------------------------------------------------ window robustness
def test_language_switch_leaves_each_shortcut_once(qapp, settings):
    win = MainWindow(settings)
    for language in ("zh", "en", "zh"):
        win._set_language(language)
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    owners = [a for a in win.findChildren(QtGui.QAction) if a.shortcut() == QtGui.QKeySequence("Ctrl+O")]
    assert len(owners) == 1
    assert len(win.findChildren(QtWidgets.QToolBar)) == 1
    win.close()


def test_window_starts_with_a_damaged_settings_file(qapp, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"line_width": "thick", "background": "not a colour", "palette": [[1, 2]],
                                "legend_font_size": None, "legend_value_format": "%q", "digital_fraction": 7}))
    win = MainWindow(Settings(path))
    state = win.view.state
    assert state.line_width == 1.0 and state.background == (0, 0, 0) and state.value_format == "%g"
    assert state.digital_fraction == 0.6 and len(state.palette) == 32
    path.write_text("{ this is not json")
    MainWindow(Settings(path)).close()
    win.close()


def test_computed_signal_is_placed_below_the_signal_it_uses(qapp, settings):
    win = MainWindow(settings)
    win.dataset = DemoDataset()
    win.script = parse_plt(SCRIPT.replace("pressure nobit 4", "pressure nobit 4 key=p_bar"))
    win.rebuild(keep_view=False)
    dialog = ComputedSignalDialog(win.signals, 0, win)
    dialog.kind.setCurrentIndex(dialog.kind.findData("minus"))
    dialog.other.setCurrentIndex(2)                         # 'p_bar', a display name defined below the source
    entry = dialog.entry()
    assert entry.ops[-1].refs == {"x1": "p_bar"}
    items = win.script.items
    used = [e for e in (dialog.source_entry(), dialog.other_entry()) if e is not None]
    items.insert(max(items.index(e) for e in used) + 1, entry)
    win.rebuild(keep_view=True)
    new = next(s for s in win.signals if s.entry is entry)
    assert new.error is None and new.has_data
    win.script.modified = False
    win.close()
