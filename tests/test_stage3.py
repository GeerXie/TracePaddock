"""Stage 3: value mappings, state bands, XY view, export, printing, script history, favourites, settings."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from tracepaddock.core.export import ExportError, export_axis, held_values, matlab_names, write_csv, write_mat, write_mdf
from tracepaddock.core.model import Dataset, EnumTable, SignalData, SignalInfo
from tracepaddock.core.registry import open_dataset
from tracepaddock.display import DisplaySignal
from tracepaddock.i18n_zh import ZH
from tracepaddock.plt.model import PltEntry
from tracepaddock.plt.parser import load_plt, parse_plt, save_plt
from tracepaddock.plt.process import build_display_signals
from tracepaddock.settings import Settings
from tracepaddock.ui.dialogs import FavouritesDialog, TimeRangeDialog
from tracepaddock.ui.exporting import ExportDialog, picture_state, save_picture
from tracepaddock.ui.mainwindow import MainWindow
from tracepaddock.ui.render import (ViewState, compute_layout, state_fill, state_segments, xy_axis_index, xy_curve)
from tracepaddock.ui.settings_dialog import SettingsDialog
from tracepaddock.ui.valuemaps import ValueMapDialog, signal_pattern
from tracepaddock.valuemap import (MapAssignment, MappingImport, ValueMap, ValueMapLibrary, parse_raw_value,
                               read_mapping_file, write_mapping_csv)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------- a synthetic recording with enumerations
class TrainingTrace(Dataset):
    """Generate synthetic state and speed samples at two rates."""

    def __init__(self):
        super().__init__("training.mf4")
        t10 = np.round(np.arange(0.0, 20.0, 0.01), 6)
        t100 = np.round(np.arange(0.0, 20.0, 0.1), 6)
        road = np.where(t100 < 5, 1.0, np.where(t100 < 12, 2.0, 3.0))           # 1 highway, 2 rural, 3 urban
        sign = np.where((t100 >= 4) & (t100 < 15), 7.0, 0.0)                    # 7 = 120 km/h sign seen
        limit = np.where(t10 >= 4.35, np.where(t10 >= 15.2, 0.0, 12.0), 0.0)    # code 12 = 120 km/h on CAN
        self._data = {
            "RoadType": (t100, road), "SignType": (t100, sign), "DemoSpeedCode": (t10, limit),
            "v_x": (t10, 25 + 3 * np.sin(t10)),
        }
        own = EnumTable(values={0.0: "none", 7.0: "limit 120"})
        self.signals = [SignalInfo("RoadType", unit="-"), SignalInfo("SignType", unit="-", enum=own),
                        SignalInfo("DemoSpeedCode", unit="-"), SignalInfo("v_x", unit="m/s")]
        self._finish_catalogue()

    def read(self, signal):
        t, y = self._data[signal.name]
        return SignalData(t, y, signal.enum)

    def time_range(self):
        return (0.0, 19.99)

    def common_time_base(self):
        return np.unique(np.concatenate([t for t, _ in self._data.values()]))


TRAINING_VIEW = ("RoadType nobit 2\nSignType nobit 3\nDemoSpeedCode nobit 4\n"
              "v_x nobit 5 scmin=0 scmax=50\nDemoSpeedCode nobit 6 key=LimitTimes2 exp=x*2\n")


def _library(tmp_path) -> ValueMapLibrary:
    lib = ValueMapLibrary(tmp_path / "valuemaps.json", autoload=False)
    lib.tables["路况"] = ValueMap("路况", [(1, 1, "高速"), (2, 2, "乡村道路"), (3, 3, "城市")], default="未知")
    lib.tables["限速码"] = ValueMap("限速码", [(0, 0, "0"), (12, 12, "120"), (6, 6, "60")], unit="km/h")
    lib.assign("RoadType", "路况", "mapped")
    lib.assign("DemoSpeed*", "限速码", "both")
    return lib


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path / "settings.json")


@pytest.fixture
def training_window(qapp, settings, tmp_path):
    win = MainWindow(settings)
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


# ---------------------------------------------------------------------- value maps: data layer
@pytest.mark.parametrize("text, expected", [
    ("3", (3.0, 3.0)), (" -2.5 ", (-2.5, -2.5)), ("0x1F", (31.0, 31.0)), ("0b101", (5.0, 5.0)),
    ("10..20", (10.0, 20.0)), ("20...10", (10.0, 20.0)), ("1~3", (1.0, 3.0)), ("-5..-1", (-5.0, -1.0)),
    ("1e3", (1000.0, 1000.0)), (7, (7.0, 7.0)), ("*", None), ("default", None), ("其它", None), ("其他", None),
])
def test_raw_values_are_numbers_ranges_or_the_default(text, expected):
    assert parse_raw_value(text) == expected


@pytest.mark.parametrize("text", ["abc", "", "1..x", "nan", "inf", float("nan")])
def test_bad_raw_values_are_rejected_with_a_message(text):
    with pytest.raises(ValueError):
        parse_raw_value(text)


def test_value_map_texts_ranges_default_and_numbers():
    table = ValueMap("t", [(0, 0, "aus"), (1, 9, "klein"), (10, 10, "zehn")], default="sonst")
    assert table.text(0) == "aus" and table.text(5) == "klein" and table.text(10) == "zehn"
    assert table.text(99) == "sonst"
    assert not table.is_numeric                          # a range and texts: no numeric curve
    speed = ValueMap("speed", [(12, 12, "120"), (0, 0, "0"), (6, 6, "60")], unit="km/h")
    assert speed.numbers() == [(0.0, 0.0), (6.0, 60.0), (12.0, 120.0)]
    assert speed.expression() == "map(x,0,0,6,60,12,120)"
    with pytest.raises(ValueError):
        table.expression()


def test_library_round_trip_keeps_chinese_and_rules(tmp_path):
    lib = _library(tmp_path)
    lib.save()
    text = (tmp_path / "valuemaps.json").read_text(encoding="utf-8")
    assert "乡村道路" in text                             # written as text, not as \\u escapes
    again = ValueMapLibrary(tmp_path / "valuemaps.json")
    assert again.tables["路况"].rows == lib.tables["路况"].rows
    assert again.tables["路况"].default == "未知"
    assert [(a.pattern, a.table, a.mode) for a in again.assignments] == [("RoadType", "路况", "mapped"),
                                                                         ("DemoSpeed*", "限速码", "both")]


def test_damaged_library_file_gives_an_empty_library(tmp_path):
    path = tmp_path / "valuemaps.json"
    path.write_text("{not json", encoding="utf-8")
    assert ValueMapLibrary(path).tables == {}
    path.write_text(json.dumps({"tables": [{"name": "a", "rows": [[1, 1, "x"], ["bad"], [2, "y"]]}],
                                "assignments": [{"pattern": "s", "table": "missing"}, {"pattern": "t", "table": "a"}]}),
                    encoding="utf-8")
    lib = ValueMapLibrary(path)
    assert lib.tables["a"].rows == [(1.0, 1.0, "x")]
    assert [a.pattern for a in lib.assignments] == ["t"]   # a rule for an unknown table is dropped


def test_exact_names_win_over_wildcards_and_names_are_case_insensitive(tmp_path):
    lib = _library(tmp_path)
    lib.tables["other"] = ValueMap("other", [(1, 1, "x")])
    lib.assign("demospeedcode", "other", "raw")
    table, rule = lib.find("DemoSpeedCode")
    assert table.name == "other" and rule.mode == "raw"
    table, rule = lib.find("DemoSpeedCodeFront")
    assert table.name == "限速码"
    assert lib.find("Unknown") is None
    lib.assign("Mode only", "", "raw")                    # keeps the recording's own text table
    table, rule = lib.find("mode only")
    assert table is None and rule.mode == "raw"


def test_rename_and_remove_tables_update_the_rules(tmp_path):
    lib = _library(tmp_path)
    assert lib.rename_table("路况", "RoadTypeMap")
    assert lib.find("RoadType")[0].name == "RoadTypeMap"
    assert not lib.rename_table("RoadTypeMap", "限速码")   # name taken
    lib.remove_table("RoadTypeMap")
    assert lib.find("RoadType") is None


def test_csv_import_with_header_merged_names_ranges_and_default(tmp_path):
    path = tmp_path / "training.csv"
    path.write_text("信号,原始值,逻辑值\nRoadType,1,高速\n,2,乡村道路\n,3,城市\n,*,未知\n"
                    "SpeedCode,0,0\n,1..5,低速\n", encoding="utf-8")
    result = read_mapping_file(path)
    tables = {t.name: t for t in result.tables}
    assert tables["RoadType"].rows == [(1.0, 1.0, "高速"), (2.0, 2.0, "乡村道路"), (3.0, 3.0, "城市")]
    assert tables["RoadType"].default == "未知"
    assert tables["SpeedCode"].rows == [(0.0, 0.0, "0"), (1.0, 5.0, "低速")]
    assert [(a.pattern, a.table) for a in result.assignments] == [("RoadType", "RoadType"), ("SpeedCode", "SpeedCode")]
    assert result.problems == []


def test_gbk_csv_two_columns_without_header_and_bad_rows(tmp_path):
    path = tmp_path / "道路类型.txt"
    path.write_bytes("0;无效\n1;高速\nzwei;城市\n".encode("gbk"))
    result = read_mapping_file(path)
    assert len(result.tables) == 1 and result.tables[0].name == "道路类型"
    assert result.tables[0].rows == [(0.0, 0.0, "无效"), (1.0, 1.0, "高速")]
    assert len(result.problems) == 1 and "zwei" in result.problems[0]
    assert result.assignments == []                      # no signal column: the table is not assigned


def test_dbc_value_tables(tmp_path):
    path = tmp_path / "training.dbc"
    path.write_text('VERSION ""\nBO_ 100 DemoMsg: 8 Controller\n SG_ DemoLimit : 0|8@1+ (1,0) [0|255] "" X\n'
                    'VAL_TABLE_ YesNo 1 "Yes" 0 "No" ;\n'
                    'VAL_ 100 DemoLimit 0 "no limit" 12 "120 km/h" 6 "60 km/h" ;\n'
                    'VAL_ 200 DemoLimit 0 "off" ;\n', encoding="utf-8")
    result = read_mapping_file(path)
    names = [t.name for t in result.tables]
    assert names == ["YesNo", "DemoLimit", "DemoLimit@200"]
    assert result.tables[1].rows == [(0.0, 0.0, "no limit"), (6.0, 6.0, "60 km/h"), (12.0, 12.0, "120 km/h")]
    assert [(a.pattern, a.table) for a in result.assignments] == [("DemoLimit", "DemoLimit")]


def test_excel_import(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Training"
    for row in (["Signal", "Value", "Text"], ["RoadType", 1, "高速"], [None, 2, "城市"]):
        sheet.append(row)
    path = tmp_path / "maps.xlsx"
    book.save(path)
    result = read_mapping_file(path)
    assert result.tables[0].rows == [(1.0, 1.0, "高速"), (2.0, 2.0, "城市")]
    assert result.assignments[0].pattern == "RoadType"


def test_csv_export_reads_back_and_json_import(tmp_path):
    lib = _library(tmp_path)
    write_mapping_csv(tmp_path / "out.csv", list(lib.tables.values()), lib.assignments)
    back = read_mapping_file(tmp_path / "out.csv")
    tables = {t.name: t for t in back.tables}
    assert tables["路况"].rows == lib.tables["路况"].rows and tables["路况"].default == "未知"
    assert tables["限速码"].unit == "km/h"
    lib.save(tmp_path / "copy.json")
    again = read_mapping_file(tmp_path / "copy.json")
    assert {t.name for t in again.tables} == {"路况", "限速码"}
    with pytest.raises(ValueError):
        (tmp_path / "broken.json").write_text("[", encoding="utf-8")
        read_mapping_file(tmp_path / "broken.json")


def test_merge_keeps_how_a_signal_is_shown(tmp_path):
    lib = _library(tmp_path)
    lib.assign("RoadType", "路况", "raw", True)
    imported = MappingImport([ValueMap("路况", [(1, 1, "highway")])], [MapAssignment("RoadType", "路况")])
    assert lib.merge(imported) == (1, 1)
    table, rule = lib.find("RoadType")
    assert table.rows == [(1.0, 1.0, "highway")] and rule.mode == "raw" and rule.lane


# ---------------------------------------------------------------------- value maps in the display signals
def test_display_signals_take_tables_and_modes(tmp_path):
    signals = build_display_signals(TrainingTrace(), parse_plt(TRAINING_VIEW), valuemaps=_library(tmp_path))
    road, sign, code, speed, times2 = signals
    assert road.map_name == "路况" and road.format_value(2.0) == "乡村道路"
    assert road.format_value(9.0) == "未知"
    assert code.format_value(12.0) == "120 km/h (12)"   # mode "both", the table's unit
    assert sign.value_mode is None and sign.format_value(7.0) == "limit 120 (7)"   # the recording's own table
    assert speed.enum is None and speed.format_value(25.0) == "25"
    # a line that changes the values does not take the table of the recorded name
    assert times2.map_name == "" and times2.format_value(24.0) == "24"


def test_mode_raw_and_rule_for_the_display_name(tmp_path):
    lib = _library(tmp_path)
    lib.assign("SignType", "", "raw")
    lib.tables["x2"] = ValueMap("x2", [(24, 24, "zwei mal zwölf")])
    lib.assign("LimitTimes2", "x2", "mapped")
    signals = build_display_signals(TrainingTrace(), parse_plt(TRAINING_VIEW), valuemaps=lib)
    assert signals[1].format_value(7.0) == "7"
    assert signals[4].format_value(24.0) == "zwei mal zwölf"
    assert signals[0].state_text(3.0) == "城市"


def test_bit_field_lines_use_rules_for_their_token_only(tmp_path):
    lib = _library(tmp_path)
    lib.tables["bits"] = ValueMap("bits", [(1, 1, "set")])
    lib.assign("RoadType:0:1", "bits", "mapped")
    script = parse_plt("RoadType:0:1 nobit 2\nRoadType nobit 3\n")
    bit, whole = build_display_signals(TrainingTrace(), script, valuemaps=lib)
    assert bit.map_name == "bits" and whole.map_name == "路况"


def test_state_band_signals_leave_the_analog_window(tmp_path, qapp):
    lib = _library(tmp_path)
    lib.assign("RoadType", "路况", "mapped", lane=True)
    lib.assign("SignType", "", "mapped", lane=True)
    lib.assign("DemoSpeedCode", "限速码", "mapped", lane=True)
    signals = build_display_signals(TrainingTrace(), parse_plt(TRAINING_VIEW), valuemaps=lib)
    assert signals[0].lane and not signals[0].in_analog and signals[3].in_analog
    state = ViewState(x0=0, x1=20)
    plain = compute_layout(QtCore.QRectF(0, 0, 1000, 600), [s for s in signals if not s.lane], state)
    banded = compute_layout(QtCore.QRectF(0, 0, 1000, 600), signals, state)
    assert banded.digital.height() > plain.digital.height()    # room for the labels
    assert banded.analog.height() > 300


def test_state_segments_exact_and_decimated():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.0, 2.0, np.nan, 3.0])
    start, end, value, busy = state_segments(t, y, 0.5, 4.5, 800)
    assert start.tolist() == [0.5, 2.0, 3.0, 4.0] and end.tolist() == [2.0, 3.0, 4.0, 4.5]
    assert value[0] == 1.0 and value[1] == 2.0 and np.isnan(value[2]) and value[3] == 3.0
    assert not busy.any()
    tt = np.arange(0.0, 10.0, 0.001)
    yy = (np.arange(len(tt)) // 3 % 2).astype(float)        # changes every 3 ms: far more than pixels
    start, end, value, busy = state_segments(tt, yy, 0.0, 10.0, 200)
    assert len(start) <= 200 and busy.all()
    assert abs(end[-1] - 10.0) < 1e-9


def test_state_colours_are_stable():
    sig = DisplaySignal(entry=PltEntry(name="s"), enum=EnumTable(values={1.0: "a", 2.0: "b", 3.0: "c"}))
    assert state_fill(sig, 2.0) == state_fill(sig, 2.0) != state_fill(sig, 3.0)
    assert state_fill(sig, 0.0) == (217, 217, 217)


def test_signal_pattern_of_lines():
    road = DisplaySignal(entry=PltEntry(name="RoadType\\Sensor:1"), info=SignalInfo("RoadType", source_name="RoadType\\Sensor:1"))
    assert signal_pattern(road) == "RoadType"
    calc = DisplaySignal(entry=parse_plt("RoadType nobit 1 key=R2 exp=x*2").items[0])
    assert signal_pattern(calc) == "R2"


# ---------------------------------------------------------------------- the window: value mapping, bands, XY
def test_legend_shows_mapped_values_and_the_mode_can_be_switched(training_window, qapp):
    win = training_window
    view = win.view
    view.set_reference_line(True)
    lay = view._lay()
    x = lay.analog.left() + lay.analog.width() * (8.0 / 19.99)
    view._mouse = QtCore.QPointF(x, lay.analog.center().y())
    assert "乡村道路" in view.values_text()
    win.set_value_mode(0, "raw")
    assert win.signals[0].value_mode == "raw"
    assert "乡村道路" not in win.view.values_text()
    assert ValueMapLibrary(win.valuemaps.path).exact("RoadType").mode == "raw"    # saved at once


def test_state_band_from_the_legend_menu_function(training_window, qapp):
    win = training_window
    win.set_state_band(0, True)
    assert win.signals[0].lane and win.signals[0].value_mode == "mapped"
    image = win.view.grab_image()
    assert not image.isNull()
    win.set_state_band(0, False)
    assert not win.signals[0].lane


def test_value_map_dialog_edits_a_copy(training_window, qapp):
    win = training_window
    dialog = ValueMapDialog(win.valuemaps, win.signals, 2, win)
    assert dialog.pattern_edit.text() == "DemoSpeed*"          # the signal follows a wildcard rule
    assert dialog.use_combo.currentData() == "限速码"
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("mapped"))
    assert dialog.result_library().find("DemoSpeedCode")[1].mode == "mapped"
    assert win.valuemaps.find("DemoSpeedCode")[1].mode == "both"         # the original is untouched
    # edit a row of the shown table in Chinese
    items = dialog.table_list.findItems("限速码", QtCore.Qt.MatchExactly)
    dialog.table_list.setCurrentItem(items[0])
    dialog.rows.item(0, 1).setText("无限速")
    dialog._store_table()
    assert (0.0, 0.0, "无限速") in dialog.result_library().tables["限速码"].rows
    dialog.reject()


def test_new_table_from_the_values_of_a_signal(training_window, qapp, monkeypatch):
    win = training_window
    dialog = ValueMapDialog(win.valuemaps, win.signals, 1, win)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("标志类型", True))
    dialog._table_from_values()
    table = dialog.result_library().tables["标志类型"]
    assert table.rows == [(0.0, 0.0, "none"), (7.0, 7.0, "limit 120")]   # texts from the recording's table
    assert dialog.result_library().find("SignType")[0].name == "标志类型"
    dialog.reject()


def test_numeric_signal_from_a_table(training_window, qapp):
    win = training_window
    entry = win.add_mapped_signal(2, "限速码")
    assert entry is not None and entry.key == "DemoSpeedCode_限速码" and entry.unit == "km/h"
    sig = next(s for s in win.signals if s.entry is entry)
    assert sig.error is None
    assert sig.value_at(5.0) == 120.0 and sig.value_at(1.0) == 0.0
    items = win.script.items
    assert items.index(entry) == items.index(win.signals[2].entry) + 1


def test_xy_view(training_window, qapp):
    win = training_window
    win.act_xy.setChecked(True)
    win._toggle_xy()
    assert win.view.is_xy() and not win.act_zoom_x.isEnabled()
    assert xy_axis_index(win.signals) == 0
    xs, ys = xy_curve(win.signals[3], win.signals[2], 0.0, 20.0)
    assert len(xs) == len(ys) == len(win.signals[3].t)      # DemoSpeedCode shares the 10 ms raster of v_x
    win.view.grab()
    win.view.start_zoom("x")
    assert win.view._mode == ""                              # no zoom in the XY view
    win.act_xy.setChecked(False)
    win._toggle_xy()
    assert not win.view.is_xy() and win.act_zoom_x.isEnabled()


def test_xy_hover_shows_the_time_of_the_sample_under_the_mouse(training_window, qapp):
    from tracepaddock.ui.render import xy_pixels
    win = training_window
    win.act_xy.setChecked(True)
    win._toggle_xy()
    view = win.view
    lay = view._lay()
    px, py, times = xy_pixels(lay, win.signals[0], win.signals[1], view.state)
    assert len(px) == len(py) == len(times) > 0 and times[0] >= 0.0
    k = len(px) // 2
    view._mouse = QtCore.QPointF(px[k], py[k])
    hit = view._xy_nearest()
    assert hit is not None and (hit[0], hit[1]) == (px[k], py[k]) and hit[2] in times
    view._emit_status()
    view.grab()
    win._set_option("xy_points", True)
    win._set_option("xy_time_color", False)
    assert view.state.xy_points and not view.state.xy_time_color
    view.grab()


def test_xy_view_needs_two_analog_signals(qapp, settings, monkeypatch):
    win = MainWindow(settings)
    win.dataset = TrainingTrace()
    win.script = parse_plt("v_x nobit 5\n")
    win.rebuild(keep_view=False)
    errors = []
    monkeypatch.setattr(win, "_error", errors.append)
    win.act_xy.setChecked(True)
    win._toggle_xy()
    assert not win.view.is_xy() and errors
    win.close()


def test_xy_curve_holds_both_signals():
    a = DisplaySignal(entry=PltEntry(name="a"), t=np.array([0.0, 1.0, 2.0]), y=np.array([10.0, 20.0, 30.0]))
    b = DisplaySignal(entry=PltEntry(name="b"), t=np.array([0.5, 1.5]), y=np.array([1.0, 2.0]))
    xs, ys = xy_curve(a, b, 0.0, 2.0)
    assert xs.tolist() == [10.0, 10.0, 20.0, 20.0, 30.0] and ys.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]


def test_time_range_dialog(qapp):
    dialog = TimeRangeDialog((0.0, 100.0), (10.0, 20.0))
    assert dialog.values() == (10.0, 20.0)
    dialog.t1.setValue(5.0)
    dialog.accept()
    assert dialog.result() != QtWidgets.QDialog.Accepted and dialog.note.text()
    dialog._everything()
    assert dialog.values() == (0.0, 100.0)


# ---------------------------------------------------------------------- export
def _sig(name, t, y, unit="", enum=None):
    return DisplaySignal(entry=PltEntry(name=name), name=name, unit=unit, t=np.asarray(t, float),
                         y=np.asarray(y, float), enum=enum)


def test_csv_union_axis_holds_values_and_writes_texts(tmp_path):
    a = _sig("车速", [0.0, 0.1, 0.2], [1.0, 2.5, 3.0], "km/h")
    b = _sig("Road", [0.05, 0.15], [1.0, 2.0], enum=EnumTable(values={1.0: "高速", 2.0: "城市"}))
    rows = write_csv(tmp_path / "a.csv", [a, b], 0.0, 0.2, text_values="mapped")
    lines = (tmp_path / "a.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows == 5
    assert lines[:2] == ["TIME;车速;Road", "s;km/h;"]
    assert lines[2:] == ["0;1;", "0.05;1;高速", "0.1;2.5;高速", "0.15;2.5;城市", "0.2;3;城市"]


def test_csv_fixed_step_decimal_comma_and_quotes(tmp_path):
    a = _sig("a;b", [0.0, 1.0], [0.5, 1.5])
    write_csv(tmp_path / "b.txt", [a], 0.0, 1.0, step=0.5, separator="\t", decimal=",", header=True)
    lines = (tmp_path / "b.txt").read_text(encoding="utf-8-sig").splitlines()
    assert lines == ["TIME\ta;b", "s\t", "0\t0,5", "0,5\t0,5", "1\t1,5"]
    write_csv(tmp_path / "c.csv", [a], 0.0, 1.0, header=True)
    assert (tmp_path / "c.csv").read_text(encoding="utf-8-sig").splitlines()[0] == 'TIME;"a;b"'
    with pytest.raises(ExportError):
        write_csv(tmp_path / "d.csv", [a], 0.0, 1.0, separator=",", decimal=",")
    with pytest.raises(ExportError):
        write_csv(tmp_path / "e.csv", [a], 5.0, 6.0)
    with pytest.raises(ExportError):
        export_axis([a], 0.0, 1.0, step=0.0)


def test_held_values_are_empty_before_the_first_sample():
    a = _sig("a", [1.0, 2.0], [5.0, 6.0])
    out = held_values(a, np.array([0.5, 1.0, 1.5, 3.0]))
    assert np.isnan(out[0]) and out[1:].tolist() == [5.0, 5.0, 6.0]


def test_mat_export_own_samples_and_common_axis(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    a = _sig("1st.speed", [0.0, 0.1, 0.2], [1.0, 2.0, 3.0], "m/s")
    b = _sig("1st speed", [0.05], [7.0])
    write_mat(tmp_path / "x.mat", [a, b], 0.0, 0.2)
    m = scipy_io.loadmat(tmp_path / "x.mat")
    assert m["xst_speed"].tolist() == [[0.0, 1.0], [0.1, 2.0], [0.2, 3.0]]
    assert m["xst_speed_2"].tolist() == [[0.05, 7.0]]
    write_mat(tmp_path / "y.mat", [a, b], 0.0, 0.2, common_axis=True)
    m = scipy_io.loadmat(tmp_path / "y.mat")
    assert m["q_T0"].ravel().tolist() == [0.0, 0.05, 0.1, 0.2]
    assert np.isnan(m["xst_speed_2"][0, 0]) and m["xst_speed_2"].ravel()[1:].tolist() == [7.0, 7.0, 7.0]


def test_matlab_names_are_sanitized():
    assert matlab_names(["1abc", "a.b", "a_b", "x" * 80]) == ["xabc", "a_b", "a_b_2", "x" * 63]


def test_mf4_export_keeps_text_tables_and_mdf3_export(tmp_path):
    road = _sig("Road", [0.0, 1.0], [1.0, 2.0], "-", EnumTable(values={1.0: "高速", 2.0: "城市"}, default="其它"))
    speed = _sig("Speed", [0.0, 0.5, 1.0], [1.0, 2.0, 3.0], "km/h")
    assert write_mdf(tmp_path / "x.mf4", [road, speed], 0.0, 1.0) == 2
    with open_dataset(tmp_path / "x.mf4") as ds:
        data = ds.read(ds.find("Road"))
        assert data.enum.values == {1.0: "高速", 2.0: "城市"} and data.enum.default == "其它"
        assert ds.read(ds.find("Speed")).y.tolist() == [1.0, 2.0, 3.0]
    assert write_mdf(tmp_path / "x.mdf", [road, speed], 0.5, 1.0, version="3.30") == 2
    with open_dataset(tmp_path / "x.mdf") as ds:
        assert ds.read(ds.find("Speed")).t.tolist() == [0.5, 1.0]


def test_export_from_the_window(training_window, tmp_path, qapp):
    win = training_window
    dialog = ExportDialog(win.signals, 0, win.dataset.time_range(), (2.0, 6.0), win.settings,
                          str(tmp_path / "training_export.csv"), win)
    dialog.rb_view.setChecked(True)
    dialog.rb_step.setChecked(True)
    dialog.step_box.setValue(500.0)
    dialog.text_combo.setCurrentIndex(dialog.text_combo.findData("mapped"))
    options = dialog.options()
    assert options["range"] == (2.0, 6.0) and options["step"] == 0.5
    win.write_export(options)
    lines = (tmp_path / "training_export.csv").read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].split(";")[:3] == ["TIME", "RoadType", "SignType"]
    assert lines[2].split(";")[1] == "高速" and lines[-1].split(";")[1] == "乡村道路"
    assert len(lines) == 2 + 9
    dialog.reject()


def test_pictures(training_window, tmp_path, qapp):
    win = training_window
    state = picture_state(win.view.state, True)
    assert state.background == (255, 255, 255) and win.view.state.background != (255, 255, 255)
    for ext in (".png", ".pdf", ".svg"):
        path = tmp_path / ("view" + ext)
        save_picture(path, win.view.size(), win.signals, state, ["a", "b"])
        assert path.stat().st_size > 1000


# ---------------------------------------------------------------------- script history, favourites, settings
def _scripts(tmp_path):
    first, second = tmp_path / "first.plt", tmp_path / "second.plt"
    first.write_text("RoadType nobit 2\n", encoding="utf-8")
    second.write_text("v_x nobit 5\nSignType nobit 3\n", encoding="utf-8")
    return first, second


def test_script_history_back_next_and_key_zero(training_window, tmp_path, qapp):
    win = training_window
    win.script.modified = False
    first, second = _scripts(tmp_path)
    assert win.open_plt(first) and win.open_plt(second)
    assert [s.name for s in win.signals] == ["v_x", "SignType"]
    assert win.act_back.isEnabled() and not win.act_forward.isEnabled()
    win.history_step(-1)
    assert [s.name for s in win.signals] == ["RoadType"] and win.act_forward.isEnabled()
    win.history_step(1)
    assert [s.name for s in win.signals] == ["v_x", "SignType"]
    win.toggle_last_script()                                # key 0
    assert [s.name for s in win.signals] == ["RoadType"]
    win.toggle_last_script()
    assert [s.name for s in win.signals] == ["v_x", "SignType"]
    assert win.settings.get("plt_history")[-2:] == [str(first), str(second)]
    win._fill_history()                                     # the menu can be built
    win.clear_history()
    assert win.settings.get("plt_history") == [str(second)]


def test_loading_another_script_asks_about_changes(training_window, tmp_path, qapp, monkeypatch):
    win = training_window
    first, _second = _scripts(tmp_path)
    win.script.modified = True
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.Cancel)
    assert not win.open_plt(first)
    assert len(win.signals) == 5


def test_favourites(training_window, tmp_path, qapp):
    win = training_window
    first, second = _scripts(tmp_path)
    win.script.modified = False
    win.open_plt(first)
    win._add_current_favourite()
    win._add_current_favourite()
    assert win.settings.get("favourite_plts") == [str(first)]
    dialog = FavouritesDialog([str(first)], str(second), str(tmp_path))
    dialog._add([str(second), str(first)])
    dialog.list.setCurrentRow(1)
    dialog._move(-1)
    assert dialog.paths() == [str(second), str(first)]
    win._fill_favourites()


def test_settings_dialog_switches_processing_to_the_common_time_axis(training_window, qapp, monkeypatch):
    win = training_window
    win.script = parse_plt("RoadType nobit 5 key=d diff\n")        # 100 ms raster, common axis 10 ms
    win.rebuild()
    own_samples = len(win.signals[0].t)
    dialog = SettingsDialog(win.settings, win)
    dialog.checks["ops_on_common_time_base"].setChecked(True)
    dialog.keep_spin.setValue(2)
    dialog.accept()
    assert "ops_on_common_time_base" in dialog.changed_keys()
    assert win.settings.get("short_name_keep") == 2
    assert dialog.result.text() == "DemoRig.Thermal"
    win.rebuild()
    assert len(win.signals[0].t) == len(win.dataset.common_time_base()) > own_samples


def test_continuation_lines_keep_the_operations_in_order():
    script = parse_plt(
        "BenchEnabled nobit 2 key=Derived\n+ diff\n+ mavr(-3)\n+ integ\n"
    )
    entry = next(script.entries())
    assert entry.display_name == "Derived"
    assert [op.text for op in entry.ops] == ["diff", "mavr(-3)", "integ"]


def test_calculus_uses_the_common_recording_raster_when_requested(mf4_file):
    with open_dataset(mf4_file) as ds:
        script = parse_plt("BenchEnabled nobit 2 key=Integrated integ\n")
        signal = build_display_signals(ds, script, {"ops_on_common_time_base": True})[0]
        np.testing.assert_array_equal(signal.t, [0., 1., 2., 3., 4.])
        np.testing.assert_array_equal(signal.y, [0., 0., 0., 1., 2.])


def test_mf4_text_table_with_a_numeric_default_conversion(tmp_path):
    """CAN databases written to MF4: 'raw 8190 = Abnormal, 8191 = Invalid, else 0.05625 * raw km/h'."""
    from asammdf import MDF, Signal
    conversion = {"val_0": 8190, "text_0": "Abnormal", "val_1": 8191, "text_1": "Invalid value",
                  "default_addr": {"a": 0.05625, "b": 0.0, "unit": "km/h"}}
    mdf = MDF(version="4.10")
    mdf.append([Signal(samples=np.array([1600, 1658, 8191], dtype=np.uint16), timestamps=np.array([0.0, 0.1, 0.2]),
                       name="VehicleSpeed", conversion=conversion)])
    mdf.save(str(tmp_path / "speed.mf4"), overwrite=True)
    mdf.close()
    with open_dataset(tmp_path / "speed.mf4") as ds:
        info = ds.find("VehicleSpeed")
        data = ds.read(info)
        assert data.y.tolist() == pytest.approx([90.0, 93.2625, 8191 * 0.05625])
        assert data.enum.text(8191 * 0.05625) == "Invalid value"
        assert info.unit == "km/h"


# ---------------------------------------------------------------------- Chinese texts
def _literal_tr_strings():
    out = {}
    for path in sorted((ROOT / "tracepaddock").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "tr" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.setdefault(arg.value, path.name)
    return out


_NOT_TRANSLATED = {"<", ">", "<<", ">>", "PLT <", "PLT >", "XY", "Tab", "%g px", "0", "C", "E", "H", "I", "Q", "S", "X",
                   "Ctrl+O", "Ctrl+S", "Ctrl+C", "Ctrl+P", "ESC", "F11", "SPACE", "Z / Y / U", "N / P",
                   "L R T B, RETURN", "-"}


def test_every_text_of_the_interface_has_a_chinese_translation():
    missing = sorted(text for text, where in _literal_tr_strings().items()
                     if text not in ZH and text not in _NOT_TRANSLATED)
    assert missing == []


def test_menu_texts_of_the_window_are_translated(qapp, settings):
    from tracepaddock.i18n import set_language
    set_language("zh")
    try:
        win = MainWindow(settings)
        texts = [a.text() for a in win.findChildren(QtGui.QAction) if a.text()]
        english = [t for t in texts if t.split("\t")[0].isascii() and any(c.isalpha() for c in t)
                   and t.split("\t")[0] not in ("English", "XY") and not t.startswith(("%", "1 ", "2 "))
                   and "px" not in t]
        assert english == []
        win.close()
    finally:
        set_language("en")
