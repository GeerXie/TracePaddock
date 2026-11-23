"""Stage 4: rule scripts and plug-ins, signal databases, pictures, calculation raster, rule UI, Excel report."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from tracepaddock.candb import SignalDatabase, parse_value_description
from tracepaddock.core.export import write_mdf
from tracepaddock.display import DisplaySignal
from tracepaddock.plt.model import PltEntry
from tracepaddock.plt.parser import parse_plt
from tracepaddock.plt.process import build_display_signals
from tracepaddock.rules.engine import SignalSource, evaluate
from tracepaddock.rules.model import Condition, Group, Operand, Rule, RuleSet
from tracepaddock.rules.plugins import builtin_folder, discover, read_plugin
from tracepaddock.rules.report import RunResult, write_xlsx
from tracepaddock.rules.runner import JobError, run_job
from tracepaddock.rules.script import ScriptApi, ScriptError, Sig
from tracepaddock.signs import auto_speed_spec, sign_pixmap
from tracepaddock.valuemap import ValueMap, ValueMapLibrary, read_mapping_file

from test_stage3 import TrainingTrace, TRAINING_VIEW, _library


# ---------------------------------------------------------------------- script vocabulary
def test_signal_expressions_become_conditions():
    v, lim = Sig("v"), Sig("lim")
    c = v > lim + 3
    assert (c.signal, c.op, c.value.signal, c.value.scale, c.value.offset) == ("v", ">", "lim", 1.0, 3.0)
    c = v * 3.6 >= 90
    assert (c.signal, c.op, c.value, c.scale) == ("v", ">=", 90.0, 3.6)
    c = -v < lim                                   # -v < lim  <=>  v > -lim
    assert c.op == ">" and c.value.scale == -1.0
    c = (v * 2 + 1) == (lim * 4)                   # 2v + 1 == 4 lim  <=>  v == 2 lim - 0.5
    assert c.value.scale == 2.0 and c.value.offset == -0.5
    g = (v > 1) & ((lim < 2) | ~(v == 5))
    assert isinstance(g, Group) and g.mode == "and" and g.items[1].mode == "or" and g.items[1].items[1].negate
    assert v.isin([1, 2]).op == "in" and v.changes_from(1, 2).previous == 1.0
    assert v.equals(lim, tol=0.5).tolerance == 0.5
    with pytest.raises(ScriptError):
        v + lim
    with pytest.raises(ScriptError):
        v == [1, 2]


def test_script_api_builds_rules_and_events():
    t = np.arange(0.0, 5.0, 1.0)
    api = ScriptApi(lambda name: (t, t * 2) if name == "a" else None, "demo", {"k": 1})
    ns = api.namespace()
    exec("r = rule('x', trigger=becomes(S('a') > 3, hold=0.5), expect=S('a') >= 4, within=(0.1, 2))\n"
         "tt, yy = data('a')\nevent(tt[2], 'two', verdict='pass', a=float(yy[2]))\nlog('n', len(tt))", ns)
    rule = api.rules[0]
    assert (rule.trigger_mode, rule.trigger_hold, rule.within_min, rule.within_max) == ("becomes_true", 0.5, 0.1, 2.0)
    assert rule.origin == "script:demo"
    assert api.events[0].verdict == "PASS" and api.events[0].values == {"a": 4.0} and api.messages == ["n 5"]
    with pytest.raises(ScriptError):
        api.data("missing")
    with pytest.raises(ScriptError):
        api.event(1.0, verdict="maybe")


# ---------------------------------------------------------------------- runner process
@pytest.fixture(scope="module")
def training_mf4(tmp_path_factory):
    folder = tmp_path_factory.mktemp("training")
    ds = TrainingTrace()
    signals = [DisplaySignal(entry=PltEntry(name=s.name), name=s.name, unit=s.unit, t=d.t, y=d.y)
               for s in ds.signals for d in [ds.read(s)]]
    path = folder / "training.mf4"
    write_mdf(path, signals, 0.0, 20.0)
    return path


def test_runner_runs_a_script_on_the_recording(training_mf4):
    script = ("rule('limit', trigger=becomes(S('SignType') == 7), expect=S('DemoSpeedCode') == 12, within=0.5,\n"
              "     record=['RoadType'])\nlog('done')")
    result = run_job({"dataset": str(training_mf4), "plt_text": TRAINING_VIEW, "script": script, "script_name": "s"}, timeout=90)
    assert result["error"] is None and result["log"] == ["done"]
    events = result["results"][0]["events"]
    assert [(e["t0"], e["verdict"], round(e["delay"], 3)) for e in events] == [(4.0, "PASS", 0.35)]
    assert events[0]["values"]["RoadType"] == 1.0


def test_runner_reports_the_line_of_a_mistake(training_mf4):
    result = run_job({"dataset": str(training_mf4), "script": "a = 1\nb = S('x') + S('y')\n"}, timeout=90)
    assert "line 2: b = S('x') + S('y')" in result["error"] and "ScriptError" in result["error"]
    result = run_job({"dataset": str(training_mf4), "script": "rule('x', trigger=\n"}, timeout=90)
    assert "SyntaxError" in result["error"]


def test_runner_stops_a_script_that_does_not_end(training_mf4):
    started = time.perf_counter()
    with pytest.raises(JobError) as info:
        run_job({"dataset": str(training_mf4), "script": "while True:\n    pass\n"}, timeout=4)
    assert "4 s" in str(info.value) and time.perf_counter() - started < 30


def test_plugins_are_found_and_run(training_mf4, tmp_path):
    names = {p.name: p for p in discover()}
    assert {"State transitions", "Response timing"} <= set(names)
    transitions = names["State transitions"]
    assert transitions.problem is None and names["Response timing"].problem is None
    result = run_job({"dataset": str(training_mf4), "plugin": str(transitions.path), "params": {"signal": "RoadType", "record": []},
                      "script_name": transitions.name}, timeout=90)
    assert result["error"] is None
    assert [round(e["t0"], 3) for e in result["events"]] == [5.0, 12.0]
    bad = tmp_path / "broken.py"
    bad.write_text("NAME = 'x'\nPARAMETERS = [{'name': 'a', 'type': 'colour'}]\n", encoding="utf-8")
    plugin = read_plugin(bad)
    assert plugin.problem and "colour" in plugin.problem
    (tmp_path / "norun.py").write_text("NAME = 'y'\n", encoding="utf-8")
    assert "run" in read_plugin(tmp_path / "norun.py").problem


# ---------------------------------------------------------------------- signal databases
DBC = '''VERSION ""
BO_ 340 BenchInputs: 8 TrainingSender
 SG_ BenchLeverPosition : 8|8@1+ (1,0) [0|100] "%"  TrainingReceiver
 SG_ BenchLeverTemperature : 28|1@1+ (1,0) [0|1] ""  TrainingReceiver
BO_ 372 BenchSpeed: 8 TrainingStation
 SG_ ShaftSpeed : 0|13@1+ (0.125,0) [0|1023.75] "rpm"  TrainingReceiver
BA_DEF_ BO_  "GenMsgCycleTime" INT 0 65535;
BA_DEF_ BO_  "GenMsgSendType" ENUM  "Cyclic","Event";
BA_DEF_DEF_  "GenMsgCycleTime" 0;
BA_DEF_DEF_  "GenMsgSendType" "Cyclic";
BA_ "GenMsgCycleTime" BO_ 340 40;
BA_ "GenMsgCycleTime" BO_ 372 15;
CM_ SG_ 340 BenchLeverPosition "Bench lever opening";
VAL_ 340 BenchLeverTemperature 0 "Not high" 1 "Temperature too high" ;
VAL_ 372 ShaftSpeed 8191 "Invalid" ;
'''


def test_dbc_database(tmp_path):
    path = tmp_path / "car.dbc"
    path.write_text(DBC, encoding="utf-8")
    db = SignalDatabase()
    assert db.load(path) == 3
    speed = db.find("CAN1.BenchSpeed.ShaftSpeed")
    assert speed.cycle_ms == 15.0 and speed.factor == 0.125 and speed.unit == "rpm"
    assert speed.tolerance_hint() == pytest.approx(0.0625)
    assert db.find("ShaftSpeed\\Sensor:1") is speed
    temp = db.find("BenchLeverTemperature")
    assert temp.values == {0.0: "Not high", 1.0: "Temperature too high"} and temp.cycle_ms == 40.0
    assert db.find("BenchLeverPosition").description == "Bench lever opening"
    imported = db.to_mapping_import()
    tables = {t.name: t for t in imported.tables}
    assert tables["ShaftSpeed"].rows == [(1023.875, 1023.875, "Invalid")]
    assert [s.name for s in db.search("lever")] == ["BenchLeverPosition", "BenchLeverTemperature"]


def test_excel_can_matrix_with_chinese_titles(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["版本说明"])
    sheet.append(["报文名称", "报文ID", "周期", "信号名称", "信号描述", "精度", "偏移量", "最小值", "最大值", "单位", "值描述"])
    sheet.append(["TrainingFrame", "0x2A0", 100, "DemoLimit", "限速输出", 5, 0, 0, 150, "km/h", "0x0: 无限速; 0x1E: 无效"])
    sheet.append([None, None, None, "DemoFlag", "", 1, 0, 0, 1, "", "'0x0': 'Off', '0x1': 'On'"])
    path = tmp_path / "matrix.xlsx"
    book.save(path)
    db = SignalDatabase()
    assert db.load(path) == 2
    limit = db.find("DemoLimit")
    assert (limit.message, limit.message_id, limit.cycle_ms, limit.factor, limit.unit) == ("TrainingFrame", 0x2A0, 100.0, 5.0, "km/h")
    assert limit.values == {0.0: "无限速", 30.0: "无效"}
    assert limit.physical_values() == {0.0: "无限速", 150.0: "无效"}
    assert db.find("DemoFlag").message == "TrainingFrame" and db.find("DemoFlag").values == {0.0: "Off", 1.0: "On"}


def test_value_descriptions():
    assert parse_value_description("0=Off;1=On; 2 = Fault") == {0.0: "Off", 1.0: "On", 2.0: "Fault"}
    assert parse_value_description('0x0: "Not high"\n0x1: "Too high"') == {0.0: "Not high", 1.0: "Too high"}
    assert parse_value_description(None) == {} and parse_value_description("n/a") == {}


def test_bad_database_files_give_a_message(tmp_path):
    from tracepaddock.candb import CanDbError
    empty = tmp_path / "empty.csv"
    empty.write_text("a;b\n1;2\n", encoding="utf-8")
    with pytest.raises(CanDbError):
        SignalDatabase().load(empty)


# ---------------------------------------------------------------------- pictures
def test_builtin_signs(qapp):
    for spec in ("speed:50", "speed:120", "end", "end:80", "warn", "warn:限", "none"):
        pix = sign_pixmap(spec, 64)
        assert pix is not None and pix.width() == 64 and pix.height() == 64
    cond = sign_pixmap("cond:60:雨天", 64)
    assert cond.height() > 64
    assert sign_pixmap("C:/no/such/picture.png", 64) is None
    assert auto_speed_spec(80.4) == "speed:80" and auto_speed_spec(0) == "end" and auto_speed_spec(float("nan")) is None


def test_pictures_of_value_maps(tmp_path, qapp):
    table = ValueMap("sign", [(1, 1, "非条件"), (2, 2, "条件"), (5, 9, "其它限速")], images={"1": "speed:80", "5..9": "warn",
                                                                                     "*": "none"})
    assert table.image_for(1) == "speed:80" and table.image_for(2) is None and table.image_for(7) == "warn"
    assert table.image_for(42) == "none"
    speed = ValueMap("code", [(0, 0, "0"), (12, 12, "120")], unit="km/h", auto_image="speed")
    assert speed.image_for(12) == "speed:120" and speed.image_for(0) == "end" and speed.image_for(3) == "speed:3"
    back = ValueMap.from_json(table.to_json())
    assert back.images == table.images
    path = tmp_path / "map.csv"
    path.write_text("信号,原始值,逻辑值,图片\nSign,1,非条件,speed:80\n,*,其它,none\n", encoding="utf-8")
    imported = read_mapping_file(path).tables[0]
    assert imported.images == {"1": "speed:80", "*": "none"}


def test_display_signals_carry_their_pictures(tmp_path, qapp):
    lib = _library(tmp_path)
    lib.tables["限速码"].auto_image = "speed"
    signals = build_display_signals(TrainingTrace(), parse_plt(TRAINING_VIEW), valuemaps=lib)
    code = signals[2]
    assert code.image_for(12.0) == "speed:120" and signals[3].image_for(25.0) is None


# ---------------------------------------------------------------------- calculation raster
def test_calculation_raster_of_formulas():
    script = ("RoadType nobit 2 key=fixed exp(x1=DemoSpeedCode)=x+x1  // @raster=fixed:0.5\n"
              "RoadType nobit 3 key=follow exp(x1=DemoSpeedCode)=x+x1  // @raster=signal:x\n"
              "RoadType nobit 4 key=union exp(x1=DemoSpeedCode)=x+x1\n"
              "RoadType nobit 5 key=bad exp=x*2  // @raster=sometimes\n")
    fixed, follow, union, bad = build_display_signals(TrainingTrace(), parse_plt(script))
    assert np.allclose(np.diff(fixed.t), 0.5) and fixed.t[0] == 0.0
    assert len(follow.t) == 200 and len(union.t) == 2000
    assert bad.error and "@raster" in bad.error


# ---------------------------------------------------------------------- rule form, logic chain
def test_rule_form_round_trip(qapp):
    from tracepaddock.ui.rule_form import RuleFormDialog, parse_value
    rule = Rule("Training", trigger=Group("or", [Condition("SignType", "in", [1.0, 2.0]), Condition("Road", "==", 3.0)]),
                expect=Condition("Out", "==", Operand("Value", 3.6, 1.0), tolerance=0.5), within_min=0.1,
                within_max=0.8, already="new_sample", trigger_hold=0.2, scope=Condition("On", "==", 1.0),
                record=["Road", "Day"], note="n")
    enums = {"SignType": {1.0: "非条件限速", 2.0: "条件限速"}}
    dialog = RuleFormDialog(rule, ["SignType", "Road", "Out", "Value", "On", "Day"], lambda n: enums.get(n, {}),
                            lambda n: "0.5|hint", lambda n: 0.1)
    dialog.accept()
    assert dialog.rule().to_json() == rule.to_json()
    assert parse_value("in", "1 = 非条件限速, 条件限速", enums["SignType"]) == ([1.0, 2.0], None)
    assert parse_value("changes from", "1 -> 2") == (2.0, 1.0)
    assert parse_value("between", "10..20") == ([10.0, 20.0], None)
    with pytest.raises(ValueError):
        parse_value("==", "fast")
    with pytest.raises(ValueError):
        parse_value("in", "=Other")


def test_rule_form_validates(qapp):
    from tracepaddock.ui.rule_form import RuleFormDialog
    dialog = RuleFormDialog(None, ["a"], lambda n: {}, lambda n: "", lambda n: None, taken=["New rule"])
    dialog.accept()
    assert dialog.rule() is None and dialog.error.text()           # the name is taken
    dialog.name.setText("x")
    dialog.trigger.table.cellWidget(0, 2).setCurrentText("not a number")
    dialog.accept()
    assert dialog.rule() is None and "row 1" in dialog.error.text()


def test_logic_chain(qapp):
    from tracepaddock.ui.rule_graph import RuleGraphView, build_graph
    rule = Rule("Training", trigger=Group("and", [Condition("SignType", "in", [1.0]), Condition("Road", "==", 2.0)]),
                expect=Condition("Out", "==", Operand("Value")), scope=Condition("On", "==", 1.0))
    nodes, edges = build_graph(rule, {"events": 5, "PASS": 4, "FAIL": 1, "delay_min": 0.1, "delay_max": 0.4})
    kinds = [n["kind"] for n in nodes]
    assert kinds.count("signal") == 5 and kinds.count("condition") == 4 and "verdict_bad" in kinds
    assert any("4" in n["text"] and "1" in n["text"] for n in nodes if n["kind"] == "verdict_bad")
    view = RuleGraphView()
    view.show_rule(rule)
    assert len(view._items) == len(nodes)
    assert not view.picture().isNull()


def test_logic_chain_nodes_do_not_overlap_with_long_texts(qapp):
    from tracepaddock.ui.rule_graph import RuleGraphView, build_graph
    texts = {"SignType": {1.0: "非条件限速", 2.0: "条件限速（雨天）", 3.0: "隐形限速", 4.0: "限速结束"}}
    rule = Rule("Training", trigger=Group("and", [Condition("SignType", "in", [1.0, 2.0, 3.0, 4.0]),
                                             Condition("RoadType_with_a_rather_long_name", "==", 2.0)]),
                expect=Group("and", [Condition("RigSpeedClass_RigSpeedClass", "==", Operand("Rig_MarkerValue"),
                                               tolerance=0.5), Condition("SignType", "!=", 0.0)]),
                scope=Condition("On", "==", 1.0), within_max=0.5, expect_hold=0.2)
    view = RuleGraphView()
    view.show_rule(rule, {"events": 5, "PASS": 4, "FAIL": 1, "INCONCLUSIVE": 2, "delay_min": 0.1,
                          "delay_max": 0.4}, texts)
    rects = list(view.node_rects().values())
    assert len(rects) == len(build_graph(rule)[0])
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.adjusted(1, 1, -1, -1).intersects(b), (a, b)
    assert any(chr(10) in t for t in view._texts.values())              # the value list is wrapped


# ---------------------------------------------------------------------- rules panel in the window
@pytest.fixture
def window(qapp, tmp_path):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    win = MainWindow(Settings(tmp_path / "settings.json"))
    win.valuemaps = _library(tmp_path)
    win.resize(1400, 900)
    win.dataset = TrainingTrace()
    win.script = parse_plt(TRAINING_VIEW)
    win.show()
    win.rebuild(keep_view=False)
    qapp.processEvents()
    yield win
    win.script.modified = False
    win.close()


def test_rules_panel_runs_shows_events_and_jumps(window, qapp, tmp_path):
    panel = window.rules_panel
    window.ruleset.rules.append(Rule("限速", trigger=Condition("SignType", "==", 7.0),
                                     expect=Condition("DemoSpeedCode", "==", 12.0), within_max=0.5, record=["RoadType"]))
    window.ruleset.rules.append(Rule("road", trigger=Condition("RoadType", "changes")))
    panel.fill_tree()
    panel.run_all()
    panel.wait()
    assert set(panel.results) == {"限速", "road"}
    assert panel.table.rowCount() == 3                          # one check + two road changes
    assert len(window.view.state.rule_lanes) == 2
    window.view.grab()
    first = panel.table.item(0, 0)
    panel._jump(first)
    assert window.view.state.marker == pytest.approx(4.0)
    assert "乡村" not in window.view.values_text() and "高速" in window.view.values_text()
    panel.filter.setCurrentIndex(panel.filter.findData("PASS"))
    assert panel.table.rowCount() == 1
    item = panel.tree.topLevelItem(1)
    item.setCheckState(0, QtCore.Qt.Unchecked)                  # switching a rule off removes its lane
    assert len(window.view.state.rule_lanes) == 1
    assert RuleSet(window.ruleset.path).rules[1].enabled is False
    path = tmp_path / "report.xlsx"
    write_xlsx(path, panel._visible_results(), window.report_meta())
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["概览", "事件", "设置"]
    rows = list(book["事件"].iter_rows(values_only=True))
    assert rows[1][1] == "限速" and rows[1][4] == "通过" and rows[1][5] == pytest.approx(350.0)
    assert "RoadType" in rows[0]


def test_rules_panel_runs_scripts_in_the_runner(window, qapp, training_mf4):
    from tracepaddock.core.registry import open_dataset
    window.dataset = open_dataset(training_mf4)
    window.rebuild(keep_view=False)
    window.ruleset.scripts["s"] = "rule('x', trigger=becomes(S('SignType') == 7), expect=S('DemoSpeedCode') == 12, within=0.5)"
    plugin = next(p for p in discover() if p.name == "State transitions")
    window.ruleset.plugin_runs["p"] = {"path": str(plugin.path), "params": {"signal": "RoadType", "record": []}}
    window.rules_panel.fill_tree()
    window.rules_panel.run_all()
    window.rules_panel.wait(180000)
    assert [e.verdict for e in window.rules_panel.results["s"].events] == ["PASS"]
    assert len(window.rules_panel.results["p"].events) == 2
    window.dataset.close()


def test_rule_context_hints(window):
    ctx = window.rule_context()
    assert ctx["enums"]("SignType") == {0.0: "none", 7.0: "limit 120"}
    assert ctx["hint"]("SignType").startswith("0|")                # value texts: exact
    assert ctx["hint"]("RoadType").startswith("0|")                # a value mapping: exact as well
    assert ctx["hint"]("v_x").startswith("|")                      # continuous: no suggestion from the data
    assert ctx["period"]("DemoSpeedCode") == pytest.approx(0.01)


def test_image_panel_follows_the_cursor(window, qapp):
    lib = window.valuemaps
    lib.tables["限速码"].auto_image = "speed"
    window._valuemaps_changed()
    panel = window.image_panel
    assert len(panel.tiles) == 1 and panel.tiles[0].sig.name == "DemoSpeedCode"
    panel.show_time(1.0)
    assert panel.tiles[0].picture.pixmap() is not None             # 0 km/h -> "end of restriction"
    panel.show_time(5.0)
    assert panel.tiles[0].last == 12.0 and "120" in panel.tiles[0].value.text()
    assert panel.tiles[0].styleSheet()                              # flashes after the change


def test_settings_dialog_restores_the_defaults(qapp, tmp_path):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.settings_dialog import SettingsDialog
    settings = Settings(tmp_path / "settings.json")
    settings.set("ops_on_common_time_base", True)
    settings.set("statistics_on_common_time_base", False)
    dialog = SettingsDialog(settings)
    assert dialog.checks["ops_on_common_time_base"].isChecked()
    dialog.restore_defaults()
    assert not dialog.checks["ops_on_common_time_base"].isChecked()
    assert dialog.checks["statistics_on_common_time_base"].isChecked()
    dialog.accept()
    assert settings.get("ops_on_common_time_base") is False


# ---------------------------------------------------------------------- signals that stopped being sent
def test_a_stopped_signal_is_drawn_dashed_after_its_end(qapp):
    from tracepaddock.ui.render import ViewState, _curve_parts, legend_value_text
    t = np.round(np.arange(0.0, 43.57, 0.02), 6)                     # ends at 43.56 s, recording runs to 65 s
    sig = DisplaySignal(entry=PltEntry(name="v"), name="v", unit="km/h", t=t, y=np.full(len(t), 85.78))
    assert sig.valid_until() == pytest.approx(43.56 + 0.1)
    state = ViewState(x0=30.0, x1=65.0, cursor=53.3)
    (xs, ys), tail = _curve_parts(sig, state, 1000.0, True)
    assert xs.max() <= 43.66 + 1e-9 and tail[0].tolist() == pytest.approx([43.66, 65.0])
    assert "(no new data)" in legend_value_text(sig, state) and "85.78" in legend_value_text(sig, state)
    state.stopped = "cut"
    (xs, ys), tail = _curve_parts(sig, state, 1000.0, True)
    assert tail is None and xs.max() <= 43.66 + 1e-9 and legend_value_text(sig, state) == "no data"
    state.stopped = "hold"
    (xs, ys), tail = _curve_parts(sig, state, 1000.0, True)
    assert tail is None and xs.max() == pytest.approx(65.0) and legend_value_text(sig, state) == "85.78"
    on_change = DisplaySignal(entry=PltEntry(name="w"), name="w", t=np.array([0.0, 3.0]), y=np.array([1.0, 2.0]))
    assert on_change.valid_until() == float("inf")                  # sent on change: keeps its value


def test_the_stopped_signal_option_is_in_the_appearance_dialog(qapp, tmp_path):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.dialogs import AppearanceDialog
    settings = Settings(tmp_path / "settings.json")
    assert settings.get("stopped_signals") == "dashed"
    dialog = AppearanceDialog(settings, lambda: None)
    dialog.stopped_combo.setCurrentIndex(dialog.stopped_combo.findData("cut"))
    assert settings.get("stopped_signals") == "cut"


# ---------------------------------------------------------------------- layouts (panels shown with one click)
def test_layouts_start_with_the_signals_only_and_switch_with_one_click(qapp, tmp_path):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    settings = Settings(tmp_path / "settings.json")
    win = MainWindow(settings)
    win.show()
    qapp.processEvents()
    assert win.rules_dock.isHidden() and win.image_dock.isHidden() and win.toolbar.isVisible()
    win.layout_actions["review"].trigger()
    qapp.processEvents()
    assert win.rules_dock.isVisible() and win.image_dock.isVisible()
    win.layout_actions["focus"].trigger()
    qapp.processEvents()
    assert win.rules_dock.isHidden() and not win.toolbar.isVisible() and not win.statusBar().isVisible()
    win._set_language("zh")                                   # menus are rebuilt: the tool bar stays hidden
    qapp.processEvents()
    assert not win.toolbar.isVisible() and not win.act_toolbar.isChecked()
    win._set_language("en")
    win.act_toolbar.trigger()
    win.act_statusbar.trigger()
    win.act_images.trigger()                                  # a panel on its own, from the menu
    qapp.processEvents()
    assert win.toolbar.isVisible() and win.statusBar().isVisible() and win.image_dock.isVisible()
    win.image_dock.close()                                    # the close button of the panel
    qapp.processEvents()
    assert win.image_dock.isHidden() and not win.act_images.isChecked()
    win.close()


def test_saved_layouts_and_the_startup_layout(qapp, tmp_path, monkeypatch):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    settings = Settings(tmp_path / "settings.json")
    win = MainWindow(settings)
    win.show()
    win.apply_layout("rules")
    qapp.processEvents()
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Mine", True))
    win.save_layout()
    assert "Mine" in win._saved_layouts()
    win.apply_layout("signals")
    qapp.processEvents()
    assert win.rules_dock.isHidden()
    win.apply_layout("custom:Mine")
    qapp.processEvents()
    assert win.rules_dock.isVisible() and win.image_dock.isHidden()
    win._fill_layout_menu()
    texts = [a.text() for a in win.layout_menu.actions()]
    assert "Mine" in texts
    settings.set("startup_layout", "last")
    win.apply_layout("images")
    qapp.processEvents()
    win.script.modified = False
    win.close()                                               # remembers the layout it had
    again = MainWindow(settings)
    again.show()
    qapp.processEvents()
    assert again.image_dock.isVisible() and again.rules_dock.isHidden()
    settings.set("startup_layout", "custom:Mine")
    again.delete_layout("Mine")
    assert settings.get("startup_layout") == "signals" and "Mine" not in again._saved_layouts()
    again.close()


# ---------------------------------------------------------------------- themes
def _contrast(a, b):
    def lum(rgb):
        out = []
        for c in rgb:
            c = c / 255.0
            out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_theme_signal_colours_keep_their_meaning_and_read_on_the_plot():
    from tracepaddock.plt.palette import DEFAULT_PALETTE
    from tracepaddock.ui.themes import THEMES, hex_rgb, theme_palette
    assert theme_palette(THEMES["classic"]) == list(DEFAULT_PALETTE)
    for theme in THEMES.values():
        if not theme.run1:
            continue
        palette = theme_palette(theme)
        plot = hex_rgb(theme.plot)
        for index in list(range(2, 8)) + list(range(10, 16)):
            assert _contrast(palette[index], plot) >= 3.0, (theme.key, index)
        red, green, blue = palette[2], palette[3], palette[5]
        assert red[0] > red[2] and green[1] > green[0] and blue[2] > blue[0]      # 2 red, 3 green, 5 blue
        assert _contrast(palette[1], plot) >= 7.0                                 # colour 1: the text colour


def test_switching_themes_and_back_restores_the_classic_look(qapp, tmp_path, demo_pair):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    settings = Settings(tmp_path / "settings.json")
    settings.set("theme", "classic")
    settings.set("background", "#c7edcc")                       # a user-defined plot background
    win = MainWindow(settings)
    win.show()
    win.set_theme("racing")
    qapp.processEvents()
    assert win.view.state.legend_style == "chip" and win.view.state.backdrop is not None
    assert {"zoom", "pointer"} <= set(win.view.theme_cursors) and "QMenuBar" in qapp.styleSheet()
    assert win.theme_actions["racing"].isChecked() and set(win.theme_actions) == {"classic", "daylight", "obsidian", "racing"}
    assert win.open_dataset(*demo_pair)
    win.view.start_zoom("x")
    assert win.view.cursor().pixmap().width() > 0               # the cartoon zoom pointer
    win.view.cancel()
    win._set_pointer("")                                        # system pointers
    assert win.view.theme_cursors == {}
    win._set_backdrop("")
    assert win.view.state.backdrop is None
    win.set_theme("obsidian")
    from tracepaddock.plt.palette import DEFAULT_PALETTE
    from tracepaddock.ui.themes import THEMES, hex_rgb
    assert win.view.state.background == (0x1b, 0x1d, 0x21)
    assert win.view.state.palette[2] == hex_rgb(THEMES["obsidian"].run1[0])      # the theme's signal colours
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    assert len(win.findChildren(QtWidgets.QToolBar)) == 1                     # the old ones are deleted
    win.set_theme("classic")
    qapp.processEvents()
    assert qapp.styleSheet() == "" and win.view.state.legend_style == "block"
    assert list(win.view.state.palette) == list(DEFAULT_PALETTE)
    # the classic look: legend rows fill the height, Arial, no time labels, icon tool bar, clock
    assert win.view.state.legend_fill and win.view.state.font_family == "Arial" and not win.view.state.time_labels
    assert win.toolbar.toolButtonStyle() == QtCore.Qt.ToolButtonIconOnly and not win.clock_label.isHidden()
    assert settings.get("background") == "#c7edcc" and win.view.state.backdrop is None
    win.close()


def test_themed_views_paint_with_legend_marks_and_pictures(qapp):
    from tracepaddock.ui import cute_art
    from tracepaddock.ui.render import ViewState, paint_view
    from tracepaddock.ui.themes import THEMES, hex_rgb, theme_palette
    t = np.linspace(0.0, 10.0, 200)
    signals = [DisplaySignal(entry=PltEntry(name="a"), name="a", unit="km/h", t=t, y=np.sin(t), color=2,
                             scmin=-1.0, scmax=1.0),
               DisplaySignal(entry=PltEntry(name="b"), name="b", t=t, y=(t > 5).astype(float), color=5, digital=True)]
    for key in ("daylight", "obsidian", "racing"):
        theme = THEMES[key]
        state = ViewState(x0=0.0, x1=10.0, background=hex_rgb(theme.plot), palette=theme_palette(theme),
                          legend_style="chip", legend_bg=hex_rgb(theme.legend_bg), text_color=hex_rgb(theme.text),
                          grid_color=hex_rgb(theme.grid), value_font=theme.font_values, cursor=4.0,
                          backdrop=cute_art.backdrop_image(theme.backdrop, 800) if theme.backdrop else None,
                          backdrop_opacity=theme.backdrop_opacity)
        image = QtGui.QImage(900, 400, QtGui.QImage.Format_ARGB32)
        painter = QtGui.QPainter(image)
        paint_view(painter, QtCore.QRectF(0, 0, 900, 400), signals, state)
        painter.end()
        assert QtGui.QColor(image.pixel(895, 395)).name() == theme.legend_bg
    for character in cute_art.CHARACTERS:
        for kind in ("zoom", "pointer"):
            assert not cute_art.cursor(kind, character).pixmap().isNull()
    for scene in cute_art.SCENES:
        assert not cute_art.backdrop_image(scene, 400).isNull()


# ---------------------------------------------------------------------- status bar, own pointer, recordings of the folder
def test_status_bar_shows_time_field_clock_version_and_author(qapp, tmp_path):
    import re
    import tracepaddock
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    win = MainWindow(Settings(tmp_path / "settings.json"))
    win.show()
    qapp.processEvents()
    assert tracepaddock.__version__ in win.credit_label.text() and "Geer Xie" in win.credit_label.text()
    assert re.fullmatch(r"\d\d:\d\d:\d\d", win.clock_label.text())
    win._on_status({"x": "12.5", "t": "12.5 s"})
    assert win._status_fields["t"][0].text().strip() == "12.5 s" and win._status_fields["t"][1] == "T="
    win.set_theme("daylight")
    assert win.clock_label.isHidden()                       # the clock belongs to the classic look
    win.close()


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0])
def test_ordinary_picture_cursor_contains_only_the_picture(qapp, tmp_path, ratio):
    from tracepaddock.ui import cute_art

    image = QtGui.QImage(64, 40, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("#ff8800"))
    picture = tmp_path / "picture.png"
    assert image.save(str(picture))
    custom = cute_art.image_cursor("pointer", str(picture), picture.stat().st_mtime, ratio)
    assert custom is not None
    for pointer in [custom] + [cute_art.cursor("pointer", key, ratio) for key in cute_art.CHARACTERS]:
        assert pointer.hotSpot() == QtCore.QPoint(24, 24)
        pixels = pointer.pixmap().toImage()
        assert pixels.pixelColor(round(24 * ratio), round(24 * ratio)).alpha() > 0
        assert pixels.pixelColor(round(4 * ratio), round(12 * ratio)).alpha() == 0


def test_own_picture_as_mouse_pointer(qapp, tmp_path):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    picture = tmp_path / "me.png"
    image = QtGui.QImage(64, 40, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("#ff8800"))
    image.save(str(picture))
    settings = Settings(tmp_path / "settings.json")
    win = MainWindow(settings)
    settings.set("cursor_image", str(picture))
    win._set_pointer("custom")
    assert set(win.view.theme_cursors) == {"zoom", "pointer"}
    hot = win.view.theme_cursors["zoom"].hotSpot()
    assert (hot.x(), hot.y()) == (9, 9)                     # the precise cross stays the hot spot
    settings.set("cursor_image", str(tmp_path / "missing.png"))
    win._set_pointer("custom")
    assert win.view.theme_cursors == {}                     # unreadable picture: system pointers
    win.close()


def test_recordings_of_the_folder_in_the_tool_bar(qapp, tmp_path, demo_pair):
    import shutil
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    for name in ("a_drive.mf4", "b_drive.mf4"):
        shutil.copy(demo_pair[0], tmp_path / name)
    win = MainWindow(Settings(tmp_path / "settings.json"))
    assert win.open_dataset(tmp_path / "a_drive.mf4", demo_pair[1])
    combo = win.files_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["a_drive.mf4", "b_drive.mf4"]
    assert combo.currentText() == "a_drive.mf4"
    win._files_combo_chosen(1)
    assert Path(win.dataset.path).name == "b_drive.mf4" and win.files_combo.currentText() == "b_drive.mf4"
    win.script.modified = False
    win.close()


@pytest.mark.parametrize(("code", "title", "filter_name"), [
    ("en", "Open PLT", "PLT files"), ("zh", "打开 PLT", "PLT 文件"),
])
def test_open_plt_menu_and_dialog(qapp, tmp_path, monkeypatch, code, title, filter_name):
    from tracepaddock.i18n import language, set_language
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow

    path = tmp_path / "test.plt"
    path.write_text("speed nobit 2\n", encoding="utf-8")
    seen = []

    def choose_plt(_parent, caption, _start, filters):
        seen.append((caption, filters.split(";;")[0]))
        return str(path), ""

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", choose_plt)
    previous = language()
    set_language(code)
    win = MainWindow(Settings(tmp_path / "settings.json"))
    try:
        assert win.act_open_plt.text().replace("&", "").startswith(title)
        win.act_open_plt.trigger()
        assert seen == [(title, f"{filter_name} (*.plt)")]
        assert win.script.path == path
    finally:
        win.close()
        set_language(previous)


def test_startup_has_a_black_plot_until_a_recording_is_open(qapp, tmp_path, demo_pair):
    from PySide6.QtTest import QTest
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow

    settings = Settings(tmp_path / "settings.json")
    settings.set("theme", "daylight")
    win = MainWindow(settings)
    win.show()
    try:
        assert {16, 32}.issubset(size.width() for size in win.windowIcon().availableSizes())
        assert win.open_plt(demo_pair[1])
        QTest.keyClick(win.view, QtCore.Qt.Key_X)
        QTest.keyClick(win.view, QtCore.Qt.Key_Z)
        move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(win.view.rect().center()), QtCore.Qt.NoButton,
                                 QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
        qapp.sendEvent(win.view, move)
        assert not win.act_ref.isChecked() and not win.view.reference_line
        assert win.view.cursor_time() is None
        image = win.view.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        black = QtGui.QImage(image.size(), QtGui.QImage.Format_RGB32)
        black.fill(QtCore.Qt.black)
        assert image == black
        assert win.view.grab_image().pixelColor(win.view.rect().center()) == QtGui.QColor(QtCore.Qt.black)

        assert win.open_dataset(demo_pair[0])
        assert win.view.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32) != black
        QTest.keyClick(win.view, QtCore.Qt.Key_X)
        assert win.act_ref.isChecked() and win.view.reference_line
    finally:
        win.close()


def test_missing_plt_signals_are_explained_in_the_editor_and_skipped_in_the_plot(qapp, tmp_path, monkeypatch, demo_pair):
    from tracepaddock.i18n import tr
    from tracepaddock.settings import Settings
    from tracepaddock.ui.editor import PltEditorDialog
    from tracepaddock.ui.mainwindow import MainWindow

    missing = ["BenchMissingSpeed", "BenchMissingLimit", "BenchMissingMarker", "BenchMissingState",
               "BenchMissingIndicator", "BenchMissingBuzzer"]
    path = tmp_path / "mixed.plt"
    path.write_text("".join(f"{name} nobit 2 ref=25\n" for name in missing)
                    + (demo_pair[1]).read_text(encoding="utf-8"), encoding="utf-8")

    def filter_and_apply(dialog):
        dialog.filter_edit.setCurrentText("Rig_*")
        dialog._refresh_filter()
        assert dialog.name_model.rowCount() == 5
        assert dialog.missing_note.text() == tr(
            "%d PLT signals in red are missing from this recording and will be skipped in the plot.") % len(missing)
        for index, name in enumerate(missing):
            row = dialog.entry_list.item(index)
            assert row.text().startswith(name)
            assert row.toolTip() == tr("This signal is not contained in the dataset")
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(PltEditorDialog, "exec_", filter_and_apply)
    win = MainWindow(Settings(tmp_path / "settings.json"))
    try:
        assert win.open_dataset(demo_pair[0], path)
        win.edit_signals()
        shown = [signal.name for signal in win.view.signals]
        assert shown == ["Rig_Zone", "Rig_Surface", "Rig_Lighting", "Rig_MarkerType", "RigVelocity", "Rig_MarkerValue",
                         "RigSpeedClass"]
        assert [entry.name for entry in win.script.entries()] == missing + shown
        assert win.statusBar().currentMessage() == tr(
            "Skipped %d PLT signals missing from this recording, including %s") % (len(missing), missing[0])
    finally:
        win.close()


def test_themes_can_be_deleted_and_restored_but_not_the_classic_one(qapp, tmp_path, monkeypatch):
    from tracepaddock.settings import Settings
    from tracepaddock.ui.mainwindow import MainWindow
    from tracepaddock.ui.theme_dialog import ThemeDialog
    from tracepaddock.ui.themes import visible_themes
    assert visible_themes(["classic", "racing"]) == ["classic", "daylight", "obsidian"]     # classic always stays
    dialog = ThemeDialog("obsidian", [])
    keys = [dialog.list.item(i).data(QtCore.Qt.UserRole) for i in range(dialog.list.count())]
    assert keys == ["classic", "daylight", "obsidian", "racing"]
    dialog.list.setCurrentRow(0)
    assert not dialog.b_delete.isEnabled()                                  # the classic theme
    dialog.list.setCurrentRow(3)
    dialog._delete(confirm=False)
    assert dialog.hidden == {"racing"} and dialog.list.count() == 3 and dialog.b_restore.isEnabled()
    dialog._restore()
    assert dialog.hidden == set() and dialog.list.count() == 4
    # in the window: delete the theme in use -> back to the classic theme, menu without it
    settings = Settings(tmp_path / "settings.json")
    win = MainWindow(settings)
    win.set_theme("racing")

    class Fake:
        def __init__(self, *a, **k):
            self.hidden, self.applied = {"racing", "classic"}, None

        def exec_(self):
            return 1

    import tracepaddock.ui.theme_dialog as td
    monkeypatch.setattr(td, "ThemeDialog", Fake)
    win.theme_dialog()
    assert settings.get("theme") == "classic" and "racing" not in win.theme_actions and "classic" in win.theme_actions
    win.close()


def test_classic_legend_shows_values_under_the_mouse_and_no_ranges_when_idle(qapp):
    from tracepaddock.ui.render import ViewState, legend_value_text
    sig = DisplaySignal(entry=PltEntry(name="v"), name="v", t=np.array([0.0, 1.0]), y=np.array([3.0, 4.0]),
                        scmin=0.0, scmax=10.0)
    assert legend_value_text(sig, ViewState()) == "0 - 10 [1/div]"
    assert legend_value_text(sig, ViewState(legend_idle_ranges=False)) == ""
    assert legend_value_text(sig, ViewState(legend_idle_ranges=False, cursor=0.5)) == "3"
