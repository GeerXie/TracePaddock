"""Regression tests: rule runner, plug-ins, Excel report, settings / value-map / rule files, export, start-up."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from tracepaddock.display import DisplaySignal
from tracepaddock.plt.model import PltEntry

from test_stage3 import TrainingTrace, TRAINING_VIEW

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
BUILTIN_PLUGIN = "state_transitions.py"


@pytest.fixture(scope="module")
def training_mf4(tmp_path_factory):
    from tracepaddock.core.export import write_mdf
    folder = tmp_path_factory.mktemp("training_io")
    ds = TrainingTrace()
    signals = [DisplaySignal(entry=PltEntry(name=s.name), name=s.name, unit=s.unit, t=d.t, y=d.y)
               for s in ds.signals for d in [ds.read(s)]]
    path = folder / "training.mf4"
    write_mdf(path, signals, 0.0, 20.0)
    return path


# ---------------------------------------------------------------------- helpers
def _alive(pid: int) -> bool:
    if not WINDOWS:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000 | 0x00100000, False, pid)   # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102   # WAIT_TIMEOUT: still running
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def _kill(pid) -> None:
    """Never leave an endless script running after a test."""
    if pid:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _endless(pid_file: Path) -> str:
    """A rule script that writes the id of its process into ``pid_file`` and then never ends."""
    return "import os\nopen(%r, 'w').write(str(os.getpid()))\nwhile True:\n    pass\n" % str(pid_file)


def _read_pid(pid_file: Path, seconds: float = 90.0) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            text = pid_file.read_text().strip()
        except OSError:
            text = ""
        if text:
            return int(text)
        time.sleep(0.05)
    raise AssertionError("the script did not start")


def _pid_or_none(pid_file: Path):
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def _env(tmp_path: Path) -> dict:
    temp = tmp_path / "temp"                                  # job folders of a viewer that ends abruptly stay here
    temp.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(QT_QPA_PLATFORM="offscreen", APPDATA=str(tmp_path / "appdata"), PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(ROOT), TEMP=str(temp), TMP=str(temp))
    env.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"))
    return env


def _run_python(code: str, tmp_path: Path, timeout: float):
    """Run ``code`` in another Python process; returns (exit code, stdout, stderr).

    The output goes to files, not pipes: when a process crashes, Windows Error Reporting keeps a suspended
    copy of it that holds its pipes open, and reading them would block long after the process ended."""
    out_path, err_path = tmp_path / "child_out.txt", tmp_path / "child_err.txt"
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), env=_env(tmp_path), stdout=out, stderr=err)
        try:
            code = proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = "timeout"
    return (code, out_path.read_text(encoding="utf-8", errors="replace"),
            err_path.read_text(encoding="utf-8", errors="replace"))


@contextmanager
def _locked(path: Path):
    """Another program holds ``path`` open without sharing it (it can be neither read nor renamed)."""
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)   # GENERIC_READ, no sharing
    assert handle not in (None, ctypes.c_void_p(-1).value)
    try:
        yield
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _sig(name, t, y, unit="", info=None, key=None):
    entry = PltEntry(name=info.name if info is not None else ("*" if key else name), key=key)
    return DisplaySignal(entry=entry, info=info, name=name, unit=unit, t=np.asarray(t, float), y=np.asarray(y, float))


class _Ctx(QtWidgets.QMainWindow):
    """The parts of the main window that the rules panel uses."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__()
        self.plugin_calls = []
        self.report_path = str(tmp_path / "report.xlsx")

    def rule_value_names(self, rule):
        return {}

    def rule_context(self):
        return {"names": ["RoadType", "SignType"], "enums": lambda name: {}, "hint": lambda name: "",
                "period": lambda name: None, "all_names": ["RoadType", "SignType"]}

    def has_data(self):
        return True

    def view_range(self):
        return (0.0, 20.0)

    def rule_source(self):
        return None

    def run_plugin(self, path, params, name, rng=None):
        self.plugin_calls.append(path)
        return {"error": None, "log": [], "results": [], "events": []}

    def run_script(self, code, name, rng=None):
        return {"error": None, "log": [], "results": [], "events": []}

    def set_rule_lanes(self, lanes):
        pass

    def report_default_path(self):
        return self.report_path

    def report_meta(self):
        return {"数据文件": "x.mf4"}


# ---------------------------------------------------------------------- plug-in references (finding 0)
def test_plugin_rules_refer_to_plugins_independently_of_the_program_folder(tmp_path, qapp, monkeypatch):
    from tracepaddock.rules import plugins
    from tracepaddock.rules.model import RuleSet
    from tracepaddock.ui import rules_panel as panel_module
    from tracepaddock.ui.script_dialog import PluginDialog
    user = tmp_path / "user_plugins"
    user.mkdir()
    monkeypatch.setattr(plugins, "user_folder", lambda: user)
    builtin = plugins.builtin_folder() / BUILTIN_PLUGIN
    (user / "mine.py").write_text("NAME = 'mine'\ndef run(api, p):\n    pass\n", encoding="utf-8")

    # a rule list written before the program folder was moved, or by another copy of the program
    moved = tmp_path / "Old" / "TracePaddock" / "_internal" / "tracepaddock" / "plugins" / BUILTIN_PLUGIN
    other_copy = tmp_path / "B" / "tracepaddock" / "plugins" / BUILTIN_PLUGIN
    other_copy.parent.mkdir(parents=True)
    other_copy.write_text("NAME = 'old copy'\ndef run(api, p):\n    api.log('old copy')\n", encoding="utf-8")
    moved_user = tmp_path / "Old" / "TracePaddock" / "TracePaddockData" / "plugins" / "mine.py"
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"plugins": {
        "moved": {"path": str(moved), "params": {"signal": "RoadType"}},
        "other copy": {"path": str(other_copy), "params": {}},
        "user": {"path": str(moved_user), "params": {}},
        "same": {"path": str(builtin), "params": {}}}}), encoding="utf-8")
    ruleset = RuleSet(rules_path)
    assert {k: plugins.resolve_plugin(v["path"]) for k, v in ruleset.plugin_runs.items()} == {
        "moved": builtin, "other copy": builtin, "user": user / "mine.py", "same": builtin}
    ruleset.save()
    text = rules_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(plugins.builtin_folder()) not in text.replace("\\\\", "\\")

    # "Run all" passes the plug-in file of this program
    ctx = _Ctx(tmp_path)
    panel = panel_module.RulesPanel(ctx, ruleset)
    panel.run_all()
    panel.wait()
    assert sorted(Path(p) for p in ctx.plugin_calls) == sorted([builtin, builtin, user / "mine.py", builtin])

    # editing a rule with an old path selects its plug-in and keeps its parameters
    dialog = PluginDialog(plugins.discover(), ["RoadType"], lambda p, v: {}, None,
                          {"path": str(moved), "params": {"signal": "RoadType"}})
    assert dialog.plugin().path == builtin and dialog.values()["signal"] == "RoadType"

    # a plug-in added to the list is stored by name, not by the folder of the program
    def accept(self):
        self.list.setCurrentRow(next(i for i, p in enumerate(self.plugins) if p.path == builtin))
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PluginDialog, "exec_", accept)
    panel.edit_plugin(None)
    added = [v["path"] for k, v in ruleset.plugin_runs.items() if k not in ("moved", "other copy", "user", "same")]
    assert added == ["builtin:" + BUILTIN_PLUGIN]


def test_the_runner_finds_a_plugin_by_its_reference(training_mf4):
    from tracepaddock.rules.runner import run_job
    result = run_job({"dataset": str(training_mf4), "plugin": "builtin:" + BUILTIN_PLUGIN,
                      "params": {"signal": "RoadType", "record": []}, "script_name": "p"}, timeout=90)
    assert result["error"] is None and [round(e["t0"], 3) for e in result["events"]] == [5.0, 12.0]


# ---------------------------------------------------------------------- the script process (findings 1, 5)
@pytest.mark.skipif(not WINDOWS, reason="Windows job object")
def test_the_script_process_ends_when_the_viewer_ends(training_mf4, tmp_path):
    pid_file = tmp_path / "pid.txt"
    job = {"dataset": str(training_mf4), "script": _endless(pid_file)}
    viewer = textwrap.dedent("""
        import json, os, sys, threading, time
        from tracepaddock.rules.runner import run_job
        job = json.loads(%r)
        threading.Thread(target=lambda: run_job(job, timeout=600), daemon=True).start()
        deadline = time.time() + 90
        while not (os.path.exists(%r) and open(%r).read().strip()) and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        os._exit(3)                       # the viewer ends (or crashes) while the script still runs
    """) % (json.dumps(job), str(pid_file), str(pid_file))
    _run_python(viewer, tmp_path, 180)
    pid = _pid_or_none(pid_file)
    try:
        assert pid is not None
        assert _gone(pid, 15), "the script process outlived the viewer"
    finally:
        _kill(pid)


def test_a_script_that_runs_too_long_is_ended_and_leaves_no_job_folder(training_mf4, tmp_path, monkeypatch):
    from tracepaddock.rules.runner import JobError, run_job
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    pid_file = tmp_path / "pid.txt"
    try:
        with pytest.raises(JobError) as info:
            run_job({"dataset": str(training_mf4), "script": _endless(pid_file)}, timeout=12)
        assert "12 s" in str(info.value)
        pid = _pid_or_none(pid_file)
        assert pid is not None and _gone(pid, 10)
    finally:
        _kill(_pid_or_none(pid_file))
    assert list(temp.iterdir()) == []                              # job.json with the script is gone
    with pytest.raises(TypeError):                                  # a job that cannot be written
        run_job({"dataset": str(training_mf4), "script": "log(1)", "params": {1, 2}}, timeout=20)
    assert list(temp.iterdir()) == []


def test_running_scripts_are_stopped_on_request(training_mf4, tmp_path):
    from tracepaddock.rules import runner
    pid_file = tmp_path / "pid.txt"
    out = {}

    def work():
        try:
            out["result"] = runner.run_job({"dataset": str(training_mf4), "script": _endless(pid_file)}, timeout=600)
        except Exception as exc:
            out["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    pid = _read_pid(pid_file)
    try:
        assert runner.stop_all_jobs() == 1
        thread.join(20)
        assert not thread.is_alive()
        assert isinstance(out.get("error"), runner.JobError) and "stopped" in str(out["error"])
        assert _gone(pid, 10)
    finally:
        _kill(pid)
        thread.join(30)


def test_closing_the_viewer_during_a_run_ends_the_script_and_the_program_cleanly(training_mf4, tmp_path):
    pid_file = tmp_path / "pid.txt"
    viewer = textwrap.dedent("""
        import gc, os, sys, time
        from PySide6 import QtCore, QtWidgets
        app = QtWidgets.QApplication([])
        from tracepaddock.core.registry import open_dataset
        from tracepaddock.plt.parser import parse_plt
        from tracepaddock.settings import Settings
        from tracepaddock.ui.mainwindow import MainWindow
        win = MainWindow(Settings(%(settings)r))
        win.dataset = open_dataset(%(dataset)r)
        win.script = parse_plt(%(plt)r)
        win.rebuild(keep_view=False)
        win.ruleset.scripts["endless"] = %(script)r
        win.show()
        win.rules_panel.run_all()

        def close_when_running():
            if os.path.exists(%(pid)r) and open(%(pid)r).read().strip():
                win.script.modified = False
                win.close()
            else:
                QtCore.QTimer.singleShot(100, close_when_running)

        QtCore.QTimer.singleShot(100, close_when_running)
        started = time.time()
        code = app.exec_()
        print("event loop ended after %%.1f s" %% (time.time() - started), flush=True)
        del win
        gc.collect()
        print("window destroyed", flush=True)
        sys.exit(code)
    """) % {"settings": str(tmp_path / "settings.json"), "dataset": str(training_mf4), "plt": TRAINING_VIEW,
            "script": _endless(pid_file), "pid": str(pid_file)}
    started = time.monotonic()
    try:
        code, out, err = _run_python(viewer, tmp_path, 240)
        pid = _pid_or_none(pid_file)
        assert pid is not None, err[-2000:]
        assert code == 0 and "window destroyed" in out, (code, err[-2000:])
        assert time.monotonic() - started < 100                   # not after the time limit of 120 s
        assert _gone(pid, 10)
    finally:
        _kill(_pid_or_none(pid_file))


# ---------------------------------------------------------------------- settings / value maps / rules (finding 2)
_BOM = b"\xef\xbb\xbf"


def test_files_with_a_byte_order_mark_are_read(tmp_path):
    from tracepaddock.rules.model import RuleSet
    from tracepaddock.settings import Settings
    from tracepaddock.valuemap import ValueMapLibrary
    (tmp_path / "settings.json").write_bytes(_BOM + json.dumps(
        {"language": "zh", "recent_datasets": ["C:\\数据\\a.mf4"], "grid": False}, ensure_ascii=False).encode("utf-8"))
    settings = Settings(tmp_path / "settings.json")
    assert settings.get("language") == "zh" and settings.get("recent_datasets") == ["C:\\数据\\a.mf4"]
    assert settings.get("grid") is False
    (tmp_path / "valuemaps.json").write_bytes(_BOM + json.dumps(
        {"tables": [{"name": "Gear", "rows": [[1, 1, "一档"]]}], "assignments": [{"pattern": "gear", "table": "Gear"}]},
        ensure_ascii=False).encode("utf-8"))
    library = ValueMapLibrary(tmp_path / "valuemaps.json")
    assert list(library.tables) == ["Gear"] and [a.pattern for a in library.assignments] == ["gear"]
    (tmp_path / "rules.json").write_bytes(_BOM + json.dumps(
        {"rules": [{"name": "sign", "trigger": {"items": [{"signal": "SignType", "op": "==", "value": 7}]}}],
         "scripts": {"training script": "log(1)"}}).encode("utf-8"))
    rules = RuleSet(tmp_path / "rules.json")
    assert [r.name for r in rules.rules] == ["sign"] and list(rules.scripts) == ["training script"]


@pytest.mark.parametrize("kind", ["settings", "valuemaps", "rules"])
def test_a_file_with_a_mistake_is_kept_and_reported_instead_of_being_overwritten(tmp_path, kind):
    from tracepaddock.rules.model import Rule, RuleSet
    from tracepaddock.settings import Settings, unreadable_files
    from tracepaddock.valuemap import ValueMap, ValueMapLibrary
    unreadable_files()                                             # forget reports of earlier tests
    path = tmp_path / (kind + ".json")
    text = '{"language": "zh", "recent_datasets": ["a.mf4"],}' if kind == "settings" else \
        '{"tables": [{"name": "Gear", "rows": [[1, 1, "一档"]]}], }' if kind == "valuemaps" else \
        '{"rules": [{"name": "sign"}], "scripts": {"s": "log(1)"},}'
    path.write_text(text, encoding="utf-8")
    if kind == "settings":
        obj = Settings(path)
        assert obj.get("recent_datasets") == []
        obj.set("window/width", 1200)
    elif kind == "valuemaps":
        obj = ValueMapLibrary(path)
        assert obj.tables == {}
        obj.tables["new"] = ValueMap("new", [(0.0, 0.0, "x")])
    else:
        obj = RuleSet(path)
        assert obj.rules == []
        obj.rules.append(Rule("new"))
    obj.save()
    kept = [p for p in tmp_path.iterdir() if p != path]
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == text   # the user's text is not lost
    reports = unreadable_files()
    assert [(r.path, r.backup) for r in reports] == [(path, kept[0])] and reports[0].reason
    assert unreadable_files() == []
    assert json.loads(path.read_text(encoding="utf-8"))                     # saving goes on normally


def test_rules_that_cannot_be_read_are_kept_in_a_copy(tmp_path):
    from tracepaddock.rules.model import RuleSet
    from tracepaddock.settings import unreadable_files
    unreadable_files()
    path = tmp_path / "rules.json"
    text = json.dumps({"rules": [{"name": "typo", "trigger": {"items": [{"signal": "a", "op": "=", "value": 1}]}},
                                 {"name": "fine", "trigger": {"items": [{"signal": "a", "op": "==", "value": 1}]}}]})
    path.write_text(text, encoding="utf-8")
    rules = RuleSet(path)
    RuleSet(path)                                                  # the next start uses the same copy
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rules.json", "rules.json.bad"]
    assert [r.name for r in rules.rules] == ["fine"]
    rules.save()
    reports = unreadable_files()
    assert len(reports) == 2 and reports[0].skipped == 1 and "typo" in reports[0].reason
    assert reports[0].backup == reports[1].backup and reports[0].backup.read_text(encoding="utf-8") == text


@pytest.mark.skipif(not WINDOWS, reason="exclusive file lock")
def test_a_settings_file_that_cannot_be_opened_is_not_written_over(tmp_path):
    from tracepaddock.settings import Settings, unreadable_files
    unreadable_files()
    path = tmp_path / "settings.json"
    path.write_text('{"language": "zh", "grid": false}', encoding="utf-8")
    with _locked(path):
        settings = Settings(path)
    settings.set("window/width", 1200)
    settings.save()
    assert json.loads(path.read_text(encoding="utf-8")) == {"language": "zh", "grid": False}
    reports = unreadable_files()
    assert [(r.path, r.backup) for r in reports] == [(path, None)]


def test_the_start_tells_which_files_could_not_be_read(tmp_path, qapp, monkeypatch):
    import tracepaddock.__main__ as start
    from tracepaddock.settings import Settings, unreadable_files
    unreadable_files()
    (tmp_path / "settings.json").write_text("{ not json", encoding="utf-8")
    Settings(tmp_path / "settings.json")
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: shown.append(args[2]))
    start.show_load_problems(None)
    assert len(shown) == 1 and "settings.json" in shown[0] and "settings.json.bad" in shown[0]
    start.show_load_problems(None)                                  # told once
    assert len(shown) == 1


# ---------------------------------------------------------------------- plug-in files (findings 3, 12)
def test_plugins_whose_parameters_are_not_a_list_are_reported_not_raised(tmp_path, qapp, monkeypatch):
    from tracepaddock.rules import plugins
    from tracepaddock.ui import rules_panel as panel_module
    from tracepaddock.ui.script_dialog import PluginDialog
    values = ["None", "0", "True", "1.5", "'abc'", "{'a': 1}", "{[]: 1}"]
    for i, value in enumerate(values):
        (tmp_path / ("p%d.py" % i)).write_text("NAME = 'p%d'\nPARAMETERS = %s\ndef run(api, p):\n    pass\n"
                                               % (i, value), encoding="utf-8")
    found = {p.path.name: p.problem for p in plugins.discover([tmp_path]) if p.path.parent == tmp_path}
    assert len(found) == len(values) and all(found.values())
    monkeypatch.setattr(plugins, "user_folder", lambda: tmp_path)
    monkeypatch.setattr(panel_module, "discover", plugins.discover)
    shown = []
    monkeypatch.setattr(PluginDialog, "exec_", lambda self: shown.append(len(self.plugins)) or 0)
    from tracepaddock.rules.model import RuleSet
    panel = panel_module.RulesPanel(_Ctx(tmp_path), RuleSet(tmp_path / "rules.json", autoload=False))
    panel.edit_plugin(None)                                        # the plug-in window opens
    assert shown and shown[0] >= 2                                 # the two shipped plug-ins can still be used


def test_plugins_saved_with_a_byte_order_mark_or_a_coding_line_work(training_mf4, tmp_path):
    from tracepaddock.rules.plugins import builtin_folder, read_plugin
    from tracepaddock.rules.runner import run_job
    source = (builtin_folder() / BUILTIN_PLUGIN).read_text(encoding="utf-8")
    source = source.replace('"State transitions"', '"状态切换"')
    bom = tmp_path / "bom_copy.py"
    bom.write_bytes(_BOM + source.encode("utf-8"))
    gbk = tmp_path / "gbk_copy.py"
    gbk.write_bytes(("# -*- coding: gbk -*-\n" + source).encode("gbk"))
    for path in (bom, gbk):
        plugin = read_plugin(path)
        assert plugin.problem is None and plugin.name == "状态切换"
    result = run_job({"dataset": str(training_mf4), "plugin": str(bom), "params": {"signal": "RoadType", "record": []},
                      "script_name": "bom"}, timeout=90)
    assert result["error"] is None and len(result["events"]) == 2
    result = run_job({"dataset": str(training_mf4), "plugin": str(gbk), "params": {"signal": "RoadType", "record": []},
                      "script_name": "gbk"}, timeout=90)
    assert result["error"] is None and len(result["events"]) == 2


# ---------------------------------------------------------------------- Excel report (findings 4, 11)
def test_the_report_is_written_with_control_characters_in_the_texts(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from tracepaddock.rules.engine import Event
    from tracepaddock.rules.report import RunResult, write_xlsx
    event = Event("r\x07", 1.0, "PASS", note="note \x00 end", values={"v\x1b": 1.0}, texts={"v\x1b": "ON\x1b"})
    result = RunResult("rule \x01", "script", [event], "bad \x1b[0m thing", "t\x02", "e\x03", None,
                       ["value \x1b[31mred"])
    path = tmp_path / "report.xlsx"
    write_xlsx(path, [result], {"数据文件": "x\x0b.mf4"})
    book = openpyxl.load_workbook(path)
    cells = [c for sheet in book.worksheets for row in sheet.iter_rows(values_only=True) for c in row
             if isinstance(c, str)]
    assert "value ?[31mred" in cells and "note ? end" in cells and "bad ?[0m thing" in cells
    assert not any(ch in c for c in cells for ch in "\x00\x01\x02\x03\x07\x0b\x1b")


def test_a_report_that_cannot_be_written_gives_a_message(tmp_path, qapp, monkeypatch):
    from tracepaddock.rules.model import RuleSet
    from tracepaddock.rules.report import RunResult
    from tracepaddock.ui import rules_panel as panel_module
    panel = panel_module.RulesPanel(_Ctx(tmp_path), RuleSet(tmp_path / "rules.json", autoload=False))
    panel.ruleset.scripts["s"] = "log(1)"
    panel.results["s"] = RunResult("s", "script", [], None)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "r.xlsx"), ""))
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: shown.append(args[2]))

    def broken(*args):
        raise RuntimeError("unexpected trouble")

    monkeypatch.setattr(panel_module, "write_xlsx", broken)
    panel.export_xlsx()                                            # must not raise out of the button's slot
    assert shown and "unexpected trouble" in shown[0]


def test_the_overview_does_not_show_rules_that_checked_nothing_as_passed(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from tracepaddock.rules.engine import Event
    from tracepaddock.rules.report import RunResult, write_xlsx
    results = [RunResult("undecided", "form", [Event("undecided", 1.0, "INCONCLUSIVE"), Event("undecided", 2.0, "INCONCLUSIVE")]),
               RunResult("nothing", "form", []),
               RunResult("passed", "form", [Event("passed", 1.0, "PASS"), Event("passed", 2.0, "INCONCLUSIVE")]),
               RunResult("failed", "form", [Event("failed", 1.0, "FAIL")]),
               RunResult("broken", "form", [], "signal not found"),
               RunResult("found", "form", [Event("found", 1.0, "FOUND")])]
    path = tmp_path / "report.xlsx"
    write_xlsx(path, results, {})
    sheet = openpyxl.load_workbook(path)["概览"]
    fills = {row[0].value: (row[0].fill.fill_type, row[0].fill.fgColor.rgb) for row in sheet.iter_rows(min_row=2)}
    assert fills["undecided"] == ("solid", "00E7E6E6")
    assert fills["nothing"][0] is None
    assert fills["passed"] == ("solid", "00C6EFCE") and fills["found"] == ("solid", "00C6EFCE")
    assert fills["failed"] == ("solid", "00FFC7CE") and fills["broken"] == ("solid", "00FFC7CE")


# ---------------------------------------------------------------------- export (findings 6, 7, 8)
def test_mdf_export_writes_the_file_name_that_was_chosen(tmp_path):
    from tracepaddock.core.export import write_mdf
    from tracepaddock.core.registry import open_dataset
    speed = _sig("Speed", [0.0, 0.5, 1.0], [1.0, 2.0, 3.0], "km/h")
    (tmp_path / "trace.mdf").write_bytes(b"OLD-MDF-")
    (tmp_path / "trace.mf4").write_bytes(b"OLD-MF4-")
    assert write_mdf(tmp_path / "trace.dat", [speed], 0.0, 1.0, version="3.30") == 1
    assert write_mdf(tmp_path / "x.mdf", [speed], 0.0, 1.0, version="4.10") == 1
    assert (tmp_path / "trace.mdf").read_bytes() == b"OLD-MDF-" and (tmp_path / "trace.mf4").read_bytes() == b"OLD-MF4-"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trace.dat", "trace.mdf", "trace.mf4", "x.mdf"]
    assert (tmp_path / "trace.dat").read_bytes()[:8] == b"MDF     " and (tmp_path / "x.mdf").read_bytes()[8:12] == b"4.10"
    with open_dataset(tmp_path / "trace.dat") as ds:
        assert ds.read(ds.find("Speed")).y.tolist() == [1.0, 2.0, 3.0]


def test_csv_export_with_an_integer_format_writes_infinite_values(tmp_path):
    from tracepaddock.core.export import ExportError, write_csv
    inv = _sig("inv", [0.0, 1.0, 2.0, 3.0], [1.0, np.inf, -np.inf, 4.0], "1/s")
    for fmt in ("%d", "%i", "%5d"):
        write_csv(tmp_path / "a.csv", [inv], 0.0, 3.0, number_format=fmt)
        lines = (tmp_path / "a.csv").read_text(encoding="utf-8-sig").splitlines()
        assert [line.split(";")[1].strip() for line in lines[2:]] == ["1", "inf", "-inf", "4"]
    with pytest.raises(ExportError):                               # a failed export leaves no half written file
        write_csv(tmp_path / "b.csv", [_sig("车速", [0.0], [1.0])], 0.0, 1.0, encoding="ascii")
    assert not (tmp_path / "b.csv").exists()


def test_mdf3_export_keeps_signals_with_chinese_names_apart(tmp_path):
    from tracepaddock.core.export import write_mdf
    from tracepaddock.core.model import SignalInfo
    from tracepaddock.core.registry import open_dataset
    t = [0.0, 1.0]
    signals = [_sig("车速", t, [1.0, 2.0], "km/h", SignalInfo("v_veh"), "车速"),
               _sig("转速", t, [3.0, 4.0], "1/min", SignalInfo("n_eng"), "转速"),
               _sig("功率", t, [5.0, 6.0], "kW", None, "功率"),
               _sig("扭矩", t, [7.0, 8.0], "Nm", None, "扭矩"),
               _sig("v_veh", t, [9.0, 10.0], "m/s")]
    renamed = {}
    assert write_mdf(tmp_path / "x.mdf", signals, 0.0, 1.0, version="3.30", renamed=renamed) == 5
    with open_dataset(tmp_path / "x.mdf") as ds:
        names = [s.name for s in ds.signals if s.name.lower() != "time"]
        values = {s.name: ds.read(s).y.tolist() for s in ds.signals if s.name.lower() != "time"}
    assert len(names) == 5 and len(set(names)) == 5 and not any("?" in n for n in names)
    assert values["v_veh"] == [9.0, 10.0] and values[renamed["车速"]] == [1.0, 2.0]
    assert values[renamed["转速"]] == [3.0, 4.0] and renamed["转速"] == "n_eng"
    assert set(renamed) == {"车速", "转速", "功率", "扭矩"}


# ---------------------------------------------------------------------- messages in the user's language (finding 9)
def test_runner_and_plugin_messages_are_translated(training_mf4, tmp_path, monkeypatch):
    from tracepaddock import i18n
    from tracepaddock.rules import runner
    from tracepaddock.rules.plugins import read_plugin
    table = {"no function run(api, params)": "缺少函数 run(api, params)",
             "parameter %s has an unknown type %s": "参数 %s 的类型 %s 未知",
             "cannot be read: %s": "无法读取：%s",
             "the plug-in has no function run(api, params)": "插件缺少函数 run(api, params)"}
    monkeypatch.setattr(i18n, "_table", dict(table))
    (tmp_path / "norun.py").write_text("NAME = 'x'\n", encoding="utf-8")
    assert read_plugin(tmp_path / "norun.py").problem == "缺少函数 run(api, params)"
    (tmp_path / "type.py").write_text("PARAMETERS = [{'name': 'a', 'type': 'colour'}]\ndef run(api, p):\n    pass\n",
                                      encoding="utf-8")
    assert read_plugin(tmp_path / "type.py").problem == "参数 a 的类型 colour 未知"
    (tmp_path / "syntax.py").write_text("def run(:\n", encoding="utf-8")
    assert read_plugin(tmp_path / "syntax.py").problem.startswith("无法读取：")
    monkeypatch.setattr(i18n, "set_language", lambda lang: None)   # the runner process selects the language
    with pytest.raises(runner.JobError) as info:
        runner._execute({"dataset": str(training_mf4), "plugin": str(tmp_path / "norun.py"), "language": "zh"})
    assert str(info.value) == "插件缺少函数 run(api, params)"


def test_the_time_limit_message_is_translated(training_mf4, tmp_path, monkeypatch):
    from tracepaddock import i18n
    from tracepaddock.rules import runner
    monkeypatch.setattr(i18n, "_table", {"the script did not finish within %g s and was stopped": "脚本未在 %g 秒内结束，已被停止"})
    pid_file = tmp_path / "pid.txt"
    try:
        with pytest.raises(runner.JobError) as info:
            runner.run_job({"dataset": str(training_mf4), "script": _endless(pid_file)}, timeout=5)
        assert str(info.value) == "脚本未在 5 秒内结束，已被停止"
    finally:
        _kill(_pid_or_none(pid_file))


# ---------------------------------------------------------------------- portable program (finding 10)
def _build_portable():
    spec = importlib.util.spec_from_file_location("build_portable_under_test", ROOT / "tools" / "build_portable.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_portable_program_contains_the_modules_scripts_may_import(training_mf4, qapp):
    from tracepaddock.rules.runner import SCRIPT_MODULES
    from tracepaddock.ui.script_dialog import ScriptDialog
    build = _build_portable()
    args = build.pyinstaller_args(Path("icon.ico"))
    hidden = {args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--hidden-import"}
    assert {"numpy", "math", "statistics", "collections", "re", "json", "datetime"} <= set(SCRIPT_MODULES)
    assert set(SCRIPT_MODULES) <= hidden
    assert all(importlib.util.find_spec(name) is not None for name in SCRIPT_MODULES)
    assert build.check_script_modules([sys.executable, "-m", "tracepaddock.rules.runner"], training_mf4) is None
    dialog = ScriptDialog("s", "", [], lambda code, name: {})
    help_text = dialog.findChild(QtWidgets.QTextBrowser).toPlainText()
    assert "statistics" in help_text and "datetime" in help_text
    dialog.deleteLater()


# ---------------------------------------------------------------------- errors in the window (excepthook)
def test_an_error_in_the_window_is_logged_and_shown_instead_of_ending_the_program(tmp_path):
    program = textwrap.dedent("""
        import sys, types
        from PySide6 import QtCore, QtWidgets
        stub = types.ModuleType("tracepaddock.ui.mainwindow")

        class Worker(QtCore.QThread):
            def run(self):
                raise KeyError("in a thread")

        class MainWindow(QtWidgets.QMainWindow):
            def __init__(self, settings):
                super().__init__()
                self.worker = Worker(self)
                QtCore.QTimer.singleShot(50, self.fail)
                QtCore.QTimer.singleShot(100, self.worker.start)
                QtCore.QTimer.singleShot(900, self.finish)

            def fail(self):
                raise ZeroDivisionError("broken slot")

            def finish(self):
                self.worker.wait()
                print("still running", flush=True)
                QtWidgets.QApplication.instance().quit()

            def open_dataset(self, *args):
                return False

            def open_plt(self, *args):
                pass

        stub.MainWindow = MainWindow
        import tracepaddock.ui
        sys.modules["tracepaddock.ui.mainwindow"] = stub
        shown = []
        QtWidgets.QMessageBox.critical = staticmethod(lambda *args: shown.append(args[2]) or QtWidgets.QMessageBox.Ok)
        import tracepaddock.__main__ as start
        code = start.main([])
        print("messages:", len(shown), "|".join(shown), flush=True)
        sys.exit(code)
    """)
    code, out, err = _run_python(program, tmp_path, 120)
    assert code == 0, (code, err[-2000:])
    assert "still running" in out and "messages: 2" in out
    assert "broken slot" in out and "in a thread" in out
    log = (tmp_path / "appdata" / "TracePaddock" / "logs" / "tracepaddock.log").read_text(encoding="utf-8")
    assert "ZeroDivisionError: broken slot" in log and "KeyError: 'in a thread'" in log


def test_rule_jobs_and_imports_keep_the_standard_error_handling(training_mf4, tmp_path):
    import tracepaddock.__main__ as start
    before = sys.excepthook
    from tracepaddock import i18n
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"dataset": str(training_mf4), "script": "log('ok')", "output": str(tmp_path / "out.json")}),
                   encoding="utf-8")
    try:
        assert start.main(["--rule-job", str(job)]) == 0
    finally:
        i18n.set_language("en")
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["log"] == ["ok"]
    assert sys.excepthook is before
