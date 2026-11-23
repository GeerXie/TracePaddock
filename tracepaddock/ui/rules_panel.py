"""Rule list, event table and logic chain (docked under the plot)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from ..rules.engine import Event, evaluate
from ..rules.model import Rule, RuleSet
from ..rules.plugins import Plugin, discover, plugin_ref, read_plugin, resolve_plugin
from ..rules.report import VERDICT_TEXT, RunResult, write_xlsx
from .render import VERDICT_COLORS
from .rule_form import RuleFormDialog
from .rule_graph import RuleGraphView
from .script_dialog import PluginDialog, ScriptDialog, describe_result, end_thread

__all__ = ["RulesPanel"]

_KIND_TEXT = {"form": "Form", "script": "Script", "plugin": "Plug-in"}
_VERDICT_LABEL = {"PASS": "Pass", "FAIL": "Fail", "TOO_EARLY": "Too early", "INCONCLUSIVE": "Not decided",
                  "FOUND": "Found"}


def _events(result: dict, name: str) -> Tuple[List[Event], Optional[str], List[Rule], Optional[float]]:
    """Events, error, rules and response period of a runner result."""
    if result.get("error"):
        return [], str(result["error"]), [], None
    events: List[Event] = []
    errors = []
    period = None
    for res in result.get("results", []):
        if res.get("error"):
            errors.append("%s: %s" % (res["rule"], res["error"]))
        events += [Event.from_json(e) for e in res.get("events", [])]
        period = period or res.get("response_period")
    events += [Event.from_json(e) for e in result.get("events", [])]
    rules = []
    for data in result.get("rules", []):
        try:
            rules.append(Rule.from_json(data))
        except (ValueError, TypeError):
            pass
    return sorted(events, key=lambda e: e.t0), ("; ".join(errors) or None), rules, period


class _JobsThread(QtCore.QThread):
    finished_all = QtCore.Signal(object)

    def __init__(self, jobs: List[Tuple[str, str, Callable[[], dict]]], parent=None) -> None:
        super().__init__(parent)
        self.jobs = jobs
        #: set when the viewer closes: the jobs not started yet are left out
        self.stopped = False

    def run(self) -> None:
        out = []
        for name, kind, fn in self.jobs:
            if self.stopped:
                break
            try:
                out.append((name, kind, fn()))
            except Exception as exc:
                out.append((name, kind, {"error": str(exc)}))
        self.finished_all.emit(out)


class RulesPanel(QtWidgets.QWidget):
    """``ctx`` is the main window; it provides the signals, the recording and the jobs (see MainWindow)."""

    def __init__(self, ctx, ruleset: RuleSet, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.ruleset = ruleset
        self.results: Dict[str, RunResult] = {}
        self.script_rules: Dict[str, List[Rule]] = {}
        self._thread: Optional[_JobsThread] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.b_new = QtWidgets.QPushButton(tr("New rule"))
        self.b_edit = QtWidgets.QPushButton(tr("Edit"))
        self.b_copy = QtWidgets.QPushButton(tr("Copy"))
        self.b_delete = QtWidgets.QPushButton(tr("Delete"))
        self.b_script = QtWidgets.QPushButton(tr("Python script..."))
        self.b_plugin = QtWidgets.QPushButton(tr("Plug-ins..."))
        self.b_run = QtWidgets.QPushButton(tr("Run all"))
        self.b_run.setToolTip("F5")
        self.range = QtWidgets.QComboBox()
        self.range.addItem(tr("whole recording"), "full")
        self.range.addItem(tr("visible time range"), "view")
        self.b_export = QtWidgets.QPushButton(tr("Excel report..."))
        for w in (self.b_new, self.b_edit, self.b_copy, self.b_delete):
            bar.addWidget(w)
        bar.addSpacing(12)
        bar.addWidget(self.b_script)
        bar.addWidget(self.b_plugin)
        bar.addSpacing(12)
        bar.addWidget(self.b_run)
        bar.addWidget(self.range)
        bar.addStretch(1)
        bar.addWidget(self.b_export)
        layout.addLayout(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels([tr("Rule"), tr("Type"), tr("Result")])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 220)
        self.tree.setColumnWidth(1, 60)
        split.addWidget(self.tree)
        self.tabs = QtWidgets.QTabWidget()
        events_page = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(events_page)
        el.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QComboBox()
        self.filter.addItem(tr("all events"), "")
        for key in ("FAIL", "PASS", "TOO_EARLY", "INCONCLUSIVE", "FOUND"):
            self.filter.addItem(tr(_VERDICT_LABEL[key]), key)
        self.only_selected = QtWidgets.QCheckBox(tr("only the selected rule"))
        self.summary = QtWidgets.QLabel()
        row.addWidget(self.filter)
        row.addWidget(self.only_selected)
        row.addStretch(1)
        row.addWidget(self.summary)
        el.addLayout(row)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([tr("Rule"), tr("Time [s]"), tr("Verdict"), tr("Delay [ms]"),
                                              tr("At the trigger"), tr("When decided"), tr("Note"), "#"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        for col, w in enumerate((170, 90, 70, 80, 300, 200, 200, 40)):
            self.table.setColumnWidth(col, w)
        self.table.setColumnHidden(7, True)
        self.table.setSortingEnabled(True)
        el.addWidget(self.table)
        self.tabs.addTab(events_page, tr("Events"))
        self.graph = RuleGraphView()
        self.tabs.addTab(self.graph, tr("Logic chain"))
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        layout.addWidget(split, 1)

        self.b_new.clicked.connect(self.new_rule)
        self.b_edit.clicked.connect(self.edit_current)
        self.b_copy.clicked.connect(self.copy_current)
        self.b_delete.clicked.connect(self.delete_current)
        self.b_script.clicked.connect(lambda: self.edit_script(None))
        self.b_plugin.clicked.connect(lambda: self.edit_plugin(None))
        self.b_run.clicked.connect(self.run_all)
        self.b_export.clicked.connect(self.export_xlsx)
        self.tree.itemDoubleClicked.connect(lambda *_: self.edit_current())
        self.tree.itemChanged.connect(self._item_checked)
        self.tree.currentItemChanged.connect(lambda *_: self._selection_changed())
        self.table.itemDoubleClicked.connect(self._jump)
        self.table.itemActivated.connect(self._jump)
        self.filter.currentIndexChanged.connect(self._fill_events)
        self.only_selected.toggled.connect(self._fill_events)
        self.graph.nodeActivated.connect(lambda _kind: self.edit_current())
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop_jobs)
        self.fill_tree()

    # ------------------------------------------------------------------ list
    def _items(self) -> List[Tuple[str, str]]:
        return ([("form", r.name) for r in self.ruleset.rules] + [("script", n) for n in self.ruleset.scripts]
                + [("plugin", n) for n in self.ruleset.plugin_runs])

    def _enabled(self, kind: str, name: str) -> bool:
        if kind == "form":
            rule = self._rule(name)
            return rule is not None and rule.enabled
        return name not in self.ruleset.disabled

    def _rule(self, name: str) -> Optional[Rule]:
        return next((r for r in self.ruleset.rules if r.name == name), None)

    def fill_tree(self, select: Optional[str] = None) -> None:
        current_item = self.tree.currentItem()
        current = select or (current_item.text(0) if current_item is not None else None)
        self.tree.blockSignals(True)
        self.tree.clear()
        for kind, name in self._items():
            res = self.results.get(name)
            text = ""
            if res is not None:
                text = res.error or ("%d · %s %d · %s %d" % (len(res.events), tr("pass"), res.count("PASS"), tr("fail"),
                                                             res.count("FAIL") + res.count("TOO_EARLY"))
                                     if any(e.verdict != "FOUND" for e in res.events) else
                                     "%d %s" % (len(res.events), tr("events")))
            item = QtWidgets.QTreeWidgetItem([name, tr(_KIND_TEXT[kind]), text])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (kind, name))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            checks = QtCore.Qt.CheckState
            item.setCheckState(0, checks.Checked if self._enabled(kind, name) else checks.Unchecked)
            if res is not None and (res.error or res.count("FAIL") or res.count("TOO_EARLY")):
                item.setForeground(2, QtGui.QBrush(QtGui.QColor(*VERDICT_COLORS["FAIL"])))
            elif res is not None and res.events:
                item.setForeground(2, QtGui.QBrush(QtGui.QColor(*VERDICT_COLORS["PASS"])))
            self.tree.addTopLevelItem(item)
            if name == current:
                self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        self._selection_changed()

    def _item_checked(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        kind, name = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        on = item.checkState(0) == QtCore.Qt.CheckState.Checked
        if kind == "form":
            rule = self._rule(name)
            if rule is not None:
                rule.enabled = on
        elif on:
            self.ruleset.disabled.discard(name)
        else:
            self.ruleset.disabled.add(name)
        self.ruleset.save()
        self._update_lanes()

    def current(self) -> Optional[Tuple[str, str]]:
        item = self.tree.currentItem()
        return item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item is not None else None

    def _selection_changed(self) -> None:
        cur = self.current()
        rule = None
        if cur is not None:
            kind, name = cur
            rule = self._rule(name) if kind == "form" else next(iter(self.script_rules.get(name, [])), None)
        res = self.results.get(cur[1]) if cur else None
        summary: Optional[Dict[str, object]] = None
        if res is not None and rule is not None:
            summary = {v: res.count(v) for v in VERDICT_TEXT}
            summary["events"] = len(res.events)
            d = res.delays()
            if d:
                summary.update(delay_min=min(d), delay_max=max(d))
        self.graph.show_rule(rule, summary, self.ctx.rule_value_names(rule) if rule else None)
        for b in (self.b_edit, self.b_copy, self.b_delete):
            b.setEnabled(cur is not None)
        if self.only_selected.isChecked():
            self._fill_events()

    # ------------------------------------------------------------------ editing
    def _form(self, rule: Optional[Rule]) -> Optional[Rule]:
        c = self.ctx.rule_context()
        dialog = RuleFormDialog(rule, c["names"], c["enums"], c["hint"], c["period"], c["all_names"],
                                self.ruleset.all_names(), self)
        return dialog.rule() if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted else None

    def new_rule(self) -> None:
        rule = self._form(None)
        if rule is not None:
            self.ruleset.rules.append(rule)
            self._changed(rule.name)

    def edit_current(self) -> None:
        cur = self.current()
        if cur is None:
            return
        kind, name = cur
        if kind == "form":
            old = self._rule(name)
            if old is None:
                return
            new = self._form(old)
            if new is not None:
                new.enabled = old.enabled
                self.ruleset.rules[self.ruleset.rules.index(old)] = new
                self.results.pop(name, None)
                self._changed(new.name)
        elif kind == "script":
            self.edit_script(name)
        else:
            self.edit_plugin(name)

    def edit_script(self, name: Optional[str]) -> None:
        code = self.ruleset.scripts.get(name, "") if name else ""
        c = self.ctx.rule_context()
        dialog = ScriptDialog(name or self.ruleset.unique_name(tr("Script")), code, c["names"],
                              lambda code, n: self.ctx.run_script(code, n), self)
        if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            new = dialog.script_name()
            if name and new != name:
                self.ruleset.scripts.pop(name, None)
                self.results.pop(name, None)
            if new != name and new in self.ruleset.all_names():
                new = self.ruleset.unique_name(new)
            self.ruleset.scripts[new] = dialog.code()
            if dialog.last_result is not None:
                self._take(new, "script", dialog.last_result)
            self._changed(new)

    def edit_plugin(self, name: Optional[str]) -> None:
        preset = self.ruleset.plugin_runs.get(name) if name else None
        c = self.ctx.rule_context()
        dialog = PluginDialog(discover(), c["names"], lambda p, params: self.ctx.run_plugin(str(p.path), params, p.name),
                              self, preset)
        if dialog.exec_() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        plugin = dialog.plugin()
        if plugin is None:
            return
        new = name or self.ruleset.unique_name(plugin.name)
        self.ruleset.plugin_runs[new] = {"path": plugin_ref(plugin.path), "params": dialog.values()}
        if dialog.last_result is not None:
            self._take(new, "plugin", dialog.last_result)
        self._changed(new)

    def copy_current(self) -> None:
        cur = self.current()
        if cur is None:
            return
        kind, name = cur
        new = self.ruleset.unique_name(name)
        if kind == "form":
            source = self._rule(name)
            if source is None:
                return
            rule = copy.deepcopy(source)
            rule.name = new
            self.ruleset.rules.append(rule)
        elif kind == "script":
            self.ruleset.scripts[new] = self.ruleset.scripts[name]
        else:
            self.ruleset.plugin_runs[new] = copy.deepcopy(self.ruleset.plugin_runs[name])
        self._changed(new)

    def delete_current(self) -> None:
        cur = self.current()
        if cur is None:
            return
        kind, name = cur
        if (QtWidgets.QMessageBox.question(self, tr("Rules"), tr("Delete '%s'?") % name)
                != QtWidgets.QMessageBox.StandardButton.Yes):
            return
        if kind == "form":
            self.ruleset.rules = [r for r in self.ruleset.rules if r.name != name]
        elif kind == "script":
            self.ruleset.scripts.pop(name, None)
        else:
            self.ruleset.plugin_runs.pop(name, None)
        self.ruleset.disabled.discard(name)
        self.results.pop(name, None)
        self._changed(None)

    def _changed(self, select: Optional[str]) -> None:
        self.ruleset.save()
        self.fill_tree(select)
        self._fill_events()
        self._update_lanes()

    # ------------------------------------------------------------------ running
    def time_range(self) -> Optional[Tuple[float, float]]:
        return self.ctx.view_range() if self.range.currentData() == "view" else None

    def run_all(self) -> None:
        if not self.ctx.has_data() or (self._thread is not None and self._thread.isRunning()):
            return
        rng = self.time_range()
        source = self.ctx.rule_source()
        for rule in self.ruleset.rules:
            if rule.enabled:
                res = evaluate(rule, source, rng)
                names = self.ctx.rule_value_names(rule)
                self.results[rule.name] = RunResult(rule.name, "form", res.events, res.error, rule.trigger.text(names),
                                                    rule.expect.text(names) if rule.expect else "", res.response_period)
            else:
                self.results.pop(rule.name, None)
        jobs = []
        for name, code in self.ruleset.scripts.items():
            if name not in self.ruleset.disabled:
                jobs.append((name, "script", lambda c=code, n=name: self.ctx.run_script(c, n, rng)))
        for name, run in self.ruleset.plugin_runs.items():
            if name not in self.ruleset.disabled:
                jobs.append((name, "plugin", lambda r=run, n=name: self.ctx.run_plugin(str(resolve_plugin(r["path"])),
                                                                                       r["params"], n, rng)))
        self.fill_tree()
        self._fill_events()
        self._update_lanes()
        if jobs:
            self.summary.setText(tr("running %d scripts / plug-ins ...") % len(jobs))
            self.b_run.setEnabled(False)
            self._thread = _JobsThread(jobs, self)
            self._thread.finished_all.connect(self._jobs_done)
            self._thread.start()

    def wait(self, ms: int = 120000) -> None:
        """Wait for running scripts (tests, demo)."""
        if self._thread is not None:
            self._thread.wait(ms)
            QtWidgets.QApplication.processEvents()

    def stop_jobs(self) -> None:
        """The viewer closes: end the running scripts and plug-ins at once instead of waiting for them."""
        if self._thread is not None:
            self._thread.stopped = True
        end_thread(self._thread)

    def _jobs_done(self, out) -> None:
        for name, kind, result in out:
            self._take(name, kind, result)
        self.b_run.setEnabled(True)
        self.fill_tree()
        self._fill_events()
        self._update_lanes()

    def _take(self, name: str, kind: str, result: dict) -> None:
        events, error, rules, period = _events(result, name)
        for e in events:
            if not e.rule or e.rule != name and len(rules) <= 1:
                e.rule = name
        self.script_rules[name] = rules
        trig = "; ".join(r.trigger.text() for r in rules)
        exp = "; ".join(r.expect.text() for r in rules if r.expect is not None)
        self.results[name] = RunResult(name, kind, events, error, trig, exp, period, list(result.get("log", [])))

    def clear_results(self) -> None:
        self.results.clear()
        self.fill_tree()
        self._fill_events()
        self._update_lanes()

    # ------------------------------------------------------------------ events
    def _visible_results(self) -> List[RunResult]:
        names = [n for k, n in self._items() if self._enabled(k, n)]
        return [self.results[n] for n in names if n in self.results]

    def _fill_events(self, *_args) -> None:
        wanted = self.filter.currentData()
        current = self.current() if self.only_selected.isChecked() else None
        only = current[1] if current is not None else None
        rows = []
        for res in self._visible_results():
            if only and res.name != only:
                continue
            for e in res.events:
                if not wanted or e.verdict == wanted:
                    rows.append((res, e))
        rows.sort(key=lambda r: r[1].t0)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        self._rows = rows
        for i, (res, e) in enumerate(rows):
            at = "; ".join("%s=%s" % (k, e.texts.get(k, "%g" % v)) for k, v in e.values.items())
            when = "; ".join("%s=%s" % (k, e.actual_texts.get(k, "%g" % v)) for k, v in e.actual.items())
            note = e.note + (("  " + tr("(already met at the trigger)")) if e.already else "")
            cells = [res.name, "%.3f" % e.t0, tr(_VERDICT_LABEL.get(e.verdict) or e.verdict),
                     "" if e.delay is None else "%.0f" % (e.delay * 1000), at, when, note.strip(), str(i)]
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if col in (1, 3) and text:
                    item.setData(QtCore.Qt.ItemDataRole.EditRole, float(text))
                if col == 2:
                    item.setBackground(QtGui.QColor(*[min(255, c + 120) for c in VERDICT_COLORS.get(e.verdict, (200, 200, 200))]))
                    item.setForeground(QtGui.QColor(30, 30, 30))
                self.table.setItem(i, col, item)
        self.table.setSortingEnabled(True)
        results = self._visible_results()
        n_events = sum(len(r.events) for r in results)
        passed = sum(r.count("PASS") for r in results)
        failed = sum(r.count("FAIL") + r.count("TOO_EARLY") for r in results)
        open_ = sum(r.count("INCONCLUSIVE") for r in results)
        delays = [d for r in results for d in r.delays()]
        text = tr("%d events · pass %d · fail %d · not decided %d") % (n_events, passed, failed, open_)
        if delays:
            text += "  ·  " + tr("delay %.0f / %.0f / %.0f ms (min / mean / max)") % (
                min(delays) * 1000, sum(delays) / len(delays) * 1000, max(delays) * 1000)
        self.summary.setText(text)

    def _jump(self, item: QtWidgets.QTableWidgetItem) -> None:
        index = self.table.item(item.row(), 7)
        if index is None:
            return
        res, e = self._rows[int(index.text())]
        self.ctx.jump_to(e.t0, e.t1 if e.t1 is not None else e.t_end)

    def _update_lanes(self) -> None:
        lanes = []
        for res in self._visible_results():
            if res.events:
                lanes.append((res.name, [(e.t0, e.t1 if e.t1 is not None else e.t_end, e.verdict) for e in res.events]))
        self.ctx.set_rule_lanes(lanes[:8])

    # ------------------------------------------------------------------ report
    def export_xlsx(self) -> None:
        results = self._visible_results()
        if not results:
            QtWidgets.QMessageBox.information(self, tr("Rules"), tr("Run the rules first."))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("Excel report"), self.ctx.report_default_path(),
                                                        "Excel (*.xlsx)")
        if not path:
            return
        try:
            write_xlsx(path, results, self.ctx.report_meta())
        except (OSError, ImportError) as exc:
            QtWidgets.QMessageBox.warning(self, tr("Rules"), str(exc))
            return
        except Exception as exc:                            # an unexpected problem: report it, keep the viewer
            QtWidgets.QMessageBox.warning(self, tr("Rules"), "%s: %s" % (type(exc).__name__, exc))
            return
        self.ctx.statusBar().showMessage(tr("Saved %s") % path, 6000)
