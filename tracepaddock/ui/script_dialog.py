"""Script window and plug-in window for rules (the code runs in a separate process)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from ..rules.plugins import Plugin, plugin_ref, user_folder
from ..rules.runner import SCRIPT_MODULES, stop_all_jobs

__all__ = ["ScriptDialog", "PluginDialog", "PythonHighlighter", "EXAMPLE_SCRIPT", "describe_result", "end_thread"]

EXAMPLE_SCRIPT = '''# Open lesson 03 (Trigger and response) from Help > Quick warm-up.
# S("name") refers to a recorded signal or a name in the current view.
request = S("TutorialRequest")
ack = S("TutorialAck")

rule("Request acknowledgement",
     trigger=becomes(request == 1),
     expect=ack == 1,
     within=0.5,
     already="hold",
     scope=S("TutorialEnabled") == 1,
     record=["TutorialAckStatus"])

# Use the original sample arrays to add your own events.
t, y = data("TutorialInputTravel")
changes = np.flatnonzero(np.abs(np.diff(y)) > 5)
for k in changes:
    event(t[k + 1], "input changes by %.1f percentage points" % (y[k + 1] - y[k]))
log("%d input changes" % len(changes))
'''

HELP = """<h3>%s</h3>
 <p><b>%s</b><br><code>S("name")</code> %s</p>
 <p><b>%s</b><br>
 <code>S("a") &gt; 90</code>, <code>S("a") == S("b")</code><br>
 <code>S("v_ms") * 3.6 &gt; S("lim") + 3</code><br>
 <code>S("a").isin([1, 2])</code>, <code>.between(10, 20)</code><br>
 <code>.equals(S("b"), tol=0.5)</code><br>
 <code>.changes()</code>, <code>.changes_to(2)</code>, <code>.changes_from(1, 2)</code><br>
 <code>.rises()</code>, <code>.falls()</code><br>
 <code>cond1 &amp; cond2</code>, <code>cond1 | cond2</code>, <code>~cond</code></p>
 <p><b>%s</b><br>
 <code>becomes(cond, hold=1.0)</code> %s<br>
 <code>while_(cond)</code> %s</p>
 <p><code>rule(<br>
 &nbsp;&nbsp;name,<br>
 &nbsp;&nbsp;trigger=...,<br>
 &nbsp;&nbsp;expect=...,<br>
 &nbsp;&nbsp;within=(0, 0.5),<br>
 &nbsp;&nbsp;already="hold",<br>
 &nbsp;&nbsp;scope=...,<br>
 &nbsp;&nbsp;record=[...],<br>
 &nbsp;&nbsp;expect_hold=0,<br>
 &nbsp;&nbsp;count_initial=False,<br>
 )</code><br>
 <code>already</code> · <code>"hold"</code>, <code>"new_sample"</code>, <code>"change"</code></p>
 <p><b>%s</b><br>
 <code>t, y = data("name")</code> (numpy)<br>
 <code>event(time, "text", verdict="FOUND")</code><br>
 <code>log(...)</code> · <code>np</code></p>
 <p><b>%s</b><br><code>%s</code></p>
"""


def describe_result(result: dict) -> str:
    """Short text about a runner result for the output pane."""
    if result.get("error"):
        return tr("Error:") + "\n" + str(result["error"])
    lines = []
    for res in result.get("results", []):
        if res.get("error"):
            lines.append("%s: %s" % (res["rule"], res["error"]))
            continue
        verdicts: Dict[str, int] = {}
        for e in res.get("events", []):
            verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
        lines.append("%s: %d %s %s" % (res["rule"], len(res.get("events", [])), tr("events"),
                                       " ".join("%s %d" % (k, v) for k, v in sorted(verdicts.items()))))
    if result.get("events"):
        lines.append(tr("%d events from the script itself") % len(result["events"]))
    for message in result.get("log", []):
        lines.append("> " + message)
    if result.get("seconds") is not None:
        lines.append(tr("finished in %.1f s") % result["seconds"])
    return "\n".join(lines) or tr("no rules and no events")


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    KEYWORDS = ("and", "or", "not", "if", "else", "elif", "for", "while", "in", "def", "return", "import", "from",
                "as", "True", "False", "None", "lambda", "try", "except", "with", "pass", "break", "continue")
    API = ("S", "rule", "becomes", "while_", "data", "event", "log", "np", "params")

    def __init__(self, document: QtGui.QTextDocument) -> None:
        super().__init__(document)

        def fmt(color: str, bold: bool = False) -> QtGui.QTextCharFormat:
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            if bold:
                f.setFontWeight(QtGui.QFont.Bold)
            return f

        dark = QtWidgets.QApplication.palette().color(QtGui.QPalette.Base).lightness() < 128
        colours = ("#569cd6", "#c586c0", "#b5cea8", "#ce9178", "#6a9955") if dark else \
            ("#0033b3", "#7a3e9d", "#1750eb", "#067d17", "#8c8c8c")
        self.rules = [(QtCore.QRegularExpression(r"\b(%s)\b" % "|".join(self.KEYWORDS)), fmt(colours[0], True)),
                      (QtCore.QRegularExpression(r"\b(%s)\b(?=\s*[(.])" % "|".join(self.API)), fmt(colours[1], True)),
                      (QtCore.QRegularExpression(r"\b\d+(\.\d+)?\b"), fmt(colours[2])),
                      (QtCore.QRegularExpression(r"\"[^\"]*\"|'[^']*'"), fmt(colours[3])),
                      (QtCore.QRegularExpression(r"#[^\n]*"), fmt(colours[4]))]

    def highlightBlock(self, text: str) -> None:
        for regex, form in self.rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), form)


def end_thread(thread: Optional[QtCore.QThread], wait_ms: int = 10000) -> None:
    """The viewer closes: end the script processes and wait until ``thread`` (which waits for them) is done.

    A thread still running when its window is destroyed would end the whole program with an error."""
    if thread is None or not thread.isRunning():
        return
    deadline = time.monotonic() + wait_ms / 1000.0
    while thread.isRunning() and time.monotonic() < deadline:
        stop_all_jobs()                                    # again: the thread may just have started the next one
        thread.wait(100)


class _Worker(QtCore.QThread):
    done = QtCore.Signal(object)

    def __init__(self, fn: Callable[[], dict], parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self.fn())
        except Exception as exc:                            # time limit, crash of the child process
            self.done.emit({"error": str(exc)})


class _RunMixin:
    def _start(self, fn: Callable[[], dict]) -> None:
        if getattr(self, "_worker", None) is not None and self._worker.isRunning():
            return
        self.output.setPlainText(tr("running ..."))
        self._set_busy(True)
        self._worker = _Worker(fn, self)
        self._worker.done.connect(self._finished)
        if not getattr(self, "_ends_on_quit", False):
            QtWidgets.QApplication.instance().aboutToQuit.connect(self._end_on_quit)
            self._ends_on_quit = True
        self._worker.start()

    def _end_on_quit(self) -> None:
        end_thread(getattr(self, "_worker", None))

    def _finished(self, result: dict) -> None:
        self._set_busy(False)
        self.last_result = result
        self.output.setPlainText(describe_result(result))


class ScriptDialog(QtWidgets.QDialog, _RunMixin):
    """Edit a rule script. ``run(code, name) -> result dict`` runs it in the runner process."""

    def __init__(self, name: str, code: str, names: Sequence[str], run: Callable[[str, str], dict],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Rule script (Python)"))
        self.resize(1300, 850)
        self.run_fn = run
        self.last_result: Optional[dict] = None
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("Name")))
        self.name = QtWidgets.QLineEdit(name)
        row.addWidget(self.name, 1)
        layout.addLayout(row)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        self.editor = QtWidgets.QPlainTextEdit(code or EXAMPLE_SCRIPT)
        font = QtGui.QFont("Consolas")
        font.setPointSize(10)
        self.editor.setFont(font)
        self.editor.setTabStopDistance(4 * QtGui.QFontMetricsF(font).horizontalAdvance(" "))
        self.highlighter = PythonHighlighter(self.editor.document())
        split.addWidget(self.editor)
        side = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        side.setChildrenCollapsible(False)
        help_view = QtWidgets.QTextBrowser()
        help_view.setMinimumHeight(105)
        help_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        help_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        help_view.setWordWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        help_view.setHtml(HELP % (tr("Vocabulary"), tr("Signal references"), tr("a signal"),
                                  tr("Comparisons and conditions"), tr("Triggers and rules"),
                                  tr("trigger with minimum duration"), tr("one event per interval"),
                                  tr("Data and events"),
                                  tr("Modules that can be imported (also in the portable program):"),
                                  ", ".join(SCRIPT_MODULES)))
        side.addWidget(help_view)
        catalogue = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(catalogue)
        sl.setContentsMargins(0, 0, 0, 0)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText(tr("filter signals; double click inserts S(\"name\")"))
        sl.addWidget(self.filter)
        self.signals = QtWidgets.QListWidget()
        self.signals.addItems(list(names))
        sl.addWidget(self.signals, 1)
        side.addWidget(catalogue)
        side.setSizes([390, 170])
        split.addWidget(side)
        split.setSizes([790, 480])
        layout.addWidget(split, 1)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(170)
        self.output.setFont(font)
        layout.addWidget(self.output)
        buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton(tr("Run (preview)"))
        self.save_button = QtWidgets.QPushButton(tr("Save to the rule list"))
        close = QtWidgets.QPushButton(tr("Close"))
        buttons.addWidget(self.run_button)
        buttons.addStretch(1)
        buttons.addWidget(self.save_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.run_button.clicked.connect(lambda: self._start(lambda: self.run_fn(self.code(), self.script_name())))
        self.save_button.clicked.connect(self._save)
        close.clicked.connect(self.reject)
        self.filter.textChanged.connect(self._filter)
        self.signals.itemDoubleClicked.connect(lambda item: self.editor.insertPlainText('S("%s")' % item.text()))

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)

    def _filter(self, text: str) -> None:
        needle = text.lower()
        for i in range(self.signals.count()):
            item = self.signals.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _save(self) -> None:
        if not self.script_name():
            self.output.setPlainText(tr("The script needs a name."))
            return
        self.accept()

    def code(self) -> str:
        return self.editor.toPlainText()

    def script_name(self) -> str:
        return self.name.text().strip()


class PluginDialog(QtWidgets.QDialog, _RunMixin):
    """Choose a plug-in, set its parameters, run it once or add it to the rule list."""

    def __init__(self, plugins: Sequence[Plugin], names: Sequence[str], run: Callable[[Plugin, dict], dict],
                 parent: Optional[QtWidgets.QWidget] = None, preset: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Rule plug-ins"))
        self.resize(1050, 700)
        self.plugins = [p for p in plugins if p.problem is None]
        self.names = list(names)
        self.run_fn = run
        self.last_result: Optional[dict] = None
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        self.list = QtWidgets.QListWidget()
        for p in self.plugins:
            self.list.addItem(p.name)
        left.addWidget(self.list, 1)
        folder = QtWidgets.QPushButton(tr("Open the plug-in folder"))
        left.addWidget(folder)
        broken = [p for p in plugins if p.problem]
        if broken:
            note = QtWidgets.QLabel(tr("Not usable:") + "\n" + "\n".join("%s: %s" % (p.path.name, p.problem) for p in broken))
            note.setStyleSheet("color: #c00000")
            note.setWordWrap(True)
            left.addWidget(note)
        layout.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        right.addWidget(self.description)
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        right.addWidget(self.form_host)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        right.addWidget(self.output, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton(tr("Run (preview)"))
        self.add_button = QtWidgets.QPushButton(tr("Add to the rule list"))
        close = QtWidgets.QPushButton(tr("Close"))
        buttons.addWidget(self.run_button)
        buttons.addStretch(1)
        buttons.addWidget(self.add_button)
        buttons.addWidget(close)
        right.addLayout(buttons)
        layout.addLayout(right, 2)
        self.list.currentRowChanged.connect(self._show)
        self.run_button.clicked.connect(lambda: self._start(lambda: self.run_fn(self.plugin(), self.values())))
        self.add_button.clicked.connect(self.accept)
        close.clicked.connect(self.reject)
        folder.clicked.connect(self._open_folder)
        self._preset = preset or {}
        if self.plugins:
            index = 0
            if preset and preset.get("path"):
                index = next((i for i, p in enumerate(self.plugins) if self._is_preset(p)), 0)
            self.list.setCurrentRow(index)
        else:
            self.description.setText(tr("No plug-ins found."))
            self.run_button.setEnabled(False)
            self.add_button.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)

    def _is_preset(self, plugin: Plugin) -> bool:
        """``plugin`` is the one of the edited rule (stored as a reference or as an older absolute path)."""
        given = self._preset.get("path")
        return bool(given) and plugin_ref(given) == plugin_ref(plugin.path)

    def plugin(self) -> Optional[Plugin]:
        row = self.list.currentRow()
        return self.plugins[row] if 0 <= row < len(self.plugins) else None

    def _show(self, row: int) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self._widgets.clear()
        plugin = self.plugin()
        if plugin is None:
            return
        self.description.setText("<b>%s</b><br>%s<br><span style='color:gray'>%s</span>" % (
            plugin.name, plugin.description, plugin.path))
        given = self._preset.get("params", {}) if self._is_preset(plugin) else {}
        for par in plugin.parameters:
            key, kind = str(par["name"]), str(par.get("type", "text"))
            value = given.get(key, par.get("default"))
            if kind == "signal":
                w = QtWidgets.QComboBox()
                w.setEditable(True)
                w.addItems(self.names)
                w.setCurrentText("" if value is None else str(value))
            elif kind == "number":
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(-1e9, 1e9)
                w.setDecimals(6)
                w.setValue(float(value or 0.0))
            elif kind == "choice":
                w = QtWidgets.QComboBox()
                w.addItems([str(c) for c in par.get("choices", [])])
                w.setCurrentText(str(value))
            elif kind == "bool":
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(value))
            elif kind == "signals":
                w = QtWidgets.QLineEdit(", ".join(value) if isinstance(value, (list, tuple)) else str(value or ""))
            else:
                w = QtWidgets.QLineEdit("" if value is None else str(value))
            self.form.addRow(str(par.get("label", key)), w)
            self._widgets[key] = w

    def values(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        plugin = self.plugin()
        for par in (plugin.parameters if plugin else []):
            key, kind = str(par["name"]), str(par.get("type", "text"))
            w = self._widgets.get(key)
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                out[key] = w.value()
            elif isinstance(w, QtWidgets.QCheckBox):
                out[key] = w.isChecked()
            elif isinstance(w, QtWidgets.QComboBox):
                out[key] = w.currentText().strip()
            elif kind == "signals":
                out[key] = [p.strip() for p in w.text().split(",") if p.strip()]
            else:
                out[key] = w.text().strip()
        return out

    def _open_folder(self) -> None:
        folder = user_folder()
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
