"""Form editor for rules: trigger -> expectation within a time window, optional scope."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from ..rules.model import ALREADY_MODES, OPERATORS, Condition, Group, Operand, Rule

__all__ = ["ConditionsEditor", "RuleFormDialog", "parse_value", "value_text"]

_OP_LABELS = {
    "==": "=", "!=": "≠", "<": "<", "<=": "≤", ">": ">", ">=": "≥", "in": "in (one of)", "not in": "not in",
    "between": "between", "changes": "changes", "changes to": "changes to", "changes from": "changes from",
    "rises": "rises", "falls": "falls",
}
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_OPERAND = re.compile(r"^=\s*(?:(?P<k1>" + _NUM + r")\s*\*\s*)?(?P<name>[^*+]+?)\s*(?:\*\s*(?P<k2>" + _NUM + r"))?"
                      r"\s*(?:(?P<sign>[-+])\s*(?P<off>" + _NUM + r"))?\s*$")


def _token(text: str, enum: Optional[Dict[float, str]]) -> float:
    tok = text.strip()
    if "=" in tok:                                       # "1 = 非条件限速" from the value list
        tok = tok.split("=", 1)[0].strip()
    try:
        return float(int(tok, 16)) if tok.lower().startswith("0x") else float(tok)
    except ValueError:
        pass
    for value, name in (enum or {}).items():
        if name.strip().lower() == text.strip().lower():
            return float(value)
    raise ValueError(tr("'%s' is not a number or a value text of the signal") % text.strip())


def parse_value(op: str, text: str, enum: Optional[Dict[float, str]] = None):
    """Text of the value cell -> (value, previous). Raises ``ValueError`` with a message."""
    text = (text or "").strip()
    if op in ("changes", "rises", "falls"):
        return None, None
    if text.startswith("="):
        if op not in ("==", "!=", "<", "<=", ">", ">="):
            raise ValueError(tr("only =, ≠, <, ≤, >, ≥ can compare with another signal"))
        m = _OPERAND.match(text)
        if not m:
            raise ValueError(tr("write another signal as =name, =3.6*name or =name*3.6+1"))
        k = float(m.group("k1") or 1.0) * float(m.group("k2") or 1.0)
        off = float(m.group("off") or 0.0) * (-1.0 if m.group("sign") == "-" else 1.0)
        return Operand(m.group("name").strip(), k, off), None
    if not text:
        raise ValueError(tr("a value is needed"))
    if op == "between":
        parts = [p for p in re.split(r"\.\.|,|;|~", text) if p.strip()]
        if len(parts) != 2:
            raise ValueError(tr("write a range as 10..20"))
        return [_token(parts[0], enum), _token(parts[1], enum)], None
    if op == "changes from":
        parts = [p for p in re.split(r"->|→|>", text) if p.strip()]
        old = _token(parts[0], enum)
        new = _token(parts[1], enum) if len(parts) > 1 else None
        return new, old
    if op in ("in", "not in", "changes to"):
        values = [_token(p, enum) for p in re.split(r"[,;]", text) if p.strip()]
        if not values:
            raise ValueError(tr("a value is needed"))
        return (values if op != "changes to" or len(values) > 1 else values[0]), None
    return _token(text, enum), None


def value_text(cond: Condition, enum: Optional[Dict[float, str]] = None) -> str:
    def one(v):
        name = (enum or {}).get(float(v)) if v is not None else None
        return ("%g = %s" % (v, name)) if name else "%g" % v

    if cond.op in ("changes", "rises", "falls"):
        return ""
    if isinstance(cond.value, Operand):
        o = cond.value
        out = "=" + ("%g*" % o.scale if o.scale != 1.0 else "") + o.signal
        if o.offset:
            out += ("+%g" % o.offset) if o.offset > 0 else ("%g" % o.offset)
        return out
    if cond.op == "between" and isinstance(cond.value, list):
        return "%g..%g" % tuple(cond.value)
    if cond.op == "changes from":
        return "%g -> %s" % (cond.previous if cond.previous is not None else 0, "%g" % cond.value if cond.value is not None else "")
    if isinstance(cond.value, list):
        return ", ".join("%g" % v for v in cond.value) if len(cond.value) > 1 else one(cond.value[0])
    return one(cond.value) if cond.value is not None else ""


class ConditionsEditor(QtWidgets.QWidget):
    """Rows "signal | comparison | value | tolerance", combined with AND or OR."""

    changed = QtCore.Signal()

    def __init__(self, names: Sequence[str], enums: Callable[[str], Dict[float, str]],
                 hint: Callable[[str], str], completer: Optional[QtWidgets.QCompleter] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.names, self.enums, self.hint, self.completer = list(names), enums, hint, completer
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem(tr("all rows hold (AND)"), "and")
        self.mode.addItem(tr("any row holds (OR)"), "or")
        self.negate = QtWidgets.QCheckBox(tr("negate (NOT)"))
        add = QtWidgets.QPushButton(tr("+ Condition"))
        top.addWidget(self.mode)
        top.addWidget(self.negate)
        top.addStretch(1)
        top.addWidget(add)
        layout.addLayout(top)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([tr("Signal"), tr("Comparison"), tr("Value"), tr("Tolerance"), ""])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(3, 90)
        self.table.setColumnWidth(4, 30)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(96)
        layout.addWidget(self.table)
        self.note = QtWidgets.QLabel()
        self.note.setStyleSheet("color: gray")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        add.clicked.connect(lambda: self.add_row())
        self.mode.currentIndexChanged.connect(self.changed)
        self.negate.toggled.connect(self.changed)

    # ------------------------------------------------------------------ rows
    def add_row(self, cond: Optional[Condition] = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        sig = QtWidgets.QComboBox()
        sig.setEditable(True)
        sig.addItems(self.names)
        if self.completer is not None:
            sig.setCompleter(self.completer)
        sig.setCurrentText(cond.signal if cond else (self.names[0] if self.names else ""))
        op = QtWidgets.QComboBox()
        for key in OPERATORS:
            op.addItem(tr(_OP_LABELS[key]), key)
        op.setCurrentIndex(max(0, op.findData(cond.op if cond else "==")))
        value = QtWidgets.QComboBox()
        value.setEditable(True)
        value.setToolTip(tr("A number, a value text of the signal, a list 1, 2, 3, a range 10..20, "
                            "or =name to compare with another signal (=3.6*v_ms+1 also works)"))
        tol = QtWidgets.QLineEdit("%g" % cond.tolerance if cond and cond.tolerance else "")
        delete = QtWidgets.QToolButton()
        delete.setText("✕")
        self.table.setCellWidget(row, 0, sig)
        self.table.setCellWidget(row, 1, op)
        self.table.setCellWidget(row, 2, value)
        self.table.setCellWidget(row, 3, tol)
        self.table.setCellWidget(row, 4, delete)
        self._fill_values(sig, value)
        if cond is not None:
            value.setCurrentText(value_text(cond, self.enums(cond.signal)))
        sig.currentTextChanged.connect(lambda _t, s=sig, v=value, tl=tol: self._signal_changed(s, v, tl))
        for w in (op, value):
            w.currentIndexChanged.connect(self.changed)
        value.editTextChanged.connect(self.changed)
        tol.textChanged.connect(self.changed)
        delete.clicked.connect(lambda _c=False, d=delete: self._delete(d))
        self._signal_changed(sig, value, tol, keep_value=True)
        self.changed.emit()

    def _fill_values(self, sig: QtWidgets.QComboBox, value: QtWidgets.QComboBox) -> None:
        current = value.currentText()
        value.blockSignals(True)
        value.clear()
        for v, text in sorted((self.enums(sig.currentText()) or {}).items()):
            value.addItem("%g = %s" % (v, text))
        value.setCurrentText(current)
        value.blockSignals(False)

    def _signal_changed(self, sig, value, tol, keep_value: bool = False) -> None:
        self._fill_values(sig, value)
        hint = self.hint(sig.currentText())
        tol.setPlaceholderText(hint.split("|")[0] if hint else "")
        tol.setToolTip(hint.split("|")[-1] if hint else "")
        self.changed.emit()

    def _delete(self, button: QtWidgets.QWidget) -> None:
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 4) is button:
                self.table.removeRow(r)
                break
        self.changed.emit()

    # ------------------------------------------------------------------ value
    def set_group(self, group: Optional[Group]) -> None:
        self.table.setRowCount(0)
        if group is None:
            return
        self.mode.setCurrentIndex(0 if group.mode == "and" else 1)
        self.negate.setChecked(group.negate)
        for item in group.items:
            if isinstance(item, Condition):
                self.add_row(item)
        if any(isinstance(item, Group) for item in group.items):
            self.note.setText(tr("Nested groups of this rule are not shown here; edit them in the script."))

    def group(self) -> Group:
        """The rows as a group. Raises ``ValueError`` naming the row."""
        items: List[Condition] = []
        for r in range(self.table.rowCount()):
            name = self.table.cellWidget(r, 0).currentText().strip()
            op = self.table.cellWidget(r, 1).currentData()
            text = self.table.cellWidget(r, 2).currentText()
            tol_text = self.table.cellWidget(r, 3).text().strip()
            if not name:
                raise ValueError(tr("row %d: choose a signal") % (r + 1))
            try:
                value, previous = parse_value(op, text, self.enums(name))
                tol = float(tol_text) if tol_text else 0.0
            except ValueError as exc:
                raise ValueError(tr("row %d (%s): %s") % (r + 1, name, exc)) from exc
            items.append(Condition(name, op, value, tol, previous=previous))
        return Group(self.mode.currentData(), items, self.negate.isChecked())

    def is_empty(self) -> bool:
        return self.table.rowCount() == 0


class RuleFormDialog(QtWidgets.QDialog):
    """Create or edit a rule. ``hint(name)`` returns "placeholder|tooltip" for the tolerance, ``period(name)``
    the sampling period of a signal in seconds (or None)."""

    def __init__(self, rule: Optional[Rule], names: Sequence[str], enums: Callable[[str], Dict[float, str]],
                 hint: Callable[[str], str], period: Callable[[str], Optional[float]],
                 all_names: Sequence[str] = (), taken: Sequence[str] = (),
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Rule"))
        self.resize(980, 860)
        self.period = period
        self.taken = {t for t in taken if not rule or t != rule.name}
        completer = None
        if all_names:
            completer = QtWidgets.QCompleter(sorted(set(all_names) | set(names)), self)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        outer = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name = QtWidgets.QLineEdit(rule.name if rule else tr("New rule"))
        self.note = QtWidgets.QLineEdit(rule.note if rule else "")
        form.addRow(tr("Name"), self.name)
        form.addRow(tr("Note"), self.note)
        outer.addLayout(form)

        box = QtWidgets.QGroupBox(tr("Trigger"))
        bl = QtWidgets.QVBoxLayout(box)
        self.trigger = ConditionsEditor(names, enums, hint, completer)
        bl.addWidget(self.trigger)
        row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem(tr("when the condition becomes true (one event per rising edge)"), "becomes_true")
        self.mode.addItem(tr("while the condition is true (one event per interval)"), "while_true")
        self.hold = QtWidgets.QDoubleSpinBox()
        self.hold.setRange(0.0, 3600.0)
        self.hold.setDecimals(3)
        self.hold.setSuffix(" s")
        self.initial = QtWidgets.QCheckBox(tr("count it if it holds already when its signals start"))
        row.addWidget(self.mode, 1)
        row.addWidget(QtWidgets.QLabel(tr("at least for")))
        row.addWidget(self.hold)
        bl.addLayout(row)
        bl.addWidget(self.initial)
        outer.addWidget(box)

        self.expect_box = QtWidgets.QGroupBox(tr("Expectation (check the response)"))
        self.expect_box.setCheckable(True)
        el = QtWidgets.QVBoxLayout(self.expect_box)
        self.expect = ConditionsEditor(names, enums, hint, completer)
        el.addWidget(self.expect)
        row = QtWidgets.QHBoxLayout()
        self.within_min = QtWidgets.QDoubleSpinBox()
        self.within_max = QtWidgets.QDoubleSpinBox()
        self.expect_hold = QtWidgets.QDoubleSpinBox()
        for spin in (self.within_min, self.within_max, self.expect_hold):
            spin.setRange(0.0, 3600.0)
            spin.setDecimals(3)
            spin.setSuffix(" s")
        row.addWidget(QtWidgets.QLabel(tr("met within")))
        row.addWidget(self.within_min)
        row.addWidget(QtWidgets.QLabel(tr("to")))
        row.addWidget(self.within_max)
        row.addWidget(QtWidgets.QLabel(tr("and then kept at least")))
        row.addWidget(self.expect_hold)
        row.addStretch(1)
        el.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.already = QtWidgets.QComboBox()
        for key, label in (("hold", "already right at the trigger: counts, delay 0 (the value is simply kept)"),
                           ("new_sample", "a new sample (frame) after the trigger must show it"),
                           ("change", "it must change into the expected state after the trigger")):
            self.already.addItem(tr(label), key)
        row.addWidget(QtWidgets.QLabel(tr("If the expectation already holds")))
        row.addWidget(self.already, 1)
        el.addLayout(row)
        self.period_note = QtWidgets.QLabel()
        self.period_note.setStyleSheet("color: gray")
        el.addWidget(self.period_note)
        outer.addWidget(self.expect_box)

        self.scope_box = QtWidgets.QGroupBox(tr("Scope: only check while (optional)"))
        self.scope_box.setCheckable(True)
        sl = QtWidgets.QVBoxLayout(self.scope_box)
        self.scope = ConditionsEditor(names, enums, hint, completer)
        sl.addWidget(self.scope)
        outer.addWidget(self.scope_box)

        row = QtWidgets.QHBoxLayout()
        self.record = QtWidgets.QLineEdit(", ".join(rule.record) if rule else "")
        self.record.setPlaceholderText(tr("signals whose values at the trigger go into the event table, separated by commas"))
        pick = QtWidgets.QPushButton(tr("Add signal..."))
        row.addWidget(QtWidgets.QLabel(tr("Record")))
        row.addWidget(self.record, 1)
        row.addWidget(pick)
        outer.addLayout(row)
        self.error = QtWidgets.QLabel()
        self.error.setStyleSheet("color: #c00000")
        self.error.setWordWrap(True)
        outer.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._names = list(names)
        pick.clicked.connect(self._pick)
        self.expect.changed.connect(self._update_period)

        # fill
        if rule is None:
            rule = Rule(tr("New rule"))
            self.trigger.add_row()
            self.expect.add_row()
            self.expect_box.setChecked(True)
            self.scope_box.setChecked(False)
        else:
            self.trigger.set_group(rule.trigger)
            self.expect.set_group(rule.expect)
            self.scope.set_group(rule.scope)
            self.expect_box.setChecked(rule.expect is not None and bool(rule.expect.items))
            self.scope_box.setChecked(rule.scope is not None and bool(rule.scope.items))
            if self.expect.is_empty():
                self.expect.add_row()
            if self.scope.is_empty():
                self.scope.add_row()
        self.mode.setCurrentIndex(max(0, self.mode.findData(rule.trigger_mode)))
        self.hold.setValue(rule.trigger_hold)
        self.initial.setChecked(rule.count_initial)
        self.within_min.setValue(rule.within_min)
        self.within_max.setValue(rule.within_max)
        self.expect_hold.setValue(rule.expect_hold)
        self.already.setCurrentIndex(max(0, self.already.findData(rule.already)))
        self._rule: Optional[Rule] = None
        self._update_period()

    def _pick(self) -> None:
        name, ok = QtWidgets.QInputDialog.getItem(self, tr("Record"), tr("Signal"), self._names, 0, True)
        if ok and name:
            parts = [p.strip() for p in self.record.text().split(",") if p.strip()]
            if name not in parts:
                parts.append(name)
            self.record.setText(", ".join(parts))

    def _update_period(self) -> None:
        try:
            names = self.expect.group().signals()
        except ValueError:
            names = []
        period = self.period(names[0]) if names else None
        self.period_note.setText(
            tr("The response signal %s is sampled about every %.0f ms: delays are measured to this resolution.")
            % (names[0], period * 1000) if period else "")

    def rule(self) -> Optional[Rule]:
        return self._rule

    def accept(self) -> None:
        name = self.name.text().strip()
        try:
            if not name:
                raise ValueError(tr("the rule needs a name"))
            if name in self.taken:
                raise ValueError(tr("a rule with this name exists already"))
            trigger = self.trigger.group()
            if not trigger.items:
                raise ValueError(tr("the trigger needs at least one condition"))
            expect = self.expect.group() if self.expect_box.isChecked() else None
            if expect is not None and not expect.items:
                expect = None
            scope = self.scope.group() if self.scope_box.isChecked() else None
            if scope is not None and not scope.items:
                scope = None
            if self.within_max.value() < self.within_min.value():
                raise ValueError(tr("the time window ends before it starts"))
            record = [p.strip() for p in self.record.text().split(",") if p.strip()]
            self._rule = Rule(name=name, trigger=trigger, trigger_mode=self.mode.currentData(),
                              trigger_hold=self.hold.value(), expect=expect, within_min=self.within_min.value(),
                              within_max=self.within_max.value(), expect_hold=self.expect_hold.value(),
                              already=self.already.currentData(), scope=scope, count_initial=self.initial.isChecked(),
                              record=record, note=self.note.text().strip(), origin="form")
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()
