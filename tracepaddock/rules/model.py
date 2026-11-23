"""Data model of rules: conditions, groups, rules and rule sets (JSON).

A rule says: *whenever the trigger becomes true (inside the scope), the
expected condition must hold within [within_min, within_max] seconds*. A
rule without an expectation only finds events.

Conditions compare a signal with a number, a list of numbers or another
signal. Both sides may be scaled (``scale * signal + offset``), which covers
unit conversions such as m/s -> km/h.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..settings import keep_copy, read_json_object
from .plugins import plugin_ref

__all__ = ["Operand", "Condition", "Group", "Rule", "RuleSet", "OPERATORS", "INSTANT_OPERATORS",
           "TRIGGER_MODES", "ALREADY_MODES", "default_rules_path"]

#: operator -> needs a value
OPERATORS = {
    "==": True, "!=": True, "<": True, "<=": True, ">": True, ">=": True,
    "in": True, "not in": True, "between": True,
    "changes": False, "changes to": True, "changes from": True, "rises": False, "falls": False,
}
#: operators that mark instants (a change) instead of a state that lasts
INSTANT_OPERATORS = {"changes", "changes to", "changes from", "rises", "falls"}
#: "becomes_true": every rising edge of the trigger is an event; "while_true": every interval is an event
TRIGGER_MODES = ("becomes_true", "while_true")
#: how an expectation that already holds at the trigger counts:
#: "hold" = satisfied with delay 0; "new_sample" = a new sample of the response signal after the trigger
#: must show it; "change" = the response must change into the expected state after the trigger
ALREADY_MODES = ("hold", "new_sample", "change")


def default_rules_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "TracePaddock" / "rules.json"


@dataclass
class Operand:
    """``scale * signal + offset``: the right-hand side of a comparison with another signal."""

    signal: str
    scale: float = 1.0
    offset: float = 0.0

    def to_json(self) -> dict:
        return {"signal": self.signal, "scale": self.scale, "offset": self.offset}

    @classmethod
    def from_json(cls, data: dict) -> "Operand":
        return cls(str(data.get("signal", "")), float(data.get("scale", 1.0)), float(data.get("offset", 0.0)))

    def text(self) -> str:
        out = self.signal
        if self.scale != 1.0:
            out = "%g*%s" % (self.scale, out)
        if self.offset:
            out += " %s %g" % ("+" if self.offset > 0 else "-", abs(self.offset))
        return out


Value = Union[None, float, List[float], Operand]


@dataclass
class Condition:
    """One row: ``scale * signal + offset  <op>  value`` (with an absolute tolerance for equality)."""

    signal: str
    op: str = "=="
    value: Value = 0.0
    tolerance: float = 0.0
    scale: float = 1.0
    offset: float = 0.0
    #: for "changes from": the value it changes from (``value`` is the value it changes to)
    previous: Optional[float] = None

    @property
    def instant(self) -> bool:
        return self.op in INSTANT_OPERATORS

    def signals(self) -> List[str]:
        out = [self.signal]
        if isinstance(self.value, Operand):
            out.append(self.value.signal)
        return out

    def to_json(self) -> dict:
        value = self.value.to_json() if isinstance(self.value, Operand) else self.value
        data = {"signal": self.signal, "op": self.op, "value": value}
        if self.tolerance:
            data["tolerance"] = self.tolerance
        if self.scale != 1.0:
            data["scale"] = self.scale
        if self.offset:
            data["offset"] = self.offset
        if self.previous is not None:
            data["previous"] = self.previous
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Condition":
        value = data.get("value")
        if isinstance(value, dict):
            value = Operand.from_json(value)
        elif isinstance(value, list):
            value = [float(v) for v in value]
        elif value is not None:
            value = float(value)
        op = str(data.get("op", "=="))
        if op not in OPERATORS:
            raise ValueError(f"unknown operator '{op}'")
        prev = data.get("previous")
        return cls(signal=str(data.get("signal", "")), op=op, value=value,
                   tolerance=float(data.get("tolerance", 0.0) or 0.0), scale=float(data.get("scale", 1.0)),
                   offset=float(data.get("offset", 0.0) or 0.0), previous=None if prev is None else float(prev))

    def text(self, names: Optional[Dict[str, Dict[float, str]]] = None) -> str:
        """Readable form; ``names`` maps a signal to its value texts (for enumerations)."""
        left = self.signal
        if self.scale != 1.0:
            left = "%g*%s" % (self.scale, left)
        if self.offset:
            left += " %s %g" % ("+" if self.offset > 0 else "-", abs(self.offset))
        texts = (names or {}).get(self.signal, {})

        def one(v: float) -> str:
            t = texts.get(float(v))
            return "%g (%s)" % (v, t) if t else "%g" % v

        if self.op in ("changes", "rises", "falls"):
            return "%s %s" % (left, self.op)
        if isinstance(self.value, Operand):
            right = self.value.text()
        elif isinstance(self.value, list):
            right = ("%s .. %s" % (one(self.value[0]), one(self.value[1])) if self.op == "between" and len(self.value) == 2
                     else "{%s}" % ", ".join(one(v) for v in self.value))
        elif self.value is None:
            right = ""
        else:
            right = one(self.value)
        if self.op == "changes from":
            return "%s changes from %s to %s" % (left, one(self.previous) if self.previous is not None else "?", right)
        tol = " ±%g" % self.tolerance if self.tolerance else ""
        return "%s %s %s%s" % (left, self.op, right, tol)


@dataclass
class Group:
    """AND / OR of conditions and groups, optionally negated."""

    mode: str = "and"
    items: List[Union[Condition, "Group"]] = field(default_factory=list)
    negate: bool = False

    def signals(self) -> List[str]:
        out: List[str] = []
        for item in self.items:
            for name in item.signals():
                if name not in out:
                    out.append(name)
        return out

    def to_json(self) -> dict:
        return {"mode": self.mode, "negate": self.negate, "items": [i.to_json() for i in self.items]}

    @classmethod
    def from_json(cls, data: dict) -> "Group":
        items: List[Union[Condition, Group]] = []
        for item in data.get("items", []) or []:
            items.append(Group.from_json(item) if "items" in item else Condition.from_json(item))
        mode = str(data.get("mode", "and")).lower()
        return cls(mode="or" if mode == "or" else "and", items=items, negate=bool(data.get("negate", False)))

    def text(self, names: Optional[Dict[str, Dict[float, str]]] = None) -> str:
        joiner = " AND " if self.mode == "and" else " OR "
        parts = [("(%s)" % i.text(names)) if isinstance(i, Group) else i.text(names) for i in self.items]
        body = joiner.join(parts) if parts else "(empty)"
        return ("NOT (%s)" % body) if self.negate else body


def _group(value: Union[Condition, Group, None]) -> Optional[Group]:
    if value is None or isinstance(value, Group):
        return value
    return Group("and", [value])


@dataclass
class Rule:
    name: str
    trigger: Group = field(default_factory=Group)
    trigger_mode: str = "becomes_true"
    #: the trigger must stay true at least this long [s] (debounce); the event time is its start
    trigger_hold: float = 0.0
    #: an expectation; ``None`` = only find the events
    expect: Optional[Group] = None
    within_min: float = 0.0
    within_max: float = 0.5
    #: the expectation must then stay true at least this long [s]
    expect_hold: float = 0.0
    already: str = "hold"
    #: only triggers while this holds (e.g. function active)
    scope: Optional[Group] = None
    #: also count a trigger that is already true at the start of the recording
    count_initial: bool = False
    #: signals whose values at the trigger time go into the event table
    record: List[str] = field(default_factory=list)
    enabled: bool = True
    note: str = ""
    #: "form", "script" or "plugin:<name>" - who made the rule (scripts are not edited in the form)
    origin: str = "form"

    def __post_init__(self) -> None:
        self.trigger = _group(self.trigger) or Group()
        self.expect = _group(self.expect)
        self.scope = _group(self.scope)
        if self.trigger_mode not in TRIGGER_MODES:
            raise ValueError(f"unknown trigger mode '{self.trigger_mode}'")
        if self.already not in ALREADY_MODES:
            raise ValueError(f"unknown mode '{self.already}' (hold, new_sample or change)")

    def signals(self) -> List[str]:
        out: List[str] = []
        for part in (self.trigger, self.expect, self.scope):
            if part is not None:
                for name in part.signals():
                    if name not in out:
                        out.append(name)
        for name in self.record:
            if name not in out:
                out.append(name)
        return out

    def to_json(self) -> dict:
        return {"name": self.name, "enabled": self.enabled, "note": self.note, "origin": self.origin,
                "trigger": self.trigger.to_json(), "trigger_mode": self.trigger_mode,
                "trigger_hold": self.trigger_hold,
                "expect": self.expect.to_json() if self.expect is not None else None,
                "within_min": self.within_min, "within_max": self.within_max, "expect_hold": self.expect_hold,
                "already": self.already, "scope": self.scope.to_json() if self.scope is not None else None,
                "count_initial": self.count_initial, "record": list(self.record)}

    @classmethod
    def from_json(cls, data: dict) -> "Rule":
        def grp(value):
            return Group.from_json(value) if isinstance(value, dict) else None

        return cls(name=str(data.get("name", "rule")), trigger=grp(data.get("trigger")) or Group(),
                   trigger_mode=str(data.get("trigger_mode", "becomes_true")),
                   trigger_hold=float(data.get("trigger_hold", 0.0) or 0.0), expect=grp(data.get("expect")),
                   within_min=float(data.get("within_min", 0.0) or 0.0),
                   within_max=float(data.get("within_max", 0.5) if data.get("within_max") is not None else 0.5),
                   expect_hold=float(data.get("expect_hold", 0.0) or 0.0), already=str(data.get("already", "hold")),
                   scope=grp(data.get("scope")), count_initial=bool(data.get("count_initial", False)),
                   record=[str(r) for r in data.get("record", []) or []], enabled=bool(data.get("enabled", True)),
                   note=str(data.get("note", "") or ""), origin=str(data.get("origin", "form") or "form"))


class RuleSet:
    """The user's rules (form rules, script rules are kept as their script text)."""

    def __init__(self, path: "Optional[str | Path]" = None, *, autoload: bool = True) -> None:
        self.path = Path(path) if path is not None else default_rules_path()
        self.rules: List[Rule] = []
        #: name -> script text of scripts the user wrote in the script window
        self.scripts: Dict[str, str] = {}
        #: name -> {"path": plug-in reference (see plugins.plugin_ref), "params": {...}} of plug-ins in the list
        self.plugin_runs: Dict[str, dict] = {}
        #: names of scripts and plug-in runs that are switched off
        self.disabled: set = set()
        #: False when the file could not be read and must not be written over
        self._writable = True
        if autoload:
            self.load()

    def load(self) -> None:
        """Read the rule file (UTF-8, also with BOM). A file that cannot be read is kept, not overwritten;
        rules that cannot be read are kept in a copy of the file (the next save would drop them)."""
        data, self._writable = read_json_object(self.path)
        if data is None:
            return
        problems = self.take(data)
        if problems:
            self._writable = keep_copy(self.path, "; ".join(problems), move=False, skipped=len(problems)) is not None

    def take(self, data: dict) -> List[str]:
        """Read rules from a dict; returns problems (rules that could not be read)."""
        problems: List[str] = []
        self.rules = []
        for item in (data.get("rules", []) if isinstance(data, dict) else []) or []:
            try:
                self.rules.append(Rule.from_json(item))
            except (ValueError, TypeError, AttributeError) as exc:
                problems.append("%s: %s" % (item.get("name", "?") if isinstance(item, dict) else "?", exc))
        scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
        self.scripts = {str(k): str(v) for k, v in scripts.items()} if isinstance(scripts, dict) else {}
        plugins = data.get("plugins", {}) if isinstance(data, dict) else {}
        self.plugin_runs = {str(k): {"path": plugin_ref(str(v.get("path", ""))), "params": dict(v.get("params") or {})}
                            for k, v in plugins.items() if isinstance(v, dict)} if isinstance(plugins, dict) else {}
        disabled = data.get("disabled", []) if isinstance(data, dict) else []
        self.disabled = {str(d) for d in disabled} if isinstance(disabled, list) else set()
        return problems

    def to_json(self) -> dict:
        return {"format": "tracepaddock-rules", "version": 1, "rules": [r.to_json() for r in self.rules],
                "scripts": dict(self.scripts), "plugins": dict(self.plugin_runs), "disabled": sorted(self.disabled)}

    def all_names(self) -> List[str]:
        return [r.name for r in self.rules] + list(self.scripts) + list(self.plugin_runs)

    def save(self, path: "Optional[str | Path]" = None) -> None:
        if path is None and not getattr(self, "_writable", True):
            return                                             # the file could not be read: keep it as it is
        target = Path(path) if path is not None else self.path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, indent=1, ensure_ascii=False)
            os.replace(tmp, target)
        except OSError:
            if path is not None:
                raise

    def unique_name(self, wanted: str) -> str:
        names = set(self.all_names())
        name, n = wanted, 2
        while name in names:
            name, n = "%s (%d)" % (wanted, n), n + 1
        return name

    def form_rules(self) -> Sequence[Rule]:
        return [r for r in self.rules if r.origin == "form"]
