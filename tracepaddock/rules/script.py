"""The Python interface for rule scripts and plug-ins.

A script builds rules with a small vocabulary and may add events of its own::

    request = S("TutorialRequest")
    rule("Request acknowledgement",
         trigger=becomes(request == 1),
         expect=S("TutorialAck") == 1,
         within=0.5, record=["TutorialAckStatus"])

    t, y = data("TutorialInputTravel")
    for k in np.flatnonzero(np.abs(np.diff(y)) > 5):
        event(t[k + 1], "change in input travel", verdict="FOUND")

Comparisons of ``S(...)`` with numbers, lists (``isin``) or other signals give
conditions; ``&``, ``|`` and ``~`` combine them. Signals may be scaled:
``S("v_ms") * 3.6``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .engine import Event
from .model import Condition, Group, Operand, Rule

__all__ = ["ScriptApi", "Sig", "ScriptError"]


class ScriptError(Exception):
    """A mistake in a rule script (shown to the user with the line)."""


class Sig:
    """A signal reference in a script: ``scale * signal + offset``."""

    def __init__(self, name: str, scale: float = 1.0, offset: float = 0.0) -> None:
        self.name, self.scale, self.offset = str(name), float(scale), float(offset)

    # arithmetic: only linear scaling, which conditions can carry
    def __mul__(self, k):
        self._number(k, "*")
        return Sig(self.name, self.scale * k, self.offset * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        self._number(k, "/")
        return Sig(self.name, self.scale / k, self.offset / k)

    def __add__(self, k):
        self._number(k, "+")
        return Sig(self.name, self.scale, self.offset + k)

    __radd__ = __add__

    def __sub__(self, k):
        self._number(k, "-")
        return Sig(self.name, self.scale, self.offset - k)

    def __neg__(self):
        return Sig(self.name, -self.scale, -self.offset)

    @staticmethod
    def _number(k, op: str) -> None:
        if not isinstance(k, (int, float)) or isinstance(k, bool):
            raise ScriptError(f"a signal can only be combined with a number by '{op}' "
                              "(compare two signals with ==, <, ... instead)")

    # comparisons -> conditions
    def _cond(self, op: str, other, tolerance: float = 0.0) -> Condition:
        if isinstance(other, Sig):
            if other.scale == 0:
                raise ScriptError("cannot compare with a signal scaled by 0")
            # (a*x + b) op (c*y + d)  ->  x op' (c*y + d - b) / a, where op' flips for a < 0
            value = Operand(other.name, other.scale / self.scale, (other.offset - self.offset) / self.scale)
            return Condition(self.name, _flip(op) if self.scale < 0 else op, value, tolerance / abs(self.scale))
        if isinstance(other, (list, tuple, set)):
            raise ScriptError("use .isin([...]) to compare with several values")
        if not isinstance(other, (int, float)) or isinstance(other, bool):
            raise ScriptError(f"cannot compare a signal with {type(other).__name__}")
        return Condition(self.name, op, float(other), tolerance, self.scale, self.offset)

    def __eq__(self, other):  # type: ignore[override]
        return self._cond("==", other)

    def __ne__(self, other):  # type: ignore[override]
        return self._cond("!=", other)

    def __lt__(self, other):
        return self._cond("<", other)

    def __le__(self, other):
        return self._cond("<=", other)

    def __gt__(self, other):
        return self._cond(">", other)

    def __ge__(self, other):
        return self._cond(">=", other)

    __hash__ = object.__hash__

    def equals(self, other, tol: float = 0.0) -> Condition:
        """Equal within an absolute tolerance."""
        return self._cond("==", other, tol)

    def isin(self, values: Sequence[float], tol: float = 0.0) -> Condition:
        return Condition(self.name, "in", [float(v) for v in values], tol, self.scale, self.offset)

    def between(self, lo: float, hi: float) -> Condition:
        return Condition(self.name, "between", [float(lo), float(hi)], 0.0, self.scale, self.offset)

    # changes of the scaled signal (values in the scaled unit)
    def changes(self) -> Condition:
        return Condition(self.name, "changes", None, 0.0, self.scale, self.offset)

    def changes_to(self, value, tol: float = 0.0) -> Condition:
        values = [float(v) for v in value] if isinstance(value, (list, tuple)) else float(value)
        return Condition(self.name, "changes to", values, tol, self.scale, self.offset)

    def changes_from(self, old: float, new: Optional[float] = None) -> Condition:
        return Condition(self.name, "changes from", None if new is None else float(new), 0.0, self.scale, self.offset,
                         previous=float(old))

    def rises(self) -> Condition:
        return Condition(self.name, "rises", None, 0.0, self.scale, self.offset)

    def falls(self) -> Condition:
        return Condition(self.name, "falls", None, 0.0, self.scale, self.offset)


def _flip(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<="}.get(op, op)


# Conditions and groups combine with & | ~ in scripts.
def _and(a, b):
    return Group("and", [_part(a), _part(b)])


def _or(a, b):
    return Group("or", [_part(a), _part(b)])


def _not(a):
    g = _part(a)
    g = g if isinstance(g, Group) else Group("and", [g])
    return Group(g.mode, g.items, not g.negate)


def _part(x):
    if isinstance(x, (Condition, Group)):
        return x
    raise ScriptError(f"expected a condition, got {type(x).__name__}")


for _cls in (Condition, Group):
    _cls.__and__ = _and          # type: ignore[attr-defined]
    _cls.__or__ = _or            # type: ignore[attr-defined]
    _cls.__invert__ = _not       # type: ignore[attr-defined]


class _Becomes:
    """Marks a trigger condition with its options (``becomes(cond, hold=...)`` / ``while_(cond)``)."""

    def __init__(self, cond, mode: str, hold: float) -> None:
        self.cond, self.mode, self.hold = _part(cond), mode, float(hold)


class ScriptApi:
    """What a script sees. ``data`` reads arrays through ``reader(name) -> (t, y) or None``."""

    def __init__(self, reader, script_name: str = "script", params: Optional[dict] = None) -> None:
        self._reader = reader
        self.name = script_name
        self.rules: List[Rule] = []
        self.events: List[Event] = []
        self.messages: List[str] = []
        self.params = dict(params or {})

    # vocabulary --------------------------------------------------------
    def S(self, name: str) -> Sig:
        return Sig(name)

    def becomes(self, cond, hold: float = 0.0) -> _Becomes:
        return _Becomes(cond, "becomes_true", hold)

    def while_(self, cond, hold: float = 0.0) -> _Becomes:
        return _Becomes(cond, "while_true", hold)

    def rule(self, name: str, trigger, expect=None, within: Union[float, Tuple[float, float]] = 0.5,
             already: str = "hold", scope=None, record: Sequence[str] = (), hold: float = 0.0,
             expect_hold: float = 0.0, count_initial: bool = False, note: str = "") -> Rule:
        mode, trig_hold = "becomes_true", float(hold)
        if isinstance(trigger, _Becomes):
            mode, trig_hold, trigger = trigger.mode, max(trig_hold, trigger.hold), trigger.cond
        lo, hi = (0.0, float(within)) if not isinstance(within, (tuple, list)) else (float(within[0]), float(within[1]))
        try:
            rule = Rule(name=str(name), trigger=_part(trigger), trigger_mode=mode, trigger_hold=trig_hold,
                        expect=None if expect is None else _part(expect), within_min=lo, within_max=hi,
                        already=already, scope=None if scope is None else _part(scope),
                        record=[str(r.name if isinstance(r, Sig) else r) for r in record],
                        expect_hold=float(expect_hold), count_initial=bool(count_initial), note=note,
                        origin="script:" + self.name)
        except ValueError as exc:
            raise ScriptError(str(exc)) from exc
        self.rules.append(rule)
        return rule

    def data(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        got = self._reader(name)
        if got is None:
            raise ScriptError(f"signal '{name}' not found")
        return got

    def event(self, time: float, text: str = "", verdict: str = "FOUND", rule: Optional[str] = None,
              **values: float) -> Event:
        verdict = str(verdict).upper()
        if verdict not in ("PASS", "FAIL", "TOO_EARLY", "INCONCLUSIVE", "FOUND"):
            raise ScriptError(f"unknown verdict '{verdict}' (PASS, FAIL, INCONCLUSIVE or FOUND)")
        ev = Event(rule=rule or self.name, t0=float(time), verdict=verdict, note=str(text),
                   values={k: float(v) for k, v in values.items()})
        self.events.append(ev)
        return ev

    def log(self, *parts) -> None:
        self.messages.append(" ".join(str(p) for p in parts))

    def namespace(self) -> Dict[str, object]:
        return {"S": self.S, "sig": self.S, "rule": self.rule, "becomes": self.becomes, "while_": self.while_,
                "data": self.data, "event": self.event, "log": self.log, "print": self.log, "np": np,
                "params": self.params, "Condition": Condition, "Group": Group}
