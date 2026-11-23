"""Evaluate rules on recorded signals.

All signals of a rule are step functions. A rule is evaluated on the union
of the time stamps of the signals it uses (each holds its last sample).
A measured delay is only as fine as the sampling of the signals involved.

A signal has no value before its first sample and - if it is sent cyclically -
from ``LOST_PERIODS`` sampling periods after its last sample on (the sender
stopped; holding the last value would invent data). Conditions use three-valued
logic: true, false or unknown (no value). "Becomes true" needs a known false
before it; a condition that is true when its signals start is the initial
state and counts only with ``count_initial``. An expectation that could not be
checked because its signals had no value is INCONCLUSIVE, not FAIL.

Verdicts: PASS, FAIL, TOO_EARLY (the response came before ``within_min``),
INCONCLUSIVE (the recording ended before the time limit or during the hold
time, or the signals had no value) and FOUND (a rule without expectation that
only finds events).

Signals a rule only records go into the event table and change nothing else.
With a time range only triggers inside it are events; their expectation may
use the data after it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.model import Dataset, EnumTable
from ..core.sampling import LOST_PERIODS, is_cyclic, valid_until
from ..display import DisplaySignal
from ..i18n import tr
from .model import OPERATORS, Condition, Group, Operand, Rule

__all__ = ["Step", "SignalSource", "Event", "RuleResult", "evaluate", "evaluate_all", "VERDICTS", "describe_delay"]

VERDICTS = ("PASS", "FAIL", "TOO_EARLY", "INCONCLUSIVE", "FOUND")
_EPS = 1e-9


@dataclass
class Step:
    """A signal as step function: time stamps, values, optional text table."""

    name: str
    t: np.ndarray
    y: np.ndarray
    enum: Optional[EnumTable] = None
    unit: str = ""
    _valid_until: Optional[float] = field(default=None, init=False, repr=False, compare=False)

    def text(self, value: float) -> Optional[str]:
        if value is None or value != value or not self.enum:
            return None
        return self.enum.text(value)

    @property
    def period(self) -> Optional[float]:
        if len(self.t) < 3:
            return None
        return float(np.median(np.diff(self.t)))

    @property
    def cyclic(self) -> bool:
        """Sampled at a steady rate (see :mod:`tracepaddock.core.sampling`)."""
        return is_cyclic(self.t)

    @property
    def valid_until(self) -> float:
        """Time from which the last sample is no longer valid (the signal stopped being sent)."""
        if self._valid_until is None:
            self._valid_until = valid_until(self.t) if len(self.t) else float("inf")
        return self._valid_until

    def value_at(self, time: float) -> float:
        """The held value at ``time``; NaN before the first sample and after ``valid_until``."""
        if not len(self.t) or time >= self.valid_until - _EPS:
            return float("nan")
        i = int(np.searchsorted(self.t, time, side="right")) - 1
        return float(self.y[i]) if i >= 0 else float("nan")


class SignalSource:
    """Finds the signals a rule names: first among the signals shown (display names, keys,
    recorded names - so computed signals and value mappings are available), then in the recording."""

    def __init__(self, signals: Sequence[DisplaySignal] = (), dataset: Optional[Dataset] = None) -> None:
        self.dataset = dataset
        self._shown: Dict[str, DisplaySignal] = {}
        for sig in signals:
            if not getattr(sig, "has_data", False):
                continue
            entry = sig.entry
            names = [sig.name, entry.key, entry.token]
            if not entry.ops and entry.bit_pos is None:
                names.append(entry.name)
                if sig.info is not None:
                    names += [sig.info.name, sig.info.source_name]
            for name in names:
                if name:
                    self._shown.setdefault(name.lower(), sig)
        self._cache: Dict[str, Optional[Step]] = {}

    def names(self) -> List[str]:
        return sorted({getattr(s, "name", "") for s in self._shown.values()})

    def get(self, name: str) -> Optional[Step]:
        key = name.lower()
        if key in self._cache:
            return self._cache[key]
        step = None
        sig = self._shown.get(key)
        if sig is not None:
            step = Step(name, np.asarray(sig.t, dtype=np.float64), np.asarray(sig.y, dtype=np.float64),
                        sig.enum, sig.unit)
        elif self.dataset is not None:
            info = self.dataset.find(name)
            if info is not None:
                try:
                    data = self.dataset.read(info)
                    step = Step(name, np.asarray(data.t, dtype=np.float64), np.asarray(data.y, dtype=np.float64),
                                data.enum or info.enum, info.unit)
                except Exception:
                    step = None
        self._cache[key] = step
        return step


@dataclass
class Event:
    rule: str
    #: trigger time (start of the interval in "while_true" mode)
    t0: float
    verdict: str = "FOUND"
    #: end of the trigger interval ("while_true" mode)
    t_end: Optional[float] = None
    #: when the expectation was met
    t1: Optional[float] = None
    delay: Optional[float] = None
    #: the expectation held already at the trigger
    already: bool = False
    #: values at the trigger: signal -> value / text
    values: Dict[str, float] = field(default_factory=dict)
    texts: Dict[str, str] = field(default_factory=dict)
    #: values of the expectation's signals when it was decided (t1, or the end of the time window)
    actual: Dict[str, float] = field(default_factory=dict)
    actual_texts: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> dict:
        return {"rule": self.rule, "t0": self.t0, "verdict": self.verdict, "t_end": self.t_end, "t1": self.t1,
                "delay": self.delay, "already": self.already, "values": self.values, "texts": self.texts,
                "actual": self.actual, "actual_texts": self.actual_texts, "note": self.note}

    @classmethod
    def from_json(cls, data: dict) -> "Event":
        def num(v):
            return None if v is None else float(v)

        return cls(rule=str(data.get("rule", "")), t0=float(data.get("t0", 0.0)), verdict=str(data.get("verdict", "FOUND")),
                   t_end=num(data.get("t_end")), t1=num(data.get("t1")), delay=num(data.get("delay")),
                   already=bool(data.get("already", False)),
                   values={str(k): float(v) for k, v in (data.get("values") or {}).items() if v is not None},
                   texts={str(k): str(v) for k, v in (data.get("texts") or {}).items()},
                   actual={str(k): float(v) for k, v in (data.get("actual") or {}).items() if v is not None},
                   actual_texts={str(k): str(v) for k, v in (data.get("actual_texts") or {}).items()},
                   note=str(data.get("note", "") or ""))


@dataclass
class RuleResult:
    rule: Rule
    events: List[Event] = field(default_factory=list)
    error: Optional[str] = None
    #: sampling period of the expectation's first signal (resolution of the measured delays)
    response_period: Optional[float] = None
    time_range: Tuple[float, float] = (0.0, 0.0)

    def count(self, verdict: str) -> int:
        return sum(1 for e in self.events if e.verdict == verdict)

    def delays(self) -> np.ndarray:
        return np.array([e.delay for e in self.events if e.delay is not None and e.verdict == "PASS"], dtype=np.float64)

    def summary(self) -> Dict[str, object]:
        d = self.delays()
        out: Dict[str, object] = {v: self.count(v) for v in VERDICTS}
        out["events"] = len(self.events)
        if len(d):
            out.update(delay_min=float(d.min()), delay_max=float(d.max()), delay_mean=float(d.mean()),
                       delay_median=float(np.median(d)))
        return out


def describe_delay(seconds: Optional[float]) -> str:
    return "" if seconds is None else "%.0f ms" % (seconds * 1000.0)


# ---------------------------------------------------------------------- evaluation on the grid
class _Grid:
    def __init__(self, steps: Dict[str, Step], t0: float, t1: float) -> None:
        parts = [s.t[(s.t >= t0) & (s.t <= t1)] for s in steps.values()]
        # the moments from which a stopped cyclic signal has no value belong to the grid as well
        lost = [s.valid_until for s in steps.values() if t0 <= s.valid_until <= t1]
        parts.append(np.array([t0] + lost, dtype=np.float64))
        self.t = np.unique(np.concatenate(parts))
        self.start, self.end = t0, t1
        self._values: Dict[str, np.ndarray] = {}
        self.steps = steps

    def values(self, name: str) -> np.ndarray:
        key = name.lower()
        if key not in self._values:
            step = self.steps[key]
            if not len(step.t):
                out = np.full(len(self.t), np.nan)
            else:
                idx = np.searchsorted(step.t, self.t, side="right") - 1
                out = np.asarray(step.y, dtype=np.float64)[np.clip(idx, 0, len(step.t) - 1)].copy()
                out[idx < 0] = np.nan                               # no value before the first sample
                out[self.t >= step.valid_until - _EPS] = np.nan     # a cyclic signal that stopped
            self._values[key] = out
        return self._values[key]

    def missing(self, names: Sequence[str], mask: np.ndarray | slice) -> List[str]:
        """The signals among ``names`` without a value somewhere in ``mask``."""
        out: List[str] = []
        for name in names:
            if name not in out and np.isnan(self.values(name)[mask]).any():
                out.append(name)
        return out


def _close(a: np.ndarray, b, tol: float) -> np.ndarray:
    scale = np.maximum(np.abs(a), np.abs(b)) if np.ndim(b) else max(1.0, abs(float(b)))
    return np.abs(a - b) <= tol + 1e-9 * np.maximum(1.0, scale)


def _condition(cond: Condition, grid: _Grid) -> Tuple[str, np.ndarray, np.ndarray]:
    """``(kind, true, known)``: a "state" or an "instant" (a change), where the condition is true and
    where it is decided at all (its signals have a value - for a change also at the previous point)."""
    op, value = cond.op, cond.value
    # ("changes from 1" may go to any value; "between" says that it needs two)
    if value is None and OPERATORS.get(op) and op not in ("changes from", "between"):
        raise ValueError(tr("'%s %s' needs a value") % (cond.signal, op))
    raw = grid.values(cond.signal)
    left = cond.scale * raw + cond.offset
    finite = np.isfinite(left)
    tol = float(cond.tolerance or 0.0)
    if isinstance(value, Operand):
        right = value.scale * grid.values(value.signal) + value.offset
        finite = finite & np.isfinite(right)
    else:
        right = value
    prev_left = np.concatenate(([np.nan], left[:-1]))
    both = finite & np.isfinite(prev_left)                 # a change can be decided here
    changed = both & ~_close(left, prev_left, 0.0)
    with np.errstate(invalid="ignore"):
        if op == "==":
            out = _close(left, right, tol)
        elif op == "!=":
            out = ~_close(left, right, tol)
        elif op in ("<", "<=", ">", ">="):
            if right is None or isinstance(right, list):
                raise ValueError(tr("'%s %s' needs a value") % (cond.signal, op))
            if op == "<":
                out = left < right
            elif op == "<=":
                out = left <= right + tol
            elif op == ">":
                out = left > right
            else:
                out = left >= right - tol
        elif op in ("in", "not in", "changes to"):
            values = right if isinstance(right, list) else [right]
            hit = np.zeros(len(left), dtype=bool)
            for v in values:
                hit |= _close(left, v, tol)
            out = hit if op != "not in" else ~hit
            if op == "changes to":
                # into the band of a value: a move inside the band is no change to it
                entered = np.zeros(len(left), dtype=bool)
                for v in values:
                    entered |= _close(left, v, tol) & ~_close(prev_left, v, tol)
                return "instant", both & entered, both
        elif op == "between":
            if not isinstance(right, list) or len(right) != 2:
                raise ValueError(tr("'%s between' needs two values") % cond.signal)
            lo, hi = sorted(right)
            out = (left >= lo - tol) & (left <= hi + tol)
        elif op == "changes":
            return "instant", changed, both
        elif op == "changes from":
            if cond.previous is None:
                raise ValueError(tr("'%s changes from' needs the value it changes from") % cond.signal)
            to_hit = _close(left, right, tol) if right is not None else np.ones(len(left), dtype=bool)
            # out of the band of the old value
            away = _close(prev_left, cond.previous, tol) & ~_close(left, cond.previous, tol)
            return "instant", both & away & to_hit, both
        elif op == "rises":
            return "instant", both & (left > prev_left), both
        elif op == "falls":
            return "instant", both & (left < prev_left), both
        else:
            raise ValueError(tr("unknown operator '%s'") % op)
    return "state", np.asarray(out, dtype=bool) & finite, finite


def _rising(true: np.ndarray, known: np.ndarray, initial: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Where a state becomes true after it was known to be false, and where that is decided. With
    ``initial`` also where it is true at the start of the data or after a time without value."""
    edges = np.zeros(len(true), dtype=bool)
    edges[1:] = true[1:] & known[:-1] & ~true[:-1]
    if initial and len(true):
        edges[0] = true[0]
        edges[1:] |= true[1:] & ~known[:-1]
    decided = np.zeros(len(true), dtype=bool)
    decided[1:] = known[1:] & known[:-1]
    return edges, decided


def _group(group: Group, grid: _Grid, initial: bool = False) -> Tuple[str, np.ndarray, np.ndarray]:
    """AND / OR / NOT in three-valued logic: ``(kind, true, known)``. AND is false as soon as one
    part is known false, OR is true as soon as one part is true; otherwise a part without value
    leaves the result unknown (and NOT of unknown stays unknown). With ``initial`` a state ORed with
    a change also counts where it is the initial state (true when its signals start)."""
    parts = [(_group(item, grid, initial) if isinstance(item, Group) else _condition(item, grid))
             for item in group.items]
    n = len(grid.t)
    if not parts:
        return "state", np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    instants = [(a, k) for kind, a, k in parts if kind == "instant"]
    states = [(a, k) for kind, a, k in parts if kind == "state"]
    if group.mode == "and":
        items, kind = instants + states, ("instant" if instants else "state")
        true = np.logical_and.reduce([a for a, _ in items])
        false = np.logical_or.reduce([k & ~a for a, k in items])
    else:
        if instants:
            items, kind = instants + [_rising(a, k, initial) for a, k in states], "instant"
        else:
            items, kind = states, "state"
        true = np.logical_or.reduce([a for a, _ in items])
        false = np.logical_and.reduce([k & ~a for a, k in items])
    if group.negate:
        if kind == "instant":
            raise ValueError(tr("NOT cannot be applied to a change (an instant)"))
        true, false = false, true
    return kind, np.asarray(true, dtype=bool), np.asarray(true | false, dtype=bool)


def _runs(state: np.ndarray, times: np.ndarray, end: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intervals where ``state`` is true: start index, stop index (the first index after it, ``len``
    for an interval that is still open), start time and end time (``end`` for an open interval)."""
    if not len(state):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0), np.empty(0)
    s = state.astype(np.int8)
    d = np.diff(np.concatenate(([0], s, [0])))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)                        # first index after the run
    end_times = np.where(stops < len(times), times[np.clip(stops, 0, len(times) - 1)], end)
    return starts, stops, times[starts], end_times


def _initial(starts: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Runs without a known false before them - at the start of the data or after a time without
    value: the initial state, not a change."""
    if not len(starts):
        return np.zeros(0, dtype=bool)
    return (starts == 0) | ~known[np.clip(starts - 1, 0, None)]


# ---------------------------------------------------------------------- rule
def _texts(step: Step, value: float, out_v: Dict[str, float], out_t: Dict[str, str]) -> None:
    if value is None or value != value:
        return
    out_v[step.name] = float(value)
    text = step.text(value)
    if text is not None:
        out_t[step.name] = text


def _add_note(event: "Event", text: str) -> None:
    event.note = (event.note + "; " + text) if event.note else text


def _condition_signals(rule: Rule) -> List[str]:
    """The signals of the trigger, the expectation and the scope (not the ones only recorded)."""
    out: List[str] = []
    for part in (rule.trigger, rule.expect, rule.scope):
        if part is not None:
            for name in part.signals():
                if name not in out:
                    out.append(name)
    return out


def evaluate(rule: Rule, source: SignalSource, time_range: Optional[Tuple[float, float]] = None) -> RuleResult:
    """Evaluate one rule. Problems (missing signals, wrong operands) go into ``RuleResult.error``.

    With ``time_range`` only triggers inside it are events; the expectation of a trigger near its end
    may use the data up to ``within_max`` (and ``expect_hold``) after it. A visible interval is
    clipped at the right edge, but a state response must still occur before the trigger is known
    to have ended. The recorded signals only go into the event table (one that is missing is left out)."""
    steps: Dict[str, Step] = {}
    missing = []
    for name in _condition_signals(rule):
        step = source.get(name)
        if step is None or not len(step.t):
            missing.append(name)
        else:
            steps[name.lower()] = step
    result = RuleResult(rule)
    if missing:
        result.error = tr("signal not found or without data: %s") % ", ".join(missing)
        return result
    if not steps:
        result.error = tr("the rule uses no signal")
        return result
    lo = min(float(s.t[0]) for s in steps.values())
    hi = max(float(s.t[-1]) for s in steps.values())
    end = hi                                                # the data for the expectations ends here
    if time_range is not None:
        lo, hi = max(lo, float(time_range[0])), min(hi, float(time_range[1]))
        margin = max(rule.within_max, 0.0) + max(rule.expect_hold, 0.0)
        if margin < float("inf"):
            end = min(end, hi + margin)
    result.time_range = (lo, hi)
    if not hi >= lo:
        result.error = tr("no data in the time range")
        return result
    grid_start = lo
    try:
        if time_range is not None:
            # The preceding grid point detects an edge at lo without counting a state already true at lo.
            past = []
            for step in steps.values():
                previous = int(np.searchsorted(step.t, lo, side="left")) - 1
                if previous >= 0:
                    past.append(float(step.t[previous]))
                if step.valid_until < lo:
                    past.append(step.valid_until)
            grid_start = max(past, default=lo)
        grid = _Grid(steps, grid_start, end)
        t = grid.t
        n = int(np.searchsorted(t, hi + _EPS, side="right"))     # triggers: the grid points up to hi
        kind, trig, trig_known = _group(rule.trigger, grid)
        trig_initial = (_group(rule.trigger, grid, initial=True)[1]
                        if kind == "instant" and rule.count_initial else trig)
        scope = None
        if rule.scope is not None and rule.scope.items:
            skind, scope, _known = _group(rule.scope, grid)
            if skind != "state":
                raise ValueError(tr("the scope must be a state (no change operator)"))
        expectation = (_group(rule.expect, grid) if rule.expect is not None and rule.expect.items else None)
    except (ValueError, KeyError, TypeError) as exc:
        result.error = str(exc)
        return result

    # ---- trigger instants / intervals (an interval still open at hi ends there)
    if kind == "instant":
        starts = np.flatnonzero(trig[:n])
        initial = np.zeros(len(starts), dtype=bool)
        if rule.count_initial:
            starts = np.flatnonzero(trig_initial[:n])
            initial = ~trig[starts]
        stops, start_t, end_t = starts + 1, t[starts], t[starts]
    else:
        starts, stops, start_t, end_t = _runs(trig[:n], t[:n], hi)
        initial = _initial(starts, trig_known)
        if not rule.count_initial:
            keep = ~initial
            starts, stops, start_t, end_t, initial = (a[keep] for a in (starts, stops, start_t, end_t, initial))
    if grid_start < lo:
        keep = start_t >= lo
        starts, stops, start_t, end_t, initial = (a[keep] for a in (starts, stops, start_t, end_t, initial))
    if rule.trigger_hold > 0 and kind == "state":
        keep = (end_t - start_t) >= rule.trigger_hold - _EPS
        starts, stops, start_t, end_t, initial = (a[keep] for a in (starts, stops, start_t, end_t, initial))
    if scope is not None:
        keep = scope[starts]
        starts, stops, start_t, end_t, initial = (a[keep] for a in (starts, stops, start_t, end_t, initial))

    # the event table: the trigger's signals and the recorded ones (looked up apart from the grid)
    shown: Dict[str, Step] = {n.lower(): steps[n.lower()] for n in rule.trigger.signals()}
    for name in rule.record:
        key = name.lower()
        if key not in shown:
            step = steps[key] if key in steps else source.get(name)
            if step is not None and len(step.t):
                shown[key] = step
    expect_names = list(rule.expect.signals()) if rule.expect is not None else []
    expect_steps = [steps[n.lower()] for n in expect_names]
    if expect_steps:
        result.response_period = expect_steps[0].period
    response = (_Expectation(rule, *expectation, grid, expect_names, expect_steps[0] if expect_steps else None)
                if expectation is not None else None)

    for k, stop, t_start, t_stop, first in zip(starts, stops, start_t, end_t, initial):
        k, t0 = int(k), float(t_start)
        event = Event(rule=rule.name, t0=t0, t_end=float(t_stop) if rule.trigger_mode == "while_true" else None)
        if first:
            _add_note(event, tr("already true when the data started"))
        for step in shown.values():
            _texts(step, step.value_at(t0), event.values, event.texts)
        if response is None:
            event.verdict = "FOUND"
            result.events.append(event)
            continue
        expect_kind, expect, expect_known = response.kind, response.true, response.known
        limit = t0 + rule.within_max
        response_end = len(t)
        if rule.trigger_mode == "while_true" and kind == "state" and expect_kind == "state":
            trigger_end = int(stop)
            if trigger_end == n and n < len(t):
                ended = np.flatnonzero(~trig[n:])
                trigger_end = n + int(ended[0]) if len(ended) else len(t)
            if trigger_end < len(t) and trig_known[trigger_end]:
                response_end = trigger_end      # the first false sample is outside the interval
                limit = min(limit, float(t[trigger_end]))
        t1, undecided = response.met(k, t0, limit, response_end)
        if t1 is not None and t1 <= limit + _EPS:
            event.t1, event.delay = t1, t1 - t0
            event.already = t1 <= t0 + _EPS and expect_kind == "state"
            if event.delay < rule.within_min - _EPS:
                event.verdict = "TOO_EARLY"
            elif rule.trigger_mode == "while_true" and expect_kind == "state":
                # from t1 to the end of the interval (an open one includes its last point)
                inside = slice(int(np.searchsorted(t, t1 - _EPS, side="left")), int(stop))
                kept = _kept(expect, expect_known, inside)
                if kept is None:
                    event.verdict = "INCONCLUSIVE"
                    _add_note(event, tr("no value for %s during the interval")
                              % (", ".join(grid.missing(expect_names, inside)) or "?"))
                elif not kept:
                    event.verdict = "FAIL"
                    _add_note(event, tr("expectation not kept until the end of the interval"))
                else:
                    event.verdict = "PASS"
            else:
                event.verdict = "PASS"
            decide = t1
        elif end < limit - _EPS:
            event.verdict, decide = "INCONCLUSIVE", end
            _add_note(event, tr("the recording ends before the time limit"))
        elif undecided:
            # it may have been met, but that cannot be seen (no value, the recording ends during the hold time)
            event.verdict, decide = "INCONCLUSIVE", limit
            _add_note(event, undecided)
        else:
            window_end = int(np.searchsorted(t, limit + _EPS, side="right"))
            window = slice(k, min(window_end, response_end))
            decide = float(t[response_end - 1]) if response_end < window_end else limit
            if not expect_known[window].all():
                # never met, but not known to be wrong all the time either: the signals had no value
                event.verdict = "INCONCLUSIVE"
                _add_note(event, tr("no value for %s within the time limit")
                          % (", ".join(grid.missing(expect_names, window)) or "?"))
            else:
                event.verdict = "FAIL"
        for step in expect_steps:
            _texts(step, step.value_at(decide), event.actual, event.actual_texts)
        result.events.append(event)
    return result


def _kept(expect: np.ndarray, known: np.ndarray, inside) -> Optional[bool]:
    """Whether the expectation stays true on ``inside`` (a mask or a slice): True, False (known false
    somewhere) or None (without value somewhere, never known false)."""
    true, decided = expect[inside], known[inside]
    if (decided & ~true).any():
        return False
    if not decided.all():
        return None
    return True


class _Expectation:
    """The expectation of a rule on the grid: when it is met after a trigger."""

    def __init__(self, rule: Rule, kind: str, true: np.ndarray, known: np.ndarray, grid: _Grid,
                 names: List[str], own: Optional[Step]) -> None:
        self.rule, self.kind, self.true, self.known = rule, kind, true, known
        self.grid, self.t, self.names = grid, grid.t, names
        #: the response signal (the expectation's first signal) - its samples count in "new_sample" mode
        self.own = own
        self.runs = _runs(true, grid.t, grid.end) if kind == "state" else None

    def met(self, k0: int, t0: float, limit: float, end_index: int) -> Tuple[Optional[float], str]:
        """``(time, "")`` when the expectation is met for the trigger at grid point ``k0`` at the latest
        at ``limit`` (in the way ``rule.already`` asks for, and then kept ``expect_hold`` long).
        Otherwise ``(None, note)``: the note says why it may have been met although that cannot be
        seen; it is empty when it was not met."""
        t, rule = self.t, self.rule
        k_end = min(end_index, int(np.searchsorted(t, limit + _EPS, side="right")))
        if self.runs is None:
            hits = np.flatnonzero(self.true[k0:k_end])
            return (float(t[k0 + hits[0]]), "") if len(hits) else (None, "")
        starts, stops, start_t, _end_t = self.runs
        note = ""
        if rule.already == "new_sample":
            # a new sample of the response signal at or after the trigger that shows the expected state
            own = self.own
            if own is None:
                return None, note
            own_t = own.t
            for ts in own_t[int(np.searchsorted(own_t, t0 - _EPS, side="left")):]:
                if ts > limit + _EPS:
                    break
                j = int(np.searchsorted(t, ts, side="left"))
                if j >= k_end:
                    break
                if self.true[j]:
                    held, why = self._held(int(np.searchsorted(starts, j, side="right")) - 1, float(t[j]))
                    if held:
                        return float(ts), ""
                    note = note or why
            if not note and len(own_t) and own.valid_until < float("inf") and own_t[-1] < limit - _EPS:
                # a cyclic response signal that stopped being sent: no new sample could show the state
                note = tr("no value for %s within the time limit") % own.name
            return None, note
        # runs of the expectation that are still true at or after the trigger and start before the limit
        for r in range(int(np.searchsorted(stops, k0, side="right")), len(starts)):
            s = int(starts[r])
            if s >= k_end:
                break
            if rule.already == "change":
                if s < k0:
                    continue                           # true since before the trigger: a change is required
                if s == 0 or not self.known[s - 1]:
                    # true after a time without value: whether it changed cannot be seen (a time without
                    # value after the trigger is noted by the time window)
                    if s == k0 and not note:
                        missing = self.grid.missing(self.names, slice(s - 1, s)) if s else self.names
                        note = tr("no value for %s before the trigger, so a change cannot be seen") % (
                            ", ".join(missing) or "?")
                    continue
            t1 = max(float(start_t[r]), t0)
            held, why = self._held(r, t1)
            if held:
                return t1, ""
            note = note or why
        return None, note

    def _held(self, r: int, t1: float) -> Tuple[bool, str]:
        """Whether run ``r`` of the expectation lasts ``expect_hold`` from ``t1`` on; when it does not,
        why that cannot be decided (empty: it is known false before the hold time is over)."""
        hold = self.rule.expect_hold
        assert self.runs is not None
        _starts, stops, _start_t, end_t = self.runs
        if hold <= 0 or float(end_t[r]) - t1 >= hold - _EPS:
            return True, ""
        stop = int(stops[r])
        rest = slice(stop, int(np.searchsorted(self.t, t1 + hold - _EPS, side="left")))   # until the hold time is over
        if (self.known[rest] & ~self.true[rest]).any():
            return False, ""
        if stop < len(self.t):                                 # it lost its value, and no known false follows
            missing = self.grid.missing(self.names, rest)
            return False, tr("no value for %s during the hold time") % (", ".join(missing) or "?")
        return False, tr("the recording ends during the hold time")


def evaluate_all(rules: Sequence[Rule], source: SignalSource,
                 time_range: Optional[Tuple[float, float]] = None) -> List[RuleResult]:
    return [evaluate(rule, source, time_range) for rule in rules if rule.enabled]
