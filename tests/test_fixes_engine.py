"""Regression tests for the review findings of the rule engine (engine, script API, sampling)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from tracepaddock.core.sampling import is_cyclic, valid_until
from tracepaddock.rules.engine import SignalSource, Step, evaluate
from tracepaddock.rules.model import Condition, Group, Rule
from tracepaddock.rules.script import Sig


class Source(SignalSource):
    """A signal source from plain arrays."""

    def __init__(self, **signals):
        super().__init__()
        for name, (t, y) in signals.items():
            self._cache[name.lower()] = Step(name, np.asarray(t, float), np.asarray(y, float))


def raster(start: float, stop: float, step: float) -> np.ndarray:
    """Time stamps ``start .. stop`` (both included)."""
    return np.round(start + step * np.arange(int(round((stop - start) / step)) + 1), 6)


def from_(t: np.ndarray, when: float, until: float = None) -> np.ndarray:
    """1 from ``when`` (until ``until``), else 0."""
    y = t >= when - 1e-9
    if until is not None:
        y &= t < until - 1e-9
    return y.astype(float)


def check(**kw) -> Rule:
    args = dict(name="r", trigger=Condition("trig", "==", 1.0), expect=Condition("out", "==", 1.0), within_max=0.5)
    args.update(kw)
    return Rule(**args)


def events(result):
    return [(round(e.t0, 6), e.verdict) for e in result.events]


# ---------------------------------------------------------------------- expect_hold cut off by missing data
@pytest.mark.parametrize("already", ["hold", "change", "new_sample"])
def test_a_hold_time_cut_off_by_the_end_of_the_recording_is_inconclusive(already):
    t = raster(0.0, 10.0, 0.1)
    src = Source(trig=(t, from_(t, 9.0)), out=(t, from_(t, 9.1)))      # out kept to the end (0.9 s)
    event = evaluate(check(expect_hold=2.0, already=already), src).events[0]
    assert event.verdict == "INCONCLUSIVE" and "hold time" in event.note


def test_a_hold_time_cut_off_by_a_signal_that_stops_is_inconclusive():
    t = raster(0.0, 20.0, 0.1)
    tout = raster(0.0, 9.5, 0.1)                                        # sent until 9.5 s: no value from 10.0 s
    src = Source(trig=(t, from_(t, 9.0)), out=(tout, from_(tout, 9.1)))
    for already in ("hold", "new_sample"):
        event = evaluate(check(expect_hold=2.0, already=already), src).events[0]
        assert event.verdict == "INCONCLUSIVE" and "out" in event.note and "hold time" in event.note


def test_a_hold_time_that_ends_at_a_known_false_stays_fail():
    t = raster(0.0, 20.0, 0.1)
    src = Source(trig=(t, from_(t, 9.0)), out=(t, from_(t, 9.1, 9.5)))  # drops to a known 0 at 9.5 s
    event = evaluate(check(expect_hold=2.0), src).events[0]
    assert event.verdict == "FAIL"


def test_a_gap_during_the_hold_time_is_inconclusive_unless_a_known_false_follows():
    t = raster(0.0, 10.0, 0.1)
    trig = from_(t, 5.0)
    out = from_(t, 5.4)
    out[np.isclose(t, 5.6)] = np.nan                                    # no value at 5.6 s, then 1 again
    gap = evaluate(check(expect_hold=0.5), Source(trig=(t, trig), out=(t, out))).events[0]
    assert gap.verdict == "INCONCLUSIVE" and "out" in gap.note and "hold time" in gap.note
    out[t >= 5.7 - 1e-9] = 0.0                                          # known false at 5.7 s, before 5.9 s
    dropped = evaluate(check(expect_hold=0.5), Source(trig=(t, trig), out=(t, out))).events[0]
    assert dropped.verdict == "FAIL"


def test_a_later_run_may_meet_the_hold_time_after_one_was_cut_off_by_a_gap():
    t = raster(0.0, 10.0, 0.1)
    out = from_(t, 5.1)
    out[np.isclose(t, 5.3)] = np.nan                                    # no value at 5.3 s
    src = Source(trig=(t, from_(t, 5.0)), out=(t, out))
    event = evaluate(check(expect_hold=0.3), src).events[0]
    assert event.verdict == "PASS" and event.delay == pytest.approx(0.4)


# ---------------------------------------------------------------------- a trigger at the last time stamp
def test_a_trigger_at_the_last_time_stamp_finds_an_expectation_that_holds():
    t = raster(0.0, 5.0, 0.5)
    src = Source(trig=(t, from_(t, 5.0)), out=(t, np.ones(len(t))))
    for within in (0.0, 0.5):
        for trigger in (Condition("trig", "==", 1.0), Condition("trig", "changes")):
            event = evaluate(check(trigger=trigger, within_max=within), src).events[0]
            assert (event.t0, event.verdict, event.delay) == (5.0, "PASS", 0.0)
    single = Source(trig=([3.0], [1.0]), out=([3.0], [1.0]))
    event = evaluate(check(count_initial=True, within_max=0.0), single).events[0]
    assert (event.verdict, event.delay) == ("PASS", 0.0)
    both = Source(trig=(t, from_(t, 5.0)), out=(t, from_(t, 5.0)))     # change mode: both change at the end
    event = evaluate(check(already="change"), both).events[0]
    assert (event.verdict, event.delay) == ("PASS", 0.0)


# ---------------------------------------------------------------------- already="change" needs a known false
def test_change_mode_does_not_count_a_run_after_a_time_without_value_as_a_change():
    t = raster(0.0, 9.9, 0.1)
    trig = from_(t, 5.0)
    out = np.ones(len(t))
    out[np.isclose(t, 5.2)] = np.nan
    gap = evaluate(check(already="change"), Source(trig=(t, trig), out=(t, out))).events[0]
    assert gap.verdict == "INCONCLUSIVE" and "out" in gap.note
    late_t = raster(5.2, 9.9, 0.1)                                      # out recorded only from 5.2 s on
    late = evaluate(check(already="change"), Source(trig=(t, trig), out=(late_t, np.ones(len(late_t))))).events[0]
    assert late.verdict == "INCONCLUSIVE" and "out" in late.note
    at_t = raster(5.0, 9.9, 0.1)                                        # out recorded from the trigger on
    at = evaluate(check(already="change"), Source(trig=(t, trig), out=(at_t, np.ones(len(at_t))))).events[0]
    assert at.verdict == "INCONCLUSIVE" and "out" in at.note and "before the trigger" in at.note
    real = evaluate(check(already="change"), Source(trig=(t, trig), out=(t, from_(t, 5.2)))).events[0]
    assert real.verdict == "PASS" and real.delay == pytest.approx(0.2)


# ---------------------------------------------------------------------- 'changes to' / 'changes from' with a tolerance
def test_changes_to_and_from_with_a_tolerance_fire_when_the_band_is_entered_or_left():
    t = np.arange(0.0, 6.0, 1.0)

    def times(values, cond):
        return [e.t0 for e in evaluate(Rule("r", trigger=cond), Source(m=(t, values))).events]

    assert times([0, 1.0, 1.1, 0.95, 1.05, 1.0], Condition("m", "changes to", 1.0, tolerance=0.2)) == [1.0]
    assert times([1, 1.1, 1.05, 0.9, 3, 3], Condition("m", "changes from", None, tolerance=0.2, previous=1.0)) == [4.0]
    # from one listed value to another
    assert times([0, 1, 2, 0, 0, 0], Condition("m", "changes to", [1.0, 2.0])) == [1.0, 2.0]
    assert times([0, 1, 2, 2, 1, 0], Condition("m", "changes from", 1.0, previous=2.0)) == [4.0]


# ---------------------------------------------------------------------- new_sample: the response stopped being sent
def test_new_sample_mode_with_a_response_that_stopped_being_sent_is_inconclusive():
    tt = raster(0.0, 20.0, 0.05)
    out_t = raster(0.0, 10.0, 0.1)                                      # last sample 10.0 s (value 1)
    src = Source(trig=(tt, from_(tt, 10.15)), out=(out_t, np.ones(len(out_t))))
    event = evaluate(check(within_max=0.3, already="new_sample"), src).events[0]
    assert event.verdict == "INCONCLUSIVE" and "out" in event.note
    slow_t = np.arange(0.0, 11.0, 1.0)                                  # a 1 s frame that stops at 10 s
    tt2 = raster(0.0, 59.95, 0.05)
    slow = Source(trig=(tt2, from_(tt2, 11.5, 12.0)), out=(slow_t, np.ones(len(slow_t))))
    assert events(evaluate(check(within_max=2.0, already="new_sample"), slow)) == [(11.5, "INCONCLUSIVE")]
    # the sender is alive: no new sample in time is a real FAIL, a new sample with the state a PASS
    alive_t = raster(0.0, 20.0, 0.1)
    wrong = Source(trig=(tt, from_(tt, 10.15)), out=(alive_t, np.where(alive_t < 10.0, 1.0, 0.0)))
    assert events(evaluate(check(within_max=0.3, already="new_sample"), wrong)) == [(10.15, "FAIL")]
    right = Source(trig=(tt, from_(tt, 10.15)), out=(alive_t, np.ones(len(alive_t))))
    assert events(evaluate(check(within_max=0.3, already="new_sample"), right)) == [(10.15, "PASS")]
    frames_t = np.arange(0.0, 60.0, 1.0)                                # alive, but slower than the time limit
    frames = Source(trig=(tt2, from_(tt2, 2.05, 2.5)), out=(frames_t, np.ones(len(frames_t))))
    assert events(evaluate(check(within_max=0.5, already="new_sample"), frames)) == [(2.05, "FAIL")]


# ---------------------------------------------------------------------- record signals are information only
def test_record_signals_do_not_change_events_or_verdicts_and_may_be_missing():
    t = [0.0, 1.0, 9.8, 10.0]                                           # sent on change
    extra_t = raster(0.0, 19.9, 0.1)
    src = Source(trig=(t, [0, 0, 1, 1]), out=(t, [0, 0, 0, 0]), extra=(extra_t, np.full(len(extra_t), 7.0)))
    base = check(within_max=1.0)
    plain = evaluate(base, src)
    assert events(plain) == [(9.8, "INCONCLUSIVE")]
    for record in (["extra"], ["nope"], ["extra", "nope"]):
        result = evaluate(check(within_max=1.0, record=record), src)
        assert result.error is None and events(result) == events(plain) and result.time_range == plain.time_range
        assert result.events[0].note == plain.events[0].note
        assert ("extra" in result.events[0].values) == ("extra" in record) and "nope" not in result.events[0].values
    assert evaluate(check(within_max=1.0, record=["extra"]), src).events[0].values["extra"] == 7.0
    debounce = dict(name="d", trigger=Condition("trig", "==", 1.0), trigger_hold=0.5)
    assert evaluate(Rule(**debounce, record=["extra"]), src).events == evaluate(Rule(**debounce), src).events == []
    interval = dict(name="w", trigger=Condition("trig", "==", 1.0), trigger_mode="while_true")
    assert [e.t_end for e in evaluate(Rule(**interval, record=["extra"]), src).events] == [10.0]


# ---------------------------------------------------------------------- an expect_hold that is never met is fast
def test_an_expect_hold_that_is_never_met_does_not_scan_the_whole_recording():
    n = 100000
    t = np.round(np.arange(n) * 0.01, 6)
    trig = ((np.arange(n) // 50) % 2).astype(float)                   # 1000 rising edges
    out = (np.arange(n) % 2).astype(float)                              # toggles at every sample
    src = Source(trig=(t, trig), out=(t, out))
    started = time.perf_counter()
    result = evaluate(check(within_max=0.2, expect_hold=0.05), src)
    assert time.perf_counter() - started < 3.0
    assert len(result.events) == 1000 and result.count("FAIL") == 1000


# ---------------------------------------------------------------------- script API keeps the scaling
def test_script_change_operators_keep_the_scaling_of_the_signal():
    t = np.arange(4.0)
    speed = Source(v_ms=(t, [0.0, 10.0, 10.0, 36.0]))
    flips = Source(x=(t, [0.0, 1.0, 0.0, 0.0]))

    def times(cond, src):
        return [e.t0 for e in evaluate(Rule("r", trigger=cond), src).events]

    kmh = Sig("v_ms") * 3.6
    assert kmh.changes_to(36).scale == 3.6 and times(kmh.changes_to(36), speed) == [1.0]
    assert times(kmh.changes_from(0, 36), speed) == [1.0]
    assert times((-Sig("x")).rises(), flips) == [2.0] and times((-Sig("x")).falls(), flips) == [1.0]
    changes = (Sig("x") * 2 + 5).changes()
    assert (changes.scale, changes.offset) == (2.0, 5.0) and changes.text() == "2*x + 5 changes"


# ---------------------------------------------------------------------- the visible time range only picks the triggers
def test_a_time_range_picks_the_triggers_but_the_expectation_may_use_later_data():
    t = raster(0.0, 19.9, 0.1)
    trig = from_(t, 9.8)
    src = Source(trig=(t, trig), out=(t, from_(t, 10.2)))
    for rng in (None, (0.0, 10.0)):
        event = evaluate(check(), src, rng).events[0]
        assert (event.verdict, round(event.delay, 6)) == ("PASS", 0.4)
    never = Source(trig=(t, trig), out=(t, np.zeros(len(t))))
    assert events(evaluate(check(), never, (0.0, 10.0))) == [(9.8, "FAIL")]
    assert events(evaluate(check(expect_hold=0.5), src, (0.0, 10.3))) == [(9.8, "PASS")]
    assert evaluate(check(), src, (0.0, 9.7)).events == []              # the trigger lies after the range
    short_t = raster(0.0, 10.0, 0.1)                                    # the recording really ends at 10.0 s
    short = Source(trig=(short_t, from_(short_t, 9.8)), out=(short_t, np.zeros(len(short_t))))
    event = evaluate(check(), short, (0.0, 10.0)).events[0]
    assert event.verdict == "INCONCLUSIVE" and "recording ends" in event.note


# ---------------------------------------------------------------------- while_true: the last sample of an open interval
def test_while_true_checks_the_last_sample_of_an_interval_open_at_the_end():
    t = raster(0.0, 10.0, 0.1)
    trig = from_(t, 5.0)
    out = from_(t, 5.0)
    out[-1] = 0.0                                                       # wrong only at the last sample
    rule = check(trigger_mode="while_true")
    event = evaluate(rule, Source(trig=(t, trig), out=(t, out))).events[0]
    assert event.verdict == "FAIL" and "kept" in event.note
    assert events(evaluate(rule, Source(trig=(t, trig), out=(t, from_(t, 5.0))))) == [(5.0, "PASS")]


# ---------------------------------------------------------------------- repeated or interleaved time stamps
def test_cyclic_signals_with_repeated_or_interleaved_time_stamps():
    a = raster(0.0, 4.99, 0.01)
    assert is_cyclic(np.repeat(a, 2)) and valid_until(np.repeat(a, 2)) == pytest.approx(4.99 + 0.05)
    for offset in (0.001, 0.003):                                       # union of two 10 ms rasters
        union = np.union1d(a, np.round(a + offset, 6))
        assert is_cyclic(union) and valid_until(union) == pytest.approx(union[-1] + 0.05)
    assert is_cyclic(np.union1d(a, np.round(a + 0.007, 6)))            # steady enough as it is (7 ms, 3 ms)
    irregular = np.cumsum([0.5, 3.1, 0.2, 7.0, 1.3, 0.05, 2.2, 4.4, 0.9, 6.1, 0.3, 2.8, 0.7, 5.0])
    assert not is_cyclic(irregular) and valid_until(irregular) == float("inf")
    t = raster(0.0, 19.9, 0.1)
    src = Source(trig=(t, from_(t, 15.0, 16.0)), out=(np.repeat(a, 2), np.zeros(2 * len(a))))
    event = evaluate(check(), src).events[0]
    assert event.verdict == "INCONCLUSIVE" and "out" in event.note    # stopped at 5 s, not held to 15 s


# ---------------------------------------------------------------------- count_initial for an OR of a change and a state
def test_count_initial_counts_the_state_part_of_an_or_with_a_change():
    t = np.arange(0.0, 6.0, 1.0)
    src = Source(a=(t, [0, 0, 0, 1, 1, 1]), b=(t, [1, 1, 0, 0, 0, 0]))
    trigger = Group("or", [Condition("a", "changes"), Condition("b", "==", 1.0)])
    assert [e.t0 for e in evaluate(Rule("r", trigger=trigger), src).events] == [3.0]
    found = evaluate(Rule("r", trigger=trigger, count_initial=True), src).events
    assert [e.t0 for e in found] == [0.0, 3.0]
    assert "already true when the data started" in found[0].note and not found[1].note
    late = Source(a=(t, [0, 0, 0, 0, 1, 1]), b=(np.arange(2.0, 6.0), [1, 1, 0, 0]))   # b starts at 2 s
    found = evaluate(Rule("r", trigger=trigger, count_initial=True), late).events
    assert [e.t0 for e in found] == [2.0, 4.0] and "already true" in found[0].note


# ---------------------------------------------------------------------- a comparison without a value
@pytest.mark.parametrize("cond", [{"signal": "a", "op": "=="}, {"signal": "a", "op": ">", "value": None},
                                  {"signal": "a", "op": "in"}, {"signal": "a", "op": "changes to"}])
def test_a_comparison_without_a_value_is_a_rule_error(cond):
    t = np.arange(0.0, 5.0)
    src = Source(a=(t, [0, 1, 0, 1, 0]))
    rule = Rule.from_json({"name": "r", "trigger": {"mode": "and", "items": [cond]}})
    result = evaluate(rule, src)
    assert result.error and "'a %s'" % cond["op"] in result.error and not result.events
    expect = Rule.from_json({"name": "r", "trigger": {"items": [{"signal": "a", "op": "rises"}]},
                             "expect": {"items": [cond]}})
    assert evaluate(expect, src).error


def test_changes_from_without_a_new_value_is_no_error():
    t = np.arange(0.0, 5.0)
    rule = Rule.from_json({"name": "r", "trigger": {"items": [{"signal": "a", "op": "changes from", "previous": 1}]}})
    result = evaluate(rule, Source(a=(t, [0, 1, 0, 1, 2])))
    assert result.error is None and [e.t0 for e in result.events] == [2.0, 4.0]
    between = Rule.from_json({"name": "r", "trigger": {"items": [{"signal": "a", "op": "between"}]}})
    assert "needs two values" in evaluate(between, Source(a=(t, [0, 1, 0, 1, 2]))).error
