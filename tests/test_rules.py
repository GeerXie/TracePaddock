"""Stage 4: the rule engine (conditions, triggers, expectations, verdicts, delays)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tracepaddock.core.model import EnumTable
from tracepaddock.rules.engine import Event, SignalSource, Step, evaluate
from tracepaddock.rules.model import Condition, Group, Operand, Rule, RuleSet


class Source(SignalSource):
    """A signal source from plain arrays."""

    def __init__(self, **signals):
        super().__init__()
        for name, (t, y, *rest) in signals.items():
            enum = rest[0] if rest else None
            self._cache[name.lower()] = Step(name, np.asarray(t, float), np.asarray(y, float), enum)


def training_source() -> Source:
    t100 = np.round(np.arange(0.0, 60.0, 0.1), 6)
    t20 = np.round(np.arange(0.0, 60.0, 0.02), 6)

    def steps(t, changes, start=0.0):
        y = np.full(len(t), float(start))
        for when, value in changes:
            y[t >= when - 1e-9] = value
        return y

    sign = steps(t100, [(5.0, 1), (7.0, 0), (22.0, 1), (24.0, 0), (30.0, 2), (32.0, 0), (41.0, 3), (43.0, 0),
                        (55.0, 4), (57.0, 0)])
    value = steps(t100, [(5.0, 120), (22.0, 80), (30.0, 60), (41.0, 50), (55.0, 0)])
    out = steps(t20, [(5.34, 120), (22.18, 80), (41.42, 50), (55.26, 0)])
    road = steps(t100, [(20.0, 2), (40.0, 3)], 1.0)
    names = EnumTable(values={0.0: "无标志", 1.0: "非条件限速", 2.0: "条件限速", 3.0: "隐性限速", 4.0: "限速解除"})
    roads = EnumTable(values={1.0: "高速", 2.0: "乡村", 3.0: "城市"})
    return Source(SignType=(t100, sign, names), SignValue=(t100, value), BenchOutput=(t20, out), Road=(t100, road, roads))


def training_rule(**kw) -> Rule:
    args = dict(name="Synthetic speed response", trigger=Group("and", [Condition("SignType", "in", [1, 2, 3, 4])]),
                expect=Group("and", [Condition("BenchOutput", "==", Operand("SignValue"), tolerance=0.5)]),
                within_max=0.5, record=["Road"])
    args.update(kw)
    return Rule(**args)


def test_trigger_response_with_delays_and_texts():
    result = evaluate(training_rule(), training_source())
    assert result.error is None
    got = [(round(e.t0, 3), e.verdict, None if e.delay is None else round(e.delay * 1000)) for e in result.events]
    assert got == [(5.0, "PASS", 340), (22.0, "PASS", 180), (30.0, "FAIL", None), (41.0, "PASS", 420),
                   (55.0, "PASS", 260)]
    fail = result.events[2]
    assert fail.texts["SignType"] == "条件限速" and fail.texts["Road"] == "乡村"
    assert fail.actual["BenchOutput"] == 80.0                    # what the output showed at the time limit
    summary = result.summary()
    assert summary["PASS"] == 4 and summary["FAIL"] == 1 and summary["delay_max"] == pytest.approx(0.42)
    assert result.response_period == pytest.approx(0.02)


def test_time_limit_and_too_early():
    late = evaluate(training_rule(within_max=0.3), training_source())
    assert [e.verdict for e in late.events] == ["FAIL", "PASS", "FAIL", "FAIL", "PASS"]
    early = evaluate(training_rule(within_min=0.2, within_max=0.5), training_source())
    assert [e.verdict for e in early.events] == ["PASS", "TOO_EARLY", "FAIL", "PASS", "PASS"]


def test_already_satisfied_modes():
    t = np.arange(0.0, 10.0, 0.1)
    trigger = (t >= 2.0).astype(float) - (t >= 4.0) + (t >= 6.0)     # rising at 2.0 and 6.0
    out_t = np.arange(0.0, 10.0, 1.0)                                # sent every second, already 1 since 1.0
    out = (out_t >= 1.0).astype(float)
    src = Source(Trig=(t, trigger), Out=(out_t, out))
    base = dict(name="r", trigger=Condition("Trig", "==", 1.0), expect=Condition("Out", "==", 1.0), within_max=2.0)
    hold = evaluate(Rule(**base, already="hold"), src)
    assert [(e.verdict, e.delay, e.already) for e in hold.events] == [("PASS", 0.0, True), ("PASS", 0.0, True)]
    new = evaluate(Rule(**base, already="new_sample"), src)
    assert [(e.verdict, round(e.delay, 6)) for e in new.events] == [("PASS", 0.0), ("PASS", 0.0)]
    src2 = Source(Trig=(t + 0.05, trigger), Out=(out_t, out))         # trigger between two frames
    new2 = evaluate(Rule(**base, already="new_sample"), src2)
    assert [round(e.delay, 6) for e in new2.events] == [0.95, 0.95]
    change = evaluate(Rule(**base, already="change"), src)
    assert [e.verdict for e in change.events] == ["FAIL", "FAIL"]   # never changes after the trigger


def test_change_at_the_trigger_instant_counts_as_a_change():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    src = Source(A=(t, [0, 1, 1, 1]), B=(t, [0, 1, 1, 1]))
    r = evaluate(Rule("r", trigger=Condition("A", "==", 1), expect=Condition("B", "==", 1), already="change"), src)
    assert [(e.verdict, e.delay) for e in r.events] == [("PASS", 0.0)]


def test_debounce_scope_and_initial_state():
    t = np.round(np.arange(0.0, 10.0, 0.01), 6)
    speed = np.where((t > 1) & (t < 1.1), 95, np.where((t > 3) & (t < 6), 96, 85)).astype(float)
    speed[:20] = 99                                               # already over at the start
    active = (t < 5.0).astype(float)
    src = Source(v=(t, speed), active=(t, active))
    over = Condition("v", ">", 90.0)
    r = evaluate(Rule("over", trigger=over), src)
    assert [round(e.t0, 2) for e in r.events] == [1.01, 3.01]      # the start state is not an event
    r = evaluate(Rule("over", trigger=over, count_initial=True), src)
    assert [round(e.t0, 2) for e in r.events] == [0.0, 1.01, 3.01]
    r = evaluate(Rule("over", trigger=over, trigger_hold=0.5), src)
    assert [round(e.t0, 2) for e in r.events] == [3.01]           # the 0.1 s spike is filtered
    r = evaluate(Rule("over", trigger=over, scope=Condition("active", "==", 1)), src)
    assert [round(e.t0, 2) for e in r.events] == [1.01, 3.01]
    r = evaluate(Rule("over", trigger=over, trigger_mode="while_true"), src)
    assert [(round(e.t0, 2), round(e.t_end, 2)) for e in r.events] == [(1.01, 1.1), (3.01, 6.0)]


def test_while_true_needs_the_expectation_to_last():
    t = np.round(np.arange(0.0, 10.0, 0.1), 6)
    over = ((t >= 2) & (t < 6)).astype(float)
    warn = ((t >= 2.3) & (t < 4.0)).astype(float)                   # drops too early
    warn2 = ((t >= 2.3) & (t < 6.2)).astype(float)
    src = Source(over=(t, over), warn=(t, warn), warn2=(t, warn2))
    base = dict(trigger=Condition("over", "==", 1), trigger_mode="while_true", within_max=0.5)
    bad = evaluate(Rule("r", expect=Condition("warn", "==", 1), **base), src)
    good = evaluate(Rule("r", expect=Condition("warn2", "==", 1), **base), src)
    assert bad.events[0].verdict == "FAIL" and "kept" in bad.events[0].note
    assert good.events[0].verdict == "PASS" and good.events[0].delay == pytest.approx(0.3)


@pytest.mark.parametrize("already", ["hold", "new_sample", "change"])
def test_interval_response_must_occur_before_the_first_false_trigger_sample(already):
    t = np.array([0.0, 1.0, 2.0, 2.75, 3.0, 4.0, 5.0])
    trigger = ((t >= 1.0) & (t < 3.0)).astype(float)
    rule = Rule("r", trigger=Condition("trig", "==", 1), expect=Condition("out", "==", 1),
                trigger_mode="while_true", within_max=1.75, already=already)

    for time_range in (None, (1.0, 2.0)):
        # An expectation met right at the configured time limit counts, including past a plot cut.
        src = Source(trig=(t, trigger), out=(t, (t >= 2.75).astype(float)))
        event, = evaluate(rule, src, time_range).events
        assert (event.verdict, event.t1, event.t_end) == ("PASS", 2.75, 3.0 if time_range is None else 2.0)

        # An expectation first met exactly when the trigger becomes false no longer counts.
        rule.within_max = 4.0
        src = Source(trig=(t, trigger), out=(t, (t >= 3.0).astype(float)))
        event, = evaluate(rule, src, time_range).events
        assert event.verdict == "FAIL" and event.t1 is None and event.actual["out"] == 0.0
        rule.within_max = 1.75


def test_visible_range_uses_the_recorded_state_before_its_left_edge():
    t = np.arange(6.0)
    src = Source(trig=(t, [0, 1, 1, 1, 0, 0]), out=(t, [0, 1, 1, 1, 0, 0]))
    rule = Rule("r", trigger=Condition("trig", "==", 1), expect=Condition("out", "==", 1), already="change")
    event, = evaluate(rule, src, (1.0, 3.0)).events
    assert (event.t0, event.t1, event.verdict) == (1.0, 1.0, "PASS")
    rule.count_initial = True
    assert evaluate(rule, src, (2.0, 3.0)).events == []
    rule.trigger = Group("and", [Condition("trig", "rises")])
    assert [event.t0 for event in evaluate(rule, src, (1.0, 3.0)).events] == [1.0]


def test_change_operators_and_groups():
    t = np.arange(0.0, 6.0, 1.0)
    src = Source(m=(t, [0, 1, 2, 2, 1, 0]), f=(t, [0, 0, 1, 1, 1, 0]))
    def times(cond):
        return [e.t0 for e in evaluate(Rule("r", trigger=cond), src).events]
    assert times(Condition("m", "changes")) == [1.0, 2.0, 4.0, 5.0]
    assert times(Condition("m", "changes to", 2.0)) == [2.0]
    assert times(Condition("m", "changes from", 2.0, previous=2.0)) == []
    assert times(Condition("m", "changes from", 1.0, previous=2.0)) == [4.0]
    assert times(Condition("m", "rises")) == [1.0, 2.0] and times(Condition("m", "falls")) == [4.0, 5.0]
    both = Group("and", [Condition("m", "changes"), Condition("f", "==", 1)])
    assert times(both) == [2.0, 4.0]
    either = Group("or", [Condition("m", "changes to", 1.0), Condition("f", "==", 1)])
    assert times(either) == [1.0, 2.0, 4.0]
    assert times(Group("and", [Condition("m", "between", [1, 2])], negate=True)) == [5.0]
    with pytest.raises(ValueError):
        Group.from_json({"mode": "and", "items": [{"signal": "m", "op": "~~"}]})


def test_scaled_operands_and_tolerance():
    t = np.arange(0.0, 4.0, 1.0)
    src = Source(v_ms=(t, [10.0, 20.0, 25.0, 30.0]), v_kmh=(t, [36.2, 71.0, 90.0, 108.0]))
    r = evaluate(Rule("r", trigger=Condition("v_kmh", "==", Operand("v_ms", 3.6), tolerance=0.5),
                      count_initial=True, trigger_mode="while_true"), src)
    assert [(e.t0, e.t_end) for e in r.events] == [(0.0, 1.0), (2.0, 3.0)]   # open at the end of the data


def test_missing_signals_and_values_before_the_first_sample():
    src = Source(a=([5.0, 6.0], [1.0, 1.0]), b=([0.0, 10.0], [0.0, 0.0]))
    r = evaluate(Rule("r", trigger=Condition("nope", "==", 1)), src)
    assert r.error and "nope" in r.error and not r.events
    r = evaluate(Rule("r", trigger=Condition("a", "==", 1), count_initial=True), src)
    assert [e.t0 for e in r.events] == [5.0]                         # no value before its first sample


def test_inconclusive_at_the_end_of_the_recording():
    t = np.arange(0.0, 5.0, 0.1)
    src = Source(trig=(t, (t >= 4.8).astype(float)), out=(t, np.zeros(len(t))))
    r = evaluate(Rule("r", trigger=Condition("trig", "==", 1), expect=Condition("out", "==", 1), within_max=1.0), src)
    assert [e.verdict for e in r.events] == ["INCONCLUSIVE"]


def test_rules_round_trip_json(tmp_path):
    rules = RuleSet(tmp_path / "rules.json", autoload=False)
    rules.rules.append(training_rule(scope=Condition("Road", "in", [1, 2]), trigger_hold=0.2, expect_hold=0.1,
                                already="new_sample", note="限速输出检查"))
    rules.scripts["mine"] = "rule('x', trigger=S('a') > 1)"
    rules.save()
    again = RuleSet(tmp_path / "rules.json")
    assert [r.to_json() for r in again.rules] == [r.to_json() for r in rules.rules]
    assert again.scripts == {"mine": "rule('x', trigger=S('a') > 1)"}
    assert "限速输出检查" in (tmp_path / "rules.json").read_text(encoding="utf-8")
    bad = RuleSet(tmp_path / "x.json", autoload=False)
    problems = bad.take({"rules": [{"name": "broken", "trigger_mode": "sometimes"}, training_rule().to_json()]})
    assert len(bad.rules) == 1 and problems and "broken" in problems[0]
    event = evaluate(training_rule(), training_source()).events[0]
    assert Event.from_json(json.loads(json.dumps(event.to_json()))) == event


def test_rule_texts():
    names = {"SignType": {1.0: "非条件限速"}}
    text = training_rule().trigger.text(names)
    assert "非条件限速" in text and "SignType in" in text
    assert Condition("v", "==", Operand("w", 3.6, -1.0), tolerance=0.5).text() == "v == 3.6*w - 1 ±0.5"


# ---------------------------------------------------------------------- signals without value (start, stop)
def test_a_condition_true_when_its_signal_starts_is_the_initial_state():
    t = np.round(np.arange(0.0, 10.0, 0.1), 6)
    ta = np.round(np.arange(2.0, 10.0, 0.1), 6)                      # starts 2 s after the other signal
    a = np.where((ta >= 4.0) & (ta < 6.0), 0.0, 1.0)
    src = Source(a=(ta, a), b=(t, np.zeros(len(t))))
    rule = Rule("r", trigger=Group("and", [Condition("a", "==", 1), Condition("b", "==", 0)]))
    assert [e.t0 for e in evaluate(rule, src).events] == [6.0]      # 2.0 s is not a rising edge
    rule.count_initial = True
    events = evaluate(rule, src).events
    assert [e.t0 for e in events] == [2.0, 6.0]
    assert "already true when the data started" in events[0].note and not events[1].note


def test_not_of_a_signal_without_value_stays_unknown():
    t = np.round(np.arange(0.0, 10.0, 0.1), 6)
    ta = np.round(np.arange(3.0, 10.0, 0.1), 6)
    src = Source(a=(ta, (ta >= 5.0).astype(float)), b=(t, np.zeros(len(t))))
    rule = Rule("r", trigger=Group("and", [Condition("a", "==", 0)], negate=True), count_initial=True,
                record=["b"])
    assert [e.t0 for e in evaluate(rule, src).events] == [5.0]      # not "true" before a has a value


def test_a_cyclic_signal_that_stops_has_no_value_and_gives_inconclusive():
    t100 = np.round(np.arange(0.0, 20.0, 0.1), 6)
    t20 = np.round(np.arange(0.0, 10.0, 0.02), 6)                    # the speed stops being sent at 10 s
    warn = (((t100 >= 5.0) & (t100 < 6.0)) | ((t100 >= 15.0) & (t100 < 16.0))).astype(float)
    src = Source(warn=(t100, warn), speed=(t20, np.full(len(t20), 95.0)), limit=(t100, np.full(len(t100), 90.0)))
    step = src.get("speed")
    assert step.cyclic and step.valid_until == pytest.approx(10.0 - 0.02 + 5 * 0.02)
    assert np.isnan(step.value_at(12.0)) and step.value_at(9.5) == 95.0
    rule = Rule("r", trigger=Condition("warn", "==", 1), expect=Condition("speed", ">", Operand("limit")),
                within_max=0.0, record=["speed"])
    events = evaluate(rule, src).events
    assert [(e.t0, e.verdict) for e in events] == [(5.0, "PASS"), (15.0, "INCONCLUSIVE")]
    assert "speed" in events[1].note and "speed" not in events[1].values and "speed" not in events[1].actual
    # a signal sent on change only is held to the end of the recording
    on_change = Source(warn=(t100, warn), speed=([0.0, 3.0], [95.0, 95.0]), limit=(t100, np.full(len(t100), 90.0)))
    assert not on_change.get("speed").cyclic
    assert [e.verdict for e in evaluate(rule, on_change).events] == ["PASS", "PASS"]


def test_known_wrong_stays_fail_and_missing_data_during_an_interval_is_inconclusive():
    t = np.round(np.arange(0.0, 20.0, 0.1), 6)
    tout = np.round(np.arange(0.0, 12.0, 0.1), 6)                   # the output stops at 12 s
    trig = (((t >= 2.0) & (t < 4.0)) | ((t >= 10.0) & (t < 14.0))).astype(float)
    src = Source(trig=(t, trig), out=(tout, np.where(tout < 3.0, 0.0, 1.0)))
    rule = Rule("r", trigger=Condition("trig", "==", 1), expect=Condition("out", "==", 1), within_max=0.5)
    assert [e.verdict for e in evaluate(rule, src).events] == ["FAIL", "PASS"]   # 0 at 2.5 s is known
    rule.trigger_mode = "while_true"
    events = evaluate(rule, src).events
    assert [e.verdict for e in events] == ["FAIL", "INCONCLUSIVE"]
    assert "out" in events[1].note
