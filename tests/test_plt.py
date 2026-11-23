"""PLT script parser, formula engine and processing pipeline."""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from tracepaddock.core.model import KIND_BIT, Dataset, EnumTable, SignalData, SignalInfo
from tracepaddock.display import DisplaySignal
from tracepaddock.plt.expr import ExprError, compile_expr
from tracepaddock.plt.model import Op, PltEntry, PltScript
from tracepaddock.plt.parser import dump_plt, format_entry, load_plt, parse_plt, save_plt, short_name
from tracepaddock.plt.process import build_display_signals, resample_hold, signal_statistics


# ------------------------------------------------------------------ parser
@pytest.mark.parametrize("fixture", ["d97_plt", "mf4_plt"])
def test_generated_views_round_trip_byte_exact(fixture, request):
    path = request.getfixturevalue(fixture)
    script = load_plt(path)
    assert dump_plt(script).encode(script.encoding) == path.read_bytes()


def test_generated_view_parses_entries(mf4_plt):
    script = load_plt(mf4_plt)
    entries = list(script.entries())
    assert len(entries) == 3 and all(e.enabled for e in entries)
    first = entries[0]
    assert (first.name, first.view, first.color, first.unit) == ("BenchAcceleration", "nobit", 1, "m/s²")
    assert first.scmin == -2 and first.scmax == 10


def test_combinations_of_view_attributes():
    script = parse_plt("\n".join([
        "DemoTorque 2 scmax=200",
        "DemoSpeed unit=km/s",
        "+ exp=3.6*x",
        "~ DemoSpeed 2 key=distance unit=m scmin=0 scmax=500",
        "~ + integ",
        "BenchRequest nobit 2 key=Braking unit=- exp(x1=moving,x2=StoppingDistance)=ne(x1*x2,0)",
        "AxleSpin key=Rotation rate unit=grad/s integ",
        "DriveCommand green ref1=min ref2=max",
        "SIGNAL_1 DGF=BTH:10:2 REVERT DGF=BTH:10:2 REVERT",
        "PackedStatus:4:-3 nobit",
        "x unit= scmin=-1 scmax=1",
        "a exp=x + 3 5",
    ]))
    e = list(script.entries())
    assert (e[0].name, e[0].color, e[0].scmax) == ("DemoTorque", 2, 200.0)
    assert e[1].unit == "km/s" and [o.expr for o in e[1].ops] == ["3.6*x"]
    assert not e[2].enabled and e[2].key == "distance" and [o.kind for o in e[2].ops] == ["integ"]
    assert e[3].ops[0].refs == {"x1": "moving", "x2": "StoppingDistance"} and e[3].ops[0].expr == "ne(x1*x2,0)"
    assert e[4].key == "Rotation rate" and e[4].unit == "grad/s"
    assert e[5].color == 11 and e[5].refs == {1: "MIN", 2: "MAX"}
    assert [o.kind for o in e[6].ops] == ["dgf", "revert", "dgf", "revert"] and e[6].ops[0].args == ("BTH", 10.0, 2.0)
    assert (e[7].name, e[7].bit_pos, e[7].bit_count, e[7].bit_signed) == ("PackedStatus", 4, 3, True)
    assert e[8].unit == "" and e[8].scmin == -1.0
    assert e[9].ops[0].expr == "x + 3" and e[9].color == 5


def test_changed_entry_is_rewritten_in_standard_layout(tmp_path):
    script = parse_plt('sig  nobit 1 unit=m scmin=0 scmax=1\r\n//BEGIN_GROUP "g" ENABLED\r\nb 3 bit\r\n//END_GROUP\r\n')
    first = next(script.entries())
    first.scmax, first.key = 2.5e6, "short"
    first.ops.append(Op("exp", "exp=x*2", expr="x*2"))
    text = dump_plt(script)
    assert text.splitlines()[0] == "sig  nobit 1 key=short unit=m scmin=0 scmax=2.5e+006"
    assert text.splitlines()[1] == "+ exp=x*2"
    assert "\r\n" in text and list(script.entries())[1].group == "g"
    target = tmp_path / "out.plt"
    save_plt(script, target)
    assert dump_plt(load_plt(target)) == text and not script.modified


def test_short_names():
    assert short_name("#10.Thermal.Bench_Heat_0", drop_bus_prefix=True) == "#Bench_Heat_0"
    assert short_name("DemoProbe_phi_[0]\\Session_1", drop_device=True) == "DemoProbe_phi_[0]"
    assert short_name("A.B.C.D", keep_elements=2) == "A.B"
    assert short_name("A.B.C.D", keep_elements=2, keep_from_left=False) == "C.D"


# ------------------------------------------------------------------ expressions
def ev(text, **env):
    return compile_expr(text).evaluate(env)


def test_arithmetic_and_precedence():
    assert ev("2+3*4") == 14
    assert ev("2**3**2") == 512            # right associative
    assert ev("-2**2") == -4
    assert ev("(1+2)*3/4") == pytest.approx(2.25)
    assert ev("3.6*x", x=np.array([1.0, 2.0])).tolist() == [3.6, 7.2]
    assert ev("sin(2*PI*t/2)", t=np.array([0.5]))[0] == pytest.approx(1.0)
    assert ev("E") == pytest.approx(math.e)


def test_functions():
    x = np.array([-2.5, -0.5, 0.0, 0.5, 2.5])
    assert ev("max(-1,min(1,x))", x=x).tolist() == [-1, -0.5, 0, 0.5, 1]
    assert ev("ior(ge(x,1),le(x,-1))", x=x).tolist() == [1, 0, 0, 0, 1]
    assert ev("aint(x)", x=x).tolist() == [-2, 0, 0, 0, 2]
    assert ev("anint(x)", x=x).tolist() == [-3, -1, 0, 1, 3]
    assert ev("sgn(x)", x=x).tolist() == [-1, -1, 1, 1, 1]
    assert ev("switch(gt(x,0),10,20)", x=x).tolist() == [20, 20, 20, 10, 10]
    assert ev("mod(7,3)") == 1 and ev("abs(-3)") == 3


def test_bit_functions_match_integer_arithmetic():
    v = np.array([0.0, 5.0, 255.0, 4660.0])
    assert ev("iand(x,15)", x=v).tolist() == [0, 5, 15, 4660 & 15]
    assert ev("ieor(x,255)", x=v).tolist() == [255, 250, 0, 4660 ^ 255]
    assert ev("btest(x,2)", x=v).tolist() == [0, 1, 1, (4660 >> 2) & 1]
    assert ev("ibits(x,4,8)", x=v).tolist() == [0, 0, 15, (4660 >> 4) & 255]
    assert ev("sbits(x,0,8)", x=np.array([255.0, 127.0, 128.0])).tolist() == [-1, 127, -128]
    assert ev("ishft(x,4)", x=v).tolist() == [0, 80, 4080, 4660 << 4]
    assert ev("ishft(x,-2)", x=v).tolist() == [0, 1, 63, 4660 >> 2]
    assert ev("ibset(x,3)", x=v).tolist() == [8, 13, 255, 4660 | 8]
    assert ev("ibclr(x,0)", x=v).tolist() == [0, 4, 254, 4660]
    assert ev("not(x)", x=np.array([0.0])).tolist() == [-1]
    assert ev("ishftc(x,1,8)", x=np.array([129.0])).tolist() == [3]


def test_dword_to_float_and_map():
    bits = struct.unpack("<I", struct.pack("<f", 1.5))[0]
    assert ev("(DWORD~FLOAT)x", x=np.array([float(bits)]))[0] == 1.5
    x = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    assert ev("map(x,-10,-10,10,10)", x=x).tolist() == [-10, -10, 0, 10, 10]
    assert ev("map(x,-10,0,-10,1,10,1,10,0)", x=np.array([-11.0, 0.0, 11.0])).tolist() == [0, 1, 0]


@pytest.mark.parametrize("bad", ["", "x +", "foo(x)", "__import__('os')", "x.real", "1 2", "min()", "y", "lambda: 1"])
def test_invalid_expressions_are_rejected(bad):
    with pytest.raises(ExprError):
        compile_expr(bad).evaluate({"x": np.zeros(2), "t": np.zeros(2)})


# ------------------------------------------------------------------ pipeline
class FakeDataset(Dataset):
    """A fast 10 ms signal, a slow 40 ms signal and a status word."""

    def __init__(self):
        super().__init__("fake.d97")
        self._t_fast = np.arange(0, 1.0, 0.01)
        self._t_slow = np.arange(0.005, 1.0, 0.04)
        self._data = {
            "fast": (self._t_fast, np.arange(len(self._t_fast), dtype=float)),
            "slow": (self._t_slow, np.arange(len(self._t_slow), dtype=float) * 10),
            "status": (self._t_fast, (np.arange(len(self._t_fast)) % 16).astype(float)),
            "flag": (self._t_fast, (np.arange(len(self._t_fast)) % 2).astype(float)),
        }
        self.signals = [SignalInfo("fast", unit="m/s", scmin=0, scmax=200), SignalInfo("slow", unit="bar"),
                        SignalInfo("status", enum=EnumTable(values={0.0: "off"})),
                        SignalInfo("flag", kind=KIND_BIT)]
        self._finish_catalogue()

    def read(self, signal):
        t, y = self._data[signal.name]
        return SignalData(t, y, signal.enum)

    def time_range(self):
        return (0.0, 0.99)

    def common_time_base(self):
        return np.union1d(self._t_fast, self._t_slow)


def build(text, **settings):
    class S:
        def get(self, key, default=None):
            return settings.get(key, default)
    return build_display_signals(FakeDataset(), parse_plt(text), S())


def test_plain_entry_takes_unit_and_range_from_the_dataset():
    (sig,) = build("fast 3")
    assert (sig.name, sig.unit, sig.color, sig.digital) == ("fast", "m/s", 3, False)
    assert (sig.scmin, sig.scmax) == (0.0, 200.0) and len(sig.y) == 100 and not sig.modified


def test_missing_signal_and_bad_expression_do_not_stop_the_others():
    a, b, c = build("nothere 1\nfast exp=foo(x)\nslow 2")
    assert a.error == "not contained in the dataset" and not a.has_data
    assert "unknown function" in b.error
    assert c.has_data and c.error is None


def test_ops_are_applied_in_order():
    (a,) = build("fast exp=x*2 diff")
    (b,) = build("fast diff exp=x*2")
    assert a.y[1] == pytest.approx(200.0) and b.y[1] == pytest.approx(200.0)
    (c,) = build("fast -min exp=x+1")
    (d,) = build("fast exp=x+1 -min")
    assert c.y[0] == 1.0 and d.y[0] == 0.0 and c.modified


def test_expression_over_two_sample_rates_uses_hold_on_the_union():
    sigs = build("fast 1\nslow 2\nfast key=sum exp(x1=slow)=x+x1")
    total = sigs[2]
    ds = FakeDataset()
    tf, yf = ds._data["fast"]
    ts, ys = ds._data["slow"]
    assert np.array_equal(total.t, np.union1d(tf, ts))
    assert np.allclose(total.y, resample_hold(tf, yf, total.t) + resample_hold(ts, ys, total.t))


def test_forward_reference_falls_back_to_dataset_or_reports():
    (a,) = build("fast key=k exp(x1=later)=x+x1")
    assert "later" in a.error
    (b,) = build("fast key=k exp(x1=slow)=x1")
    assert b.has_data


def test_placeholder_bit_extraction_and_digital_placement():
    sigs = build("fast 1\n* 2 key=T_SIN exp=sin(2*PI*t/2)\nstatus:1 3\nstatus:0:-2 4\nflag 5\nflag 6 exp=x*2\nflag nobit 7")
    sine, bit1, field, flag, flag2, flag3 = sigs[1:]
    base = FakeDataset().common_time_base()          # a placeholder lives on the common time axis
    assert sine.has_data and np.array_equal(sine.t, base)
    assert sine.y[np.searchsorted(base, 0.25)] == pytest.approx(math.sin(math.pi * 0.25))
    assert bit1.digital and set(bit1.y) == {0.0, 1.0} and bit1.enum is None
    assert not field.digital and set(field.y) == {0.0, 1.0, -2.0, -1.0}
    assert flag.digital and not flag2.digital and not flag3.digital


def test_moving_average_at_the_edges():
    (fwd,) = build("fast mavr(4)")
    (bwd,) = build("fast mavr(-4)")
    x = np.arange(100, dtype=float)
    n = len(x)
    want_f = [x[k:min(n, k + 4)].mean() for k in range(n)]
    want_b = [x[max(0, k - 3):k + 1].mean() for k in range(n)]
    assert np.allclose(fwd.y, want_f) and np.allclose(bwd.y, want_b)


def test_integ_diff_shift_revert_and_reference_lines():
    (i,) = build("fast integ")
    assert i.y[0] == 0 and i.y[-1] == pytest.approx(sum(range(0, 99)) * 0.01)   # left rectangles
    (s,) = build("fast shift(2)")
    assert s.y[:4].tolist() == [0, 0, 0, 1]
    (r,) = build("fast revert ref0=min ref1=max ref2=5")
    assert r.y[0] == 99 and r.ref_lines == [0.0, 99.0, 5.0]


def test_filter_identities():
    # a cascade of two PDT2 sections equals the backward moving average of length 4
    (casc,) = build("fast dgf=PDT2:0.25:0:0.25:0:0 dgf=PDT2:1:1:0:0:0")
    (mavr,) = build("fast mavr(-4)")
    assert np.allclose(casc.y[3:], mavr.y[3:])
    # three PDT2 sections equal the 5th order Butterworth at 10 Hz for 100 Hz sampling
    text = ("fast dgf=PDT2:0.2452:0.4905:0.2452:0.4905:-0.5095 dgf=PDT2:0.0647:0.1294:0.0647:-1.0966:0.3555 "
            "dgf=PDT2:0.0808:0.1616:0.0808:-1.3693:0.6926")
    (sections,) = build(text)
    (bth,) = build("fast dgf=bth:10:5")
    assert np.allclose(sections.y, bth.y, rtol=2e-2, atol=0.5)


def test_statistics_with_common_time_base():
    ds = FakeDataset()
    (slow,) = build_display_signals(ds, parse_plt("slow"))
    base = ds.common_time_base()
    stats = signal_statistics(slow, 0.0, 0.99, base)
    held = resample_hold(slow.t, slow.y, base)
    assert stats["mean"] == pytest.approx(held.mean()) and stats["std"] == pytest.approx(held.std(ddof=1))
    window = signal_statistics(slow, 0.5, 0.6, base)
    assert window["min"] == pytest.approx(resample_hold(slow.t, slow.y, np.array([0.5]))[0])
