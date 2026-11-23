"""Tests for measurement data readers and the file-type registry."""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tracepaddock.core.d97 import D97Dataset, parse_enum, write_d97
from tracepaddock.core.model import KIND_BIT, KIND_FLOAT, DatasetError
from tracepaddock.core.registry import list_datasets_in_directory, open_dataset
from tracepaddock.plt.parser import load_plt
from tracepaddock.plt.process import build_display_signals, signal_statistics

# ------------------------------------------------------------------ signal records
def test_synthetic_d97_catalogue_and_values(d97_file):
    with D97Dataset(d97_file) as ds:
        assert [(s.name, s.kind) for s in ds.signals] == [("MotorSpeed", KIND_FLOAT), ("Ready", KIND_BIT)]
        speed = ds.find("MotorSpeed")
        assert speed.unit == "m/s" and speed.sample_count == 5
        assert ds.read(speed).y.tolist() == [0., 2., 4., 6., 8.]
        assert ds.read(ds.find("Ready")).y.tolist() == [0., 1., 1.]
        np.testing.assert_array_equal(ds.common_time_base(), np.arange(5.0))
        assert ds.time_range() == (0.0, 4.0)


def test_d97_script_builds_display_signals(d97_file, d97_plt):
    with D97Dataset(d97_file) as ds:
        signals = build_display_signals(ds, load_plt(d97_plt))
        assert [s.name for s in signals if s.has_data] == ["MotorSpeed", "Ready"]


def _write_d97(path, precision="DOUBLE", newline="\r\n", extra_control=()):
    fmt = "<d" if precision == "DOUBLE" else "<f"
    head = ["; test", "[control]", "FLOAT=INTEL387", f"PRECISION={precision}", "DATAMATRIX=SV_BY_*", *extra_control,
            "[SIGNAL0]", "NAME=speed", "UNIT=m/s", "TYP=FLT", "DPOS=1", "TMSV=0", "SCMIN=0", "SCMAX=50",
            "[SIGNAL1]", "NAME=bit3", "TYP=D0", "DPOS=2", "TMSV=0", "BITP=3",
            "[SIGNAL2]", "NAME=field", "TYP=DS", "DPOS=2", "TPOS=0", "BITP=0", "BITC=3",
            '[SIGNAL3]', "NAME=state", "TYP=FLT", "DPOS=2", "TMSV=0", 'ENUM={="other",0="off",8:15="on &quotx&quot"}',
            "[DATA]"]
    rows = [[0.0, 0.1, 0.2, 0.3], [1.0, 2.0, 3.0, 4.0], [0.0, 8.0, 7.0, 15.0]]
    with open(path, "wb") as f:
        f.write((newline.join(head) + newline).encode("latin-1"))
        f.write(struct.pack(fmt, len(rows)))
        for row in rows:
            f.write(struct.pack(fmt, len(row)))
            f.write(struct.pack("<%d%s" % (len(row), fmt[1]), *row))
        f.write(b"\r\n")


@pytest.mark.parametrize("precision,newline", [("DOUBLE", "\r\n"), ("SINGLE", "\n")])
def test_d97_variants(tmp_path, precision, newline):
    path = tmp_path / "v.d97"
    _write_d97(path, precision, newline)
    with D97Dataset(path) as ds:
        assert [s.name for s in ds.signals] == ["speed", "bit3", "field", "state"]
        assert ds.read(ds.find("speed")).y.tolist() == [1, 2, 3, 4]
        assert ds.read(ds.find("bit3")).y.tolist() == [0, 1, 0, 1]
        assert ds.read(ds.find("field")).y.tolist() == [0, 0, -1, -1]
        enum = ds.find("state").enum
        assert enum.text(0) == "off" and enum.text(9) == 'on "x"' and enum.text(3) == "other"
        assert ds.read(ds.find("speed")).t == pytest.approx([0, 0.1, 0.2, 0.3])
        assert ds.time_range() == pytest.approx((0.0, 0.3))
    path.unlink()                       # the file is released after close()


def test_d97_truncated_and_foreign_files_give_clear_errors(tmp_path):
    good = tmp_path / "good.d97"
    _write_d97(good)
    cut = tmp_path / "cut.d97"
    cut.write_bytes(good.read_bytes()[:-30])
    with pytest.raises(DatasetError, match="truncated"):
        D97Dataset(cut)
    other = tmp_path / "other.d97"
    other.write_text("hello")
    with pytest.raises(DatasetError, match="DATA"):
        D97Dataset(other)
    empty = tmp_path / "empty.d97"
    empty.write_bytes(b"")
    with pytest.raises(DatasetError):
        D97Dataset(empty)


def test_d97_write_and_read_back(tmp_path):
    class Sig:
        def __init__(self, name, t, y, **kw):
            self.name, self.t, self.y = name, np.asarray(t, float), np.asarray(y, float)
            self.__dict__.update(kw)

    a = Sig("a", [0, 0.01, 0.02], [1.5, 2.5, 3.5], unit="bar", scmin=0.0, scmax=10.0)
    b = Sig("b", [0, 0.01, 0.02], [0, 1, 1], digital=True)
    c = Sig("c", [0, 0.5], [7, 8], unit="V")
    path = tmp_path / "out.d97"
    assert write_d97(path, [a, b, c, Sig("empty", [], [])], description="demo [1]") == ["a", "b", "c"]
    with D97Dataset(path) as ds:
        assert [(s.name, s.kind, s.unit) for s in ds.signals] == [("a", "FLT", "bar"), ("b", "D0", ""), ("c", "FLT", "V")]
        assert ds.read(ds.find("a")).y.tolist() == [1.5, 2.5, 3.5]
        assert ds.read(ds.find("b")).y.tolist() == [0, 1, 1]
        assert ds.read(ds.find("c")).t.tolist() == [0, 0.5]
        assert ds.info()["Description"] == "demo [1]"


def test_enum_grammar():
    assert parse_enum('{0="no action",1="stoplight on",3="signal invalid"}').values == {
        0.0: "no action", 1.0: "stoplight on", 3.0: "signal invalid"}
    table = parse_enum('{[:]="signal valid",0:30="corr&#44 value",31:31="default"}')
    assert table.default == "signal valid" and table.ranges == [(0.0, 30.0, "corr, value"), (31.0, 31.0, "default")]
    assert parse_enum("{}") is None


# ------------------------------------------------------------------ MDF
def test_mf4_statistics_on_own_and_common_rasters(mf4_file, mf4_plt):
    with open_dataset(mf4_file) as ds:
        assert ds.format_name == "MF4 4.10"
        signals = build_display_signals(ds, load_plt(mf4_plt))
        fast, temperature, slow = signals
        assert [sig.name for sig in signals if sig.has_data] == ["BenchAcceleration", "BenchTemperature",
                                                                "BenchEnabled"]
        assert (fast.unit, temperature.unit) == ("m/s²", "°C")
        assert fast.y.tolist() == [0., 2., 4., 6., 8.]
        own = signal_statistics(slow, *ds.time_range())
        shared = signal_statistics(slow, *ds.time_range(), ds.common_time_base())
        assert own["mean"] == pytest.approx(1 / 2)
        assert shared["mean"] == pytest.approx(3 / 5)


def test_mdf3_device_suffix(dat_file):
    with open_dataset(dat_file) as ds:
        assert ds.format_name == "MDF 3.00"
        full = ds.find("BenchTorque\\Sensor:1")
        assert full is not None and full.name.endswith("\\Sensor:1")
        assert ds.find("BenchTorque") is full and ds.find("benchtorque") is full
        assert len(ds.read(full)) == full.sample_count > 0

    class Strip:
        def get(self, key, default=None):
            return True if key == "strip_device_suffix" else default

    with open_dataset(dat_file, Strip()) as ds:
        short = ds.find("BenchTorque")
        assert short.name == "BenchTorque" and short.source_name.endswith("\\Sensor:1")


# ------------------------------------------------------------------ registry
def test_registry_sniffs_content_and_handles_archives(tmp_path, dat_file):
    renamed = tmp_path / "renamed.xyz"
    renamed.write_bytes(dat_file.read_bytes())
    with open_dataset(renamed) as ds:
        assert ds.format_name == "MDF 3.00"
    d97 = tmp_path / "rec.d97"
    _write_d97(d97)
    archive = tmp_path / "pack age.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(d97, "sub/..\\..\\rec.d97")          # hostile member name: must stay inside the cache
        zf.writestr("rec.plt", "speed 2\n")
    with open_dataset(archive) as ds:
        assert [s.name for s in ds.signals][:1] == ["speed"]
        assert ds.archive_path == archive and ds.archive_plt.name == "rec.plt"
        assert ds.path.parent.name != tmp_path.name and "TracePaddock" in str(ds.path)
    with pytest.raises(DatasetError, match="not found"):
        open_dataset(tmp_path / "nope.mf4")
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00" * 100)
    with pytest.raises(DatasetError, match="unknown file type"):
        open_dataset(junk)
    assert d97 in list_datasets_in_directory(d97) and renamed not in list_datasets_in_directory(tmp_path)
