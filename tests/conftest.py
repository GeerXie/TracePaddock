"""Provide isolated Qt and synthetic recording fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"))

if sys.platform == "win32":
    import ctypes

    # Report native test crashes without opening a dialog on the user's desktop.
    _kernel32 = ctypes.windll.kernel32
    _kernel32.SetErrorMode(_kernel32.GetErrorMode() | 0x8003)


@pytest.fixture(scope="session")
def recording_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("recordings")


@pytest.fixture(scope="session", autouse=True)
def isolate_settings(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("APPDATA", str(tmp_path_factory.mktemp("appdata")))
        yield


@pytest.fixture(scope="session")
def d97_file(recording_dir) -> Path:
    from tracepaddock.core.d97 import write_d97
    from tracepaddock.display import DisplaySignal
    from tracepaddock.plt.model import PltEntry

    path = recording_dir / "recording.d97"
    speed = DisplaySignal(PltEntry(name="MotorSpeed"), name="MotorSpeed", t=np.arange(5.0), y=np.arange(5.0) * 2,
                          unit="m/s", scmin=0.0, scmax=10.0)
    ready = DisplaySignal(PltEntry(name="Ready"), name="Ready", t=np.array([0., 2., 4.]), y=np.array([0., 1., 1.]),
                          digital=True)
    write_d97(path, [speed, ready], description="Synthetic ramp")
    return path


@pytest.fixture(scope="session")
def d97_plt(recording_dir) -> Path:
    path = recording_dir / "recording.plt"
    path.write_bytes(b"// Synthetic ramp\r\nMotorSpeed nobit 2 scmin=0 scmax=10\r\nReady bit 3\r\n")
    return path


@pytest.fixture(scope="session")
def mf4_file(recording_dir) -> Path:
    from asammdf import MDF, Signal

    path = recording_dir / "multi-rate.mf4"
    with MDF(version="4.10") as recording:
        times = np.arange(5.0)
        recording.append([
            Signal(samples=np.arange(5.0) * 2, timestamps=times, name="BenchAcceleration", unit="m/s²"),
            Signal(samples=np.arange(5.0) + 20, timestamps=times, name="BenchTemperature", unit="°C"),
        ], common_timebase=True, acq_name="FastGroup")
        recording.append([
            Signal(samples=np.array([0., 1., 1.]), timestamps=np.array([0., 2., 4.]), name="BenchEnabled"),
        ], acq_name="SlowGroup")
        recording.save(path)
    return path


@pytest.fixture(scope="session")
def mf4_plt(recording_dir) -> Path:
    path = recording_dir / "multi-rate.plt"
    path.write_bytes(("// 温度和加速度\nBenchAcceleration nobit 1 unit=m/s² scmin=-2 scmax=10\n"
                      "BenchTemperature nobit 2\nBenchEnabled nobit 3\n").encode("utf-8"))
    return path


@pytest.fixture(scope="session")
def dat_file(recording_dir) -> Path:
    from asammdf import MDF, Signal

    path = recording_dir / "suffixed.dat"
    with MDF(version="3.00") as recording:
        times = np.arange(10.0) / 100
        recording.append([
            Signal(samples=np.arange(10, dtype=float) * 3, timestamps=times, name="BenchTorque\\Sensor:1", unit="Nm"),
            Signal(samples=np.arange(10, dtype=float) + 10, timestamps=times, name="BenchVoltage", unit="V"),
        ], common_timebase=True)
        generated = recording.save(path.with_suffix(".mdf"))
    Path(generated).rename(path)
    return path


@pytest.fixture(scope="session")
def demo_pair(recording_dir) -> tuple[Path, Path]:
    from asammdf import MDF, Signal

    path = recording_dir / "workbench.mf4"
    view = recording_dir / "workbench.plt"
    names = ("Rig_Zone", "Rig_Surface", "Rig_Lighting", "Rig_MarkerType", "RigVelocity", "Rig_MarkerValue",
             "RigSpeedClass")
    times = np.arange(0., 8., .1)
    with MDF(version="4.10") as recording:
        signals = [Signal(samples=np.sin(times) + 15 if name == "RigVelocity" else
                          np.mod(np.arange(len(times)) // (10 + i), i + 2),
                          timestamps=times, name=name, unit="m/s" if name == "RigVelocity" else "-")
                   for i, name in enumerate(names)]
        recording.append(signals, common_timebase=True)
        recording.save(path)
    view.write_text("".join(f"{name} nobit {i + 2}\n" for i, name in enumerate(names)), encoding="utf-8")
    return path, view


@pytest.fixture(scope="session")
def qapp():
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def move_pointer(qapp):
    from PySide6 import QtCore, QtGui

    def move(widget, point):
        event = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(point),
                                  QtCore.QPointF(widget.mapToGlobal(point)), QtCore.Qt.NoButton,
                                  QtCore.Qt.NoButton, QtCore.Qt.NoModifier)
        qapp.sendEvent(widget, event)

    return move
