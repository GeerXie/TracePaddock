"""Rebuild the bundled warm-up recordings from invented signals only."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from asammdf import MDF, Signal


DESTINATION = Path(__file__).resolve().parents[1] / "tracepaddock" / "samples" / "warmup"


def channel(name, time, values, *, unit="", labels=()):
    conversion = {}
    for number, label in enumerate(labels):
        conversion[f"val_{number}"] = number
        conversion[f"text_{number}"] = label
    return Signal(np.asarray(values), time, name=name, unit=unit, conversion=conversion or None)


def save(name, signals):
    with MDF(version="4.10") as mdf:
        mdf.start_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        mdf.append(signals, common_timebase=True, comment="Synthetic warm-up signals")
        mdf.save(DESTINATION / f"{name}.mf4", overwrite=True)


def plot_lesson():
    time = np.arange(601) / 50
    target = np.select([time < 2, time < 8], [36., 54.], default=42.)
    accel = 18 * (1 - np.exp(-np.maximum(time - 2.18, 0) / .65))
    decel = 12 * (1 - np.exp(-np.maximum(time - 8.22, 0) / .75))
    speed = 36 + accel - decel + .25 * np.sin(2 * np.pi * .75 * time)
    mode = np.select([time < 2, (time >= 7) & (time < 9)], [0, 2], default=1).astype(np.uint8)
    trim = np.select([time < 2, time < 7, time < 9], [1, 3, 5], default=2).astype(np.uint8)
    packed = (trim << 4) | (mode << 2) | ((mode == 2).astype(np.uint8) << 1) | (mode != 0)
    save("01-plot", [channel("TutorialSpeed", time, speed, unit="km/h"),
                     channel("TutorialTargetSpeed", time, target, unit="km/h"),
                     channel("TutorialDriveMode", time, mode, labels=("Standby", "Tracking", "Limited")),
                     channel("TutorialFlags", time, packed)])


def processing_lesson():
    time = np.arange(1601) / 100
    omega = 2 * np.pi * .17
    speed = 8 + .45 * time + .8 * np.sin(omega * time)
    noisy = speed + .45 * np.sin(2 * np.pi * 9 * time) + .18 * np.sin(2 * np.pi * 18 * time)
    acceleration = .45 + .8 * omega * np.cos(omega * time)
    save("02-processing", [channel("TutorialNoisySpeed", time, noisy, unit="m/s"),
                           channel("TutorialSpeedBaseline", time, speed, unit="m/s"),
                           channel("TutorialAcceleration", time, acceleration, unit="m/s2")])


def timing_lesson():
    time = np.arange(501) / 50
    enabled = ((time >= 1) & (time < 8.5)).astype(np.uint8)
    requested = (((time >= 2) & (time < 2.9)) | ((time >= 6) & (time < 6.9))).astype(np.uint8)
    acknowledged = (((time >= 2.24) & (time < 2.8)) | ((time >= 6.72) & (time < 7.3))).astype(np.uint8)
    status = np.select([acknowledged == 1, (time >= 6.5) & (time < 6.72), requested == 1],
                       [2, 3, 1], default=0).astype(np.uint8)
    travel = np.round(2 * (15 + 10 * np.sin(.85 * time))) / 2
    save("03-timing", [channel("TutorialEnabled", time, enabled),
                       channel("TutorialRequest", time, requested),
                       channel("TutorialAck", time, acknowledged),
                       channel("TutorialAckStatus", time, status),
                       channel("TutorialInputTravel", time, travel, unit="%")])


if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plot_lesson()
    processing_lesson()
    timing_lesson()
