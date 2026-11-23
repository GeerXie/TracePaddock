"""When does a recorded signal stop having a valid value?

A signal that is sampled at a steady rate (a cyclic CAN frame, a measurement
raster) and whose samples stop before the recording ends was no longer sent:
after its last sample it has no new value. Holding the last value there would
show data that does not exist. A signal sent only on change keeps its last value
(an unchanged value is not sent again).

The rule engine treats the time from ``valid_until`` on as "no value"; the plot
draws it dashed (or not at all, or held - a display option).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["LOST_PERIODS", "is_cyclic", "valid_until"]

#: a cyclically sent signal has no value from this many sampling periods after its last sample
LOST_PERIODS = 5.0


def _period(t: np.ndarray) -> Optional[float]:
    """The sampling period of a signal sampled at a steady rate, ``None`` for other signals.

    Steady: at least 10 sampling times and 90 % of the intervals at most 1.5 times the median
    interval; a time stamp that occurs more than once is one sampling time. Two interleaved rasters
    of the same rate are steady as well (a formula of two cyclic signals is computed at the union of
    their time stamps): the time from each sample to the next but one is then the period of the
    rasters (80 % of these times within 25 % of their median)."""
    if t is None or len(t) < 10:
        return None
    t = np.asarray(t, dtype=np.float64)
    dt = np.diff(t)
    if not (dt > 0).all():
        t = t[np.concatenate(([True], dt > 0))]
        dt = np.diff(t)
        if len(t) < 10:
            return None
    median = float(np.median(dt))
    if median > 0 and float(np.percentile(dt, 90)) <= 1.5 * median:
        return median
    pairs = t[2:] - t[:-2]
    median = float(np.median(pairs))
    if median > 0 and 0.75 * median <= float(np.percentile(pairs, 10)) \
            and float(np.percentile(pairs, 90)) <= 1.25 * median:
        return median
    return None


def is_cyclic(t: np.ndarray) -> bool:
    """Sampled at a steady rate (at least 10 sampling times; see :func:`_period`). Signals sent on
    change are not."""
    return _period(t) is not None


def valid_until(t: np.ndarray) -> float:
    """Time from which the last sample is no longer valid: ``LOST_PERIODS`` periods after the last
    sample of a cyclic signal, infinity for other signals."""
    period = _period(t)
    if period is None:
        return float("inf")
    return float(np.asarray(t, dtype=np.float64)[-1]) + LOST_PERIODS * period
