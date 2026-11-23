"""Automatic view ranges and vertical arrangement of signals.

A signal is positioned purely through its view range: the value ``scmin`` is
drawn at the bottom edge of the analog window and ``scmax`` at the top edge.
Placing a signal into a horizontal band therefore means computing a view
range from its own value range. All functions write the result into the
display signal *and* into its PLT entry, so saving the script keeps it.

Modes:

* :func:`autoscale` - every signal fills the window with its own range.
* :func:`scale_by_unit` - signals with the same unit share one range, so their
  curves can be compared directly.
* :func:`arrange_bands` - every signal gets its own band, top to bottom in
  legend order.
* :func:`arrange_bands_by_unit` - one band per unit; signals of that unit
  share the band and a common scale.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..display import DisplaySignal
from .palette import CYCLIC_RANGE, cyclic_colors

__all__ = ["data_range", "nice_range", "autoscale", "scale_by_unit", "arrange_bands",
           "arrange_bands_by_unit", "apply_cyclic_colors", "place_in_band"]


def data_range(sig: DisplaySignal, t0: Optional[float] = None, t1: Optional[float] = None
               ) -> Optional[Tuple[float, float]]:
    """Minimum and maximum of a signal, optionally only inside ``[t0, t1]``."""
    if not sig.has_data:
        return None
    y = np.asarray(sig.y, dtype=np.float64)
    if t0 is not None and t1 is not None:
        i0 = max(0, int(np.searchsorted(sig.t, t0, side="right")) - 1)
        i1 = max(i0 + 1, int(np.searchsorted(sig.t, t1, side="right")))
        y = y[i0:i1]
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return None
    return float(finite.min()), float(finite.max())


def nice_range(lo: float, hi: float) -> Tuple[float, float]:
    """Widen ``[lo, hi]`` to limits that give a round value per grid division (1, 2, 2.5 or 5 x 10^n)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0, 1.0
    if hi <= lo:
        pad = abs(lo) * 0.5 or 0.5
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / 10.0
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * power
        low = math.floor(lo / step + 1e-9) * step
        if low + 10.0 * step >= hi - 1e-12 * abs(hi):
            return low, low + 10.0 * step
    return lo, hi


def _set(sig: DisplaySignal, lo: float, hi: float) -> None:
    sig.scmin, sig.scmax = float(lo), float(hi)
    sig.entry.scmin, sig.entry.scmax = float(lo), float(hi)
    sig.entry.scansc = False


def _analog(signals: Sequence[DisplaySignal]) -> List[DisplaySignal]:
    return [s for s in signals if s.in_analog and s.has_data]


def autoscale(signals: Sequence[DisplaySignal], *, nice: bool = True,
              t0: Optional[float] = None, t1: Optional[float] = None) -> int:
    """Give every analog signal a view range that fits its values. Returns the number changed."""
    count = 0
    for sig in _analog(signals):
        rng = data_range(sig, t0, t1)
        if rng is None:
            continue
        lo, hi = rng
        if nice:
            lo, hi = nice_range(lo, hi)
        else:
            pad = (hi - lo) * 0.05 or abs(lo) * 0.05 or 0.5
            lo, hi = lo - pad, hi + pad
        _set(sig, lo, hi)
        count += 1
    return count


def _unit_groups(signals: Sequence[DisplaySignal]) -> List[List[DisplaySignal]]:
    """Analog signals grouped by unit, groups in order of first appearance.

    Signals without a unit are not comparable with each other, so each forms
    its own group.
    """
    groups: Dict[str, List[DisplaySignal]] = {}
    order: List[List[DisplaySignal]] = []
    for sig in _analog(signals):
        unit = sig.unit.strip()
        if unit in ("", "-", "1"):
            order.append([sig])
            continue
        if unit not in groups:
            groups[unit] = []
            order.append(groups[unit])
        groups[unit].append(sig)
    return order


def _common_range(group: Sequence[DisplaySignal]) -> Optional[Tuple[float, float]]:
    ranges = [r for r in (data_range(s) for s in group) if r is not None]
    if not ranges:
        return None
    return min(r[0] for r in ranges), max(r[1] for r in ranges)


def scale_by_unit(signals: Sequence[DisplaySignal], *, nice: bool = True) -> int:
    """Signals with the same unit get one common view range. Returns the number of unit groups."""
    groups = _unit_groups(signals)
    for group in groups:
        rng = _common_range(group)
        if rng is None:
            continue
        lo, hi = nice_range(*rng) if nice else rng
        for sig in group:
            _set(sig, lo, hi)
    return len(groups)


def place_in_band(lo: float, hi: float, b0: float, b1: float, margin: float = 0.08) -> Tuple[float, float]:
    """View range that maps the values ``[lo, hi]`` into the band ``[b0, b1]``.

    ``b0`` / ``b1`` are fractions of the window height (0 = bottom, 1 = top);
    ``margin`` is the share of the band left free at its top and bottom.
    """
    if hi <= lo:
        pad = abs(lo) * 0.5 or 0.5
        lo, hi = lo - pad, hi + pad
    inner0 = b0 + (b1 - b0) * margin
    inner1 = b1 - (b1 - b0) * margin
    span = (hi - lo) / max(1e-12, inner1 - inner0)
    scmin = lo - inner0 * span
    return scmin, scmin + span


def arrange_bands(signals: Sequence[DisplaySignal]) -> int:
    """Every analog signal gets its own horizontal band, top to bottom in legend order."""
    analog = _analog(signals)
    n = len(analog)
    for i, sig in enumerate(analog):
        rng = data_range(sig)
        if rng is None:
            continue
        _set(sig, *place_in_band(rng[0], rng[1], 1.0 - (i + 1) / n, 1.0 - i / n))
    return n


def arrange_bands_by_unit(signals: Sequence[DisplaySignal]) -> int:
    """One band per unit; the signals of a unit share the band and one scale."""
    groups = _unit_groups(signals)
    n = len(groups)
    for i, group in enumerate(groups):
        rng = _common_range(group)
        if rng is None:
            continue
        lo, hi = place_in_band(rng[0], rng[1], 1.0 - (i + 1) / n, 1.0 - i / n)
        for sig in group:
            _set(sig, lo, hi)
    return n


def apply_cyclic_colors(signals: Sequence[DisplaySignal], color_range: Tuple[int, int] = CYCLIC_RANGE) -> None:
    """Give the signals consecutive colour numbers from ``color_range`` (inclusive), cycling."""
    for sig, color in zip(signals, cyclic_colors(color_range)):
        sig.color = color
        sig.entry.color = color
