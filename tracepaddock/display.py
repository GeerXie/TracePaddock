"""What the plot, the legend and the exporters work with: one *display signal*
per enabled PLT entry, i.e. the samples after all PLT attributes were applied.

``tracepaddock.plt.process.build_display_signals`` creates these objects; the UI
only reads them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .core.model import EnumTable, SignalInfo
from .plt.model import PltEntry

__all__ = ["DisplaySignal", "format_number", "value_at", "c_exponent"]

_EXPONENT2 = re.compile(r"([eE][+-])(\d\d)(?!\d)")


def c_exponent(text: str) -> str:
    """Write exponents with three digits (``1e+05`` -> ``1e+005``).

    A field padded with blanks keeps its width.
    """
    fixed, count = _EXPONENT2.subn(r"\g<1>0\2", text)
    if count:
        if fixed.startswith(" "):
            fixed = fixed[1:]
        elif fixed.endswith(" "):
            fixed = fixed[:-1]
    return fixed


def format_number(value: float, fmt: Optional[str] = None, default: str = "%g") -> str:
    """Format ``value`` with a PLT ``fmt=`` specification.

    ``fmt`` is ``[width][.prec]type`` with type ``f e g E G d x b`` (printf
    like; ``b`` = binary digits, precision ignored). ``None`` or an invalid
    specification falls back to ``default`` (a printf format such as ``"%g"``).
    Exponents have three digits. Non finite values give ``"nan"`` / ``"inf"``.
    """
    return c_exponent(_format_number(value, fmt, default))


def _format_number(value: float, fmt: Optional[str], default: str) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
    if fmt:
        spec = fmt.strip()
        if spec.startswith("%"):
            spec = spec[1:]
        kind = spec[-1:] if spec else ""
        body = spec[:-1]
        try:
            if kind in ("f", "e", "g", "E", "G"):
                return ("%" + body + kind) % v
            if kind in ("d", "x", "X"):
                return ("%" + body + kind) % int(round(v))
            if kind == "b":
                width = int(body.split(".")[0]) if body.split(".")[0] else 0
                iv = int(round(v))
                digits = bin(abs(iv))[2:]
                return ("-" if iv < 0 else "") + digits.rjust(width, "0")
        except (ValueError, TypeError):
            pass
    try:
        return default % v
    except (ValueError, TypeError):
        return "%g" % v


def value_at(t: np.ndarray, y: np.ndarray, time: float) -> Optional[float]:
    """Value of a sampled signal at ``time``: the last sample at or before it.

    Before the first sample the first value is returned; ``None`` for an
    empty signal.
    """
    n = len(t)
    if n == 0:
        return None
    i = int(np.searchsorted(t, time, side="right")) - 1
    if i < 0:
        i = 0
    return float(y[i])


@dataclass
class DisplaySignal:
    """One row of the legend / one curve."""

    #: The PLT definition this signal was built from.
    entry: PltEntry
    #: Catalogue entry of the underlying signal; ``None`` for placeholders
    #: (``*``) and for signals that are missing in the dataset.
    info: Optional[SignalInfo] = None
    #: Legend name (``key=`` or the signal name).
    name: str = ""
    unit: str = ""
    #: Palette index.
    color: int = 1
    #: ``True`` = DIGITAL window, ``False`` = analog (FLOAT) window.
    digital: bool = False
    #: Processed samples; ``None`` when the signal could not be built.
    t: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    #: Effective view range: the value ``scmin`` is drawn at the bottom edge
    #: of the analog window and ``scmax`` at the top edge (before Y zoom).
    scmin: float = 0.0
    scmax: float = 1.0
    enum: Optional[EnumTable] = None
    fmt: Optional[str] = None
    #: Horizontal reference lines in signal units.
    ref_lines: List[float] = field(default_factory=list)
    #: 0 none, 1 emphasized, 2 emphasized with focus.
    emphasize: int = 0
    #: ``True`` when value changing attributes were applied (legend mark).
    modified: bool = False
    #: Why the signal has no data (missing in dataset, expression error, ...).
    error: Optional[str] = None
    group: Optional[str] = None
    #: How values with a text table are written: ``"mapped"`` text only, ``"both"`` text with the
    #: number, ``"raw"`` number only. ``None`` = the text table came with the recording (``"both"``).
    value_mode: Optional[str] = None
    #: Name of the user's mapping table in use ("" = none) and the unit of its mapped values.
    map_name: str = ""
    map_unit: str = ""
    #: Shown as a band of labelled states in the upper window instead of a curve.
    lane: bool = False
    #: The user's mapping table of this signal (texts, pictures), ``None`` if there is none.
    value_map: Optional[object] = None

    def image_for(self, value: Optional[float]) -> Optional[str]:
        """Picture spec for a value (see :mod:`tracepaddock.signs`), from the mapping table."""
        if self.value_map is None or value is None:
            return None
        return self.value_map.image_for(value)

    @property
    def has_data(self) -> bool:
        return self.t is not None and self.y is not None and len(self.t) > 0

    @property
    def in_analog(self) -> bool:
        """True when the signal is drawn as a curve in the analog window."""
        return not self.digital and not self.lane

    @property
    def label(self) -> str:
        """Legend text: ``name (unit)``."""
        return f"{self.name} ({self.unit})"

    def value_at(self, time: float) -> Optional[float]:
        if not self.has_data:
            return None
        return value_at(self.t, self.y, time)

    def valid_until(self) -> float:
        """Time from which the signal has no new value (it stopped being sent); infinity if it did not."""
        if not self.has_data:
            return float("inf")
        key = (id(self.t), len(self.t))
        cached = self.__dict__.get("_valid_until")
        if cached is None or cached[0] != key:
            from .core.sampling import valid_until
            cached = (key, valid_until(self.t))
            self.__dict__["_valid_until"] = cached
        return cached[1]

    def stopped_at(self, time: float) -> bool:
        """True when ``time`` lies after the end of a signal that stopped being sent."""
        return time >= self.valid_until()

    def format_value(self, value: Optional[float], default: str = "%g", use_enum: bool = True) -> str:
        """Text for the legend value column (number, or the ENUM text with the number)."""
        if value is None:
            return ""
        text = format_number(value, self.fmt, default)
        mode = self.value_mode or "both"
        if use_enum and self.enum and mode != "raw":
            name = self.enum.text(value)
            if name is not None:
                shown = f"{name} {self.map_unit}" if self.map_unit else name
                return shown if mode == "mapped" else f"{shown} ({text})"
        return text

    def state_text(self, value: float) -> str:
        """Short text of one state for the state band: the mapped text, else the number."""
        if self.enum and self.value_mode != "raw":
            name = self.enum.text(value)
            if name is not None:
                return name
        return format_number(value, self.fmt, "%g")

    def normalized(self) -> Optional[np.ndarray]:
        """Samples mapped so that ``scmin`` -> 0.0 and ``scmax`` -> 1.0."""
        if not self.has_data:
            return None
        span = self.scmax - self.scmin
        if span == 0 or not np.isfinite(span):
            span = 1.0
        return (np.asarray(self.y, dtype=np.float64) - self.scmin) / span

    def denormalize(self, fraction: float) -> float:
        """Inverse of :meth:`normalized` for one value."""
        return self.scmin + fraction * (self.scmax - self.scmin)
