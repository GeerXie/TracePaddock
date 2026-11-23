"""Data model shared by every dataset reader and by the rest of the application.

A *dataset* is one measurement file. It exposes a catalogue of signals
(:class:`SignalInfo`) and reads the samples of one signal on demand
(:class:`SignalData`). Readers never load the whole file up front: a recording
can hold tens of thousands of signals and several gigabytes.
"""
from __future__ import annotations

import abc
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DatasetError",
    "EnumTable",
    "SignalInfo",
    "SignalData",
    "Dataset",
    "KIND_FLOAT",
    "KIND_BIT",
    "KIND_BITFIELD",
    "KIND_BITFIELD_SIGNED",
]

#: Standard signal (analog value).
KIND_FLOAT = "FLT"
#: Digital signal spanning one bit.
KIND_BIT = "D0"
#: Unsigned bit field.
KIND_BITFIELD = "DN"
#: Signed bit field.
KIND_BITFIELD_SIGNED = "DS"


class DatasetError(Exception):
    """A dataset could not be opened or a signal could not be read.

    The message is shown to the user as is, so it must say what was wrong
    with which file or signal.
    """


@dataclass
class EnumTable:
    """Text labels for individual signal values and ranges.

    ``values`` maps a single value to a text, ``ranges`` maps the closed
    interval ``lo <= v <= hi`` to a text, ``default`` is used when nothing
    else matches (``None`` = show the number).
    """

    values: Dict[float, str] = field(default_factory=dict)
    ranges: List[Tuple[float, float, str]] = field(default_factory=list)
    default: Optional[str] = None

    def text(self, value: float) -> Optional[str]:
        """Return the replacement text for ``value`` or ``None`` if there is none."""
        try:
            hit = self.values.get(float(value))
        except (TypeError, ValueError):
            return self.default
        if hit is not None:
            return hit
        for lo, hi, txt in self.ranges:
            if lo <= value <= hi:
                return txt
        return self.default

    def __bool__(self) -> bool:
        return bool(self.values or self.ranges or self.default is not None)


@dataclass
class SignalInfo:
    """Catalogue entry of one signal. Cheap to create; holds no samples."""

    #: Name used to address the signal from a PLT script. Readers normalise
    #: source specific decoration here (e.g. the ``\\CCP:1`` device suffix of
    #: INCA/CANape MDF3 files is removed).
    name: str
    #: Name exactly as stored in the file.
    source_name: str = ""
    unit: str = ""
    description: str = ""
    #: One of ``KIND_FLOAT``, ``KIND_BIT``, ``KIND_BITFIELD``, ``KIND_BITFIELD_SIGNED``.
    kind: str = KIND_FLOAT
    bit_pos: int = 0
    bit_count: int = 0
    #: Default display range stored in the file (``None`` = not stored).
    scmin: Optional[float] = None
    scmax: Optional[float] = None
    #: Default colour index stored in the file (``None`` = not stored).
    color: Optional[int] = None
    enum: Optional[EnumTable] = None
    #: Human readable label of the time base / acquisition raster, e.g. ``"10 ms"``.
    raster: str = ""
    #: Label of the source / group / device the signal came from.
    group: str = ""
    #: Number of samples if known without reading the data, else ``None``.
    sample_count: Optional[int] = None
    #: Reader private handle (row index, (group, channel) pair, ...).
    handle: object = None
    #: Position in ``Dataset.signals``.
    index: int = -1
    channel_origin: str = ""
    origin_label: str = ""
    shared_name: bool = False

    @property
    def is_digital(self) -> bool:
        """True for signals that belong into the DIGITAL window by default."""
        return self.kind == KIND_BIT


@dataclass
class SignalData:
    """Samples of one signal in physical units.

    ``t`` and ``y`` are one dimensional ``float64`` arrays of equal length;
    ``t`` is in seconds and non decreasing. They may be read-only views into
    a memory mapped file, so callers must copy before modifying in place.
    """

    t: np.ndarray
    y: np.ndarray
    enum: Optional[EnumTable] = None

    def __post_init__(self) -> None:
        if self.t.shape != self.y.shape:
            raise ValueError(f"time/value length mismatch: {self.t.shape} vs {self.y.shape}")

    def __len__(self) -> int:
        return int(self.t.shape[0])


def _strip_device_suffix(name: str) -> str:
    """``'v_x\\\\CCP:1'`` -> ``'v_x'`` (INCA / CANape style ``name\\device``)."""
    cut = name.find("\\")
    return name if cut < 0 else name[:cut]


class Dataset(abc.ABC):
    """One opened measurement file.

    Sub-classes fill ``self.signals`` in ``__init__`` (catalogue only) and
    implement :meth:`read`. They must be safe to use from one thread only and
    must release file handles in :meth:`close`.
    """

    #: Short label for the actual file format shown in the UI.
    format_name: str = ""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self.signals: List[SignalInfo] = []
        self._by_name: Optional[Dict[str, SignalInfo]] = None
        self._by_lower: Optional[Dict[str, SignalInfo]] = None
        self._by_stripped: Optional[Dict[str, SignalInfo]] = None

    # ------------------------------------------------------------------ catalogue
    def _finish_catalogue(self) -> None:
        """Call at the end of ``__init__`` after ``self.signals`` is complete."""
        for i, sig in enumerate(self.signals):
            sig.index = i
            if not sig.source_name:
                sig.source_name = sig.name
        self._by_name = self._by_lower = self._by_stripped = None

    def _build_lookup(self) -> None:
        by_name: Dict[str, SignalInfo] = {}
        by_lower: Dict[str, SignalInfo] = {}
        by_stripped: Dict[str, SignalInfo] = {}
        for sig in self.signals:
            # Unqualified names select the first occurrence.
            by_name.setdefault(sig.name, sig)
            by_name.setdefault(sig.source_name, sig)
            by_lower.setdefault(sig.name.lower(), sig)
            by_lower.setdefault(sig.source_name.lower(), sig)
            by_stripped.setdefault(_strip_device_suffix(sig.source_name).lower(), sig)
        self._by_name, self._by_lower, self._by_stripped = by_name, by_lower, by_stripped

    def find(self, name: str, origin: str = "") -> Optional[SignalInfo]:
        """Look up a signal by the name written in a PLT script.

        Order: exact match, case-insensitive match, match ignoring a
        ``\\device`` suffix on either side. Returns ``None`` if absent.
        """
        if origin:
            candidates = self.source_choices(name)
            exact = next((signal for signal in candidates if signal.channel_origin == origin), None)
            if exact is not None:
                return exact
            key = origin.rsplit(".", 1)[-1]
            if key != "-":
                matches = [signal for signal in candidates if signal.channel_origin.rsplit(".", 1)[-1] == key]
                if len(matches) == 1:
                    return matches[0]
            return None
        if self._by_name is None:
            self._build_lookup()
        assert self._by_name is not None and self._by_lower is not None and self._by_stripped is not None
        hit = self._by_name.get(name)
        if hit is None:
            hit = self._by_lower.get(name.lower())
        if hit is None:
            hit = self._by_stripped.get(_strip_device_suffix(name).lower())
        return hit

    def source_choices(self, name: str) -> List[SignalInfo]:
        folded = name.casefold()
        return [signal for signal in self.signals if signal.channel_origin
                and folded in (signal.name.casefold(), signal.source_name.casefold())]

    def filter(self, pattern: str, *, regex: bool = False, case_sensitive: bool = False) -> List[SignalInfo]:
        """Signals whose name matches ``pattern``.

        ``regex=False``: wildcard pattern with ``*`` (any run of characters)
        and ``?`` (exactly one character), matched against the whole name.
        ``regex=True``: regular expression, matched against the whole name.
        An empty pattern or ``*`` returns all signals. An invalid regular
        expression raises ``re.error``.
        """
        if pattern in ("", "*") and not regex:
            return list(self.signals)
        flags = 0 if case_sensitive else re.IGNORECASE
        rx = re.compile(pattern if regex else fnmatch.translate(pattern), flags)
        match = rx.fullmatch if regex else rx.match
        return [s for s in self.signals if match(s.name)]

    # ------------------------------------------------------------------ data
    @abc.abstractmethod
    def read(self, signal: SignalInfo) -> SignalData:
        """Read all samples of ``signal`` (physical values, own time axis).

        Bit signals are already extracted (0/1, or the bit-field value).
        Raises :class:`DatasetError` if the signal cannot be read.
        """

    def read_by_name(self, name: str) -> SignalData:
        sig = self.find(name)
        if sig is None:
            raise DatasetError(f"Signal '{name}' is not contained in {self.path.name}")
        return self.read(sig)

    @abc.abstractmethod
    def time_range(self) -> Tuple[float, float]:
        """``(t_min, t_max)`` in seconds over the whole recording (cheap)."""

    def common_time_base(self) -> np.ndarray:
        """Sorted union of the time stamps of *all* time bases in the recording.

        Common-time-base statistics evaluate each signal at these timestamps
        by holding the last recorded sample. An empty base uses each
        signal's own samples instead.
        """
        return np.empty(0, dtype=np.float64)

    def info(self) -> "Dict[str, str]":
        """Texts for the *dataset info* dialog: section title -> multi line text.

        The first entries should be ``"File"`` and ``"Description"``.
        """
        return {"File": str(self.path)}

    def close(self) -> None:
        """Release file handles / memory maps. Safe to call twice."""

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.signals)


def unique_names(signals: Iterable[SignalInfo]) -> Sequence[str]:
    """Names in catalogue order without duplicates (first occurrence kept)."""
    seen = set()
    out = []
    for sig in signals:
        if sig.name not in seen:
            seen.add(sig.name)
            out.append(sig.name)
    return out
