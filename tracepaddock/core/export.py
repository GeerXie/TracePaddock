"""Export of the displayed signals (after all PLT processing) to other files.

* CSV / TXT: one time column and one column per signal. Signals with
  different time stamps are put on one axis: the union of their time stamps
  (each signal holds its last sample) or a fixed step.
* MATLAB ``.mat``: one ``N x 2`` matrix ``[time, value]`` per signal on its own
  samples, plus ``q_KEY`` (names) and ``q_UNIT`` (units); or, with a common
  axis, the vector ``q_T0`` and one column vector per signal.
* MF4 / MDF 3: every signal with its own time stamps; text tables (also the
  user's value mappings) are written as value-to-text conversions in MF4.

All writers take the time window ``[t0, t1]`` and the list of display
signals. They raise :class:`ExportError` with a message for the user.
"""
from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..display import DisplaySignal, format_number

__all__ = ["ExportError", "export_axis", "held_values", "write_csv", "write_mat", "write_mdf",
           "matlab_names", "EXPORT_FORMATS", "estimate_rows"]

#: extension -> label
EXPORT_FORMATS = {
    ".csv": "CSV (Excel)",
    ".txt": "TXT (tab separated)",
    ".mat": "MATLAB (*.mat)",
    ".mf4": "MDF 4.10 (*.mf4)",
    ".mdf": "MDF 3.30 (*.mdf)",
}

Progress = Optional[Callable[[int, int], None]]


class ExportError(Exception):
    """The export could not be written; the message is shown to the user."""


def _window(sig: DisplaySignal, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(sig.t, dtype=np.float64)
    keep = (t >= t0) & (t <= t1)
    return t[keep], np.asarray(sig.y, dtype=np.float64)[keep]


def _usable(signals: Sequence[DisplaySignal]) -> List[DisplaySignal]:
    return [s for s in signals if s.has_data]


def export_axis(signals: Sequence[DisplaySignal], t0: float, t1: float, step: Optional[float] = None) -> np.ndarray:
    """Common time axis for a table export: a fixed ``step`` from ``t0``, or the union of all time stamps."""
    if step is not None:
        if not step > 0:
            raise ExportError("the time step must be greater than 0")
        count = int(math.floor((t1 - t0) / step + 1e-9)) + 1
        if count > 50_000_000:
            raise ExportError("the time step is too small for this time range (more than 50 million rows)")
        return t0 + step * np.arange(max(1, count))
    parts = [_window(s, t0, t1)[0] for s in _usable(signals)]
    if not parts:
        return np.empty(0)
    return np.unique(np.concatenate(parts))


def estimate_rows(signals: Sequence[DisplaySignal], t0: float, t1: float, step: Optional[float] = None) -> int:
    if step is not None and step > 0:
        return int(math.floor((t1 - t0) / step + 1e-9)) + 1
    return int(sum(len(_window(s, t0, t1)[0]) for s in _usable(signals)))


def held_values(sig: DisplaySignal, times: np.ndarray) -> np.ndarray:
    """Value of the step signal at ``times``; NaN before its first sample."""
    t = np.asarray(sig.t, dtype=np.float64)
    y = np.asarray(sig.y, dtype=np.float64)
    if len(t) == 0:
        return np.full(len(times), np.nan)
    idx = np.searchsorted(t, times, side="right") - 1
    out = y[np.clip(idx, 0, len(t) - 1)].copy()
    out[idx < 0] = np.nan
    return out


# ---------------------------------------------------------------------- CSV / TXT
def _cell_writer(sig: DisplaySignal, number_format: str, decimal: str, text_values: str) -> Callable[[float], str]:
    use_text = text_values in ("mapped", "both") and bool(sig.enum)
    try:
        number_format % 1.0
    except (TypeError, ValueError):
        number_format = "%.9g"

    def cell(value: float) -> str:
        if value != value:
            return ""
        if math.isinf(value):                          # e.g. a division by zero in a computed signal
            text = "inf" if value > 0 else "-inf"
        else:
            try:
                text = number_format % value
            except (OverflowError, ValueError, TypeError):
                text = "%.9g" % value
        if decimal != ".":
            text = text.replace(".", decimal)
        if use_text:
            name = sig.enum.text(value)
            if name is not None:
                return name if text_values == "mapped" else "%s (%s)" % (name, text)
        return text
    return cell


def write_csv(path: "str | Path", signals: Sequence[DisplaySignal], t0: float, t1: float, *,
              step: Optional[float] = None, separator: str = ";", decimal: str = ".",
              number_format: str = "%.9g", quote: str = '"', header: bool = True,
              text_values: str = "raw", encoding: str = "utf-8-sig", progress: Progress = None) -> int:
    """Write a table: first column the time in seconds, then one column per signal.

    ``step=None`` uses the union of the time stamps of the exported signals.
    ``text_values``: ``"raw"`` numbers, ``"mapped"`` the text of the text table /
    value mapping instead of the number, ``"both"`` text and number.
    Returns the number of data rows.
    """
    used = _usable(signals)
    if not used:
        raise ExportError("none of the chosen signals has data")
    if separator == decimal:
        raise ExportError("column separator and decimal separator must differ")
    times = export_axis(used, t0, t1, step)
    if len(times) == 0:
        raise ExportError("there are no samples in the chosen time range")
    columns = [held_values(sig, times) for sig in used]
    writers = [_cell_writer(sig, number_format, decimal, text_values) for sig in used]
    time_cell = _cell_writer(DisplaySignal(entry=used[0].entry), "%.9g", decimal, "raw")
    quoting = csv.QUOTE_MINIMAL if quote else csv.QUOTE_NONE
    opened = written = False
    try:
        with open(path, "w", encoding=encoding, newline="") as f:
            opened = True
            writer = csv.writer(f, delimiter=separator, quotechar=quote or '"', quoting=quoting,
                                escapechar=None if quote else "\\", lineterminator="\r\n")
            if header:
                writer.writerow(["TIME"] + [sig.name for sig in used])
                writer.writerow(["s"] + [sig.unit for sig in used])
            total = len(times)
            chunk = 20000
            for start in range(0, total, chunk):
                stop = min(total, start + chunk)
                rows = []
                for k in range(start, stop):
                    rows.append([time_cell(times[k])] + [w(col[k]) for w, col in zip(writers, columns)])
                writer.writerows(rows)
                if progress is not None:
                    progress(stop, total)
        written = True
    except OSError as exc:
        raise ExportError(f"{Path(path).name}: cannot be written ({exc.strerror or exc})") from exc
    except UnicodeEncodeError as exc:
        raise ExportError(f"{Path(path).name}: a name or text cannot be written in {encoding}: {exc.object[exc.start:exc.end]!r}") from exc
    finally:
        if opened and not written:                     # no half written table is left behind
            try:
                os.remove(path)
            except OSError:
                pass
    return len(times)


# ---------------------------------------------------------------------- MATLAB
_NOT_MATLAB = re.compile(r"[^0-9A-Za-z_]")


def matlab_names(names: Sequence[str]) -> List[str]:
    """Form unique MATLAB variable names of at most 63 characters. A first
    character that is not a letter becomes ``x``, other characters outside ``[0-9A-Za-z_]`` become ``_``."""
    out: List[str] = []
    seen = set()
    for name in names:
        base = _NOT_MATLAB.sub("_", name or "x")
        if not base[:1].isalpha() or not base[:1].isascii():
            base = "x" + base[1:] if base else "x"
        base = base[:63]
        candidate, n = base, 2
        while candidate.lower() in seen or candidate in ("q_KEY", "q_UNIT", "q_T0", "q_NAME"):
            suffix = "_%d" % n
            candidate, n = base[:63 - len(suffix)] + suffix, n + 1
        seen.add(candidate.lower())
        out.append(candidate)
    return out


def write_mat(path: "str | Path", signals: Sequence[DisplaySignal], t0: float, t1: float, *,
              common_axis: bool = False, step: Optional[float] = None, progress: Progress = None) -> int:
    """MATLAB 5 file. Returns the number of signals written.

    Own samples (default): ``<name> = [t, y]`` (N x 2) per signal. With
    ``common_axis``: ``q_T0`` (N x 1) and ``<name>`` (N x 1) on that axis.
    ``q_KEY`` / ``q_UNIT`` / ``q_NAME`` are cell arrays with the display
    names, units and the original names of the variables.
    """
    try:
        from scipy.io import savemat
    except ImportError as exc:                                     # pragma: no cover
        raise ExportError("MATLAB export needs scipy") from exc
    used = _usable(signals)
    if not used:
        raise ExportError("none of the chosen signals has data")
    names = matlab_names([s.name for s in used])
    data: Dict[str, object] = {}
    if common_axis:
        times = export_axis(used, t0, t1, step)
        if len(times) == 0:
            raise ExportError("there are no samples in the chosen time range")
        data["q_T0"] = times.reshape(-1, 1)
        for k, (var, sig) in enumerate(zip(names, used)):
            data[var] = held_values(sig, times).reshape(-1, 1)
            if progress is not None:
                progress(k + 1, len(used))
    else:
        for k, (var, sig) in enumerate(zip(names, used)):
            t, y = _window(sig, t0, t1)
            data[var] = np.column_stack((t, y)) if len(t) else np.empty((0, 2))
            if progress is not None:
                progress(k + 1, len(used))
    data["q_KEY"] = np.array([s.name for s in used], dtype=object)
    data["q_UNIT"] = np.array([s.unit for s in used], dtype=object)
    data["q_NAME"] = np.array(names, dtype=object)
    try:
        savemat(str(path), data, do_compression=True, long_field_names=True, oned_as="column")
    except OSError as exc:
        raise ExportError(f"{Path(path).name}: cannot be written ({exc.strerror or exc})") from exc
    return len(used)


# ---------------------------------------------------------------------- MF4 / MDF
def _conversion(sig: DisplaySignal) -> Optional[dict]:
    """Value-to-text conversion of an asammdf channel for the signal's text table (MF4 only)."""
    enum = sig.enum
    if not enum:
        return None
    conv: Dict[str, object] = {}
    if enum.ranges:
        rows = [(v, v, t) for v, t in sorted(enum.values.items())] + [tuple(r) for r in enum.ranges]
        for i, (lo, hi, text) in enumerate(rows):
            conv[f"lower_{i}"], conv[f"upper_{i}"], conv[f"text_{i}"] = float(lo), float(hi), str(text)
    else:
        for i, (value, text) in enumerate(sorted(enum.values.items())):
            conv[f"val_{i}"], conv[f"text_{i}"] = float(value), str(text)
    if not conv:
        return None
    conv["default_addr"] = enum.default or ""
    return conv


def _latin1(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _is_latin1(text: str) -> bool:
    try:
        text.encode("latin-1")
        return True
    except UnicodeEncodeError:
        return False


def _mdf3_names(signals: Sequence[DisplaySignal]) -> List[str]:
    """Channel names for MDF 3, which holds Latin-1 text only. A display name outside Latin-1 (a Chinese
    ``key=``) becomes the recorded name of the signal, else its characters are written as ``_XXXX`` code
    points; names that would occur twice get ``_2``, ``_3`` ..."""
    taken = {sig.name for sig in signals if _is_latin1(sig.name)}
    out: List[str] = []
    for sig in signals:
        if _is_latin1(sig.name):
            out.append(sig.name)
            continue
        candidates = [sig.info.name if sig.info is not None else "", "" if sig.entry.is_placeholder else sig.entry.token]
        base = next((c for c in candidates if c and _is_latin1(c)), "") or \
            "".join(ch if ord(ch) < 256 else "_%04X" % ord(ch) for ch in sig.name)
        name, number = base, 2
        while name in taken:
            name, number = "%s_%d" % (base, number), number + 1
        taken.add(name)
        out.append(name)
    return out


def write_mdf(path: "str | Path", signals: Sequence[DisplaySignal], t0: float, t1: float, *,
              version: str = "4.10", progress: Progress = None, renamed: Optional[Dict[str, str]] = None) -> int:
    """MF4 (``version`` 4.x) or MDF 3 file with every signal on its own samples. Returns the count.

    The file gets exactly the name ``path`` (asammdf itself would change the extension to .mf4 / .mdf).
    MDF 3 holds Latin-1 names only: ``renamed`` (if given) receives ``{display name: name in the file}`` of
    the signals that had to be written under another name."""
    from asammdf import MDF, Signal

    used = _usable(signals)
    if not used:
        raise ExportError("none of the chosen signals has data")
    old = version < "4"
    names = _mdf3_names(used) if old else [sig.name for sig in used]
    out = MDF(version=version)
    groups: Dict[Tuple[int, float, float], List] = {}
    written = 0
    target = Path(path)
    temporary: Optional[str] = None
    try:
        for k, sig in enumerate(used):
            t, y = _window(sig, t0, t1)
            if len(t) == 0:
                continue
            name, unit = names[k], sig.unit
            comment = " ".join(op.text for op in sig.entry.ops)
            conversion = None if old else _conversion(sig)
            if old:
                unit, comment = _latin1(unit), _latin1(comment)
                if name != sig.name and renamed is not None:
                    renamed[sig.name] = name
            item = Signal(samples=np.ascontiguousarray(y), timestamps=np.ascontiguousarray(t), name=name,
                          unit=unit, comment=comment, conversion=conversion)
            groups.setdefault((len(t), float(t[0]), float(t[-1])), []).append(item)
            written += 1
            if progress is not None:
                progress(k + 1, len(used))
        if not written:
            raise ExportError("there are no samples in the chosen time range")
        for items in groups.values():
            out.append(items, comment="TracePaddock export")
        # asammdf gives the file the extension of its version: write a temporary file with that extension
        # next to the target, then give it the chosen name (an existing x.mdf / x.mf4 is not touched)
        handle, temporary = tempfile.mkstemp(prefix=target.stem + ".", suffix=".mdf" if old else ".mf4",
                                             dir=str(target.parent))
        os.close(handle)
        real = Path(out.save(temporary, overwrite=True) or temporary)
        os.replace(str(real), str(target))
        if real == Path(temporary):
            temporary = None
    except ExportError:
        raise
    except OSError as exc:
        raise ExportError(f"{target.name}: cannot be written ({exc.strerror or exc})") from exc
    except Exception as exc:
        raise ExportError(f"{target.name}: export failed ({type(exc).__name__}: {exc})") from exc
    finally:
        out.close()
        if temporary is not None:
            try:
                os.remove(temporary)
            except OSError:
                pass
    return written
