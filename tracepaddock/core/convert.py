"""Convert a whole recording into another file format.

* MDF 3 (``.dat`` / ``.mdf``) -> MF4: a complete conversion that keeps every
  channel with its time stamps, units, text tables and comments.
* The signal-by-signal conversion writes numeric values, time stamps, units
  and descriptions. This path omits text tables and raw bus frames. MDF 3
  output replaces characters outside Latin-1 in channel metadata.

BLF stores raw bus frames. Conversion of measured signals cannot reconstruct
the source frames required to write BLF.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .model import Dataset, DatasetError

__all__ = ["convert_recording", "TARGETS"]

#: Target extension -> (label, MDF version written)
TARGETS = {
    ".mf4": ("MF4 (MDF 4.10)", "4.10"),
    ".mdf": ("MDF 3.30", "3.30"),
    ".dat": ("MDF 3.30 (.dat)", "3.30"),
}

#: Signals are handed to the writer in portions of about this many bytes.
_BATCH_BYTES = 150 * 1024 * 1024


def _latin1(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _signal_by_signal(dataset: Dataset, target: Path, version: str,
                      progress: Optional[Callable[[int, int], None]]) -> int:
    """Write every signal of ``dataset``; signals sharing a time axis form one channel group."""
    from asammdf import MDF, Signal

    old = version < "4"
    out = MDF(version=version)
    written = 0
    total = len(dataset.signals)
    groups: Dict[object, List] = {}
    for sig in dataset.signals:
        # The sample record handle stores the time row second. Channel handles store the group first.
        handle = sig.handle if isinstance(sig.handle, tuple) else (sig.handle,)
        key = handle[1] if len(handle) == 3 and not isinstance(handle[2], bool) else handle[0]
        groups.setdefault(key, []).append(sig)
    done = 0
    for members in groups.values():
        batch: List = []
        size = 0
        reference = None

        def flush() -> None:
            nonlocal batch, size, reference
            if batch:
                out.append(batch, comment=members[0].raster or "")
            batch, size, reference = [], 0, None

        for sig in members:
            done += 1
            if progress is not None and done % 200 == 0:
                progress(done, total)
            try:
                data = dataset.read(sig)
            except DatasetError:
                continue
            if len(data) == 0:
                continue
            if reference is not None and not (len(data.t) == len(reference) and data.t[0] == reference[0]
                                              and data.t[-1] == reference[-1]):
                flush()                                    # another time axis: start a new channel group
            if reference is None:
                reference = data.t
            name = sig.source_name or sig.name
            unit, comment = sig.unit, sig.description
            if old:
                name, unit, comment = _latin1(name), _latin1(unit), _latin1(comment)
            batch.append(Signal(samples=np.ascontiguousarray(data.y), timestamps=np.ascontiguousarray(data.t),
                                name=name, unit=unit, comment=comment))
            written += 1
            size += data.y.nbytes
            if size >= _BATCH_BYTES:
                flush()
        flush()
    if progress is not None:
        progress(total, total)
    if not written:
        raise DatasetError(f"{dataset.path.name}: no signal with data to convert")
    out.save(str(target), overwrite=True)
    out.close()
    return written


def convert_recording(source: "str | Path", target: "str | Path", *,
                      progress: Optional[Callable[[int, int], None]] = None) -> int:
    """Write the recording ``source`` as ``target``; the target extension selects the format.

    Returns the number of signals written. Raises :class:`DatasetError` with a
    message for the user when the conversion is not possible.
    """
    src, dst = Path(source), Path(target)
    ext = dst.suffix.lower()
    if ext == ".blf":
        raise DatasetError(
            "BLF files store raw bus frames, not measured signals. This recording holds signals, "
            "so it cannot be written as BLF. MF4 is the format to use for signals.")
    if ext not in TARGETS:
        raise DatasetError(f"Cannot convert to '{ext}'. Possible targets: " + ", ".join(TARGETS))
    if src.exists() and dst.exists() and src.resolve() == dst.resolve():
        raise DatasetError("Source and target are the same file")
    version = TARGETS[ext][1]
    from .registry import open_dataset

    dataset = open_dataset(src)
    try:
        if dataset.format_name.startswith("MDF ") and version >= "4":
            # Same family, old to new: let the MDF library copy everything.
            from asammdf import MDF

            try:
                with MDF(str(dataset.path)) as mdf:
                    count = sum(max(0, len(g.channels) - 1) for g in mdf.groups)
                    converted = mdf.convert(version)
                    converted.save(str(dst), overwrite=True)
                    converted.close()
                return count
            except Exception:
                pass                                        # fall back to the signal by signal route
        return _signal_by_signal(dataset, dst, version, progress)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"{src.name}: conversion failed ({type(exc).__name__}: {exc})") from exc
    finally:
        dataset.close()
