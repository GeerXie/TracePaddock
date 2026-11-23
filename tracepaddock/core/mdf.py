"""Reader for MDF recordings (MDF 2.x/3.x ``.mdf .dat .mf3`` and MDF 4.x ``.mf4``).

The file parsing is done by the *asammdf* library; this module adapts it to
the application's :class:`~tracepaddock.core.model.Dataset` interface: a flat
catalogue of numeric signals, each read on demand with its own time axis.

Naming: a signal keeps the name it was recorded with. Measurement tools such
as INCA or CANape append the device to the name (``v_x\\CCP:1``); scripts can
address such a signal with or without that suffix (see ``Dataset.find``). With
``strip_device_suffix=True`` the catalogue lists the short form instead.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from hashlib import blake2s
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import Dataset, DatasetError, EnumTable, SignalData, SignalInfo

__all__ = ["MdfDataset"]

# asammdf data type codes of channels that are not numbers over time.
_NON_NUMERIC_V4 = {6, 7, 8, 9, 10, 11, 12, 13, 14}   # strings, byte arrays, MIME, CANopen date/time
_NON_NUMERIC_V3 = {7, 8}                              # string, byte array
# Conversion types that only attach texts to numeric values.
_TEXT_TABLE_V4 = {7, 8}        # value to text, value range to text
_TEXT_TABLE_V3 = {11, 12}      # text table, text range table
_TX = re.compile(r"<TX>(.*?)</TX>", re.S)
_CAN_BUS = re.compile(r'<e\s+name="ChannelNo"[^>]*>\s*(\d+)\s*</e>')
_CAN_ID = re.compile(r'<e\s+name="MessageID"[^>]*>\s*(\d+)\s*</e>')


def _plain_comment(comment: object) -> str:
    """Channel comments of MF4 files are XML; return the human readable text."""
    text = str(comment or "").strip()
    if text.startswith("<"):
        m = _TX.search(text)
        text = m.group(1).strip() if m else ""
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return text


def _text(value: object, utf8: bool = False) -> str:
    """Text of a block. MDF 4 stores UTF-8 (falls back to Latin-1 for broken files), MDF 3 Latin-1."""
    if isinstance(value, bytes):
        if utf8:
            try:
                return value.decode("utf-8").strip("\x00 ")
            except UnicodeDecodeError:
                pass
        return value.decode("latin-1", "replace").strip("\x00 ")
    return str(value).strip("\x00 ") if value is not None else ""


def _repair_text(text: str) -> str:
    """Undo a common recording fault in units such as ``m/s²`` and ``°``.

    A recorder running on a Chinese Windows reads the UTF-8 bytes of such
    characters in the GBK code page and stores the result, e.g. ``m/s虏``.
    If turning the text back into GBK bytes gives valid UTF-8, that is what
    was meant. Genuine Chinese text does not survive this test and is kept.
    """
    if text.isascii():
        return text
    try:
        fixed = text.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Accept the repair only if it yields typical unit characters; a genuine
    # Chinese text can also pass the round trip by chance (one character).
    if all(ch.isascii() or "\u0080" <= ch <= "ÿ" or ch in _UNIT_CHARACTERS for ch in fixed):
        return fixed
    return text


_UNIT_CHARACTERS = "Ωμ‰℃€Ωµ"


#: MDF 4 conversion types that turn numbers into numbers (1:1, linear, rational, algebraic, tables).
_NUMERIC_V4 = {0, 1, 2, 3, 4, 5, 6}


def _numeric_default(conv: object) -> Optional[object]:
    """The numeric conversion a value-to-text table applies to all other values, if it has one.

    CAN databases turned into MF4 often store a signal as "8190 = Abnormal,
    8191 = Invalid, anything else: physical = 0.05625 * raw". The texts are
    then a value-to-text table whose default is a linear conversion.
    """
    refs = getattr(conv, "referenced_blocks", None) or {}
    default = refs.get("default_addr")
    if default is not None and getattr(default, "conversion_type", None) in _NUMERIC_V4:
        return default
    return None


def _convert(default: object, values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = default.convert(np.asarray(values))
    return np.asarray(out, dtype=np.float64)


def _enum_to_physical(table: Optional[EnumTable], default: object) -> Optional[EnumTable]:
    """Text table keyed by physical values: the raw keys run through the default conversion."""
    if table is None:
        return None
    try:
        keys = list(table.values)
        values = dict(zip(_convert(default, np.array(keys, dtype=np.float64)).tolist(), table.values.values())) if keys else {}
        ranges = []
        for lo, hi, text in table.ranges:
            a, b = _convert(default, np.array([lo, hi], dtype=np.float64)).tolist()
            ranges.append((min(a, b), max(a, b), text))
    except Exception:
        return table
    return EnumTable(values=values, ranges=ranges, default=table.default)


def _enum_from_conversion(conv: object, version4: bool) -> Optional[EnumTable]:
    """Build an :class:`EnumTable` from a value-to-text conversion block."""
    table = EnumTable()
    refs = getattr(conv, "referenced_blocks", None) or {}
    try:
        ctype = conv.conversion_type
        if version4:
            count = int(getattr(conv, "val_param_nr", 0))
            if ctype == 7:
                for i in range(count):
                    txt = refs.get(f"text_{i}")
                    if isinstance(txt, (bytes, str)):
                        table.values[float(conv[f"val_{i}"])] = _text(txt, utf8=True)
            else:
                for i in range(count // 2):
                    txt = refs.get(f"text_{i}")
                    if isinstance(txt, (bytes, str)):
                        table.ranges.append((float(conv[f"lower_{i}"]), float(conv[f"upper_{i}"]),
                                             _text(txt, utf8=True)))
            default = refs.get("default_addr")
            if isinstance(default, (bytes, str)) and _text(default, utf8=True):
                table.default = _text(default, utf8=True)
        else:
            count = int(getattr(conv, "ref_param_nr", 0))
            if ctype == 11:
                for i in range(count):
                    table.values[float(conv[f"param_val_{i}"])] = _text(conv[f"text_{i}"])
            else:
                for i in range(count):
                    txt = refs.get(f"text_{i}")
                    if isinstance(txt, (bytes, str)):
                        table.ranges.append((float(conv[f"lower_{i}"]), float(conv[f"upper_{i}"]), _text(txt)))
                default = refs.get("default_addr")
                if isinstance(default, (bytes, str)) and _text(default):
                    table.default = _text(default)
    except (KeyError, AttributeError, TypeError, ValueError):
        pass
    return table if table else None


class MdfDataset(Dataset):
    """An MDF / MF4 recording."""

    def __init__(self, path: "str | Path", *, strip_device_suffix: bool = False) -> None:
        super().__init__(path)
        self._mdf = None
        self._masters: Dict[int, np.ndarray] = {}
        self._range: Optional[Tuple[float, float]] = None
        self._common: Optional[np.ndarray] = None
        #: (group, channel) -> numeric conversion applied to the raw values of a text-table channel
        self._defaults: Dict[Tuple[int, int], object] = {}
        try:
            from asammdf import MDF
        except ImportError as exc:  # pragma: no cover - dependency problem
            raise DatasetError("The asammdf package is required to read MDF files") from exc
        # asammdf reports problems twice: on its logger and as an exception.
        # The exception is turned into a message for the user below.
        logging.getLogger("asammdf").setLevel(logging.CRITICAL)
        try:
            self._mdf = MDF(str(self.path))
        except Exception as exc:
            raise DatasetError(f"{self.path.name}: cannot read MDF file ({type(exc).__name__}: {exc})") from exc
        self._v4 = str(self._mdf.version) >= "4"
        self.format_name = ("MF4 " if self._v4 else "MDF ") + str(self._mdf.version)
        try:
            self._build_catalogue(strip_device_suffix)
        except Exception as exc:
            self.close()
            raise DatasetError(f"{self.path.name}: cannot list the signals ({type(exc).__name__}: {exc})") from exc

    def _build_catalogue(self, strip: bool) -> None:
        mdf = self._mdf
        non_numeric = _NON_NUMERIC_V4 if self._v4 else _NON_NUMERIC_V3
        text_tables = _TEXT_TABLE_V4 if self._v4 else _TEXT_TABLE_V3
        seen: Dict[str, int] = {}
        signals: List[SignalInfo] = []
        for gi, group in enumerate(mdf.groups):
            cg = group.channel_group
            cycles = int(getattr(cg, "cycles_nr", 0) or 0)
            raster = _text(getattr(cg, "acq_name", "")) or _plain_comment(getattr(cg, "comment", ""))
            source = getattr(cg, "acq_source", None)
            group_label = _text(getattr(source, "name", "")) if source is not None else ""
            master = mdf.masters_db.get(gi)
            for ci, ch in enumerate(group.channels):
                if ci == master or getattr(ch, "channel_type", 0) in ((2, 3) if self._v4 else (1,)):
                    continue
                if getattr(ch, "data_type", 0) in non_numeric:
                    continue
                conv = ch.conversion
                enum = None
                raw = False
                default = None
                if conv is not None and conv.conversion_type in text_tables:
                    enum = _enum_from_conversion(conv, self._v4)
                    raw = True
                    default = _numeric_default(conv) if self._v4 else None
                    if default is not None:
                        self._defaults[(gi, ci)] = default
                        enum = _enum_to_physical(enum, default)
                unit = _repair_text(_text(getattr(ch, "unit", "")) or (
                    _text(getattr(conv, "unit", "")) if conv is not None else "") or (
                    _text(getattr(default, "unit", "")) if default is not None else ""))
                src_name = ch.name
                name = src_name.split("\\", 1)[0] if strip and "\\" in src_name else src_name
                descr = _plain_comment(getattr(ch, "comment", ""))
                dup = seen.get(src_name, 0)
                seen[src_name] = dup + 1
                if dup:
                    descr = (descr + "\n" if descr else "") + f"[occurrence {dup + 1} of this name, group {gi}: {raster}]"
                signals.append(SignalInfo(
                    name=name, source_name=src_name, unit=unit, description=descr, enum=enum,
                    raster=raster, group=group_label, sample_count=cycles, handle=(gi, ci, raw),
                ))
        self.signals = signals
        if self._v4:
            self._set_channel_origins()
        self._finish_catalogue()

    def _set_channel_origins(self) -> None:
        """Persist physical group and channel choices, plus a portable source when unique."""
        with self.path.open("rb") as stream:
            header = stream.read(4096)
        identity = blake2s(header + str(self.path.stat().st_size).encode(), digest_size=12).hexdigest()
        can_groups: Dict[int, list] = {}
        for bus, ids in getattr(self._mdf, "bus_logging_map", {}).get("CAN", {}).items():
            for identifier, group in ids.items():
                if isinstance(group, int):
                    can_groups.setdefault(group, []).append((int(bus), int(identifier)))
        group_origins = []
        for gi, group in enumerate(self._mdf.groups):
            cg = group.channel_group
            source = getattr(cg, "acq_source", None)
            name = _text(getattr(cg, "acq_name", ""))
            device = _text(getattr(source, "name", ""))
            path = _text(getattr(source, "path", ""))
            buses = can_groups.get(gi, ())
            bus = _CAN_BUS.search(_text(getattr(source, "comment", "")))
            identifier = _CAN_ID.search(_text(getattr(cg, "comment", "")))
            if bus and identifier:
                message = "CAN %d · 0x%X" % (int(bus[1]), int(identifier[1]))
            elif len(buses) == 1:
                message = "CAN %d · 0x%X" % buses[0]
            elif buses and len({channel for channel, _ in buses}) == 1:
                message = "CAN %d" % buses[0][0]
            else:
                message = "CAN %d" % int(bus[1]) if bus else ""
            label = " · ".join(part for part in (f"#{gi}", message, name or device) if part)
            stable = (name, device, path, message)
            identifies_group = name.casefold() not in ("", "can_dataframe") or "0x" in message or device
            key = blake2s(repr(stable).encode(), digest_size=12).hexdigest() if identifies_group else "-"
            group_origins.append((label, key))
        counts = Counter(signal.name.casefold() for signal in self.signals)
        candidates = Counter((signal.source_name.casefold(), group_origins[signal.handle[0]][1])
                             for signal in self.signals)
        for signal in self.signals:
            gi, ci, _ = signal.handle
            label, key = group_origins[gi]
            if candidates[signal.source_name.casefold(), key] > 1:
                key = "-"
            signal.channel_origin = f"{identity}.{gi}.{ci}.{key}"
            signal.origin_label = label
            signal.shared_name = counts[signal.name.casefold()] > 1

    # ------------------------------------------------------------------ reading
    def _master(self, gi: int) -> np.ndarray:
        t = self._masters.get(gi)
        if t is None:
            if int(getattr(self._mdf.groups[gi].channel_group, "cycles_nr", 0) or 0) == 0:
                t = np.empty(0, dtype=np.float64)
            else:
                t = np.asarray(self._mdf.get_master(gi), dtype=np.float64)
            self._masters[gi] = t
        return t

    def read(self, signal: SignalInfo) -> SignalData:
        if self._mdf is None:
            raise DatasetError(f"{self.path.name}: file is closed")
        gi, ci, raw = signal.handle
        if not signal.sample_count:
            empty = np.empty(0, dtype=np.float64)
            return SignalData(empty, empty, signal.enum)
        try:
            with np.errstate(all="ignore"):
                sig = self._mdf.get(group=gi, index=ci, raw=raw)
                samples = sig.samples
                if samples.dtype.kind not in "fiub":
                    # A conversion produced texts: fall back to the stored numbers.
                    sig = self._mdf.get(group=gi, index=ci, raw=True)
                    samples = sig.samples
        except Exception as exc:
            raise DatasetError(f"Signal '{signal.name}' cannot be read ({type(exc).__name__}: {exc})") from exc
        if samples.dtype.kind not in "fiub" or samples.ndim != 1:
            raise DatasetError(f"Signal '{signal.name}' is not a numeric signal over time")
        default = self._defaults.get((gi, ci))
        if default is not None:
            try:
                samples = _convert(default, samples)            # raw -> physical, as the text table's default says
            except Exception as exc:
                raise DatasetError(f"Signal '{signal.name}': conversion failed ({type(exc).__name__}: {exc})") from exc
        return SignalData(np.asarray(sig.timestamps, dtype=np.float64),
                          np.asarray(samples, dtype=np.float64), signal.enum)

    def time_range(self) -> Tuple[float, float]:
        if self._range is None:
            lo, hi = np.inf, -np.inf
            for gi in range(len(self._mdf.groups)):
                t = self._master(gi)
                if len(t):
                    lo, hi = min(lo, float(t[0])), max(hi, float(t[-1]))
            self._range = (lo, hi) if np.isfinite(lo) else (0.0, 0.0)
        return self._range

    def common_time_base(self) -> np.ndarray:
        if self._common is None:
            parts = [self._master(gi) for gi in range(len(self._mdf.groups))]
            parts = [p for p in parts if len(p)]
            self._common = np.unique(np.concatenate(parts)) if parts else np.empty(0)
        return self._common

    def info(self) -> Dict[str, str]:
        mdf = self._mdf
        out = {"File": str(self.path)}
        header = mdf.header
        rasters: Dict[str, List[int]] = {}
        for group in mdf.groups:
            cg = group.channel_group
            label = _text(getattr(cg, "acq_name", "")) or _plain_comment(getattr(cg, "comment", "")) or "(unnamed)"
            entry = rasters.setdefault(label, [0, 0])
            entry[0] += 1
            entry[1] += max(0, len(group.channels) - 1)
        out["Format"] = (f"{self.format_name}, {len(self.signals)} signals in {len(mdf.groups)} groups, "
                         f"recorded {getattr(header, 'start_time', '')}")
        comment = _plain_comment(getattr(header, "comment", ""))
        if comment:
            out["Description"] = comment
        out["Acquisition rasters"] = "\n".join(
            f"{label}: {n_groups} group(s), {n_ch} signal(s)" for label, (n_groups, n_ch) in rasters.items())
        return out

    def close(self) -> None:
        if self._mdf is not None:
            try:
                self._mdf.close()
            except Exception:
                pass
            self._mdf = None
