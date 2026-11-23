"""Reader and writer for recordings with a text header and sample records.

File structure::

    ; comment lines
    [CONTROL]        FLOAT=INTEL387  PRECISION=DOUBLE|SINGLE  DATAMATRIX=SV_BY_*  [TC=..]
    [DESCRIPTION]    free text
    [INFO<n>]        free text
    [SIGNAL<n>]      NAME= DESCR= UNIT= COLOR= TYP=FLT|D0|DN|DS DPOS= TMSV=|TPOS= SCMIN= SCMAX= BITP= BITC= RPT= ENUM={..}
    [DATA]
    <n> then n records, each: <m> followed by m values     (binary little endian, or formatted text)

``TMSV`` and ``TPOS`` can identify the data record holding a signal's time
stamps. One-bit signals (``TYP=D0``) can share a data record and use different
``BITP`` offsets. A text table can span several ``ENUM=`` lines. Header text
may use ISO-8859-1 or the system's CJK code page.

Opening a binary recording parses the header and indexes the data records.
Samples are copied from the memory map on demand so closing the reader can
release the file.
"""
from __future__ import annotations

import codecs
import locale
import math
import mmap
import re
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .model import (
    KIND_BIT, KIND_BITFIELD, KIND_BITFIELD_SIGNED, KIND_FLOAT,
    Dataset, DatasetError, EnumTable, SignalData, SignalInfo,
)

__all__ = ["D97Dataset", "write_d97", "parse_enum", "INFO_LINE_LIMIT"]

_INFO_TITLES = {
    0: "Created",
    1: "Configuration files",
    2: "Tool configuration",
    3: "Tools used",
    4: "Change history",
    5: "Identifier",
    6: "Recorded data traces",
    7: "Measurement hardware",
    8: "Data sources",
    9: "GPS",
}

#: Limit on lines shown from an [INFO<n>] section.
INFO_LINE_LIMIT = 2047

#: ENUM text escapes omit the semicolon. Decode ``&amp`` last.
_ENUM_ENTITIES = (("&quot", '"'), ("&#44", ","), ("&amp", "&"))
#: Bracket escapes for [DESCRIPTION] and [INFO<n>] lines.
_TEXT_ENTITIES = (("&#91", "["), ("&#93", "]"))
_VALUE_BLANKS = " \t\r\n"


def _cjk_codec() -> Optional[str]:
    """Multi-byte code page of this PC for texts that are not ISO-8859-1, or ``None``."""
    try:
        name = codecs.lookup(locale.getpreferredencoding(False) or "latin-1").name
    except LookupError:
        return None
    if name in ("gbk", "gb2312", "gb18030", "cp936"):
        # Only the GB2312 part: its byte pairs (both bytes >= 0xA1) are rarely
        # formed by Latin-1 text, unlike the GBK extensions ("f\xfcr" = "für").
        return "gb2312"
    if name in ("cp932", "shift_jis", "cp949", "euc_kr", "cp950", "big5"):
        return name
    return None


#: Optional local CJK encoding for files that mix encodings across header fields.
TEXT_CODEC = _cjk_codec()


def _decode_text(text: str) -> str:
    """Header bytes are split as Latin-1; a value that is valid in the PC's CJK code page is re-read in it."""
    if text.isascii() or TEXT_CODEC is None:
        return text
    try:
        return text.encode("latin-1").decode(TEXT_CODEC)
    except (UnicodeError, LookupError):
        return text


def _replace_all(text: str, table: Sequence[Tuple[str, str]]) -> str:
    if "&" not in text:
        return text
    for src, dst in table:
        text = text.replace(src, dst)
    return text


def _parse_enum_parts(spec: str, table: EnumTable) -> None:
    s = spec.strip()
    if s.startswith("{"):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]
    s = s.replace("“", '"').replace("”", '"').replace("„", '"')
    for m in re.finditer(r'\s*([^=,"]*)=\s*"([^"]*)"\s*(?:,|$)', s):
        key, text = m.group(1).strip(), _replace_all(m.group(2), _ENUM_ENTITIES)
        if key in ("", "*", "[:]", "*:*", ":"):
            table.default = text
            continue
        try:
            if ":" in key:
                lo, hi = key.split(":", 1)
                table.ranges.append((float(lo), float(hi), text))
            else:
                table.values[float(key)] = text
        except ValueError:
            continue


def parse_enum(*specs: str) -> Optional[EnumTable]:
    """Parse ``{0="a",1="b"}`` / ``{0:30="a",31:31="b"}`` text replacements.

    Multiple ``ENUM=`` lines are merged into one table. The first element may define a
    default text: ``="x"`` or ``*="x"`` for the value form, ``[:]="x"`` or
    ``*:*="x"`` for the range form. Typographic quotes are accepted too.
    """
    table = EnumTable()
    for spec in specs:
        _parse_enum_parts(spec, table)
    return table if table else None


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None or text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _to_int(text: Optional[str], default: int) -> int:
    try:
        return int(float(text)) if text not in (None, "") else default
    except (ValueError, OverflowError):
        return default


def _is_int(text: Optional[str]) -> bool:
    return bool(text) and re.fullmatch(r"\s*-?\d+\s*", text) is not None


def _first_backstep(t: np.ndarray) -> int:
    """Length of the leading part of a time record that does not run backwards."""
    if len(t) < 2:
        return len(t)
    back = np.flatnonzero(np.diff(t) < 0)
    return int(back[0]) + 1 if len(back) else len(t)


class D97Dataset(Dataset):
    """A recording whose binary samples are read on demand from a memory map."""

    format_name = "D97"

    def __init__(self, path: "str | Path") -> None:
        super().__init__(path)
        self._file = None
        self._mm: Optional[mmap.mmap] = None
        self._rows: List[Tuple[int, int]] = []       # (byte offset, value count) per data record
        self._ascii_rows: Optional[List[np.ndarray]] = None
        self._dtype = np.dtype("<f8")
        self._control: Dict[str, str] = {}
        self._comments: List[str] = []
        self._description = ""
        self._info: Dict[int, str] = {}
        self._tc = 0.0
        self._time_rows: List[int] = []
        self._fixed_axes: List[Tuple[int, float]] = []    # (sample count, sample time) of fixed-rate signals
        self._time_cache: Dict[int, np.ndarray] = {}
        self._common: Optional[np.ndarray] = None
        self._range: Optional[Tuple[float, float]] = None
        try:
            self._open()
        except DatasetError:
            self.close()
            raise
        except (OSError, ValueError, struct.error, IndexError, ArithmeticError, MemoryError) as exc:
            self.close()
            raise DatasetError(f"{self.path.name}: cannot read recording ({exc})") from exc

    # ------------------------------------------------------------------ opening
    def _open(self) -> None:
        self._file = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as exc:
            raise DatasetError(f"{self.path.name}: file is empty") from exc
        mm = self._mm
        m = re.search(rb"(?im)^\[DATA\][ \t]*\r?\n?", mm)
        if m is None:
            raise DatasetError(f"{self.path.name}: missing [DATA] section in a truncated or invalid recording")
        self._parse_header(mm[:m.start()].decode("latin-1"))
        self._index_data(m.end())
        self._build_catalogue()

    def _parse_header(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        chunks = ("\n" + text).split("\n[")
        self._comments = [_decode_text(ln) for ln in chunks[0].split("\n") if ln.startswith(";")]
        self._signal_chunks: List[Tuple[int, str]] = []
        for chunk in chunks[1:]:
            end = chunk.find("]")
            if end < 0:
                continue
            title = chunk[:end].strip().upper()
            body = chunk[end + 1:].lstrip("\n")
            if title.startswith("SIGNAL") and title[6:].isdigit():
                self._signal_chunks.append((int(title[6:]), body))
            elif title == "CONTROL":
                for line in body.split("\n"):
                    key, sep, value = line.partition("=")
                    if sep:
                        self._control[key.strip().upper()] = value.strip(_VALUE_BLANKS)
            elif title == "DESCRIPTION":
                self._description = _decode_text(_replace_all(body.strip("\n"), _TEXT_ENTITIES))
            elif title.startswith("INFO") and title[4:].isdigit():
                self._info[int(title[4:])] = body.strip("\n")
        ctl = self._control
        coding = (ctl.get("FLOAT") or ctl.get("FORMAT") or "INTEL387").upper()
        self._binary = coding in ("INTEL387", "BINARY")
        if not self._binary and "FORMATTED" not in coding:
            raise DatasetError(f"{self.path.name}: unknown data coding FLOAT={coding}")
        layout = (ctl.get("DATAMATRIX") or ctl.get("LAYOUT") or "SV_BY_*").upper()
        if layout.startswith("FRAME"):
            raise DatasetError(
                f"{self.path.name}: data layout {layout} (frame by frame) is not supported yet; "
                "only signal-by-signal recordings (SV_BY_*) can be read")
        single = ctl.get("PRECISION", "SINGLE").upper() == "SINGLE"
        self._dtype = np.dtype("<f4" if single else "<f8")
        self._tc = _to_float(ctl.get("TC")) or 0.0

    def _index_data(self, pos: int) -> None:
        mm = self._mm
        assert mm is not None
        size = len(mm)
        if not self._binary:
            self._index_ascii(pos)
            return
        width = self._dtype.itemsize
        fmt = "<f" if width == 4 else "<d"
        if pos + width > size:
            raise DatasetError(f"{self.path.name}: [DATA] section is empty")
        count = struct.unpack_from(fmt, mm, pos)[0]
        pos += width
        if not math.isfinite(count) or not (0 <= count < 1e9) or count != int(count):
            raise DatasetError(f"{self.path.name}: invalid record count {count!r} in [DATA]")
        rows: List[Tuple[int, int]] = []
        unpack = struct.Struct(fmt).unpack_from
        for i in range(int(count)):
            if pos + width > size:
                raise DatasetError(
                    f"{self.path.name}: file is truncated - data record {i} of {int(count)} is missing")
            m = unpack(mm, pos)[0]
            pos += width
            if not math.isfinite(m) or m < 0 or m != int(m) or pos + int(m) * width > size:
                raise DatasetError(
                    f"{self.path.name}: file is truncated or damaged in data record {i} (length {m!r})")
            n = int(m)
            rows.append((pos, n))
            pos += n * width
        self._rows = rows

    def _index_ascii(self, pos: int) -> None:
        mm = self._mm
        assert mm is not None
        end_tag = re.compile(rb"\[/DATA\]", re.I).search(mm, pos)
        tokens = mm[pos:end_tag.start() if end_tag else len(mm)].split()
        try:
            values = np.array(tokens, dtype=np.float64)
        except ValueError as exc:
            raise DatasetError(f"{self.path.name}: formatted [DATA] section contains non numeric text") from exc
        if len(values) == 0:
            raise DatasetError(f"{self.path.name}: [DATA] section is empty")
        if not math.isfinite(values[0]) or values[0] < 0:
            raise DatasetError(f"{self.path.name}: invalid record count in [DATA]")
        count, p = int(values[0]), 1
        rows: List[np.ndarray] = []
        for i in range(count):
            if p >= len(values):
                raise DatasetError(f"{self.path.name}: file is truncated - data record {i} of {count} is missing")
            if not math.isfinite(values[p]) or values[p] < 0:
                raise DatasetError(f"{self.path.name}: damaged length of data record {i}")
            n = int(values[p])
            if p + 1 + n > len(values):
                raise DatasetError(f"{self.path.name}: file is truncated in data record {i}")
            rows.append(values[p + 1:p + 1 + n])
            p += 1 + n
        self._ascii_rows = rows
        self._rows = [(0, len(r)) for r in rows]

    def _build_catalogue(self) -> None:
        rows = self._rows
        nrows = len(rows)
        ctl = self._control
        # [CONTROL] can identify the common time record by row number or signal name.
        control_row = -1
        for key in ("TMSV", "TPOS"):
            if _is_int(ctl.get(key)):
                control_row = int(ctl[key])
                break
        control_name = ctl.get("TMSV", "") if not _is_int(ctl.get("TMSV")) else ""
        time_rows = set()
        fixed = set()
        signals: List[SignalInfo] = []
        pending_named_time: List[SignalInfo] = []
        for number, body in self._signal_chunks:
            attr: Dict[str, str] = {}
            enum_parts: List[str] = []
            for line in body.split("\n"):
                key, sep, value = line.partition("=")
                if not sep:
                    continue
                key = key.strip().upper()
                value = value.strip(_VALUE_BLANKS)
                if key == "ENUM":
                    enum_parts.append(value)
                else:
                    attr[key] = value
            name = _decode_text(attr.get("NAME", ""))
            if not name:
                continue
            typ = attr.get("TYP", attr.get("TYPE", "FLT")).upper()
            kind = {"D0": KIND_BIT, "DN": KIND_BITFIELD, "DS": KIND_BITFIELD_SIGNED}.get(typ, KIND_FLOAT)
            dpos_text = attr.get("DPOS")
            dpos = number if dpos_text in (None, "") else _to_int(dpos_text, -1)   # damaged: unreadable
            tpos = _to_int(attr.get("TMSV", attr.get("TPOS")), -1)
            tc = _to_float(attr.get("TC")) or 0.0
            rpt = max(1, _to_int(attr.get("RPT"), 1))
            step = tc if tc else self._tc * rpt
            if tpos < 0 and not step and control_row >= 0:
                tpos = control_row
            count = rows[dpos][1] if 0 <= dpos < nrows else 0
            if tpos >= 0:
                time_rows.add(tpos)
            elif step:
                fixed.add((count, step))
            enum = parse_enum(*(_decode_text(p) for p in enum_parts)) if enum_parts else None
            descr = _replace_all(_decode_text(attr.get("DESCR", attr.get("DESCRIPTION", ""))).replace("\\n", "\n"),
                                 _ENUM_ENTITIES)
            sig = SignalInfo(
                name=name,
                unit=_decode_text(attr.get("UNIT", "")),
                description=descr,
                kind=kind,
                bit_pos=_to_int(attr.get("BITP"), 0),
                bit_count=1 if kind == KIND_BIT else _to_int(attr.get("BITC"), 0),
                scmin=_to_float(attr.get("SCMIN")),
                scmax=_to_float(attr.get("SCMAX")),
                color=_to_int(attr.get("COLOR"), 1) if "COLOR" in attr else None,
                enum=enum,
                sample_count=count,
                handle=(dpos, tpos, step),
            )
            signals.append(sig)
            if tpos < 0 and not step and control_name:
                pending_named_time.append(sig)
        del self._signal_chunks
        if pending_named_time:
            carrier = next((s for s in signals if s.name == control_name), None)
            if carrier is not None:
                row = carrier.handle[0]
                time_rows.add(row)
                for sig in pending_named_time:
                    sig.handle = (sig.handle[0], row, 0.0)
        self._time_rows = sorted(r for r in time_rows if 0 <= r < nrows)
        self._fixed_axes = sorted(fixed)
        labels = {r: self._raster_label(r) for r in self._time_rows}
        for sig in signals:
            dpos, tpos, step = sig.handle
            sig.raster = labels.get(tpos, f"{step * 1000:g} ms" if step else "")
        self.signals = signals
        self._finish_catalogue()

    def _raster_label(self, row: int) -> str:
        t = self._time(row)
        if len(t) < 2:
            return ""
        head = t[: min(len(t), 2001)]
        return f"{float(np.median(np.diff(head))) * 1000:.4g} ms"

    # ------------------------------------------------------------------ reading
    def _view(self, index: int) -> np.ndarray:
        """Values of a data record as stored (a temporary view; copy before keeping it)."""
        if not 0 <= index < len(self._rows):
            raise DatasetError(f"{self.path.name}: data record {index} does not exist")
        if self._ascii_rows is not None:
            return self._ascii_rows[index]
        if self._mm is None:
            raise DatasetError(f"{self.path.name}: file is closed")
        offset, count = self._rows[index]
        return np.frombuffer(self._mm, dtype=self._dtype, count=count, offset=offset)

    def _time(self, index: int) -> np.ndarray:
        """A time record as float64, cut where it runs backwards (inconsistent time information)."""
        t = self._time_cache.get(index)
        if t is None:
            t = np.array(self._view(index), dtype=np.float64)
            t = t[:_first_backstep(t)]
            t.setflags(write=False)                  # shared by every signal of this raster
            self._time_cache[index] = t
        return t

    def _check_open(self) -> None:
        if self._mm is None and self._ascii_rows is None:
            raise DatasetError(f"{self.path.name}: file is closed")

    def read(self, signal: SignalInfo) -> SignalData:
        self._check_open()
        dpos, tpos, step = signal.handle
        raw = np.array(self._view(dpos), dtype=np.float64)          # a copy: the file stays free
        if signal.kind == KIND_FLOAT:
            y = raw
        else:
            bits = np.nan_to_num(raw).astype(np.int64)
            if signal.kind == KIND_BIT:
                y = ((bits >> signal.bit_pos) & 1).astype(np.float64)
            else:
                width = max(1, signal.bit_count)
                field = (bits >> signal.bit_pos) & ((1 << width) - 1)
                if signal.kind == KIND_BITFIELD_SIGNED:
                    field = np.where(field >= (1 << (width - 1)), field - (1 << width), field)
                y = field.astype(np.float64)
        if tpos >= 0:
            t = self._time(tpos)
        elif step:
            t = np.arange(len(y), dtype=np.float64) * step
        else:
            raise DatasetError(f"Signal '{signal.name}' has neither a time record nor a sample time")
        n = min(len(t), len(y))
        return SignalData(t[:n], y[:n], signal.enum)

    def time_range(self) -> Tuple[float, float]:
        self._check_open()
        if self._range is None:
            lo, hi = np.inf, -np.inf
            for r in self._time_rows:
                t = self._time(r)
                if len(t):
                    lo, hi = min(lo, float(t[0])), max(hi, float(t[-1]))
            for count, step in self._fixed_axes:
                if count:
                    lo, hi = min(lo, 0.0), max(hi, (count - 1) * step)
            self._range = (lo, hi) if np.isfinite(lo) else (0.0, 0.0)
        return self._range

    def common_time_base(self) -> np.ndarray:
        if self._common is None:
            self._check_open()
            parts = [self._time(r) for r in self._time_rows]
            parts += [np.arange(count, dtype=np.float64) * step for count, step in self._fixed_axes if count]
            self._common = np.unique(np.concatenate(parts)) if parts else np.empty(0)
        return self._common

    def info(self) -> Dict[str, str]:
        out = {"File": str(self.path)}
        out["Format"] = (f"D97, {len(self.signals)} signals, {len(self._rows)} data records, "
                         f"{len(self._time_rows) + len(self._fixed_axes)} time bases, "
                         f"{self._dtype.itemsize * 8} bit values")
        if self._description.strip():
            out["Description"] = self._description
        if self._comments:
            out["Recorder comments"] = "\n".join(self._comments)
        for n in sorted(self._info):
            text = self._info[n]
            if not text.strip():
                continue
            lines = text.split("\n")
            if len(lines) > INFO_LINE_LIMIT:
                hidden = len(lines) - INFO_LINE_LIMIT + 1
                lines = lines[:INFO_LINE_LIMIT - 1] + [f"... ({hidden} more lines not shown)"]
            shown = _decode_text(_replace_all("\n".join(lines), _TEXT_ENTITIES))
            out[f"INFO{n} - {_INFO_TITLES.get(n, 'Information')}"] = shown
        return out

    def close(self) -> None:
        self._ascii_rows = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                pass        # a view handed out by an older caller is still alive; it keeps the map
            self._mm = None
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------- writing
_TRANSLITERATION = {"²": "^2", "³": "^3", "°": "deg", "µ": "u", "μ": "u",
                    "Ω": "Ohm", "℃": "degC", "‰": "permille", "€": "EUR"}


def _one_line(text: object) -> str:
    """A value for one header line: line breaks would start a new attribute or section."""
    return str(text or "").replace("\r", " ").replace("\n", " ").strip(_VALUE_BLANKS)


def _encode_text(text: str) -> bytes:
    """Encode header text in ISO-8859-1 or the system's CJK code page.

    A few unit characters that neither can store are spelled out; anything
    else raises :class:`DatasetError` instead of being replaced silently.
    """
    for codec in ("latin-1", TEXT_CODEC):
        if codec:
            try:
                return text.encode(codec)
            except UnicodeEncodeError:
                pass
    spelled = "".join(_TRANSLITERATION.get(ch, ch) for ch in text)
    bad = ""
    for codec in ("latin-1", TEXT_CODEC):
        if codec:
            try:
                return spelled.encode(codec)
            except UnicodeEncodeError as exc:
                bad = spelled[exc.start]
    raise DatasetError(f"'{text}' contains '{bad}', which the recording's text encoding cannot store")


def _enum_text(enum: EnumTable) -> str:
    def esc(s: str) -> str:
        return _one_line(s).replace("&", "&amp").replace('"', "&quot").replace(",", "&#44")
    parts = []
    if enum.default is not None:
        parts.append(('[:]' if enum.ranges and not enum.values else '') + f'="{esc(enum.default)}"')
    parts += [f'{v:.15g}="{esc(t)}"' for v, t in enum.values.items()]
    parts += [f'{lo:.15g}:{hi:.15g}="{esc(t)}"' for lo, hi, t in enum.ranges]
    return "{" + ",".join(parts) + "}"


def write_d97(path: "str | Path", signals: Sequence[object], *, description: str = "",
              comments: Iterable[str] = (), infos: Optional[Mapping[int, str]] = None) -> List[str]:
    """Write signals in a binary layout with separate sample and time records.

    ``signals``: objects with ``name``, ``t``, ``y`` and optionally ``unit``,
    ``description``, ``color``, ``scmin``, ``scmax``, ``digital``, ``enum``
    (:class:`~tracepaddock.display.DisplaySignal` fits). Signals without samples
    are skipped. One time record is written per distinct time axis. ``infos``
    are [INFO<n>] texts to keep (e.g. from the recording the data came from);
    the creation date (INFO0) and this tool (INFO4) are added. Returns the
    names written. Texts are stored in the code page of this PC.
    """
    from .. import APP_NAME, __version__

    rows: List[np.ndarray] = []
    time_index: Dict[Tuple[int, float, float], int] = {}
    entries = []
    for sig in signals:
        t, y = getattr(sig, "t", None), getattr(sig, "y", None)
        if t is None or y is None or len(t) == 0:
            continue
        t = np.ascontiguousarray(t, dtype="<f8")
        y = np.ascontiguousarray(y, dtype="<f8")
        key = (len(t), float(t[0]), float(t[-1]))
        row = time_index.get(key)
        if row is None or not np.array_equal(rows[row], t):
            row = len(rows)
            rows.append(t)
            time_index[key] = row
        entries.append((sig, row, len(rows)))
        rows.append(y)
    now = time.ctime()
    info = {int(k): str(v) for k, v in (infos or {}).items()}
    info.setdefault(0, f"CREATED <{now}>")
    history = info.get(4, "")
    info[4] = (history + "\n" if history.strip() else "") + f"MODIFIED <{now}> BY {APP_NAME} {__version__}"
    def block(text: str) -> bytes:
        """Multi-line text of [DESCRIPTION] / [INFO<n>]: no line may start a section."""
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("[", "&#91").replace("]", "&#93")
        return _encode_text(text.replace("\n", "\r\n"))

    head: List[bytes] = [b"; " + _encode_text(_one_line(c)) for c in comments]
    head += [b"", b"[CONTROL]", b"FILEKEY=0x0100", b"LEVEL=3", b"VDIM=%d" % len(entries), b"FLOAT=INTEL387",
             b"PRECISION=DOUBLE", b"DATAMATRIX=SV_BY_*", b"", b"[DESCRIPTION]", block(description), b""]
    for n in sorted(info):
        head += [b"[INFO%d]" % n, block(info[n]), b""]
    for i, (sig, trow, drow) in enumerate(entries):
        digital = bool(getattr(sig, "digital", False))
        color = getattr(sig, "color", None)
        descr = str(getattr(sig, "description", "") or "").replace("&", "&amp")
        descr = descr.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
        head += [b"[SIGNAL%d]" % i, b"NAME=" + _encode_text(_one_line(sig.name)), b"DESCR=" + _encode_text(descr),
                 b"COLOR=%d" % (int(color) if color is not None else 1), b"TYP=" + (b"D0" if digital else b"FLT"),
                 b"DPOS=%d" % drow, b"TMSV=%d" % trow]
        if digital:
            head.append(b"BITP=0")
        else:
            head.append(b"UNIT=" + _encode_text(_one_line(getattr(sig, "unit", ""))))
            lo, hi = getattr(sig, "scmin", None), getattr(sig, "scmax", None)
            if lo is not None and hi is not None:
                head += [b"SCMIN=" + (b"%.15g" % lo), b"SCMAX=" + (b"%.15g" % hi)]
        enum = getattr(sig, "enum", None)
        if enum:
            head.append(b"ENUM=" + _encode_text(_enum_text(enum)))
        head.append(b"")
    head.append(b"[DATA]")
    with open(path, "wb") as f:
        f.write(b"\r\n".join(head) + b"\r\n")
        f.write(struct.pack("<d", len(rows)))
        for row in rows:
            f.write(struct.pack("<d", len(row)))
            f.write(row.tobytes())
        f.write(b"\r\n")
    return [e[0].name for e in entries]
