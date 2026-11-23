"""Signal databases: DBC files and CAN matrices in Excel.

What the viewer takes from them: the resolution (factor) of a signal, which
suggests the tolerance for comparisons; offset, unit, minimum and maximum;
the cycle time and send type of its message (how fine a measured delay can
be, whether a value is repeated); and its value table (raw value -> text),
which can be taken over as a value mapping.

DBC, ARXML, KCD and SYM are read with the ``canmatrix`` package. Excel and
CSV CAN matrices are read by their column titles (English and Chinese titles
are recognised), so the usual OEM layouts work without configuration.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["SignalSpec", "SignalDatabase", "parse_value_description", "CanDbError"]


class CanDbError(Exception):
    """A database file could not be read; the message is shown to the user."""


@dataclass
class SignalSpec:
    name: str
    message: str = ""
    message_id: Optional[int] = None
    cycle_ms: Optional[float] = None
    send_type: str = ""
    signal_send_type: str = ""
    length: Optional[int] = None
    factor: Optional[float] = None
    offset: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = ""
    #: raw value -> text
    values: Dict[float, str] = field(default_factory=dict)
    description: str = ""
    source: str = ""

    @property
    def resolution(self) -> Optional[float]:
        return abs(self.factor) if self.factor else None

    def physical(self, raw: float) -> float:
        return raw * (self.factor if self.factor else 1.0) + (self.offset or 0.0)

    def physical_values(self) -> Dict[float, str]:
        return {round(self.physical(k), 12): v for k, v in self.values.items()}

    def tolerance_hint(self) -> Optional[float]:
        """Half a resolution step: the tolerance for "equal" between physical values of this signal."""
        return None if not self.resolution else self.resolution / 2.0

    def summary(self) -> str:
        parts = []
        if self.resolution:
            parts.append("resolution %g %s" % (self.resolution, self.unit))
        if self.cycle_ms:
            parts.append("cycle %g ms" % self.cycle_ms)
        if self.send_type:
            parts.append(self.send_type)
        if self.values:
            parts.append("%d value texts" % len(self.values))
        return ", ".join(parts)


_NUMBER = r"(0x[0-9a-fA-F]+|-?\d+(?:\.\d+)?)"
_VALUE_ITEM = re.compile(r"['\"]?" + _NUMBER + r"['\"]?\s*[:=]\s*(?:'([^']*)'|\"([^\"]*)\"|([^,;\n\r]+))")


def parse_value_description(text: object) -> Dict[float, str]:
    """``"'0x0': 'Off', '0x1': 'On'"``, ``"0=Off;1=On"``, ``"0x0: Off\\n0x1: On"`` -> {0: 'Off', 1: 'On'}."""
    out: Dict[float, str] = {}
    if text is None:
        return out
    for m in _VALUE_ITEM.finditer(str(text)):
        key = m.group(1)
        value = (m.group(2) if m.group(2) is not None else m.group(3) if m.group(3) is not None else m.group(4) or "")
        value = value.strip().strip("'\"").strip()
        try:
            number = float(int(key, 16)) if key.lower().startswith("0x") else float(key)
        except ValueError:
            continue
        if value:
            out[number] = value
    return out


def _float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        text = str(value).strip()
        return float(int(text, 16)) if text.lower().startswith("0x") else float(text)
    except ValueError:
        return None


# column title -> field; exact titles first, then words a title may contain
_COLUMNS: Sequence[Tuple[str, Sequence[str], Sequence[str]]] = (
    ("message", ("msg name", "message name", "message", "报文名称", "报文名", "报文"), ()),
    ("message_id", ("msg id(hex)", "msg id", "message id", "can id", "id", "报文id", "报文标识符"), ()),
    ("cycle_ms", ("genmsgcycletime", "cycle time", "cycle time [ms]", "cycle time(ms)", "周期", "发送周期",
                  "周期(ms)"), ()),
    ("send_type", ("genmsgsendtype", "msg send type", "message send type", "报文发送类型"), ()),
    ("name", ("signal name", "signal", "信号名称", "信号名", "信号"), ()),
    ("description", ("signal description", "description", "信号描述", "描述", "comment"), ()),
    ("signal_send_type", ("signal type", "signal send type", "gensigsendtype", "信号发送类型"), ()),
    ("length", ("signal length", "length", "信号长度", "长度", "bit length"), ()),
    ("unit", ("signal unit", "unit", "单位"), ()),
    ("factor", ("resolution", "factor", "精度", "分辨率", "系数", "因子"), ()),
    ("offset", ("offset", "偏移量", "偏移"), ()),
    ("minimum", ("min", "minimum", "最小值", "物理最小值"), ("最小", "min ")),
    ("maximum", ("max", "maximum", "最大值", "物理最大值"), ("最大", "max ")),
    ("values", ("signal value description(dec)", "signal value description", "value description", "value table",
                "值描述", "信号值描述", "数值描述"), ("value description", "值描述")),
)


def _map_columns(header: Sequence[object]) -> Dict[str, int]:
    titles = [str(h or "").strip().lower().replace("\n", " ") for h in header]
    found: Dict[str, int] = {}
    for key, exact, contains in _COLUMNS:
        for i, title in enumerate(titles):
            if i in found.values():
                continue
            if title in exact or any(word in title for word in contains):
                found[key] = i
                break
    return found


class SignalDatabase:
    """All loaded databases. A signal name may occur in several messages: the first one found wins in
    :meth:`find`; :meth:`search` lists all."""

    def __init__(self) -> None:
        self.files: List[str] = []
        self._by_name: Dict[str, List[SignalSpec]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_name.values())

    def _add(self, spec: SignalSpec) -> None:
        self._by_name.setdefault(spec.name.lower(), []).append(spec)

    def load(self, path: "str | Path") -> int:
        """Load one file. Returns the number of signals read. Raises :class:`CanDbError`."""
        p = Path(path)
        ext = p.suffix.lower()
        try:
            if ext in (".xlsx", ".xlsm", ".csv", ".txt"):
                specs = list(self._read_table(p))
            else:
                specs = list(self._read_canmatrix(p))
        except CanDbError:
            raise
        except Exception as exc:
            raise CanDbError("%s: cannot be read (%s: %s)" % (p.name, type(exc).__name__, exc)) from exc
        if not specs:
            raise CanDbError("%s: no signals found (is this a CAN matrix with a 'Signal Name' column?)" % p.name)
        for spec in specs:
            self._add(spec)
        self.files.append(str(p))
        return len(specs)

    def _read_canmatrix(self, path: Path) -> Iterable[SignalSpec]:
        try:
            import canmatrix.formats as cf
        except ImportError as exc:                                      # pragma: no cover
            raise CanDbError("reading %s needs the 'canmatrix' package" % path.suffix) from exc
        import logging
        logging.getLogger("canmatrix").setLevel(logging.ERROR)
        dbs = cf.loadp(str(path))
        for db in dbs.values():
            for frame in db.frames:
                try:
                    send = frame.attribute("GenMsgSendType", db) or ""
                except Exception:
                    send = ""
                cycle = getattr(frame, "cycle_time", 0) or None
                for sig in frame.signals:
                    try:
                        sig_send = sig.attribute("GenSigSendType", db) or ""
                    except Exception:
                        sig_send = ""
                    values = {}
                    for k, v in (sig.values or {}).items():
                        try:
                            values[float(k)] = str(v)
                        except (TypeError, ValueError):
                            continue
                    yield SignalSpec(name=sig.name, message=frame.name,
                                     message_id=int(frame.arbitration_id.id) if frame.arbitration_id else None,
                                     cycle_ms=float(cycle) if cycle else None, send_type=str(send),
                                     signal_send_type=str(sig_send), length=int(sig.size),
                                     factor=float(sig.factor), offset=float(sig.offset),
                                     minimum=_float(sig.min), maximum=_float(sig.max), unit=str(sig.unit or ""),
                                     values=values, description=str(sig.comment or ""), source=path.name)

    def _rows(self, path: Path) -> Iterable[Tuple[str, List[Sequence[object]]]]:
        if path.suffix.lower() in (".xlsx", ".xlsm"):
            import openpyxl
            book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            try:
                for sheet in book.worksheets:
                    yield sheet.title, [list(r) for r in sheet.iter_rows(values_only=True)]
            finally:
                book.close()
        else:
            raw = path.read_bytes()
            for codec in ("utf-8-sig", "gb18030", "latin-1"):
                try:
                    text = raw.decode(codec)
                    break
                except UnicodeDecodeError:
                    continue
            sample = "\n".join(text.splitlines()[:10])
            delimiter = max(("\t", ";", ","), key=sample.count)
            yield path.stem, list(csv.reader(io.StringIO(text), delimiter=delimiter))

    def _read_table(self, path: Path) -> Iterable[SignalSpec]:
        for sheet, rows in self._rows(path):
            header_index = None
            for i, row in enumerate(rows[:30]):
                columns = _map_columns(row)
                if "name" in columns and ("factor" in columns or "message" in columns):
                    header_index = i
                    break
            if header_index is None:
                continue
            col = _map_columns(rows[header_index])

            def get(row, key):
                i = col.get(key)
                return row[i] if i is not None and i < len(row) else None

            message = ""
            for row in rows[header_index + 1:]:
                name = get(row, "name")
                message = str(get(row, "message") or message or "").strip()
                if not name or not str(name).strip():
                    continue
                msg_id = get(row, "message_id")
                yield SignalSpec(
                    name=str(name).strip(), message=message, message_id=int(_float(msg_id)) if _float(msg_id) is not None else None,
                    cycle_ms=_float(get(row, "cycle_ms")) or None, send_type=str(get(row, "send_type") or ""),
                    signal_send_type=str(get(row, "signal_send_type") or ""),
                    length=int(_float(get(row, "length"))) if _float(get(row, "length")) is not None else None,
                    factor=_float(get(row, "factor")), offset=_float(get(row, "offset")),
                    minimum=_float(get(row, "minimum")), maximum=_float(get(row, "maximum")),
                    unit=str(get(row, "unit") or "").strip(), values=parse_value_description(get(row, "values")),
                    description=str(get(row, "description") or "").strip(), source="%s [%s]" % (path.name, sheet))

    # ------------------------------------------------------------------ lookup
    def find(self, name: str) -> Optional[SignalSpec]:
        """The database entry of a recorded signal: exact name, the last part of a dotted name
        (``CAN1.Msg.Signal``), or the name without device suffix (``Signal\\CCP:1``)."""
        if not name:
            return None
        candidates = [name, name.split("\\", 1)[0]]
        for sep in ("::", "."):
            if sep in name:
                candidates.append(name.split("\\", 1)[0].rsplit(sep, 1)[-1])
        for c in candidates:
            hit = self._by_name.get(c.lower())
            if hit:
                return hit[0]
        return None

    def search(self, text: str, limit: int = 500) -> List[SignalSpec]:
        needle = text.strip().lower()
        out: List[SignalSpec] = []
        for key, specs in self._by_name.items():
            if not needle or needle in key or any(needle in s.message.lower() or needle in s.description.lower()
                                                  for s in specs):
                out.extend(specs)
                if len(out) >= limit:
                    break
        return sorted(out, key=lambda s: (s.name.lower(), s.message))[:limit]

    def to_mapping_import(self, names: Optional[Sequence[str]] = None):
        """Value tables as value mappings (physical value -> text), one table per signal."""
        from .valuemap import MapAssignment, MappingImport, ValueMap
        result = MappingImport()
        wanted = {n.lower() for n in names} if names else None
        for key, specs in self._by_name.items():
            if wanted is not None and key not in wanted:
                continue
            spec = next((s for s in specs if s.values), None)
            if spec is None:
                continue
            rows = sorted((v, v, t) for v, t in spec.physical_values().items())
            result.tables.append(ValueMap(spec.name, rows, unit=""))
            result.assignments.append(MapAssignment(spec.name, spec.name))
        return result
