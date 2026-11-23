"""User defined value mappings: raw value -> logical value or text.

A recording often holds raw enumeration values (0, 1, 2 ...) whose meaning is
defined elsewhere: a road type, a speed limit, a state. A :class:`ValueMap` is
one such table. The :class:`ValueMapLibrary` keeps the tables and says which
signal uses which table and how it is shown (:class:`MapAssignment`).

The library is a JSON file in the user's profile, independent of PLT scripts.
Tables can be imported from CSV / TXT, Excel, DBC (``VAL_`` and ``VAL_TABLE_``) and JSON.
"""
from __future__ import annotations

import csv
import fnmatch
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .core.model import EnumTable
from .settings import read_json_object

__all__ = ["ValueMap", "MapAssignment", "ValueMapLibrary", "MappingImport", "read_mapping_file",
           "write_mapping_csv", "parse_raw_value", "MODES", "default_library_path"]

#: How the value of a mapped signal is written: text only, text with the raw number, number only.
MODES = ("mapped", "both", "raw")

_DEFAULT_WORDS = {"*", "default", "else", "other", "others", "其它", "其他", "默认", "缺省"}
_RANGE_SEPARATORS = ("...", "..", "…", "~", "～")


def default_library_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "TracePaddock" / "valuemaps.json"


def _number(text: str) -> float:
    """``'3'``, ``'3.5'``, ``'0x1F'``, ``'0b101'``, ``'1e3'`` -> float. Raises ``ValueError``."""
    s = text.strip().replace("−", "-")
    if not s:
        raise ValueError("empty")
    low = s.lower()
    sign = 1.0
    if low[0] in "+-" and low[1:3] in ("0x", "0b"):
        sign, low = (-1.0 if low[0] == "-" else 1.0), low[1:]
    if low.startswith("0x"):
        return sign * int(low[2:], 16)
    if low.startswith("0b"):
        return sign * int(low[2:], 2)
    value = float(s)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("not finite")
    return value


def parse_raw_value(text: object) -> "Optional[Tuple[float, float]]":
    """Raw value cell of a mapping table.

    Returns ``(lo, hi)`` (equal for a single value), or ``None`` for the
    "all other values" row (``*``, ``default``, ``其它`` ...). Raises
    ``ValueError`` when the text is neither.
    """
    if isinstance(text, bool):
        return (float(text), float(text))
    if isinstance(text, (int, float)):
        value = float(text)
        if value != value:
            raise ValueError("not a number")
        return (value, value)
    s = str(text).strip()
    if s.lower() in _DEFAULT_WORDS:
        return None
    try:
        value = _number(s)
        return (value, value)
    except ValueError:
        pass
    for separator in _RANGE_SEPARATORS:
        if separator in s:
            left, right = s.split(separator, 1)
            try:
                lo, hi = _number(left), _number(right)
            except ValueError:
                break
            return (lo, hi) if lo <= hi else (hi, lo)
    raise ValueError(f"'{s}' is not a number, a range a..b or '*'")


def _g(value: float) -> str:
    return "%.15g" % value


@dataclass
class ValueMap:
    """One mapping table. ``rows`` are ``(lo, hi, text)``; ``lo == hi`` for a single value."""

    name: str
    rows: List[Tuple[float, float, str]] = field(default_factory=list)
    #: Text for values that no row covers (``None`` = show the number).
    default: Optional[str] = None
    #: Unit of the mapped values when they are numbers, e.g. ``km/h``.
    unit: str = ""
    #: Pictures for the image panel: raw value text ("80", "10..20", "*") -> picture spec (see tracepaddock.signs).
    images: Dict[str, str] = field(default_factory=dict)
    #: "speed": show a speed limit sign for the logical value (km/h) when no picture is given.
    auto_image: str = ""

    @staticmethod
    def key(lo: float, hi: float) -> str:
        """Text of a row's raw value as used for :attr:`images`."""
        return _g(lo) if lo == hi else "%s..%s" % (_g(lo), _g(hi))

    def image_for(self, value: Optional[float]) -> Optional[str]:
        """Picture for a raw value: the row's picture, the "*" picture, or the automatic sign."""
        if value is None or value != value:
            return None
        for lo, hi, _text in self.rows:
            if lo == hi and lo == value:
                spec = self.images.get(self.key(lo, hi))
                if spec:
                    return spec
                break
        else:
            for lo, hi, _text in self.rows:
                if lo != hi and lo <= value <= hi and self.images.get(self.key(lo, hi)):
                    return self.images[self.key(lo, hi)]
        if self.images.get("*"):
            covered = any(lo <= value <= hi for lo, hi, _t in self.rows)
            if not covered:
                return self.images["*"]
        if self.auto_image == "speed":
            from .signs import auto_speed_spec
            mapped = self.numbers()
            number = dict(mapped).get(float(value)) if mapped else None
            return auto_speed_spec(number if number is not None else value)
        return None

    @property
    def has_images(self) -> bool:
        return bool(self.images) or bool(self.auto_image)

    def enum(self) -> EnumTable:
        table = EnumTable(default=self.default if self.default else None)
        for lo, hi, text in self.rows:
            if lo == hi:
                table.values.setdefault(float(lo), text)
            else:
                table.ranges.append((float(lo), float(hi), text))
        return table

    def text(self, value: float) -> Optional[str]:
        return self.enum().text(value)

    def numbers(self) -> "Optional[List[Tuple[float, float]]]":
        """``[(raw, mapped), ...]`` sorted by raw value when every row is a single value with a
        numeric text, else ``None``."""
        pairs = []
        for lo, hi, text in self.rows:
            if lo != hi:
                return None
            try:
                pairs.append((float(lo), _number(text)))
            except ValueError:
                return None
        return sorted(pairs) if pairs else None

    @property
    def is_numeric(self) -> bool:
        return self.numbers() is not None

    def expression(self, symbol: str = "x") -> str:
        """PLT formula that converts the raw value into the mapped number: ``map(x,0,0,1,30,...)``.

        Raises ``ValueError`` when the table is not purely numeric.
        """
        pairs = self.numbers()
        if pairs is None:
            raise ValueError(f"table '{self.name}' does not map every value to a number")
        if len(pairs) == 1:
            return _g(pairs[0][1])
        return "map(%s,%s)" % (symbol, ",".join("%s,%s" % (_g(p), _g(q)) for p, q in pairs))

    def to_json(self) -> dict:
        data = {"name": self.name, "unit": self.unit, "default": self.default,
                "rows": [[lo, hi, text] for lo, hi, text in self.rows]}
        if self.images:
            data["images"] = dict(self.images)
        if self.auto_image:
            data["auto_image"] = self.auto_image
        return data

    @classmethod
    def from_json(cls, data: dict) -> "ValueMap":
        rows = []
        for row in data.get("rows", []) or []:
            try:
                lo, hi, text = float(row[0]), float(row[1]), str(row[2])
            except (TypeError, ValueError, IndexError):
                continue
            rows.append((lo, hi, text) if lo <= hi else (hi, lo, text))
        default = data.get("default")
        images = data.get("images") or {}
        return cls(name=str(data.get("name", "")), rows=rows,
                   default=str(default) if default not in (None, "") else None, unit=str(data.get("unit", "") or ""),
                   images={str(k): str(v) for k, v in images.items() if v} if isinstance(images, dict) else {},
                   auto_image=str(data.get("auto_image", "") or ""))


@dataclass
class MapAssignment:
    """A signal (name or wildcard pattern) uses a table.

    ``table`` may be empty: then the signal keeps the text table that came with
    the recording, and only ``mode`` and ``lane`` apply.
    """

    pattern: str
    table: str
    #: ``"mapped"`` text only, ``"both"`` text with the raw number, ``"raw"`` number only.
    mode: str = "mapped"
    #: Show the signal as a band of labelled states instead of a curve.
    lane: bool = False

    def to_json(self) -> dict:
        return {"pattern": self.pattern, "table": self.table, "mode": self.mode, "lane": self.lane}

    @classmethod
    def from_json(cls, data: dict) -> "MapAssignment":
        mode = str(data.get("mode", "mapped"))
        return cls(pattern=str(data.get("pattern", "")), table=str(data.get("table", "")),
                   mode=mode if mode in MODES else "mapped", lane=bool(data.get("lane", False)))


@dataclass
class MappingImport:
    """What a mapping file contained."""

    tables: List[ValueMap] = field(default_factory=list)
    assignments: List[MapAssignment] = field(default_factory=list)
    #: Rows that were skipped and why (for the user).
    problems: List[str] = field(default_factory=list)


class ValueMapLibrary:
    """All tables and assignments of the user."""

    def __init__(self, path: "Optional[str | Path]" = None, *, autoload: bool = True) -> None:
        self.path = Path(path) if path is not None else default_library_path()
        self.tables: Dict[str, ValueMap] = {}
        self.assignments: List[MapAssignment] = []
        #: False when the file could not be read and must not be written over
        self._writable = True
        if autoload:
            self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        """Read the library file (UTF-8, also with BOM). A file that cannot be read is kept, not overwritten."""
        data, self._writable = read_json_object(self.path)
        if data is not None:
            self._take(data)

    def _take(self, data: dict) -> None:
        self.tables, self.assignments = {}, []
        for item in data.get("tables", []) or []:
            if isinstance(item, dict):
                table = ValueMap.from_json(item)
                if table.name:
                    self.tables[table.name] = table
        for item in data.get("assignments", []) or []:
            if isinstance(item, dict):
                assignment = MapAssignment.from_json(item)
                if assignment.pattern and (not assignment.table or assignment.table in self.tables):
                    self.assignments.append(assignment)

    def to_json(self) -> dict:
        return {"format": "tracepaddock-valuemaps", "version": 1,
                "tables": [t.to_json() for t in self.tables.values()],
                "assignments": [a.to_json() for a in self.assignments]}

    def save(self, path: "Optional[str | Path]" = None) -> None:
        """Write the library. Errors are raised only when ``path`` is given (an explicit export)."""
        if path is None and not getattr(self, "_writable", True):
            return                                             # the file could not be read: keep it as it is
        target = Path(path) if path is not None else self.path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, indent=1, ensure_ascii=False)
            os.replace(tmp, target)
        except OSError:
            if path is not None:
                raise

    # ------------------------------------------------------------------ lookup
    def _usable(self, assignment: MapAssignment) -> bool:
        return not assignment.table or assignment.table in self.tables

    def assignment_for(self, *names: Optional[str]) -> Optional[MapAssignment]:
        """Assignment for a signal known under ``names``: an exact name wins over a wildcard pattern,
        and the first name given wins over the later ones."""
        wanted = [n.lower() for n in names if n]
        if not wanted:
            return None
        exact = {a.pattern.lower(): a for a in reversed(self.assignments) if self._usable(a)}
        for name in wanted:
            if name in exact:
                return exact[name]
        for assignment in self.assignments:
            pattern = assignment.pattern.lower()
            if ("*" in pattern or "?" in pattern) and self._usable(assignment):
                if any(fnmatch.fnmatchcase(name, pattern) for name in wanted):
                    return assignment
        return None

    def find(self, *names: Optional[str]) -> "Optional[Tuple[Optional[ValueMap], MapAssignment]]":
        """``(table, assignment)`` for a signal; ``table`` is ``None`` when the assignment only sets
        how the recording's own text table is shown."""
        assignment = self.assignment_for(*names)
        if assignment is None:
            return None
        return self.tables.get(assignment.table) if assignment.table else None, assignment

    def exact(self, pattern: str) -> Optional[MapAssignment]:
        low = pattern.lower()
        return next((a for a in self.assignments if a.pattern.lower() == low), None)

    # ------------------------------------------------------------------ changes
    def assign(self, pattern: str, table: str, mode: str = "mapped", lane: bool = False) -> MapAssignment:
        """Let ``pattern`` use ``table`` (replaces an assignment with the same pattern)."""
        if mode not in MODES:
            mode = "mapped"
        current = self.exact(pattern)
        if current is not None:
            current.table, current.mode, current.lane = table, mode, lane
            return current
        assignment = MapAssignment(pattern, table, mode, lane)
        self.assignments.append(assignment)
        return assignment

    def unassign(self, pattern: str) -> None:
        low = pattern.lower()
        self.assignments = [a for a in self.assignments if a.pattern.lower() != low]

    def unique_name(self, wanted: str) -> str:
        name, n = wanted or "table", 2
        while name in self.tables:
            name, n = "%s (%d)" % (wanted or "table", n), n + 1
        return name

    def remove_table(self, name: str) -> None:
        self.tables.pop(name, None)
        self.assignments = [a for a in self.assignments if a.table != name]

    def rename_table(self, old: str, new: str) -> bool:
        if old == new:
            return True
        if not new or new in self.tables or old not in self.tables:
            return False
        self.tables = {(new if k == old else k): v for k, v in self.tables.items()}
        self.tables[new].name = new
        for assignment in self.assignments:
            if assignment.table == old:
                assignment.table = new
        return True

    def merge(self, imported: MappingImport) -> Tuple[int, int]:
        """Take over imported tables (same name = replaced) and assignments. Returns the two counts."""
        for table in imported.tables:
            self.tables[table.name] = table
        count = 0
        for item in imported.assignments:
            if self._usable(item):
                current = self.exact(item.pattern)
                if current is not None:                    # keep how the user chose to show the signal
                    current.table = item.table
                else:
                    self.assignments.append(MapAssignment(item.pattern, item.table, item.mode, item.lane))
                count += 1
        return len(imported.tables), count


# ---------------------------------------------------------------------- import
_ALIASES = {
    "signal": {"signal", "signal name", "signalname", "name", "信号", "信号名", "信号名称", "变量", "变量名"},
    "value": {"value", "raw", "raw value", "rawvalue", "enum", "enum value", "原始值", "枚举值", "值", "数值"},
    "text": {"text", "label", "meaning", "logical", "logical value", "description", "显示值", "逻辑值",
             "含义", "映射值", "说明", "文本", "显示文本", "意义"},
    "table": {"table", "map", "table name", "表", "映射表", "表名"},
    "unit": {"unit", "单位"},
    "image": {"image", "picture", "icon", "图片", "图示", "图标"},
}


def _decode(raw: bytes) -> str:
    for codec in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(codec)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return str(value).strip()


def _rows_to_import(rows: Iterable[Sequence[object]], default_table: str, where: str) -> MappingImport:
    """Turn the rows of a sheet (lists of cells) into tables and assignments."""
    result = MappingImport()
    cleaned = [[_cell(c) for c in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return result
    columns: Dict[str, int] = {}
    first = [c.lower() for c in cleaned[0]]
    for key, names in _ALIASES.items():
        for index, cell in enumerate(first):
            if cell in names and key not in columns and index not in columns.values():
                columns[key] = index
    if "value" in columns and "text" in columns:
        body = cleaned[1:]
    else:
        width = max(len(row) for row in cleaned)
        columns = {"signal": 0, "value": 1, "text": 2} if width >= 3 else {"value": 0, "text": 1}
        body = cleaned
        if width < 2:
            result.problems.append(f"{where}: two columns (value, text) or three (signal, value, text) are needed")
            return result

    def get(row: Sequence[str], key: str) -> str:
        index = columns.get(key)
        return row[index] if index is not None and index < len(row) else ""

    tables: Dict[str, ValueMap] = {}
    assigned: Dict[str, str] = {}
    signal = table_name = ""
    for number, row in enumerate(body, start=2 if body is not cleaned else 1):
        signal = get(row, "signal") or signal                  # merged cells: the name is written once
        table_name = get(row, "table") or (table_name if "table" in columns else "")
        name = table_name or signal or default_table
        raw, text = get(row, "value"), get(row, "text")
        if not raw and not text:
            continue
        try:
            span = parse_raw_value(raw)
        except ValueError as exc:
            result.problems.append(f"{where}, row {number}: {exc}")
            continue
        table = tables.setdefault(name, ValueMap(name))
        if get(row, "unit"):
            table.unit = get(row, "unit")
        image = get(row, "image")
        if span is None:
            table.default = text or None
            if image:
                table.images["*"] = image
        else:
            table.rows.append((span[0], span[1], text))
            if image:
                table.images[ValueMap.key(span[0], span[1])] = image
        if signal:
            assigned.setdefault(signal, name)
    result.tables = [t for t in tables.values() if t.rows or t.default]
    kept = {t.name for t in result.tables}
    result.assignments = [MapAssignment(sig, name) for sig, name in assigned.items() if name in kept]
    return result


def _read_csv(path: Path) -> MappingImport:
    text = _decode(path.read_bytes())
    sample = "\n".join(text.splitlines()[:20])
    counts = {d: sample.count(d) for d in ("\t", ";", ",")}
    delimiter = max(counts, key=counts.get) if any(counts.values()) else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return _rows_to_import(list(reader), path.stem, path.name)


def _read_xlsx(path: Path) -> MappingImport:
    try:
        import openpyxl
    except ImportError as exc:                              # pragma: no cover - depends on the installation
        raise ValueError("Excel files need the Python package 'openpyxl'; save the table as CSV instead") from exc
    result = MappingImport()
    try:
        book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError(f"{path.name}: not a readable Excel workbook ({exc})") from exc
    try:
        for sheet in book.worksheets:
            part = _rows_to_import(list(sheet.iter_rows(values_only=True)), sheet.title, f"{path.name} [{sheet.title}]")
            result.tables += part.tables
            result.assignments += part.assignments
            result.problems += part.problems
    finally:
        book.close()
    return result


_DBC_STATEMENT = re.compile(r"^[ \t]*VAL_(TABLE_)?[ \t]+(.*?);[ \t]*$", re.MULTILINE | re.DOTALL)
_DBC_PAIR = re.compile(r'(-?\d+(?:\.\d+)?)\s+"((?:[^"\\]|\\.)*)"')


def _read_dbc(path: Path) -> MappingImport:
    """Value tables of a CAN database: ``VAL_ <message id> <signal> <value> "<text>" ... ;``."""
    text = _decode(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    result = MappingImport()
    seen: Dict[str, ValueMap] = {}
    for match in _DBC_STATEMENT.finditer(text):
        is_table, body = bool(match.group(1)), match.group(2).strip()
        head = body.split('"', 1)[0].split()
        if is_table:
            if len(head) < 1:
                continue
            name, message = head[0], ""
        else:
            if len(head) < 2:
                continue
            message, name = head[0], head[1]
        rows = [(float(v), float(v), t.replace('\\"', '"')) for v, t in _DBC_PAIR.findall(body)]
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        table = ValueMap(name, rows)
        known = seen.get(name)
        if known is not None:
            if known.rows == rows:
                continue
            table.name = "%s@%s" % (name, message or len(seen))  # same signal name in another message
        seen[table.name] = table
        result.tables.append(table)
        if not is_table and known is None:
            result.assignments.append(MapAssignment(name, table.name))
    if not result.tables:
        result.problems.append(f"{path.name}: no VAL_ or VAL_TABLE_ entries found")
    return result


def _read_json(path: Path) -> MappingImport:
    try:
        data = json.loads(_decode(path.read_bytes()))
    except ValueError as exc:
        raise ValueError(f"{path.name}: not valid JSON ({exc})") from exc
    library = ValueMapLibrary(path, autoload=False)
    if isinstance(data, dict):
        library._take(data)
    result = MappingImport(list(library.tables.values()), list(library.assignments))
    if not result.tables:
        result.problems.append(f"{path.name}: no tables found")
    return result


def read_mapping_file(path: "str | Path") -> MappingImport:
    """Read tables (and signal assignments, if the file names signals) from a mapping file.

    * ``.csv`` / ``.txt``: columns *signal, value, text* (or only *value, text*), with or
      without a header row; separator tab, semicolon or comma; UTF-8 or GBK.
    * ``.xlsx``: the same columns, every sheet.
    * ``.dbc``: ``VAL_`` and ``VAL_TABLE_`` statements.
    * ``.json``: a library written by :meth:`ValueMapLibrary.save`.

    Raises ``ValueError`` with a message for the user when the file cannot be read.
    """
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".dbc":
            return _read_dbc(p)
        if ext == ".json":
            return _read_json(p)
        if ext in (".xlsx", ".xlsm"):
            return _read_xlsx(p)
        return _read_csv(p)
    except OSError as exc:
        raise ValueError(f"{p.name}: cannot be read ({exc.strerror or exc})") from exc


def write_mapping_csv(path: "str | Path", tables: Sequence[ValueMap],
                      assignments: Sequence[MapAssignment] = ()) -> None:
    """Write tables as a CSV that :func:`read_mapping_file` (and Excel) reads back."""
    by_table: Dict[str, List[str]] = {}
    for assignment in assignments:
        by_table.setdefault(assignment.table, []).append(assignment.pattern)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "signal", "value", "text", "unit", "image"])
        for table in tables:
            for signal in by_table.get(table.name) or [""]:
                for lo, hi, text in table.rows:
                    raw = ValueMap.key(lo, hi)
                    writer.writerow([table.name, signal, raw, text, table.unit, table.images.get(raw, "")])
                if table.default or table.images.get("*"):
                    writer.writerow([table.name, signal, "*", table.default or "", table.unit, table.images.get("*", "")])
