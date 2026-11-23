"""Read and write PLT scripts.

See :mod:`tracepaddock.plt.model` for the format. The parser is tolerant: a line
it cannot interpret is kept as a comment and tokens it does not know are kept
verbatim, so loading and saving a script never loses anything. Definitions
that were not changed are written back exactly as they were read.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.model import KIND_BIT, SignalInfo
from .model import Op, PltComment, PltEntry, PltGroupBegin, PltGroupEnd, PltItem, PltScript
from .palette import COLOR_NAMES

__all__ = [
    "PltError", "parse_plt", "load_plt", "dump_plt", "save_plt", "parse_attributes",
    "format_attributes", "format_entry", "entry_from_signal", "short_name", "split_name", "value_problem",
]


class PltError(ValueError):
    """A PLT script could not be read or written."""


_FLAG_OPS = ("min", "max", "-mean", "-min", "-max", "diff", "integ", "revert")
#: Legacy display attributes preserved when saving a view.
_LEGACY_FLAGS = ("noscale", "grid", "log")
_RE_BITSUFFIX = re.compile(r"^(.*?):(\d+)(?::(-?\d+))?$")
#: ``name\DEVICE:1`` - the first ``:<digits>`` after the last backslash belongs
#: to the device name of MDF recordings, a bit suffix may follow it.
_RE_DEVICE_NAME = re.compile(r"^(?P<base>.*\\[^\\:\s]*:\d+)(?::(?P<pos>\d+)(?::(?P<width>-?\d+))?)?$")
_RE_KEYVAL = re.compile(r"^(key|unit|scmin|scmax|fmt|emphasize|dgf|channel|ref[0-2]?)=(.*)$", re.I)
_RE_EXP = re.compile(r"^exp\s*(\(|=)", re.I)
_RE_CALL = re.compile(r"^(mavr|shift)\(\s*(-?\d+)\s*\)$", re.I)
_RE_GROUP = re.compile(r'^//\s*BEGIN_GROUP\s+"([^"]*)"(.*)$', re.I)
_RE_GROUP_END = re.compile(r"^//\s*END_GROUP\b", re.I)
_RE_SIGNALGROUP = re.compile(r"^//\s*<\s*signalgroup_(\d+)\s*>", re.I)
_RE_TRAILING_COMMENT = re.compile(r"(^|\s)//")
_BOM = b"\xef\xbb\xbf"


def _is_keyword(token: str) -> bool:
    """True if ``token`` starts a new attribute (and therefore ends a free text value)."""
    low = token.lower()
    return bool(
        low in ("bit", "nobit", "scansc") or low in _FLAG_OPS or low in _LEGACY_FLAGS
        or low.upper() in COLOR_NAMES or token.startswith("//")
        or re.fullmatch(r"\d+", token) or _RE_KEYVAL.match(token) or _RE_EXP.match(token)
        or _RE_CALL.match(token)
    )


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def _expression_open(text: str) -> bool:
    """True while an ``exp`` attribute collected so far cannot be complete."""
    if text.count("(") > text.count(")"):
        return True
    head = text.lower()
    if head.startswith("exp") and "=" not in text:
        return True
    return text.rstrip()[-1:] in "+-*/,=(" or text.rstrip().endswith("**")


def _make_exp(text: str) -> Op:
    """Split ``exp=<expr>`` / ``exp(x1=a,x2=b)=<expr>`` into its parts."""
    body = text[3:].lstrip()
    refs = {}
    if body.startswith("("):
        depth, end = 0, -1
        for i, ch in enumerate(body):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > 0:
            for part in body[1:end].split(","):
                sym, sep, name = part.partition("=")
                if sep and sym.strip():
                    refs[sym.strip().lower()] = name.strip()
            body = body[end + 1:].lstrip()
    expr = body[1:].strip() if body.startswith("=") else body.strip()
    return Op("exp", text, expr=expr, refs=refs)


def _make_dgf(text: str) -> Op:
    parts = text.split("=", 1)[1].split(":")
    args: Tuple = (parts[0].strip().upper(),) + tuple(
        v for v in (_number(p) for p in parts[1:]) if v is not None)
    return Op("dgf", text, args=args)


def _split_comment(text: str) -> Tuple[str, Optional[str]]:
    """``"a 1 unit=m  // note"`` -> ``("a 1 unit=m", "// note")``."""
    m = _RE_TRAILING_COMMENT.search(text)
    if not m:
        return text, None
    start = m.start() if not m.group(1) else m.start() + 1
    return text[:start].rstrip(), text[start:].strip()


def parse_attributes(text: str, entry: PltEntry) -> None:
    """Parse an attribute string (everything after the signal name) into ``entry``.

    Attributes add to what the entry already holds; value changing operations
    are appended to ``entry.ops`` in the order written.
    """
    tokens = text.split()
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        low = tok.lower()
        i += 1
        if re.fullmatch(r"\d+", tok):
            entry.color = int(tok)
        elif low in ("bit", "nobit"):
            entry.view = low
            entry.view_pos = len(entry.ops) if entry.ops else None
        elif low == "scansc":
            entry.scansc = True
        elif low in _FLAG_OPS:
            entry.ops.append(Op(low, low))
        elif _RE_CALL.match(tok):
            m = _RE_CALL.match(tok)
            entry.ops.append(Op(m.group(1).lower(), tok, args=(int(m.group(2)),)))
        elif _RE_EXP.match(tok):
            parts = [tok]
            while i < n:
                # An expression may contain spaces ("exp=x + 3"). Once it is
                # complete, the next attribute keyword (a colour number, -min,
                # unit= ...) ends it; only a token that continues it with an
                # operator is still part of the expression.
                if not _expression_open(" ".join(parts)) and (
                        _is_keyword(tokens[i]) or tokens[i][:1] not in "+-*/,)"):
                    break
                parts.append(tokens[i])
                i += 1
            entry.ops.append(_make_exp(" ".join(parts)))
        elif _RE_KEYVAL.match(tok):
            m = _RE_KEYVAL.match(tok)
            key, value = m.group(1).lower(), m.group(2)
            if key in ("key", "unit"):
                parts = [value] if value else []
                while i < n and not _is_keyword(tokens[i]):
                    parts.append(tokens[i])
                    i += 1
                setattr(entry, key, " ".join(parts))
            elif key in ("scmin", "scmax"):
                num = _number(value)
                if num is None or not math.isfinite(num):
                    entry.extra.append(tok)
                else:
                    setattr(entry, key, num)
            elif key == "fmt":
                entry.fmt = value
            elif key == "channel":
                entry.channel_origin = value
            elif key == "emphasize":
                num = _number(value)
                if num is None or not math.isfinite(num):
                    entry.extra.append(tok)
                else:
                    entry.emphasize = int(num)
            elif key == "dgf":
                entry.ops.append(_make_dgf(tok))
            else:  # ref / ref0..ref2
                index = int(key[3]) if len(key) > 3 else 0
                num = _number(value)
                entry.refs[index] = num if num is not None and math.isfinite(num) else value.upper()
        elif low.upper() in COLOR_NAMES:
            entry.color = COLOR_NAMES[low.upper()]
        else:
            entry.extra.append(tok)


def split_name(token: str) -> Tuple[str, Optional[int], Optional[int], bool]:
    """``name[:pos[:[-]width]]`` -> ``(name, pos, width, signed)``.

    A device suffix of MDF recordings (``name\\CCP:1``) is part of the name.
    """
    if "\\" in token:
        m = _RE_DEVICE_NAME.match(token)
        if m:
            pos, width = m.group("pos"), m.group("width")
            if pos is None:
                return m.group("base"), None, None, False
            count = abs(int(width)) if width is not None else 1
            return m.group("base"), int(pos), count, bool(width and width.startswith("-"))
    m = _RE_BITSUFFIX.match(token)
    if not m or not m.group(1):
        return token, None, None, False
    width = m.group(3)
    count = abs(int(width)) if width is not None else 1
    return m.group(1), int(m.group(2)), count, bool(width and width.startswith("-"))


def _apply_group_states(script: PltScript) -> None:
    """``//BEGIN_GROUP "g" ENABLED|DISABLED`` enables / disables all signals of the group when read.

    The original lines stay as they are (the group line decides on every read),
    so an unchanged script is still written back byte for byte.
    """
    state: Optional[str] = None
    for item in script.items:
        if isinstance(item, PltGroupBegin):
            state = item.state
        elif isinstance(item, PltGroupEnd):
            state = None
        elif isinstance(item, PltEntry) and state in ("ENABLED", "DISABLED"):
            wanted = state == "ENABLED"
            if item.enabled != wanted:
                item.enabled = wanted
                item.raw_state = item.state()


def parse_plt(text: str, *, path: "Optional[str | Path]" = None) -> PltScript:
    """Parse the text of a PLT script."""
    newline = "\r\n" if "\r\n" in text else "\n"
    script = PltScript(path=Path(path) if path else None, newline=newline)
    current: Optional[PltEntry] = None
    pending: List[PltComment] = []        # comment / blank lines after the current definition
    group: Optional[str] = None
    comment_group: Optional[int] = None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            m = _RE_GROUP.match(stripped)
            if m:
                tail = m.group(2).upper()
                state = "ENABLED" if "ENABLED" in tail else ("DISABLED" if "DISABLED" in tail else None)
                cm = re.search(r"COLOR\s*=\s*(\d+)", tail)
                script.items.append(PltGroupBegin(m.group(1), state, int(cm.group(1)) if cm else None, raw=line))
                group, current, pending = m.group(1), None, []
                continue
            if _RE_GROUP_END.match(stripped):
                script.items.append(PltGroupEnd(raw=line))
                group, current, pending = None, None, []
                continue
            if stripped:
                # //<signalgroup_n> opens a comment group; //<endsignalgroup>
                # and every other comment close it (help, arrange signals).
                sg = _RE_SIGNALGROUP.match(stripped)
                comment_group = int(sg.group(1)) if sg else None
            item = PltComment(line)
            script.items.append(item)
            if current is not None:
                pending.append(item)
            continue
        body, enabled = stripped, True
        if body.startswith("~"):
            enabled, body = False, body[1:].lstrip()
        if body.startswith("+"):
            if current is None:
                script.items.append(PltComment(line))
                continue
            if pending:
                # Blank and comment lines are skipped when reading, so they do not
                # end the definition: they stay between its lines.
                del script.items[-len(pending):]
                current.inner_lines.extend(c.text for c in pending)
                current.raw_lines.extend(c.text for c in pending)
                pending = []
            attrs, comment = _split_comment(body[1:])
            parse_attributes(attrs, current)
            if comment:
                current.comment = comment if not current.comment else current.comment + "  " + comment
            current.raw_lines.append(line)
            current.raw_state = current.state()
            continue
        pending = []
        body, comment = _split_comment(body)
        parts = body.split(None, 1)
        if not parts:
            script.items.append(PltComment(line))
            current = None
            continue
        name, bit_pos, bit_count, signed = split_name(parts[0])
        current = PltEntry(name=name, enabled=enabled, bit_pos=bit_pos, bit_count=bit_count,
                           bit_signed=signed, group=group, comment_group=comment_group, comment=comment)
        if len(parts) > 1:
            parse_attributes(parts[1], current)
        current.raw_lines = [line]
        current.raw_state = current.state()
        script.items.append(current)
    _apply_group_states(script)
    return script


def load_plt(path: "str | Path") -> PltScript:
    """Read a PLT script: UTF-8 (with or without byte order mark), else the system code page."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise PltError(f"{path}: cannot read the view file ({exc.strerror or exc})") from exc
    # Older scripts may use a local code page. Latin-1 is the last resort
    # because it maps every byte, so nothing is lost when the file is saved.
    text = encoding = None
    for candidate in ("utf-8-sig", "mbcs", "latin-1"):
        try:
            text, encoding = raw.decode(candidate), ("utf-8" if candidate == "utf-8-sig" else candidate)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    script = parse_plt(text, path=path)
    script.encoding = encoding
    script.bom = raw.startswith(_BOM)
    return script


def _g(value: float) -> str:
    """``%g`` with a three-digit exponent."""
    text = "%g" % value
    return re.sub(r"e([-+])(\d\d)$", r"e\g<1>0\2", text)


def value_problem(value: str) -> Optional[str]:
    """Why a ``key=`` / ``unit=`` value would not read back unchanged, or ``None``.

    A value runs until the next attribute keyword, so a later word such as
    ``max``, ``2`` or ``bit`` would be read as an attribute of its own.
    """
    if "\n" in value or "\r" in value:
        return "it contains a line break"
    words = value.split()
    for word in words[1:]:
        if _is_keyword(word):
            return f"the word '{word}' would be read as an attribute; use no blanks (e.g. '{'_'.join(words)}')"
    return None


def _safe_value(value: str, what: str) -> str:
    problem = value_problem(value)
    if problem:
        raise PltError(f"{what} '{value}' cannot be written into a view file: {problem}")
    return " ".join(value.split()) if value.strip() != value or "  " in value else value


def format_attributes(entry: PltEntry) -> str:
    """The value changing attributes of ``entry`` as one string (editor text box)."""
    return " ".join(op.text for op in entry.ops)


def format_entry(entry: PltEntry) -> List[str]:
    """Serialize one signal definition to PLT lines."""
    parts = [entry.token + " "]
    view_first = bool(entry.view) and (not entry.ops or not entry.view_pos)
    if view_first:
        parts.append(entry.view)
    if entry.color is not None:
        parts.append(str(entry.color))
    parts.extend(entry.extra)               # before key= / unit=, whose values run to the next keyword
    if entry.key:
        parts.append(f"key={_safe_value(entry.key, 'Display name')}")
    if entry.unit is not None:
        parts.append(f"unit={_safe_value(entry.unit, 'Unit')}")
    if entry.channel_origin:
        parts.append(f"channel={entry.channel_origin}")
    if entry.scmin is not None:
        parts.append(f"scmin={_g(entry.scmin)}")
    if entry.scmax is not None:
        parts.append(f"scmax={_g(entry.scmax)}")
    if entry.scansc:
        parts.append("scansc")
    for index in sorted(entry.refs):
        value = entry.refs[index]
        parts.append(f"ref{index}={_g(value) if isinstance(value, float) else value}")
    if entry.fmt:
        parts.append(f"fmt={entry.fmt}")
    if entry.emphasize:
        parts.append(f"emphasize={entry.emphasize}")
    prefix = "" if entry.enabled else "~ "
    first = prefix + " ".join(parts)
    if entry.comment:
        first += "  " + entry.comment
    lines = [first]
    lines.extend(entry.inner_lines)
    for index, op in enumerate(entry.ops):
        if not view_first and entry.view and entry.view_pos == index:
            lines.append(f"{prefix}+ {entry.view}")
        lines.append(f"{prefix}+ {op.text}")
    if not view_first and entry.view and (entry.view_pos or 0) >= len(entry.ops):
        lines.append(f"{prefix}+ {entry.view}")
    return lines


def dump_plt(script: PltScript) -> str:
    """Text of a script. Unchanged definitions keep their original lines."""
    out: List[str] = []
    for item in script.items:
        if isinstance(item, PltEntry):
            if item.raw_lines and item.raw_state == item.state():
                out.extend(item.raw_lines)
            else:
                out.extend(format_entry(item))
        elif isinstance(item, PltGroupBegin):
            text = f'//BEGIN_GROUP "{item.name}"'
            if item.state:
                text += f" {item.state}"
            if item.color is not None:
                text += f" COLOR={item.color}"
            # Keep the original spelling unless the group was changed.
            if item.raw is not None and _RE_GROUP.match(item.raw.strip()) and _group_equal(item.raw, item):
                text = item.raw
            out.append(text)
        elif isinstance(item, PltGroupEnd):
            out.append(item.raw if item.raw is not None else "//END_GROUP")
        else:
            out.append(item.text)
    return script.newline.join(out) + (script.newline if out else "")


def _group_equal(raw: str, item: PltGroupBegin) -> bool:
    m = _RE_GROUP.match(raw.strip())
    tail = m.group(2).upper()
    state = "ENABLED" if "ENABLED" in tail else ("DISABLED" if "DISABLED" in tail else None)
    cm = re.search(r"COLOR\s*=\s*(\d+)", tail)
    return (m.group(1), state, int(cm.group(1)) if cm else None) == (item.name, item.state, item.color)


def save_plt(script: PltScript, path: "Optional[str | Path]" = None) -> None:
    """Write the script (atomically) and clear its modified flag.

    A script read in a code page that cannot hold its new text (for example a
    unit ``m/s²`` copied into an old script) is switched to UTF-8 instead of
    losing characters.
    """
    target = Path(path) if path is not None else script.path
    if target is None:
        raise PltError("No file name given for the view file")
    text = dump_plt(script)
    encoding = script.encoding or "utf-8"
    try:
        data = text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        encoding = "utf-8"
        data = text.encode(encoding)
    if script.bom and encoding == "utf-8":
        data = _BOM + data
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise PltError(f"{target}: cannot write the view file ({exc.strerror or exc})") from exc
    script.path = target
    script.encoding = encoding
    script.modified = False


def short_name(name: str, *, drop_bus_prefix: bool = False, drop_device: bool = False,
               keep_elements: int = 0, keep_from_left: bool = True) -> str:
    """Shorten a signal's display name using the configured editor options.

    ``drop_bus_prefix``: ``#10.DCW.NAME`` -> ``#NAME`` (signals starting with ``#`` or ``$``).
    ``drop_device``: ``name\\device\\more`` -> ``name``.
    ``keep_elements`` > 0: keep that many dot separated elements from the left or right.
    """
    out = name
    if drop_bus_prefix and out[:1] in "#$" and "." in out:
        out = out[0] + out.rsplit(".", 1)[1]
    if drop_device and "\\" in out:
        out = out.split("\\", 1)[0]
    if keep_elements > 0 and out[:1] not in "#$":
        parts = out.split(".")
        if len(parts) > keep_elements:
            out = ".".join(parts[:keep_elements] if keep_from_left else parts[-keep_elements:])
    return out


def entry_from_signal(info: SignalInfo, *, color: Optional[int] = None,
                      display_name: Optional[str] = None, name: Optional[str] = None) -> PltEntry:
    """A new definition for a catalogue signal, as when it is added in the editor.

    ``name`` is the name written into the script (default: the catalogue name);
    ``display_name`` becomes ``key=`` when it differs from it.
    """
    entry = PltEntry(
        name=name or info.name,
        view="bit" if info.kind == KIND_BIT else "nobit",
        color=color if color is not None else (info.color if info.color is not None else 1),
        unit=info.unit,
        scmin=info.scmin,
        scmax=info.scmax,
        channel_origin=info.channel_origin if info.shared_name else "",
    )
    if display_name and display_name != entry.name:
        entry.key = display_name
    if info.kind == KIND_BIT:
        entry.scmin, entry.scmax = 0.0, 1.0
    return entry
