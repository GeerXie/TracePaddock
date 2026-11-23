"""In-memory model of a PLT script (the signal selection / layout file).

A PLT script is a text file with one signal definition per line::

    [~] <name>[:<bit>[:[-]<width>]]  [bit|nobit] [<colour>] [key=..] [unit=..] [scmin=..] [scmax=..] ...  [// comment]
    [~] + <more attributes>            (continuation of the previous definition)
    // comment                         (some comments carry meaning, see below)

``~`` disables a definition. Attributes are processed in the order written;
the value changing ones (``exp=``, ``dgf=``, ``diff`` ...) are kept as an
ordered list of :class:`Op`.

The parser (``tracepaddock.plt.parser``) produces a :class:`PltScript`; the writer
turns it back into PLT text. Unknown attributes are preserved when loading
and saving a script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

__all__ = [
    "Op",
    "PltEntry",
    "PltComment",
    "PltGroupBegin",
    "PltGroupEnd",
    "PltItem",
    "PltScript",
    "OP_KINDS",
    "BIT_DELETING_OPS",
]

#: Value changing operations, in the spelling used by ``Op.kind``.
OP_KINDS = (
    "exp",      # exp=<expr of x,t>  or  exp(x1=<name>,..)=<expr of x,x1..x9,t>
    "dgf",      # dgf=bth|chb1|chb2:<fc>:<order>[:<ripple>]  /  dgf=pdt1:..  /  dgf=pdt2:..
    "min",      # constant: minimum of the signal
    "max",      # constant: maximum of the signal
    "-mean",    # subtract mean
    "-min",     # subtract minimum
    "-max",     # subtract maximum
    "diff",     # derivative
    "integ",    # integral
    "mavr",     # mavr(<n>) moving average, n>0 forward, n<0 backward
    "shift",    # shift(<n>) by n samples
    "revert",   # reverse in time
)

#: Operations that "implicitly delete the BIT option" (help, PLT attributes).
#: REVERT, MIN and MAX are not in that list: a reversed or constant one-bit
#: signal is still a one-bit signal.
BIT_DELETING_OPS = frozenset({"exp", "dgf", "-mean", "-min", "-max", "diff", "integ", "mavr", "shift"})

_RE_VIDEO = re.compile(r"^\s*//\s*VIDEO_SYNCH\s*=\s*(.*)$", re.I)


@dataclass
class Op:
    """One value changing attribute of a signal definition."""

    kind: str
    #: The attribute exactly as written, e.g. ``"exp(x1=vf)=x-x1"`` or ``"mavr(10)"``.
    text: str
    #: ``exp`` only: the expression right of the ``=``.
    expr: str = ""
    #: ``exp`` only: symbol -> signal name, e.g. ``{"x1": "vf"}``.
    refs: Dict[str, str] = field(default_factory=dict)
    #: Parsed arguments. ``dgf``: ``(type_upper, float, ...)``; ``mavr``/``shift``: ``(int,)``.
    args: Tuple = ()


@dataclass
class PltEntry:
    """One signal definition (possibly spread over continuation lines)."""

    #: Signal name as written, without the ``:bit:width`` part. ``"*"`` is a
    #: placeholder for a signal that exists only through its ``exp=``.
    name: str = ""
    #: ``False`` when the definition is prefixed with ``~``.
    enabled: bool = True
    #: Colour index into the palette, ``None`` = not given (default 1).
    color: Optional[int] = None
    #: ``"bit"`` = DIGITAL window, ``"nobit"`` = analog (FLOAT) window,
    #: ``None`` = not given (decided by the signal type).
    view: Optional[str] = None
    #: Number of value changing operations written before the ``bit``/``nobit``
    #: keyword (``None`` = the keyword came before all of them). Attributes are
    #: processed in the order written: ``exp=.. bit`` shows the result in the
    #: DIGITAL window, ``bit exp=..`` does not (the expression deletes BIT).
    view_pos: Optional[int] = None
    #: Bit extraction from the ``name:pos[:width]`` form; ``None`` = none.
    bit_pos: Optional[int] = None
    bit_count: Optional[int] = None
    bit_signed: bool = False
    #: ``key=`` display name override.
    key: Optional[str] = None
    #: ``unit=`` override (may be the empty string, which is different from ``None``).
    unit: Optional[str] = None
    scmin: Optional[float] = None
    scmax: Optional[float] = None
    #: ``scansc``: automatic scaling from the signal's minimum and maximum.
    scansc: bool = False
    #: Reference lines: index (0..2) -> number or ``"MIN"`` / ``"MAX"`` / ``"MEAN"``.
    refs: Dict[int, Union[float, str]] = field(default_factory=dict)
    #: ``fmt=`` number format such as ``".2f"``, ``"8.3e"``, ``"x"``, ``"b"``.
    fmt: Optional[str] = None
    #: ``emphasize=`` 0 none, 1 emphasized, 2 emphasized and focused.
    emphasize: int = 0
    #: Ordered value changing operations.
    ops: List[Op] = field(default_factory=list)
    #: Attribute tokens the parser did not recognise, kept verbatim for writing.
    extra: List[str] = field(default_factory=list)
    #: Exact MDF channel selection, when more than one recorded channel shares the name.
    channel_origin: str = ""
    #: Comment written behind the definition, starting with ``//``.
    comment: Optional[str] = None
    #: Comment or blank lines that stood between the definition and its
    #: continuation lines; written back between them.
    inner_lines: List[str] = field(default_factory=list)
    #: Name of the enclosing ``//BEGIN_GROUP`` or ``None``. Set by the parser.
    group: Optional[str] = None
    #: Number ``n`` of the enclosing ``//<signalgroup_n>`` comment group or ``None``.
    comment_group: Optional[int] = None
    #: The text lines this definition was parsed from and a snapshot of its
    #: fields at that moment. While the fields still equal the snapshot the
    #: writer emits the original lines, so untouched definitions keep their
    #: exact spelling and layout.
    raw_lines: List[str] = field(default_factory=list, compare=False, repr=False)
    raw_state: Optional[Tuple] = field(default=None, compare=False, repr=False)

    def state(self) -> Tuple:
        """All fields that end up in the file, as one comparable value."""
        return (
            self.name, self.enabled, self.color, self.view, self.view_pos, self.bit_pos, self.bit_count,
            self.bit_signed, self.key, self.unit, self.scmin, self.scmax, self.scansc,
            tuple(sorted(self.refs.items())), self.fmt, self.emphasize,
            tuple(op.text for op in self.ops), tuple(self.extra), self.channel_origin, self.comment, tuple(self.inner_lines),
        )

    @property
    def token(self) -> str:
        """The signal name as written in the script, including a ``:pos[:width]`` bit suffix."""
        if self.bit_pos is None:
            return self.name
        text = f"{self.name}:{self.bit_pos}"
        if self.bit_signed or (self.bit_count or 1) != 1:
            text += f":{'-' if self.bit_signed else ''}{self.bit_count or 1}"
        return text

    @property
    def display_name(self) -> str:
        """Name shown in the legend: ``key=`` if given, else the signal name with its bit suffix."""
        return self.key if self.key else self.token

    @property
    def is_placeholder(self) -> bool:
        return self.name == "*"

    @property
    def modifies_values(self) -> bool:
        return bool(self.ops)

    def referenced_names(self) -> List[str]:
        """Signal names referenced from ``exp(x1=..)`` operations."""
        out: List[str] = []
        for op in self.ops:
            out.extend(op.refs.values())
        return out


@dataclass
class PltComment:
    """A comment or blank line, kept verbatim (without the line ending)."""

    text: str = ""


@dataclass
class PltGroupBegin:
    """``//BEGIN_GROUP "<name>" [ENABLED|DISABLED] [COLOR=<n>]``."""

    name: str = ""
    #: ``"ENABLED"``, ``"DISABLED"`` or ``None`` (group partly visible).
    state: Optional[str] = None
    color: Optional[int] = None
    #: The line as read, written back while name, state and colour are unchanged.
    raw: Optional[str] = field(default=None, compare=False, repr=False)


@dataclass
class PltGroupEnd:
    """``//END_GROUP``."""

    #: The line as read (other spellings and a comment behind it are kept).
    raw: Optional[str] = field(default=None, compare=False, repr=False)


PltItem = Union[PltEntry, PltComment, PltGroupBegin, PltGroupEnd]


@dataclass
class PltScript:
    """A whole PLT script in file order."""

    items: List[PltItem] = field(default_factory=list)
    path: Optional[Path] = None
    #: Write UTF-8 for new files and preserve the encoding of loaded files.
    encoding: str = "utf-8"
    #: The file started with a UTF-8 byte order mark (written back the same way).
    bom: bool = False
    newline: str = "\r\n"
    #: Set by the UI when the user changed something that is not saved yet.
    modified: bool = False

    def entries(self) -> Iterator[PltEntry]:
        """All signal definitions, enabled or not, in file order."""
        return (it for it in self.items if isinstance(it, PltEntry))

    def enabled_entries(self) -> Iterator[PltEntry]:
        return (it for it in self.items if isinstance(it, PltEntry) and it.enabled)

    def groups(self) -> List[PltGroupBegin]:
        return [it for it in self.items if isinstance(it, PltGroupBegin)]

    def find(self, display_name: str) -> Optional[PltEntry]:
        """First definition whose display name (``key`` or name) equals ``display_name``."""
        for ent in self.entries():
            if ent.display_name == display_name:
                return ent
        return None

    def video_synch_signals(self) -> List[str]:
        """Signal names given by ``//VIDEO_SYNCH=<name>[;<name>]...`` comments, in order."""
        names: List[str] = []
        for item in self.items:
            if isinstance(item, PltComment):
                m = _RE_VIDEO.match(item.text)
                if m:
                    names.extend(n.strip() for n in m.group(1).split(";") if n.strip())
        return names

    def signal_groups(self) -> Dict[int, List[PltEntry]]:
        """Definitions grouped by their ``//<signalgroup_n>`` comment group number."""
        out: Dict[int, List[PltEntry]] = {}
        for ent in self.entries():
            if ent.comment_group is not None:
                out.setdefault(ent.comment_group, []).append(ent)
        return out

    def __len__(self) -> int:
        return sum(1 for _ in self.entries())
