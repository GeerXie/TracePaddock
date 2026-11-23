"""Application settings: a JSON file in the user's profile, no Qt dependency.

Use ``settings.get("key")`` / ``settings.set("key", value)``. Keys listed in
``DEFAULTS`` are shared between modules; a module may also use private keys
by passing its own default: ``settings.get("editor/geometry", None)``.

The settings, value-mapping and rule files may be edited by hand. A file that
cannot be read is never written over silently: it is kept under a backup name
(see :func:`read_json_object`) and the window tells the user at the start.
"""
from __future__ import annotations

import json
import locale
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .plt.palette import CYCLIC_RANGE, MAX_PALETTE_COLOR

__all__ = ["Settings", "DEFAULTS", "default_settings_path", "data_folder", "read_json_object", "keep_copy",
           "UnreadableFile", "unreadable_files"]


def _default_language() -> str:
    try:
        lang = (locale.getdefaultlocale()[0] or "").lower()
    except Exception:
        lang = ""
    return "zh" if lang.startswith("zh") else "en"


DEFAULTS: Dict[str, Any] = {
    # --- general
    "language": _default_language(),          # "en" | "zh"
    "recent_datasets": [],                    # newest first
    "recent_plts": [],                        # newest first
    "favourite_plts": [],
    "plt_history": [],                        # oldest first; back / next walk through it
    "history_max": 20,
    "stopped_signals": "dashed",            # after the end of a signal that stopped being sent: dashed / hold / cut
    "recent_max": 32,
    "print_white_background": True,           # printing, PDF and picture export on white
    "last_dataset_dir": "",
    "last_plt_dir": "",
    "current_plt": "",                        # PLT script used for the next dataset
    # --- view
    "font_family": "Arial",
    "background": "#000000",
    "invert_bw": True,                        # swap black/white so colour 1 stays visible
    "grid": True,
    "line_width": 1,
    "legend_fraction": 0.757,                 # divider plot | legend (fraction of width)
    "digital_fraction": 0.097,                # divider digital | analog (fraction of height)
    "legend_value_format": "%g",
    "legend_font_size": 11,                   # largest legend font in pixels (rows shrink with many signals)
    "time_labels": False,                     # show the time at the vertical grid lines
    "xy_time_color": True,                    # XY view: light (early) to full colour (late), start / end marks
    "xy_points": False,                       # XY view: points instead of lines
    # --- display names proposed when a signal is added in the editor
    "short_name_bus_prefix": False,           # '#10.DCW.NAME' -> '#NAME'
    "short_name_device": True,                # 'name\\CCP:1' -> 'name'
    "short_name_keep": 0,                     # >0: keep this many dot separated elements
    "short_name_keep_left": True,
    "mark_modified_signals": False,           # prefix processed signals with a mark
    "enum_mode": 0,                           # 0 text (value), 1 text only, 2 value only
    "caption_format": "%V - [%P(%T)]",
    "time_format": "%d.%m.%Y %H:%M:%S",
    "palette": None,                          # None = built-in 32 colour palette, else [[r,g,b],...]
    "cyclic_color_range": list(CYCLIC_RANGE),
    "step_mode_analog": True,                 # hold each sample until the next one
    "statistics_on_common_time_base": True,   # mean and standard deviation on common timestamps
    # --- loading / PLT handling
    "filter_method": 0,                       # 0 wildcard * ?, 1 wildcard * ? %, 2 regular expression
    "filter_case_sensitive": False,
    "filter_history": [],
    "use_current_signal_selection": True,     # keep the current PLT when another dataset is loaded
    "load_disabled_signals": False,           # also read ~ signals (needed for exp(x1=..) on them)
    "ignore_incomplete_computed": False,      # skip computed signals whose inputs are missing
    "calculus_on_nonequidistant": True,       # allow diff/integ/mavr on non equidistant signals
    "strip_device_suffix": False,             # True: list MDF signals as 'name' instead of 'name\\CCP:1' (scripts match either way)
    "arrange_mode": 0,                        # 0 from PLT, 1 equally spaced, 2 postfix, 3 range/unit, 4 comments, 5 groups
}


def data_folder() -> Path:
    """Where settings, value maps, rules and own plug-ins live. The portable program keeps them in
    ``TracePaddockData`` next to TracePaddock.exe (copy the folder and everything goes along); if that folder cannot
    be written (e.g. under Program Files) and when run from the sources: %APPDATA%\\TracePaddock."""
    if getattr(sys, "frozen", False):
        portable = Path(sys.executable).resolve().parent / "TracePaddockData"
        try:
            portable.mkdir(exist_ok=True)
            probe = portable / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return portable
        except OSError:
            pass
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "TracePaddock"


def default_settings_path() -> Path:
    return data_folder() / "settings.json"


# ---------------------------------------------------------------------- files that cannot be read
#: reason of a file whose JSON is valid but not an object (translated when it is shown)
NOT_AN_OBJECT = "the file does not contain a JSON object"


@dataclass
class UnreadableFile:
    """A settings, value-mapping or rule file that could not be read (shown to the user at the start)."""

    path: Path
    #: why: the text of the error
    reason: str
    #: where the original content was kept; ``None``: the file stays as it is and is not written over
    backup: Optional[Path] = None
    #: entries that could not be read while the rest of the file is used (0: the whole file is unusable)
    skipped: int = 0


_unreadable: List[UnreadableFile] = []


def unreadable_files() -> List[UnreadableFile]:
    """The files that could not be read since the last call (each one is reported once)."""
    out = list(_unreadable)
    del _unreadable[:]
    return out


def keep_copy(path: "str | Path", reason: str, *, move: bool, skipped: int = 0) -> Optional[Path]:
    """Keep the content of ``path`` as ``<name>.bad`` (``.bad2`` ... when taken) and report it.

    ``move`` renames the file, for a file whose content is not used at all (a copy is made when the file
    cannot be renamed); otherwise it is copied, because the readable part is used and written again. A backup
    with the same content (kept at an earlier start) is used again. Returns the backup, or ``None`` when no
    backup could be made: then the file must not be written over."""
    path = Path(path)
    try:
        content: Optional[bytes] = path.read_bytes()
    except OSError:
        content = None
    backup: Optional[Path] = path.with_name(path.name + ".bad")
    number = 2
    while backup.exists():
        try:
            if content is not None and backup.read_bytes() == content:
                break                                          # kept already: no second copy
        except OSError:
            pass
        backup, number = path.with_name("%s.bad%d" % (path.name, number)), number + 1
    else:
        try:
            if move:
                try:
                    os.rename(path, backup)
                except OSError:                                # e.g. open in an editor: keep a copy instead
                    shutil.copy2(path, backup)
            else:
                shutil.copy2(path, backup)
        except OSError:
            backup = None
    _unreadable.append(UnreadableFile(path, reason, backup, skipped))
    return backup


def read_json_object(path: "str | Path") -> Tuple[Optional[dict], bool]:
    """Read the JSON object of a file written by the program or edited by hand (UTF-8, with or without BOM).

    Returns ``(data, writable)``. ``data`` is ``None`` when the file does not exist or cannot be used. A file
    with a mistake (or in another encoding) is renamed out of the way by :func:`keep_copy`, so the next save
    does not destroy it; a file that cannot even be opened (locked by another program) is left alone. In both
    cases the user is told. ``writable`` is ``False`` when the file must not be written over."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, True
    except (ValueError, RecursionError) as exc:               # a typo, a GBK file, a truncated file ...
        return None, keep_copy(path, str(exc), move=True) is not None
    except OSError as exc:
        _unreadable.append(UnreadableFile(path, exc.strerror or str(exc)))
        return None, False
    if not isinstance(data, dict):
        return None, keep_copy(path, NOT_AN_OBJECT, move=True) is not None
    return data, True


class Settings:
    """Dictionary backed settings with defaults and JSON persistence."""

    def __init__(self, path: "Optional[str | Path]" = None, *, autoload: bool = True) -> None:
        self.path = Path(path) if path is not None else default_settings_path()
        self._data: Dict[str, Any] = {}
        #: False when the file could not be read and must not be written over
        self._writable = True
        if autoload:
            self.load()

    def load(self) -> None:
        data, self._writable = read_json_object(self.path)
        self._data = data if data is not None else {}

    def save(self) -> None:
        if not self._writable:
            return                                             # the file could not be read: keep it as it is
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass  # settings are a convenience; never fail the application over them

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._data[key]
        if key in DEFAULTS:
            value = DEFAULTS[key]
            return list(value) if isinstance(value, list) else value
        return default

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def set_default(self, key: str, value: Any) -> None:
        self._data.setdefault(key, value)

    def reset(self, key: str) -> None:
        self._data.pop(key, None)

    # Typed access: the file can be edited by hand, so a value of the wrong
    # type or outside its range gives the default instead of an error.
    def number(self, key: str, default: float, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
        try:
            value = float(self.get(key, default))
        except (TypeError, ValueError):
            return default
        if value != value:
            return default
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def text_list(self, key: str) -> List[str]:
        value = self.get(key, [])
        return [str(v) for v in value if isinstance(v, (str, int, float))] if isinstance(value, list) else []

    def color_range(self) -> Tuple[int, int]:
        """Inclusive range of visible colour numbers used for automatic colours."""
        value = self.get("cyclic_color_range", CYCLIC_RANGE)
        try:
            lo, hi = int(value[0]), int(value[1])
        except (TypeError, ValueError, OverflowError, IndexError, KeyError):
            return CYCLIC_RANGE
        lo = max(1, min(MAX_PALETTE_COLOR, lo))
        hi = max(1, min(MAX_PALETTE_COLOR, hi))
        return (lo, hi) if lo <= hi else (hi, lo)

    def push_recent(self, key: str, path: "str | Path") -> None:
        """Move ``path`` to the front of the list stored under ``key``."""
        p = str(path)
        items = [x for x in self.text_list(key) if os.path.normcase(x) != os.path.normcase(p)]
        items.insert(0, p)
        self.set(key, items[: int(self.number("recent_max", 32, 1, 100))])
