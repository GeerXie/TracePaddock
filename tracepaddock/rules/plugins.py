"""Rule plug-ins: Python files that build rules from a few parameters.

A plug-in file sets ``NAME``, ``DESCRIPTION`` and ``PARAMETERS`` (plain
literals, read here without running the file) and defines
``run(api, params)``, which is executed by the separate runner process.
Each parameter is a dict with ``name``, ``label``, ``type`` (``signal``,
``signals``, ``number``, ``text``, ``choice``, ``bool``) and ``default``
(``choices`` for ``choice``).

Plug-ins come from ``tracepaddock/plugins`` (examples shipped with the viewer)
and from the user's folder next to the settings file. The rule list refers to
them with :func:`plugin_ref` (``builtin:<file>`` / ``user:<file>``), so the
references survive moving or updating the program folder.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..i18n import tr

__all__ = ["Plugin", "discover", "builtin_folder", "user_folder", "read_plugin", "plugin_ref", "resolve_plugin"]

PARAMETER_TYPES = ("signal", "signals", "number", "text", "choice", "bool")

_BUILTIN, _USER = "builtin:", "user:"


@dataclass
class Plugin:
    path: Path
    name: str
    description: str = ""
    parameters: List[Dict[str, object]] = field(default_factory=list)
    problem: Optional[str] = None

    def defaults(self) -> Dict[str, object]:
        return {str(p["name"]): p.get("default") for p in self.parameters}


def builtin_folder() -> Path:
    return Path(__file__).resolve().parents[1] / "plugins"


def user_folder() -> Path:
    from ..settings import data_folder
    return data_folder() / "plugins"


def _same_folder(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def plugin_ref(path: "str | Path") -> str:
    """How the rule list refers to a plug-in file: ``builtin:<file>`` for a plug-in shipped with the program,
    ``user:<file>`` for the user's plug-in folder, the absolute path for any other file.

    Absolute paths written before the program folder was moved or updated (or by a copy of the program in
    another folder) are turned into the same form: a shipped plug-in (``.../tracepaddock/plugins/<file>``) is
    the one of this program, a file that no longer exists is looked for by its name."""
    text = str(path or "")
    if not text or text.startswith((_BUILTIN, _USER)):
        return text
    p = Path(text)
    builtin, user = builtin_folder(), user_folder()
    if _same_folder(p.parent, builtin):
        return _BUILTIN + p.name
    if _same_folder(p.parent, user):
        return _USER + p.name
    shipped = p.parent.name == "plugins" and p.parent.parent.name == "tracepaddock"
    if shipped and (builtin / p.name).is_file():
        return _BUILTIN + p.name
    if not p.is_file():
        places = [(_BUILTIN, builtin), (_USER, user)]
        for prefix, folder in (places if shipped else places[::-1]):
            if (folder / p.name).is_file():
                return prefix + p.name
    return text


def resolve_plugin(ref: "str | Path") -> Path:
    """The file a plug-in reference (or an old absolute path) stands for in this program."""
    text = plugin_ref(ref)
    if text.startswith(_BUILTIN):
        return builtin_folder() / Path(text[len(_BUILTIN):]).name
    if text.startswith(_USER):
        return user_folder() / Path(text[len(_USER):]).name
    return Path(text)


def read_plugin(path: "str | Path") -> Plugin:
    """Read the metadata of a plug-in without executing it."""
    p = Path(path)
    plugin = Plugin(p, p.stem)
    try:
        tree = ast.parse(p.read_bytes(), str(p))            # bytes: a BOM and a coding line are honoured
    except (OSError, SyntaxError, ValueError) as exc:
        plugin.problem = tr("cannot be read: %s") % exc
        return plugin
    has_run = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            has_run = True
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ("NAME", "DESCRIPTION", "PARAMETERS"):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError, MemoryError, RecursionError):
                    plugin.problem = tr("%s must be a plain literal") % key
                    continue
                if key == "NAME":
                    plugin.name = str(value)
                elif key == "DESCRIPTION":
                    plugin.description = str(value)
                elif not isinstance(value, (list, tuple)):
                    plugin.problem = tr("PARAMETERS must be a list")
                else:
                    plugin.parameters = [dict(v) for v in value if isinstance(v, dict) and "name" in v]
    for par in plugin.parameters:
        if par.get("type", "text") not in PARAMETER_TYPES:
            plugin.problem = tr("parameter %s has an unknown type %s") % (par.get("name"), par.get("type"))
    if not has_run and plugin.problem is None:
        plugin.problem = tr("no function run(api, params)")
    return plugin


def discover(extra: Optional[List[Path]] = None) -> List[Plugin]:
    """All plug-ins of the built-in and the user folder (and ``extra`` folders), sorted by name."""
    out: List[Plugin] = []
    for folder in [builtin_folder(), user_folder()] + list(extra or []):
        if folder.is_dir():
            for path in sorted(folder.glob("*.py")):
                if not path.name.startswith("_"):
                    try:
                        out.append(read_plugin(path))
                    except Exception as exc:                # one broken file must not hide the others
                        out.append(Plugin(path, path.stem, problem=tr("cannot be read: %s") % exc))
    return sorted(out, key=lambda p: p.name.lower())
