"""Window layouts: which panels are shown, chosen with one click (like the layout presets of VS Code).

A preset names the panels it shows; the user can save the current arrangement
(dock positions, sizes, floating panels, tool bar, status bar) under a name.
The program starts with the layout chosen under "When the program starts" -
by default only the signals.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = ["PRESETS", "PRESET_KEYS", "DEFAULT_STARTUP", "LAST", "preset_parts"]

#: (key, menu text, shortcut, panels shown)
PRESETS: Tuple[Tuple[str, str, str, Dict[str, bool]], ...] = (
    ("signals", "Signals only (default)", "Ctrl+Shift+1",
     {"rules": False, "images": False, "toolbar": True, "statusbar": True}),
    ("rules", "Signals and rules / events", "Ctrl+Shift+2",
     {"rules": True, "images": False, "toolbar": True, "statusbar": True}),
    ("images", "Signals and image panel", "Ctrl+Shift+3",
     {"rules": False, "images": True, "toolbar": True, "statusbar": True}),
    ("review", "Rule review: signals, rules / events and image panel", "Ctrl+Shift+4",
     {"rules": True, "images": True, "toolbar": True, "statusbar": True}),
    ("focus", "Focus: only the signals (no tool bar, no status bar)", "Ctrl+Shift+5",
     {"rules": False, "images": False, "toolbar": False, "statusbar": False}),
)
PRESET_KEYS: List[str] = [p[0] for p in PRESETS]
DEFAULT_STARTUP = "signals"
#: start with the layout the window had when it was closed
LAST = "last"


def preset_parts(key: str) -> Dict[str, bool]:
    for k, _text, _shortcut, parts in PRESETS:
        if k == key:
            return dict(parts)
    return dict(PRESETS[0][3])
