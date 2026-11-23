"""Minimal UI translation: English source strings, optional Chinese table.

Wrap every user visible string in ``tr("...")``. The English text is the key;
``tracepaddock/i18n_zh.py`` holds ``ZH = {"English text": "中文"}``. Unknown
strings fall back to English, so a missing translation never breaks the UI.
"""
from __future__ import annotations

from typing import Dict

__all__ = ["tr", "set_language", "language"]

_language = "en"
_table: Dict[str, str] = {}


def set_language(lang: str) -> None:
    """Select ``"en"`` or ``"zh"``. Takes effect for strings translated afterwards."""
    global _language, _table
    _language = "zh" if str(lang).lower().startswith("zh") else "en"
    _table = {}
    if _language == "zh":
        try:
            from .i18n_zh import ZH  # type: ignore
            _table = dict(ZH)
        except Exception:
            _table = {}


def language() -> str:
    return _language


def tr(text: str) -> str:
    """Translate ``text`` (English) into the selected language."""
    return _table.get(text, text)
