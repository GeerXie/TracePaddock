"""Look of the program: colours of the window and of the plot, signal colours, fonts, legend style and -
in the cute theme - a picture behind the curves, mouse pointers and decorations of the menu bar.

"classic" uses the system window style, icon tool bar, the 32-colour palette, black plot, Arial legend
with coloured name fields filling the legend height, dotted 10 x 10 grid, no time labels, clock in the status bar. The signal colours of the other themes keep the meaning of the PLT colour numbers (2 = red,
3 = green, 4 = yellow, 5 = blue ...) with tones chosen for their background; the runs 2..7 and 10..15 were
checked with the data-visualisation palette validator (OKLab distance >= 8 between neighbours under
protan / deutan simulation, >= 15 for normal vision, >= 3:1 contrast against the plot).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..plt.palette import DEFAULT_PALETTE

__all__ = ["Theme", "THEMES", "THEME_KEYS", "get_theme", "apply_app_theme", "theme_palette", "hex_rgb",
           "visible_themes", "DEFAULT_THEME", "PROTECTED_THEME", "plot_palette", "apply_plot_colours"]

#: First launch uses Obsidian. Keep Classic available for the system window style.
DEFAULT_THEME = "obsidian"
PROTECTED_THEME = "classic"

RGB = Tuple[int, int, int]


def hex_rgb(text: str) -> RGB:
    text = text.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


@dataclass(frozen=True)
class Theme:
    key: str
    name: str                                     # English menu text (translated with tr)
    dark: bool = False
    # ---- window
    window: str = ""                              # "" = system look (classic)
    bar: str = ""                                 # menu bar, tool bar, status bar (colour or Qt gradient)
    panel: str = ""
    base: str = ""
    alt_base: str = ""
    border: str = ""
    text: str = ""
    text_dim: str = ""
    accent: str = ""
    accent_text: str = "#ffffff"
    selection: str = ""
    hover: str = ""
    radius: int = 6
    font_ui: str = ""                             # "" = keep the application font
    font_bar: str = ""                            # menu bar / tool bar font ("" = font_ui)
    # ---- plot
    plot: str = ""                                # "" = background from the settings
    legend_bg: str = ""
    grid: str = ""
    axis_text: str = ""
    cursor: str = ""
    marker: str = ""
    legend_style: str = "block"                   # "block": coloured name field; "chip": colour mark + text
    font_values: str = ""                         # legend values (fixed digits), "" = legend font
    #: signal colours 2..7 and 10..15 (red, green, yellow, blue, magenta, cyan; red, green, brown, blue,
    #: magenta, cyan), ink (1) and the two greys (8, 9)
    run1: Sequence[str] = ()
    run2: Sequence[str] = ()
    ink: str = ""
    greys: Sequence[str] = ()
    state_fills: Sequence[str] = ()
    # ---- classic look: icons in the tool bar, legend rows filling the height, small legend font, no time
    #      labels under the grid, a clock in the status bar
    toolbar_icons: bool = False
    legend_fill: bool = False
    legend_font_px: int = 13
    font_legend: str = ""
    time_labels: bool = True
    clock: bool = False
    #: the legend shows the values under the mouse while it is over the plot (no reference line needed)
    hover_values: bool = False
    #: without cursor the legend shows each signal's range ("0 - 150 [15/div]"); off: the value column stays empty
    legend_idle_ranges: bool = True
    # ---- decorations
    backdrop: str = ""                            # default picture behind the curves (scene key)
    backdrop_opacity: float = 0.0
    cursor_character: str = ""                    # default mouse pointer character
    bar_pattern: str = ""                         # decoration of menu bar / tool bar: "racing"


THEMES: Dict[str, Theme] = {
    "classic": Theme("classic", "Classic", toolbar_icons=True, legend_fill=True, legend_font_px=11,
                     font_legend="Arial", time_labels=False, clock=True, hover_values=True,
                     legend_idle_ranges=False),
    "daylight": Theme(
        "daylight", "Daylight (light)", dark=False,
        window="#f3f4f6", bar="#f3f4f6", panel="#f8f9fa", base="#ffffff", alt_base="#f6f8fa", border="#d8dce1",
        text="#1f2328", text_dim="#59636e", accent="#0b6bcb", selection="#d3e5fb", hover="#e6e9ed",
        font_ui="Microsoft YaHei UI",
        plot="#ffffff", legend_bg="#f8f9fa", grid="#e3e7eb", axis_text="#6e7781", cursor="#1f2328", marker="#e8590c",
        legend_style="chip", font_values="Cascadia Mono",
        run1=("#e64535", "#13936b", "#b96a16", "#0055c6", "#a94676", "#1f93b8"),
        run2=("#cc490d", "#10853e", "#7c4100", "#7667e2", "#7c2477", "#029180"),
        ink="#1f2328", greys=("#6e7781", "#9ea6af"),
        state_fills=("#dce5ef", "#f5d8cb", "#dbe8df", "#edcfcc", "#e6ddeb", "#f1e1cb", "#d2e6e3", "#ebcfba",
                     "#ead8df", "#dbe2f1", "#f2decf", "#e1e9d5")),
    "obsidian": Theme(
        "obsidian", "Obsidian (default)", dark=True,
        window="#202225", bar="#202225", panel="#26282c", base="#1b1d21", alt_base="#222429", border="#34373d",
        text="#e3e5e8", text_dim="#9aa0a8", accent="#4c8dff", selection="#2d4f86", hover="#2f3237",
        font_ui="Microsoft YaHei UI",
        plot="#1b1d21", legend_bg="#202225", grid="#2c3036", axis_text="#8b919a", cursor="#e3e5e8", marker="#f0a030",
        legend_style="chip", font_values="Cascadia Mono", toolbar_icons=True, hover_values=True,
        run1=("#ec544b", "#009d7b", "#c18417", "#007ae0", "#ac4e87", "#01a2c5"),
        run2=("#bb3f00", "#24ae56", "#9d5d03", "#8a81ef", "#a2419d", "#009c89"),
        ink="#e3e5e8", greys=("#7d8590", "#b1b8c0"),
        state_fills=("#405062", "#714d40", "#385b50", "#673f46", "#534764", "#64553f", "#35595b", "#73493e",
                     "#624951", "#43506a", "#6a4d45", "#4d5941")),
    "racing": Theme(
        "racing", "Italian racing (red / carbon)", dark=True,
        window="#17171a",
        bar="qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d1d21, stop:0.75 #232327, stop:1 #2e1418)",
        panel="#1f1f23", base="#141416", alt_base="#1a1a1d", border="#34343a",
        text="#ececef", text_dim="#a0a0a8", accent="#d7263d", accent_text="#ffffff", selection="#5a1a22",
        hover="#2a2a30", radius=4, font_ui="Microsoft YaHei UI", font_bar="Bahnschrift",
        plot="#121214", legend_bg="#18181b", grid="#26262b", axis_text="#8d8d96", cursor="#ececef", marker="#ffcc00",
        legend_style="chip", font_values="Cascadia Mono",
        run1=("#ec544b", "#009d7b", "#c18417", "#007ae0", "#ac4e87", "#01a2c5"),
        run2=("#bb3f00", "#24ae56", "#9d5d03", "#8a81ef", "#a2419d", "#009c89"),
        ink="#ececef", greys=("#7d7d86", "#b4b4bc"),
        state_fills=("#3a4a5c", "#7a5a12", "#2f5f3f", "#8a2a32", "#4a3f5c", "#7a6a1c", "#1f5a52", "#8a4a1c",
                     "#5c3a42", "#3b4a6a", "#6a4a32", "#3f5a2c"),
        backdrop="racing", backdrop_opacity=0.22, cursor_character="helmet", bar_pattern="racing"),
}
THEME_KEYS: List[str] = list(THEMES)


def get_theme(key: Optional[str]) -> Theme:
    return THEMES.get(str(key or DEFAULT_THEME), THEMES[PROTECTED_THEME])


def visible_themes(hidden) -> List[str]:
    """Theme keys the user kept (``hidden``: keys of deleted themes); the classic theme is always kept."""
    gone = {str(k) for k in (hidden or []) if str(k) != PROTECTED_THEME}
    return [k for k in THEME_KEYS if k not in gone]


def theme_palette(theme: Theme) -> List[RGB]:
    """The 32-entry palette (indexed by PLT colour number) of a theme."""
    if not theme.run1:
        return list(DEFAULT_PALETTE)
    out = list(DEFAULT_PALETTE)
    out[0] = hex_rgb(theme.plot)
    out[1] = hex_rgb(theme.ink)
    for k, value in enumerate(theme.run1):
        out[2 + k] = hex_rgb(value)
    out[8], out[9] = hex_rgb(theme.greys[0]), hex_rgb(theme.greys[1])
    for k, value in enumerate(theme.run2):
        out[10 + k] = hex_rgb(value)
    if not theme.dark:                             # the pastel range is too light on a light plot: darker
        out[16:] = [tuple(int(c * 0.62) for c in rgb) for rgb in DEFAULT_PALETTE[16:]]
    return out


#: the built-in themes whose signal colours are used on a plot background that does not suit a theme
#: (dark background: True, light background: False)
_SIGNAL_COLOURS_FOR = {True: "obsidian", False: "daylight"}


def _plot_sources(theme: Theme, dark: bool) -> Tuple[Theme, Theme]:
    """``(signal colours, plot colours)`` of ``theme`` on a dark / light plot background: its own colours on a
    background that suits it. On one that does not (a dark background chosen in a light theme, a light one in
    a dark theme, a picture printed on white) the signal colours of the built-in theme made for that
    background, and legend, text, grid, cursor and marker derived from the background as in the classic look."""
    if not theme.run1 or theme.dark == dark:
        return theme, theme
    return THEMES[_SIGNAL_COLOURS_FOR[dark]], THEMES[PROTECTED_THEME]


def plot_palette(theme: Theme, dark: bool) -> List[RGB]:
    """The 32-entry palette ``theme`` draws the signals with on a dark / light plot background."""
    return theme_palette(_plot_sources(theme, dark)[0])


def apply_plot_colours(state, theme: Theme) -> None:
    """Fill the theme colours of a view state (``render.ViewState``, its background must be set) for that
    background: signal colours, state bands, legend, text, grid, time labels, cursor and marker (``None`` =
    derived from the background). ``state.theme`` remembers the theme, so a picture on another background
    (on white) can be coloured for it again."""
    signals, plot = _plot_sources(theme, state.dark)

    def rgb(value: str) -> Optional[RGB]:
        return hex_rgb(value) if value else None

    state.theme = theme.key
    state.palette = theme_palette(signals)
    state.state_fills = [hex_rgb(c) for c in signals.state_fills] or None
    state.legend_bg, state.text_color = rgb(plot.legend_bg), rgb(plot.text)
    state.grid_color, state.axis_color = rgb(plot.grid), rgb(plot.axis_text)
    state.cursor_color, state.marker_color = rgb(plot.cursor), rgb(plot.marker)


# ---------------------------------------------------------------------- window style
_CLASSIC: Dict[str, object] = {}


def _qpalette(t: Theme) -> QtGui.QPalette:
    p = QtGui.QPalette()
    c = QtGui.QColor
    roles = {
        QtGui.QPalette.Window: t.window, QtGui.QPalette.WindowText: t.text, QtGui.QPalette.Base: t.base,
        QtGui.QPalette.AlternateBase: t.alt_base, QtGui.QPalette.ToolTipBase: t.panel,
        QtGui.QPalette.ToolTipText: t.text, QtGui.QPalette.Text: t.text, QtGui.QPalette.Button: t.panel,
        QtGui.QPalette.ButtonText: t.text, QtGui.QPalette.BrightText: "#ff4d4f", QtGui.QPalette.Highlight: t.accent,
        QtGui.QPalette.HighlightedText: t.accent_text, QtGui.QPalette.Link: t.accent,
        QtGui.QPalette.PlaceholderText: t.text_dim, QtGui.QPalette.Light: t.panel, QtGui.QPalette.Midlight: t.border,
        QtGui.QPalette.Mid: t.border, QtGui.QPalette.Dark: t.border, QtGui.QPalette.Shadow: t.border,
    }
    for role, value in roles.items():
        p.setColor(role, c(value))
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, c(t.text_dim))
    return p


def _stylesheet(t: Theme, assets: Dict[str, str]) -> str:
    r = t.radius
    bar_font = f'font-family: "{t.font_bar}"; font-size: 10pt;' if t.font_bar else ""
    pattern = ""
    if "bar" in assets:
        repeat = "repeat-x" if assets.get("repeat") == "1" else "no-repeat"
        where = "left center" if repeat == "repeat-x" else "right center"
        pattern = f'background-image: url("{assets["bar"]}"); background-repeat: {repeat}; background-position: {where};'
    return f"""
QMainWindow, QDialog {{ background: {t.window}; }}
QMenuBar {{ background-color: {t.bar}; {pattern} border-bottom: 1px solid {t.border}; padding: 3px 6px; {bar_font} }}
QMenuBar QMenu {{ font-family: "{t.font_ui or 'Microsoft YaHei UI'}"; font-size: 9pt; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: {r}px; background: transparent; color: {t.text}; }}
QMenuBar::item:selected {{ background: {t.hover}; }}
QMenuBar::item:pressed {{ background: {t.selection}; }}
QMenu {{ background: {t.panel}; border: 1px solid {t.border}; border-radius: {r}px; padding: 6px; }}
QMenu::item {{ padding: 5px 26px 5px 24px; border-radius: {max(4, r - 2)}px; color: {t.text}; }}
QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
QMenu::item:disabled {{ color: {t.text_dim}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 5px 8px; }}
QToolBar {{ background-color: {t.bar}; border: none; border-bottom: 1px solid {t.border}; spacing: 1px; padding: 2px 4px; }}
QToolBar::separator {{ width: 1px; background: {t.border}; margin: 5px 6px; }}
QToolButton {{ padding: 3px 6px; border-radius: {r}px; border: 1px solid transparent; color: {t.text}; }}
QToolButton:hover {{ background: {t.hover}; border-color: {t.border}; }}
QToolButton:checked, QToolButton:pressed {{ background: {t.selection}; border-color: {t.accent}; }}
QToolButton:disabled {{ color: {t.text_dim}; }}
QStatusBar {{ background-color: {t.bar}; border-top: 1px solid {t.border}; color: {t.text_dim}; }}
QStatusBar QLabel {{ color: {t.text_dim}; }}
QDockWidget {{ color: {t.text}; }}
QDockWidget::title {{ background: {t.panel}; padding: 6px 8px; border-bottom: 1px solid {t.border}; }}
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: {r}px; top: -1px; background: {t.panel}; }}
QTabBar::tab {{ padding: 6px 14px; margin-right: 2px; color: {t.text_dim}; background: transparent;
               border: 1px solid transparent; border-top-left-radius: {r}px; border-top-right-radius: {r}px; }}
QTabBar::tab:selected {{ color: {t.text}; background: {t.panel}; border-color: {t.border}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:hover {{ color: {t.text}; }}
QPushButton {{ padding: 4px 10px; border-radius: {r}px; border: 1px solid {t.border}; background: {t.panel}; color: {t.text}; }}
QPushButton:hover {{ border-color: {t.accent}; background: {t.hover}; }}
QPushButton:pressed {{ background: {t.selection}; }}
QPushButton:default {{ background: {t.accent}; color: {t.accent_text}; border-color: {t.accent}; }}
QPushButton:disabled {{ color: {t.text_dim}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {t.base}; border: 1px solid {t.border}; border-radius: {max(4, r - 2)}px; padding: 3px 6px;
    color: {t.text}; selection-background-color: {t.selection}; selection-color: {t.text}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {t.accent}; }}
QComboBox QAbstractItemView {{ background: {t.panel}; border: 1px solid {t.border}; selection-background-color: {t.selection};
    selection-color: {t.text}; }}
QHeaderView::section {{ background: {t.panel}; color: {t.text}; border: none; border-bottom: 1px solid {t.border};
    border-right: 1px solid {t.border}; padding: 4px 6px; font-weight: 600; }}
QTableView, QTreeView, QListView, QTableWidget, QTreeWidget, QListWidget {{
    background: {t.base}; alternate-background-color: {t.alt_base}; border: 1px solid {t.border}; border-radius: {r}px;
    gridline-color: {t.border}; color: {t.text}; selection-background-color: {t.selection}; selection-color: {t.text}; }}
QScrollBar:vertical {{ width: 11px; background: transparent; margin: 2px; }}
QScrollBar:horizontal {{ height: 11px; background: transparent; margin: 2px; }}
QScrollBar::handle {{ background: {t.border}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {t.text_dim}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ background: {t.panel}; color: {t.text}; border: 1px solid {t.border}; border-radius: {r}px; padding: 4px 6px; }}
QGroupBox {{ border: 1px solid {t.border}; border-radius: {r}px; margin-top: 12px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.text_dim}; }}
QSplitter::handle {{ background: {t.border}; }}
QGraphicsView {{ border: 1px solid {t.border}; border-radius: {r}px; }}
"""


def _asset_folder() -> Path:
    return Path(tempfile.gettempdir()) / "tracepaddock_theme"


def apply_app_theme(app: QtWidgets.QApplication, theme: Theme) -> None:
    """Window style of the whole program (style, palette, style sheet, font)."""
    if not _CLASSIC:                                         # remember the system look once
        _CLASSIC.update(style=app.style().objectName(), font=QtGui.QFont(app.font()))
    if theme.key == "classic" or not theme.window:
        # Clear the custom palette before Qt applies the current system colours to existing menus.
        app.setPalette(QtGui.QPalette())
        app.setStyleSheet("")
        style = QtWidgets.QStyleFactory.create(str(_CLASSIC["style"]))
        if style is not None:
            app.setStyle(style)
        app.setFont(_CLASSIC["font"])                        # type: ignore[arg-type]
        return
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    app.setPalette(_qpalette(theme))
    if theme.font_ui:
        families = set(QtGui.QFontDatabase().families())
        if theme.font_ui in families:
            app.setFont(QtGui.QFont(theme.font_ui, 9))
    assets: Dict[str, str] = {}
    if theme.bar_pattern:
        from .cute_art import bar_pattern_svg, write_asset
        svg, repeat = bar_pattern_svg(theme.bar_pattern, 30)
        try:
            assets["bar"] = write_asset(_asset_folder(), "bar_%s.svg" % theme.bar_pattern, svg).as_posix()
            assets["repeat"] = "1" if repeat else "0"
        except (OSError, ValueError):
            assets.clear()                       # the decoration needs a file: without one the bars stay plain
    app.setStyleSheet(_stylesheet(theme, assets))
