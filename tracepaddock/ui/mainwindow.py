"""The application main window."""
from __future__ import annotations

import copy
import re
from weakref import WeakSet
from pathlib import Path
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from .. import APP_NAME, AUTHOR, __version__
from ..core.export import ExportError, write_csv, write_mat, write_mdf
from ..core.model import Dataset, DatasetError
from ..core.registry import is_supported, list_datasets_in_directory, open_dataset
from ..display import DisplaySignal
from ..i18n import language, set_language, tr
from ..plt import arrange
from ..plt.model import PltEntry, PltScript
from ..plt.parser import PltError, load_plt, parse_attributes, save_plt
from ..plt.process import build_display_signals, signal_statistics, statistics_text, value_map_names
from ..settings import Settings, default_settings_path
from ..valuemap import ValueMapLibrary
from ..rules.engine import SignalSource
from ..rules.model import RuleSet
from .candb_dialog import SignalDatabaseDialog
from .image_panel import ImagePanel
from .layouts import DEFAULT_STARTUP, LAST, PRESET_KEYS, PRESETS, preset_parts
from .icons import application_icon, icon
from .themes import DEFAULT_THEME, PROTECTED_THEME, THEME_KEYS, THEMES, apply_app_theme, get_theme, visible_themes
from .rules_panel import RulesPanel
from .dialogs import (AppearanceDialog, ComputedSignalDialog, FavouritesDialog, InfoDialog, ReferenceLineDialog,
                      TimeRangeDialog, signal_info_sections)
from .editor import PltEditorDialog
from .exporting import ExportDialog, header_lines, picture_state, print_preview, print_view, save_picture
from .open_view import OpenViewDialog, recording_file_filters
from .plotview import PlotView
from .render import BACKGROUNDS
from .settings_dialog import PROCESSING_KEYS, SettingsDialog
from .valuemaps import ValueMapDialog, entry_pattern
from .warmup import WarmupDialog

__all__ = ["MainWindow"]

SHORTCUT_HELP = [
    ("Ctrl+O", "Open a recording"), ("E", "Choose / edit signals"), ("Ctrl+S", "Save the view"),
    ("Q", "Discard changes of the script and reload"), ("I", "Information (signal under selection, else recording)"),
    ("X", "Reference line on / off"), ("Z / Y / U", "Zoom in time / in value / both: drag a frame"),
    ("-", "Undo the last zoom"), ("SPACE", "Show everything"), ("N / P", "Next / previous page"),
    ("L R T B, RETURN", "Mark the limits at the mouse position, RETURN applies them"),
    ("TAB, mouse wheel", "Select the next / previous signal (while signals are emphasized it is shown too; "
                         "a single emphasis moves along)"),
    ("H", "Emphasize / end emphasis: the signal under the mouse in the legend, else the selected one. "
          "Without a selection: select the curve under the mouse"),
    ("Up / Down", "Move the range of the selected (else the emphasized) signals by one pixel; "
                  "without such a signal: move the mouse pointer"),
    ("Shift + Up / Down", "Selected (else emphasized) signals: taller / flatter by one pixel at each end"),
    ("Left / Right", "Previous / next sample of the selected signal; without selection: move the pointer one pixel"),
    ("C", "Copy the values at the cursor"), ("S", "Copy statistics of all signals"),
    ("Ctrl+C", "Copy the name of the selected signal"), ("ESC", "Cancel zoom / measurement / selection"),
    ("Shift+Left / Right", "Previous / next recording in the folder"),
    ("Alt+Left / Right", "Previous / next view of the history"),
    ("0", "Switch between the two last used views"),
    ("Ctrl + wheel", "Zoom in time around the mouse"), ("Shift + wheel", "Scroll in time"),
    ("drag in the plot", "Zoom in time"), ("Ctrl+P", "Print"), ("F11", "Full screen"),
    ("click on a legend row", "Select the signal; while others are emphasized it is shown only after H"),
    ("right click on a legend row", "Edit the signal (view editor, also value display and state band)"),
    ("double click on a legend row", "Edit the signal"),
    ("F5", "Run all rules"), ("double click on an event", "Show the event (marker in the plot)"),
]

_MODE_LABELS = (("mapped", "Show mapped value"), ("both", "Show mapped value (raw value)"), ("raw", "Show raw value"))

#: what the user may change in the classic look (Appearance dialog, View menu) besides the plot background
#: ("classic_background"): kept in "classic_look" while another theme is in use, given back on return
_CLASSIC_LOOK = ("time_labels", "legend_font_size", "backdrop", "backdrop_opacity", "cursor_character")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__()
        self.setWindowIcon(application_icon())
        self.settings = settings or Settings()
        if self.settings.get("theme") is None:
            startup_theme = THEMES[DEFAULT_THEME]
            self.settings.set("theme", DEFAULT_THEME)
            self.settings.set_default("background", startup_theme.plot)
            self.settings.set_default("time_labels", startup_theme.time_labels)
            self.settings.set_default("legend_font_size", startup_theme.legend_font_px)
        if str(self.settings.get("theme") or PROTECTED_THEME) not in THEMES:
            self.settings.set("theme", PROTECTED_THEME)          # a theme that no longer exists: the classic look
            self._restore_classic_look()
        self.valuemaps = ValueMapLibrary(self._side_file("valuemaps.json"))
        self.ruleset = RuleSet(self._side_file("rules.json"))
        self._candb = None                                # signal databases, loaded when first needed
        self._warmup_dialog: Optional[WarmupDialog] = None
        self.dataset: Optional[Dataset] = None
        self.script = PltScript()
        self.signals: List[DisplaySignal] = []
        # script history: oldest first, _history_pos = the script shown now
        self._history: List[str] = self.settings.text_list("plt_history")
        self._history_pos = len(self._history) - 1
        self._last_two: List[str] = []                 # the two most recently used scripts (key 0)
        #: tool bar / status bar shown (the tool bar is rebuilt on a language change)
        self._bars = {"toolbar": True, "statusbar": True}
        self.view = PlotView(self)
        self._pointer = None
        self._pointer_windows = WeakSet()
        self._pointer_filter_installed = False
        self.setCentralWidget(self.view)
        self.setAcceptDrops(True)
        self.rules_panel = RulesPanel(self, self.ruleset)
        self.rules_dock = QtWidgets.QDockWidget(self)
        self.rules_dock.setObjectName("rules")
        self.rules_dock.setWidget(self.rules_panel)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.rules_dock)
        self.rules_dock.hide()
        self.image_panel = ImagePanel(lambda index: self.value_mapping_dialog(index))
        self.image_dock = QtWidgets.QDockWidget(self)
        self.image_dock.setObjectName("images")
        self.image_dock.setWidget(self.image_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.image_dock)
        self.image_dock.hide()
        self._build_status_bar()
        self._build_actions()
        self.view.apply_settings(self.settings)
        self.view.statusChanged.connect(self._on_status)
        self.view.editRequested.connect(self._edit_signal_at)
        self.view.commandRequested.connect(self._on_view_command)
        self.view.dividersChanged.connect(self._on_dividers)
        self.view.rangesChanged.connect(self._on_ranges_changed)
        self.view.viewChanged.connect(self._update_actions)
        self.view.cursorTimeChanged.connect(self.image_panel.show_time)
        self.rules_dock.visibilityChanged.connect(lambda on: self.act_rules.setChecked(on))
        self.image_dock.visibilityChanged.connect(lambda on: self.act_images.setChecked(on))
        self.resize(int(self.settings.number("window/width", 1400, 400, 10000)),
                    int(self.settings.number("window/height", 850, 300, 10000)))
        QtCore.QTimer.singleShot(300, _warm_up_imports)
        self._update_title()
        self._update_actions()
        self.apply_layout(str(self.settings.get("startup_layout", DEFAULT_STARTUP) or DEFAULT_STARTUP), startup=True)
        self.apply_theme()

    def _side_file(self, name: str) -> Path:
        """A file next to the settings file (value mappings live there)."""
        base = getattr(self.settings, "path", None) or default_settings_path()
        return Path(base).with_name(name)

    # ------------------------------------------------------------------ construction
    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self._status_fields = {}
        for key, caption, width in (("x", "X=", 110), ("y", "Y=", 150), ("dydx", "dY/dX", 90), ("t", "T=", 100),
                                    ("xrange", "<X>", 190), ("yrange", "<Y>", 110)):
            label = QtWidgets.QLabel(caption)
            value = QtWidgets.QLabel("")
            value.setObjectName("statusValue")
            value.setMinimumWidth(width)
            value.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
            bar.addPermanentWidget(label)
            bar.addPermanentWidget(value)
            self._status_fields[key] = (value, caption)
        self.clock_label = QtWidgets.QLabel("")
        self.clock_label.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        bar.addPermanentWidget(self.clock_label)
        self.credit_label = QtWidgets.QLabel("%s %s · %s" % (APP_NAME, __version__, AUTHOR))
        font = QtGui.QFont(self.credit_label.font())
        font.setPointSizeF(max(6.5, font.pointSizeF() - 1.5))
        self.credit_label.setFont(font)
        self.credit_label.setStyleSheet("color: #8a8f96")
        bar.addPermanentWidget(self.credit_label)
        self._clock = QtCore.QTimer(self)
        self._clock.timeout.connect(self._tick)
        self._clock.start(1000)
        self._tick()

    def _tick(self) -> None:
        if self.clock_label.isVisible() or not self.clock_label.text():
            self.clock_label.setText(QtCore.QTime.currentTime().toString("HH:mm:ss"))

    def _menu(self, parent, title: str) -> QtWidgets.QMenu:
        menu = parent.addMenu(tr(title))
        self._built.append(menu)
        return menu

    def _group(self) -> QtGui.QActionGroup:
        group = QtGui.QActionGroup(self)
        self._built.append(group)
        return group

    def _action(self, text: str, slot, shortcut: str = "", *, key_hint: str = "", checkable: bool = False,
                tip: str = "") -> QtGui.QAction:
        """``shortcut`` is a real window shortcut; ``key_hint`` only shows the key the view handles itself."""
        action = QtGui.QAction(tr(text) + ("\t" + key_hint if key_hint else ""), self)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.setCheckable(checkable)
        action.setStatusTip(tr(tip) if tip else "")
        action.triggered.connect(slot)
        self._built.append(action)
        return action

    def _build_actions(self) -> None:
        # Rebuilt on a language change: remove the previous menus, toolbar and actions
        # completely, otherwise their shortcuts would stay active twice.
        for obj in getattr(self, "_built", []):
            try:
                if isinstance(obj, QtWidgets.QToolBar):
                    self.removeToolBar(obj)
                    obj.hide()
                obj.deleteLater()
            except RuntimeError:
                pass                                   # already deleted
        self._built: list = []
        bar = self.menuBar()
        bar.clear()
        a = self._action

        # ---- File
        m = self._menu(bar, "&File")
        self.act_open = a("&Open recording and view...", self.open_dialog, "Ctrl+O")
        self.act_open_plt = a("Open P&LT...", self.open_plt_dialog)
        self.act_edit = a("Choose / &edit signals...", self.edit_signals, key_hint="E")
        self.act_save_plt = a("&Save view", self.save_plt, "Ctrl+S")
        self.act_save_plt_as = a("Save view &as...", self.save_plt_as)
        self.act_reload = a("Disca&rd script changes and reload", self.reload_script, key_hint="Q")
        self.act_info = a("&Info...", self.show_info, key_hint="I")
        self.act_prev_file = a("Previous recording in folder", lambda: self.step_dataset(-1), "Shift+Left")
        self.act_next_file = a("Next recording in folder", lambda: self.step_dataset(1), "Shift+Right")
        m.addActions([self.act_open, self.act_open_plt])
        self.favourites_menu = m.addMenu(tr("Favourite views"))
        self.favourites_menu.aboutToShow.connect(self._fill_favourites)
        m.addSeparator()
        m.addActions([self.act_edit, self.act_save_plt, self.act_save_plt_as, self.act_reload])
        m.addSeparator()
        self.act_export = a("&Export data...", self.export_data,
                            tip="Write the signals of the legend as CSV, TXT, MATLAB, MF4 or MDF")
        self.act_picture = a("Export &picture...", self.export_picture, tip="PNG, PDF or SVG of the current view")
        self.act_print = a("&Print...", self.print_current, "Ctrl+P")
        self.act_preview = a("Print pre&view...", self.print_preview_current)
        m.addActions([self.act_export, self.act_picture, self.act_print, self.act_preview])
        m.addSeparator()
        self.act_convert = a("&Convert recording to another format...", self.convert_dialog,
                             tip="Write the whole recording as MF4 or MDF 3, e.g. .dat -> .mf4")
        m.addActions([self.act_info, self.act_convert, self.act_prev_file, self.act_next_file])
        m.addSeparator()
        self.recent_data_menu = m.addMenu(tr("Recent recordings"))
        self.recent_plt_menu = m.addMenu(tr("Recent views"))
        self.recent_data_menu.aboutToShow.connect(lambda: self._fill_recent(self.recent_data_menu, "recent_datasets", self.open_dataset))
        self.recent_plt_menu.aboutToShow.connect(lambda: self._fill_recent(self.recent_plt_menu, "recent_plts", self.open_plt))
        m.addSeparator()
        m.addAction(a("E&xit", self.close, "Ctrl+Q"))

        # ---- Edit
        m = self._menu(bar, "&Edit")
        m.addAction(a("Copy &picture", self.copy_picture))
        m.addAction(a("Copy &values at the cursor", self.copy_values, key_hint="C"))
        m.addAction(a("Copy &statistics", self.copy_statistics, key_hint="S"))

        # ---- View
        m = self._menu(bar, "&View")
        view_menu = m
        self.act_zoom_xy = a("Zoom (drag a frame)", lambda: self.view.start_zoom("xy"), key_hint="U")
        self.act_zoom_x = a("Zoom in time", lambda: self.view.start_zoom("x"), key_hint="Z")
        self.act_zoom_y = a("Zoom in value", lambda: self.view.start_zoom("y"), key_hint="Y")
        self.act_zoom_undo = a("Undo last zoom", self.view.undo_zoom, key_hint="-")
        self.act_zoom_reset = a("Show everything", self.view.reset_zoom, key_hint="SPACE")
        self.act_time_range = a("Set visible time range...", self.time_range_dialog,
                                tip="Enter the start and end time (T0 / T1) of the visible range")
        self.act_ref = a("Reference line", self._toggle_reference_line, key_hint="X", checkable=True,
                         tip="A vertical line follows the mouse; the legend shows the values at that time")
        self.act_page_prev = a("Previous page", lambda: self.view.page("left"), key_hint="P")
        self.act_page_next = a("Next page", lambda: self.view.page("right"), key_hint="N")
        self.act_page_up = a("Page up (value zoom)", lambda: self.view.page("up"))
        self.act_page_down = a("Page down (value zoom)", lambda: self.view.page("down"))
        self.act_xy = a("XY view (first signal = X axis)", self._toggle_xy, checkable=True,
                        tip="The first signal of the legend is the X axis; all others are drawn over it")
        self.act_xy.setChecked(self.view.is_xy())
        m.addActions([self.act_zoom_xy, self.act_zoom_x, self.act_zoom_y, self.act_zoom_undo, self.act_zoom_reset,
                      self.act_time_range])
        m.addSeparator()
        m.addActions([self.act_ref, self.act_page_prev, self.act_page_next, self.act_page_up, self.act_page_down])
        m.addSeparator()
        m.addAction(self.act_xy)
        self.act_xy_time = a("XY: colour by time (light = early, dark = late)",
                             lambda c: self._set_option("xy_time_color", bool(c)), checkable=True)
        self.act_xy_time.setChecked(bool(self.settings.get("xy_time_color", True)))
        self.act_xy_points = a("XY: points only", lambda c: self._set_option("xy_points", bool(c)), checkable=True)
        self.act_xy_points.setChecked(bool(self.settings.get("xy_points", False)))
        m.addActions([self.act_xy_time, self.act_xy_points])
        m.addSeparator()
        width_menu = m.addMenu(tr("Line width"))
        group = self._group()
        for width in (1.0, 1.5, 2.0, 3.0, 4.0):
            act = a("%g px" % width, lambda _c=False, w=width: self._set_option("line_width", w), checkable=True)
            act.setChecked(self.settings.number("line_width", 1.0) == width)
            group.addAction(act)
            width_menu.addAction(act)
        bg_menu = m.addMenu(tr("Background"))
        group = self._group()
        current = str(self.settings.get("background", "#000000")).lower()
        for key, (label, rgb) in BACKGROUNDS.items():
            name = "#%02x%02x%02x" % rgb
            act = a(label, lambda _c=False, n=name: self._set_option("background", n), checkable=True)
            pix = QtGui.QPixmap(16, 16)
            pix.fill(QtGui.QColor(*rgb))
            act.setIcon(QtGui.QIcon(pix))
            act.setChecked(name == current)
            group.addAction(act)
            bg_menu.addAction(act)
        self.act_grid = a("Grid", lambda c: self._set_option("grid", bool(c)), checkable=True)
        self.act_grid.setChecked(bool(self.settings.get("grid", True)))
        self.act_steps = a("Hold samples (steps)", lambda c: self._set_option("step_mode_analog", bool(c)), checkable=True,
                           tip="On (default): a value holds until the next sample. Off: straight lines between samples")
        self.act_steps.setChecked(bool(self.settings.get("step_mode_analog", True)))
        m.addActions([self.act_grid, self.act_steps])
        m.addAction(a("Appearance...", self.appearance_dialog))
        theme_menu = m.addMenu(tr("Theme"))
        group = self._group()
        current_theme = get_theme(self.settings.get("theme")).key
        self.theme_actions = {}
        for key in visible_themes(self.settings.text_list("hidden_themes")):
            act = a(THEMES[key].name, lambda _c=False, k=key: self.set_theme(k), checkable=True)
            act.setChecked(key == current_theme)
            group.addAction(act)
            theme_menu.addAction(act)
            self.theme_actions[key] = act
        theme_menu.addSeparator()
        theme_menu.addAction(a("Manage themes...", self.theme_dialog,
                               tip="Apply a theme, delete the themes you do not want, restore deleted ones"))
        self.backdrop_menu = m.addMenu(tr("Picture behind the curves"))
        self.backdrop_menu.aboutToShow.connect(self._fill_backdrop_menu)
        self._fill_backdrop_menu()
        self.pointer_menu = m.addMenu(tr("Mouse pointer"))
        self.pointer_menu.aboutToShow.connect(self._fill_pointer_menu)
        self._fill_pointer_menu()
        m.addSeparator()
        m.addAction(a("Full screen", self._toggle_full_screen, "F11"))

        # ---- Script history
        self.history_menu = self._menu(bar, "View &history")
        self.act_back = a("Back", lambda: self.history_step(-1), "Alt+Left", tip="Load the previous view")
        self.act_forward = a("Next", lambda: self.history_step(1), "Alt+Right", tip="Load the next view")
        self.act_clear_history = a("Clear history", self.clear_history)
        self.history_menu.aboutToShow.connect(self._fill_history)
        self._fill_history()

        # ---- Signals
        m = self._menu(bar, "&Signals")
        self.act_auto = a("Auto range: every signal fills the window", lambda: self._arrange("auto"),
                          tip="Each signal gets a rounded display range that fits its values")
        self.act_unit = a("Same range for signals with the same unit", lambda: self._arrange("unit"),
                          tip="Signals of one unit share one scale, so their curves can be compared directly")
        self.act_bands = a("Arrange in bands (one per signal)", lambda: self._arrange("bands"),
                           tip="Every signal gets its own horizontal band, top to bottom in legend order")
        self.act_unit_bands = a("Arrange in bands by unit", lambda: self._arrange("unit_bands"),
                                tip="One band per unit; signals of that unit share the band and one scale")
        self.act_view_auto = a("Auto range for the visible time window", lambda: self._arrange("auto_view"))
        self.act_restore = a("Restore ranges from the script file", lambda: self._arrange("restore"))
        self.act_colors = a("Apply cyclic colours", lambda: self._arrange("colors"))
        m.addActions([self.act_auto, self.act_view_auto, self.act_unit, self.act_bands, self.act_unit_bands])
        m.addSeparator()
        m.addActions([self.act_restore, self.act_colors])
        m.addSeparator()
        self.act_refline = a("Reference lines of the selected signal...", self.reference_lines,
                             tip="Horizontal lines at a value or at the signal's minimum / maximum / mean")
        self.act_computed = a("Add reference / computed signal...", self.add_computed_signal,
                              tip="A constant, ramp, sine or step for comparison - or a processed copy of a signal")
        self.act_duplicate = a("Duplicate selected signal with processing...", lambda: self.add_computed_signal(True),
                               tip="Copy the selected signal, apply a function (filter, scaling, derivative...) and compare")
        m.addActions([self.act_refline, self.act_computed, self.act_duplicate])
        m.addSeparator()
        self.act_valuemaps = a("Value mapping (raw value -> logical value)...", lambda: self.value_mapping_dialog(),
                               tip="Import or enter what raw enumeration values mean; show raw or mapped values")
        m.addAction(self.act_valuemaps)
        self.value_display_menu = m.addMenu(tr("Value display of the selected signal"))
        self.value_display_menu.aboutToShow.connect(self._fill_value_display_menu)
        self._fill_value_display_menu()

        # ---- Analysis
        m = self._menu(bar, "&Analysis")
        self.act_rules = a("Rules and events panel", lambda c: self.rules_dock.setVisible(bool(c)), checkable=True,
                           tip="Rules over several signals: find events, check responses and their delay")
        self.act_rules.setChecked(self.rules_dock.isVisible())
        self.act_new_rule = a("New rule (form)...", lambda: self._rules_do("new"))
        self.act_script = a("Python script...", lambda: self._rules_do("script"))
        self.act_plugins = a("Plug-ins...", lambda: self._rules_do("plugin"))
        self.act_run_rules = a("Run all rules", lambda: self._rules_do("run"), "F5")
        self.act_report = a("Excel report of the events...", lambda: self._rules_do("report"))
        self.act_images = a("Image panel", lambda c: self.image_dock.setVisible(bool(c)), checkable=True,
                            tip="Pictures for signal values at the cursor, e.g. the speed limit sign")
        self.act_images.setChecked(self.image_dock.isVisible())
        m.addActions([self.act_rules, self.act_new_rule, self.act_script, self.act_plugins])
        m.addSeparator()
        m.addActions([self.act_run_rules, self.act_report])
        m.addSeparator()
        m.addAction(self.act_images)
        self.rules_dock.setWindowTitle(tr("Rules and events"))
        self.image_dock.setWindowTitle(tr("Image panel"))

        # ---- View: layouts and panels (at the top of the menu)
        self.layout_actions = {}
        for key, text, shortcut, _parts in PRESETS:
            self.layout_actions[key] = a(text, lambda _c=False, k=key: self.apply_layout(k), shortcut)
        self.layout_menu = QtWidgets.QMenu(tr("Layout"), self)
        self._built.append(self.layout_menu)
        self.layout_menu.aboutToShow.connect(self._fill_layout_menu)
        self._fill_layout_menu()
        self.act_toolbar = a("Tool bar", lambda c: self._set_bars(toolbar=bool(c)), checkable=True)
        self.act_statusbar = a("Status bar", lambda c: self._set_bars(statusbar=bool(c)), checkable=True)
        self.act_toolbar.setChecked(self._bars["toolbar"])
        self.act_statusbar.setChecked(self._bars["statusbar"])
        panels = QtWidgets.QMenu(tr("Panels"), self)
        self._built.append(panels)
        panels.addActions([self.act_rules, self.act_images])
        panels.addSeparator()
        panels.addActions([self.act_toolbar, self.act_statusbar])
        first = view_menu.actions()[0]
        view_menu.insertMenu(first, self.layout_menu)
        view_menu.insertMenu(first, panels)
        view_menu.insertSeparator(first)

        # ---- Configuration / Language / Help
        m = self._menu(bar, "&Configuration")
        m.addAction(a("Settings...", self.settings_dialog))
        m.addAction(self.act_valuemaps)
        m.addAction(a("Signal databases (DBC / CAN matrix)...", self.signal_database_dialog,
                      tip="Resolution, cycle time, send type and value tables of the signals"))
        m = self._menu(bar, "&Language")
        group = self._group()
        for code, label in (("en", "English"), ("zh", "中文")):
            act = QtGui.QAction(label, self, checkable=True)
            self._built.append(act)
            act.setChecked(language() == code)
            act.triggered.connect(lambda _c=False, c=code: self._set_language(c))
            group.addAction(act)
            m.addAction(act)
        m = self._menu(bar, "&Help")
        m.addAction(a("Quick warm-up...", self._show_warmup))
        m.addAction(a("Keyboard and mouse...", self.show_shortcuts, "F1"))
        m.addAction(a("About...", self.show_about))

        # ---- toolbar
        tb = self.addToolBar(tr("Standard"))
        self._built.append(tb)
        tb.setObjectName("standard")
        tb.setMovable(False)
        icons = get_theme(self.settings.get("theme")).toolbar_icons
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly if icons else QtCore.Qt.ToolButtonTextOnly)
        tb.setIconSize(QtCore.QSize(20, 20))
        for act, label, name in ((self.act_open, "Open", "open"), (self.act_prev_file, "<", "prev_file"),
                                 (self.act_next_file, ">", "next_file"), ("files", "", ""),
                                 (self.act_edit, "Signals", "signals"), (self.act_save_plt, "Save view", "save_view"),
                                 (None, "", ""),
                                 (self.act_back, "View <", "view_back"), (self.act_forward, "View >", "view_forward"),
                                 (None, "", ""),
                                 (self.act_zoom_xy, "Zoom", "zoom"), (self.act_zoom_x, "Zoom X", "zoom_x"),
                                 (self.act_zoom_y, "Zoom Y", "zoom_y"), (self.act_zoom_undo, "Undo", "zoom_undo"),
                                 (self.act_zoom_reset, "All", "zoom_all"), (None, "", ""),
                                 (self.act_ref, "Ref line", "ref"), (self.act_page_prev, "<<", "page_prev"),
                                 (self.act_page_next, ">>", "page_next"), (self.act_xy, "XY", "xy"), (None, "", ""),
                                 (self.act_auto, "Auto range", "auto_range"), (self.act_unit, "By unit", "by_unit"),
                                 (self.act_bands, "Bands", "bands"), (self.act_unit_bands, "Unit bands", "unit_bands"),
                                 (None, "", ""),
                                 (self.act_computed, "+ Signal", "add_signal"), (self.act_refline, "+ Ref line", "add_ref"),
                                 (self.act_valuemaps, "Mapping", "mapping"), (None, "", ""),
                                 (self.act_rules, "Rules", "rules"), (self.act_images, "Images", "images"), (None, "", ""),
                                 (self.act_export, "Export", "export"), (self.act_info, "Info", "info")):
            if act is None:
                tb.addSeparator()
                continue
            if act == "files":
                tb.addWidget(self._files_combo())
                continue
            act.setIconText(tr(label))              # short caption on the toolbar, full text in the menu
            act.setIcon(icon(name))
            act.setIconVisibleInMenu(False)
            act.setToolTip(act.text().replace("&", "").replace("\t", "   "))
            tb.addAction(act)
        # the layout button sits in the top right corner of the menu bar (always visible, as in VS Code)
        self.layout_button = QtWidgets.QToolButton()
        self.layout_button.setText(tr("Layout"))
        self.layout_button.setToolTip(tr("Show other panels with one click; choose the layout used at the start"))
        self.layout_button.setMenu(self.layout_menu)
        self.layout_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.layout_button.setAutoRaise(True)
        self.menuBar().setCornerWidget(self.layout_button, QtCore.Qt.TopRightCorner)
        self._built.append(self.layout_button)
        self.toolbar = tb
        tb.setVisible(self._bars["toolbar"])

    # ------------------------------------------------------------------ recordings of the current folder
    def _files_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setObjectName("filesCombo")
        combo.setMinimumContentsLength(22)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setToolTip(tr("Recordings in the folder of the current recording"))
        combo.activated.connect(self._files_combo_chosen)
        self._built.append(combo)
        self.files_combo = combo
        self._fill_files_combo()
        return combo

    def _fill_files_combo(self) -> None:
        combo = getattr(self, "files_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        current = Path(getattr(self.dataset, "archive_path", None) or self.dataset.path) if self.dataset is not None else None
        if current is not None:
            try:
                files = list_datasets_in_directory(current)
            except OSError:
                files = [current]
            if current not in files:
                files = [current] + list(files)
            for path in files:
                combo.addItem(path.name, str(path))
                combo.setItemData(combo.count() - 1, str(path), QtCore.Qt.ToolTipRole)
            combo.setCurrentIndex(max(0, combo.findData(str(current))))
        combo.blockSignals(False)

    def _files_combo_chosen(self, index: int) -> None:
        path = self.files_combo.itemData(index)
        current = str(getattr(self.dataset, "archive_path", None) or self.dataset.path) if self.dataset is not None else ""
        if path and path != current:
            self.open_dataset(path)

    # ------------------------------------------------------------------ themes (see themes.py, cute_art.py)
    def set_theme(self, key: str) -> None:
        """Switch the look; plot background, time labels, legend font, picture and pointer follow the theme.
        The classic look comes back as the user left it; choosing the theme in use changes nothing."""
        stored = str(self.settings.get("theme") or PROTECTED_THEME)
        new = get_theme(key)
        if new.key == stored:
            return
        if stored == PROTECTED_THEME:                        # leaving the classic look: remember it
            self.settings.set("classic_background", self.settings.get("background", "#000000"))
            self.settings.set("classic_look", {name: self.settings.get(name) for name in _CLASSIC_LOOK})
        self.settings.set("theme", new.key)
        if new.key == PROTECTED_THEME:
            self._restore_classic_look()
        else:
            self.settings.set("background", new.plot or self.settings.get("classic_background", "#000000"))
            self.settings.set("time_labels", new.time_labels)
            self.settings.set("legend_font_size", new.legend_font_px)
            for setting in ("backdrop", "backdrop_opacity", "cursor_character"):
                self.settings.set(setting, None)             # the theme's own picture and pointer
        self.apply_theme()
        self._build_actions()                                # menus in the new style, check marks
        self._update_actions()

    def _restore_classic_look(self) -> None:
        """The user's classic look as it was when another theme was chosen (else the classic theme's own)."""
        classic = THEMES[PROTECTED_THEME]
        self.settings.set("background", self.settings.get("classic_background", "#000000"))
        saved = self.settings.get("classic_look", None)
        saved = saved if isinstance(saved, dict) else {}
        own = {"time_labels": classic.time_labels, "legend_font_size": classic.legend_font_px}
        for name in _CLASSIC_LOOK:
            self.settings.set(name, saved.get(name, own.get(name)))

    def theme_dialog(self) -> None:
        from .theme_dialog import ThemeDialog
        dialog = ThemeDialog(get_theme(self.settings.get("theme")).key, self.settings.text_list("hidden_themes"), self)
        dialog.exec_()
        self.settings.set("hidden_themes", sorted(dialog.hidden))
        current = get_theme(self.settings.get("theme")).key
        target = dialog.applied or current
        if target in dialog.hidden:                 # the theme in use was deleted: back to the classic one
            target = PROTECTED_THEME
        if target != current:
            self.set_theme(target)
        else:
            self._build_actions()
            self._update_actions()

    def apply_theme(self) -> None:
        theme = get_theme(self.settings.get("theme"))
        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply_app_theme(app, theme)
        self.view.apply_settings(self.settings)
        character = self.settings.get("cursor_character", None)
        character = theme.cursor_character if character is None else str(character)
        cursors = {}
        ratio = float(self.devicePixelRatioF() or 1.0)
        if character == "custom":
            from .cute_art import image_cursor
            path = str(self.settings.get("cursor_image", "") or "")
            try:
                stamp = Path(path).stat().st_mtime if path else 0.0
            except OSError:
                stamp = -1.0
            made = {kind: image_cursor(kind, path, stamp, ratio) for kind in ("zoom", "pointer")} if stamp >= 0 else {}
            cursors = {k: v for k, v in made.items() if v is not None}
        elif character:
            from .cute_art import CHARACTERS, cursor
            if character in CHARACTERS:
                cursors = {"zoom": cursor("zoom", character, ratio), "pointer": cursor("pointer", character, ratio)}
        self.view.set_theme_cursors(cursors)
        self._pointer = cursors.get("pointer")
        self._set_pointer_filter(self._pointer is not None)
        for widget in (self, *self._pointer_windows):
            # Rebuilt menus may have lost their Qt object while Python still holds the wrapper.
            if not isValid(widget):
                self._pointer_windows.discard(widget)
                continue
            if self._pointer is None:
                widget.unsetCursor()
            else:
                widget.setCursor(self._pointer)
        if self._pointer is None:
            self._pointer_windows.clear()
        classic = not theme.window
        for value, _caption in self._status_fields.values():
            value.setStyleSheet("QLabel { background: #ffffff; color: #000000; padding: 0 2px; }" if classic else "")
        self.clock_label.setVisible(theme.clock)
        graph = getattr(self.rules_panel, "graph", None)
        if graph is not None:
            graph.setBackgroundBrush(QtGui.QColor(theme.base) if theme.base else QtGui.QColor(250, 250, 250))
        self.view.refresh()

    @staticmethod
    def _clear_menu(menu: QtWidgets.QMenu) -> None:
        """Empty a menu that is filled each time it opens. ``clear()`` deletes only the menu's own actions, so
        the submenus and action groups of the last filling are deleted here."""
        for child in menu.children():
            if isinstance(child, (QtWidgets.QMenu, QtGui.QActionGroup)):
                child.deleteLater()
        menu.clear()

    def _fill_backdrop_menu(self) -> None:
        from .cute_art import SCENES
        menu = self.backdrop_menu
        self._clear_menu(menu)
        theme = get_theme(self.settings.get("theme"))
        chosen = self.settings.get("backdrop", None)
        current = theme.backdrop if chosen is None else str(chosen)
        group = QtGui.QActionGroup(menu)
        for key, text in [("", tr("none"))] + [(k, tr(v)) for k, v in SCENES.items()]:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(current == key)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, k=key: self._set_backdrop(k))
        custom = menu.addAction(tr("Own picture..."))
        custom.setCheckable(True)
        custom.setChecked(bool(current) and current not in SCENES)
        group.addAction(custom)
        custom.triggered.connect(self._choose_backdrop_file)
        menu.addSeparator()
        opacity_menu = menu.addMenu(tr("Opacity"))
        value = self.view.state.backdrop_opacity
        group2 = QtGui.QActionGroup(opacity_menu)
        for pct in (10, 15, 20, 30, 45, 60):
            act = opacity_menu.addAction("%d %%" % pct)
            act.setCheckable(True)
            act.setChecked(abs(value * 100 - pct) < 0.5)
            group2.addAction(act)
            act.triggered.connect(lambda _c=False, p=pct: self._set_option("backdrop_opacity", p / 100.0))

    def _set_backdrop(self, key: str) -> None:
        self._set_option("backdrop", key)

    def _choose_backdrop_file(self) -> None:
        from .cute_art import IMAGE_FILTER, backdrop_image
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("Picture behind the curves"), "",
                                                        tr("Pictures") + " (%s)" % IMAGE_FILTER)
        if path and backdrop_image(path, self.view.width()) is None:
            self._error(tr("The picture %s cannot be read.") % path)
            path = ""
        if path:
            self._set_backdrop(path)
        else:
            self._fill_backdrop_menu()

    def _fill_pointer_menu(self) -> None:
        from .cute_art import CHARACTERS
        menu = self.pointer_menu
        self._clear_menu(menu)
        theme = get_theme(self.settings.get("theme"))
        chosen = self.settings.get("cursor_character", None)
        current = theme.cursor_character if chosen is None else str(chosen)
        group = QtGui.QActionGroup(menu)
        for key, text in [("", tr("system pointers"))] + [(k, tr(v)) for k, v in CHARACTERS.items()]:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(current == key)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, k=key: self._set_pointer(k))
        own = menu.addAction(tr("Own picture..."))
        own.setCheckable(True)
        own.setChecked(current == "custom")
        group.addAction(own)
        own.triggered.connect(self._choose_pointer_file)

    def _choose_pointer_file(self) -> None:
        from .cute_art import IMAGE_FILTER, image_cursor
        start = str(Path(str(self.settings.get("cursor_image", "") or "")).parent) if self.settings.get("cursor_image") else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("Picture for the mouse pointer"), start,
                                                        tr("Pictures") + " (%s)" % IMAGE_FILTER)
        if not path:
            self._fill_pointer_menu()
            return
        if image_cursor("pointer", path, Path(path).stat().st_mtime, 1.0) is None:
            self._error(tr("The picture %s cannot be read.") % path)
            return
        self.settings.set("cursor_image", path)
        self._set_pointer("custom")

    def _set_pointer(self, key: str) -> None:
        self.settings.set("cursor_character", key)
        self.apply_theme()

    # ------------------------------------------------------------------ layouts (see layouts.py)
    def _fill_layout_menu(self) -> None:
        menu = self.layout_menu
        self._clear_menu(menu)
        for key in PRESET_KEYS:
            menu.addAction(self.layout_actions[key])
        saved = self._saved_layouts()
        if saved:
            menu.addSeparator()
            for name in sorted(saved):
                act = menu.addAction(name)
                act.triggered.connect(lambda _c=False, n=name: self.apply_layout("custom:" + n))
        menu.addSeparator()
        menu.addAction(tr("Save current layout..."), self.save_layout)
        if saved:
            delete = menu.addMenu(tr("Delete saved layout"))
            for name in sorted(saved):
                act = delete.addAction(name)
                act.triggered.connect(lambda _c=False, n=name: self.delete_layout(n))
        menu.addSeparator()
        start = menu.addMenu(tr("When the program starts"))
        group = QtGui.QActionGroup(start)
        current = str(self.settings.get("startup_layout", DEFAULT_STARTUP) or DEFAULT_STARTUP)
        choices = [(key, tr(text).replace("&", "")) for key, text, _s, _p in PRESETS]
        choices += [("custom:" + n, n) for n in sorted(saved)]
        choices.append((LAST, tr("the layout when the program was last closed")))
        for key, text in choices:
            act = start.addAction(text)
            act.setCheckable(True)
            act.setChecked(key == current)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, k=key: self.settings.set("startup_layout", k))

    def _saved_layouts(self) -> Dict[str, dict]:
        data = self.settings.get("layouts", {})
        return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}

    def _set_bars(self, toolbar: Optional[bool] = None, statusbar: Optional[bool] = None) -> None:
        if toolbar is not None:
            self._bars["toolbar"] = bool(toolbar)
        if statusbar is not None:
            self._bars["statusbar"] = bool(statusbar)
        self.toolbar.setVisible(self._bars["toolbar"])
        self.statusBar().setVisible(self._bars["statusbar"])
        for act, key in ((self.act_toolbar, "toolbar"), (self.act_statusbar, "statusbar")):
            act.blockSignals(True)
            act.setChecked(self._bars[key])
            act.blockSignals(False)

    def layout_state(self) -> dict:
        """The current arrangement: dock positions and sizes, tool bar and status bar."""
        state = bytes(self.saveState().toBase64()).decode("ascii")
        return {"state": state, "toolbar": self._bars["toolbar"], "statusbar": self._bars["statusbar"],
                "rules": not self.rules_dock.isHidden(), "images": not self.image_dock.isHidden()}

    def _restore_layout_state(self, data: dict) -> bool:
        try:
            ok = self.restoreState(QtCore.QByteArray.fromBase64(str(data.get("state", "")).encode("ascii")))
        except (TypeError, ValueError):
            ok = False
        if not ok:                                 # unreadable: at least the panels that were shown
            self.rules_dock.setVisible(bool(data.get("rules", False)))
            self.image_dock.setVisible(bool(data.get("images", False)))
        self._set_bars(bool(data.get("toolbar", True)), bool(data.get("statusbar", True)))
        return ok

    def apply_layout(self, key: str, startup: bool = False) -> None:
        """Show the panels of a preset, a saved layout ("custom:<name>") or the last one (``LAST``)."""
        if key == LAST or key.startswith("custom:"):
            data = self.settings.get("window/layout") if key == LAST else self._saved_layouts().get(key[7:])
            if isinstance(data, dict):
                self._restore_layout_state(data)
                return
            key = DEFAULT_STARTUP
        parts = preset_parts(key)
        for dock, area, on in ((self.rules_dock, QtCore.Qt.BottomDockWidgetArea, parts["rules"]),
                               (self.image_dock, QtCore.Qt.RightDockWidgetArea, parts["images"])):
            if dock.isFloating():
                dock.setFloating(False)
            if self.dockWidgetArea(dock) != area:
                self.addDockWidget(area, dock)
            dock.setVisible(on)
        self._set_bars(parts["toolbar"], parts["statusbar"])
        QtCore.QTimer.singleShot(0, self._size_docks)

    def _size_docks(self) -> None:
        if not self.rules_dock.isHidden():
            self.resizeDocks([self.rules_dock], [max(220, int(self.height() * 0.32))], QtCore.Qt.Vertical)
        if not self.image_dock.isHidden():
            self.resizeDocks([self.image_dock], [270], QtCore.Qt.Horizontal)

    def save_layout(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, tr("Save current layout"), tr("Name of the layout:"))
        name = name.strip()
        if not ok or not name:
            return
        saved = self._saved_layouts()
        if name in saved and QtWidgets.QMessageBox.question(
                self, APP_NAME, tr("Replace the saved layout '%s'?") % name) != QtWidgets.QMessageBox.Yes:
            return
        saved[name] = self.layout_state()
        self.settings.set("layouts", saved)

    def delete_layout(self, name: str) -> None:
        saved = self._saved_layouts()
        saved.pop(name, None)
        self.settings.set("layouts", saved)
        if self.settings.get("startup_layout") == "custom:" + name:
            self.settings.set("startup_layout", DEFAULT_STARTUP)

    # ------------------------------------------------------------------ helpers
    def _update_title(self) -> None:
        title = "%s %s" % (APP_NAME, __version__)
        if self.dataset is not None:
            plt = ("*" if self.script.modified else "") + (str(self.script.path) if self.script.path else tr("(new view)"))
            title += " - [%s (%s)]" % (self.dataset.path, plt)
        self.setWindowTitle(title)

    def _update_actions(self) -> None:
        loaded = self.dataset is not None
        xy = self.view.is_xy()
        for act in (self.act_edit, self.act_info, self.act_ref, self.act_time_range,
                    self.act_auto, self.act_unit, self.act_bands, self.act_unit_bands, self.act_view_auto,
                    self.act_restore, self.act_colors, self.act_computed, self.act_prev_file, self.act_next_file,
                    self.act_export, self.act_picture, self.act_print, self.act_preview):
            act.setEnabled(loaded)
        for act in (self.act_zoom_xy, self.act_zoom_x, self.act_zoom_y):
            act.setEnabled(loaded and not xy)
        zoomed = loaded and self.view.is_zoomed()
        for act in (self.act_zoom_undo, self.act_zoom_reset, self.act_page_prev, self.act_page_next,
                    self.act_page_up, self.act_page_down):
            act.setEnabled(zoomed)
        has_script = any(True for _ in self.script.entries())
        self.act_save_plt.setEnabled(has_script)
        self.act_save_plt_as.setEnabled(has_script)
        self.act_reload.setEnabled(self.script.path is not None)
        selected = 0 <= self.view.selected() < len(self.signals)
        self.act_refline.setEnabled(loaded and bool(self.signals))
        self.act_duplicate.setEnabled(loaded and (selected or bool(self.signals)))
        self.act_xy.setEnabled(loaded and (xy or self.view.xy_possible()))
        self.act_xy.setChecked(xy)
        self.act_back.setEnabled(loaded and self._history_pos > 0)
        self.act_forward.setEnabled(loaded and 0 <= self._history_pos < len(self._history) - 1)
        self.act_valuemaps.setEnabled(True)
        for act in (self.act_new_rule, self.act_script, self.act_plugins, self.act_run_rules, self.act_report):
            act.setEnabled(loaded)

    def _on_status(self, info: dict) -> None:
        for key, (label, _caption) in self._status_fields.items():
            label.setText(" %s" % info.get(key, "") if info.get(key, "") else "")
        hint = tr(info["hint"]) if info.get("hint") else ""
        if hint:
            self.statusBar().showMessage(hint)
        elif getattr(self, "_hint", ""):
            self.statusBar().clearMessage()
        self._hint = hint

    def _error(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, APP_NAME, text)

    def _busy(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        return _Restore()

    def _fill_recent(self, menu: QtWidgets.QMenu, key: str, opener) -> None:
        menu.clear()
        for path in self.settings.text_list(key)[:int(self.settings.number("recent_max", 32, 4, 100))]:
            act = menu.addAction(path)
            act.triggered.connect(lambda _c=False, p=path: opener(p))
        if menu.isEmpty():
            menu.addAction(tr("(none)")).setEnabled(False)

    # ------------------------------------------------------------------ loading
    def open_dialog(self) -> None:
        current_data = getattr(self.dataset, "archive_path", None) or getattr(self.dataset, "path", None)
        current_view = self.script.path
        dialog = OpenViewDialog(self.settings, str(current_data or ""), str(current_view or ""), self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        data, view = dialog.paths()
        new_data = bool(data and (current_data is None or Path(data).absolute() != current_data.absolute()))
        new_view = bool(view and (current_view is None or Path(view).absolute() != current_view.absolute()))
        if new_data:
            if not self.open_dataset(data, view if new_view else None, keep_current=bool(view and not new_view)):
                return
        elif new_view and not self.open_plt(view):
            return
        for key, value in dialog.defaults.items():
            self.settings.set(key, value)
        if dialog.defaults:
            self.settings.save()

    def open_dataset(self, path: "str | Path", plt: "Optional[str | Path]" = None, *,
                     keep_current: bool = False) -> bool:
        """Load a recording; ``plt`` optionally names the script to use."""
        with self._busy():
            try:
                dataset = open_dataset(path, self.settings)
            except DatasetError as exc:
                dataset, message = None, str(exc)
        if dataset is None:
            self._error(message)
            return False
        script_path = plt or (None if keep_current else getattr(dataset, "archive_plt", None))
        keep = keep_current or bool(self.settings.get("use_current_signal_selection", True))
        if not script_path and keep and not self.script.items:
            recent_view = self.settings.get("current_plt", "")
            if recent_view and Path(recent_view).is_file():
                script_path = recent_view
        script = None
        if script_path:
            try:
                script = load_plt(script_path)
            except PltError as exc:
                dataset.close()
                self._error(str(exc))
                return False
        elif not keep:
            script = PltScript()
        if script is not None and not self._ask_save_changes():
            dataset.close()
            return False
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = dataset
        self.rules_panel.results.clear()
        self.view.state.marker = None
        source = Path(getattr(dataset, "archive_path", None) or path)
        self.settings.push_recent("recent_datasets", source)
        self.settings.set("last_dataset_dir", str(source.parent))
        self._fill_files_combo()
        if script is not None:
            self._use_script(script)
        self.rebuild(keep_view=False)
        if not any(True for _ in self.script.entries()):
            self.statusBar().showMessage(tr("No signals chosen yet - press E to choose signals"), 8000)
            QtCore.QTimer.singleShot(0, self.edit_signals)
        return True

    def _load_script(self, path: "str | Path", *, remember: bool = True) -> bool:
        try:
            script = load_plt(path)
        except PltError as exc:
            self._error(str(exc))
            return False
        self._use_script(script, remember=remember)
        return True

    def _use_script(self, script: PltScript, *, remember: bool = True) -> None:
        self.script = script
        self._script_snapshot = copy.deepcopy(script)
        if script.path is not None:
            self.settings.push_recent("recent_plts", script.path)
            self.settings.set("current_plt", str(script.path))
            self.settings.set("last_plt_dir", str(script.path.parent))
            self._note_script(str(script.path), remember)

    def _note_script(self, path: str, remember: bool) -> None:
        """Keep the script history (back / next) and the two last used scripts (key 0) up to date."""
        key = path.lower()
        if not self._last_two or self._last_two[-1].lower() != key:
            self._last_two = (self._last_two + [path])[-2:]
        if remember:
            if not (0 <= self._history_pos < len(self._history) and self._history[self._history_pos].lower() == key):
                del self._history[self._history_pos + 1:]
                self._history = [p for p in self._history if p.lower() != key] + [path]
                limit = int(self.settings.number("history_max", 20, 4, 100))
                self._history = self._history[-limit:]
                self._history_pos = len(self._history) - 1
            self.settings.set("plt_history", list(self._history))

    def open_plt_dialog(self) -> None:
        start = self.settings.get("last_plt_dir", "") or self.settings.get("last_dataset_dir", "") or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("Open PLT"), start,
                                                        tr("PLT files") + " (*.plt);;" + tr("All files") + " (*)")
        if path:
            self.open_plt(path)

    def open_plt(self, path: "str | Path", *, remember: bool = True) -> bool:
        if not self._ask_save_changes():
            return False
        if self._load_script(path, remember=remember):
            self.rebuild(keep_view=True)
            return True
        return False

    def _ask_save_changes(self) -> bool:
        """Before another script replaces a changed one: save, discard or cancel. ``False`` = cancel."""
        if not (self.script.modified and any(True for _ in self.script.entries())):
            return True
        answer = QtWidgets.QMessageBox.question(
            self, APP_NAME, tr("The view was changed. Save it?"),
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            self.save_plt()
            return not self.script.modified
        return True

    def rebuild(self, *, keep_view: bool = True) -> None:
        """Recompute the display signals from recording and script and show them."""
        if self.dataset is None:
            self.signals = []
            self.view.set_signals([], None)
        else:
            missing = {id(entry): entry.display_name for entry in self.script.entries()
                       if entry.enabled and not entry.is_placeholder and not entry.channel_origin
                       and self.dataset.find(entry.name) is None}
            with self._busy():
                signals = build_display_signals(self.dataset, self.script, self.settings, valuemaps=self.valuemaps)
            self.signals = [signal for signal in signals if id(signal.entry) not in missing]
            self.view.set_signals(self.signals, self.dataset.time_range(), keep_view=keep_view)
            self.view.setFocus()
            self.image_panel.set_signals(self.signals)
            self.rules_panel.fill_tree()
            self.rules_panel._fill_events()
            self.rules_panel._update_lanes()
            notices = []
            if missing:
                notices.append(tr("Skipped %d PLT signals missing from this recording, including %s")
                               % (len(missing), next(iter(missing.values()))))
            errors = [signal.name for signal in self.signals if signal.error]
            if errors:
                notices.append(tr("%d of %d signals could not be shown, e.g. %s")
                               % (len(errors), len(self.signals), errors[0]))
            self.statusBar().showMessage("  ·  ".join(notices), 10000)
        self._update_title()
        self._update_actions()

    def convert_dialog(self) -> None:
        """Write a whole recording in another format (the open one, or one chosen from disk)."""
        from ..core.convert import TARGETS, convert_recording

        if self.dataset is not None:
            source = Path(self.dataset.path)
        else:
            start = self.settings.get("last_dataset_dir", "") or str(Path.home())
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("Recording to convert"), start,
                                                              recording_file_filters())
            if not chosen:
                return
            source = Path(chosen)
        filters = ";;".join("%s (*%s)" % (label, ext) for ext, (label, _v) in TARGETS.items())
        default = ".mdf" if source.suffix.lower() == ".mf4" else ".mf4"
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Convert %s to") % source.name, str(source.with_suffix(default)), filters,
            "%s (*%s)" % (TARGETS[default][0], default))
        if not target:
            return
        progress = self._progress(tr("Converting %s ...") % source.name)
        try:
            with self._busy():
                count = convert_recording(source, target, progress=self._reporter(progress))
        except DatasetError as exc:
            progress.close()
            self._error(str(exc))
            return
        progress.close()
        self.statusBar().showMessage(tr("%d signals written to %s") % (count, target), 10000)
        QtWidgets.QMessageBox.information(self, APP_NAME, tr("%d signals written to %s") % (count, target))

    def _progress(self, text: str) -> QtWidgets.QProgressDialog:
        progress = QtWidgets.QProgressDialog(text, "", 0, 100, self)
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(400)
        return progress

    @staticmethod
    def _reporter(progress: QtWidgets.QProgressDialog):
        def report(done: int, total: int) -> None:
            progress.setValue(int(100 * done / max(1, total)))
            QtWidgets.QApplication.processEvents()
        return report

    def step_dataset(self, step: int) -> None:
        if self.dataset is None:
            return
        current = Path(getattr(self.dataset, "archive_path", None) or self.dataset.path)
        files = list_datasets_in_directory(current)
        if current in files and len(files) > 1:
            self.open_dataset(files[(files.index(current) + step) % len(files)])

    # ------------------------------------------------------------------ script history / favourites
    def history_step(self, step: int) -> None:
        target = self._history_pos + step
        if self.dataset is None or not 0 <= target < len(self._history):
            return
        path = self._history[target]
        if not Path(path).exists():
            self._error(tr("The view %s does not exist any more") % path)
            return
        old = self._history_pos
        self._history_pos = target
        if not self.open_plt(path, remember=False):
            self._history_pos = old
        self._update_actions()

    def history_jump(self, index: int) -> None:
        self.history_step(index - self._history_pos)

    def clear_history(self) -> None:
        current = self._history[self._history_pos] if 0 <= self._history_pos < len(self._history) else None
        self._history = [current] if current else []
        self._history_pos = len(self._history) - 1
        self.settings.set("plt_history", list(self._history))
        self._update_actions()

    def toggle_last_script(self) -> None:
        """Key 0: switch between the two last used PLT scripts."""
        if self.dataset is not None and len(self._last_two) == 2 and Path(self._last_two[0]).exists():
            self.open_plt(self._last_two[0])

    def _fill_history(self) -> None:
        menu = self.history_menu
        menu.clear()
        menu.addActions([self.act_back, self.act_forward, self.act_clear_history])
        menu.addSeparator()
        if not self._history:
            menu.addAction(tr("(none)")).setEnabled(False)
        for number, index in enumerate(reversed(range(len(self._history))), start=1):
            mark = " *" if index == self._history_pos else ""
            act = menu.addAction("%d  %s%s" % (number, self._history[index], mark))
            act.triggered.connect(lambda _c=False, i=index: self.history_jump(i))
            act.setEnabled(self.dataset is not None)

    def _fill_favourites(self) -> None:
        menu = self.favourites_menu
        menu.clear()
        menu.addAction(tr("Administrate..."), self.favourites_dialog)
        current = str(self.script.path) if self.script.path else ""
        add = menu.addAction(tr("Add current script"), self._add_current_favourite)
        add.setEnabled(bool(current))
        menu.addSeparator()
        paths = self.settings.text_list("favourite_plts")
        if not paths:
            menu.addAction(tr("(none)")).setEnabled(False)
        for number, path in enumerate(paths, start=1):
            act = menu.addAction("%d  %s" % (number, path))
            act.triggered.connect(lambda _c=False, p=path: self.open_plt(p) if self.dataset is not None else None)
            act.setEnabled(self.dataset is not None)

    def _add_current_favourite(self) -> None:
        if self.script.path is None:
            return
        paths = self.settings.text_list("favourite_plts")
        current = str(self.script.path)
        if current.lower() not in (p.lower() for p in paths):
            self.settings.set("favourite_plts", paths + [current])
            self.statusBar().showMessage(tr("Added to the favourites: %s") % current, 5000)

    def favourites_dialog(self) -> None:
        start = self.settings.get("last_plt_dir", "") or str(Path.home())
        dialog = FavouritesDialog(self.settings.text_list("favourite_plts"),
                                  str(self.script.path) if self.script.path else None, start, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.settings.set("favourite_plts", dialog.paths())

    # ------------------------------------------------------------------ script
    def edit_signals(self, select: Optional[PltEntry] = None) -> None:
        """The view editor, opened on the line ``select`` (right / double click on a legend row), else on the
        selected signal. Value display chosen there is applied with OK; the selected and emphasized signals
        stay marked."""
        if self.dataset is None:
            return
        if not isinstance(select, PltEntry):
            index = self.view.selected()
            select = self.signals[index].entry if 0 <= index < len(self.signals) else None
        marked = self._marked_entries()
        dialog = PltEditorDialog(self.dataset, self.script, self.settings, self, select_entry=select,
                                 value_texts=self._value_texts_state, value_mapping=self._editor_value_mapping,
                                 view_state=self.view.state)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._apply_value_texts(dialog.value_text_changes())      # on the shown signals, before the rebuild
            self.script = dialog.script()
            self.rebuild(keep_view=True)
            self._restore_marks(marked, dialog.original_entry)
            # a line added with the mapping dialog's "numeric signal" is selected and shown, as from the menu
            added = [i for i, s in enumerate(self.signals) if any(s.entry is line for line in dialog.mapped_lines)]
            if added:
                st = self.view.state
                if st.emphasized:
                    st.emphasized = st.emphasized | {i for i in added if self.signals[i].in_analog}
                self.view.select(added[-1])

    def _marked_entries(self):
        """The lines that are selected, emphasized and shown for the moment (TAB), to find them again after
        lines were added, removed or moved."""
        st = self.view.state
        selected = self.signals[st.selected].entry if 0 <= st.selected < len(self.signals) else None
        emphasized = [self.signals[i].entry for i in sorted(st.emphasized) if 0 <= i < len(self.signals)]
        temporary = self.signals[st.temporary].entry if 0 <= st.temporary < len(self.signals) else None
        return selected, emphasized, temporary

    def _restore_marks(self, marked, original) -> None:
        """After a rebuild the same lines are selected / emphasized / shown for the moment, wherever they are
        now. ``original(line)`` gives the line of the old view a line stands for (itself, or the line the
        editor copied it from)."""
        selected, emphasized, temporary = marked
        where: Dict[int, int] = {}
        for index, sig in enumerate(self.signals):
            source = original(sig.entry)
            if source is not None:
                where.setdefault(id(source), index)
        view = self.view
        st = view.state
        st.emphasized = frozenset(where[id(e)] for e in emphasized
                                  if id(e) in where and self.signals[where[id(e)]].in_analog)
        view.select(where.get(id(selected), -1) if selected is not None else -1)
        shown = where.get(id(temporary), -1) if temporary is not None else -1
        # set or end the mark either way: indices of the old view must not leave it on another line
        keep = (shown >= 0 and shown == st.selected and bool(st.emphasized) and shown not in st.emphasized
                and view.shows_curve(shown))
        st.temporary = shown if keep else -1
        view.refresh()

    def _value_texts_state(self, entry: PltEntry) -> Optional[dict]:
        """For the editor: how the shown signal of a view line writes its values and the rule a change goes to
        ("" when the line cannot use one); ``None`` when the line is not shown."""
        sig = next((s for s in self.signals if s.entry is entry), None)
        if sig is None:
            return None
        pattern, _rule = self._rule_for(sig)
        return {"pattern": pattern, "text": bool(sig.enum), "mode": sig.value_mode or "both", "lane": bool(sig.lane),
                "lane_possible": self._band_possible(sig)}

    def _editor_value_mapping(self, entry: PltEntry, parent: QtWidgets.QWidget):
        """The editor's "Value mapping..." button for a view line: the library is saved at once (as from the
        menu) and the window shows it. A requested line with the mapped numbers is returned to the editor
        together with the line it belongs below: ``(new line, source line)``."""
        index = next((i for i, s in enumerate(self.signals) if s.entry is entry), -1)
        dialog = ValueMapDialog(self.valuemaps, self.signals, index, parent)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        self.valuemaps = dialog.result_library()
        request = dialog.numeric_request()
        line = self._mapped_entry(*request) if request is not None and self.dataset is not None else None
        source = self.signals[request[0]].entry if line is not None else None
        self._valuemaps_changed()
        return (line, source) if line is not None else None

    def _apply_value_texts(self, changes) -> None:
        """Value display chosen in the editor: ``(line, fields)`` with only the fields the user changed. Stored
        in the rule of the line's shown signal (it applies wherever that signal is shown); a field left alone
        keeps the rule's value."""
        stored = False
        for line, fields in changes:
            sig = next((s for s in self.signals if s.entry is line), None)
            if sig is None:
                continue
            pattern, rule = self._rule_for(sig)
            if not pattern:
                continue                               # a computed line without key=: no rule can reach it
            mode = fields.get("mode", rule.mode if rule is not None else (sig.value_mode or "both"))
            lane = fields.get("lane", rule.lane if rule is not None else False)
            self.valuemaps.assign(pattern, rule.table if rule is not None else "", mode, bool(lane))
            stored = True
        if stored:
            self.valuemaps.save()

    def _edit_signal_at(self, index: int) -> None:
        if 0 <= index < len(self.signals):
            self.edit_signals(self.signals[index].entry)

    def save_plt(self) -> None:
        if self.script.path is None:
            self.save_plt_as()
            return
        try:
            save_plt(self.script)
        except PltError as exc:
            self._error(str(exc))
            return
        self._script_snapshot = copy.deepcopy(self.script)
        self.statusBar().showMessage(tr("Saved %s") % self.script.path, 5000)
        self._update_title()

    def save_plt_as(self) -> None:
        start = str(self.script.path or Path(self.settings.get("last_plt_dir", "") or Path.home()) / "signals.plt")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("Save view"), start, tr("Views") + " (*.plt)")
        if not path:
            return
        try:
            save_plt(self.script, path)
        except PltError as exc:
            self._error(str(exc))
            return
        self._script_snapshot = copy.deepcopy(self.script)
        self.settings.push_recent("recent_plts", path)
        self.settings.set("current_plt", str(path))
        self._note_script(str(path), True)
        self._update_title()
        self._update_actions()

    def reload_script(self) -> None:
        if self.script.path is not None and self._load_script(self.script.path, remember=False):
            self.rebuild(keep_view=True)

    # ------------------------------------------------------------------ view commands
    def _toggle_reference_line(self, checked: bool = False) -> None:
        self.view.set_reference_line(self.act_ref.isChecked())

    def _toggle_xy(self, checked: bool = False) -> None:
        wanted = self.act_xy.isChecked()
        on = self.view.set_xy(wanted)
        if wanted and not on:
            self._error(tr("The XY view needs at least two analog signals with data"))
        self._update_actions()

    def time_range_dialog(self) -> None:
        if self.dataset is None:
            return
        dialog = TimeRangeDialog(self.dataset.time_range(), self.view.x_range(), self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.view.set_x_range(*dialog.values())

    def _on_view_command(self, command: str) -> None:
        if command == "reference_line_toggle":
            if self.act_ref.isEnabled():
                self.act_ref.setChecked(not self.act_ref.isChecked())
                self.view.set_reference_line(self.act_ref.isChecked())
        elif command == "reference_line_off":
            self.act_ref.setChecked(False)
        elif command == "edit":
            self.edit_signals()
        elif command == "reload":
            self.reload_script()
        elif command == "info":
            self.show_info()
        elif command == "copy_values":
            self.copy_values()
        elif command == "statistics":
            self.copy_statistics()
        elif command == "toggle_script":
            self.toggle_last_script()
        elif command == "copy_name":
            index = self.view.selected()
            if 0 <= index < len(self.signals):
                QtWidgets.QApplication.clipboard().setText(self.signals[index].name)

    def _on_ranges_changed(self) -> None:
        """A range was moved or scaled with the keys: the view (.plt) is changed."""
        if not self.script.modified:
            self.script.modified = True
            self._update_title()

    def _on_dividers(self, legend: float, digital: float) -> None:
        self.settings.set("legend_fraction", legend)
        self.settings.set("digital_fraction", digital)

    def _set_option(self, key: str, value) -> None:
        self.settings.set(key, value)
        self.view.apply_settings(self.settings)

    def appearance_dialog(self) -> None:
        AppearanceDialog(self.settings, lambda: self.view.apply_settings(self.settings), self).exec_()
        self._build_actions()                       # menu check marks follow the new values
        self._update_actions()

    def settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.settings.save()
        if any(key in PROCESSING_KEYS for key in dialog.changed_keys()):
            self.rebuild(keep_view=True)
        self.view.apply_settings(self.settings)

    def _toggle_full_screen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _arrange(self, mode: str) -> None:
        if not self.signals:
            return
        if mode == "auto":
            arrange.autoscale(self.signals)
        elif mode == "auto_view":
            x0, x1 = self.view.x_range()
            arrange.autoscale(self.signals, t0=x0, t1=x1)
        elif mode == "unit":
            arrange.scale_by_unit(self.signals)
        elif mode == "bands":
            arrange.arrange_bands(self.signals)
        elif mode == "unit_bands":
            arrange.arrange_bands_by_unit(self.signals)
        elif mode == "colors":
            arrange.apply_cyclic_colors(self.signals, self.settings.color_range())
        elif mode == "restore":
            snapshot = getattr(self, "_script_snapshot", None)
            if snapshot is None:
                return
            for old, new in zip(snapshot.entries(), self.script.entries()):
                if old.name == new.name:
                    new.scmin, new.scmax, new.scansc = old.scmin, old.scmax, old.scansc
            self.rebuild(keep_view=True)
            return
        self.script.modified = True
        self.view.refresh()
        self._update_title()

    def reference_lines(self) -> None:
        index = self.view.selected()
        if not 0 <= index < len(self.signals):
            names = [s.name for s in self.signals]
            name, ok = QtWidgets.QInputDialog.getItem(self, tr("Reference lines"), tr("Signal"), names, 0, False)
            if not ok:
                return
            index = names.index(name)
        sig = self.signals[index]
        dialog = ReferenceLineDialog(sig, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            sig.entry.refs = dialog.refs()
            self.script.modified = True
            self.rebuild(keep_view=True)

    def add_computed_signal(self, duplicate: bool = False) -> None:
        if self.dataset is None:
            return
        source = self.view.selected() if duplicate is True else -1
        if duplicate is True and source < 0 and self.signals:
            source = 0
        used = sum(1 for _ in self.script.entries())
        lo, hi = self.settings.color_range()
        dialog = ComputedSignalDialog(self.signals, source, self, color=lo + (used + 1) % (hi - lo + 1))
        if dialog.exec_() != QtWidgets.QDialog.Accepted or dialog.entry() is None:
            return
        entry = dialog.entry()
        # Below its source and below a second signal it uses (x1), which must be defined first.
        self._insert_entry(entry, [dialog.source_entry(), dialog.other_entry()])

    def _insert_entry(self, entry: PltEntry, after: List[Optional[PltEntry]]) -> None:
        """Insert a new line below the last of ``after`` (by identity) and select it. Selection and emphasis
        stay with their lines; while signals are emphasized the new signal is emphasized too, so it is seen."""
        items = self.script.items
        used = [i for i, item in enumerate(items) if any(item is e for e in after if e is not None)]
        position = max(used) + 1 if used else len(items)
        items.insert(position, entry)
        self.script.modified = True
        _selected, emphasized, _temporary = self._marked_entries()
        self.rebuild(keep_view=True)
        self._restore_marks((entry, emphasized + ([entry] if emphasized else []), None), lambda line: line)
        sig = next((s for s in self.signals if s.entry is entry), None)
        if sig is not None and sig.error:
            self._error(tr("The new signal could not be computed: %s") % sig.error)

    # ------------------------------------------------------------------ value mappings
    def _fill_value_display_menu(self) -> None:
        """Signals > value display of the selected signal: text / text with number / number, state band.
        (A right click on a legend row opens the view editor, which offers the same.)"""
        menu = self.value_display_menu
        menu.clear()
        index = self.view.selected()
        sig = self.signals[index] if 0 <= index < len(self.signals) else None
        has_text = sig is not None and bool(sig.enum)
        current = (sig.value_mode or "both") if sig is not None else ""
        old = getattr(self, "_value_display_group", None)
        if old is not None:
            try:
                old.deleteLater()
            except RuntimeError:
                pass                                   # went with the menu bar of another language
        group = self._value_display_group = QtGui.QActionGroup(menu)
        for mode, label in _MODE_LABELS:
            act = menu.addAction(tr(label))
            act.setCheckable(True)
            act.setChecked(has_text and mode == current)
            act.setEnabled(has_text)
            group.addAction(act)
            act.triggered.connect(lambda _c=False, m=mode, i=index: self.set_value_mode(i, m))
        menu.addSeparator()
        lane = menu.addAction(tr("Show as state band"))
        lane.setCheckable(True)
        lane.setChecked(sig is not None and sig.lane)
        lane.setEnabled(sig is not None and self._band_possible(sig))
        lane.triggered.connect(lambda checked, i=index: self.set_state_band(i, checked))
        if sig is None:
            menu.addSeparator()
            menu.addAction(tr("(select a signal in the legend first)")).setEnabled(False)

    def _band_possible(self, sig: DisplaySignal) -> bool:
        """A state band needs data, a place outside the DIGITAL window and a name a mapping rule can use."""
        return sig.has_data and not sig.digital and bool(self._rule_for(sig)[0])

    def _rule_for_entry(self, entry: PltEntry, info):
        """The mapping rule in effect for a view line and the name a change is stored under: an exact rule in
        effect is changed itself; otherwise (no rule, or a wildcard rule) a rule for this signal is made.
        ``("", None)`` for a line no rule can reach (a computed line or placeholder without key=)."""
        names = value_map_names(entry, info)
        if not names:
            return "", None
        rule = self.valuemaps.assignment_for(*names)
        pattern = entry_pattern(entry, info)
        if rule is not None and rule.pattern.lower() in {name.lower() for name in names}:
            pattern = rule.pattern
        if "*" in pattern or "?" in pattern:
            return "", None                            # a rule under this name would apply to other signals
        return pattern, rule

    def _rule_for(self, sig: DisplaySignal):
        """Current rule of a signal (exact, or the wildcard rule that matches it) and its pattern."""
        return self._rule_for_entry(sig.entry, sig.info)

    def set_value_mode(self, index: int, mode: str) -> None:
        """Show the values of a signal as mapped text, text with number, or raw number."""
        if not 0 <= index < len(self.signals):
            return
        sig = self.signals[index]
        pattern, rule = self._rule_for(sig)
        if not pattern:
            return                                     # a computed line without key=: no rule can reach it
        table = rule.table if rule is not None else (sig.map_name or "")
        lane = rule.lane if rule is not None else sig.lane
        self.valuemaps.assign(pattern, table, mode, lane)
        self._valuemaps_changed()

    def set_state_band(self, index: int, on: bool) -> None:
        if not 0 <= index < len(self.signals):
            return
        sig = self.signals[index]
        pattern, rule = self._rule_for(sig)
        if not pattern:
            return
        table = rule.table if rule is not None else (sig.map_name or "")
        mode = rule.mode if rule is not None else (sig.value_mode or "both")       # the band alone: texts unchanged
        self.valuemaps.assign(pattern, table, mode, bool(on))
        self._valuemaps_changed()

    def _valuemaps_changed(self) -> None:
        self.valuemaps.save()
        selected = self.view.selected()
        self.rebuild(keep_view=True)
        if 0 <= selected < len(self.signals):
            self.view.select(selected)

    def value_mapping_dialog(self, index: int = -1) -> None:
        current = index if 0 <= index < len(self.signals) else self.view.selected()
        dialog = ValueMapDialog(self.valuemaps, self.signals, current, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.valuemaps = dialog.result_library()
        self.valuemaps.save()
        request = dialog.numeric_request()
        if request is not None and self.dataset is not None:
            self.add_mapped_signal(*request)
        else:
            self._valuemaps_changed()

    def add_mapped_signal(self, index: int, table_name: str) -> Optional[PltEntry]:
        """Add a script line that turns the raw values of signal ``index`` into the numbers of a table
        (``exp=map(x, ...)``), so the logical values can be drawn and compared as a curve."""
        entry = self._mapped_entry(index, table_name)
        if entry is not None:
            self._insert_entry(entry, [self.signals[index].entry])
        return entry

    def _mapped_entry(self, index: int, table_name: str) -> Optional[PltEntry]:
        """The line of :meth:`add_mapped_signal`, not inserted yet."""
        if not 0 <= index < len(self.signals) or table_name not in self.valuemaps.tables:
            return None
        table = self.valuemaps.tables[table_name]
        try:
            formula = table.expression()
        except ValueError as exc:
            self._error(str(exc))
            return None
        sig = self.signals[index]
        lo, hi = self.settings.color_range()
        used = sum(1 for _ in self.script.entries())
        entry = PltEntry(name=sig.entry.name, view="nobit", color=lo + (used + 1) % (hi - lo + 1),
                         unit=table.unit or None, channel_origin=sig.entry.channel_origin)
        entry.bit_pos, entry.bit_count, entry.bit_signed = sig.entry.bit_pos, sig.entry.bit_count, sig.entry.bit_signed
        entry.ops = list(sig.entry.ops)
        entry.key = re.sub(r"[\s=]+", "_", "%s_%s" % (sig.name, table_name))
        parse_attributes("exp=" + formula, entry)
        return entry

    # ------------------------------------------------------------------ rules (context of the rules panel)
    def _rules_do(self, what: str) -> None:
        self.rules_dock.show()
        panel = self.rules_panel
        {"new": panel.new_rule, "script": lambda: panel.edit_script(None), "plugin": lambda: panel.edit_plugin(None),
         "run": panel.run_all, "report": panel.export_xlsx}[what]()

    def has_data(self) -> bool:
        return self.dataset is not None

    def view_range(self):
        return self.view.x_range()

    def rule_source(self) -> SignalSource:
        return SignalSource(self.signals, self.dataset)

    def rule_value_names(self, rule) -> dict:
        source = self.rule_source()
        out = {}
        for name in rule.signals():
            step = source.get(name)
            if step is not None and step.enum:
                out[name] = dict(step.enum.values)
        return out

    def signal_database(self):
        """The signal databases of the settings (loaded on first use)."""
        if self._candb is None:
            from ..candb import CanDbError, SignalDatabase
            self._candb = SignalDatabase()
            for path in self.settings.text_list("signal_databases"):
                try:
                    self._candb.load(path)
                except CanDbError as exc:
                    self.statusBar().showMessage(str(exc), 8000)
        return self._candb

    def rule_context(self) -> dict:
        source = self.rule_source()
        names = [s.name for s in self.signals if s.has_data]
        all_names = []
        if self.dataset is not None and len(self.dataset.signals) <= 200000:
            all_names = [s.name for s in self.dataset.signals]
        db = self.signal_database()

        def enums(name: str) -> dict:
            step = source.get(name) if name else None
            return dict(step.enum.values) if step is not None and step.enum else {}

        def period(name: str):
            step = source.get(name) if name else None
            return step.period if step is not None else None

        def hint(name: str) -> str:
            if not name:
                return ""
            spec = db.find(name) if db is not None else None
            if spec is not None and spec.resolution:
                text = tr("resolution %g %s (%s)") % (spec.resolution, spec.unit, spec.source)
                if spec.cycle_ms:
                    text += ", " + tr("cycle %g ms") % spec.cycle_ms
                if spec.values:
                    return "0|" + text + "; " + tr("value texts: compare exactly")
                return "%g|%s" % (spec.tolerance_hint(), text)
            step = source.get(name)
            if step is None or len(step.y) < 2:
                return ""
            import numpy as np
            values = np.unique(step.y[np.isfinite(step.y)])
            if len(values) < 2:
                return ""
            if enums(name):
                return "0|" + tr("value texts: compare exactly")
            step = float(np.min(np.diff(values)))
            multiples = (values - values[0]) / step
            if len(values) <= 5000 and np.allclose(multiples, np.round(multiples), atol=1e-6):
                return "%g|%s" % (step / 2.0, tr("the data changes in steps of %g (no database entry)") % step)
            return "|" + tr("continuous values without a fixed step: choose the tolerance from the requirement, "
                            "e.g. 1 % of the value (or add the DBC / CAN matrix under Configuration)")

        return {"names": names, "all_names": all_names, "enums": enums, "hint": hint, "period": period}

    def _job(self, rng=None) -> dict:
        from ..plt.parser import dump_plt
        from .settings_dialog import PROCESSING_KEYS
        self.valuemaps.save()
        return {"dataset": str(self.dataset.path), "plt_text": dump_plt(self.script),
                "valuemaps": str(self.valuemaps.path),
                "settings": {k: self.settings.get(k) for k in PROCESSING_KEYS}, "range": list(rng) if rng else None,
                "language": language()}

    def run_script(self, code: str, name: str, rng=None) -> dict:
        from ..rules.runner import run_job
        job = self._job(rng)
        job.update(script=code, script_name=name)
        return run_job(job, timeout=float(self.settings.number("rule_timeout", 120, 5, 3600)))

    def run_plugin(self, path: str, params: dict, name: str, rng=None) -> dict:
        from ..rules.runner import run_job
        job = self._job(rng)
        job.update(plugin=path, params=params, script_name=name)
        return run_job(job, timeout=float(self.settings.number("rule_timeout", 120, 5, 3600)))

    def jump_to(self, t0: float, t1=None) -> None:
        """Show an event: a marker at its time, the view around it, the legend with its values."""
        span = self.view.x_range()[1] - self.view.x_range()[0]
        want = max(2.0, 4.0 * ((t1 - t0) if t1 is not None and t1 > t0 else 0.5))
        self.view.set_marker(t0, span=want if span > 20 or span < want else None)
        self.image_panel.show_time(t0)
        self.statusBar().showMessage(tr("Event at %.3f s") % t0, 5000)

    def set_rule_lanes(self, lanes) -> None:
        self.view.state.rule_lanes = list(lanes)
        self.view.refresh()

    def report_default_path(self) -> str:
        folder = Path(self.settings.get("export/dir", "") or (self.dataset.path.parent if self.dataset else Path.home()))
        stem = self.dataset.path.stem if self.dataset else "rules"
        return str(folder / (stem + "_rules.xlsx"))

    def report_meta(self) -> dict:
        from .. import __version__ as version
        meta = {"数据文件": str(self.dataset.path) if self.dataset else "",
                "View": str(self.script.path or tr("(new view)")),
                "时间范围": "%s" % (self.rules_panel.range.currentText()),
                "可见时间段 [s]": "%.6g ... %.6g" % self.view.x_range(),
                "数值映射库": str(self.valuemaps.path), "规则库": str(self.ruleset.path),
                "信号数据库": "; ".join(self.settings.text_list("signal_databases")),
                "TracePaddock 版本": version}
        return meta

    def signal_database_dialog(self) -> None:
        start = self.settings.get("last_dataset_dir", "") or str(Path.home())

        def take_values(db) -> None:
            names = [s.info.name for s in self.signals if s.info is not None] or None
            imported = db.to_mapping_import(names)
            if not imported.tables:
                QtWidgets.QMessageBox.information(self, APP_NAME, tr("None of the shown signals has a value table in the database."))
                return
            tables, rules = self.valuemaps.merge(imported)
            self._valuemaps_changed()
            QtWidgets.QMessageBox.information(self, APP_NAME, tr("%d value tables taken over as value mappings") % tables)

        dialog = SignalDatabaseDialog(self.settings.text_list("signal_databases"), start, take_values, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.settings.set("signal_databases", dialog.files())
            self.settings.save()
            self._candb = dialog.database()

    # ------------------------------------------------------------------ export / print
    def export_data(self) -> None:
        if self.dataset is None or not self.signals:
            return
        folder = Path(self.settings.get("export/dir", "") or self.dataset.path.parent)
        ext = str(self.settings.get("export/format", ".csv"))
        default = folder / (self.dataset.path.stem + "_export" + ext)
        dialog = ExportDialog(self.signals, self.view.selected(), self.dataset.time_range(), self.view.x_range(),
                              self.settings, str(default), self, emphasized=sorted(self.view.state.emphasized))
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        o = dialog.options()
        try:
            count = self.write_export(o)
        except ExportError as exc:
            self._error(str(exc))
            return
        self.statusBar().showMessage(tr("%d signals written to %s") % (count, o["path"]), 10000)

    def write_export(self, o: dict) -> int:
        """Write an export described by :meth:`ExportDialog.options`. Raises :class:`ExportError`."""
        fmt, path, signals = o["format"], o["path"], o["signals"]
        t0, t1 = o["range"]
        progress = self._progress(tr("Exporting %s ...") % Path(path).name)
        report = self._reporter(progress)
        try:
            with self._busy():
                if fmt in (".csv", ".txt"):
                    write_csv(path, signals, t0, t1, step=o["step"], separator=o["separator"], decimal=o["decimal"],
                              number_format=o["number_format"], quote=o["quote"], header=o["header"],
                              text_values=o["text_values"], progress=report)
                elif fmt == ".mat":
                    write_mat(path, signals, t0, t1, common_axis=o["mat_common"], step=o["step"], progress=report)
                elif fmt == ".mf4":
                    write_mdf(path, signals, t0, t1, version="4.10", progress=report)
                elif fmt == ".mdf":
                    write_mdf(path, signals, t0, t1, version="3.30", progress=report)
                else:
                    raise ExportError(tr("Unknown export format %s") % fmt)
        finally:
            progress.close()
        return len(signals)

    def _picture_state(self):
        state = picture_state(self.view.state, bool(self.settings.get("print_white_background", True)))
        state.cursor = self.view.cursor_time()
        return state

    def _header(self) -> List[str]:
        return header_lines(self.dataset.path if self.dataset else "", self.script.path, self.view.x_range())

    def export_picture(self) -> None:
        if self.dataset is None:
            return
        start = str(Path(self.settings.get("export/dir", "") or self.dataset.path.parent) / (self.dataset.path.stem + ".png"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("Export picture"), start,
                                                        "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;JPEG (*.jpg)")
        if not path:
            return
        try:
            save_picture(path, self.view.size(), self.signals, self._picture_state(), self._header())
        except OSError as exc:
            self._error(str(exc))
            return
        self.settings.set("export/dir", str(Path(path).parent))
        self.statusBar().showMessage(tr("Saved %s") % path, 6000)

    def print_current(self) -> None:
        if self.dataset is not None:
            print_view(self, self.settings, self.view.size(), self.signals, self._picture_state(), self._header())

    def print_preview_current(self) -> None:
        if self.dataset is not None:
            print_preview(self, self.settings, self.view.size(), self.signals, self._picture_state(), self._header())

    # ------------------------------------------------------------------ clipboard / info
    def copy_picture(self) -> None:
        QtWidgets.QApplication.clipboard().setImage(self.view.grab_image())
        self.statusBar().showMessage(tr("Picture copied to the clipboard"), 4000)

    def copy_values(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.view.values_text())
        self.statusBar().showMessage(tr("Values copied to the clipboard"), 4000)

    def copy_statistics(self) -> None:
        if self.dataset is None:
            return
        base = self.dataset.common_time_base() if self.settings.get("statistics_on_common_time_base", True) else None
        with self._busy():
            text = statistics_text(self.dataset.path.name, self.signals, self.dataset.time_range(),
                                   self.view.x_range(), base)
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage(tr("Statistics of %d signals copied to the clipboard") % len(self.signals), 4000)

    def show_info(self) -> None:
        if self.dataset is None:
            return
        index = self.view.selected()
        if 0 <= index < len(self.signals):
            sig = self.signals[index]
            base = self.dataset.common_time_base() if self.settings.get("statistics_on_common_time_base", True) else None
            stats = signal_statistics(sig, *self.dataset.time_range(), base)
            InfoDialog(signal_info_sections(sig, self.dataset, stats), tr("Signal info") + " - " + sig.name, self).exec_()
        else:
            InfoDialog(self.dataset.info(), tr("Data set properties") + " - " + self.dataset.path.name, self).exec_()

    def show_shortcuts(self) -> None:
        rows = "".join("<tr><td><b>%s</b>&nbsp;&nbsp;</td><td>%s</td></tr>" % (tr(key), tr(text))
                       for key, text in SHORTCUT_HELP)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(tr("Keyboard and mouse"))
        box.setTextFormat(QtCore.Qt.RichText)
        box.setText("<table>%s</table>" % rows)
        box.exec_()

    def _show_warmup(self) -> None:
        if self._warmup_dialog is None:
            self._warmup_dialog = WarmupDialog(self.open_dataset, self)
        self._warmup_dialog.update_language()
        self._warmup_dialog.show()
        self._warmup_dialog.raise_()
        self._warmup_dialog.activateWindow()

    def show_about(self) -> None:
        from .. import BUILD
        QtWidgets.QMessageBox.about(self, APP_NAME, "<b>%s %s</b><br>%s<br><br>%s<br>%s<br><br>%s" % (
            APP_NAME, __version__, tr("build %s") % BUILD,
            tr("Viewer and analysis tool for automotive measurement data."),
            tr("A view (.plt file) stores which signals are shown and how."),
            tr("Made by %s") % AUTHOR))

    def _set_language(self, code: str) -> None:
        self.settings.set("language", code)
        set_language(code)
        self._build_actions()
        self._update_actions()
        self._update_title()
        if self._warmup_dialog is not None:
            self._warmup_dialog.update_language()

    # ------------------------------------------------------------------ events
    def _set_pointer_filter(self, enabled: bool) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None and enabled != self._pointer_filter_installed:
            if enabled:
                app.installEventFilter(self)
            else:
                app.removeEventFilter(self)
            self._pointer_filter_installed = enabled

    def eventFilter(self, watched, event) -> bool:
        if (event.type() == QtCore.QEvent.Show and self._pointer is not None
                and isinstance(watched, (QtWidgets.QDialog, QtWidgets.QMenu))
                and not watched.testAttribute(QtCore.Qt.WA_SetCursor)):
            owner = watched.parentWidget()
            while owner is not None and owner is not self:
                owner = owner.parentWidget()
            if owner is self:
                watched.setCursor(self._pointer)
                self._pointer_windows.add(watched)
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        plt = next((p for p in paths if p.lower().endswith(".plt")), None)
        data = next((p for p in paths if is_supported(p)), None)
        if data:
            self.open_dataset(data, plt)
        elif plt and self.dataset is not None:
            self.open_plt(plt)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.script.modified and any(True for _ in self.script.entries()):
            answer = QtWidgets.QMessageBox.question(
                self, APP_NAME, tr("The view was changed. Save it?"),
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
            if answer == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QtWidgets.QMessageBox.Save:
                self.save_plt()
                if self.script.modified:
                    event.ignore()
                    return
        self.settings.set("window/width", self.width())
        self.settings.set("window/height", self.height())
        self.settings.set("window/layout", self.layout_state())
        self.settings.save()
        if self.dataset is not None:
            self.dataset.close()
        self._set_pointer_filter(False)
        super().closeEvent(event)


def _warm_up_imports() -> None:
    """Import the MDF library in the background: it takes seconds, and the first MF4 would wait for it."""
    import threading

    def work() -> None:
        try:
            import asammdf  # noqa: F401
        except Exception:
            pass

    threading.Thread(target=work, name="warm-up", daemon=True).start()


class _Restore:
    """Context manager that removes the wait cursor again."""

    def __enter__(self) -> "_Restore":
        return self

    def __exit__(self, *exc: object) -> None:
        QtWidgets.QApplication.restoreOverrideCursor()
