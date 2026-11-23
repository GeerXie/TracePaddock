"""The signal selection / PLT script editor dialog.

Left: all signals of the recording with a filter. Right: the signals of the
script with their properties. The dialog works on a copy of the script; the
caller takes the result with :meth:`PltEditorDialog.script` after ``exec_()``.
"""
from __future__ import annotations

import copy
import math
import fnmatch
import re
from typing import Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.model import Dataset, SignalInfo
from ..i18n import tr
from ..plt.model import PltComment, PltEntry, PltGroupBegin, PltGroupEnd, PltScript
from ..plt.palette import DEFAULT_COLOR, DEFAULT_PALETTE, HIDDEN_COLOR, color_rgb, cyclic_colors
from ..core.model import KIND_BIT
from ..plt.expr import ExprError, compile_expr
from ..plt.model import BIT_DELETING_OPS
from ..plt.parser import entry_from_signal, format_attributes, parse_attributes, short_name, value_problem
from ..plt.process import build_display_signals
from .color_dialog import CyclicColorsDialog
from .render import ViewState, signal_color

__all__ = ["PltEditorDialog", "color_combo"]

FILTER_MODES = ["?,*", "?,* (a!=A)", "REGEX", "REGEX (a!=A)"]
#: how a signal with a text table writes its values (value display row)
VALUE_MODES = (("mapped", "Mapped value"), ("both", "Mapped value (raw value)"), ("raw", "Raw value"))


def _color_icon(rgb, width: int, height: int) -> QtGui.QIcon:
    pix = QtGui.QPixmap(width, height)
    pix.fill(QtGui.QColor(*rgb))
    painter = QtGui.QPainter(pix)
    painter.setPen(QtGui.QColor(120, 120, 120))
    painter.drawRect(0, 0, width - 1, height - 1)
    painter.end()
    return QtGui.QIcon(pix)


def color_combo(parent: Optional[QtWidgets.QWidget] = None, palette=DEFAULT_PALETTE, *,
                view_state: Optional[ViewState] = None) -> QtWidgets.QComboBox:
    """Combo box with one swatch per palette colour (item data = colour number)."""
    combo = QtWidgets.QComboBox(parent)
    combo.setIconSize(QtCore.QSize(36, 14))
    for index in range(len(palette)):
        if view_state is None:
            rgb = color_rgb(index, palette)
        else:
            rgb = view_state.background if index == HIDDEN_COLOR else signal_color(index, view_state)
        label = ("%d" % index) + ("  " + tr("(not drawn)") if index == HIDDEN_COLOR else "")
        combo.addItem(_color_icon(rgb, 36, 14), label, index)
    return combo


class _NameModel(QtCore.QAbstractListModel):
    """List model over catalogue signals; cheap enough for 60 000 rows."""

    def __init__(self, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self.rows: List[SignalInfo] = []
        self.used: set = set()

    def set_rows(self, rows: List[SignalInfo]) -> None:
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        sig = self.rows[index.row()]
        if role == QtCore.Qt.DisplayRole:
            return sig.name + ("    [" + self.source_label(sig) + "]" if sig.shared_name else "")
        if role == QtCore.Qt.ToolTipRole:
            parts = [sig.name]
            if sig.origin_label:
                parts.append(self.source_label(sig))
            if sig.unit:
                parts.append(tr("Unit") + ": " + sig.unit)
            if sig.raster:
                parts.append(tr("Raster") + ": " + sig.raster)
            if sig.sample_count is not None:
                parts.append(tr("Samples") + ": %d" % sig.sample_count)
            if sig.description:
                parts.append(sig.description)
            return "\n".join(parts)
        if role == QtCore.Qt.ForegroundRole and sig.index in self.used:
            return QtGui.QBrush(QtGui.QColor(0, 140, 0))
        return None

    @staticmethod
    def source_label(info: SignalInfo) -> str:
        channel = info.channel_origin.split(".")[2]
        return info.origin_label + " · " + tr("Channel %s") % channel


class PltEditorDialog(QtWidgets.QDialog):
    """Choose the signals to show and edit their properties."""

    def __init__(self, dataset: Optional[Dataset], script: PltScript, settings: object,
                 parent: Optional[QtWidgets.QWidget] = None, *, select_entry: Optional[PltEntry] = None,
                 value_texts: Optional[Callable[[PltEntry], Optional[dict]]] = None,
                 value_mapping: Optional[Callable[[PltEntry, QtWidgets.QWidget],
                                                   Optional[Tuple[PltEntry, Optional[PltEntry]]]]] = None,
                 view_state: Optional[ViewState] = None) -> None:
        """``value_texts(line)`` tells how the shown signal of a line of ``script`` writes its values:
        ``{"pattern", "text", "mode", "lane", "lane_possible"}`` (``pattern`` = name of the mapping rule a change
        goes to, "" = none possible), ``None`` = not shown. ``value_mapping(line, parent)`` opens the value
        mapping for it and may return ``(new line, line of script it belongs below)``. Without them the value
        display row stays disabled."""
        super().__init__(parent)
        self.setWindowTitle(tr("Signals (view editor)"))
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMaximizeButtonHint)
        self.dataset = dataset
        self.settings = settings
        self._color_range = settings.color_range()
        self._color_state = view_state or ViewState()
        self._script = copy.deepcopy(script)
        #: (line of the copy, line of the caller's view) - the caller knows the shown signal of its own lines;
        #: the list keeps the copies alive, so their ids stay unique keys
        self._pairs = list(zip(self._script.items, script.items))
        self._origin = {id(mine): source for mine, source in self._pairs}
        self._value_texts = value_texts
        self._value_mapping = value_mapping
        #: value display chosen here, applied by the caller after OK, per mapping rule (lines that share a rule
        #: share the change): pattern.lower() -> {"fields": only what the user changed ("mode", "lane"),
        #: "lines": the caller's lines it was chosen on}
        self._value_changes: Dict[str, dict] = {}
        self._value_state: Optional[dict] = None
        self._value_origin: Optional[PltEntry] = None
        #: lines added with the value mapping dialog's "numeric signal" (the caller selects them after OK)
        self.mapped_lines: List[PltEntry] = []
        self._loading = False
        self._build_ui()
        self._refresh_filter()
        self._refill_entries()
        if select_entry is not None:
            row = next((i for i, item in enumerate(script.items) if item is select_entry), -1)
            if row >= 0:                                 # by identity: two equal lines are still two lines
                self.entry_list.setCurrentRow(row)
        elif self.entry_list.count():
            self.entry_list.setCurrentRow(0)
        self.resize(int(settings.number("editor/width", 1150, 500, 8000)),
                    int(settings.number("editor/height", 720, 400, 8000)))

    # ------------------------------------------------------------------ result
    def script(self) -> PltScript:
        return self._script

    def original_entry(self, entry: object) -> Optional[PltEntry]:
        """The line of the caller's view that ``entry`` (a line of the result) was copied from; ``None`` for a
        line added here."""
        return self._origin.get(id(entry))

    def value_text_changes(self) -> List[Tuple[PltEntry, dict]]:
        """Value display chosen in the editor: ``(line of the caller's view, {"mode": ..., "lane": ...})`` with
        only the fields the user changed. A change stays while a line of that signal is left in the view."""
        alive_lines = [self._origin[id(item)] for item in self._script.items if id(item) in self._origin]
        alive = {id(line) for line in alive_lines}
        out = []
        for key, change in self._value_changes.items():
            if not change["fields"]:
                continue
            line = next((line for line in change["lines"] if id(line) in alive), None)
            if line is None and self._value_texts is not None:     # chosen on a line that was deleted since
                line = next((other for other in alive_lines
                             if str((self._value_texts(other) or {}).get("pattern") or "").lower() == key), None)
            if line is not None:
                out.append((line, dict(change["fields"])))
        return out

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        outer.addWidget(splitter, 1)

        # ---- left: recording
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        name = self.dataset.path.name if self.dataset is not None else tr("(no recording loaded)")
        lv.addWidget(QtWidgets.QLabel("<b>%s</b><br>%s" % (tr("Signals within dataset"), name)))
        row = QtWidgets.QHBoxLayout()
        self.filter_edit = QtWidgets.QComboBox()
        self.filter_edit.setEditable(True)
        self.filter_edit.addItems(self.settings.text_list("filter_history"))
        self.filter_edit.setCurrentText("*")
        self.filter_edit.setToolTip(tr("Filter: * = any characters, ? = one character; or a regular expression"))
        self.filter_mode = QtWidgets.QComboBox()
        self.filter_mode.addItems(FILTER_MODES)
        self.filter_mode.setCurrentIndex(int(self.settings.number("editor/filter_mode", 0, 0, len(FILTER_MODES) - 1)))
        row.addWidget(self.filter_edit, 1)
        row.addWidget(self.filter_mode)
        lv.addLayout(row)
        self.name_model = _NameModel(self)
        self.name_view = QtWidgets.QListView()
        self.name_view.setModel(self.name_model)
        self.name_view.setUniformItemSizes(True)
        self.name_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.name_view.doubleClicked.connect(lambda _i: self._add_selected())
        lv.addWidget(self.name_view, 1)
        self.count_label = QtWidgets.QLabel()
        lv.addWidget(self.count_label)
        splitter.addWidget(left)

        # ---- middle: buttons
        mid = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(mid)
        mv.setContentsMargins(4, 30, 4, 0)

        def button(text: str, slot, tip: str = "") -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            mv.addWidget(b)
            return b

        button(tr("Add") + "  >>", self._add_selected, tr("Add the marked signals to the script (double click)"))
        button(tr("Delete"), self._delete_entries, tr("Remove the marked script entries (Del)"))
        button(tr("Duplicate"), self._duplicate_entry, tr("Copy the marked entry, e.g. to process the copy (Ctrl+D)"))
        button(tr("Enable / disable"), self._toggle_entries, tr("Hide or show the marked entries without removing them (~)"))
        button(tr("Insert placeholder"), self._add_placeholder,
               tr("A signal that exists only through its formula, e.g. exp=sin(2*PI*t)"))
        button(tr("Add comment"), self._add_comment)
        mv.addSpacing(12)
        button(tr("Up"), lambda: self._move(-1))
        button(tr("Down"), lambda: self._move(1))
        mv.addStretch(1)
        splitter.addWidget(mid)

        # ---- right: script
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.script_label = QtWidgets.QLabel()
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.script_label, 1)
        self.cyclic_color_button = QtWidgets.QPushButton(tr("Cyclic colours..."))
        self.cyclic_color_button.clicked.connect(self._open_cyclic_colors)
        header.addWidget(self.cyclic_color_button)
        rv.addLayout(header)
        self.entry_list = QtWidgets.QListWidget()
        self.entry_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.entry_list.currentRowChanged.connect(self._load_properties)
        self.entry_list.itemDoubleClicked.connect(lambda _i: self._toggle_entries())
        self._entry_viewport = self.entry_list.viewport()
        self._entry_viewport.installEventFilter(self)
        rv.addWidget(self.entry_list, 1)
        copy_options = QtWidgets.QHBoxLayout()
        copy_options.addWidget(QtWidgets.QLabel(tr("Copy on right-click")))
        self.copy_range = QtWidgets.QCheckBox(tr("Display range"))
        self.copy_range.setChecked(bool(self.settings.get("editor/copy_range", True)))
        self.copy_unit = QtWidgets.QCheckBox(tr("Unit"))
        self.copy_unit.setChecked(bool(self.settings.get("editor/copy_unit", False)))
        copy_options.addWidget(self.copy_range)
        copy_options.addWidget(self.copy_unit)
        copy_options.addStretch(1)
        rv.addLayout(copy_options)
        self.copy_hint = QtWidgets.QLabel(tr("Left-click one source signal, then right-click a target signal."))
        self.copy_hint.setWordWrap(True)
        rv.addWidget(self.copy_hint)
        self.missing_note = QtWidgets.QLabel()
        self.missing_note.setWordWrap(True)
        rv.addWidget(self.missing_note)

        self.props = QtWidgets.QGroupBox(tr("Edit signal properties"))
        form = QtWidgets.QGridLayout(self.props)
        self.key_edit = QtWidgets.QLineEdit()
        self.key_edit.setPlaceholderText(tr("(recorded name)"))
        self.unit_edit = QtWidgets.QLineEdit()
        self.min_edit = QtWidgets.QLineEdit()
        self.max_edit = QtWidgets.QLineEdit()
        self.auto_check = QtWidgets.QCheckBox(tr("AUTOSCALING"))
        self.color_box = color_combo(self, view_state=self._color_state)
        self.ref_edit = QtWidgets.QLineEdit()
        self.ref_edit.setPlaceholderText(tr("e.g.  0; min; max"))
        self.digital_check = QtWidgets.QCheckBox(tr("DIGITAL-window"))
        self.fmt_edit = QtWidgets.QLineEdit()
        self.fmt_edit.setPlaceholderText(".2f   8.3e   x   b")
        self.bit_type = QtWidgets.QComboBox()
        self.bit_type.addItems([tr("<none>"), "DIGITAL-1-Bit", "DIGITAL-N-Bit", "DIGITAL-N-Bit SIGNED"])
        self.bit_pos = QtWidgets.QSpinBox()
        self.bit_pos.setRange(0, 31)
        self.bit_count = QtWidgets.QSpinBox()
        self.bit_count.setRange(1, 32)
        self.attr_edit = QtWidgets.QPlainTextEdit()
        self.attr_edit.setPlaceholderText("exp=3.6*x    exp(x1=other)=x-x1    dgf=bth:10:2    mavr(10)    diff    integ    -mean")
        self.attr_edit.setMaximumHeight(70)
        self.attr_error = QtWidgets.QLabel()
        self.attr_error.setStyleSheet("color: #c00000")
        self.name_label = QtWidgets.QLabel()
        self.name_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.source_button = QtWidgets.QPushButton(tr("Choose channel source..."))
        self.source_button.hide()
        self.source_button.clicked.connect(self._choose_source)
        self.mode_box = QtWidgets.QComboBox()
        for mode, label in VALUE_MODES:
            self.mode_box.addItem(tr(label), mode)
        self.mode_box.setToolTip(tr("How the legend and the status bar write a value that has a text"))
        self.lane_check = QtWidgets.QCheckBox(tr("Show as state band"))
        self.lane_check.setToolTip(tr("Labelled segments in the upper window instead of a curve"))
        self.mapping_button = QtWidgets.QPushButton(tr("Value mapping..."))
        self.mapping_button.setToolTip(tr("Assign or edit the text table of this signal (raw value -> text)"))
        self.value_hint = QtWidgets.QLabel()
        self.value_hint.setStyleSheet("color: #707070")
        self.value_hint.setMinimumHeight(self.value_hint.fontMetrics().height())
        r = 0
        form.addWidget(QtWidgets.QLabel(tr("SIGNAL")), r, 0)
        form.addWidget(self.name_label, r, 1, 1, 4)
        form.addWidget(self.source_button, r, 5)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("DISPLAY NAME")), r, 0)
        form.addWidget(self.key_edit, r, 1, 1, 5)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("UNIT")), r, 0)
        form.addWidget(self.unit_edit, r, 1)
        form.addWidget(QtWidgets.QLabel(tr("NUMBER FORMAT")), r, 3)
        form.addWidget(self.fmt_edit, r, 4, 1, 2)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("RANGE")), r, 0)
        form.addWidget(self.min_edit, r, 1)
        form.addWidget(QtWidgets.QLabel("..."), r, 2)
        form.addWidget(self.max_edit, r, 3, 1, 2)
        form.addWidget(self.auto_check, r, 5)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("COLOR")), r, 0)
        form.addWidget(self.color_box, r, 1)
        form.addWidget(QtWidgets.QLabel(tr("REFERENCE LINES")), r, 3)
        form.addWidget(self.ref_edit, r, 4, 1, 2)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("VIEW")), r, 0)
        form.addWidget(self.digital_check, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("VALUE DISPLAY")), r, 0)
        form.addWidget(self.mode_box, r, 1)
        form.addWidget(self.lane_check, r, 3)
        form.addWidget(self.mapping_button, r, 4, 1, 2)
        r += 1
        form.addWidget(self.value_hint, r, 1, 1, 5)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("BITFIELD")), r, 0)
        form.addWidget(self.bit_type, r, 1)
        form.addWidget(QtWidgets.QLabel("BIT-P"), r, 2)
        form.addWidget(self.bit_pos, r, 3)
        form.addWidget(QtWidgets.QLabel("BIT-C"), r, 4)
        form.addWidget(self.bit_count, r, 5)
        r += 1
        form.addWidget(QtWidgets.QLabel(tr("ATTRIBUTES")), r, 0)
        form.addWidget(self.attr_edit, r, 1, 1, 5)
        r += 1
        form.addWidget(self.attr_error, r, 1, 1, 5)
        rv.addWidget(self.props)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(2, 6)

        for widget in (self.key_edit, self.unit_edit, self.min_edit, self.max_edit, self.ref_edit, self.fmt_edit):
            widget.editingFinished.connect(self._store_properties)
        self.auto_check.toggled.connect(self._store_properties)
        self.digital_check.toggled.connect(self._store_properties)
        self.color_box.currentIndexChanged.connect(self._store_properties)
        self.bit_type.currentIndexChanged.connect(self._store_properties)
        self.bit_pos.valueChanged.connect(self._store_properties)
        self.bit_count.valueChanged.connect(self._store_properties)
        self.attr_edit.textChanged.connect(self._store_properties)
        self.mode_box.currentIndexChanged.connect(self._store_value_texts)
        self.lane_check.toggled.connect(self._store_value_texts)
        self.mapping_button.clicked.connect(self._open_value_mapping)
        self.filter_edit.editTextChanged.connect(self._schedule_filter)
        self.filter_mode.currentIndexChanged.connect(self._refresh_filter)
        self._filter_timer = QtCore.QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(180)
        self._filter_timer.timeout.connect(self._refresh_filter)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self._duplicate_entry)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self.entry_list, self._delete_entries,
                            context=QtCore.Qt.WidgetShortcut)

    # ------------------------------------------------------------------ left list
    def _schedule_filter(self) -> None:
        self._filter_timer.start()

    def _refresh_filter(self) -> None:
        if self.dataset is None:
            self.name_model.set_rows([])
            self.count_label.setText("")
            return
        pattern = self.filter_edit.currentText().strip() or "*"
        mode = self.filter_mode.currentIndex()
        regex, case = mode >= 2, bool(mode % 2)
        try:
            rows = self.dataset.filter(pattern, regex=regex, case_sensitive=case)
            if not regex and not rows and "*" not in pattern and "?" not in pattern:
                rows = self.dataset.filter("*" + pattern + "*", case_sensitive=case)   # plain text = "contains"
            self.filter_edit.setStyleSheet("")
        except re.error:
            rows = []
            self.filter_edit.setStyleSheet("QComboBox { background: #ffd0d0; color: #000000 }")
        self.name_model.used = self._used_indices()
        self.name_model.set_rows(rows)
        self.count_label.setText(tr("%d of %d signals") % (len(rows), len(self.dataset.signals)))

    def _used_indices(self) -> set:
        """Catalogue positions of the signals the script already shows (they are listed green)."""
        used = set()
        if self.dataset is not None:
            for entry in self._script.entries():
                info = self.dataset.find(entry.name, entry.channel_origin) if not entry.is_placeholder else None
                if info is not None:
                    used.add(info.index)
        return used

    def _script_name(self, info) -> str:
        """Name written into the script for a catalogue signal.

        MDF 3 recordings append the device (``v_x\\CCP:1``). The short name is
        written when it identifies the signal unambiguously, so the script also
        works with recordings that store the name without the suffix.
        """
        if info.shared_name:
            return info.name
        if "\\" in info.source_name:
            short = info.source_name.split("\\", 1)[0]
            if self.dataset is not None and self.dataset.find(short) is info:
                return short
        return info.name

    def _add_selected(self) -> None:
        rows = sorted(i.row() for i in self.name_view.selectionModel().selectedIndexes())
        if not rows:
            return
        insert_at = self.entry_list.currentRow() + 1 if self.entry_list.currentRow() >= 0 else len(self._script.items)
        used = sum(1 for _ in self._script.entries())
        lo, hi = self._color_range
        for offset, row in enumerate(rows):
            info = self.name_model.rows[row]
            name = self._script_name(info)
            display = short_name(
                name,
                drop_bus_prefix=bool(self.settings.get("short_name_bus_prefix", False)),
                drop_device=bool(self.settings.get("short_name_device", True)),
                keep_elements=int(self.settings.number("short_name_keep", 0, 0, 20)),
                keep_from_left=bool(self.settings.get("short_name_keep_left", True)))
            if info.shared_name:
                _, group, channel, _ = info.channel_origin.split(".")
                display = f"{display or name} [#{group}:{channel}]"
            color = info.color if info.color not in (None, 1) else lo + (used + offset) % (hi - lo + 1)
            self._script.items.insert(insert_at + offset,
                                      entry_from_signal(info, color=color, display_name=display, name=name))
        self._script.modified = True
        self._refill_entries(select=insert_at + len(rows) - 1)
        self._refresh_filter()

    # ------------------------------------------------------------------ right list
    def _entry_color_icon(self, entry: PltEntry) -> QtGui.QIcon:
        index = entry.color if entry.color is not None else DEFAULT_COLOR
        rgb = self._color_state.background if index == HIDDEN_COLOR else signal_color(index, self._color_state)
        return _color_icon(rgb, 12, 12)

    def _open_cyclic_colors(self) -> None:
        self._store_properties()
        all_rows = [row for row, entry in enumerate(self._script.items) if isinstance(entry, PltEntry)]
        selected_rows = [row for row in self._selected_rows() if isinstance(self._script.items[row], PltEntry)]
        if not all_rows:
            return
        dialog = CyclicColorsDialog(self._color_range, self._color_state, len(selected_rows), len(all_rows), self)
        result = dialog.exec_()
        if result not in (QtWidgets.QDialog.Accepted, CyclicColorsDialog.Selected):
            return
        self._color_range = dialog.color_range()
        rows = selected_rows if result == CyclicColorsDialog.Selected else all_rows
        refresh_current = False
        for row, color in zip(rows, cyclic_colors(self._color_range)):
            entry = self._script.items[row]
            if entry.color == color:
                continue
            entry.color = color
            self._script.modified = True
            if entry.enabled:
                self.entry_list.item(row).setIcon(self._entry_color_icon(entry))
            refresh_current |= row == self.entry_list.currentRow()
        if refresh_current:
            self._load_properties(self.entry_list.currentRow())

    def _entry_text(self, item) -> str:
        if isinstance(item, PltEntry):
            text = ("" if item.enabled else "~ ") + item.display_name
            if item.key and item.key != item.name and not item.is_placeholder:
                text += "    [" + item.name + "]"
            if item.ops:
                text += "    " + format_attributes(item)
            return text
        if isinstance(item, PltGroupBegin):
            return '//BEGIN_GROUP "%s"' % item.name
        if isinstance(item, PltGroupEnd):
            return "//END_GROUP"
        return item.text

    def _refill_entries(self, select: Optional[int] = None) -> None:
        self._loading = True
        current = self.entry_list.currentRow() if select is None else select
        self.entry_list.clear()
        missing = 0
        for item in self._script.items:
            row = QtWidgets.QListWidgetItem(self._entry_text(item))
            if isinstance(item, PltEntry):
                if not item.enabled:
                    row.setForeground(QtGui.QColor(60, 90, 220))
                elif item.is_placeholder:
                    row.setForeground(QtGui.QColor(0, 150, 190))
                elif self.dataset is not None and self.dataset.find(item.name, item.channel_origin) is None:
                    missing += 1
                    row.setForeground(QtGui.QColor(210, 0, 0))
                    row.setToolTip(tr("Saved channel source is unavailable; choose the source in the view editor")
                                   if item.channel_origin else tr("This signal is not contained in the dataset"))
                if item.enabled:
                    row.setIcon(self._entry_color_icon(item))
            elif isinstance(item, (PltGroupBegin, PltGroupEnd)):
                row.setForeground(QtGui.QColor(150, 0, 170))
            else:
                row.setForeground(QtGui.QColor(190, 110, 0))
            self.entry_list.addItem(row)
        self._loading = False
        count = sum(1 for _ in self._script.entries())
        self.cyclic_color_button.setEnabled(count > 0)
        path = self._script.path.name if self._script.path else tr("(new script)")
        self.script_label.setText("<b>%s</b> (%d)<br>%s" % (tr("Signals in the view"), count, path))
        self.missing_note.setText(tr("%d PLT signals in red are missing from this recording and will be skipped "
                                     "in the plot.") % missing if missing else "")
        self.missing_note.setVisible(bool(missing))
        if self.entry_list.count():
            self.entry_list.setCurrentRow(min(max(current, 0), self.entry_list.count() - 1))
        else:
            self._load_properties(-1)

    def _selected_rows(self) -> List[int]:
        return sorted(self.entry_list.row(i) for i in self.entry_list.selectedItems())

    def _current_entry(self) -> Optional[PltEntry]:
        row = self.entry_list.currentRow()
        if 0 <= row < len(self._script.items) and isinstance(self._script.items[row], PltEntry):
            return self._script.items[row]
        return None

    def eventFilter(self, watched, event) -> bool:
        if (watched is self._entry_viewport
                and event.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease,
                                     QtCore.QEvent.MouseButtonDblClick)
                and event.button() == QtCore.Qt.RightButton):
            if event.type() == QtCore.QEvent.MouseButtonPress:
                row = self.entry_list.indexAt(event.pos()).row()
                if row >= 0:
                    self._copy_properties_to(row)
            return True
        return super().eventFilter(watched, event)

    def _copy_properties_to(self, row: int) -> None:
        target = self._script.items[row]
        if not isinstance(target, PltEntry) or not (self.copy_range.isChecked() or self.copy_unit.isChecked()):
            return
        selected = self._selected_rows()
        if len(selected) != 1:
            self.copy_hint.setText(tr("Select exactly one source signal to copy its properties."))
            return
        source_row = selected[0]
        source = self._script.items[source_row]
        if not isinstance(source, PltEntry) or source is target:
            return
        self._store_properties()
        before = target.state()
        notice = ""
        if self.copy_range.isChecked():
            limits = self._display_range(source, source_row)
            if limits is None:
                notice = tr("The source display range is unavailable.")
            else:
                target.scmin, target.scmax = limits
                target.scansc = False
        if self.copy_unit.isChecked():
            info = (self.dataset.find(source.name, source.channel_origin)
                    if self.dataset is not None and not source.is_placeholder else None)
            unit = source.unit if source.unit is not None else (info.unit if info is not None else "")
            if value_problem(unit):
                notice = tr("The source unit cannot be stored in a view file.")
            else:
                target.unit = unit
        self.copy_hint.setText(notice or tr("Left-click one source signal, then right-click a target signal."))
        if target.state() != before:
            self._script.modified = True
        self.entry_list.setCurrentRow(row, QtCore.QItemSelectionModel.ClearAndSelect)

    def _display_range(self, entry: PltEntry, row: int) -> Optional[Tuple[float, float]]:
        lo, hi = entry.scmin, entry.scmax
        if not entry.scansc and lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            return lo, hi
        if self.dataset is None:
            return None
        preview = copy.copy(self._script)
        source = copy.copy(entry)
        source.enabled = True
        preview.items = (self._script.items[:row] if source.ops else []) + [source]
        for signal in build_display_signals(self.dataset, preview, self.settings):
            if signal.entry is source and signal.has_data:
                lo, hi = signal.scmin, signal.scmax
                if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
                    return lo, hi
        return None

    def _delete_entries(self) -> None:
        rows = self._selected_rows()
        for row in reversed(rows):
            del self._script.items[row]
        if rows:
            self._script.modified = True
            self._refill_entries(select=rows[0])
            self._refresh_filter()

    def _toggle_entries(self) -> None:
        rows = self._selected_rows()
        for row in rows:
            item = self._script.items[row]
            if isinstance(item, PltEntry):
                item.enabled = not item.enabled
        if rows:
            self._script.modified = True
            self._refill_entries(select=rows[-1])

    def _duplicate_entry(self) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        row = self.entry_list.currentRow()
        clone = copy.deepcopy(entry)
        clone.raw_lines, clone.raw_state = [], None
        clone.key = (entry.display_name + "_2")
        self._script.items.insert(row + 1, clone)
        self._script.modified = True
        self._refill_entries(select=row + 1)
        self.attr_edit.setFocus()

    def _add_placeholder(self) -> None:
        row = self.entry_list.currentRow() + 1
        entry = PltEntry(name="*", view="nobit", color=2, key="T_SIN", unit="", scmin=-2.0, scmax=2.0)
        parse_attributes("exp=sin(2*PI*t/2)", entry)
        self._script.items.insert(row, entry)
        self._script.modified = True
        self._refill_entries(select=row)

    def _add_comment(self) -> None:
        row = self.entry_list.currentRow() + 1
        text, ok = QtWidgets.QInputDialog.getText(self, tr("Add comment"), tr("Comment text"))
        if ok:
            self._script.items.insert(row, PltComment("// " + text))
            self._script.modified = True
            self._refill_entries(select=row)

    def _move(self, step: int) -> None:
        rows = self._selected_rows()
        if not rows or rows[0] + step < 0 or rows[-1] + step >= len(self._script.items):
            return
        items = self._script.items
        for row in (rows if step < 0 else reversed(rows)):
            items[row + step], items[row] = items[row], items[row + step]
        self._script.modified = True
        self._refill_entries(select=rows[0] + step)
        for row in rows:
            self.entry_list.item(row + step).setSelected(True)

    # ------------------------------------------------------------------ properties
    def _widget_values(self) -> dict:
        return {
            "key": self.key_edit.text(), "unit": self.unit_edit.text(),
            "scmin": self.min_edit.text(), "scmax": self.max_edit.text(),
            "scansc": self.auto_check.isChecked(), "color": self.color_box.currentData(),
            "refs": self.ref_edit.text(), "digital": self.digital_check.isChecked(), "fmt": self.fmt_edit.text(),
            "bits": (self.bit_type.currentIndex(), self.bit_pos.value(), self.bit_count.value()),
            "attrs": self.attr_edit.toPlainText(),
        }

    def _choose_source(self) -> None:
        entry = self._current_entry()
        if entry is None or self.dataset is None:
            return
        choices = self.dataset.source_choices(entry.name)
        if not choices:
            return
        labels = [self.name_model.source_label(info) for info in choices]
        current = self.dataset.find(entry.name, entry.channel_origin)
        selected = next((index for index, info in enumerate(choices) if info is current), 0)
        label, ok = QtWidgets.QInputDialog.getItem(self, tr("Choose channel source..."), tr("Channel source"),
                                                  labels, selected, False)
        if ok:
            entry.channel_origin = choices[labels.index(label)].channel_origin
            self._script.modified = True
            self._refill_entries()
            self._refresh_filter()

    def _load_properties(self, row: int) -> None:
        entry = self._current_entry()
        self.props.setEnabled(entry is not None)
        if entry is None:
            return
        self._loading = True
        info = (self.dataset.find(entry.name, entry.channel_origin)
                if self.dataset is not None and not entry.is_placeholder else None)
        self.source_button.setVisible(bool((entry.channel_origin or (info and info.shared_name))
                                           and self.dataset is not None and self.dataset.source_choices(entry.name)))
        detail = (self.name_model.source_label(info) if info is not None and info.shared_name
                  else info.raster if info is not None else "")
        self.name_label.setText(entry.token + ("    (%s)" % detail if detail else ""))
        self.key_edit.setText(entry.key or "")
        self.unit_edit.setText(entry.unit if entry.unit is not None else (info.unit if info is not None else ""))
        self.min_edit.setText("" if entry.scmin is None else "%g" % entry.scmin)
        self.max_edit.setText("" if entry.scmax is None else "%g" % entry.scmax)
        self.auto_check.setChecked(entry.scansc)
        self.color_box.setCurrentIndex(max(0, self.color_box.findData(entry.color if entry.color is not None else 1)))
        self.ref_edit.setText("; ".join(
            ("%g" % v if isinstance(v, float) else str(v).lower()) for _, v in sorted(entry.refs.items())))
        one_bit = (info is not None and info.kind == KIND_BIT) or (entry.bit_pos is not None and (entry.bit_count or 1) == 1)
        implicit = entry.view is None and one_bit and not any(op.kind in BIT_DELETING_OPS for op in entry.ops)
        self.digital_check.setChecked(entry.view == "bit" or implicit)
        self.fmt_edit.setText(entry.fmt or "")
        if entry.bit_pos is None:
            self.bit_type.setCurrentIndex(0)
        elif entry.bit_signed:
            self.bit_type.setCurrentIndex(3)
        else:
            self.bit_type.setCurrentIndex(1 if (entry.bit_count or 1) == 1 else 2)
        self.bit_pos.setValue(entry.bit_pos or 0)
        self.bit_count.setValue(entry.bit_count or 1)
        self.attr_edit.setPlainText(format_attributes(entry))
        self.attr_error.setText("")
        for widget in (self.key_edit, self.unit_edit, self.min_edit, self.max_edit, self.ref_edit):
            self._mark(widget, None)
        self._load_value_texts(entry)
        self._shown = self._widget_values()
        self._loading = False

    # ------------------------------------------------------------------ value display
    def _load_value_texts(self, entry: PltEntry) -> None:
        """The value display row: how the shown signal of the line writes its values, with the changes chosen
        here (also on another line of the same signal); applied with OK."""
        origin = self.original_entry(entry)
        state = self._value_texts(origin) if self._value_texts is not None and origin is not None else None
        pattern = str(state.get("pattern") or "") if state is not None else ""
        self._value_state, self._value_origin = (state if pattern else None), origin
        change = self._value_changes.get(pattern.lower()) if pattern else None
        if change is not None:                       # a change chosen on this or another line of the signal
            mode = change["fields"].get("mode", change["base"]["mode"])
            lane = change["fields"].get("lane", change["base"]["lane"])
        else:
            mode = str(state.get("mode") or "both") if state is not None else "both"
            lane = bool(state.get("lane")) if state is not None else False
        self.mode_box.setCurrentIndex(max(0, self.mode_box.findData(mode)))
        self.lane_check.setChecked(lane)
        has_text = bool(pattern) and bool(state.get("text"))
        self.mode_box.setEnabled(has_text)
        self.lane_check.setEnabled(bool(pattern) and bool(state.get("lane_possible")))
        self.mapping_button.setEnabled(bool(pattern) and self._value_mapping is not None)
        if state is None:
            hint = tr("Value display can be set after OK: the signal is not shown yet")
        elif not pattern:
            hint = tr("A computed line needs a display name (key=) for texts or a state band")
        elif not has_text:
            hint = tr('This signal has no text table; "Value mapping..." assigns one')
        else:
            hint = ""
        self.value_hint.setText(hint)

    def _store_value_texts(self, *_args) -> None:
        """Keep what the user changed in the value display row - per field, so a field left alone keeps what
        the rule has when OK is pressed (e.g. after the value mapping dialog changed it)."""
        state, origin = self._value_state, self._value_origin
        if self._loading or state is None or origin is None:
            return
        key = str(state["pattern"]).lower()
        change = self._value_changes.get(key)
        if change is None:                           # compared with what the line showed when it was first changed
            change = self._value_changes[key] = {
                "fields": {}, "lines": [], "base": {"mode": str(state.get("mode") or "both"),
                                                    "lane": bool(state.get("lane"))}}
        fields, base = change["fields"], change["base"]
        if self.mode_box.isEnabled():
            mode = str(self.mode_box.currentData() or "both")
            if mode != base["mode"]:
                fields["mode"] = mode
            else:
                fields.pop("mode", None)
        if self.lane_check.isEnabled():
            if self.lane_check.isChecked() != base["lane"]:
                fields["lane"] = self.lane_check.isChecked()
            else:
                fields.pop("lane", None)
        if not any(line is origin for line in change["lines"]):
            change["lines"].append(origin)
        if not fields:
            del self._value_changes[key]                      # back to what is shown: nothing to change

    def _open_value_mapping(self) -> None:
        entry = self._current_entry()
        origin = self.original_entry(entry) if entry is not None else None
        if origin is None or self._value_mapping is None:
            return
        result = self._value_mapping(origin, self)
        if result is not None:                                 # a line with the mapped numbers as a curve,
            line, source = result                              # placed below the line it is computed from
            row = next((i for i, item in enumerate(self._script.items)
                        if source is not None and self._origin.get(id(item)) is source), self.entry_list.currentRow())
            self._script.items.insert(row + 1, line)
            self._script.modified = True
            self.mapped_lines.append(line)
            self._revalidate_value_changes()
            self._refill_entries(select=row + 1)
            self._refresh_filter()
            return
        self._revalidate_value_changes()
        self._loading = True
        try:
            self._load_value_texts(entry)                       # the table may have changed
        finally:
            self._loading = False

    def _revalidate_value_changes(self) -> None:
        """After the value mapping dialog changed rules: a change is kept only while its lines still go to the
        same rule, and only the fields that still apply; what it is compared with is taken from the rules again."""
        if self._value_texts is None:
            self._value_changes.clear()
            return
        for key, change in list(self._value_changes.items()):
            states = [(line, self._value_texts(line)) for line in change["lines"]]
            states = [(line, state) for line, state in states
                      if state is not None and str(state.get("pattern") or "").lower() == key]
            if not states:
                del self._value_changes[key]
                continue
            state = states[0][1]
            base = {"mode": str(state.get("mode") or "both"), "lane": bool(state.get("lane"))}
            fields = change["fields"]
            if "mode" in fields and (not state.get("text") or fields["mode"] == base["mode"]):
                del fields["mode"]
            if "lane" in fields and (not state.get("lane_possible") or fields["lane"] == base["lane"]):
                del fields["lane"]
            change.update(base=base, lines=[line for line, _state in states])
            if not fields:
                del self._value_changes[key]

    @staticmethod
    def _float(text: str) -> Optional[float]:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None

    @staticmethod
    def _mark(widget: QtWidgets.QWidget, problem: Optional[str]) -> None:
        """Red background with dark text (readable in every theme) and the reason as tooltip for a value that
        cannot be taken over."""
        widget.setStyleSheet("background: #ffd0d0; color: #000000" if problem else "")
        widget.setToolTip(problem or "")

    def _store_properties(self, *_args) -> None:
        """Take over the fields the user changed since the entry was shown, after checking them."""
        if self._loading:
            return
        entry = self._current_entry()
        if entry is None:
            return
        now = self._widget_values()
        shown = getattr(self, "_shown", now)
        changed = {key for key in now if now[key] != shown.get(key)}
        if not changed:
            return
        before = entry.state()
        taken = set()
        if "key" in changed:
            text = now["key"].strip()
            problem = value_problem(text)
            self._mark(self.key_edit, problem)
            if not problem:
                entry.key = text if text and text != entry.name else None
                taken.add("key")
        if "unit" in changed:
            text = now["unit"].strip()
            problem = value_problem(text)
            self._mark(self.unit_edit, problem)
            if not problem:
                entry.unit = text
                taken.add("unit")
        for key, widget in (("scmin", self.min_edit), ("scmax", self.max_edit)):
            if key in changed:
                text = now[key].strip()
                value = self._float(text) if text else None
                bad = bool(text) and (value is None or value != value or value in (float("inf"), float("-inf")))
                self._mark(widget, tr("Not a number") if bad else None)
                if not bad:
                    setattr(entry, key, value)
                    taken.add(key)
        if "scansc" in changed:
            entry.scansc = now["scansc"]
            taken.add("scansc")
        if "color" in changed:
            entry.color = now["color"]
            taken.add("color")
        if "digital" in changed:
            entry.view = "bit" if now["digital"] else "nobit"
            entry.view_pos = len(entry.ops) if now["digital"] and entry.ops else None   # BIT after the operations
            taken.add("digital")
        if "fmt" in changed:
            entry.fmt = now["fmt"].strip() or None
            taken.add("fmt")
        if "bits" in changed:
            kind, pos, count = now["bits"]
            if kind == 0:
                entry.bit_pos = entry.bit_count = None
                entry.bit_signed = False
            else:
                entry.bit_pos = pos
                entry.bit_count = 1 if kind == 1 else count
                entry.bit_signed = kind == 3
            taken.add("bits")
        if "refs" in changed:
            refs, bad = {}, []
            for index, part in enumerate(p.strip() for p in now["refs"].split(";") if p.strip()):
                value = self._float(part)
                if value is None and part.upper() not in ("MIN", "MAX", "MEAN"):
                    bad.append(part)
                elif index <= 2:
                    refs[index] = value if value is not None else part.upper()
            problem = (tr("Allowed: numbers, min, max, mean (up to three, separated by ;)") + " - " + ", ".join(bad)
                       if bad else None)
            self._mark(self.ref_edit, problem)
            if not bad:
                entry.refs = refs
                taken.add("refs")
        if "attrs" in changed:
            # parse into a scratch entry first so a typo never destroys the old attributes
            scratch = PltEntry()
            parse_attributes(now["attrs"].replace("\n", " "), scratch)
            problems = list(scratch.extra)
            for op in scratch.ops:
                if op.kind == "exp":
                    try:
                        compile_expr(op.expr)
                    except ExprError as exc:
                        problems.append("%s: %s" % (op.text, exc))
            if problems:
                self.attr_error.setText(tr("Not understood") + ": " + "; ".join(problems))
            else:
                self.attr_error.setText("")
                was_last = entry.view_pos is not None and entry.view_pos >= len(entry.ops)
                entry.ops = scratch.ops
                if entry.view_pos is not None:
                    entry.view_pos = (len(entry.ops) or None) if was_last else min(entry.view_pos, len(entry.ops)) or None
                taken.add("attrs")
        for key in taken:
            shown[key] = now[key]
        self._shown = shown
        if entry.state() != before:
            self._script.modified = True
            item = self.entry_list.item(self.entry_list.currentRow())
            if item is not None:
                item.setText(self._entry_text(entry))
                if "color" in taken and entry.enabled:
                    item.setIcon(self._entry_color_icon(entry))

    def accept(self) -> None:
        self._store_properties()
        if self._color_range != self.settings.color_range():
            self.settings.set("cyclic_color_range", list(self._color_range))
        self.settings.set("editor/copy_range", self.copy_range.isChecked())
        self.settings.set("editor/copy_unit", self.copy_unit.isChecked())
        self.settings.set("editor/width", self.width())
        self.settings.set("editor/height", self.height())
        self.settings.set("editor/filter_mode", self.filter_mode.currentIndex())
        text = self.filter_edit.currentText().strip()
        if text and text != "*":
            history = [text] + [x for x in self.settings.text_list("filter_history") if x != text]
            self.settings.set("filter_history", history[:10])
        super().accept()
