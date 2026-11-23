"""Dialog for the user's value mappings: raw value -> logical value / text.

Upper part: the signal chosen in the legend - which table it uses and how its
values are shown (mapped text, text with the raw number, raw number, state band).
Middle: the tables (import, export, new, copy, rename, delete) and the rows of
the table being edited. Lower part: all assignments signal -> table.

The dialog works on a copy of the library; OK takes the copy over.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..display import DisplaySignal
from ..i18n import tr
from ..valuemap import (MODES, MappingImport, ValueMap, ValueMapLibrary, parse_raw_value, read_mapping_file,
                        write_mapping_csv)

__all__ = ["ValueMapDialog", "signal_pattern"]

_NO_TABLE = "\x00none"
_OWN_TABLE = "\x00own"
_MODE_LABELS = {"mapped": "Mapped value", "both": "Mapped value (raw value)", "raw": "Raw value"}


def signal_pattern(sig: DisplaySignal) -> str:
    """Name under which an assignment for ``sig`` is stored: the display name of a processed
    line, else the recorded name without device suffix."""
    return entry_pattern(sig.entry, sig.info)


def entry_pattern(entry, info) -> str:
    """:func:`signal_pattern` of a view line and its catalogue entry (``None`` when it is not recorded)."""
    if entry.ops or entry.is_placeholder:
        return entry.key or entry.token
    if entry.bit_pos is not None:
        return entry.token
    if info is not None:
        return info.source_name.split("\\")[0] or info.name
    return entry.name


def _g(value: float) -> str:
    return "%.15g" % value


class ValueMapDialog(QtWidgets.QDialog):
    """Edit tables and assignments. ``result_library()`` is the edited copy after ``exec_()``."""

    def __init__(self, library: ValueMapLibrary, signals: Sequence[DisplaySignal], current: int = -1,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Value mapping (raw value -> logical value)"))
        self.resize(1080, 760)
        self.library = copy.deepcopy(library)
        self.library.path = library.path
        self.signals = list(signals)
        self._table: Optional[str] = None             # table shown in the row editor
        self._loading = False
        self._numeric_request: Optional[tuple] = None

        outer = QtWidgets.QVBoxLayout(self)

        # ---- the chosen signal
        box = QtWidgets.QGroupBox(tr("Signal"))
        grid = QtWidgets.QGridLayout(box)
        self.signal_combo = QtWidgets.QComboBox()
        for sig in self.signals:
            self.signal_combo.addItem(sig.label)
        grid.addWidget(QtWidgets.QLabel(tr("Signal")), 0, 0)
        grid.addWidget(self.signal_combo, 0, 1, 1, 3)
        self.pattern_edit = QtWidgets.QLineEdit()
        self.pattern_edit.setToolTip(tr("The rule applies to every signal with this name; * and ? are wildcards"))
        grid.addWidget(QtWidgets.QLabel(tr("Applies to")), 1, 0)
        grid.addWidget(self.pattern_edit, 1, 1, 1, 3)
        self.use_combo = QtWidgets.QComboBox()
        grid.addWidget(QtWidgets.QLabel(tr("Table")), 2, 0)
        grid.addWidget(self.use_combo, 2, 1)
        self.mode_combo = QtWidgets.QComboBox()
        for mode in MODES:
            self.mode_combo.addItem(tr(_MODE_LABELS[mode]), mode)
        grid.addWidget(QtWidgets.QLabel(tr("Show")), 2, 2)
        grid.addWidget(self.mode_combo, 2, 3)
        self.lane_check = QtWidgets.QCheckBox(tr("Show as state band (labelled segments above the curves)"))
        grid.addWidget(self.lane_check, 3, 1, 1, 3)
        self.signal_note = QtWidgets.QLabel()
        self.signal_note.setStyleSheet("color: gray")
        self.signal_note.setWordWrap(True)
        grid.addWidget(self.signal_note, 4, 1, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        outer.addWidget(box)

        # ---- tables and rows
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QtWidgets.QLabel(tr("Tables")))
        self.table_list = QtWidgets.QListWidget()
        lv.addWidget(self.table_list, 1)
        buttons = QtWidgets.QGridLayout()
        self.btn_new = QtWidgets.QPushButton(tr("New"))
        self.btn_from_values = QtWidgets.QPushButton(tr("New from the signal's values"))
        self.btn_from_own = QtWidgets.QPushButton(tr("Copy the recording's text table"))
        self.btn_copy = QtWidgets.QPushButton(tr("Copy"))
        self.btn_rename = QtWidgets.QPushButton(tr("Rename"))
        self.btn_delete = QtWidgets.QPushButton(tr("Delete"))
        self.btn_import = QtWidgets.QPushButton(tr("Import..."))
        self.btn_export = QtWidgets.QPushButton(tr("Export..."))
        self.btn_import.setToolTip(tr("CSV / TXT (signal, value, text or value, text), Excel, DBC (VAL_), JSON"))
        for i, button in enumerate((self.btn_new, self.btn_copy, self.btn_rename, self.btn_delete,
                                    self.btn_import, self.btn_export)):
            buttons.addWidget(button, i // 2, i % 2)
        buttons.addWidget(self.btn_from_values, 3, 0, 1, 2)
        buttons.addWidget(self.btn_from_own, 4, 0, 1, 2)
        lv.addLayout(buttons)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.table_title = QtWidgets.QLabel()
        rv.addWidget(self.table_title)
        form = QtWidgets.QFormLayout()
        self.unit_edit = QtWidgets.QLineEdit()
        self.unit_edit.setToolTip(tr("Unit of the logical values when they are numbers, e.g. km/h"))
        form.addRow(tr("Unit of the logical values"), self.unit_edit)
        self.default_edit = QtWidgets.QLineEdit()
        self.default_edit.setPlaceholderText(tr("(empty = show the number)"))
        form.addRow(tr("Text for all other values"), self.default_edit)
        rv.addLayout(form)
        self.auto_speed = QtWidgets.QCheckBox(tr("Without a picture show a speed limit sign (logical value = km/h)"))
        form.addRow("", self.auto_speed)
        self.rows = QtWidgets.QTableWidget(0, 3)
        self.rows.setHorizontalHeaderLabels([tr("Raw value"), tr("Logical value / text"), tr("Picture")])
        self.rows.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.rows.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.rows.setColumnWidth(2, 200)
        self.rows.setIconSize(QtCore.QSize(22, 22))
        self.rows.verticalHeader().setVisible(False)
        self.rows.setToolTip(tr("Raw value: a number (also 0x1F), a range 10..20, or * for all other values"))
        rv.addWidget(self.rows, 1)
        row_buttons = QtWidgets.QHBoxLayout()
        self.btn_add_row = QtWidgets.QPushButton(tr("Add row"))
        self.btn_del_row = QtWidgets.QPushButton(tr("Delete rows"))
        self.btn_sort = QtWidgets.QPushButton(tr("Sort"))
        self.btn_numeric = QtWidgets.QPushButton(tr("Add the logical values as a numeric signal"))
        self.btn_numeric.setToolTip(tr("Only for tables whose logical values are all numbers: adds a script line "
                                       "with exp=map(x, ...), a curve that can be plotted and compared"))
        self.btn_picture = QtWidgets.QPushButton(tr("Picture file..."))
        self.btn_builtin = QtWidgets.QPushButton(tr("Built-in sign"))
        builtin_menu = QtWidgets.QMenu(self.btn_builtin)
        from ..signs import BUILTIN_EXAMPLES, sign_pixmap
        for spec in BUILTIN_EXAMPLES:
            pix = sign_pixmap(spec, 24)
            act = builtin_menu.addAction(QtGui.QIcon(pix) if pix is not None else QtGui.QIcon(), spec)
            act.triggered.connect(lambda _c=False, sp=spec: self._set_picture(sp))
        self.btn_builtin.setMenu(builtin_menu)
        self.btn_builtin.setToolTip(tr("speed:80 = limit sign, end = end of restriction, cond:60:雨天 = with a panel, "
                                       "warn = warning triangle; the text can also be typed into the cell"))
        for button in (self.btn_add_row, self.btn_del_row, self.btn_sort, self.btn_picture, self.btn_builtin):
            row_buttons.addWidget(button)
        row_buttons.addStretch(1)
        row_buttons.addWidget(self.btn_numeric)
        rv.addLayout(row_buttons)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        outer.addWidget(split, 1)

        # ---- all assignments
        outer.addWidget(QtWidgets.QLabel(tr("Rules (signal -> table)")))
        self.assign_list = QtWidgets.QTreeWidget()
        self.assign_list.setRootIsDecorated(False)
        self.assign_list.setHeaderLabels([tr("Signal"), tr("Table"), tr("Show"), tr("State band")])
        self.assign_list.setMaximumHeight(150)
        outer.addWidget(self.assign_list)
        bottom = QtWidgets.QHBoxLayout()
        self.btn_unassign = QtWidgets.QPushButton(tr("Delete rule"))
        bottom.addWidget(self.btn_unassign)
        bottom.addStretch(1)
        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bottom.addWidget(self.button_box)
        outer.addLayout(bottom)

        # ---- wiring
        self.signal_combo.currentIndexChanged.connect(self._show_signal)
        self.use_combo.currentIndexChanged.connect(self._assignment_changed)
        self.mode_combo.currentIndexChanged.connect(self._assignment_changed)
        self.lane_check.toggled.connect(self._assignment_changed)
        self.pattern_edit.editingFinished.connect(self._pattern_changed)
        self.table_list.currentTextChanged.connect(self._show_table)
        self.rows.cellChanged.connect(self._cell_changed)
        self.unit_edit.editingFinished.connect(self._store_table)
        self.default_edit.editingFinished.connect(self._store_table)
        self.auto_speed.toggled.connect(lambda _c: self._store_table())
        self.btn_picture.clicked.connect(self._choose_picture_file)
        self.btn_new.clicked.connect(self._new_table)
        self.btn_from_values.clicked.connect(self._table_from_values)
        self.btn_from_own.clicked.connect(self._table_from_own)
        self.btn_copy.clicked.connect(self._copy_table)
        self.btn_rename.clicked.connect(self._rename_table)
        self.btn_delete.clicked.connect(self._delete_table)
        self.btn_import.clicked.connect(self._import)
        self.btn_export.clicked.connect(self._export)
        self.btn_add_row.clicked.connect(self._add_row)
        self.btn_del_row.clicked.connect(self._delete_rows)
        self.btn_sort.clicked.connect(self._sort_rows)
        self.btn_numeric.clicked.connect(self._numeric_signal)
        self.btn_unassign.clicked.connect(self._unassign)
        self.assign_list.itemDoubleClicked.connect(self._assignment_clicked)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self._fill_tables()
        self._fill_assignments()
        if self.signals:
            self.signal_combo.setCurrentIndex(current if 0 <= current < len(self.signals) else 0)
            self._show_signal()
        else:
            for w in (self.signal_combo, self.pattern_edit, self.use_combo, self.mode_combo, self.lane_check,
                      self.btn_from_values, self.btn_from_own):
                w.setEnabled(False)
        if self.table_list.count() and self.table_list.currentRow() < 0:
            self.table_list.setCurrentRow(0)
        self._update_buttons()

    # ------------------------------------------------------------------ results
    def result_library(self) -> ValueMapLibrary:
        return self.library

    def numeric_request(self) -> Optional[tuple]:
        """``(signal index, table name)`` when the user asked for a numeric signal, else ``None``."""
        return self._numeric_request

    # ------------------------------------------------------------------ signal part
    def _signal(self) -> Optional[DisplaySignal]:
        i = self.signal_combo.currentIndex()
        return self.signals[i] if 0 <= i < len(self.signals) else None

    def _own_enum(self, sig: DisplaySignal):
        """The text table that came with the recording (not one of the user's tables)."""
        if sig.info is not None and sig.info.enum:
            return sig.info.enum
        if sig.enum and not sig.map_name:
            return sig.enum
        return None

    def _fill_use_combo(self, sig: Optional[DisplaySignal]) -> None:
        self.use_combo.blockSignals(True)
        self.use_combo.clear()
        self.use_combo.addItem(tr("(no mapping)"), _NO_TABLE)
        if sig is not None and self._own_enum(sig):
            self.use_combo.addItem(tr("(text table of the recording)"), _OWN_TABLE)
        for name in self.library.tables:
            self.use_combo.addItem(name, name)
        self.use_combo.blockSignals(False)

    def _show_signal(self, *_args) -> None:
        sig = self._signal()
        if sig is None:
            return
        self._loading = True
        pattern = signal_pattern(sig)
        assignment = self.library.exact(pattern) or self.library.assignment_for(pattern)
        if assignment is not None and assignment.pattern.lower() != pattern.lower():
            pattern = assignment.pattern                       # matched by a wildcard rule
        self.pattern_edit.setText(pattern)
        self._fill_use_combo(sig)
        if assignment is None:
            self.use_combo.setCurrentIndex(0)
            self.mode_combo.setCurrentIndex(self.mode_combo.findData("both"))
            self.lane_check.setChecked(False)
        else:
            data = assignment.table or (_OWN_TABLE if self._own_enum(sig) else _NO_TABLE)
            index = self.use_combo.findData(data)
            self.use_combo.setCurrentIndex(max(0, index))
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(assignment.mode)))
            self.lane_check.setChecked(assignment.lane)
            if assignment.table and assignment.table in self.library.tables:
                items = self.table_list.findItems(assignment.table, QtCore.Qt.MatchExactly)
                if items:
                    self.table_list.setCurrentItem(items[0])
        own = self._own_enum(sig)
        notes = []
        if sig.info is not None:
            notes.append(tr("Recorded name: %s") % sig.info.source_name)
        if own:
            notes.append(tr("The recording has its own text table with %d entries") % (len(own.values) + len(own.ranges)))
        if sig.entry.ops:
            notes.append(tr("This line changes the values (%s): rules for the recorded name do not apply to it")
                         % " ".join(op.text for op in sig.entry.ops))
        self.signal_note.setText("   ".join(notes))
        self._loading = False
        self._update_buttons()

    def _pattern_changed(self) -> None:
        if not self._loading:
            self._assignment_changed()

    def _assignment_changed(self, *_args) -> None:
        if self._loading:
            return
        sig = self._signal()
        pattern = self.pattern_edit.text().strip()
        if sig is None or not pattern:
            return
        data = self.use_combo.currentData()
        mode, lane = self.mode_combo.currentData(), self.lane_check.isChecked()
        if data == _NO_TABLE and mode == "both" and not lane:
            self.library.unassign(pattern)                 # the default: nothing to store
        elif data == _NO_TABLE:
            if self._own_enum(sig) is None and mode != "raw" and not lane:
                self.library.unassign(pattern)
            else:
                self.library.assign(pattern, "", mode, lane)
        else:
            self.library.assign(pattern, "" if data == _OWN_TABLE else data, mode, lane)
        self._fill_assignments()
        if data not in (_NO_TABLE, _OWN_TABLE):
            items = self.table_list.findItems(data, QtCore.Qt.MatchExactly)
            if items:
                self.table_list.setCurrentItem(items[0])
        self._update_buttons()

    # ------------------------------------------------------------------ lists
    def _fill_tables(self, select: Optional[str] = None) -> None:
        current = select or (self.table_list.currentItem().text() if self.table_list.currentItem() else None)
        self.table_list.blockSignals(True)
        self.table_list.clear()
        for name in self.library.tables:
            self.table_list.addItem(name)
        self.table_list.blockSignals(False)
        items = self.table_list.findItems(current, QtCore.Qt.MatchExactly) if current else []
        if items:
            self.table_list.setCurrentItem(items[0])
        elif self.table_list.count():
            self.table_list.setCurrentRow(0)
        else:
            self._show_table("")
        sig = self._signal()
        keep = self.use_combo.currentData()
        self._loading = True
        self._fill_use_combo(sig)
        self.use_combo.setCurrentIndex(max(0, self.use_combo.findData(keep)))
        self._loading = False

    def _fill_assignments(self) -> None:
        self.assign_list.clear()
        for a in self.library.assignments:
            table = a.table or tr("(text table of the recording)")
            item = QtWidgets.QTreeWidgetItem([a.pattern, table, tr(_MODE_LABELS.get(a.mode, a.mode)),
                                              tr("yes") if a.lane else ""])
            item.setData(0, QtCore.Qt.UserRole, a.pattern)
            self.assign_list.addTopLevelItem(item)
        for column in range(4):
            self.assign_list.resizeColumnToContents(column)

    def _assignment_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        pattern = item.data(0, QtCore.Qt.UserRole)
        for i, sig in enumerate(self.signals):
            if signal_pattern(sig).lower() == str(pattern).lower():
                self.signal_combo.setCurrentIndex(i)
                return
        table = item.text(1)
        items = self.table_list.findItems(table, QtCore.Qt.MatchExactly)
        if items:
            self.table_list.setCurrentItem(items[0])

    def _unassign(self) -> None:
        item = self.assign_list.currentItem()
        if item is None:
            return
        self.library.unassign(str(item.data(0, QtCore.Qt.UserRole)))
        self._fill_assignments()
        self._show_signal()

    # ------------------------------------------------------------------ table editor
    def _show_table(self, name: str) -> None:
        self._store_table()
        self._table = name if name in self.library.tables else None
        table = self.library.tables.get(name) if name else None
        self._loading = True
        self.rows.setRowCount(0)
        enabled = table is not None
        for w in (self.rows, self.unit_edit, self.default_edit, self.btn_add_row, self.btn_del_row, self.btn_sort,
                  self.auto_speed, self.btn_picture, self.btn_builtin):
            w.setEnabled(enabled)
        if table is None:
            self.table_title.setText(tr("(no table)"))
            self.unit_edit.clear()
            self.default_edit.clear()
        else:
            self.table_title.setText("<b>%s</b>" % table.name.replace("<", "&lt;"))
            self.unit_edit.setText(table.unit)
            self.default_edit.setText(table.default or "")
            self.auto_speed.setChecked(table.auto_image == "speed")
            for lo, hi, text in table.rows:
                key = ValueMap.key(lo, hi)
                self._append_row(key, text, table.images.get(key, ""))
            if table.images.get("*"):
                self._append_row("*", "", table.images["*"])
        self._loading = False
        self._update_buttons()

    def _append_row(self, raw: str, text: str, picture: str = "") -> None:
        row = self.rows.rowCount()
        self.rows.insertRow(row)
        self.rows.setItem(row, 0, QtWidgets.QTableWidgetItem(raw))
        self.rows.setItem(row, 1, QtWidgets.QTableWidgetItem(text))
        self.rows.setItem(row, 2, QtWidgets.QTableWidgetItem(picture))
        self._picture_icon(row)

    def _picture_icon(self, row: int) -> None:
        from ..signs import sign_pixmap
        item = self.rows.item(row, 2)
        if item is None:
            return
        spec = item.text().strip()
        pix = sign_pixmap(spec, 22) if spec else None
        self._loading, loading = True, self._loading
        item.setIcon(QtGui.QIcon(pix) if pix is not None else QtGui.QIcon())
        item.setToolTip("" if not spec or pix is not None else tr("picture not found"))
        self._loading = loading

    def _set_picture(self, spec: str) -> None:
        row = self.rows.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, self.windowTitle(), tr("Choose a row of the table first"))
            return
        self.rows.item(row, 2).setText(spec)

    def _choose_picture_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, tr("Picture file..."), "",
                                                        tr("Pictures") + " (*.png *.jpg *.jpeg *.bmp *.svg *.gif)")
        if path:
            self._set_picture(path)

    def _cell_changed(self, row: int, column: int) -> None:
        if self._loading:
            return
        if column == 0:
            item = self.rows.item(row, 0)
            self._check_raw(item)
        if column == 2:
            self._picture_icon(row)
        self._store_table()

    def _check_raw(self, item: Optional[QtWidgets.QTableWidgetItem]) -> bool:
        if item is None:
            return True
        text = item.text().strip()
        ok = True
        if text:
            try:
                parse_raw_value(text)
            except ValueError as exc:
                ok = False
                item.setToolTip(str(exc))
        self._loading, loading = True, self._loading
        item.setBackground(QtGui.QBrush() if ok else QtGui.QColor(255, 200, 200))
        item.setForeground(QtGui.QBrush() if ok else QtGui.QColor(30, 30, 30))
        if ok:
            item.setToolTip("")
        self._loading = loading
        return ok

    def _store_table(self) -> None:
        """Take the editor content into the working copy (rows with an invalid raw value are kept out)."""
        if self._loading or self._table is None or self._table not in self.library.tables:
            return
        table = self.library.tables[self._table]
        rows, default, images = [], None, {}
        for r in range(self.rows.rowCount()):
            raw_item, text_item, pic_item = self.rows.item(r, 0), self.rows.item(r, 1), self.rows.item(r, 2)
            raw = raw_item.text().strip() if raw_item else ""
            text = text_item.text().strip() if text_item else ""
            picture = pic_item.text().strip() if pic_item else ""
            if not raw:
                continue
            try:
                span = parse_raw_value(raw)
            except ValueError:
                continue
            if span is None:
                default = text or None
                if picture:
                    images["*"] = picture
            else:
                rows.append((span[0], span[1], text))
                if picture:
                    images[ValueMap.key(span[0], span[1])] = picture
        table.rows = rows
        table.images = images
        table.auto_image = "speed" if self.auto_speed.isChecked() else ""
        table.unit = self.unit_edit.text().strip()
        default_text = self.default_edit.text().strip()
        table.default = default_text or default
        self._update_buttons()

    def _add_row(self) -> None:
        self._loading = True
        self._append_row("", "", "")
        self._loading = False
        self.rows.setCurrentCell(self.rows.rowCount() - 1, 0)
        self.rows.editItem(self.rows.item(self.rows.rowCount() - 1, 0))

    def _delete_rows(self) -> None:
        rows = sorted({i.row() for i in self.rows.selectedIndexes()}, reverse=True)
        self._loading = True
        for r in rows:
            self.rows.removeRow(r)
        self._loading = False
        self._store_table()

    def _sort_rows(self) -> None:
        self._store_table()
        if self._table in self.library.tables:
            table = self.library.tables[self._table]
            table.rows.sort(key=lambda r: (r[0], r[1]))
            self._show_table(self._table)

    # ------------------------------------------------------------------ table commands
    def _ask_name(self, title: str, default: str) -> Optional[str]:
        name, ok = QtWidgets.QInputDialog.getText(self, title, tr("Name of the table"), text=default)
        name = name.strip()
        return name if ok and name else None

    def _add_table(self, table: ValueMap) -> None:
        table.name = self.library.unique_name(table.name)
        self.library.tables[table.name] = table
        self._fill_tables(select=table.name)

    def _new_table(self) -> None:
        sig = self._signal()
        name = self._ask_name(tr("New table"), signal_pattern(sig) if sig is not None else "table")
        if name:
            self._add_table(ValueMap(name))
            self._add_row()

    def _table_from_values(self) -> None:
        """A table with one row per value the signal takes (up to 512); the texts are filled from a
        text table of the recording when there is one."""
        sig = self._signal()
        if sig is None or not sig.has_data:
            return
        y = np.asarray(sig.y, dtype=np.float64)
        values = np.unique(y[np.isfinite(y)])
        if len(values) > 512:
            QtWidgets.QMessageBox.information(self, self.windowTitle(),
                                              tr("The signal takes %d different values; the first 512 are used") % len(values))
            values = values[:512]
        own = self._own_enum(sig)
        rows = [(float(v), float(v), (own.text(v) if own else None) or "") for v in values]
        name = self._ask_name(tr("New table"), signal_pattern(sig))
        if name:
            self._add_table(ValueMap(name, rows))
            self._assign_current(name)

    def _table_from_own(self) -> None:
        sig = self._signal()
        own = self._own_enum(sig) if sig is not None else None
        if own is None:
            QtWidgets.QMessageBox.information(self, self.windowTitle(), tr("This signal has no text table in the recording"))
            return
        rows = [(v, v, t) for v, t in sorted(own.values.items())] + [tuple(r) for r in own.ranges]
        name = self._ask_name(tr("New table"), signal_pattern(sig))
        if name:
            self._add_table(ValueMap(name, rows, own.default))
            self._assign_current(name)

    def _assign_current(self, table: str) -> None:
        index = self.use_combo.findData(table)
        if index >= 0:
            self.use_combo.setCurrentIndex(index)
            if self.mode_combo.currentData() == "raw":
                self.mode_combo.setCurrentIndex(self.mode_combo.findData("mapped"))
            self._assignment_changed()

    def _copy_table(self) -> None:
        if self._table in self.library.tables:
            self._store_table()
            table = copy.deepcopy(self.library.tables[self._table])
            table.name = table.name + " (2)"
            self._add_table(table)

    def _rename_table(self) -> None:
        old = self._table
        if old not in self.library.tables:
            return
        self._store_table()
        new = self._ask_name(tr("Rename table"), old)
        if not new or new == old:
            return
        if not self.library.rename_table(old, new):
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), tr("A table with this name exists already"))
            return
        self._table = new
        self._fill_tables(select=new)
        self._fill_assignments()
        self._show_signal()

    def _delete_table(self) -> None:
        name = self._table
        if name not in self.library.tables:
            return
        users = [a.pattern for a in self.library.assignments if a.table == name]
        text = tr("Delete the table '%s'?") % name
        if users:
            text += "\n" + tr("It is used by: %s") % ", ".join(users[:8])
        if QtWidgets.QMessageBox.question(self, self.windowTitle(), text) != QtWidgets.QMessageBox.Yes:
            return
        self._table = None
        self.library.remove_table(name)
        self._fill_tables()
        self._fill_assignments()
        self._show_signal()

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("Import value mappings"), "",
            tr("Mapping files") + " (*.csv *.txt *.xlsx *.dbc *.json);;" + tr("All files") + " (*)")
        if not path:
            return
        self._store_table()
        try:
            imported: MappingImport = read_mapping_file(path)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        replaced = [t.name for t in imported.tables if t.name in self.library.tables]
        tables, rules = self.library.merge(imported)
        message = tr("%d tables and %d rules imported from %s") % (tables, rules, Path(path).name)
        if replaced:
            message += "\n" + tr("Replaced tables with the same name: %s") % ", ".join(replaced[:10])
        if imported.problems:
            message += "\n\n" + tr("Skipped:") + "\n" + "\n".join(imported.problems[:15])
            if len(imported.problems) > 15:
                message += "\n..."
        self._fill_tables(select=imported.tables[0].name if imported.tables else None)
        self._fill_assignments()
        self._show_signal()
        QtWidgets.QMessageBox.information(self, self.windowTitle(), message)

    def _export(self) -> None:
        if not self.library.tables:
            return
        self._store_table()
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("Export value mappings"), "valuemaps.csv",
            "CSV (*.csv);;" + tr("TracePaddock value mappings") + " (*.json)")
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                self.library.save(path)
            else:
                write_mapping_csv(path, list(self.library.tables.values()), self.library.assignments)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), str(exc))

    def _numeric_signal(self) -> None:
        self._store_table()
        sig_index = self.signal_combo.currentIndex()
        if self._table in self.library.tables and self.library.tables[self._table].is_numeric:
            self._numeric_request = (sig_index, self._table)
            self.accept()

    # ------------------------------------------------------------------ state
    def _update_buttons(self) -> None:
        has = self._table in self.library.tables
        for w in (self.btn_copy, self.btn_rename, self.btn_delete):
            w.setEnabled(has)
        self.btn_export.setEnabled(bool(self.library.tables))
        sig = self._signal()
        self.btn_from_values.setEnabled(sig is not None and sig.has_data)
        self.btn_from_own.setEnabled(sig is not None and self._own_enum(sig) is not None)
        self.btn_numeric.setEnabled(has and sig is not None and self.library.tables[self._table].is_numeric)
        self.btn_unassign.setEnabled(bool(self.library.assignments))

    def accept(self) -> None:
        self._store_table()
        bad = []
        for r in range(self.rows.rowCount()):
            if not self._check_raw(self.rows.item(r, 0)):
                bad.append(self.rows.item(r, 0).text())
        if bad:
            answer = QtWidgets.QMessageBox.question(
                self, self.windowTitle(),
                tr("These raw values are not understood and are left out: %s\nContinue?") % ", ".join(bad[:8]))
            if answer != QtWidgets.QMessageBox.Yes:
                return
        super().accept()
