"""Dialog of the signal databases (DBC / CAN matrix): files, lookup, value tables -> value mappings."""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6 import QtCore, QtWidgets

from ..candb import CanDbError, SignalDatabase
from ..i18n import tr

__all__ = ["SignalDatabaseDialog"]


class SignalDatabaseDialog(QtWidgets.QDialog):
    """``files()`` after ``exec_()`` is the new file list; ``import_values`` takes over value tables."""

    def __init__(self, files: List[str], start_dir: str, import_values: Callable[[SignalDatabase], None],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Signal databases (DBC / CAN matrix)"))
        self.resize(1250, 720)
        self.start_dir = start_dir
        self.import_values = import_values
        self.db = SignalDatabase()
        self._files: List[str] = []
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(tr("From a DBC, ARXML or an Excel / CSV CAN matrix the viewer takes the resolution "
                                   "(suggested tolerance), unit, range, cycle time and send type of each signal, and "
                                   "its value table. Signals of the recording are found by name."))
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(110)
        row.addWidget(self.list, 1)
        side = QtWidgets.QVBoxLayout()
        add = QtWidgets.QPushButton(tr("Add file..."))
        remove = QtWidgets.QPushButton(tr("Remove"))
        side.addWidget(add)
        side.addWidget(remove)
        side.addStretch(1)
        row.addLayout(side)
        layout.addLayout(row)
        self.problems = QtWidgets.QLabel()
        self.problems.setStyleSheet("color: #c00000")
        self.problems.setWordWrap(True)
        layout.addWidget(self.problems)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText(tr("search signal, message or description"))
        layout.addWidget(self.search)
        self.table = QtWidgets.QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([tr("Signal"), tr("Message"), "ID", tr("Cycle [ms]"), tr("Send type"),
                                              tr("Resolution"), tr("Offset"), tr("Min"), tr("Max"), tr("Unit"),
                                              tr("Value texts")])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(10, 320)
        layout.addWidget(self.table, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.take = QtWidgets.QPushButton(tr("Take the value tables over as value mappings"))
        bottom.addWidget(self.take)
        bottom.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bottom.addWidget(buttons)
        layout.addLayout(bottom)
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        self.search.textChanged.connect(self._fill)
        self.take.clicked.connect(lambda: self.import_values(self.db))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        for path in files:
            self._load(path)
        self._fill()

    def files(self) -> List[str]:
        return list(self._files)

    def database(self) -> SignalDatabase:
        return self.db

    def _load(self, path: str) -> None:
        try:
            count = self.db.load(path)
        except CanDbError as exc:
            self.problems.setText(str(exc))
            return
        self._files.append(path)
        self.list.addItem("%s  (%d %s)" % (path, count, tr("signals")))

    def _add(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, tr("Add file..."), self.start_dir,
            tr("Signal databases") + " (*.dbc *.arxml *.kcd *.sym *.xlsx *.xlsm *.csv);;" + tr("All files") + " (*)")
        for path in paths:
            if path not in self._files:
                self._load(path)
        self._fill()

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        del self._files[row]
        self.list.takeItem(row)
        files, self.db = list(self._files), SignalDatabase()
        self._files = []
        self.list.clear()
        for path in files:
            self._load(path)
        self._fill()

    def _fill(self, *_args) -> None:
        specs = self.db.search(self.search.text(), limit=800)
        self.table.setRowCount(len(specs))
        for i, s in enumerate(specs):
            values = "; ".join("%g=%s" % (k, v) for k, v in sorted(s.values.items())[:8])
            if len(s.values) > 8:
                values += " ..."
            cells = [s.name, s.message, "" if s.message_id is None else "0x%X" % s.message_id,
                     "" if s.cycle_ms is None else "%g" % s.cycle_ms, s.send_type or s.signal_send_type,
                     "" if s.factor is None else "%g" % s.factor, "" if s.offset is None else "%g" % s.offset,
                     "" if s.minimum is None else "%g" % s.minimum, "" if s.maximum is None else "%g" % s.maximum,
                     s.unit, values]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 0:
                    item.setToolTip(s.description or s.source)
                self.table.setItem(i, c, item)
        self.take.setEnabled(len(self.db) > 0)
