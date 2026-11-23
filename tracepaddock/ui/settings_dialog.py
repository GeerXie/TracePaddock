"""Settings dialog: general options, short names for new signals, processing and statistics, printing."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6 import QtWidgets

from ..i18n import tr
from ..plt.parser import short_name

__all__ = ["SettingsDialog", "PROCESSING_KEYS"]

#: Settings that change how the display signals are computed (a change needs a rebuild).
PROCESSING_KEYS = ("ops_on_common_time_base", "calculus_on_nonequidistant", "load_disabled_signals",
                   "ignore_incomplete_computed")

_CHECKS: Tuple[Tuple[str, str, str, bool], ...] = (
    # (tab, key, text, default)
    ("processing", "ops_on_common_time_base",
     "Compute DIFF, INTEG, MAVR, DGF, SHIFT ... on the common time axis of the recording", False),
    ("processing", "calculus_on_nonequidistant",
     "Allow INTEG, DIFF and MAVR on signals without a constant sample time", True),
    ("processing", "statistics_on_common_time_base",
     "Statistics (key S, signal info) over the common time axis", True),
    ("processing", "load_disabled_signals", "Also read disabled (~) signals so that formulas can use them", False),
    ("processing", "ignore_incomplete_computed", "Leave out computed signals whose inputs are missing (no error row)", False),
    ("general", "use_current_signal_selection", "Keep the current view when another recording is opened", True),
    ("general", "print_white_background", "White background for printing, PDF and picture export", True),
    ("names", "short_name_bus_prefix", "Remove .<s>. elements of bus signals: #10.DCW.NAME -> #NAME", False),
    ("names", "short_name_device", "Remove what follows a backslash: name\\CCP:1 -> name", True),
)


class SettingsDialog(QtWidgets.QDialog):
    """Edits a :class:`~tracepaddock.settings.Settings` object; ``changed_keys()`` after ``exec_()``."""

    def __init__(self, settings: object, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Settings"))
        self.settings = settings
        self._before: Dict[str, object] = {}
        self.checks: Dict[str, QtWidgets.QCheckBox] = {}
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)
        pages = {}
        for key, title in (("general", "General"), ("names", "Names of new signals"),
                           ("processing", "Processing and statistics")):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            pages[key] = form
            tabs.addTab(page, tr(title))

        g = pages["general"]
        self.recent_spin = QtWidgets.QSpinBox()
        self.recent_spin.setRange(4, 100)
        self.recent_spin.setValue(int(settings.number("recent_max", 32, 4, 100)))
        g.addRow(tr("Length of the recent file lists"), self.recent_spin)
        self.history_spin = QtWidgets.QSpinBox()
        self.history_spin.setRange(4, 100)
        self.history_spin.setValue(int(settings.number("history_max", 20, 4, 100)))
        g.addRow(tr("Length of the script history"), self.history_spin)

        for tab, key, text, default in _CHECKS:
            box = QtWidgets.QCheckBox(tr(text))
            box.setChecked(bool(settings.get(key, default)))
            box.setToolTip(tr("Default: on") if default else tr("Default: off"))
            self._before[key] = bool(settings.get(key, default))
            pages[tab].addRow("", box)
            self.checks[key] = box

        n = pages["names"]
        row = QtWidgets.QHBoxLayout()
        self.keep_spin = QtWidgets.QSpinBox()
        self.keep_spin.setRange(0, 10)
        self.keep_spin.setSpecialValueText(tr("off"))
        self.keep_spin.setValue(int(settings.number("short_name_keep", 0, 0, 20)))
        self.keep_side = QtWidgets.QComboBox()
        self.keep_side.addItem(tr("from the left"), True)
        self.keep_side.addItem(tr("from the right"), False)
        self.keep_side.setCurrentIndex(0 if settings.get("short_name_keep_left", True) else 1)
        row.addWidget(self.keep_spin)
        row.addWidget(self.keep_side)
        row.addStretch(1)
        n.addRow(tr("Keep at most this many dot separated elements"), row)
        self.example = QtWidgets.QLineEdit(r"DemoRig.Thermal.CoolingFan.Speed\DemoDevice:1")
        self.result = QtWidgets.QLineEdit()
        self.result.setReadOnly(True)
        n.addRow(tr("Example"), self.example)
        n.addRow(tr("Proposed display name"), self.result)
        n.addRow(QtWidgets.QLabel(tr("Used when signals are added in the editor; the recorded name stays in the script.")))

        p = pages["processing"]
        note = QtWidgets.QLabel(tr(
            "By default, sample based operations run on each signal's own samples. With the first option they "
            "run on the common time axis of the recording (the union of the time stamps of all signals); with "
            "several sample rates a derivative there is mostly 0 with large spikes."))
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        p.addRow(note)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                                             | QtWidgets.QDialogButtonBox.RestoreDefaults)
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).setText(tr("Restore defaults"))
        buttons.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for w in (self.keep_spin,):
            w.valueChanged.connect(self._preview)
        self.keep_side.currentIndexChanged.connect(self._preview)
        self.example.textChanged.connect(self._preview)
        self.checks["short_name_bus_prefix"].toggled.connect(self._preview)
        self.checks["short_name_device"].toggled.connect(self._preview)
        self._changed: Tuple[str, ...] = ()
        self._preview()

    def restore_defaults(self) -> None:
        """Every option of the dialog back to its default (applied with OK)."""
        for _tab, key, _text, default in _CHECKS:
            self.checks[key].setChecked(default)
        self.recent_spin.setValue(32)
        self.history_spin.setValue(20)
        self.keep_spin.setValue(0)
        self.keep_side.setCurrentIndex(0)

    def _preview(self, *_args) -> None:
        self.result.setText(short_name(self.example.text(),
                                       drop_bus_prefix=self.checks["short_name_bus_prefix"].isChecked(),
                                       drop_device=self.checks["short_name_device"].isChecked(),
                                       keep_elements=self.keep_spin.value(),
                                       keep_from_left=bool(self.keep_side.currentData())))

    def accept(self) -> None:
        s = self.settings
        changed = []
        for key, box in self.checks.items():
            if box.isChecked() != self._before[key]:
                changed.append(key)
            s.set(key, box.isChecked())
        s.set("recent_max", self.recent_spin.value())
        s.set("history_max", self.history_spin.value())
        s.set("short_name_keep", self.keep_spin.value())
        s.set("short_name_keep_left", bool(self.keep_side.currentData()))
        self._changed = tuple(changed)
        super().accept()

    def changed_keys(self) -> Tuple[str, ...]:
        return self._changed
