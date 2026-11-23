"""Smaller dialogs: appearance, computed / reference signals, reference lines, information."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.model import Dataset
from ..display import DisplaySignal
from ..i18n import tr
from ..plt.expr import FUNCTIONS, ExprError, compile_expr
from ..plt.model import PltEntry
from ..plt.parser import format_entry, parse_attributes
from .editor import color_combo
from .render import BACKGROUNDS

__all__ = ["AppearanceDialog", "ComputedSignalDialog", "ReferenceLineDialog", "InfoDialog", "TEMPLATES",
           "TimeRangeDialog", "FavouritesDialog", "NumberBox"]


class NumberBox(QtWidgets.QDoubleSpinBox):
    """Number field that shows ``2`` instead of ``2,000000``.

    Values use a dot as decimal separator, as in PLT scripts and formulas.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setLocale(QtCore.QLocale.c())
        self.setRange(-1e12, 1e12)
        self.setDecimals(9)
        self.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)

    def textFromValue(self, value: float) -> str:
        return "%.10g" % value

    def valueFromText(self, text: str) -> float:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        candidate = text.replace(",", ".").strip()
        if candidate in ("", "-", "+", ".", "-.", "+.") or candidate.lower().endswith(("e", "e-", "e+")):
            return QtGui.QValidator.Intermediate, text, pos
        try:
            float(candidate)
        except ValueError:
            return QtGui.QValidator.Invalid, text, pos
        return QtGui.QValidator.Acceptable, text, pos


def _num(value: float) -> str:
    """A number for use inside a formula (negative values in parentheses)."""
    text = "%.10g" % value
    return "(%s)" % text if value < 0 else text


# ---------------------------------------------------------------------- appearance
class AppearanceDialog(QtWidgets.QDialog):
    """Line width, background colour, grid, drawing style and legend font.

    Changes are applied immediately through ``on_change`` so the effect is
    visible behind the dialog; Cancel restores the previous values.
    """

    KEYS = ("line_width", "background", "grid", "step_mode_analog", "time_labels",
            "legend_font_size", "legend_value_format", "stopped_signals")

    def __init__(self, settings: object, on_change: Callable[[], None],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Appearance"))
        self.settings = settings
        self.on_change = on_change
        self._before = {k: settings.get(k) for k in self.KEYS}
        form = QtWidgets.QFormLayout(self)

        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(1.0, 8.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(float(settings.get("line_width", 1)))
        form.addRow(tr("Line width (pixels)"), self.width_spin)

        row = QtWidgets.QHBoxLayout()
        self.bg_combo = QtWidgets.QComboBox()
        for key, (label, rgb) in BACKGROUNDS.items():
            pix = QtGui.QPixmap(28, 14)
            pix.fill(QtGui.QColor(*rgb))
            self.bg_combo.addItem(QtGui.QIcon(pix), tr(label), "#%02x%02x%02x" % rgb)
        self.bg_combo.addItem(tr("Custom..."), "custom")
        row.addWidget(self.bg_combo, 1)
        self.bg_button = QtWidgets.QPushButton(tr("Choose colour..."))
        row.addWidget(self.bg_button)
        form.addRow(tr("Background"), row)
        current = str(settings.get("background", "#000000")).lower()
        index = self.bg_combo.findData(current)
        self._custom = current
        self.bg_combo.setCurrentIndex(index if index >= 0 else self.bg_combo.count() - 1)

        self.grid_check = QtWidgets.QCheckBox(tr("Show grid (10 x 10 divisions)"))
        self.grid_check.setChecked(bool(settings.get("grid", True)))
        form.addRow("", self.grid_check)
        self.step_check = QtWidgets.QCheckBox(tr("Hold each sample until the next one (steps, default)"))
        self.step_check.setChecked(bool(settings.get("step_mode_analog", True)))
        self.step_check.setToolTip(tr("Off: connect the samples with straight lines"))
        form.addRow("", self.step_check)
        self.stopped_combo = QtWidgets.QComboBox()
        for label, key in (("dashed line (default)", "dashed"), ("hold the last value", "hold"),
                           ("do not draw", "cut")):
            self.stopped_combo.addItem(tr(label), key)
        index = self.stopped_combo.findData(str(settings.get("stopped_signals", "dashed")))
        self.stopped_combo.setCurrentIndex(max(0, index))
        self.stopped_combo.setToolTip(tr("A signal sent at a steady rate whose samples end before the recording "
                                         "ends was no longer sent: from 5 sampling periods after its last sample "
                                         "it has no new value. Signals sent only on change keep their last value."))
        form.addRow(tr("After a signal stops"), self.stopped_combo)
        self.time_check = QtWidgets.QCheckBox(tr("Show the time at the grid lines"))
        self.time_check.setChecked(bool(settings.get("time_labels", True)))
        form.addRow("", self.time_check)
        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(8, 24)
        self.font_spin.setValue(int(settings.get("legend_font_size", 13)))
        form.addRow(tr("Legend font size (max.)"), self.font_spin)
        self.format_edit = QtWidgets.QLineEdit(str(settings.get("legend_value_format", "%g")))
        self.format_edit.setToolTip(tr("printf format for values in the legend, e.g. %g  %.3f  %.6g"))
        form.addRow(tr("Value format"), self.format_edit)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.width_spin.valueChanged.connect(self._apply)
        self.bg_combo.currentIndexChanged.connect(self._apply)
        self.bg_button.clicked.connect(self._choose_colour)
        for check in (self.grid_check, self.step_check, self.time_check):
            check.toggled.connect(self._apply)
        self.font_spin.valueChanged.connect(self._apply)
        self.format_edit.editingFinished.connect(self._apply)
        self.stopped_combo.currentIndexChanged.connect(self._apply)

    def _choose_colour(self) -> None:
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._custom), self, tr("Background colour"))
        if colour.isValid():
            self._custom = colour.name()
            self.bg_combo.setCurrentIndex(self.bg_combo.count() - 1)
            self._apply()

    def _apply(self, *_args) -> None:
        s = self.settings
        s.set("line_width", self.width_spin.value())
        data = self.bg_combo.currentData()
        s.set("background", self._custom if data == "custom" else data)
        s.set("grid", self.grid_check.isChecked())
        s.set("step_mode_analog", self.step_check.isChecked())
        s.set("stopped_signals", self.stopped_combo.currentData())
        s.set("time_labels", self.time_check.isChecked())
        s.set("legend_font_size", self.font_spin.value())
        fmt = self.format_edit.text().strip() or "%g"
        try:
            fmt % 1.0
        except (TypeError, ValueError):
            fmt = "%g"
        s.set("legend_value_format", fmt)
        self.on_change()

    def reject(self) -> None:
        for key, value in self._before.items():
            if value is None:
                self.settings.reset(key)
            else:
                self.settings.set(key, value)
        self.on_change()
        super().reject()


# ---------------------------------------------------------------------- computed signals
#: key -> (label, needs source signal, needs second signal, [(parameter label, default)], builder)
#: The builder returns (attribute text, name suffix, unit rule) where the unit
#: rule is "" (keep), "/s", "*s" or "none".
TEMPLATES: Dict[str, Tuple[str, bool, bool, List[Tuple[str, float]], Callable]] = {
    "const": ("Constant value (reference signal)", False, False, [("Value", 0.0)],
              lambda p, o: ("exp=%s" % _num(p[0]), "CONST", "none")),
    "ramp": ("Ramp  a*t + b", False, False, [("Slope a", 1.0), ("Offset b", 0.0)],
             lambda p, o: ("exp=%s*t+%s" % (_num(p[0]), _num(p[1])), "RAMP", "none")),
    "sine": ("Sine wave", False, False, [("Amplitude", 1.0), ("Frequency [Hz]", 1.0), ("Phase [rad]", 0.0), ("Offset", 0.0)],
             lambda p, o: ("exp=%s*sin(2*PI*%s*t+%s)+%s" % tuple(_num(v) for v in p), "SINE", "none")),
    "step": ("Step at time T", False, False, [("Time T [s]", 1.0), ("Value before", 0.0), ("Value after", 1.0)],
             lambda p, o: ("exp=switch(ge(t,%s),%s,%s)" % (_num(p[0]), _num(p[2]), _num(p[1])), "STEP", "none")),
    "pulse": ("Square wave", False, False, [("Period [s]", 1.0), ("Duty (0..1)", 0.5), ("Low value", 0.0), ("High value", 1.0)],
              lambda p, o: ("exp=switch(lt(mod(t,%s),%s),%s,%s)" % (
                  _num(p[0]), _num(p[0] * p[1]), _num(p[3]), _num(p[2])), "PULSE", "none")),
    "gain": ("Scale and offset  a*x + b", True, False, [("Factor a", 1.0), ("Offset b", 0.0)],
             lambda p, o: ("exp=%s*x+%s" % (_num(p[0]), _num(p[1])), "scaled", "")),
    "lowpass": ("Low-pass filter (Butterworth)", True, False, [("Cut-off frequency [Hz]", 5.0), ("Order", 2.0)],
                lambda p, o: ("dgf=bth:%g:%d" % (p[0], max(1, int(p[1]))), "filt", "")),
    "mavr": ("Moving average", True, False, [("Number of samples", 10.0)],
             lambda p, o: ("mavr(-%d)" % max(2, int(p[0])), "avg", "")),
    "diff": ("Derivative  d/dt", True, False, [], lambda p, o: ("diff", "diff", "/s")),
    "integ": ("Integral over time", True, False, [], lambda p, o: ("integ", "integ", "*s")),
    "nomean": ("Remove mean value", True, False, [], lambda p, o: ("-mean", "zero_mean", "")),
    "abs": ("Absolute value", True, False, [], lambda p, o: ("exp=abs(x)", "abs", "")),
    "limit": ("Limit to a range", True, False, [("Lower limit", -1.0), ("Upper limit", 1.0)],
              lambda p, o: ("exp=max(%s,min(%s,x))" % (_num(p[0]), _num(p[1])), "lim", "")),
    "shift": ("Shift by samples", True, False, [("Samples (+ = later)", 1.0)],
              lambda p, o: ("shift(%d)" % int(p[0]), "shift", "")),
    "minus": ("Difference to another signal  x - x1", True, True, [],
              lambda p, o: ("exp(x1=%s)=x-x1" % o, "diff_to", "")),
    "plus": ("Sum with another signal  x + x1", True, True, [],
             lambda p, o: ("exp(x1=%s)=x+x1" % o, "sum", "")),
    "ratio": ("Ratio to another signal  x / x1", True, True, [],
              lambda p, o: ("exp(x1=%s)=x/x1" % o, "ratio", "none")),
    "custom": ("Custom formula", False, False, [], None),
}


class ComputedSignalDialog(QtWidgets.QDialog):
    """Create a reference / standard signal, or a processed copy of a signal.

    The result is a PLT entry that can be saved with the view.
    """

    def __init__(self, signals: Sequence[DisplaySignal], source_index: int = -1,
                 parent: Optional[QtWidgets.QWidget] = None, *, color: int = 2) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Add computed signal"))
        self.signals = list(signals)
        self._entry: Optional[PltEntry] = None
        form = QtWidgets.QFormLayout(self)

        self.source = QtWidgets.QComboBox()
        self.source.addItem(tr("(none - new standard signal)"), -1)
        for i, sig in enumerate(self.signals):
            self.source.addItem(sig.name, i)
        self.source.setCurrentIndex(source_index + 1 if 0 <= source_index < len(self.signals) else 0)
        form.addRow(tr("Based on signal"), self.source)

        self.kind = QtWidgets.QComboBox()
        form.addRow(tr("Function"), self.kind)
        self.other = QtWidgets.QComboBox()
        for sig in self.signals:
            self.other.addItem(sig.name)
        self.other_label = QtWidgets.QLabel(tr("Other signal (x1)"))
        form.addRow(self.other_label, self.other)

        self.param_labels: List[QtWidgets.QLabel] = []
        self.param_edits: List[QtWidgets.QDoubleSpinBox] = []
        for _ in range(4):
            label, spin = QtWidgets.QLabel(), NumberBox()
            form.addRow(label, spin)
            self.param_labels.append(label)
            self.param_edits.append(spin)
            spin.valueChanged.connect(self._update)
        self.formula = QtWidgets.QLineEdit()
        self.formula.setPlaceholderText("x*3.6     max(0,x)     sin(2*PI*t)")
        self.formula_label = QtWidgets.QLabel(tr("Formula of x, t"))
        form.addRow(self.formula_label, self.formula)
        self.functions = QtWidgets.QLabel(tr("Functions: ") + ", ".join(sorted(FUNCTIONS)))
        self.functions.setWordWrap(True)
        self.functions.setStyleSheet("color: gray")
        form.addRow("", self.functions)

        self.name_edit = QtWidgets.QLineEdit()
        form.addRow(tr("Display name"), self.name_edit)
        self.unit_edit = QtWidgets.QLineEdit()
        form.addRow(tr("Unit"), self.unit_edit)
        self.color_box = color_combo(self)
        self.color_box.setCurrentIndex(max(0, self.color_box.findData(color)))
        form.addRow(tr("Colour"), self.color_box)
        self.same_range = QtWidgets.QCheckBox(tr("Same display range as the source signal (curves are comparable)"))
        self.same_range.setChecked(True)
        form.addRow("", self.same_range)
        raster_row = QtWidgets.QHBoxLayout()
        self.raster = QtWidgets.QComboBox()
        self.raster.addItem(tr("time stamps of all input signals (default)"), "union")
        self.raster.addItem(tr("fixed step [ms]"), "fixed")
        self.raster.addItem(tr("time stamps of the source signal (x)"), "signal:x")
        self.raster.addItem(tr("time stamps of the other signal (x1)"), "signal:x1")
        self.raster.setToolTip(tr("At which times the result has values. Saved as a comment // @raster=..., "
                                  "which other tools reading the script ignore."))
        self.raster_step = NumberBox()
        self.raster_step.setRange(0.001, 1e7)
        self.raster_step.setValue(10.0)
        raster_row.addWidget(self.raster, 1)
        raster_row.addWidget(self.raster_step)
        form.addRow(tr("Calculation raster"), raster_row)
        self.preview = QtWidgets.QLabel()
        self.preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("font-family: Consolas, monospace")
        form.addRow(tr("Line in the view file"), self.preview)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

        self._name_touched = self._unit_touched = False
        self.name_edit.textEdited.connect(lambda _t: setattr(self, "_name_touched", True))
        self.unit_edit.textEdited.connect(lambda _t: setattr(self, "_unit_touched", True))
        self.source.currentIndexChanged.connect(self._fill_kinds)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.other.currentIndexChanged.connect(self._update)
        self.formula.textChanged.connect(self._update)
        self.name_edit.textChanged.connect(self._update)
        self.unit_edit.textChanged.connect(self._update)
        self.color_box.currentIndexChanged.connect(self._update)
        self.same_range.toggled.connect(self._update)
        self.raster.currentIndexChanged.connect(self._update)
        self.raster_step.valueChanged.connect(self._update)
        self._fill_kinds()

    def _source(self) -> Optional[DisplaySignal]:
        index = self.source.currentData()
        return self.signals[index] if index is not None and index >= 0 else None

    def _fill_kinds(self) -> None:
        based = self._source() is not None
        self.kind.blockSignals(True)
        self.kind.clear()
        for key, (label, needs_source, _two, _params, _build) in TEMPLATES.items():
            if key == "custom" or needs_source == based:
                self.kind.addItem(tr(label), key)
        self.kind.blockSignals(False)
        self.same_range.setVisible(based)
        self._kind_changed()

    def _kind_changed(self) -> None:
        key = self.kind.currentData()
        _label, _src, two, params, build = TEMPLATES[key]
        for i, (label, spin) in enumerate(zip(self.param_labels, self.param_edits)):
            visible = i < len(params)
            label.setVisible(visible)
            spin.setVisible(visible)
            if visible:
                label.setText(tr(params[i][0]))
                spin.blockSignals(True)
                spin.setValue(params[i][1])
                spin.blockSignals(False)
        self.other.setVisible(two)
        self.other_label.setVisible(two)
        custom = build is None
        self.formula.setVisible(custom)
        self.formula_label.setVisible(custom)
        self.functions.setVisible(custom)
        self._update()

    def _update(self, *_args) -> None:
        key = self.kind.currentData()
        if key is None:
            return
        _label, _src, _two, params, build = TEMPLATES[key]
        src = self._source()
        error = ""
        if build is None:
            text = self.formula.text().strip()
            try:
                compile_expr(text)
            except ExprError as exc:
                error = str(exc)
            attrs, suffix, unit_rule = "exp=" + text, "calc", ""
        else:
            values = [spin.value() for spin in self.param_edits[:len(params)]]
            attrs, suffix, unit_rule = build(values, self.other.currentText())
        if not self._name_touched:
            self.name_edit.setText((src.name + "_" + suffix) if src is not None else "REF_" + suffix.upper())
        if not self._unit_touched:
            base = src.unit if src is not None else ""
            unit = {"": base, "/s": (base + "/s") if base else "1/s",
                    "*s": (base + "*s") if base else "s", "none": ""}[unit_rule]
            self.unit_edit.setText(unit)
        entry = PltEntry(name=src.entry.name if src is not None else "*", view="nobit",
                         color=self.color_box.currentData(), unit=self.unit_edit.text().strip())
        entry.key = self.name_edit.text().strip() or None
        if src is not None:
            entry.channel_origin = src.entry.channel_origin
            entry.bit_pos, entry.bit_count, entry.bit_signed = src.entry.bit_pos, src.entry.bit_count, src.entry.bit_signed
            entry.ops = list(src.entry.ops)                 # the copy starts from what the source shows
            if self.same_range.isChecked() and unit_rule == "":
                entry.scmin, entry.scmax = src.scmin, src.scmax
        parse_attributes(attrs, entry)
        mode = self.raster.currentData() if hasattr(self, "raster") else "union"
        self.raster_step.setEnabled(mode == "fixed") if hasattr(self, "raster_step") else None
        if mode == "fixed":
            entry.comment = "// @raster=fixed:%g" % (self.raster_step.value() / 1000.0)
        elif mode == "signal:x1" and not (entry.ops and any(op.refs for op in entry.ops)):
            entry.comment = None
        elif mode and mode != "union":
            entry.comment = "// @raster=" + mode
        self._entry = None if error else entry
        self.preview.setText(error or "\n".join(format_entry(entry)))
        self.preview.setStyleSheet("font-family: Consolas, monospace" + ("; color: #c00000" if error else ""))
        self.preview.setMinimumHeight(self.preview.heightForWidth(max(200, self.preview.width())))
        self.adjustSize()
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(not error and bool(entry.key))

    def entry(self) -> Optional[PltEntry]:
        """The new script entry (``None`` if the dialog content is not valid)."""
        return self._entry

    def source_entry(self) -> Optional[PltEntry]:
        src = self._source()
        return src.entry if src is not None else None

    def other_entry(self) -> Optional[PltEntry]:
        """The second signal of a two-signal function (x1), else ``None``."""
        key = self.kind.currentData()
        if key is None or not TEMPLATES[key][2] or not self.signals:
            return None
        index = self.other.currentIndex()
        return self.signals[index].entry if 0 <= index < len(self.signals) else None


# ---------------------------------------------------------------------- reference lines
class ReferenceLineDialog(QtWidgets.QDialog):
    """Up to three horizontal reference lines of one signal: a value, or its MIN / MAX / MEAN."""

    def __init__(self, signal: DisplaySignal, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Reference lines"))
        self.signal = signal
        form = QtWidgets.QFormLayout(self)
        form.addRow(QtWidgets.QLabel("<b>%s</b>" % signal.label))
        self.rows: List[Tuple[QtWidgets.QComboBox, QtWidgets.QDoubleSpinBox]] = []
        for index in range(3):
            mode = QtWidgets.QComboBox()
            mode.addItems([tr("(none)"), tr("Value"), "MIN", "MAX", "MEAN"])
            spin = NumberBox()
            current = signal.entry.refs.get(index)
            if isinstance(current, str):
                mode.setCurrentText(current)
            elif current is not None:
                mode.setCurrentIndex(1)
                spin.setValue(float(current))
            mode.currentIndexChanged.connect(lambda i, s=spin: s.setEnabled(i == 1))
            spin.setEnabled(mode.currentIndex() == 1)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(mode)
            row.addWidget(spin, 1)
            form.addRow(tr("Line %d") % (index + 1), row)
            self.rows.append((mode, spin))
        form.addRow(QtWidgets.QLabel(tr("Lines are drawn dashed in the signal's colour, at its own scale.")))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def refs(self) -> Dict[int, object]:
        out: Dict[int, object] = {}
        for index, (mode, spin) in enumerate(self.rows):
            if mode.currentIndex() == 1:
                out[index] = float(spin.value())
            elif mode.currentIndex() > 1:
                out[index] = mode.currentText()
        return out


# ---------------------------------------------------------------------- time range
class TimeRangeDialog(QtWidgets.QDialog):
    """Set the visible time range exactly using its endpoints."""

    def __init__(self, full: Tuple[float, float], current: Tuple[float, float],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Visible time range"))
        self.full = (float(full[0]), float(full[1]))
        form = QtWidgets.QFormLayout(self)
        form.addRow(QtWidgets.QLabel(tr("Recording: %.9g ... %.9g s") % self.full))
        self.t0, self.t1 = NumberBox(), NumberBox()
        self.t0.setValue(float(current[0]))
        self.t1.setValue(float(current[1]))
        form.addRow("T0 [s]", self.t0)
        form.addRow("T1 [s]", self.t1)
        self.note = QtWidgets.QLabel()
        self.note.setStyleSheet("color: #c00000")
        form.addRow("", self.note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        everything = buttons.addButton(tr("Whole recording"), QtWidgets.QDialogButtonBox.ResetRole)
        everything.clicked.connect(self._everything)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _everything(self) -> None:
        self.t0.setValue(self.full[0])
        self.t1.setValue(self.full[1])

    def values(self) -> Tuple[float, float]:
        return float(self.t0.value()), float(self.t1.value())

    def accept(self) -> None:
        t0, t1 = self.values()
        if not t1 > t0:
            self.note.setText(tr("T1 must be greater than T0"))
            return
        if t1 < self.full[0] or t0 > self.full[1]:
            self.note.setText(tr("The range lies outside the recording"))
            return
        super().accept()


# ---------------------------------------------------------------------- favourites
class FavouritesDialog(QtWidgets.QDialog):
    """Administrate the favourite PLT scripts: add, remove, reorder."""

    def __init__(self, paths: Sequence[str], current: Optional[str], start_dir: str,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Favourite views"))
        self.resize(760, 420)
        self.current = current
        self.start_dir = start_dir
        layout = QtWidgets.QHBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for p in paths:
            self.list.addItem(p)
        layout.addWidget(self.list, 1)
        side = QtWidgets.QVBoxLayout()
        self.btn_current = QtWidgets.QPushButton(tr("Add current script"))
        self.btn_current.setEnabled(bool(current))
        self.btn_files = QtWidgets.QPushButton(tr("Add script files..."))
        self.btn_remove = QtWidgets.QPushButton(tr("Remove"))
        self.btn_up = QtWidgets.QPushButton(tr("Up"))
        self.btn_down = QtWidgets.QPushButton(tr("Down"))
        for b in (self.btn_current, self.btn_files, self.btn_remove, self.btn_up, self.btn_down):
            side.addWidget(b)
        side.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
                                             QtCore.Qt.Vertical)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        side.addWidget(buttons)
        layout.addLayout(side)
        self.btn_current.clicked.connect(lambda: self._add([self.current] if self.current else []))
        self.btn_files.clicked.connect(self._add_files)
        self.btn_remove.clicked.connect(self._remove)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))

    def paths(self) -> List[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _add(self, paths: Sequence[str]) -> None:
        known = {p.lower() for p in self.paths()}
        for p in paths:
            if p and p.lower() not in known:
                self.list.addItem(p)
                known.add(p.lower())

    def _add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, tr("Add script files..."), self.start_dir,
                                                          tr("Views") + " (*.plt)")
        self._add(files)

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _move(self, step: int) -> None:
        row = self.list.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.list.count():
            return
        item = self.list.takeItem(row)
        self.list.insertItem(target, item)
        self.list.setCurrentRow(target)


# ---------------------------------------------------------------------- information
class InfoDialog(QtWidgets.QDialog):
    """Texts about the recording (and optionally one signal) in a list of sections."""

    def __init__(self, sections: Dict[str, str], title: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        layout = QtWidgets.QHBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(260)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.text.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(self.list)
        layout.addWidget(self.text, 1)
        self._sections = sections
        for name in sections:
            self.list.addItem(name)
        self.list.currentTextChanged.connect(self._show)
        if sections:
            self.list.setCurrentRow(0)

    def _show(self, name: str) -> None:
        text = self._sections.get(name, "")
        if len(text) > 2_000_000:
            text = text[:2_000_000] + "\n..."
        self.text.setPlainText(text)


def signal_info_sections(sig: DisplaySignal, dataset: Optional[Dataset], stats: Optional[Dict[str, float]]) -> Dict[str, str]:
    """Sections for :class:`InfoDialog` describing one signal."""
    info = sig.info
    lines = [f"Display name : {sig.name}", f"Unit         : {sig.unit}"]
    if info is not None:
        lines += [f"Recorded name: {info.source_name}", f"Raster       : {info.raster}",
                  f"Source       : {info.group}", f"Samples      : {info.sample_count}",
                  f"Kind         : {info.kind}"]
    lines += [f"View range   : {sig.scmin:g} ... {sig.scmax:g}",
              f"Window       : {'DIGITAL' if sig.digital else 'analog'}"]
    if sig.entry.ops:
        lines.append("Processing   : " + " ".join(op.text for op in sig.entry.ops))
    if sig.error:
        lines.append("Problem      : " + sig.error)
    if stats:
        lines += ["", f"Minimum      : {stats['min']:g}", f"Maximum      : {stats['max']:g}",
                  f"Mean         : {stats['mean']:g}", f"Std deviation: {stats['std']:g}"]
    out = {"Signal": "\n".join(lines)}
    if info is not None and info.description:
        out["Description"] = info.description
    if sig.enum:
        rows = [f"{v:g} = {t}" for v, t in sorted(sig.enum.values.items())]
        rows += [f"{lo:g} ... {hi:g} = {t}" for lo, hi, t in sig.enum.ranges]
        if sig.enum.default is not None:
            rows.append(f"(other) = {sig.enum.default}")
        out["Text table"] = "\n".join(rows)
    return out
