"""Export dialog for the displayed signals, picture export and printing."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtPrintSupport, QtWidgets

from ..core.export import EXPORT_FORMATS, estimate_rows
from ..display import DisplaySignal
from ..i18n import tr
from .dialogs import NumberBox
from .render import ViewState, paint_view
from .themes import apply_plot_colours, get_theme, plot_palette

__all__ = ["ExportDialog", "save_picture", "print_view", "print_preview", "picture_state"]

_SEPARATORS = [(";", ";"), (",", ","), ("\t", "Tab"), (" ", "Space")]


class ExportDialog(QtWidgets.QDialog):
    """Which signals, which time range, which format and options. ``options()`` after ``exec_()``."""

    def __init__(self, signals: Sequence[DisplaySignal], selected: int, full: Tuple[float, float],
                 view: Tuple[float, float], settings: object, default_path: str,
                 parent: Optional[QtWidgets.QWidget] = None, emphasized: Sequence[int] = ()) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Export data"))
        self.signals = list(signals)
        self.selected = selected
        self.emphasized = [i for i in emphasized if 0 <= i < len(self.signals)]
        self.full, self.view = full, view
        self.settings = settings
        form = QtWidgets.QFormLayout(self)

        self.format_combo = QtWidgets.QComboBox()
        for ext, label in EXPORT_FORMATS.items():
            self.format_combo.addItem(tr(label), ext)
        form.addRow(tr("Format"), self.format_combo)

        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(default_path)
        browse = QtWidgets.QPushButton(tr("Browse..."))
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form.addRow(tr("File"), path_row)

        self.which = QtWidgets.QButtonGroup(self)
        box = QtWidgets.QVBoxLayout()
        with_data = sum(1 for s in self.signals if s.has_data)
        self.rb_all = QtWidgets.QRadioButton(tr("All signals of the legend (%d)") % with_data)
        name = self.signals[selected].name if 0 <= selected < len(self.signals) else ""
        self.rb_selected = QtWidgets.QRadioButton(tr("Only the selected signal") + (": " + name if name else ""))
        self.rb_selected.setEnabled(bool(name))
        self.rb_emph = QtWidgets.QRadioButton(tr("Only the emphasized signals (%d)") % len(self.emphasized))
        self.rb_emph.setEnabled(bool(self.emphasized))
        for i, rb in enumerate((self.rb_all, self.rb_selected, self.rb_emph)):
            self.which.addButton(rb, i)
            box.addWidget(rb)
        self.rb_all.setChecked(True)
        form.addRow(tr("Signals"), box)

        self.range_group = QtWidgets.QButtonGroup(self)
        rbox = QtWidgets.QVBoxLayout()
        self.rb_full = QtWidgets.QRadioButton(tr("Whole recording (%.6g ... %.6g s)") % full)
        self.rb_view = QtWidgets.QRadioButton(tr("Visible time range (%.6g ... %.6g s)") % view)
        self.range_group.addButton(self.rb_full, 0)
        self.range_group.addButton(self.rb_view, 1)
        rbox.addWidget(self.rb_full)
        rbox.addWidget(self.rb_view)
        (self.rb_view if view != full else self.rb_full).setChecked(True)
        form.addRow(tr("Time range"), rbox)

        self.axis_group = QtWidgets.QButtonGroup(self)
        abox = QtWidgets.QHBoxLayout()
        self.rb_union = QtWidgets.QRadioButton(tr("Time stamps of all exported signals (values held)"))
        self.rb_step = QtWidgets.QRadioButton(tr("Fixed step [ms]"))
        self.step_box = NumberBox()
        self.step_box.setRange(0.001, 1e7)
        self.step_box.setValue(float(settings.number("export/step_ms", 10.0, 0.001, 1e7)))
        self.axis_group.addButton(self.rb_union, 0)
        self.axis_group.addButton(self.rb_step, 1)
        abox.addWidget(self.rb_union)
        abox.addWidget(self.rb_step)
        abox.addWidget(self.step_box)
        self.rb_union.setChecked(not settings.get("export/fixed_step", False))
        self.rb_step.setChecked(bool(settings.get("export/fixed_step", False)))
        self.axis_label = QtWidgets.QLabel(tr("Time axis"))
        form.addRow(self.axis_label, abox)
        self.mat_common = QtWidgets.QCheckBox(tr("All signals on one time axis (q_T0); off: [t, y] per signal on its own samples"))
        form.addRow("", self.mat_common)

        self.csv_box = QtWidgets.QGroupBox(tr("Table options"))
        cform = QtWidgets.QFormLayout(self.csv_box)
        self.sep_combo = QtWidgets.QComboBox()
        for value, label in _SEPARATORS:
            self.sep_combo.addItem(tr(label), value)
        self.dec_combo = QtWidgets.QComboBox()
        self.dec_combo.addItem(".", ".")
        self.dec_combo.addItem(",", ",")
        self.fmt_edit = QtWidgets.QLineEdit(str(settings.get("export/number_format", "%.9g")))
        self.fmt_edit.setToolTip(tr("printf format for values in the legend, e.g. %g  %.3f  %.6g"))
        self.quote_combo = QtWidgets.QComboBox()
        self.quote_combo.addItem('"', '"')
        self.quote_combo.addItem("'", "'")
        self.quote_combo.addItem(tr("(none)"), "")
        self.header_check = QtWidgets.QCheckBox(tr("1st row names, 2nd row units"))
        self.header_check.setChecked(bool(settings.get("export/header", True)))
        self.text_combo = QtWidgets.QComboBox()
        self.text_combo.addItem(tr("Raw value"), "raw")
        self.text_combo.addItem(tr("Mapped value"), "mapped")
        self.text_combo.addItem(tr("Mapped value (raw value)"), "both")
        cform.addRow(tr("Column separator"), self.sep_combo)
        cform.addRow(tr("Decimal separator"), self.dec_combo)
        cform.addRow(tr("Number format"), self.fmt_edit)
        cform.addRow(tr("Text quote"), self.quote_combo)
        cform.addRow("", self.header_check)
        cform.addRow(tr("Signals with a text table / value mapping"), self.text_combo)
        for combo, key, default in ((self.sep_combo, "export/separator", ";"), (self.dec_combo, "export/decimal", "."),
                                    (self.quote_combo, "export/quote", '"'), (self.text_combo, "export/text_values", "raw")):
            index = combo.findData(settings.get(key, default))
            combo.setCurrentIndex(max(0, index))
        form.addRow(self.csv_box)

        self.estimate = QtWidgets.QLabel()
        self.estimate.setStyleSheet("color: gray")
        form.addRow("", self.estimate)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        last = str(settings.get("export/format", ".csv"))
        self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(last)))
        if Path(default_path).suffix.lower() != self.format_combo.currentData():
            self.path_edit.setText(str(Path(default_path).with_suffix(self.format_combo.currentData())))
        browse.clicked.connect(self._browse)
        self.format_combo.currentIndexChanged.connect(self._format_changed)
        for group in (self.which, self.range_group, self.axis_group):
            group.buttonClicked.connect(self._update)
        self.step_box.valueChanged.connect(self._update)
        self.mat_common.toggled.connect(self._update)
        self._format_changed()

    # ------------------------------------------------------------------ state
    def _format_changed(self, *_args) -> None:
        ext = self.format_combo.currentData()
        path = self.path_edit.text().strip()
        if path:
            self.path_edit.setText(str(Path(path).with_suffix(ext)))
        table = ext in (".csv", ".txt")
        self.csv_box.setVisible(table)
        self.mat_common.setVisible(ext == ".mat")
        if ext == ".txt" and self.sep_combo.currentData() == ";":
            self.sep_combo.setCurrentIndex(self.sep_combo.findData("\t"))
        self._update()

    def _update(self, *_args) -> None:
        ext = self.format_combo.currentData()
        axis = ext in (".csv", ".txt") or (ext == ".mat" and self.mat_common.isChecked())
        for w in (self.axis_label, self.rb_union, self.rb_step, self.step_box):
            w.setVisible(axis)
        self.step_box.setEnabled(self.rb_step.isChecked())
        signals = self.chosen_signals()
        t0, t1 = self.time_range()
        if axis:
            rows = estimate_rows(signals, t0, t1, self.step_seconds())
            note = tr("about %s rows x %d columns") % (format(rows, ","), len(signals) + 1)
            if not self.rb_step.isChecked():
                note += "  " + tr("(at most; equal time stamps share a row)")
        else:
            note = tr("%d signals, each with its own time stamps") % len(signals)
        self.estimate.setText(note)

    def _browse(self) -> None:
        ext = self.format_combo.currentData()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("Export data"), self.path_edit.text(),
                                                        "%s (*%s)" % (tr(EXPORT_FORMATS[ext]), ext))
        if path:
            self.path_edit.setText(path)

    # ------------------------------------------------------------------ results
    def chosen_signals(self) -> List[DisplaySignal]:
        if self.rb_selected.isChecked() and 0 <= self.selected < len(self.signals):
            chosen = [self.signals[self.selected]]
        elif self.rb_emph.isChecked() and self.emphasized:
            chosen = [self.signals[i] for i in self.emphasized]
        else:
            chosen = list(self.signals)
        return [s for s in chosen if s.has_data]

    def time_range(self) -> Tuple[float, float]:
        return self.view if self.rb_view.isChecked() else self.full

    def step_seconds(self) -> Optional[float]:
        return self.step_box.value() / 1000.0 if self.rb_step.isChecked() else None

    def options(self) -> Dict[str, object]:
        ext = self.format_combo.currentData()
        return {"format": ext, "path": self.path_edit.text().strip(), "signals": self.chosen_signals(),
                "range": self.time_range(), "step": self.step_seconds(), "mat_common": self.mat_common.isChecked(),
                "separator": self.sep_combo.currentData(), "decimal": self.dec_combo.currentData(),
                "number_format": self.fmt_edit.text().strip() or "%.9g", "quote": self.quote_combo.currentData(),
                "header": self.header_check.isChecked(), "text_values": self.text_combo.currentData()}

    def accept(self) -> None:
        o = self.options()
        if not o["path"]:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), tr("Please choose a file"))
            return
        if not o["signals"]:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(), tr("None of the chosen signals has data"))
            return
        if o["format"] in (".csv", ".txt") and o["separator"] == o["decimal"]:
            QtWidgets.QMessageBox.warning(self, self.windowTitle(),
                                          tr("Column separator and decimal separator must differ"))
            return
        s = self.settings
        s.set("export/format", o["format"])
        s.set("export/separator", o["separator"])
        s.set("export/decimal", o["decimal"])
        s.set("export/number_format", o["number_format"])
        s.set("export/quote", o["quote"])
        s.set("export/header", o["header"])
        s.set("export/text_values", o["text_values"])
        s.set("export/fixed_step", self.rb_step.isChecked())
        s.set("export/step_ms", self.step_box.value())
        s.set("export/dir", str(Path(o["path"]).parent))
        super().accept()


# ---------------------------------------------------------------------- pictures and printing
def picture_state(state: ViewState, white: bool) -> ViewState:
    """Copy of the view state for a picture or print: no flashing, optionally on white - then with the colours
    the theme uses on a light plot (legend, text, grid, cursor, marker, signal colours)."""
    out = copy.copy(state)
    out.flash = True
    if white:
        out.background = (255, 255, 255)
        if out.theme:
            theme = get_theme(out.theme)
            own_palette = list(state.palette) != plot_palette(theme, state.dark)    # settings "palette"
            apply_plot_colours(out, theme)
            if own_palette:
                out.palette = state.palette
    return out


def _header(painter: QtGui.QPainter, rect: QtCore.QRectF, lines: Sequence[str]) -> None:
    font = QtGui.QFont("Microsoft YaHei UI")
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(20, 20, 20))
    painter.drawText(rect, int(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter), "    ".join(lines))


def _paint_page(painter: QtGui.QPainter, page: QtCore.QRectF, size: QtCore.QSizeF, signals: Sequence[DisplaySignal],
                state: ViewState, header: Sequence[str]) -> None:
    """Paint header and view into ``page``, scaled like the screen (logical pixels of ``size``)."""
    head_h = 26.0 if header else 0.0
    virtual = QtCore.QSizeF(size.width(), size.height() + head_h)
    scale = min(page.width() / virtual.width(), page.height() / virtual.height())
    painter.save()
    painter.translate(page.left(), page.top())
    painter.scale(scale, scale)
    if header:
        painter.fillRect(QtCore.QRectF(0, 0, size.width(), head_h), QtGui.QColor(255, 255, 255))
        _header(painter, QtCore.QRectF(4, 0, size.width() - 8, head_h), header)
    paint_view(painter, QtCore.QRectF(0, head_h, size.width(), size.height()), signals, state)
    painter.restore()


def save_picture(path: "str | Path", size: QtCore.QSize, signals: Sequence[DisplaySignal], state: ViewState,
                 header: Sequence[str] = ()) -> None:
    """PNG / JPG (pixels as on the screen), SVG or PDF (vector) of the view. Raises ``OSError``."""
    p = Path(path)
    ext = p.suffix.lower()
    logical = QtCore.QSizeF(size)
    if ext == ".pdf":
        writer = QtGui.QPdfWriter(str(p))
        writer.setResolution(96)
        writer.setPageOrientation(QtGui.QPageLayout.Landscape)
        painter = QtGui.QPainter()
        if not painter.begin(writer):
            raise OSError(f"{p.name} cannot be written")
        try:
            page = writer.pageLayout().paintRectPixels(writer.resolution())
            _paint_page(painter, QtCore.QRectF(0, 0, page.width(), page.height()), logical, signals, state, header)
        finally:
            painter.end()
        return
    if ext == ".svg":
        from PySide6 import QtSvg

        generator = QtSvg.QSvgGenerator()
        generator.setFileName(str(p))
        head_h = 26 if header else 0
        generator.setSize(QtCore.QSize(size.width(), size.height() + head_h))
        generator.setViewBox(QtCore.QRect(0, 0, size.width(), size.height() + head_h))
        generator.setTitle(p.stem)
        painter = QtGui.QPainter()
        if not painter.begin(generator):
            raise OSError(f"{p.name} cannot be written")
        try:
            _paint_page(painter, QtCore.QRectF(0, 0, size.width(), size.height() + head_h), logical, signals, state, header)
        finally:
            painter.end()
        return
    head_h = 26 if header else 0
    image = QtGui.QImage(size.width(), size.height() + head_h, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(*state.background))
    painter = QtGui.QPainter(image)
    try:
        _paint_page(painter, QtCore.QRectF(0, 0, image.width(), image.height()), logical, signals, state, header)
    finally:
        painter.end()
    if not image.save(str(p)):
        raise OSError(f"{p.name} cannot be written")


def _render_to_printer(printer: QtPrintSupport.QPrinter, size: QtCore.QSize, signals: Sequence[DisplaySignal],
                       state: ViewState, header: Sequence[str]) -> None:
    painter = QtGui.QPainter()
    if not painter.begin(printer):
        return
    try:
        page = QtCore.QRectF(printer.pageRect(QtPrintSupport.QPrinter.DevicePixel))
        _paint_page(painter, page.translated(-page.topLeft()), QtCore.QSizeF(size), signals, state, header)
    finally:
        painter.end()


def _printer(settings: object) -> QtPrintSupport.QPrinter:
    printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.ScreenResolution)
    printer.setPageOrientation(QtGui.QPageLayout.Landscape)
    name = str(settings.get("print/printer", "") or "")
    if name:
        printer.setPrinterName(name)
    return printer


def print_view(parent: QtWidgets.QWidget, settings: object, size: QtCore.QSize, signals: Sequence[DisplaySignal],
               state: ViewState, header: Sequence[str]) -> bool:
    printer = _printer(settings)
    dialog = QtPrintSupport.QPrintDialog(printer, parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return False
    settings.set("print/printer", printer.printerName())
    _render_to_printer(printer, size, signals, state, header)
    return True


def print_preview(parent: QtWidgets.QWidget, settings: object, size: QtCore.QSize, signals: Sequence[DisplaySignal],
                  state: ViewState, header: Sequence[str]) -> None:
    printer = _printer(settings)
    dialog = QtPrintSupport.QPrintPreviewDialog(printer, parent)
    dialog.paintRequested.connect(lambda p: _render_to_printer(p, size, signals, state, header))
    dialog.resize(1100, 800)
    dialog.exec_()


def header_lines(dataset_path: object, script_path: object, x_range: Tuple[float, float]) -> List[str]:
    return [str(dataset_path or ""), str(script_path or tr("(new script)")),
            "%.6g ... %.6g s" % x_range, time.strftime("%Y-%m-%d %H:%M")]
