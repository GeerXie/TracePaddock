"""Choose a range of visible PLT colours and the signals to colour."""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from ..plt.palette import MAX_PALETTE_COLOR, text_color_on
from .render import ViewState, signal_color


class _ColorRangeSlider(QtWidgets.QWidget):
    """A horizontal range with two thumbs. The paired spin boxes support keyboard entry."""

    rangeChanged = QtCore.Signal(int, int)

    def __init__(self, color_range: Tuple[int, int], parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.values = color_range
        self._active: Optional[int] = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(280, self.style().pixelMetric(QtWidgets.QStyle.PM_SliderThickness) + 6)

    def set_values(self, first: int, last: int) -> None:
        if (first, last) != self.values:
            self.values = (first, last)
            self.update()
            self.rangeChanged.emit(first, last)

    def _option(self, value: int) -> QtWidgets.QStyleOptionSlider:
        option = QtWidgets.QStyleOptionSlider()
        option.initFrom(self)
        option.orientation = QtCore.Qt.Horizontal
        option.minimum = 1
        option.maximum = MAX_PALETTE_COLOR
        option.sliderPosition = option.sliderValue = value
        option.upsideDown = self.layoutDirection() == QtCore.Qt.RightToLeft
        return option

    def _thumb_center(self, value: int) -> int:
        style = QtWidgets.QStyle
        return self.style().subControlRect(style.CC_Slider, self._option(value), style.SC_SliderHandle, self).center().x()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtWidgets.QStylePainter(self)
        style = QtWidgets.QStyle
        option = self._option(self.values[0])
        option.subControls = style.SC_SliderGroove
        painter.drawComplexControl(style.CC_Slider, option)
        groove = self.style().subControlRect(style.CC_Slider, option, style.SC_SliderGroove, self)
        painter.setPen(QtGui.QPen(self.palette().highlight().color(), 3))
        painter.drawLine(self._thumb_center(self.values[0]), groove.center().y(),
                         self._thumb_center(self.values[1]), groove.center().y())
        for index, value in enumerate(self.values):
            option = self._option(value)
            option.subControls = style.SC_SliderHandle
            if index == self._active:
                option.activeSubControls = style.SC_SliderHandle
                option.state |= style.State_Sunken
            painter.drawComplexControl(style.CC_Slider, option)

    def _slide(self, x: int) -> None:
        left, right = sorted((self._thumb_center(1), self._thumb_center(MAX_PALETTE_COLOR)))
        value = QtWidgets.QStyle.sliderValueFromPosition(
            1, MAX_PALETTE_COLOR, x - left, right - left, self._option(1).upsideDown)
        first, last = self.values
        if first == last and value != first:
            self._active = 0 if value < first else 1
        if self._active == 0:
            self.set_values(min(value, last), last)
        else:
            self.set_values(first, max(first, value))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            x = event.pos().x()
            self._active = min(range(2), key=lambda i: abs(x - self._thumb_center(self.values[i])))
            self._slide(x)
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._active is not None:
            self._slide(event.pos().x())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._active is not None and event.button() == QtCore.Qt.LeftButton:
            self._slide(event.pos().x())
            self._active = None
            self.update()
        else:
            super().mouseReleaseEvent(event)


class CyclicColorsDialog(QtWidgets.QDialog):
    Selected = 2

    def __init__(self, color_range: Tuple[int, int], state: ViewState, selected_count: int, total_count: int,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Cyclic colours"))
        self._state = state
        self.slider = _ColorRangeSlider(color_range, self)
        self.first = QtWidgets.QSpinBox()
        self.last = QtWidgets.QSpinBox()
        row = QtWidgets.QHBoxLayout()
        for text, spin in ((tr("First colour"), self.first), (tr("Last colour"), self.last)):
            label = QtWidgets.QLabel(text)
            label.setBuddy(spin)
            row.addWidget(label)
            row.addWidget(spin)
            row.addStretch()
            spin.setRange(1, MAX_PALETTE_COLOR)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.slider)
        self.summary = QtWidgets.QLabel()
        layout.addWidget(self.summary)
        previews = QtWidgets.QGridLayout()
        self._swatches = []
        for number in range(MAX_PALETTE_COLOR):
            swatch = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
            swatch.setMinimumSize(32, 22)
            previews.addWidget(swatch, number // 8, number % 8)
            self._swatches.append(swatch)
        layout.addLayout(previews)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        selected = buttons.addButton(tr("Colour selected signals"), QtWidgets.QDialogButtonBox.ActionRole)
        all_signals = buttons.addButton(tr("Colour all signals"), QtWidgets.QDialogButtonBox.ActionRole)
        selected.setEnabled(selected_count > 0)
        all_signals.setEnabled(total_count > 0)
        selected.clicked.connect(lambda: self.done(self.Selected))
        all_signals.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.slider.rangeChanged.connect(self._update_preview)
        self.first.valueChanged.connect(lambda value: self.slider.set_values(value, max(value, self.last.value())))
        self.last.valueChanged.connect(lambda value: self.slider.set_values(min(value, self.first.value()), value))
        self._update_preview(*color_range)
        self.resize(470, self.sizeHint().height())

    def color_range(self) -> Tuple[int, int]:
        return self.slider.values

    def _update_preview(self, first: int, last: int) -> None:
        with QtCore.QSignalBlocker(self.first), QtCore.QSignalBlocker(self.last):
            self.first.setValue(first)
            self.last.setValue(last)
        self.summary.setText(tr("Colour range %d–%d · %d colours") % (first, last, last - first + 1))
        for color, swatch in enumerate(self._swatches, first):
            swatch.setVisible(color <= last)
            if color <= last:
                rgb = signal_color(color, self._state)
                fill = QtGui.QColor(*rgb).name()
                ink = QtGui.QColor(*text_color_on(rgb)).name()
                swatch.setText(str(color))
                swatch.setToolTip(fill)
                swatch.setStyleSheet(f"background-color: {fill}; color: {ink}; border-radius: 3px;")
