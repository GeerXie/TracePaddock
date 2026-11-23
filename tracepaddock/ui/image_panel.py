"""Image panel: a picture per signal for the value at the cursor (e.g. the speed limit sign).

Signals take part when their mapping table has pictures or shows speed signs
automatically (value mapping dialog, column "Picture"). The panel follows the
time of the cursor / marker; a tile flashes when its value changes.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..display import DisplaySignal
from ..i18n import tr
from ..signs import sign_pixmap

__all__ = ["ImagePanel"]


class _Tile(QtWidgets.QFrame):
    def __init__(self, sig: DisplaySignal, size: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.sig = sig
        self.size_px = size
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        self.title = QtWidgets.QLabel(sig.name)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setStyleSheet("font-weight: bold")
        self.picture = QtWidgets.QLabel()
        self.picture.setAlignment(QtCore.Qt.AlignCenter)
        self.picture.setMinimumSize(size + 8, int(size * 1.45) + 8)
        self.value = QtWidgets.QLabel()
        self.value.setAlignment(QtCore.Qt.AlignCenter)
        self.value.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.picture, 1)
        layout.addWidget(self.value)
        self.last: Optional[float] = None
        self._flash = QtCore.QTimer(self)
        self._flash.setSingleShot(True)
        self._flash.timeout.connect(lambda: self.setStyleSheet(""))

    def show_time(self, time: Optional[float]) -> bool:
        """Update for ``time``; returns True if the value changed."""
        value = self.sig.value_at(time) if time is not None else None
        changed = value is not None and self.last is not None and value != self.last
        if value == self.last and self.picture.pixmap() is not None:
            return False
        self.last = value
        spec = self.sig.image_for(value) if value is not None else None
        pix = sign_pixmap(spec, self.size_px) if spec else None
        if pix is not None:
            self.picture.setPixmap(pix)
        else:
            self.picture.clear()
            self.picture.setText("—" if value is None else "")
        self.value.setText("" if value is None else self.sig.format_value(value))
        if changed:
            self.setStyleSheet("_Tile, QFrame { border: 3px solid #f08c00; border-radius: 4px; }")
            self._flash.start(1200)
        return changed


class ImagePanel(QtWidgets.QWidget):
    """Tiles of the signals that have pictures. ``configure(index)`` opens the mapping dialog."""

    def __init__(self, configure: Callable[[int], None], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.configure = configure
        self.signals: List[DisplaySignal] = []
        self.tiles: List[_Tile] = []
        self.size_px = 96
        self.time: Optional[float] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.time_label = QtWidgets.QLabel()
        setup = QtWidgets.QPushButton(tr("Pictures..."))
        setup.setToolTip(tr("Give the values of a signal pictures (value mapping, column Picture)"))
        self.size_box = QtWidgets.QComboBox()
        for px in (64, 96, 128, 192):
            self.size_box.addItem("%d px" % px, px)
        self.size_box.setCurrentIndex(1)
        bar.addWidget(self.time_label, 1)
        bar.addWidget(self.size_box)
        bar.addWidget(setup)
        layout.addLayout(bar)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.host)
        self.grid.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.scroll.setWidget(self.host)
        layout.addWidget(self.scroll, 1)
        self.empty = QtWidgets.QLabel(tr("No signal has pictures yet. Right click a legend row -> 'Value mapping...' "
                                         "in the editor, or press 'Pictures...': choose a picture per value, or let "
                                         "the panel show speed limit signs automatically."))
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet("color: gray")
        layout.addWidget(self.empty)
        setup.clicked.connect(lambda: self.configure(-1))
        self.size_box.currentIndexChanged.connect(self._size_changed)

    def _size_changed(self) -> None:
        self.size_px = int(self.size_box.currentData())
        self.set_signals(self.signals)

    def set_signals(self, signals: Sequence[DisplaySignal]) -> None:
        self.signals = list(signals)
        for tile in self.tiles:
            tile.setParent(None)
            tile.deleteLater()
        self.tiles = []
        shown = [s for s in self.signals if s.has_data and s.value_map is not None and s.value_map.has_images]
        columns = max(1, (self.width() - 20) // (self.size_px + 40))
        for i, sig in enumerate(shown):
            tile = _Tile(sig, self.size_px, self.host)
            tile.mouseDoubleClickEvent = lambda _e, s=sig: self.configure(self.signals.index(s))
            self.grid.addWidget(tile, i // columns, i % columns)
            self.tiles.append(tile)
        self.empty.setVisible(not shown)
        self.show_time(self.time)

    def show_time(self, time: Optional[float]) -> None:
        if time is not None:
            self.time = time
        self.time_label.setText(tr("Time: %.3f s") % self.time if self.time is not None else tr("Move the cursor"))
        for tile in self.tiles:
            tile.show_time(self.time)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.tiles:
            columns = max(1, (self.width() - 20) // (self.size_px + 40))
            for i, tile in enumerate(self.tiles):
                self.grid.addWidget(tile, i // columns, i % columns)
