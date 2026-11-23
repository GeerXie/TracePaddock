"""Theme management: apply a theme, delete themes the user does not want, restore deleted ones.

The classic theme cannot be deleted. Deleting only hides a built-in theme (settings "hidden_themes"),
so "Restore deleted themes" brings it back.
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import tr
from .themes import PROTECTED_THEME, THEME_KEYS, THEMES, Theme, hex_rgb, theme_palette

__all__ = ["ThemeDialog", "theme_preview"]


def theme_preview(theme: Theme, width: int = 96, height: int = 36) -> QtGui.QPixmap:
    """A small picture of a theme: window colour, plot colour with a few signal colours, accent."""
    pix = QtGui.QPixmap(width, height)
    pix.fill(QtGui.QColor(theme.window or "#f0f0f0"))
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    plot = QtCore.QRectF(4, 6, width * 0.62, height - 10)
    painter.fillRect(plot, QtGui.QColor(theme.plot or "#000000"))
    palette = theme_palette(theme)
    for k, index in enumerate((7, 4, 2, 3)):
        painter.setPen(QtGui.QPen(QtGui.QColor(*palette[index]), 1.6))
        y = plot.top() + 4 + k * (plot.height() - 8) / 3.0
        painter.drawLine(QtCore.QPointF(plot.left() + 3, y), QtCore.QPointF(plot.right() - 3, y + (-3 if k % 2 else 3)))
    legend = QtCore.QRectF(plot.right() + 3, 6, width - plot.right() - 7, height - 10)
    for k, index in enumerate((2, 3, 5)):
        r = QtCore.QRectF(legend.left(), legend.top() + k * legend.height() / 3.0, legend.width(), legend.height() / 3.0 - 1)
        painter.fillRect(r, QtGui.QColor(*palette[index]))
    painter.fillRect(QtCore.QRectF(0, 0, width, 3), QtGui.QColor(theme.accent or "#0078d4"))
    painter.end()
    return pix


class ThemeDialog(QtWidgets.QDialog):
    """``applied``: the theme chosen with Apply (or ``None``); ``hidden``: keys of deleted themes."""

    def __init__(self, current: str, hidden: Iterable[str], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Themes"))
        self.current = current
        self.hidden: Set[str] = {str(k) for k in hidden if str(k) in THEMES and str(k) != PROTECTED_THEME}
        self.applied: Optional[str] = None
        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(tr("Double click or Apply: use the theme. Delete removes it from the menu; "
                                   "the classic theme always stays."))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.list = QtWidgets.QListWidget()
        self.list.setIconSize(QtCore.QSize(96, 36))
        self.list.setSpacing(3)
        layout.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_apply = QtWidgets.QPushButton(tr("Apply"))
        self.b_delete = QtWidgets.QPushButton(tr("Delete theme"))
        self.b_restore = QtWidgets.QPushButton(tr("Restore deleted themes"))
        close = QtWidgets.QPushButton(tr("Close"))
        for b in (self.b_apply, self.b_delete, self.b_restore):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        layout.addLayout(row)
        self.b_apply.clicked.connect(self._apply)
        self.b_delete.clicked.connect(self._delete)
        self.b_restore.clicked.connect(self._restore)
        close.clicked.connect(self.accept)
        self.list.itemDoubleClicked.connect(lambda _item: self._apply())
        self.list.currentItemChanged.connect(lambda *_: self._update_buttons())
        self.resize(460, 380)
        self._fill()

    def _fill(self, select: Optional[str] = None) -> None:
        self.list.clear()
        for key in THEME_KEYS:
            if key in self.hidden:
                continue
            theme = THEMES[key]
            text = tr(theme.name) + ("  " + tr("(in use)") if key == self.current else "")
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(theme_preview(theme)), text)
            item.setData(QtCore.Qt.UserRole, key)
            self.list.addItem(item)
            if key == (select or self.current):
                self.list.setCurrentItem(item)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        self._update_buttons()

    def _key(self) -> Optional[str]:
        item = self.list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _update_buttons(self) -> None:
        key = self._key()
        protected = key == PROTECTED_THEME
        self.b_delete.setEnabled(key is not None and not protected)
        self.b_delete.setToolTip(tr("The classic theme cannot be deleted.") if protected else "")
        self.b_restore.setEnabled(bool(self.hidden))
        self.b_apply.setEnabled(key is not None)

    def _apply(self) -> None:
        key = self._key()
        if key is not None:
            self.applied = key
            self.accept()

    def _delete(self, confirm: bool = True) -> None:
        key = self._key()
        if key is None or key == PROTECTED_THEME:
            return
        if confirm and QtWidgets.QMessageBox.question(
                self, tr("Delete theme"), tr("Delete the theme '%s'? It can be restored later.") % tr(THEMES[key].name)
        ) != QtWidgets.QMessageBox.Yes:
            return
        self.hidden.add(key)
        if self.applied == key:
            self.applied = None
        self._fill()

    def _restore(self) -> None:
        self.hidden.clear()
        self._fill(self._key())
