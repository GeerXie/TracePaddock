"""A bundled walkthrough the user can keep open while trying the examples."""
from pathlib import Path
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import language, tr


WARMUP_DIR = Path(__file__).resolve().parents[1] / "samples" / "warmup"
LESSONS = (("01-plot", "plot"), ("02-processing", "processing"), ("03-timing", "timing"))


class WarmupDialog(QtWidgets.QDialog):
    def __init__(self, open_recording: Callable[[Path, Path], bool], parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self._open_recording = open_recording
        self._language = ""
        self.resize(800, 680)

        self.lesson = QtWidgets.QComboBox()
        self.open_button = QtWidgets.QPushButton()
        self.open_button.clicked.connect(self._open_lesson)
        self.lesson.currentIndexChanged.connect(self._show_lesson)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.lesson, 1)
        top.addWidget(self.open_button)

        self.guide = QtWidgets.QTextBrowser()
        self.guide.setOpenExternalLinks(False)
        self.guide.document().setDocumentMargin(12)

        self.folder_button = QtWidgets.QPushButton()
        self.folder_button.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(WARMUP_DIR))))
        self.close_button = QtWidgets.QPushButton()
        self.close_button.clicked.connect(self.close)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.folder_button)
        bottom.addStretch(1)
        bottom.addWidget(self.close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.guide, 1)
        layout.addLayout(bottom)
        self.update_language()

    def update_language(self) -> None:
        code = language()
        if self._language == code:
            return
        self._language = code
        selected = max(0, self.lesson.currentIndex())
        self.setWindowTitle(tr("Quick warm-up"))
        self.open_button.setText(tr("Open sample"))
        self.folder_button.setText(tr("Open sample folder"))
        self.close_button.setText(tr("Close"))
        self.lesson.clear()
        self.lesson.addItems([tr("Plot, cursor and bit fields"), tr("Processing and derived signals"),
                              tr("DBC, states and timing checks")])
        self.lesson.setCurrentIndex(selected)
        try:
            guide = (WARMUP_DIR / f"guide-{code}.html").read_text(encoding="utf-8")
        except OSError:
            self.guide.setPlainText(tr("The warm-up guide file is missing."))
        else:
            self.guide.setHtml(guide)
        if selected:
            self._show_lesson(selected)

    def _show_lesson(self, index: int) -> None:
        if 0 <= index < len(LESSONS):
            self.guide.scrollToAnchor(LESSONS[index][1])

    def _open_lesson(self) -> None:
        index = self.lesson.currentIndex()
        if 0 <= index < len(LESSONS):
            stem = LESSONS[index][0]
            self._open_recording(WARMUP_DIR / (stem + ".mf4"), WARMUP_DIR / (stem + ".plt"))
