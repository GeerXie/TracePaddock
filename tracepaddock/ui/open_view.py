"""Open a recording and a view from the same dialog."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from ..core.registry import file_dialog_filters
from ..i18n import tr


def recording_file_filters() -> str:
    return ";;".join(
        "%s (%s)" % (tr(label), " ".join("*" + e if e != "*" else "*" for e in extensions))
        for label, extensions in file_dialog_filters()
    )


class OpenViewDialog(QtWidgets.QDialog):
    def __init__(self, settings, recording: str, view: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Open recording and view"))
        self.settings = settings
        self.defaults = {}
        layout = QtWidgets.QGridLayout(self)
        view_filters = tr("PLT files") + " (*.plt);;" + tr("All files") + " (*)"
        self.recording = self._file_row(layout, 0, tr("Recording"), "dataset", recording,
                                        "recent_datasets", recording_file_filters())
        self.view = self._file_row(layout, 1, tr("View (.plt)"), "plt", view, "recent_plts", view_filters)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Open).setText(tr("Open"))
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 2, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        self.resize(760, self.sizeHint().height())

    def _file_row(self, layout, row: int, label: str, kind: str, current: str, recent: str, filters: str):
        def existing_file(path: str) -> bool:
            return bool(path) and Path(path).is_file()

        default_file = self.settings.get("default_" + kind, "")
        candidates = [default_file, *self.settings.text_list(recent)]
        edit = QtWidgets.QLineEdit(current or next((path for path in candidates if existing_file(path)), ""))
        directory_key = "default_" + kind + "_dir"
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)

        def browse_start() -> str:
            return next((str(path) for path in (self.defaults.get(directory_key, self.settings.get(directory_key, "")),
                                               self.settings.get("last_" + kind + "_dir", ""), Path.home())
                         if path and Path(path).is_dir()), "")

        def browse() -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, label, browse_start(), filters)
            if path:
                edit.setText(path)

        def default_directory() -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, tr("Set default browse folder"), browse_start())
            if path:
                self.defaults[directory_key] = path

        button = QtWidgets.QPushButton(tr("Browse..."))
        button.clicked.connect(browse)
        layout.addWidget(button, row, 2)
        options = QtWidgets.QToolButton()
        options.setText("…")
        options.setToolTip(tr("File and folder defaults"))
        options.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(options)
        use_file = menu.addAction(tr("Set as default file"))
        use_file.triggered.connect(lambda: self.defaults.update({"default_" + kind: edit.text().strip()}))
        menu.aboutToShow.connect(lambda: use_file.setEnabled(existing_file(edit.text().strip())))
        menu.addAction(tr("Set default browse folder"), default_directory)
        menu.addAction(tr("Clear defaults"), lambda: self.defaults.update({"default_" + kind: "", directory_key: ""}))
        options.setMenu(menu)
        layout.addWidget(options, row, 3)
        return edit

    def paths(self) -> tuple[str, str]:
        return self.recording.text().strip(), self.view.text().strip()
