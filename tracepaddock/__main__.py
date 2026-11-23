"""Start the viewer:  python -m tracepaddock [recording] [script.plt]

Named options can set paths and a time range::

    python -m tracepaddock IFN=<recording> PLT=<script> T0=<start> T1=<end>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def log_file() -> Path:
    """The log of the viewer: TracePaddockData\\logs\\tracepaddock.log (portable) or %APPDATA%\\TracePaddock\\logs."""
    from .settings import data_folder
    return data_folder() / "logs" / "tracepaddock.log"


def install_excepthook() -> None:
    """For the window only: an error that escapes a Qt slot or a thread is written to the log file and shown
    in a short message, instead of ending the program (PySide6 ends it when no hook is installed)."""
    import threading
    import time
    import traceback

    from PySide6 import QtCore, QtWidgets

    from .i18n import tr

    class Relay(QtCore.QObject):
        message = QtCore.Signal(str)

    showing: List[bool] = []
    last = {"text": "", "time": 0.0}

    def show(text: str) -> None:
        if showing:                                    # one message at a time (an error in painting repeats)
            return
        showing.append(True)
        try:
            QtWidgets.QMessageBox.critical(QtWidgets.QApplication.activeWindow(), "TracePaddock", text)
        finally:
            showing.clear()

    relay = Relay()
    relay.message.connect(show, QtCore.Qt.QueuedConnection)   # shown by the window thread, after the event

    def hook(kind, value, tb) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        details = "".join(traceback.format_exception(kind, value, tb))
        now = time.monotonic()
        if details == last["text"] and now - last["time"] < 2.0:
            return                                     # the same error again at once (e.g. every repaint)
        last.update(text=details, time=now)
        path = log_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n--- %s unexpected error (%s)\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                 threading.current_thread().name, details))
        except OSError:
            pass
        stream = sys.stderr                            # also on the console, unless that is the log itself
        if stream is not None and os.path.normcase(os.path.abspath(str(getattr(stream, "name", "")))) \
                != os.path.normcase(os.path.abspath(str(path))):
            try:
                stream.write(details)
            except Exception:
                pass
        relay.message.emit(tr("An unexpected error occurred: %s\nTracePaddock keeps running. The details were "
                              "written to %s.") % ("%s: %s" % (kind.__name__, value), path))

    hook.relay = relay                                 # keep the relay alive as long as the hook
    sys.excepthook = hook


def show_load_problems(parent=None) -> None:
    """Tell the user which settings, value-mapping or rule files could not be read, and where they were kept."""
    from PySide6 import QtWidgets

    from .i18n import tr
    from .settings import unreadable_files

    lines = []
    for item in unreadable_files():
        reason = tr(item.reason)
        if item.skipped and item.backup is not None:
            lines.append(tr("%s: %d entries could not be read (%s).\nA copy of the file was kept as %s.")
                         % (item.path, item.skipped, reason, item.backup))
        elif item.backup is not None:
            lines.append(tr("%s could not be read (%s).\nThe file was kept as %s; TracePaddock continues without "
                            "its content.") % (item.path, reason, item.backup))
        else:
            lines.append(tr("%s could not be read (%s).\nTracePaddock does not write over it in this session.")
                         % (item.path, reason))
    if lines:
        QtWidgets.QMessageBox.warning(parent, "TracePaddock", "\n\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--rule-job"] and len(args) > 1:          # child process of a rule script (portable program)
        from .rules.runner import main as run_rule_job
        return run_rule_job(args[1:2])
    if args[:1] == ["--self-test"]:
        from .selftest import run as self_test
        return self_test(args[1] if len(args) > 1 else None)
    if os.name == "nt":
        import ctypes

        # Windows gives the app its own taskbar button, including when it runs from Python.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GeerXie.TracePaddock")
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("TracePaddock")
    install_excepthook()
    # Qt's default dialog font is a bitmap-like fallback on Chinese Windows; use the system UI font.
    from PySide6 import QtGui

    families = set(QtGui.QFontDatabase().families())
    for family in ("Microsoft YaHei UI", "Segoe UI"):
        if family in families:
            app.setFont(QtGui.QFont(family, 9))
            break

    from .i18n import set_language
    from .settings import Settings
    from .ui.mainwindow import MainWindow

    settings = Settings()
    set_language(settings.get("language", "en"))
    window = MainWindow(settings)

    recording = script = None
    t0 = t1 = None
    for arg in args:
        upper = arg.upper()
        if upper.startswith("IFN="):
            recording = arg[4:]
        elif upper.startswith("PLT="):
            script = arg[4:]
        elif upper.startswith("T0="):
            t0 = arg[3:]
        elif upper.startswith("T1="):
            t1 = arg[3:]
        elif arg.startswith(("/", "-")) and not Path(arg).exists():
            continue                                  # ignore unrecognised switches
        elif arg.lower().endswith(".plt"):
            script = arg
        else:
            recording = arg
    window.show()
    QtCore.QTimer.singleShot(0, lambda: show_load_problems(window))
    if recording:
        if window.open_dataset(recording, script):
            try:
                if t0 is not None and t1 is not None:
                    window.view.set_x_range(float(t0), float(t1))
            except ValueError:
                pass
    elif script:
        window.open_plt(script)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
