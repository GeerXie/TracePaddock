"""Self test of an installation:  TracePaddock.exe --self-test [output folder]

Runs without a window: opens the bundled sample recording with its view, checks values, evaluates a rule
in the program and a rule script in the rule process (started the way the installed program starts it),
finds the built-in plug-ins, exports CSV and MF4 and reads the MF4 back, writes the Excel report and a
picture. The results and timings go to ``selftest_report.txt`` in the output folder (default: a new
folder in the temporary directory). Exit code 0 when every check passed.
"""
from __future__ import annotations

import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional, Tuple

__all__ = ["run", "sample_folder", "WATCHDOG"]

#: seconds after which a hanging self test writes the stacks of all threads and stops (exit code 1)
WATCHDOG = 180


def sample_folder() -> Path:
    return Path(__file__).resolve().parent / "samples" / "warmup"


SCRIPT = '''rule("Request acknowledgement", trigger=becomes(S("TutorialRequest") == 1),
     expect=S("TutorialAck") == 1, within=0.5)
'''


def run(out: Optional[str] = None) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    folder = Path(out) if out else Path(tempfile.mkdtemp(prefix="tracepaddock_selftest_"))
    folder.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[str, bool, str, float]] = []
    # never hang: after WATCHDOG seconds write where every thread is and stop
    import faulthandler
    hang = open(folder / "selftest_hang_traceback.txt", "w", encoding="utf-8")
    faulthandler.dump_traceback_later(WATCHDOG, exit=True, file=hang)
    progress = folder / "selftest_progress.txt"
    progress.write_text("", encoding="utf-8")

    def check(name: str, fn: Callable[[], str]) -> bool:
        with open(progress, "a", encoding="utf-8") as f:              # the last line names a hanging step
            f.write("%s ...\n" % name)
        started = time.perf_counter()
        try:
            detail, ok = str(fn() or "ok"), True
        except Exception as exc:                                    # every failure is reported, none stops
            detail, ok = "%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc(limit=4)), False
        results.append((name, ok, detail, time.perf_counter() - started))
        return ok

    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["TracePaddock"])
    from . import BUILD, __version__
    from .core.export import write_csv, write_mdf
    from .core.registry import open_dataset
    from .rules.engine import SignalSource, evaluate
    from .rules.model import Condition, Rule
    from .rules.plugins import discover
    from .rules.report import RunResult, write_xlsx
    from .settings import Settings
    from .ui.mainwindow import MainWindow

    settings = Settings(folder / "settings.json", autoload=False)
    win = MainWindow(settings)
    win.resize(1400, 900)
    errors: List[str] = []
    win._error = lambda text: errors.append(str(text))          # no message boxes: the text goes to the report
    sample = sample_folder()
    state = {}

    def open_sample(stem: str, expected: int) -> str:
        source = sample / (stem + ".mf4")
        view = sample / (stem + ".plt")
        if not win.open_dataset(source, view):
            try:
                open_dataset(source).close()
            except Exception:
                raise RuntimeError("the sample recording could not be opened: %s\n%s"
                                   % ("; ".join(errors), traceback.format_exc()))
            raise RuntimeError("the sample recording could not be opened: %s" % "; ".join(errors))
        shown = [s for s in win.signals if s.has_data]
        if len(shown) != expected:
            raise RuntimeError("expected %d signals, got %d" % (expected, len(shown)))
        return "%d signals" % len(shown)

    def values() -> str:
        speed = next(s for s in win.signals if s.name == "TutorialSpeed")
        value = speed.value_at(4.0)
        if value is None or not 25.0 < value < 65.0:
            raise RuntimeError("TutorialSpeed at 4 s: %r" % value)
        return "TutorialSpeed(4 s) = %.3f km/h" % value

    def warmup() -> str:
        from .ui.warmup import WarmupDialog

        if settings.get("theme") != "obsidian":
            raise RuntimeError("The default theme is not Obsidian")
        guide = WarmupDialog(win.open_dataset, win)
        if guide.lesson.count() != 3 or "TutorialSpeed" not in guide.guide.toPlainText():
            raise RuntimeError("The three warm-up lessons could not be loaded")
        guide.lesson.setCurrentIndex(1)
        guide.open_button.click()
        if win.dataset is None or win.dataset.path.name != "02-processing.mf4":
            raise RuntimeError("The guide could not open its processing exercise")
        return "Obsidian, three lessons, processing recording opens from the guide"

    def rule_in_program() -> str:
        rule = Rule("Request acknowledgement", trigger=Condition("TutorialRequest", "==", 1.0),
                    expect=Condition("TutorialAck", "==", 1.0), within_max=0.5)
        result = evaluate(rule, SignalSource(win.signals, win.dataset))
        verdicts = [event.verdict for event in result.events]
        if result.error or verdicts != ["PASS", "FAIL"]:
            raise RuntimeError("%s, %s" % (result.error, verdicts))
        state["events"] = result.events
        return "1 pass, 1 fail"

    def rule_process() -> str:
        result = win.run_script(SCRIPT, "self test")
        if result.get("error"):
            raise RuntimeError(result["error"])
        verdicts = [event.get("verdict") for rule in result.get("results", []) for event in rule.get("events", [])]
        if verdicts != ["PASS", "FAIL"]:
            raise RuntimeError("%s" % verdicts)
        return "1 pass, 1 fail from the rule process"

    def plugins() -> str:
        found = [p.name for p in discover()]
        if len(found) < 2:
            raise RuntimeError("plug-ins found: %s" % found)
        return ", ".join(found)

    def export() -> str:
        signals = [s for s in win.signals if s.has_data]
        csv = folder / "selftest_export.csv"
        write_csv(csv, signals, 0.0, 8.0, step=0.05)
        mf4 = folder / "selftest_export.mf4"
        write_mdf(mf4, signals, 0.0, 10.0)
        back = open_dataset(mf4)
        try:
            names = {info.name for info in back.signals}
        finally:
            back.close()
        missing = {s.name for s in signals} - names
        if csv.stat().st_size < 1000 or missing:
            raise RuntimeError("csv %d bytes, missing in MF4: %s" % (csv.stat().st_size, sorted(missing)))
        return "csv %d bytes, mf4 %d bytes, %d signals read back" % (csv.stat().st_size, mf4.stat().st_size, len(names))

    def report() -> str:
        path = folder / "selftest_report.xlsx"
        write_xlsx(path, [RunResult("Request acknowledgement", "form", state.get("events", []),
                                    None, "", "", None, [])], {"test": "self"})
        return "%d bytes" % path.stat().st_size

    def picture() -> str:
        win.show()
        app.processEvents()
        path = folder / "selftest_picture.png"
        if not win.grab().save(str(path)):
            raise RuntimeError("picture not written")
        return "%d bytes" % path.stat().st_size

    def modules() -> str:
        """Every binary module the program needs loads (a missing DLL shows up here by name)."""
        import importlib
        names = ("bz2", "lzma", "zlib", "ssl", "ctypes", "xml.parsers.expat", "hashlib", "numpy", "scipy.signal",
                 "scipy.io", "pandas", "lxml.etree", "lz4.frame", "isal.isal_zlib", "numexpr", "asammdf",
                 "canmatrix.formats.dbc", "openpyxl", "PySide6.QtSvg", "PySide6.QtPrintSupport")
        failed = []
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as exc:                            # report all, not only the first
                failed.append("%s (%s: %s)" % (name, type(exc).__name__, exc))
        if failed:
            raise RuntimeError("cannot load: " + "; ".join(failed))
        return "%d modules" % len(names)

    for name, fn in (("binary modules", modules), ("open plot recording", lambda: open_sample("01-plot", 6)),
                     ("values", values), ("quick warm-up and default theme", warmup),
                     ("open timing recording", lambda: open_sample("03-timing", 5)),
                     ("rule in the program", rule_in_program), ("rule script in the rule process", rule_process),
                     ("built-in plug-ins", plugins), ("export CSV and MF4, read back", export),
                     ("Excel report", report), ("picture", picture)):
        check(name, fn)
    win.script.modified = False
    win.close()

    from PySide6 import __version__ as binding_version
    ok = all(r[1] for r in results)
    lines = ["TracePaddock %s (build %s) self test: %s" % (__version__, BUILD, "PASSED" if ok else "FAILED"),
             "program: %s%s" % (sys.executable, " (portable)" if getattr(sys, "frozen", False) else ""),
             "Python %s, Qt %s, PySide %s, %s" % (platform.python_version(), QtCore.qVersion(), binding_version,
                                                  platform.platform()), ""]
    for name, passed, detail, seconds in results:
        lines.append("[%s] %-34s %7.2f s  %s" % ("OK  " if passed else "FAIL", name, seconds, detail.splitlines()[0]))
        if not passed:
            lines += ["       " + line for line in detail.splitlines()[1:]]
    (folder / "selftest_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    faulthandler.cancel_dump_traceback_later()
    hang.close()
    try:
        (folder / "selftest_hang_traceback.txt").unlink()
    except OSError:
        pass
    try:
        print("\n".join(lines))
    except Exception:
        pass
    del QtCore
    return 0 if ok else 1
