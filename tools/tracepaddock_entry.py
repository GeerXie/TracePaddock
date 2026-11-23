"""Entry point of the portable program (PyInstaller): TracePaddock.exe [recording] [view.plt] | --self-test | --rule-job."""
import os
import sys


def _streams() -> None:
    """A program without console has no stdout / stderr: text goes to a log file next to the settings."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        try:
            from tracepaddock.settings import data_folder
            folder = data_folder() / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            sys.stderr = open(folder / "tracepaddock.log", "a", encoding="utf-8", buffering=1)
        except Exception:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _crash_log() -> None:
    """A native crash (access violation in a library) writes the Python stack of every thread to
    TracePaddockData\\logs\\crash.log instead of ending silently."""
    try:
        import faulthandler
        from tracepaddock.settings import data_folder
        folder = data_folder() / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        log = open(folder / "crash.log", "a", encoding="utf-8", buffering=1)
        log.write("\n--- start %s pid %d args %s\n" % (__import__("time").strftime("%Y-%m-%d %H:%M:%S"), os.getpid(),
                                                          sys.argv[1:]))
        faulthandler.enable(file=log, all_threads=True)
        globals()["_CRASH_LOG"] = log                     # keep the file open for the whole run
    except Exception:
        pass


if __name__ == "__main__":
    _streams()
    _crash_log()
    from tracepaddock.__main__ import main
    sys.exit(main())
