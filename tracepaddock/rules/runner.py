"""Run a rule script or plug-in in a separate process.

User code never runs inside the window process: a script with an endless
loop or a crash cannot freeze or kill the viewer. The window writes a job
file, starts ``python -m tracepaddock.rules.runner <job.json>`` and waits with a
time limit; the runner opens the recording itself, builds the signals of
the current script (computed signals and value mappings included), runs the
user code, evaluates the rules it made and writes the result file.

The child process never outlives the viewer: it is ended at the time limit,
by :func:`stop_all_jobs` (the viewer closes) and, on Windows, by a job object
that ends it together with the viewer process, also when the viewer crashes.

Job keys: dataset, plt_text (optional), valuemaps (path, optional), settings
(processing options), script + script_name, or plugin + params, range
(optional [t0, t1]), output (path of the result JSON).
"""
from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

from ..i18n import tr

__all__ = ["run_job", "main", "JobError", "stop_all_jobs", "SCRIPT_MODULES"]

ROOT = Path(__file__).resolve().parents[2]

#: Modules a script or plug-in may import; the portable program contains them too (tools/build_portable.py).
SCRIPT_MODULES = ("numpy", "math", "cmath", "statistics", "bisect", "heapq", "collections", "itertools",
                  "functools", "operator", "re", "string", "textwrap", "json", "csv", "datetime", "time",
                  "calendar", "decimal", "fractions", "random", "struct", "copy", "dataclasses", "enum", "typing")


class JobError(Exception):
    """The script could not be run; the message is shown to the user."""


def _user_traceback(exc: BaseException, filename: str, code: str = "") -> str:
    """Only the frames of the user's code, so the message points at the user's line."""
    lines = code.splitlines()

    def text(lineno: Optional[int], given: Optional[str]) -> str:
        if given:
            return given.strip()
        return lines[lineno - 1].strip() if lineno and 0 < lineno <= len(lines) else ""

    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == filename]
    where = "".join("  line %d: %s\n" % (f.lineno, text(f.lineno, f.line)) for f in frames[-3:])
    if isinstance(exc, SyntaxError):
        where = "  line %s: %s\n" % (exc.lineno, text(exc.lineno, exc.text))
        return "%sSyntaxError: %s" % (where, exc.msg)
    return "%s%s: %s" % (where, type(exc).__name__, exc)


def _execute(job: dict) -> dict:
    from ..core.registry import open_dataset
    from ..plt.parser import parse_plt
    from ..plt.process import build_display_signals
    from ..valuemap import ValueMapLibrary
    from .engine import SignalSource, evaluate_all
    from .plugins import resolve_plugin
    from .script import ScriptApi

    from ..i18n import set_language
    set_language(str(job.get("language") or "en"))       # notes and messages of the engine
    started = time.perf_counter()
    dataset = open_dataset(job["dataset"])
    try:
        signals = []
        if job.get("plt_text"):
            maps = ValueMapLibrary(job["valuemaps"]) if job.get("valuemaps") else None
            signals = build_display_signals(dataset, parse_plt(job["plt_text"]), job.get("settings") or {},
                                            valuemaps=maps)
        source = SignalSource(signals, dataset)

        def reader(name: str):
            step = source.get(name)
            return None if step is None else (step.t, step.y)

        if job.get("plugin"):
            path = resolve_plugin(job["plugin"])
            if not path.is_file():
                raise JobError(tr("the plug-in file %s was not found") % path)
            filename = str(path)
            code = importlib.util.decode_source(path.read_bytes())   # honours a BOM and a coding line
            name = job.get("script_name") or path.stem
        else:
            filename = "<script %s>" % (job.get("script_name") or "script")
            code = str(job.get("script", ""))
            name = job.get("script_name") or "script"
        api = ScriptApi(reader, name, job.get("params") or {})
        namespace = api.namespace()
        namespace["__name__"] = "__tracepaddock_script__"
        try:
            exec(compile(code, filename, "exec"), namespace)
            if job.get("plugin"):
                run = namespace.get("run")
                if not callable(run):
                    raise JobError(tr("the plug-in has no function run(api, params)"))
                run(api, api.params)
        except JobError:
            raise
        except BaseException as exc:                       # the user's mistake: report it, do not crash
            return {"error": _user_traceback(exc, filename, code), "log": api.messages}
        rng = job.get("range")
        results = evaluate_all(api.rules, source, tuple(rng) if rng else None)
        return {"error": None, "log": api.messages,
                "rules": [r.to_json() for r in api.rules],
                "results": [{"rule": res.rule.name, "error": res.error, "response_period": res.response_period,
                             "events": [e.to_json() for e in res.events]} for res in results],
                "events": [e.to_json() for e in api.events],
                "seconds": time.perf_counter() - started}
    finally:
        dataset.close()


def main(argv: Optional[list] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    try:
        out = _execute(job)
    except JobError as exc:
        out = {"error": str(exc), "log": []}
    except Exception as exc:                               # problems with the data, not the script
        out = {"error": "%s: %s" % (type(exc).__name__, exc), "log": []}
    Path(job["output"]).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------- child processes of the viewer
_lock = threading.Lock()
#: running child processes -> True when they were stopped on purpose
_running: Dict[subprocess.Popen, bool] = {}
_kernel32: Optional[ctypes.WinDLL] = None
_job_object: Optional[int] = None


def _create_job_object() -> int:
    """A Windows job object that ends its processes when its last handle closes, i.e. when this process ends."""
    global _kernel32
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return 0
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x2000                  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(handle)
        return 0
    _kernel32 = kernel32
    return handle                                                   # never closed: it lives as long as we do


def _end_with_this_process(proc: subprocess.Popen) -> None:
    """Windows: put the child into the job object, so it ends with the viewer, also when the viewer crashes."""
    global _job_object
    if os.name != "nt":
        return
    try:
        with _lock:
            if _job_object is None:
                _job_object = _create_job_object()
        kernel32 = _kernel32
        handle = getattr(proc, "_handle", None)
        if _job_object and kernel32 is not None and handle is not None:
            kernel32.AssignProcessToJobObject(_job_object, int(handle))
    except Exception:                                               # best effort: the time limit still applies
        pass


def stop_all_jobs() -> int:
    """End every running script process at once (the viewer closes). Returns how many were running."""
    with _lock:
        procs = list(_running)
        for proc in procs:
            _running[proc] = True
    for proc in procs:
        try:
            proc.kill()
        except OSError:
            pass
    return len(procs)


def run_job(job: Dict[str, object], timeout: float = 120.0) -> dict:
    """Run ``job`` in a child process. Raises :class:`JobError` (time limit, crash, stopped)."""
    folder = Path(tempfile.mkdtemp(prefix="tracepaddock_rules_"))
    try:
        job = dict(job)
        job["output"] = str(folder / "result.json")
        job_path = folder / "job.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        flags = 0x08000000 if os.name == "nt" else 0              # CREATE_NO_WINDOW
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("QT_QPA_PLATFORM", None)
        if getattr(sys, "frozen", False):                       # the portable program runs the job itself
            command, cwd = [sys.executable, "--rule-job", str(job_path)], str(Path(sys.executable).parent)
        else:
            command, cwd = [sys.executable, "-m", "tracepaddock.rules.runner", str(job_path)], str(ROOT)
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=flags, env=env)
        _end_with_this_process(proc)
        with _lock:
            _running[proc] = False
        try:
            try:
                _out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                proc.communicate()
                raise JobError(tr("the script did not finish within %g s and was stopped") % timeout) from exc
        finally:
            with _lock:
                stopped = _running.pop(proc, False)
        if stopped:
            raise JobError(tr("the script was stopped"))
        try:
            return json.loads(Path(job["output"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            tail = err.decode("utf-8", "replace").strip().splitlines()[-6:]
            raise JobError(tr("the script process ended without a result (exit code %s)") % proc.returncode
                           + "\n" + "\n".join(tail))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
