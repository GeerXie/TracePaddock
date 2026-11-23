"""Build and verify the portable Windows release in a clean CPython 3.11 environment.

Run ``python -m pip install ".[build]"``, then ``python tools/build_portable.py``.
The executable, notices and verified ZIP are written to ``dist``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WORK = ROOT / "build"

EXCLUDES = ["tkinter", "matplotlib", "IPython", "pytest", "_pytest", "asammdf.gui.widgets", "asammdf.gui.dialogs",
            "asammdf.gui.ui", "asammdf.gui.asammdfgui", "pyqtgraph", "PyQt5", "PyQt6", "PySide2",
            "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtDesigner",
            "PySide6.QtHelp", "PySide6.QtSql", "PySide6.QtTest"]

WARMUP_FILES = (
    "01-plot.mf4", "01-plot.plt", "02-processing.mf4", "02-processing.plt",
    "03-timing.mf4", "03-timing.plt", "guide-en.html", "guide-zh.html",
    "warmup-signals.dbc", "warmup-mapping.json",
)

README = """TracePaddock {version}（免安装版，构建 {build}）

使用方法
  1. 把整个文件夹解压到任意位置（例如 D:\\Tools\\TracePaddock）。不需要安装，也不需要 Python。
  2. 双击 TracePaddock.exe 启动；也可以把数据文件拖到窗口里，或用命令行：
       TracePaddock.exe <数据文件> [<PLT 文件>]
  3. 从「帮助 → 快速上手」打开练习录制和配套视图。
  4. 设置、数值映射、规则和自己的插件保存在本文件夹的 TracePaddockData 子文件夹里，
     复制整个文件夹即可带走全部设置；如果文件夹不可写（例如放在 Program Files 下），
     则保存在 %APPDATA%\\TracePaddock。

自检
  TracePaddock.exe --self-test [输出文件夹]
  不显示窗口，打开自带的示例数据，检查数值、规则、插件、导出（CSV / MF4）、Excel 报告和截图，
  结果写入 selftest_report.txt；全部通过时返回码为 0。

制作人：Geer Xie
使用与分发条款见 LICENSE。第三方组件的许可与源码见 THIRD_PARTY_NOTICES.md。
"""


def app_icon(path: Path) -> None:
    """Embed the icon used by the main window in the Windows executable."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from tracepaddock.ui.icons import application_icon

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["build"])
    image = application_icon().pixmap(256, 256).toImage()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image                                     # several sizes in one .ico
    except ImportError:
        if not image.save(str(path), "ICO"):
            raise RuntimeError("Could not write the TracePaddock icon")
    else:
        png = path.with_suffix(".png")
        image.save(str(png))
        Image.open(png).save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    del app


def folder_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def pyinstaller_args(icon: Path) -> List[str]:
    """The PyInstaller command line of the portable program."""
    from tracepaddock.rules.runner import SCRIPT_MODULES
    args = ["--noconfirm", "--windowed", "--debug", "noarchive", "--noupx",
            "--name", "TracePaddock", "--icon", str(icon),
            "--distpath", str(DIST), "--workpath", str(WORK / "pyinstaller"), "--specpath", str(WORK),
            "--paths", str(ROOT),
            # modules that are loaded by name at run time
            "--collect-submodules", "canmatrix.formats", "--hidden-import", "isal.igzip_lib",
            "--hidden-import", "isal._isal", "--hidden-import", "isal.isal_zlib", "--hidden-import", "asammdf.gui.plot"]
    for name in WARMUP_FILES:
        args += ["--add-data", f"{ROOT / 'tracepaddock/samples/warmup' / name}:tracepaddock/samples/warmup"]
    for name in ("response_timing.py", "state_transitions.py"):
        args += ["--add-data", f"{ROOT / 'tracepaddock/plugins' / name}:tracepaddock/plugins"]
    for module in SCRIPT_MODULES:                    # rule scripts and plug-ins may import these
        args += ["--hidden-import", module]
    for module in EXCLUDES:
        args += ["--exclude-module", module]
    args.append(str(ROOT / "tools" / "tracepaddock_entry.py"))
    return args


def check_script_modules(command: Sequence[str], dataset: Path, timeout: float = 300.0) -> Optional[str]:
    """Run a rule job that imports every module of SCRIPT_MODULES. ``command`` starts a job file, e.g.
    ``[TracePaddock.exe, "--rule-job"]``. Returns ``None`` when all imports work, else what went wrong."""
    from tracepaddock.rules.runner import SCRIPT_MODULES
    folder = Path(tempfile.mkdtemp(prefix="tracepaddock_build_check_"))
    try:
        job = {"dataset": str(dataset), "script_name": "imports", "output": str(folder / "result.json"),
               "script": "".join("import %s\n" % name for name in SCRIPT_MODULES) + "log('ok')\n"}
        (folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
        subprocess.run(list(command) + [str(folder / "job.json")], cwd=str(ROOT), timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        return None if result.get("error") is None and result.get("log") == ["ok"] else str(result.get("error"))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def copy_notices(out: Path) -> None:
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff"):
        shutil.copyfile(ROOT / name, out / name)
    notices = out / "third-party"
    shutil.copytree(ROOT / "third-party", notices)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    shutil.copyfile(python_license, notices / "Python-LICENSE.txt")
    pending = ["numpy", "scipy", "PySide6-Essentials", "asammdf", "canmatrix", "openpyxl", "PyInstaller", "Pillow"]
    versions = {}
    while pending:
        dist = metadata.distribution(pending.pop())
        name = dist.metadata["Name"]
        if name in versions:
            continue
        versions[name] = dist.version
        for req in map(Requirement, dist.requires or []):
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                pending.append(req.name)
        for file in dist.files or []:
            if ".dist-info" in file.parts[0] and (
                    "licenses" in file.parts or file.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))):
                destination = notices / name / Path(*file.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dist.locate_file(file), destination)
    (notices / "dependency-versions.txt").write_text(
        "\n".join(f"{name}=={versions[name]}" for name in sorted(versions, key=str.casefold)) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Name of a new folder inside dist")
    options = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import tracepaddock
    version = tracepaddock.__version__
    name = options.out or f"TracePaddock-{version}"
    if Path(name).name != name or name in {".", ".."}:
        parser.error("--out must be a folder name, without a path")
    out = DIST / name
    archive = DIST / f"{name}-win64.zip"
    if out.exists() or archive.exists():
        parser.error("The output already exists. Choose a new --out name.")
    WORK.mkdir(exist_ok=True)
    DIST.mkdir(exist_ok=True)
    build = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    stamp = ROOT / "tracepaddock" / "_build.py"
    stamp.write_text('BUILD = "%s"\n' % build, encoding="utf-8")
    icon = WORK / "tracepaddock.ico"
    app_icon(icon)
    try:
        import PyInstaller.__main__
        with tempfile.TemporaryDirectory(prefix="stage-", dir=WORK) as directory:
            stage = Path(directory)
            args = pyinstaller_args(icon)
            args[args.index("--distpath") + 1] = str(stage)
            PyInstaller.__main__.run(args)
            release = stage / "TracePaddock"
            copy_notices(release)
            (release / "使用说明.txt").write_text(
                README.format(version=version, build=build), encoding="utf-8-sig")

            executable = release / "TracePaddock.exe"
            problem = check_script_modules([str(executable), "--rule-job"],
                                           release / "_internal/tracepaddock/samples/warmup/03-timing.mf4")
            if problem is not None:
                raise RuntimeError("Bundled rule modules: " + problem)
            checks = stage / "self-test"
            windows = Path(os.environ.get("SystemRoot", "C:/Windows"))
            env = dict(os.environ, PATH=os.pathsep.join(map(str, (windows / "System32", windows))))
            result = subprocess.run([str(executable), "--self-test", str(checks)],
                                    cwd=release, env=env, timeout=240, check=False)
            report = (checks / "selftest_report.txt").read_text(encoding="utf-8")
            (WORK / "release-selftest.txt").write_text(report, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(report)
            shutil.rmtree(release / "TracePaddockData", ignore_errors=True)
            with zipfile.ZipFile(stage / archive.name, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for path in sorted(release.rglob("*")):
                    if path.is_file():
                        zf.write(path, Path("TracePaddock") / path.relative_to(release))
            release.rename(out)
            (stage / archive.name).rename(archive)
    finally:
        stamp.unlink(missing_ok=True)
    size, zipped = folder_size(out), archive.stat().st_size
    print("folder %s: %.1f MB in %d files" % (out, size / 1e6, sum(1 for p in out.rglob('*') if p.is_file())))
    print("zip    %s: %.1f MB" % (archive, zipped / 1e6))
    print(f"Portable self-test passed. Report: {WORK / 'release-selftest.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
