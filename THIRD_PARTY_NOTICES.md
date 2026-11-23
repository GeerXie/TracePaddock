# TracePaddock third-party notices

The Windows x64 portable distribution uses the libraries below. Their licenses apply independently of the [TracePaddock application license](LICENSE). The GNU [Lesser General Public License version 3](third-party/LGPL-3.0.txt) and the [General Public License version 3](third-party/GPL-3.0.txt) that it incorporates are included in full.

| Component | Version and license for this distribution | Corresponding source |
| --- | --- | --- |
| Qt runtime | 6.11.3 · LGPL-3.0-only option · The Qt Company Ltd. and contributors | [Qt 6.11.3 source archive](https://download.qt.io/official_releases/qt/6.11/6.11.3/single/qt-everywhere-src-6.11.3.tar.xz) |
| PySide6 Essentials and Shiboken6 | 6.11.3 · LGPL-3.0-only option · The Qt Company Ltd. and contributors | [Qt for Python 6.11.3 source archive, including Shiboken](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.3-src/pyside-setup-everywhere-src-6.11.3.tar.xz) |
| asammdf | 7.4.5 · LGPL-3.0-or-later · Copyright 2017–2023 Daniel Hrisca | [asammdf 7.4.5 source distribution](https://files.pythonhosted.org/packages/3b/2a/dd0e1e65a7b7d196d3114b3b1819c329f5576282cf377b55376572b84455/asammdf-7.4.5.tar.gz) |

The asammdf source archive's SHA-256 is `b0c149daea22c5b2a47ceb2f63099f78adbd68117894473f21ed9dfc2c5217a4`. [PyPI release metadata](https://pypi.org/project/asammdf/7.4.5/#files)

See the accompanying third-party license files for other dependencies and upstream attributions for code included in Qt. [Qt licensing](https://doc.qt.io/qt-6.11/licensing.html) · [Qt third-party attributions](https://doc.qt.io/qt-6.11/licenses-used-in-qt.html) · [Qt for Python third-party attributions](https://doc.qt.io/qtforpython-6.11/licenses.html)

## Running with your own build of an LGPL library

You may modify and replace the LGPL libraries and reverse-engineer the combined work to debug modifications to those libraries, as provided by the LGPL. TracePaddock loads matching shared libraries and Python modules from separate files in the portable folder.

1. Close TracePaddock and make a copy of the extracted portable folder. Use the Windows x64 and CPython 3.11 ABI of this release and keep the public interfaces used by the application compatible.
2. For Qt and Qt for Python, build replacements using the matching sources above. Copy the corresponding DLLs, platform or feature plugins, and Python binding extensions to their existing locations under `_internal/PySide6` or `_internal/shiboken6`, preserving filenames and the folder layout. Keep Qt runtime, plugins, PySide and Shiboken compatible with one another. [Qt for Python build instructions](https://doc.qt.io/qtforpython-6.11/building_from_source/index.html) · [Qt for Windows deployment paths](https://doc.qt.io/qt-6.11/windows-deployment.html)
3. For asammdf, copy the compatible package modules and compiled extensions to their corresponding paths under `_internal/asammdf`. To match the portable layout for an edited Python module, use pristine CPython 3.11 with `python -m compileall -b path/to/module.py`, then replace the corresponding on-disk `.pyc` with the generated file. Rebuild any changed native extension for CPython 3.11 on Windows x64 and keep its module filename.
4. Restart `TracePaddock.exe` from the copied folder. Run `TracePaddock.exe --self-test <output-folder>` to exercise the main workflows with the replacement files.

## Source availability for redistribution

The links above identify the exact upstream source versions for this portable distribution. When offering this binary as a download, keep these directions next to it and ensure equivalent no-charge access to the corresponding source. If the upstream files become unavailable, provide the matching source from another accessible location. Include applicable source changes and build instructions when redistributing modified library builds. [GNU GPLv3, section 6(d)](third-party/GPL-3.0.txt)

## 简体中文

便携版使用 Qt、PySide6、Shiboken6 和 asammdf。每个组件的准确版本、许可证及源码下载地址见上表。这些组件继续适用各自许可证。LGPL 允许修改或替换库，也允许为调试库的修改对组合程序进行逆向分析。可复制解压目录，用匹配的 Windows x64 和 CPython 3.11 架构构建替换文件，按原目录结构放进 `_internal`，再从副本启动程序。对外提供便携包下载时，应在下载入口旁保留源码说明，并确保同版本源码可免费获取。
