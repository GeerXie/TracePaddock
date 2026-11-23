"""Choose the right reader for a file and open it.

The file content decides first (an MDF file is recognised by its first bytes
whatever its extension), then the extension. A recording in a ZIP archive is
unpacked into a cache folder and opened from there.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from .model import Dataset, DatasetError

__all__ = ["open_dataset", "file_dialog_filters", "is_supported", "list_datasets_in_directory", "PLT_EXTENSIONS"]

MDF_EXTENSIONS = (".mf4", ".mdf", ".mf3", ".dat")
D97_EXTENSIONS = (".d97",)
ARCHIVE_EXTENSIONS = (".zip", ".zi_")
PLT_EXTENSIONS = (".plt",)
SUPPORTED = MDF_EXTENSIONS + D97_EXTENSIONS + ARCHIVE_EXTENSIONS


def file_dialog_filters() -> List[Tuple[str, List[str]]]:
    """Named filters for common recording types and the unrestricted file filter."""
    return [
        ("Recording files", list(MDF_EXTENSIONS + ARCHIVE_EXTENSIONS)),
        ("MF4 files", [".mf4"]),
        ("MDF files", [".mdf", ".mf3", ".dat"]),
        ("ZIP archives", list(ARCHIVE_EXTENSIONS)),
        ("All files", ["*"]),
    ]


def is_supported(path: "str | Path") -> bool:
    return Path(path).suffix.lower() in SUPPORTED


def list_datasets_in_directory(path: "str | Path") -> List[Path]:
    """Supported recordings in the folder of ``path`` (or in ``path`` if it is a folder), sorted by name."""
    folder = Path(path)
    if not folder.is_dir():
        folder = folder.parent
    try:
        return sorted((p for p in folder.iterdir() if p.is_file() and is_supported(p)),
                      key=lambda p: p.name.lower())
    except OSError:
        return []


def _sniff(path: Path) -> str:
    """Identify the recording format from its header, if recognized."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError as exc:
        raise DatasetError(f"{path.name}: cannot open file ({exc.strerror or exc})") from exc
    if head[:3] == b"MDF" and head[:8].rstrip() in (b"MDF", b"MDF"):
        return "mdf"
    if head[:8] == b"UnFinMF ":
        raise DatasetError(f"{path.name}: the MDF file was not finalised by the recorder and cannot be read")
    text = head.lstrip()
    if text[:1] in (b";", b"[") and b"[CONTROL]" in head.upper():
        return "d97"
    return ""


def _open_plain(path: Path, settings: Optional[object]) -> Dataset:
    kind = _sniff(path)
    ext = path.suffix.lower()
    if kind == "mdf" or (not kind and ext in MDF_EXTENSIONS):
        from .mdf import MdfDataset
        strip = bool(settings.get("strip_device_suffix", False)) if settings is not None else False
        return MdfDataset(path, strip_device_suffix=strip)
    if kind == "d97" or ext in D97_EXTENSIONS:
        from .d97 import D97Dataset
        return D97Dataset(path)
    raise DatasetError(f"{path.name}: unknown file type. Choose an MF4 or MDF recording, or a ZIP archive containing a recording.")


def _cache_dir(archive: Path) -> Path:
    st = archive.stat()
    key = hashlib.sha1(f"{archive.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "TracePaddock" / "unzip" / key


def _extract(zf: zipfile.ZipFile, member: zipfile.ZipInfo, folder: Path) -> Path:
    """Unpack one member into ``folder`` using only its base name (no path traversal)."""
    name = Path(member.filename.replace("\\", "/")).name
    target = folder / name
    if target.exists() and target.stat().st_size == member.file_size:
        return target
    free = shutil.disk_usage(folder).free
    if member.file_size > free * 0.95:
        raise DatasetError(
            f"{name}: unpacking needs {member.file_size / 1e9:.1f} GB but only {free / 1e9:.1f} GB "
            f"are free in {folder}")
    tmp = target.with_suffix(target.suffix + ".part")
    with zf.open(member) as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
    os.replace(tmp, target)
    return target


def _open_archive(path: Path, settings: Optional[object]) -> Dataset:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise DatasetError(f"{path.name}: not a readable ZIP archive ({exc})") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if not members:
            raise DatasetError(f"{path.name}: the archive is empty")
        first = members[0]
        if Path(first.filename).suffix.lower() in PLT_EXTENSIONS:
            raise DatasetError(
                f"{path.name}: the first file in the archive is a view (.plt); the recording must come first")
        folder = _cache_dir(path)
        folder.mkdir(parents=True, exist_ok=True)
        data_path = _extract(zf, first, folder)
        plt = next((m for m in members if Path(m.filename).suffix.lower() in PLT_EXTENSIONS), None)
        plt_path = _extract(zf, plt, folder) if plt is not None else None
    dataset = _open_plain(data_path, settings)
    dataset.archive_path = path          # type: ignore[attr-defined]
    dataset.archive_plt = plt_path       # type: ignore[attr-defined]
    return dataset


def open_dataset(path: "str | Path", settings: Optional[object] = None) -> Dataset:
    """Open a recording. Raises :class:`DatasetError` with a message for the user."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"{p}: file not found")
    if p.is_dir():
        raise DatasetError(f"{p}: is a folder, not a recording")
    if p.suffix.lower() in ARCHIVE_EXTENSIONS and zipfile.is_zipfile(p):
        dataset = _open_archive(p, settings)
    else:
        dataset = _open_plain(p, settings)
        dataset.archive_path = None      # type: ignore[attr-defined]
        dataset.archive_plt = None       # type: ignore[attr-defined]
    return dataset
