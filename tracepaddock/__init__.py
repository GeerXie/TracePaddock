"""TracePaddock - viewer for automotive measurement data (MDF, MF4)."""

__version__ = "0.5.2"
#: author shown in the status bar and the About box
AUTHOR = "Geer Xie"
try:                                                     # written by tools/build_portable.py
    from ._build import BUILD  # type: ignore
except ImportError:
    BUILD = "source"
APP_NAME = "TracePaddock"
