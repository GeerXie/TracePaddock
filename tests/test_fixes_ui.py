"""Regression tests for the review of themes, pictures, legend and menus: switching themes keeps the user's
own look, theme colours that suit the plot background (on the screen and on white pictures), the chip legend,
background pictures, memory of rendered pictures, menus filled when they open, the menu-bar decoration file
and the red marks of the signal editor."""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from tracepaddock.core.model import Dataset, SignalData, SignalInfo
from tracepaddock.display import DisplaySignal
from tracepaddock.i18n import tr
from tracepaddock.plt.model import PltEntry
from tracepaddock.plt.parser import parse_plt
from tracepaddock.settings import Settings
from tracepaddock.ui import cute_art, themes
from tracepaddock.ui.editor import PltEditorDialog
from tracepaddock.ui.mainwindow import MainWindow
from tracepaddock.ui.plotview import PlotView
from tracepaddock.ui.render import (BACKGROUNDS, compute_layout, legend_font, legend_value_text, paint_legend,
                                paint_view, signal_color)
from tracepaddock.ui.themes import THEME_KEYS, THEMES, apply_app_theme

WHITE = (255, 255, 255)


# ---------------------------------------------------------------------- helpers
def _luminance(rgb) -> float:
    out = []
    for c in rgb:
        c = c / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def _contrast(a, b) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _rgb(colour: QtGui.QColor):
    return colour.red(), colour.green(), colour.blue()


@pytest.fixture(autouse=True)
def _classic_look_afterwards(qapp):
    """A test may change the look of the whole program: the next test starts with the classic look again."""
    yield
    apply_app_theme(qapp, THEMES["classic"])


def _window(tmp_path: Path, **values) -> MainWindow:
    settings = Settings(tmp_path / "settings.json")
    settings.set("theme", "classic")
    for key, value in values.items():
        settings.set(key, value)
    return MainWindow(settings)


def _close(win: MainWindow) -> None:
    win.script.modified = False
    win.close()


def _signals(rows, scale=(-1.0, 1.0)):
    """Display signals from ``(name, unit[, colour[, digital]])`` rows."""
    t = np.linspace(0.0, 10.0, 500)
    out = []
    for k, row in enumerate(rows):
        name, unit = row[0], row[1]
        colour = row[2] if len(row) > 2 else 1 + k
        digital = row[3] if len(row) > 3 else False
        lo, hi = scale
        y = (t > 5.0).astype(float) if digital else lo + (hi - lo) * (0.5 + 0.4 * np.sin(t + k))
        out.append(DisplaySignal(entry=PltEntry(name=name), name=name, unit=unit, t=t, y=y, color=colour,
                                 digital=digital, scmin=0.0 if digital else lo, scmax=1.0 if digital else hi))
    return out


_PLOT_SIGNALS = [("speed", "km/h", 1), ("pressure", "bar", 2), ("brake", "", 5, True)]


def _pixels(image: QtGui.QImage) -> np.ndarray:
    image = image.convertToFormat(QtGui.QImage.Format_RGB32)
    ptr = image.constBits()
    raw = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
    return raw[:, :image.width(), [2, 1, 0]].copy()


def _paint(state, signals, size=(1000, 500)):
    image = QtGui.QImage(size[0], size[1], QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(*state.background))
    painter = QtGui.QPainter(image)
    lay = paint_view(painter, QtCore.QRectF(0, 0, size[0], size[1]), signals, state)
    painter.end()
    return _pixels(image), lay


def _region(pixels: np.ndarray, rect: QtCore.QRectF) -> np.ndarray:
    return pixels[int(rect.top()):int(rect.bottom()), int(rect.left()):int(rect.right())]


def _most_common(colours: np.ndarray):
    values, counts = np.unique(colours.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(c) for c in values[int(np.argmax(counts))])


def _drawn_colour(before: np.ndarray, after: np.ndarray, rect: QtCore.QRectF):
    """The colour of what was added between two pictures inside ``rect`` (the most common changed pixel)."""
    a, b = _region(before, rect), _region(after, rect)
    changed = np.any(a != b, axis=2)
    assert changed.any(), "nothing was drawn"
    return _most_common(b[changed])


def _measure(state, signals, size=(1000, 500)) -> dict:
    """Contrasts in a picture of ``state``: legend background, value text on it, cursor, marker and grid."""
    base_state = copy.copy(state)
    base_state.cursor, base_state.marker = None, None
    base_state.x0, base_state.x1, base_state.y0, base_state.y1 = 0.0, 10.0, 0.0, 1.0
    base, lay = _paint(base_state, signals, size)
    plot = lay.digital.united(lay.analog)
    bg = tuple(state.background)
    with_cursor = copy.copy(base_state)
    with_cursor.cursor = 4.0
    cursor, _ = _paint(with_cursor, signals, size)
    with_marker = copy.copy(base_state)
    with_marker.marker = 6.5
    marker, _ = _paint(with_marker, signals, size)
    no_grid = copy.copy(base_state)
    no_grid.grid = False
    plain, _ = _paint(no_grid, signals, size)
    legend = lay.legend
    corner = QtCore.QRectF(legend.right() - 12, legend.bottom() - 8, 10, 6)
    value_area = QtCore.QRectF(legend.left() + lay.name_width + 5, legend.top(), legend.width() - lay.name_width - 6,
                               lay.row_height)
    area = _region(cursor, value_area)
    area_bg = _most_common(area)
    text = max(_contrast(tuple(int(c) for c in p), area_bg) for p in np.unique(area.reshape(-1, 3), axis=0))
    return {"legend": _most_common(_region(base, corner)), "text": text,
            "cursor": _contrast(_drawn_colour(base, cursor, plot), bg),
            "marker": _contrast(_drawn_colour(base, marker, plot), bg),
            "grid": _contrast(_drawn_colour(plain, base, plot), bg)}


def _picture_file(path: Path, size=(320, 200), colour: str = "#3366cc") -> Path:
    image = QtGui.QImage(size[0], size[1], QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor(colour))
    assert image.save(str(path))
    return path


# ---------------------------------------------------------------------- choosing a theme keeps the user's look
def test_switching_back_to_classic_after_a_pointer_menu_was_deleted(qapp, tmp_path):
    classic = _window(tmp_path)
    native_palette = QtGui.QPalette(qapp.palette())
    native_style = qapp.style().objectName()
    classic.set_theme("racing")
    menu = classic.backdrop_menu
    # The custom pointer is applied on Show. Retain the Python wrapper across a menu rebuild.
    QtWidgets.QApplication.sendEvent(menu, QtCore.QEvent(QtCore.QEvent.Type.Show))
    assert menu in classic._pointer_windows
    classic._build_actions()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert not isValid(menu)

    classic.set_theme("classic")
    assert menu not in classic._pointer_windows
    assert not qapp.styleSheet()
    assert qapp.style().objectName() == native_style
    assert qapp.palette() == native_palette
    assert classic.toolbar.toolButtonStyle() == QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
    assert classic.view.state.theme == "classic"
    assert classic.theme_actions["classic"].isChecked()
    _close(classic)


def test_classic_menu_uses_the_current_windows_colours(qapp, tmp_path):
    if qapp.platformName() != "windows" or qapp.style().objectName() != "windows11":
        pytest.skip("Requires the native Windows 11 Qt style")

    hints = qapp.styleHints()
    scheme = QtCore.Qt.ColorScheme
    roles = QtGui.QPalette.ColorRole

    def bar_colours(bar):
        return bar.palette().color(roles.Window), bar.palette().color(roles.WindowText)

    win = _window(tmp_path)
    try:
        hints.setColorScheme(scheme.Dark)
        win.menuBar().ensurePolished()
        qapp.processEvents()
        dark_colours = bar_colours(win.menuBar())

        hints.setColorScheme(scheme.Light)
        qapp.processEvents()
        assert bar_colours(win.menuBar()) != dark_colours
        win.theme_actions["obsidian"].trigger()
        hints.setColorScheme(scheme.Dark)
        qapp.processEvents()
        win.theme_actions["classic"].trigger()
        assert bar_colours(win.menuBar()) == dark_colours
    finally:
        _close(win)
        hints.setColorScheme(scheme.Unknown)


def test_clicking_the_theme_in_use_keeps_the_users_choices(qapp, tmp_path):
    win = _window(tmp_path, background="#c7edcc")                # a user-defined plot background
    win.theme_actions["classic"].trigger()                       # already checked: Qt still sends "triggered"
    assert win.settings.get("background") == "#c7edcc" and win.view.state.background == (199, 237, 204)
    win.set_theme("daylight")
    win.set_theme("classic")
    win._set_option("background", "#ffffff")                     # View > Background > White
    win.theme_actions["classic"].trigger()
    assert win.settings.get("background") == "#ffffff"           # not the background saved by the round trip
    win.set_theme("obsidian")
    win._set_option("legend_font_size", 17)
    win._set_option("time_labels", False)
    win.theme_actions["obsidian"].trigger()
    assert win.settings.get("legend_font_size") == 17 and win.settings.get("time_labels") is False
    _close(win)


def test_a_theme_that_no_longer_exists_does_not_replace_the_classic_background(qapp, tmp_path):
    win = _window(tmp_path, theme="cosmic", background="#0d1024", classic_background="#c7edcc")
    assert win.settings.get("theme") == "classic"
    assert win.view.state.background == (199, 237, 204)          # the classic look with the user's background
    win.set_theme("daylight")
    win.set_theme("classic")
    assert win.settings.get("background") == "#c7edcc" and win.view.state.background == (199, 237, 204)
    _close(win)


def test_returning_to_the_classic_theme_gives_back_the_users_appearance(qapp, tmp_path):
    picture = _picture_file(tmp_path / "background.png")
    pointer = _picture_file(tmp_path / "pointer.png", (64, 40), "#ff8800")
    win = _window(tmp_path)
    for key, value in (("legend_font_size", 16), ("time_labels", True), ("background", "#ffffff"),
                       ("backdrop_opacity", 0.45)):
        win._set_option(key, value)                              # Appearance dialog / View menu
    win._set_backdrop(str(picture))
    win.settings.set("cursor_image", str(pointer))
    win._set_pointer("custom")
    keys = ("background", "legend_font_size", "time_labels", "backdrop", "backdrop_opacity", "cursor_character")
    before = {key: win.settings.get(key) for key in keys}
    win.set_theme("daylight")
    assert win.settings.get("legend_font_size") == THEMES["daylight"].legend_font_px   # the theme's own look
    win.set_theme("racing")
    win.set_theme("classic")
    assert {key: win.settings.get(key) for key in keys} == before
    state = win.view.state
    assert state.font_size == 16 and state.time_labels and state.background == WHITE
    assert state.backdrop is not None and abs(state.backdrop_opacity - 0.45) < 1e-9
    assert set(win.view.theme_cursors) == {"zoom", "pointer"}
    _close(win)


# ---------------------------------------------------------------------- theme colours that suit the background
@pytest.mark.parametrize("key, background", [("daylight", "black"), ("daylight", "dark"), ("obsidian", "white"),
                                             ("racing", "green")])
def test_a_background_that_does_not_suit_the_theme_keeps_signals_legend_and_cursor_readable(qapp, tmp_path, key,
                                                                                         background):
    rgb = BACKGROUNDS[background][1]
    win = _window(tmp_path)
    win.set_theme(key)
    win._set_option("background", "#%02x%02x%02x" % rgb)        # View > Background
    state = win.view.state
    assert state.background == rgb
    assert _contrast(signal_color(1, state), rgb) >= 4.5        # colour 1: the colour of a signal without one
    assert min(_contrast(signal_color(i, state), rgb) for i in range(2, 16)) >= 2.0
    m = _measure(state, _signals(_PLOT_SIGNALS))                # as copied to the clipboard
    assert m["text"] >= 4.5 and m["cursor"] >= 3.0, m
    _close(win)


@pytest.mark.parametrize("key", THEME_KEYS)
def test_pictures_on_white_are_light_and_readable_in_every_theme(qapp, tmp_path, key):
    win = _window(tmp_path)
    win.set_theme(key)
    state = win._picture_state()                                # print, PDF, picture: on white by default
    assert state.background == WHITE
    m = _measure(state, _signals(_PLOT_SIGNALS))
    assert _luminance(m["legend"]) >= 0.75, m                   # a light legend, not a dark block on the paper
    assert m["text"] >= 4.5, m
    assert m["cursor"] >= 3.0 and m["marker"] >= 2.0, m         # cursor and marker lines can be seen
    assert m["grid"] <= 4.5, m                                  # the grid stays light, not near-black lines
    if THEMES[key].run1:                                        # the theme's signal colours made for a light plot
        assert min(_contrast(signal_color(i, state), WHITE) for i in range(1, 16)) >= 3.0
    assert win.view.state.background != WHITE or key == "daylight"   # the screen keeps its own colours
    _close(win)


def test_pictures_on_white_keep_a_palette_of_the_user(qapp, tmp_path):
    palette = [[(17 * k) % 256, (53 * k) % 256, (97 * k) % 256] for k in range(32)]
    win = _window(tmp_path, palette=palette)
    win.set_theme("obsidian")
    state = win._picture_state()
    assert [tuple(c) for c in state.palette] == [tuple(c) for c in palette]
    _close(win)


# ---------------------------------------------------------------------- chip legend
class _TextRecorder(QtGui.QPainter):
    """A painter that remembers the texts it draws."""

    def __init__(self, device: QtGui.QPaintDevice) -> None:
        super().__init__(device)
        self.texts = []

    def drawText(self, *args):
        self.texts.append(args[-1])
        return super().drawText(*args)


def _themed_view(tmp_path: Path, key: str, rows, size, scale=(-1.0, 1.0)) -> PlotView:
    theme = THEMES[key]
    settings = Settings(tmp_path / ("view_%s.json" % key))
    for name, value in (("theme", key), ("background", theme.plot or "#000000"),
                        ("legend_font_size", theme.legend_font_px), ("time_labels", theme.time_labels)):
        settings.set(name, value)
    view = PlotView()
    view.resize(*size)
    view.apply_settings(settings)
    view.set_signals(_signals(rows, scale), (0.0, 10.0))
    return view


def _legend_as_painted(view: PlotView):
    """(name, value) of every legend row as the view paints it (with the layout painting uses)."""
    view._layout = compute_layout(QtCore.QRectF(view.rect()), view.signals, view.state)
    image = QtGui.QImage(view.width(), view.height(), QtGui.QImage.Format_RGB32)
    painter = _TextRecorder(image)
    view.state.cursor = view.cursor_time()
    paint_legend(painter, view._layout, view.signals, view.state)
    view.state.cursor = None
    painter.end()
    return list(zip(painter.texts[0::2], painter.texts[1::2]))


def _record_tips(monkeypatch):
    calls = []
    monkeypatch.setattr(QtWidgets.QToolTip, "showText", staticmethod(lambda *a, **k: calls.append("show")))
    monkeypatch.setattr(QtWidgets.QToolTip, "hideText", staticmethod(lambda *a, **k: calls.append("hide")))
    return calls


def _tip_shown(view: PlotView, row: int, calls) -> bool:
    view._legend_tip(row, QtCore.QPoint(5, 5))
    return calls[-1] == "show"


@pytest.mark.parametrize("key", ["daylight", "obsidian", "racing"])
def test_the_chip_legend_draws_every_name_in_full_when_there_is_room(qapp, tmp_path, key, monkeypatch):
    calls = _record_tips(monkeypatch)
    view = _themed_view(tmp_path, key, [("VehicleSpeed", "km/h"), ("BenchLeverPosition", "%"), ("v", ""),
                                        ("SteeringWheelAngle", "deg")], (1600, 600))
    rows = _legend_as_painted(view)
    assert [name for name, _value in rows] == [sig.label for sig in view.signals]
    assert not any(_tip_shown(view, row, calls) for row in range(len(view.signals)))


@pytest.mark.parametrize("key", ["daylight", "obsidian", "racing"])
def test_a_shortened_chip_legend_row_shows_its_full_text_as_a_tip(qapp, tmp_path, key, monkeypatch):
    calls = _record_tips(monkeypatch)
    long = ("Signal_name_far_too_long_for_the_name_column_" * 3, "")      # wider than the column: cut anyway
    view = _themed_view(tmp_path, key, [long, ("x", "")], (2000, 400), scale=(0.0, 100.0))
    _legend_as_painted(view)
    metrics = QtGui.QFontMetricsF(legend_font(view.state, view._layout.row_height))
    probe = "m"                    # a name a little too long for the room after the colour mark
    while metrics.horizontalAdvance(probe + " ()") < view._layout.name_width - 20:
        probe += "m"
    view.set_signals(_signals([long, (probe, "")], (0.0, 100.0)), (0.0, 10.0))
    seen = set()
    for width in [2000] + list(range(700, 2400, 10)):
        view.resize(width, 400)
        rows = _legend_as_painted(view)
        for row, (name, value) in enumerate(rows):
            sig = view.signals[row]
            shortened = (name != sig.label, value != legend_value_text(sig, view.state))
            expected = any(shortened) or view._layout.font_px < 11
            assert _tip_shown(view, row, calls) == expected, (width, row, name, value)
            seen.add((width == 2000, row, shortened))
    assert (True, 1, (True, False)) in seen                     # only the probe's name was shortened
    assert {True, False} <= {s[1] for w, r, s in seen if r == 1}   # the value was cut at some widths only


# ---------------------------------------------------------------------- pictures behind the curves
def test_a_background_picture_that_cannot_be_read_is_refused_with_a_message(qapp, tmp_path, monkeypatch):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not a picture")
    good = _picture_file(tmp_path / "good.png")
    win = _window(tmp_path)
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kw: messages.append(args[2]))
    chosen = [str(broken)]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args, **kw: (chosen[0], ""))
    win._choose_backdrop_file()
    assert len(messages) == 1 and str(broken) in messages[0]
    assert win.settings.get("backdrop") is None and win.view.state.backdrop is None
    assert [a.text() for a in win.backdrop_menu.actions() if a.isChecked()] == [tr("none")]
    chosen[0] = str(good)
    win._choose_backdrop_file()
    assert len(messages) == 1 and win.settings.get("backdrop") == str(good)
    assert win.view.state.backdrop is not None
    _close(win)


def test_the_own_background_picture_is_read_once_and_again_only_when_the_file_changes(qapp, tmp_path):
    picture = _picture_file(tmp_path / "own.png", (320, 200))
    settings = Settings(tmp_path / "settings.json")
    settings.set("backdrop", str(picture))
    view = PlotView()
    view.resize(900, 500)
    view.show()
    view.apply_settings(settings)
    qapp.processEvents()
    first = view.state.backdrop
    assert first is not None and first.width() == 320
    for width in (950, 1000, 1080, 1200):                       # dragging the window edge
        view.resize(width, 500)
        qapp.processEvents()
    assert view.width() == 1200
    settings.set("grid", False)
    view.apply_settings(settings)                               # another View option
    assert view.state.backdrop.cacheKey() == first.cacheKey()
    _picture_file(picture, (640, 400), "#cc3366")
    stamp = time.time() + 5
    os.utime(picture, (stamp, stamp))
    view.apply_settings(settings)
    assert view.state.backdrop.width() == 640                   # a changed file is read again
    view.close()


def test_scene_pictures_kept_in_memory_stay_few_and_go_with_the_theme(qapp, tmp_path):
    settings = Settings(tmp_path / "settings.json")
    settings.set("theme", "racing")
    view = PlotView()
    view.resize(900, 500)
    view.show()
    view.apply_settings(settings)
    assert view.state.backdrop is not None
    for width in range(900, 1900, 50):                          # every 50 px a new width of the scene
        view.resize(width, 500)
        qapp.processEvents()
    assert view.width() == 1850
    assert cute_art._scene_image.cache_info().currsize <= 2
    settings.set("theme", "classic")
    view.apply_settings(settings)
    assert view.state.backdrop is None and cute_art._scene_image.cache_info().currsize == 0
    view.close()


# ---------------------------------------------------------------------- menus filled when they open
def test_menus_filled_when_they_open_do_not_pile_up_submenus_and_groups(qapp, tmp_path):
    win = _window(tmp_path, layouts={"Mine": {"state": "", "toolbar": True, "statusbar": True}})
    menus = {"picture": win.backdrop_menu, "pointer": win.pointer_menu, "layout": win.layout_menu}

    def objects():
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        return {name: tuple(len(menu.findChildren(kind))
                            for kind in (QtWidgets.QMenu, QtGui.QActionGroup, QtGui.QAction))
                for name, menu in menus.items()}

    before = objects()
    for _ in range(5):
        for menu in menus.values():
            menu.aboutToShow.emit()                             # every opening fills the menu again
    assert objects() == before
    _close(win)


# ---------------------------------------------------------------------- menu-bar decoration of the racing theme
def test_the_racing_theme_starts_although_its_menu_bar_picture_cannot_be_written(qapp, tmp_path, monkeypatch):
    blocker = tmp_path / "tracepaddock_theme"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    monkeypatch.setattr(themes, "_asset_folder", lambda: blocker)
    win = _window(tmp_path, theme="racing")                     # must not stop the program
    sheet = qapp.styleSheet()
    assert "QMenuBar" in sheet and "background-image" not in sheet
    _close(win)
    folder = tmp_path / "assets"
    folder.mkdir()
    (folder / "bar_racing.svg").write_bytes(b"\xff\xfe damaged \x00")   # not even UTF-8
    monkeypatch.setattr(themes, "_asset_folder", lambda: folder)
    apply_app_theme(qapp, THEMES["racing"])
    assert "background-image" in qapp.styleSheet()
    assert (folder / "bar_racing.svg").read_bytes().startswith(b"<svg")


# ---------------------------------------------------------------------- red marks of the signal editor
class _Recording(Dataset):
    def __init__(self) -> None:
        super().__init__("demo.d97")
        t = np.arange(0.0, 10.0, 0.1)
        self._data = {"speed": (t, 20 + 5 * np.sin(t)), "pressure": (t, 100 + 50 * np.cos(t))}
        self.signals = [SignalInfo("speed", unit="m/s"), SignalInfo("pressure", unit="bar")]
        self._finish_catalogue()

    def read(self, signal):
        t, y = self._data[signal.name]
        return SignalData(t, y)

    def time_range(self):
        return (0.0, 9.9)


@pytest.mark.parametrize("key", ["obsidian", "racing"])
def test_red_marks_of_the_signal_editor_stay_readable_in_dark_themes(qapp, tmp_path, key):
    apply_app_theme(qapp, THEMES[key])
    dialog = PltEditorDialog(_Recording(), parse_plt("speed nobit 2\npressure nobit 3\n"),
                             Settings(tmp_path / "settings.json"))
    dialog.show()
    dialog.entry_list.setCurrentRow(0)
    dialog.min_edit.setText("abc")                              # not a number: marked red
    dialog._store_properties()
    dialog.filter_mode.setCurrentIndex(2)                       # regular expression ...
    dialog.filter_edit.setCurrentText("(unclosed")              # ... that cannot be read: marked red
    dialog._refresh_filter()
    for widget in (dialog.min_edit, dialog.filter_edit.lineEdit()):
        palette = widget.palette()
        assert palette.color(QtGui.QPalette.Base).name() == "#ffd0d0"
        assert _contrast(_rgb(palette.color(QtGui.QPalette.Text)), _rgb(palette.color(QtGui.QPalette.Base))) >= 4.5
    dialog.reject()
