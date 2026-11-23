from tracepaddock.core.registry import open_dataset
from tracepaddock.i18n import language, set_language
from tracepaddock.plt.parser import load_plt
from tracepaddock.plt.process import build_display_signals
from tracepaddock.settings import Settings
from tracepaddock.valuemap import ValueMapLibrary, read_mapping_file
from tracepaddock.ui.mainwindow import MainWindow
from tracepaddock.ui.warmup import LESSONS, WARMUP_DIR


def test_bundled_mapping_shows_the_timeout_as_a_state_band(tmp_path):
    library = ValueMapLibrary(tmp_path / "maps.json", autoload=False)
    assert library.merge(read_mapping_file(WARMUP_DIR / "warmup-mapping.json")) == (1, 1)
    with open_dataset(WARMUP_DIR / "03-timing.mf4") as dataset:
        script = load_plt(WARMUP_DIR / "03-timing.plt")
        signals = build_display_signals(dataset, script, valuemaps=library)
        status = next(signal for signal in signals if signal.name == "TutorialAckStatus")
        assert status.lane and not status.digital
        assert status.value_at(6.6) == 3
        assert status.state_text(3) == "Timed out"


def test_warmup_menu_opens_the_selected_recording_and_view(qapp, tmp_path):
    original = language()
    settings = Settings(tmp_path / "settings.json", autoload=False)
    window = MainWindow(settings)
    try:
        assert settings.get("theme") == "obsidian"
        window._set_language("zh")
        help_menu = next(action.menu() for action in window.menuBar().actions() if "帮助" in action.text())
        next(action for action in help_menu.actions() if "快速上手" in action.text()).trigger()
        dialog = window._warmup_dialog
        assert dialog.isVisible() and not dialog.isModal()
        assert "快速上手" in dialog.guide.toPlainText()
        for index, (stem, _) in enumerate(LESSONS):
            dialog.lesson.setCurrentIndex(index)
            dialog.open_button.click()
            assert window.dataset.path.stem == window.script.path.stem == stem
            assert window.signals and all(signal.has_data for signal in window.signals)
        window._set_language("en")
        assert dialog.windowTitle() == "Quick warm-up"
        assert dialog.lesson.currentIndex() == 2
        assert "Obsidian" in dialog.guide.toPlainText()
    finally:
        window.close()
        set_language(original)
