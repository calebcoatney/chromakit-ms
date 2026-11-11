"""GC-only files hide the MS UI; a later GC-MS file shows it again."""
import numpy as np
import pytest

pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QCheckBox

from ui.app import ChromaKitApp


@pytest.fixture
def app(qtbot, tmp_path, monkeypatch):
    window = ChromaKitApp()
    qtbot.addWidget(window)
    window.show()

    def fake_load(path, detector=None):
        has_ms = "gcms" in path
        window.data_handler.has_ms_data = has_ms
        x = np.linspace(0, 10, 200)
        return {
            "chromatogram": {"x": x, "y": np.sin(x) + 2},
            "tic": {"x": x, "y": x} if has_ms else {},
        }

    monkeypatch.setattr(window.data_handler, "load_data_directory", fake_load)

    def fail(parent, title, text, *a, **k):
        raise AssertionError(text)       # a load error would otherwise open a modal and hang

    monkeypatch.setattr("ui.app.QMessageBox.critical", fail)
    return window, tmp_path


def _load(window, tmp_path, name):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    window.on_file_selected(str(d))


def _ms_widgets(window):
    pf, bf = window.parameters_frame, window.button_frame
    return {
        "batch": bf.batch_search_button,
        "deconvolve": bf.deconvolve_ms_button,
        "baseline": pf.ms_baseline_button,
    }


def test_gc_only_hides_ms_ui_and_gcms_shows_it_again(app):
    window, tmp_path = app
    ms_idx = window.right_tabs.indexOf(window.ms_frame)

    _load(window, tmp_path, "gcms_a.D")
    assert window.right_tabs.isTabVisible(ms_idx)
    assert all(not w.isHidden() for w in _ms_widgets(window).values())

    _load(window, tmp_path, "gconly.D")
    assert not window.right_tabs.isTabVisible(ms_idx)
    assert all(w.isHidden() for w in _ms_widgets(window).values())

    _load(window, tmp_path, "gcms_b.D")
    assert window.right_tabs.isTabVisible(ms_idx)
    assert all(not w.isHidden() for w in _ms_widgets(window).values())
    # visible does not mean enabled: still waiting for integrated peaks
    assert not window.button_frame.batch_search_button.isEnabled()


def test_corrected_ms_checkbox_follows_the_gate(app):
    window, tmp_path = app
    pf = window.parameters_frame
    _load(window, tmp_path, "gconly.D")
    assert not hasattr(pf, "use_corrected_ms_data_check")   # not created yet: no error

    _load(window, tmp_path, "gcms_a.D")
    pf.use_corrected_ms_data_check = QCheckBox("Use Baseline-Corrected MS Data", pf)
    pf.use_corrected_ms_data_check.show()
    _load(window, tmp_path, "gconly.D")
    assert pf.use_corrected_ms_data_check.isHidden()

    _load(window, tmp_path, "gcms_b.D")
    assert not pf.use_corrected_ms_data_check.isHidden()
