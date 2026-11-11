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
    assert pf.use_corrected_ms_data_check is None   # not created yet: no error

    _load(window, tmp_path, "gcms_a.D")
    pf.use_corrected_ms_data_check = QCheckBox("Use Baseline-Corrected MS Data", pf)
    pf.use_corrected_ms_data_check.show()
    _load(window, tmp_path, "gconly.D")
    assert pf.use_corrected_ms_data_check.isHidden()

    _load(window, tmp_path, "gcms_b.D")
    assert not pf.use_corrected_ms_data_check.isHidden()


def _load_method(window, tmp_path, monkeypatch, name="m", **kw):
    from logic.method import ChromaMethod
    p = tmp_path / f"{name}.chromethod"
    ChromaMethod(name=name, signal_type="gc", **kw).to_file(p)
    monkeypatch.setattr("ui.app.QFileDialog.getOpenFileName", lambda *a, **k: (str(p), ""))
    window.load_method()


def test_method_load_keeps_ms_baseline_button_hidden_for_gc_only(app, monkeypatch):
    # apply_method rebuilds the parameter widgets, including a fresh (visible)
    # ms_baseline_button; the MS gate must run again afterwards.
    window, tmp_path = app
    _load(window, tmp_path, "gconly.D")
    assert window.parameters_frame.ms_baseline_button.isHidden()

    _load_method(window, tmp_path, monkeypatch)      # same signal factor: reprocess path
    assert window.parameters_frame.ms_baseline_button.isHidden()


class _FakeMS:
    def __init__(self):
        self.xlabels = np.linspace(0, 10, 5)
        self.data = np.ones((5, 3))


class _FakeDialog:
    def close(self):
        pass


def _complete_ms_baseline(window, monkeypatch):
    """Drive the real MS-baseline completion handler (creates the checkbox)."""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr("ui.app.QMessageBox.question", lambda *a, **k: QMessageBox.No)
    ms = _FakeMS()
    window._on_ms_baseline_correction_completed(ms, ms.data * 0.5, ms.data * 0.5, _FakeDialog())
    return window.parameters_frame.use_corrected_ms_data_check


def test_corrected_ms_checkbox_lives_in_the_baseline_section(app, monkeypatch):
    window, tmp_path = app
    pf = window.parameters_frame
    _load(window, tmp_path, "gcms_a.D")

    cb = _complete_ms_baseline(window, monkeypatch)
    assert cb is not None
    assert not cb.isWindow()                          # parented, not a stray top-level window
    assert pf.section_groups['baseline'].isAncestorOf(cb)
    assert not cb.isHidden()

    _load(window, tmp_path, "gconly.D")
    assert cb.isHidden()


def test_corrected_ms_checkbox_survives_a_method_rebuild(app, monkeypatch, qtbot):
    from PySide6.QtCore import QCoreApplication, QEvent
    window, tmp_path = app
    _load(window, tmp_path, "gcms_a.D")
    _complete_ms_baseline(window, monkeypatch)

    _load_method(window, tmp_path, monkeypatch)       # rebuilds the baseline section
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)   # run the deleteLater()s
    _load(window, tmp_path, "gcms_b.D")               # gate must not touch the deleted checkbox
    _load(window, tmp_path, "gconly.D")

    _load(window, tmp_path, "gcms_c.D")
    cb = _complete_ms_baseline(window, monkeypatch)   # created again in the new section
    assert window.parameters_frame.section_groups['baseline'].isAncestorOf(cb)


def test_gate_skips_a_deleted_or_unparented_checkbox(app, capsys):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    window, tmp_path = app
    pf = window.parameters_frame

    holder = QWidget()
    pf.use_corrected_ms_data_check = QCheckBox("Use Baseline-Corrected MS Data", holder)
    holder.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    window._apply_ms_visibility(True)                 # no "Internal C++ object already deleted"

    stray = QCheckBox("Use Baseline-Corrected MS Data")
    pf.use_corrected_ms_data_check = stray
    window._apply_ms_visibility(True)
    assert not stray.isVisible()                      # never shown as its own window
    assert capsys.readouterr().out.count("WARNING: MS gate skipped") == 2
