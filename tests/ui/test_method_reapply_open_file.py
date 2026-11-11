"""Loading a method re-applies it to the open file, through the real on_file_selected.

Only the loader boundary is faked: DataHandler.load_data_directory for .D
folders, CFolder.open for .C folders.
"""
import numpy as np
import pytest

pytest.importorskip("pytestqt")

from logic.method import ChromaMethod
from logic.signal_profiles import SignalProfileRegistry
from ui.app import ChromaKitApp

DETECTORS = ["FID1A", "FID2B"]
DEFAULT_CHANNEL = "FID1A"


def _trace(detector, signal_factor):
    x = np.linspace(0, 10, 2000)
    scale = {"FID1A": 100.0, "FID2B": 10000.0}[detector]
    y = scale * np.exp(-0.5 * ((x - 5) / 0.1) ** 2) + scale * 0.5
    return x, y * signal_factor


class _FakeCFolder:
    """A .C folder whose loader auto-detects DEFAULT_CHANNEL when no detector is given."""

    loads = []

    def __init__(self, path):
        self.path = path

    @property
    def profile(self):
        return SignalProfileRegistry.get("gc")

    def get_available_detectors(self):
        return list(DETECTORS)

    def load_signal(self, signal_factor=1.0, detector=None):
        channel = detector or DEFAULT_CHANNEL
        self.loads.append((self.path, detector, channel, signal_factor))
        x, y = _trace(channel, signal_factor)
        return {"x": x, "y": y, "metadata": {"detector": channel, "has_ms_data": False}}


@pytest.fixture
def app(qtbot, tmp_path, monkeypatch):
    window = ChromaKitApp()
    qtbot.addWidget(window)
    window.show()
    loads = []

    def fake_load(path, detector=None):
        # mirrors DataHandler.load_data_directory: keep the channel unless told otherwise
        if detector is not None:
            window.data_handler.current_detector = detector
        elif window.data_handler.current_detector == "Unknown":
            window.data_handler.current_detector = DEFAULT_CHANNEL
        has_ms = "gcms" in path
        window.data_handler.has_ms_data = has_ms
        loads.append((path, detector, window.data_handler.current_detector, window.signal_factor))
        x, y = _trace(window.data_handler.current_detector, window.signal_factor)
        return {"chromatogram": {"x": x, "y": y}, "tic": {"x": x, "y": x} if has_ms else {}}

    monkeypatch.setattr(window.data_handler, "load_data_directory", fake_load)
    monkeypatch.setattr(window.data_handler, "get_available_detectors", lambda *a: list(DETECTORS))
    _FakeCFolder.loads = []
    monkeypatch.setattr("logic.c_folder.CFolder.open", staticmethod(lambda path: _FakeCFolder(path)))

    def fail(parent, title, text, *a, **k):
        raise AssertionError(text)       # a modal would hang the run

    monkeypatch.setattr("ui.app.QMessageBox.critical", fail)
    return window, tmp_path, loads, monkeypatch


def _open(window, tmp_path, name, detector=None):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    window.on_file_selected(str(d), detector=detector)
    return str(d)


def _load_method(window, tmp_path, monkeypatch, **kw):
    p = tmp_path / "applied.chromethod"
    ChromaMethod(name="applied", signal_type="gc", **kw).to_file(p)
    monkeypatch.setattr("ui.app.QFileDialog.getOpenFileName", lambda *a, **k: (str(p), ""))
    window.load_method()


# ── T1: the method's detector reloads the real file once ────────────────────

@pytest.mark.parametrize("name, has_ms", [("gconly.D", False), ("gcms.D", True)])
def test_method_detector_reloads_open_file_once(app, name, has_ms):
    window, tmp_path, loads, mp = app
    path = _open(window, tmp_path, name)
    assert window.data_handler.current_detector == "FID1A"
    loads.clear()

    _load_method(window, tmp_path, mp, detector="FID2B")

    assert [(p, d) for p, d, _, _ in loads] == [(path, "FID2B")]
    assert window.data_handler.current_detector == "FID2B"
    assert window._method_dirty is False
    assert window.method_label.text() == "Method: applied"
    ms_idx = window.right_tabs.indexOf(window.ms_frame)
    assert window.right_tabs.isTabVisible(ms_idx) is has_ms
    assert window.parameters_frame.ms_baseline_button.isHidden() is not has_ms
    assert "Applied method 'applied'" in window.status_bar.currentMessage()


# ── Issue 4: a reload must keep the channel the user is looking at ──────────

def test_c_folder_signal_factor_reload_keeps_channel(app):
    window, tmp_path, loads, mp = app
    path = _open(window, tmp_path, "sample.C", detector="FID2B")
    assert window.data_handler.current_detector == "FID2B"
    _FakeCFolder.loads.clear()

    _load_method(window, tmp_path, mp, signal_factor=2.0)

    assert [(d, ch) for _, d, ch, _ in _FakeCFolder.loads] == [("FID2B", "FID2B")]
    assert window.data_handler.current_detector == "FID2B"


def test_c_folder_absent_method_detector_keeps_channel(app):
    window, tmp_path, loads, mp = app
    _open(window, tmp_path, "sample.C", detector="FID2B")
    _FakeCFolder.loads.clear()

    # detector absent from the file, plus a signal-factor change that forces a reload
    _load_method(window, tmp_path, mp, detector="TCD3C", signal_factor=2.0)

    assert [ch for _, _, ch, _ in _FakeCFolder.loads] == ["FID2B"]
    assert "TCD3C not in this file; showing FID2B" in window.status_bar.currentMessage()


def test_c_folder_scaling_dialog_reload_keeps_channel(app):
    window, tmp_path, loads, mp = app
    _open(window, tmp_path, "sample.C", detector="FID2B")
    _FakeCFolder.loads.clear()

    window._on_scaling_factors_changed(3.0, 1.0)

    assert [(d, ch, sf) for _, d, ch, sf in _FakeCFolder.loads] == [("FID2B", "FID2B", 3.0)]
