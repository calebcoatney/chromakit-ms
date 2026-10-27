import numpy as np
import pytest
pytest.importorskip('pytestqt')

from pathlib import Path
from ui.app import ChromaKitApp


def _app_with_processed(qtbot, signal_range):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_processed = {'x': np.array([0.0, 1.0]), 'corrected_y': np.array([0.0, signal_range])}
    return app


def _params(min_prominence):
    return {'peaks': {'enabled': True, 'min_prominence': min_prominence}}


def test_reason_names_missing_method(qtbot):
    app = _app_with_processed(qtbot, 3141.5)
    assert app.current_method_path is None
    assert "no method is loaded" in app._no_peaks_reason(_params(100000.0))


def test_reason_with_method_loaded_points_at_threshold(qtbot):
    app = _app_with_processed(qtbot, 3141.5)
    app.current_method_path = Path("GC2-FIDA.chromethod")
    msg = app._no_peaks_reason(_params(100000.0))
    assert "no method is loaded" not in msg
    assert "above the signal range" in msg


def test_integrate_shows_reason_when_no_peaks(qtbot, monkeypatch):
    app = _app_with_processed(qtbot, 3141.5)
    app.parameters_frame.current_params['peaks']['enabled'] = True   # no toggle signal: no raw data to reprocess
    monkeypatch.setattr("ui.app.QMessageBox.warning", lambda *a, **k: pytest.fail(f"unexpected dialog: {a[1:3]}"))
    monkeypatch.setattr(app, "integrate_peaks_no_ui", lambda **k: None)
    app.on_integrate()
    assert "no method is loaded" in app.status_bar.currentMessage()
