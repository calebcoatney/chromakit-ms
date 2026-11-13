"""A parameter change before any file is open does nothing and raises nothing."""
import pytest

pytest.importorskip("pytestqt")

from ui.app import ChromaKitApp


def test_parameter_change_before_a_file_is_open(qtbot, monkeypatch):
    window = ChromaKitApp()
    qtbot.addWidget(window)
    processed = []
    monkeypatch.setattr(window, "process_and_display", lambda *a, **k: processed.append(a))
    assert window.current_x is None and window.current_y is None
    window.on_parameters_changed(window.parameters_frame.current_params)   # raised AttributeError before
    assert processed == []
