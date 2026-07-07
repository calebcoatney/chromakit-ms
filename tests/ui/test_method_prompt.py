import pytest
pytest.importorskip('pytestqt')

from ui.app import ChromaKitApp
from PySide6.QtWidgets import QMessageBox

# Capture the real method at import time, BEFORE the autouse _no_close_prompt
# fixture (in tests/ui/conftest.py) patches it per-test.
_REAL_MAYBE_PROMPT = ChromaKitApp._maybe_prompt_save


def _make(qtbot, monkeypatch):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    # Re-install the real _maybe_prompt_save for these tests (override the autouse stub).
    monkeypatch.setattr(ChromaKitApp, "_maybe_prompt_save", _REAL_MAYBE_PROMPT)
    return app


def test_prompt_returns_true_when_not_dirty(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = False
    assert app._maybe_prompt_save() is True


def test_prompt_cancel_returns_false(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = True
    monkeypatch.setattr("ui.app.QMessageBox.question", lambda *a, **k: QMessageBox.Cancel)
    assert app._maybe_prompt_save() is False


def test_prompt_discard_returns_true(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = True
    monkeypatch.setattr("ui.app.QMessageBox.question", lambda *a, **k: QMessageBox.Discard)
    assert app._maybe_prompt_save() is True


def test_prompt_save_success_returns_true(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = True
    monkeypatch.setattr("ui.app.QMessageBox.question", lambda *a, **k: QMessageBox.Save)
    # Simulate a successful save that clears dirty:
    def fake_save():
        app._method_dirty = False
    monkeypatch.setattr(app, "save_method", fake_save)
    assert app._maybe_prompt_save() is True


def test_prompt_save_cancelled_dialog_returns_false(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = True
    monkeypatch.setattr("ui.app.QMessageBox.question", lambda *a, **k: QMessageBox.Save)
    # Simulate a save whose file dialog was cancelled -> dirty stays True:
    monkeypatch.setattr(app, "save_method", lambda: None)  # no-op, dirty remains True
    assert app._maybe_prompt_save() is False


def test_close_event_ignored_when_prompt_false(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = True
    monkeypatch.setattr("ui.app.QMessageBox.question", lambda *a, **k: QMessageBox.Cancel)
    from PySide6.QtGui import QCloseEvent
    ev = QCloseEvent()
    app.closeEvent(ev)
    assert ev.isAccepted() is False  # event.ignore() was called


def test_close_event_accepted_when_not_dirty(qtbot, monkeypatch):
    app = _make(qtbot, monkeypatch)
    app._method_dirty = False
    from PySide6.QtGui import QCloseEvent
    ev = QCloseEvent()
    ev.accept()
    app.closeEvent(ev)
    assert ev.isAccepted() is True  # super().closeEvent ran (no ignore)
