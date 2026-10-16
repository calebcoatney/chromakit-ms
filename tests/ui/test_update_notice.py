import pytest

pytest.importorskip("pytestqt")

from ui.app import ChromaKitApp


def test_newer_release_shows_a_button(qtbot, monkeypatch):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    opened = []
    monkeypatch.setattr("ui.app.QDesktopServices.openUrl", lambda url: opened.append(url.toString()))
    app._on_update_found("9.9.9", "https://github.com/calebcoatney/chromakit-ms/releases/tag/v9.9.9")
    btn = app.update_button
    # the window is never shown in tests, so isVisible() would always be False
    assert not btn.isHidden() and btn.text() == "Update available: v9.9.9"
    btn.click()
    assert opened == ["https://github.com/calebcoatney/chromakit-ms/releases/tag/v9.9.9"]


def test_same_or_older_release_shows_nothing(qtbot):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app._on_update_found("0.0.1", "https://x")
    assert getattr(app, "update_button", None) is None


def test_startup_check_is_off_in_tests(qtbot, monkeypatch):
    called = []
    monkeypatch.setattr("logic.update_check.latest_release", lambda *a, **k: called.append(1))
    app = ChromaKitApp()
    qtbot.addWidget(app)
    qtbot.wait(50)
    assert called == []
