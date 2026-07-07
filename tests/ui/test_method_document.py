import pytest
pytest.importorskip('pytestqt')

from ui.app import ChromaKitApp


def _make(qtbot):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    return app


def test_boots_with_untitled_method(qtbot):
    app = _make(qtbot)
    assert app.current_method is not None
    assert app.current_method.name == "Untitled"
    assert app.current_method.signal_type == "gc"
    assert app.current_method_path is None
    assert app._method_dirty is False


def test_title_shows_untitled(qtbot):
    app = _make(qtbot)
    assert "Untitled" in app.windowTitle()
    assert "*" not in app.windowTitle()


def test_mark_dirty_updates_title(qtbot):
    app = _make(qtbot)
    app._mark_dirty(True)
    assert app._method_dirty is True
    assert "Untitled*" in app.windowTitle()


def test_apply_method_to_frames_guards_writeback(qtbot):
    app = _make(qtbot)
    # Programmatic population must not flip dirty.
    app._method_dirty = False
    app._apply_method_to_frames()
    assert app._method_dirty is False
