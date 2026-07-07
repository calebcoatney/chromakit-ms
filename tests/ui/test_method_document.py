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


def test_param_edit_preserves_method_metadata(qtbot):
    from logic.method import ChromaMethod
    app = _make(qtbot)
    original = ChromaMethod(name="Loaded", signal_type="gc", version="9")
    fixed_created = original.created_at  # capture the (fresh) creation time
    app.current_method = original
    app.current_method_path = None
    app._method_dirty = False
    # Simulate a user parameter edit: drive the write-back slot directly with
    # the frame's current params (this is what parameters_changed carries).
    app._loading_method = False
    app._on_params_writeback(app.parameters_frame.current_params)
    assert app.current_method.version == "9"
    assert app.current_method.created_at == fixed_created
    assert app.current_method.name == "Loaded"          # identity preserved
    assert app.current_method.signal_type == "gc"
    assert app._method_dirty is True                     # edit still marks dirty
