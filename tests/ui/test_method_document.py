import pytest
pytest.importorskip('pytestqt')

from ui.app import ChromaKitApp
from ui.frames.ms import MSFrame


@pytest.fixture(autouse=True)
def _no_ms_autoload(monkeypatch):
    """Neutralize MSFrame's background NIST library loader thread.

    ChromaKitApp constructs an MSFrame, which schedules a QThread that
    deserializes the full NIST14 library via ms-toolkit. Under pytest-qt
    (event loop pumped between tests + GC), that native load races and
    segfaults the interpreter. The document-model tests don't need the
    library, so we stub the auto-load to a no-op.
    """
    monkeypatch.setattr(MSFrame, "_try_autoload_library", lambda self: None)


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
