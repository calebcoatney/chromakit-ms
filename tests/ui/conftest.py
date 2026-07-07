"""Shared fixtures for ui/ tests.

Scoped to tests/ui/ so logic/ and api/ tests (which never build the Qt app)
are unaffected.
"""
import pytest

pytest.importorskip("pytestqt")


@pytest.fixture(autouse=True)
def _no_ms_autoload(monkeypatch):
    """Neutralize MSFrame's background NIST library loader thread.

    ChromaKitApp constructs an MSFrame, which schedules a QThread that
    deserializes the full NIST14 library via ms-toolkit. Under pytest-qt
    (event loop pumped between tests + GC, repeated app construction), that
    native load races and segfaults the interpreter. UI tests don't need the
    library, so we stub the auto-load to a no-op for every ui/ test.
    """
    from ui.frames.ms import MSFrame
    monkeypatch.setattr(MSFrame, "_try_autoload_library", lambda self: None)


@pytest.fixture(autouse=True)
def _no_close_prompt(monkeypatch):
    """Neutralize the document dirty-save prompt during teardown.

    ChromaKitApp.closeEvent calls _maybe_prompt_save(), which pops a modal
    QMessageBox.question when the method is dirty. Tests routinely leave a
    window dirty (e.g. mark_dirty checks), and pytest-qt closes every managed
    widget at teardown -- headlessly the modal blocks forever and hangs the
    run. Stub the prompt to "proceed" (return True) for every ui/ test.

    Tests that need to exercise a specific prompt outcome override
    _maybe_prompt_save on the instance themselves (this class-level default is
    compatible with such per-instance monkeypatches).
    """
    from ui.app import ChromaKitApp
    monkeypatch.setattr(ChromaKitApp, "_maybe_prompt_save", lambda self: True)
