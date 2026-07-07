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
