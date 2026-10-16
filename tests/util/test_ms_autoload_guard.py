import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QThread
from PySide6.QtTest import QTest

from ui.app import ChromaKitApp


class _SavedPaths:
    def __init__(self, path):
        self._path = str(path)

    def value(self, key, default=None):
        return self._path if key.startswith("ms/") else default


class _RecordingThread:
    """Stands in for LibraryLoadThread; records the start instead of loading."""
    started = 0

    def __init__(self, *a, **k):
        from PySide6.QtCore import QObject, Signal

        class _Sig(QObject):
            progress_update = Signal(str, int)
            finished = Signal(bool, str)
        self._sig = _Sig()
        self.progress_update = self._sig.progress_update
        self.finished = self._sig.finished

    def start(self):
        type(self).started += 1

    def isRunning(self):
        return False


@pytest.fixture
def app_with_saved_library(qtbot, tmp_path, monkeypatch):
    f = tmp_path / "lib.json"
    f.write_text("{}")
    _RecordingThread.started = 0
    monkeypatch.setattr("ui.frames.ms.LibraryLoadThread", _RecordingThread)
    win = ChromaKitApp()
    qtbot.addWidget(win)
    monkeypatch.setattr(win.ms_frame, "settings", _SavedPaths(f))
    return win


def test_autoload_is_skipped_when_the_env_var_is_set(app_with_saved_library, monkeypatch):
    monkeypatch.setenv("CHROMAKIT_NO_MS_AUTOLOAD", "1")
    app_with_saved_library.ms_frame._try_autoload_library()
    QTest.qWait(250)
    assert _RecordingThread.started == 0


def test_autoload_runs_without_the_env_var(app_with_saved_library, monkeypatch):
    monkeypatch.delenv("CHROMAKIT_NO_MS_AUTOLOAD", raising=False)
    app_with_saved_library.ms_frame._try_autoload_library()
    assert _RecordingThread.started == 1


def test_closing_the_app_waits_for_a_running_library_load(qtbot, monkeypatch):
    win = ChromaKitApp()
    qtbot.addWidget(win)

    class SlowLoad(QThread):
        done = False

        def run(self):
            self.msleep(300)
            SlowLoad.done = True

    t = SlowLoad()
    win.ms_frame.load_thread = t
    t.start()
    assert t.isRunning()
    win.close()
    assert SlowLoad.done and t.isFinished()
