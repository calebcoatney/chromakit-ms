"""Shared pytest fixtures for ChromaKit test suite."""
import pytest
from logic.signal_profiles import SignalProfileRegistry


@pytest.fixture
def isolated_registry():
    """Clear the profile registry for this test, then restore built-ins afterwards."""
    saved = dict(SignalProfileRegistry._profiles)
    SignalProfileRegistry._profiles.clear()
    yield
    SignalProfileRegistry._profiles.clear()
    SignalProfileRegistry._profiles.update(saved)


import os
import pathlib


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "realdata: needs CHROMAKIT_GCXGC_RUN pointing at a SepSolve .C folder",
    )


@pytest.fixture
def real_gcxgc_run() -> pathlib.Path:
    """The real GCxGC run named by CHROMAKIT_GCXGC_RUN, or skip."""
    path = os.environ.get("CHROMAKIT_GCXGC_RUN")
    if not path or not os.path.isdir(path):
        pytest.skip("CHROMAKIT_GCXGC_RUN is not set to a .C folder")
    return pathlib.Path(path)


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch):
    """Tests never call GitHub or auto-load the user's saved MS library."""
    monkeypatch.setenv("CHROMAKIT_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("CHROMAKIT_NO_MS_AUTOLOAD", "1")


@pytest.fixture(autouse=True)
def _isolated_batch_settings(tmp_path, monkeypatch):
    """Tests that build BatchJobDialog never touch the user's real QSettings."""
    try:
        from PySide6.QtCore import QSettings
        from ui.dialogs import batch_job_dialog
    except ImportError:  # non-Qt environment
        return
    ini = str(tmp_path / "batch_dialog_settings.ini")
    monkeypatch.setattr(batch_job_dialog, "_settings",
                        lambda: QSettings(ini, QSettings.IniFormat))
