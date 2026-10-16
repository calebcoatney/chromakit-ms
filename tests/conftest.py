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
    """Tests never call GitHub."""
    monkeypatch.setenv("CHROMAKIT_NO_UPDATE_CHECK", "1")
