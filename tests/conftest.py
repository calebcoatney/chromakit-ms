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
