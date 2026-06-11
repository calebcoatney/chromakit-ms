"""Tests for api/ms_toolkit_singleton.

Mocks out the actual MSToolkit class so the tests run without ms-toolkit-nrel
or any library files. We test the singleton lifecycle, not the library load.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch

from api import ms_toolkit_singleton as singleton


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level state before and after each test."""
    singleton._toolkit = None
    singleton._loaded_paths = None
    yield
    singleton._toolkit = None
    singleton._loaded_paths = None


def test_is_loaded_returns_false_initially():
    assert singleton.is_loaded() is False
    assert singleton.loaded_paths() is None


def test_get_toolkit_raises_when_not_loaded():
    with pytest.raises(RuntimeError, match="not loaded"):
        singleton.get_toolkit()


def test_load_library_creates_toolkit_and_returns_summary(tmp_path):
    """load_library() returns a status dict and marks the singleton loaded."""
    cache_file = tmp_path / "lib.json"
    cache_file.write_text('{"Hexane": {"formula": "C6H14", "mw": 86.18, "spectrum": {}}}')

    fake_toolkit = MagicMock()
    fake_toolkit.library = {}
    fake_compound_cls = MagicMock()
    fake_compound_cls.from_json.return_value = MagicMock()

    with patch.object(singleton, 'MSToolkit', return_value=fake_toolkit), \
         patch.object(singleton, 'Compound', fake_compound_cls):
        result = singleton.load_library(
            library_path=str(cache_file),
            cache_path=str(cache_file),
            preselector_path=None,
            w2v_path=None,
        )

    assert singleton.is_loaded() is True
    assert result['status'] == 'loaded'
    assert result['compound_count'] == 1
    assert result['preselector_loaded'] is False
    assert result['w2v_loaded'] is False
    assert 'elapsed_seconds' in result
    assert singleton.loaded_paths()['library_path'] == str(cache_file)


def test_load_library_raises_when_cache_missing():
    """A nonexistent cache file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        singleton.load_library(
            library_path="/nonexistent/lib.json",
            cache_path=None,
            preselector_path=None,
            w2v_path=None,
        )


def test_load_library_replaces_previous_load(tmp_path):
    """A second load_library() call swaps the singleton in place."""
    cache1 = tmp_path / "lib1.json"
    cache1.write_text('{"Hexane": {"formula": "C6H14"}}')
    cache2 = tmp_path / "lib2.json"
    cache2.write_text('{"Benzene": {"formula": "C6H6"}, "Toluene": {"formula": "C7H8"}}')

    fake_toolkit = MagicMock()
    fake_toolkit.library = {}
    fake_compound_cls = MagicMock()
    fake_compound_cls.from_json.return_value = MagicMock()

    with patch.object(singleton, 'MSToolkit', return_value=fake_toolkit), \
         patch.object(singleton, 'Compound', fake_compound_cls):
        r1 = singleton.load_library(str(cache1), str(cache1), None, None)
        assert r1['compound_count'] == 1

        r2 = singleton.load_library(str(cache2), str(cache2), None, None)
        assert r2['compound_count'] == 2

    assert singleton.loaded_paths()['library_path'] == str(cache2)


def test_load_library_loads_preselector_when_provided(tmp_path):
    """When preselector_path exists, it's loaded onto the toolkit."""
    cache_file = tmp_path / "lib.json"
    cache_file.write_text('{}')
    preselector_file = tmp_path / "pre.pkl"
    preselector_file.write_bytes(b'\x80\x04\x95\x00')  # valid pickle prefix bytes

    fake_toolkit = MagicMock()
    fake_toolkit.library = {}
    fake_compound_cls = MagicMock()

    with patch.object(singleton, 'MSToolkit', return_value=fake_toolkit), \
         patch.object(singleton, 'Compound', fake_compound_cls):
        result = singleton.load_library(
            library_path=str(cache_file),
            cache_path=str(cache_file),
            preselector_path=str(preselector_file),
            w2v_path=None,
        )

    fake_toolkit.load_preselector.assert_called_once_with(str(preselector_file))
    assert result['preselector_loaded'] is True
