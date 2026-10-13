"""MSFrame's own copy of the search options: default and loading from settings.

MSFrame reads the same QSettings keys as the MS Options dialog. Here both
QSettings names are redirected to a temp INI so nothing touches Caleb's real
ChromaKit preferences.
"""
import types

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings


@pytest.fixture
def ini_settings(tmp_path, monkeypatch):
    path = str(tmp_path / "t.ini")
    factory = lambda *a, **k: QSettings(path, QSettings.IniFormat)
    import PySide6.QtCore as qtcore
    import ui.frames.ms as ms_mod
    monkeypatch.setattr(qtcore, "QSettings", factory)   # _load_search_options_from_settings imports it per call
    monkeypatch.setattr(ms_mod, "QSettings", factory)   # MSFrame.__init__
    return path


def _loaded_options(ini_settings):
    """Run the real loader against a bare options dict (no widgets needed)."""
    from ui.frames.ms import MSFrame
    fake = types.SimpleNamespace(search_options={})
    MSFrame._load_search_options_from_settings(fake)
    return fake.search_options


@pytest.mark.parametrize("index, expected", [(0, "None"), (1, "NIST11"), (2, "NIST_GC")])
def test_old_stored_index_loads_as_a_name(ini_settings, index, expected):
    store = QSettings(ini_settings, QSettings.IniFormat)
    store.setValue("ms_search/weighting", index)
    store.sync()
    assert _loaded_options(ini_settings)["weighting"] == expected


@pytest.mark.parametrize("name", ["NIST11", "NIST_GC", "MassBank", "SQRT", "None"])
def test_stored_name_loads_as_that_name(ini_settings, name):
    store = QSettings(ini_settings, QSettings.IniFormat)
    store.setValue("ms_search/weighting_scheme", name)
    store.sync()
    assert _loaded_options(ini_settings)["weighting"] == name


def test_nothing_stored_loads_vector_with_nist11(ini_settings):
    opts = _loaded_options(ini_settings)
    assert opts["search_method"] == "vector"
    assert opts["weighting"] == "NIST11"


def test_a_saved_word2vec_method_still_loads_as_word2vec(ini_settings):
    store = QSettings(ini_settings, QSettings.IniFormat)
    store.setValue("ms_search/method", 1)
    store.sync()
    assert _loaded_options(ini_settings)["search_method"] == "w2v"


def test_frame_starts_with_vector_and_nist11(qtbot, ini_settings):
    from ui.frames.ms import MSFrame
    frame = MSFrame()
    qtbot.addWidget(frame)
    assert frame.search_options["search_method"] == "vector"
    assert frame.search_options["weighting"] == "NIST11"

