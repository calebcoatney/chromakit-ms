"""MS Options: the weighting dropdown, its stored format and the NIST11 default.

Every QSettings here is a temp INI (the module's QSettings is monkeypatched),
never Caleb's real ChromaKit preferences.
"""
import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings


@pytest.fixture
def ini_settings(tmp_path, monkeypatch):
    path = str(tmp_path / "t.ini")
    import ui.dialogs.ms_options_dialog as mod
    monkeypatch.setattr(mod, "QSettings", lambda *a, **k: QSettings(path, QSettings.IniFormat))
    return path


def _store(path):
    return QSettings(path, QSettings.IniFormat)


def _dialog(qtbot):
    from ui.dialogs.ms_options_dialog import MSOptionsDialog
    dlg = MSOptionsDialog()
    qtbot.addWidget(dlg)
    return dlg


def _scheme(dlg):
    return dlg.weighting_combo.currentData()


def test_every_dropdown_option_is_a_real_toolkit_scheme(qtbot, ini_settings):
    preprocessing = pytest.importorskip("ms_toolkit.preprocessing")
    dlg = _dialog(qtbot)
    names = [dlg.weighting_combo.itemData(i) for i in range(dlg.weighting_combo.count())]
    assert names and set(names) <= set(preprocessing.weighting_schemes)
    assert "NIST" not in names


def test_options_are_labelled_with_their_exponents(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    combo = dlg.weighting_combo
    labels = {combo.itemData(i): combo.itemText(i) for i in range(combo.count())}
    assert labels["NIST11"] == "NIST11 (m/z^1.3 · I^0.53)"
    assert labels["NIST_GC"] == "NIST_GC (m/z^3 · I^0.6)"


def test_nist11_is_selectable(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    dlg.weighting_combo.setCurrentIndex(dlg.weighting_combo.findData("NIST_GC"))
    assert dlg.get_options()["weighting"] == "NIST_GC"
    dlg.weighting_combo.setCurrentIndex(dlg.weighting_combo.findData("NIST11"))
    assert dlg.get_options()["weighting"] == "NIST11"


def test_every_option_is_what_get_options_returns(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    for i in range(dlg.weighting_combo.count()):
        dlg.weighting_combo.setCurrentIndex(i)
        assert dlg.get_options()["weighting"] == dlg.weighting_combo.itemData(i)


def test_default_is_vector_search_with_nist11(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    opts = dlg.get_options()
    assert opts["search_method"] == "vector"
    assert opts["weighting"] == "NIST11"
    assert _scheme(dlg) == "NIST11"


def test_restore_defaults_is_vector_search_with_nist11(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    dlg.search_method_combo.setCurrentIndex(1)       # Word2Vec
    dlg.weighting_combo.setCurrentIndex(dlg.weighting_combo.findData("None"))
    dlg._restore_defaults()
    opts = dlg.get_options()
    assert opts["search_method"] == "vector"
    assert opts["weighting"] == "NIST11"


def test_the_other_algorithm_defaults_are_unchanged(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    dlg._restore_defaults()
    opts = dlg.get_options()
    assert opts["similarity"] == "composite"
    assert opts["unmatched"] == "keep_all"
    assert opts["top_k_clusters"] == 10
    assert dlg.top_k_clusters_spin.value() == 10


def test_nist11_round_trips_through_save_and_load_as_a_name(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    dlg.weighting_combo.setCurrentIndex(dlg.weighting_combo.findData("NIST_GC"))
    dlg._save_settings()
    dlg.weighting_combo.setCurrentIndex(dlg.weighting_combo.findData("NIST11"))
    dlg._save_settings()
    store = _store(ini_settings)
    store.sync()
    assert store.value("ms_search/weighting_scheme") == "NIST11"
    assert not store.contains("ms_search/weighting")

    again = _dialog(qtbot)
    assert _scheme(again) == "NIST11"


@pytest.mark.parametrize("name", ["NIST11", "NIST_GC", "MassBank", "SQRT", "None"])
def test_each_scheme_round_trips(qtbot, ini_settings, name):
    dlg = _dialog(qtbot)
    dlg.weighting_combo.setCurrentIndex(dlg.weighting_combo.findData(name))
    dlg._save_settings()
    assert _scheme(_dialog(qtbot)) == name


@pytest.mark.parametrize("index, expected", [(0, "None"), (1, "NIST11"), (2, "NIST_GC")])
def test_old_stored_index_loads_as_a_name(qtbot, ini_settings, index, expected):
    store = _store(ini_settings)
    store.setValue("ms_search/weighting", index)
    store.sync()
    dlg = _dialog(qtbot)
    assert _scheme(dlg) == expected
    assert dlg.get_options()["weighting"] == expected


def test_old_index_is_rewritten_as_a_name_on_save(qtbot, ini_settings):
    store = _store(ini_settings)
    store.setValue("ms_search/weighting", 2)
    store.sync()
    dlg = _dialog(qtbot)
    dlg._save_settings()
    store = _store(ini_settings)
    store.sync()
    assert store.value("ms_search/weighting_scheme") == "NIST_GC"
    assert not store.contains("ms_search/weighting")


def test_saved_search_method_is_not_disturbed(qtbot, ini_settings):
    store = _store(ini_settings)
    store.setValue("ms_search/method", 1)            # Word2Vec, as a user may have saved
    store.sync()
    dlg = _dialog(qtbot)
    assert dlg.get_options()["search_method"] == "w2v"
    assert _scheme(dlg) == "NIST11"


def test_weighting_tooltip_explains_the_default(qtbot, ini_settings):
    dlg = _dialog(qtbot)
    tip = dlg.weighting_combo.toolTip()
    assert "NIST11" in tip and "default" in tip.lower()
    assert "81.7" in tip and "75.3" in tip and "NIST_GC" in tip
    assert "FPO" in tip
    assert "Clusters to Consider" in tip and "50" in tip
    assert "4" in tip
