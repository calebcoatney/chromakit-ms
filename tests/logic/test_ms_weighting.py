"""The MS library-search weighting setting: names, labels, storage, default.

ChromaKit passes the scheme *name* to ms-toolkit's search_vector /
search_hybrid. Only the keys of ms_toolkit.preprocessing.weighting_schemes are
real names; any other string silently searches unweighted. These tests pin
every option ChromaKit offers to a real key, the stored format (a name, with
the old integer index still readable), and the NIST11 default.

QSettings here are temp INI files, never the real ChromaKit preferences.
"""
import pytest
from PySide6.QtCore import QSettings

import logic.ms_weighting as mw


@pytest.fixture
def ini(tmp_path):
    return QSettings(str(tmp_path / "t.ini"), QSettings.IniFormat)


def _toolkit_schemes():
    preprocessing = pytest.importorskip("ms_toolkit.preprocessing")
    return preprocessing.weighting_schemes


def test_default_is_nist11():
    assert mw.DEFAULT_WEIGHTING == "NIST11"


def test_every_option_is_a_real_toolkit_scheme():
    schemes = _toolkit_schemes()
    names = mw.weighting_scheme_names()
    assert names, "the dropdown has no options"
    assert set(names) <= set(schemes)


def test_options_cover_every_toolkit_scheme():
    # Built from the toolkit's own dict, so a scheme added there appears here.
    assert set(mw.weighting_scheme_names()) == set(_toolkit_schemes())


def test_nist11_and_nist_gc_are_selectable_and_nist_is_gone():
    names = mw.weighting_scheme_names()
    assert "NIST11" in names and "NIST_GC" in names
    assert "NIST" not in names


def test_default_scheme_is_listed_first():
    assert mw.weighting_scheme_names()[0] == mw.DEFAULT_WEIGHTING


def test_fallback_table_matches_the_toolkit():
    """The built-in copy (used when ms-toolkit is not installed) must not drift."""
    assert dict(mw._FALLBACK_SCHEMES) == dict(_toolkit_schemes())


def test_labels_state_the_exponents():
    assert mw.weighting_label("NIST11") == "NIST11 (m/z^1.3 · I^0.53)"
    assert mw.weighting_label("NIST_GC") == "NIST_GC (m/z^3 · I^0.6)"
    assert mw.weighting_label("MassBank") == "MassBank (m/z^2 · I^0.5)"
    assert mw.weighting_label("SQRT") == "SQRT (I^0.5)"
    assert mw.weighting_label("None") == "None (unweighted)"


def test_labels_come_from_the_toolkit_dict():
    schemes = _toolkit_schemes()
    for name, (a, b) in schemes.items():
        label = mw.weighting_label(name)
        assert label.startswith(name)
        if (a, b) == (0, 1):          # "None": plain intensity, no exponents to state
            assert "unweighted" in label
            continue
        assert f"I^{b:g}" in label
        if a:
            assert f"m/z^{a:g}" in label


def test_write_then_read_round_trips_a_name(ini):
    for name in mw.weighting_scheme_names():
        mw.write_weighting_scheme(ini, name)
        ini.sync()
        assert mw.read_weighting_scheme(ini) == name


def test_stored_format_is_the_name_under_a_new_key(ini):
    mw.write_weighting_scheme(ini, "NIST11")
    ini.sync()
    assert ini.value("ms_search/weighting_scheme") == "NIST11"


def test_write_rejects_a_name_the_toolkit_does_not_know(ini):
    with pytest.raises(ValueError):
        mw.write_weighting_scheme(ini, "NIST")


def test_nothing_stored_reads_the_default(ini):
    assert mw.read_weighting_scheme(ini) == "NIST11"


@pytest.mark.parametrize("index, expected", [(0, "None"), (1, "NIST11"), (2, "NIST_GC")])
def test_old_integer_index_reads_as_a_name(ini, index, expected):
    ini.setValue("ms_search/weighting", index)
    ini.sync()
    assert mw.read_weighting_scheme(ini) == expected


def test_old_index_given_as_text_still_reads(ini):
    # An INI/plist round trip can hand back "2" rather than 2.
    ini.setValue("ms_search/weighting", "2")
    ini.sync()
    assert mw.read_weighting_scheme(ini) == "NIST_GC"


@pytest.mark.parametrize("junk", [3, -1, 99, "abc", ""])
def test_unreadable_old_index_reads_the_default(ini, junk):
    ini.setValue("ms_search/weighting", junk)
    ini.sync()
    assert mw.read_weighting_scheme(ini) == "NIST11"


def test_new_key_wins_over_old_index(ini):
    ini.setValue("ms_search/weighting", 0)
    ini.setValue("ms_search/weighting_scheme", "NIST_GC")
    ini.sync()
    assert mw.read_weighting_scheme(ini) == "NIST_GC"


def test_unknown_stored_name_falls_through_to_the_old_index_then_default(ini):
    ini.setValue("ms_search/weighting_scheme", "NIST")
    ini.sync()
    assert mw.read_weighting_scheme(ini) == "NIST11"
    ini.setValue("ms_search/weighting", 2)
    ini.sync()
    assert mw.read_weighting_scheme(ini) == "NIST_GC"


def test_saving_retires_the_old_integer_key(ini):
    ini.setValue("ms_search/weighting", 2)
    ini.sync()
    mw.write_weighting_scheme(ini, "NIST11")
    ini.sync()
    assert not ini.contains("ms_search/weighting")
    assert mw.read_weighting_scheme(ini) == "NIST11"

