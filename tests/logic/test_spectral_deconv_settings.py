import pytest

from logic.correlation_grouping import CorrelationDeconvParams
from logic.spectral_deconv_runner import WindowGroupingParams
from logic.spectral_deconv_settings import (
    PREFIX, parse_mz_list, load_spectral_deconv_settings, save_spectral_deconv_settings,
)


class FakeSettings:
    """Mimics QSettings.value(key, default, type) / setValue."""

    def __init__(self, data=None):
        self.data = dict(data or {})

    def value(self, key, default=None, type=None):
        if key not in self.data:
            return default
        v = self.data[key]
        return type(v) if type is not None else v

    def setValue(self, key, value):
        self.data[key] = value


def test_defaults_when_empty():
    dp, gp = load_spectral_deconv_settings(FakeSettings())
    assert dp == CorrelationDeconvParams()
    assert gp == WindowGroupingParams()


def test_round_trip():
    dp = CorrelationDeconvParams(noise_percentile=0.7, search_range=0.03, min_peak_height=500.0,
                                 min_top_edge_ratio=2.0, min_points=5, rt_tolerance=0.02,
                                 min_r=0.9, min_ions=3, excluded_mz=[73.0, 147.0])
    gp = WindowGroupingParams(gap_tolerance=0.2, padding_fraction=0.3,
                              rt_match_tolerance=0.04, max_window_peaks=6)
    s = FakeSettings()
    save_spectral_deconv_settings(s, dp, gp)
    assert s.data[PREFIX + "excluded_mz"] == "73.0, 147.0"
    assert load_spectral_deconv_settings(s) == (dp, gp)


def test_gap_zero_means_auto():
    s = FakeSettings({PREFIX + "gap_tolerance": 0.0})
    _, gp = load_spectral_deconv_settings(s)
    assert gp.gap_tolerance is None
    save_spectral_deconv_settings(s, CorrelationDeconvParams(), WindowGroupingParams())
    assert s.data[PREFIX + "gap_tolerance"] == 0.0


def test_old_adap_keys_ignored():
    s = FakeSettings({PREFIX + "shape_sim_threshold": 18.0, PREFIX + "min_cluster_size": 7,
                      PREFIX + "model_peak_choice": "mz"})
    dp, _ = load_spectral_deconv_settings(s)
    assert dp == CorrelationDeconvParams()


def test_parse_mz_list_lenient_skips_garbage():
    assert parse_mz_list("73, abc, , 147.5") == [73.0, 147.5]


def test_parse_mz_list_strict_raises():
    with pytest.raises(ValueError, match="abc"):
        parse_mz_list("73, abc", strict=True)


def test_load_skips_bad_stored_excluded_mz():
    dp, _ = load_spectral_deconv_settings(FakeSettings({PREFIX + "excluded_mz": "73, junk"}))
    assert dp.excluded_mz == [73.0]


@pytest.mark.parametrize("text", ["73 147 207", "73, 147, 207", "73,147 207", " 73\t147 ,, 207 "])
def test_parse_mz_list_accepts_commas_and_or_spaces(text):
    assert parse_mz_list(text, strict=True) == [73.0, 147.0, 207.0]
    assert parse_mz_list(text) == [73.0, 147.0, 207.0]


def test_parse_mz_list_semicolon_is_not_a_separator():
    with pytest.raises(ValueError, match="73;147"):
        parse_mz_list("73;147", strict=True)


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "0", "-5", "0.0"])
def test_parse_mz_list_strict_rejects_non_finite_and_non_positive(token):
    with pytest.raises(ValueError, match=token):
        parse_mz_list(f"73, {token}", strict=True)


def test_parse_mz_list_lenient_skips_non_finite_and_non_positive():
    assert parse_mz_list("73, nan, -5, 0, inf, 147") == [73.0, 147.0]
