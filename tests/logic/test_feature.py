import pytest
from logic.feature import Feature, SpectralFeature


def test_feature_position_property():
    f = Feature(
        feature_id=1, position=1.23, position_units="min",
        area=100.0, width=0.05, start=1.20, end=1.26,
        start_index=10, end_index=20
    )
    assert f.position == 1.23


def test_feature_defaults():
    f = Feature(
        feature_id=1, position=1.0, position_units="min",
        area=50.0, width=0.02, start=0.99, end=1.01,
        start_index=5, end_index=10
    )
    assert f.is_shoulder is False
    assert f.is_negative is False
    assert f.quality_issues == []


def test_spectral_feature_inherits_feature():
    sf = SpectralFeature(
        feature_id=2, position=1600.0, position_units="cm⁻¹",
        area=250.0, width=10.0, start=1595.0, end=1605.0,
        start_index=100, end_index=110
    )
    assert isinstance(sf, Feature)
    assert sf.position == 1600.0
    assert sf.band_assignment == ""
    assert sf.absorbance == 0.0
    assert sf.transmittance == 0.0


def test_spectral_feature_fields():
    sf = SpectralFeature(
        feature_id=3, position=1720.0, position_units="cm⁻¹",
        area=300.0, width=15.0, start=1712.0, end=1728.0,
        start_index=200, end_index=215,
        band_assignment="C=O stretch", absorbance=0.85
    )
    assert sf.band_assignment == "C=O stretch"
    assert sf.absorbance == 0.85


def test_spectral_feature_as_dict():
    sf = SpectralFeature(
        feature_id=4, position=1600.0, position_units="cm⁻¹",
        area=100.0, width=8.0, start=1596.0, end=1604.0,
        start_index=50, end_index=58,
        band_assignment="aromatic C=C"
    )
    d = sf.as_dict()
    assert d["position"] == 1600.0
    assert d["position_units"] == "cm⁻¹"
    assert d["band_assignment"] == "aromatic C=C"
    assert "area" in d


def test_spectral_feature_as_row():
    sf = SpectralFeature(
        feature_id=5, position=1601.5678, position_units="cm⁻¹",
        area=100.0, width=8.0, start=1597.0, end=1606.0,
        start_index=60, end_index=70
    )
    row = sf.as_row()
    assert isinstance(row, list)
    assert row[0] == round(1601.5678, 2)
