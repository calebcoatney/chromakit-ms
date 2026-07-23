import numpy as np
import pytest

from logic.integration import integrate_bands
from logic.method import BandWindow
from logic.feature import SpectralFeature
from logic.signal_profiles import SignalProfileRegistry


def _uvvis_profile():
    return SignalProfileRegistry.get("uvvis")


def _ftir_profile():
    return SignalProfileRegistry.get("ftir")


def test_returns_one_feature_per_band_in_order():
    x = np.linspace(300, 600, 301)  # nm
    y = np.ones_like(x)
    bands = [
        BandWindow(name="a", x_min=350, x_max=400),
        BandWindow(name="b", x_min=450, x_max=500),
    ]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert len(feats) == 2
    assert all(isinstance(f, SpectralFeature) for f in feats)
    assert [f.band_assignment for f in feats] == ["a", "b"]


def test_area_of_flat_unit_band_equals_width():
    x = np.linspace(300, 600, 3001)
    y = np.ones_like(x)
    bands = [BandWindow(name="a", x_min=350, x_max=400)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert feats[0].area == pytest.approx(50.0, rel=1e-3)


def test_absorbance_and_position_at_max():
    x = np.linspace(300, 500, 2001)
    y = np.zeros_like(x)
    idx = np.argmin(np.abs(x - 400))
    y[idx] = 2.5
    bands = [BandWindow(name="peak", x_min=350, x_max=450)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert feats[0].absorbance == pytest.approx(2.5, rel=1e-6)
    assert feats[0].position == pytest.approx(400.0, abs=0.2)


def test_inverted_x_ftir_area_is_positive():
    x = np.linspace(2200, 1800, 2001)  # descending wavenumbers
    y = np.ones_like(x)
    bands = [BandWindow(name="co", x_min=1970, x_max=2005)]
    feats = integrate_bands(x, y, bands, _ftir_profile())
    assert feats[0].area == pytest.approx(35.0, rel=1e-2)
    assert feats[0].area > 0


def test_empty_window_emits_zero_feature_with_quality_issue():
    x = np.linspace(300, 500, 201)
    y = np.ones_like(x)
    bands = [BandWindow(name="offscale", x_min=800, x_max=900)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert len(feats) == 1
    assert feats[0].area == 0.0
    assert feats[0].absorbance == 0.0
    assert feats[0].band_assignment == "offscale"
    assert any("no samples" in q.lower() for q in feats[0].quality_issues)


def test_feature_bounds_and_units():
    x = np.linspace(300, 600, 301)
    y = np.ones_like(x)
    bands = [BandWindow(name="a", x_min=350, x_max=400)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    f = feats[0]
    assert f.start == 350
    assert f.end == 400
    assert f.width == pytest.approx(50.0)
    assert f.position_units == "Wavelength (nm)"
