"""Tests for the area_factor guard in the integration layer."""
import logging
import numpy as np
import pytest

from logic.integration import Integrator


def _simple_processed():
    """One clean Gaussian peak on a flat baseline, pre-detected."""
    x = np.linspace(0, 10, 1001)
    y = 100.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.1 ** 2))
    apex_idx = int(np.argmax(y))
    return {
        "x": x,
        "original_y": y,
        "corrected_y": y,
        "baseline_y": np.zeros_like(y),
        "peaks_x": np.array([x[apex_idx]]),
        "peaks_y": np.array([y[apex_idx]]),
        "peak_metadata": [{"left_base": apex_idx - 40, "right_base": apex_idx + 40}],
    }


def _area(processed, area_factor):
    result = Integrator.integrate(
        processed_data=processed, area_factor=area_factor, verbose=False,
    )
    peak = result["peaks"][0]
    return getattr(peak, "integrator_area", None) or getattr(peak, "area", None)


def test_none_area_factor_is_identity():
    p = _simple_processed()
    base = _area(p, None)
    assert base > 0


def test_area_factor_scales():
    p = _simple_processed()
    base = _area(p, None)
    scaled = _area(p, 600.0)
    assert scaled == pytest.approx(base * 600.0, rel=1e-9)


def test_zero_area_factor_is_identity_with_warning(caplog):
    p = _simple_processed()
    base = _area(p, None)
    with caplog.at_level(logging.WARNING):
        zeroed = _area(p, 0.0)
    assert zeroed == pytest.approx(base, rel=1e-9)
    assert any("area_factor" in r.message and "1" in r.message for r in caplog.records)
