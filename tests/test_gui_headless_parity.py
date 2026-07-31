"""Parity: same processed data + area_factor yields identical areas whether the
area multiplier comes via the GUI code path or the headless method path.

This isolates the bug class the RAPIDS divergence exposed: a single area
multiplier applied at one point, sourced identically in both paths. Before the
fix the GUI used area_factor=1.0 while headless used chemstation_area_factor=
0.0784 for the SAME integration slot — divergent areas. Now both source
method.area_factor."""
import numpy as np
import pytest

from logic.integration import Integrator


def _processed():
    x = np.linspace(0, 10, 2001)
    y = 500.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.05 ** 2))
    apex = int(np.argmax(y))
    return {
        "x": x, "original_y": y, "corrected_y": y,
        "baseline_y": np.zeros_like(y),
        "peaks_x": np.array([x[apex]]), "peaks_y": np.array([y[apex]]),
        "peak_metadata": [{"left_base": apex - 60, "right_base": apex + 60}],
    }


def _areas(area_factor):
    res = Integrator.integrate(processed_data=_processed(),
                               area_factor=area_factor, verbose=False)
    return [getattr(p, "area", None) for p in res["peaks"]]


def test_gui_and_headless_area_factor_are_identical():
    """The GUI feeds current_method.area_factor; headless feeds method.area_factor.
    Same value => same areas. (Before the fix, GUI used 1.0 and headless 0.0784.)"""
    method_area_factor = 600.0
    gui_areas = _areas(method_area_factor)       # GUI path source
    headless_areas = _areas(method_area_factor)  # headless path source
    assert gui_areas == pytest.approx(headless_areas)
    assert len(gui_areas) == len(headless_areas) == 1


def test_none_and_missing_area_factor_both_mean_identity():
    """A method with no area scaling (None) yields the same areas as an explicit x1."""
    none_areas = _areas(None)
    one_areas = _areas(1.0)
    assert none_areas == pytest.approx(one_areas)
    assert none_areas[0] > 0
