"""Peak range filters must constrain integration, not just which apexes survive.

Reported by Anh at the 2026-11-10 demo:
  1. range filters still allow peak bounds outside the time window;
  2. when the left neighbour of the first in-window peak is excluded by the
     filter, the valley between them is no longer used as the left bound,
     so the peak integrates across its excluded neighbour.
"""
import copy

import numpy as np
import pytest

from logic.processor import ChromatogramProcessor


def build_setup():
    """(processor, x, y, params) for two overlapping peaks and one isolated peak, no baseline or smoothing."""
    proc = ChromatogramProcessor()
    x = np.linspace(0, 10, 6001)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    # Two overlapping peaks (valley well above baseline) and one isolated peak.
    y = g(4.0, 0.08, 100) + g(4.35, 0.08, 80) + g(7.0, 0.08, 50)
    params = copy.deepcopy(proc.default_params)
    params['baseline']['enabled'] = False
    params['smoothing']['enabled'] = False
    return proc, x, y, params


@pytest.fixture
def setup():
    return build_setup()


def _integrate(proc, x, y, params):
    processed = proc.process(x, y, params)
    return proc.integrate_peaks(processed)['peaks']


def test_bounds_stay_inside_range_filter_window(setup):
    proc, x, y, params = setup
    params['peaks']['range_filters'] = [[6.9, 7.03]]
    peaks = _integrate(proc, x, y, params)
    assert len(peaks) == 1
    assert peaks[0].start_time >= 6.9
    assert peaks[0].end_time <= 7.03


def test_valley_to_excluded_neighbour_is_left_bound(setup):
    proc, x, y, params = setup
    # Unfiltered: the 4.35 peak's left bound is the valley between the pair.
    params['peaks']['range_filters'] = []
    unfiltered = {round(p.retention_time, 2): p for p in _integrate(proc, x, y, params)}
    valley = unfiltered[4.35].start_time

    # Filter cuts out the 4.0 peak; the window starts left of the valley so
    # the window clamp alone would not hide the bug.
    params['peaks']['range_filters'] = [[4.1, 5.0]]
    peaks = _integrate(proc, x, y, params)
    assert [round(p.retention_time, 2) for p in peaks] == [4.35]
    assert peaks[0].start_time == pytest.approx(valley)
    assert peaks[0].area == pytest.approx(unfiltered[4.35].area)


def test_no_range_filter_unchanged(setup):
    proc, x, y, params = setup
    params['peaks']['range_filters'] = []
    peaks = _integrate(proc, x, y, params)
    assert [round(p.retention_time, 2) for p in peaks] == [4.0, 4.35, 7.0]


def test_valley_to_ms_range_dropped_neighbour_is_left_bound(setup):
    proc, x, y, params = setup
    params['peaks']['range_filters'] = []
    unfiltered = {round(p.retention_time, 2): p for p in _integrate(proc, x, y, params)}
    valley = unfiltered[4.35].start_time

    # MS data starts at 4.1 (solvent delay): the 4.0 apex is dropped by the
    # MS-range filter, which must carry it as a valley neighbour just like a
    # range-filter exclusion does.
    processed = proc.process(x, y, params, ms_range=(4.1, 10.0))
    assert 4.0 in np.round(processed['range_excluded_peaks_x'], 2)
    peaks = proc.integrate_peaks(processed)['peaks']
    kept = {round(p.retention_time, 2): p for p in peaks}
    assert 4.0 not in kept
    assert kept[4.35].start_time == pytest.approx(valley)
    assert kept[4.35].area == pytest.approx(unfiltered[4.35].area)


def test_ms_range_dropped_neighbour_in_masked_region_keeps_area_finite():
    """Baseline on + ms_range masks corrected_y to NaN before the MS window.

    A negative peak detected in that masked region is dropped by the MS-range
    filter and carried as a valley neighbour. The valley search between it and
    the first kept peak must skip the NaN samples; otherwise the bound lands
    in the masked region and the exported area is NaN.
    """
    proc = ChromatogramProcessor()
    x = np.linspace(0, 10, 10000)

    def g(c, s, a):
        return a * np.exp(-0.5 * ((x - c) / s) ** 2)

    # Solvent peak, a dip after it (before MS-on), a solvent tail, the first
    # analyte just after MS-on, and a later peak.
    y = (g(1.0, 0.1, 1000) - g(1.6, 0.05, 60)
         + 40 * np.exp(-(x - 1.0) / 0.8) * (x > 1.0)
         + g(2.3, 0.05, 100) + g(5, 0.05, 80) + 5)
    params = copy.deepcopy(proc.default_params)
    params['baseline']['enabled'] = True
    params['smoothing']['enabled'] = True
    params['negative_peaks'] = dict(params.get('negative_peaks', {}),
                                    enabled=True, min_prominence=0.01)

    processed = proc.process(x, y, params, ms_range=(2.28, 10.0))
    assert np.isnan(processed['corrected_y']).any()
    excluded = np.asarray(processed['range_excluded_peaks_x'])
    in_masked = excluded[np.isnan(processed['corrected_y'][np.searchsorted(x, excluded)])]
    assert len(in_masked) > 0, "fixture must carry an apex from the NaN region"

    with_carry = proc.integrate_peaks(dict(processed))['peaks']
    no_carry_data = dict(processed)
    no_carry_data['range_excluded_peaks_x'] = np.array([])
    without_carry = proc.integrate_peaks(no_carry_data)['peaks']

    assert len(with_carry) == len(without_carry) > 0
    for p, q in zip(with_carry, without_carry):
        assert np.isfinite(p.area), f"NaN area at RT {p.retention_time:.3f}"
        assert p.start_time == pytest.approx(q.start_time)
        assert p.area == pytest.approx(q.area)
