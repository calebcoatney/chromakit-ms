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


@pytest.fixture
def setup():
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
