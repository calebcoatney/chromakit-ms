"""Tests for logic.ion_peak_resolver (port of mzmine MinimumSearchFeatureResolver).

Each test names the mzmine behaviour it mirrors:
MinimumSearchFeatureResolver.resolve / tryToFinalizePeak / checkPeakShape /
adjustStartAndEnd, and MathUtils.calcQuantile / calcQuantileSorted.
"""
import numpy as np
import pytest

from logic.ion_peak_resolver import (
    IonPeak, mzmine_quantile, whole_run_noise_levels, resolve_ion_peaks,
)

X = np.round(np.arange(0.0, 2.0, 0.005), 6)     # 400 scans, 0.005 min apart
KW = dict(search_range=0.05, min_peak_height=1000.0, min_top_edge_ratio=1.7, min_points=4)


def g(center, height, sigma=0.015):
    return height * np.exp(-0.5 * ((X - center) / sigma) ** 2)


# --- MathUtils.calcQuantile -------------------------------------------------

def test_mzmine_quantile_midpoint_rule():
    assert mzmine_quantile([4, 1, 3, 2], 0.5) == 2.5     # (n-1)q = 1.5 -> (s[1]+s[2])/2
    assert mzmine_quantile([4, 1, 3, 2], 0.85) == 3.5    # 2.55 -> (s[2]+s[3])/2
    assert mzmine_quantile([], 0.85) == 0.0
    assert mzmine_quantile([7.0], 0.85) == 7.0
    assert mzmine_quantile([1, 2, 3, 4], 2.0) == 4.0     # q clamped to 1
    assert mzmine_quantile([1, 2, 3, 4], -1.0) == 1.0    # q clamped to 0


def test_whole_run_noise_levels_per_column():
    d = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=float)
    np.testing.assert_allclose(whole_run_noise_levels(d, 0.5), [2.5, 25.0])
    assert whole_run_noise_levels(np.zeros((0, 3)), 0.85).tolist() == [0.0, 0.0, 0.0]


# --- MinimumSearchFeatureResolver.resolve ---------------------------------------

def test_single_gaussian_one_peak_to_baseline():
    y = g(1.0, 10000.0)
    peaks = resolve_ion_peaks(X, y, 57.0, 50.0, **KW)
    assert len(peaks) == 1
    p = peaks[0]
    assert isinstance(p, IonPeak)
    assert p.mz == 57.0
    assert abs(p.rt_apex - 1.0) < 0.003
    assert p.height == pytest.approx(10000.0, rel=1e-6)
    # adjustStartAndEnd pulls in the zeroed neighbour: edges sit at baseline
    assert p.intensity_array[0] < 50.0 and p.intensity_array[-1] < 50.0
    assert len(p.rt_array) == len(p.intensity_array) == p.end_idx - p.start_idx + 1
    np.testing.assert_allclose(p.rt_array, X[p.start_idx:p.end_idx + 1])


def test_two_gaussians_split_at_local_minimum():
    y = g(1.0, 10000.0) + g(1.07, 6000.0)
    kw = dict(KW, search_range=0.02)
    peaks = resolve_ion_peaks(X, y, 57.0, 50.0, **kw)
    assert len(peaks) == 2
    assert abs(peaks[0].rt_apex - 1.0) < 0.003
    assert abs(peaks[1].rt_apex - 1.07) < 0.003
    # the next region starts at the minimum that closed the previous one
    assert peaks[0].end_idx == peaks[1].start_idx


def test_noise_only_trace_gives_nothing():
    y = np.random.default_rng(0).uniform(0.0, 40.0, X.size)
    assert resolve_ion_peaks(X, y, 57.0, 50.0, **KW) == []


def test_three_point_spike_below_min_points():
    y = np.zeros(X.size)
    y[100:103] = 5000.0
    assert resolve_ion_peaks(X, y, 57.0, 50.0, **KW) == []


def test_top_edge_ratio_failure():
    # hump on a high plateau: 5000 top vs 3000 edges < 1.7x
    y = 3000.0 + g(1.0, 2000.0, sigma=0.05)
    assert resolve_ion_peaks(X, y, 57.0, 0.0, **KW) == []


def test_below_min_peak_height():
    assert resolve_ion_peaks(X, g(1.0, 800.0), 57.0, 50.0, **KW) == []


def test_noise_level_argument_is_used():
    # mzmine computes the level from the trace; ChromaKit passes the whole-run level in
    y = g(1.0, 10000.0)
    assert len(resolve_ion_peaks(X, y, 57.0, 50.0, **KW)) == 1
    assert resolve_ion_peaks(X, y, 57.0, 20000.0, **KW) == []


def test_short_trace_returns_nothing():
    assert resolve_ion_peaks(X[:2], np.array([5000.0, 6000.0]), 57.0, 0.0, **KW) == []


def test_plateau_apex_is_first_max():
    y = g(1.0, 10000.0)
    y[198:203] = 10000.0                     # flat (saturated) top
    peaks = resolve_ion_peaks(X, y, 57.0, 50.0, **KW)
    assert len(peaks) == 1
    assert peaks[0].apex_idx == 198
    assert peaks[0].height == 10000.0


def test_peak_cut_at_window_start_is_rejected():
    # max at index 0: top == left edge, fails the top/edge ratio (as in mzmine)
    y = g(0.0, 10000.0)
    assert resolve_ion_peaks(X, y, 57.0, 50.0, **KW) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        resolve_ion_peaks(X, np.zeros(10), 57.0, 0.0, **KW)
