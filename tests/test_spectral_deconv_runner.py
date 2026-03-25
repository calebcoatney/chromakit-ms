import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock
from logic.spectral_deconv_runner import _group_peaks_into_windows, WindowGroupingParams


def _make_peak(rt, start, end):
    p = MagicMock()
    p.retention_time = rt
    p.start_time = start
    p.end_time = end
    return p


def test_single_peak_one_window():
    peaks = [_make_peak(1.0, 0.9, 1.1)]
    params = WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 1
    w_start, w_end, w_peaks = windows[0]
    assert len(w_peaks) == 1
    assert w_start < 1.0 < w_end  # padding applied


def test_overlapping_peaks_merge():
    # end_time of first > start_time of second → gap < 0
    peaks = [_make_peak(1.0, 0.9, 1.15), _make_peak(1.2, 1.1, 1.3)]
    params = WindowGroupingParams(gap_tolerance=0.05, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 1
    assert len(windows[0][2]) == 2


def test_far_peaks_separate_windows():
    peaks = [_make_peak(1.0, 0.9, 1.1), _make_peak(5.0, 4.9, 5.1)]
    params = WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 2


def test_window_clamped_to_rt_bounds():
    peaks = [_make_peak(0.1, 0.05, 0.15)]  # close to start
    params = WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    w_start, w_end, _ = windows[0]
    assert w_start >= 0.0


def test_auto_gap_tolerance_two_peaks():
    # Peaks of width 0.2 min → median width = 0.2 → gap_tolerance = 0.1
    peaks = [_make_peak(1.0, 0.9, 1.1), _make_peak(5.0, 4.9, 5.1)]
    params = WindowGroupingParams(gap_tolerance=None, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 2  # gap 3.8 >> auto tolerance ~0.1


def test_auto_gap_tolerance_fallback_single_peak():
    # Only 1 peak → fallback to 0.3 min
    peaks = [_make_peak(1.0, 0.9, 1.1)]
    params = WindowGroupingParams(gap_tolerance=None, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 1
