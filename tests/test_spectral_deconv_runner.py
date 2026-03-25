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


import numpy as np
from unittest.mock import MagicMock, patch
from logic.spectral_deconvolution import DeconvolutionParams, DeconvolutedComponent
from logic.spectral_deconv_runner import run_spectral_deconvolution, WindowGroupingParams


def _make_chromatographic_peak(rt, start, end):
    """Mock ChromatographicPeak with required attributes."""
    p = MagicMock()
    p.retention_time = rt
    p.start_time = start
    p.end_time = end
    p.deconvolved_spectrum = None
    p.deconvolution_component_count = None
    return p


def _make_component(rt, spectrum=None):
    c = MagicMock(spec=DeconvolutedComponent)
    c.rt = rt
    c.spectrum = spectrum or {50.0: 1000.0, 73.0: 500.0}
    return c


def test_nearest_component_assigned():
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)
    components = [_make_component(1.01), _make_component(1.5)]
    grouping = WindowGroupingParams(rt_match_tolerance=0.05)

    # call the internal matching function directly
    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak], components, grouping.rt_match_tolerance)

    assert peak.deconvolved_spectrum is not None
    assert peak.deconvolution_component_count == 2


def test_no_match_beyond_tolerance():
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)
    components = [_make_component(1.2)]  # 0.2 min away > 0.05 tolerance
    grouping = WindowGroupingParams(rt_match_tolerance=0.05)

    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak], components, grouping.rt_match_tolerance)

    assert peak.deconvolved_spectrum is None
    assert peak.deconvolution_component_count == 1  # ran but no match


def test_one_to_one_assignment():
    # Two peaks, two components — each should get its nearest
    peak_a = _make_chromatographic_peak(1.0, 0.9, 1.1)
    peak_b = _make_chromatographic_peak(1.3, 1.2, 1.4)
    comp_a = _make_component(1.01, spectrum={50.0: 1000.0})
    comp_b = _make_component(1.31, spectrum={73.0: 2000.0})

    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak_a, peak_b], [comp_a, comp_b], rt_match_tolerance=0.05)

    assert peak_a.deconvolved_spectrum is not None
    assert peak_b.deconvolved_spectrum is not None
    # Each gets a distinct spectrum
    assert set(peak_a.deconvolved_spectrum['mz']) != set(peak_b.deconvolved_spectrum['mz'])


def test_deconvolved_spectrum_format():
    """Deconvolved spectrum must be {'mz': np.ndarray, 'intensities': np.ndarray}."""
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)
    comp = _make_component(1.01, spectrum={50.0: 1000.0, 73.0: 500.0})

    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak], [comp], rt_match_tolerance=0.05)

    spec = peak.deconvolved_spectrum
    assert 'mz' in spec and 'intensities' in spec
    assert isinstance(spec['mz'], np.ndarray)
    assert isinstance(spec['intensities'], np.ndarray)
    assert len(spec['mz']) == len(spec['intensities'])
    assert list(spec['mz']) == sorted(spec['mz'])  # sorted by m/z


def test_run_spectral_deconvolution_empty_components_sets_count_zero():
    """When deconvolve() returns empty, component_count must be 0 (not None)."""
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)

    fake_ms = MagicMock()
    fake_ms.xlabels = np.linspace(0.0, 5.0, 100)
    fake_ms.data = np.zeros((100, 200))
    # Add a non-zero trace so EIC extraction doesn't short-circuit
    fake_ms.data[40:60, 49] = 500.0

    fake_data_dir = MagicMock()
    fake_data_dir.get_file.return_value = fake_ms

    fake_eic_peak = MagicMock()

    with patch('logic.spectral_deconv_runner.rb.read', return_value=fake_data_dir), \
         patch('logic.spectral_deconv_runner.extract_eic_peaks', return_value=[fake_eic_peak]), \
         patch('logic.spectral_deconv_runner.deconvolve', return_value=[]):
        run_spectral_deconvolution(
            [peak], ms_data_path='/fake/path.D',
            grouping_params=WindowGroupingParams(gap_tolerance=0.3)
        )

    assert peak.deconvolution_component_count == 0
    assert peak.deconvolved_spectrum is None
