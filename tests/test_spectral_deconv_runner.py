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
from logic.correlation_grouping import CorrelationDeconvParams, DeconvolutedComponent
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
    """When deconvolve_window() returns nothing, component_count must be 0 (not None)."""
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)

    fake_ms = MagicMock()
    fake_ms.xlabels = np.linspace(0.0, 5.0, 100)
    fake_ms.ylabels = np.arange(1.0, 201.0)
    fake_ms.data = np.zeros((100, 200))
    fake_data_dir = MagicMock()
    fake_data_dir.get_file.return_value = fake_ms

    with patch('logic.spectral_deconv_runner.rb.read', return_value=fake_data_dir), \
         patch('logic.spectral_deconv_runner.deconvolve_window', return_value=[]):
        run_spectral_deconvolution(
            [peak], ms_data_path='/fake/path.D',
            grouping_params=WindowGroupingParams(gap_tolerance=0.3)
        )

    assert peak.deconvolution_component_count == 0
    assert peak.deconvolved_spectrum is None


def _synthetic_ms(compounds, mz_axis, t0=0.0, t1=2.0, dt=0.005, seed=0):
    from types import SimpleNamespace
    x = np.round(np.arange(t0, t1, dt), 6)
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 30.0, (x.size, len(mz_axis)))
    col = {m: j for j, m in enumerate(mz_axis)}
    for center, ions in compounds:
        for m, h in ions.items():
            data[:, col[m]] += h * np.exp(-0.5 * ((x - center) / 0.015) ** 2)
    return SimpleNamespace(xlabels=x, ylabels=np.asarray(mz_axis, dtype=float), data=data)


def test_runner_end_to_end_spectra_match_planted_ions():
    mz_axis = [29.0, 43.0, 57.0, 71.0, 85.0, 99.0]
    ms = _synthetic_ms([(0.8, {43: 8000, 57: 20000, 71: 6000, 85: 5000}),
                        (1.4, {29: 7000, 43: 9000, 57: 15000, 99: 6000})], mz_axis)
    p1 = _make_chromatographic_peak(0.8, 0.76, 0.84)
    p2 = _make_chromatographic_peak(1.4, 1.36, 1.44)
    data_dir = MagicMock()
    data_dir.get_file.return_value = ms
    with patch('logic.spectral_deconv_runner.rb.read', return_value=data_dir):
        run_spectral_deconvolution([p1, p2], '/fake.D',
                                   deconv_params=CorrelationDeconvParams(search_range=0.03),
                                   grouping_params=WindowGroupingParams(gap_tolerance=0.1))
    assert set(p1.deconvolved_spectrum['mz']) == {43.0, 57.0, 71.0, 85.0}
    assert set(p2.deconvolved_spectrum['mz']) == {29.0, 43.0, 57.0, 99.0}


def test_runner_applies_ms_time_offset():
    mz_axis = [43.0, 57.0, 71.0, 85.0]
    ms = _synthetic_ms([(0.80, {43: 8000, 57: 20000, 71: 6000, 85: 5000})], mz_axis)
    peak = _make_chromatographic_peak(0.85, 0.81, 0.89)     # FID RT = MS RT + 0.05
    data_dir = MagicMock()
    data_dir.get_file.return_value = ms
    with patch('logic.spectral_deconv_runner.rb.read', return_value=data_dir):
        run_spectral_deconvolution([peak], '/fake.D',
                                   deconv_params=CorrelationDeconvParams(search_range=0.03),
                                   grouping_params=WindowGroupingParams(gap_tolerance=0.1,
                                                                        rt_match_tolerance=0.02),
                                   ms_time_offset=0.05)
    assert peak.deconvolved_spectrum is not None
    assert set(peak.deconvolved_spectrum['mz']) == {43.0, 57.0, 71.0, 85.0}


def test_max_window_peaks_splits_large_cluster():
    """A chain of 20 overlapping peaks with max_window_peaks=8 should split."""
    peaks = [_make_peak(1.0 + i * 0.05, 0.97 + i * 0.05, 1.03 + i * 0.05)
             for i in range(20)]
    params = WindowGroupingParams(gap_tolerance=0.1, padding_fraction=0.0,
                                  max_window_peaks=8)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=20.0)

    assert len(windows) >= 3  # 20 peaks / 8 max = at least 3 windows
    for _, _, w_peaks in windows:
        assert len(w_peaks) <= 8
    # All peaks accounted for
    assert sum(len(w[2]) for w in windows) == 20


def test_max_window_peaks_zero_disables_splitting():
    """max_window_peaks=0 means unlimited — no splitting."""
    peaks = [_make_peak(1.0 + i * 0.05, 0.97 + i * 0.05, 1.03 + i * 0.05)
             for i in range(20)]
    params = WindowGroupingParams(gap_tolerance=0.1, padding_fraction=0.0,
                                  max_window_peaks=0)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=20.0)

    assert len(windows) == 1
    assert len(windows[0][2]) == 20


def test_split_at_largest_gap():
    """Splitting prefers the largest internal gap."""
    # 3 peaks close, then a gap, then 2 peaks close
    peaks = [
        _make_peak(1.0, 0.95, 1.05),
        _make_peak(1.1, 1.05, 1.15),
        _make_peak(1.2, 1.15, 1.25),
        # gap of 0.3 min here
        _make_peak(1.6, 1.55, 1.65),
        _make_peak(1.7, 1.65, 1.75),
    ]
    params = WindowGroupingParams(gap_tolerance=0.5, padding_fraction=0.0,
                                  max_window_peaks=3)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=20.0)

    assert len(windows) == 2
    assert len(windows[0][2]) == 3
    assert len(windows[1][2]) == 2


def _two_run_peak_and_ms():
    mz_axis = [43.0, 57.0, 71.0, 85.0]
    ms = _synthetic_ms([(0.80, {43: 8000, 57: 20000, 71: 6000, 85: 5000})], mz_axis)
    peak = _make_chromatographic_peak(0.80, 0.76, 0.84)
    data_dir = MagicMock()
    data_dir.get_file.return_value = ms
    return peak, data_dir


def test_rerun_without_a_match_clears_the_previous_spectrum():
    """A rerun replaces results: a peak that loses its match must not keep the old spectrum."""
    peak, data_dir = _two_run_peak_and_ms()
    grouping = WindowGroupingParams(gap_tolerance=0.1)
    with patch('logic.spectral_deconv_runner.rb.read', return_value=data_dir):
        run_spectral_deconvolution([peak], '/fake.D',
                                   deconv_params=CorrelationDeconvParams(search_range=0.03),
                                   grouping_params=grouping)
        assert peak.deconvolved_spectrum is not None
        assert peak.deconvolution_component_count == 1

        # Second run: min_ions above the ion count of the planted compound -> no component.
        run_spectral_deconvolution([peak], '/fake.D',
                                   deconv_params=CorrelationDeconvParams(search_range=0.03,
                                                                         min_ions=10),
                                   grouping_params=grouping)
    assert peak.deconvolved_spectrum is None
    assert peak.deconvolution_component_count == 0


def test_rerun_with_shifted_ms_offset_clears_the_previous_spectrum():
    """The inspector's offset 'Apply' reruns: a component pushed out of reach must not linger."""
    peak, data_dir = _two_run_peak_and_ms()
    grouping = WindowGroupingParams(gap_tolerance=0.1, rt_match_tolerance=0.02)
    params = CorrelationDeconvParams(search_range=0.03)
    with patch('logic.spectral_deconv_runner.rb.read', return_value=data_dir):
        run_spectral_deconvolution([peak], '/fake.D', deconv_params=params,
                                   grouping_params=grouping)
        assert peak.deconvolved_spectrum is not None

        run_spectral_deconvolution([peak], '/fake.D', deconv_params=params,
                                   grouping_params=grouping, ms_time_offset=0.05)
    assert peak.deconvolved_spectrum is None
    assert peak.deconvolution_component_count == 1      # a component exists, none within reach


def test_cancelled_rerun_leaves_unprocessed_peaks_on_the_apex_fallback():
    """Results are reset up front, so a run cancelled before a window leaves None, not stale data."""
    peak, data_dir = _two_run_peak_and_ms()
    grouping = WindowGroupingParams(gap_tolerance=0.1)
    params = CorrelationDeconvParams(search_range=0.03)
    with patch('logic.spectral_deconv_runner.rb.read', return_value=data_dir):
        run_spectral_deconvolution([peak], '/fake.D', deconv_params=params,
                                   grouping_params=grouping)
        assert peak.deconvolved_spectrum is not None
        run_spectral_deconvolution([peak], '/fake.D', deconv_params=params,
                                   grouping_params=grouping, should_cancel=lambda: True)
    assert peak.deconvolved_spectrum is None
    assert peak.deconvolution_component_count is None


def test_runner_passes_whole_run_noise_to_every_window():
    """Noise levels come from the whole run, never from the window slice (spec 3, step 2)."""
    from logic.correlation_grouping import deconvolve_window as real_deconvolve_window
    from logic.ion_peak_resolver import whole_run_noise_levels
    mz_axis = [29.0, 43.0, 57.0, 71.0, 85.0, 99.0]
    ms = _synthetic_ms([(0.8, {43: 8000, 57: 20000, 71: 6000, 85: 5000}),
                        (1.4, {29: 7000, 43: 9000, 57: 15000, 99: 6000})], mz_axis)
    p1 = _make_chromatographic_peak(0.8, 0.76, 0.84)
    p2 = _make_chromatographic_peak(1.4, 1.36, 1.44)
    data_dir = MagicMock()
    data_dir.get_file.return_value = ms
    params = CorrelationDeconvParams(search_range=0.03, noise_percentile=0.7)

    with patch('logic.spectral_deconv_runner.rb.read', return_value=data_dir), \
         patch('logic.spectral_deconv_runner.deconvolve_window',
               wraps=real_deconvolve_window) as spy:
        run_spectral_deconvolution([p1, p2], '/fake.D', deconv_params=params,
                                   grouping_params=WindowGroupingParams(gap_tolerance=0.1))

    assert spy.call_count == 2
    full = np.asarray(ms.data, dtype=float)
    expected = whole_run_noise_levels(full, params.noise_percentile)
    for call in spy.call_args_list:
        rt_window, ms_slice, _mz_axis, noise_levels, _params = call.args
        np.testing.assert_array_equal(noise_levels, expected)
        assert ms_slice.shape[0] < full.shape[0]            # it really was a window slice...
        # ...whose own noise would have differed, so this test can tell the two apart
        assert not np.allclose(whole_run_noise_levels(ms_slice, params.noise_percentile), expected)
