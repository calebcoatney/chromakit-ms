import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock
from logic.spectral_deconvolution import EICPeak


def _make_mock_ms(n_scans=60, n_mz=200, rt_start=1.0, rt_end=2.0):
    """Mock rainbow DataFile object with .xlabels and .data."""
    ms = MagicMock()
    ms.xlabels = np.linspace(rt_start, rt_end, n_scans)
    ms.data = np.zeros((n_scans, n_mz), dtype=float)
    return ms


def _gaussian(n, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((np.arange(n) - center) / sigma) ** 2)


def test_returns_eic_peaks():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)  # peak at m/z 50, apex at scan 30

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)

    assert len(peaks) >= 1
    assert all(isinstance(p, EICPeak) for p in peaks)


def test_mz_is_one_based():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 0] = _gaussian(60, 30, 4, 1000.0)   # column 0 → m/z 1
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)  # column 49 → m/z 50

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)
    mzs = {p.mz for p in peaks}

    assert 1.0 in mzs
    assert 50.0 in mzs


def test_min_intensity_filter():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 10] = _gaussian(60, 30, 4, 50.0)   # max ~50, below threshold
    ms.data[:, 20] = _gaussian(60, 30, 4, 1000.0)  # max ~1000, above threshold

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=200.0)
    mzs = {p.mz for p in peaks}

    assert 11.0 not in mzs  # filtered out
    assert 21.0 in mzs      # kept


def test_boundary_indices_are_valid_integers():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)

    for p in peaks:
        n = len(p.rt_array)
        assert isinstance(p.left_boundary_idx, (int, np.integer))
        assert isinstance(p.right_boundary_idx, (int, np.integer))
        assert isinstance(p.apex_idx, (int, np.integer))
        assert 0 <= p.left_boundary_idx <= p.apex_idx <= p.right_boundary_idx < n


def test_rt_array_spans_full_window():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(rt_start=2.0, rt_end=3.0)
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)

    peaks = extract_eic_peaks(ms, t_start=2.0, t_end=3.0, min_intensity=100.0)

    for p in peaks:
        assert p.rt_array[0] >= 2.0 - 1e-9
        assert p.rt_array[-1] <= 3.0 + 1e-9


def test_empty_window_returns_empty_list():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(rt_start=1.0, rt_end=2.0)
    # No signal anywhere

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)

    assert peaks == []


def test_prominence_filter_rejects_noise_bumps():
    """Tiny bumps on a noisy baseline are rejected by prominence filter."""
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(n_scans=100, n_mz=10)

    # m/z column 0: real peak — tall Gaussian well above zero baseline
    ms.data[:, 0] = _gaussian(100, 50, 5, 10000.0)

    # m/z column 1: noise — small random bumps on a flat ~500-count baseline
    np.random.seed(42)
    ms.data[:, 1] = 500.0 + np.random.normal(0, 50, 100)
    # Ensure trace max > min_intensity so it passes the trace-level pre-filter
    ms.data[30, 1] = 800.0  # single spike; prominence ≈ 300

    peaks = extract_eic_peaks(
        ms, t_start=1.0, t_end=2.0, min_intensity=200.0, min_prominence=1000.0
    )
    mzs = {p.mz for p in peaks}
    assert 1.0 in mzs       # real peak has prominence ~10000 → kept
    assert 2.0 not in mzs   # noise bump has prominence ~300 → filtered


def test_per_peak_height_filter():
    """Peaks whose individual apex is below min_intensity are rejected,
    even when the trace max (a different peak) is above threshold."""
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(n_scans=100, n_mz=10)

    # Single m/z trace with one tall peak + one tiny peak
    ms.data[:, 0] = _gaussian(100, 30, 5, 5000.0) + _gaussian(100, 70, 3, 100.0)

    peaks = extract_eic_peaks(
        ms, t_start=1.0, t_end=2.0, min_intensity=200.0, min_prominence=50.0
    )
    # Only the tall peak at scan 30 should survive; the tiny peak at scan 70 is below height
    assert len(peaks) == 1
    assert abs(peaks[0].rt_apex - ms.xlabels[30]) < 0.01


def test_default_prominence_is_zero():
    """With default min_prominence=0, behavior matches original (all peaks kept)."""
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(n_scans=60, n_mz=5)
    ms.data[:, 0] = _gaussian(60, 30, 4, 1000.0)

    # No prominence arg → default 0 → no prominence filtering
    peaks_default = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)
    peaks_explicit = extract_eic_peaks(
        ms, t_start=1.0, t_end=2.0, min_intensity=100.0, min_prominence=0.0
    )
    assert len(peaks_default) == len(peaks_explicit)
