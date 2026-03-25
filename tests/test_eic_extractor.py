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
