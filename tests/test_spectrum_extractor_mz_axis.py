"""Tests that SpectrumExtractor reads the real m/z axis from ms.ylabels.

Regression coverage for the bug where the extractor assumed
``mz_values = np.arange(len(spectrum)) + 1`` (column 0 → m/z 1), which
silently mis-labelled spectra on instruments that scan above m/z 1 (e.g.
Agilent ChemStation with low_mass=20) or store only m/z bins above a
detection threshold (sparse, non-uniform m/z axes from threshold-scan
acquisition).

The fix is to derive m/z values from ``ms.ylabels`` (the m/z axis that
rainbow parses out of the file) instead of column indices.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_sparse_ms_mock(n_scans=20, ylabels=None, signal_mz=None,
                         signal_scan=10, signal_intensity=1000.0,
                         rt_start=1.0, rt_end=2.0):
    """Mock a rainbow data.ms DataFile with a non-contiguous m/z axis.

    ``ylabels`` is the actual m/z axis as rainbow would parse it (e.g.
    ``[24, 25, 26, 28, 32, 43, 60, 73, 74, 87]`` for a sparse / above-
    threshold-only scan). The column index of any given m/z is ``ylabels.index(mz)``.

    If ``signal_mz`` is given, a single signal point of ``signal_intensity``
    is placed at ``signal_scan`` in the column whose ylabels entry equals
    ``signal_mz``.
    """
    if ylabels is None:
        ylabels = np.array([24, 25, 26, 28, 32, 43, 60, 73, 74, 87], dtype=float)
    else:
        ylabels = np.asarray(ylabels, dtype=float)

    ms = MagicMock()
    ms.xlabels = np.linspace(rt_start, rt_end, n_scans)
    ms.ylabels = ylabels
    ms.data = np.zeros((n_scans, len(ylabels)), dtype=float)
    if signal_mz is not None:
        col = int(np.where(ylabels == signal_mz)[0][0])
        # Place a small triangular peak so background-subtraction has a
        # nonzero apex even after subtracting boundary scans.
        for k, dk in enumerate(range(-3, 4)):
            i = signal_scan + dk
            if 0 <= i < n_scans:
                ms.data[i, col] = signal_intensity * (1 - abs(dk) / 4.0)

    datadir = MagicMock()
    datadir.get_file = MagicMock(return_value=ms)
    return datadir, ms


def test_extract_at_rt_uses_real_mz_axis_from_ylabels():
    """A signal in column 5 (m/z=43 in our sparse axis) must be reported
    as m/z=43, not m/z=6 (which is what column-index + 1 would give).
    """
    from logic.spectrum_extractor import SpectrumExtractor

    ylabels = [24, 25, 26, 28, 32, 43, 60, 73, 74, 87]
    # Signal at m/z = 43, which is column index 5
    datadir, ms = _make_sparse_ms_mock(
        ylabels=ylabels, signal_mz=43, signal_scan=10, signal_intensity=5000.0,
    )

    with patch('logic.spectrum_extractor.rb.read', return_value=datadir):
        extractor = SpectrumExtractor()
        result = extractor.extract_at_rt(
            data_directory='/fake/path.D',
            retention_time=1.5,   # middle of the rt window
            intensity_threshold=0.5,  # keep only the dominant peak
        )

    assert result is not None
    assert 43 in result['mz'].tolist(), (
        f"Extracted m/z values should include 43 (the signal's real m/z), "
        f"but got {result['mz'].tolist()}"
    )
    assert 6 not in result['mz'].tolist(), (
        f"Extracted m/z values must NOT include 6 (the broken column-index+1 "
        f"label for column 5). Got {result['mz'].tolist()}"
    )


def test_extract_for_peak_uses_real_mz_axis_from_ylabels():
    """The peak-bounded extractor must also use ms.ylabels for m/z."""
    from logic.spectrum_extractor import SpectrumExtractor

    ylabels = [24, 25, 26, 28, 32, 43, 60, 73, 74, 87]
    # Signal at m/z = 60 (carboxylic acid McLafferty fragment, column 6)
    datadir, ms = _make_sparse_ms_mock(
        ylabels=ylabels, signal_mz=60, signal_scan=10, signal_intensity=8000.0,
    )

    class _Peak:
        start_time = 1.4
        apex_time = 1.5
        end_time = 1.6

    with patch('logic.spectrum_extractor.rb.read', return_value=datadir):
        extractor = SpectrumExtractor()
        result = extractor.extract_for_peak(
            data_directory='/fake/path.D',
            peak=_Peak(),
            options={'extraction_method': 'apex',
                     'intensity_threshold': 0.5,
                     'subtract_enabled': False},
        )

    assert result is not None
    assert 60 in result['mz'].tolist(), (
        f"Extracted m/z values should include 60 (the signal's real m/z), "
        f"but got {result['mz'].tolist()}"
    )
    assert 7 not in result['mz'].tolist(), (
        f"Extracted m/z values must NOT include 7 (column-index+1 label "
        f"for column 6). Got {result['mz'].tolist()}"
    )


def test_extract_for_peak_handles_contiguous_axis_correctly():
    """If the instrument scans contiguously from m/z 1 (the old assumption),
    behavior should still be correct: m/z = ms.ylabels[col_idx] = col_idx + 1.
    """
    from logic.spectrum_extractor import SpectrumExtractor

    n_mz = 200
    ylabels = np.arange(n_mz, dtype=float) + 1  # contiguous: 1..200
    datadir, ms = _make_sparse_ms_mock(
        ylabels=ylabels, signal_mz=43.0, signal_scan=10, signal_intensity=3000.0,
    )

    class _Peak:
        start_time = 1.4
        apex_time = 1.5
        end_time = 1.6

    with patch('logic.spectrum_extractor.rb.read', return_value=datadir):
        extractor = SpectrumExtractor()
        result = extractor.extract_for_peak(
            data_directory='/fake/path.D',
            peak=_Peak(),
            options={'extraction_method': 'apex',
                     'intensity_threshold': 0.5,
                     'subtract_enabled': False},
        )

    assert result is not None
    assert 43 in result['mz'].tolist()
