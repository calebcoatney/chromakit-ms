# deconvolution/test_spectral_deconvolution.py
"""Tests for ADAP-GC 3.2 spectral deconvolution module."""
import numpy as np
import pytest
from spectral_deconvolution import (
    EICPeak, DeconvolutedComponent, DeconvolutionParams,
    sharpness_yang,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gaussian_peak(rt_center=5.0, width=0.5, height=1000.0,
                       mz=100.0, n_points=50) -> EICPeak:
    """Create a synthetic Gaussian EIC peak."""
    rts = np.linspace(rt_center - 2 * width, rt_center + 2 * width, n_points)
    ints = height * np.exp(-0.5 * ((rts - rt_center) / (width / 2)) ** 2)
    apex_idx = int(np.argmax(ints))
    return EICPeak(
        rt_apex=rt_center,
        mz=mz,
        rt_array=rts,
        intensity_array=ints,
        left_boundary_idx=0,
        right_boundary_idx=n_points - 1,
        apex_idx=apex_idx,
    )


class TestSharpnessYang:
    def test_gaussian_peak_is_sharp(self):
        # A symmetric Gaussian should score well above 10
        peak = make_gaussian_peak(rt_center=5.0, width=0.3, height=1000.0, n_points=50)
        score = sharpness_yang(peak.rt_array, peak.intensity_array,
                               peak.left_boundary_idx, peak.right_boundary_idx)
        assert score > 10.0

    def test_flat_signal_returns_negative_one(self):
        rt = np.linspace(0, 1, 20)
        ints = np.ones(20) * 500.0
        assert sharpness_yang(rt, ints, 0, 19) == -1.0

    def test_one_sided_returns_median_not_negative_one(self):
        # All right-side points below p25 → only left side has slopes
        # Construct: apex at index 10, steeply rising left, flat right (below p25)
        rt = np.linspace(0, 1, 21)
        ints = np.zeros(21)
        ints[0] = 10.0   # left boundary
        ints[10] = 1000.0  # apex
        ints[20] = 10.0  # right boundary (right side all flat, below p25=257.5)
        # Only left side will have points above p25=0.25*(1000-10)+10=257.5
        # Fill left side with rising slope
        for i in range(1, 10):
            ints[i] = 10.0 + (i / 10.0) * 990.0
        # Right side: all at 10 (below p25) except boundary
        score = sharpness_yang(rt, ints, 0, 20)
        # Left side has slopes; right side empty → returns median_left (not -1.0)
        assert score != -1.0
        assert score > 0.0
