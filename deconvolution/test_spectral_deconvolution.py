# deconvolution/test_spectral_deconvolution.py
"""Tests for ADAP-GC 3.2 spectral deconvolution module."""
import numpy as np
import pytest
from spectral_deconvolution import (
    EICPeak, DeconvolutedComponent, DeconvolutionParams,
    sharpness_yang, is_shared, shape_similarity_angle,
    _merge_peaks, _cluster_by_rt, _cluster_by_shape,
    _filter_peaks, _find_model_peak,
    _build_components, deconvolve,
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
