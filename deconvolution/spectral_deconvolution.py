# deconvolution/spectral_deconvolution.py
"""ADAP-GC 3.2 Spectral Deconvolution.

Reference: Smirnov et al., J. Proteome Res. 2018, 17, 470-478.
Java source: deconvolution/adap-gc-source-reference/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import nnls
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN


# ─── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class EICPeak:
    """One detected EIC peak: a single m/z's intensity profile across a peak window."""
    rt_apex: float               # retention time of apex (minutes)
    mz: float                    # m/z of this EIC
    rt_array: np.ndarray         # RT time points spanning the peak window
    intensity_array: np.ndarray  # intensities at those time points
    left_boundary_idx: int       # index of left apex window boundary
    right_boundary_idx: int      # index of right apex window boundary
    apex_idx: int                # index of apex


@dataclass
class DeconvolutedComponent:
    """One deconvoluted analyte: model peak + fragmentation spectrum."""
    rt: float                              # apex RT of model peak (minutes)
    spectrum: dict                         # {mz: intensity} fragmentation spectrum
    model_peak_mz: float                   # m/z of representative EIC peak
    model_peak_rt_array: np.ndarray        # model peak elution profile RT axis
    model_peak_intensity_array: np.ndarray # model peak elution profile intensities


@dataclass
class DeconvolutionParams:
    """All tunable parameters for the ADAP-GC 3.2 algorithm."""
    min_cluster_distance: float = 0.005    # DBSCAN eps (minutes)
    min_cluster_size: int = 2              # DBSCAN min_samples
    min_cluster_intensity: float = 200.0   # drop clusters below this max intensity
    use_is_shared: bool = True             # filter chromatographically unresolved peaks
    edge_to_height_ratio: float = 0.3      # boundary/apex threshold for is_shared
    delta_to_height_ratio: float = 0.3     # |left-right|/apex threshold for is_shared
    min_model_peak_sharpness: float = 10.0 # minimum sharpness score for model peaks
    shape_sim_threshold: float = 30.0      # max angle (degrees) within one shape cluster
    model_peak_choice: str = "sharpness"   # "sharpness", "intensity", or "mz"
    excluded_mz: list = field(default_factory=list)  # empty = no exclusions
    excluded_mz_tolerance: float = 0.5     # ± tolerance for excluded_mz matching


# ─── Internal dataclass ────────────────────────────────────────────────────────

@dataclass
class _PeakData:
    """Internal: EICPeak wrapper with mutable shared-window RT bounds for merge step."""
    source: EICPeak
    left_peak_rt: float      # shared window left (RT minutes); may expand
    right_peak_rt: float     # shared window right (RT minutes); may expand
    rt_array: np.ndarray     # chromatogram RT axis (may be merged from multiple peaks)
    intensity_array: np.ndarray
    apex_intensity: float    # max intensity in this chromatogram


# ─── Layer 1: Math primitives ─────────────────────────────────────────────────

def sharpness_yang(rt_array: np.ndarray, intensity_array: np.ndarray,
                   left: int, right: int) -> float:
    """Peak quality metric. Port of FeatureTools.sharpnessYang() (line 542).

    Computes median slope on each side of the apex for points above the
    25th-percentile height above baseline. Higher = sharper, better model peak.
    Uses index deltas (not time deltas) matching Java lines 591, 607.

    Returns -1.0 if both sides are empty (degenerate peak).
    Returns median of available side if only one side has data above p25.
    """
    apex_idx = left + int(np.argmax(intensity_array[left:right + 1]))
    apex_intensity = intensity_array[apex_idx]

    if apex_intensity <= 0:
        return -1.0

    left_h = intensity_array[left]
    right_h = intensity_array[right]
    left_rt = rt_array[left]
    right_rt = rt_array[right]

    if right_rt == left_rt:
        return -1.0

    slope_bl = (right_h - left_h) / (right_rt - left_rt)
    intercept_bl = left_h - slope_bl * left_rt
    baseline_at_apex = slope_bl * rt_array[apex_idx] + intercept_bl

    p25_offset = 0.25 * (apex_intensity - baseline_at_apex)
    if p25_offset <= 0.0:
        return -1.0
    p25 = p25_offset + baseline_at_apex

    left_slopes = [
        (apex_intensity - intensity_array[i]) / float(apex_idx - i)
        for i in range(left, apex_idx)
        if intensity_array[i] >= p25
    ]
    right_slopes = [
        (intensity_array[i] - apex_intensity) / float(i - apex_idx)
        for i in range(apex_idx + 1, right + 1)
        if intensity_array[i] >= p25
    ]

    if not left_slopes and not right_slopes:
        return -1.0
    if not right_slopes:
        return float(np.median(left_slopes))
    if not left_slopes:
        return float(np.median(right_slopes))
    return (float(np.median(left_slopes)) - float(np.median(right_slopes))) / 2.0
