"""Spectral deconvolution runner for ChromaKit-MS.

Orchestrates: FID peak window grouping → EIC extraction → deconvolve() →
component-to-peak matching → ChromatographicPeak.deconvolved_spectrum update.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import rainbow as rb

from logic.spectral_deconvolution import DeconvolutionParams, deconvolve
from logic.eic_extractor import extract_eic_peaks


@dataclass
class WindowGroupingParams:
    """Parameters for grouping FID peaks into deconvolution windows."""
    gap_tolerance: float | None = None
    # None = 0.5 × median FID peak width; fallback 0.3 min if < 2 peaks
    padding_fraction: float = 0.5
    # Pad each side of cluster window by this fraction of cluster width
    rt_match_tolerance: float = 0.05
    # Max RT distance (min) to assign a DeconvolutedComponent to a FID peak


def _group_peaks_into_windows(
    peaks: list,
    params: WindowGroupingParams,
    rt_min: float,
    rt_max: float,
) -> list[tuple[float, float, list]]:
    """Group ChromatographicPeak objects into RT cluster windows.

    Returns list of (window_start, window_end, [peaks_in_window]) tuples,
    sorted by window_start.
    """
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda p: p.retention_time)

    # Compute gap tolerance
    if params.gap_tolerance is None:
        widths = [p.end_time - p.start_time for p in sorted_peaks]
        gap_tol = 0.5 * float(np.median(widths)) if len(widths) >= 2 else 0.3
    else:
        gap_tol = params.gap_tolerance

    # Merge adjacent peaks into clusters
    clusters: list[tuple[float, float, list]] = []
    cluster_start = sorted_peaks[0].start_time
    cluster_end = sorted_peaks[0].end_time
    cluster_peaks = [sorted_peaks[0]]

    for peak in sorted_peaks[1:]:
        gap = peak.start_time - cluster_end
        if gap <= gap_tol:
            cluster_end = max(cluster_end, peak.end_time)
            cluster_peaks.append(peak)
        else:
            clusters.append((cluster_start, cluster_end, cluster_peaks))
            cluster_start = peak.start_time
            cluster_end = peak.end_time
            cluster_peaks = [peak]
    clusters.append((cluster_start, cluster_end, cluster_peaks))

    # Pad each cluster and clamp to RT axis bounds
    windows = []
    for cl_start, cl_end, cl_peaks in clusters:
        width = cl_end - cl_start
        pad = max(params.padding_fraction * width, 1e-6)
        w_start = max(rt_min, cl_start - pad)
        w_end = min(rt_max, cl_end + pad)
        windows.append((w_start, w_end, cl_peaks))

    return windows
