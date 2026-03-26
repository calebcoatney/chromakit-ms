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
    max_window_peaks: int = 10
    # Force a split when a window accumulates this many peaks (0 = unlimited)


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
    raw_clusters: list[list] = []
    cluster_end = sorted_peaks[0].end_time
    cluster_peaks = [sorted_peaks[0]]

    for peak in sorted_peaks[1:]:
        gap = peak.start_time - cluster_end
        if gap <= gap_tol:
            cluster_end = max(cluster_end, peak.end_time)
            cluster_peaks.append(peak)
        else:
            raw_clusters.append(cluster_peaks)
            cluster_end = peak.end_time
            cluster_peaks = [peak]
    raw_clusters.append(cluster_peaks)

    # Split oversized clusters at their largest internal RT gap
    max_peaks = params.max_window_peaks
    if max_peaks > 0:
        split_clusters: list[list] = []
        for cl in raw_clusters:
            split_clusters.extend(_split_cluster(cl, max_peaks))
    else:
        split_clusters = raw_clusters

    # Build windows with padding
    windows = []
    for cl_peaks in split_clusters:
        cl_start = min(p.start_time for p in cl_peaks)
        cl_end = max(p.end_time for p in cl_peaks)
        width = cl_end - cl_start
        pad = max(params.padding_fraction * width, 1e-6)
        w_start = max(rt_min, cl_start - pad)
        w_end = min(rt_max, cl_end + pad)
        windows.append((w_start, w_end, cl_peaks))

    return windows


def _split_cluster(peaks: list, max_peaks: int) -> list[list]:
    """Recursively split a peak cluster at its largest internal RT gap.

    Returns a list of sub-clusters, each with at most max_peaks peaks.
    """
    if len(peaks) <= max_peaks:
        return [peaks]

    # Find the largest gap between consecutive peak RTs
    rts = [p.retention_time for p in peaks]
    gaps = [(rts[i + 1] - rts[i], i) for i in range(len(rts) - 1)]
    _, split_idx = max(gaps, key=lambda x: x[0])

    left = peaks[:split_idx + 1]
    right = peaks[split_idx + 1:]

    return _split_cluster(left, max_peaks) + _split_cluster(right, max_peaks)


def _assign_components_to_peaks(
    fid_peaks: list,
    components: list,
    rt_match_tolerance: float,
) -> None:
    """Assign DeconvolutedComponents to FID peaks in-place (one-to-one, greedy by RT proximity).

    Sets peak.deconvolved_spectrum = {'mz': array, 'intensities': array}
    and peak.deconvolution_component_count = len(components) on each peak,
    regardless of whether a match was found (count lets Phase C show
    "X found, none matched" vs "not run").
    """
    n_components = len(components)

    # Build all (distance, component_idx, peak_idx) pairs
    pairs = []
    for ci, comp in enumerate(components):
        for pi, peak in enumerate(fid_peaks):
            dist = abs(comp.rt - peak.retention_time)
            # Tie-break: higher total intensity wins (negate for sort)
            intensity = -sum(comp.spectrum.values())
            pairs.append((dist, intensity, ci, pi))

    pairs.sort()  # sort by distance, then by descending intensity

    assigned_comps = set()
    assigned_peaks = set()

    for dist, _, ci, pi in pairs:
        if ci in assigned_comps or pi in assigned_peaks:
            continue
        if dist <= rt_match_tolerance:
            comp = components[ci]
            if not comp.spectrum:
                continue  # skip components with empty spectra
            peak = fid_peaks[pi]
            mz_arr = np.array(sorted(comp.spectrum.keys()))
            int_arr = np.array([comp.spectrum[m] for m in mz_arr])
            peak.deconvolved_spectrum = {'mz': mz_arr, 'intensities': int_arr}
            assigned_comps.add(ci)
            assigned_peaks.add(pi)

    # Set component count on all peaks (even unmatched ones)
    for peak in fid_peaks:
        peak.deconvolution_component_count = n_components


def run_spectral_deconvolution(
    peaks: list,
    ms_data_path: str,
    deconv_params: DeconvolutionParams | None = None,
    grouping_params: WindowGroupingParams | None = None,
    progress_callback=None,
    should_cancel=None,
) -> list:
    """Run ADAP-GC spectral deconvolution on all FID peaks.

    Args:
        peaks: List of ChromatographicPeak objects (already integrated).
        ms_data_path: Path to the Agilent .D directory.
        deconv_params: ADAP-GC parameters. Uses defaults if None.
        grouping_params: Window grouping parameters. Uses defaults if None.
        progress_callback: Optional callable(int) -> None receiving 0-100 progress.
        should_cancel: Optional callable() -> bool; if True, abort early.

    Returns:
        The same peaks list with deconvolved_spectrum / deconvolution_component_count
        populated in-place on matched peaks.
    """
    if deconv_params is None:
        deconv_params = DeconvolutionParams()
    if grouping_params is None:
        grouping_params = WindowGroupingParams()

    # Open MS data once
    data_dir = rb.read(ms_data_path)
    ms = data_dir.get_file('data.ms')

    rt_min = float(ms.xlabels[0])
    rt_max = float(ms.xlabels[-1])

    windows = _group_peaks_into_windows(peaks, grouping_params, rt_min, rt_max)
    total = max(len(windows), 1)

    for i, (window_start, window_end, window_peaks) in enumerate(windows):
        if should_cancel is not None and should_cancel():
            break

        eic_peaks = extract_eic_peaks(
            ms,
            t_start=window_start,
            t_end=window_end,
            min_intensity=deconv_params.min_cluster_intensity,
            min_prominence=deconv_params.min_eic_prominence,
        )

        if not eic_peaks:
            if progress_callback is not None:
                progress_callback(int(100 * (i + 1) / total))
            continue

        components = deconvolve(eic_peaks, deconv_params)

        if not components:
            # Ran but found nothing — still record count as 0
            for peak in window_peaks:
                peak.deconvolution_component_count = 0
        else:
            _assign_components_to_peaks(window_peaks, components, grouping_params.rt_match_tolerance)

        if progress_callback is not None:
            progress_callback(int(100 * (i + 1) / total))

    return peaks
