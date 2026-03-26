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
    min_eic_prominence: float = 1000.0     # min prominence for EIC peak detection
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
    if left < 0 or right >= len(intensity_array) or left > right:
        return -1.0

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


def is_shared(intensity_array: np.ndarray,
              edge_to_height: float, delta_to_height: float) -> bool:
    """Detect chromatographically unresolved peaks.
    Port of FeatureTools.isShared(List<Double>, ...) (line 185).

    Call with: peak.intensity_array[left_boundary_idx : right_boundary_idx + 1]

    Returns True if the peak has multiple local maxima OR any boundary ratio
    exceeds its threshold (indicating co-elution with an adjacent peak).
    """
    size = len(intensity_array)
    if size < 2:
        return False

    left_intensity = float(intensity_array[0])
    right_intensity = float(intensity_array[-1])
    absolute_maximum = max(left_intensity, right_intensity)
    local_maxima_count = 0
    index = 1

    while index < size - 1:
        current = float(intensity_array[index])
        if current > absolute_maximum:
            absolute_maximum = current

        prev_idx = index - 1
        next_idx = index + 1
        # Skip plateau runs (equal consecutive values)
        while next_idx + 1 < size and current == float(intensity_array[next_idx]):
            next_idx += 1

        if float(intensity_array[prev_idx]) < current > float(intensity_array[next_idx]):
            local_maxima_count += 1

        index = next_idx

    if local_maxima_count > 1:
        return True

    if absolute_maximum == 0:
        return False

    left_to_apex = left_intensity / absolute_maximum
    right_to_apex = right_intensity / absolute_maximum
    delta_to_apex = abs(left_intensity - right_intensity) / absolute_maximum

    return (left_to_apex >= edge_to_height
            or right_to_apex >= edge_to_height
            or delta_to_apex >= delta_to_height)


def shape_similarity_angle(peak_a: EICPeak, peak_b: EICPeak) -> float:
    """Angle between two chromatogram elution profiles (degrees, [0°, 90°]).

    Port of Math.continuous_dot_product() + angle from
    TwoStepDecomposition.getShapeClusters() (line 448-453).

    Norm uses the continuous (trapz) inner product, matching peak.getNorm()
    in Java: norm = sqrt(continuous_dot_product(chrom, chrom)).
    np.interp clamps to boundary values outside each peak's RT range.
    """
    all_rt = np.union1d(peak_a.rt_array, peak_b.rt_array)
    a = np.interp(all_rt, peak_a.rt_array, peak_a.intensity_array)
    b = np.interp(all_rt, peak_b.rt_array, peak_b.intensity_array)

    norm_a = np.sqrt(np.trapezoid(a ** 2, all_rt))
    norm_b = np.sqrt(np.trapezoid(b ** 2, all_rt))

    if norm_a == 0.0 or norm_b == 0.0:
        return 90.0

    dot = np.trapezoid(a * b, all_rt)
    cos_angle = np.clip(dot / (norm_a * norm_b), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ─── Layer 2: Internal boundary/merge helpers ─────────────────────────────────

def _correct_peak_boundaries(peak_data_list: list,
                              edge_ratio: float, delta_ratio: float) -> None:
    """Expand shared-window RT bounds for adjacent same-m/z peaks that are close.
    Port of FeatureTools.correctPeakBoundaries().

    Mutates left_peak_rt / right_peak_rt on _PeakData objects in place.

    Java fidelity note: the Java computes mergeRight/mergeLeft boolean conditions
    but their if(!mergeRight) guards are commented out (lines 295, 322). Only the
    1.1× combined-width span check gates the merge. This implementation matches
    that relaxed (span-only) behavior.
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for pd in peak_data_list:
        groups[pd.source.mz].append(pd)

    for group in groups.values():
        group.sort(key=lambda p: p.source.rt_apex)
        for i in range(1, len(group)):
            prev, cur = group[i - 1], group[i]
            combined = ((prev.right_peak_rt - prev.left_peak_rt) +
                        (cur.right_peak_rt - cur.left_peak_rt))
            total = cur.right_peak_rt - prev.left_peak_rt
            if combined > 0 and total < 1.1 * combined:
                prev.right_peak_rt = cur.right_peak_rt
                cur.left_peak_rt = prev.left_peak_rt


def _merge_peaks(peaks: list, edge_ratio: float, delta_ratio: float) -> list:
    """Merge adjacent same-m/z peaks whose shared windows overlap.
    Port of TwoStepDecomposition.mergePeaks().

    Returns _PeakData list for use as 'other_peaks' in _build_components.
    Adjacent peaks at the same m/z with total span < 1.1× combined widths are
    merged into a single wider chromatogram (union of RT points, last-write wins
    on RT collision, matching Java TreeMap.putAll behavior).
    """
    from collections import defaultdict

    peak_data_list = [
        _PeakData(
            source=p,
            left_peak_rt=float(p.rt_array[p.left_boundary_idx]),
            right_peak_rt=float(p.rt_array[p.right_boundary_idx]),
            rt_array=p.rt_array.copy(),
            intensity_array=p.intensity_array.copy(),
            apex_intensity=float(p.intensity_array[p.apex_idx]),
        )
        for p in peaks
    ]

    _correct_peak_boundaries(peak_data_list, edge_ratio, delta_ratio)

    groups: dict = defaultdict(list)
    for pd in peak_data_list:
        key = (pd.source.mz, pd.left_peak_rt, pd.right_peak_rt)
        groups[key].append(pd)

    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
            continue

        group.sort(key=lambda p: p.source.rt_apex)

        # Merge: union of RT points; later entries overwrite on collision
        rt_to_int: dict = {}
        for pd in group:
            for rt, intensity in zip(pd.rt_array, pd.intensity_array):
                rt_to_int[float(rt)] = float(intensity)

        merged_rts = np.array(sorted(rt_to_int.keys()))
        merged_ints = np.array([rt_to_int[rt] for rt in merged_rts])
        apex_intensity = float(merged_ints.max())

        # Source from highest-intensity peak (per PeakInfo.merge() in Java)
        best = max(group, key=lambda p: p.apex_intensity)

        result.append(_PeakData(
            source=best.source,
            left_peak_rt=group[0].left_peak_rt,
            right_peak_rt=group[-1].right_peak_rt,
            rt_array=merged_rts,
            intensity_array=merged_ints,
            apex_intensity=apex_intensity,
        ))

    return result


# ─── Layer 3: Clustering ──────────────────────────────────────────────────────

def _cluster_by_rt(peaks: list, eps: float, min_samples: int,
                   min_intensity: float) -> tuple[list, list]:
    """Cluster EIC peaks by apex RT using 1D DBSCAN.
    Port of TwoStepDecomposition.getRetTimeClusters().

    Returns (clusters, noise_peaks) where noise_peaks are DBSCAN label=-1 points.
    Clusters whose max apex intensity is below min_intensity are dropped.
    Returns clusters sorted by mean RT.
    """
    if not peaks:
        return [], []

    rt_values = np.array([[p.rt_apex] for p in peaks])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(rt_values)

    groups: dict = {}
    noise_peaks: list = []
    for peak, label in zip(peaks, labels):
        if label == -1:
            noise_peaks.append(peak)
            continue
        groups.setdefault(label, []).append(peak)

    filtered = [
        cluster for cluster in groups.values()
        if max(p.intensity_array[p.apex_idx] for p in cluster) >= min_intensity
    ]

    filtered.sort(key=lambda c: float(np.mean([p.rt_apex for p in c])))
    return filtered, noise_peaks


def _cluster_by_shape(peaks: list, threshold: float) -> list:
    """Cluster peaks by elution profile similarity using complete-linkage hierarchical clustering.
    Port of TwoStepDecomposition.getShapeClusters().

    Distance metric: shape_similarity_angle (degrees, [0°, 90°]).
    threshold is the max cluster diameter (complete-linkage criterion).
    """
    n = len(peaks)
    if n == 1:
        return [peaks]

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            angle = shape_similarity_angle(peaks[i], peaks[j])
            dist_matrix[i, j] = angle
            dist_matrix[j, i] = angle

    Z = linkage(squareform(dist_matrix), method='complete')
    labels = fcluster(Z, t=threshold, criterion='distance')

    groups: dict = {}
    for peak, label in zip(peaks, labels):
        groups.setdefault(int(label), []).append(peak)
    return list(groups.values())


# ─── Layer 4: Peak filtering and selection ────────────────────────────────────

def _filter_peaks(cluster: list, params: DeconvolutionParams) -> list:
    """Filter model peak candidates by isShared, sharpness, and excluded m/z.
    Port of TwoStepDecomposition.filterPeaks().
    """
    result = []
    for peak in cluster:
        sliced = peak.intensity_array[peak.left_boundary_idx:peak.right_boundary_idx + 1]

        if params.use_is_shared and is_shared(
                sliced, params.edge_to_height_ratio, params.delta_to_height_ratio):
            continue

        if sharpness_yang(peak.rt_array, peak.intensity_array,
                          peak.left_boundary_idx, peak.right_boundary_idx) \
                < params.min_model_peak_sharpness:
            continue

        if any(abs(peak.mz - excl) <= params.excluded_mz_tolerance
               for excl in params.excluded_mz):
            continue

        result.append(peak)
    return result


def _find_model_peak(peaks: list, choice: str) -> Optional[EICPeak]:
    """Select model peak from a cluster by choice criterion.
    Port of TwoStepDecomposition.findModelPeak().
    """
    if not peaks:
        return None
    if choice == 'intensity':
        return max(peaks, key=lambda p: float(p.intensity_array[p.apex_idx]))
    if choice == 'mz':
        return max(peaks, key=lambda p: p.mz)
    # default: 'sharpness'
    return max(peaks, key=lambda p: sharpness_yang(
        p.rt_array, p.intensity_array, p.left_boundary_idx, p.right_boundary_idx))


# ─── Layer 5: NNLS decomposition ─────────────────────────────────────────────

def _build_components(model_peaks: list, other_peaks: list) -> list:
    """Decompose each EIC peak into model peak contributions via NNLS.
    Port of TwoStepDecomposition.buildComponents().

    For each other_peak, finds model peaks whose rt_apex falls within the
    other peak's original apex window (left_boundary_idx..right_boundary_idx).
    Normalizes all chromatograms by apex intensity, solves NNLS to find
    non-negative coefficients, and accumulates mz contributions into spectra.

    Boundary check uses the ORIGINAL apex window from EICPeak
    (left_boundary_idx/right_boundary_idx), NOT the expanded shared-window
    left_peak_rt/right_peak_rt. This matches Java leftApexIndex/rightApexIndex.
    """
    spectra: dict = {id(mp): {} for mp in model_peaks}

    for other in other_peaks:
        src = other.source
        left_rt = float(src.rt_array[src.left_boundary_idx])
        right_rt = float(src.rt_array[src.right_boundary_idx])

        candidates = [mp for mp in model_peaks if left_rt <= mp.rt_apex <= right_rt]
        if not candidates:
            continue

        all_rts = other.rt_array.copy()
        for mp in candidates:
            all_rts = np.union1d(all_rts, mp.rt_array)

        s0_raw = np.interp(all_rts, other.rt_array, other.intensity_array)
        s0 = s0_raw / other.apex_intensity if other.apex_intensity > 0 else s0_raw

        S = np.zeros((len(all_rts), len(candidates)))
        for j, mp in enumerate(candidates):
            col_raw = np.interp(all_rts, mp.rt_array, mp.intensity_array)
            apex_int = float(mp.intensity_array[mp.apex_idx])
            S[:, j] = col_raw / apex_int if apex_int > 0 else col_raw

        coeffs, _ = nnls(S, s0)

        for j, mp in enumerate(candidates):
            # += accumulation: multiple _PeakData objects may share the same src.mz
            spectra[id(mp)][src.mz] = (
                spectra[id(mp)].get(src.mz, 0.0) + float(coeffs[j]) * other.apex_intensity
            )

    return [
        DeconvolutedComponent(
            rt=mp.rt_apex,
            spectrum=spectra[id(mp)],
            model_peak_mz=mp.mz,
            model_peak_rt_array=mp.rt_array,
            model_peak_intensity_array=mp.intensity_array,
        )
        for mp in model_peaks
    ]


# ─── Entry point ──────────────────────────────────────────────────────────────

def deconvolve(peaks: list,
               params: Optional[DeconvolutionParams] = None,
               return_intermediates: bool = False) -> list | tuple:
    """Spectral deconvolution via the ADAP-GC 3.2 two-step decomposition algorithm.

    Clusters EIC peaks by retention time, selects model peaks by shape and
    sharpness, then decomposes all peaks into linear combinations of model peaks
    (NNLS) to construct fragmentation spectra for each detected analyte.

    Reference: Smirnov et al., J. Proteome Res. 2018, 17, 470-478.
    Java source: dulab.adap.workflow.TwoStepDecomposition.run()

    Args:
        peaks: Detected EIC peaks (one per m/z per chromatographic feature).
        params: Algorithm parameters. Uses default DeconvolutionParams if None.
        return_intermediates: If True, returns (components, intermediates_dict).

    Returns:
        List of DeconvolutedComponent, OR (list, dict) if return_intermediates=True.
    """
    if params is None:
        params = DeconvolutionParams()
    if not peaks:
        return []

    rt_clusters, noise_peaks = _cluster_by_rt(
        peaks, params.min_cluster_distance,
        params.min_cluster_size, params.min_cluster_intensity,
    )

    model_peaks: list = []
    result: list = []

    for cluster in rt_clusters:
        candidates = _filter_peaks(cluster, params)

        if len(candidates) == 0:
            # Fallback: no model peak candidates survived filtering.
            # Pick best from the full cluster and build a direct spectrum
            # by interpolating ALL input peaks (not just this cluster) at
            # the model peak's RT. This matches Java TwoStepDecomposition
            # lines 102-126 which iterate the full `peaks` list.
            model_peak = _find_model_peak(cluster, params.model_peak_choice)
            if model_peak is None:
                continue

            span_rt = model_peak.rt_apex
            spectrum: dict = {}
            for p in peaks:  # ALL input peaks
                p_left_rt = float(p.rt_array[p.left_boundary_idx])
                p_right_rt = float(p.rt_array[p.right_boundary_idx])
                if p_left_rt <= span_rt <= p_right_rt:
                    spectrum[p.mz] = float(np.interp(span_rt, p.rt_array, p.intensity_array))

            result.append(DeconvolutedComponent(
                rt=model_peak.rt_apex,
                spectrum=spectrum,
                model_peak_mz=model_peak.mz,
                model_peak_rt_array=model_peak.rt_array,
                model_peak_intensity_array=model_peak.intensity_array,
            ))

        elif len(candidates) == 1:
            model_peaks.append(candidates[0])

        else:
            shape_clusters = _cluster_by_shape(candidates, params.shape_sim_threshold)
            for sc in shape_clusters:
                mp = _find_model_peak(sc, params.model_peak_choice)
                if mp is not None:
                    model_peaks.append(mp)

    other_peaks = _merge_peaks(
        peaks, params.edge_to_height_ratio, params.delta_to_height_ratio
    )
    result += _build_components(model_peaks, other_peaks)

    if return_intermediates:
        intermediates = {
            'rt_clusters': rt_clusters,
            'noise_peaks': noise_peaks,
            'model_peaks': model_peaks,
        }
        return result, intermediates

    return result
