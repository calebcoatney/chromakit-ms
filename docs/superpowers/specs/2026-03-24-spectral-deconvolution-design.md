# Spectral Deconvolution Module — Design Spec

**Date:** 2026-03-24
**Algorithm:** ADAP-GC 3.2 (Smirnov et al., J. Proteome Res. 2018, 17, 470–478)
**Reference implementation:** `deconvolution/adap-gc-source-reference/`
**Output:** `deconvolution/spectral_deconvolution.py` + `deconvolution/test_spectral_deconvolution.py`

---

## Purpose

Implement spectral deconvolution for GC-MS data: given a list of detected EIC peaks (one peak per m/z per chromatographic feature), group peaks from co-eluting analytes and construct a fragmentation spectrum for each detected compound. This module is standalone; integration into chromakit-ms is deferred.

The algorithm is a Python translation of `TwoStepDecomposition.java` from the `dulab.adap` library (GPL v2). The implementation is idiomatic Python using numpy, scipy, and scikit-learn — not a 1:1 Java translation.

---

## Source References

| Java file | Role |
|-----------|------|
| `core_algorithm/TwoStepDecomposition.java` | Main pipeline |
| `core_algorithm/Optimization.java` | NNLS decomposition (replaced by `scipy.optimize.nnls`) |
| `core_algorithm/Clustering.java` | Hierarchical shape clustering (replaced by `scipy.cluster.hierarchy`) |
| `utilities/FeatureTools.java` | `sharpnessYang()`, `isShared()`, `correctPeakBoundaries()` |
| `utilities/Math.java` | `continuous_dot_product()`, `interpolate()` |
| `data_model/Peak.java`, `PeakInfo.java`, `Component.java` | Data structures |

---

## Public API

### Input

```python
@dataclass
class EICPeak:
    rt_apex: float               # retention time of peak apex (minutes)
    mz: float                    # m/z of this extracted ion chromatogram
    rt_array: np.ndarray         # RT time points spanning the peak window
    intensity_array: np.ndarray  # intensities at those time points
    left_boundary_idx: int       # index into rt_array / intensity_array (apex window left)
    right_boundary_idx: int      # index into rt_array / intensity_array (apex window right)
    apex_idx: int                # index of apex in rt_array / intensity_array
```

`left_boundary_idx` and `right_boundary_idx` define the apex integration window (Java `leftApexIndex`/`rightApexIndex`). These are different from the internal "shared window" indices used by `_merge_peaks` (Java `leftPeakIndex`/`rightPeakIndex`), which are computed and tracked internally and never exposed in the public API.

### Output

```python
@dataclass
class DeconvolutedComponent:
    rt: float                              # apex RT of the model peak (minutes)
    spectrum: dict[float, float]           # m/z -> intensity (fragmentation spectrum)
    model_peak_mz: float                   # m/z of the representative EIC peak
    model_peak_rt_array: np.ndarray        # elution profile RT axis of model peak
    model_peak_intensity_array: np.ndarray # elution profile intensities of model peak
```

### Parameters

```python
@dataclass
class DeconvolutionParams:
    min_cluster_distance: float = 0.005    # DBSCAN eps (minutes)
    min_cluster_size: int = 2              # DBSCAN min_samples
    min_cluster_intensity: float = 200.0   # drop clusters below this peak intensity
    use_is_shared: bool = True             # filter chromatographically unresolved peaks
    edge_to_height_ratio: float = 0.3      # boundary/apex threshold for is_shared
    delta_to_height_ratio: float = 0.3     # |left-right|/apex threshold for is_shared
    min_model_peak_sharpness: float = 10.0 # minimum sharpness for model peak candidates
    shape_sim_threshold: float = 30.0      # max angle (degrees) for same shape cluster
    model_peak_choice: str = "sharpness"   # "sharpness", "intensity", or "mz"
    excluded_mz: list[float] = field(default_factory=list)  # empty = no exclusions
    excluded_mz_tolerance: float = 0.5     # ± tolerance for excluded_mz matching
```

`excluded_mz` defaults to empty. Callers who process TMS-derivatized metabolomics data should pass `[73, 147, 221]` explicitly.

### Entry point

```python
def deconvolve(peaks: list[EICPeak], params: DeconvolutionParams) -> list[DeconvolutedComponent]:
    ...
```

---

## Module Structure

Single file `deconvolution/spectral_deconvolution.py`. All functions are module-level (no classes). Private helpers are prefixed with `_`.

### Layer 1: Math primitives

**`sharpness_yang(rt_array, intensity_array, left, right) -> float`**

Port of `FeatureTools.sharpnessYang()` (line 542). Algorithm:

1. Find the apex index (max intensity) within `[left, right]`.
2. Compute the baseline line connecting `intensity_array[left]` to `intensity_array[right]`.
3. Compute `p25_height = 0.25 * (apex_intensity - baseline_at_apex) + baseline_at_apex`. If this is negative, return -1.0.
4. Collect left slopes `(apex_intensity - intensity_array[i]) / (apex_idx - i)` for all `i` in `[left, apex_idx)` where `intensity_array[i] >= p25_height`.
5. Collect right slopes `(intensity_array[i] - apex_intensity) / (i - apex_idx)` for all `i` in `(apex_idx, right]` where `intensity_array[i] >= p25_height`.
6. Sort each list and compute medians.
7. Return values:
   - Both empty → return -1.0
   - Only left non-empty → return `median_left`
   - Only right non-empty → return `median_right` (note: right slopes are negative for a well-formed peak, so this value will be negative; callers compare against a positive threshold)
   - Both non-empty → return `(median_left - median_right) / 2.0`

**Slope uses index deltas, not time deltas** (matching Java lines 591, 607). This makes sharpness unit-independent of scan rate.

**`is_shared(intensity_array, edge_to_height, delta_to_height) -> bool`**

Port of `FeatureTools.isShared(List<Double>, ...)` (line 185). `intensity_array` is pre-sliced to `peak.intensity_array[peak.left_boundary_idx : peak.right_boundary_idx + 1]` by the caller. Algorithm:

1. Track the running absolute maximum across the array.
2. Count local maxima (points strictly greater than both neighbors, handling plateau runs via skip-ahead).
3. If `local_maxima_count > 1` → return True.
4. Compute `left_to_apex = intensity_array[0] / absolute_maximum`.
5. Compute `right_to_apex = intensity_array[-1] / absolute_maximum`.
6. Compute `delta_to_apex = abs(intensity_array[0] - intensity_array[-1]) / absolute_maximum`.
7. Return `left_to_apex >= edge_to_height or right_to_apex >= edge_to_height or delta_to_apex >= delta_to_height`.

**`shape_similarity_angle(peak_a: EICPeak, peak_b: EICPeak) -> float`**

Port of `Math.continuous_dot_product()` + angle computation from `TwoStepDecomposition.getShapeClusters()`. Algorithm:

1. Union of RT grids: `all_rt = np.union1d(peak_a.rt_array, peak_b.rt_array)`.
2. Interpolate: `a = np.interp(all_rt, peak_a.rt_array, peak_a.intensity_array)`, same for `b`. (`np.interp` clamps to boundary values outside the range.)
3. Compute continuous norms: `norm_a = sqrt(np.trapz(a**2, all_rt))`, same for `norm_b`.
4. Continuous dot: `dot = np.trapz(a * b, all_rt)`.
5. `cos_angle = np.clip(dot / (norm_a * norm_b), 0.0, 1.0)`.
6. Return `degrees(arccos(cos_angle))`.

The angle is in [0°, 90°]. Note: norms are computed from the continuous (trapz) inner product, **not** from apex intensity scalars. This matches `peak.getNorm()` in the Java, which is `sqrt(continuous_dot_product(chromatogram, chromatogram))`.

---

### Layer 2: Internal boundary/merge helpers

**`_PeakData`** — private dataclass, never exposed in public API:

```python
@dataclass
class _PeakData:
    source: EICPeak          # original input peak (rt_apex, mz, left/right_boundary_idx preserved)
    left_peak_rt: float      # shared window left boundary in RT minutes (Java leftPeakIndex→RT)
    right_peak_rt: float     # shared window right boundary in RT minutes (Java rightPeakIndex→RT)
    rt_array: np.ndarray     # chromatogram RT axis (may be merged)
    intensity_array: np.ndarray  # chromatogram intensities (may be merged)
    apex_intensity: float    # max intensity in this chromatogram
```

`left_peak_rt` and `right_peak_rt` store RT minutes (not indices) to avoid ambiguity across peaks with different `rt_array` grids. They are initialized from `source.rt_array[source.left_boundary_idx]` and `source.rt_array[source.right_boundary_idx]`. `_correct_peak_boundaries` may expand them.

**`_correct_peak_boundaries(peak_data_list, edge_ratio, delta_ratio) -> None`**

Port of `FeatureTools.correctPeakBoundaries()`. Mutates `left_peak_rt`/`right_peak_rt` in place. Algorithm:

1. Group `_PeakData` objects by `source.mz`. Sort each group by `source.rt_apex`.
2. For each adjacent pair `(prev, cur)` in each group:
   a. Compute `combined_width = (prev.right_peak_rt - prev.left_peak_rt) + (cur.right_peak_rt - cur.left_peak_rt)`.
   b. Compute `total_width = cur.right_peak_rt - prev.left_peak_rt`.
   c. If `total_width < 1.1 * combined_width`: expand shared windows — set `prev.right_peak_rt = cur.right_peak_rt` and `cur.left_peak_rt = prev.left_peak_rt`.

**Note on Java fidelity:** The Java computes `mergeRight` and `mergeLeft` boolean conditions (checking boundary-to-apex ratios) but their `if (!mergeRight) continue` guards are commented out (lines 295, 322). The actual merge is gated solely by the `1.1 × combined_width` span check. This Python translation matches the commented-out (relaxed) behavior — the boolean conditions are not computed, only the span check is used.

**`_merge_peaks(peaks, edge_ratio, delta_ratio) -> list[_PeakData]`**

Port of `TwoStepDecomposition.mergePeaks()`. Algorithm:

1. Wrap each `EICPeak` in a `_PeakData` with `left_peak_rt` / `right_peak_rt` initialized from its `left_boundary_idx` / `right_boundary_idx`.
2. Call `_correct_peak_boundaries(peak_data_list, edge_ratio, delta_ratio)`.
3. Group by `(source.mz, left_peak_rt, right_peak_rt)`.
4. For each group: merge chromatograms by building a combined `rt_array` / `intensity_array` from the union of all group members' RT points. If two peaks in the group have the same RT point, the last-write wins (matching Java `TreeMap.putAll` behavior; group is sorted by `source.rt_apex`, so higher-RT peaks overwrite). Set `apex_intensity` to the maximum intensity in the merged chromatogram (from the highest-intensity peak in the group, per `PeakInfo.merge()` in Java).
5. Return the list of merged `_PeakData` objects.

This merged list is the "other peaks" passed to `_build_components` for NNLS decomposition.

---

### Layer 3: Clustering

**`_cluster_by_rt(peaks, eps, min_samples, min_intensity) -> list[list[EICPeak]]`**

`sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)` on the 1D array of `[p.rt_apex for p in peaks]`. Noise points (label -1) are discarded. Each cluster is filtered: if `max(p.intensity_array[p.apex_idx] for p in cluster) < min_intensity`, drop it. Returns clusters sorted by mean `rt_apex`. A cluster of exactly `min_samples` points is valid (it forms a core cluster). A single isolated peak becomes noise and is dropped.

**`_cluster_by_shape(peaks, threshold) -> list[list[EICPeak]]`**

1. Compute pairwise `shape_similarity_angle` for all `(i, j)` pairs → square distance matrix `D`.
2. `Z = scipy.cluster.hierarchy.linkage(squareform(D), method='complete')`.
3. `labels = fcluster(Z, t=threshold, criterion='distance')`.
4. Return list of groups, one per unique label.

---

### Layer 4: Peak filtering and selection

**`_filter_peaks(cluster, params) -> list[EICPeak]`**

For each peak in the cluster:
- If `params.use_is_shared` and `is_shared(peak.intensity_array[peak.left_boundary_idx : peak.right_boundary_idx + 1], params.edge_to_height_ratio, params.delta_to_height_ratio)` → skip.
- If `sharpness_yang(peak.rt_array, peak.intensity_array, peak.left_boundary_idx, peak.right_boundary_idx) < params.min_model_peak_sharpness` → skip.
- If `any(abs(peak.mz - excl) <= params.excluded_mz_tolerance for excl in params.excluded_mz)` → skip.

Returns surviving peaks.

**`_find_model_peak(peaks, choice) -> EICPeak`**

Returns the peak with the maximum value of: `sharpness_yang(...)` (choice="sharpness"), `peak.intensity_array[peak.apex_idx]` (choice="intensity"), or `peak.mz` (choice="mz").

---

### Layer 5: NNLS decomposition

**`_build_components(model_peaks: list[EICPeak], other_peaks: list[_PeakData]) -> list[DeconvolutedComponent]`**

Port of `TwoStepDecomposition.buildComponents()`. Accumulates a spectrum dict per model peak, then returns components.

For each `other` in `other_peaks`:
1. Find candidate model peaks: those where `other.source.rt_array[other.source.left_boundary_idx] <= model.rt_apex <= other.source.rt_array[other.source.right_boundary_idx]`. Use `model.rt_apex` directly — no index indirection needed. The boundary RT values come from the **original** `source.left_boundary_idx`/`source.right_boundary_idx` (the apex integration window), **not** from the expanded `left_peak_rt`/`right_peak_rt` shared-window fields.
2. If no candidate model peaks, skip this `other` peak (no NNLS to solve).
3. Build union RT grid from `other.rt_array` plus all candidate model `rt_array` fields.
4. Interpolate the other peak chromatogram onto the union grid: `s0_raw = np.interp(union_rt, other.rt_array, other.intensity_array)`.
5. Normalize: `s0 = s0_raw / other.apex_intensity`.
6. For each candidate model peak `m_i`: interpolate its chromatogram onto the union grid, normalize by `m_i.intensity_array[m_i.apex_idx]`. Stack as column `i` of matrix `S` (shape `[n_grid_points, n_candidates]`).
7. `coeffs, _ = scipy.optimize.nnls(S, s0)`.
8. For each candidate `m_i`: `spectra[m_i][other.source.mz] += coeffs[i] * other.apex_intensity`.

After all `other` peaks are processed, construct one `DeconvolutedComponent` per model peak using its accumulated spectrum dict.

---

### Entry point

**`deconvolve(peaks: list[EICPeak], params: DeconvolutionParams) -> list[DeconvolutedComponent]`**

```
1. rt_clusters = _cluster_by_rt(peaks, params.min_cluster_distance,
                                params.min_cluster_size, params.min_cluster_intensity)

2. model_peaks = []
   result = []

   For each rt_cluster:
     a. candidates = _filter_peaks(rt_cluster, params)

     b. if len(candidates) == 0:
          model_peak = _find_model_peak(rt_cluster, params.model_peak_choice)
          if model_peak is None: continue
          Build direct spectrum:
            - span_rt = model_peak.rt_apex
            - spectrum = {}
            - For each p in ALL INPUT PEAKS (not just rt_cluster):
                if p.rt_array[p.left_boundary_idx] <= span_rt <= p.rt_array[p.right_boundary_idx]:
                    spectrum[p.mz] = np.interp(span_rt, p.rt_array, p.intensity_array)
            - result.append(DeconvolutedComponent(
                rt=model_peak.rt_apex,
                spectrum=spectrum,
                model_peak_mz=model_peak.mz,
                model_peak_rt_array=model_peak.rt_array,
                model_peak_intensity_array=model_peak.intensity_array,
              ))

     c. elif len(candidates) == 1:
          model_peaks.append(candidates[0])

     d. else:
          shape_clusters = _cluster_by_shape(candidates, params.shape_sim_threshold)
          for sc in shape_clusters:
            model_peaks.append(_find_model_peak(sc, params.model_peak_choice))

3. other_peaks = _merge_peaks(peaks, params.edge_to_height_ratio, params.delta_to_height_ratio)

4. result += _build_components(model_peaks, other_peaks)

5. return result
```

---

## Testing Plan

File: `deconvolution/test_spectral_deconvolution.py`
Framework: pytest, no external fixtures, all data constructed inline.

### Math primitive tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_sharpness_yang_gaussian` | Gaussian-shaped array (steep symmetric peak) | sharpness > 10 |
| `test_sharpness_yang_flat` | Constant array (all same intensity) | returns -1.0 |
| `test_sharpness_yang_one_sided` | All points on left side below p25; right side has slope data | returns `median_right` value (not -1.0) |
| `test_is_shared_clean_peak` | Symmetric peak, boundaries < 0.3× apex | False |
| `test_is_shared_high_boundary` | Left boundary intensity = 0.5× apex, right = 0.05× | True (edge ratio exceeded) |
| `test_is_shared_bimodal` | Two-hump intensity array with two clear local maxima | True (multiple local maxima) |
| `test_shape_similarity_angle_identical` | Same `EICPeak` passed twice | angle ≈ 0° (< 1°) |
| `test_shape_similarity_angle_max_dissimilar` | One peak is early sharp spike, other is late sharp spike on the same shared RT grid (both present, overlapping but different shape) | angle > 45° |
| `test_shape_similarity_angle_range` | Arbitrary valid pair | angle ∈ [0°, 90°] |

Note on `test_shape_similarity_angle_max_dissimilar`: use two peaks with overlapping RT ranges but maximally different shapes (e.g., one peaks early, one peaks late) rather than non-overlapping peaks. `np.interp` clamps to boundary values outside each peak's RT range, so truly non-overlapping peaks will not give exactly 90° — an angle > 45° is the correct relaxed assertion.

### Boundary/merge tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_merge_peaks_adjacent_same_mz` | Two peaks at same m/z, close in RT with high right/left edges, span check passes | merged into 1 `_PeakData`; merged `rt_array` spans both windows |
| `test_merge_peaks_non_adjacent` | Two peaks at same m/z but RT span >> 1.1× combined widths | remain 2 separate `_PeakData` objects |
| `test_merge_peaks_different_mz` | Two peaks at different m/z at same RT | remain 2 separate `_PeakData` objects |

### Clustering tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_cluster_by_rt_two_groups` | 4 peaks: 2 near 1.0 min, 2 near 5.0 min, eps=0.01 | 2 clusters |
| `test_cluster_by_rt_intensity_filter` | One cluster with max intensity < `min_cluster_intensity` | that cluster dropped; 0 returned |
| `test_cluster_by_rt_noise_dropped` | Single isolated peak with `min_samples=2` | 0 clusters returned |
| `test_cluster_by_shape_groups_similar` | 3 peaks with near-identical shape + 1 clearly different | 2 shape clusters |

### NNLS decomposition tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_build_components_single_model` | 1 model peak, 1 `_PeakData` whose apex window spans model peak's RT | component spectrum has `other.source.mz` with intensity > 0 |
| `test_build_components_no_overlap` | Model peak RT outside `[left_boundary, right_boundary]` of other peak | other peak not included in NNLS; spectrum for that m/z is 0 or absent |
| `test_build_components_two_models_separated` | 2 model peaks at different RTs; 1 EIC peak spanning only the first model's RT | second model's spectrum has no contribution from the EIC peak |

### Fallback and end-to-end tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_deconvolve_zero_candidates_fallback` | All peaks in a cluster fail sharpness/isShared filters; construct peaks with `excluded_mz` set so all are rejected | `deconvolve()` returns at least 1 component built via direct spectrum (no NNLS), spectrum populated by all input peaks spanning the fallback model's RT |
| `test_deconvolve_synthetic` | ~10 `EICPeak` objects: 2 analytes at 2.0 and 2.05 min, each with 4–5 distinct m/z values; ensure peaks are sharp enough to pass filter | returns 2 components; each component's spectrum contains the m/z values for that analyte; no exceptions |

---

## Constraints and Explicit Non-Goals

- No MZmine GUI, file I/O, or framework code.
- No Java class hierarchies — use dataclasses or plain dicts.
- No JOptimizer or Colt — use `scipy.optimize.nnls`.
- No assumption of TMS derivatization in defaults — `excluded_mz` is empty by default.
- No chromakit-ms integration in this PR — wiring deferred.
- Module must be importable without chromakit-ms installed (no imports from `logic/`, `ui/`, or `api/`).

---

## Dependencies

All available in the existing `chromakit-env` conda environment:
- `numpy`
- `scipy` (`optimize.nnls`, `cluster.hierarchy`, `spatial.distance.squareform`)
- `scikit-learn` (`sklearn.cluster.DBSCAN`)
