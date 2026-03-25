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
    rt_apex: float           # retention time of peak apex (minutes)
    mz: float                # m/z of this extracted ion chromatogram
    rt_array: np.ndarray     # RT time points spanning the peak window
    intensity_array: np.ndarray  # intensities at those time points
    left_boundary_idx: int   # index into rt_array / intensity_array
    right_boundary_idx: int
    apex_idx: int            # index of apex in rt_array / intensity_array
```

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
Port of `FeatureTools.sharpnessYang()` (line 542). Finds the baseline between left/right boundary points, computes 25th-percentile height above baseline, collects slopes from apex to all points above that height on each side, returns `(median_left_slope - median_right_slope) / 2`. Uses index deltas (not time deltas) for slope computation, matching the Java. Returns -1.0 for degenerate peaks (flat, zero-height, or insufficient points).

**`is_shared(intensity_array, edge_to_height, delta_to_height) -> bool`**
Port of `FeatureTools.isShared(List<Double>, ...)` (line 185). Returns True if: (a) more than one local maximum exists in the array, or (b) any of left/apex, right/apex, or |left-right|/apex ratios exceed their respective thresholds. Called on the full windowed intensity array (already sliced to left..right).

**`shape_similarity_angle(peak_a: EICPeak, peak_b: EICPeak) -> float`**
Port of `Math.continuous_dot_product()`. Interpolates both chromatograms onto the union of their RT grids using `numpy.interp`. Computes norms via `numpy.trapz`. Returns `degrees(arccos(dot / (norm_a * norm_b)))`, clipped to [0°, 90°]. This angle is the pairwise distance metric for shape clustering.

### Layer 2: Internal boundary/merge helpers

**`_correct_peak_boundaries(peak_data_list, edge_ratio, delta_ratio) -> None`**
Port of `FeatureTools.correctPeakBoundaries()`. Groups `_PeakData` objects by m/z, sorts each group by RT. For each adjacent pair: checks if the shared windows should be expanded (i.e., `total_span < 1.1 × combined_widths` where spans are measured in RT minutes). If so, sets `left_peak_idx` of the right peak to `left_peak_idx` of the left peak, and vice versa. Mutates in place.

**`_merge_peaks(peaks, edge_ratio, delta_ratio) -> list[_PeakData]`**
Wraps all input `EICPeak` objects in `_PeakData` (which carries mutable `left_peak_idx`, `right_peak_idx` initialized to the input boundaries). Calls `_correct_peak_boundaries`. Groups by `(mz, left_peak_idx, right_peak_idx)`. Merges chromatograms within each group by taking the union of RT points (same m/z, so no conflict). Returns merged `_PeakData` list. This is the "other peaks" list consumed by NNLS decomposition.

`_PeakData` is a private dataclass — not part of the public API:

```python
@dataclass
class _PeakData:
    source: EICPeak          # original input peak
    left_peak_idx: int       # shared window left (may be expanded by _correct_peak_boundaries)
    right_peak_idx: int      # shared window right
    # merged chromatogram fields (set during merge step):
    rt_array: np.ndarray
    intensity_array: np.ndarray
    apex_intensity: float
    apex_idx: int            # index of apex in merged rt_array
```

### Layer 3: Clustering

**`_cluster_by_rt(peaks, eps, min_samples, min_intensity) -> list[list[EICPeak]]`**
sklearn `DBSCAN(eps=eps, min_samples=min_samples)` on the 1D array of `rt_apex` values. Noise points (label -1) are discarded. Each cluster is filtered by its maximum peak intensity against `min_intensity`. Returns clusters sorted by mean RT.

**`_cluster_by_shape(peaks, threshold) -> list[list[EICPeak]]`**
Computes pairwise `shape_similarity_angle` for all peak pairs. Builds a square distance matrix. Calls `scipy.cluster.hierarchy.linkage(squareform(dist_matrix), method='complete')` then `fcluster(Z, t=threshold, criterion='distance')`. Returns list of clusters.

### Layer 4: Peak filtering and selection

**`_filter_peaks(cluster, params) -> list[EICPeak]`**
For each peak in the cluster: skip if `use_is_shared` and `is_shared(...)` is True; skip if `sharpness_yang(...) < min_model_peak_sharpness`; skip if mz is within `excluded_mz_tolerance` of any value in `excluded_mz`. Returns surviving peaks.

**`_find_model_peak(peaks, choice) -> EICPeak`**
Picks the peak with highest value of: sharpness (default), intensity, or mz depending on `choice`.

### Layer 5: NNLS decomposition

**`_build_components(model_peaks, other_peaks) -> list[DeconvolutedComponent]`**
Port of `TwoStepDecomposition.buildComponents()`. For each `_PeakData` in `other_peaks`:
1. Find model peaks whose `apex_idx` (mapped to global RT) falls within `[left_boundary, right_boundary]` of this other peak. Use RT comparison (`model_peak.rt_apex` between `other_peak.rt_array[left]` and `other_peak.rt_array[right]`).
2. Build union RT grid from other peak + all candidate model peak RT arrays.
3. Interpolate each to the union grid via `numpy.interp`.
4. Normalize each by its apex intensity.
5. Assemble matrix `S` (shape: `[n_timepoints, n_model_peaks]`), target vector `s0`.
6. Call `scipy.optimize.nnls(S, s0)` → coefficients.
7. For each model peak `i`: add `coeff[i] * other_peak.apex_intensity` to `spectra[model_peak][other_peak.mz]`.

Returns one `DeconvolutedComponent` per model peak.

### Entry point

**`deconvolve(peaks, params) -> list[DeconvolutedComponent]`**
Orchestrates the full pipeline:

```
1. _cluster_by_rt(peaks, ...) → rt_clusters
2. For each rt_cluster:
   a. _filter_peaks(cluster, params) → candidates
   b. if len(candidates) == 0:
        model_peak = _find_model_peak(cluster, params.model_peak_choice)
        Build direct spectrum: for all peaks whose boundary spans model_peak.rt_apex,
        interpolate intensity at rt_apex and add mz → intensity to spectrum.
        Append DeconvolutedComponent directly to result (skip buildComponents).
   c. elif len(candidates) == 1:
        model_peaks.append(candidates[0])
   d. else:
        shape_clusters = _cluster_by_shape(candidates, params.shape_sim_threshold)
        for each shape_cluster: model_peaks.append(_find_model_peak(...))
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
| `test_sharpness_yang_gaussian` | Gaussian-shaped array | sharpness > 10 |
| `test_sharpness_yang_flat` | Constant array | returns -1.0 |
| `test_sharpness_yang_asymmetric` | Steep left, gradual right | 0 < sharpness < Gaussian case |
| `test_is_shared_clean_peak` | Symmetric peak, low boundaries | False |
| `test_is_shared_high_boundary` | Left boundary > 0.3 × apex | True |
| `test_is_shared_bimodal` | Two-hump intensity array | True |
| `test_shape_similarity_angle_identical` | Same peak twice | angle ≈ 0° |
| `test_shape_similarity_angle_non_overlapping` | Peaks at opposite ends of RT range | angle ≈ 90° |
| `test_shape_similarity_angle_range` | Any valid pair | angle ∈ [0°, 90°] |

### Clustering tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_cluster_by_rt_two_groups` | 4 peaks: 2 near 1.0 min, 2 near 5.0 min, eps=0.01 | 2 clusters |
| `test_cluster_by_rt_intensity_filter` | Cluster with max intensity < threshold | cluster dropped |
| `test_cluster_by_rt_noise_dropped` | Single isolated peak | 0 clusters returned |
| `test_cluster_by_shape_similar` | 3 identical-shape + 1 different-shape peaks | 2 shape clusters |

### NNLS decomposition tests

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_build_components_single_model` | 1 model peak, 1 other peak at same RT | spectrum has that m/z with intensity > 0 |
| `test_build_components_no_overlap` | Model peak apex outside other peak boundary | that m/z not in spectrum (or coeff ≈ 0) |
| `test_build_components_two_models` | 2 non-overlapping model peaks, 1 EIC overlapping only first | second model spectrum entry for that m/z ≈ 0 |

### End-to-end smoke test

`test_deconvolve_synthetic`: Construct ~10 `EICPeak` objects representing 2 simulated analytes co-eluting at 2.0 min and 2.05 min, each with 4–5 distinct m/z values. Run `deconvolve()` with default params. Verify: (a) returns 2 components, (b) each component's spectrum contains the m/z values associated with that analyte, (c) no exceptions raised.

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
