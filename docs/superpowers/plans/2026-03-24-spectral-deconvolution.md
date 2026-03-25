# Spectral Deconvolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the ADAP-GC 3.2 spectral deconvolution algorithm as a standalone Python module that accepts detected EIC peaks and returns deconvoluted components with fragmentation spectra.

**Architecture:** Flat module (no classes), all functions module-level. Three layers: math primitives → pipeline stages → single entry point `deconvolve()`. Internal `_PeakData` dataclass carries mutable shared-window bounds for the merge step; never exposed publicly.

**Tech Stack:** Python, numpy, scipy (`nnls`, `cluster.hierarchy`, `spatial.distance`), scikit-learn (`DBSCAN`). All available in `chromakit-env`.

**Spec:** `docs/superpowers/specs/2026-03-24-spectral-deconvolution-design.md`

**Run all tests:** `cd deconvolution && conda run -n chromakit-env pytest test_spectral_deconvolution.py -v`

> All pytest commands must be run from the `deconvolution/` subdirectory. The test file uses a bare `from spectral_deconvolution import ...` which requires the working directory to be `deconvolution/`.

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `deconvolution/spectral_deconvolution.py` | **Create** | Full algorithm: dataclasses, math primitives, clustering, NNLS, entry point |
| `deconvolution/test_spectral_deconvolution.py` | **Create** | All pytest tests for above |

No other files are created or modified. This module has zero imports from `logic/`, `ui/`, or `api/`.

---

## Task 1: Scaffold both files

**Files:**
- Create: `deconvolution/spectral_deconvolution.py`
- Create: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Create the module scaffold**

```python
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
```

- [ ] **Step 2: Create the test file scaffold**

```python
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
```

- [ ] **Step 3: Verify imports work**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env python -c "import spectral_deconvolution; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit scaffold**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: scaffold spectral_deconvolution module and test file"
```

---

## Task 2: `sharpness_yang`

Port of `FeatureTools.sharpnessYang()` (line 542). Measures peak quality as average absolute slope near the apex. Uses **index deltas** (not time), matching the Java.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

Add to `test_spectral_deconvolution.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestSharpnessYang -v
```

Expected: 3 errors (`ImportError: cannot import name 'sharpness_yang'`)

- [ ] **Step 3: Implement `sharpness_yang`**

Add to `spectral_deconvolution.py` after the `_PeakData` dataclass:

```python
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

    p25 = 0.25 * (apex_intensity - baseline_at_apex)
    if p25 < 0.0:
        return -1.0
    p25 += baseline_at_apex

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestSharpnessYang -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement sharpness_yang (FeatureTools.sharpnessYang port)"
```

---

## Task 3: `is_shared`

Port of `FeatureTools.isShared(List<Double>, ...)` (line 185). Detects chromatographically unresolved peaks (multi-modal or high boundary ratios).

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestIsShared:
    def test_clean_symmetric_peak_not_shared(self):
        # Symmetric Gaussian with low boundaries → not shared
        peak = make_gaussian_peak(rt_center=5.0, width=0.3, height=1000.0, n_points=50)
        sliced = peak.intensity_array[peak.left_boundary_idx:peak.right_boundary_idx + 1]
        assert is_shared(sliced, 0.3, 0.3) is False

    def test_high_left_boundary_is_shared(self):
        # Left boundary = 50% of apex → edge_to_height ratio exceeded
        ints = np.array([500.0, 600.0, 800.0, 1000.0, 700.0, 400.0, 50.0])
        assert is_shared(ints, 0.3, 0.3) is True

    def test_bimodal_is_shared(self):
        # Two clear peaks → multiple local maxima
        ints = np.array([10.0, 500.0, 200.0, 600.0, 10.0])
        assert is_shared(ints, 0.3, 0.3) is True

    def test_high_delta_is_shared(self):
        # Left=400, right=50, apex=1000 → |400-50|/1000 = 0.35 > 0.3
        ints = np.array([400.0, 700.0, 1000.0, 600.0, 50.0])
        assert is_shared(ints, 0.3, 0.3) is True
```

- [ ] **Step 2: Run to verify failure**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestIsShared -v
```

Expected: 4 errors (ImportError)

- [ ] **Step 3: Implement `is_shared`**

Add after `sharpness_yang` in `spectral_deconvolution.py`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestIsShared -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement is_shared (FeatureTools.isShared port)"
```

---

## Task 4: `shape_similarity_angle`

Port of `Math.continuous_dot_product()` + angle from `TwoStepDecomposition.getShapeClusters()`. Returns degrees in [0°, 90°].

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestShapeSimilarityAngle:
    def test_identical_peaks_angle_near_zero(self):
        peak = make_gaussian_peak(rt_center=5.0, mz=100.0)
        angle = shape_similarity_angle(peak, peak)
        assert angle < 1.0  # degrees

    def test_angle_always_in_valid_range(self):
        peak_a = make_gaussian_peak(rt_center=4.0, mz=100.0)
        peak_b = make_gaussian_peak(rt_center=6.0, mz=200.0)
        angle = shape_similarity_angle(peak_a, peak_b)
        assert 0.0 <= angle <= 90.0

    def test_very_different_shapes_large_angle(self):
        # peak_a: early sharp spike; peak_b: late sharp spike on shared RT range
        rts = np.linspace(0, 10, 100)
        ints_a = np.zeros(100)
        ints_a[10] = 1000.0  # spike near start
        ints_b = np.zeros(100)
        ints_b[90] = 1000.0  # spike near end
        peak_a = EICPeak(rt_apex=rts[10], mz=100.0, rt_array=rts,
                         intensity_array=ints_a,
                         left_boundary_idx=0, right_boundary_idx=99, apex_idx=10)
        peak_b = EICPeak(rt_apex=rts[90], mz=200.0, rt_array=rts,
                         intensity_array=ints_b,
                         left_boundary_idx=0, right_boundary_idx=99, apex_idx=90)
        angle = shape_similarity_angle(peak_a, peak_b)
        # Both on the same RT grid with non-overlapping spikes → near 90°
        assert angle > 45.0
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestShapeSimilarityAngle -v
```

Expected: 3 errors (ImportError)

- [ ] **Step 3: Implement `shape_similarity_angle`**

Add after `is_shared`:

```python
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

    norm_a = np.sqrt(np.trapz(a ** 2, all_rt))
    norm_b = np.sqrt(np.trapz(b ** 2, all_rt))

    if norm_a == 0.0 or norm_b == 0.0:
        return 90.0

    dot = np.trapz(a * b, all_rt)
    cos_angle = np.clip(dot / (norm_a * norm_b), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestShapeSimilarityAngle -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement shape_similarity_angle (Math.continuous_dot_product port)"
```

---

## Task 5: `_correct_peak_boundaries` + `_merge_peaks`

Internal merge step: expands shared-window RT bounds for adjacent same-m/z peaks, then merges overlapping ones into a single wider chromatogram. Port of `FeatureTools.correctPeakBoundaries()` + `TwoStepDecomposition.mergePeaks()`.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestMergePeaks:
    def _make_adjacent_same_mz_peaks(self):
        """Two peaks at m/z=100 that are close enough to merge."""
        # Peak 1: rt 1.0–1.4, apex at 1.2
        rts1 = np.linspace(1.0, 1.4, 20)
        ints1 = 1000.0 * np.exp(-0.5 * ((rts1 - 1.2) / 0.08) ** 2)
        p1 = EICPeak(rt_apex=1.2, mz=100.0, rt_array=rts1, intensity_array=ints1,
                     left_boundary_idx=0, right_boundary_idx=19, apex_idx=int(np.argmax(ints1)))
        # Peak 2: rt 1.3–1.7, apex at 1.5 (overlapping window with p1)
        rts2 = np.linspace(1.3, 1.7, 20)
        ints2 = 800.0 * np.exp(-0.5 * ((rts2 - 1.5) / 0.08) ** 2)
        p2 = EICPeak(rt_apex=1.5, mz=100.0, rt_array=rts2, intensity_array=ints2,
                     left_boundary_idx=0, right_boundary_idx=19, apex_idx=int(np.argmax(ints2)))
        return p1, p2

    def test_adjacent_same_mz_merged(self):
        p1, p2 = self._make_adjacent_same_mz_peaks()
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert len(merged) == 1
        # Merged rt_array should span from p1 start to p2 end
        assert merged[0].rt_array.min() <= 1.0 + 1e-9
        assert merged[0].rt_array.max() >= 1.7 - 1e-9

    def test_non_overlapping_same_mz_not_merged(self):
        p1 = make_gaussian_peak(rt_center=1.0, width=0.1, mz=100.0, n_points=20)
        p2 = make_gaussian_peak(rt_center=5.0, width=0.1, mz=100.0, n_points=20)
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert len(merged) == 2

    def test_different_mz_not_merged(self):
        p1 = make_gaussian_peak(rt_center=2.0, mz=100.0, n_points=20)
        p2 = make_gaussian_peak(rt_center=2.0, mz=200.0, n_points=20)
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert len(merged) == 2

    def test_merged_apex_from_highest_intensity_peak(self):
        p1, p2 = self._make_adjacent_same_mz_peaks()
        # p1 apex intensity ~1000, p2 ~800; merged should use p1's apex_intensity
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert merged[0].apex_intensity == pytest.approx(
            max(p1.intensity_array.max(), p2.intensity_array.max()), rel=0.01
        )
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestMergePeaks -v
```

Expected: 4 errors (ImportError)

- [ ] **Step 3: Implement `_correct_peak_boundaries` and `_merge_peaks`**

Add after `shape_similarity_angle`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestMergePeaks -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement _correct_peak_boundaries and _merge_peaks"
```

---

## Task 6: `_cluster_by_rt`

Port of `TwoStepDecomposition.getRetTimeClusters()`. 1D DBSCAN on apex RT values.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestClusterByRT:
    def test_two_well_separated_groups(self):
        peaks = [
            make_gaussian_peak(rt_center=1.000, height=500.0, mz=float(i))
            for i in range(100, 103)
        ] + [
            make_gaussian_peak(rt_center=5.000, height=500.0, mz=float(i))
            for i in range(200, 203)
        ]
        clusters = _cluster_by_rt(peaks, eps=0.01, min_samples=2, min_intensity=100.0)
        assert len(clusters) == 2
        # Clusters should be sorted by mean RT
        mean_rts = [np.mean([p.rt_apex for p in c]) for c in clusters]
        assert mean_rts[0] < mean_rts[1]

    def test_low_intensity_cluster_dropped(self):
        # Two groups: one with intensity 500, one with intensity 50 (below threshold)
        high = [make_gaussian_peak(rt_center=1.0, height=500.0, mz=float(i)) for i in range(2)]
        low = [make_gaussian_peak(rt_center=5.0, height=50.0, mz=float(i)) for i in range(2)]
        clusters = _cluster_by_rt(high + low, eps=0.01, min_samples=2, min_intensity=100.0)
        assert len(clusters) == 1

    def test_isolated_peak_dropped_as_noise(self):
        # One peak alone with min_samples=2 → DBSCAN noise → dropped
        peaks = [make_gaussian_peak(rt_center=1.0, height=500.0, mz=100.0)]
        clusters = _cluster_by_rt(peaks, eps=0.01, min_samples=2, min_intensity=100.0)
        assert len(clusters) == 0

    def test_empty_input(self):
        assert _cluster_by_rt([], eps=0.01, min_samples=2, min_intensity=100.0) == []
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestClusterByRT -v
```

Expected: 4 errors (ImportError)

- [ ] **Step 3: Implement `_cluster_by_rt`**

Add after `_merge_peaks`:

```python
# ─── Layer 3: Clustering ──────────────────────────────────────────────────────

def _cluster_by_rt(peaks: list, eps: float, min_samples: int,
                   min_intensity: float) -> list:
    """Cluster EIC peaks by apex RT using 1D DBSCAN.
    Port of TwoStepDecomposition.getRetTimeClusters().

    Noise points (label -1) are discarded. Clusters whose max apex intensity
    is below min_intensity are dropped. Returns clusters sorted by mean RT.
    """
    if not peaks:
        return []

    rt_values = np.array([[p.rt_apex] for p in peaks])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(rt_values)

    groups: dict = {}
    for peak, label in zip(peaks, labels):
        if label == -1:
            continue
        groups.setdefault(label, []).append(peak)

    filtered = [
        cluster for cluster in groups.values()
        if max(p.intensity_array[p.apex_idx] for p in cluster) >= min_intensity
    ]

    filtered.sort(key=lambda c: float(np.mean([p.rt_apex for p in c])))
    return filtered
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestClusterByRT -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement _cluster_by_rt (DBSCAN on apex RT)"
```

---

## Task 7: `_cluster_by_shape`

Complete-linkage hierarchical clustering on `shape_similarity_angle` pairwise distances. Port of `TwoStepDecomposition.getShapeClusters()`.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestClusterByShape:
    def test_identical_shapes_in_one_cluster(self):
        # Three peaks with same shape → all in one cluster at threshold=30°
        peaks = [make_gaussian_peak(rt_center=5.0, mz=float(m)) for m in [100, 200, 300]]
        clusters = _cluster_by_shape(peaks, threshold=30.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_different_shapes_split_into_clusters(self):
        rts = np.linspace(0, 10, 100)
        # Early spike
        ints_early = np.zeros(100); ints_early[10] = 1000.0
        # Late spike
        ints_late = np.zeros(100); ints_late[90] = 1000.0

        def make_spike(ints, mz):
            return EICPeak(rt_apex=rts[np.argmax(ints)], mz=mz,
                           rt_array=rts, intensity_array=ints,
                           left_boundary_idx=0, right_boundary_idx=99,
                           apex_idx=int(np.argmax(ints)))

        early = [make_spike(ints_early.copy(), float(m)) for m in [100, 101, 102]]
        late = [make_spike(ints_late.copy(), float(m)) for m in [200, 201, 202]]
        clusters = _cluster_by_shape(early + late, threshold=30.0)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [3, 3]

    def test_single_peak_returns_one_cluster(self):
        peak = make_gaussian_peak()
        clusters = _cluster_by_shape([peak], threshold=30.0)
        assert len(clusters) == 1
        assert clusters[0][0] is peak
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestClusterByShape -v
```

Expected: 3 errors (ImportError)

- [ ] **Step 3: Implement `_cluster_by_shape`**

Add after `_cluster_by_rt`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestClusterByShape -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement _cluster_by_shape (hierarchical shape clustering)"
```

---

## Task 8: `_filter_peaks` + `_find_model_peak`

Port of `TwoStepDecomposition.filterPeaks()` and the model peak selection logic.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestFilterPeaks:
    def test_sharp_clean_peak_passes(self):
        peak = make_gaussian_peak(rt_center=5.0, width=0.2, height=1000.0, n_points=50)
        params = DeconvolutionParams(min_model_peak_sharpness=1.0)
        result = _filter_peaks([peak], params)
        assert len(result) == 1

    def test_low_sharpness_removed(self):
        # Near-flat peak won't pass high sharpness threshold
        rt = np.linspace(0, 1, 20)
        ints = np.ones(20) * 500.0
        ints[10] = 510.0  # tiny apex
        peak = EICPeak(rt_apex=0.5, mz=100.0, rt_array=rt, intensity_array=ints,
                       left_boundary_idx=0, right_boundary_idx=19, apex_idx=10)
        params = DeconvolutionParams(min_model_peak_sharpness=10.0)
        result = _filter_peaks([peak], params)
        assert len(result) == 0

    def test_excluded_mz_removed(self):
        peak = make_gaussian_peak(mz=73.0, width=0.2, height=1000.0, n_points=50)
        params = DeconvolutionParams(
            min_model_peak_sharpness=1.0,
            excluded_mz=[73.0], excluded_mz_tolerance=0.5
        )
        result = _filter_peaks([peak], params)
        assert len(result) == 0

    def test_shared_peak_removed_when_enabled(self):
        # Bimodal peak should be filtered when use_is_shared=True
        rt = np.linspace(0, 1, 20)
        ints = np.zeros(20)
        ints[0] = 10.0; ints[5] = 500.0; ints[10] = 200.0; ints[15] = 600.0; ints[19] = 10.0
        peak = EICPeak(rt_apex=0.789, mz=100.0, rt_array=rt, intensity_array=ints,
                       left_boundary_idx=0, right_boundary_idx=19, apex_idx=15)
        params = DeconvolutionParams(use_is_shared=True, min_model_peak_sharpness=0.0)
        result = _filter_peaks([peak], params)
        assert len(result) == 0


class TestFindModelPeak:
    def test_picks_sharpest_peak(self):
        sharp = make_gaussian_peak(rt_center=5.0, width=0.1, height=1000.0, n_points=50)
        broad = make_gaussian_peak(rt_center=5.0, width=0.5, height=1000.0, n_points=50)
        result = _find_model_peak([broad, sharp], 'sharpness')
        assert result is sharp

    def test_picks_highest_intensity(self):
        low = make_gaussian_peak(height=500.0, mz=100.0)
        high = make_gaussian_peak(height=2000.0, mz=200.0)
        result = _find_model_peak([low, high], 'intensity')
        assert result is high

    def test_returns_none_for_empty_list(self):
        assert _find_model_peak([], 'sharpness') is None
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestFilterPeaks test_spectral_deconvolution.py::TestFindModelPeak -v
```

Expected: 7 errors (ImportError)

- [ ] **Step 3: Implement `_filter_peaks` and `_find_model_peak`**

Add after `_cluster_by_shape`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestFilterPeaks test_spectral_deconvolution.py::TestFindModelPeak -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement _filter_peaks and _find_model_peak"
```

---

## Task 9: `_build_components`

NNLS decomposition of each EIC peak into model peak contributions. Port of `TwoStepDecomposition.buildComponents()`. Replaces ~200 lines of JOptimizer/Colt with `scipy.optimize.nnls`.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestBuildComponents:
    def _make_other_peak(self, source_peak: EICPeak) -> '_PeakData':
        from spectral_deconvolution import _PeakData
        return _PeakData(
            source=source_peak,
            left_peak_rt=float(source_peak.rt_array[source_peak.left_boundary_idx]),
            right_peak_rt=float(source_peak.rt_array[source_peak.right_boundary_idx]),
            rt_array=source_peak.rt_array.copy(),
            intensity_array=source_peak.intensity_array.copy(),
            apex_intensity=float(source_peak.intensity_array[source_peak.apex_idx]),
        )

    def test_single_model_peak_has_mz_in_spectrum(self):
        model = make_gaussian_peak(rt_center=5.0, mz=100.0, height=1000.0, n_points=50)
        other_src = make_gaussian_peak(rt_center=5.0, mz=200.0, height=800.0, n_points=50)
        other = self._make_other_peak(other_src)
        components = _build_components([model], [other])
        assert len(components) == 1
        assert 200.0 in components[0].spectrum
        assert components[0].spectrum[200.0] > 0.0

    def test_model_apex_outside_other_boundary_gives_zero(self):
        # Other peak: RT 1.0–2.0; model peak: RT=5.0 (outside)
        model = make_gaussian_peak(rt_center=5.0, mz=100.0)
        other_src = make_gaussian_peak(rt_center=1.5, mz=200.0)
        other = self._make_other_peak(other_src)
        components = _build_components([model], [other])
        # model.rt_apex=5.0 is outside other's boundary [~1.0, ~2.0]
        assert len(components) == 1
        assert components[0].spectrum.get(200.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_two_models_separated_contributions(self):
        # model1 at RT=2.0, model2 at RT=8.0
        # other peaks: one at RT=2.0 (only within model1's range),
        #              one at RT=8.0 (only within model2's range)
        model1 = make_gaussian_peak(rt_center=2.0, mz=100.0, height=1000.0, n_points=30)
        model2 = make_gaussian_peak(rt_center=8.0, mz=200.0, height=1000.0, n_points=30)

        # other peak at RT=2.0–2.5 → only model1's RT=2.0 falls inside
        rts_other = np.linspace(1.5, 2.5, 30)
        ints_other = 500.0 * np.exp(-0.5 * ((rts_other - 2.0) / 0.2) ** 2)
        other_src = EICPeak(rt_apex=2.0, mz=300.0, rt_array=rts_other,
                            intensity_array=ints_other, left_boundary_idx=0,
                            right_boundary_idx=29, apex_idx=int(np.argmax(ints_other)))
        from spectral_deconvolution import _PeakData
        other = _PeakData(source=other_src, left_peak_rt=1.5, right_peak_rt=2.5,
                          rt_array=rts_other, intensity_array=ints_other,
                          apex_intensity=float(ints_other.max()))

        components = _build_components([model1, model2], [other])
        assert len(components) == 2
        comp1 = next(c for c in components if c.model_peak_mz == 100.0)
        comp2 = next(c for c in components if c.model_peak_mz == 200.0)
        # model1 should have m/z=300 in its spectrum; model2 should not
        assert comp1.spectrum.get(300.0, 0.0) > 0.0
        assert comp2.spectrum.get(300.0, 0.0) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestBuildComponents -v
```

Expected: 3 errors (ImportError)

- [ ] **Step 3: Implement `_build_components`**

Add after `_find_model_peak`:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestBuildComponents -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement _build_components (NNLS spectral decomposition)"
```

---

## Task 10: `deconvolve()` entry point

Main pipeline orchestrator. Port of `TwoStepDecomposition.run()`. Includes the 0-candidates fallback (direct spectrum path) and the normal NNLS path.

**Files:**
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

- [ ] **Step 1: Write failing tests**

```python
class TestDeconvolve:
    def test_empty_input_returns_empty(self):
        assert deconvolve([]) == []

    def test_zero_candidates_fallback_builds_direct_spectrum(self):
        # Make peaks that will ALL fail the sharpness filter
        # Use near-flat peaks: tiny apex above a high baseline
        params = DeconvolutionParams(
            min_cluster_size=2,
            min_cluster_distance=0.05,
            min_cluster_intensity=10.0,
            min_model_peak_sharpness=1000.0,  # impossibly high threshold
            use_is_shared=False,
        )
        # Three peaks at same RT cluster, different m/z — all fail sharpness
        peaks = []
        for mz in [100.0, 200.0, 300.0]:
            p = make_gaussian_peak(rt_center=5.0, mz=mz, height=500.0, n_points=30)
            peaks.append(p)

        result = deconvolve(peaks, params)
        # Fallback should produce exactly 1 component (direct spectrum)
        assert len(result) == 1
        # Direct spectrum: all m/z values whose boundary spans the model RT
        assert set(result[0].spectrum.keys()) == {100.0, 200.0, 300.0}

    def test_single_analyte_two_mz_values(self):
        # Two peaks at same RT → should produce 1 component
        params = DeconvolutionParams(
            min_cluster_distance=0.05,
            min_cluster_size=2,
            min_cluster_intensity=100.0,
            min_model_peak_sharpness=1.0,
            use_is_shared=False,
        )
        peaks = [
            make_gaussian_peak(rt_center=5.0, mz=100.0, width=0.3, height=1000.0, n_points=40),
            make_gaussian_peak(rt_center=5.0, mz=200.0, width=0.3, height=800.0, n_points=40),
        ]
        result = deconvolve(peaks, params)
        assert len(result) >= 1
        # The component's spectrum should contain at least one of the m/z values
        all_mzs = set()
        for comp in result:
            all_mzs.update(comp.spectrum.keys())
        assert len(all_mzs) > 0

    def test_default_params_used_when_none(self):
        peaks = [make_gaussian_peak(rt_center=5.0, mz=float(m), height=1000.0)
                 for m in [100, 200]]
        # Should not raise
        result = deconvolve(peaks)
        assert isinstance(result, list)

    def test_synthetic_two_analytes(self):
        """Two analytes at well-separated RTs, each with 3 m/z values.

        Use eps=0.01 and RT gap of 0.5 min so each analyte forms its own
        RT cluster — guaranteeing two components with distinct spectra.
        """
        params = DeconvolutionParams(
            min_cluster_distance=0.01,   # eps < 0.5 min gap → two clusters
            min_cluster_size=2,
            min_cluster_intensity=100.0,
            min_model_peak_sharpness=1.0,
            use_is_shared=False,
            shape_sim_threshold=30.0,
        )
        # Analyte A at RT=2.0: m/z 100, 150, 200
        # Analyte B at RT=2.5: m/z 250, 300, 350  (0.5 min gap >> eps=0.01)
        mz_a = {100.0, 150.0, 200.0}
        mz_b = {250.0, 300.0, 350.0}
        peaks = []
        for mz in mz_a:
            peaks.append(make_gaussian_peak(rt_center=2.0, mz=mz,
                                            width=0.2, height=1000.0, n_points=40))
        for mz in mz_b:
            peaks.append(make_gaussian_peak(rt_center=2.5, mz=mz,
                                            width=0.2, height=800.0, n_points=40))

        result = deconvolve(peaks, params)
        assert len(result) == 2

        # Sort by RT so we can check each component
        result.sort(key=lambda c: c.rt)
        comp_a, comp_b = result[0], result[1]

        # Each component's spectrum should contain at least the m/z from its analyte
        # (the NNLS will distribute contributions; we check that each component's
        # dominant m/z values come from the correct analyte group)
        assert any(mz in comp_a.spectrum and comp_a.spectrum[mz] > 0 for mz in mz_a)
        assert any(mz in comp_b.spectrum and comp_b.spectrum[mz] > 0 for mz in mz_b)
        # RTs should be distinct and near their respective analyte RTs
        assert comp_a.rt == pytest.approx(2.0, abs=0.1)
        assert comp_b.rt == pytest.approx(2.5, abs=0.1)
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestDeconvolve -v
```

Expected: 5 errors (ImportError)

- [ ] **Step 3: Implement `deconvolve()`**

Add after `_build_components`:

```python
# ─── Entry point ──────────────────────────────────────────────────────────────

def deconvolve(peaks: list,
               params: Optional[DeconvolutionParams] = None) -> list:
    """Spectral deconvolution via the ADAP-GC 3.2 two-step decomposition algorithm.

    Clusters EIC peaks by retention time, selects model peaks by shape and
    sharpness, then decomposes all peaks into linear combinations of model peaks
    (NNLS) to construct fragmentation spectra for each detected analyte.

    Reference: Smirnov et al., J. Proteome Res. 2018, 17, 470-478.
    Java source: dulab.adap.workflow.TwoStepDecomposition.run()

    Args:
        peaks: Detected EIC peaks (one per m/z per chromatographic feature).
        params: Algorithm parameters. Uses default DeconvolutionParams if None.

    Returns:
        List of DeconvolutedComponent, each with RT and fragmentation spectrum.
    """
    if params is None:
        params = DeconvolutionParams()
    if not peaks:
        return []

    rt_clusters = _cluster_by_rt(
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

    return result
```

- [ ] **Step 4: Run to verify pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestDeconvolve -v
```

Expected: 5 passed

- [ ] **Step 5: Run full test suite**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: implement deconvolve() entry point — ADAP-GC 3.2 pipeline complete"
```

---

## Final check

- [ ] **Run full suite one last time**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env pytest test_spectral_deconvolution.py -v --tb=short
```

Expected: all tests pass, no warnings about imports from logic/ui/api.

- [ ] **Verify module is importable standalone (no chromakit-ms dependency)**

```bash
conda run -n chromakit-env python -c "
import sys
sys.path.insert(0, 'deconvolution')
from spectral_deconvolution import deconvolve, DeconvolutionParams, EICPeak
print('Standalone import OK')
"
```

Expected: `Standalone import OK`
