# Spectral Deconvolution Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the already-implemented ADAP-GC 3.2 spectral deconvolution algorithm into ChromaKit-MS so each FID peak gets a deconvolved MS spectrum for library search, with a raw/deconvolved toggle in the MS tab.

**Architecture:** FID peaks are grouped into cluster windows using gap-merge logic, EIC traces are extracted per m/z from the raw MS data, `deconvolve()` runs per window, dominant components are matched back to FID peaks by RT proximity, and the resulting spectra replace point-in-time extraction in the batch search pipeline. A QRunnable worker handles threading; a new tab in the MS Options dialog exposes tunable parameters.

**Tech Stack:** Python, PySide6, numpy, scipy, scikit-learn, rainbow. All in `chromakit-env` conda environment.

**Spec:** `docs/superpowers/specs/2026-03-25-spectral-deconvolution-integration-design.md`

**Run all logic tests:** `conda run -n chromakit-env pytest tests/ -v`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `logic/spectral_deconvolution.py` | **Create** (copy) | ADAP-GC algorithm (copy of `deconvolution/spectral_deconvolution.py`) |
| `logic/eic_extractor.py` | **Create** | `extract_eic_peaks()` — raw MS data → `list[EICPeak]` |
| `logic/spectral_deconv_runner.py` | **Create** | `WindowGroupingParams`, window grouping, `run_spectral_deconvolution()` |
| `logic/spectral_deconv_worker.py` | **Create** | `SpectralDeconvWorker` QRunnable |
| `tests/test_eic_extractor.py` | **Create** | Unit tests for EIC extractor |
| `tests/test_spectral_deconv_runner.py` | **Create** | Unit tests for window grouping and component matching |
| `tests/test_batch_search_deconv.py` | **Create** | Unit tests for batch search spectrum selection |
| `logic/integration.py` | **Modify** | Add `deconvolved_spectrum` and `deconvolution_component_count` fields to `ChromatographicPeak` |
| `logic/batch_search.py` | **Modify** | Use `peak.deconvolved_spectrum` when available (line ~94) |
| `ui/frames/parameters.py` | **Modify** | Rename all `deconv_*` → `peak_splitting_*`; rename `'deconvolution'` mode string and params key |
| `ui/app.py` | **Modify** | Rename U-Net methods; dispatch `SpectralDeconvWorker` |
| `ui/frames/plot.py` | **Modify** | Rename `"Deconv"` legend label to `"Peak Split"` |
| `ui/frames/buttons.py` | **Modify** | Add `deconvolve_ms_clicked` signal + "Deconvolve MS" button |
| `ui/dialogs/ms_options_dialog.py` | **Modify** | Add "Spectral Deconvolution" tab; rename fallback tabs |
| `ui/frames/ms.py` | **Modify** | Add raw/deconvolved toggle; store current data directory path |

---

## Task 1: Rename U-Net peak splitting throughout the codebase

**Files:**
- Modify: `ui/frames/parameters.py`
- Modify: `ui/app.py`
- Modify: `ui/frames/plot.py`

This is a pure rename task. No new functionality. No tests — verify manually by running the app after.

- [ ] **Step 1: Rename in `parameters.py`**

Open `ui/frames/parameters.py`. Make all of the following changes (use your editor's find-and-replace):

| Find | Replace |
|------|---------|
| `'mode': 'classical',  # 'classical' or 'deconvolution'` | `'mode': 'classical',  # 'classical' or 'peak_splitting'` |
| `self.peak_mode_combo.addItem("Deconvolution (U-Net)", "deconvolution")` | `self.peak_mode_combo.addItem("Peak Splitting (U-Net)", "peak_splitting")` |
| Key `'deconvolution':` in `current_params` dict | Key `'peak_splitting':` |
| `self.deconv_controls_frame` | `self.peak_splitting_controls_frame` |
| `self.deconv_advanced_frame` | `self.peak_splitting_advanced_frame` |
| `self.deconv_advanced_toggle` | `self.peak_splitting_advanced_toggle` |
| `self.deconv_reset_btn` | `self.peak_splitting_reset_btn` |
| `self.deconv_windows_table` | `self.peak_splitting_windows_table` |
| `self._add_deconv_window_row` | `self._add_peak_splitting_window_row` |
| `self._remove_deconv_window_row` | `self._remove_peak_splitting_window_row` |
| `self._on_deconv_windows_changed` | `self._on_peak_splitting_windows_changed` |
| `self._on_deconv_param_changed` | `self._on_peak_splitting_param_changed` |
| `self._toggle_deconv_advanced` | `self._toggle_peak_splitting_advanced` |
| `self._update_deconv_method_visibility` | `self._update_peak_splitting_method_visibility` |
| `self._reset_deconv_defaults` | `self._reset_peak_splitting_defaults` |
| `self._sync_deconv_spinboxes` | `self._sync_peak_splitting_spinboxes` |
| `_on_pre_fit_signal_changed` | `_on_peak_splitting_pre_fit_signal_changed` |
| `.connect(self._on_pre_fit_signal_changed)` | `.connect(self._on_peak_splitting_pre_fit_signal_changed)` |
| `== 'deconvolution'` (mode comparisons) | `== 'peak_splitting'` |
| `current_params['deconvolution']` | `current_params['peak_splitting']` |
| `params.get('deconvolution'` | `params.get('peak_splitting'` |

Also update the `is_deconv` variable name where it appears (just a local, fine to leave or rename to `is_peak_splitting` for clarity).

- [ ] **Step 2: Rename in `app.py`**

Open `ui/app.py`. Make all of the following changes:

| Find | Replace |
|------|---------|
| `peak_mode == 'deconvolution'` | `peak_mode == 'peak_splitting'` |
| `def _apply_deconvolution(` | `def _apply_peak_splitting(` |
| `self._apply_deconvolution(` | `self._apply_peak_splitting(` |
| `def _integrate_deconvolution(` | `def _integrate_peak_splitting(` |
| `self._integrate_deconvolution(` | `self._integrate_peak_splitting(` |
| `self._deconv_cache` | `self._peak_splitting_cache` |
| `'deconvolution'` (in params.get calls for U-Net params) | `'peak_splitting'` |

- [ ] **Step 3: Rename in `plot.py`**

Open `ui/frames/plot.py`. Change the legend label:

```python
# Find:
label='Deconv' if i == 0 and 'Deconv' not in ...
# Replace:
label='Peak Split' if i == 0 and 'Peak Split' not in ...
```

- [ ] **Step 4: Verify app still launches**

```bash
conda run -n chromakit-env python main.py
```

Load a file, switch to "Peak Splitting (U-Net)" mode in the parameters panel, confirm no errors. Then switch back to classical. Close.

- [ ] **Step 5: Commit**

```bash
git add ui/frames/parameters.py ui/app.py ui/frames/plot.py
git commit -m "refactor: rename U-Net deconvolution → peak splitting throughout UI"
```

---

## Task 2: Copy `spectral_deconvolution.py` to `logic/`

**Files:**
- Create: `logic/spectral_deconvolution.py`

- [ ] **Step 1: Copy the file**

```bash
cp "deconvolution/spectral_deconvolution.py" "logic/spectral_deconvolution.py"
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
conda run -n chromakit-env python -c "from logic.spectral_deconvolution import deconvolve, EICPeak, DeconvolutedComponent, DeconvolutionParams; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add logic/spectral_deconvolution.py
git commit -m "feat: add spectral_deconvolution.py to logic/ for app integration"
```

---

## Task 3: Add new fields to `ChromatographicPeak`

**Files:**
- Modify: `logic/integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chromatic_peak_fields.py`:

```python
# tests/test_chromatic_peak_fields.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.integration import ChromatographicPeak


def test_deconvolved_spectrum_defaults_to_none():
    peak = ChromatographicPeak(
        compound_id='test', peak_number=1, retention_time=1.0,
        integrator='test', width=0.1, area=1000.0,
        start_time=0.95, end_time=1.05
    )
    assert peak.deconvolved_spectrum is None


def test_deconvolution_component_count_defaults_to_none():
    peak = ChromatographicPeak(
        compound_id='test', peak_number=1, retention_time=1.0,
        integrator='test', width=0.1, area=1000.0,
        start_time=0.95, end_time=1.05
    )
    assert peak.deconvolution_component_count is None
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
conda run -n chromakit-env pytest tests/test_chromatic_peak_fields.py -v
```

Expected: `AttributeError: 'ChromatographicPeak' object has no attribute 'deconvolved_spectrum'`

- [ ] **Step 3: Add the fields to `ChromatographicPeak.__init__`**

In `logic/integration.py`, find the `__init__` method of `ChromatographicPeak`. After the existing `self.is_grouped = False` / `self.grouped_peak_count = None` block, add:

```python
        # MS spectral deconvolution results (ADAP-GC)
        self.deconvolved_spectrum = None
        # When populated: {'mz': np.ndarray, 'intensities': np.ndarray}
        # Same format as SpectrumExtractor.extract_for_peak()

        self.deconvolution_component_count = None
        # int: ADAP-GC components found in this peak's window; None = not yet run
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n chromakit-env pytest tests/test_chromatic_peak_fields.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add logic/integration.py tests/test_chromatic_peak_fields.py
git commit -m "feat: add deconvolved_spectrum and deconvolution_component_count fields to ChromatographicPeak"
```

---

## Task 4: EIC Extractor

**Files:**
- Create: `logic/eic_extractor.py`
- Create: `tests/test_eic_extractor.py`

The extractor takes an open rainbow `data.ms` DataFile object and an RT window, and returns a `list[EICPeak]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eic_extractor.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock
from logic.spectral_deconvolution import EICPeak


def _make_mock_ms(n_scans=60, n_mz=200, rt_start=1.0, rt_end=2.0):
    """Mock rainbow DataFile object with .xlabels and .data."""
    ms = MagicMock()
    ms.xlabels = np.linspace(rt_start, rt_end, n_scans)
    ms.data = np.zeros((n_scans, n_mz), dtype=float)
    return ms


def _gaussian(n, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((np.arange(n) - center) / sigma) ** 2)


def test_returns_eic_peaks():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)  # peak at m/z 50, apex at scan 30

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)

    assert len(peaks) >= 1
    assert all(isinstance(p, EICPeak) for p in peaks)


def test_mz_is_one_based():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 0] = _gaussian(60, 30, 4, 1000.0)   # column 0 → m/z 1
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)  # column 49 → m/z 50

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)
    mzs = {p.mz for p in peaks}

    assert 1.0 in mzs
    assert 50.0 in mzs


def test_min_intensity_filter():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 10] = _gaussian(60, 30, 4, 50.0)   # max ~50, below threshold
    ms.data[:, 20] = _gaussian(60, 30, 4, 1000.0)  # max ~1000, above threshold

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=200.0)
    mzs = {p.mz for p in peaks}

    assert 11.0 not in mzs  # filtered out
    assert 21.0 in mzs      # kept


def test_boundary_indices_are_valid_integers():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms()
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)

    for p in peaks:
        n = len(p.rt_array)
        assert isinstance(p.left_boundary_idx, (int, np.integer))
        assert isinstance(p.right_boundary_idx, (int, np.integer))
        assert isinstance(p.apex_idx, (int, np.integer))
        assert 0 <= p.left_boundary_idx <= p.apex_idx <= p.right_boundary_idx < n


def test_rt_array_spans_full_window():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(rt_start=2.0, rt_end=3.0)
    ms.data[:, 49] = _gaussian(60, 30, 4, 1000.0)

    peaks = extract_eic_peaks(ms, t_start=2.0, t_end=3.0, min_intensity=100.0)

    for p in peaks:
        assert p.rt_array[0] >= 2.0 - 1e-9
        assert p.rt_array[-1] <= 3.0 + 1e-9


def test_empty_window_returns_empty_list():
    from logic.eic_extractor import extract_eic_peaks
    ms = _make_mock_ms(rt_start=1.0, rt_end=2.0)
    # No signal anywhere

    peaks = extract_eic_peaks(ms, t_start=1.0, t_end=2.0, min_intensity=100.0)

    assert peaks == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n chromakit-env pytest tests/test_eic_extractor.py -v
```

Expected: `ModuleNotFoundError: No module named 'logic.eic_extractor'`

- [ ] **Step 3: Implement `logic/eic_extractor.py`**

```python
"""EIC peak extractor for ADAP-GC spectral deconvolution.

Extracts per-m/z ion chromatogram peaks from a rainbow data.ms object
within a specified RT window, returning EICPeak objects for deconvolve().
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from logic.spectral_deconvolution import EICPeak


def extract_eic_peaks(
    ms,
    t_start: float,
    t_end: float,
    min_intensity: float = 200.0,
) -> list[EICPeak]:
    """Extract EIC peaks from a rainbow data.ms DataFile within [t_start, t_end].

    Args:
        ms: Open rainbow DataFile (.xlabels = RT axis, .data = [scans x mz] array).
        t_start: Window start in minutes (inclusive).
        t_end: Window end in minutes (inclusive).
        min_intensity: Skip m/z traces whose max intensity is below this value.

    Returns:
        List of EICPeak objects. rt_array and intensity_array span the full
        window so ADAP-GC has full chromatographic context. m/z values are
        1-based integers stored as float (column j → mz = float(j+1)).
    """
    xlabels = np.asarray(ms.xlabels, dtype=float)
    data = np.asarray(ms.data, dtype=float)

    # Slice to window
    mask = (xlabels >= t_start) & (xlabels <= t_end)
    if not np.any(mask):
        return []

    rt_window = xlabels[mask]
    ms_slice = data[mask, :]   # shape: [n_window_scans, n_mz]
    n_window = len(rt_window)

    eic_peaks: list[EICPeak] = []

    for j in range(ms_slice.shape[1]):
        trace = ms_slice[:, j]

        if trace.max() < min_intensity:
            continue

        apex_indices, _ = find_peaks(trace)
        if len(apex_indices) == 0:
            # Single scan maximum — treat the argmax as the apex
            apex_indices = np.array([int(np.argmax(trace))])

        # Get half-max widths for each apex
        try:
            widths_out = peak_widths(trace, apex_indices, rel_height=0.5)
            left_ips = widths_out[2]   # float array
            right_ips = widths_out[3]  # float array
        except Exception:
            continue

        mz = float(j + 1)

        for k, apex_idx in enumerate(apex_indices):
            left_idx = int(np.floor(left_ips[k]))
            right_idx = int(np.ceil(right_ips[k]))
            # Clamp to valid range
            left_idx = max(0, min(left_idx, n_window - 1))
            right_idx = max(0, min(right_idx, n_window - 1))
            apex_idx_int = int(apex_idx)

            eic_peaks.append(EICPeak(
                rt_apex=float(rt_window[apex_idx_int]),
                mz=mz,
                rt_array=rt_window.copy(),
                intensity_array=trace.copy(),
                left_boundary_idx=left_idx,
                right_boundary_idx=right_idx,
                apex_idx=apex_idx_int,
            ))

    return eic_peaks
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n chromakit-env pytest tests/test_eic_extractor.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add logic/eic_extractor.py tests/test_eic_extractor.py
git commit -m "feat: implement EIC extractor for ADAP-GC spectral deconvolution"
```

---

## Task 5: Window grouping and `WindowGroupingParams`

**Files:**
- Create: `logic/spectral_deconv_runner.py` (scaffold + window grouping only)
- Create: `tests/test_spectral_deconv_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spectral_deconv_runner.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock
from logic.spectral_deconv_runner import _group_peaks_into_windows, WindowGroupingParams


def _make_peak(rt, start, end):
    p = MagicMock()
    p.retention_time = rt
    p.start_time = start
    p.end_time = end
    return p


def test_single_peak_one_window():
    peaks = [_make_peak(1.0, 0.9, 1.1)]
    params = WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 1
    w_start, w_end, w_peaks = windows[0]
    assert len(w_peaks) == 1
    assert w_start < 1.0 < w_end  # padding applied


def test_overlapping_peaks_merge():
    # end_time of first > start_time of second → gap < 0
    peaks = [_make_peak(1.0, 0.9, 1.15), _make_peak(1.2, 1.1, 1.3)]
    params = WindowGroupingParams(gap_tolerance=0.05, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 1
    assert len(windows[0][2]) == 2


def test_far_peaks_separate_windows():
    peaks = [_make_peak(1.0, 0.9, 1.1), _make_peak(5.0, 4.9, 5.1)]
    params = WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 2


def test_window_clamped_to_rt_bounds():
    peaks = [_make_peak(0.1, 0.05, 0.15)]  # close to start
    params = WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    w_start, w_end, _ = windows[0]
    assert w_start >= 0.0


def test_auto_gap_tolerance_two_peaks():
    # Peaks of width 0.2 min → median width = 0.2 → gap_tolerance = 0.1
    peaks = [_make_peak(1.0, 0.9, 1.1), _make_peak(5.0, 4.9, 5.1)]
    params = WindowGroupingParams(gap_tolerance=None, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 2  # gap 3.8 >> auto tolerance ~0.1


def test_auto_gap_tolerance_fallback_single_peak():
    # Only 1 peak → fallback to 0.3 min
    peaks = [_make_peak(1.0, 0.9, 1.1)]
    params = WindowGroupingParams(gap_tolerance=None, padding_fraction=0.5)
    windows = _group_peaks_into_windows(peaks, params, rt_min=0.0, rt_max=10.0)
    assert len(windows) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n chromakit-env pytest tests/test_spectral_deconv_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'logic.spectral_deconv_runner'`

- [ ] **Step 3: Implement `logic/spectral_deconv_runner.py` (window grouping only)**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n chromakit-env pytest tests/test_spectral_deconv_runner.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add logic/spectral_deconv_runner.py tests/test_spectral_deconv_runner.py
git commit -m "feat: implement WindowGroupingParams and FID peak window grouping"
```

---

## Task 6: Full deconvolution runner + component matching

**Files:**
- Modify: `logic/spectral_deconv_runner.py` (add `run_spectral_deconvolution`)
- Modify: `tests/test_spectral_deconv_runner.py` (add matching tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spectral_deconv_runner.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch
from logic.spectral_deconvolution import DeconvolutionParams, DeconvolutedComponent
from logic.spectral_deconv_runner import run_spectral_deconvolution, WindowGroupingParams


def _make_chromatographic_peak(rt, start, end):
    """Mock ChromatographicPeak with required attributes."""
    p = MagicMock()
    p.retention_time = rt
    p.start_time = start
    p.end_time = end
    p.deconvolved_spectrum = None
    p.deconvolution_component_count = None
    return p


def _make_component(rt, spectrum=None):
    c = MagicMock(spec=DeconvolutedComponent)
    c.rt = rt
    c.spectrum = spectrum or {50.0: 1000.0, 73.0: 500.0}
    return c


def test_nearest_component_assigned():
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)
    components = [_make_component(1.01), _make_component(1.5)]
    grouping = WindowGroupingParams(rt_match_tolerance=0.05)

    # call the internal matching function directly
    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak], components, grouping.rt_match_tolerance)

    assert peak.deconvolved_spectrum is not None
    assert peak.deconvolution_component_count == 2


def test_no_match_beyond_tolerance():
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)
    components = [_make_component(1.2)]  # 0.2 min away > 0.05 tolerance
    grouping = WindowGroupingParams(rt_match_tolerance=0.05)

    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak], components, grouping.rt_match_tolerance)

    assert peak.deconvolved_spectrum is None
    assert peak.deconvolution_component_count == 1  # ran but no match


def test_one_to_one_assignment():
    # Two peaks, two components — each should get its nearest
    peak_a = _make_chromatographic_peak(1.0, 0.9, 1.1)
    peak_b = _make_chromatographic_peak(1.3, 1.2, 1.4)
    comp_a = _make_component(1.01, spectrum={50.0: 1000.0})
    comp_b = _make_component(1.31, spectrum={73.0: 2000.0})

    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak_a, peak_b], [comp_a, comp_b], rt_match_tolerance=0.05)

    assert peak_a.deconvolved_spectrum is not None
    assert peak_b.deconvolved_spectrum is not None
    # Each gets a distinct spectrum
    assert set(peak_a.deconvolved_spectrum['mz']) != set(peak_b.deconvolved_spectrum['mz'])


def test_deconvolved_spectrum_format():
    """Deconvolved spectrum must be {'mz': np.ndarray, 'intensities': np.ndarray}."""
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)
    comp = _make_component(1.01, spectrum={50.0: 1000.0, 73.0: 500.0})

    from logic.spectral_deconv_runner import _assign_components_to_peaks
    _assign_components_to_peaks([peak], [comp], rt_match_tolerance=0.05)

    spec = peak.deconvolved_spectrum
    assert 'mz' in spec and 'intensities' in spec
    assert isinstance(spec['mz'], np.ndarray)
    assert isinstance(spec['intensities'], np.ndarray)
    assert len(spec['mz']) == len(spec['intensities'])
    assert list(spec['mz']) == sorted(spec['mz'])  # sorted by m/z


def test_run_spectral_deconvolution_empty_components_sets_count_zero():
    """When deconvolve() returns empty, component_count must be 0 (not None)."""
    peak = _make_chromatographic_peak(1.0, 0.9, 1.1)

    fake_ms = MagicMock()
    fake_ms.xlabels = np.linspace(0.0, 5.0, 100)
    fake_ms.data = np.zeros((100, 200))
    # Add a non-zero trace so EIC extraction doesn't short-circuit
    fake_ms.data[40:60, 49] = 500.0

    fake_data_dir = MagicMock()
    fake_data_dir.get_file.return_value = fake_ms

    with patch('logic.spectral_deconv_runner.rb.read', return_value=fake_data_dir), \
         patch('logic.spectral_deconv_runner.deconvolve', return_value=[]):
        run_spectral_deconvolution(
            [peak], ms_data_path='/fake/path.D',
            grouping_params=WindowGroupingParams(gap_tolerance=0.3)
        )

    assert peak.deconvolution_component_count == 0
    assert peak.deconvolved_spectrum is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n chromakit-env pytest tests/test_spectral_deconv_runner.py::test_nearest_component_assigned -v
```

Expected: `ImportError: cannot import name '_assign_components_to_peaks'`

- [ ] **Step 3: Add `_assign_components_to_peaks` and `run_spectral_deconvolution` to the runner**

Append to `logic/spectral_deconv_runner.py`:

```python
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
        ms_data_path: Path to the Agilent .D directory (DataHandler.current_directory_path).
        deconv_params: ADAP-GC parameters. Uses defaults if None.
        grouping_params: Window grouping parameters. Uses defaults if None.
        progress_callback: Optional callable(int) → None receiving 0–100 progress.
        should_cancel: Optional callable() → bool; if True, abort early.

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
```

- [ ] **Step 4: Run all runner tests**

```bash
conda run -n chromakit-env pytest tests/test_spectral_deconv_runner.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add logic/spectral_deconv_runner.py tests/test_spectral_deconv_runner.py
git commit -m "feat: implement run_spectral_deconvolution and component-to-peak matching"
```

---

## Task 7: Batch search integration

**Files:**
- Modify: `logic/batch_search.py`
- Create: `tests/test_batch_search_deconv.py`

The change is at line ~94 in `batch_search.py` where `extract_for_peak` is called. When `peak.deconvolved_spectrum` is set, use it directly; otherwise fall back.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_batch_search_deconv.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_peak(with_deconvolved=False):
    peak = MagicMock()
    peak.retention_time = 1.0
    peak.peak_number = 1
    peak.manual_assignment = False
    peak.is_saturated = False
    if with_deconvolved:
        peak.deconvolved_spectrum = {
            'mz': np.array([50.0, 73.0]),
            'intensities': np.array([1000.0, 500.0]),
        }
    else:
        peak.deconvolved_spectrum = None
    return peak


def test_uses_deconvolved_spectrum_when_available():
    """When peak.deconvolved_spectrum is set, extract_for_peak must NOT be called."""
    from logic.batch_search import BatchSearchWorker

    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = [('Compound A', 0.95)]

    peak = _make_peak(with_deconvolved=True)
    worker = BatchSearchWorker(ms_toolkit, [peak], '/fake/path.D', options={'search_method': 'vector', 'top_n': 5})

    with patch.object(worker.spectrum_extractor, 'extract_for_peak') as mock_extract:
        worker.run()
        mock_extract.assert_not_called()


def test_falls_back_to_extraction_when_no_deconvolved_spectrum():
    """When peak.deconvolved_spectrum is None, extract_for_peak must be called."""
    from logic.batch_search import BatchSearchWorker

    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = [('Compound B', 0.90)]

    peak = _make_peak(with_deconvolved=False)
    worker = BatchSearchWorker(ms_toolkit, [peak], '/fake/path.D', options={'search_method': 'vector', 'top_n': 5})

    fake_spectrum = {'mz': np.array([50.0]), 'intensities': np.array([1000.0])}
    with patch.object(worker.spectrum_extractor, 'extract_for_peak', return_value=fake_spectrum):
        worker.run()
        worker.spectrum_extractor.extract_for_peak.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n chromakit-env pytest tests/test_batch_search_deconv.py -v
```

Expected: `test_uses_deconvolved_spectrum_when_available` FAILS because `extract_for_peak` is still called.

- [ ] **Step 3: Modify `batch_search.py` lines ~91–103**

Find the spectrum extraction block in `BatchSearchWorker.run()` and replace it:

```python
                # Extract spectrum: use deconvolved if available, otherwise point-in-time
                if hasattr(peak, 'deconvolved_spectrum') and peak.deconvolved_spectrum is not None:
                    spectrum = peak.deconvolved_spectrum
                    # Deconvolved spectrum is already clean — skip background subtraction
                else:
                    spectrum = self.spectrum_extractor.extract_for_peak(
                        self.data_directory,
                        peak,
                        self.options
                    )
```

Keep the existing `if not spectrum or 'mz' not in spectrum or 'intensities' not in spectrum:` check immediately after — it will still work correctly for both branches.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n chromakit-env pytest tests/test_batch_search_deconv.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add logic/batch_search.py tests/test_batch_search_deconv.py
git commit -m "feat: use deconvolved spectrum in batch search when available"
```

---

## Task 8: Worker

**Files:**
- Create: `logic/spectral_deconv_worker.py`

No unit tests for Qt threading — verify via manual test in Task 9.

- [ ] **Step 1: Create `logic/spectral_deconv_worker.py`**

```python
"""QRunnable worker for spectral deconvolution.

Wraps run_spectral_deconvolution() for use on QThreadPool.
Follows the same pattern as BatchSearchWorker.
"""
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from logic.spectral_deconvolution import DeconvolutionParams
from logic.spectral_deconv_runner import run_spectral_deconvolution, WindowGroupingParams


class SpectralDeconvWorkerSignals(QObject):
    progress = Signal(int)    # 0–100 percentage
    finished = Signal()       # peaks mutated in place; caller accesses them directly
    error = Signal(str)


class SpectralDeconvWorker(QRunnable):
    """Run ADAP-GC spectral deconvolution on a background thread."""

    def __init__(
        self,
        peaks: list,
        ms_data_path: str,
        deconv_params: DeconvolutionParams | None = None,
        grouping_params: WindowGroupingParams | None = None,
    ):
        super().__init__()
        self.peaks = peaks
        self.ms_data_path = ms_data_path
        self.deconv_params = deconv_params or DeconvolutionParams()
        self.grouping_params = grouping_params or WindowGroupingParams()
        self.signals = SpectralDeconvWorkerSignals()
        self.cancelled = False

    @Slot()
    def run(self):
        try:
            run_spectral_deconvolution(
                peaks=self.peaks,
                ms_data_path=self.ms_data_path,
                deconv_params=self.deconv_params,
                grouping_params=self.grouping_params,
                progress_callback=self.signals.progress.emit,
                should_cancel=lambda: self.cancelled,
            )
            self.signals.finished.emit()

        except Exception as e:
            import traceback
            self.signals.error.emit(traceback.format_exc())
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
conda run -n chromakit-env python -c "from logic.spectral_deconv_worker import SpectralDeconvWorker; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add logic/spectral_deconv_worker.py
git commit -m "feat: implement SpectralDeconvWorker QRunnable"
```

---

## Task 9: "Deconvolve MS" button and app wiring

**Files:**
- Modify: `ui/frames/buttons.py`
- Modify: `ui/app.py`

- [ ] **Step 1: Add signal and button to `ButtonFrame`**

In `ui/frames/buttons.py`:

1. Add signal after `batch_search_clicked`:
```python
    deconvolve_ms_clicked = Signal()
```

2. In `__init__`, after the `batch_search_button` block (look for where `self.batch_search_button` is created and added), add:
```python
        self.deconvolve_ms_button = QPushButton("Deconvolve MS")
        self.deconvolve_ms_button.setToolTip("Run ADAP-GC spectral deconvolution on integrated peaks")
        self.deconvolve_ms_button.setEnabled(False)
        self.deconvolve_ms_button.clicked.connect(self.deconvolve_ms_clicked)
        self.row2_layout.addWidget(self.deconvolve_ms_button)
```

3. Find the method that enables the `batch_search_button` (called when integration completes — search for `batch_search_button.setEnabled`). Enable the new button in the same place:
```python
        self.deconvolve_ms_button.setEnabled(True)
```

- [ ] **Step 2: Wire the button in `app.py`**

In `ui/app.py`:

1. Connect the signal in the section where button signals are connected (search for `batch_search_clicked`):
```python
        self.button_frame.deconvolve_ms_clicked.connect(self._on_deconvolve_ms_clicked)
```

2. Add the handler method near `_on_batch_search_clicked`:
```python
    def _on_deconvolve_ms_clicked(self):
        """Dispatch spectral deconvolution worker."""
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            return
        if not self.data_handler.current_directory_path:
            return

        from logic.spectral_deconv_worker import SpectralDeconvWorker

        # Read params from MS options dialog settings
        deconv_params, grouping_params = self._get_spectral_deconv_params()

        worker = SpectralDeconvWorker(
            peaks=self.integrated_peaks,
            ms_data_path=self.data_handler.current_directory_path,
            deconv_params=deconv_params,
            grouping_params=grouping_params,
        )
        worker.signals.progress.connect(self._on_deconvolve_ms_progress)
        worker.signals.finished.connect(self._on_deconvolve_ms_finished)
        worker.signals.error.connect(self._on_deconvolve_ms_error)

        print(f"Running MS spectral deconvolution on {len(self.integrated_peaks)} peaks...")
        self.thread_pool.start(worker)

    def _get_spectral_deconv_params(self):
        """Read DeconvolutionParams and WindowGroupingParams from QSettings."""
        from PySide6.QtCore import QSettings
        from logic.spectral_deconvolution import DeconvolutionParams
        from logic.spectral_deconv_runner import WindowGroupingParams

        s = QSettings("CalebCoatney", "ChromaKit")

        deconv = DeconvolutionParams(
            min_cluster_distance=s.value("ms_spectral_deconv/min_cluster_distance", 0.005, float),
            min_cluster_size=s.value("ms_spectral_deconv/min_cluster_size", 2, int),
            min_cluster_intensity=s.value("ms_spectral_deconv/min_cluster_intensity", 200.0, float),
            shape_sim_threshold=s.value("ms_spectral_deconv/shape_sim_threshold", 30.0, float),
            model_peak_choice=s.value("ms_spectral_deconv/model_peak_choice", "sharpness", str),
            excluded_mz=[
                float(x.strip())
                for x in s.value("ms_spectral_deconv/excluded_mz", "", str).split(",")
                if x.strip()
            ],
        )

        raw_gap = s.value("ms_spectral_deconv/gap_tolerance", 0.0, float)
        grouping = WindowGroupingParams(
            gap_tolerance=None if raw_gap == 0.0 else raw_gap,
            padding_fraction=s.value("ms_spectral_deconv/padding_fraction", 0.5, float),
            rt_match_tolerance=s.value("ms_spectral_deconv/rt_match_tolerance", 0.05, float),
        )

        return deconv, grouping

    def _on_deconvolve_ms_progress(self, pct: int):
        # Update progress bar if available; print as fallback
        print(f"Deconvolution progress: {pct}%")

    def _on_deconvolve_ms_finished(self):
        print("MS spectral deconvolution complete.")
        # Refresh the MS tab to pick up deconvolved spectra
        self._refresh_current_peak_ms_display()

    def _on_deconvolve_ms_error(self, msg: str):
        print(f"Spectral deconvolution error:\n{msg}")

    def _refresh_current_peak_ms_display(self):
        """Re-display spectrum for the last-viewed peak (picks up deconvolved spectrum)."""
        if hasattr(self, '_last_ms_peak_index') and self._last_ms_peak_index is not None:
            self.on_peak_spectrum_requested(self._last_ms_peak_index)
```

> **Note on `_last_ms_peak_index` tracking:** At the top of the existing `on_peak_spectrum_requested` method, add `self._last_ms_peak_index = peak_index` so the last-viewed peak index is always known. This lets `_refresh_current_peak_ms_display` re-trigger the display after deconvolution completes. Add `self._last_ms_peak_index = None` to `ChromaKitApp.__init__` for initialization.

- [ ] **Step 3: Manual test**

```bash
conda run -n chromakit-env python main.py
```

1. Load a `.D` file with GC-MS data
2. Run integration — confirm "Deconvolve MS" button becomes enabled
3. Click "Deconvolve MS" — confirm no crash, progress prints in console
4. Select a peak in the RT table — confirm MS tab still shows a spectrum

- [ ] **Step 4: Commit**

```bash
git add ui/frames/buttons.py ui/app.py
git commit -m "feat: add Deconvolve MS button and app wiring for spectral deconvolution"
```

---

## Task 10: MS Options dialog — Spectral Deconvolution tab

**Files:**
- Modify: `ui/dialogs/ms_options_dialog.py`

- [ ] **Step 1: Rename fallback tabs and add new tab**

In `ui/dialogs/ms_options_dialog.py`:

1. In `__init__`, change the tab order/calls:
```python
        self._create_general_tab()
        self._create_extraction_tab()       # was already here
        self._create_subtraction_tab()      # was already here
        self._create_spectral_deconv_tab()  # NEW — add this call
        self._create_algorithm_tab()
        self._create_quality_checks_tab()
```

2. In `_create_extraction_tab`, change the tab widget add call:
```python
        self.tab_widget.addTab(tab, "Spectrum Extraction (Fallback)")
```

3. In `_create_subtraction_tab`, change the tab widget add call:
```python
        self.tab_widget.addTab(tab, "Background Subtraction (Fallback)")
```

4. Add the new method `_create_spectral_deconv_tab`:

```python
    def _create_spectral_deconv_tab(self):
        """Create the Spectral Deconvolution (ADAP-GC) options tab."""
        from PySide6.QtWidgets import QGroupBox, QFormLayout

        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Deconvolved spectra replace point-in-time extraction for library search.\n"
            "Run via 'Deconvolve MS' button after FID integration."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- Peak Window Grouping ---
        grouping_group = QGroupBox("Peak Window Grouping")
        grouping_layout = QFormLayout(grouping_group)

        self.gap_tolerance_spin = QDoubleSpinBox()
        self.gap_tolerance_spin.setRange(0.0, 5.0)
        self.gap_tolerance_spin.setSingleStep(0.05)
        self.gap_tolerance_spin.setDecimals(3)
        self.gap_tolerance_spin.setSpecialValueText("Auto")
        self.gap_tolerance_spin.setToolTip("0 = auto (0.5× median FID peak width)")
        grouping_layout.addRow("Gap tolerance (min):", self.gap_tolerance_spin)

        self.padding_fraction_spin = QDoubleSpinBox()
        self.padding_fraction_spin.setRange(0.0, 2.0)
        self.padding_fraction_spin.setSingleStep(0.1)
        self.padding_fraction_spin.setDecimals(2)
        grouping_layout.addRow("Padding fraction:", self.padding_fraction_spin)

        self.rt_match_tolerance_spin = QDoubleSpinBox()
        self.rt_match_tolerance_spin.setRange(0.001, 1.0)
        self.rt_match_tolerance_spin.setSingleStep(0.005)
        self.rt_match_tolerance_spin.setDecimals(3)
        self.rt_match_tolerance_spin.setToolTip("Max RT gap (min) between FID peak and MS component")
        grouping_layout.addRow("RT match tolerance (min):", self.rt_match_tolerance_spin)

        layout.addWidget(grouping_group)

        # --- ADAP-GC Parameters ---
        adap_group = QGroupBox("ADAP-GC Parameters")
        adap_layout = QFormLayout(adap_group)

        self.min_cluster_distance_spin = QDoubleSpinBox()
        self.min_cluster_distance_spin.setRange(0.0001, 1.0)
        self.min_cluster_distance_spin.setSingleStep(0.001)
        self.min_cluster_distance_spin.setDecimals(4)
        self.min_cluster_distance_spin.setToolTip("DBSCAN eps — max RT gap within a cluster (min)")
        adap_layout.addRow("Min cluster distance (min):", self.min_cluster_distance_spin)

        self.min_cluster_size_spin = QSpinBox()
        self.min_cluster_size_spin.setRange(1, 20)
        self.min_cluster_size_spin.setToolTip("DBSCAN min_samples — minimum EIC peaks per cluster")
        adap_layout.addRow("Min cluster size:", self.min_cluster_size_spin)

        self.min_cluster_intensity_spin = QDoubleSpinBox()
        self.min_cluster_intensity_spin.setRange(0.0, 1e8)
        self.min_cluster_intensity_spin.setSingleStep(100.0)
        self.min_cluster_intensity_spin.setDecimals(0)
        self.min_cluster_intensity_spin.setToolTip("Drop clusters below this max intensity (counts)")
        adap_layout.addRow("Min cluster intensity:", self.min_cluster_intensity_spin)

        self.shape_sim_threshold_spin = QDoubleSpinBox()
        self.shape_sim_threshold_spin.setRange(1.0, 90.0)
        self.shape_sim_threshold_spin.setSingleStep(1.0)
        self.shape_sim_threshold_spin.setDecimals(1)
        self.shape_sim_threshold_spin.setToolTip("Max angle (°) between EIC shapes in the same cluster")
        adap_layout.addRow("Shape similarity threshold (°):", self.shape_sim_threshold_spin)

        self.model_peak_choice_combo = QComboBox()
        self.model_peak_choice_combo.addItems(["Sharpness", "Intensity", "m/z"])
        self.model_peak_choice_combo.setToolTip("Criterion for selecting the representative EIC peak")
        adap_layout.addRow("Model peak choice:", self.model_peak_choice_combo)

        self.excluded_mz_edit = QLineEdit()
        self.excluded_mz_edit.setPlaceholderText("e.g. 73, 147, 221  (leave blank for none)")
        self.excluded_mz_edit.setToolTip("Comma-separated m/z values to exclude (e.g. TMS artifacts)")
        adap_layout.addRow("Excluded m/z:", self.excluded_mz_edit)

        layout.addWidget(adap_group)
        layout.addStretch()

        self.tab_widget.addTab(tab, "Spectral Deconvolution")
```

5. In `_load_settings`, add at the end:
```python
        # Spectral Deconvolution tab
        self.gap_tolerance_spin.setValue(
            self.settings.value("ms_spectral_deconv/gap_tolerance", 0.0, float))
        self.padding_fraction_spin.setValue(
            self.settings.value("ms_spectral_deconv/padding_fraction", 0.5, float))
        self.rt_match_tolerance_spin.setValue(
            self.settings.value("ms_spectral_deconv/rt_match_tolerance", 0.05, float))
        self.min_cluster_distance_spin.setValue(
            self.settings.value("ms_spectral_deconv/min_cluster_distance", 0.005, float))
        self.min_cluster_size_spin.setValue(
            self.settings.value("ms_spectral_deconv/min_cluster_size", 2, int))
        self.min_cluster_intensity_spin.setValue(
            self.settings.value("ms_spectral_deconv/min_cluster_intensity", 200.0, float))
        self.shape_sim_threshold_spin.setValue(
            self.settings.value("ms_spectral_deconv/shape_sim_threshold", 30.0, float))
        choice_map = {"sharpness": 0, "intensity": 1, "mz": 2}
        self.model_peak_choice_combo.setCurrentIndex(
            choice_map.get(
                self.settings.value("ms_spectral_deconv/model_peak_choice", "sharpness", str), 0))
        self.excluded_mz_edit.setText(
            self.settings.value("ms_spectral_deconv/excluded_mz", "", str))
```

6. In `_save_settings`, add at the end:
```python
        # Spectral Deconvolution tab
        self.settings.setValue("ms_spectral_deconv/gap_tolerance",
                               self.gap_tolerance_spin.value())
        self.settings.setValue("ms_spectral_deconv/padding_fraction",
                               self.padding_fraction_spin.value())
        self.settings.setValue("ms_spectral_deconv/rt_match_tolerance",
                               self.rt_match_tolerance_spin.value())
        self.settings.setValue("ms_spectral_deconv/min_cluster_distance",
                               self.min_cluster_distance_spin.value())
        self.settings.setValue("ms_spectral_deconv/min_cluster_size",
                               self.min_cluster_size_spin.value())
        self.settings.setValue("ms_spectral_deconv/min_cluster_intensity",
                               self.min_cluster_intensity_spin.value())
        self.settings.setValue("ms_spectral_deconv/shape_sim_threshold",
                               self.shape_sim_threshold_spin.value())
        choice_list = ["sharpness", "intensity", "mz"]
        self.settings.setValue("ms_spectral_deconv/model_peak_choice",
                               choice_list[self.model_peak_choice_combo.currentIndex()])
        self.settings.setValue("ms_spectral_deconv/excluded_mz",
                               self.excluded_mz_edit.text().strip())
```

7. In `_restore_defaults`, add at the end:
```python
        # Spectral Deconvolution tab
        self.gap_tolerance_spin.setValue(0.0)
        self.padding_fraction_spin.setValue(0.5)
        self.rt_match_tolerance_spin.setValue(0.05)
        self.min_cluster_distance_spin.setValue(0.005)
        self.min_cluster_size_spin.setValue(2)
        self.min_cluster_intensity_spin.setValue(200.0)
        self.shape_sim_threshold_spin.setValue(30.0)
        self.model_peak_choice_combo.setCurrentIndex(0)
        self.excluded_mz_edit.clear()
```

- [ ] **Step 2: Manual test**

```bash
conda run -n chromakit-env python main.py
```

Open "MS Search Options" → confirm new "Spectral Deconvolution" tab appears with all controls. Change a value → OK → reopen dialog → confirm value persisted. Click "Restore Defaults" → confirm values reset.

- [ ] **Step 3: Commit**

```bash
git add ui/dialogs/ms_options_dialog.py
git commit -m "feat: add Spectral Deconvolution tab to MS Options dialog"
```

---

## Task 11: Raw/deconvolved toggle in MS frame

**Files:**
- Modify: `ui/frames/ms.py`

- [ ] **Step 1: Store current data directory in `MSFrame`**

In `ui/frames/ms.py`, add `self._current_data_path = None` in `__init__`. Add a method:
```python
    def set_data_path(self, path: str):
        """Called by ChromaKitApp when a file is loaded."""
        self._current_data_path = path
```

In `ui/app.py`, find where data is loaded and the MS frame is updated. Connect the data path:
```python
        self.ms_frame.set_data_path(self.data_handler.current_directory_path)
```

(Search for where `ms_frame` is updated after a file load — likely in `_on_data_loaded` or equivalent.)

- [ ] **Step 2: Add the toggle button**

In `ui/frames/ms.py`, locate `_create_ms_tools` (or wherever the spectrum display controls are built). Add a toggle button:

```python
        self.spectrum_toggle_btn = QPushButton("Deconvolved")
        self.spectrum_toggle_btn.setCheckable(True)
        self.spectrum_toggle_btn.setChecked(True)  # default: show deconvolved
        self.spectrum_toggle_btn.setVisible(False)  # hidden until deconvolved spectrum available
        self.spectrum_toggle_btn.setFixedWidth(120)
        self.spectrum_toggle_btn.clicked.connect(self._on_spectrum_toggle)
```

Add to the layout near the spectrum plot area (alongside whatever existing label/button row is near the spectrum plot).

- [ ] **Step 3: Update peak selection to show/hide toggle and use correct spectrum**

Find the method that fires when a peak is selected and a spectrum is displayed (likely something like `display_peak_spectrum`, `_on_peak_selected`, or where `extract_for_peak` is called from the MS frame).

Replace/wrap the spectrum display logic:

```python
    def display_peak_spectrum(self, peak, data_directory=None):
        """Display spectrum for a peak. Uses deconvolved if available."""
        self._current_peak = peak
        if data_directory:
            self._current_data_path = data_directory

        has_deconvolved = (
            hasattr(peak, 'deconvolved_spectrum')
            and peak.deconvolved_spectrum is not None
        )
        self.spectrum_toggle_btn.setVisible(has_deconvolved)

        if has_deconvolved and self.spectrum_toggle_btn.isChecked():
            spectrum = peak.deconvolved_spectrum
        else:
            # Point-in-time extraction at TIC apex
            if self._current_data_path:
                from logic.spectrum_extractor import SpectrumExtractor
                extractor = SpectrumExtractor()
                spectrum = extractor.extract_at_rt(
                    self._current_data_path,
                    retention_time=peak.retention_time,
                )
            else:
                return

        self._plot_spectrum(spectrum)

    def _on_spectrum_toggle(self):
        """Re-display current peak with toggled spectrum source, updating button label."""
        checked = self.spectrum_toggle_btn.isChecked()
        self.spectrum_toggle_btn.setText("Deconvolved" if checked else "Raw Apex")
        if hasattr(self, '_current_peak') and self._current_peak is not None:
            self.display_peak_spectrum(self._current_peak)
```

> **Note:** The exact method names and call sites depend on the current `ms.py` implementation. Read the file to find where `extract_for_peak` / `extract_at_rt` is called from the MS frame. Adapt the above to fit the existing structure — the principle is: intercept before the extraction call, check `peak.deconvolved_spectrum`, show/hide the toggle button accordingly.

- [ ] **Step 4: Manual test**

```bash
conda run -n chromakit-env python main.py
```

1. Load a GC-MS file, integrate, click "Deconvolve MS"
2. Select a peak — confirm MS tab shows spectrum (no crash)
3. If deconvolution produced a result for this peak: confirm toggle button appears
4. Click toggle — confirm spectrum changes between deconvolved and raw apex views
5. For a peak with no deconvolved spectrum: confirm toggle button is hidden

- [ ] **Step 5: Commit**

```bash
git add ui/frames/ms.py ui/app.py
git commit -m "feat: add raw/deconvolved spectrum toggle to MS tab"
```

---

## Task 12: End-to-end test on real data

No code changes. Validate the full pipeline on actual GC-MS data.

- [ ] **Step 1: Run on a real `.D` file**

```bash
conda run -n chromakit-env python main.py
```

1. Load a `.D` file with known GC-MS data
2. Run integration
3. Click "Deconvolve MS"
4. Wait for completion — check console for errors
5. Open MS Search Options → Spectral Deconvolution tab — note the default params
6. Run "MS Search All" — confirm library matches are produced
7. Compare match scores to a previous run without deconvolution (qualitative check)

- [ ] **Step 2: Inspect a co-eluting peak region**

1. Find a region in the chromatogram where two FID peaks are close together (gap < 0.1 min)
2. Select each peak → confirm both have deconvolved spectra (toggle button visible)
3. Toggle between raw and deconvolved — visually confirm deconvolved looks cleaner

- [ ] **Step 3: Commit any parameter tweaks discovered during testing**

If default `DeconvolutionParams` or `WindowGroupingParams` values need adjustment based on real data, update the defaults in `logic/spectral_deconv_runner.py` and `ui/dialogs/ms_options_dialog.py`.

```bash
git add -p
git commit -m "tune: adjust spectral deconvolution default parameters based on real data testing"
```
