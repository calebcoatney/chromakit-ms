# Spectral Deconvolution Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two Phase A integration bugs (missing toggle, no progress bar) and implement the Phase C chunk inspector dialog with DBSCAN scatter + EIC trace plots, per-window parameter tuning, and Apply-to-All rerun flow.

**Architecture:** The algorithm layer gains a `return_intermediates` mode so `deconvolve()` can expose RT clusters, noise peaks, and model peaks for visualization. A new non-modal `SpectralDeconvInspectorDialog` runs per-window EIC extraction + deconvolution on demand in a background thread, rendering a 2-row matplotlib canvas (DBSCAN scatter top, EIC traces bottom). The existing `on_peak_spectrum_requested` flow is routed through `display_peak_spectrum()` to activate the already-written toggle button.

**Tech Stack:** PySide6, matplotlib (embedded FigureCanvas), numpy, scipy, scikit-learn, rainbow. All in `chromakit-env`.

**Spec:** `docs/superpowers/specs/2026-03-26-spectral-deconv-inspector-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `logic/spectral_deconvolution.py` | Modify | `_cluster_by_rt` returns `(clusters, noise_peaks)` tuple; `deconvolve()` gains `return_intermediates` param |
| `deconvolution/spectral_deconvolution.py` | Modify | Same change — keep in sync with `logic/` copy |
| `deconvolution/test_spectral_deconvolution.py` | Modify | Add tests for tuple return and `return_intermediates` |
| `ui/frames/ms.py` | Modify | `display_peak_spectrum()` calls `set_current_spectrum()` on success; add "Inspect" button + `inspect_requested` signal |
| `ui/app.py` | Modify | Route `on_peak_spectrum_requested` through `display_peak_spectrum()`; add `QProgressDialog` to deconvolution flow; wire inspector |
| `ui/dialogs/spectral_deconv_inspector.py` | Create | `SpectralDeconvInspectorDialog` — layout, preview worker, plot rendering |

---

## Task 1: Algorithm — `_cluster_by_rt` tuple return + `deconvolve()` intermediates

**Files:**
- Modify: `logic/spectral_deconvolution.py`
- Modify: `deconvolution/spectral_deconvolution.py`
- Modify: `deconvolution/test_spectral_deconvolution.py`

> Tests run from `deconvolution/` directory (the test file imports `from spectral_deconvolution import ...` directly). Always `cd deconvolution` before running pytest.

- [ ] **Step 1: Write failing tests for `_cluster_by_rt` tuple return**

Add to `TestClusterByRT` in `deconvolution/test_spectral_deconvolution.py`:

```python
def test_returns_tuple_of_clusters_and_noise(self):
    # One isolated peak (noise) + two clustered peaks
    noise = make_gaussian_peak(rt_center=10.0, height=500.0, mz=99.0)  # alone → noise
    clustered = [
        make_gaussian_peak(rt_center=1.0, height=500.0, mz=float(i))
        for i in range(100, 103)
    ]
    clusters, noise_peaks = _cluster_by_rt(
        clustered + [noise], eps=0.01, min_samples=2, min_intensity=100.0
    )
    assert len(clusters) == 1
    assert len(noise_peaks) == 1
    assert noise_peaks[0].mz == 99.0

def test_no_noise_returns_empty_noise_list(self):
    peaks = [
        make_gaussian_peak(rt_center=1.0, height=500.0, mz=float(i))
        for i in range(100, 103)
    ]
    clusters, noise_peaks = _cluster_by_rt(peaks, eps=0.01, min_samples=2, min_intensity=100.0)
    assert len(clusters) == 1
    assert noise_peaks == []

def test_empty_input_returns_empty_tuple(self):
    clusters, noise_peaks = _cluster_by_rt([], eps=0.01, min_samples=2, min_intensity=100.0)
    assert clusters == []
    assert noise_peaks == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestClusterByRT -v
```

Expected: `ValueError: not enough values to unpack` — `_cluster_by_rt` currently returns a plain list, so tuple unpacking fails with a `ValueError`, not a `TypeError`.

- [ ] **Step 3: Modify `_cluster_by_rt` in `logic/spectral_deconvolution.py` to return `(clusters, noise_peaks)`**

In `logic/spectral_deconvolution.py`, replace the `_cluster_by_rt` function body. The key change: collect label=-1 peaks into a separate list and return both:

```python
def _cluster_by_rt(peaks: list, eps: float, min_samples: int,
                   min_intensity: float) -> tuple[list, list]:
    """Cluster EIC peaks by apex RT using 1D DBSCAN.
    ...
    Returns (clusters, noise_peaks) where noise_peaks are DBSCAN label=-1 points.
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
```

- [ ] **Step 4: Update the `_cluster_by_rt` call site in `deconvolve()` in the same file**

In `deconvolve()`, the call at `rt_clusters = _cluster_by_rt(...)` (currently line ~485) becomes:

```python
rt_clusters, noise_peaks = _cluster_by_rt(
    peaks, params.min_cluster_distance,
    params.min_cluster_size, params.min_cluster_intensity,
)
```

- [ ] **Step 5: Add `return_intermediates` to `deconvolve()` signature and return logic**

Change the function signature:

```python
def deconvolve(peaks: list,
               params: Optional[DeconvolutionParams] = None,
               return_intermediates: bool = False) -> list | tuple:
```

Accumulate `model_peaks` list (already done in the existing code) and return intermediates when requested. Add just before the final `return result`:

```python
    if return_intermediates:
        intermediates = {
            'rt_clusters': rt_clusters,
            'noise_peaks': noise_peaks,
            'model_peaks': model_peaks,  # NNLS-path only; fallback-path not included
        }
        return result, intermediates
    return result
```

- [ ] **Step 6: Write failing tests for `return_intermediates=True`**

Add to `TestDeconvolve` in `deconvolution/test_spectral_deconvolution.py`:

```python
def test_return_intermediates_gives_tuple(self):
    peaks = [
        make_gaussian_peak(rt_center=5.0, mz=float(m), height=1000.0)
        for m in [100, 200]
    ]
    result = deconvolve(peaks, return_intermediates=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    components, intermediates = result
    assert isinstance(components, list)
    assert 'rt_clusters' in intermediates
    assert 'noise_peaks' in intermediates
    assert 'model_peaks' in intermediates

def test_return_intermediates_false_gives_list(self):
    peaks = [
        make_gaussian_peak(rt_center=5.0, mz=float(m), height=1000.0)
        for m in [100, 200]
    ]
    result = deconvolve(peaks, return_intermediates=False)
    assert isinstance(result, list)

def test_intermediates_noise_peaks_are_eic_peaks(self):
    # One isolated noise peak (min_samples=2, only 1 peak at that RT)
    noise = make_gaussian_peak(rt_center=10.0, height=500.0, mz=99.0)
    clustered = [
        make_gaussian_peak(rt_center=1.0, mz=float(i), height=500.0)
        for i in range(100, 103)
    ]
    params = DeconvolutionParams(
        min_cluster_distance=0.01, min_cluster_size=2,
        min_cluster_intensity=100.0, min_model_peak_sharpness=1.0,
        use_is_shared=False,
    )
    _, intermediates = deconvolve(clustered + [noise], params, return_intermediates=True)
    assert any(p.mz == 99.0 for p in intermediates['noise_peaks'])
```

- [ ] **Step 7: Run tests — expect failures (function not yet updated to return intermediates)**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt/deconvolution"
conda run -n chromakit-env pytest test_spectral_deconvolution.py::TestDeconvolve -v
```

Expected: new tests fail, existing tests pass (they call `deconvolve()` without `return_intermediates`).

- [ ] **Step 8: Run full test suite — all existing tests should still pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py -v
```

Expected: only the new 3 `TestDeconvolve` tests fail. If other tests fail, the `_cluster_by_rt` tuple change broke something — fix before proceeding.

- [ ] **Step 9: Implement `return_intermediates` in `deconvolve()` (Step 5 above)**

Apply the change described in Step 5.

- [ ] **Step 10: Run full test suite — all tests should pass**

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py -v
```

Expected: all tests pass.

- [ ] **Step 11: Apply identical changes to `deconvolution/spectral_deconvolution.py`**

This is the standalone reference copy. Apply the exact same changes:
- `_cluster_by_rt` returns `(clusters, noise_peaks)` tuple
- `deconvolve()` call site unpacks tuple
- `deconvolve()` gains `return_intermediates` param

Run tests one more time to confirm the standalone copy's tests still pass (they test the same logic):

```bash
conda run -n chromakit-env pytest test_spectral_deconvolution.py -v
```

- [ ] **Step 12: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
git add logic/spectral_deconvolution.py deconvolution/spectral_deconvolution.py deconvolution/test_spectral_deconvolution.py
git commit -m "feat: _cluster_by_rt returns (clusters, noise_peaks); deconvolve() gains return_intermediates param"
```

---

## Task 2: Bug Fix — `display_peak_spectrum()` calls `set_current_spectrum()` + route peak-click through it

**Files:**
- Modify: `ui/frames/ms.py` (lines 451–492)
- Modify: `ui/app.py` (lines 757–860)

- [ ] **Step 1: Extend `display_peak_spectrum()` to call `set_current_spectrum()` on success**

In `ui/frames/ms.py`, after `self.plot_mass_spectrum(...)` at line 480, add the `set_current_spectrum` call before the implicit return:

```python
    # Use same label format as old code — preserves peak_number for search results tree
    label = f"Peak {getattr(peak, 'peak_number', '')} (RT={peak.retention_time:.3f})"
    self.plot_mass_spectrum(spectrum['mz'], spectrum['intensities'], label)
    # Populate single-peak search field with the spectrum just displayed.
    # Called here (not in plot_mass_spectrum) because plot_mass_spectrum does not
    # call set_current_spectrum — they are separate methods.
    self.set_current_spectrum(
        spectrum['mz'],
        spectrum['intensities'],
        label,
        rt=peak.retention_time,
    )
```

The two early-return paths (`return` at line 475 when `_current_data_path` is missing, and `return` at line 478 when `spectrum is None`) are left untouched — `set_current_spectrum` is only called on the success path.

The status note ("deconvolved spectrum available…") cannot move here — `MSFrame` has no status bar reference. It stays in `on_peak_spectrum_requested`.

- [ ] **Step 3: Route `on_peak_spectrum_requested` through `display_peak_spectrum()`**

In `ui/app.py`, replace the spectrum extraction block in `on_peak_spectrum_requested` (currently lines ~800–853) with:

```python
        # Route through display_peak_spectrum() — handles toggle, deconvolved vs raw,
        # and calls set_current_spectrum() internally.
        ms_path = self._get_ms_data_path()
        self.ms_frame.display_peak_spectrum(peak, data_directory=ms_path)

        # Update RT entry (display_peak_spectrum does not have access to rt_entry)
        self.ms_frame.rt_entry.setText(f"{peak.retention_time:.3f}")

        # Update status bar
        has_deconv = getattr(peak, 'deconvolved_spectrum', None) is not None
        deconv_note = " (deconvolved spectrum available — use Search All to apply)" if has_deconv else ""
        self.status_bar.showMessage(
            f"Extracted mass spectrum for peak {peak.peak_number} at RT={peak.retention_time:.3f}{deconv_note}"
        )
```

**Exact diff — what to delete vs keep in `on_peak_spectrum_requested`:**
- Lines 761–763 (`has_ms_data` guard): **keep**
- Lines 764–767 (`integrated_peaks` bounds guard): **keep**
- Line 770 (`peak = self.integrated_peaks[peak_index]`): **keep**
- Lines 772–773 (status bar "Extracting…" message): **keep** (will be overwritten on success)
- Lines 775–778 (`ms_toolkit` guard): **DELETE** — `display_peak_spectrum` uses `SpectrumExtractor` from `logic/`, not `ms_toolkit`
- Lines 780–782 (`current_directory_path` guard): **keep**
- Lines 784–798 (`search_options` block): **DELETE** — not used by `display_peak_spectrum`
- Lines 800–end-of-method (the entire `try` block with `extract_for_peak`, `plot_mass_spectrum`, `set_current_spectrum`, and all `print` statements): **DELETE** and replace with the code block above

- [ ] **Step 4: Manual test — toggle button appears after deconvolution**

Launch the app:
```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
conda run -n chromakit-env python main.py
```

1. Load a `.D` file with MS data, run FID integration
2. Click "Deconvolve MS" and wait for completion
3. Click a peak in the chromatogram
4. Verify: the toggle button ("Deconvolved" / "Raw Apex") appears in the MS tab toolbar
5. Click the toggle — spectrum should change between deconvolved and raw apex
6. Click a peak that was not deconvolved — toggle should be hidden

- [ ] **Step 5: Commit**

```bash
git add ui/frames/ms.py ui/app.py
git commit -m "fix: route on_peak_spectrum_requested through display_peak_spectrum; show deconvolved toggle"
```

---

## Task 3: Bug Fix — `QProgressDialog` for deconvolution

**Files:**
- Modify: `ui/app.py` (lines 2777–2852)

- [ ] **Step 1: Refactor `_on_deconvolve_ms_finished` and `_on_deconvolve_ms_error` signatures**

Use `progress_dialog=None` as the default so any future direct call won't crash:

```python
def _on_deconvolve_ms_finished(self, progress_dialog=None):
    if progress_dialog:
        progress_dialog.setValue(100)
        progress_dialog.close()
    # ... existing body (status bar summary, _refresh_current_peak_ms_display, inspector refresh)

def _on_deconvolve_ms_error(self, msg: str, progress_dialog=None):
    if progress_dialog:
        progress_dialog.close()
    self.status_bar.showMessage("Spectral deconvolution failed")
    QMessageBox.critical(self, "Deconvolution Error", msg)
```

Remove `_on_deconvolve_ms_progress` entirely (inlined as a lambda in the next step).

- [ ] **Step 2: Replace signal connections in `_on_deconvolve_ms_clicked` with lambda closures**

Replace lines 2795–2797:
```python
worker.signals.progress.connect(self._on_deconvolve_ms_progress)
worker.signals.finished.connect(self._on_deconvolve_ms_finished)
worker.signals.error.connect(self._on_deconvolve_ms_error)
```

with:
```python
        progress_dialog = QProgressDialog(
            "Running spectral deconvolution...", "Cancel", 0, 100, self
        )
        progress_dialog.setWindowTitle("Spectral Deconvolution")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.canceled.connect(lambda: setattr(worker, 'cancelled', True))

        worker.signals.progress.connect(
            lambda pct: (progress_dialog.setValue(pct),
                         self.status_bar.showMessage(f"Spectral deconvolution: {pct}%"))
        )
        worker.signals.finished.connect(
            lambda: self._on_deconvolve_ms_finished(progress_dialog)
        )
        worker.signals.error.connect(
            lambda msg: self._on_deconvolve_ms_error(msg, progress_dialog)
        )
```

Note: `QProgressDialog` must be imported at the top of `app.py` if not already present. Check existing imports — it's likely already there from the cross-file assignment dialog.

- [ ] **Step 3: Manual test — progress dialog appears**

Launch the app, load a file with MS data, integrate FID peaks, click "Deconvolve MS":
- A modal progress dialog titled "Spectral Deconvolution" should appear at 0%
- It should update to 100% as windows are processed
- Clicking "Cancel" should stop the worker (may not cancel immediately mid-window)
- Status bar shows summary on completion

- [ ] **Step 4: Commit**

```bash
git add ui/app.py
git commit -m "fix: add QProgressDialog to spectral deconvolution worker flow"
```

---

## Task 4: Inspector Dialog — Skeleton + Layout

**Files:**
- Create: `ui/dialogs/spectral_deconv_inspector.py`

- [ ] **Step 1: Create the dialog file with skeleton class**

```python
"""Spectral deconvolution chunk inspector dialog.

Shows DBSCAN RT clustering and EIC traces for each peak window,
with parameter controls for interactive tuning.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QRunnable, Slot
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QSplitter, QWidget,
    QGroupBox, QFormLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QProgressBar,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from logic.spectral_deconvolution import DeconvolutionParams
from logic.spectral_deconv_runner import WindowGroupingParams, _group_peaks_into_windows


class SpectralDeconvInspectorDialog(QDialog):
    """Non-modal dialog for inspecting ADAP-GC deconvolution per window."""

    rerun_requested = Signal(object, object)  # (DeconvolutionParams, WindowGroupingParams)

    def __init__(
        self,
        peaks: list,
        ms_data_path: str,
        deconv_params: DeconvolutionParams,
        grouping_params: WindowGroupingParams,
        initial_peak_index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectral Deconvolution Inspector")
        self.setMinimumSize(1100, 650)
        self.setModal(False)

        self._peaks = peaks           # live reference — not a copy
        self._ms_data_path = ms_data_path
        self._ms = None               # opened once on first use
        self._windows: list = []      # [(w_start, w_end, [peaks_in_window]), ...]
        self._current_window_idx: int = 0
        self._preview_worker = None

        # Build UI first, then populate params from arguments
        self._build_ui()
        self._load_params(deconv_params, grouping_params)
        self._rebuild_windows()

        # Navigate to the window containing initial_peak_index
        target_window = self._window_for_peak_index(initial_peak_index)
        self._navigate_to(target_window, auto_run=True)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_params_panel())
        splitter.addWidget(self._build_plot_panel())
        splitter.setSizes([320, 780])

    def _build_params_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # Window Grouping group
        wg_group = QGroupBox("Window Grouping")
        wg_form = QFormLayout(wg_group)

        self._gap_spin = QDoubleSpinBox()
        self._gap_spin.setRange(0.0, 10.0)
        self._gap_spin.setDecimals(3)
        self._gap_spin.setSingleStep(0.01)
        self._gap_spin.setSpecialValueText("Auto")
        wg_form.addRow("Gap tolerance (min):", self._gap_spin)

        self._padding_spin = QDoubleSpinBox()
        self._padding_spin.setRange(0.0, 5.0)
        self._padding_spin.setDecimals(2)
        wg_form.addRow("Padding fraction:", self._padding_spin)

        self._rt_match_spin = QDoubleSpinBox()
        self._rt_match_spin.setRange(0.001, 1.0)
        self._rt_match_spin.setDecimals(3)
        self._rt_match_spin.setSingleStep(0.005)
        wg_form.addRow("RT match tolerance (min):", self._rt_match_spin)
        layout.addWidget(wg_group)

        # ADAP-GC Parameters group
        adap_group = QGroupBox("ADAP-GC Parameters")
        adap_form = QFormLayout(adap_group)

        self._min_dist_spin = QDoubleSpinBox()
        self._min_dist_spin.setRange(0.0001, 1.0)
        self._min_dist_spin.setDecimals(4)
        self._min_dist_spin.setSingleStep(0.001)
        adap_form.addRow("Min cluster distance (min):", self._min_dist_spin)

        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(1, 50)
        adap_form.addRow("Min cluster size:", self._min_size_spin)

        self._min_intensity_spin = QDoubleSpinBox()
        self._min_intensity_spin.setRange(0.0, 1e8)
        self._min_intensity_spin.setDecimals(0)
        self._min_intensity_spin.setSingleStep(100.0)
        adap_form.addRow("Min cluster intensity:", self._min_intensity_spin)

        self._shape_sim_spin = QDoubleSpinBox()
        self._shape_sim_spin.setRange(0.0, 90.0)
        self._shape_sim_spin.setDecimals(1)
        adap_form.addRow("Shape similarity (°):", self._shape_sim_spin)

        self._model_peak_combo = QComboBox()
        self._model_peak_combo.addItems(["sharpness", "intensity", "mz"])
        adap_form.addRow("Model peak choice:", self._model_peak_combo)

        self._excluded_mz_edit = QLineEdit()
        self._excluded_mz_edit.setPlaceholderText("e.g. 73, 147, 221")
        adap_form.addRow("Excluded m/z:", self._excluded_mz_edit)
        layout.addWidget(adap_group)

        # Top N traces
        top_n_layout = QFormLayout()
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 200)
        self._top_n_spin.setValue(20)
        top_n_layout.addRow("Top N EIC traces:", self._top_n_spin)
        layout.addLayout(top_n_layout)

        layout.addStretch()

        # Buttons
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        layout.addWidget(self._preview_btn)

        self._apply_btn = QPushButton("Apply to All && Rerun")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self._apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        return panel

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Navigation row
        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.clicked.connect(self._on_prev)
        nav_layout.addWidget(self._prev_btn)

        self._window_combo = QComboBox()
        self._window_combo.setSizePolicy(
            self._window_combo.sizePolicy().horizontalPolicy(),
            self._window_combo.sizePolicy().verticalPolicy()
        )
        self._window_combo.currentIndexChanged.connect(self._on_window_selected)
        nav_layout.addWidget(self._window_combo, stretch=1)

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        nav_layout.addWidget(self._next_btn)
        layout.addLayout(nav_layout)

        # Matplotlib canvas — 2-row subplot
        self._fig = Figure(figsize=(8, 6), tight_layout=True)
        self._ax_scatter = self._fig.add_subplot(2, 1, 1)
        self._ax_eic = self._fig.add_subplot(2, 1, 2, sharex=self._ax_scatter)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

        # Status label
        self._status_label = QLabel("Select a window to preview.")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        return panel

    # ── Params load/save ───────────────────────────────────────────────────────

    def _load_params(self, dp: DeconvolutionParams, gp: WindowGroupingParams):
        self._gap_spin.setValue(gp.gap_tolerance or 0.0)
        self._padding_spin.setValue(gp.padding_fraction)
        self._rt_match_spin.setValue(gp.rt_match_tolerance)

        self._min_dist_spin.setValue(dp.min_cluster_distance)
        self._min_size_spin.setValue(dp.min_cluster_size)
        self._min_intensity_spin.setValue(dp.min_cluster_intensity)
        self._shape_sim_spin.setValue(dp.shape_sim_threshold)
        idx = self._model_peak_combo.findText(dp.model_peak_choice)
        if idx >= 0:
            self._model_peak_combo.setCurrentIndex(idx)
        self._excluded_mz_edit.setText(
            ", ".join(str(m) for m in dp.excluded_mz)
        )

    def _read_params(self) -> tuple[DeconvolutionParams, WindowGroupingParams]:
        raw_gap = self._gap_spin.value()
        gp = WindowGroupingParams(
            gap_tolerance=None if raw_gap == 0.0 else raw_gap,
            padding_fraction=self._padding_spin.value(),
            rt_match_tolerance=self._rt_match_spin.value(),
        )
        excluded = [
            float(x.strip())
            for x in self._excluded_mz_edit.text().split(",")
            if x.strip()
        ]
        dp = DeconvolutionParams(
            min_cluster_distance=self._min_dist_spin.value(),
            min_cluster_size=self._min_size_spin.value(),
            min_cluster_intensity=self._min_intensity_spin.value(),
            shape_sim_threshold=self._shape_sim_spin.value(),
            model_peak_choice=self._model_peak_combo.currentText(),
            excluded_mz=excluded,
        )
        return dp, gp

    # ── Window management ──────────────────────────────────────────────────────

    def _rebuild_windows(self):
        """Recompute window list from current peaks + grouping params."""
        import rainbow as rb
        if self._ms is None:
            data_dir = rb.read(self._ms_data_path)
            self._ms = data_dir.get_file('data.ms')

        _, gp = self._read_params()
        rt_min = float(self._ms.xlabels[0])
        rt_max = float(self._ms.xlabels[-1])
        self._windows = _group_peaks_into_windows(self._peaks, gp, rt_min, rt_max)
        self._populate_combo()

    def _populate_combo(self):
        self._window_combo.blockSignals(True)
        self._window_combo.clear()
        for i, (w_start, w_end, win_peaks) in enumerate(self._windows):
            peak_rts = ", ".join(f"{p.retention_time:.3f}" for p in win_peaks)
            label = (
                f"Window {i+1} "
                f"(RT {w_start:.2f}–{w_end:.2f}, "
                f"{len(win_peaks)} peak(s): {peak_rts} min)"
            )
            self._window_combo.addItem(label)
        self._window_combo.blockSignals(False)

    def _window_for_peak_index(self, peak_index: int) -> int:
        """Return window index containing the peak at peak_index, or 0."""
        if not self._windows or peak_index < 0 or peak_index >= len(self._peaks):
            return 0
        target_peak = self._peaks[peak_index]
        for i, (_, _, win_peaks) in enumerate(self._windows):
            if target_peak in win_peaks:
                return i
        return 0

    def _navigate_to(self, window_idx: int, auto_run: bool = False):
        if not self._windows:
            return
        window_idx = max(0, min(window_idx, len(self._windows) - 1))
        self._current_window_idx = window_idx
        self._window_combo.blockSignals(True)
        self._window_combo.setCurrentIndex(window_idx)
        self._window_combo.blockSignals(False)
        self._prev_btn.setEnabled(window_idx > 0)
        self._next_btn.setEnabled(window_idx < len(self._windows) - 1)
        if auto_run:
            self._run_preview()

    def navigate_to_peak(self, peak_index: int):
        """Called by ChromaKitApp when user clicks a peak while dialog is open."""
        self._navigate_to(self._window_for_peak_index(peak_index), auto_run=True)

    def refresh_current_window(self):
        """Called by ChromaKitApp after a full rerun to update plots."""
        self._run_preview()

    def set_controls_enabled(self, enabled: bool):
        """Enable/disable all interactive controls (during full rerun)."""
        for w in (self._preview_btn, self._apply_btn, self._prev_btn,
                  self._next_btn, self._window_combo,
                  self._gap_spin, self._padding_spin, self._rt_match_spin,
                  self._min_dist_spin, self._min_size_spin, self._min_intensity_spin,
                  self._shape_sim_spin, self._model_peak_combo,
                  self._excluded_mz_edit, self._top_n_spin):
            w.setEnabled(enabled)

    # ── Navigation slots ───────────────────────────────────────────────────────

    def _on_prev(self):
        self._navigate_to(self._current_window_idx - 1, auto_run=True)

    def _on_next(self):
        self._navigate_to(self._current_window_idx + 1, auto_run=True)

    def _on_window_selected(self, idx: int):
        if idx != self._current_window_idx:
            self._navigate_to(idx, auto_run=True)

    # ── Preview / Apply ────────────────────────────────────────────────────────

    def _on_preview_clicked(self):
        self._rebuild_windows()   # re-group in case grouping params changed
        self._run_preview()

    def _on_apply_clicked(self):
        dp, gp = self._read_params()
        from PySide6.QtCore import QSettings
        s = QSettings("CalebCoatney", "ChromaKit")
        s.setValue("ms_spectral_deconv/min_cluster_distance", dp.min_cluster_distance)
        s.setValue("ms_spectral_deconv/min_cluster_size", dp.min_cluster_size)
        s.setValue("ms_spectral_deconv/min_cluster_intensity", dp.min_cluster_intensity)
        s.setValue("ms_spectral_deconv/shape_sim_threshold", dp.shape_sim_threshold)
        s.setValue("ms_spectral_deconv/model_peak_choice", dp.model_peak_choice)
        s.setValue("ms_spectral_deconv/excluded_mz",
                   ", ".join(str(m) for m in dp.excluded_mz))
        s.setValue("ms_spectral_deconv/gap_tolerance", self._gap_spin.value())
        s.setValue("ms_spectral_deconv/padding_fraction", gp.padding_fraction)
        s.setValue("ms_spectral_deconv/rt_match_tolerance", gp.rt_match_tolerance)
        # Note: DeconvolutionParams fields use_is_shared, edge_to_height_ratio,
        # delta_to_height_ratio, min_model_peak_sharpness, and excluded_mz_tolerance
        # are not exposed in this dialog (consistent with MSOptionsDialog) and are
        # always constructed at their dataclass defaults.
        self.rerun_requested.emit(dp, gp)

    def _run_preview(self):
        """Launch background worker for the current window."""
        if not self._windows:
            return
        if self._preview_worker is not None:
            return  # already running
        if self._ms is None:
            self._status_label.setText("Error: MS data file could not be opened.")
            return

        self.set_controls_enabled(False)
        self._status_label.setText("Running deconvolution…")

        w_start, w_end, win_peaks = self._windows[self._current_window_idx]
        dp, _ = self._read_params()

        from PySide6.QtCore import QThreadPool
        self._preview_worker = _PreviewWorker(
            ms=self._ms,
            w_start=w_start,
            w_end=w_end,
            win_peaks=win_peaks,
            deconv_params=dp,
            top_n=self._top_n_spin.value(),
        )
        self._preview_worker.signals.finished.connect(self._on_preview_finished)
        self._preview_worker.signals.error.connect(self._on_preview_error)
        QThreadPool.globalInstance().start(self._preview_worker)

    def _on_preview_finished(self, result: dict):
        self._preview_worker = None
        self.set_controls_enabled(True)
        self._render_plots(result)

    def _on_preview_error(self, msg: str):
        self._preview_worker = None
        self.set_controls_enabled(True)
        self._status_label.setText(f"Error: {msg}")

    def closeEvent(self, event):
        self._ms = None  # release rainbow data
        super().closeEvent(event)
```

- [ ] **Step 2: Manual smoke-test — dialog opens without crashing**

```bash
# Quick Python smoke test (no GUI yet — just import check)
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
conda run -n chromakit-env python -c "from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog; print('OK')"
```

Expected: `OK` with no ImportError.

Note: `ui/dialogs/__init__.py` exists and is empty — no update needed. The import path works as-is.

- [ ] **Step 3: Commit skeleton**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat: add SpectralDeconvInspectorDialog skeleton with layout and navigation"
```

---

## Task 5: Inspector Dialog — Preview Worker

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py`

Add `_PreviewWorkerSignals` and `_PreviewWorker` classes to the file. Place them just above `SpectralDeconvInspectorDialog`.

- [ ] **Step 1: Add `_PreviewWorkerSignals` and `_PreviewWorker` classes**

```python
class _PreviewWorkerSignals(QObject):
    finished = Signal(dict)   # result dict passed to _on_preview_finished
    error = Signal(str)


class _PreviewWorker(QRunnable):
    """Background worker: extract EICs + run deconvolve() for one window."""

    def __init__(self, ms, w_start: float, w_end: float,
                 win_peaks: list, deconv_params: DeconvolutionParams,
                 top_n: int):
        super().__init__()
        self._ms = ms
        self._w_start = w_start
        self._w_end = w_end
        self._win_peaks = win_peaks
        self._deconv_params = deconv_params
        self._top_n = top_n
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            from logic.eic_extractor import extract_eic_peaks
            from logic.spectral_deconvolution import deconvolve

            eic_peaks = extract_eic_peaks(
                self._ms,
                t_start=self._w_start,
                t_end=self._w_end,
                min_intensity=self._deconv_params.min_cluster_intensity,
            )

            if not eic_peaks:
                self.signals.finished.emit({
                    'eic_peaks': [],
                    'components': [],
                    'intermediates': {'rt_clusters': [], 'noise_peaks': [], 'model_peaks': []},
                    'win_peaks': self._win_peaks,
                    'top_n': self._top_n,
                    'w_start': self._w_start,
                    'w_end': self._w_end,
                    'empty': True,
                })
                return

            # Sort by descending max intensity; keep top_n for EIC plot
            eic_sorted = sorted(eic_peaks, key=lambda p: -p.intensity_array.max())
            top_eic = eic_sorted[:self._top_n]

            # Run deconvolution on all EIC peaks (not just top N) for accuracy
            components, intermediates = deconvolve(
                eic_peaks, self._deconv_params, return_intermediates=True
            )

            self.signals.finished.emit({
                'eic_peaks': eic_peaks,
                'top_eic': top_eic,
                'components': components,
                'intermediates': intermediates,
                'win_peaks': self._win_peaks,
                'top_n': self._top_n,
                'w_start': self._w_start,
                'w_end': self._w_end,
                'empty': False,
            })
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())
```

- [ ] **Step 2: Manual smoke-test — preview runs without crash**

Launch the app, load a file, integrate FID peaks, click "Deconvolve MS" once, then open the inspector (you'll wire the button in Task 7 — for now, test the dialog directly in a Python session or temporarily add a quick-open shortcut). Alternatively, defer this smoke test to Task 7 when the button is wired.

- [ ] **Step 3: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat: add _PreviewWorker to inspector dialog for per-window EIC extraction + deconvolution"
```

---

## Task 6: Inspector Dialog — Plot Rendering

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py`

Add `_render_plots` method to `SpectralDeconvInspectorDialog`.

- [ ] **Step 1: Add `_render_plots` method**

```python
    def _render_plots(self, result: dict):
        """Render DBSCAN scatter (top) and EIC traces (bottom) from preview result."""
        self._ax_scatter.clear()
        self._ax_eic.clear()

        win_peaks = result['win_peaks']
        fid_rts = [p.retention_time for p in win_peaks]

        if result.get('empty'):
            self._ax_scatter.set_title("No EIC peaks found in this window")
            self._ax_eic.set_title("(check Min Cluster Intensity setting)")
            self._canvas.draw()
            self._status_label.setText(
                "No EIC peaks found in this window — try lowering Min Cluster Intensity."
            )
            return

        intermediates = result['intermediates']
        rt_clusters = intermediates['rt_clusters']  # list[list[EICPeak]]
        noise_peaks = intermediates['noise_peaks']
        model_peaks = intermediates['model_peaks']
        top_eic = result['top_eic']
        components = result['components']

        # Build color map: cluster index → matplotlib color
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab10')
        cluster_colors: dict[int, tuple] = {}
        eic_to_color: dict[int, tuple] = {}  # id(EICPeak) → color

        for ci, cluster in enumerate(rt_clusters):
            color = cmap(ci % 10)
            cluster_colors[ci] = color
            for peak in cluster:
                eic_to_color[id(peak)] = color

        model_peak_ids = {id(mp) for mp in model_peaks}

        # ── Top subplot: DBSCAN scatter ────────────────────────────────────────
        # Noise points
        if noise_peaks:
            noise_rts = [p.rt_apex for p in noise_peaks]
            noise_mzs = [p.mz for p in noise_peaks]
            self._ax_scatter.scatter(
                noise_rts, noise_mzs, color='gray', s=10, alpha=0.5,
                label='Noise', zorder=2
            )

        # Clustered points
        for ci, cluster in enumerate(rt_clusters):
            color = cluster_colors[ci]
            rts = [p.rt_apex for p in cluster]
            mzs = [p.mz for p in cluster]
            self._ax_scatter.scatter(rts, mzs, color=color, s=18, zorder=3)
            # Model peaks: star marker
            for peak in cluster:
                if id(peak) in model_peak_ids:
                    self._ax_scatter.scatter(
                        [peak.rt_apex], [peak.mz],
                        color=color, marker='*', s=80, zorder=4
                    )

        # FID peak RT lines with RT labels
        for rt in fid_rts:
            self._ax_scatter.axvline(rt, color='gray', linestyle='--', alpha=0.6, linewidth=1)
            self._ax_scatter.text(
                rt, 1.01, f"{rt:.3f}",
                transform=self._ax_scatter.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7, color='gray',
            )

        n_noise = len(noise_peaks)
        self._ax_scatter.set_ylabel("m/z")
        self._ax_scatter.set_title(
            f"RT Clusters — {len(rt_clusters)} cluster(s), {n_noise} noise point(s)"
        )

        # ── Bottom subplot: EIC traces ─────────────────────────────────────────
        for peak in top_eic:
            color = eic_to_color.get(id(peak), 'gray')
            lw = 2.5 if id(peak) in model_peak_ids else 1.0
            self._ax_eic.plot(
                peak.rt_array, peak.intensity_array,
                color=color, linewidth=lw, alpha=0.8
            )

        # FID peak RT lines
        for rt in fid_rts:
            self._ax_eic.axvline(rt, color='gray', linestyle='--', alpha=0.6, linewidth=1)

        shown = len(top_eic)
        total = len(result['eic_peaks'])
        self._ax_eic.set_xlabel("Retention Time (min)")
        self._ax_eic.set_ylabel("Intensity")
        self._ax_eic.set_title(
            f"EIC Traces (top {shown} of {total} shown)"
        )

        self._canvas.draw()

        # ── Status line ────────────────────────────────────────────────────────
        n_comp = len(components)
        matched = [
            p for p in win_peaks
            if getattr(p, 'deconvolved_spectrum', None) is not None
        ]
        # Note: win_peaks here are from the window grouping at the time
        # of dialog open; deconvolved_spectrum reflects the last full run,
        # not this preview run
        if n_comp == 0:
            self._status_label.setText("0 components found (all peaks filtered)")
        else:
            n_matched = len(matched)
            matched_rts = ", ".join(f"{p.retention_time:.3f}" for p in matched)
            unmatched = len(win_peaks) - n_matched
            self._status_label.setText(
                f"{n_comp} component(s) found — "
                f"{n_matched} matched to FID peak(s)"
                + (f" (RT {matched_rts} min)" if matched_rts else "")
                + (f" — {unmatched} unmatched" if unmatched > 0 else "")
            )
```

- [ ] **Step 2: Manual test — plots render correctly**

After wiring the Inspect button in Task 7, open the inspector and verify:
- Top subplot shows colored scatter (each RT cluster a different color, noise gray, model peaks as stars)
- Bottom subplot shows EIC traces (colored by cluster, model peaks bold)
- FID peak RT dashed lines appear in both subplots
- Status line shows component/match counts
- Prev/Next/dropdown navigate between windows and trigger re-render

- [ ] **Step 3: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat: add _render_plots to inspector dialog — DBSCAN scatter + EIC traces"
```

---

## Task 7: "Inspect" Button in MS Tab + App Wiring

**Files:**
- Modify: `ui/frames/ms.py`
- Modify: `ui/app.py`

- [ ] **Step 1: Add "Inspect" button and `inspect_requested` signal to `MSFrame`**

In `ui/frames/ms.py`, add the signal declaration near the top of the class (with the other signals):
```python
inspect_requested = Signal()
```

In `_build_ui` (or wherever the toolbar row is built, around line 172 where `spectrum_toggle_btn` is added), add after the toggle button:

```python
        self.inspect_btn = QPushButton("Inspect")
        self.inspect_btn.setEnabled(False)   # enabled by ChromaKitApp after first deconvolution
        self.inspect_btn.setFixedWidth(80)
        self.inspect_btn.clicked.connect(self.inspect_requested)
        mz_shift_layout.addWidget(self.inspect_btn)
```

- [ ] **Step 2: Add `deconvolve_ms_run` flag + enable Inspect button in `ChromaKitApp`**

In `ui/app.py`, add to `__init__` (near `_last_ms_peak_index`):
```python
        self._deconvolve_ms_run = False
        self._deconv_inspector = None
```

In `_on_deconvolve_ms_finished(self, progress_dialog)`, after the existing status bar message, add:
```python
        self._deconvolve_ms_run = True
        self.ms_frame.inspect_btn.setEnabled(True)
```

- [ ] **Step 3: Wire `inspect_requested` signal in `ChromaKitApp.__init__`**

Add alongside the other `ms_frame` signal connections (around line 152):
```python
        self.ms_frame.inspect_requested.connect(self._on_inspect_requested)
```

- [ ] **Step 4: Implement `_on_inspect_requested`**

```python
    def _on_inspect_requested(self):
        """Open (or bring to front) the spectral deconvolution inspector dialog."""
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            return
        if not hasattr(self.data_handler, 'current_directory_path') \
                or not self.data_handler.current_directory_path:
            return

        peak_index = self._last_ms_peak_index or 0

        ms_path = self._get_ms_data_path()
        if not ms_path:
            self.status_bar.showMessage("Cannot open inspector: no MS data path available")
            return

        # If already open, navigate rather than open a second instance
        if self._deconv_inspector is not None and self._deconv_inspector.isVisible():
            self._deconv_inspector.navigate_to_peak(peak_index)
            self._deconv_inspector.raise_()
            self._deconv_inspector.activateWindow()
            return

        from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog
        deconv_params, grouping_params = self._get_spectral_deconv_params()

        self._deconv_inspector = SpectralDeconvInspectorDialog(
            peaks=self.integrated_peaks,
            ms_data_path=ms_path,  # already validated above
            deconv_params=deconv_params,
            grouping_params=grouping_params,
            initial_peak_index=peak_index,
            parent=self,
        )
        self._deconv_inspector.rerun_requested.connect(self._on_inspector_rerun_requested)
        self._deconv_inspector.show()
```

- [ ] **Step 5: Implement `_on_inspector_rerun_requested`**

```python
    def _on_inspector_rerun_requested(self, deconv_params, grouping_params):
        """Dispatch full spectral deconvolution worker with inspector's params."""
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            return
        if not hasattr(self.data_handler, 'current_directory_path') \
                or not self.data_handler.current_directory_path:
            return

        if self._deconv_inspector:
            self._deconv_inspector.set_controls_enabled(False)

        from logic.spectral_deconv_worker import SpectralDeconvWorker

        worker = SpectralDeconvWorker(
            peaks=self.integrated_peaks,
            ms_data_path=self._get_ms_data_path(),
            deconv_params=deconv_params,
            grouping_params=grouping_params,
        )

        progress_dialog = QProgressDialog(
            "Running spectral deconvolution...", "Cancel", 0, 100, self
        )
        progress_dialog.setWindowTitle("Spectral Deconvolution")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.canceled.connect(lambda: setattr(worker, 'cancelled', True))

        worker.signals.progress.connect(
            lambda pct: (progress_dialog.setValue(pct),
                         self.status_bar.showMessage(f"Spectral deconvolution: {pct}%"))
        )
        worker.signals.finished.connect(
            lambda: self._on_deconvolve_ms_finished(progress_dialog)
        )
        worker.signals.error.connect(
            lambda msg: self._on_deconvolve_ms_error(msg, progress_dialog)
        )

        QThreadPool.globalInstance().start(worker)
```

- [ ] **Step 6: Update `_on_deconvolve_ms_finished` to refresh inspector**

At the end of `_on_deconvolve_ms_finished`, after the existing status bar summary and `_refresh_current_peak_ms_display()` call, add:

```python
        # Re-enable inspector and refresh its current window plot
        if self._deconv_inspector is not None and self._deconv_inspector.isVisible():
            self._deconv_inspector.set_controls_enabled(True)
            self._deconv_inspector.refresh_current_window()
```

- [ ] **Step 7: Full end-to-end manual test**

Launch the app:
```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt"
conda run -n chromakit-env python main.py
```

Test sequence:
1. Load a `.D` file with MS data
2. Run FID integration — peaks appear in RT table
3. Click "Deconvolve MS" — progress dialog appears, completes, status bar shows "N of M peaks deconvolved"
4. Click a peak in the chromatogram — toggle button ("Deconvolved") appears in MS tab; spectrum shows deconvolved data
5. Click toggle → spectrum switches to "Raw Apex"; click again → back to "Deconvolved"
6. Click "Inspect" button — inspector dialog opens, showing the window for the selected peak
7. Top subplot: colored scatter of EIC peaks by RT cluster, stars on model peaks, dashed FID RT lines
8. Bottom subplot: colored EIC traces, model peaks bold, dashed FID RT lines
9. Navigate prev/next windows — plots update
10. Change a parameter (e.g., lower Min Cluster Intensity) → click Preview → plots update for that window only
11. Click "Apply to All & Rerun" → progress dialog in main window, inspector re-renders on completion
12. Status line shows component/match counts
13. Peaks without deconvolved spectrum: toggle hidden, Inspect still works (shows window)

- [ ] **Step 8: Commit**

```bash
git add ui/frames/ms.py ui/app.py ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat: wire SpectralDeconvInspectorDialog — Inspect button, app wiring, Apply-to-All flow"
```

---

## Future: MS Tab Thumbnail

Deferred. A small read-only DBSCAN scatter embedded below the MS spectrum plot in the MS tab. Would show the cluster scatter for the current peak's window without opening the inspector dialog. Implementation requires caching per-window intermediates on `ChromaKitApp` after each full run. Tackle in a future session once the inspector dialog is validated in use.
