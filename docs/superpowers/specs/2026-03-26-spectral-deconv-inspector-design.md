# Spectral Deconvolution Inspector — Design Spec

**Date:** 2026-03-26
**Scope:** (1) Fix two integration bugs from Phase A, (2) implement Phase C chunk inspector dialog (`SpectralDeconvInspectorDialog`)
**Depends on:** Phase A implementation (complete as of 2026-03-25)

---

## Context

Phase A wired ADAP-GC spectral deconvolution into ChromaKit. Two bugs were found post-integration:

1. **Toggle not showing:** `on_peak_spectrum_requested` in `app.py` bypasses `ms_frame.display_peak_spectrum()` — it calls `extract_for_peak()` and `plot_mass_spectrum()` directly. The toggle button and deconvolved-spectrum display logic live entirely in `display_peak_spectrum()`, which is never called from the peak-click path.

2. **No progress bar:** The deconvolution worker's progress signal connects only to `status_bar.showMessage()`. Deconvolution takes 1–2 minutes; without a `QProgressDialog` (like batch search has), there is no visible indication that work is in progress.

Phase C adds a **chunk inspector dialog** so users can visualize DBSCAN clustering, EIC traces, and component-to-peak matching on a per-window basis — and interactively tune parameters before committing a full rerun.

---

## Changes

### 1. Bug fix — toggle not showing (`ui/app.py`)

Replace the body of `on_peak_spectrum_requested` from the spectrum extraction call onward with a single call to `ms_frame.display_peak_spectrum(peak, data_directory)`. Remove the direct calls to `extract_for_peak()`, `plot_mass_spectrum()`, and `set_current_spectrum()` that currently bypass the toggle logic.

`display_peak_spectrum()` already handles:
- Showing/hiding the toggle button based on `peak.deconvolved_spectrum`
- Using the deconvolved spectrum when available and toggle is on
- Falling back to `SpectrumExtractor.extract_at_rt()` otherwise
- Calling `plot_mass_spectrum()` internally

The status bar note ("deconvolved spectrum available — use Search All to apply") moves inside `display_peak_spectrum()`.

After `display_peak_spectrum()` returns, `set_current_spectrum()` (used to populate the single-peak search field) must still be called with the spectrum that was actually displayed. `plot_mass_spectrum()` does not call `set_current_spectrum()` internally. The clean fix is to extend `display_peak_spectrum()` to call `self.set_current_spectrum()` on the success path only — immediately before the final return, after `plot_mass_spectrum()` has been called. The two early-return paths (no data path available, spectrum is None) must not call `set_current_spectrum()`. This makes `display_peak_spectrum()` fully self-contained on success, and the caller only needs to follow up with `ms_frame.rt_entry.setText(f"{peak.retention_time:.3f}")`.

### 2. Bug fix — progress bar for deconvolution (`ui/app.py`)

In `_on_deconvolve_ms_clicked`, create a `QProgressDialog` before dispatching the worker, matching the batch search pattern. Worker signal connections are switched from named methods to lambda closures so the dialog reference is captured:

```python
progress_dialog = QProgressDialog("Running spectral deconvolution...", "Cancel", 0, 100, self)
progress_dialog.setWindowTitle("Spectral Deconvolution")
progress_dialog.setWindowModality(Qt.WindowModal)
progress_dialog.setMinimumDuration(0)
progress_dialog.canceled.connect(lambda: setattr(worker, 'cancelled', True))

worker.signals.progress.connect(lambda pct: progress_dialog.setValue(pct))
worker.signals.finished.connect(lambda: self._on_deconvolve_ms_finished(progress_dialog))
worker.signals.error.connect(lambda msg: self._on_deconvolve_ms_error(msg, progress_dialog))
```

`SpectralDeconvWorker` exposes cancellation via `self.cancelled` flag (not a `cancel()` method) — the lambda above sets it directly, matching how `BatchSearchWorker` cancellation works.

`_on_deconvolve_ms_finished(progress_dialog)` calls `progress_dialog.setValue(100)`, closes it, then shows the status bar summary. `_on_deconvolve_ms_error(msg, progress_dialog)` closes the dialog and shows `QMessageBox.critical`.

`_on_deconvolve_ms_finished` and `_on_deconvolve_ms_error` are updated to accept `progress_dialog` as a required parameter: `_on_deconvolve_ms_finished(self, progress_dialog)` and `_on_deconvolve_ms_error(self, msg, progress_dialog)`. These methods are only ever called via the lambda closures defined in `_on_deconvolve_ms_clicked` and `_on_inspector_rerun_requested` — they are never called directly elsewhere — so making `progress_dialog` required is safe and resolves any signature ambiguity. `_on_deconvolve_ms_progress` is removed; its body (one `status_bar.showMessage` line) is inlined into the lambda.

### 3. Algorithm change — `deconvolve()` intermediates (`logic/spectral_deconvolution.py`)

Add `return_intermediates: bool = False` parameter to `deconvolve()`.

When `True`, return a tuple `(components, intermediates)` instead of just `components`:

```python
intermediates = {
    'rt_clusters': list[list[EICPeak]],  # from _cluster_by_rt; index = cluster label
    'noise_peaks': list[EICPeak],        # DBSCAN noise points (label == -1)
    'model_peaks': list[EICPeak],        # selected model peaks (one per shape cluster)
}
```

`_cluster_by_rt` is modified to always return a tuple `(clusters, noise_peaks)` — `noise_peaks` is the list of `EICPeak` objects that received DBSCAN label -1. The call site in `deconvolve()` unpacks both values; when `return_intermediates=False` the noise list is simply unused. This avoids conditional return types inside `_cluster_by_rt`.

`intermediates['model_peaks']` captures model peaks selected via the normal (NNLS) path only. Model peaks selected via the fallback path (clusters where all candidates fail filtering — lines ~502–520 in the current implementation) are not included, because they bypass the `model_peaks` accumulator list and go directly to `result` as `DeconvolutedComponent` objects. This is acceptable for visualization: fallback-path components are a minority edge case, and their model peak's RT and m/z can still be recovered from the returned `DeconvolutedComponent.model_peak_mz` and `.rt` fields if needed.

All existing call sites pass no `return_intermediates` argument and are unaffected.

The same change is applied to `deconvolution/spectral_deconvolution.py` (the standalone reference copy) to keep them in sync.

### 4. Chunk inspector dialog (`ui/dialogs/spectral_deconv_inspector.py`)

New file: `SpectralDeconvInspectorDialog(QDialog)`.

#### Construction

```python
SpectralDeconvInspectorDialog(
    peaks,              # list[ChromatographicPeak] — live reference to integrated_peaks (not a copy)
    ms_data_path,       # str — .D directory path
    deconv_params,      # DeconvolutionParams — copied from current QSettings
    grouping_params,    # WindowGroupingParams — copied from current QSettings
    initial_peak_index=0,  # index into integrated_peaks for the current peak; dialog resolves
                           # this to a window index during __init__; defaults to 0
    parent=None,
)
```

`peaks` is stored as a live reference (`self._peaks = peaks`), not a copy. This ensures the status line's component-to-peak matching ("2 matched to FID peaks") reflects the current `integrated_peaks` state after a full rerun.

`initial_peak_index` is the raw index into `integrated_peaks` (i.e., `_last_ms_peak_index` from `ChromaKitApp`). During `__init__`, the dialog computes its window list and finds which window contains the peak at that index, using that as the initial selected window. This avoids requiring `app.py` to compute and cache the window list. If `initial_peak_index` is 0 or the peak is not found in any window, the dialog defaults to window 0.

Non-modal (`setModal(False)`). Opens alongside the main window; stays open while the user clicks other peaks.

#### Layout

Two-column `QSplitter` (horizontal, resizable):

**Left column — Parameters panel:**

- `QGroupBox("Window Grouping")` — gap tolerance, padding fraction, RT match tolerance (same controls as the MS Options Spectral Deconvolution tab)
- `QGroupBox("ADAP-GC Parameters")` — all `DeconvolutionParams` user-tunable fields (min cluster distance, min cluster size, min cluster intensity, shape similarity threshold, model peak choice, excluded m/z)
- `QSpinBox` — "Top N EIC traces" (default 20, range 1–200)
- `QPushButton("Preview")` — re-runs just the selected window with current params; updates plots; does not commit to peaks

**Right column — Navigation + plots:**

Chunk selector row:
```
[← Prev]  [QComboBox: "Window 3 (RT 12.41–13.08, 2 peaks: 12.61, 12.89 min)"]  [Next →]
```
Each dropdown entry: `"Window N (RT X.XX–Y.YY, K peak(s): RT1, RT2, ...)"`.
Selecting a window or clicking Prev/Next triggers an automatic preview re-run.

**Matplotlib canvas** — 2-row subplot (shared x-axis):

*Top subplot — RT Cluster scatter:*
- X: EICPeak apex RT, Y: m/z
- Points colored by RT cluster index (tab10 palette, cycling if >10 clusters)
- Noise points: gray, marker size 3
- Cluster points: marker size 5
- Model peak apices: same cluster color, star marker (`*`), size 10
- FID peak RTs: vertical dashed lines (gray, alpha=0.6), labeled with peak RT
- Title: `"RT Clusters — N clusters, M noise points"`

*Bottom subplot — EIC traces:*
- Top N EIC traces by `max(intensity_array)`, drawn as lines
- Line color: matches the RT cluster color of each EICPeak (gray if noise)
- Model peak traces: linewidth 2.5 (vs. 1.0 for others)
- X: RT across full window, Y: intensity
- FID peak RTs: vertical dashed lines (same as top)
- Title: `"EIC Traces (top N of M total shown)"`

**Status line** below canvas (plain `QLabel`):
`"3 components found — 2 matched to FID peaks (RT 12.61, 12.89 min) — 1 unmatched"`
When no EIC peaks found: `"No EIC peaks found in this window (check min cluster intensity)"`
When deconvolution returned no components: `"0 components found (all peaks filtered)"`

**Bottom button row:**
```
[Apply to All & Rerun]                          [Close]
```
"Apply to All & Rerun": saves the dialog's current params to QSettings, then emits `rerun_requested = Signal(object, object)` with `(deconv_params, grouping_params)`. `ChromaKitApp` connects this signal to a slot that dispatches `SpectralDeconvWorker` with the provided params (same flow as `_on_deconvolve_ms_clicked`, but using the dialog's params instead of reading from QSettings). The signal approach is consistent with the rest of the codebase's signal/slot pattern. The `QProgressDialog` is shown in the main window as usual. Dialog stays open.

After the full rerun completes (`_on_deconvolve_ms_finished` fires), the dialog automatically re-runs a preview for the currently selected window to reflect the updated `peak.deconvolved_spectrum` values. `ChromaKitApp` triggers this by calling `inspector.refresh_current_window()` from `_on_deconvolve_ms_finished` if `self._deconv_inspector` is open.

#### Threading

The preview re-run (EIC extraction + deconvolve) runs in a `QRunnable` on `QThreadPool` to keep the dialog responsive. A "Running..." label replaces the status line while the worker runs. The "Preview" button, chunk navigation controls, and "Apply to All & Rerun" button are all disabled during a preview run.

**Mutual exclusion with the main-window full rerun:** When "Apply to All & Rerun" is triggered, the dialog disables all its interactive controls (including "Preview") for the duration of the full worker run. The full worker runs on the main window's `QThreadPool` and mutates `integrated_peaks` in-place; the dialog must not start a preview run while that mutation is in progress. The "Apply to All & Rerun" button re-enables the dialog controls only after `_on_deconvolve_ms_finished` fires. A preview run in progress when the user clicks "Apply to All & Rerun" is not possible because the button is disabled while a preview is running.

#### MS data file

Opened once on dialog open (`rb.read(ms_data_path)`) and stored as `self._ms`. Goes out of scope when the dialog closes.

#### Initial window selection

When opened from the MS tab "Inspect" button, `ChromaKitApp` computes `initial_window_index` by finding which window contains the currently displayed peak, and passes it to the constructor. The dialog opens already showing that window.

### 5. "Inspect" button in MS tab (`ui/frames/ms.py`)

A small `QPushButton("Inspect")` added to the MS tab toolbar row (alongside the existing spectrum toggle button). Always visible; disabled until `deconvolve_ms_run` flag is set (toggled by `ChromaKitApp` after the first successful deconvolution run — or enabled unconditionally if we want to allow pre-run inspection). Emits a signal `inspect_requested = Signal()` — `ChromaKitApp` handles opening the dialog.

### 6. App wiring (`ui/app.py`)

- Connect `ms_frame.inspect_requested` → `_on_inspect_requested()`
- `_on_inspect_requested()`: passes `_last_ms_peak_index` (or 0 if None) directly to `SpectralDeconvInspectorDialog` as `initial_peak_index` — the dialog resolves this to a window index during init. Constructs `SpectralDeconvInspectorDialog`, connects `inspector.rerun_requested` → `_on_inspector_rerun_requested(deconv_params, grouping_params)`, stores reference as `self._deconv_inspector` (to prevent GC), calls `show()`
- If inspector is already open (`self._deconv_inspector` is not None and `isVisible()`): call `self._deconv_inspector.navigate_to_peak(_last_ms_peak_index or 0)` (the dialog resolves to window index internally) and bring to front instead of opening a second instance
- `_on_inspector_rerun_requested(deconv_params, grouping_params)`: same body as `_on_deconvolve_ms_clicked` but uses the provided params instead of reading from QSettings; calls `self._deconv_inspector.set_controls_enabled(False)` before dispatching the worker; uses the same lambda-captured `progress_dialog` pattern. `_on_deconvolve_ms_finished(progress_dialog)` always includes: `if self._deconv_inspector and self._deconv_inspector.isVisible(): self._deconv_inspector.set_controls_enabled(True); self._deconv_inspector.refresh_current_window()` — this re-enables inspector controls and refreshes the plot regardless of which path initiated the rerun (button in main window or "Apply to All & Rerun" in dialog)

---

## Non-Goals / Deferred

- **MS tab thumbnail:** A small read-only DBSCAN scatter embedded below the MS spectrum plot in the MS tab. Deferred to a future session. Would require caching per-window intermediates on `ChromaKitApp` after each full run. The "Inspect" button is the primary entry point for now.
- **Raw vs. deconvolved spectrum side-by-side panel** inside the inspector — deferred; the existing toggle in the MS tab covers this for now.
- **DeconvolutionParams tuning persisted per-file** — parameters are global (QSettings); file-specific param profiles deferred.

---

## Files Changed

| File | Change |
|------|--------|
| `logic/spectral_deconvolution.py` | Add `return_intermediates` param to `deconvolve()`; expose noise peaks from `_cluster_by_rt` |
| `deconvolution/spectral_deconvolution.py` | Same change (keep in sync) |
| `ui/app.py` | Fix `on_peak_spectrum_requested`; add `QProgressDialog` to deconvolution flow; add `_on_inspect_requested()`; add `_deconv_inspector` reference |
| `ui/frames/ms.py` | Add "Inspect" button; emit `inspect_requested` signal |
| `ui/dialogs/spectral_deconv_inspector.py` | New file — `SpectralDeconvInspectorDialog` |

---

## Dependencies

All available in `chromakit-env`: `matplotlib`, `numpy`, `scipy`, `scikit-learn`, `rainbow`, `PySide6`.
