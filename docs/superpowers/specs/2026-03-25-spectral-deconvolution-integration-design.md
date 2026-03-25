# Spectral Deconvolution Integration — Design Spec

**Date:** 2026-03-25
**Algorithm:** ADAP-GC 3.2 (already implemented in `deconvolution/spectral_deconvolution.py`)
**Scope:** Phase A — wire ADAP-GC into the app for better MS identification; 1:1 FID-to-MS mapping preserved
**End goal (Phase C):** Full chunk inspector (EIC trace viewer, DBSCAN cluster coloring, raw vs. clean spectrum browser) — not built in this phase, but all architecture decisions are made with it in mind

---

## Context

ChromaKit processes **GC-MS** data with two parallel signals:
- **FID** — used for quantitation. Peaks are detected and integrated here.
- **MS (TIC + EIC)** — used for compound identification. Currently, for each FID peak, a spectrum is extracted at the TIC argmax within the peak's RT window and searched against an MS library.

The existing extraction approach is a point-in-time snapshot, which can be corrupted by co-eluting compounds. ADAP-GC deconvolutes per-m/z ion chromatograms (EICs) within each peak region to reconstruct a clean fragmentation spectrum per compound. Phase A replaces the point-in-time extraction with this deconvolved spectrum.

**Note on terminology:** The codebase contains a pre-existing "deconvolution" feature (`logic/deconvolution.py`) based on a U-Net + EMG pipeline for **chromatographic peak splitting** on the FID signal. This is a different feature. As part of this work, all references to that feature are renamed to **"Peak Splitting (U-Net)"** throughout the codebase and UI to eliminate ambiguity. The new ADAP-GC feature is called **"MS Spectral Deconvolution"** everywhere.

---

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| A | ADAP-GC wired into app; deconvolved spectrum replaces point-in-time extraction; raw/deconvolved toggle in MS tab | **This spec** |
| B | Full FID/MS co-registration: MS component count informs FID peak splitting/merging decisions | Future |
| C | Chunk inspector: EIC trace viewer, DBSCAN cluster coloring, raw vs. clean spectrum side-by-side, `DeconvolutionParams` tuning | End goal |

---

## Architecture

### New and moved files (all in `logic/`)

| File | Status | Responsibility |
|------|--------|---------------|
| `logic/spectral_deconvolution.py` | **Move** from `deconvolution/` | ADAP-GC algorithm — `EICPeak`, `DeconvolutedComponent`, `DeconvolutionParams`, `deconvolve()`. Unchanged from standalone version. |
| `logic/eic_extractor.py` | **New** | Extracts per-m/z EIC traces from `data.ms` within an RT window and converts them to `list[EICPeak]` |
| `logic/spectral_deconv_runner.py` | **New** | Orchestrates the full pipeline: group FID peaks into windows → EIC extraction → `deconvolve()` → match components to FID peaks → update `ChromatographicPeak` objects. Also defines `WindowGroupingParams`. |
| `logic/spectral_deconv_worker.py` | **New** | `QRunnable` wrapping the runner; emits progress/completion signals |

`deconvolution/spectral_deconvolution.py` remains as the standalone reference copy with its own tests. It starts identical to the `logic/` version; divergence only if app-specific changes are needed.

### Modified files

| File | Change |
|------|--------|
| `logic/integration.py` (`ChromatographicPeak`) | Add `deconvolved_spectrum` and `deconvolution_component_count` fields |
| `logic/batch_search.py` | Use `peak.deconvolved_spectrum` if available; fall back to `SpectrumExtractor.extract_for_peak()` |
| `ui/frames/buttons.py` | Add "Deconvolve MS" button |
| `ui/frames/ms.py` | Add raw/deconvolved toggle when `peak.deconvolved_spectrum` is not None |
| `ui/dialogs/ms_options_dialog.py` | Add "Spectral Deconvolution" tab with `DeconvolutionParams` and window-grouping controls; rename extraction/subtraction tabs to clarify fallback role |
| `ui/app.py` | Dispatch `SpectralDeconvWorker`; rename U-Net methods (`_apply_deconvolution` → `_apply_peak_splitting`, etc.) |
| `ui/frames/parameters.py` | Rename "Deconvolution (U-Net)" combo item to "Peak Splitting (U-Net)"; rename `deconvolution` key in `current_params` to `peak_splitting`; rename all `deconv_*` attributes |
| `ui/frames/plot.py` | Rename `"Deconv"` plot legend label to `"Peak Split"` |

---

## Component Design

### 1. EIC Extractor (`logic/eic_extractor.py`)

**Function:** `extract_eic_peaks(ms, t_start, t_end, min_intensity) -> list[EICPeak]`

- `ms`: open rainbow `DataFile` object for `data.ms` (`.xlabels` = RT axis as 1D array, `.data` = 2D array [scans × m/z])
- Slice scan axis to `[t_start, t_end]` → `rt_window` (1D), `ms_slice` (2D)
- **m/z indexing:** use `float(j + 1)` as the m/z value for column `j` (1-based integer, consistent with `SpectrumExtractor` which uses `np.arange(len(spectrum)) + 1` — `ms.ylabels` is not used)
- For each m/z column `j`:
  - Skip if `max(ms_slice[:, j]) < min_intensity`
  - Run `scipy.signal.find_peaks` on the EIC trace
  - For each apex: get `left_boundary_idx`/`right_boundary_idx` via `scipy.signal.peak_widths` at `rel_height=0.5`. `peak_widths` returns float arrays — convert with `left_boundary_idx = int(np.floor(left_ip))`, `right_boundary_idx = int(np.ceil(right_ip))`, then clip to `[0, len(rt_window)-1]`
  - Construct `EICPeak(rt_apex=rt_window[apex_idx], mz=float(j+1), rt_array=rt_window, intensity_array=ms_slice[:,j].astype(float), left_boundary_idx=..., right_boundary_idx=..., apex_idx=apex_idx)`
- `rt_array` and `intensity_array` span the full window (not just the peak) so ADAP-GC has full chromatographic context for shape similarity and NNLS decomposition
- **m/z column convention:** column 0 = m/z 1.0, column 1 = m/z 2.0, … — matching the existing `SpectrumExtractor` convention (`np.arange(len(spectrum)) + 1`)

The function does **not** open the file — it receives an already-open `DataFile` object.

### 2. `WindowGroupingParams` dataclass (defined in `logic/spectral_deconv_runner.py`)

```python
@dataclass
class WindowGroupingParams:
    gap_tolerance: float | None = None  # None = 0.5× median FID peak width at runtime;
                                        # falls back to 0.3 min if fewer than 2 peaks
    padding_fraction: float = 0.5       # pad each side by this fraction of cluster width
    rt_match_tolerance: float = 0.05    # max RT distance (min) to assign a DeconvolutedComponent
                                        # to a FID peak; tight because FID and MS share the same
                                        # elution event through a GC splitter
```

### 3. Window grouping (inside `spectral_deconv_runner.py`)

Groups already-detected `ChromatographicPeak` objects into cluster windows. Reuses the chunker's gap-merge logic without re-running peak detection on the TIC.

1. Sort FID peaks by `retention_time`
2. Compute `gap_tolerance`: if `grouping_params.gap_tolerance is None`, use `0.5 × median(peak.end_time - peak.start_time for peak in peaks)`, with fallback of `0.3` min if fewer than 2 peaks
3. Merge adjacent peaks where gap between `peak_n.end_time` and `peak_{n+1}.start_time` < `gap_tolerance`
4. Each cluster window = `[cluster_start - padding, cluster_end + padding]` where `padding = grouping_params.padding_fraction × cluster_width`, clamped to the RT axis bounds of the MS data
5. Result: `list[(window_start, window_end, [fid_peaks_in_window])]`

### 4. Deconvolution runner (`logic/spectral_deconv_runner.py`)

**Function:** `run_spectral_deconvolution(peaks, ms_data_path, deconv_params, grouping_params) -> list[ChromatographicPeak]`

`ms_data_path` is the `.D` directory path string — the same value as `DataHandler.current_directory_path`. The runner opens it once before the window loop: `data_dir = rainbow.read(ms_data_path); ms = data_dir.get_file('data.ms')`. `data_dir` and `ms` go out of scope after the runner returns; rainbow data containers hold data in memory and have no explicit close method.

For each cluster window:
1. Call `extract_eic_peaks(ms, window_start, window_end, deconv_params.min_cluster_intensity)` → `list[EICPeak]`
2. If empty: skip — FID peaks in this window keep `deconvolved_spectrum = None`
3. Call `deconvolve(eic_peaks, deconv_params)` → `list[DeconvolutedComponent]`
4. Assignment of components to FID peaks is one-to-one. Algorithm: build all `(component, fid_peak)` pairs with their RT distance `|component.rt - peak.retention_time|`. Sort all pairs by RT distance ascending. Iterate in that order: when a pair is reached and neither the component nor the FID peak has been assigned yet, assign them. Mark both as used. Continue until all pairs are exhausted. If multiple components have identical RT distance to the same FID peak, break ties by picking the component with higher `sum(component.spectrum.values())`.
5. For each assigned (component, FID peak) pair where `|component.rt - peak.retention_time| <= grouping_params.rt_match_tolerance`: convert `component.spectrum` (`dict[float, float]`) to the same format as `SpectrumExtractor.extract_for_peak()` returns, then store:
   ```python
   mz_arr = np.array(sorted(component.spectrum.keys()))
   int_arr = np.array([component.spectrum[m] for m in mz_arr])
   peak.deconvolved_spectrum = {'mz': mz_arr, 'intensities': int_arr}
   peak.deconvolution_component_count = len(components_in_window)
   ```
6. For FID peaks that received no assignment within tolerance: set `peak.deconvolved_spectrum = None` and `peak.deconvolution_component_count = len(components_in_window)`. Setting the count even when no spectrum was assigned allows Phase C to display "X components found, none matched" vs. "deconvolution not run" (count is None).

Re-running `run_spectral_deconvolution` **unconditionally overwrites** all `peak.deconvolved_spectrum` values for peaks in processed windows. Previously run library search results are not invalidated automatically; stale-result indication is deferred to Phase C.

Returns the mutated peaks list.

### 5. Worker (`logic/spectral_deconv_worker.py`)

`QRunnable` following the same pattern as `BatchSearchWorker`:
- Signal `progress = Signal(int)`: emits 0–100 (percentage), calculated as `int(100 * windows_done / total_windows)` after each window
- Signal `finished = Signal()`: no payload — peaks are mutated in place; `ChromaKitApp` accesses them directly after the signal fires, matching the existing `BatchSearchWorker` pattern
- Signal `error = Signal(str)`: emits on unhandled exception
- Cancellation via `self.cancelled` flag checked between windows
- Dispatched from `ChromaKitApp` when "Deconvolve MS" is clicked; `app.py` passes `DataHandler.current_directory_path` as `ms_data_path`

### 6. `ChromatographicPeak` additions (`logic/integration.py`)

Two new fields initialized in `__init__`:
```python
self.deconvolved_spectrum = None
# When populated: {'mz': np.ndarray, 'intensities': np.ndarray}
# Same format as SpectrumExtractor.extract_for_peak() — directly usable in batch_search.py

self.deconvolution_component_count = None
# int: how many ADAP-GC components were found in this peak's window; None if not yet run
```

### 7. Library search integration (`logic/batch_search.py`)

When building the spectrum for a peak, replace the unconditional `SpectrumExtractor.extract_for_peak()` call with:

```python
if peak.deconvolved_spectrum is not None:
    spectrum = peak.deconvolved_spectrum  # {'mz': array, 'intensities': array}
    # background subtraction is skipped — ADAP-GC NNLS reconstruction is the denoising step
else:
    spectrum = spectrum_extractor.extract_for_peak(data_dir, peak, options)
    # background subtraction runs as before, inside extract_for_peak
```

`spectrum` in both branches is in the same `{'mz': array, 'intensities': array}` format. All downstream processing (weighting, query vector construction, library search) is unchanged.

---

## UI Changes

### "Deconvolve MS" button (`ui/frames/buttons.py`)

- Added alongside the existing "Search MS" button
- Disabled until FID peaks are integrated (same guard as "Search MS")
- Clicking dispatches `SpectralDeconvWorker` and shows a progress bar (0–100%)

### Raw/deconvolved toggle (`ui/frames/ms.py`)

- When a peak is selected and `peak.deconvolved_spectrum is not None`: display deconvolved spectrum by default
- A small toggle button ("Deconvolved / Raw") switches to the point-in-time TIC apex extraction for comparison: calls `SpectrumExtractor.extract_at_rt(data_directory, retention_time=peak.retention_time)`. `SpectrumExtractor.extract_at_rt` already exists in `logic/spectrum_extractor.py` with signature `extract_at_rt(self, data_directory: str, retention_time: float, intensity_threshold: float = 0.01)`. `data_directory` is the current `.D` directory path, received by `MSFrame` from `ChromaKitApp` via the existing signal/slot that fires when a file is loaded (same path used for all other spectrum extraction in `MSFrame`)
- This raw comparison is a live extraction, not stored state
- Toggle is only visible when a deconvolved spectrum is available; otherwise the MS tab behaves exactly as today

### MS Options dialog — new "Spectral Deconvolution" tab (`ui/dialogs/ms_options_dialog.py`)

New tab added via `_create_spectral_deconv_tab()`. Controls and their `QSettings` keys:

**Peak Window Grouping** (maps to `WindowGroupingParams`):

| Control | Field | QSettings key | Default |
|---------|-------|---------------|---------|
| Gap tolerance (min), 0 = auto | `gap_tolerance` | `ms_spectral_deconv/gap_tolerance` | `0.0` (0 = use auto heuristic) |
| Padding fraction | `padding_fraction` | `ms_spectral_deconv/padding_fraction` | `0.5` |
| RT match tolerance (min) | `rt_match_tolerance` | `ms_spectral_deconv/rt_match_tolerance` | `0.05` |

**ADAP-GC Parameters** (maps to `DeconvolutionParams`) — user-tunable:

| Control | `DeconvolutionParams` field | QSettings key | Default |
|---------|----------------------------|---------------|---------|
| Min cluster distance (min) | `min_cluster_distance` | `ms_spectral_deconv/min_cluster_distance` | `0.005` |
| Min cluster size | `min_cluster_size` | `ms_spectral_deconv/min_cluster_size` | `2` |
| Min cluster intensity (counts) | `min_cluster_intensity` | `ms_spectral_deconv/min_cluster_intensity` | `200.0` |
| Shape similarity threshold (°) | `shape_sim_threshold` | `ms_spectral_deconv/shape_sim_threshold` | `30.0` |
| Model peak choice | `model_peak_choice` | `ms_spectral_deconv/model_peak_choice` | `"sharpness"` |
| Excluded m/z (comma-separated) | `excluded_mz` | `ms_spectral_deconv/excluded_mz` | `""` (empty = no exclusions) |

**ADAP-GC Parameters — hard-coded in Phase A** (not user-tunable; use `DeconvolutionParams` defaults):

| Field | Default value | Rationale |
|-------|--------------|-----------|
| `use_is_shared` | `True` | Rarely needs changing |
| `edge_to_height_ratio` | `0.3` | Algorithm default; expose in Phase C |
| `delta_to_height_ratio` | `0.3` | Algorithm default; expose in Phase C |
| `min_model_peak_sharpness` | `10.0` | Algorithm default; expose in Phase C |
| `excluded_mz_tolerance` | `0.5` | Algorithm default; expose in Phase C |

All settings are loaded in `_load_settings()` and saved in `_save_settings()` following the existing dialog pattern. The "Restore Defaults" button resets all `ms_spectral_deconv/` keys.

Tab note displayed at top: *"Deconvolved spectra replace point-in-time extraction for library search. Run via 'Deconvolve MS' button after FID integration."*

### MS Options dialog — existing tabs

- "Spectrum Extraction" tab → renamed "Spectrum Extraction (Fallback)"
- "Background Subtraction" tab → renamed "Background Subtraction (Fallback)"
- Both remain fully functional as the fallback path used when `peak.deconvolved_spectrum is None`

---

## Cleanup / Rename

All references to the U-Net chromatographic peak splitting feature are renamed throughout. The `peaks.mode = 'deconvolution'` string and the `'deconvolution'` params dict key live in `ui/frames/parameters.py` (`current_params`) — not in `logic/integration.py`.

| Old name | New name | Location |
|----------|----------|----------|
| `peaks.mode = 'deconvolution'` | `peaks.mode = 'peak_splitting'` | `parameters.py` `current_params` |
| params key `'deconvolution'` | `'peak_splitting'` | `parameters.py` `current_params` |
| `_apply_deconvolution()` | `_apply_peak_splitting()` | `app.py` |
| `_integrate_deconvolution()` | `_integrate_peak_splitting()` | `app.py` |
| `_deconv_cache` | `_peak_splitting_cache` | `app.py` |
| `"Deconvolution (U-Net)"` combo item | `"Peak Splitting (U-Net)"` | `parameters.py` |
| `deconv_controls_frame` | `peak_splitting_controls_frame` | `parameters.py` |
| `deconv_advanced_frame` | `peak_splitting_advanced_frame` | `parameters.py` |
| `deconv_advanced_toggle` | `peak_splitting_advanced_toggle` | `parameters.py` |
| `deconv_reset_btn` | `peak_splitting_reset_btn` | `parameters.py` |
| `_on_deconv_param_changed` | `_on_peak_splitting_param_changed` | `parameters.py` |
| `_on_pre_fit_signal_changed` | `_on_peak_splitting_pre_fit_signal_changed` | `parameters.py` |
| `_add_deconv_window_row` | `_add_peak_splitting_window_row` | `parameters.py` |
| `_remove_deconv_window_row` | `_remove_peak_splitting_window_row` | `parameters.py` |
| `_on_deconv_windows_changed` | `_on_peak_splitting_windows_changed` | `parameters.py` |
| `_toggle_deconv_advanced` | `_toggle_peak_splitting_advanced` | `parameters.py` |
| `_update_deconv_method_visibility` | `_update_peak_splitting_method_visibility` | `parameters.py` |
| `_reset_deconv_defaults` | `_reset_peak_splitting_defaults` | `parameters.py` |
| `_sync_deconv_spinboxes` | `_sync_peak_splitting_spinboxes` | `parameters.py` |
| `"Deconv"` plot legend label | `"Peak Split"` | `plot.py` |

`logic/deconvolution.py` and `logic/chunker.py` are **not** renamed or removed — they implement the peak splitting pipeline and are still called by the renamed methods in `app.py`. Only surface-facing identifiers change.

---

## Constraints and Non-Goals (Phase A)

- No FID/MS co-registration — FID peaks are not split or merged based on MS component count (Phase B)
- No chunk inspector UI — deferred to Phase C (end goal)
- No stale-result indicator when deconvolution is re-run after library search — deferred to Phase C
- `deconvolution/spectral_deconvolution.py` standalone copy and its tests are not modified
- No changes to the API layer (`api/`) — deconvolution is GUI-only for now
- No assumption of TMS derivatization in defaults — `excluded_mz` defaults to empty

---

## Dependencies

All available in `chromakit-env`:
- `numpy`, `scipy`, `scikit-learn` (already used by `spectral_deconvolution.py`)
- `rainbow` (already used by `DataHandler` and `SpectrumExtractor`)
