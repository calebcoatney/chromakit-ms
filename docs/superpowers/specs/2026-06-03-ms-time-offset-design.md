# MS Time Offset — Design

**Date:** 2026-06-03
**Status:** Draft, pending implementation plan

## 1. Problem

In ChromaKit-MS, FID peak retention times and MS-derived retention times (from deconvolved components, EIC apices, raw TIC) can disagree by ~0.03–0.07 min due to small clock/timing skew between detectors. When the user runs spectral deconvolution and inspects the result, the dashed FID peak lines visibly miss the clustered m/z components by ~0.05 min — exactly the magnitude of the default greedy nearest-RT match tolerance (`WindowGroupingParams.rt_match_tolerance = 0.05 min` in `logic/spectral_deconv_runner.py:23`). For tightly spaced peaks this produces ambiguous or wrong FID↔MS assignments.

Existing alignment infrastructure is dead code: a `ChromatogramProcessor.align_tic_to_fid` cross-correlation function (`logic/processor.py:657`), a `Method.align_tic` flag (`logic/method.py:55`), a checkbox in `ui/frames/parameters.py:440`, a `PlotFrame.set_aligned_tic_data` receiver, an `aligned_tic_data` kwarg in `DataHandler.extract_spectrum_at_rt`, and a `POST /api/align-tic` endpoint. None of them connect to the processing pipeline; commit `03834f7` removed the React UI control as "non-functional".

## 2. Goal

A user-controlled constant time shift applied to the MS data so that MS-derived retention times align with FID-derived retention times throughout the application. The shift is:

- **Set interactively** in the existing Spectral Deconvolution Inspector with live preview.
- **Applied globally** so every MS consumer (TIC plot, deconvolution, EIC, spectrum extraction) sees a consistent time axis.
- **Persisted per file** in a sidecar so reopening a file restores the offset.
- **Logged** in exported results JSON metadata for traceability.

Out of scope: non-linear / piecewise / DTW alignment, per-peak nudge overrides, FID-side shifts, method-level (sequence-wide) defaults, REST API surface, auto-cascading to MS library search and quantitation reruns.

## 3. Architecture

### 3.1 Single source of truth

Add `DataHandler.ms_time_offset: float` (minutes; defaults to `0.0`). Every consumer that currently reads `ms.xlabels` reads through a helper that adds this offset.

### 3.2 Helper

New module `logic/ms_time.py`:

```python
def shifted_xlabels(ms_data, offset_min: float) -> np.ndarray:
    """Return ms_data.xlabels shifted by offset_min (in minutes)."""
    return ms_data.xlabels + offset_min
```

Every consumer calls `shifted_xlabels(ms, handler.ms_time_offset)` instead of `ms.xlabels` directly. Grep-auditable; a single line to change if the offset model ever becomes more complex.

### 3.3 Touch points

The following currently read `ms.xlabels` directly and must be migrated to `shifted_xlabels`:

- `logic/data_handler.py::_get_tic_data` (around line 127) — TIC time vector for the main plot.
- `logic/spectrum_extractor.py` (lines 46, 49, 180–419) — per-peak MS spectrum extraction.
- `logic/eic_extractor.py` (lines 39–47) — EIC time axis for deconvolution input.
- `logic/spectral_deconv_runner.py` (lines 188–189) — window-bound resolution and downstream component RTs.
- `ui/dialogs/spectral_deconv_inspector.py` (lines 375–380) — the inspector independently re-opens MS; reuse the same helper, parameterized by the inspector's current preview offset (see §4).

`ChromatogramProcessor.align_tic_to_fid` (`logic/processor.py:657`) is **kept** as the backend for the inspector's "Auto" button. It is no longer applied as a transformation; it only suggests an offset value.

## 4. UX — Spectral Deconvolution Inspector

A new "MS Time Offset" group is added at the top of the existing parameters panel in `SpectralDeconvInspectorDialog` (`ui/dialogs/spectral_deconv_inspector.py:95`):

```
┌─ MS Time Offset ────────────────────────────────────┐
│  [───────────●────────────]   −0.0480 min (−2.88 s) │
│  [Auto]  [Reset]  [Apply globally]                  │
└─────────────────────────────────────────────────────┘
```

- **Slider:** integer steps over the range ±500, mapped to ±0.500 min in 0.001 min increments (1 ms resolution, ample for GC). Live readout in both minutes (4 decimal places) and seconds (2 decimal places).
- **Auto button:** runs `ChromatogramProcessor.align_tic_to_fid` on the current file's FID and TIC, sets the slider to the returned lag. Records the source as `"auto"`.
- **Reset button:** sets slider to 0.
- **Live preview:** moving the slider re-renders the inspector's scatter and EIC subplots with shifted MS data while keeping the FID dashed lines fixed. Cheap because the deconvolution result is cached on the dialog; only the x-axis vector for plotted MS-side items needs to be recomputed.
- **Apply globally:** writes the current slider value to `DataHandler.ms_time_offset`, writes the sidecar (§5.1), triggers a `SpectralDeconvWorker` rerun via the existing `_on_inspector_rerun_requested` path (`ui/app.py:3147`) so `peak.deconvolved_spectrum` and `peak.deconvolution_component_count` reflect the new matches, then closes the inspector. A status bar message reports the applied value and warns that downstream library search results may be stale.

## 5. Persistence

### 5.1 Per-file sidecar

New file `overrides/ms_time_offsets.json` (parallel to existing `overrides/manual_assignments.json` convention but keyed by absolute `.D` path):

```json
{
  "/abs/path/to/sample01.D": {
    "offset_min": -0.048,
    "timestamp": 1735000000.0,
    "source": "manual"
  },
  "/abs/path/to/sample02.D": {
    "offset_min": -0.052,
    "timestamp": 1735000100.0,
    "source": "auto"
  }
}
```

- `source: "manual" | "auto"` for traceability (was the slider dialed in by hand, or seeded from cross-correlation?).
- Loaded by `DataHandler` on file open; if a matching entry exists, `ms_time_offset` is set accordingly and the visual indicator (§7) is shown.
- Written on every "Apply globally". An entry with `offset_min: 0.0` is permitted and means "explicitly no shift" (suppresses any future auto-suggest UI nudge, if added later).

Trade-off accepted: absolute-path keying breaks if files are moved between machines. The alternative — a sidecar file inside each `.D` directory — pollutes the raw data directory and conflicts with the existing `overrides/`-centric convention. The central file is preferred for consistency with `manual_assignments.json`.

### 5.2 Results JSON metadata

`logic/json_exporter.py` writes two new fields into the metadata block:

- `ms_time_offset: float` (minutes; `0.0` if none)
- `ms_time_offset_source: "manual" | "auto" | null`

These replace the existing `bl_info['align_tic']` boolean, which is deleted.

## 6. Batch automation

`AutomationWorker` reads each `.D`'s sidecar entry as part of its existing file-open path. If present, the offset is applied silently and recorded in the per-file results JSON metadata. If absent, no shift is applied (same behavior as today). No auto-compute happens in batch — users opt in by setting offsets in the inspector first. A future "batch auto-align" feature is out of scope.

## 7. Visual indicator

When `DataHandler.ms_time_offset != 0`:

- **Status bar** on file load: *"MS offset active: −0.048 min"*.
- **Plot legend** entry for the TIC trace appends the offset, e.g. `"TIC (offset −0.048 min)"`. This puts the indicator directly next to the trace it affects.

No additional widgets are introduced.

## 8. Dead code removal

The following are deleted as part of this change:

- `align_tic` field in `logic/method.py:55`.
- `align_tic` checkbox and `_on_align_tic_toggled` handler in `ui/frames/parameters.py` (lines 47, 440–449, 1570–1586).
- `PlotFrame.aligned_tic_data`, `PlotFrame.tic_alignment_info`, and `PlotFrame.set_aligned_tic_data` in `ui/frames/plot.py` (lines 71–72, 978–982).
- `aligned_tic_data` kwarg in `DataHandler.extract_spectrum_at_rt` (`logic/data_handler.py:265`).
- `bl_info['align_tic']` write in `logic/json_exporter.py:134`.
- `AlignTICRequest`, `AlignTICResponse`, and `POST /api/align-tic` in `api/models.py:103–108, 190–194` and `api/main.py:248–266`.
- Corresponding test fixture entry in `tests/logic/test_method.py:28`.

`ChromatogramProcessor.align_tic_to_fid` is **kept** — it becomes the "Auto" backend.

## 9. File-by-file change summary

| File | Change |
|---|---|
| `logic/ms_time.py` | **NEW** — `shifted_xlabels(ms_data, offset_min)` helper. |
| `logic/data_handler.py` | Add `ms_time_offset: float = 0.0`. Load from sidecar on file open. Use `shifted_xlabels` in `_get_tic_data`. Remove dead `aligned_tic_data` kwarg from `extract_spectrum_at_rt`. |
| `logic/spectrum_extractor.py` | Replace `ms.xlabels` reads with `shifted_xlabels(ms, handler.ms_time_offset)`. |
| `logic/eic_extractor.py` | Same. |
| `logic/spectral_deconv_runner.py` | Same. |
| `logic/method.py` | Delete `align_tic` field. |
| `logic/json_exporter.py` | Delete `bl_info['align_tic']`. Add `ms_time_offset` and `ms_time_offset_source` to metadata. |
| `logic/sidecar_offsets.py` | **NEW** — read/write helpers for `overrides/ms_time_offsets.json`. |
| `ui/dialogs/spectral_deconv_inspector.py` | Add offset group (slider, Auto, Reset, Apply globally). Live preview replot. Wire "Apply globally" to write sidecar, set `DataHandler.ms_time_offset`, trigger rerun. Use `shifted_xlabels` in the dialog's own MS re-read. |
| `ui/frames/parameters.py` | Delete align-TIC checkbox and handler. |
| `ui/frames/plot.py` | Delete `aligned_tic_data`/`set_aligned_tic_data`. Append offset to TIC legend label when nonzero. |
| `ui/app.py` | Show status bar message on file load when sidecar offset is present. Wire inspector "Apply globally" handler. |
| `api/models.py` | Delete `AlignTICRequest`, `AlignTICResponse`. |
| `api/main.py` | Delete `POST /api/align-tic`. |
| `tests/logic/test_method.py` | Remove `align_tic` from fixture. |
| `tests/logic/test_ms_time_offset.py` | **NEW** — unit tests for `shifted_xlabels`, sidecar round-trip, `DataHandler.ms_time_offset` integration. |

## 10. Open questions

None at design time. Implementation may surface small wiring questions (e.g. exact placement of the inspector's offset group within its existing splitter), to be resolved in the plan or during implementation.
