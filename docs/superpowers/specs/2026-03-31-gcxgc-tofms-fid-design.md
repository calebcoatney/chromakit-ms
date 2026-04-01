# GCxGC-TOFMS/FID Support Design

**Date:** 2026-03-31  
**Instrument:** Agilent 8860/8890 GC + SepSolve INSIGHT Modulator + Bench TOFMS  
**Approach:** Minimal parallel path — new loader, processor, and peak model; UI adapts via `ui_mode="gcxgc"`

---

## Background

GCxGC adds a second separation dimension via a flow modulator between two columns. The result is a 2D chromatogram: first-dimension RT (1tR, minutes, volatility) × second-dimension RT (2tR, seconds, polarity). The instrument produces two detectors in parallel:

- **FID** (`.rsd`) — quantitative signal; integration is a volume (sum of 2D peak intensities)
- **TOFMS** (`.dbc.lsc`) — identification via MS library matching; already DBC-processed (baseline-corrected, deconvolved) by the instrument

Raw instrument files: `.rsd`, `.dbc.lsc`, `.HDR`, `.DAT`, `.DAX`, `.lsc`  
Only `.rsd` and `.dbc.lsc` are needed for processing. `.HDR` is needed for acquisition parameters.

---

## Data Parameters

From the `.HDR` file (INI-style text):
- `Data Rate` — acquisition rate in Hz (e.g. `50`)
- `cal modulation pulse` — modulation period PM in seconds (e.g. `0.5`)

These determine the 2D array shape:
```
scans_per_mod = int(PM * Hz)
n_mods = len(signal_1d) // scans_per_mod
fid_2d.shape = (n_mods, scans_per_mod)   # e.g. (19514, 25)
tic_2d.shape = (n_mods, scans_per_mod)
```

The `.HDR` parser should fall back to raising a clear error if either field is missing, prompting the user to verify the file. No silent defaults — wrong PM produces a meaningless 2D reshape.

---

## Section 1: Signal Profile

Register a `gcxgc` profile in `logic/signal_profiles.py`:

```python
SignalProfileRegistry.register(SignalProfile(
    name="gcxgc",
    display_name="GCxGC-TOFMS/FID",
    feature_class=None,           # set by _update_gcxgc_profile() after GCxGC2DPeak is defined
    loader_class=SepSolveLoader,
    x_label="1st Dimension RT (min)",
    y_label="2nd Dimension RT (s)",
    pipeline_stages=[
        PipelineStage.PEAKS,
        PipelineStage.MS_SEARCH,
        PipelineStage.QUANTITATION,
    ],
    ui_mode="gcxgc",              # new mode; triggers heatmap rendering and parameter visibility
    default_params={},
))
```

No new `PipelineStage` variants needed. `_update_gcxgc_profile(GCxGC2DPeak)` follows the same pattern as `_update_chromatographic_profiles`.

---

## Section 2: Data Loader (`logic/loaders/sepsolve_loader.py`)

`SepSolveLoader(DataLoader)` receives the `.C` folder path and finds source files inside `data/` by extension.

**HDR parsing:** Read as INI-style text, extract `Data Rate` (Hz) and `cal modulation pulse` (PM seconds).

**FID parsing (`.rsd`):** Block-header structure: 12-byte header (`u32, u32, count`) followed by `count` float64 values. Blocks are walked sequentially; `count > 2000` triggers a seek-forward scan for the next valid header. Result: 1D float64 array → reshape to `(n_mods, scans_per_mod)`.

**TIC parsing (`.dbc.lsc`):** SD04 block structure, same as `.lsc`. Seek to offset 12568, read `SD04` tags. Each block yields one TIC value (sum of `peak_count` intensities). Result: 1D int array → reshape to `(n_mods, scans_per_mod)`.

**Spectrum extraction:** Given a scan index, seek to that SD04 block in the `.dbc.lsc` and decode `(m/z × 1000 → float, intensity as float64)` pairs. Unit-mass bin (round m/z, sum intensities per bin). Returns `list[tuple[int, float]]` — the format `ms_toolkit.search_w2v` already accepts.

**Return dict:**
```python
{
    "x": np.ndarray,   # 1st-dim RT axis in minutes, length = n_mods
    "y": np.ndarray,   # FID 1D (for any fallback compatibility use)
    "metadata": {
        "is_gcxgc": True,
        "has_ms_data": True,
        "fid_2d": np.ndarray,   # (n_mods, scans_per_mod)
        "tic_2d": np.ndarray,   # (n_mods, scans_per_mod)
        "pm": float,             # modulation period in seconds
        "hz": float,             # acquisition rate in Hz
        "lsc_path": str,         # absolute path to .dbc.lsc for spectrum extraction
        "filename": str,
    }
}
```

**`CFolder` changes:**  
`CFolder.create_multi(source_paths, signal_type, **metadata)` — moves a list of files into `data/`, sets `source_format="sepsolve"` in the manifest. `CFolder.load_signal()` uses the existing generic fallback branch (`loader = profile.loader_class(); loader.load(self.path)`) — no changes needed there.  
`CFolder.extract()` is **not supported** for `source_format="sepsolve"` (multi-file sources); it raises `NotImplementedError` with a clear message if called on such a folder.

---

## Section 3: Data Model (`logic/gcxgc_peak.py`)

```python
class GCxGC2DPeak(Feature):
    """A compound peak in a GCxGC chromatogram.

    rt1: 1st dimension retention time (minutes)
    rt2: 2nd dimension retention time (seconds) — apex within modulation
    volume: sum of FID intensities across all contributing modulation slices
            mapped to Feature.area for QuantitationCalculator compatibility
    """
```

Key fields:
- `rt1` (float, min) — 1st dim RT; also `Feature.position`
- `rt2` (float, s) — 2nd dim RT apex
- `volume` (float) — 2D integrated intensity; identical to `Feature.area`
- `n_sub_peaks` (int) — number of modulation slices contributing
- `mod_start`, `mod_end` (int) — modulation index range
- MS: `compound_name`, `match_score`, `casno`
- Quantitation: `mol_C`, `mol_C_percent`, `num_carbons`, `mol`, `mass_mg`, `mol_percent`, `wt_percent`

`Feature.area = volume` ensures `QuantitationCalculator` works without modification.  
`as_dict()` is inherited and includes all instance attributes — `rt2`, `volume`, `n_sub_peaks` appear in JSON/CSV exports automatically.

---

## Section 4: Processing Pipeline (`logic/gcxgc_processor.py`)

`GCxGCProcessor` takes the loader metadata dict and produces `list[GCxGC2DPeak]`.

### Step 1 — Column-by-column peak detection
```python
for i in range(n_mods):
    peaks, props = find_peaks(fid_2d[i],
                              height=min_height,
                              prominence=min_prominence,
                              distance=min_distance_scans)
    sub_peaks[i] = list(zip(peaks, fid_2d[i][peaks]))
```

### Step 2 — Sub-peak grouping
Sweep adjacent modulations. Two sub-peaks in neighboring columns belong to the same compound if their scan index (2tR position) differs by ≤ `rt2_grouping_tolerance` scans (default: 2). Chain these forward-only into groups. Discard groups with fewer than `min_sub_peaks` modulations (default: 2) — eliminates single-column noise spikes.

### Step 3 — Peak construction
For each group:
- `rt1` = intensity-weighted mean modulation index × PM / 60
- `rt2` = scan index of highest sub-peak × (1 / Hz)
- `volume` = sum of per-slice areas across all modulations. Per-slice area: `np.trapz(fid_2d[i][left_base:right_base+1])` where `left_base`/`right_base` come from `find_peaks(..., width=True)` properties. Summed across all modulations in the group.
- `n_sub_peaks` = number of modulations in group

### Step 4 — MS spectrum extraction (deferred, on demand)
`GCxGCProcessor.extract_spectrum(peak) -> list[tuple[int, float]]`  
Computes the absolute scan index in the `.dbc.lsc` from `(apex_mod_index × scans_per_mod) + apex_scan_within_mod`, reads the SD04 block at that position, decodes and bins the spectrum. The result feeds directly into `BatchSearchWorker` — no changes to the MS search pipeline.

### Parameters exposed to UI
| Parameter | Default | Description |
|---|---|---|
| `min_height` | — | Absolute FID threshold |
| `min_prominence` | — | scipy prominence threshold |
| `rt2_grouping_tolerance` | 2 scans | Max 2tR shift between adjacent sub-peaks |
| `min_sub_peaks` | 2 | Min modulations per compound |

---

## Section 5: UI Adaptations

`ChromaKitApp` sets `self._gcxgc_mode: bool` from `metadata['is_gcxgc']` after loading. All UI branching derives from this flag.

### `PlotFrame`
New method `render_gcxgc(fid_2d, tic_2d, pm, hz)` alongside the existing line-plot path:
- Two vertically stacked axes: TIC heatmap (top), FID heatmap (bottom)
- `imshow` with log normalization (floor at 10th percentile), viridis colormap
- X-axis: 1st dim RT (min), Y-axis: 2nd dim RT (s)
- Click handler maps pixel → nearest `GCxGC2DPeak` by (rt1, rt2), highlights it
- Detected peaks overlaid as scatter markers on both axes at their (rt1, rt2)

### `ParametersFrame`
When `ui_mode == "gcxgc"`: hide smoothing and baseline correction widgets. Show four GCxGC controls: min height, min prominence, 2tR grouping tolerance (scans), min sub-peaks per compound.

### `RTTableFrame`
When GCxGC mode: add `rt2` column ("2D RT (s)"), rename area column to "Volume". All other columns (compound name, match score, CAS#, quantitation) unchanged.

### `MSFrame`
Unchanged. Spectrum extraction produces the same `list[tuple[int, float]]` format that `BatchSearchWorker` and `ms_toolkit` already consume.

### `BatchConvertDialog`
New row type `"sepsolve"`: detect stems where a directory contains both `<stem>.rsd` and `<stem>.dbc.lsc`. Group all associated files (`*.rsd`, `*.dbc.lsc`, `*.HDR`, `*.DAT`, `*.DAX`, `*.lsc`) by stem and propose a `.C` conversion using `CFolder.create_multi()` with `signal_type="gcxgc"`.

---

## File Changes Summary

| File | Change |
|---|---|
| `logic/loaders/sepsolve_loader.py` | New — SepSolveLoader |
| `logic/gcxgc_peak.py` | New — GCxGC2DPeak |
| `logic/gcxgc_processor.py` | New — GCxGCProcessor |
| `logic/signal_profiles.py` | Add gcxgc profile + _update_gcxgc_profile() |
| `logic/c_folder.py` | Add create_multi() |
| `ui/frames/plot.py` | Add render_gcxgc() + gcxgc click handler |
| `ui/frames/parameters.py` | Add gcxgc parameter visibility branch |
| `ui/frames/rt_table.py` | Add rt2 column + volume label in gcxgc mode |
| `ui/app.py` | Detect is_gcxgc, set mode flag, route to GCxGCProcessor |
| `ui/dialogs/batch_convert_dialog.py` | Add sepsolve stem detection + create_multi path |
