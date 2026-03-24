# Signal Generalization Design

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Extend ChromaKit-MS to support multiple scientific signal types (GC, GC-MS, FTIR, UV-Vis) through a shared processing engine, a `.C` container format, and a `SignalProfile` registry.

---

## Background

ChromaKit-MS currently processes GC and GC-MS data from Agilent `.D` directories. The core processing pipeline (smoothing → baseline correction → peak detection → integration → export) is signal-agnostic at the algorithmic level, but the codebase is tightly coupled to Agilent file formats, GC/MS terminology, and GC-MS-specific UI frames.

The goal is to generalize the system to support FTIR and UV-Vis signals — starting with file-based data — without duplicating the processing engine. Real-time/streaming data acquisition is explicitly out of scope for this design.

---

## Design Goals

1. Support loading signals from multiple source formats (Agilent `.D`, CSV) through a unified container format.
2. Share the existing processing pipeline across all signal types with minimal changes.
3. Generalize the data model so feature objects are typed appropriately for their domain.
4. Allow the desktop UI to adapt its layout and labeling based on signal type.
5. Provide a non-destructive migration path for existing Agilent `.D` workflows.

---

## 1. `.C` Folder Format

ChromaKit manages `.C` directories as its native container format. Each `.C` folder represents one sample (a single signal acquisition) and contains the raw data source, processing results, and a manifest declaring the signal type.

### Layout

```
MySample.C/
  manifest.json          ← signal type, source format, instrument metadata
  data/
    MySample.D/      ← Agilent: original .D folder nested inside
    — or —
    spectrum.csv         ← FTIR/UV-Vis: raw signal file
  results/
    features.json        ← feature list + full processing metadata
    features.csv         ← flat table export
```

### `manifest.json` schema

```json
{
  "signal_type": "ftir",
  "source_format": "csv",
  "created": "2026-03-24T10:00:00",
  "instrument": "Mettler Toledo ReactIR",
  "sample_id": "RXN-042",
  "csv_columns": {
    "x_column": "wavenumber",
    "y_column": "absorbance"
  }
}
```

The `signal_type` field is the key dispatch value — it maps to a registered `SignalProfile` that controls how the folder is loaded, processed, and displayed.

### Migration from `.D` folders

A `CFolderMigrationDialog` is presented when the user opens a directory containing `.D` folders with no corresponding `.C` wrappers. It lists detected `.D` folders, the user confirms, and `CFolder.create()` wraps each one:

- A new `MySample.C/` folder is created alongside `MySample.D/`
- `MySample.D/` is **copied** (not moved) into `MySample.C/data/` — the original `.D` folder remains at its original path until the user explicitly chooses to clean up
- `manifest.json` is written with `signal_type: "gcms"` or `"gc"` (user-selectable per-folder in the dialog)
- The dialog displays the source path and destination path for each folder before committing
- The dialog warns that external tools or scripts referencing the original `.D` path will need to be updated if the user later removes the originals
- Migration is atomic per-folder: if an error occurs mid-copy, the partially created `.C` folder is removed and the original `.D` is untouched

---

## 2. `Feature` Class Hierarchy

The current `Peak` class accumulates GC, MS, and quantitation fields in a single flat object. This design replaces it with a typed hierarchy.

### Base class: `Feature` (`logic/feature.py`)

```python
@dataclass
class Feature:
    feature_id: int
    position: float          # x-axis value (RT in min, wavenumber in cm⁻¹, wavelength in nm)
    position_units: str      # derived from SignalProfile, used in exports
    area: float
    width: float
    start: float
    end: float
    start_index: int
    end_index: int
    is_shoulder: bool = False
    is_negative: bool = False
    quality_issues: list[str] = field(default_factory=list)
```

### `ChromatographicPeak(Feature)` (`logic/integration.py`)

Carries all current `Peak` fields using their **existing names** — `retention_time`, `start_time`, `end_time`, `compound_id`, `peak_number`, `asymmetry`, `is_saturated`, `is_grouped`, `is_convoluted`, `spectral_coherence`, `integrator` — plus the optional MS match fields (`compound_name`, `match_score`, `casno`) and Polyarc quantitation fields (`mol_C`, `mol_C_percent`, `num_carbons`, `mol`, `mass_mg`, `mol_percent`, `wt_percent`).

Field names are **not renamed** on `ChromatographicPeak`. Instead, `position` on the base `Feature` class is a computed property that `ChromatographicPeak` satisfies by returning `self.retention_time`. This means generic code that reads `feature.position` works across all feature types, while all existing call sites that read `peak.retention_time` continue to work unchanged.

`as_row()` and `as_dict()` are serialization methods that live on `ChromatographicPeak` (not the base class), since their field names and column headers are domain-specific. `SpectralFeature` defines its own `as_row()`/`as_dict()` methods.

MS and quantitation fields remain on `ChromatographicPeak` (not a further subclass) because GC and GC-MS share the same peak object — the MS fields are simply unpopulated for GC-only runs, which is already the current behavior.

```python
Peak = ChromatographicPeak  # backward-compatibility alias — no call sites need to change
```

### `SpectralFeature(Feature)` (`logic/feature.py`)

```python
@dataclass
class SpectralFeature(Feature):
    band_assignment: str = ""    # functional group label or compound name
    absorbance: float = 0.0
    transmittance: float = 0.0
```

---

## 3. `SignalProfile` Registry

### `SignalProfile` dataclass (`logic/signal_profiles.py`)

```python
class PipelineStage(str, Enum):
    SMOOTHING     = "smoothing"
    BASELINE      = "baseline"
    PEAKS         = "peaks"
    MS_SEARCH     = "ms_search"
    QUANTITATION  = "quantitation"

@dataclass
class SignalProfile:
    name: str                          # registry key: "gcms", "gc", "ftir", "uvvis"
    display_name: str                  # human label: "GC-MS", "GC", "FTIR", "UV-Vis"
    feature_class: type                # ChromatographicPeak or SpectralFeature
    loader_class: type                 # AgilentLoader or CSVLoader
    x_label: str                       # axis label for PlotFrame and exports
    y_label: str                       # axis label for PlotFrame
    pipeline_stages: list[PipelineStage]  # ordered list of active stages
    ui_mode: str                       # "chromatography" or "spectroscopy"
    default_params: dict               # default processing parameters
```

`SignalProfileRegistry.register()` validates that all `pipeline_stages` values are valid `PipelineStage` enum members, raising `ValueError` on registration if not.

### Built-in profiles (registered at module import)

| Profile | `pipeline_stages` | `ui_mode` |
|---------|-------------------|-----------|
| `gcms`  | smoothing, baseline, peaks, ms_search, quantitation | chromatography |
| `gc`    | smoothing, baseline, peaks, quantitation | chromatography |
| `ftir`  | smoothing, baseline, peaks | spectroscopy |
| `uvvis` | smoothing, baseline, peaks | spectroscopy |

### `SignalProfileRegistry`

```python
class SignalProfileRegistry:
    _profiles: dict[str, SignalProfile] = {}

    @classmethod
    def register(cls, profile: SignalProfile): ...

    @classmethod
    def get(cls, name: str) -> SignalProfile: ...

    @classmethod
    def list_profiles(cls) -> list[str]: ...
```

---

## 4. Data Loaders

New module `logic/loaders/` with an abstract base and two concrete implementations.

### `DataLoader` ABC (`logic/loaders/base.py`)

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self, c_folder_path: str) -> dict:
        # Returns: {'x': ndarray, 'y': ndarray, 'metadata': dict}
        pass
```

### `AgilentLoader` (`logic/loaders/agilent_loader.py`)

Wraps existing `DataHandler` logic. All rainbow/detector-detection behavior is preserved, just relocated behind the `DataLoader` interface. `DataHandler` continues to exist internally.

`AgilentLoader.load()` returns the flat `{'x', 'y', 'metadata'}` dict specified by the ABC, with `metadata` carrying Agilent-specific keys that downstream pipeline stages consume when available:

```python
{
    'x': ndarray,             # retention time (minutes)
    'y': ndarray,             # detector signal
    'metadata': {
        'has_ms_data': bool,
        'tic_x': ndarray,     # TIC time axis (None if no MS data)
        'tic_y': ndarray,     # TIC intensity (None if no MS data)
        'detector': str,      # e.g. "FID1A"
        'sample_id': str,
        'filename': str,
    }
}
```

MS-specific pipeline stages (ms_search, quantitation) read from `metadata['has_ms_data']` before attempting to access TIC or spectrum data. If `has_ms_data` is False or the key is absent, those stages are skipped regardless of `profile.pipeline_stages`.

### `CSVLoader` (`logic/loaders/csv_loader.py`)

Reads `data/*.csv`. Column mapping is read from `manifest.json` (`x_column`, `y_column`). Returns the same `{'x': ndarray, 'y': ndarray, 'metadata': dict}` dict. `metadata` contains `{'filename': str, 'has_ms_data': False}`.

**Guaranteed metadata keys across all loaders:** `has_ms_data` (bool) and `filename` (str). All other keys are loader-specific and must be accessed with `.get()` to avoid `KeyError` in generic pipeline code.

---

## 5. `CFolder` Manager

New module `logic/c_folder.py`. Manages the on-disk `.C` structure and serves as the entry point for all data loading.

```python
class CFolder:
    @staticmethod
    def create(source_path: str, signal_type: str, **metadata) -> 'CFolder':
        # Creates .C directory, copies source into data/, writes manifest.json
        # signal_type is required; additional metadata keys (instrument, sample_id, etc.)
        # map directly to manifest.json fields (see Section 1 manifest schema)

    @staticmethod
    def open(path: str) -> 'CFolder':
        # Reads existing .C directory, validates manifest

    @property
    def profile(self) -> SignalProfile:
        # Reads signal_type from manifest, returns SignalProfileRegistry.get(signal_type)

    def load_signal(self) -> dict:
        # Delegates to self.profile.loader_class().load(self.path)

    def save_results(self, features: list[Feature], processing_metadata: dict):
        # Writes results/features.json and results/features.csv

    def get_manifest(self) -> dict: ...
```

The app loading flow becomes:

```
CFolder.open(path)
  → folder.profile          (reads manifest → registry lookup)
  → folder.load_signal()    (profile.loader_class → {'x', 'y', 'metadata'})
  → ChromatogramProcessor.process(x, y, params, profile)
  → folder.save_results(features, metadata)
```

---

## 6. Processing Pipeline Adaptations

`ChromatogramProcessor` requires minimal changes:

1. **Stage gating**: `process()` accepts a `profile` argument and skips stages not in `profile.pipeline_stages`. MS search and quantitation are already separate post-process calls — they simply aren't invoked.

2. **Default parameters**: `profile.default_params` populates the Parameters panel on load, replacing the current hardcoded defaults.

3. **Feature instantiation**: the integrator calls `profile.feature_class(...)` instead of `Peak(...)` directly — one-line change in `logic/integration.py`.

No new processor class is needed.

---

## 7. UI Mode System

`ChromaKitApp` gains `set_mode(ui_mode: str)` called after `CFolder.open()`:

### Frame visibility by mode

| Frame | `chromatography` | `spectroscopy` |
|-------|-----------------|----------------|
| `FileTreeFrame` | ✓ | ✓ |
| `PlotFrame` | ✓ | ✓ |
| `ParametersFrame` | ✓ | ✓ |
| `RTTableFrame` | ✓ (relabeled) | ✓ (relabeled "Feature Table") |
| `MSFrame` | ✓ (gcms only) | hidden |
| `QuantitationFrame` | ✓ (gcms only) | hidden |
| `ButtonFrame` | ✓ | ✓ |

`RTTableFrame` is reused with relabeled columns — "Retention Time" → "Position", units drawn from `profile.x_label`. No new frame class is needed for the basic spectroscopy case.

`PlotFrame` receives `profile.x_label` and `profile.y_label` to update axis labels.

`set_mode()` reads the profile once on load and distributes display config to each frame. The profile is the single source of truth for all display adaptation.

---

## 8. New Files + Modified Files

### New files

| Path | Purpose |
|------|---------|
| `logic/feature.py` | `Feature` base class, `SpectralFeature` |
| `logic/signal_profiles.py` | `SignalProfile` dataclass, `SignalProfileRegistry`, built-in registrations |
| `logic/c_folder.py` | `CFolder` manager |
| `logic/loaders/__init__.py` | loader package |
| `logic/loaders/base.py` | `DataLoader` ABC |
| `logic/loaders/agilent_loader.py` | `AgilentLoader` wrapping `DataHandler` |
| `logic/loaders/csv_loader.py` | `CSVLoader` for CSV files |
| `ui/dialogs/c_folder_migration_dialog.py` | Migration utility dialog |

### Modified files

| Path | Change |
|------|--------|
| `logic/integration.py` | `Peak` → `ChromatographicPeak(Feature)`, `Peak` alias added |
| `logic/processor.py` | Accept `profile` param, gate stages, use `profile.feature_class` |
| `logic/automation_worker.py` | Scan for `.C` folders instead of `.D` folders; load via `CFolder.open()` |
| `logic/json_exporter.py` | Receive instrument metadata from `CFolder.get_manifest()` instead of calling `rb.read()` directly |
| `ui/app.py` | Add `set_mode()`, call on folder load, wire migration dialog |
| `ui/frames/tree.py` | Change name filter from `["*.D"]` to `["*.C"]`; double-click routes to `CFolder.open()` instead of `DataHandler.load_data_directory()`; direct double-click on a `.D` folder prompts migration |
| `ui/frames/plot_frame.py` | Accept axis label updates from profile |
| `ui/frames/rt_table_frame.py` | Accept column label updates from profile |

---

## Out of Scope

- Real-time / streaming data acquisition (separate future design)
- NMR, Raman, or other signal types beyond FTIR and UV-Vis
- New quantitation methods for spectroscopy (Beer-Lambert, etc.)
- REST API updates (follow-on work once logic layer is stable)
