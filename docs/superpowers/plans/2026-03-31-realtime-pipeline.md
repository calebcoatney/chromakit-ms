# Real-Time Pipeline & Method File Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `.chromethod` file format and `POST /api/run` endpoint so external scripts can drive the full GC/GC-MS/FTIR/UV-Vis processing pipeline programmatically.

**Architecture:** A new `logic/method.py` becomes the single source of truth for all processing param models (currently duplicated in `api/models.py`). `ChromaMethod` wraps those models with file I/O and GUI ↔ method conversion. The `POST /api/run` endpoint chains the existing load/process/integrate/export logic in one call using a method file as its parameter carrier. The GUI gains Save/Load Method buttons that serialize/deserialize `ParametersFrame.current_params`.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI, PySide6. No new dependencies introduced.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `logic/method.py` | **Create** | `ChromaMethod` class + all processing param models |
| `tests/logic/test_method.py` | **Create** | Unit tests for ChromaMethod |
| `api/models.py` | **Modify** | Import param models from `logic/method.py`; add `RunRequest`, `RunResponse` |
| `api/main.py` | **Modify** | Implement `POST /api/export`; add `POST /api/run` |
| `ui/frames/parameters.py` | **Modify** | Add Save/Load Method buttons + `_reinitialize_controls()` |
| `tools/watch_and_run.py` | **Create** | Reference file-watcher script (not part of core) |

---

## Task 1: Create `logic/method.py` — param models + ChromaMethod

**Files:**
- Create: `logic/method.py`
- Create: `tests/logic/test_method.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/logic/test_method.py`:

```python
"""Tests for logic/method.py — ChromaMethod and param models."""
import json
import tempfile
from pathlib import Path
import pytest

# These imports will fail until logic/method.py is created — that's expected.
from logic.method import (
    ChromaMethod,
    SmoothingParams,
    BaselineParams,
    PeakParams,
    DeconvolutionParams,
    NegativePeakParams,
    ShoulderParams,
    IntegrationSubParams,
)

# Minimal GUI params dict (mirrors ParametersFrame.current_params)
_GUI_PARAMS = {
    "smoothing": {
        "enabled": False, "method": "whittaker", "median_enabled": False,
        "median_kernel": 5, "lambda": 0.1, "diff_order": 1,
        "savgol_window": 3, "savgol_polyorder": 1,
    },
    "baseline": {
        "show_corrected": False, "method": "arpls", "lambda": 1e4,
        "asymmetry": 0.01, "baseline_offset": 0.0, "align_tic": False,
        "break_points": [], "fastchrom": {"half_window": None, "smooth_half_window": None},
    },
    "peaks": {
        "enabled": False, "mode": "classical", "min_prominence": 1e5,
        "min_height": 0.0, "min_width": 0.0, "range_filters": [],
    },
    "peak_splitting": {
        "splitting_method": "geometric", "windows": [],
        "heatmap_threshold": 0.36, "pre_fit_signal_threshold": 0.001,
        "min_area_frac": 0.15, "valley_threshold_frac": 0.48,
        "mu_bound_factor": 0.68, "fat_threshold_frac": 0.44,
        "dedup_sigma_factor": 1.32, "dedup_rt_tolerance": 0.005,
    },
    "negative_peaks": {"enabled": False, "min_prominence": 1e5},
    "shoulders": {
        "enabled": False, "window_length": 41, "polyorder": 3,
        "sensitivity": 8, "apex_distance": 10,
    },
    "integration": {"peak_groups": []},
}


def test_creates_with_defaults():
    m = ChromaMethod(name="Test", signal_type="gc")
    assert m.name == "Test"
    assert m.signal_type == "gc"
    assert m.version == "1"
    assert m.chemstation_area_factor == pytest.approx(0.0784)


def test_invalid_signal_type_raises():
    with pytest.raises(Exception):  # ValueError or KeyError from registry
        ChromaMethod(name="Bad", signal_type="nonexistent_instrument")


def test_round_trip_to_from_file():
    m = ChromaMethod(name="CO2 Hydro GC", signal_type="gc")
    m.smoothing.enabled = True
    m.baseline.method = "snip"
    m.chemstation_area_factor = 0.05

    with tempfile.NamedTemporaryFile(suffix=".chromethod", delete=False, mode="w") as f:
        path = f.name

    m.to_file(path)
    loaded = ChromaMethod.from_file(path)

    assert loaded.name == "CO2 Hydro GC"
    assert loaded.signal_type == "gc"
    assert loaded.smoothing.enabled is True
    assert loaded.baseline.method == "snip"
    assert loaded.chemstation_area_factor == pytest.approx(0.05)


def test_to_processor_params_excludes_metadata():
    m = ChromaMethod(name="Test", signal_type="gc")
    p = m.to_processor_params()
    for key in ("name", "signal_type", "created_at", "version",
                "chemstation_area_factor", "export_output_dir"):
        assert key not in p, f"metadata key '{key}' should not be in processor params"
    for key in ("smoothing", "baseline", "peaks", "deconvolution",
                "negative_peaks", "shoulders", "integration"):
        assert key in p, f"param key '{key}' should be in processor params"


def test_to_processor_params_lambda_alias():
    """Lambda must serialize as 'lambda' (the alias), not 'lambda_'."""
    m = ChromaMethod(name="Test", signal_type="gc")
    m.smoothing.lambda_ = 0.5
    m.baseline.lambda_ = 1e5
    p = m.to_processor_params()
    assert "lambda" in p["smoothing"], "smoothing lambda key should be 'lambda'"
    assert "lambda_" not in p["smoothing"]
    assert "lambda" in p["baseline"], "baseline lambda key should be 'lambda'"


def test_from_gui_params_renames_peak_splitting():
    m = ChromaMethod.from_gui_params(_GUI_PARAMS, name="Test", signal_type="gc")
    assert m.deconvolution.splitting_method == "geometric"
    assert m.smoothing.enabled is False
    assert m.peaks.min_prominence == pytest.approx(1e5)


def test_to_gui_params_renames_deconvolution():
    m = ChromaMethod(name="Test", signal_type="gc")
    m.deconvolution.splitting_method = "emg"
    gui = m.to_gui_params()
    assert "peak_splitting" in gui, "GUI expects 'peak_splitting', not 'deconvolution'"
    assert "deconvolution" not in gui
    assert gui["peak_splitting"]["splitting_method"] == "emg"


def test_from_gui_to_gui_roundtrip():
    m = ChromaMethod.from_gui_params(_GUI_PARAMS, name="Test", signal_type="gc")
    result = m.to_gui_params()
    assert result["smoothing"]["enabled"] is False
    assert result["smoothing"]["method"] == "whittaker"
    assert result["baseline"]["method"] == "arpls"
    assert result["peaks"]["min_prominence"] == pytest.approx(1e5)
    assert "peak_splitting" in result
    assert result["peak_splitting"]["splitting_method"] == "geometric"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
conda run -n chromakit-env pytest tests/logic/test_method.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'ChromaMethod' from 'logic.method'` (or ModuleNotFoundError).

- [ ] **Step 3: Create `logic/method.py`**

```python
"""Method file format for ChromaKit processing pipelines.

A ChromaMethod is a named, persisted snapshot of all processing parameters.
It is the single source of truth for parameter models — imported by api/ and
read/written by the GUI's Save/Load Method buttons.

File format: JSON with .chromethod extension.

Layer rule: this module is in logic/ and must NOT import from api/ or ui/.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Processing Parameter Sub-Models ────────────────────────────────────────────
# These are the canonical definitions. api/models.py imports from here.


class SmoothingParams(BaseModel):
    enabled: bool = False
    method: str = Field(default="whittaker", description="'whittaker' or 'savgol'")
    median_enabled: bool = Field(default=False, description="Apply median pre-filter")
    median_kernel: int = Field(default=5, ge=3, description="Median filter kernel (odd)")
    lambda_: float = Field(default=1e-1, alias="lambda", description="Whittaker lambda")
    diff_order: int = Field(default=1, ge=1, le=2, description="Whittaker difference order")
    savgol_window: int = Field(default=3, ge=3, description="Savitzky-Golay window (odd)")
    savgol_polyorder: int = Field(default=1, ge=1, description="Savitzky-Golay poly order")

    model_config = {"populate_by_name": True}


class BreakPoint(BaseModel):
    time: float = Field(..., description="Break point time in minutes")
    tolerance: float = Field(default=0.1, description="Tolerance window around break point")


class FastchromParams(BaseModel):
    half_window: Optional[int] = None
    smooth_half_window: Optional[int] = None


class BaselineParams(BaseModel):
    show_corrected: bool = False
    method: str = Field(
        default="arpls",
        description="asls|arpls|airpls|imodpoly|modpoly|snip|mixture_model|irsqr|fastchrom",
    )
    lambda_: float = Field(default=1e4, alias="lambda")
    asymmetry: float = 0.01
    baseline_offset: float = Field(default=0.0)
    align_tic: bool = Field(default=False, description="Align MS TIC to FID time axis")
    break_points: Optional[List[BreakPoint]] = Field(default_factory=list)
    fastchrom: Optional[FastchromParams] = Field(default_factory=FastchromParams)

    model_config = {"populate_by_name": True}


class PeakParams(BaseModel):
    enabled: bool = False
    mode: str = Field(default="classical", description="'classical' or 'deconvolution'")
    window_length: int = 41
    polyorder: int = 3
    peak_prominence: float = 0.05
    peak_width: int = 5
    min_prominence: Optional[float] = Field(default=1e5)
    min_height: Optional[float] = 0.0
    min_width: Optional[float] = 0.0
    range_filters: Optional[List[List[float]]] = Field(default_factory=list)


class DeconvolutionParams(BaseModel):
    splitting_method: str = Field(default="geometric", description="'geometric' or 'emg'")
    windows: Optional[List[List[float]]] = Field(default_factory=list)
    heatmap_threshold: float = 0.36
    pre_fit_signal_threshold: float = 0.001
    min_area_frac: float = 0.15
    valley_threshold_frac: float = 0.48
    mu_bound_factor: float = 0.68
    fat_threshold_frac: float = 0.44
    dedup_sigma_factor: float = 1.32
    dedup_rt_tolerance: float = 0.005


class NegativePeakParams(BaseModel):
    enabled: bool = False
    min_prominence: float = 1e5


class ShoulderParams(BaseModel):
    enabled: bool = False
    window_length: int = 41
    polyorder: int = 3
    sensitivity: int = Field(default=8, ge=1, le=10, description="Detection sensitivity 1–10")
    apex_distance: int = 10


class IntegrationSubParams(BaseModel):
    peak_groups: Optional[List[List[float]]] = Field(
        default_factory=list,
        description="[start, end] time windows for peak grouping",
    )


# ── ChromaMethod ────────────────────────────────────────────────────────────────

_METADATA_FIELDS = frozenset({
    "name", "version", "signal_type", "created_at",
    "chemstation_area_factor", "export_output_dir",
})


class ChromaMethod(BaseModel):
    """Named snapshot of all ChromaKit processing parameters.

    Usage:
        ChromaMethod.from_file("run.chromethod")   # load from disk
        method.to_file("run.chromethod")            # save to disk
        method.to_processor_params()                # dict for convert_params_for_processor()
        ChromaMethod.from_gui_params(params, ...)   # build from ParametersFrame.current_params
        method.to_gui_params()                      # restore to ParametersFrame.current_params
    """

    name: str
    version: str = "1"
    signal_type: str = Field(..., description="Registered SignalProfileRegistry name")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    smoothing: SmoothingParams = Field(default_factory=SmoothingParams)
    baseline: BaselineParams = Field(default_factory=BaselineParams)
    peaks: PeakParams = Field(default_factory=PeakParams)
    deconvolution: DeconvolutionParams = Field(default_factory=DeconvolutionParams)
    negative_peaks: NegativePeakParams = Field(default_factory=NegativePeakParams)
    shoulders: ShoulderParams = Field(default_factory=ShoulderParams)
    integration: IntegrationSubParams = Field(default_factory=IntegrationSubParams)
    chemstation_area_factor: float = Field(
        default=0.0784,
        description="Chemstation area conversion factor applied during integration",
    )
    export_output_dir: Optional[str] = Field(
        default=None,
        description="Output directory for exported JSON; None = same dir as data file",
    )

    @field_validator("signal_type")
    @classmethod
    def _validate_signal_type(cls, v: str) -> str:
        from logic.signal_profiles import SignalProfileRegistry
        SignalProfileRegistry.get(v)  # raises KeyError if unregistered
        return v

    @classmethod
    def from_file(cls, path: str | Path) -> "ChromaMethod":
        """Load a .chromethod JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def to_file(self, path: str | Path) -> None:
        """Write this method to a .chromethod JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, by_alias=True))

    def to_processor_params(self) -> dict:
        """Return a params dict ready for convert_params_for_processor().

        Excludes all method metadata (name, signal_type, etc.) and
        serializes lambda fields using their 'lambda' alias so the
        processor receives the expected key names.
        """
        return self.model_dump(by_alias=True, exclude=_METADATA_FIELDS)

    @classmethod
    def from_gui_params(
        cls,
        params: dict,
        name: str,
        signal_type: str,
        chemstation_area_factor: float = 0.0784,
        export_output_dir: Optional[str] = None,
    ) -> "ChromaMethod":
        """Build a ChromaMethod from ParametersFrame.current_params.

        The GUI stores deconvolution params under the key 'peak_splitting'.
        This method renames that key to 'deconvolution' for the method schema.
        """
        d = dict(params)
        d["deconvolution"] = d.pop("peak_splitting", d.get("deconvolution", {}))
        return cls(
            name=name,
            signal_type=signal_type,
            chemstation_area_factor=chemstation_area_factor,
            export_output_dir=export_output_dir,
            **d,
        )

    def to_gui_params(self) -> dict:
        """Return a dict compatible with ParametersFrame.current_params.

        Renames 'deconvolution' back to 'peak_splitting' for GUI compatibility.
        Excludes all method metadata fields.
        """
        d = self.model_dump(by_alias=True, exclude=_METADATA_FIELDS)
        d["peak_splitting"] = d.pop("deconvolution", {})
        return d
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
conda run -n chromakit-env pytest tests/logic/test_method.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
git add logic/method.py tests/logic/test_method.py && \
git commit -m "feat: add logic/method.py with ChromaMethod and consolidated param models"
```

---

## Task 2: Update `api/models.py` to import param models from `logic/method.py`

**Files:**
- Modify: `api/models.py`

The param sub-models (`SmoothingParams`, `BaselineParams`, etc.) currently defined in `api/models.py` are moved to `logic/method.py` in Task 1. This task replaces the duplicate definitions with imports and adds the two new models (`RunRequest`, `RunResponse`) needed by `POST /api/run`.

- [ ] **Step 1: Replace param model definitions with imports**

In `api/models.py`, find the block starting at line 1 (`"""Pydantic models...`). Replace the entire param model section (lines 18–116, from `class SmoothingParams` through `class IntegrationSubParams`) with a single import block:

```python
# ── Processing param models — canonical definitions live in logic/method.py ──
from logic.method import (
    SmoothingParams,
    BreakPoint,
    FastchromParams,
    BaselineParams,
    PeakParams,
    DeconvolutionParams,
    NegativePeakParams,
    ShoulderParams,
    IntegrationSubParams,
)
```

Keep `ProcessingParams` (line 108–116) in `api/models.py` — it is an API-specific composition that imports its sub-models. After inserting the import block, `ProcessingParams` will reference the imported names correctly without any other changes.

- [ ] **Step 2: Add `RunRequest` and `RunResponse` to `api/models.py`**

Append these two classes after the existing `NavigationResponse` class at the end of the response models section:

```python
class RunRequest(BaseModel):
    """Request to run the full Phase 1 pipeline: load → process → integrate → export JSON."""
    data_path: str = Field(..., description="Path to Agilent .D or .C directory")
    method_path: str = Field(..., description="Path to .chromethod file")
    detector: Optional[str] = Field(
        None,
        description="Detector to use (e.g. 'FID1A'). Auto-detected if omitted.",
    )


class RunResponse(BaseModel):
    """Response from POST /api/run."""
    status: str = Field(..., description="'complete' or 'error'")
    data_path: str
    method: str = Field(..., description="Method name from the .chromethod file")
    signal_type: str
    peak_count: int
    peaks: List[Dict[str, Any]]
    output_files: List[str] = Field(
        ..., description="Absolute paths of JSON files written to disk"
    )
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
conda run -n chromakit-env pytest tests/ -v
```

Expected: all existing tests PASS (no regressions from the import refactor).

- [ ] **Step 4: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
git add api/models.py && \
git commit -m "refactor: import param models from logic/method.py in api/models.py; add RunRequest/RunResponse"
```

---

## Task 3: Implement `POST /api/export`

**Files:**
- Modify: `api/main.py`

The endpoint stub at line ~360 (`POST /api/export`) is replaced with a real implementation. It accepts already-serialized peak dicts and the source data path, then writes a JSON result file using the same export context logic as the GUI.

- [ ] **Step 1: Replace the stub**

In `api/main.py`, find the existing stub:

```python
@app.post("/api/export")
async def export_results(request: dict):
    """Export integration results to JSON/CSV/XLSX."""
    # TODO: Wire up ExportManager + json_exporter
    raise HTTPException(status_code=501, detail="Export endpoint not yet implemented")
```

Replace it with:

```python
@app.post("/api/export")
async def export_results(request: ExportRequest):
    """Export serialized peak results to a JSON file.

    Accepts peak dicts from a prior /api/integrate call.
    Writes to the same location as the GUI exporter (inside the .D directory,
    or inside {.C}/results/ for .C folders).

    Only 'json' format is supported in Phase 1.
    """
    from logic.json_exporter import _resolve_export_context

    try:
        if request.format != "json":
            raise HTTPException(
                status_code=400,
                detail=f"Format '{request.format}' not supported. Only 'json' is available.",
            )

        metadata, output_path = _resolve_export_context(
            request.file_path, data_handler.current_detector
        )

        result_data = {
            **metadata,
            "peaks": request.peaks,
        }

        import json as _json
        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(result_data, f, indent=4)

        return {"status": "exported", "output_file": output_path}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")
```

Also update the import line near the top of `api/main.py` to include `ExportRequest`:

```python
from api.models import (
    BrowseResponse, FileEntry,
    LoadFileRequest, LoadFileResponse, ChromatogramData, TICData,
    ProcessRequest, ProcessResponse,
    IntegrateRequest, IntegrateResponse,
    SpectrumRequest, SpectrumResponse,
    AlignTICRequest, AlignTICResponse,
    AssignmentRequest, ScalingFactorsRequest,
    NavigationResponse,
    ExportRequest,                           # ← add this
)
```

- [ ] **Step 2: Smoke-test manually**

Start the API server in one terminal:

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
conda run -n chromakit-env python api/main.py
```

In another terminal, run a minimal test (replace the file path with any .D file on disk):

```bash
curl -s -X POST http://127.0.0.1:8000/api/export \
  -H "Content-Type: application/json" \
  -d '{
    "peaks": [{"rt": 1.5, "area": 10000.0}],
    "file_path": "/path/to/sample.D",
    "format": "json"
  }' | python -m json.tool
```

Expected: `{"status": "exported", "output_file": "/path/to/sample.D/...json"}`.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
git add api/main.py && \
git commit -m "feat: implement POST /api/export endpoint"
```

---

## Task 4: Implement `POST /api/run`

**Files:**
- Modify: `api/main.py`

This is the high-level endpoint for the automation use case. It reads a `.chromethod` file, runs the full Phase 1 pipeline (load → process → integrate → export JSON), and returns the peak table and output file path in one response.

- [ ] **Step 1: Add the import for RunRequest/RunResponse**

In `api/main.py`, extend the existing `from api.models import (...)` block to include `RunRequest` and `RunResponse`:

```python
from api.models import (
    BrowseResponse, FileEntry,
    LoadFileRequest, LoadFileResponse, ChromatogramData, TICData,
    ProcessRequest, ProcessResponse,
    IntegrateRequest, IntegrateResponse,
    SpectrumRequest, SpectrumResponse,
    AlignTICRequest, AlignTICResponse,
    AssignmentRequest, ScalingFactorsRequest,
    NavigationResponse,
    ExportRequest,
    RunRequest, RunResponse,               # ← add these
)
```

- [ ] **Step 2: Add `POST /api/run` to `api/main.py`**

Add the following endpoint after the `POST /api/export` implementation from Task 3:

```python
# ─── Full Pipeline Run ────────────────────────────────────────────────

@app.post("/api/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest):
    """Run the full Phase 1 pipeline in one call: load → process → integrate → export JSON.

    Requires a .chromethod file (see POST /api/method/validate or save from the GUI).
    Writes a JSON result file to the same directory as the data file and returns
    the peak table plus the output file path.
    """
    import numpy as np
    from logic.method import ChromaMethod
    from logic.json_exporter import export_integration_results_to_json, _resolve_export_context

    try:
        # 1. Load method
        try:
            method = ChromaMethod.from_file(request.method_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Method file not found: {request.method_path}"
            )

        # 2. Load data
        try:
            data = data_handler.load_data_directory(
                request.data_path, detector=request.detector
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Data file not found: {request.data_path}"
            )

        x = np.array(data["chromatogram"]["x"])
        y = np.array(data["chromatogram"]["y"])

        # 3. Process (smoothing + baseline + peak detection)
        raw_params = method.to_processor_params()
        proc_params = convert_params_for_processor(raw_params)
        processed = processor.process(x, y, params=proc_params)

        # 4. Integrate
        integrated = processor.integrate_peaks(
            processed_data=processed,
            rt_table=None,
            chemstation_area_factor=method.chemstation_area_factor,
            peak_groups=method.integration.peak_groups or [],
        )
        peaks = integrated.get("peaks", [])

        # 5. Export JSON (writes to same directory as data file, mirroring GUI behavior)
        export_integration_results_to_json(
            peaks=peaks,
            d_path=request.data_path,
            detector=data_handler.current_detector,
            processing_params=raw_params,
        )

        # 6. Determine where the file was written
        _, output_file = _resolve_export_context(
            request.data_path, data_handler.current_detector
        )

        # 7. Serialize peaks for response
        peaks_dicts = []
        for peak in peaks:
            d = peak.as_dict() if hasattr(peak, "as_dict") else peak
            peaks_dicts.append(serialize_numpy(d))

        return RunResponse(
            status="complete",
            data_path=request.data_path,
            method=method.name,
            signal_type=method.signal_type,
            peak_count=len(peaks_dicts),
            peaks=peaks_dicts,
            output_files=[output_file],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
```

- [ ] **Step 3: Smoke-test manually**

First save a method file from the GUI (or create one manually with the correct schema). Then:

```bash
curl -s -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "/path/to/sample.D",
    "method_path": "/path/to/my_method.chromethod"
  }' | python -m json.tool
```

Expected: JSON response with `"status": "complete"`, a `peaks` array, and an `output_files` list.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
git add api/main.py && \
git commit -m "feat: implement POST /api/run full pipeline endpoint"
```

---

## Task 5: GUI Save/Load Method buttons

**Files:**
- Modify: `ui/frames/parameters.py`

Add two buttons above the parameter scroll area. "Save Method…" captures the current widget state; "Load Method…" restores it. Loading rebuilds the widgets from scratch using the existing `_init_*_controls()` methods (which all read from `self.current_params`).

- [ ] **Step 1: Add the method button bar**

In `ui/frames/parameters.py`, in `ParametersFrame.__init__`, find the final two lines of `__init__` (where the scroll area is wired up):

```python
        self.params_scroll.setWidget(self.params_widget)

        # Add scroll area to main layout
        self.layout.addWidget(self.params_scroll)
```

Insert a button bar **before** `self.layout.addWidget(self.params_scroll)`:

```python
        self.params_scroll.setWidget(self.params_widget)

        # ── Method file buttons ─────────────────────────────────────────
        _method_bar = QHBoxLayout()
        self._load_method_btn = QPushButton("Load Method…")
        self._save_method_btn = QPushButton("Save Method…")
        _method_bar.addWidget(self._load_method_btn)
        _method_bar.addWidget(self._save_method_btn)
        self._load_method_btn.clicked.connect(self._on_load_method)
        self._save_method_btn.clicked.connect(self._on_save_method)
        self.layout.addLayout(_method_bar)
        # ───────────────────────────────────────────────────────────────

        # Add scroll area to main layout
        self.layout.addWidget(self.params_scroll)
```

- [ ] **Step 2: Add `_on_save_method`, `_on_load_method`, and `_reinitialize_controls`**

Append these three methods to the `ParametersFrame` class (anywhere after `__init__`):

```python
    # ── Method file I/O ────────────────────────────────────────────────────

    def _on_save_method(self):
        """Serialize current widget state to a .chromethod file."""
        from PySide6.QtWidgets import QFileDialog, QInputDialog
        from logic.method import ChromaMethod

        name, ok = QInputDialog.getText(self, "Save Method", "Method name:")
        if not ok or not name.strip():
            return

        signal_type, ok = QInputDialog.getItem(
            self, "Save Method", "Signal type:",
            ["gc", "gcms", "ftir", "uvvis"], 0, False,
        )
        if not ok:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Method", f"{name.strip()}.chromethod",
            "ChromaKit Method (*.chromethod)",
        )
        if not path:
            return

        try:
            method = ChromaMethod.from_gui_params(
                self.current_params, name=name.strip(), signal_type=signal_type,
            )
            method.to_file(path)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Method Failed", str(e))

    def _on_load_method(self):
        """Load a .chromethod file and apply it to the parameter widgets."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from logic.method import ChromaMethod

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Method", "", "ChromaKit Method (*.chromethod)",
        )
        if not path:
            return

        try:
            method = ChromaMethod.from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Method Failed", str(e))
            return

        self.current_params = method.to_gui_params()
        self._reinitialize_controls()
        self.parameters_changed.emit(self.current_params)

    def _reinitialize_controls(self):
        """Rebuild all parameter widgets from self.current_params.

        Called after loading a method file. Because all _init_*_controls()
        methods read from self.current_params when creating their widgets,
        clearing the layout and re-running them is sufficient to restore
        the full widget state.
        """
        # Remove all existing widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Rebuild title
        from PySide6.QtWidgets import QLabel
        parameters_label = QLabel("Integration Parameters")
        parameters_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.params_layout.addWidget(parameters_label)

        # Rebuild all control sections
        self.section_groups = {}
        self._init_smoothing_controls()
        self._init_baseline_controls()
        self._init_peaks_controls()
        self._init_negative_peaks_controls()
        self._init_shoulder_controls()
        self._init_range_filter_controls()
        self._init_peak_grouping_controls()
        self.params_layout.addStretch()
```

- [ ] **Step 3: Manual test**

Launch the GUI:

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
conda run -n chromakit-env chromakit-ms
```

1. Change a few parameters (e.g., enable smoothing, change baseline method to "snip").
2. Click "Save Method…" → enter a name → pick "gc" → save as `test.chromethod`.
3. Change the parameters again (to confirm they change).
4. Click "Load Method…" → pick `test.chromethod`.
5. Verify all widgets match the originally saved values.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
git add ui/frames/parameters.py && \
git commit -m "feat: add Save/Load Method buttons to ParametersFrame"
```

---

## Task 6: Reference file watcher script

**Files:**
- Create: `tools/watch_and_run.py`

A standalone script users can adapt for their automation loops. Not part of ChromaKit core — it is not imported by any other module and is not installed by `setup.py`.

- [ ] **Step 1: Create `tools/watch_and_run.py`**

```python
#!/usr/bin/env python3
"""
watch_and_run.py — Reference file watcher for ChromaKit automated pipelines.

Monitors a directory for new Agilent .D directories. When one appears,
calls POST /api/run on a running ChromaKit API server using a specified
.chromethod file. Results (JSON) are written by the API server alongside
the data file.

Usage:
    python tools/watch_and_run.py \\
        --watch-dir /path/to/gc_data \\
        --method    /path/to/my_method.chromethod \\
        --api-url   http://127.0.0.1:8000

Dependencies (not in setup.py — install manually):
    pip install watchdog requests

This script is a starting point. Adapt it for your own automation loop
(e.g., forward results to a Bayesian optimizer, log to a database, etc.).
"""
import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    sys.exit(
        "Missing dependencies. Install with:\n"
        "    pip install watchdog requests\n"
    )


class _DDirectoryHandler(FileSystemEventHandler):
    """Fires POST /api/run when a new .D directory is created."""

    def __init__(self, method_path: str, api_url: str):
        self.method_path = method_path
        self.api_url = api_url.rstrip("/")
        self._seen: set[str] = set()

    def on_created(self, event):
        if not event.is_directory:
            return
        path = event.src_path
        if not path.endswith(".D"):
            return
        if path in self._seen:
            return
        self._seen.add(path)
        print(f"[watcher] New .D detected: {path}")
        self._run(path)

    def _run(self, data_path: str):
        payload = {"data_path": data_path, "method_path": self.method_path}
        try:
            resp = requests.post(
                f"{self.api_url}/api/run",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            peak_count = result.get("peak_count", "?")
            output_files = result.get("output_files", [])
            print(
                f"[watcher] {Path(data_path).name}: "
                f"{peak_count} peaks → {output_files}"
            )
        except requests.exceptions.Timeout:
            print(f"[watcher] ERROR: API timed out processing {data_path}")
        except requests.exceptions.ConnectionError:
            print(f"[watcher] ERROR: Could not connect to API at {self.api_url}")
        except Exception as e:
            print(f"[watcher] ERROR processing {data_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Watch a directory for new .D files and process them via ChromaKit API."
    )
    parser.add_argument(
        "--watch-dir", required=True, help="Directory to watch for new .D files"
    )
    parser.add_argument(
        "--method", required=True, help="Path to .chromethod file"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="ChromaKit API base URL (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    watch_dir = Path(args.watch_dir)
    if not watch_dir.is_dir():
        sys.exit(f"ERROR: watch-dir does not exist: {watch_dir}")
    if not Path(args.method).is_file():
        sys.exit(f"ERROR: method file does not exist: {args.method}")

    # Verify API is reachable
    try:
        resp = requests.get(f"{args.api_url.rstrip('/')}/api/health", timeout=5)
        resp.raise_for_status()
        print(f"[watcher] API healthy at {args.api_url}")
    except Exception as e:
        sys.exit(f"ERROR: Cannot reach ChromaKit API at {args.api_url}: {e}")

    handler = _DDirectoryHandler(
        method_path=str(Path(args.method).resolve()),
        api_url=args.api_url,
    )
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    print(f"[watcher] Watching {watch_dir} for new .D directories. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test**

With the API server running and a test `.D` file available:

```bash
# In terminal 1: start the API
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
conda run -n chromakit-env python api/main.py

# In terminal 2: start the watcher
conda run -n chromakit-env pip install watchdog requests --quiet
conda run -n chromakit-env python tools/watch_and_run.py \
    --watch-dir /path/to/watch_dir \
    --method    /path/to/my_method.chromethod

# In terminal 3: simulate a new file arriving (copy a .D into the watched dir)
cp -r /path/to/existing_sample.D /path/to/watch_dir/new_sample.D
```

Expected: watcher prints `[watcher] new_sample.D: N peaks → ['/path/to/watch_dir/new_sample.D/...json']`.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && \
git add tools/watch_and_run.py && \
git commit -m "feat: add reference file watcher script (tools/watch_and_run.py)"
```

---

## Self-Review Notes

**Spec coverage:**
- ✅ Section 1 (method file format) → Task 1
- ✅ Section 2 (logic/method.py param consolidation) → Tasks 1–2
- ✅ Section 3 (GUI Save/Load) → Task 5
- ✅ Section 4 (POST /api/run) → Task 4
- ✅ Section 5 (POST /api/export) → Task 3
- ✅ Section 6 (reference watcher) → Task 6

**Type consistency check:**
- `ChromaMethod.to_processor_params()` → returns `dict` → passed to `convert_params_for_processor()` in Task 4 ✅
- `ChromaMethod.from_gui_params()` → used in Task 5 `_on_save_method` ✅
- `ChromaMethod.to_gui_params()` → used in Task 5 `_on_load_method` ✅
- `RunRequest` / `RunResponse` → defined Task 2, imported Task 4 ✅
- `ExportRequest` → already existed in models.py; imported in Task 3 ✅
- `_resolve_export_context` → private function in json_exporter; used Tasks 3 & 4 ✅

**Dependency order:** Tasks 1 → 2 → 3 → 4 must run in sequence (each builds on prior). Task 5 (GUI) depends on Task 1 only. Task 6 is fully independent.
