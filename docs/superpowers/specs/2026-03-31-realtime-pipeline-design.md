# Real-Time Pipeline & Method File Design

**Date:** 2026-03-31
**Status:** Approved

## Overview

Add programmatic real-time analysis capability to ChromaKit by: (1) formalizing a method file format that captures all processing parameters, and (2) completing the REST API with a high-level `POST /api/run` endpoint that executes the full pipeline in one call. Together these allow external automation loops (Bayesian optimizers, reactor scripts, file watchers) to use ChromaKit as a reliable, reproducible analysis engine without any GUI interaction.

The feature is decomposed into three phases:

- **Phase 1 (this spec):** Method file + `POST /api/run` → load, process, integrate, export JSON
- **Phase 2:** MS library search via API
- **Phase 3:** Quantitation methods via API

---

## Motivation

ChromaKit already implements the full GC/GC-MS/FTIR/UV-Vis processing pipeline. The missing pieces for programmatic use are:

1. No way to express "use these parameters" without specifying every value in code
2. No single API endpoint that runs the full pipeline end-to-end
3. Processing param models are duplicated between `api/models.py` and the GUI

A method file solves (1), `POST /api/run` solves (2), and consolidating param models into `logic/method.py` solves (3).

---

## Section 1: Method File Format

### File Extension

`.chromethod` (JSON internally)

### Schema Envelope

```json
{
  "name": "CO2 Hydrogenation GC",
  "version": "1",
  "signal_type": "gc",
  "created_at": "2026-03-31T14:22:00",
  "smoothing": { ... },
  "baseline": { ... },
  "peaks": { ... },
  "shoulders": { ... },
  "deconvolution": { ... },
  "integration": { ... },
  "export": { "output_dir": null }
}
```

The envelope fields are fixed; the nested parameter objects contain all fields from the existing Pydantic models (see Section 2). The example above is illustrative — the actual schema is defined by the Pydantic models, which cover all advanced options (break points, shoulder detection, deconvolution windows, fastchrom params, etc.).

### `signal_type`

Maps directly to `SignalProfileRegistry` names: `"gc"`, `"gcms"`, `"ftir"`, `"uvvis"`. The registry determines which pipeline stages apply for that signal type. Validation fails fast at load time if the value is not a registered profile name.

---

## Section 2: `logic/method.py` — Param Model Consolidation

### Problem

Processing param models currently live in `api/models.py` with a note that they "mirror" `ui/frames/parameters.py`. Adding method files creates a third consumer. The duplication should be resolved now.

### Solution

Move all processing param models into `logic/method.py` as the single source of truth:

- `SmoothingParams`
- `BaselineParams` (including `BreakPoint`, `FastchromParams`)
- `PeakParams`
- `ShoulderParams`
- `DeconvolutionParams`
- `IntegrationParams` (new — captures `chemstation_area_factor` and related)
- `ExportParams` (new — captures `output_dir` and format preferences)

`api/models.py` imports these from `logic/method.py` rather than redefining them.

### `ChromaMethod` Class

```python
class ChromaMethod(BaseModel):
    name: str
    version: str = "1"
    signal_type: str          # validated against SignalProfileRegistry
    created_at: datetime = Field(default_factory=datetime.utcnow)
    smoothing: SmoothingParams = Field(default_factory=SmoothingParams)
    baseline: BaselineParams = Field(default_factory=BaselineParams)
    peaks: PeakParams = Field(default_factory=PeakParams)
    shoulders: ShoulderParams = Field(default_factory=ShoulderParams)
    deconvolution: DeconvolutionParams = Field(default_factory=DeconvolutionParams)
    integration: IntegrationParams = Field(default_factory=IntegrationParams)
    export: ExportParams = Field(default_factory=ExportParams)

    @classmethod
    def from_file(cls, path: str | Path) -> "ChromaMethod": ...
    def to_file(self, path: str | Path) -> None: ...
    def to_processor_params(self) -> dict: ...
```

`to_processor_params()` converts the Pydantic model tree into the flat dict that `ChromatogramProcessor.process()` already accepts, so no changes are needed to the processor.

---

## Section 3: GUI — Save/Load Method

The interactive parameter widgets in `ParametersFrame` are unchanged. Two buttons are added to the parameters panel (or the File menu):

- **Save Method...** — opens a file dialog, writes the current widget state to a `.chromethod` file via `ChromaMethod.to_file()`
- **Load Method...** — opens a file dialog, reads a `.chromethod` file, and restores all parameter values to the widgets

This is a read/write bridge between the widget state and `ChromaMethod`. No new parameter storage logic is introduced.

---

## Section 4: `POST /api/run` Endpoint

### Request

```
POST /api/run
{
  "data_path": "/path/to/sample.D",
  "method_path": "/path/to/co2_hydro.chromethod",
  "output_path": "/path/to/results/"   // optional; defaults to data_path directory
}
```

### Behavior

Chains the existing logic in sequence:

1. Load `ChromaMethod` from `method_path`
2. Load data via `DataHandler.load_data_directory(data_path)`
3. Call `ChromatogramProcessor.process()` with `method.to_processor_params()`
4. Call `processor.integrate_peaks()`
5. Write JSON output via the export logic (see Section 5)
6. Return peak table + output file paths

### Response

```json
{
  "status": "complete",
  "data_path": "...",
  "method": "...",
  "peak_count": 12,
  "peaks": [ { "rt": 2.31, "area": 45230.1, ... } ],
  "output_files": ["/path/to/results/sample.json"]
}
```

### Existing Endpoints

`/api/load`, `/api/process`, `/api/integrate` are unchanged and remain available for clients that want step-by-step control. `/api/run` is a convenience wrapper for the automation use case only.

---

## Section 5: Export Endpoint

The currently-stubbed `POST /api/export` endpoint is implemented as part of this work. It wraps the existing `logic/json_exporter` (the same exporter used by the GUI's trigger-based exports), ensuring the programmatic and GUI output formats are identical.

`POST /api/run` calls this internally. It can also be called standalone by clients that ran the pipeline step-by-step via the individual endpoints.

---

## Section 6: Reference File Watcher

`tools/watch_and_run.py` — a thin standalone script, not part of ChromaKit core.

- Uses `watchdog` to monitor a directory for new `.D` files
- On detection, calls `POST /api/run` with a specified method file
- Prints results to stdout; writes nothing itself (ChromaKit handles output)
- Serves as a starting point for users building automation loops

Dependencies: `watchdog`, `requests` (both optional/external, not added to `setup.py`).

---

## Out of Scope (this phase)

- MS library search via API (Phase 2)
- Quantitation methods via API (Phase 3)
- ChromaKit writing commands or setpoints back to any instrument or control system
- WebSocket / streaming progress for long-running runs (batch use cases only; single-sample runs are fast enough for polling)
- Multi-user or remote deployment considerations (API remains single-user local)

---

## File Inventory

| File | Change |
|------|--------|
| `logic/method.py` | **New** — `ChromaMethod` + all param models |
| `api/models.py` | **Modify** — import param models from `logic/method.py`; add `RunRequest`, `RunResponse`, `ExportRequest` |
| `api/main.py` | **Modify** — add `POST /api/run`, implement `POST /api/export` |
| `ui/frames/parameters.py` | **Modify** — add Save/Load Method buttons |
| `tools/` | **New directory** |
| `tools/watch_and_run.py` | **New** — reference watcher script |
