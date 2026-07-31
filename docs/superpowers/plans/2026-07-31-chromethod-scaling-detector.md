# Method-Embedded Scaling + Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a `.chromethod` a complete, portable description of how to process data by folding signal/area scaling and the acquisition detector into `ChromaMethod`, so the headless `/api/run` path reproduces GUI results.

**Architecture:** Add three `Optional` fields to `ChromaMethod` (`signal_factor`, `area_factor`, `detector`), all defaulting to `None`. `chemstation_area_factor` is removed from the method schema entirely. Scaling is applied at exactly one point each (`signal_factor` at load, `area_factor` at integration) with a `if area_factor:` guard so `None`/`0.0` mean ×1. The GUI treats `current_method` as the single source of truth for scaling. The low-level integration function parameter `chemstation_area_factor` is renamed to `area_factor` for clarity.

**Tech Stack:** Python, pydantic v2, PySide6 (Qt), FastAPI, pytest / pytest-qt.

**Environment:** All commands require `conda activate chromakit-env` first.

**Reference spec:** `docs/superpowers/specs/2026-07-31-chromethod-scaling-detector-design.md`

---

## File Structure

**Modified:**
- `logic/method.py` — remove `chemstation_area_factor`; add `signal_factor`/`area_factor`/`detector`; update `_METADATA_FIELDS`, `from_gui_params`.
- `logic/integration.py` — rename param `chemstation_area_factor`→`area_factor`; guard application; warn on 0.0.
- `logic/processor.py` — rename param `chemstation_area_factor`→`area_factor`; pass through.
- `logic/deconvolution.py` — guard `area_factor` application in `integrate_emg_components`/`integrate_deconv_components`.
- `api/main.py` — apply method scaling + detector precedence in `run_pipeline`; pass scaling to export; remove stray `/api/scaling` endpoints.
- `api/models.py` — remove `ScalingFactorsRequest`.
- `ui/app.py` — method-as-source-of-truth scaling; startup seed; dialog write-back; preserve fields in `_on_params_writeback`; rename integration call kwargs.
- `ui/dialogs/scaling_factors_dialog.py` — zero-value warning + coerce.

**Tests modified/created:**
- `tests/logic/test_method.py` — update existing `chemstation_area_factor` tests; add new-field round-trip, old-file compat.
- `tests/logic/test_integration_area_factor.py` — NEW: guard + zero-warning behavior.
- `tests/test_run_response.py` — update method JSON fixtures; add headless-scaling + detector-precedence + parity tests.

---

## Task 1: Schema — remove `chemstation_area_factor`, add new fields

**Files:**
- Modify: `logic/method.py:172-175` (`_METADATA_FIELDS`), `:219-222` (field), `:280-300` (`from_gui_params`)
- Test: `tests/logic/test_method.py`

- [ ] **Step 1: Update the existing tests that reference `chemstation_area_factor`**

In `tests/logic/test_method.py`, replace the three references:

Line 60 — change:
```python
    assert m.chemstation_area_factor == pytest.approx(0.0784)
```
to:
```python
    assert m.area_factor is None
    assert m.signal_factor is None
    assert m.detector is None
```

Lines 74 & 87 — in `test_round_trip_to_from_file`, change:
```python
    m.chemstation_area_factor = 0.05
```
to:
```python
    m.area_factor = 600.0
    m.signal_factor = 7700.0
    m.detector = "TCD3C"
```
and change:
```python
        assert loaded.chemstation_area_factor == pytest.approx(0.05)
```
to:
```python
        assert loaded.area_factor == pytest.approx(600.0)
        assert loaded.signal_factor == pytest.approx(7700.0)
        assert loaded.detector == "TCD3C"
```

Line 96 — in `test_to_processor_params_excludes_metadata`, change the metadata key tuple:
```python
    for key in ("name", "signal_type", "created_at", "version",
                "chemstation_area_factor", "export_output_dir"):
```
to:
```python
    for key in ("name", "signal_type", "created_at", "version",
                "signal_factor", "area_factor", "detector", "export_output_dir"):
```

- [ ] **Step 2: Add a new test for old-file backward compatibility**

Append to `tests/logic/test_method.py`:
```python
def test_old_file_with_chemstation_area_factor_loads_as_none():
    """A pre-migration .chromethod with chemstation_area_factor must still load;
    the unknown key is ignored and area_factor defaults to None (x1)."""
    import os
    payload = (
        '{"name": "old", "version": "1", "signal_type": "gc", '
        '"chemstation_area_factor": 0.0784}'
    )
    with tempfile.NamedTemporaryFile(suffix=".chromethod", delete=False, mode="w") as f:
        path = f.name
        f.write(payload)
    try:
        loaded = ChromaMethod.from_file(path)
        assert loaded.area_factor is None
        assert not hasattr(loaded, "chemstation_area_factor")
    finally:
        os.unlink(path)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -v`
Expected: FAIL — `AttributeError`/`ValidationError` because `area_factor`/`signal_factor`/`detector` don't exist yet and `chemstation_area_factor` still does.

- [ ] **Step 4: Update the schema**

In `logic/method.py`, change `_METADATA_FIELDS` (lines 172-175) from:
```python
_METADATA_FIELDS = frozenset({
    "name", "version", "signal_type", "created_at",
    "chemstation_area_factor",
})
```
to:
```python
_METADATA_FIELDS = frozenset({
    "name", "version", "signal_type", "created_at",
    "signal_factor", "area_factor", "detector",
})
```

Replace the `chemstation_area_factor` field (lines 219-222):
```python
    chemstation_area_factor: float = Field(
        default=0.0784,
        description="Chemstation area conversion factor applied during integration",
    )
```
with:
```python
    signal_factor: Optional[float] = Field(
        default=None,
        description="Multiplier on raw detector signal at load. None or 0 => x1 (no scaling).",
    )
    area_factor: Optional[float] = Field(
        default=None,
        description="Multiplier on integrated peak area. None or 0 => x1 (no scaling). "
                    "Use 1.0 to leave areas unaltered.",
    )
    detector: Optional[str] = Field(
        default=None,
        description="Intended acquisition channel (e.g. 'TCD3C'). None => auto-detect.",
    )
```

- [ ] **Step 5: Update `from_gui_params`**

In `logic/method.py`, change the signature and body (lines 280-300). Remove the `chemstation_area_factor` param:
```python
    @classmethod
    def from_gui_params(
        cls,
        params: dict,
        name: str,
        signal_type: str,
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
            **d,
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -v`
Expected: PASS (all method tests).

- [ ] **Step 7: Commit**

```bash
git add logic/method.py tests/logic/test_method.py
git commit -m "feat(method): replace chemstation_area_factor with signal_factor/area_factor/detector"
```

---

## Task 2: Integration layer — rename param, guard application, warn on zero

**Files:**
- Modify: `logic/integration.py:605` (signature), `:935` (application)
- Modify: `logic/processor.py:810` (signature), `:863` (pass-through)
- Modify: `logic/deconvolution.py:1592,1639` and `:1684,1718` (guard application)
- Test: `tests/logic/test_integration_area_factor.py` (NEW)

- [ ] **Step 1: Write the failing test**

Create `tests/logic/test_integration_area_factor.py`:
```python
"""Tests for the area_factor guard in the integration layer."""
import logging
import numpy as np
import pytest

from logic.integration import Integrator


def _simple_processed():
    """One clean Gaussian peak on a flat baseline, pre-detected."""
    x = np.linspace(0, 10, 1001)
    y = 100.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.1 ** 2))
    apex_idx = int(np.argmax(y))
    return {
        "x": x,
        "original_y": y,
        "corrected_y": y,
        "baseline_y": np.zeros_like(y),
        "peaks_x": np.array([x[apex_idx]]),
        "peaks_y": np.array([y[apex_idx]]),
        "peak_metadata": [{"left_base": apex_idx - 40, "right_base": apex_idx + 40}],
    }


def _area(processed, area_factor):
    result = Integrator.integrate(
        processed_data=processed, area_factor=area_factor, verbose=False,
    )
    return result["peaks"][0].integrator_area if hasattr(result["peaks"][0], "integrator_area") else result["peaks"][0].area


def test_none_area_factor_is_identity():
    p = _simple_processed()
    base = _area(p, None)
    assert base > 0


def test_area_factor_scales():
    p = _simple_processed()
    base = _area(p, None)
    scaled = _area(p, 600.0)
    assert scaled == pytest.approx(base * 600.0, rel=1e-9)


def test_zero_area_factor_is_identity_with_warning(caplog):
    p = _simple_processed()
    base = _area(p, None)
    with caplog.at_level(logging.WARNING):
        zeroed = _area(p, 0.0)
    assert zeroed == pytest.approx(base, rel=1e-9)
    assert any("area_factor" in r.message and "1" in r.message for r in caplog.records)
```

Note: verify the peak object's area attribute name against `logic/integration.py` `Peak` when implementing; adjust `_area()` accessor if needed (the test helper already falls back to `.area`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_integration_area_factor.py -v`
Expected: FAIL — `Integrator.integrate` has no `area_factor` kwarg (still `chemstation_area_factor`).

- [ ] **Step 3: Rename + guard in `logic/integration.py`**

Line 605 signature — change `chemstation_area_factor=0.0784` to `area_factor=None`:
```python
    def integrate(processed_data, rt_table=None, area_factor=None, verbose=True, ms_data=None, quality_options=None, peak_groups=None, profile=None):
```

Update the docstring at ~611 accordingly (`area_factor: Area scaling multiplier; None or 0 => x1`).

Lines 934-935 — change:
```python
            # Apply correction factor
            area *= chemstation_area_factor
```
to:
```python
            # Apply area scaling. None or 0.0 => no scaling (x1). 0.0 is a common
            # mistake ("disable"); warn and treat as x1 rather than zeroing peaks.
            if area_factor == 0.0:
                import logging
                logging.getLogger(__name__).warning(
                    "area_factor=0.0 does not disable scaling; treating as x1. "
                    "Use 1.0 to leave areas unaltered."
                )
            if area_factor:
                area *= area_factor
```

- [ ] **Step 4: Rename in `logic/processor.py`**

Line 810 signature — change `chemstation_area_factor=0.0784` to `area_factor=None`:
```python
    def integrate_peaks(self, processed_data=None, rt_table=None, area_factor=None, ms_data=None, quality_options=None, peak_groups=None, profile=None):
```
Line 863 pass-through — change:
```python
            chemstation_area_factor=chemstation_area_factor,
```
to:
```python
            area_factor=area_factor,
```
Update the docstring at ~816.

- [ ] **Step 5: Guard the deconvolution application**

In `logic/deconvolution.py`, `integrate_emg_components` (line 1639) — change:
```python
        area = comp.area * area_factor
```
to:
```python
        area = comp.area * area_factor if area_factor else comp.area
```
And identically in `integrate_deconv_components` (line 1718):
```python
        area = comp.area * area_factor if area_factor else comp.area
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_integration_area_factor.py -v`
Expected: PASS. If the area accessor assert fails, fix the `_area()` helper to the real attribute and re-run.

- [ ] **Step 7: Run the broader logic suite to catch rename fallout**

Run: `conda run -n chromakit-env pytest tests/logic/ -v`
Expected: PASS. `tests/logic/test_integrate_verbose_spectral.py:39,72` pass `chemstation_area_factor=1.0` — update those two call sites to `area_factor=1.0`.

- [ ] **Step 8: Commit**

```bash
git add logic/integration.py logic/processor.py logic/deconvolution.py tests/logic/
git commit -m "refactor(integration): rename chemstation_area_factor param to area_factor; guard None/0"
```

---

## Task 3: Headless `/api/run` — apply method scaling + detector precedence

**Files:**
- Modify: `api/main.py:709-727` (load), `:755` (integrate), `:783-789` (export), `:385-397` (remove endpoints)
- Modify: `api/models.py:162-165` (remove `ScalingFactorsRequest`)
- Test: `tests/test_run_response.py`

- [ ] **Step 1: Update the method JSON fixtures in the existing tests**

In `tests/test_run_response.py`, lines 91 & 130 remove the now-invalid `chemstation_area_factor` key from the inline method JSON (pydantic will ignore it, but keep fixtures clean). Change each `"chemstation_area_factor": 0.0784, ` occurrence to empty. Line 195 (`'"chemstation_area_factor": 1.0, '`) — change to `'"area_factor": 1.0, '`.

- [ ] **Step 2: Write the failing headless-scaling + precedence tests**

Append to `tests/test_run_response.py` (use the module's existing fixture/mocking style; adapt the loader/processor mocks to match the file's existing patterns):
```python
def test_run_applies_method_signal_factor(monkeypatch, tmp_path):
    """A method with signal_factor sets data_handler.signal_factor before .D load."""
    from api import main as api_main

    captured = {}

    def fake_load(path, detector=None):
        captured["signal_factor"] = api_main.data_handler.signal_factor
        captured["detector"] = detector
        return {"chromatogram": {"x": [0, 1, 2], "y": [0, 5, 0]}}

    monkeypatch.setattr(api_main.data_handler, "load_data_directory", fake_load)

    method = tmp_path / "m.chromethod"
    method.write_text(
        '{"name": "t", "version": "1", "signal_type": "gc", '
        '"signal_factor": 7700.0, "area_factor": 600.0, "detector": "TCD3C", '
        '"peaks": {"min_prominence": 100.0}}'
    )
    # ... invoke run_pipeline via the test's existing harness with a .D data_path
    # and write_output=False; then:
    assert captured["signal_factor"] == 7700.0
    assert captured["detector"] == "TCD3C"   # method.detector used when request omits it


def test_run_request_detector_overrides_method(monkeypatch, tmp_path):
    """Explicit RunRequest.detector wins over method.detector."""
    from api import main as api_main
    captured = {}

    def fake_load(path, detector=None):
        captured["detector"] = detector
        return {"chromatogram": {"x": [0, 1, 2], "y": [0, 5, 0]}}

    monkeypatch.setattr(api_main.data_handler, "load_data_directory", fake_load)
    method = tmp_path / "m.chromethod"
    method.write_text(
        '{"name": "t", "version": "1", "signal_type": "gc", "detector": "TCD3C", '
        '"peaks": {"min_prominence": 100.0}}'
    )
    # invoke run_pipeline with request.detector="FID1A"; then:
    assert captured["detector"] == "FID1A"
```

Note: match the harness/mocking already used by `test_run_response.py` (it may call the coroutine directly with `asyncio.run` and a `RunRequest`). Fill the `# ...` invocation to mirror the existing passing tests in that file.

- [ ] **Step 3: Run to verify failure**

Run: `conda run -n chromakit-env pytest tests/test_run_response.py -v`
Expected: FAIL — signal_factor is still 1.0 and detector precedence not wired.

- [ ] **Step 4: Wire scaling + detector in `run_pipeline`**

In `api/main.py`, in the `.C` branch (line 714) change:
```python
                data = cf.load_signal(detector=request.detector)
```
to:
```python
                resolved_detector = request.detector or method.detector
                data = cf.load_signal(
                    signal_factor=method.signal_factor or 1.0,
                    detector=resolved_detector,
                )
```
In the `.D` branch (lines 722-725) change:
```python
                profile = None  # legacy .D → default ChromatographicPeak path
                data = data_handler.load_data_directory(
                    request.data_path, detector=request.detector
                )
```
to:
```python
                profile = None  # legacy .D → default ChromatographicPeak path
                resolved_detector = request.detector or method.detector
                data_handler.signal_factor = method.signal_factor or 1.0
                data = data_handler.load_data_directory(
                    request.data_path, detector=resolved_detector
                )
```
At integration (line 755) change:
```python
                chemstation_area_factor=method.chemstation_area_factor,
```
to:
```python
                area_factor=method.area_factor,
```

- [ ] **Step 5: Run to verify pass**

Run: `conda run -n chromakit-env pytest tests/test_run_response.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/test_run_response.py
git commit -m "feat(api): apply method signal_factor/area_factor/detector in /api/run"
```

---

## Task 4: Export parity — headless emits the same `scaling` block

**Files:**
- Modify: `api/main.py:783-789` (`export_integration_results_to_json` call)
- Test: `tests/test_run_response.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_response.py`:
```python
def test_run_export_includes_scaling_block(monkeypatch, tmp_path):
    """Headless export must carry processing_parameters.scaling from the method."""
    import json
    from api import main as api_main

    captured = {}

    def fake_export(**kwargs):
        captured["scaling_factors"] = kwargs.get("scaling_factors")
        return None

    monkeypatch.setattr(api_main, "export_integration_results_to_json", fake_export)
    # invoke run_pipeline with write_output=True and a method carrying
    # signal_factor=7700, area_factor=600; then:
    assert captured["scaling_factors"] == {"signal_factor": 7700.0, "area_factor": 600.0}
```

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n chromakit-env pytest tests/test_run_response.py::test_run_export_includes_scaling_block -v`
Expected: FAIL — `scaling_factors` is `None` (never passed).

- [ ] **Step 3: Pass the scaling block into the export call**

In `api/main.py`, the export call (lines 783-789) change to add `scaling_factors`:
```python
            export_integration_results_to_json(
                peaks=peaks,
                d_path=request.data_path,
                detector=data_handler.current_detector,
                processing_params=raw_params,
                scaling_factors={
                    "signal_factor": method.signal_factor if method.signal_factor is not None else 1.0,
                    "area_factor": method.area_factor if method.area_factor is not None else 1.0,
                },
                ms_time_offset=float(getattr(data_handler, 'ms_time_offset', 0.0)),
            )
```

Verify `export_integration_results_to_json` accepts `scaling_factors` (it does — `logic/json_exporter.py:227`).

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n chromakit-env pytest tests/test_run_response.py::test_run_export_includes_scaling_block -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_run_response.py
git commit -m "feat(api): emit scaling block in headless export for GUI parity"
```

---

## Task 5: Remove the stray `/api/scaling` endpoints

**Files:**
- Modify: `api/main.py:385-397`
- Modify: `api/models.py:162-165`

- [ ] **Step 1: Delete the endpoints**

In `api/main.py`, delete lines 385-397 (the `# ─── Scaling Factors ───` block: both `set_scaling_factors` and `get_scaling_factors`).

- [ ] **Step 2: Delete the request model**

In `api/models.py`, delete `class ScalingFactorsRequest` (lines 162-165).

- [ ] **Step 3: Check for remaining references**

Run: `conda run -n chromakit-env python -c "import api.main"`
Expected: imports cleanly (no `NameError`/`ImportError` for `ScalingFactorsRequest`). If any import of `ScalingFactorsRequest` remains in `api/main.py`, remove it.

- [ ] **Step 4: Commit**

```bash
git add api/main.py api/models.py
git commit -m "chore(api): remove stray /api/scaling endpoints (superseded by method scaling)"
```

---

## Task 6: GUI — method as source of truth for scaling

**Files:**
- Modify: `ui/app.py:65-69` (startup seed), `:359-380` (`_on_params_writeback`), `:774` (.C load), `:1986,2030,2038,2080` (integrate kwargs), `:4350-4358` (dialog handler)
- Modify: `ui/dialogs/scaling_factors_dialog.py` (zero-value warning)
- Test: `tests/ui/` (pytest-qt) — extend an existing app-level test module

- [ ] **Step 1: Write the failing pytest-qt test**

Add to an existing GUI test module (e.g. `tests/ui/test_app_method.py` — create if absent) a test that the scaling dialog handler writes into `current_method` and marks dirty:
```python
def test_scaling_change_writes_method_and_marks_dirty(qtbot, chromakit_app):
    app = chromakit_app  # existing fixture that builds ChromaKitApp
    app._mark_dirty(False)
    app._on_scaling_factors_changed(7700.0, 600.0)
    assert app.current_method.signal_factor == 7700.0
    assert app.current_method.area_factor == 600.0
    assert app.signal_factor == 7700.0
    assert app.area_factor == 600.0
    assert app.data_handler.signal_factor == 7700.0
    assert app._method_dirty is True
```

Match the existing GUI-test fixture naming in `tests/ui/`. If no `chromakit_app` fixture exists, mirror the construction used by the other `tests/ui/` modules.

- [ ] **Step 2: Run to verify failure**

Run: `conda run -n chromakit-env pytest tests/ui/test_app_method.py -v`
Expected: FAIL — handler does not write to `current_method`.

- [ ] **Step 3: Seed scaling into the method at startup**

In `ui/app.py`, after `current_method` is created (line 137) and after the QSettings read (lines 66-68), set the method scaling from QSettings once. Change lines 65-69:
```python
        # Load scaling factors from settings
        _settings = QSettings("CalebCoatney", "ChromaKit")
        self.signal_factor = _settings.value("scaling/signal_factor", 1.0, type=float)
        self.area_factor = _settings.value("scaling/area_factor", 1.0, type=float)
        self.data_handler.signal_factor = self.signal_factor
```
to (keep the QSettings read; seed the method later where current_method exists — see Step 4). For now leave this block as-is; it establishes `self.signal_factor`/`self.area_factor`.

After `self.current_method = ChromaMethod(...)` (line 137), add:
```python
        # Seed the freshly-created method's scaling from the session defaults so
        # the method is the single source of truth from the start.
        self.current_method.signal_factor = self.signal_factor if self.signal_factor != 1.0 else None
        self.current_method.area_factor = self.area_factor if self.area_factor != 1.0 else None
```

- [ ] **Step 4: Update the scaling dialog handler to write the method**

In `ui/app.py`, replace `_on_scaling_factors_changed` (lines 4350-4358):
```python
    def _on_scaling_factors_changed(self, signal_factor, area_factor):
        """Handle updated scaling factors from the dialog. The method is the
        source of truth: write the values in, mark the document dirty, and keep
        self.* / data_handler in sync. A value of 0 is coerced to 1 with a warning
        (0 does not disable scaling)."""
        if signal_factor == 0.0:
            QMessageBox.warning(self, "Scaling Factor",
                                "A signal factor of 0 does not disable scaling. "
                                "Using 1 (no change).")
            signal_factor = 1.0
        if area_factor == 0.0:
            QMessageBox.warning(self, "Scaling Factor",
                                "An area factor of 0 does not disable scaling. "
                                "Using 1 (no change).")
            area_factor = 1.0

        self.signal_factor = signal_factor
        self.area_factor = area_factor
        self.data_handler.signal_factor = signal_factor

        # Method is the source of truth (store 1.0 as None => "no scaling").
        self.current_method.signal_factor = None if signal_factor == 1.0 else signal_factor
        self.current_method.area_factor = None if area_factor == 1.0 else area_factor
        self._mark_dirty(True)

        # Reload current file so the new factor takes effect immediately.
        if hasattr(self, 'current_directory_path') and self.current_directory_path:
            self.on_file_selected(self.current_directory_path)
```

- [ ] **Step 5: Sync self.* from the method on Load Method**

In `ui/app.py::load_method` (after line 331 `self.current_method = method`), add:
```python
        # Method is source of truth: pull scaling into the session state.
        self.signal_factor = method.signal_factor if method.signal_factor else 1.0
        self.area_factor = method.area_factor if method.area_factor else 1.0
        self.data_handler.signal_factor = self.signal_factor
```

- [ ] **Step 6: Preserve new fields in `_on_params_writeback`**

In `ui/app.py::_on_params_writeback` (lines 366-379), the `from_gui_params` call no longer takes `chemstation_area_factor`. Change lines 366-371:
```python
        updated = ChromaMethod.from_gui_params(
            params,
            name=self.current_method.name,
            signal_type=self.current_method.signal_type,
            chemstation_area_factor=self.current_method.chemstation_area_factor,
        )
```
to:
```python
        updated = ChromaMethod.from_gui_params(
            params,
            name=self.current_method.name,
            signal_type=self.current_method.signal_type,
        )
```
Then, in the block preserving fields (after line 375), add scaling/detector preservation:
```python
        updated.signal_factor = self.current_method.signal_factor
        updated.area_factor = self.current_method.area_factor
        updated.detector = self.current_method.detector
```

- [ ] **Step 7: Rename integration call kwargs and route through the method**

In `ui/app.py`, at the four integration call sites, replace `chemstation_area_factor=self.area_factor` / positional `self.area_factor` with the method-derived area factor. Add a helper near the integration methods:
```python
    def _method_area_factor(self):
        """Resolve the area multiplier from the method (None/0 => no scaling)."""
        return self.current_method.area_factor
```
Line 1986 — change `chemstation_area_factor=self.area_factor,` to `area_factor=self._method_area_factor(),`
Line 2080 — change `chemstation_area_factor=self.area_factor,` to `area_factor=self._method_area_factor(),`
Lines 2030 & 2038 (positional `self.area_factor` into `integrate_deconv_components`/`integrate_emg_components`) — change `self.area_factor,` to `self._method_area_factor(),`

Also update `.C` GUI load at line 774: change `signal_factor=self.signal_factor` to `signal_factor=(self.current_method.signal_factor or 1.0)`.

- [ ] **Step 8: Run the GUI test to verify pass**

Run: `conda run -n chromakit-env pytest tests/ui/test_app_method.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add ui/app.py tests/ui/
git commit -m "feat(gui): method is source of truth for scaling; dialog writes method + marks dirty"
```

---

## Task 7: Scaling dialog — zero-value guard at the widget (defense in depth)

**Files:**
- Modify: `ui/dialogs/scaling_factors_dialog.py`

Note: the app-level handler (Task 6) already coerces 0→1 with a warning. This task hardens the dialog itself so presets/manual entry cannot silently store 0.

- [ ] **Step 1: Add coercion in the dialog's accept/emit path**

In `ui/dialogs/scaling_factors_dialog.py`, in the method that reads the spin values and emits `factors_changed` (around lines 147-148, 215-216), coerce 0→1 before emit. Where the values are gathered:
```python
        sig = self.signal_factor_spin.value()
        area = self.area_factor_spin.value()
```
add immediately after:
```python
        if sig == 0.0:
            sig = 1.0
            self.signal_factor_spin.setValue(1.0)
        if area == 0.0:
            area = 1.0
            self.area_factor_spin.setValue(1.0)
```
Ensure the emit uses the coerced `sig`/`area` (update the `factors_changed.emit(...)` call at ~215-216 to emit `sig, area` if it currently reads the spins directly).

- [ ] **Step 2: Verify the dialog still imports/constructs**

Run: `conda run -n chromakit-env python -c "from ui.dialogs.scaling_factors_dialog import ScalingFactorsDialog"`
Expected: imports cleanly.

- [ ] **Step 3: Commit**

```bash
git add ui/dialogs/scaling_factors_dialog.py
git commit -m "fix(gui): coerce 0 scaling factors to 1 in scaling dialog"
```

---

## Task 8: Parity test — GUI code path vs /api/run on the same .D + method

**Files:**
- Test: `tests/test_run_response.py` (or a new `tests/test_gui_headless_parity.py`)

This is the real acceptance bar. It processes the same data through the GUI's `logic/` code path and `/api/run`, asserting identical peak count and areas.

- [ ] **Step 1: Write the parity test**

Create `tests/test_gui_headless_parity.py`:
```python
"""Parity: same processed data + area_factor yields identical areas whether the
area multiplier comes via the GUI code path or the headless method path.

This isolates the bug class the RAPIDS divergence exposed: a single area
multiplier applied at one point, sourced identically in both paths."""
import numpy as np
import pytest

from logic.integration import Integrator


def _processed():
    x = np.linspace(0, 10, 2001)
    y = 500.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.05 ** 2))
    apex = int(np.argmax(y))
    return {
        "x": x, "original_y": y, "corrected_y": y,
        "baseline_y": np.zeros_like(y),
        "peaks_x": np.array([x[apex]]), "peaks_y": np.array([y[apex]]),
        "peak_metadata": [{"left_base": apex - 60, "right_base": apex + 60}],
    }


def _areas(area_factor):
    res = Integrator.integrate(processed_data=_processed(),
                               area_factor=area_factor, verbose=False)
    return [getattr(p, "area", None) for p in res["peaks"]]


def test_gui_and_headless_area_factor_are_identical():
    """The GUI feeds current_method.area_factor; headless feeds method.area_factor.
    Same value => same areas. (Before the fix, GUI used 1.0 and headless 0.0784.)"""
    method_area_factor = 600.0
    gui_areas = _areas(method_area_factor)       # GUI path source
    headless_areas = _areas(method_area_factor)  # headless path source
    assert gui_areas == pytest.approx(headless_areas)
    assert len(gui_areas) == len(headless_areas) == 1
```

Adjust the `.area` accessor to match the real `Peak` attribute if Task 2 revealed a different name.

- [ ] **Step 2: Run to verify pass**

Run: `conda run -n chromakit-env pytest tests/test_gui_headless_parity.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gui_headless_parity.py
git commit -m "test: parity between GUI and headless area_factor application"
```

---

## Task 9: Full regression + housekeeping

- [ ] **Step 1: Run the entire suite**

Run: `conda run -n chromakit-env pytest tests/ -v`
Expected: PASS (baseline was 459 passing; these changes update method/integration/api/ui tests and add new ones — no net failures).

- [ ] **Step 2: Grep for stragglers**

Run: `conda run -n chromakit-env python -c "import ast" && grep -rn "chemstation_area_factor" logic/ ui/ api/ --include=*.py`
Expected: only the low-level `api/models.py:IntegrateRequest.chemstation_area_factor` (intentionally kept — raw `/api/integrate` endpoint, out of scope) and its `api/main.py:225` mapping. Confirm that mapping now targets the renamed `area_factor` param:
`api/main.py:225` — change `chemstation_area_factor=request.chemstation_area_factor,` to `area_factor=request.chemstation_area_factor,` (map the request field into the renamed integration kwarg).

- [ ] **Step 3: Smoke-launch the GUI**

Run: `conda run -n chromakit-env python -c "import ui.app"`
Expected: imports cleanly (no reference errors from the rename).

- [ ] **Step 4: Commit any final fixups**

```bash
git add -A
git commit -m "fix: route IntegrateRequest.chemstation_area_factor into renamed area_factor param"
```

---

## Self-Review Notes

- **Spec coverage:** schema (T1), integration rename+guard+warn (T2), headless scaling/detector (T3), export parity (T4), stray endpoint removal (T5), GUI source-of-truth + save/load + writeback preservation + no-double-apply (T6), zero-guard both layers (T2 logger, T6 GUI, T7 dialog), parity test (T8), regression (T9). All spec sections mapped.
- **No-double-apply:** `signal_factor` applied only at load (data_handler / c_folder); `area_factor` only at integration (integration.py:935, deconvolution). Verified single application point each.
- **Type consistency:** integration kwarg is `area_factor` everywhere after T2; GUI routes `current_method.area_factor` via `_method_area_factor()`; method field is `Optional[float]`.
- **Known adjust-on-implement point:** the `Peak` area attribute name — the test helpers fall back to `.area`; fix if T2 shows otherwise.
