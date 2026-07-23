# Fixed-Window Band Integration for Spectroscopy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-window band-integration mode to ChromaKit so a `.chromethod` can declare named x-windows and `/api/run` returns one `SpectralFeature` per band (independent of peak detection), plus the guard/validation fixes that stop spectroscopy methods from 500ing.

**Architecture:** New `BandWindow` model + `bands` list on `ChromaMethod`; new `baseline.enabled` toggle; a pure `integrate_bands()` function in `logic/integration.py`; a bands branch in `/api/run` that short-circuits peak detection/rt-matching/quant when bands are present; defensive guards in `apply_rt_matching` and `_apply_peak_grouping`; a `min_prominence` non-null validator.

**Tech Stack:** Python 3.13, Pydantic v2, NumPy, SciPy (`simpson`), pytest. Conda env `chromakit-env`.

**Spec:** `docs/superpowers/specs/2026-07-23-spectroscopy-band-integration-design.md`

**Conventions verified in repo:**
- `processor.process()` returns dict with keys `x`, `original_y`, `smoothed_y`, `baseline_y`, `corrected_y`, `peaks_x`, `peaks_y`, `peak_metadata` (`logic/processor.py:579`).
- Baseline always runs at `logic/processor.py:460-471`.
- `SpectralFeature(feature_id, position, position_units, area, width, start, end, start_index, end_index, is_shoulder=False, is_negative=False, quality_issues=None, band_assignment="", absorbance=0.0, transmittance=0.0)` (`logic/feature.py:65`).
- `simpson` imported at `logic/integration.py:2`.
- `/api/run` pipeline at `api/main.py:686`; band branch goes after `processed = processor.process(...)` (`:737`) and replaces the `integrate_peaks`/rt-assign/quant block (`:740-767`).
- Tests live in `tests/logic/` and `tests/api/`; plain pytest, direct imports.

**Run all tests with:** `conda run -n chromakit-env pytest tests/ -q`

---

### Task 1: `BandWindow` model + `bands` field on `ChromaMethod`

**Files:**
- Modify: `logic/method.py` (add `BandWindow`, add `bands` field to `ChromaMethod`)
- Test: `tests/logic/test_method.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/logic/test_method.py`:

```python
def test_bandwindow_valid():
    from logic.method import BandWindow
    b = BandWindow(name="np_broad", x_min=350.0, x_max=500.0)
    assert b.name == "np_broad"
    assert b.x_min == 350.0
    assert b.x_max == 500.0


def test_bandwindow_rejects_non_increasing():
    from logic.method import BandWindow
    with pytest.raises(ValueError):
        BandWindow(name="bad", x_min=500.0, x_max=350.0)


def test_bandwindow_rejects_equal_bounds():
    from logic.method import BandWindow
    with pytest.raises(ValueError):
        BandWindow(name="bad", x_min=400.0, x_max=400.0)


def test_method_bands_default_empty():
    m = ChromaMethod(name="m", signal_type="gcms")
    assert m.bands == []


def test_method_bands_roundtrip():
    from logic.method import BandWindow
    m = ChromaMethod(
        name="ir", signal_type="ftir",
        bands=[BandWindow(name="precursor_CO", x_min=1970, x_max=2005)],
    )
    js = m.model_dump_json(by_alias=True)
    m2 = ChromaMethod.model_validate_json(js)
    assert m2.bands[0].name == "precursor_CO"
    assert m2.bands[0].x_min == 1970
    assert m2.bands[0].x_max == 2005
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -k "band" -v`
Expected: FAIL with `ImportError: cannot import name 'BandWindow'`.

- [ ] **Step 3: Write minimal implementation**

In `logic/method.py`, add after the `RFTableEntry` class (around line 120):

```python
class BandWindow(BaseModel):
    """A named fixed x-window for spectroscopy band integration.

    x_min/x_max are in the signal profile's native x-units (cm-1 for FTIR,
    nm for UV-Vis). Bounds are stored ascending regardless of axis direction.
    """
    name: str = Field(..., description="Band name -> SpectralFeature.band_assignment")
    x_min: float = Field(..., description="Window lower bound (native x-units)")
    x_max: float = Field(..., description="Window upper bound (native x-units)")

    @field_validator("x_max")
    @classmethod
    def _check_bounds(cls, v: float, info) -> float:
        x_min = info.data.get("x_min")
        if x_min is not None and v <= x_min:
            raise ValueError(f"x_max ({v}) must be greater than x_min ({x_min})")
        return v
```

In `ChromaMethod`, add this field after `rf_table` (around line 173):

```python
    bands: List[BandWindow] = Field(
        default_factory=list,
        description="Fixed-window bands for spectroscopy integration. When "
                    "non-empty, band integration replaces peak detection.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -k "band" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add logic/method.py tests/logic/test_method.py
git commit -m "feat(method): add BandWindow model and ChromaMethod.bands field"
```

---

### Task 2: `baseline.enabled` toggle on `BaselineParams`

**Files:**
- Modify: `logic/method.py` (`BaselineParams`)
- Test: `tests/logic/test_method.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/logic/test_method.py`:

```python
def test_baseline_enabled_defaults_true():
    from logic.method import BaselineParams
    assert BaselineParams().enabled is True


def test_baseline_enabled_false_roundtrip():
    from logic.method import BaselineParams
    bp = BaselineParams(enabled=False)
    assert bp.enabled is False
    bp2 = BaselineParams.model_validate_json(bp.model_dump_json(by_alias=True))
    assert bp2.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -k "baseline_enabled" -v`
Expected: FAIL with `AttributeError` / validation error (no `enabled` field).

- [ ] **Step 3: Write minimal implementation**

In `logic/method.py`, `BaselineParams` (around line 49), add as the first field:

```python
    enabled: bool = Field(default=True, description="Run baseline correction. False integrates raw signal.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -k "baseline_enabled" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add logic/method.py tests/logic/test_method.py
git commit -m "feat(method): add BaselineParams.enabled toggle (default true)"
```

---

### Task 3: `min_prominence` non-null validator

**Files:**
- Modify: `logic/method.py` (`PeakParams`)
- Test: `tests/logic/test_method.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/logic/test_method.py`:

```python
def test_min_prominence_rejects_null():
    from logic.method import PeakParams
    with pytest.raises(ValueError):
        PeakParams(min_prominence=None)


def test_min_prominence_accepts_fractional():
    from logic.method import PeakParams
    assert PeakParams(min_prominence=0.02).min_prominence == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -k "min_prominence" -v`
Expected: `test_min_prominence_rejects_null` FAILS (currently `None` is allowed).

- [ ] **Step 3: Write minimal implementation**

In `logic/method.py`, `PeakParams` (around line 64), change the field type and add a validator. Replace:

```python
    min_prominence: Optional[float] = Field(default=1e5)
```

with:

```python
    min_prominence: float = Field(default=1e5, description="Prominence threshold. Must be non-null; values <=1 are treated as a fraction of signal range.")

    @field_validator("min_prominence")
    @classmethod
    def _min_prominence_not_null(cls, v):
        if v is None:
            raise ValueError(
                "min_prominence must not be null; use a fractional value "
                "(e.g. 0.02) for spectroscopy or a large value for chromatography."
            )
        return v
```

Confirm `field_validator` is already imported at top of `logic/method.py` (it is: line 17). If `Optional` becomes unused elsewhere, leave the import — other fields still use it.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -k "min_prominence" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run full method tests to confirm no regression**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add logic/method.py tests/logic/test_method.py
git commit -m "feat(method): reject null min_prominence with actionable error"
```

---

### Task 4: `integrate_bands()` pure function

**Files:**
- Modify: `logic/integration.py` (add module-level function)
- Test: `tests/logic/test_integrate_bands.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/logic/test_integrate_bands.py`:

```python
import numpy as np
import pytest

from logic.integration import integrate_bands
from logic.method import BandWindow
from logic.feature import SpectralFeature
from logic.signal_profiles import SignalProfileRegistry


def _uvvis_profile():
    return SignalProfileRegistry.get("uvvis")


def _ftir_profile():
    return SignalProfileRegistry.get("ftir")


def test_returns_one_feature_per_band_in_order():
    x = np.linspace(300, 600, 301)  # nm
    y = np.ones_like(x)
    bands = [
        BandWindow(name="a", x_min=350, x_max=400),
        BandWindow(name="b", x_min=450, x_max=500),
    ]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert len(feats) == 2
    assert all(isinstance(f, SpectralFeature) for f in feats)
    assert [f.band_assignment for f in feats] == ["a", "b"]


def test_area_of_flat_unit_band_equals_width():
    # integral of y=1 over [350,400] == 50
    x = np.linspace(300, 600, 3001)
    y = np.ones_like(x)
    bands = [BandWindow(name="a", x_min=350, x_max=400)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert feats[0].area == pytest.approx(50.0, rel=1e-3)


def test_absorbance_and_position_at_max():
    x = np.linspace(300, 500, 2001)
    y = np.zeros_like(x)
    # inject a spike at 400 nm
    idx = np.argmin(np.abs(x - 400))
    y[idx] = 2.5
    bands = [BandWindow(name="peak", x_min=350, x_max=450)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert feats[0].absorbance == pytest.approx(2.5, rel=1e-6)
    assert feats[0].position == pytest.approx(400.0, abs=0.2)


def test_inverted_x_ftir_area_is_positive():
    # FTIR x descends high->low; band still integrates positive
    x = np.linspace(2200, 1800, 2001)  # descending wavenumbers
    y = np.ones_like(x)
    bands = [BandWindow(name="co", x_min=1970, x_max=2005)]
    feats = integrate_bands(x, y, bands, _ftir_profile())
    assert feats[0].area == pytest.approx(35.0, rel=1e-2)
    assert feats[0].area > 0


def test_empty_window_emits_zero_feature_with_quality_issue():
    x = np.linspace(300, 500, 201)
    y = np.ones_like(x)
    bands = [BandWindow(name="offscale", x_min=800, x_max=900)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    assert len(feats) == 1
    assert feats[0].area == 0.0
    assert feats[0].absorbance == 0.0
    assert feats[0].band_assignment == "offscale"
    assert any("no samples" in q.lower() for q in feats[0].quality_issues)


def test_feature_bounds_and_units():
    x = np.linspace(300, 600, 301)
    y = np.ones_like(x)
    bands = [BandWindow(name="a", x_min=350, x_max=400)]
    feats = integrate_bands(x, y, bands, _uvvis_profile())
    f = feats[0]
    assert f.start == 350
    assert f.end == 400
    assert f.width == pytest.approx(50.0)
    assert f.position_units == "Wavelength (nm)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_integrate_bands.py -v`
Expected: FAIL with `ImportError: cannot import name 'integrate_bands'`.

- [ ] **Step 3: Write minimal implementation**

In `logic/integration.py`, add this module-level function (place it near the top after the imports/`ChromatographicPeak`, e.g. after line 6's imports — it needs `SpectralFeature`, import it lazily inside to avoid any ordering issues):

```python
def integrate_bands(x, y, bands, profile):
    """Integrate fixed x-windows, independent of peak detection.

    Args:
        x: 1-D array of x positions (native units; may be ascending or
           descending, e.g. inverted-x FTIR wavenumbers).
        y: 1-D array of the signal to integrate. Caller passes the
           baseline-corrected signal when baseline is enabled, else raw.
        bands: list of BandWindow (name, x_min, x_max) with x_min < x_max.
        profile: SignalProfile (used for position_units via x_label).

    Returns:
        list[SpectralFeature] — exactly one per band, in input order. A band
        with no samples in range yields a zero-area feature carrying a
        quality issue rather than being dropped.
    """
    from logic.feature import SpectralFeature

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    units = profile.x_label if profile is not None else ""

    features = []
    for i, band in enumerate(bands):
        x_min = min(band.x_min, band.x_max)
        x_max = max(band.x_min, band.x_max)
        mask = (x >= x_min) & (x <= x_max)
        idx = np.nonzero(mask)[0]

        if idx.size == 0:
            features.append(SpectralFeature(
                feature_id=i,
                position=(x_min + x_max) / 2.0,
                position_units=units,
                area=0.0, width=(x_max - x_min),
                start=x_min, end=x_max,
                start_index=-1, end_index=-1,
                band_assignment=band.name,
                absorbance=0.0,
                quality_issues=[f"no samples in [{x_min}, {x_max}]"],
            ))
            continue

        x_win = x[idx]
        y_win = y[idx]

        # Direction-safe integration: sort ascending by x before simpson.
        order = np.argsort(x_win)
        x_sorted = x_win[order]
        y_sorted = y_win[order]
        area = float(abs(simpson(y_sorted, x=x_sorted)))

        max_pos = int(np.argmax(y_win))
        absorbance = float(y_win[max_pos])
        position = float(x_win[max_pos])

        features.append(SpectralFeature(
            feature_id=i,
            position=position,
            position_units=units,
            area=area,
            width=(x_max - x_min),
            start=x_min, end=x_max,
            start_index=int(idx.min()), end_index=int(idx.max()),
            band_assignment=band.name,
            absorbance=absorbance,
        ))

    return features
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_integrate_bands.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add logic/integration.py tests/logic/test_integrate_bands.py
git commit -m "feat(integration): add integrate_bands fixed-window integrator"
```

---

### Task 5: Baseline `enabled` toggle honored in `processor.process()`

**Files:**
- Modify: `logic/processor.py` (`process`, around line 460-471)
- Test: `tests/logic/test_processor_baseline_toggle.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/logic/test_processor_baseline_toggle.py`:

```python
import numpy as np

from logic.processor import ChromatogramProcessor


def _params(baseline_enabled):
    return {
        "smoothing": {"enabled": False, "method": "whittaker", "lambda": 0.1, "diff_order": 1},
        "baseline": {"enabled": baseline_enabled, "method": "arpls", "lambda": 1e4},
        "peaks": {"enabled": False, "min_prominence": 0.02},
    }


def test_baseline_disabled_leaves_signal_uncorrected():
    x = np.linspace(0, 10, 500)
    y = 5.0 + np.exp(-((x - 5.0) ** 2) / 0.5)  # offset baseline of 5
    proc = ChromatogramProcessor()
    out = proc.process(x, y, params=_params(baseline_enabled=False))
    # corrected_y must equal the (unsmoothed) input, baseline all-zero
    np.testing.assert_allclose(out["corrected_y"], y, rtol=1e-9)
    np.testing.assert_allclose(out["baseline_y"], np.zeros_like(y))


def test_baseline_enabled_still_corrects():
    x = np.linspace(0, 10, 500)
    y = 5.0 + np.exp(-((x - 5.0) ** 2) / 0.5)
    proc = ChromatogramProcessor()
    out = proc.process(x, y, params=_params(baseline_enabled=True))
    # corrected signal should be pulled toward zero at the flat regions
    assert out["corrected_y"][0] < y[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_processor_baseline_toggle.py -v`
Expected: `test_baseline_disabled_leaves_signal_uncorrected` FAILS (baseline currently always runs).

- [ ] **Step 3: Write minimal implementation**

In `logic/processor.py`, `process()`, replace the STEP 2 block (lines 460-471):

```python
        # STEP 2: Always calculate baseline
        # Pass ms_range so the baseline is not fit through the solvent peak
        # (the pre-MS region of the FID is masked with NaN).
        baseline_y, baseline_corrected_y = self._apply_baseline_correction(
            x_values, smoothed_y,
            method=params['baseline']['method'],
            lam=params['baseline']['lambda'],
            fastchrom_params=params['baseline'].get('fastchrom'),
            break_points=params['baseline'].get('break_points', []),
            baseline_offset=params['baseline'].get('baseline_offset', 0.0),
            ms_range=ms_range,
        )
```

with:

```python
        # STEP 2: Baseline correction (skippable via baseline.enabled=False).
        # When disabled, corrected_y IS the smoothed signal and baseline is zero,
        # so downstream (band integration, plotting) has a well-defined corrected_y.
        if params['baseline'].get('enabled', True):
            # Pass ms_range so the baseline is not fit through the solvent peak
            # (the pre-MS region of the FID is masked with NaN).
            baseline_y, baseline_corrected_y = self._apply_baseline_correction(
                x_values, smoothed_y,
                method=params['baseline']['method'],
                lam=params['baseline']['lambda'],
                fastchrom_params=params['baseline'].get('fastchrom'),
                break_points=params['baseline'].get('break_points', []),
                baseline_offset=params['baseline'].get('baseline_offset', 0.0),
                ms_range=ms_range,
            )
        else:
            baseline_y = np.zeros_like(smoothed_y)
            baseline_corrected_y = np.copy(smoothed_y)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_processor_baseline_toggle.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add logic/processor.py tests/logic/test_processor_baseline_toggle.py
git commit -m "feat(processor): honor baseline.enabled=False (integrate raw signal)"
```

---

### Task 6: Guard `apply_rt_matching` against non-chromatographic features

**Files:**
- Modify: `logic/rt_matching.py` (`apply_rt_matching`, around line 104-108)
- Test: `tests/logic/test_rt_matching.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/logic/test_rt_matching.py`:

```python
def test_apply_rt_matching_skips_spectral_features():
    import pandas as pd
    from logic.rt_matching import apply_rt_matching
    from logic.method import RTMatchingParams
    from logic.feature import SpectralFeature

    sf = SpectralFeature(
        feature_id=1, position=1987.0, position_units="cm-1",
        area=10.0, width=5.0, start=1980.0, end=1994.0,
        start_index=0, end_index=10,
    )
    rt_df = pd.DataFrame(
        [["precursor", 1.0, 1.1, 1.2]],
        columns=["Compound", "Start", "Apex", "End"],
    )
    # Must NOT raise AttributeError on peak.retention_time
    apply_rt_matching([sf], rt_df, RTMatchingParams())
    assert not hasattr(sf, "compound_id") or getattr(sf, "compound_id", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_rt_matching.py -k spectral -v`
Expected: FAIL with `AttributeError: 'SpectralFeature' object has no attribute 'retention_time'`.

- [ ] **Step 3: Write minimal implementation**

In `logic/rt_matching.py`, `apply_rt_matching`, after the existing empty-table guard (lines 104-105), add a feature-type guard before the loop:

```python
    if rt_table is None or len(rt_table) == 0:
        return

    # RT matching is only meaningful for chromatographic peaks (which carry a
    # retention_time). Spectroscopy features (SpectralFeature) have no RT — skip
    # rather than raising AttributeError. Band naming for spectra is handled by
    # the bands mechanism, not rt_table.
    if not all(hasattr(p, "retention_time") for p in peaks):
        return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_rt_matching.py -k spectral -v`
Expected: PASS.

- [ ] **Step 5: Run full rt_matching tests**

Run: `conda run -n chromakit-env pytest tests/logic/test_rt_matching.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add logic/rt_matching.py tests/logic/test_rt_matching.py
git commit -m "fix(rt_matching): skip non-chromatographic features instead of crashing"
```

---

### Task 7: Guard `_apply_peak_grouping` against non-chromatographic features

**Files:**
- Modify: `logic/integration.py` (`_apply_peak_grouping`, around line 406-412)
- Test: `tests/logic/test_peak_grouping_guard.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/logic/test_peak_grouping_guard.py`:

```python
import numpy as np

from logic.integration import Integrator
from logic.feature import SpectralFeature


def test_peak_grouping_skips_spectral_features():
    sf = SpectralFeature(
        feature_id=1, position=1987.0, position_units="cm-1",
        area=10.0, width=5.0, start=1980.0, end=1994.0,
        start_index=0, end_index=10,
    )
    x = np.linspace(1800, 2200, 100)
    y = np.ones_like(x)
    baseline_y = np.zeros_like(x)
    # Must not raise AttributeError on peak.retention_time
    result = Integrator._apply_peak_grouping(
        [sf], [[1950.0, 2000.0]], x, y, baseline_y,
        [x], [y], [baseline_y], [1987.0], [10.0], [(1980.0, 1994.0)],
        profile=None,
    )
    # returns the peaks_list unchanged (first element of the returned tuple)
    assert result[0] == [sf]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/logic/test_peak_grouping_guard.py -v`
Expected: FAIL with `AttributeError: 'SpectralFeature' object has no attribute 'retention_time'`.

- [ ] **Step 3: Write minimal implementation**

In `logic/integration.py`, `_apply_peak_grouping`, add a guard immediately after the `feature_cls = ...` line (line 406), before `consumed_indices = set()`:

```python
        feature_cls = profile.feature_class if (profile is not None and profile.feature_class is not None) else Peak

        # Peak grouping keys on retention_time; it is only valid for
        # chromatographic peaks. Spectroscopy features have no RT — return the
        # inputs unchanged rather than raising AttributeError.
        if not all(hasattr(p, "retention_time") for p in peaks_list):
            return (peaks_list, x_peaks, y_peaks, baseline_peaks,
                    ret_times, integrated_areas, integration_bounds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/logic/test_peak_grouping_guard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add logic/integration.py tests/logic/test_peak_grouping_guard.py
git commit -m "fix(integration): skip peak grouping for non-chromatographic features"
```

---

### Task 8: Wire bands branch into `/api/run`

**Files:**
- Modify: `api/main.py` (`run_pipeline`, around lines 735-767)
- Test: `tests/api/test_run_bands.py` (create)

**Note on test approach:** This is a unit test of the bands branch logic exercised through `run_pipeline` via FastAPI's `TestClient`, using a synthetic CSV `.C`-style path is heavy; instead we test the branch by constructing a method with bands and a small synthetic signal through a helper. Because `/api/run` requires a real `.C`/`.D` folder loader, we test the *decision logic* by asserting that when `method.bands` is present the response peaks carry `band_assignment` and that peak-detection params are irrelevant. We use the existing `.C` test fixture pattern from `tests/logic/test_c_folder.py`. If no fixture folder exists, this task's integration test is marked to build a minimal 2-column CSV `.C` via `CFolder`.

- [ ] **Step 1: Inspect available `.C` fixtures**

Run: `conda run -n chromakit-env python -c "import glob; print(glob.glob('tests/**/*.C', recursive=True))"`
Expected: prints any existing `.C` fixture folders. Record the path if one exists.

- [ ] **Step 2: Write the failing test**

Create `tests/api/test_run_bands.py`. If Step 1 found a `.C` fixture, set `DATA_C` to it; otherwise the test builds a synthetic FTIR CSV `.C` folder in a tmp dir. This test writes a `.chromethod` with `bands` and asserts the response.

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _make_ftir_c_folder(tmp: Path) -> Path:
    """Create a minimal FTIR .C folder with a single spectrum CSV.

    Mirrors the reactir CSV shape: 2-col headerless wavenumber,absorbance.
    Uses CFolder's expected layout. If the project's CFolder requires a
    specific manifest, adapt here.
    """
    from logic.c_folder import CFolder  # noqa: F401
    c_dir = tmp / "spectrum.C"
    c_dir.mkdir()
    x = np.linspace(2200, 1800, 401)  # descending wavenumbers
    y = np.zeros_like(x)
    # Inject a band around 1987 cm-1
    y += 1.5 * np.exp(-((x - 1987.0) ** 2) / (2 * 4.0 ** 2))
    csv = c_dir / "data.csv"
    csv.write_text("\n".join(f"{xi},{yi}" for xi, yi in zip(x, y)))
    return c_dir


@pytest.mark.skipif(
    not hasattr(__import__("logic.c_folder", fromlist=["CFolder"]), "CFolder"),
    reason="CFolder loader unavailable",
)
def test_run_with_bands_returns_named_features(tmp_path):
    data_c = _make_ftir_c_folder(tmp_path)

    method = {
        "name": "ir_bands",
        "signal_type": "ftir",
        "baseline": {"enabled": False},
        "peaks": {"enabled": False, "min_prominence": 0.02},
        "bands": [
            {"name": "precursor_CO", "x_min": 1970, "x_max": 2005},
            {"name": "np_broad", "x_min": 800, "x_max": 900},  # empty window
        ],
    }
    method_path = tmp_path / "ir.chromethod"
    method_path.write_text(json.dumps(method))

    resp = client.post("/api/run", json={
        "method_path": str(method_path),
        "data_path": str(data_c),
        "write_output": False,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["peak_count"] == 2
    names = [p["band_assignment"] for p in body["peaks"]]
    assert names == ["precursor_CO", "np_broad"]
    # First band has real signal -> non-zero area; second is empty -> zero.
    areas = {p["band_assignment"]: p["area"] for p in body["peaks"]}
    assert areas["precursor_CO"] > 0
    assert areas["np_broad"] == 0.0
```

If the synthetic `.C` construction does not match this repo's `CFolder` layout, adjust `_make_ftir_c_folder` to match `tests/logic/test_c_folder.py` fixture construction (read that file first). Keep the assertions identical.

- [ ] **Step 3: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/api/test_run_bands.py -v`
Expected: FAIL — either 200 with wrong shape (bands ignored) or an error, because `/api/run` does not yet branch on `method.bands`.

- [ ] **Step 4: Write minimal implementation**

In `api/main.py`, `run_pipeline`, locate the block after `processed = processor.process(...)` (line 737) and the integrate/rt-assign/quant block (lines 739-767). Wrap the existing integration in an `else` and add the bands branch:

```python
        processed = processor.process(x, y, params=proc_params, profile=profile)

        run_quant_summary = None

        if method.bands:
            # Fixed-window band integration replaces peak detection entirely.
            from logic.integration import integrate_bands
            y_int = processed["corrected_y"] if method.baseline.enabled else processed["original_y"]
            peaks = integrate_bands(processed["x"], y_int, method.bands, profile)
        else:
            # 4. Integrate
            integrated = processor.integrate_peaks(
                processed_data=processed,
                rt_table=None,
                chemstation_area_factor=method.chemstation_area_factor,
                peak_groups=method.integration.peak_groups or [],
                profile=profile,
            )
            peaks = integrated.get("peaks", [])

            # 4b. RT-assign — populate compound_id from the method's embedded RT table.
            if method.rt_table:
                rt_df = method.rt_table_as_dataframe()
                apply_rt_matching(peaks, rt_df, method.rt_matching)

            # 4c. Quantitate — RF-table strategy only (Phase 1a).
            if method.quant_strategy == "rf_table" and method.rf_table:
                rf_summary = quantitate_rf(peaks, method.rf_table, rf_unit=method.rf_unit, normalize=True)
                run_quant_summary = RunQuantSummary(
                    strategy=rf_summary.strategy,
                    peaks_quantitated=rf_summary.peaks_quantitated,
                    peaks_skipped_unassigned=len(rf_summary.skipped_unassigned),
                    peaks_skipped_no_rf=len(rf_summary.skipped_no_rf),
                    normalized=rf_summary.normalized,
                    warnings=list(rf_summary.warnings),
                    rf_unit=rf_summary.rf_unit,
                    composition_basis=rf_summary.composition_basis,
                )
```

Delete the now-duplicated original lines 739-767 (the old un-wrapped integrate/rt-assign/quant block and the earlier `run_quant_summary = None` at line 755). Ensure `run_quant_summary` is defined exactly once (moved above the branch). Leave the export block (lines 769+) and response construction unchanged — they already handle `peaks` polymorphically via `as_dict()`.

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/api/test_run_bands.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full api + run tests to confirm no regression**

Run: `conda run -n chromakit-env pytest tests/api/ -q`
Expected: all PASS (pre-existing `test_api.py` connection-based failures, if collected, are unrelated — do not run `api/test_api.py`).

- [ ] **Step 7: Commit**

```bash
git add api/main.py tests/api/test_run_bands.py
git commit -m "feat(api): band-integration branch in /api/run replaces peak detection when bands present"
```

---

### Task 9: Full-suite regression + push

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `conda run -n chromakit-env pytest tests/ -q`
Expected: all PASS (462+ prior tests plus the new ones). If any pre-existing test fails, investigate before proceeding — it may be an unintended regression from Task 5 (baseline) or Task 8 (api restructure).

- [ ] **Step 2: Confirm no stray behavioral change for chromatography**

Run: `conda run -n chromakit-env pytest tests/logic/test_method.py tests/api/test_run_quantitation.py tests/logic/test_rt_matching.py -q`
Expected: all PASS — confirms `bands`-empty methods and RF/RT paths are untouched.

- [ ] **Step 3: Push**

```bash
git push origin main
```

Expected: push succeeds.

---

## Self-Review

**Spec coverage:**
- §1 data model (BandWindow, bands, baseline.enabled) → Tasks 1, 2.
- §2 integrate_bands (direction-safe, absorbance/position at max, empty-window quality issue) → Task 4.
- §3 pipeline wiring (baseline toggle in processor; bands branch replaces peak detection) → Tasks 5, 8.
- §4 guards (apply_rt_matching, _apply_peak_grouping) + min_prominence validation → Tasks 6, 7, 3.
- §5 testing / acceptance criteria → covered per-task + Task 9.

**Placeholder scan:** Task 8 Step 2 contains a conditional ("if the synthetic `.C` construction does not match … adjust to match `tests/logic/test_c_folder.py`") — this is a real, bounded instruction to match an existing fixture pattern, not a placeholder. The worker must read `test_c_folder.py` if the synthetic path fails. All other steps contain concrete code and commands.

**Type consistency:** `BandWindow(name, x_min, x_max)` used identically in Tasks 1, 4, 8. `integrate_bands(x, y, bands, profile)` signature matches its call site in Task 8. `SpectralFeature` constructor kwargs match `logic/feature.py:65`. `process()` output keys (`corrected_y`, `original_y`, `x`) match `logic/processor.py:579`. `run_quant_summary` defined exactly once (Task 8 moves it above the branch).
