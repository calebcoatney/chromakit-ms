# Fixed-Window Band Integration for ChromaKit Spectroscopy — Design

**Date:** 2026-07-23
**Status:** Approved (brainstorming), ready for implementation planning
**Source handoff:** `NLR Obsidian/chromakit-spectroscopy-handoff.md`
**Repo:** `chromakit-ms`, branch `main` (TDD)

## Context

The ALchemist Bayesian-optimization loop needs a few reproducible scalar
features (band areas) from co-registered FTIR and UV-Vis spectra of the
Mo(CO)₆ → Mo(0)/MoOₓ nanoparticle reaction. Two spectroscopy shapes must be
captured:

- **FTIR:** sharp carbonyl bands (~1987, ~1932, ~1881 cm⁻¹) in a mostly-flat,
  already-background-subtracted spectrum. Region ~1800–2200 cm⁻¹.
- **UV-Vis:** a **broad, non-peak-like** absorption continuum growing across
  ~300–500 nm (the NP-formation signal), plus structured bands near ~285–300 nm
  and ~400 nm.

ChromaKit today integrates only detected peaks (classical detection →
2nd-derivative bound-walking). There is **no fixed-window integral mode**, so
the broad UV-Vis band cannot be captured robustly and IR bounds drift
spectrum-to-spectrum. Several spectroscopy-configured code paths also crash.

The consumer is the headless `/api/run` endpoint (`api/main.py`), called by
`spectro_bridge` in `nanoparticle_hmi`; results flow to `reactor/spectro/results`.

### Verified current state (this repo, 2026-07-23)

- `SpectralFeature` (`logic/feature.py:62`) declares `band_assignment`,
  `absorbance`, `transmittance` but they are never populated; it has no
  `retention_time`/`compound_id`.
- `/api/run` (`api/main.py:686`) runs `process → integrate_peaks →
  apply_rt_matching (if rt_table) → quantitate_rf`. `apply_rt_matching` on a
  `SpectralFeature` raises `AttributeError` (no `retention_time`).
- Baseline **always** runs (`processor.py:460`); no way to disable it.
- `min_prominence: null` crashes (`processor.py` reads it directly and compares
  to an int).
- `ChromaMethod` (`logic/method.py:148`) has no band concept.

The handoff is accurate against the code.

## Goals

1. A first-class fixed-window band-integration mode: declare named windows in a
   method, get one integral per window, independent of peak detection.
2. Populate `band_assignment` and `absorbance` on emitted features.
3. Allow disabling baseline so the broad UV-Vis band is not subtracted.
4. Make misconfigured spectroscopy methods fail clearly, not with HTTP 500.

## Non-Goals

- Dilution / Beer–Lambert correction (downstream, uses pump-flow log).
- Any `spectro_bridge` / HMI / vault changes.
- Choosing final band windows (workshopped separately against real spectra).
- Wiring `bands` into the **desktop GUI** live-processing path (follow-up ticket).

## Design

### 1. Data model (`logic/method.py`)

New sub-model:

```python
class BandWindow(BaseModel):
    name: str          # -> SpectralFeature.band_assignment
    x_min: float       # native x-units (cm-1 or nm)
    x_max: float
    # validator: require x_min < x_max
```

`ChromaMethod` gains:

```python
bands: List[BandWindow] = Field(default_factory=list)
```

Empty by default → no behavior change for existing chromatography methods.

`BaselineParams` gains:

```python
enabled: bool = True
```

Default `True` preserves current always-on baseline. Spectroscopy methods set
`false` to integrate raw absorbance.

Example method fragment:

```json
"baseline": { "enabled": false },
"bands": [
  { "name": "precursor_CO",  "x_min": 1970, "x_max": 2005 },
  { "name": "carbonyl_1932", "x_min": 1915, "x_max": 1950 },
  { "name": "np_broad",      "x_min": 350,  "x_max": 500 }
]
```

### 2. Band integration logic (`logic/integration.py`)

New **pure function**:

```python
def integrate_bands(x, y, bands, profile) -> List[SpectralFeature]:
    """Integrate fixed x-windows. Independent of peak detection.
    y is the signal to integrate (baseline-corrected if baseline.enabled,
    else raw). Returns exactly one SpectralFeature per band, in order."""
```

Per band:

- Select samples where `x_min <= x <= x_max` (filter by **value**, so it is
  correct for both normal and `invert_x` FTIR axes).
- **Direction-safe integral:** sort windowed `(x, y)` ascending by `x`, then
  `area = abs(simpson(y_sorted, x=x_sorted))`.
- `absorbance = max(y_window)`; `position = x` at that max (band peak location).
- `start = min(x_min, x_max)`, `end = max(...)`, `width = |x_max - x_min|`.
- `band_assignment = band.name`.
- **Empty window** (no samples in range): emit `area=0.0, absorbance=0.0` and
  append a quality issue (`"no samples in [x_min, x_max]"`). Never drop a band —
  downstream expects one feature per declared band, always.

Unit-testable in isolation with synthetic arrays; no pipeline required.

### 3. Pipeline wiring

**`processor.process()`** (`processor.py:429`, "Always calculate baseline"
`:460`): when `baseline.enabled is False`, skip baseline correction and set the
"corrected" signal equal to the raw signal so downstream `corrected_y` stays
well-defined. Default `True` → unchanged.

**`/api/run`** (`api/main.py:686`): insert a bands branch **before** peak
integration:

```
process(...)                     # smoothing + (optional) baseline
if method.bands:                 # bands present => exclusive path
    y_int  = corrected_y if baseline.enabled else raw_y
    peaks  = integrate_bands(x, y_int, method.bands, profile)
    # SKIP integrate_peaks, apply_rt_matching, quantitate_rf
else:
    ... existing integrate_peaks / rt-assign / quant path (unchanged) ...
```

When `bands` is non-empty, bands **replace** peak detection and the run
short-circuits past `integrate_peaks`, `apply_rt_matching`, and quantitation —
which also sidesteps the rt_table AttributeError on the band path. The response
serialization loop (`api/main.py:787`) already calls `peak.as_dict()`, which
`SpectralFeature` implements, so the response schema is unchanged.

### 4. Defensive guards & validation (crash-path fixes)

Protect the **non-band** (chromatography-configured-on-spectroscopy) paths:

1. **`apply_rt_matching`** (`logic/rt_matching.py:~108`): guard at entry — if
   features are non-chromatographic (lack `retention_time`/`compound_id`), skip
   with a clear warning instead of `AttributeError`.
2. **`_apply_peak_grouping`** (`logic/integration.py:~419`): same guard — skip
   grouping for non-chromatographic features instead of touching
   `peak.retention_time`.
3. **`min_prominence` validation** (`PeakParams`, `logic/method.py`): add a
   validator that **rejects `None` with an actionable error** (fail-fast at
   method load, not mid-pipeline). Chromatography default stays `1e5`;
   spectroscopy methods set a fractional value (e.g. `0.02`).

## Testing (TDD — all green before push)

- `integrate_bands`: normal window; inverted-x FTIR sign-safety; empty window →
  zero + quality issue; `absorbance`/`position` at max; multiple bands ordered.
- `baseline.enabled=false` skips correction (raw signal integrated).
- `/api/run` integration: a `.chromethod` with `bands` on ftir and uvvis `.C`
  data → one feature per band, `band_assignment` set, non-zero broad-UV-Vis
  area; peak detection off/independent.
- Guards: spectroscopy method with non-empty `rt_table`/`peak_groups` → no
  `AttributeError`.
- `min_prominence: null` → clear validation error, not HTTP 500.
- Full existing suite (462+) stays green.

## Acceptance Criteria (from handoff)

- A `.chromethod` can declare named fixed-integration windows for an `ftir` or
  `uvvis` `.C` folder; `/api/run` returns one feature per window with
  `area` = windowed (baseline-corrected or raw) integral and `band_assignment`
  set — with peak detection off/independent.
- Real reactIR IR CSV and a real UV-Vis spectrum run without crashing; the broad
  UV-Vis 350–500 nm window yields a meaningful non-zero area tracking NP growth.
- No `AttributeError` from `rt_table`/`peak_groups` on spectroscopy data.
- `min_prominence: null` no longer 500s.
- Tests added; all green; pushed to `origin/main`.

## Test Data

- reactIR IR CSVs (2-col headerless `wavenumber,absorbance`); NiCu example
  staged on synthon `~/spectro_bridge_dev/ir_data/`.
- Full dilution-run FTIR + UV-Vis in OneDrive
  `.../Data/Experiments/260611 MoCO6 300C dilution test/`.

## Follow-ups (out of scope)

- Wire `bands` into the desktop GUI live-processing path.
- Optional `transmittance` population if a downstream consumer needs it.
