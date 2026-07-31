# Design: method-embedded scaling + detector

**Date:** 2026-07-31
**Repo:** `chromakit-qt` (`origin` = `github.com/calebcoatney/chromakit-ms`)
**Motivation:** The RAPIDS headless pipeline (`raw .D` → `/api/run` → quantitated mol%)
must reproduce what the GUI produces. Today it does not, because two pieces of
processing state — signal/area **scaling** and the acquisition **detector** — live
*outside* the `.chromethod` file and are invisible to `/api/run`. This makes a
`.chromethod` an incomplete description of "how to process this data." This spec
closes that gap by folding scaling and detector into `ChromaMethod`.

---

## Problem statement (with evidence)

Two real STH `.D` files were processed in the GUI and via `/api/run` with the
*same* `.chromethod` files. Results diverged hard (GUI areas ~4e9 vs headless ~69;
GUI 27 peaks vs headless 0). Root cause, confirmed from the GUI's own JSON export:

```json
"scaling": { "signal_factor": 7700.0, "area_factor": 600.0 }
```

### The core confusion: one argument slot, two sources

There are **two independent area knobs** that have been conflated because they share
a name. `chemstation_area_factor` is a **function-parameter name** in the integration
layer — the argument slot for the area multiplier. It is not a second multiplier
applied on top of `area_factor`. What differs is *what value is fed into that slot*:

- **GUI** (`ui/app.py:1986, 2080`): passes `self.area_factor` (QSettings-backed,
  default **1.0**) into the `chemstation_area_factor=` argument. So `area *= 1.0`.
  The method's stored `chemstation_area_factor=0.0784` field is **never read**.
- **Headless** (`api/main.py:755`): passes `method.chemstation_area_factor`
  (default **0.0784**) into the same argument. So `area *= 0.0784`.

Net: the argument slot is named `chemstation_area_factor`, but the GUI feeds it
`area_factor` (1.0) while headless feeds it the method field (0.0784) — same slot,
two different sources, two different numbers. Fresh ChromaKit with no method loaded
uses `signal_factor=1.0` and `area_factor=1.0` (×1). The `0.0784` only ever bit in
the headless path.

Separately, the **detector/channel** (e.g. `TCD3C` vs `FID1A`) is not in the method
at all — chosen interactively in the GUI, passed ad hoc to `/api/run`.

---

## Where the state lives today (code map)

- **Scaling source (GUI):** `ui/app.py:67-69` reads `QSettings("CalebCoatney","ChromaKit")`
  keys `scaling/signal_factor`, `scaling/area_factor`; sets `self.signal_factor`,
  `self.area_factor`, `data_handler.signal_factor`.
- **Scaling dialog:** `ui/dialogs/scaling_factors_dialog.py` — edits those QSettings
  keys + named presets. Preset "GC1-2" = `{signal_factor:7700, area_factor:600}`.
- **signal_factor application:** `logic/data_handler.py:127` → `y = data * self.signal_factor`
  (at load). `.C` path threads `signal_factor` via `c_folder.load_signal`.
- **area application:** GUI passes `chemstation_area_factor=self.area_factor` into
  integration (`ui/app.py:1986, 2030, 2038, 2080`); `logic/integration.py:935` →
  `area *= chemstation_area_factor`.
- **Export records scaling:** `logic/export_manager.py:103-106` writes
  `signal_factor`/`area_factor` into `processing_parameters.scaling`.
- **Method schema:** `logic/method.py::ChromaMethod` — has `chemstation_area_factor`,
  but NO `detector`, NO `signal_factor`/`area_factor`.
- **API run path:** `api/main.py::run_pipeline` (`/api/run`, ~686). Loads via
  `data_handler.load_data_directory(path, detector=request.detector)` (signal_factor
  stays 1.0), integrates with `method.chemstation_area_factor`.
- **Stray endpoint:** `/api/scaling-factors` (`api/main.py:390`) sets
  `data_handler.signal_factor` but is not called by `/api/run` and discards
  `area_factor`.

---

## Design decisions (resolved)

1. **One area knob.** Remove `chemstation_area_factor` from `ChromaMethod` entirely
   (no default, no migration alias). Replace with `area_factor`. The GUI already
   used `area_factor` everywhere, so this changes nothing in the GUI; it fixes
   headless to match.
2. **`None` = no scaling.** `signal_factor`/`area_factor` default to `None`. `None`
   or `0.0` → ×1 (no scaling). `0.0` is treated as "no scaling" (not "zero
   everything") to avoid a footgun, with a warning telling the user to use `1`.
3. **Method is source of truth.** GUI `self.signal_factor`/`self.area_factor` are
   sourced from `current_method`. The scaling dialog writes into the method, marks
   the document dirty, and is saved into the `.chromethod`. QSettings remains an
   authoring convenience.
4. **Detector in the method.** Add `detector: Optional[str]`. Precedence at
   `/api/run`: `request.detector` > `method.detector` > auto-detect.
5. **Metadata classification.** All three new fields go in `_METADATA_FIELDS`
   (excluded from `to_processor_params()`); they are applied by explicit wiring at
   load/integration, consistent with how `chemstation_area_factor` was handled.

---

## Required changes

### 1. Schema (`logic/method.py`)
- **Remove** `chemstation_area_factor` field (no default, no alias). Remove it from
  `_METADATA_FIELDS`. Drop the `chemstation_area_factor` param from `from_gui_params`.
- **Add** three `Optional` fields, all defaulting to `None`, all added to
  `_METADATA_FIELDS`:
  - `signal_factor: Optional[float] = None` — raw-signal multiplier at load.
  - `area_factor: Optional[float] = None` — integrated-area multiplier.
  - `detector: Optional[str] = None` — intended channel; `None` = auto-detect.
- `from_gui_params`/`to_gui_params` round-trip all three.
- Old files carrying `chemstation_area_factor` → pydantic ignores the unknown key →
  they load as `area_factor=None` (×1), matching what the GUI always did.

### 2. Integration layer (`logic/integration.py`, `logic/processor.py`)
- Rename the function parameter `chemstation_area_factor` → `area_factor` throughout
  (`integration.py:605/935`, `processor.py:810/863`, all call sites, tests).
- Guarded, single-point application (`integration.py:935`):
  ```python
  if area_factor:              # None or 0.0 → skip (×1)
      area *= area_factor
  ```
- If `area_factor == 0.0`, emit a logger warning: `0` is not the disable mechanism;
  use `1.0` to leave areas unaltered. (`logic/` stays UI-agnostic — logger only.)

### 3. Headless path (`api/main.py::run_pipeline`)
- Detector precedence: `request.detector or method.detector` (then loader auto-detect).
- Signal scaling: `data_handler.signal_factor = method.signal_factor or 1.0` before
  `.D` load; pass `signal_factor=method.signal_factor or 1.0` into `.C` `load_signal`.
- Area scaling: pass `area_factor=method.area_factor` into integration (renamed arg).
- Remove/rewire the stray `/api/scaling-factors` endpoint; drop the now-dead
  `chemstation_area_factor` field on `RunRequest`/`api/models.py`.

### 4. GUI: method is source of truth (`ui/app.py`, `scaling_factors_dialog.py`)
- `self.signal_factor`/`self.area_factor` sourced from `current_method` (thin sync).
- **App startup:** seed `current_method.signal_factor`/`area_factor` from QSettings
  once (preserves current default-load convenience).
- **Scaling dialog accept** (`_on_scaling_factors_changed`): write values into
  `current_method`; `_mark_dirty(True)` (prompts to save on close, writes to the
  `.chromethod` on save); still update QSettings; reload current file so scaling
  takes effect live. This dirty/save behavior applies even with no data file loaded
  (it is a method change regardless). Zero value → `QMessageBox.warning` + coerce to
  `1.0` before storing.
- **Load Method:** `self.signal_factor`/`area_factor`/`data_handler.signal_factor`
  and the detector selection all come from the method — no separate scaling step.
- **`_on_params_writeback` (app.py:366):** preserve the new fields when rebuilding
  `current_method` (they would otherwise be reset).
- Integration call sites (1986, 2030, 2038, 2080) feed `current_method.area_factor`
  into the renamed `area_factor=` argument.

### 5. Export parity (`logic/export_manager.py`)
- GUI already emits `processing_parameters.scaling`. Ensure the headless `/api/run`
  export emits the same block, now sourced from `method.signal_factor`/`area_factor`,
  so GUI and headless JSONs match on scaling.

### 6. No-double-apply guarantees (explicit)
- `signal_factor`: applied **only** at load (`data_handler.py:127` / `c_folder`).
  Never in the processor, never at integration.
- `area_factor`: applied **only** at integration (`integration.py:935`). Never at load.
- One value each, sourced from `current_method`. There is no second application site.

---

## Acceptance criteria (TDD — write first)

1. **Schema round-trip:** a method with `detector="TCD3C"`, `signal_factor=7700`,
   `area_factor=600` survives `to_file`/`from_file` and `to_gui_params`/`from_gui_params`.
   (`tests/logic/test_method.py`)
2. **Old-file compat:** a file with `chemstation_area_factor` loads with
   `area_factor=None` and integrates ×1 (unknown key ignored, no crash).
3. **Zero-guard:** `area_factor=0.0` → area unchanged (×1) + warning logged; never
   zeros peaks.
4. **Headless applies scaling:** `/api/run` with `signal_factor`/`area_factor`
   scales peak areas correctly vs a ×1 baseline. (`tests/test_run_response.py`)
5. **Detector precedence:** no request detector → `method.detector` wins; explicit
   request detector overrides.
6. **Parity test (the real bar):** the same `.D` + `.chromethod` yields the **same
   peak count and same areas (float tolerance)** through the GUI processing code path
   and through `/api/run`. This is the test that would have caught the RAPIDS bug.

---

## Real-world validation (RAPIDS)

Headless on the RAPIDS GC box with NO GUI and NO QSettings:
- `GC1-TCD3C.chromethod` (`detector="TCD3C"`, `signal_factor=7700`, `area_factor=600`)
  on a GC1 STH `.D` → detects the 7 peaks; names H2/CO2/Ar/CO; areas match GUI export
  (H2 ~4.08e9, CO ~3.64e9).
- `GC2-FID2B.chromethod` → detects MeOH at RT ~21.470, area ~4.4e7.

(These numbers are illustrative validation targets, not fixtures to overfit to. The
goal is GUI/headless parity on *any* `.D` + method, not reproducing these exact values.)

---

## Out of scope
- RT-window / matching-mode tuning (RAPIDS-side method authoring).
- The RF-table quant strategy itself (already shipped).
- Removing the scaling-factor QSettings dialog (kept as authoring convenience).

## Notes / gotchas
- Normalized RF quantitation (`composition_percent`) is **invariant** to uniform
  scaling (cancels in `area_i/RF_i ÷ Σ`), so scaling does not change normalized mol%.
  It DOES change absolute areas and — critically — whether peak-detection thresholds
  (`min_prominence`, tuned against the scaled signal) find anything at all. That is
  why the headless run found 0 peaks.
- Windows OpenSSH + PowerShell `Set-Content -Encoding UTF8` writes a BOM that breaks
  pydantic JSON parsing; use
  `[System.IO.File]::WriteAllText($p,$s,(New-Object System.Text.UTF8Encoding($false)))`
  if authoring fixtures on Windows.
