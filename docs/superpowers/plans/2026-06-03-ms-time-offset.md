# MS Time Offset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-controlled constant time shift between MS and FID data, set via the Spectral Deconvolution Inspector, persisted per-file, and honored by every MS consumer in the application.

**Architecture:** A single `DataHandler.ms_time_offset` attribute (minutes) is read by every code path that touches `ms.xlabels`, via a one-line helper `shifted_xlabels(ms, offset)`. The inspector dialog provides a slider + Auto (cross-correlation) + Apply globally workflow. Offsets persist in `overrides/ms_time_offsets.json` keyed by absolute `.D` path and are logged in exported results JSON. All dead `align_tic` infrastructure is removed in a single cleanup task.

**Tech Stack:** Python, PySide6, NumPy, SciPy, pytest, rainbow (for `.D` reading).

**Spec:** `docs/superpowers/specs/2026-06-03-ms-time-offset-design.md`

---

## File Inventory

**New:**
- `logic/ms_time.py` — `shifted_xlabels()` helper
- `logic/sidecar_offsets.py` — read/write `overrides/ms_time_offsets.json`
- `tests/logic/test_ms_time.py` — unit tests for the helper
- `tests/logic/test_sidecar_offsets.py` — unit tests for sidecar I/O

**Modified:**
- `logic/data_handler.py` — add `ms_time_offset` attribute, autoload sidecar, use helper in `_get_tic_data`, remove dead `aligned_tic_data` kwarg from `extract_spectrum_at_rt`
- `logic/spectrum_extractor.py` — accept `ms_time_offset` kwarg, replace all `ms.xlabels` reads with shifted-xlabels variable
- `logic/eic_extractor.py` — accept `ms_time_offset` kwarg, replace `ms.xlabels` read
- `logic/spectral_deconv_runner.py` — accept `ms_time_offset` kwarg, propagate to `extract_eic_peaks`, shift window bounds
- `logic/method.py` — delete `align_tic` field
- `logic/json_exporter.py` — delete `bl_info['align_tic']`, add `ms_time_offset` and `ms_time_offset_source` to metadata
- `ui/dialogs/spectral_deconv_inspector.py` — add MS Time Offset group (slider, Auto, Reset, Apply globally), live preview replot, use shifted xlabels in its own MS re-read
- `ui/frames/parameters.py` — delete align-TIC checkbox and handler
- `ui/frames/plot.py` — delete `aligned_tic_data` / `set_aligned_tic_data`, append offset to TIC legend label when nonzero
- `ui/app.py` — show status bar message when sidecar offset loaded, wire inspector Apply-globally to set DataHandler offset + write sidecar + trigger deconvolution rerun
- `api/models.py` — delete `AlignTICRequest`, `AlignTICResponse`
- `api/main.py` — delete `POST /api/align-tic` endpoint
- `tests/logic/test_method.py` — remove `align_tic` key from fixture

**Convention:** every change is committed at the end of its task. Test commands assume `conda activate chromakit-env` is active.

---

## Task 1: Create the `shifted_xlabels` helper

**Files:**
- Create: `logic/ms_time.py`
- Create: `tests/logic/test_ms_time.py`

- [ ] **Step 1: Write the failing test**

Create `tests/logic/test_ms_time.py`:

```python
"""Tests for logic/ms_time.py — MS time axis shift helper."""
import numpy as np
import pytest

from logic.ms_time import shifted_xlabels


class _FakeMS:
    """Minimal stand-in for a rainbow DataFile, exposing only .xlabels."""
    def __init__(self, xlabels):
        self.xlabels = np.asarray(xlabels, dtype=float)


def test_zero_offset_returns_equal_values():
    ms = _FakeMS([1.0, 2.0, 3.0])
    out = shifted_xlabels(ms, 0.0)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_positive_offset_adds_to_each_value():
    ms = _FakeMS([1.0, 2.0, 3.0])
    out = shifted_xlabels(ms, 0.05)
    np.testing.assert_allclose(out, [1.05, 2.05, 3.05])


def test_negative_offset_subtracts_from_each_value():
    ms = _FakeMS([1.0, 2.0, 3.0])
    out = shifted_xlabels(ms, -0.05)
    np.testing.assert_allclose(out, [0.95, 1.95, 2.95])


def test_does_not_mutate_source_xlabels():
    ms = _FakeMS([1.0, 2.0, 3.0])
    _ = shifted_xlabels(ms, 0.1)
    np.testing.assert_array_equal(ms.xlabels, [1.0, 2.0, 3.0])


def test_returns_ndarray_even_for_list_input():
    ms = _FakeMS([1.0, 2.0, 3.0])
    ms.xlabels = [1.0, 2.0, 3.0]  # simulate non-array attribute
    out = shifted_xlabels(ms, 0.0)
    assert isinstance(out, np.ndarray)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/logic/test_ms_time.py -v`
Expected: ImportError / collection error — `logic.ms_time` does not exist.

- [ ] **Step 3: Implement the helper**

Create `logic/ms_time.py`:

```python
"""MS time-axis utilities.

Centralizes the application-wide convention that MS retention times may be
offset by a constant `ms_time_offset` (in minutes) to align with FID retention
times. Every consumer that reads `ms.xlabels` should go through this helper so
that the offset is honored uniformly.
"""
from __future__ import annotations

import numpy as np


def shifted_xlabels(ms_data, offset_min: float) -> np.ndarray:
    """Return the MS retention-time axis shifted by `offset_min` minutes.

    Args:
        ms_data: A rainbow MS DataFile (only `.xlabels` is read).
        offset_min: Constant shift in minutes (positive = MS times move later;
            negative = MS times move earlier). 0.0 returns the unshifted axis.

    Returns:
        A new NumPy float64 array. The source object is not mutated.
    """
    return np.asarray(ms_data.xlabels, dtype=float) + float(offset_min)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/logic/test_ms_time.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add logic/ms_time.py tests/logic/test_ms_time.py
git commit -m "feat(logic): add shifted_xlabels helper for MS time offset"
```

---

## Task 2: Create the sidecar I/O module

**Files:**
- Create: `logic/sidecar_offsets.py`
- Create: `tests/logic/test_sidecar_offsets.py`

- [ ] **Step 1: Write the failing test**

Create `tests/logic/test_sidecar_offsets.py`:

```python
"""Tests for logic/sidecar_offsets.py — per-file MS time offset sidecar."""
import json
import time
from pathlib import Path

import pytest

from logic.sidecar_offsets import (
    load_offset,
    save_offset,
    OffsetEntry,
)


def test_load_returns_none_when_sidecar_missing(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    assert load_offset("/some/file.D", sidecar_path=sidecar) is None


def test_load_returns_none_when_key_missing(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    sidecar.write_text(json.dumps({"/other/file.D": {
        "offset_min": -0.05, "timestamp": 0.0, "source": "manual"
    }}))
    assert load_offset("/some/file.D", sidecar_path=sidecar) is None


def test_save_then_load_round_trip(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/sample.D", -0.048, source="manual", sidecar_path=sidecar)
    entry = load_offset("/abs/sample.D", sidecar_path=sidecar)
    assert entry is not None
    assert entry.offset_min == pytest.approx(-0.048)
    assert entry.source == "manual"
    assert entry.timestamp > 0.0


def test_save_overwrites_existing_entry(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/sample.D", -0.048, source="manual", sidecar_path=sidecar)
    time.sleep(0.01)
    save_offset("/abs/sample.D", 0.012, source="auto", sidecar_path=sidecar)
    entry = load_offset("/abs/sample.D", sidecar_path=sidecar)
    assert entry.offset_min == pytest.approx(0.012)
    assert entry.source == "auto"


def test_save_preserves_other_entries(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/a.D", 0.01, source="manual", sidecar_path=sidecar)
    save_offset("/abs/b.D", 0.02, source="auto", sidecar_path=sidecar)
    a = load_offset("/abs/a.D", sidecar_path=sidecar)
    b = load_offset("/abs/b.D", sidecar_path=sidecar)
    assert a.offset_min == pytest.approx(0.01)
    assert b.offset_min == pytest.approx(0.02)


def test_load_tolerates_malformed_json(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    sidecar.write_text("{not valid json")
    # Must not raise; return None so the app keeps working.
    assert load_offset("/abs/a.D", sidecar_path=sidecar) is None


def test_save_rejects_invalid_source(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    with pytest.raises(ValueError):
        save_offset("/abs/a.D", 0.01, source="bogus", sidecar_path=sidecar)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/logic/test_sidecar_offsets.py -v`
Expected: ImportError — `logic.sidecar_offsets` does not exist.

- [ ] **Step 3: Implement the module**

Create `logic/sidecar_offsets.py`:

```python
"""Per-file MS time offset sidecar.

Stores user-applied MS time offsets in `overrides/ms_time_offsets.json` keyed
by absolute `.D` directory path. Format::

    {
      "/abs/path/to/sample.D": {
        "offset_min": -0.048,
        "timestamp": 1735000000.0,
        "source": "manual"   // or "auto"
      },
      ...
    }
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

DEFAULT_SIDECAR_PATH = Path("overrides") / "ms_time_offsets.json"
VALID_SOURCES = ("manual", "auto")
Source = Literal["manual", "auto"]


@dataclass(frozen=True)
class OffsetEntry:
    offset_min: float
    timestamp: float
    source: Source


def _resolve_path(sidecar_path: Optional[Path]) -> Path:
    return Path(sidecar_path) if sidecar_path is not None else DEFAULT_SIDECAR_PATH


def load_offset(data_path: str, sidecar_path: Optional[Path] = None) -> Optional[OffsetEntry]:
    """Return the saved offset for `data_path`, or None if absent / unreadable."""
    path = _resolve_path(sidecar_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = raw.get(str(data_path))
    if not isinstance(entry, dict):
        return None
    try:
        return OffsetEntry(
            offset_min=float(entry["offset_min"]),
            timestamp=float(entry.get("timestamp", 0.0)),
            source=entry.get("source", "manual"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_offset(
    data_path: str,
    offset_min: float,
    source: Source,
    sidecar_path: Optional[Path] = None,
) -> None:
    """Persist `offset_min` for `data_path` in the sidecar (creating it if needed)."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
    path = _resolve_path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    data[str(data_path)] = {
        "offset_min": float(offset_min),
        "timestamp": time.time(),
        "source": source,
    }
    path.write_text(json.dumps(data, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/logic/test_sidecar_offsets.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add logic/sidecar_offsets.py tests/logic/test_sidecar_offsets.py
git commit -m "feat(logic): add per-file MS time offset sidecar I/O"
```

---

## Task 3: Add `ms_time_offset` attribute to DataHandler + autoload

**Files:**
- Modify: `logic/data_handler.py` (add attribute, autoload on open, use helper in `_get_tic_data`, remove dead kwarg)
- Create: `tests/logic/test_data_handler_offset.py`

- [ ] **Step 1: Read the current DataHandler initializer and `extract_spectrum_at_rt`**

Run: `grep -n "def __init__\|def extract_spectrum_at_rt\|def _get_tic_data\|current_directory_path" logic/data_handler.py`

Note the line numbers; you will edit at those locations.

- [ ] **Step 2: Write the failing test**

Create `tests/logic/test_data_handler_offset.py`:

```python
"""Tests for DataHandler.ms_time_offset integration."""
from pathlib import Path

import pytest

from logic.data_handler import DataHandler
from logic.sidecar_offsets import save_offset


def test_default_offset_is_zero():
    dh = DataHandler()
    assert dh.ms_time_offset == 0.0


def test_load_ms_time_offset_from_sidecar(tmp_path, monkeypatch):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/sample.D", -0.048, source="manual", sidecar_path=sidecar)
    dh = DataHandler()
    monkeypatch.setattr("logic.data_handler.DEFAULT_OFFSET_SIDECAR", sidecar)
    dh.apply_offset_from_sidecar("/abs/sample.D")
    assert dh.ms_time_offset == pytest.approx(-0.048)


def test_apply_offset_from_sidecar_when_missing_keeps_zero(tmp_path, monkeypatch):
    sidecar = tmp_path / "ms_time_offsets.json"
    dh = DataHandler()
    monkeypatch.setattr("logic.data_handler.DEFAULT_OFFSET_SIDECAR", sidecar)
    dh.apply_offset_from_sidecar("/abs/nonexistent.D")
    assert dh.ms_time_offset == 0.0


def test_extract_spectrum_at_rt_signature_drops_aligned_tic_data():
    """The old vestigial `aligned_tic_data` kwarg must be removed."""
    import inspect
    sig = inspect.signature(DataHandler.extract_spectrum_at_rt)
    assert "aligned_tic_data" not in sig.parameters
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/logic/test_data_handler_offset.py -v`
Expected: failures — `ms_time_offset` attribute and `apply_offset_from_sidecar` method don't exist; `aligned_tic_data` kwarg still present.

- [ ] **Step 4: Modify `logic/data_handler.py`**

At the top of the file (after existing imports), add:

```python
from logic.ms_time import shifted_xlabels
from logic.sidecar_offsets import DEFAULT_SIDECAR_PATH as DEFAULT_OFFSET_SIDECAR, load_offset
```

In `DataHandler.__init__`, add after the existing attribute initialization:

```python
        self.ms_time_offset: float = 0.0
```

Add a new method to `DataHandler` (place it near `_get_tic_data`):

```python
    def apply_offset_from_sidecar(self, data_path: str) -> None:
        """If a sidecar entry exists for `data_path`, set `self.ms_time_offset` from it.
        Otherwise leave the existing value unchanged."""
        entry = load_offset(data_path, sidecar_path=DEFAULT_OFFSET_SIDECAR)
        if entry is not None:
            self.ms_time_offset = entry.offset_min
```

In `_get_tic_data` (currently at line 131 `x_tic = ms_data.xlabels`), replace that line with:

```python
            x_tic = shifted_xlabels(ms_data, self.ms_time_offset)
```

In `extract_spectrum_at_rt` (currently `def extract_spectrum_at_rt(self, retention_time, aligned_tic_data=None):` at line 265), remove the `aligned_tic_data` parameter. The new signature is:

```python
    def extract_spectrum_at_rt(self, retention_time):
```

Update the docstring/body to remove any mention of `aligned_tic_data` (it was already ignored). The body should now read (mirror what's there minus the kwarg):

```python
    def extract_spectrum_at_rt(self, retention_time):
        """Extract a mass spectrum at the given retention time using the current MS file.

        The MS time-axis offset (`self.ms_time_offset`) is applied by the underlying
        SpectrumExtractor; callers pass the FID-space retention time.
        """
        if not self.current_directory_path:
            return None
        return self.extractor.extract_at_rt(
            self.current_directory_path,
            retention_time,
            ms_time_offset=self.ms_time_offset,
        )
```

(If the existing body has additional logic beyond this, preserve it but route the call through `ms_time_offset=self.ms_time_offset`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/logic/test_data_handler_offset.py -v`
Expected: 4 passed.

(The `extract_spectrum_at_rt` change calls `extractor.extract_at_rt(..., ms_time_offset=...)` — this kwarg is added in Task 4. For now the integration is not exercised by these tests, only the signature is checked.)

- [ ] **Step 6: Verify the existing test suite still passes**

Run: `pytest tests/logic/test_method.py -v`
Expected: still passes (this task doesn't touch method.py).

- [ ] **Step 7: Commit**

```bash
git add logic/data_handler.py tests/logic/test_data_handler_offset.py
git commit -m "feat(data_handler): add ms_time_offset attribute and sidecar autoload"
```

---

## Task 4: Thread `ms_time_offset` through `SpectrumExtractor`

**Files:**
- Modify: `logic/spectrum_extractor.py`

This task adds a `ms_time_offset` kwarg (default 0.0) to every public method and replaces every direct `ms.xlabels` read with a local `xlabels` variable computed from `shifted_xlabels`.

- [ ] **Step 1: Write the failing test**

Append to `tests/logic/test_ms_time.py`:

```python
import inspect
from logic.spectrum_extractor import SpectrumExtractor


def test_spectrum_extractor_extract_at_rt_accepts_ms_time_offset():
    sig = inspect.signature(SpectrumExtractor.extract_at_rt)
    assert "ms_time_offset" in sig.parameters
    assert sig.parameters["ms_time_offset"].default == 0.0


def test_spectrum_extractor_extract_for_peak_accepts_ms_time_offset():
    sig = inspect.signature(SpectrumExtractor.extract_for_peak)
    assert "ms_time_offset" in sig.parameters
    assert sig.parameters["ms_time_offset"].default == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/logic/test_ms_time.py -v`
Expected: 2 new failures — the kwarg does not exist.

- [ ] **Step 3: Add the helper import at the top of `logic/spectrum_extractor.py`**

```python
from logic.ms_time import shifted_xlabels
```

- [ ] **Step 4: Update `extract_at_rt` signature and body**

Change the signature from:

```python
    def extract_at_rt(self, data_directory: str, retention_time: float,
                     intensity_threshold: float = 0.01) -> Dict[str, Any]:
```

to:

```python
    def extract_at_rt(self, data_directory: str, retention_time: float,
                     intensity_threshold: float = 0.01,
                     ms_time_offset: float = 0.0) -> Dict[str, Any]:
```

In the body, after `ms = datadir.get_file('data.ms')`, add:

```python
            xlabels = shifted_xlabels(ms, ms_time_offset)
```

Then replace every subsequent occurrence of `ms.xlabels` (and `np.array(ms.xlabels)`) inside this function with `xlabels`. Affected lines per Task 4 audit: 46, 49, 68, 74.

- [ ] **Step 5: Update `extract_for_peak` signature**

Change to:

```python
    def extract_for_peak(self, data_directory: str, peak: Any,
                         options: Dict[str, Any] = None,
                         ms_time_offset: float = 0.0) -> Dict[str, Any]:
```

In its body, pass the offset through to `_extract_peak_spectrum`. Add `ms_time_offset=ms_time_offset` to the call.

- [ ] **Step 6: Update `_extract_peak_spectrum` signature and body**

Add a `ms_time_offset: float = 0.0` kwarg to the signature. After the line that opens the MS file (`ms = ...`), add:

```python
        xlabels = shifted_xlabels(ms, ms_time_offset)
```

Then replace every subsequent `ms.xlabels` (and `np.array(ms.xlabels)`) in the function body with `xlabels`. Audit lines: 180, 183, 184, 189, 193, 204, 235, 247, 253, 263, 279, 300, 355, 419.

Be thorough — `grep -n "ms.xlabels" logic/spectrum_extractor.py` after the edits should return zero hits.

- [ ] **Step 7: Run tests**

```bash
pytest tests/logic/test_ms_time.py -v
grep -n "ms.xlabels" logic/spectrum_extractor.py
```

Expected: all `test_ms_time.py` tests pass; grep returns no matches.

- [ ] **Step 8: Run the broader test suite to catch regressions**

```bash
pytest tests/ -x -q
```

Expected: no new failures introduced (pre-existing skips/failures are OK; new ones are not).

- [ ] **Step 9: Commit**

```bash
git add logic/spectrum_extractor.py tests/logic/test_ms_time.py
git commit -m "feat(spectrum_extractor): route MS time axis through shifted_xlabels"
```

---

## Task 5: Thread `ms_time_offset` through `extract_eic_peaks`

**Files:**
- Modify: `logic/eic_extractor.py`

- [ ] **Step 1: Read the current signature**

Run: `sed -n '14,50p' logic/eic_extractor.py`

- [ ] **Step 2: Write the failing test**

Append to `tests/logic/test_ms_time.py`:

```python
from logic.eic_extractor import extract_eic_peaks


def test_extract_eic_peaks_accepts_ms_time_offset():
    sig = inspect.signature(extract_eic_peaks)
    assert "ms_time_offset" in sig.parameters
    assert sig.parameters["ms_time_offset"].default == 0.0
```

- [ ] **Step 3: Run test**

Run: `pytest tests/logic/test_ms_time.py::test_extract_eic_peaks_accepts_ms_time_offset -v`
Expected: FAIL.

- [ ] **Step 4: Update `logic/eic_extractor.py`**

Add at the top:

```python
from logic.ms_time import shifted_xlabels
```

Add `ms_time_offset: float = 0.0` to the `extract_eic_peaks` signature (keyword-only at the end).

Replace line 39 `xlabels = np.asarray(ms.xlabels, dtype=float)` with:

```python
    xlabels = shifted_xlabels(ms, ms_time_offset)
```

- [ ] **Step 5: Run test**

Run: `pytest tests/logic/test_ms_time.py -v`
Expected: all pass.

- [ ] **Step 6: Run the eic extractor's existing tests**

Run: `pytest tests/test_eic_extractor.py -v`
Expected: still passes (no behavior change at offset=0).

- [ ] **Step 7: Commit**

```bash
git add logic/eic_extractor.py tests/logic/test_ms_time.py
git commit -m "feat(eic_extractor): accept ms_time_offset and use shifted_xlabels"
```

---

## Task 6: Thread `ms_time_offset` through `spectral_deconv_runner`

**Files:**
- Modify: `logic/spectral_deconv_runner.py`

- [ ] **Step 1: Read context**

Run: `sed -n '157,210p' logic/spectral_deconv_runner.py`

Note the signature of `run_spectral_deconvolution` and the lines reading `ms.xlabels` (188, 189) and the call to `extract_eic_peaks`.

- [ ] **Step 2: Write the failing test**

Append to `tests/logic/test_ms_time.py`:

```python
from logic.spectral_deconv_runner import run_spectral_deconvolution


def test_run_spectral_deconvolution_accepts_ms_time_offset():
    sig = inspect.signature(run_spectral_deconvolution)
    assert "ms_time_offset" in sig.parameters
    assert sig.parameters["ms_time_offset"].default == 0.0
```

- [ ] **Step 3: Run test**

Run: `pytest tests/logic/test_ms_time.py::test_run_spectral_deconvolution_accepts_ms_time_offset -v`
Expected: FAIL.

- [ ] **Step 4: Modify `logic/spectral_deconv_runner.py`**

Add at the top:

```python
from logic.ms_time import shifted_xlabels
```

Add `ms_time_offset: float = 0.0` (keyword-only) to `run_spectral_deconvolution`'s signature.

Replace lines 188–189:

```python
    rt_min = float(ms.xlabels[0])
    rt_max = float(ms.xlabels[-1])
```

with:

```python
    _xlabels = shifted_xlabels(ms, ms_time_offset)
    rt_min = float(_xlabels[0])
    rt_max = float(_xlabels[-1])
```

Find the call to `extract_eic_peaks(...)` in this file and add `ms_time_offset=ms_time_offset` to it.

- [ ] **Step 5: Run test**

Run: `pytest tests/logic/test_ms_time.py -v`
Expected: all pass.

- [ ] **Step 6: Run the spectral deconv runner's existing tests**

Run: `pytest tests/test_spectral_deconv_runner.py -v`
Expected: still passes.

- [ ] **Step 7: Commit**

```bash
git add logic/spectral_deconv_runner.py tests/logic/test_ms_time.py
git commit -m "feat(spectral_deconv_runner): propagate ms_time_offset to EIC and windows"
```

---

## Task 7: Update `SpectralDeconvWorker` and call sites to pass the offset

**Files:**
- Modify: `logic/spectral_deconv_worker.py`
- Modify: `ui/app.py` (call sites that build/launch the worker)
- Modify: `ui/dialogs/spectral_deconv_inspector.py` (its `_PreviewWorker`)

The runner now accepts `ms_time_offset` but the worker that calls it doesn't pass anything yet. Wire it through.

- [ ] **Step 1: Inspect the worker**

Run: `grep -n "run_spectral_deconvolution\|class.*Worker\|def __init__\|def run" logic/spectral_deconv_worker.py`

- [ ] **Step 2: Add `ms_time_offset` to `SpectralDeconvWorker.__init__`**

Add a `ms_time_offset: float = 0.0` kwarg. Store it on `self`. In its `run` method (where it calls `run_spectral_deconvolution`), pass `ms_time_offset=self.ms_time_offset`.

- [ ] **Step 3: Update `ui/app.py` to pass the offset when constructing the worker**

Find every `SpectralDeconvWorker(...)` construction (use grep). At each site, add `ms_time_offset=self.data_handler.ms_time_offset` (or whatever the local DataHandler reference is named — confirm with `grep -n "self.data_handler\|self\\._data_handler" ui/app.py | head`).

- [ ] **Step 4: Update the inspector's `_PreviewWorker`**

In `ui/dialogs/spectral_deconv_inspector.py`, find the `_PreviewWorker` class (around line 31). It calls `extract_eic_peaks(...)` and `deconvolve(...)`. Add a `ms_time_offset` parameter, store on `self`, and pass to `extract_eic_peaks`. The deconvolve call does not take `ms_time_offset` (its inputs are already in time-shifted space because EIC extraction applied the shift).

In the inspector's `_run_preview` method (line 488), pass the current preview offset (`self._current_preview_offset_min`, introduced in Task 9) when constructing `_PreviewWorker`. Until Task 9, default to `0.0` here so this task is self-contained.

- [ ] **Step 5: Run the existing test suite**

```bash
pytest tests/ -x -q
```

Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add logic/spectral_deconv_worker.py ui/app.py ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat(deconv): pass ms_time_offset through worker and inspector preview"
```

---

## Task 8: Update inspector's own MS re-read to use shifted xlabels

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py`

The inspector independently opens `data.ms` and reads `xlabels` at lines 375–380 (for `_rebuild_windows` etc.).

- [ ] **Step 1: Add import**

At the top of `ui/dialogs/spectral_deconv_inspector.py`:

```python
from logic.ms_time import shifted_xlabels
```

- [ ] **Step 2: Add a preview-offset attribute to the dialog**

In `SpectralDeconvInspectorDialog.__init__`, add:

```python
        self._current_preview_offset_min: float = 0.0
```

- [ ] **Step 3: Replace direct `xlabels` reads**

At lines 379–380, change:

```python
        rt_min = float(self._ms.xlabels[0])
        rt_max = float(self._ms.xlabels[-1])
```

to:

```python
        _xlabels = shifted_xlabels(self._ms, self._current_preview_offset_min)
        rt_min = float(_xlabels[0])
        rt_max = float(_xlabels[-1])
```

(Audit with `grep -n "self._ms.xlabels" ui/dialogs/spectral_deconv_inspector.py` — fix any additional hits the same way.)

- [ ] **Step 4: Smoke-test by launching the app**

Run: `chromakit-ms`

Open a `.D` file, run Deconvolve MS, open Inspect. Confirm the inspector still renders correctly (no behavior change since `_current_preview_offset_min` is 0).

Close the app.

- [ ] **Step 5: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat(inspector): use shifted_xlabels for own MS re-reads"
```

---

## Task 9: Inspector UI — add MS Time Offset group with live preview

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py`

This task adds the slider + Auto + Reset + Apply globally controls and wires live preview to the existing render path. "Apply globally" is wired in Task 10 (this task emits a signal; the slot is added next task).

- [ ] **Step 1: Add Qt imports**

Confirm `QSlider`, `QPushButton`, `QLabel`, `QHBoxLayout`, `QGroupBox`, `Signal` are imported in the dialog file. Add what's missing.

- [ ] **Step 2: Add the `apply_offset_requested` signal**

At the class level of `SpectralDeconvInspectorDialog`, near other signals:

```python
    apply_offset_requested = Signal(float, str)  # (offset_min, source: "manual"|"auto")
```

- [ ] **Step 3: Build the UI group**

In `_build_params_panel`, insert at the top of the panel (before existing controls) a new `QGroupBox("MS Time Offset")` containing:

```python
        offset_group = QGroupBox("MS Time Offset")
        og_layout = QVBoxLayout(offset_group)

        # Slider row
        slider_row = QHBoxLayout()
        self._offset_slider = QSlider(Qt.Horizontal)
        self._offset_slider.setMinimum(-500)
        self._offset_slider.setMaximum(500)
        self._offset_slider.setValue(0)
        self._offset_slider.setSingleStep(1)
        self._offset_slider.setPageStep(10)
        self._offset_readout = QLabel("0.0000 min (0.00 s)")
        self._offset_readout.setMinimumWidth(150)
        slider_row.addWidget(self._offset_slider, 1)
        slider_row.addWidget(self._offset_readout)
        og_layout.addLayout(slider_row)

        # Button row
        btn_row = QHBoxLayout()
        self._offset_auto_btn = QPushButton("Auto")
        self._offset_reset_btn = QPushButton("Reset")
        self._offset_apply_btn = QPushButton("Apply globally")
        btn_row.addWidget(self._offset_auto_btn)
        btn_row.addWidget(self._offset_reset_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._offset_apply_btn)
        og_layout.addLayout(btn_row)

        # Place this group at the top of the existing params panel layout.
        params_layout.insertWidget(0, offset_group)
```

(Adjust `params_layout` to the actual local variable name used in `_build_params_panel`.)

- [ ] **Step 4: Track manual-vs-auto source**

Add to `__init__`:

```python
        self._offset_source: str = "manual"
```

- [ ] **Step 5: Wire slider movement → readout + live preview**

In `_build_params_panel` (after creating the controls):

```python
        self._offset_slider.valueChanged.connect(self._on_offset_slider_changed)
        self._offset_auto_btn.clicked.connect(self._on_offset_auto_clicked)
        self._offset_reset_btn.clicked.connect(self._on_offset_reset_clicked)
        self._offset_apply_btn.clicked.connect(self._on_offset_apply_clicked)
```

Add these handler methods to `SpectralDeconvInspectorDialog`:

```python
    def _on_offset_slider_changed(self, value: int) -> None:
        offset_min = value / 1000.0   # 1 ms resolution over ±0.500 min
        self._current_preview_offset_min = offset_min
        self._offset_readout.setText(
            f"{offset_min:+.4f} min ({offset_min * 60:+.2f} s)"
        )
        # Any user movement promotes source back to manual.
        self._offset_source = "manual"
        self._replot_with_current_offset()

    def _on_offset_auto_clicked(self) -> None:
        from logic.processor import ChromatogramProcessor
        # Need FID and TIC arrays for the current file.
        if self._fid_time is None or self._fid_signal is None or self._tic_time is None or self._tic_signal is None:
            return
        proc = ChromatogramProcessor()
        _, _, lag_seconds = proc.align_tic_to_fid(
            self._fid_time, self._fid_signal,
            self._tic_time, self._tic_signal,
            verbose=False,
        )
        # lag_seconds is the shift to apply to TIC time (in seconds).
        # Convert to minutes; the slider range is ±0.5 min.
        offset_min = -lag_seconds / 60.0
        clamped = max(-0.5, min(0.5, offset_min))
        self._offset_slider.setValue(int(round(clamped * 1000)))
        self._offset_source = "auto"

    def _on_offset_reset_clicked(self) -> None:
        self._offset_slider.setValue(0)
        self._offset_source = "manual"

    def _on_offset_apply_clicked(self) -> None:
        self.apply_offset_requested.emit(
            self._current_preview_offset_min,
            self._offset_source,
        )

    def _replot_with_current_offset(self) -> None:
        """Re-render the cached deconvolution result with the new preview offset.

        The FID dashed lines stay where they are (FID time is the reference);
        the m/z component scatter and EIC traces shift by `self._current_preview_offset_min`.
        """
        if not hasattr(self, "_last_render_payload") or self._last_render_payload is None:
            return
        self._render_plots(self._last_render_payload)
```

- [ ] **Step 6: Cache the render payload and apply the offset at render time**

Find the existing `_render_plots(self, payload)` method (around line 721). At the very top of the method, capture the payload for live re-renders:

```python
        self._last_render_payload = payload
```

Then locate every place inside `_render_plots` that uses an MS-side RT (component RTs from `payload["components"]`, EIC trace times, model peak times). Add the preview offset to each. Concretely, define near the top:

```python
        offset = float(getattr(self, "_current_preview_offset_min", 0.0))
```

and add `+ offset` to every MS-derived RT array/scalar plotted in `_ax_scatter` and `_ax_eic`. **Do not** add the offset to the FID dashed lines (the existing `axvline(rt, ...)` calls at lines 820 and 865 — leave those alone).

For the EIC traces, this typically means replacing things like:

```python
        self._ax_eic.plot(eic.time, eic.intensity, ...)
```

with:

```python
        self._ax_eic.plot(eic.time + offset, eic.intensity, ...)
```

(Inspect the actual variable names in `_render_plots` before editing; they may differ.)

For the cluster scatter, replace `comp.rt` with `comp.rt + offset` everywhere it is plotted.

- [ ] **Step 7: Initialize the slider from `DataHandler.ms_time_offset` when the dialog opens**

In `SpectralDeconvInspectorDialog.__init__` (or wherever the dialog is shown), accept an `initial_offset_min: float = 0.0` parameter and set the slider after construction:

```python
        self._offset_slider.setValue(int(round(initial_offset_min * 1000)))
        self._current_preview_offset_min = initial_offset_min
```

Update `ChromaKitApp._on_inspect_requested` (`ui/app.py:3112`) to pass the current `DataHandler.ms_time_offset` when constructing the inspector. Concretely:

```python
        self._deconv_inspector = SpectralDeconvInspectorDialog(
            ...existing args...,
            initial_offset_min=self.data_handler.ms_time_offset,
            parent=self,
        )
```

- [ ] **Step 8: Update the per-window Preview worker to use the current offset**

In `_run_preview` (line 488), the `_PreviewWorker` was constructed in Task 7 with a hard-coded `ms_time_offset=0.0`. Change that to:

```python
        worker = _PreviewWorker(
            ...existing args...,
            ms_time_offset=self._current_preview_offset_min,
        )
```

This ensures that if the user adjusts the slider and then clicks the dialog's "Preview" button (per-window deconvolution), the preview EIC extraction uses the live offset.

- [ ] **Step 9: Smoke-test**

Run: `chromakit-ms`

Open a `.D` file. Run Deconvolve MS. Open Inspect. Verify:
1. Slider exists at the top of the params panel with a readout.
2. Dragging the slider updates the readout in 0.001 min steps.
3. Dragging shifts the m/z component scatter and EIC traces; FID dashed lines stay still.
4. "Auto" populates a non-zero value (assuming any drift exists).
5. "Reset" returns the slider to 0.

Close the app.

- [ ] **Step 10: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py ui/app.py
git commit -m "feat(inspector): add MS time offset slider with live preview"
```

---

## Task 10: Wire "Apply globally" — sidecar write, DataHandler update, deconv rerun

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Add slot in `ChromaKitApp`**

In `ui/app.py`, near `_on_inspector_rerun_requested` (line 3147), add:

```python
    def _on_inspector_apply_offset_requested(self, offset_min: float, source: str) -> None:
        """Persist the MS time offset and trigger a deconvolution rerun."""
        from logic.sidecar_offsets import save_offset, DEFAULT_SIDECAR_PATH

        data_path = self.data_handler.current_directory_path
        if not data_path:
            self.statusBar().showMessage("No file loaded; offset not applied.", 5000)
            return

        # Persist to sidecar.
        save_offset(str(data_path), offset_min, source=source, sidecar_path=DEFAULT_SIDECAR_PATH)

        # Apply to DataHandler.
        self.data_handler.ms_time_offset = offset_min

        # Update plot legend / TIC label (Task 11 hooks this).
        if hasattr(self, "plot_frame") and hasattr(self.plot_frame, "set_ms_time_offset"):
            self.plot_frame.set_ms_time_offset(offset_min)

        # Trigger deconvolution rerun via existing path.
        self._on_inspector_rerun_requested()

        # Close the inspector.
        if self._deconv_inspector is not None:
            self._deconv_inspector.close()

        self.statusBar().showMessage(
            f"MS offset applied: {offset_min:+.4f} min ({source}). "
            "Library search results may be stale — rerun if needed.",
            10000,
        )
```

- [ ] **Step 2: Connect the signal when the inspector is created**

In `_on_inspect_requested` (line 3112), after constructing `self._deconv_inspector`, add:

```python
        self._deconv_inspector.apply_offset_requested.connect(
            self._on_inspector_apply_offset_requested
        )
```

- [ ] **Step 3: Hook sidecar autoload on file open**

Find where `DataHandler` loads a new file (typically `load_data` or similar — `grep -n "current_directory_path\s*=" logic/data_handler.py`). Immediately after `self.current_directory_path = data_path`, add:

```python
        self.apply_offset_from_sidecar(str(data_path))
```

In `ui/app.py`, after a file is loaded successfully, show the offset in the status bar:

```python
        if self.data_handler.ms_time_offset != 0.0:
            self.statusBar().showMessage(
                f"MS offset active: {self.data_handler.ms_time_offset:+.4f} min", 5000
            )
```

(Place this in the existing file-load success path; locate via `grep -n "load_data\|file_selected\|on_file" ui/app.py | head`.)

- [ ] **Step 4: Smoke-test**

Run: `chromakit-ms`

1. Load a `.D` file. No offset → no status bar message about offset.
2. Run Deconvolve MS, open Inspect, drag the slider to e.g. −0.050.
3. Click "Apply globally". Verify:
   - Inspector closes.
   - Status bar message appears.
   - Deconvolution reruns (table updates).
   - `overrides/ms_time_offsets.json` exists and contains the entry for the loaded file.
4. Close and reopen the app. Reload the same `.D` file. Verify status bar shows the offset and `DataHandler.ms_time_offset` is the saved value (add a temporary debug print if needed).

- [ ] **Step 5: Commit**

```bash
git add ui/app.py logic/data_handler.py
git commit -m "feat(app): wire Apply globally to persist offset and rerun deconvolution"
```

---

## Task 11: Plot legend annotation when offset is nonzero

**Files:**
- Modify: `ui/frames/plot.py`

- [ ] **Step 1: Add `set_ms_time_offset` to `PlotFrame`**

In `ui/frames/plot.py`, add the attribute in `__init__`:

```python
        self.ms_time_offset: float = 0.0
```

Add the method:

```python
    def set_ms_time_offset(self, offset_min: float) -> None:
        """Store the active MS time offset and trigger a replot so the legend updates."""
        self.ms_time_offset = float(offset_min)
        self.update_plot()  # or whatever the existing replot method is named
```

(Confirm the existing replot method name via `grep -n "def update_plot\|def replot\|def draw" ui/frames/plot.py | head`.)

- [ ] **Step 2: Append the offset to the TIC legend label**

Find where the TIC trace is added to the plot (search for `'TIC'` or `label='TIC'` in `plot.py`). Wrap the label:

```python
        tic_label = "TIC"
        if self.ms_time_offset != 0.0:
            tic_label = f"TIC (offset {self.ms_time_offset:+.4f} min)"
        ax.plot(self.tic_data['x'], self.tic_data['y'], label=tic_label, ...)
```

- [ ] **Step 3: Smoke-test**

Run: `chromakit-ms`. Load a file with a saved offset. Verify the TIC trace legend reads `"TIC (offset −0.0480 min)"` (or similar).

- [ ] **Step 4: Commit**

```bash
git add ui/frames/plot.py
git commit -m "feat(plot): annotate TIC legend with active MS time offset"
```

---

## Task 12: Batch automation — autoload sidecar offset

**Files:**
- Modify: any automation file that loads `.D` files (likely `ui/dialogs/automation.py` or `logic/automation_worker.py` — locate via `grep -rn "class AutomationWorker" .`)

- [ ] **Step 1: Locate the worker**

Run: `grep -rn "class AutomationWorker\|def run.*self.*Automation" .`

- [ ] **Step 2: Add sidecar autoload at file-open time**

Find the per-file processing loop. Where it currently calls something like `data_handler.load_data(path)` (or constructs a DataHandler per file), the `apply_offset_from_sidecar` call added in Task 3 (Step 4 — DataHandler load path) already triggers automatically. **Verify** the automation worker uses the same load path — if it bypasses it and reads `.D` files directly, add an explicit call:

```python
        data_handler.apply_offset_from_sidecar(str(path))
```

after the file is opened.

- [ ] **Step 3: Smoke-test**

Run a small batch automation on a folder containing at least one file with a saved sidecar offset. Confirm the offset is honored (deconvolution results match what the inspector showed).

- [ ] **Step 4: Commit**

```bash
git add <files modified>
git commit -m "feat(automation): honor per-file MS time offset in batch processing"
```

---

## Task 13: Add offset to exported results JSON

**Files:**
- Modify: `logic/json_exporter.py`
- Update: `tests/logic/test_method.py` (remove `align_tic` from fixture)

- [ ] **Step 1: Read the current metadata writer**

Run: `sed -n '125,145p' logic/json_exporter.py`

Note where `bl_info['align_tic']` is written.

- [ ] **Step 2: Replace the field**

Delete the `bl_info['align_tic'] = ...` line. Add to the metadata block (at the appropriate level — top-level metadata, not inside `bl_info`):

```python
    metadata["ms_time_offset"] = float(getattr(data_handler, "ms_time_offset", 0.0))
    metadata["ms_time_offset_source"] = (
        "manual" if metadata["ms_time_offset"] != 0.0 else None
    )
```

(Note: tracking whether the offset came from `"manual"` vs `"auto"` requires reading the sidecar entry here. For now, infer `"manual"` for any nonzero applied offset — the sidecar is the source of truth for provenance if anyone needs it. If full provenance is required, fetch via `load_offset(data_handler.current_directory_path)` and use `entry.source`.)

Prefer the precise version:

```python
    from logic.sidecar_offsets import load_offset, DEFAULT_SIDECAR_PATH
    metadata["ms_time_offset"] = float(getattr(data_handler, "ms_time_offset", 0.0))
    if data_handler.current_directory_path:
        entry = load_offset(str(data_handler.current_directory_path), sidecar_path=DEFAULT_SIDECAR_PATH)
        metadata["ms_time_offset_source"] = entry.source if entry else None
    else:
        metadata["ms_time_offset_source"] = None
```

(Adjust variable names to match what `json_exporter.py` actually receives. `data_handler` is illustrative — it may be `app` or similar.)

- [ ] **Step 3: Update `tests/logic/test_method.py`**

Remove `"align_tic": False,` from `_GUI_PARAMS["baseline"]` (line 28).

- [ ] **Step 4: Run tests**

```bash
pytest tests/logic/test_method.py -v
```

Expected: still passes (this only removes a field that's about to be removed from the model in Task 14).

- [ ] **Step 5: Smoke-test export**

In the running app, load a file with an offset, integrate, and export results. Inspect the exported JSON — confirm `ms_time_offset` and `ms_time_offset_source` are present and `align_tic` is absent.

- [ ] **Step 6: Commit**

```bash
git add logic/json_exporter.py tests/logic/test_method.py
git commit -m "feat(export): record ms_time_offset and source in results metadata"
```

---

## Task 14: Remove dead `align_tic` infrastructure

**Files:**
- Modify: `logic/method.py`
- Modify: `ui/frames/parameters.py`
- Modify: `ui/frames/plot.py`
- Modify: `api/models.py`
- Modify: `api/main.py`

- [ ] **Step 1: Delete `align_tic` field in `logic/method.py`**

Remove line 55:

```python
    align_tic: bool = Field(default=False, description="Align MS TIC to FID time axis")
```

- [ ] **Step 2: Delete checkbox + handler in `ui/frames/parameters.py`**

Remove:
- `'align_tic': False` from the default params dict (line 47).
- The `QCheckBox("Align TIC with FID")` block (lines 440–449).
- The `_on_align_tic_toggled` method (lines 1570–1586).
- Any signal connection to that method.

- [ ] **Step 3: Delete `aligned_tic_data` / `set_aligned_tic_data` in `ui/frames/plot.py`**

Remove:
- `self.aligned_tic_data = None` and `self.tic_alignment_info = None` (lines 71–72).
- The `set_aligned_tic_data` method (lines 978–982).

Keep the new `set_ms_time_offset` method added in Task 11.

- [ ] **Step 4: Delete API surface**

In `api/models.py`, remove `AlignTICRequest` (lines 103–108) and `AlignTICResponse` (lines 190–194).

In `api/main.py`, remove the `POST /api/align-tic` endpoint (lines 248–266) and its imports.

- [ ] **Step 5: Verify nothing imports the deleted symbols**

```bash
grep -rn "align_tic\|AlignTICRequest\|AlignTICResponse\|aligned_tic_data\|set_aligned_tic_data" . --include="*.py"
```

Expected: zero hits (except possibly in old docstrings / comments — clean those too).

- [ ] **Step 6: Run the test suite**

```bash
pytest tests/ -x -q
```

Expected: all pre-existing passing tests still pass.

- [ ] **Step 7: Smoke-test the GUI and API**

Run: `chromakit-ms` — confirm the parameters panel no longer shows the "Align TIC with FID" checkbox and the app launches without errors.

Run: `cd api && python main.py` — confirm the server starts and `/docs` no longer lists `/api/align-tic`.

- [ ] **Step 8: Commit**

```bash
git add logic/method.py ui/frames/parameters.py ui/frames/plot.py api/models.py api/main.py
git commit -m "chore: remove dead align_tic infrastructure (replaced by ms_time_offset)"
```

---

## Task 15: End-to-end manual verification

**Files:** none.

- [ ] **Step 1: Clean state**

Run: `rm -f overrides/ms_time_offsets.json`

- [ ] **Step 2: Apply offset workflow**

Run: `chromakit-ms`.

1. Load a `.D` file that you know has FID/MS drift (the 200-peak sample from the original report is ideal).
2. Process / integrate so FID peaks exist.
3. Click **Deconvolve MS**, wait for completion.
4. Click **Inspect**.
5. Confirm the dashed blue FID lines visibly miss the m/z components by ~0.05 min.
6. Drag the slider until the components align under the FID lines. Note the value.
7. Click **Auto** — confirm it produces a similar value to your manual adjustment.
8. Click **Apply globally**.
9. Confirm:
   - Inspector closes.
   - Status bar shows applied value.
   - Deconvolution reruns; table updates.
   - Reopening Inspect shows aligned dashed lines and components.
   - TIC trace legend in main plot reads `"TIC (offset ±0.04XX min)"`.

- [ ] **Step 3: Persistence check**

Quit the app. Inspect `overrides/ms_time_offsets.json` — confirm the entry.

Relaunch the app. Reload the file. Confirm:
- Status bar shows the offset on load.
- TIC legend already shows the offset.
- Deconvolving MS produces the aligned result without needing to set the offset again.

- [ ] **Step 4: Export check**

Export results to JSON. Inspect the file — confirm `ms_time_offset` and `ms_time_offset_source` are present in the metadata block and `align_tic` is absent.

- [ ] **Step 5: Reset check**

Reopen Inspect, click **Reset**, click **Apply globally**. Confirm the sidecar entry becomes `0.0` and the legend annotation disappears.

- [ ] **Step 6: Final test sweep**

```bash
pytest tests/ -q
```

Expected: no regressions.

- [ ] **Step 7: Commit any final tweaks discovered during E2E (if any)**

If no changes needed, no commit. Otherwise:

```bash
git add <files>
git commit -m "fix: e2e adjustments to MS time offset workflow"
```

---

## Notes for the implementer

- **Be conservative about file edits.** Confirm line numbers cited in this plan with `grep` / `sed -n` before editing. The repo evolves; lines shift.
- **Keep diffs small per commit.** The plan structures one logical change per task; resist the urge to combine.
- **The slider mapping** (`int / 1000.0` → minutes over ±0.5) is deliberate: 1 ms precision is far below GC scan rates and gives enough range for any reasonable detector skew.
- **The Auto button uses the existing `align_tic_to_fid` function**, which returns `lag_seconds` such that `aligned_tic_time = tic_time - lag/60`. The MS offset that produces the equivalent shift is `-lag_seconds / 60`. Triple-check the sign during testing.
- **Library search rerun is intentionally not auto-triggered** after Apply globally. The status bar message warns the user. This is the agreed v1 scope.
