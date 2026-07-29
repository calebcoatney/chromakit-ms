# Agilent .D → xlsx Export Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone PySide6 utility that exports one or more Agilent `.D` directories to a single `.xlsx` workbook (one sheet per folder, shared time axis, one column per selected signal).

**Architecture:** A single self-contained file `util/dxlsx_export.py`. Pure helper functions (rainbow/numpy) do the data reading, alignment, interpolation, and sheet-name sanitizing. A `QDialog` provides the GUI, and a `QThread` `ExportWorker` runs the export. Pure functions are unit-tested with stub rainbow objects; no real `.D` fixtures needed.

**Tech Stack:** Python, PySide6, rainbow, numpy, openpyxl. No scipy (uses `numpy.interp` + manual NaN masking).

**Spec:** `docs/superpowers/specs/2026-07-29-dxlsx-export-tool-design.md`

---

### Task 1: Pure data-reading helpers

Rainbow-reading helpers. Tested against small stub objects that mimic rainbow's `datafiles` / `get_file` / `metadata` / `name` API, so no real `.D` directory is required.

**Files:**
- Create: `util/dxlsx_export.py`
- Test: `tests/util/test_dxlsx_export.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/util/test_dxlsx_export.py
import numpy as np
import pytest
from util import dxlsx_export as dx


class StubDataFile:
    def __init__(self, name, xlabels, data, metadata=None):
        self.name = name
        self.xlabels = np.asarray(xlabels)
        self.data = np.asarray(data)
        self.metadata = metadata or {}


class StubDataDir:
    def __init__(self, name, datafiles, metadata=None):
        self.name = name
        self.datafiles = datafiles
        self.metadata = metadata or {}

    def get_file(self, fname):
        for f in self.datafiles:
            if f.name == fname:
                return f
        raise KeyError(fname)


def make_dir():
    fid = StubDataFile("FID1A.ch", [0.0, 0.5, 1.0], [10.0, 20.0, 30.0],
                       metadata={"notebook": "SampleA"})
    # MS: 2 time points x 3 m/z -> TIC = row sums = [6, 60]
    ms = StubDataFile("data.ms", [0.5, 1.0], [[1, 2, 3], [10, 20, 30]])
    return StubDataDir("dirname", [fid, ms], metadata={"notebook": "DirNB"})


def test_list_signals_returns_ch_and_ms():
    d = make_dir()
    assert dx.list_signals(d) == ["FID1A.ch", "data.ms"]


def test_read_signal_ch():
    d = make_dir()
    x, y = dx.read_signal(d, "FID1A.ch")
    assert np.allclose(x, [0.0, 0.5, 1.0])
    assert np.allclose(y, [10.0, 20.0, 30.0])


def test_read_signal_ms_is_tic():
    d = make_dir()
    x, y = dx.read_signal(d, "data.ms")
    assert np.allclose(x, [0.5, 1.0])
    assert np.allclose(y, [6.0, 60.0])  # TIC = sum over m/z


def test_read_notebook_prefers_detector_file():
    d = make_dir()
    # FID1A.ch metadata notebook = "SampleA"
    assert dx.read_notebook(d, "/some/path.D") == "SampleA"


def test_read_notebook_falls_back_to_dir_then_basename():
    fid = StubDataFile("FID1A.ch", [0.0], [1.0], metadata={})
    d = StubDataDir("dname", [fid], metadata={})
    # no notebook anywhere -> data_dir.name
    assert dx.read_notebook(d, "/x/basename.D") == "dname"
```

Note: `list_signals` takes an already-opened data dir in tests. The real function accepts a path and calls `rb.read`. To keep it testable, split into `list_signals_from_dir(data_dir)` (pure) and `list_signals(path)` (wrapper). Adjust the test import accordingly if you rename — but the plan uses `list_signals(data_dir)` accepting a dir object directly for simplicity. Keep the signature accepting a dir object.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'util.dxlsx_export'`

- [ ] **Step 3: Write minimal implementation**

```python
# util/dxlsx_export.py
# -*- coding: utf-8 -*-
"""Standalone utility: export Agilent .D directories to a single .xlsx workbook.

One worksheet per .D folder, a shared Time (min) column, and one column per
selected signal (GC .ch detectors and/or MS data.ms TIC), interpolated onto a
common retention-time grid.

Run: python util/dxlsx_export.py
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Data-reading helpers (rainbow + numpy only)
# ---------------------------------------------------------------------------

def list_signals(data_dir):
    """Return signal filenames in a rainbow data dir: '*.ch' plus 'data.ms'."""
    names = [str(f.name) for f in data_dir.datafiles]
    signals = [n for n in names if n.endswith(".ch")]
    if "data.ms" in names:
        signals.append("data.ms")
    return signals


def read_signal(data_dir, signal):
    """Return (x_minutes, y) for a signal. MS 'data.ms' returns the TIC."""
    f = data_dir.get_file(signal)
    x = np.asarray(f.xlabels, dtype=float).flatten()
    if signal == "data.ms":
        y = np.sum(np.asarray(f.data, dtype=float), axis=1)
    else:
        y = np.asarray(f.data, dtype=float).flatten()
    return x, y


def read_notebook(data_dir, d_path):
    """Best-effort notebook name: detector-file metadata -> dir metadata ->
    data_dir.name -> folder basename."""
    detector_files = [f for f in data_dir.datafiles if str(f.name).endswith(".ch")]
    if detector_files:
        nb = detector_files[0].metadata.get("notebook")
        if nb:
            return str(nb)
    nb = data_dir.metadata.get("notebook")
    if nb:
        return str(nb)
    name = getattr(data_dir, "name", None)
    if name:
        return str(name)
    return os.path.splitext(os.path.basename(d_path.rstrip("/\\")))[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add util/dxlsx_export.py tests/util/test_dxlsx_export.py
git commit -m "feat(util): add rainbow data-reading helpers for .D xlsx export"
```

---

### Task 2: Alignment & interpolation onto a common grid

Given multiple signals with different ranges, build a union time grid and resample each signal, masking out-of-range points to NaN.

**Files:**
- Modify: `util/dxlsx_export.py`
- Test: `tests/util/test_dxlsx_export.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/util/test_dxlsx_export.py

def test_build_grid_union_range():
    sig_x = {"A": np.array([0.0, 1.0]), "B": np.array([0.5, 2.0])}
    grid = dx.build_time_grid(sig_x, skip_solvent_delay=False, has_ms=False,
                              ms_x=None, n=5)
    assert np.isclose(grid[0], 0.0)
    assert np.isclose(grid[-1], 2.0)
    assert len(grid) == 5


def test_build_grid_clips_to_ms_start_when_enabled():
    sig_x = {"FID1A.ch": np.array([0.0, 2.0]), "data.ms": np.array([1.8, 2.0])}
    grid = dx.build_time_grid(sig_x, skip_solvent_delay=True, has_ms=True,
                              ms_x=np.array([1.8, 2.0]), n=5)
    assert np.isclose(grid[0], 1.8)
    assert np.isclose(grid[-1], 2.0)


def test_resample_masks_out_of_range_to_nan():
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    x = np.array([0.5, 1.0, 1.5])
    y = np.array([5.0, 10.0, 15.0])
    out = dx.resample_to_grid(grid, x, y)
    assert np.isnan(out[0])   # 0.0 < x.min()
    assert np.isclose(out[1], 5.0)
    assert np.isclose(out[2], 10.0)
    assert np.isclose(out[3], 15.0)
    assert np.isnan(out[4])   # 2.0 > x.max()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k "grid or resample" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_time_grid'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to util/dxlsx_export.py

def build_time_grid(sig_x, skip_solvent_delay, has_ms, ms_x, n):
    """Build a common time grid (minutes) over the union range of all signals.

    sig_x: dict signal_name -> native x array.
    If skip_solvent_delay and has_ms, clip the start to min(ms_x).
    """
    starts = [np.min(x) for x in sig_x.values() if len(x)]
    ends = [np.max(x) for x in sig_x.values() if len(x)]
    t_min = min(starts)
    t_max = max(ends)
    if skip_solvent_delay and has_ms and ms_x is not None and len(ms_x):
        t_min = float(np.min(ms_x))
    return np.linspace(t_min, t_max, int(n))


def resample_to_grid(grid, x, y):
    """np.interp onto grid, masking points outside [x.min(), x.max()] to NaN."""
    out = np.interp(grid, x, y)
    out = np.where((grid < np.min(x)) | (grid > np.max(x)), np.nan, out)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k "grid or resample" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add util/dxlsx_export.py tests/util/test_dxlsx_export.py
git commit -m "feat(util): add union time grid + NaN-masked resampling"
```

---

### Task 3: Sheet-name sanitizing & de-duplication

Excel sheet names: max 31 chars, no `[ ] : * ? / \`, unique.

**Files:**
- Modify: `util/dxlsx_export.py`
- Test: `tests/util/test_dxlsx_export.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/util/test_dxlsx_export.py

def test_sanitize_removes_invalid_and_truncates():
    used = set()
    name = dx.safe_sheet_name("a/b:c*d?e[f]g" * 5, used)
    for ch in r'[]:*?/\\':
        assert ch not in name
    assert len(name) <= 31
    assert name in used


def test_sanitize_dedupes_collisions():
    used = set()
    n1 = dx.safe_sheet_name("Sample", used)
    n2 = dx.safe_sheet_name("Sample", used)
    n3 = dx.safe_sheet_name("Sample", used)
    assert n1 == "Sample"
    assert n2 == "Sample_2"
    assert n3 == "Sample_3"


def test_sanitize_empty_falls_back():
    used = set()
    name = dx.safe_sheet_name("", used)
    assert name == "Sheet"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k sanitize -v`
Expected: FAIL — no attribute `safe_sheet_name`

- [ ] **Step 3: Write minimal implementation**

```python
# append to util/dxlsx_export.py

_INVALID_SHEET_CHARS = r'[]:*?/\\'


def safe_sheet_name(raw, used_names):
    """Sanitize to a valid, unique Excel sheet name; records result in used_names."""
    name = "".join(c for c in str(raw) if c not in _INVALID_SHEET_CHARS).strip()
    if not name:
        name = "Sheet"
    name = name[:31]
    base = name
    counter = 2
    while name in used_names:
        suffix = "_" + str(counter)
        name = base[:31 - len(suffix)] + suffix
        counter += 1
    used_names.add(name)
    return name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k sanitize -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add util/dxlsx_export.py tests/util/test_dxlsx_export.py
git commit -m "feat(util): add Excel sheet-name sanitizer with de-duplication"
```

---

### Task 4: Per-folder sheet builder (rows for openpyxl)

Combine helpers: read selected signals present in a folder, build the grid, resample each, and return a header row + data rows (NaN → None for empty cells).

**Files:**
- Modify: `util/dxlsx_export.py`
- Test: `tests/util/test_dxlsx_export.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/util/test_dxlsx_export.py

def test_build_sheet_rows_shape_and_headers():
    d = make_dir()  # FID1A.ch (0..1), data.ms (0.5..1.0)
    header, rows = dx.build_sheet_rows(
        d, selected=["FID1A.ch", "data.ms"],
        skip_solvent_delay=False, n=4)
    assert header == ["Time (min)", "FID1A.ch", "data.ms"]
    assert len(rows) == 4
    assert len(rows[0]) == 3
    # first grid point 0.0 is before MS start -> MS cell is None
    assert rows[0][2] is None


def test_build_sheet_rows_skips_absent_signals():
    fid = StubDataFile("FID1A.ch", [0.0, 1.0], [1.0, 2.0])
    d = StubDataDir("d", [fid])
    header, rows = dx.build_sheet_rows(
        d, selected=["FID1A.ch", "data.ms"],
        skip_solvent_delay=True, n=3)
    # data.ms not present -> column omitted
    assert header == ["Time (min)", "FID1A.ch"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k build_sheet_rows -v`
Expected: FAIL — no attribute `build_sheet_rows`

- [ ] **Step 3: Write minimal implementation**

```python
# append to util/dxlsx_export.py

def build_sheet_rows(data_dir, selected, skip_solvent_delay, n):
    """Return (header, rows) for one .D folder.

    header: ["Time (min)", <signal>, ...] for selected signals present here.
    rows: list of [time, val, ...] with NaN converted to None.
    """
    present = [s for s in selected if s in list_signals(data_dir)]
    sig_xy = {s: read_signal(data_dir, s) for s in present}
    sig_x = {s: xy[0] for s, xy in sig_xy.items()}

    has_ms = "data.ms" in present
    ms_x = sig_x.get("data.ms")
    grid = build_time_grid(sig_x, skip_solvent_delay, has_ms, ms_x, n)

    columns = [grid]
    header = ["Time (min)"]
    for s in present:
        x, y = sig_xy[s]
        columns.append(resample_to_grid(grid, x, y))
        header.append(s)

    rows = []
    for i in range(len(grid)):
        row = []
        for col in columns:
            v = col[i]
            row.append(None if (isinstance(v, float) and np.isnan(v)) else float(v))
        rows.append(row)
    return header, rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k build_sheet_rows -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add util/dxlsx_export.py tests/util/test_dxlsx_export.py
git commit -m "feat(util): build per-folder header + data rows for xlsx export"
```

---

### Task 5: Workbook writer (openpyxl)

Write a full workbook from a list of `(sheet_name, header, rows)` using openpyxl, with a bold header row.

**Files:**
- Modify: `util/dxlsx_export.py`
- Test: `tests/util/test_dxlsx_export.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/util/test_dxlsx_export.py
from openpyxl import load_workbook


def test_write_workbook_creates_sheets(tmp_path):
    out = tmp_path / "out.xlsx"
    sheets = [
        ("SampleA", ["Time (min)", "FID1A.ch"], [[0.0, 1.0], [0.5, None]]),
        ("SampleB", ["Time (min)", "data.ms"], [[1.0, 6.0]]),
    ]
    dx.write_workbook(str(out), sheets)
    wb = load_workbook(str(out))
    assert wb.sheetnames == ["SampleA", "SampleB"]
    ws = wb["SampleA"]
    assert ws.cell(row=1, column=1).value == "Time (min)"
    assert ws.cell(row=1, column=1).font.bold is True
    assert ws.cell(row=3, column=2).value is None  # NaN cell empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k write_workbook -v`
Expected: FAIL — no attribute `write_workbook`

- [ ] **Step 3: Write minimal implementation**

```python
# append to util/dxlsx_export.py
from openpyxl import Workbook
from openpyxl.styles import Font


def write_workbook(out_path, sheets):
    """sheets: list of (sheet_name, header, rows). Writes bold header row."""
    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)
    for sheet_name, header, rows in sheets:
        ws = wb.create_sheet(title=sheet_name)
        ws.append(header)
        for c in range(1, len(header) + 1):
            ws.cell(row=1, column=c).font = bold
        for row in rows:
            ws.append(row)
    wb.save(out_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k write_workbook -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add util/dxlsx_export.py tests/util/test_dxlsx_export.py
git commit -m "feat(util): write multi-sheet xlsx workbook with openpyxl"
```

---

### Task 6: Orchestration function (folders → sheets)

A single pure-ish function that ties folder reading to sheet building, using an injectable reader so it's testable without rainbow. This is what the worker calls.

**Files:**
- Modify: `util/dxlsx_export.py`
- Test: `tests/util/test_dxlsx_export.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/util/test_dxlsx_export.py

def test_export_folders_builds_sheets_and_skips_failures(tmp_path):
    good = make_dir()

    def reader(path):
        if path == "GOOD":
            return good
        raise IOError("bad .D")

    out = tmp_path / "wb.xlsx"
    logs = []
    result = dx.export_folders(
        ["GOOD", "BAD"], selected=["FID1A.ch", "data.ms"],
        skip_solvent_delay=False, n=4, out_path=str(out),
        reader=reader, log=logs.append, progress=lambda i: None)

    assert result["exported"] == 1
    assert result["skipped"] == 1
    assert out.exists()
    assert any("bad .D" in m or "BAD" in m for m in logs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k export_folders -v`
Expected: FAIL — no attribute `export_folders`

- [ ] **Step 3: Write minimal implementation**

```python
# append to util/dxlsx_export.py

def _default_reader(path):
    import rainbow as rb
    return rb.read(path)


def export_folders(folders, selected, skip_solvent_delay, n, out_path,
                   reader=_default_reader, log=print, progress=None):
    """Read each folder, build a sheet, write the workbook.

    Returns {'exported': int, 'skipped': int}. Failures are logged and skipped.
    """
    sheets = []
    used_names = set()
    exported = 0
    skipped = 0
    total = len(folders)
    for idx, folder in enumerate(folders):
        try:
            data_dir = reader(folder)
            header, rows = build_sheet_rows(data_dir, selected,
                                            skip_solvent_delay, n)
            if len(header) <= 1:
                log("Skipped (no selected signals present): " + str(folder))
                skipped += 1
            else:
                raw_name = read_notebook(data_dir, folder)
                sheet_name = safe_sheet_name(raw_name, used_names)
                sheets.append((sheet_name, header, rows))
                exported += 1
                log("Exported: " + str(folder) + " -> " + sheet_name)
        except Exception as e:  # noqa: BLE001 - report and continue
            log("Skipped (error): " + str(folder) + " (" + str(e) + ")")
            skipped += 1
        if progress is not None:
            progress(int((idx + 1) / max(total, 1) * 100))

    if sheets:
        write_workbook(out_path, sheets)
    else:
        log("No sheets to write.")
    return {"exported": exported, "skipped": skipped}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -k export_folders -v`
Expected: PASS

- [ ] **Step 5: Run the full util test file**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add util/dxlsx_export.py tests/util/test_dxlsx_export.py
git commit -m "feat(util): orchestrate .D folders into xlsx workbook with skip-on-error"
```

---

### Task 7: QThread export worker

Wrap `export_folders` in a `QThread` that emits progress/log/finished signals.

**Files:**
- Modify: `util/dxlsx_export.py`

- [ ] **Step 1: Add the worker (no unit test — GUI/Qt threading; verified via smoke run in Task 9)**

```python
# append to util/dxlsx_export.py
from PySide6.QtCore import QThread, Signal


class ExportWorker(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(dict)

    def __init__(self, folders, selected, skip_solvent_delay, n, out_path):
        super().__init__()
        self._folders = folders
        self._selected = selected
        self._skip = skip_solvent_delay
        self._n = n
        self._out = out_path

    def run(self):
        try:
            result = export_folders(
                self._folders, self._selected, self._skip, self._n, self._out,
                log=self.log.emit, progress=self.progress.emit)
        except Exception as e:  # noqa: BLE001
            self.log.emit("Fatal error: " + str(e))
            result = {"exported": 0, "skipped": len(self._folders), "error": str(e)}
        self.finished.emit(result)
```

- [ ] **Step 2: Verify import compiles**

Run: `conda run -n chromakit-env python -c "import util.dxlsx_export"`
Expected: no output, exit 0

- [ ] **Step 3: Commit**

```bash
git add util/dxlsx_export.py
git commit -m "feat(util): add QThread ExportWorker for .D xlsx export"
```

---

### Task 8: GUI dialog

The `QDialog`: folder list, signal-union checkboxes, options, output picker, export button, progress bar, log. Wires to `ExportWorker`. GUI wiring is verified by the smoke run in Task 9.

**Files:**
- Modify: `util/dxlsx_export.py`

- [ ] **Step 1: Add the dialog class**

```python
# append to util/dxlsx_export.py
import rainbow as rb
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QFileDialog, QTextEdit, QProgressBar, QMessageBox, QGroupBox,
    QCheckBox, QLineEdit, QScrollArea, QSpinBox, QListWidget,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QSettings


class ExportDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agilent .D → xlsx Export")
        self.resize(640, 640)
        self._settings = QSettings("CalebCoatney", "ChromaKit")
        self._folders = []          # list[str]
        self._signal_checks = {}    # signal name -> QCheckBox
        self._worker = None
        self._build_ui()

    # -- UI construction -------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Folders
        fbox = QGroupBox("Agilent .D folders")
        fl = QVBoxLayout(fbox)
        self._folder_list = QListWidget()
        self._folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        fl.addWidget(self._folder_list)
        frow = QHBoxLayout()
        add_btn = QPushButton("Add .D Folder(s)…")
        add_btn.clicked.connect(self._add_folders)
        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        frow.addWidget(add_btn)
        frow.addWidget(rm_btn)
        fl.addLayout(frow)
        layout.addWidget(fbox)

        # Signals
        sbox = QGroupBox("Signals to export")
        sl = QVBoxLayout(sbox)
        self._signal_area = QScrollArea()
        self._signal_area.setWidgetResizable(True)
        self._signal_inner = QWidget()
        self._signal_layout = QVBoxLayout(self._signal_inner)
        self._signal_area.setWidget(self._signal_inner)
        sl.addWidget(self._signal_area)
        layout.addWidget(sbox)

        # Options
        obox = QGroupBox("Options")
        ol = QHBoxLayout(obox)
        self._skip_check = QCheckBox("Skip solvent delay (clip to MS start when MS present)")
        self._skip_check.setChecked(True)
        ol.addWidget(self._skip_check)
        ol.addWidget(QLabel("Points per signal:"))
        self._points_spin = QSpinBox()
        self._points_spin.setRange(100, 200000)
        self._points_spin.setValue(int(self._settings.value("dxlsx/points", 10000)))
        ol.addWidget(self._points_spin)
        layout.addWidget(obox)

        # Output
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output:"))
        self._out_edit = QLineEdit()
        self._out_edit.setReadOnly(True)
        self._out_edit.setText(self._settings.value("dxlsx/out", ""))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_output)
        orow.addWidget(self._out_edit)
        orow.addWidget(browse)
        layout.addLayout(orow)

        # Export + progress + log
        self._export_btn = QPushButton("Export")
        self._export_btn.clicked.connect(self._start_export)
        layout.addWidget(self._export_btn)
        self._progress = QProgressBar()
        layout.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log)

    # -- Folder handling -------------------------------------------------
    def _add_folders(self):
        # QFileDialog directory mode selects one dir at a time; loop-friendly.
        d = QFileDialog.getExistingDirectory(self, "Select an Agilent .D folder")
        if d and d not in self._folders:
            self._folders.append(d)
            self._folder_list.addItem(d)
            self._rescan_signals()

    def _remove_selected(self):
        for item in self._folder_list.selectedItems():
            path = item.text()
            if path in self._folders:
                self._folders.remove(path)
            self._folder_list.takeItem(self._folder_list.row(item))
        self._rescan_signals()

    def _rescan_signals(self):
        prev = {s: cb.isChecked() for s, cb in self._signal_checks.items()}
        union = []
        for folder in self._folders:
            try:
                data_dir = rb.read(folder)
                for s in list_signals(data_dir):
                    if s not in union:
                        union.append(s)
            except Exception as e:  # noqa: BLE001
                self._log.append("Could not scan " + folder + ": " + str(e))
        # rebuild checkboxes
        while self._signal_layout.count():
            item = self._signal_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._signal_checks = {}
        for s in union:
            cb = QCheckBox(s)
            cb.setChecked(prev.get(s, True))
            self._signal_layout.addWidget(cb)
            self._signal_checks[s] = cb

    # -- Output ----------------------------------------------------------
    def _pick_output(self):
        start_dir = self._settings.value("dxlsx/out", "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save workbook", start_dir, "Excel Workbook (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self._out_edit.setText(path)

    # -- Export ----------------------------------------------------------
    def _start_export(self):
        selected = [s for s, cb in self._signal_checks.items() if cb.isChecked()]
        out_path = self._out_edit.text().strip()
        if not self._folders:
            QMessageBox.warning(self, "No folders", "Add at least one .D folder.")
            return
        if not selected:
            QMessageBox.warning(self, "No signals", "Select at least one signal.")
            return
        if not out_path:
            QMessageBox.warning(self, "No output", "Choose an output .xlsx path.")
            return

        self._settings.setValue("dxlsx/points", self._points_spin.value())
        self._settings.setValue("dxlsx/out", out_path)

        self._export_btn.setEnabled(False)
        self._progress.setValue(0)
        self._log.clear()

        self._worker = ExportWorker(
            list(self._folders), selected, self._skip_check.isChecked(),
            self._points_spin.value(), out_path)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.log.connect(self._log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, result):
        self._export_btn.setEnabled(True)
        msg = ("Exported " + str(result.get("exported", 0)) + " folder(s), skipped "
               + str(result.get("skipped", 0)) + ".")
        self._log.append(msg)
        QMessageBox.information(self, "Export complete", msg)
```

- [ ] **Step 2: Verify import compiles**

Run: `conda run -n chromakit-env python -c "import util.dxlsx_export"`
Expected: exit 0

- [ ] **Step 3: Commit**

```bash
git add util/dxlsx_export.py
git commit -m "feat(util): add ExportDialog GUI for .D xlsx export"
```

---

### Task 9: main() entry point + smoke run

**Files:**
- Modify: `util/dxlsx_export.py`

- [ ] **Step 1: Add main()**

```python
# append to util/dxlsx_export.py

def main():
    app = QApplication(sys.argv)
    dlg = ExportDialog()
    dlg.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the GUI launches headlessly**

Run: `conda run -n chromakit-env python -c "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from PySide6.QtWidgets import QApplication; import util.dxlsx_export as dx; app=QApplication([]); d=dx.ExportDialog(); print('OK', d.windowTitle())"`
Expected: prints `OK Agilent .D → xlsx Export`, exit 0

- [ ] **Step 3: Run the full util test suite once more**

Run: `conda run -n chromakit-env pytest tests/util/test_dxlsx_export.py -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add util/dxlsx_export.py
git commit -m "feat(util): add main() entry point for .D xlsx export tool"
```

---

## Self-Review

**Spec coverage:**
- Multi-folder selector → Task 8 (`_add_folders`, folder list). ✓
- Enumerate .ch/.ms signals with checkboxes, union set → Task 1 (`list_signals`) + Task 8 (`_rescan_signals`). ✓
- Skip-solvent-delay checkbox clips to MS start → Task 2 (`build_time_grid`) + Task 8 checkbox. ✓
- One sheet per folder named from notebook attr → Task 1 (`read_notebook`) + Task 3 (`safe_sheet_name`) + Task 6. ✓
- Shared Time (min) column + one column per signal, interpolated → Task 2 + Task 4. ✓
- Union-range fixed grid, numpy.interp + NaN masking → Task 2. ✓
- MS = TIC single column → Task 1 (`read_signal`). ✓
- Output workbook location picker → Task 8 (`_pick_output`). ✓
- Export button → Task 8 (`_start_export`). ✓
- QThread worker + progress/log → Task 7 + Task 8. ✓
- Skip-on-error per folder → Task 6. ✓
- util/ location, minimal rainbow reuse (no DataHandler) → all tasks; only `import rainbow`, numpy, openpyxl, PySide6. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code.

**Type consistency:** `list_signals`, `read_signal`, `read_notebook`, `build_time_grid`, `resample_to_grid`, `safe_sheet_name`, `build_sheet_rows`, `write_workbook`, `export_folders`, `ExportWorker`, `ExportDialog` — names and signatures used consistently across tasks. `export_folders` uses injectable `reader`/`log`/`progress` matching both the test (Task 6) and the worker (Task 7).

No gaps found.
