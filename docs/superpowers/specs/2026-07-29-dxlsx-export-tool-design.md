# Design: Agilent .D → xlsx Export Utility

**Date:** 2026-07-29
**Status:** Approved (design phase)

## Overview

A standalone PySide6 utility, **`util/dxlsx_export.py`**, that exports one or more
Agilent `.D` data directories to a single `.xlsx` workbook using the `rainbow`
library. Each `.D` folder becomes one worksheet; each worksheet has a shared
`Time (min)` column and one column per selected signal (GC `.ch` detectors and/or
the MS `data.ms` TIC), interpolated onto a common retention-time grid.

It matches the style and conventions of the existing `util/json_to_xlsx.py`
tool: a self-contained `QApplication` + `QDialog`, a `QThread` export worker with
a progress bar and log, and `QSettings("CalebCoatney", "ChromaKit")` for prefs.

### Placement & coupling

- Lives in `util/` alongside `json_to_xlsx.py`.
- **Minimal reuse**: lifts ~30 lines of rainbow-reading logic inline (ported from
  `logic/data_handler.py` and `logic/json_exporter.py`) rather than importing the
  heavy `DataHandler` (which transitively pulls in `spectrum_extractor` /
  `ms-toolkit`). Runs inside `chromakit-env`.
- Dependencies: `rainbow`, `numpy`, `openpyxl`, `PySide6` — all already project deps.
  **No scipy** — interpolation uses `numpy.interp` with manual NaN masking.

## Data Layer (lifted helpers)

Three pure functions at the top of the file (rainbow + numpy only):

- `list_signals(d_path) -> list[str]`
  - `rb.read(d_path).datafiles`; return names ending in `.ch`, plus `data.ms` if
    present. Names kept **with** extension (e.g. `FID1A.ch`, `data.ms`).
- `read_signal(data_dir, signal) -> (x_min, y)`
  - `.ch`: `data_dir.get_file(signal)` → `x = xlabels` (retention time, minutes),
    `y = data` (flattened).
  - `data.ms`: `data_dir.get_file('data.ms')` → `x = xlabels` (minutes),
    `y = np.sum(data, axis=1)` (TIC = sum over m/z axis).
- `read_notebook(data_dir, d_path) -> str`
  - Ported from `logic/json_exporter.py:scrape_metadata_from_d_directory`.
  - Prefer the relevant detector file's `metadata['notebook']`; fall back to
    `data_dir.metadata['notebook']`; then `getattr(data_dir, 'name', ...)`;
    finally `os.path.basename(d_path)`.

**Solvent delay note:** rainbow's MS `xlabels` already begin *after* the
acquisition solvent delay, so "MS start time" is simply `min(ms_x)`. No offset
math or sidecar files are involved.

## Alignment & Interpolation

Per `.D` folder, for the selected signals that exist in that folder:

1. Read each signal's native `(x_min, y)`.
2. **Union range**: `t_min = min(all signal starts)`, `t_max = max(all signal ends)`.
3. **Skip solvent delay**: if the option is checked AND `data.ms` is among the
   folder's selected signals, set `t_min = min(ms_x)` (clip the axis start to the
   MS start; leading FID/GC rows before MS start are dropped).
4. Build fixed grid: `grid = np.linspace(t_min, t_max, N)`, `N` configurable
   (default 10000).
5. For each signal, resample onto the grid with `np.interp(grid, x, y)`, then
   **manually mask** points outside the signal's native `[x.min(), x.max()]` range
   to `NaN` (since `np.interp` clamps to endpoints by default).
6. NaN values are written as empty cells.

Column layout: `Time (min)` first, then one column per selected-and-present
signal, header = the signal name (e.g. `FID1A.ch`, `data.ms`).

## GUI Layout (single `QDialog`)

1. **Folders group** — `QListWidget` + "Add .D Folder(s)…" (`QFileDialog`
   Directory mode; supports selecting multiple) + "Remove Selected". Adding or
   removing folders rescans and rebuilds the signal union.
2. **Signals group** — scroll area of checkboxes, one per **unique signal across
   all folders** (union set, e.g. `{FID1A.ch, TCD2B.ch, data.ms}`). All checked by
   default. A checked signal is exported for every folder that has it; folders
   lacking it simply omit that column.
3. **Options** —
   - `[x] Skip solvent delay (clip to MS start when MS present)`
   - `Points per signal` spinbox (default 10000).
4. **Output** — read-only `QLineEdit` + "Browse…" (`.xlsx` save dialog).
5. **Export** button + `QProgressBar` + log `QTextEdit`.
6. `QSettings("CalebCoatney", "ChromaKit")` remembers last output directory and
   the points value.

## Export Worker & Sheet Naming

`ExportWorker(QThread)` iterates the folders, emitting `progress(int)`,
`log(str)`, and `finished(summary)` signals consumed on the main thread.

- **Sheet name** from `read_notebook`. Sanitize Excel-invalid characters
  `[ ] : * ? / \`, truncate to 31 chars, and de-duplicate collisions with
  `_2`, `_3`, … Fall back to folder basename when notebook is missing.
- Each sheet written with `openpyxl`; bold header row; NaN → empty cell.
- Workbook saved to the chosen output path.

### Error handling

A folder that fails to read (corrupt `.D`, missing/unreadable signal) is logged
and **skipped**, not fatal — remaining folders still export. A summary
`QMessageBox` reports counts of exported vs skipped folders at the end.

## Entry Point

`main()` builds `QApplication(sys.argv)`, shows the dialog, `sys.exit(app.exec())`.
Runnable via `python util/dxlsx_export.py`.

## Out of Scope (YAGNI)

- Full MS m/z matrix export (TIC only).
- Cross-correlation (`align_tic_to_fid`) sub-second alignment.
- Per-folder independent signal selection (union model only).
- Any modification of existing ChromaKit code.
