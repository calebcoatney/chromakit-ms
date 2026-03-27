# Avantes UV-Vis Parser & Instrument-Specific CSV Parsers

**Date:** 2026-03-27
**Scope:** Add an Avantes UV-Vis parser that converts a multi-column Avantes CSV into per-spectrum `.C` folders. Simultaneously formalize the existing ReactIR FTIR CSV path as an instrument-specific parser, and simplify the generic CSV path in `BatchConvertDialog`.

---

## Background

ChromaKit's `.C` folder abstraction wraps a single signal (x/y pair + manifest metadata) into a portable container. The existing `BatchConvertDialog` supports converting Agilent `.D` folders and generic two-column CSVs. This spec adds explicit, instrument-named parsers for:

- **Mettler Toledo ReactIR** — drops one headerless two-column CSV per scan (wavenumber, absorbance). Already works with the generic CSV path; this formalizes it with a named function and instrument metadata.
- **Avantes UV-Vis** — appends a new column to a single CSV per acquisition. Requires splitting into one `.C` folder per spectrum column.

---

## New Logic-Layer Files

### `logic/loaders/reactir_parser.py`

Single function:

```python
def parse_reactir_csv(csv_path: str) -> CFolder
```

Calls `CFolder.create()` with ReactIR-specific settings baked in:
- `signal_type="ftir"`
- `instrument="Mettler Toledo ReactIR"`
- `csv_columns={"x_column": 0, "y_column": 1, "has_header": False}`

Returns the created `CFolder`. No user configuration required.

---

### `logic/loaders/avantes_parser.py`

Single function:

```python
def parse_avantes_uvvis(csv_path: str, output_dir: str = None) -> list[str]
```

`output_dir` defaults to the same directory as `csv_path`.

**File format assumed (Avantes UV-Vis export):**

| Row | Col 0 | Cols 1..N |
|-----|-------|-----------|
| 0 | `Integration Time [msec]` | value (same for all columns) |
| 1 | `Number of Averages` | value |
| 2 | `Wavelength [nm]` | `A.U.` labels |
| 3 | `Timestamp` | millisecond timestamps |
| 4 | `Date/Time` | `DD/MM/YYYY HH:MM:SS` strings |
| 5+ | wavelength (nm) | absorbance (A.U.) per spectrum |

**Steps:**
1. Read the raw CSV with pandas (no header).
2. Extract from header rows: `integration_time_ms` (row 0, col 1), `n_averages` (row 1, col 1), `Date/Time` strings (row 4, cols 1..N).
3. Rows 5+ = spectral data: col 0 = wavelength array, cols 1..N = per-spectrum intensity arrays.
4. For each spectrum column `i` (0-indexed):
   - Write a 2-column headerless CSV `{basename}_{i:03d}.csv` to `output_dir`.
   - Parse the `Date/Time` string to ISO-8601 (`sample_timestamp`).
   - Call `CFolder.create(temp_csv_path, "uvvis", instrument="Avantes", sample_timestamp=..., integration_time_ms=..., n_averages=...)`.
   - `CFolder.create()` moves the temp CSV into `data/`.
5. Return list of created `.C` paths.

---

## `BatchConvertDialog` Changes

### Buttons (top row)

| Button | Behavior |
|--------|----------|
| Add .D Folders… | Unchanged |
| Add ReactIR Files… | Multi-file picker (`.csv`); one row per file added to preview table |
| Add Avantes UV-Vis File… | Single-file picker (`.csv`); parses immediately, adds one row per spectrum column |
| Add CSV Files… | Multi-file picker (`.csv`); generic one-to-one path |

### CSV Settings Group

Shown only when generic CSV files are loaded. Simplified to:
- Signal type combo: FTIR / UV-Vis / GC / GC-MS
- "Has header row" checkbox
- X/Y column spinners **removed** — always col 0 (x) and col 1 (y)

### Preview Table Columns

`Source | Destination (.C) | Type | Timestamp`

- ReactIR rows: Type = `"ReactIR (FTIR)"`, Timestamp from filename pattern extraction (existing logic)
- Avantes rows: Type = `"Avantes UV-Vis"`, Timestamp from parsed `Date/Time` header row
- Generic CSV rows: Type = signal type from combo, Timestamp from filename pattern extraction

### Timestamp Extraction Group

Unchanged. Applies to `.D` and generic CSV rows only. Avantes rows always populate timestamp from the file header.

---

## Data Flow & Error Handling

### At "Add" time

- **ReactIR / generic CSV:** validate file exists, add row to table. No parsing.
- **Avantes:** read and parse the file immediately to determine column count and extract timestamps. On parse failure, show an error dialog and add no rows. On success, populate N rows with timestamps pre-filled.

### At "Convert" time

Iterate table rows, dispatch by type:

| Row type | Action |
|----------|--------|
| `.D` | `CFolder.create()` — unchanged |
| ReactIR | `parse_reactir_csv(path)` |
| Avantes | Write temp CSV + `CFolder.create()` using pre-parsed data stored at "Add" time |
| Generic CSV | `CFolder.create()` with manifest settings |

- `FileExistsError` → silently skip (consistent with existing behavior)
- All other exceptions → collected, shown in summary error dialog at end
- Progress bar advances **one step per `.C` folder created** (Avantes file with N columns = N steps)

---

## Manifest Fields Per Instrument

| Field | ReactIR | Avantes | Generic CSV |
|-------|---------|---------|-------------|
| `signal_type` | `"ftir"` | `"uvvis"` | user-selected |
| `instrument` | `"Mettler Toledo ReactIR"` | `"Avantes"` | *(omitted)* |
| `sample_timestamp` | from filename (if extracted) | from file header | from filename (if extracted) |
| `integration_time_ms` | — | ✓ | — |
| `n_averages` | — | ✓ | — |
| `csv_columns` | `{x:0, y:1, has_header:false}` | `{x:0, y:1, has_header:false}` | `{x:0, y:1, has_header: …}` |

---

## Files Touched

| File | Change |
|------|--------|
| `logic/loaders/reactir_parser.py` | **New** |
| `logic/loaders/avantes_parser.py` | **New** |
| `ui/dialogs/batch_convert_dialog.py` | Modified — new buttons, simplified CSV settings, dispatch logic |
