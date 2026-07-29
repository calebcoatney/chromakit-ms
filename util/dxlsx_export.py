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
    return os.path.splitext(os.path.basename(os.path.normpath(d_path)))[0]


def build_time_grid(sig_x, skip_solvent_delay, has_ms, ms_x, n):
    """Build a common time grid (minutes) over the union range of all signals.

    sig_x: dict signal_name -> native x array.
    If skip_solvent_delay and has_ms, clip the start to min(ms_x).
    """
    starts = [np.min(x) for x in sig_x.values() if len(x)]
    ends = [np.max(x) for x in sig_x.values() if len(x)]
    if not starts:
        raise ValueError("build_time_grid: no non-empty signals provided")
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


_INVALID_SHEET_CHARS = r'[]:*?/\\'


def safe_sheet_name(raw, used_names):
    """Sanitize to a valid, unique Excel sheet name (<=31 chars, no invalid
    chars, no leading/trailing spaces). Mutates used_names in place, adding the
    chosen name, and returns it."""
    name = "".join(c for c in str(raw) if c not in _INVALID_SHEET_CHARS).strip()
    name = name[:31].strip()
    if not name:
        name = "Sheet"
    base = name
    counter = 2
    while name in used_names:
        suffix = "_" + str(counter)
        name = base[:31 - len(suffix)].strip() + suffix
        counter += 1
    used_names.add(name)
    return name


def build_sheet_rows(data_dir, selected, skip_solvent_delay, n):
    """Return (header, rows) for one .D folder.

    header: ["Time (min)", <signal>, ...] for selected signals present here.
    rows: list of [time, val, ...] with NaN converted to None.
    """
    present = [s for s in selected if s in list_signals(data_dir)]
    if not present:
        return ["Time (min)"], []
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
