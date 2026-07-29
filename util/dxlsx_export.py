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
