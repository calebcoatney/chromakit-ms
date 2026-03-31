"""Parser for Mettler Toledo ReactIR CSV files."""
from __future__ import annotations
import os
from logic.c_folder import CFolder


def parse_reactir_csv(csv_path: str, sample_timestamp: str | None = None) -> CFolder:
    """Wrap a Mettler Toledo ReactIR CSV into a .C folder.

    ReactIR format: two-column headerless CSV.
      Column 0: wavenumber (cm⁻¹)
      Column 1: absorbance (A.U.)

    The source file is moved into the .C folder's data/ directory.
    Returns the created CFolder.

    sample_timestamp: ISO-format datetime string to store as the sample
        collection time.  When None the filename is searched for a timestamp
        pattern (YYYY-MM-DD_HH-MM-SS); if nothing is found the manifest falls
        back to the current wall-clock time.
    """
    import re
    from datetime import datetime as _dt

    _PATTERNS = [
        (r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", "%Y-%m-%d_%H-%M-%S"),
        (r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}", "%Y-%m-%d_%H.%M.%S"),
        (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    ]

    if sample_timestamp is None:
        filename = os.path.basename(csv_path)
        for pattern, fmt in _PATTERNS:
            m = re.search(pattern, filename)
            if m:
                try:
                    sample_timestamp = _dt.strptime(m.group(), fmt).isoformat()
                    break
                except ValueError:
                    continue

    kwargs: dict = {
        "instrument": "Mettler Toledo ReactIR",
        "csv_columns": {"x_column": 0, "y_column": 1, "has_header": False},
    }
    if sample_timestamp is not None:
        kwargs["sample_timestamp"] = sample_timestamp

    return CFolder.create(csv_path, "ftir", **kwargs)
