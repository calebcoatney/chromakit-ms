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
