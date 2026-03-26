"""EIC peak extractor for ADAP-GC spectral deconvolution.

Extracts per-m/z ion chromatogram peaks from a rainbow data.ms object
within a specified RT window, returning EICPeak objects for deconvolve().
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths

from logic.spectral_deconvolution import EICPeak


def extract_eic_peaks(
    ms,
    t_start: float,
    t_end: float,
    min_intensity: float = 200.0,
    min_prominence: float = 0.0,
) -> list[EICPeak]:
    """Extract EIC peaks from a rainbow data.ms DataFile within [t_start, t_end].

    Args:
        ms: Open rainbow DataFile (.xlabels = RT axis, .data = [scans x mz] array).
        t_start: Window start in minutes (inclusive).
        t_end: Window end in minutes (inclusive).
        min_intensity: Reject individual peaks whose apex is below this value.
            Also used as a fast pre-filter: m/z traces whose max is below this
            are skipped entirely.
        min_prominence: Reject peaks whose prominence (height above the higher
            of their two surrounding valleys) is below this value.  Set to 0
            to disable (default).  Recommended: 1000-5000 for typical GC-MS data.

    Returns:
        List of EICPeak objects. rt_array and intensity_array span the full
        window so ADAP-GC has full chromatographic context. m/z values are
        1-based integers stored as float (column j -> mz = float(j+1)).
    """
    xlabels = np.asarray(ms.xlabels, dtype=float)
    data = np.asarray(ms.data, dtype=float)

    # Slice to window
    mask = (xlabels >= t_start) & (xlabels <= t_end)
    if not np.any(mask):
        return []

    rt_window = xlabels[mask]
    ms_slice = data[mask, :]   # shape: [n_window_scans, n_mz]
    n_window = len(rt_window)

    eic_peaks: list[EICPeak] = []

    # Build find_peaks kwargs once
    fp_kwargs: dict = {'height': min_intensity}
    if min_prominence > 0:
        fp_kwargs['prominence'] = min_prominence

    for j in range(ms_slice.shape[1]):
        trace = ms_slice[:, j]

        # Fast pre-filter: skip traces that can't possibly have qualifying peaks
        if trace.max() < min_intensity:
            continue

        apex_indices, properties = find_peaks(trace, **fp_kwargs)
        if len(apex_indices) == 0:
            continue

        # Get half-max widths for each apex
        try:
            widths_out = peak_widths(trace, apex_indices, rel_height=0.5)
            left_ips = widths_out[2]
            right_ips = widths_out[3]
        except Exception:
            continue

        mz = float(j + 1)

        for k, apex_idx in enumerate(apex_indices):
            left_idx = int(np.floor(left_ips[k]))
            right_idx = int(np.ceil(right_ips[k]))
            left_idx = max(0, min(left_idx, n_window - 1))
            right_idx = max(0, min(right_idx, n_window - 1))
            apex_idx_int = int(apex_idx)

            eic_peaks.append(EICPeak(
                rt_apex=float(rt_window[apex_idx_int]),
                mz=mz,
                rt_array=rt_window.copy(),
                intensity_array=trace.copy(),
                left_boundary_idx=left_idx,
                right_boundary_idx=right_idx,
                apex_idx=apex_idx_int,
            ))

    return eic_peaks
