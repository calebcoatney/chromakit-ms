"""
Chromatogram chunking for the deconvolution pipeline.

Partitions a baseline-corrected chromatogram into windows, each containing
one peak or one cluster of overlapping peaks, with baseline-level signal
on both sides.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.signal import find_peaks, peak_widths


WINDOW_LENGTH = 256


@dataclass
class Chunk:
    """A single window extracted from a chromatogram."""
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    peak_indices: list = field(default_factory=list)
    peak_rts: list = field(default_factory=list)
    n_peaks: int = 0

    @property
    def width_points(self):
        return self.end_index - self.start_index

    @property
    def width_time(self):
        return self.end_time - self.start_time


def chunk_chromatogram(time, signal, *,
                       gap_tolerance=None,
                       padding_fraction=0.5,
                       max_chunk_width=None,
                       noise_threshold=None,
                       find_peaks_kwargs=None):
    """
    Partition a baseline-corrected chromatogram into peak-centred windows.

    Parameters
    ----------
    time : 1-D array
        Retention time axis.
    signal : 1-D array
        Baseline-corrected signal (baseline already subtracted).
    gap_tolerance : float or None
        Maximum time gap between adjacent peak bases before they are
        considered separate clusters.  None = 0.5x median peak width.
    padding_fraction : float
        Extra padding on each side as a fraction of cluster width.
    max_chunk_width : float or None
        Maximum chunk width in time units.  None = no limit.
    noise_threshold : float or None
        Signal level considered "baseline" for edge expansion.
        None = auto-estimate from signal.
    find_peaks_kwargs : dict or None
        Extra kwargs for scipy.signal.find_peaks.

    Returns
    -------
    list of Chunk
    """
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    dt = float(np.median(np.diff(time)))

    # --- Noise floor estimation ---
    if noise_threshold is None:
        sorted_abs = np.sort(np.abs(signal))
        bottom_quarter = sorted_abs[:max(1, n // 4)]
        noise_threshold = max(float(np.std(bottom_quarter)) * 3, 1e-10)

    # --- Step 1: Find peaks ---
    min_width_pts = max(3, int(0.02 / dt))
    fp_kwargs = {'prominence': noise_threshold * 5, 'distance': 10, 'width': min_width_pts}
    if find_peaks_kwargs:
        fp_kwargs.update(find_peaks_kwargs)

    peak_indices, _ = find_peaks(signal, **fp_kwargs)
    if len(peak_indices) == 0:
        return []

    # --- Step 2: Get base positions ---
    widths, _, left_ips, right_ips = peak_widths(
        signal, peak_indices, rel_height=0.9
    )
    left_bases = np.clip(np.floor(left_ips).astype(int), 0, n - 1)
    right_bases = np.clip(np.ceil(right_ips).astype(int), 0, n - 1)

    median_width_time = float(np.median(widths)) * dt
    if gap_tolerance is None:
        gap_tolerance = 0.5 * median_width_time

    # --- Step 3: Merge overlapping bases into clusters ---
    intervals = sorted(
        [(time[left_bases[i]], time[right_bases[i]], peak_indices[i])
         for i in range(len(peak_indices))],
        key=lambda x: x[0]
    )

    clusters = []
    cur_left, cur_right, cur_peaks = intervals[0][0], intervals[0][1], [intervals[0][2]]
    for lt, rt, pi in intervals[1:]:
        if lt <= cur_right + gap_tolerance:
            cur_right = max(cur_right, rt)
            cur_peaks.append(pi)
        else:
            clusters.append((cur_left, cur_right, cur_peaks))
            cur_left, cur_right, cur_peaks = lt, rt, [pi]
    clusters.append((cur_left, cur_right, cur_peaks))

    # --- Step 4: Pad around cluster base positions ---
    chunks = []
    for cl_left_time, cl_right_time, cl_peak_indices in clusters:
        cl_left_idx = max(0, int(np.searchsorted(time, cl_left_time, side='left')))
        cl_right_idx = min(n - 1, int(np.searchsorted(time, cl_right_time, side='right')))

        start_idx = cl_left_idx
        end_idx = cl_right_idx

        # padding
        cluster_width = end_idx - start_idx
        pad_pts = max(1, int(cluster_width * padding_fraction))
        start_idx = max(0, start_idx - pad_pts)
        end_idx = min(n - 1, end_idx + pad_pts)

        # enforce max chunk width by trimming padding only
        if max_chunk_width is not None:
            max_pts = int(max_chunk_width / dt)
            if end_idx - start_idx > max_pts:
                excess = (end_idx - start_idx) - max_pts
                trim_each = excess // 2
                start_idx = min(start_idx + trim_each, cl_left_idx)
                end_idx = max(end_idx - trim_each, cl_right_idx)

        chunks.append((start_idx, end_idx, cl_left_idx, cl_right_idx, cl_peak_indices))

    # --- Step 5: Resolve overlapping chunk boundaries ---
    resolved = []
    for i, (si, ei, cl_li, cl_ri, pks) in enumerate(chunks):
        if i > 0:
            prev_si, prev_ei, prev_cl_li, prev_cl_ri, prev_pks = resolved[-1]
            if prev_ei >= si:
                valley_left = prev_cl_ri
                valley_right = cl_li
                if valley_left < valley_right:
                    valley_idx = valley_left + int(np.argmin(signal[valley_left:valley_right + 1]))
                else:
                    valley_idx = (prev_ei + si) // 2
                resolved[-1] = (prev_si, valley_idx, prev_cl_li, prev_cl_ri, prev_pks)
                si = valley_idx
        resolved.append((si, ei, cl_li, cl_ri, pks))

    # --- Build final Chunk objects ---
    final_chunks = []
    for si, ei, _, _, cl_peak_indices in resolved:
        final_chunks.append(Chunk(
            start_index=si,
            end_index=ei,
            start_time=time[si],
            end_time=time[ei],
            peak_indices=cl_peak_indices,
            peak_rts=[time[pi] for pi in cl_peak_indices],
            n_peaks=len(cl_peak_indices),
        ))

    return final_chunks


def interpolate_chunk(time, signal, chunk, window_length=WINDOW_LENGTH):
    """
    Extract a chunk from the chromatogram and interpolate to fixed length.

    Returns
    -------
    t_interp : 1-D array (window_length,)
    s_interp : 1-D array (window_length,)
    s_norm : 1-D array (window_length,) — min-max normalised to [0, 1]
    scale : float — (max - min) of the interpolated signal
    offset : float — min of the interpolated signal
    """
    si, ei = chunk.start_index, chunk.end_index + 1
    t_chunk = time[si:ei]
    s_chunk = signal[si:ei]

    t_interp = np.linspace(t_chunk[0], t_chunk[-1], window_length)
    s_interp = np.interp(t_interp, t_chunk, s_chunk)

    s_min = s_interp.min()
    s_max = s_interp.max()
    scale = s_max - s_min

    if scale > 1e-12:
        s_norm = (s_interp - s_min) / scale
    else:
        s_norm = np.zeros(window_length)
        scale = 1.0

    return t_interp, s_interp, s_norm, scale, s_min
