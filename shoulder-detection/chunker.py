"""
Chromatogram chunking algorithm for the deconvolution pipeline.

Takes a baseline-corrected chromatogram and partitions it into windows,
each containing one peak or one cluster of overlapping peaks, with
baseline-level signal on both sides.

Usage:
    from chunker import chunk_chromatogram, plot_chunks

    chunks = chunk_chromatogram(time, corrected_signal)
    plot_chunks(time, corrected_signal, chunks)
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.signal import find_peaks, peak_widths


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
        considered separate clusters.  None = 1.5x median peak width.
    padding_fraction : float
        Extra padding on each side as a fraction of cluster width.
    max_chunk_width : float or None
        Maximum chunk width in time units.  Padding is reduced to fit,
        but the cluster itself is never split.  None = no limit.
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
    # Estimate noise from the quietest portion of the signal.  Sort |signal|
    # and take the std of the bottom 25% — this excludes peaks and tails,
    # giving a clean noise estimate even in dense chromatograms.
    if noise_threshold is None:
        sorted_abs = np.sort(np.abs(signal))
        bottom_quarter = sorted_abs[:max(1, n // 4)]
        noise_threshold = max(float(np.std(bottom_quarter)) * 3, 1e-10)

    # --- Step 1: Find peaks ---
    # Minimum peak width: real peaks are at least ~0.02 min wide
    min_width_pts = max(3, int(0.02 / dt))
    fp_kwargs = {'prominence': noise_threshold * 5, 'distance': 10, 'width': min_width_pts}
    if find_peaks_kwargs:
        fp_kwargs.update(find_peaks_kwargs)

    peak_indices, _ = find_peaks(signal, **fp_kwargs)
    if len(peak_indices) == 0:
        return []

    # --- Step 2: Get base positions ---
    # rel_height=0.9 gives bases at 90% depth — tight enough to avoid
    # EMG tails merging distant peaks, while the expand step (step 4)
    # still walks outward to find true baseline.
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
    # The bases from peak_widths(rel_height=0.9) already mark where the
    # peak signal drops to ~10% of peak height.  Padding adds baseline
    # margin on both sides for the CNN.
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
    # If padding causes two adjacent chunks to overlap, clamp both at the
    # argmin of the signal between them (the valley floor).
    resolved = []
    for i, (si, ei, cl_li, cl_ri, pks) in enumerate(chunks):
        if i > 0:
            prev_si, prev_ei, prev_cl_li, prev_cl_ri, prev_pks = resolved[-1]
            if prev_ei >= si:
                # find the valley between the two clusters' inner edges
                valley_left = prev_cl_ri
                valley_right = cl_li
                if valley_left < valley_right:
                    valley_idx = valley_left + int(np.argmin(signal[valley_left:valley_right + 1]))
                else:
                    valley_idx = (prev_ei + si) // 2
                # clamp previous chunk's end and this chunk's start
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


# --------------------------------------------------------------------------- #
#  Visualization
# --------------------------------------------------------------------------- #

def plot_chunks(time, signal, chunks, *, engine='plotly'):
    """Visualize chunking results."""
    if engine == 'plotly':
        return _plot_chunks_plotly(time, signal, chunks)
    return _plot_chunks_matplotlib(time, signal, chunks)


def _plot_chunks_plotly(time, signal, chunks):
    import plotly.graph_objects as go

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time, y=signal, mode='lines', name='Signal',
        line=dict(color='#333333', width=0.8), hoverinfo='skip',
    ))

    for i, chunk in enumerate(chunks):
        color = colors[i % len(colors)]
        si, ei = chunk.start_index, chunk.end_index + 1
        t_c = time[si:ei]
        s_c = signal[si:ei]

        # shaded region
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_c, t_c[::-1]]),
            y=np.concatenate([s_c, np.zeros(len(s_c))]),
            fill='toself', fillcolor=color, opacity=0.15,
            line=dict(width=0), hoverinfo='skip',
            name=f'Chunk {i} ({chunk.n_peaks} peak{"s" if chunk.n_peaks != 1 else ""})',
        ))

        # signal overlay
        fig.add_trace(go.Scattergl(
            x=t_c, y=s_c, mode='lines',
            line=dict(color=color, width=1.5),
            showlegend=False, hoverinfo='skip',
        ))

        # apex markers
        for pi in chunk.peak_indices:
            fig.add_trace(go.Scattergl(
                x=[time[pi]], y=[signal[pi]], mode='markers',
                marker=dict(symbol='diamond', size=8, color=color,
                            line=dict(color='black', width=0.5)),
                showlegend=False,
                text=f'RT={time[pi]:.2f}', hoverinfo='text',
            ))

        # boundary lines
        for edge in [chunk.start_time, chunk.end_time]:
            fig.add_vline(x=edge, line_dash='dot', line_color=color, opacity=0.4)

    fig.update_layout(
        title=f'Chromatogram Chunking — {len(chunks)} chunks',
        xaxis_title='Retention Time (min)',
        yaxis_title='Signal',
        template='plotly_white', height=450,
        legend=dict(font=dict(size=10)),
        hovermode='closest',
    )
    return fig


def _plot_chunks_matplotlib(time, signal, chunks):
    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, signal, color='#333333', linewidth=0.8, label='Signal')

    for i, chunk in enumerate(chunks):
        color = cmap(i % 10)
        si, ei = chunk.start_index, chunk.end_index + 1
        t_c, s_c = time[si:ei], signal[si:ei]
        ax.fill_between(t_c, 0, s_c, color=color, alpha=0.15)
        ax.plot(t_c, s_c, color=color, linewidth=1.5)
        for pi in chunk.peak_indices:
            ax.plot(time[pi], signal[pi], marker='D', color=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.axvline(chunk.start_time, color=color, linestyle=':', alpha=0.4)
        ax.axvline(chunk.end_time, color=color, linestyle=':', alpha=0.4)

    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Signal')
    ax.set_title(f'Chromatogram Chunking — {len(chunks)} chunks')
    ax.legend(fontsize=8, loc='upper right')
    return ax


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from synthetic_chromatogram import SyntheticChromatogramGenerator

    gen = SyntheticChromatogramGenerator(seed=42)
    result = gen.generate(num_peaks=12, merge_probability=0.4, snr=200,
                          baseline_type='flat')

    chunks = chunk_chromatogram(result['x'], result['corrected_y'])

    print(f'Found {len(chunks)} chunks:')
    for i, c in enumerate(chunks):
        print(f'  Chunk {i}: RT {c.start_time:.2f}-{c.end_time:.2f} '
              f'({c.width_time:.2f} min, {c.width_points} pts), '
              f'{c.n_peaks} peak(s) at RT {[f"{rt:.2f}" for rt in c.peak_rts]}')

    fig = plot_chunks(result['x'], result['corrected_y'], chunks)
    fig.show()
