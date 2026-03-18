"""
Training data generator for the apex-heatmap U-Net.

Generates synthetic chromatograms, chunks them, and produces
(signal_window, heatmap_target) pairs suitable for training.

Usage:
    from training_data import generate_training_data

    X, Y, metadata = generate_training_data(num_chromatograms=500, seed=42)
    # X: (N, 256, 1) — normalised signal windows
    # Y: (N, 256, 1) — gaussian-blob apex heatmaps
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from synthetic_chromatogram import SyntheticChromatogramGenerator
from chunker import chunk_chromatogram


WINDOW_LENGTH = 256


def _random_smooth(signal, rng):
    """
    Apply a random smoothing filter to simulate preprocessing artifacts.

    Probabilities: none (30%), SavGol (30%), median (20%), Whittaker (20%).
    Ground-truth apex positions are unchanged — only the signal shape is
    perturbed, teaching the model invariance to smoothing.
    """
    roll = rng.random()

    if roll < 0.30:
        # No smoothing
        return signal

    if roll < 0.60:
        # Savitzky-Golay: random odd window [11, 71], polyorder [2, 4]
        window = int(rng.integers(6, 36)) * 2 + 1  # odd in [11, 71]
        polyorder = int(rng.integers(2, 5))         # 2, 3, or 4
        if window > len(signal):
            window = max(5, len(signal) // 2 * 2 + 1)
        if polyorder >= window:
            polyorder = window - 1
        return savgol_filter(signal, window, polyorder)

    if roll < 0.80:
        # Median filter: random odd kernel [5, 21]
        kernel = int(rng.integers(3, 11)) * 2 + 1  # odd in [5, 21]
        if kernel > len(signal):
            kernel = max(3, len(signal) // 2 * 2 + 1)
        return medfilt(signal, kernel_size=kernel)

    # Whittaker-like smoothing via pybaselines
    try:
        from pybaselines.whittaker import asls
        lam = 10 ** rng.uniform(1, 4)
        smoothed, _ = asls(signal, lam=lam)
        return smoothed
    except ImportError:
        # Fallback: SavGol if pybaselines not available
        window = int(rng.integers(6, 36)) * 2 + 1
        if window > len(signal):
            window = max(5, len(signal) // 2 * 2 + 1)
        return savgol_filter(signal, window, min(3, window - 1))


def _interpolate_chunk(time, signal, chunk, window_length=WINDOW_LENGTH):
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


def _make_heatmap(t_interp, true_peaks, chunk, heatmap_sigma=2.0,
                  window_length=WINDOW_LENGTH):
    """
    Build a gaussian-blob heatmap target for a chunk.

    Maps ground-truth apex RTs into the interpolated window's coordinate
    space and places a gaussian blob at each.

    Parameters
    ----------
    t_interp : 1-D array (window_length,)
        Interpolated time grid for this chunk.
    true_peaks : list of dicts
        Ground truth peaks from the synthetic generator.
    chunk : Chunk
        The chunk object (used for time bounds).
    heatmap_sigma : float
        Width of the gaussian blob in pixels.
    window_length : int
        Number of points in the window.

    Returns
    -------
    heatmap : 1-D array (window_length,)
    n_apexes : int — number of apexes placed
    apex_indices : list of int — pixel indices of placed apexes
    """
    t_start = t_interp[0]
    t_end = t_interp[-1]
    x_grid = np.arange(window_length)
    heatmap = np.zeros(window_length)
    apex_indices = []

    for peak in true_peaks:
        rt = peak['rt']
        # check if this peak's apex falls within the chunk's time range
        if rt < t_start or rt > t_end:
            continue

        # map RT to pixel index in the interpolated window
        pixel_idx = (rt - t_start) / (t_end - t_start) * (window_length - 1)
        pixel_idx = int(round(pixel_idx))
        pixel_idx = max(0, min(window_length - 1, pixel_idx))

        blob = np.exp(-0.5 * ((x_grid - pixel_idx) / heatmap_sigma) ** 2)
        heatmap = np.maximum(heatmap, blob)
        apex_indices.append(pixel_idx)

    return heatmap, len(apex_indices), apex_indices


def _make_targets(t_interp, true_peaks, chunk, heatmap_sigma=2.0,
                  window_length=WINDOW_LENGTH, n_channels=1):
    """
    Build multi-channel targets for a chunk.

    Channel 0: apex heatmap (gaussian blobs, peak=1.0) — same as _make_heatmap.
    Channel 1: sigma map — gaussian blobs at apex positions, peak = sigma_true / chunk_width.
    Channel 2: tau map — gaussian blobs at apex positions, peak = tau_true / chunk_width.

    Parameters
    ----------
    t_interp : 1-D array (window_length,)
    true_peaks : list of dicts
        Must have keys: 'rt', 'sigma', 'tau'.
    chunk : Chunk
    heatmap_sigma : float
    window_length : int
    n_channels : int
        1 → returns (heatmap, n_apexes, apex_indices) — backward compatible.
        3 → returns (targets_3ch, n_apexes, apex_indices) where targets_3ch
            is (3, window_length).

    Returns
    -------
    target : 1-D array (window_length,) if n_channels=1,
             or 2-D array (3, window_length) if n_channels=3
    n_apexes : int
    apex_indices : list of int
    """
    if n_channels == 1:
        return _make_heatmap(t_interp, true_peaks, chunk, heatmap_sigma,
                             window_length)

    t_start = t_interp[0]
    t_end = t_interp[-1]
    chunk_width = t_end - t_start
    if chunk_width < 1e-12:
        chunk_width = 1.0

    x_grid = np.arange(window_length)
    apex_heatmap = np.zeros(window_length)
    sigma_map = np.zeros(window_length)
    tau_map = np.zeros(window_length)
    apex_indices = []

    for peak in true_peaks:
        rt = peak['rt']
        if rt < t_start or rt > t_end:
            continue

        pixel_idx = (rt - t_start) / (t_end - t_start) * (window_length - 1)
        pixel_idx = int(round(pixel_idx))
        pixel_idx = max(0, min(window_length - 1, pixel_idx))

        blob = np.exp(-0.5 * ((x_grid - pixel_idx) / heatmap_sigma) ** 2)

        # Channel 0: apex heatmap (peak=1.0)
        apex_heatmap = np.maximum(apex_heatmap, blob)

        # Channel 1: sigma map — blob scaled by normalised sigma
        sigma_norm = peak['sigma'] / chunk_width
        sigma_map = np.maximum(sigma_map, blob * sigma_norm)

        # Channel 2: tau map — blob scaled by normalised tau
        tau_norm = peak['tau'] / chunk_width
        tau_map = np.maximum(tau_map, blob * tau_norm)

        apex_indices.append(pixel_idx)

    targets = np.stack([apex_heatmap, sigma_map, tau_map], axis=0)  # (3, W)
    return targets, len(apex_indices), apex_indices


def generate_training_data(num_chromatograms=500, seed=None,
                           heatmap_sigma=2.0,
                           window_length=WINDOW_LENGTH,
                           chromatogram_kwargs=None,
                           chunker_kwargs=None,
                           smoothing_augmentation=True,
                           n_channels=1):
    """
    Generate training pairs from synthetic chromatograms.

    Parameters
    ----------
    num_chromatograms : int
        Number of synthetic chromatograms to generate and chunk.
    seed : int or None
        Random seed for reproducibility.
    heatmap_sigma : float
        Gaussian blob width in pixels for the heatmap target.
    window_length : int
        Fixed window size (must match U-Net input).
    chromatogram_kwargs : dict or None
        Override kwargs for SyntheticChromatogramGenerator.generate().
    chunker_kwargs : dict or None
        Override kwargs for chunk_chromatogram().
    smoothing_augmentation : bool
        If True, randomly apply smoothing filters to the corrected signal
        before chunking/normalization. Teaches the model invariance to
        preprocessing artifacts. Default True.
    n_channels : int
        1 → Y is (N, window_length, 1) apex heatmap only (default).
        3 → Y is (N, window_length, 3) with channels [apex, sigma, tau].

    Returns
    -------
    X : ndarray (N, window_length, 1) — normalised signal windows
    Y : ndarray (N, window_length, 1) or (N, window_length, 3) — targets
    metadata : list of dicts — per-sample metadata for debugging
        Keys: chromatogram_idx, chunk_idx, n_apexes, apex_indices,
              apex_rts, t_start, t_end
    """
    gen = SyntheticChromatogramGenerator(seed=seed)

    chrom_kw = {
        'num_peaks': (5, 20),
        'merge_probability': 0.5,
        'snr': None,  # random
        'baseline_type': 'random',
    }
    if chromatogram_kwargs:
        chrom_kw.update(chromatogram_kwargs)

    chunk_kw = {}
    if chunker_kwargs:
        chunk_kw.update(chunker_kwargs)

    all_X = []
    all_Y = []
    all_meta = []

    for ci in range(num_chromatograms):
        result = gen.generate(**chrom_kw)
        time = result['x']
        signal = result['corrected_y']
        true_peaks = result['true_peaks']

        # Apply random smoothing augmentation BEFORE chunking
        if smoothing_augmentation:
            signal = _random_smooth(signal, gen.rng)

        chunks = chunk_chromatogram(time, signal, **chunk_kw)

        for chi, chunk in enumerate(chunks):
            t_interp, s_interp, s_norm, scale, offset = _interpolate_chunk(
                time, signal, chunk, window_length
            )

            target, n_apexes, apex_indices = _make_targets(
                t_interp, true_peaks, chunk, heatmap_sigma, window_length,
                n_channels=n_channels,
            )

            # skip chunks with no ground-truth apexes (noise-only chunks)
            if n_apexes == 0:
                continue

            all_X.append(s_norm)
            all_Y.append(target)
            all_meta.append({
                'chromatogram_idx': ci,
                'chunk_idx': chi,
                'n_apexes': n_apexes,
                'apex_indices': apex_indices,
                'apex_rts': [p['rt'] for p in true_peaks
                             if t_interp[0] <= p['rt'] <= t_interp[-1]],
                't_start': t_interp[0],
                't_end': t_interp[-1],
            })

    X = np.array(all_X)[:, :, np.newaxis]  # (N, window_length, 1)

    if n_channels == 1:
        Y = np.array(all_Y)[:, :, np.newaxis]  # (N, window_length, 1)
    else:
        # all_Y items are (n_channels, window_length) → stack to (N, n_channels, W)
        # then transpose to (N, W, n_channels) to match X convention
        Y = np.array(all_Y).transpose(0, 2, 1)  # (N, window_length, n_channels)

    return X, Y, all_meta


def plot_training_samples(X, Y, metadata, n=6, engine='plotly'):
    """Visualize a few training pairs for sanity checking."""
    indices = np.linspace(0, len(X) - 1, n, dtype=int)

    if engine == 'plotly':
        return _plot_samples_plotly(X, Y, metadata, indices)
    return _plot_samples_matplotlib(X, Y, metadata, indices)


def _plot_samples_plotly(X, Y, metadata, indices):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(indices)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                        vertical_spacing=0.04)

    for row, idx in enumerate(indices, 1):
        sig = X[idx, :, 0]
        hmap = Y[idx, :, 0]
        meta = metadata[idx]
        x = np.arange(len(sig))

        fig.add_trace(go.Scattergl(
            x=x, y=sig, mode='lines', name='Signal',
            line=dict(color='#1f77b4', width=1),
            showlegend=(row == 1), hoverinfo='skip',
        ), row=row, col=1)

        fig.add_trace(go.Scattergl(
            x=x, y=hmap, mode='lines', name='Heatmap',
            line=dict(color='red', width=1.5, dash='dash'),
            showlegend=(row == 1), hoverinfo='skip',
        ), row=row, col=1)

        for ai in meta['apex_indices']:
            fig.add_trace(go.Scattergl(
                x=[ai], y=[sig[ai]], mode='markers',
                marker=dict(symbol='diamond', size=7, color='red',
                            line=dict(color='black', width=0.5)),
                showlegend=False, hoverinfo='skip',
            ), row=row, col=1)

        fig.update_yaxes(
            title_text=f'{meta["n_apexes"]} apex{"es" if meta["n_apexes"] != 1 else ""}',
            row=row, col=1
        )

    fig.update_layout(
        title=f'Training Samples ({len(X)} total)',
        height=200 * n,
        template='plotly_white',
        hovermode='closest',
    )
    return fig


def _plot_samples_matplotlib(X, Y, metadata, indices):
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        sig = X[idx, :, 0]
        hmap = Y[idx, :, 0]
        meta = metadata[idx]

        ax.plot(sig, color='#1f77b4', linewidth=0.8, label='Signal')
        ax.plot(hmap, color='red', linewidth=1.5, linestyle='--',
                label='Heatmap', alpha=0.8)

        for ai in meta['apex_indices']:
            ax.plot(ai, sig[ai], marker='D', color='red', markersize=5,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)

        ax.set_ylabel(f'{meta["n_apexes"]} apex{"es" if meta["n_apexes"] != 1 else ""}')
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Pixel Index')
    axes[0].set_title(f'Training Samples ({len(X)} total)')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    X, Y, meta = generate_training_data(num_chromatograms=50, seed=42)
    print(f'Generated {len(X)} training samples')
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')

    apex_counts = [m['n_apexes'] for m in meta]
    from collections import Counter
    print(f'Apex distribution: {dict(sorted(Counter(apex_counts).items()))}')

    fig = plot_training_samples(X, Y, meta, n=8)
    fig.show()
