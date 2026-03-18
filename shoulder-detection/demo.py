"""
End-to-end demonstration of the deconvolution pipeline:

    synthetic chromatogram → chunking → U-Net inference → EMG deconvolution

Usage:
    from demo import run_pipeline_demo
    run_pipeline_demo(model, seed=42)
"""

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import exponnorm

from synthetic_chromatogram import SyntheticChromatogramGenerator
from chunker import chunk_chromatogram, Chunk
from training_data import _interpolate_chunk, WINDOW_LENGTH
from unet_model import GCHeatmapUNet


# --------------------------------------------------------------------------- #
#  EMG fitting
# --------------------------------------------------------------------------- #

def _single_emg(t, amp, mu, sigma, tau):
    sigma = max(sigma, 1e-6)
    K = tau / sigma
    return amp * exponnorm.pdf(t, K, loc=mu, scale=sigma)


def _multi_emg(t, *params):
    n = len(params) // 4
    y = np.zeros_like(t)
    for i in range(n):
        y += _single_emg(t, *params[i * 4:(i + 1) * 4])
    return y


def deconvolve_chunk(t_chunk, s_chunk, apex_rts):
    """
    Fit a multi-EMG model to a chunk signal, seeded by apex positions.

    Parameters
    ----------
    t_chunk : 1-D array — time axis for this chunk
    s_chunk : 1-D array — signal for this chunk (not normalised)
    apex_rts : list of float — retention times of predicted apexes

    Returns
    -------
    list of dicts with keys: retention_time, area, sigma, tau, amplitude
    popt : fitted parameter array (or None on failure)
    """
    n_peaks = len(apex_rts)
    if n_peaks == 0:
        return [], None

    window_width = t_chunk[-1] - t_chunk[0]
    sig_guess = window_width / (n_peaks * 4)
    tau_guess = sig_guess * 0.3
    sig_max = window_width / 2
    tau_max = window_width / 2

    # Sort apex RTs so we can compute neighbor-based mu bounds
    sorted_rts = sorted(apex_rts)

    p0, lo, hi = [], [], []
    for i, rt in enumerate(sorted_rts):
        idx = int(np.argmin(np.abs(t_chunk - rt)))
        amp_guess = max(s_chunk[idx], 1e-6) * sig_guess * 5.0

        # Constrain mu to stay near its predicted apex — halfway to neighbors
        if i == 0:
            mu_lo = t_chunk[0]
        else:
            mu_lo = (sorted_rts[i - 1] + rt) / 2
        if i == n_peaks - 1:
            mu_hi = t_chunk[-1]
        else:
            mu_hi = (rt + sorted_rts[i + 1]) / 2

        p0.extend([amp_guess, rt, sig_guess, tau_guess])
        lo.extend([0, mu_lo, 1e-4, 1e-4])
        hi.extend([np.inf, mu_hi, sig_max, tau_max])

    try:
        popt, _ = curve_fit(
            _multi_emg, t_chunk, s_chunk,
            p0=p0, bounds=(lo, hi), maxfev=10000
        )
    except RuntimeError:
        return [], None

    results = []
    for i in range(n_peaks):
        amp, mu, sigma, tau = popt[i * 4:(i + 1) * 4]
        results.append({
            'retention_time': mu,
            'area': amp,
            'sigma': sigma,
            'tau': tau,
            'amplitude': amp,
        })

    return results, popt


# --------------------------------------------------------------------------- #
#  U-Net inference on chunks
# --------------------------------------------------------------------------- #

def infer_chunks(model, time, signal, chunks, *,
                 device=None, heatmap_height=0.15, heatmap_distance=10,
                 min_prominence=0.0):
    """
    Run U-Net inference on each chunk and extract predicted apex positions.

    Parameters
    ----------
    model : GCHeatmapUNet
    time : 1-D array — full time axis
    signal : 1-D array — full baseline-corrected signal
    chunks : list of Chunk
    device : str or None
    heatmap_height : float
    heatmap_distance : int
    min_prominence : float
        Minimum prominence of each predicted apex in the min-max normalised
        signal (s_norm, range [0, 1]).  Each heatmap apex is first snapped to
        the nearest local maximum in s_norm (within ±heatmap_distance/2 pts),
        then scipy.signal.peak_prominences is used to compute its prominence.
        Apexes below this threshold are discarded before EMG fitting.
        Set to 0.0 (default) to disable filtering.

    Returns
    -------
    list of dicts, one per chunk:
        chunk : Chunk object
        t_interp : interpolated time grid (256 pts)
        s_interp : interpolated signal (256 pts)
        s_norm : normalised signal (256 pts)
        heatmap : predicted heatmap (256 pts)
        apex_pixels : list of int — pixel indices of detected apexes
        apex_rts : list of float — retention times of detected apexes
        scale : float
        offset : float
    """
    from scipy.signal import peak_prominences

    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    model = model.to(device)
    model.eval()

    results = []
    for chunk in chunks:
        t_interp, s_interp, s_norm, scale, offset = _interpolate_chunk(
            time, signal, chunk
        )

        # run inference
        x_t = torch.tensor(s_norm, dtype=torch.float32).reshape(1, 1, -1).to(device)
        with torch.no_grad():
            heatmap = model(x_t).squeeze().cpu().numpy()

        # extract apex positions from heatmap
        apex_pixels, _ = find_peaks(heatmap, height=heatmap_height,
                                     distance=heatmap_distance)

        # prominence filter on the normalised signal
        if min_prominence > 0 and len(apex_pixels) > 0:
            # Use the full heatmap_distance as the snap radius so that false-split
            # apexes (which are at minimum heatmap_distance apart by construction)
            # still fall within each other's windows and collapse to the same
            # signal maximum.  Legitimate separate peaks snap to distinct maxima.
            half_d = max(heatmap_distance, 3)
            snap_to_orig = {}  # snapped_pixel → first original apex_pixel
            for px in apex_pixels:
                lo = max(0, int(px) - half_d)
                hi = min(len(s_norm), int(px) + half_d + 1)
                snapped_px = lo + int(np.argmax(s_norm[lo:hi]))
                if snapped_px not in snap_to_orig:
                    snap_to_orig[snapped_px] = int(px)

            unique_snapped = list(snap_to_orig.keys())
            unique_orig    = list(snap_to_orig.values())

            proms, _, _ = peak_prominences(s_norm, unique_snapped)
            keep = np.array(proms) >= min_prominence
            apex_pixels = np.array(unique_orig)[keep]

        # map pixel indices back to retention times
        apex_rts = [t_interp[int(p)] for p in apex_pixels]

        results.append({
            'chunk': chunk,
            't_interp': t_interp,
            's_interp': s_interp,
            's_norm': s_norm,
            'heatmap': heatmap,
            'apex_pixels': list(apex_pixels),
            'apex_rts': apex_rts,
            'scale': scale,
            'offset': offset,
        })

    model = model.cpu()
    return results


# --------------------------------------------------------------------------- #
#  End-to-end pipeline demo
# --------------------------------------------------------------------------- #

def run_pipeline_demo(model, seed=42, engine='plotly', **gen_kwargs):
    """
    Run the full pipeline on a synthetic chromatogram and produce
    three plots:

    1. Raw chromatogram with true components and apexes
    2. Chunked chromatogram with CNN-predicted apexes
    3. Deconvolved chromatogram with fitted EMG curves

    Parameters
    ----------
    model : GCHeatmapUNet (trained)
    seed : int
    engine : str — 'plotly' or 'matplotlib'
    **gen_kwargs — passed to generate()

    Returns
    -------
    dict with all intermediate results
    """
    defaults = dict(num_peaks=10, merge_probability=0.4, snr=200,
                    baseline_type='flat', time_range=(0, 30))
    defaults.update(gen_kwargs)

    # --- Step 1: Generate chromatogram ---
    gen = SyntheticChromatogramGenerator(seed=seed)
    result = gen.generate(**defaults)
    time = result['x']
    signal = result['corrected_y']

    # --- Step 2: Chunk ---
    chunks = chunk_chromatogram(time, signal)

    # --- Step 3: U-Net inference ---
    infer_results = infer_chunks(model, time, signal, chunks)

    # --- Step 4: EMG deconvolution per chunk ---
    deconv_results = []
    for ir in infer_results:
        chunk = ir['chunk']
        si, ei = chunk.start_index, chunk.end_index + 1
        t_chunk = time[si:ei]
        s_chunk = signal[si:ei]

        emg_fits, popt = deconvolve_chunk(t_chunk, s_chunk, ir['apex_rts'])
        deconv_results.append({
            't_chunk': t_chunk,
            's_chunk': s_chunk,
            'emg_fits': emg_fits,
            'popt': popt,
            'apex_rts': ir['apex_rts'],
        })

    # --- Plots ---
    if engine == 'plotly':
        fig1 = _plot_step1_plotly(result)
        fig2 = _plot_step2_plotly(time, signal, chunks, infer_results)
        fig3 = _plot_step3_plotly(time, signal, chunks, infer_results,
                                   deconv_results, result)
    else:
        fig1 = _plot_step1_mpl(result)
        fig2 = _plot_step2_mpl(time, signal, chunks, infer_results)
        fig3 = _plot_step3_mpl(time, signal, chunks, infer_results,
                                deconv_results, result)

    return {
        'chromatogram': result,
        'chunks': chunks,
        'infer_results': infer_results,
        'deconv_results': deconv_results,
        'figures': (fig1, fig2, fig3),
    }


# --------------------------------------------------------------------------- #
#  Plot helpers
# --------------------------------------------------------------------------- #

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# --- Step 1: Raw chromatogram with true components ---

def _plot_step1_plotly(result):
    import plotly.graph_objects as go

    t = result['x']
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=t, y=result['corrected_y'], mode='lines',
        line=dict(color='#333333', width=1),
        name='Signal', hoverinfo='skip',
    ))

    clusters_seen = set()
    for pp in result['peak_params']:
        K = pp['tau'] / max(pp['sigma'], 1e-6)
        curve = pp['amp'] * exponnorm.pdf(
            t, K, loc=pp['mu'], scale=max(pp['sigma'], 1e-6))
        color = _COLORS[pp['cluster_id'] % len(_COLORS)]
        show = pp['cluster_id'] not in clusters_seen
        clusters_seen.add(pp['cluster_id'])
        fig.add_trace(go.Scattergl(
            x=t, y=curve, mode='lines',
            line=dict(color=color, width=0.8, dash='dot'),
            name=f'Cluster {pp["cluster_id"]}',
            legendgroup=f'c{pp["cluster_id"]}',
            showlegend=show, hoverinfo='skip', opacity=0.6,
        ))

    for p in result['true_peaks']:
        color = _COLORS[p['cluster_id'] % len(_COLORS)]
        edge = 'red' if p['is_merged'] else 'black'
        fig.add_trace(go.Scattergl(
            x=[p['rt']], y=[result['corrected_y'][p['apex_index']]],
            mode='markers',
            marker=dict(symbol='diamond', size=8, color=color,
                        line=dict(color=edge, width=1)),
            showlegend=False,
            text=f'RT={p["rt"]:.2f}<br>Area={p["area"]:.1f}',
            hoverinfo='text',
        ))

    fig.update_layout(
        title='Step 1: Synthetic Chromatogram — True Components',
        xaxis_title='Retention Time (min)', yaxis_title='Signal',
        template='plotly_white', height=400, hovermode='closest',
    )
    return fig


def _plot_step1_mpl(result):
    import matplotlib.pyplot as plt
    t = result['x']
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, result['corrected_y'], color='#333333', linewidth=0.8)
    cmap = plt.cm.get_cmap('tab10')
    for pp in result['peak_params']:
        K = pp['tau'] / max(pp['sigma'], 1e-6)
        curve = pp['amp'] * exponnorm.pdf(
            t, K, loc=pp['mu'], scale=max(pp['sigma'], 1e-6))
        ax.plot(t, curve, color=cmap(pp['cluster_id'] % 10),
                linewidth=0.7, linestyle=':', alpha=0.6)
    for p in result['true_peaks']:
        ec = 'red' if p['is_merged'] else 'black'
        ax.plot(p['rt'], result['corrected_y'][p['apex_index']],
                marker='D', color=cmap(p['cluster_id'] % 10), markersize=5,
                markeredgecolor=ec, markeredgewidth=0.5, zorder=5)
    ax.set_title('Step 1: Synthetic Chromatogram — True Components')
    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Signal')
    return fig


# --- Step 2: Chunks with CNN-predicted apexes ---

def _plot_step2_plotly(time, signal, chunks, infer_results):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=time, y=signal, mode='lines',
        line=dict(color='#333333', width=0.8),
        name='Signal', hoverinfo='skip',
    ))

    for ci, (chunk, ir) in enumerate(zip(chunks, infer_results)):
        color = _COLORS[ci % len(_COLORS)]
        si, ei = chunk.start_index, chunk.end_index + 1
        t_c, s_c = time[si:ei], signal[si:ei]

        # shaded chunk
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_c, t_c[::-1]]),
            y=np.concatenate([s_c, np.zeros(len(s_c))]),
            fill='toself', fillcolor=color, opacity=0.12,
            line=dict(width=0), hoverinfo='skip',
            name=f'Chunk {ci}', showlegend=True,
        ))

        fig.add_trace(go.Scattergl(
            x=t_c, y=s_c, mode='lines',
            line=dict(color=color, width=1.5),
            showlegend=False, hoverinfo='skip',
        ))

        # heatmap overlay (scaled to signal range for visibility)
        hmap = ir['heatmap']
        t_interp = ir['t_interp']
        s_max = s_c.max() if s_c.max() > 0 else 1
        fig.add_trace(go.Scattergl(
            x=t_interp, y=hmap * s_max * 0.3, mode='lines',
            line=dict(color=color, width=1, dash='dash'),
            showlegend=False, hoverinfo='skip', opacity=0.5,
        ))

        # predicted apex markers
        for rt in ir['apex_rts']:
            idx_in_signal = int(np.argmin(np.abs(time - rt)))
            fig.add_trace(go.Scattergl(
                x=[rt], y=[signal[idx_in_signal]], mode='markers',
                marker=dict(symbol='x', size=12, color=color,
                            line=dict(width=2)),
                showlegend=False,
                text=f'Predicted RT={rt:.2f}', hoverinfo='text',
            ))

    fig.update_layout(
        title='Step 2: Chunked + U-Net Predicted Apexes',
        xaxis_title='Retention Time (min)', yaxis_title='Signal',
        template='plotly_white', height=400, hovermode='closest',
    )
    return fig


def _plot_step2_mpl(time, signal, chunks, infer_results):
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, signal, color='#333333', linewidth=0.8)

    for ci, (chunk, ir) in enumerate(zip(chunks, infer_results)):
        color = cmap(ci % 10)
        si, ei = chunk.start_index, chunk.end_index + 1
        t_c, s_c = time[si:ei], signal[si:ei]
        ax.fill_between(t_c, 0, s_c, color=color, alpha=0.1)
        ax.plot(t_c, s_c, color=color, linewidth=1.5)
        for rt in ir['apex_rts']:
            idx = int(np.argmin(np.abs(time - rt)))
            ax.plot(rt, signal[idx], 'x', color=color, markersize=10,
                    markeredgewidth=2, zorder=5)

    ax.set_title('Step 2: Chunked + U-Net Predicted Apexes')
    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Signal')
    return fig


# --- Step 3: Deconvolved chromatogram with fitted EMG curves ---

def _plot_step3_plotly(time, signal, chunks, infer_results,
                        deconv_results, result=None):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=time, y=signal, mode='lines',
        line=dict(color='#333333', width=0.8),
        name='Signal', hoverinfo='skip',
    ))

    emg_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                  '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
                  '#fabebe', '#469990']
    color_idx = 0

    for ci, (chunk, ir, dr) in enumerate(zip(chunks, infer_results,
                                              deconv_results)):
        chunk_color = _COLORS[ci % len(_COLORS)]
        si, ei = chunk.start_index, chunk.end_index + 1
        t_c, s_c = time[si:ei], signal[si:ei]

        # light shaded chunk background
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_c, t_c[::-1]]),
            y=np.concatenate([s_c, np.zeros(len(s_c))]),
            fill='toself', fillcolor=chunk_color, opacity=0.08,
            line=dict(width=0), hoverinfo='skip',
            showlegend=False,
        ))

        if dr['popt'] is None or not dr['emg_fits']:
            continue

        # plot fitted sum
        fitted_sum = _multi_emg(t_c, *dr['popt'])
        fig.add_trace(go.Scattergl(
            x=t_c, y=fitted_sum, mode='lines',
            line=dict(color=chunk_color, width=2, dash='dash'),
            name=f'Chunk {ci} fit', hoverinfo='skip',
        ))

        # plot individual EMG components
        n_peaks = len(dr['emg_fits'])
        for i, emg in enumerate(dr['emg_fits']):
            ec = emg_colors[color_idx % len(emg_colors)]
            color_idx += 1
            curve = _single_emg(t_c, emg['amplitude'], emg['retention_time'],
                                emg['sigma'], emg['tau'])
            fig.add_trace(go.Scattergl(
                x=t_c, y=curve, mode='lines',
                line=dict(color=ec, width=1.2, dash='dot'),
                name=f'EMG RT={emg["retention_time"]:.2f} '
                     f'(area={emg["area"]:.1f})',
                hoverinfo='skip',
            ))

            # apex marker on the fitted curve
            apex_idx = int(np.argmax(curve))
            fig.add_trace(go.Scattergl(
                x=[t_c[apex_idx]], y=[curve[apex_idx]], mode='markers',
                marker=dict(symbol='diamond', size=8, color=ec,
                            line=dict(color='black', width=0.5)),
                showlegend=False,
                text=f'RT={emg["retention_time"]:.2f}<br>'
                     f'Area={emg["area"]:.1f}<br>'
                     f'σ={emg["sigma"]:.3f}, τ={emg["tau"]:.3f}',
                hoverinfo='text',
            ))

    # add ground truth comparison in subtitle (only available for synthetic data)
    if result is not None:
        true_areas = [f'{p["area"]:.1f}' for p in result['true_peaks']]
        fitted_areas = []
        for dr in deconv_results:
            for emg in dr['emg_fits']:
                fitted_areas.append(f'{emg["area"]:.1f}')

    fig.update_layout(
        title='Step 3: EMG Deconvolution — Fitted Components',
        xaxis_title='Retention Time (min)', yaxis_title='Signal',
        template='plotly_white', height=450, hovermode='closest',
        legend=dict(font=dict(size=9)),
    )
    return fig


def _plot_step3_mpl(time, signal, chunks, infer_results,
                     deconv_results, result=None):
    import matplotlib.pyplot as plt

    emg_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                  '#911eb4', '#42d4f4', '#f032e6', '#bfef45']
    cmap = plt.cm.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, signal, color='#333333', linewidth=0.8, label='Signal')

    color_idx = 0
    for ci, (chunk, ir, dr) in enumerate(zip(chunks, infer_results,
                                              deconv_results)):
        si, ei = chunk.start_index, chunk.end_index + 1
        t_c, s_c = time[si:ei], signal[si:ei]
        chunk_color = cmap(ci % 10)
        ax.fill_between(t_c, 0, s_c, color=chunk_color, alpha=0.06)

        if dr['popt'] is None or not dr['emg_fits']:
            continue

        fitted_sum = _multi_emg(t_c, *dr['popt'])
        ax.plot(t_c, fitted_sum, color=chunk_color, linewidth=1.5,
                linestyle='--', alpha=0.8)

        for emg in dr['emg_fits']:
            ec = emg_colors[color_idx % len(emg_colors)]
            color_idx += 1
            curve = _single_emg(t_c, emg['amplitude'], emg['retention_time'],
                                emg['sigma'], emg['tau'])
            ax.plot(t_c, curve, color=ec, linewidth=1, linestyle=':')
            apex_idx = int(np.argmax(curve))
            ax.plot(t_c[apex_idx], curve[apex_idx], marker='D', color=ec,
                    markersize=5, markeredgecolor='black',
                    markeredgewidth=0.5, zorder=5)

    ax.set_title('Step 3: EMG Deconvolution — Fitted Components')
    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Signal')
    return fig
