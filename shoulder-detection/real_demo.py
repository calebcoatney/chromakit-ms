"""
End-to-end pipeline demonstration on real Agilent GC-FID data.

Mirrors demo.py but operates on a real .D directory instead of a synthetic
chromatogram. Data is loaded via the rainbow library and preprocessed
(Savitzky-Golay smoothing + pybaselines baseline correction) before being
passed through the chunker → U-Net → EMG deconvolution pipeline.

Usage
-----
    from real_demo import load_real_chromatogram, run_real_pipeline

    # Quick start — auto-detects detector, uses default preprocessing
    result = run_real_pipeline(model, '/path/to/sample.D')

    # With explicit options
    result = run_real_pipeline(
        model, '/path/to/sample.D',
        detector='FID1A',
        baseline_lam=1e7,
        engine='matplotlib',
    )

    # Inspect preprocessing before running the full pipeline
    raw = load_real_chromatogram('/path/to/sample.D')
"""

import os

import numpy as np
import rainbow as rb
from scipy.signal import savgol_filter
from pybaselines import Baseline

from chunker import chunk_chromatogram
from demo import (
    infer_chunks,
    deconvolve_chunk,
    _plot_step2_plotly,
    _plot_step2_mpl,
    _plot_step3_plotly,
    _plot_step3_mpl,
)

_PREFERRED_DETECTORS = ['FID1A', 'TCD2B', 'FID2B', 'TCD1A', 'TCD1B', 'FID2A']
_LAM_METHODS = {'asls', 'airpls', 'arpls', 'mixture_model', 'irsqr'}


def _normalise_path(d_path):
    """Strip shell-style backslash escapes that don't belong in Python strings."""
    return d_path.replace('\\ ', ' ').replace("\\'", "'")


# --------------------------------------------------------------------------- #
#  Data loading and preprocessing
# --------------------------------------------------------------------------- #

def load_real_chromatogram(d_path, detector=None, *,
                            smooth=True,
                            smooth_window=51,
                            smooth_order=3,
                            baseline_lam=1e6,
                            baseline_method='arpls'):
    """
    Load and preprocess a real Agilent .D GC chromatogram.

    Parameters
    ----------
    d_path : str
        Path to an Agilent .D directory (the '.D' suffix is optional).
    detector : str or None
        Detector file name without extension (e.g. 'FID1A').
        If None, auto-selects the first match from a preferred priority list:
        FID1A → TCD2B → FID2B → TCD1A → TCD1B → FID2A → first available.
    smooth : bool
        Apply Savitzky-Golay smoothing before baseline correction.
    smooth_window : int
        Window length for savgol_filter (odd; auto-corrected if even).
    smooth_order : int
        Polynomial order for savgol_filter.
    baseline_lam : float
        Regularisation parameter λ for the baseline algorithm.
    baseline_method : str
        pybaselines method name ('arpls', 'asls', 'airpls', etc.).
        Default is 'arpls' (Asymmetrically Reweighted Penalized Least Squares),
        which is more robust to chromatographic peaks than basic ASLS.

    Returns
    -------
    dict with keys:
        x           — time axis (minutes), shape (N,)
        y_raw       — raw signal as loaded from the .ch file, shape (N,)
        y_smooth    — signal after smoothing (equals y_raw if smooth=False)
        baseline    — estimated baseline curve, shape (N,)
        y_corrected — baseline-corrected signal; this is the pipeline input
        filename    — basename of the .D directory
        detector    — name of the detector that was loaded
    """
    # Normalise path — strip shell-style escapes (\ before spaces/apostrophes)
    # that users sometimes copy-paste from terminal commands
    d_path = _normalise_path(d_path)

    # Normalise path suffix
    if not d_path.endswith('.D'):
        if os.path.isdir(d_path + '.D'):
            d_path = d_path + '.D'
        elif not os.path.isdir(d_path):
            raise FileNotFoundError(f'Cannot find .D directory at: {d_path}')

    data_dir = rb.read(d_path)
    filename = os.path.basename(d_path)

    # Detector selection
    if detector is None:
        available = [str(f) for f in data_dir.datafiles if str(f).endswith('.ch')]
        for pref in _PREFERRED_DETECTORS:
            if pref + '.ch' in available:
                detector = pref
                break
        if detector is None:
            if available:
                detector = available[0].replace('.ch', '')
                print(f'No preferred detector found; using {detector}')
            else:
                raise ValueError(f'No .ch detector files found in {d_path}')

    data  = data_dir.get_file(detector + '.ch')
    x     = np.asarray(data.xlabels).flatten()
    y_raw = np.asarray(data.data).flatten().astype(float)

    # Smoothing
    if smooth:
        window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        window = min(window, len(y_raw) - (1 if len(y_raw) % 2 == 0 else 0))
        y_smooth = savgol_filter(y_raw, window_length=window, polyorder=smooth_order)
    else:
        y_smooth = y_raw.copy()

    # Baseline correction
    fitter    = Baseline()
    method_fn = getattr(fitter, baseline_method, None)
    if method_fn is None:
        print(f'Unknown baseline method "{baseline_method}", falling back to asls.')
        method_fn = fitter.asls
        baseline_method = 'asls'

    if baseline_method in _LAM_METHODS:
        baseline_arr, _ = method_fn(y_smooth, lam=baseline_lam)
    else:
        baseline_arr, _ = method_fn(y_smooth)

    y_corrected = y_smooth - baseline_arr

    print(f'Loaded  : {filename}  |  detector={detector}')
    print(f'Points  : {len(x)}    |  t=[{x[0]:.2f}, {x[-1]:.2f}] min')

    return {
        'x':           x,
        'y_raw':       y_raw,
        'y_smooth':    y_smooth,
        'baseline':    baseline_arr,
        'y_corrected': y_corrected,
        'filename':    filename,
        'detector':    detector,
    }


# --------------------------------------------------------------------------- #
#  End-to-end pipeline
# --------------------------------------------------------------------------- #

def run_real_pipeline(model, d_path, *,
                       detector=None,
                       engine='plotly',
                       smooth=True,
                       smooth_window=51,
                       smooth_order=3,
                       baseline_lam=1e6,
                       baseline_method='arpls',
                       min_peak_prominence=0.05,
                       heatmap_height=0.15,
                       heatmap_distance=10,
                       min_apex_prominence=0.0,
                       chunker_kwargs=None):
    """
    Run the full deconvolution pipeline on a real Agilent .D chromatogram.

    Produces three plots (same structure as demo.run_pipeline_demo):

      1. Preprocessing QA — raw signal, smoothed signal, baseline estimate,
                            and baseline-corrected signal
      2. Chunked chromatogram with U-Net predicted apex positions
      3. Deconvolved chromatogram with fitted EMG curves per chunk

    Parameters
    ----------
    model : GCHeatmapUNet (trained)
    d_path : str
        Path to an Agilent .D directory.
    detector : str or None
        Forwarded to load_real_chromatogram.
    engine : str — 'plotly' or 'matplotlib'
    smooth, smooth_window, smooth_order, baseline_lam, baseline_method :
        Forwarded to load_real_chromatogram.
    heatmap_height : float
        Minimum heatmap response height for apex detection.
    heatmap_distance : int
        Minimum separation (in 256-pt window pixels) between detected apexes.
        Default 10 matches the training configuration.
    min_apex_prominence : float
        Minimum prominence of each predicted apex in the normalised signal
        [0, 1].  Before the filter runs, apexes are snapped to the nearest
        s_norm maximum using a window of ±heatmap_distance pts; apexes that
        collapse to the same maximum (false splits on a single peak) are
        automatically deduplicated.  Default 0.0 (deduplication only, no
        prominence cut).  Raise to 0.05–0.10 to further suppress weak apexes.
    chunker_kwargs : dict or None
        Extra keyword arguments forwarded to chunk_chromatogram.

    Returns
    -------
    dict with keys:
        raw_data       — output of load_real_chromatogram
        chunks         — list of Chunk objects
        infer_results  — list of per-chunk U-Net inference dicts
        deconv_results — list of per-chunk EMG deconvolution dicts
        figures        — tuple (fig1, fig2, fig3)
    """
    # Step 1: load + preprocess
    d_result = load_real_chromatogram(
        d_path, detector=detector,
        smooth=smooth, smooth_window=smooth_window, smooth_order=smooth_order,
        baseline_lam=baseline_lam, baseline_method=baseline_method,
    )
    time   = d_result['x']
    signal = d_result['y_corrected']

    # Step 2: chunk
    # Apply a signal-range-relative prominence threshold (same convention as
    # production processor.py line 33) so tiny noise bumps are never chunked.
    signal_range = float(signal.max() - signal.min())
    abs_prom = (min_peak_prominence * signal_range
                if min_peak_prominence < 1 else min_peak_prominence)

    ck_kwargs = dict(chunker_kwargs or {})
    # Also support explicit min_prominence in chunker_kwargs (relative or absolute)
    if 'min_prominence' in ck_kwargs:
        val = ck_kwargs.pop('min_prominence')
        abs_prom = val * signal_range if val < 1 else val
    fp_kw = dict(ck_kwargs.pop('find_peaks_kwargs', {}) or {})
    fp_kw['prominence'] = abs_prom
    ck_kwargs['find_peaks_kwargs'] = fp_kw

    chunks = chunk_chromatogram(time, signal, **ck_kwargs)
    print(f'Chunker : {len(chunks)} chunk(s)  |  prominence threshold = {abs_prom:.4g} '
          f'({min_peak_prominence if min_peak_prominence < 1 else "absolute"} × signal range)')

    # Step 3: U-Net inference
    infer_results = infer_chunks(
        model, time, signal, chunks,
        heatmap_height=heatmap_height,
        heatmap_distance=heatmap_distance,
        min_prominence=min_apex_prominence,
    )
    n_apex_total = sum(len(ir['apex_rts']) for ir in infer_results)
    print(f'U-Net   : {n_apex_total} apex(es) predicted across {len(chunks)} chunk(s).')

    # Step 4: EMG deconvolution per chunk
    deconv_results = []
    n_fit_ok = 0
    for ir in infer_results:
        chunk = ir['chunk']
        si, ei = chunk.start_index, chunk.end_index + 1
        t_chunk = time[si:ei]
        s_chunk = signal[si:ei]
        emg_fits, popt = deconvolve_chunk(t_chunk, s_chunk, ir['apex_rts'])
        if popt is not None:
            n_fit_ok += 1
        deconv_results.append({
            't_chunk':  t_chunk,
            's_chunk':  s_chunk,
            'emg_fits': emg_fits,
            'popt':     popt,
            'apex_rts': ir['apex_rts'],
        })

    n_fit_fail = len(chunks) - n_fit_ok
    if n_fit_fail:
        print(f'Warning : EMG fitting failed for {n_fit_fail} chunk(s).')
    else:
        print(f'EMG fit : all {n_fit_ok} chunk(s) converged.')

    # Plots
    if engine == 'plotly':
        fig1 = _plot_step1_real_plotly(d_result)
        fig2 = _plot_step2_plotly(time, signal, chunks, infer_results)
        fig3 = _plot_step3_plotly(time, signal, chunks, infer_results,
                                   deconv_results)
    else:
        fig1 = _plot_step1_real_mpl(d_result)
        fig2 = _plot_step2_mpl(time, signal, chunks, infer_results)
        fig3 = _plot_step3_mpl(time, signal, chunks, infer_results,
                                deconv_results)

    return {
        'raw_data':       d_result,
        'chunks':         chunks,
        'infer_results':  infer_results,
        'deconv_results': deconv_results,
        'figures':        (fig1, fig2, fig3),
    }


# --------------------------------------------------------------------------- #
#  Step 1: preprocessing QA plot
# --------------------------------------------------------------------------- #

def _plot_step1_real_plotly(d_result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t        = d_result['x']
    y_raw    = d_result['y_raw']
    y_smooth = d_result['y_smooth']
    baseline = d_result['baseline']
    y_corr   = d_result['y_corrected']
    fname    = d_result['filename']
    det      = d_result['detector']

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            f'{fname}  ({det}) — Raw signal with baseline estimate',
            'Baseline-corrected signal  (pipeline input)',
        ],
    )

    fig.add_trace(go.Scattergl(
        x=t, y=y_raw, mode='lines',
        line=dict(color='#aaaaaa', width=0.8),
        name='Raw', opacity=0.7, hoverinfo='skip',
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=t, y=y_smooth, mode='lines',
        line=dict(color='#1f77b4', width=1.2),
        name='Smoothed', hoverinfo='skip',
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=t, y=baseline, mode='lines',
        line=dict(color='#d62728', width=1.5, dash='dash'),
        name='Baseline', hoverinfo='skip',
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=t, y=y_corr, mode='lines',
        line=dict(color='#333333', width=1),
        name='Corrected', hoverinfo='skip',
    ), row=2, col=1)

    fig.update_xaxes(title_text='Retention Time (min)', row=2, col=1)
    fig.update_yaxes(title_text='Signal', row=1, col=1)
    fig.update_yaxes(title_text='Signal', row=2, col=1)
    fig.update_layout(
        template='plotly_white',
        height=500,
        hovermode='x unified',
        legend=dict(font=dict(size=10)),
    )
    return fig


def _plot_step1_real_mpl(d_result):
    import matplotlib.pyplot as plt

    t        = d_result['x']
    y_raw    = d_result['y_raw']
    y_smooth = d_result['y_smooth']
    baseline = d_result['baseline']
    y_corr   = d_result['y_corrected']
    fname    = d_result['filename']
    det      = d_result['detector']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(t, y_raw,    color='#aaaaaa', linewidth=0.7, alpha=0.7, label='Raw')
    ax1.plot(t, y_smooth, color='#1f77b4', linewidth=1.2, label='Smoothed')
    ax1.plot(t, baseline, color='#d62728', linewidth=1.5,
             linestyle='--', label='Baseline')
    ax1.set_title(f'{fname}  ({det}) — Raw signal with baseline estimate')
    ax1.set_ylabel('Signal')
    ax1.legend(fontsize=9)

    ax2.plot(t, y_corr, color='#333333', linewidth=1, label='Corrected')
    ax2.set_title('Baseline-corrected signal  (pipeline input)')
    ax2.set_xlabel('Retention Time (min)')
    ax2.set_ylabel('Signal')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return fig
