"""
Visualization, training, and evaluation functions for the
apex-heatmap U-Net deconvolution pipeline.

Designed for use in Jupyter notebooks.

Usage:
    from pipeline_viz import (
        plot_chromatograms,
        plot_chunked_chromatograms,
        plot_training_grid,
        train_model,
        load_model,
        evaluate_model,
    )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import find_peaks
from scipy.stats import exponnorm

from synthetic_chromatogram import SyntheticChromatogramGenerator
from chunker import chunk_chromatogram
from training_data import generate_training_data
from unet_model import GCHeatmapUNet


# --------------------------------------------------------------------------- #
#  1. Raw chromatograms with components + apexes
# --------------------------------------------------------------------------- #

def plot_chromatograms(n=5, seed=None, engine='plotly', **gen_kwargs):
    """
    Plot raw synthetic chromatograms with individual EMG components
    and apex markers.

    Parameters
    ----------
    n : int
        Number of chromatograms to plot.
    seed : int or None
        Base seed; each row uses seed+i for variety.
    engine : str
        'plotly' or 'matplotlib'.
    **gen_kwargs
        Passed to SyntheticChromatogramGenerator.generate().
    """
    defaults = dict(num_peaks=10, merge_probability=0.4, snr=200,
                    baseline_type='flat', time_range=(0, 30))
    defaults.update(gen_kwargs)

    results = []
    for i in range(n):
        s = (seed + i) if seed is not None else None
        gen = SyntheticChromatogramGenerator(seed=s)
        results.append(gen.generate(**defaults))

    if engine == 'plotly':
        return _plot_chromatograms_plotly(results)
    return _plot_chromatograms_mpl(results)


def _plot_chromatograms_plotly(results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(results)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                        vertical_spacing=0.06,
                        subplot_titles=[
                            f'{len(r["true_peaks"])} peaks (SNR={r["snr"]:.0f})'
                            for r in results
                        ])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for row, result in enumerate(results, 1):
        t = result['x']

        # raw signal
        fig.add_trace(go.Scattergl(
            x=t, y=result['original_y'], mode='lines',
            line=dict(color='#333333', width=0.8),
            name='Raw signal', showlegend=(row == 1),
            legendgroup='raw', hoverinfo='skip',
        ), row=row, col=1)

        # individual EMG components
        clusters_seen = set()
        for pp in result['peak_params']:
            K = pp['tau'] / max(pp['sigma'], 1e-6)
            curve = pp['amp'] * exponnorm.pdf(
                t, K, loc=pp['mu'], scale=max(pp['sigma'], 1e-6))
            # add baseline back so components align with raw signal
            curve = curve + result['baseline_y']
            color = colors[pp['cluster_id'] % len(colors)]
            show = pp['cluster_id'] not in clusters_seen and row == 1
            clusters_seen.add(pp['cluster_id'])
            fig.add_trace(go.Scattergl(
                x=t, y=curve, mode='lines',
                line=dict(color=color, width=0.8, dash='dot'),
                name=f'Cluster {pp["cluster_id"]}',
                legendgroup=f'c{pp["cluster_id"]}',
                showlegend=False, hoverinfo='skip', opacity=0.6,
            ), row=row, col=1)

        # apex markers
        merged = [p for p in result['true_peaks'] if p['is_merged']]
        singles = [p for p in result['true_peaks'] if not p['is_merged']]

        if singles:
            fig.add_trace(go.Scattergl(
                x=[p['rt'] for p in singles],
                y=[result['original_y'][p['apex_index']] for p in singles],
                mode='markers', name='Single',
                marker=dict(symbol='diamond', size=7,
                            color=[colors[p['cluster_id'] % len(colors)]
                                   for p in singles],
                            line=dict(color='black', width=0.5)),
                showlegend=(row == 1), legendgroup='single',
                hoverinfo='skip',
            ), row=row, col=1)

        if merged:
            fig.add_trace(go.Scattergl(
                x=[p['rt'] for p in merged],
                y=[result['original_y'][p['apex_index']] for p in merged],
                mode='markers', name='Merged',
                marker=dict(symbol='diamond', size=8,
                            color=[colors[p['cluster_id'] % len(colors)]
                                   for p in merged],
                            line=dict(color='red', width=1.0)),
                showlegend=(row == 1), legendgroup='merged',
                hoverinfo='skip',
            ), row=row, col=1)

    fig.update_layout(
        height=250 * n, template='plotly_white',
        title='Synthetic Chromatograms — Raw Signal + Components',
        hovermode='closest',
        legend=dict(font=dict(size=9)),
    )
    fig.update_xaxes(title_text='Retention Time (min)', row=n, col=1)
    return fig


def _plot_chromatograms_mpl(results):
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap('tab10')

    for ax, result in zip(axes, results):
        t = result['x']
        ax.plot(t, result['original_y'], color='#333333', linewidth=0.8)

        for pp in result['peak_params']:
            K = pp['tau'] / max(pp['sigma'], 1e-6)
            curve = pp['amp'] * exponnorm.pdf(
                t, K, loc=pp['mu'], scale=max(pp['sigma'], 1e-6))
            curve = curve + result['baseline_y']
            color = cmap(pp['cluster_id'] % 10)
            ax.plot(t, curve, color=color, linewidth=0.7, linestyle=':',
                    alpha=0.6)

        for p in result['true_peaks']:
            color = cmap(p['cluster_id'] % 10)
            ec = 'red' if p['is_merged'] else 'black'
            ax.plot(p['rt'], result['original_y'][p['apex_index']],
                    marker='D', color=color, markersize=5,
                    markeredgecolor=ec, markeredgewidth=0.5, zorder=5)

        ax.set_ylabel('Signal')
        ax.set_title(f'{len(result["true_peaks"])} peaks (SNR={result["snr"]:.0f})',
                      fontsize=9)

    axes[-1].set_xlabel('Retention Time (min)')
    fig.suptitle('Synthetic Chromatograms — Raw Signal + Components', y=1.01)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  2. Chromatograms with chunk boundaries
# --------------------------------------------------------------------------- #

def plot_chunked_chromatograms(n=5, seed=None, engine='plotly',
                               chunker_kwargs=None, **gen_kwargs):
    """
    Plot synthetic chromatograms with chunk boundaries overlaid.

    Parameters
    ----------
    n : int
        Number of chromatograms.
    seed : int or None
    engine : str
    chunker_kwargs : dict or None
    **gen_kwargs
        Passed to generate().
    """
    defaults = dict(num_peaks=10, merge_probability=0.4, snr=200,
                    baseline_type='flat', time_range=(0, 30))
    defaults.update(gen_kwargs)
    chunk_kw = chunker_kwargs or {}

    rows_data = []
    for i in range(n):
        s = (seed + i) if seed is not None else None
        gen = SyntheticChromatogramGenerator(seed=s)
        result = gen.generate(**defaults)
        chunks = chunk_chromatogram(result['x'], result['corrected_y'],
                                    **chunk_kw)
        rows_data.append((result, chunks))

    if engine == 'plotly':
        return _plot_chunked_plotly(rows_data)
    return _plot_chunked_mpl(rows_data)


def _plot_chunked_plotly(rows_data):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(rows_data)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                        vertical_spacing=0.06,
                        subplot_titles=[
                            f'{len(chunks)} chunks from {len(r["true_peaks"])} peaks'
                            for r, chunks in rows_data
                        ])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for row, (result, chunks) in enumerate(rows_data, 1):
        t = result['x']
        s = result['corrected_y']

        fig.add_trace(go.Scattergl(
            x=t, y=s, mode='lines',
            line=dict(color='#333333', width=0.8),
            name='Signal', showlegend=(row == 1),
            legendgroup='sig', hoverinfo='skip',
        ), row=row, col=1)

        for ci, chunk in enumerate(chunks):
            color = colors[ci % len(colors)]
            si, ei = chunk.start_index, chunk.end_index + 1
            t_c, s_c = t[si:ei], s[si:ei]

            fig.add_trace(go.Scatter(
                x=np.concatenate([t_c, t_c[::-1]]),
                y=np.concatenate([s_c, np.zeros(len(s_c))]),
                fill='toself', fillcolor=color, opacity=0.15,
                line=dict(width=0), hoverinfo='skip',
                name=f'Chunk {ci} ({chunk.n_peaks}p)',
                showlegend=(row == 1),
                legendgroup=f'chunk_{ci}',
            ), row=row, col=1)

            fig.add_trace(go.Scattergl(
                x=t_c, y=s_c, mode='lines',
                line=dict(color=color, width=1.5),
                showlegend=False, hoverinfo='skip',
            ), row=row, col=1)

            for pi in chunk.peak_indices:
                fig.add_trace(go.Scattergl(
                    x=[t[pi]], y=[s[pi]], mode='markers',
                    marker=dict(symbol='diamond', size=7, color=color,
                                line=dict(color='black', width=0.5)),
                    showlegend=False, hoverinfo='skip',
                ), row=row, col=1)

    fig.update_layout(
        height=250 * n, template='plotly_white',
        title='Chunked Chromatograms',
        hovermode='closest',
        legend=dict(font=dict(size=9)),
    )
    fig.update_xaxes(title_text='Retention Time (min)', row=n, col=1)
    return fig


def _plot_chunked_mpl(rows_data):
    import matplotlib.pyplot as plt

    n = len(rows_data)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap('tab10')

    for ax, (result, chunks) in zip(axes, rows_data):
        t, s = result['x'], result['corrected_y']
        ax.plot(t, s, color='#333333', linewidth=0.8)

        for ci, chunk in enumerate(chunks):
            color = cmap(ci % 10)
            si, ei = chunk.start_index, chunk.end_index + 1
            ax.fill_between(t[si:ei], 0, s[si:ei], color=color, alpha=0.15)
            ax.plot(t[si:ei], s[si:ei], color=color, linewidth=1.5)
            for pi in chunk.peak_indices:
                ax.plot(t[pi], s[pi], marker='D', color=color, markersize=5,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=5)

        ax.set_ylabel('Signal')
        ax.set_title(f'{len(chunks)} chunks from {len(result["true_peaks"])} peaks',
                      fontsize=9)

    axes[-1].set_xlabel('Retention Time (min)')
    fig.suptitle('Chunked Chromatograms', y=1.01)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  3. Preprocessed training data grid (5x5)
# --------------------------------------------------------------------------- #

def plot_training_grid(X, Y, metadata, rows=5, cols=5, seed=None,
                       engine='plotly'):
    """
    Plot a grid of preprocessed training samples (normalised 256-pt windows
    with heatmap overlay).

    Parameters
    ----------
    X : ndarray (N, 256, 1)
    Y : ndarray (N, 256, 1)
    metadata : list of dicts
    rows, cols : int
    seed : int or None
        For reproducible random sampling.
    engine : str
    """
    rng = np.random.default_rng(seed)
    n = rows * cols
    indices = rng.choice(len(X), size=min(n, len(X)), replace=False)
    indices.sort()

    if engine == 'plotly':
        return _plot_grid_plotly(X, Y, metadata, indices, rows, cols)
    return _plot_grid_mpl(X, Y, metadata, indices, rows, cols)


def _plot_grid_plotly(X, Y, metadata, indices, rows, cols):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = [f'{metadata[i]["n_apexes"]} apex'
              + ('es' if metadata[i]['n_apexes'] != 1 else '')
              for i in indices]

    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.04,
                        horizontal_spacing=0.02,
                        subplot_titles=titles)

    x_axis = np.arange(256)

    for k, idx in enumerate(indices):
        r = k // cols + 1
        c = k % cols + 1
        sig = X[idx, :, 0]
        hmap = Y[idx, :, 0]
        meta = metadata[idx]

        fig.add_trace(go.Scattergl(
            x=x_axis, y=sig, mode='lines',
            line=dict(color='#1f77b4', width=0.8),
            showlegend=False, hoverinfo='skip',
        ), row=r, col=c)

        fig.add_trace(go.Scattergl(
            x=x_axis, y=hmap, mode='lines',
            line=dict(color='red', width=1.2, dash='dash'),
            showlegend=False, hoverinfo='skip', opacity=0.8,
        ), row=r, col=c)

        for ai in meta['apex_indices']:
            fig.add_trace(go.Scattergl(
                x=[ai], y=[sig[ai]], mode='markers',
                marker=dict(symbol='diamond', size=5, color='red',
                            line=dict(color='black', width=0.5)),
                showlegend=False, hoverinfo='skip',
            ), row=r, col=c)

    fig.update_layout(
        height=180 * rows, width=200 * cols,
        template='plotly_white',
        title=f'Training Data — {len(X)} samples ({rows}x{cols} shown)',
        hovermode='closest',
        margin=dict(t=60),
    )
    # hide tick labels for cleaner grid
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def _plot_grid_mpl(X, Y, metadata, indices, rows, cols):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2 * rows),
                              sharex=True, sharey=True)

    for k, idx in enumerate(indices):
        r = k // cols
        c = k % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        sig = X[idx, :, 0]
        hmap = Y[idx, :, 0]
        meta = metadata[idx]

        ax.plot(sig, color='#1f77b4', linewidth=0.6)
        ax.plot(hmap, color='red', linewidth=1.0, linestyle='--', alpha=0.8)
        for ai in meta['apex_indices']:
            ax.plot(ai, sig[ai], marker='D', color='red', markersize=3,
                    markeredgecolor='black', markeredgewidth=0.3, zorder=5)

        n_ap = meta['n_apexes']
        ax.set_title(f'{n_ap} apex{"es" if n_ap != 1 else ""}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Training Data — {len(X)} samples', fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  4. Model training
# --------------------------------------------------------------------------- #

def train_model(X, Y, *,
                model=None,
                epochs=10,
                batch_size=32,
                lr=1e-3,
                loss_fn=None,
                device=None,
                save_path=None,
                verbose=True):
    """
    Train the GCHeatmapUNet on preprocessed training data.

    Parameters
    ----------
    X : ndarray (N, 256, 1)
    Y : ndarray (N, 256, 1)
    model : GCHeatmapUNet or None
        If None, creates a new model.
    epochs : int
    batch_size : int
    lr : float
    loss_fn : nn.Module or None
        If None, uses MSELoss.
    device : str or None
        Auto-detects MPS/CUDA/CPU if None.
    save_path : str or None
        If provided, saves model weights after training.
    verbose : bool

    Returns
    -------
    model : GCHeatmapUNet (on CPU)
    history : list of float — per-epoch average loss
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if verbose:
        print(f'Device: {device}')
        print(f'Training samples: {len(X)}')

    if model is None:
        model = GCHeatmapUNet()

    model = model.to(device)

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (N, 256, 1) -> (N, 1, 256) for PyTorch conv1d
    X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    Y_t = torch.tensor(Y, dtype=torch.float32).permute(0, 2, 1)

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        history.append(avg_loss)

        if verbose:
            print(f'Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}')

    model = model.cpu()

    if save_path:
        torch.save(model.state_dict(), save_path)
        if verbose:
            print(f'Model saved to {save_path}')

    return model, history


def load_model(path, *, device=None):
    """
    Load a saved GCHeatmapUNet from disk, auto-detecting out_channels.

    Parameters
    ----------
    path : str
        Path to the .pth state dict file.
    device : str or None
        Device to map weights to. Defaults to CPU.

    Returns
    -------
    model : GCHeatmapUNet (on CPU, in eval mode)
    """
    state_dict = torch.load(path, map_location=device or 'cpu')
    # Auto-detect out_channels from final_conv.0.weight shape
    out_channels = state_dict['final_conv.0.weight'].shape[0]
    model = GCHeatmapUNet(out_channels=out_channels)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
#  5. Model evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(model, X_test=None, Y_test=None, metadata_test=None, *,
                   num_test_chromatograms=100,
                   test_seed=999,
                   heatmap_height=0.15,
                   heatmap_distance=10,
                   device=None,
                   engine='plotly',
                   chromatogram_kwargs=None,
                   chunker_kwargs=None,
                   max_count_label=4):
    """
    Evaluate the trained U-Net as a multi-class peak-count classifier and
    visualize representative examples.

    Each chunk window is classified by the exact number of peaks detected
    (capped at max_count_label, displayed as "N+").

    Parameters
    ----------
    model : GCHeatmapUNet
    X_test, Y_test, metadata_test : optional
        Pre-generated test data. If None, generates new test data.
    num_test_chromatograms : int
        Number of chromatograms to generate for test data (if not provided).
    test_seed : int
    heatmap_height : float
        Minimum height for peak detection on predicted heatmap.
    heatmap_distance : int
        Minimum distance between detected peaks.
    device : str or None
    engine : str — 'plotly' or 'matplotlib'
    chromatogram_kwargs, chunker_kwargs : dict or None
    max_count_label : int
        Peak counts above this value are clamped to this label (shown as "N+").

    Returns
    -------
    dict with keys: accuracy, true_counts, pred_counts,
                    confusion_matrix, classification_report, predictions
    """
    from collections import defaultdict
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score)

    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if X_test is None or Y_test is None:
        print(f'Generating test data from {num_test_chromatograms} chromatograms...')
        X_test, Y_test, metadata_test = generate_training_data(
            num_chromatograms=num_test_chromatograms,
            seed=test_seed,
            chromatogram_kwargs=chromatogram_kwargs,
            chunker_kwargs=chunker_kwargs,
        )

    n_samples = len(X_test)
    print(f'Evaluating on {n_samples} test samples...')

    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        preds_t = model(X_t)
    preds = preds_t.cpu().permute(0, 2, 1).numpy()
    model = model.cpu()

    y_true_counts = []
    y_pred_counts = []
    for i in range(n_samples):
        true_peaks_i, _ = find_peaks(Y_test[i, :, 0], height=0.5)
        n_true = min(len(true_peaks_i), max_count_label)

        pred_peaks_i, _ = find_peaks(preds[i, :, 0],
                                      height=heatmap_height,
                                      distance=heatmap_distance)
        n_pred = min(len(pred_peaks_i), max_count_label)

        y_true_counts.append(n_true)
        y_pred_counts.append(n_pred)

    labels = list(range(max_count_label + 1))
    label_names = [f'{l}+' if l == max_count_label else str(l) for l in labels]

    acc       = accuracy_score(y_true_counts, y_pred_counts)
    n_correct = sum(t == p for t, p in zip(y_true_counts, y_pred_counts))
    n_over    = sum(p > t   for t, p in zip(y_true_counts, y_pred_counts))
    n_under   = sum(p < t   for t, p in zip(y_true_counts, y_pred_counts))
    report    = classification_report(y_true_counts, y_pred_counts,
                                       labels=labels, target_names=label_names,
                                       zero_division=0)
    cm        = confusion_matrix(y_true_counts, y_pred_counts, labels=labels)

    print('-' * 50)
    print(f'Accuracy:         {acc:.4f}  ({n_correct}/{n_samples} exact)')
    print(f'Over-prediction:  {n_over:5d}  ({100*n_over/n_samples:.1f}%)')
    print(f'Under-prediction: {n_under:5d}  ({100*n_under/n_samples:.1f}%)')
    print()
    print('Classification Report (by peak count):')
    print(report)
    print()
    print('Confusion Matrix (rows=true, cols=predicted):')
    header = '      ' + ''.join(f'{n:>7}' for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f'{label_names[i]:>5} ' + ''.join(f'{v:>7}' for v in row))
    print('-' * 50)

    _plot_count_heatmap(cm, label_names, engine)
    _plot_count_examples(X_test, Y_test, preds, y_true_counts, y_pred_counts,
                         heatmap_height, heatmap_distance, engine)

    return {
        'accuracy':              acc,
        'true_counts':           y_true_counts,
        'pred_counts':           y_pred_counts,
        'confusion_matrix':      cm,
        'classification_report': report,
        'predictions':           preds,
    }


# --------------------------------------------------------------------------- #
#  Evaluation visualisation helpers
# --------------------------------------------------------------------------- #

def _plot_count_heatmap(cm, label_names, engine):
    if engine == 'plotly':
        _plot_count_heatmap_plotly(cm, label_names)
    else:
        _plot_count_heatmap_mpl(cm, label_names)


def _plot_count_heatmap_plotly(cm, label_names):
    import plotly.graph_objects as go

    text = [[str(cm[i, j]) for j in range(len(label_names))]
            for i in range(len(label_names))]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=label_names,
        y=label_names,
        colorscale='Blues',
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=14),
        showscale=True,
        hovertemplate='True=%{y}<br>Pred=%{x}<br>Count=%{z}<extra></extra>',
    ))
    fig.update_layout(
        title='Confusion Matrix — Peak Count (rows=true, cols=predicted)',
        xaxis_title='Predicted Peak Count',
        yaxis_title='True Peak Count',
        height=420, width=500,
        template='plotly_white',
    )
    fig.show()


def _plot_count_heatmap_mpl(cm, label_names):
    import matplotlib.pyplot as plt

    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n)))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted Peak Count', fontsize=11)
    ax.set_ylabel('True Peak Count', fontsize=11)
    ax.set_title('Confusion Matrix — Peak Count', fontsize=12)
    thresh = cm.max() * 0.6
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > thresh else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color=color, fontsize=11, fontweight=weight)
    fig.tight_layout()
    plt.show()


def _plot_count_examples(X, Y, preds, y_true_counts, y_pred_counts,
                          heatmap_height, heatmap_distance, engine,
                          n_per_group=2):
    """
    Collect and plot example cases grouped as correct, over-prediction,
    and under-prediction. Shows up to n_per_group examples per group,
    sorted by most severe/common error first.
    """
    from collections import defaultdict

    pair_to_indices = defaultdict(list)
    for i, (t, p) in enumerate(zip(y_true_counts, y_pred_counts)):
        pair_to_indices[(t, p)].append(i)

    def _pick(pairs, n):
        selected = []
        for pair in pairs:
            if pair in pair_to_indices:
                selected.append((pair, pair_to_indices[pair][0]))
            if len(selected) == n:
                break
        return selected

    correct_pairs = sorted([k for k in pair_to_indices if k[0] == k[1]],
                            key=lambda x: x[0], reverse=True)
    over_pairs    = sorted([k for k in pair_to_indices if k[1] > k[0]],
                            key=lambda x: (x[1] - x[0], len(pair_to_indices[x])),
                            reverse=True)
    under_pairs   = sorted([k for k in pair_to_indices if k[1] < k[0]],
                            key=lambda x: (x[0] - x[1], len(pair_to_indices[x])),
                            reverse=True)

    cases = []
    for pair, idx in _pick(correct_pairs, n_per_group):
        n = len(pair_to_indices[pair])
        cases.append((f'Correct (True={pair[0]}, Pred={pair[1]}, n={n})', 'correct', idx))
    for pair, idx in _pick(over_pairs, n_per_group):
        n = len(pair_to_indices[pair])
        cases.append((f'Over-pred (True={pair[0]}, Pred={pair[1]}, n={n})', 'over', idx))
    for pair, idx in _pick(under_pairs, n_per_group):
        n = len(pair_to_indices[pair])
        cases.append((f'Under-pred (True={pair[0]}, Pred={pair[1]}, n={n})', 'under', idx))

    if not cases:
        return

    if engine == 'plotly':
        _plot_count_examples_plotly(X, Y, preds, cases,
                                    heatmap_height, heatmap_distance)
    else:
        _plot_count_examples_mpl(X, Y, preds, cases,
                                  heatmap_height, heatmap_distance)


def _plot_count_examples_plotly(X, Y, preds, cases, heatmap_height, heatmap_distance):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(cases)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    titles = [c[0] for c in cases] + [''] * (nrows * ncols - n)

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=titles,
                        vertical_spacing=0.14,
                        horizontal_spacing=0.06)

    x_axis = list(range(256))
    for k, (title, kind, idx) in enumerate(cases):
        r, c = k // ncols + 1, k % ncols + 1
        show  = (k == 0)
        sig       = X[idx, :, 0].tolist()
        hmap_pred = preds[idx, :, 0].tolist()
        hmap_true = Y[idx, :, 0].tolist()
        pp, _ = find_peaks(np.array(hmap_pred), height=heatmap_height,
                            distance=heatmap_distance)
        tp, _ = find_peaks(np.array(hmap_true), height=0.5)

        fig.add_trace(go.Scattergl(
            x=x_axis, y=sig, mode='lines',
            line=dict(color='#1f77b4', width=1),
            name='Signal', showlegend=show, legendgroup='sig',
            hoverinfo='skip',
        ), row=r, col=c)
        fig.add_trace(go.Scattergl(
            x=x_axis, y=hmap_true, mode='lines',
            line=dict(color='green', width=0.8, dash='dot'),
            name='True heatmap', showlegend=show, legendgroup='true',
            opacity=0.7, hoverinfo='skip',
        ), row=r, col=c)
        fig.add_trace(go.Scattergl(
            x=x_axis, y=hmap_pred, mode='lines',
            line=dict(color='red', width=1.2, dash='dash'),
            name='Pred heatmap', showlegend=show, legendgroup='pred',
            opacity=0.8, hoverinfo='skip',
        ), row=r, col=c)
        if len(tp) > 0:
            fig.add_trace(go.Scattergl(
                x=list(tp), y=[sig[p] for p in tp], mode='markers',
                marker=dict(symbol='triangle-up', size=9, color='green'),
                name='True apex', showlegend=show, legendgroup='tapex',
                hoverinfo='skip',
            ), row=r, col=c)
        if len(pp) > 0:
            fig.add_trace(go.Scattergl(
                x=list(pp), y=[sig[p] for p in pp], mode='markers',
                marker=dict(symbol='x', size=10, color='red', line=dict(width=2)),
                name='Pred apex', showlegend=show, legendgroup='papex',
                hoverinfo='skip',
            ), row=r, col=c)

    fig.update_layout(
        height=300 * nrows + 60, width=900,
        template='plotly_white',
        title='Peak Count Prediction Examples',
        hovermode='closest',
        legend=dict(font=dict(size=9)),
    )
    fig.show()


def _plot_count_examples_mpl(X, Y, preds, cases, heatmap_height, heatmap_distance):
    import matplotlib.pyplot as plt

    n = len(cases)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    kind_colors = {'correct': '#2ca02c', 'over': '#d62728', 'under': '#ff7f0e'}
    x_axis = np.arange(256)

    for k, (title, kind, idx) in enumerate(cases):
        ax        = axes[k // ncols][k % ncols]
        sig       = X[idx, :, 0]
        hmap_pred = preds[idx, :, 0]
        hmap_true = Y[idx, :, 0]
        pp, _ = find_peaks(hmap_pred, height=heatmap_height, distance=heatmap_distance)
        tp, _ = find_peaks(hmap_true, height=0.5)

        ax.plot(x_axis, sig,       color='#1f77b4', linewidth=1,   label='Signal')
        ax.plot(x_axis, hmap_true, color='green',   linewidth=0.8,
                linestyle=':', alpha=0.7, label='True heatmap')
        ax.plot(x_axis, hmap_pred, color='red',     linewidth=1.2,
                linestyle='--', alpha=0.8, label='Pred heatmap')
        if len(tp) > 0:
            ax.plot(tp, sig[tp], '^', color='green', markersize=7, label='True apex')
        if len(pp) > 0:
            ax.plot(pp, sig[pp], 'rx', markersize=10, markeredgewidth=2, label='Pred apex')

        ax.set_title(title, fontsize=8, color=kind_colors[kind])
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.4)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle('Peak Count Prediction Examples', fontsize=11)
    fig.tight_layout()
    plt.show()
