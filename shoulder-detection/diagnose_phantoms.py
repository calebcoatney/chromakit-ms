"""
Diagnose where phantom (false positive) components come from.

Classifies each phantom by source:
  - "duplicate": within 1σ of a matched true peak (EMG fitter split one peak into two)
  - "neighbor":  within 0.5 min of a true peak but not matched (U-Net apex near real peak)
  - "hallucination": far from any true peak (U-Net or fitter invented a peak)

Also analyzes per-chunk statistics to identify whether over-prediction
comes from the U-Net (too many apexes) or the EMG fitter (splitting).

Usage:
    cd shoulder-detection
    python diagnose_phantoms.py
"""

import sys, os, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from synthetic_chromatogram import SyntheticChromatogramGenerator
from logic.deconvolution import run_deconvolution_pipeline, _extract_apices
from evaluation import match_peaks

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'eval_results/phantom_diagnosis'
WEIGHTS = 'gc_heatmap_unet_v3.pth'  # 1-channel model (the main one)
NUM_CHROMATOGRAMS = 100
SEED = 7777

CHROM_KW = dict(num_peaks=(5, 20), merge_probability=0.5, snr=None, baseline_type='random')
PIPE_KW = dict(min_prominence=0.01, heatmap_threshold=0.15, heatmap_distance=10)

os.makedirs(RESULTS_DIR, exist_ok=True)

import logic.deconvolution as _deconv
_deconv._model_cache.clear()
_deconv._model_instance = None


# ══════════════════════════════════════════════════════════════════════════════
#  Run pipeline and collect detailed per-chromatogram data
# ══════════════════════════════════════════════════════════════════════════════

print(f'Running pipeline on {NUM_CHROMATOGRAMS} chromatograms...')
gen = SyntheticChromatogramGenerator(seed=SEED)

all_phantom_types = []  # 'duplicate', 'neighbor', 'hallucination'
all_phantom_distances = []  # distance to nearest true peak
chromatogram_data = []  # for visualization

for ci in range(NUM_CHROMATOGRAMS):
    result = gen.generate(**CHROM_KW)
    time_axis = result['x']
    signal = result['corrected_y']
    true_peaks = result['true_peaks']

    _deconv._model_cache.clear()
    _deconv._model_instance = None

    try:
        deconv = run_deconvolution_pipeline(
            time_axis, signal, weights_path=WEIGHTS, **PIPE_KW
        )
        components = deconv.components
        chunks = deconv.chunks
    except Exception as e:
        components = []
        chunks = []

    matches, misses, phantoms = match_peaks(true_peaks, components, max_rt_distance=0.5)

    # Classify each phantom
    true_rts = [p['rt'] for p in true_peaks]
    true_sigmas = [p['sigma'] for p in true_peaks]
    matched_fitted_rts = set()
    for m in matches:
        if m.fitted_rt is not None:
            matched_fitted_rts.add(round(m.fitted_rt, 6))

    for ph in phantoms:
        ph_rt = ph['rt']
        if not true_rts:
            all_phantom_types.append('hallucination')
            all_phantom_distances.append(float('inf'))
            continue

        distances = [abs(ph_rt - trt) for trt in true_rts]
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]
        nearest_sigma = true_sigmas[nearest_idx]

        all_phantom_distances.append(nearest_dist)

        if nearest_dist < nearest_sigma:
            all_phantom_types.append('duplicate')
        elif nearest_dist < 0.5:
            all_phantom_types.append('neighbor')
        else:
            all_phantom_types.append('hallucination')

    # Save data for worst-case visualization
    n_phantoms = len(phantoms)
    chromatogram_data.append({
        'index': ci,
        'n_true': len(true_peaks),
        'n_fitted': len(components),
        'n_phantoms': n_phantoms,
        'n_misses': len(misses),
        'n_matched': len(matches),
        'time_axis': time_axis,
        'signal': signal,
        'true_peaks': true_peaks,
        'components': components,
        'chunks': chunks,
        'phantoms': phantoms,
        'matches': matches,
    })

    if (ci + 1) % 20 == 0:
        print(f'  {ci + 1}/{NUM_CHROMATOGRAMS}')


# ══════════════════════════════════════════════════════════════════════════════
#  Quantitative Analysis
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  PHANTOM DIAGNOSIS')
print('=' * 70)

type_counts = Counter(all_phantom_types)
total_phantoms = len(all_phantom_types)
total_fitted = sum(d['n_fitted'] for d in chromatogram_data)
total_true = sum(d['n_true'] for d in chromatogram_data)

print(f'\n  Total true peaks:    {total_true}')
print(f'  Total fitted:        {total_fitted}')
print(f'  Total phantoms:      {total_phantoms}')
print(f'  Excess components:   {total_fitted - total_true} '
      f'({(total_fitted - total_true) / total_true * 100:.0f}% over)')

print(f'\n  Phantom Classification:')
for ptype in ['duplicate', 'neighbor', 'hallucination']:
    n = type_counts.get(ptype, 0)
    pct = 100 * n / total_phantoms if total_phantoms > 0 else 0
    print(f'    {ptype:<16s}: {n:5d}  ({pct:.1f}%)')

print(f'\n  Duplicate = within 1σ of a matched peak (fitter split a real peak)')
print(f'  Neighbor  = within 0.5 min of a true peak (U-Net over-detected)')
print(f'  Hallucination = >0.5 min from any true peak')

# Distance distribution
dists = np.array(all_phantom_distances)
dists_finite = dists[np.isfinite(dists)]
if len(dists_finite) > 0:
    print(f'\n  Distance to nearest true peak (all phantoms):')
    print(f'    Median: {np.median(dists_finite):.3f} min')
    print(f'    P25:    {np.percentile(dists_finite, 25):.3f} min')
    print(f'    P75:    {np.percentile(dists_finite, 75):.3f} min')
    print(f'    P90:    {np.percentile(dists_finite, 90):.3f} min')

# Per-chromatogram over-prediction stats
over_counts = [d['n_fitted'] - d['n_true'] for d in chromatogram_data]
print(f'\n  Per-chromatogram over-prediction (fitted - true):')
print(f'    Median: {np.median(over_counts):.0f}')
print(f'    Mean:   {np.mean(over_counts):.1f}')
print(f'    Max:    {max(over_counts)}')
print(f'    % with over-prediction: '
      f'{100 * sum(1 for x in over_counts if x > 0) / len(over_counts):.0f}%')

# Fitted-to-true ratio distribution
ratios = [d['n_fitted'] / d['n_true'] if d['n_true'] > 0 else 0
          for d in chromatogram_data]
print(f'\n  Fitted/True ratio:')
print(f'    Median: {np.median(ratios):.2f}x')
print(f'    P90:    {np.percentile(ratios, 90):.2f}x')

# Save quantitative results
quant_results = {
    'total_true': total_true,
    'total_fitted': total_fitted,
    'total_phantoms': total_phantoms,
    'phantom_types': dict(type_counts),
    'distance_median': float(np.median(dists_finite)) if len(dists_finite) > 0 else None,
    'distance_p75': float(np.percentile(dists_finite, 75)) if len(dists_finite) > 0 else None,
    'over_prediction_median': float(np.median(over_counts)),
    'over_prediction_mean': float(np.mean(over_counts)),
    'fitted_true_ratio_median': float(np.median(ratios)),
}
with open(os.path.join(RESULTS_DIR, 'phantom_stats.json'), 'w') as f:
    json.dump(quant_results, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualizations
# ══════════════════════════════════════════════════════════════════════════════

# --- 1. Phantom type pie chart + distance histogram ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
labels = ['Duplicate\n(fitter split)', 'Neighbor\n(U-Net over-detect)', 'Hallucination\n(far from any peak)']
sizes = [type_counts.get('duplicate', 0), type_counts.get('neighbor', 0),
         type_counts.get('hallucination', 0)]
colors_pie = ['#ff7f0e', '#1f77b4', '#d62728']
explode = (0.05, 0.05, 0.05)
axes[0].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
axes[0].set_title(f'Phantom Sources (N={total_phantoms})', fontsize=12)

# Distance histogram
if len(dists_finite) > 0:
    axes[1].hist(dists_finite, bins=60, range=(0, 1.5), color='#1f77b4',
                 edgecolor='white', alpha=0.8)
    axes[1].axvline(0.1, color='orange', linestyle='--', linewidth=1.5, label='0.1 min (1σ typical)')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='0.5 min (match cutoff)')
    axes[1].set_xlabel('Distance to nearest true peak (min)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Phantom Distance Distribution')
    axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'phantom_classification.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: phantom_classification.png')


# --- 2. Per-chromatogram fitted vs true scatter ---
fig, ax = plt.subplots(figsize=(8, 8))
n_trues = [d['n_true'] for d in chromatogram_data]
n_fitteds = [d['n_fitted'] for d in chromatogram_data]
ax.scatter(n_trues, n_fitteds, alpha=0.5, s=30, c='#1f77b4')
max_val = max(max(n_trues), max(n_fitteds)) + 5
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect')
ax.plot([0, max_val], [0, 2 * max_val], 'r:', linewidth=0.8, alpha=0.5, label='2x over')
ax.set_xlabel('True Peak Count')
ax.set_ylabel('Fitted Component Count')
ax.set_title('Per-Chromatogram: Fitted vs True Peak Count')
ax.legend()
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val * 2.5)
ax.set_aspect('equal' if max(n_fitteds) < max_val * 1.5 else 'auto')
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'fitted_vs_true_scatter.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: fitted_vs_true_scatter.png')


# --- 3. Worst-case chromatograms (most phantoms) ---
sorted_by_phantoms = sorted(chromatogram_data, key=lambda d: d['n_phantoms'], reverse=True)
n_worst = 8

fig, axes = plt.subplots(4, 2, figsize=(18, 20))
axes = axes.flatten()

for k in range(min(n_worst, len(sorted_by_phantoms))):
    ax = axes[k]
    d = sorted_by_phantoms[k]

    ax.plot(d['time_axis'], d['signal'], color='#333333', linewidth=0.6,
            label='Signal', zorder=1)

    # True peaks
    for tp in d['true_peaks']:
        ax.axvline(tp['rt'], color='green', alpha=0.15, linewidth=0.8)
    true_rts = [tp['rt'] for tp in d['true_peaks']]
    true_heights = [d['signal'][tp['apex_index']] for tp in d['true_peaks']]
    ax.plot(true_rts, true_heights, '^', color='green', markersize=6,
            label=f'True ({d["n_true"]})', zorder=3)

    # Matched components
    matched_rts = set()
    for m in d['matches']:
        if m.fitted_rt is not None:
            matched_rts.add(round(m.fitted_rt, 6))

    # Fitted components — color by matched vs phantom
    for comp in d['components']:
        is_matched = round(comp.retention_time, 6) in matched_rts
        color = '#1f77b4' if is_matched else '#d62728'
        alpha = 0.4 if is_matched else 0.7
        ax.plot(comp.t_curve, comp.y_curve, linewidth=0.8,
                linestyle='--', alpha=alpha, color=color, zorder=2)

    # Phantom markers
    phantom_rts = [ph['rt'] for ph in d['phantoms']]
    if phantom_rts:
        phantom_heights = []
        for prt in phantom_rts:
            idx = int(np.argmin(np.abs(d['time_axis'] - prt)))
            phantom_heights.append(d['signal'][idx])
        ax.plot(phantom_rts, phantom_heights, 'x', color='red', markersize=8,
                markeredgewidth=2, label=f'Phantom ({d["n_phantoms"]})', zorder=4)

    # Chunk boundaries
    for chunk in d['chunks']:
        t_start = d['time_axis'][chunk.start_index]
        ax.axvline(t_start, color='gray', alpha=0.2, linewidth=0.5, linestyle=':')

    ax.set_title(f'#{d["index"]} — True={d["n_true"]}, Fitted={d["n_fitted"]}, '
                 f'Phantoms={d["n_phantoms"]}, Misses={d["n_misses"]}',
                 fontsize=9, color='#d62728' if d['n_phantoms'] > 10 else 'black')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlabel('RT (min)', fontsize=8)
    ax.set_ylabel('Signal', fontsize=8)

for k in range(min(n_worst, len(sorted_by_phantoms)), len(axes)):
    axes[k].set_visible(False)

fig.suptitle(f'Worst {n_worst} Chromatograms by Phantom Count', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'worst_phantom_chromatograms.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: worst_phantom_chromatograms.png')


# --- 4. Zoomed chunk-level views for the single worst chromatogram ---
worst = sorted_by_phantoms[0]
chunks_with_phantoms = []

phantom_rts_set = set(round(ph['rt'], 4) for ph in worst['phantoms'])

for chunk in worst['chunks']:
    t_start = worst['time_axis'][chunk.start_index]
    t_end = worst['time_axis'][min(chunk.end_index, len(worst['time_axis']) - 1)]

    # Count phantoms in this chunk's time range
    chunk_phantoms = [ph for ph in worst['phantoms']
                      if t_start <= ph['rt'] <= t_end]
    chunk_true = [tp for tp in worst['true_peaks']
                  if t_start <= tp['rt'] <= t_end]
    chunk_fitted = [c for c in worst['components']
                    if t_start <= c.retention_time <= t_end]

    if chunk_phantoms:
        chunks_with_phantoms.append({
            'chunk': chunk,
            't_start': t_start,
            't_end': t_end,
            'n_phantoms': len(chunk_phantoms),
            'n_true': len(chunk_true),
            'n_fitted': len(chunk_fitted),
            'phantoms': chunk_phantoms,
            'true_peaks': chunk_true,
            'fitted': chunk_fitted,
        })

chunks_with_phantoms.sort(key=lambda x: x['n_phantoms'], reverse=True)
n_zoom = min(6, len(chunks_with_phantoms))

if n_zoom > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for k in range(n_zoom):
        ax = axes[k]
        cdata = chunks_with_phantoms[k]
        t_lo, t_hi = cdata['t_start'], cdata['t_end']
        pad = (t_hi - t_lo) * 0.1
        mask = (worst['time_axis'] >= t_lo - pad) & (worst['time_axis'] <= t_hi + pad)

        ax.plot(worst['time_axis'][mask], worst['signal'][mask],
                color='#333333', linewidth=1, label='Signal')

        # Shade chunk region
        ax.axvspan(t_lo, t_hi, alpha=0.08, color='blue')

        # True peaks
        for tp in cdata['true_peaks']:
            ax.axvline(tp['rt'], color='green', alpha=0.3, linewidth=1)
            idx = tp['apex_index']
            ax.plot(tp['rt'], worst['signal'][idx], '^', color='green',
                    markersize=9, zorder=5)

        # Fitted components
        for comp in cdata['fitted']:
            ax.plot(comp.t_curve, comp.y_curve, linewidth=1,
                    linestyle='--', alpha=0.6, color='#1f77b4')

        # Phantoms
        for ph in cdata['phantoms']:
            ax.axvline(ph['rt'], color='red', alpha=0.3, linewidth=1)
            idx = int(np.argmin(np.abs(worst['time_axis'] - ph['rt'])))
            ax.plot(ph['rt'], worst['signal'][idx], 'x', color='red',
                    markersize=10, markeredgewidth=2, zorder=5)

        ax.set_title(f'True={cdata["n_true"]}, Fitted={cdata["n_fitted"]}, '
                     f'Phantoms={cdata["n_phantoms"]}',
                     fontsize=9, color='#d62728')
        ax.set_xlabel('RT (min)', fontsize=8)
        if k == 0:
            ax.legend(fontsize=7)

    for k in range(n_zoom, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f'Zoomed Chunks with Phantoms — Chromatogram #{worst["index"]}',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'worst_chunk_zoom.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: worst_chunk_zoom.png')


# --- 5. Histogram of fitted/true ratio per chunk ---
chunk_ratios = []
for d in chromatogram_data:
    for chunk in d['chunks']:
        t_start = d['time_axis'][chunk.start_index]
        t_end = d['time_axis'][min(chunk.end_index, len(d['time_axis']) - 1)]
        n_true_in_chunk = sum(1 for tp in d['true_peaks'] if t_start <= tp['rt'] <= t_end)
        n_fitted_in_chunk = sum(1 for c in d['components']
                                if t_start <= c.retention_time <= t_end)
        if n_true_in_chunk > 0:
            chunk_ratios.append(n_fitted_in_chunk / n_true_in_chunk)
        elif n_fitted_in_chunk > 0:
            chunk_ratios.append(n_fitted_in_chunk + 0.5)  # "infinite" ratio, cap for viz

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chunk-level ratio histogram
ax = axes[0]
chunk_ratios_arr = np.array(chunk_ratios)
ax.hist(chunk_ratios_arr[chunk_ratios_arr <= 5], bins=50, color='#1f77b4',
        edgecolor='white', alpha=0.8)
ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Fitted / True per chunk')
ax.set_ylabel('Count')
ax.set_title('Per-Chunk Component Ratio')

# Excess components by true peak count
ax = axes[1]
excess_by_true = defaultdict(list)
for d in chromatogram_data:
    excess_by_true[d['n_true']].append(d['n_fitted'] - d['n_true'])

true_counts_sorted = sorted(excess_by_true.keys())
means = [np.mean(excess_by_true[k]) for k in true_counts_sorted]
ax.bar(true_counts_sorted, means, color='#ff7f0e', edgecolor='white', alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('True Peak Count')
ax.set_ylabel('Mean Excess Components')
ax.set_title('Over-prediction vs True Peak Count')

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'chunk_analysis.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: chunk_analysis.png')

print(f'\nAll results saved to {RESULTS_DIR}/')
print('Done.')
