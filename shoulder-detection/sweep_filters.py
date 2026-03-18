"""
Sweep pre-fit and post-fit filter strategies to find the best phantom reduction.

Tests:
  A. Baseline (current): no pre-fit filter, min_prominence=0.01
  B. Pre-fit signal filter: skip apexes where signal < X% of global range
  C. Higher min_prominence: post-fit 0.02, 0.03, 0.05
  D. Pre-fit + post-fit combined
  E. Post-fit area filter: drop components with area < X% of median area
  F. Post-fit deduplication: merge components within 1σ of each other

Patches run_deconvolution_pipeline via monkey-patching to test pre-fit
filtering without modifying production code until we know what works.
"""

import sys, os, json, time, copy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic_chromatogram import SyntheticChromatogramGenerator
from evaluation import match_peaks

# We'll call run_deconvolution_pipeline with different settings
import logic.deconvolution as deconv_mod
from logic.deconvolution import (
    run_deconvolution_pipeline, EMGComponent, DeconvolutionResult,
    _single_emg,
)

RESULTS_DIR = 'eval_results/filter_sweep'
WEIGHTS = 'gc_heatmap_unet_v3.pth'
NUM_CHROMATOGRAMS = 100
SEED = 7777
CHROM_KW = dict(num_peaks=(5, 20), merge_probability=0.5, snr=None, baseline_type='random')

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Generate chromatograms once ──────────────────────────────────────────────
print('Generating chromatograms...')
gen = SyntheticChromatogramGenerator(seed=SEED)
chromatograms = []
for ci in range(NUM_CHROMATOGRAMS):
    result = gen.generate(**CHROM_KW)
    chromatograms.append(result)
print(f'  {len(chromatograms)} chromatograms ready')


def run_and_evaluate(chromatograms, pipeline_kwargs, post_filters=None, label=''):
    """Run pipeline on all chromatograms and return summary metrics."""
    total_true = 0
    total_fitted = 0
    total_matched = 0
    total_misses = 0
    total_phantoms = 0
    all_sigma_ratios = []
    all_area_ratios = []
    total_time = 0
    n_fat = 0

    for ci, result in enumerate(chromatograms):
        time_axis = result['x']
        signal = result['corrected_y']
        true_peaks = result['true_peaks']

        deconv_mod._model_cache.clear()
        deconv_mod._model_instance = None

        t0 = time.time()
        try:
            deconv = run_deconvolution_pipeline(time_axis, signal, **pipeline_kwargs)
            components = list(deconv.components)
        except Exception:
            components = []
        total_time += time.time() - t0

        # Apply post-filters
        if post_filters and components:
            components = apply_post_filters(components, signal, post_filters)

        matches, misses, phantoms = match_peaks(true_peaks, components, max_rt_distance=0.5)

        total_true += len(true_peaks)
        total_fitted += len(components)
        total_matched += len(matches)
        total_misses += len(misses)
        total_phantoms += len(phantoms)

        for m in matches:
            if m.sigma_ratio is not None:
                all_sigma_ratios.append(m.sigma_ratio)
                if m.sigma_ratio > 2.0:
                    n_fat += 1
            if m.area_ratio is not None:
                all_area_ratios.append(m.area_ratio)

    precision = total_matched / total_fitted if total_fitted > 0 else 0
    recall = total_matched / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    sigma_med = float(np.median(all_sigma_ratios)) if all_sigma_ratios else None
    area_med = float(np.median(all_area_ratios)) if all_area_ratios else None

    return {
        'label': label,
        'total_true': total_true,
        'total_fitted': total_fitted,
        'total_matched': total_matched,
        'total_misses': total_misses,
        'total_phantoms': total_phantoms,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_fat': n_fat,
        'sigma_ratio_median': sigma_med,
        'area_ratio_median': area_med,
        'time_s': total_time,
    }


def apply_post_filters(components, signal, filters):
    """Apply post-fit filters to a list of EMGComponents."""
    signal_range = float(np.max(signal) - np.min(signal))

    if 'min_area_frac' in filters:
        areas = [c.area for c in components]
        if areas:
            median_area = float(np.median(areas))
            threshold = filters['min_area_frac'] * median_area
            components = [c for c in components if c.area >= threshold]

    if 'dedup_sigma_factor' in filters:
        components = _deduplicate(components, filters['dedup_sigma_factor'])

    return components


def _deduplicate(components, sigma_factor=1.0):
    """Merge components whose RTs are within sigma_factor * sigma of each other."""
    if len(components) <= 1:
        return components

    components = sorted(components, key=lambda c: c.retention_time)
    kept = [components[0]]

    for c in components[1:]:
        prev = kept[-1]
        separation = abs(c.retention_time - prev.retention_time)
        merge_threshold = sigma_factor * max(c.sigma, prev.sigma)

        if separation < merge_threshold:
            # Keep the one with larger area
            if c.area > prev.area:
                kept[-1] = c
        else:
            kept.append(c)

    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  Define experiment configurations
# ══════════════════════════════════════════════════════════════════════════════

experiments = []

# A: Baseline
experiments.append({
    'label': 'A: Baseline (prom=0.01)',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.01,
        heatmap_threshold=0.15, heatmap_distance=10,
    ),
    'post_filters': None,
})

# B: Pre-fit signal filter via higher min_prominence
for prom in [0.02, 0.03, 0.05, 0.08, 0.10]:
    experiments.append({
        'label': f'B: prom={prom}',
        'pipeline_kwargs': dict(
            weights_path=WEIGHTS, min_prominence=prom,
            heatmap_threshold=0.15, heatmap_distance=10,
        ),
        'post_filters': None,
    })

# C: Higher heatmap_threshold (pre-fit: U-Net needs higher confidence)
for ht in [0.20, 0.25, 0.30, 0.40]:
    experiments.append({
        'label': f'C: heatmap_thresh={ht}',
        'pipeline_kwargs': dict(
            weights_path=WEIGHTS, min_prominence=0.01,
            heatmap_threshold=ht, heatmap_distance=10,
        ),
        'post_filters': None,
    })

# D: Post-fit area filter
for area_frac in [0.01, 0.05, 0.10]:
    experiments.append({
        'label': f'D: area_frac={area_frac}',
        'pipeline_kwargs': dict(
            weights_path=WEIGHTS, min_prominence=0.01,
            heatmap_threshold=0.15, heatmap_distance=10,
        ),
        'post_filters': {'min_area_frac': area_frac},
    })

# E: Post-fit deduplication
for sf in [0.5, 1.0, 1.5]:
    experiments.append({
        'label': f'E: dedup={sf}σ',
        'pipeline_kwargs': dict(
            weights_path=WEIGHTS, min_prominence=0.01,
            heatmap_threshold=0.15, heatmap_distance=10,
        ),
        'post_filters': {'dedup_sigma_factor': sf},
    })

# F: Best combos (we'll add these after seeing initial results)
experiments.append({
    'label': 'F: prom=0.03 + dedup=1σ',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.03,
        heatmap_threshold=0.15, heatmap_distance=10,
    ),
    'post_filters': {'dedup_sigma_factor': 1.0},
})

experiments.append({
    'label': 'F: prom=0.05 + dedup=1σ',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.05,
        heatmap_threshold=0.15, heatmap_distance=10,
    ),
    'post_filters': {'dedup_sigma_factor': 1.0},
})

experiments.append({
    'label': 'F: prom=0.03 + ht=0.25',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.03,
        heatmap_threshold=0.25, heatmap_distance=10,
    ),
    'post_filters': None,
})

experiments.append({
    'label': 'F: prom=0.05 + ht=0.25 + dedup=1σ',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.05,
        heatmap_threshold=0.25, heatmap_distance=10,
    ),
    'post_filters': {'dedup_sigma_factor': 1.0},
})

# G: Pre-fit signal filter (check raw signal at apex before EMG fitting)
for sig_thresh in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
    experiments.append({
        'label': f'G: pre_fit_sig={sig_thresh}',
        'pipeline_kwargs': dict(
            weights_path=WEIGHTS, min_prominence=0.01,
            heatmap_threshold=0.15, heatmap_distance=10,
            pre_fit_signal_threshold=sig_thresh,
        ),
        'post_filters': None,
    })

# H: Best combos with pre-fit signal filter
experiments.append({
    'label': 'H: pre_sig=0.02 + dedup=1σ',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.01,
        heatmap_threshold=0.15, heatmap_distance=10,
        pre_fit_signal_threshold=0.02,
    ),
    'post_filters': {'dedup_sigma_factor': 1.0},
})

experiments.append({
    'label': 'H: pre_sig=0.02 + prom=0.03',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.03,
        heatmap_threshold=0.15, heatmap_distance=10,
        pre_fit_signal_threshold=0.02,
    ),
    'post_filters': None,
})

experiments.append({
    'label': 'H: pre_sig=0.03 + prom=0.03 + dedup=1σ',
    'pipeline_kwargs': dict(
        weights_path=WEIGHTS, min_prominence=0.03,
        heatmap_threshold=0.15, heatmap_distance=10,
        pre_fit_signal_threshold=0.03,
    ),
    'post_filters': {'dedup_sigma_factor': 1.0},
})


# ══════════════════════════════════════════════════════════════════════════════
#  Run all experiments
# ══════════════════════════════════════════════════════════════════════════════

print(f'\nRunning {len(experiments)} experiments on {NUM_CHROMATOGRAMS} chromatograms...\n')

all_results = []
for i, exp in enumerate(experiments):
    print(f'[{i+1}/{len(experiments)}] {exp["label"]}...', end=' ', flush=True)
    r = run_and_evaluate(
        chromatograms,
        exp['pipeline_kwargs'],
        exp.get('post_filters'),
        exp['label'],
    )
    all_results.append(r)
    print(f'F1={r["f1"]:.3f}  P={r["precision"]:.3f}  R={r["recall"]:.3f}  '
          f'phantoms={r["total_phantoms"]}  time={r["time_s"]:.0f}s')

# Save raw results
with open(os.path.join(RESULTS_DIR, 'sweep_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Results table
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 100)
print(f'{"Experiment":<35} {"F1":>6} {"Prec":>6} {"Rec":>6} '
      f'{"Match":>6} {"Miss":>6} {"Phant":>6} {"Fat":>5} {"Time":>5}')
print('-' * 100)

# Sort by F1
sorted_results = sorted(all_results, key=lambda r: r['f1'], reverse=True)
for r in sorted_results:
    print(f'{r["label"]:<35} {r["f1"]:>6.3f} {r["precision"]:>6.3f} {r["recall"]:>6.3f} '
          f'{r["total_matched"]:>6d} {r["total_misses"]:>6d} {r["total_phantoms"]:>6d} '
          f'{r["n_fat"]:>5d} {r["time_s"]:>5.0f}')

print('=' * 100)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization: F1 vs Recall tradeoff
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# F1 bar chart (sorted)
ax = axes[0]
labels_sorted = [r['label'].replace('F: ', '').replace('B: ', '').replace('C: ', '')
                 .replace('D: ', '').replace('E: ', '').replace('A: ', '')
                 for r in sorted_results]
f1s = [r['f1'] for r in sorted_results]
colors = ['#2ca02c' if r['f1'] == max(f1s) else '#1f77b4' for r in sorted_results]
bars = ax.barh(range(len(f1s)), f1s, color=colors, edgecolor='white')
ax.set_yticks(range(len(f1s)))
ax.set_yticklabels(labels_sorted, fontsize=7)
ax.set_xlabel('F1 Score')
ax.set_title('F1 Score by Configuration')
ax.invert_yaxis()

# Precision vs Recall scatter
ax = axes[1]
for r in all_results:
    marker = 'o'
    color = '#1f77b4'
    if r['label'].startswith('A'):
        color = '#d62728'
        marker = 's'
    elif r['label'].startswith('F'):
        color = '#2ca02c'
        marker = 'D'
    ax.scatter(r['recall'], r['precision'], s=60, c=color, marker=marker, alpha=0.7)
    # Label the top performers
    if r['f1'] > sorted_results[3]['f1'] or r['label'].startswith('A'):
        ax.annotate(r['label'].split(':')[1].strip()[:20],
                    (r['recall'], r['precision']),
                    fontsize=6, alpha=0.8,
                    xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision vs Recall Tradeoff')

# Phantoms vs Time scatter
ax = axes[2]
for r in all_results:
    color = '#d62728' if r['label'].startswith('A') else '#1f77b4'
    if r['label'].startswith('F'):
        color = '#2ca02c'
    ax.scatter(r['time_s'], r['total_phantoms'], s=60, c=color, alpha=0.7)
ax.set_xlabel('Runtime (s)')
ax.set_ylabel('Total Phantoms')
ax.set_title('Phantoms vs Computation Time')

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'filter_sweep.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'\nResults saved to {RESULTS_DIR}/')
print('Done.')
