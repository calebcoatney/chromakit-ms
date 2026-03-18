"""
Optuna-based hyperparameter search for phantom reduction filters.

Searches over:
  - pre_fit_signal_threshold (NEW: skip noise-level apexes before EMG fitting)
  - min_prominence (post-fit height filter)
  - heatmap_threshold (U-Net confidence cutoff)
  - dedup_sigma_factor (post-fit deduplication)
  - min_area_frac (post-fit area filter)

Optimizes F1 score with a secondary objective to minimize phantoms.
Uses a smaller evaluation set (50 chromatograms) for speed during search,
then validates the best config on the full 200.
"""

import sys, os, json, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic_chromatogram import SyntheticChromatogramGenerator
from evaluation import match_peaks

import logic.deconvolution as deconv_mod
from logic.deconvolution import run_deconvolution_pipeline, _single_emg

RESULTS_DIR = 'eval_results/optuna_search'
WEIGHTS = 'gc_heatmap_unet_v3.pth'
SEARCH_N = 50       # chromatograms for search (fast)
VALIDATE_N = 200    # chromatograms for final validation
SEED = 7777
N_TRIALS = 60
CHROM_KW = dict(num_peaks=(5, 20), merge_probability=0.5, snr=None, baseline_type='random')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Generate chromatograms ───────────────────────────────────────────────────
print('Generating chromatograms...')
gen = SyntheticChromatogramGenerator(seed=SEED)
all_chromatograms = []
for ci in range(VALIDATE_N):
    result = gen.generate(**CHROM_KW)
    all_chromatograms.append(result)

search_chromatograms = all_chromatograms[:SEARCH_N]
print(f'  {len(all_chromatograms)} total, {len(search_chromatograms)} for search')


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
            if c.area > prev.area:
                kept[-1] = c
        else:
            kept.append(c)
    return kept


def evaluate(chromatograms, pipeline_kwargs, dedup_sigma=0, min_area_frac=0):
    """Run pipeline and return metrics dict."""
    total_true = total_fitted = total_matched = total_phantoms = 0

    for result in chromatograms:
        time_axis = result['x']
        signal = result['corrected_y']
        true_peaks = result['true_peaks']

        try:
            deconv = run_deconvolution_pipeline(time_axis, signal, **pipeline_kwargs)
            components = list(deconv.components)
        except Exception:
            components = []

        # Post-fit filters
        if components and dedup_sigma > 0:
            components = _deduplicate(components, dedup_sigma)
        if components and min_area_frac > 0:
            areas = [c.area for c in components]
            if areas:
                threshold = min_area_frac * float(np.median(areas))
                components = [c for c in components if c.area >= threshold]

        matches, misses, phantoms = match_peaks(true_peaks, components, max_rt_distance=0.5)
        total_true += len(true_peaks)
        total_fitted += len(components)
        total_matched += len(matches)
        total_phantoms += len(phantoms)

    precision = total_matched / total_fitted if total_fitted > 0 else 0
    recall = total_matched / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_matched': total_matched,
        'total_phantoms': total_phantoms,
        'total_fitted': total_fitted,
        'total_true': total_true,
    }


def objective(trial):
    """Optuna objective: maximize F1."""
    # Pre-fit signal filter (the key new parameter)
    pre_fit_sig = trial.suggest_float('pre_fit_signal_threshold', 0.0, 0.15)

    # U-Net confidence
    heatmap_thresh = trial.suggest_float('heatmap_threshold', 0.10, 0.50)

    # Post-fit prominence filter
    min_prom = trial.suggest_float('min_prominence', 0.005, 0.15)

    # Post-fit deduplication (0 = disabled)
    dedup_sigma = trial.suggest_float('dedup_sigma_factor', 0.0, 2.0)

    # Post-fit area filter (0 = disabled)
    min_area_frac = trial.suggest_float('min_area_frac', 0.0, 0.15)

    pipeline_kwargs = dict(
        weights_path=WEIGHTS,
        min_prominence=min_prom,
        heatmap_threshold=heatmap_thresh,
        heatmap_distance=10,
        pre_fit_signal_threshold=pre_fit_sig,
    )

    t0 = time.time()
    metrics = evaluate(search_chromatograms, pipeline_kwargs,
                       dedup_sigma=dedup_sigma, min_area_frac=min_area_frac)
    elapsed = time.time() - t0

    trial.set_user_attr('precision', metrics['precision'])
    trial.set_user_attr('recall', metrics['recall'])
    trial.set_user_attr('total_phantoms', metrics['total_phantoms'])
    trial.set_user_attr('total_matched', metrics['total_matched'])
    trial.set_user_attr('time_s', elapsed)

    print(f'  Trial {trial.number:3d}: F1={metrics["f1"]:.3f}  '
          f'P={metrics["precision"]:.3f}  R={metrics["recall"]:.3f}  '
          f'phantoms={metrics["total_phantoms"]}  ({elapsed:.0f}s)')

    return metrics['f1']


# ══════════════════════════════════════════════════════════════════════════════
#  Run Optuna search
# ══════════════════════════════════════════════════════════════════════════════

print(f'\nRunning Optuna search: {N_TRIALS} trials on {SEARCH_N} chromatograms...\n')

study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='filter_optimization',
)

# Seed with baseline config so Optuna has a reference point
study.enqueue_trial({
    'pre_fit_signal_threshold': 0.0,
    'heatmap_threshold': 0.15,
    'min_prominence': 0.01,
    'dedup_sigma_factor': 0.0,
    'min_area_frac': 0.0,
})

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best = study.best_trial
print(f'\n{"="*70}')
print(f'BEST TRIAL: #{best.number}  F1={best.value:.4f}')
print(f'{"="*70}')
print('Parameters:')
for k, v in best.params.items():
    print(f'  {k}: {v:.4f}')
print(f'Precision: {best.user_attrs["precision"]:.4f}')
print(f'Recall:    {best.user_attrs["recall"]:.4f}')
print(f'Phantoms:  {best.user_attrs["total_phantoms"]}')


# ══════════════════════════════════════════════════════════════════════════════
#  Validate best config on full dataset
# ══════════════════════════════════════════════════════════════════════════════

print(f'\nValidating best config on {VALIDATE_N} chromatograms...')
best_pipeline = dict(
    weights_path=WEIGHTS,
    min_prominence=best.params['min_prominence'],
    heatmap_threshold=best.params['heatmap_threshold'],
    heatmap_distance=10,
    pre_fit_signal_threshold=best.params['pre_fit_signal_threshold'],
)
best_dedup = best.params['dedup_sigma_factor']
best_area = best.params['min_area_frac']

t0 = time.time()
val_metrics = evaluate(all_chromatograms, best_pipeline,
                       dedup_sigma=best_dedup, min_area_frac=best_area)
val_time = time.time() - t0

print(f'\nValidation results ({VALIDATE_N} chromatograms, {val_time:.0f}s):')
print(f'  F1:        {val_metrics["f1"]:.4f}')
print(f'  Precision: {val_metrics["precision"]:.4f}')
print(f'  Recall:    {val_metrics["recall"]:.4f}')
print(f'  Matched:   {val_metrics["total_matched"]}/{val_metrics["total_true"]}')
print(f'  Phantoms:  {val_metrics["total_phantoms"]}')

# Also validate baseline for comparison
print(f'\nBaseline validation...')
t0 = time.time()
baseline_metrics = evaluate(all_chromatograms, dict(
    weights_path=WEIGHTS, min_prominence=0.01,
    heatmap_threshold=0.15, heatmap_distance=10,
    pre_fit_signal_threshold=0,
))
baseline_time = time.time() - t0

print(f'Baseline ({baseline_time:.0f}s):')
print(f'  F1:        {baseline_metrics["f1"]:.4f}')
print(f'  Precision: {baseline_metrics["precision"]:.4f}')
print(f'  Recall:    {baseline_metrics["recall"]:.4f}')
print(f'  Matched:   {baseline_metrics["total_matched"]}/{baseline_metrics["total_true"]}')
print(f'  Phantoms:  {baseline_metrics["total_phantoms"]}')


# ══════════════════════════════════════════════════════════════════════════════
#  Save results
# ══════════════════════════════════════════════════════════════════════════════

results = {
    'best_params': best.params,
    'best_f1_search': best.value,
    'validation': val_metrics,
    'baseline': baseline_metrics,
    'n_trials': N_TRIALS,
    'search_n': SEARCH_N,
    'validate_n': VALIDATE_N,
    'all_trials': [
        {
            'number': t.number,
            'params': t.params,
            'f1': t.value,
            'precision': t.user_attrs.get('precision'),
            'recall': t.user_attrs.get('recall'),
            'phantoms': t.user_attrs.get('total_phantoms'),
        }
        for t in study.trials
    ],
}

with open(os.path.join(RESULTS_DIR, 'optuna_results.json'), 'w') as f:
    json.dump(results, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. F1 over trials
ax = axes[0, 0]
f1s = [t.value for t in study.trials]
ax.plot(f1s, 'o-', markersize=3, alpha=0.7)
ax.axhline(best.value, color='green', linestyle='--', alpha=0.5, label=f'Best: {best.value:.3f}')
ax.axhline(baseline_metrics['f1'], color='red', linestyle='--', alpha=0.5, label=f'Baseline: {baseline_metrics["f1"]:.3f}')
ax.set_xlabel('Trial')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Convergence')
ax.legend(fontsize=8)

# 2. Parameter importance (simple: correlation with F1)
ax = axes[0, 1]
param_names = list(best.params.keys())
correlations = []
for pname in param_names:
    vals = [t.params[pname] for t in study.trials]
    f1_vals = [t.value for t in study.trials]
    corr = float(np.corrcoef(vals, f1_vals)[0, 1]) if len(set(vals)) > 1 else 0
    correlations.append(abs(corr))
sorted_idx = np.argsort(correlations)[::-1]
ax.barh([param_names[i] for i in sorted_idx],
        [correlations[i] for i in sorted_idx],
        color='#1f77b4')
ax.set_xlabel('|Correlation with F1|')
ax.set_title('Parameter Importance')

# 3. Precision vs Recall scatter
ax = axes[0, 2]
precs = [t.user_attrs.get('precision', 0) for t in study.trials]
recs = [t.user_attrs.get('recall', 0) for t in study.trials]
sc = ax.scatter(recs, precs, c=f1s, cmap='viridis', s=30, alpha=0.7)
ax.scatter(baseline_metrics['recall'], baseline_metrics['precision'],
           c='red', s=100, marker='s', zorder=5, label='Baseline')
ax.scatter(val_metrics['recall'], val_metrics['precision'],
           c='green', s=100, marker='*', zorder=5, label='Best')
plt.colorbar(sc, ax=ax, label='F1')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision vs Recall')
ax.legend(fontsize=8)

# 4-6. Parameter vs F1 for top 3 params
for pi, idx in enumerate(sorted_idx[:3]):
    ax = axes[1, pi]
    pname = param_names[idx]
    vals = [t.params[pname] for t in study.trials]
    ax.scatter(vals, f1s, s=20, alpha=0.6)
    ax.axvline(best.params[pname], color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel(pname)
    ax.set_ylabel('F1')
    ax.set_title(f'{pname} vs F1')

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'optuna_search.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

print(f'\nResults saved to {RESULTS_DIR}/')
print('Done.')
