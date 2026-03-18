"""Re-run 3-channel evaluation only (after bounds fix)."""
import sys, os, json, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation import (
    evaluate_pipeline, print_report,
    plot_error_distributions, plot_failure_breakdown, plot_worst_cases,
)

RESULTS_DIR = 'eval_results'
WEIGHTS_3CH = 'gc_heatmap_unet_v3_3ch.pth'
WEIGHTS_1CH = 'gc_heatmap_unet_v3.pth'
NUM_EVAL = 200
EVAL_SEED = 7777
CHROM_KW = dict(num_peaks=(5, 20), merge_probability=0.5, snr=None, baseline_type='random')
PIPE_KW = dict(min_prominence=0.01)

os.makedirs(RESULTS_DIR, exist_ok=True)

import logic.deconvolution as _deconv
_deconv._model_cache.clear()
_deconv._model_instance = None

print('=' * 70)
print('  Re-evaluating 3-Channel Model (after bounds fix)')
print('=' * 70)

t0 = time.time()
report_3ch = evaluate_pipeline(
    num_chromatograms=NUM_EVAL, seed=EVAL_SEED,
    weights_path=WEIGHTS_3CH,
    pipeline_kwargs=PIPE_KW, chromatogram_kwargs=CHROM_KW,
    verbose=True,
)
elapsed = time.time() - t0
print(f'\nCompleted in {elapsed:.1f}s')
print_report(report_3ch)

# Count crashes
n_crashes = sum(1 for ev in report_3ch.evaluations if ev.n_fitted == 0)
print(f'\nCrashes (0 fitted): {n_crashes}/200')

# Save
def report_to_dict(report):
    d = dict(
        total_true=report.total_true, total_fitted=report.total_fitted,
        total_matched=report.total_matched, total_misses=report.total_misses,
        total_phantoms=report.total_phantoms, precision=report.precision,
        recall=report.recall, f1=report.f1, n_fat=report.n_fat,
        n_narrow=report.n_narrow, n_rt_drift=report.n_rt_drift,
        n_chromatograms=len(report.evaluations),
    )
    if len(report.rt_errors) > 0:
        d['rt_error_mean'] = float(np.mean(report.rt_errors))
        d['rt_error_median'] = float(np.median(report.rt_errors))
        d['rt_error_std'] = float(np.std(report.rt_errors))
    if len(report.sigma_ratios) > 0:
        d['sigma_ratio_median'] = float(np.median(report.sigma_ratios))
        d['sigma_ratio_p25'] = float(np.percentile(report.sigma_ratios, 25))
        d['sigma_ratio_p75'] = float(np.percentile(report.sigma_ratios, 75))
    if len(report.tau_ratios) > 0:
        d['tau_ratio_median'] = float(np.median(report.tau_ratios))
        d['tau_ratio_p25'] = float(np.percentile(report.tau_ratios, 25))
        d['tau_ratio_p75'] = float(np.percentile(report.tau_ratios, 75))
    if len(report.area_ratios) > 0:
        d['area_ratio_median'] = float(np.median(report.area_ratios))
        d['area_ratio_p25'] = float(np.percentile(report.area_ratios, 25))
        d['area_ratio_p75'] = float(np.percentile(report.area_ratios, 75))
    correct = sum(1 for ev in report.evaluations if ev.n_true == ev.n_fitted)
    d['peak_count_exact'] = correct
    d['peak_count_accuracy'] = correct / len(report.evaluations)
    d['n_crashes'] = n_crashes
    return d

results_3ch = report_to_dict(report_3ch)
with open(os.path.join(RESULTS_DIR, 'eval_3ch_fixed.json'), 'w') as f:
    json.dump(results_3ch, f, indent=2)

fig = plot_error_distributions(report_3ch)
fig.savefig(os.path.join(RESULTS_DIR, 'eval_3ch_fixed_error_distributions.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

fig = plot_failure_breakdown(report_3ch)
fig.savefig(os.path.join(RESULTS_DIR, 'eval_3ch_fixed_failure_breakdown.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Load 1ch baseline for comparison
with open(os.path.join(RESULTS_DIR, 'baseline_1ch.json')) as f:
    results_1ch = json.load(f)

print('\n' + '=' * 70)
print('  COMPARISON: 1-Channel vs 3-Channel (fixed)')
print('=' * 70)
print(f'\n{"Metric":<28} {"1-Channel":>12} {"3-Channel":>12} {"Delta":>12}')
print('-' * 66)
for key in ['precision', 'recall', 'f1', 'total_matched', 'total_misses',
            'total_phantoms', 'n_fat', 'n_narrow', 'n_rt_drift',
            'peak_count_accuracy', 'sigma_ratio_median', 'tau_ratio_median',
            'area_ratio_median', 'rt_error_std']:
    v1 = results_1ch.get(key)
    v3 = results_3ch.get(key)
    if v1 is not None and v3 is not None:
        if isinstance(v1, int):
            print(f'{key:<28} {v1:>12d} {v3:>12d} {v3-v1:>+12d}')
        else:
            print(f'{key:<28} {v1:>12.4f} {v3:>12.4f} {v3-v1:>+12.4f}')

# Save combined comparison
with open(os.path.join(RESULTS_DIR, 'comparison_fixed.json'), 'w') as f:
    json.dump({'baseline_1ch': results_1ch, 'eval_3ch_fixed': results_3ch}, f, indent=2)

print(f'\nResults saved to {RESULTS_DIR}/')
print('Done.')
