"""
End-to-end evaluation + 3-channel training + comparison script.

Caches all results to eval_results/ as JSON + PNG.

Usage:
    cd shoulder-detection
    python run_evaluation.py
"""

import sys, os, json, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

from unet_model import GCHeatmapUNet, WeightedMSELoss, MultiChannelLoss
from training_data import generate_training_data
from pipeline_viz import train_model, load_model
from evaluation import (
    evaluate_pipeline, print_report,
    plot_error_distributions, plot_failure_breakdown, plot_worst_cases,
)

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = 'eval_results'
WEIGHTS_1CH = 'gc_heatmap_unet_v3.pth'
WEIGHTS_1CH_FALLBACK = 'gc_heatmap_unet_v2.pth'
WEIGHTS_3CH = 'gc_heatmap_unet_v3_3ch.pth'

NUM_TRAIN = 5000
NUM_EVAL = 200
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
EVAL_SEED = 7777

CHROM_KW = dict(
    num_peaks=(5, 20),
    merge_probability=0.5,
    snr=None,
    baseline_type='random',
)
PIPE_KW = dict(min_prominence=0.01)

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(RESULTS_DIR, exist_ok=True)


def report_to_dict(report):
    """Serialize an EvalReport to a JSON-safe dict."""
    d = dict(
        total_true=report.total_true,
        total_fitted=report.total_fitted,
        total_matched=report.total_matched,
        total_misses=report.total_misses,
        total_phantoms=report.total_phantoms,
        precision=report.precision,
        recall=report.recall,
        f1=report.f1,
        n_fat=report.n_fat,
        n_narrow=report.n_narrow,
        n_rt_drift=report.n_rt_drift,
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

    # Per-chromatogram peak count accuracy
    correct = sum(1 for ev in report.evaluations if ev.n_true == ev.n_fitted)
    d['peak_count_exact'] = correct
    d['peak_count_accuracy'] = correct / len(report.evaluations)

    return d


def save_plots(report, prefix):
    """Save evaluation plots as PNGs."""
    fig = plot_error_distributions(report)
    fig.savefig(os.path.join(RESULTS_DIR, f'{prefix}_error_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = plot_failure_breakdown(report)
    fig.savefig(os.path.join(RESULTS_DIR, f'{prefix}_failure_breakdown.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig = plot_worst_cases(report, n=6)
    if fig is not None:
        fig.savefig(os.path.join(RESULTS_DIR, f'{prefix}_worst_cases.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: Baseline evaluation (1-channel model)
# ══════════════════════════════════════════════════════════════════════════════

print('=' * 70)
print('  PHASE 1: Baseline Evaluation (1-channel model)')
print('=' * 70)

weights_1ch = WEIGHTS_1CH if os.path.isfile(WEIGHTS_1CH) else WEIGHTS_1CH_FALLBACK
if not os.path.isfile(weights_1ch):
    print(f'ERROR: No 1-channel weights found at {WEIGHTS_1CH} or {WEIGHTS_1CH_FALLBACK}')
    sys.exit(1)

print(f'Using weights: {weights_1ch}')
print(f'Device: {DEVICE}')
print(f'Evaluating on {NUM_EVAL} synthetic chromatograms...')

t0 = time.time()
report_1ch = evaluate_pipeline(
    num_chromatograms=NUM_EVAL,
    seed=EVAL_SEED,
    weights_path=weights_1ch,
    pipeline_kwargs=PIPE_KW,
    chromatogram_kwargs=CHROM_KW,
    verbose=True,
)
t_eval_1ch = time.time() - t0

print(f'\nBaseline evaluation completed in {t_eval_1ch:.1f}s')
print_report(report_1ch)

results_1ch = report_to_dict(report_1ch)
results_1ch['weights'] = weights_1ch
results_1ch['eval_time_s'] = t_eval_1ch

with open(os.path.join(RESULTS_DIR, 'baseline_1ch.json'), 'w') as f:
    json.dump(results_1ch, f, indent=2)

save_plots(report_1ch, 'baseline_1ch')
print(f'Results saved to {RESULTS_DIR}/baseline_1ch.*')


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Train 3-channel model
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  PHASE 2: Train 3-Channel Model')
print('=' * 70)

if os.path.isfile(WEIGHTS_3CH):
    print(f'3-channel weights already exist at {WEIGHTS_3CH} — skipping training')
    training_history = None
else:
    print(f'Generating 3-channel training data ({NUM_TRAIN} chromatograms)...')
    t0 = time.time()
    X_train, Y_train, meta_train = generate_training_data(
        num_chromatograms=NUM_TRAIN,
        seed=42,
        smoothing_augmentation=True,
        chromatogram_kwargs=CHROM_KW,
        n_channels=3,
    )
    t_datagen = time.time() - t0
    print(f'Data generated in {t_datagen:.1f}s: X={X_train.shape}, Y={Y_train.shape}')

    print(f'\nTraining 3-channel model ({EPOCHS} epochs, batch_size={BATCH_SIZE})...')
    loss_fn = MultiChannelLoss(peak_weight=50.0, param_weight=1.0)
    model_3ch = GCHeatmapUNet(out_channels=3)

    t0 = time.time()
    model_3ch, training_history = train_model(
        X_train, Y_train,
        model=model_3ch,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        loss_fn=loss_fn,
        device=DEVICE,
        save_path=WEIGHTS_3CH,
        verbose=True,
    )
    t_train = time.time() - t0
    print(f'Training completed in {t_train:.1f}s')

    # Save training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(training_history) + 1), training_history,
            'o-', color='#1f77b4', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('3-Channel Model Training Loss')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'training_loss_3ch.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, 'training_3ch.json'), 'w') as f:
        json.dump({
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'num_train_chromatograms': NUM_TRAIN,
            'num_train_windows': len(X_train),
            'datagen_time_s': t_datagen,
            'training_time_s': t_train,
            'loss_history': training_history,
        }, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: Evaluate 3-channel model
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  PHASE 3: Evaluate 3-Channel Model')
print('=' * 70)

# Clear model cache so we load the 3-ch weights
import logic.deconvolution as _deconv
_deconv._model_cache.clear()
_deconv._model_instance = None

print(f'Using weights: {WEIGHTS_3CH}')
print(f'Evaluating on {NUM_EVAL} synthetic chromatograms...')

t0 = time.time()
report_3ch = evaluate_pipeline(
    num_chromatograms=NUM_EVAL,
    seed=EVAL_SEED,
    weights_path=WEIGHTS_3CH,
    pipeline_kwargs=PIPE_KW,
    chromatogram_kwargs=CHROM_KW,
    verbose=True,
)
t_eval_3ch = time.time() - t0

print(f'\n3-channel evaluation completed in {t_eval_3ch:.1f}s')
print_report(report_3ch)

results_3ch = report_to_dict(report_3ch)
results_3ch['weights'] = WEIGHTS_3CH
results_3ch['eval_time_s'] = t_eval_3ch

with open(os.path.join(RESULTS_DIR, 'eval_3ch.json'), 'w') as f:
    json.dump(results_3ch, f, indent=2)

save_plots(report_3ch, 'eval_3ch')
print(f'Results saved to {RESULTS_DIR}/eval_3ch.*')


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Comparison summary
# ══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('  COMPARISON: 1-Channel vs 3-Channel')
print('=' * 70)

def fmt(val, fmt_str='.4f'):
    return f'{val:{fmt_str}}' if val is not None else 'N/A'

comparison = {}
for key in ['precision', 'recall', 'f1', 'total_matched', 'total_misses',
            'total_phantoms', 'n_fat', 'n_narrow', 'n_rt_drift',
            'peak_count_accuracy']:
    v1 = results_1ch.get(key)
    v3 = results_3ch.get(key)
    comparison[key] = {'1ch': v1, '3ch': v3}

for key in ['sigma_ratio_median', 'tau_ratio_median', 'area_ratio_median',
            'rt_error_std']:
    v1 = results_1ch.get(key)
    v3 = results_3ch.get(key)
    comparison[key] = {'1ch': v1, '3ch': v3}

print(f'\n{"Metric":<28} {"1-Channel":>12} {"3-Channel":>12} {"Delta":>12}')
print('-' * 66)
for key, vals in comparison.items():
    v1, v3 = vals['1ch'], vals['3ch']
    if v1 is not None and v3 is not None:
        if isinstance(v1, int):
            delta = v3 - v1
            print(f'{key:<28} {v1:>12d} {v3:>12d} {delta:>+12d}')
        else:
            delta = v3 - v1
            print(f'{key:<28} {v1:>12.4f} {v3:>12.4f} {delta:>+12.4f}')

with open(os.path.join(RESULTS_DIR, 'comparison.json'), 'w') as f:
    json.dump({'baseline_1ch': results_1ch, 'eval_3ch': results_3ch}, f, indent=2)

print(f'\nAll results cached in {RESULTS_DIR}/')
print('Done.')
