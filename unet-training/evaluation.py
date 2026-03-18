"""
End-to-end evaluation pipeline for the deconvolution system.

Runs the production run_deconvolution_pipeline() on synthetic chromatograms,
matches fitted EMG components to ground-truth peaks via Hungarian matching,
and computes per-peak quality metrics.

Usage:
    from evaluation import evaluate_pipeline, print_report, plot_error_distributions
    report = evaluate_pipeline(200, weights_path='gc_heatmap_unet_v3.pth')
    print_report(report)
    plot_error_distributions(report)
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath('..'))

from synthetic_chromatogram import SyntheticChromatogramGenerator
from logic.deconvolution import run_deconvolution_pipeline, EMGComponent, DeconvComponent


# =========================================================================== #
#  Data classes
# =========================================================================== #

@dataclass
class MatchedPeak:
    """One true peak paired with its fitted component (or None for misses)."""
    true_rt: float
    true_sigma: float
    true_tau: float
    true_area: float
    fitted_rt: float = None
    fitted_sigma: float = None
    fitted_tau: float = None
    fitted_area: float = None
    rt_error: float = None       # fitted - true
    sigma_ratio: float = None    # fitted / true
    tau_ratio: float = None      # fitted / true
    area_ratio: float = None     # fitted / true
    is_matched: bool = False


@dataclass
class ChromatogramEval:
    """Evaluation result for one synthetic chromatogram."""
    chromatogram_index: int
    n_true: int
    n_fitted: int
    matches: list = field(default_factory=list)     # list of MatchedPeak (matched)
    misses: list = field(default_factory=list)       # list of MatchedPeak (unmatched true)
    phantoms: list = field(default_factory=list)     # list of dicts (unmatched fitted)
    seed: int = None
    # Cached for visualization
    time_axis: np.ndarray = field(default=None, repr=False)
    signal: np.ndarray = field(default=None, repr=False)
    true_peaks: list = field(default=None, repr=False)
    components: list = field(default=None, repr=False)


@dataclass
class EvalReport:
    """Aggregate metrics across N chromatograms."""
    evaluations: list = field(default_factory=list)  # list of ChromatogramEval
    # Aggregate
    total_true: int = 0
    total_fitted: int = 0
    total_matched: int = 0
    total_misses: int = 0
    total_phantoms: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Per-match arrays (for histograms)
    rt_errors: np.ndarray = field(default=None, repr=False)
    sigma_ratios: np.ndarray = field(default=None, repr=False)
    tau_ratios: np.ndarray = field(default=None, repr=False)
    area_ratios: np.ndarray = field(default=None, repr=False)
    # Failure classification
    n_fat: int = 0        # sigma_ratio > 2
    n_narrow: int = 0     # sigma_ratio < 0.5
    n_rt_drift: int = 0   # |rt_error| > 0.1 min


# =========================================================================== #
#  Core: Hungarian matching
# =========================================================================== #

def match_peaks(true_peaks, components, max_rt_distance=0.5):
    """
    Match fitted EMG components to ground-truth peaks using the
    Hungarian algorithm on RT distance.

    Parameters
    ----------
    true_peaks : list of dicts
        Ground truth from SyntheticChromatogramGenerator (keys: rt, sigma, tau, area).
    components : list of EMGComponent
        From run_deconvolution_pipeline().
    max_rt_distance : float
        Maximum RT distance (minutes) for a valid match.

    Returns
    -------
    matches : list of MatchedPeak (paired)
    misses : list of MatchedPeak (unmatched true peaks)
    phantoms : list of dicts (unmatched fitted components)
    """
    n_true = len(true_peaks)
    n_fitted = len(components)

    if n_true == 0 and n_fitted == 0:
        return [], [], []
    if n_true == 0:
        phantoms = [{'rt': c.retention_time,
                     'sigma': getattr(c, 'sigma', None),
                     'tau': getattr(c, 'tau', None),
                     'area': c.area} for c in components]
        return [], [], phantoms
    if n_fitted == 0:
        misses = [MatchedPeak(true_rt=p['rt'], true_sigma=p['sigma'],
                              true_tau=p['tau'], true_area=p['area'])
                  for p in true_peaks]
        return [], misses, []

    # Build cost matrix: RT distance
    cost = np.zeros((n_true, n_fitted))
    for i, tp in enumerate(true_peaks):
        for j, comp in enumerate(components):
            cost[i, j] = abs(tp['rt'] - comp.retention_time)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_true = set()
    matched_fitted = set()
    matches = []

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] <= max_rt_distance:
            tp = true_peaks[i]
            comp = components[j]
            comp_sigma = getattr(comp, 'sigma', None)
            comp_tau = getattr(comp, 'tau', None)
            sigma_ratio = (comp_sigma / tp['sigma']
                           if comp_sigma is not None and tp['sigma'] > 1e-8
                           else None)
            tau_ratio = (comp_tau / tp['tau']
                         if comp_tau is not None and tp['tau'] > 1e-8
                         else None)
            area_ratio = comp.area / tp['area'] if tp['area'] > 1e-8 else None

            matches.append(MatchedPeak(
                true_rt=tp['rt'],
                true_sigma=tp['sigma'],
                true_tau=tp['tau'],
                true_area=tp['area'],
                fitted_rt=comp.retention_time,
                fitted_sigma=comp_sigma,
                fitted_tau=comp_tau,
                fitted_area=comp.area,
                rt_error=comp.retention_time - tp['rt'],
                sigma_ratio=sigma_ratio,
                tau_ratio=tau_ratio,
                area_ratio=area_ratio,
                is_matched=True,
            ))
            matched_true.add(i)
            matched_fitted.add(j)

    misses = [MatchedPeak(true_rt=true_peaks[i]['rt'],
                          true_sigma=true_peaks[i]['sigma'],
                          true_tau=true_peaks[i]['tau'],
                          true_area=true_peaks[i]['area'])
              for i in range(n_true) if i not in matched_true]

    phantoms = [{'rt': components[j].retention_time,
                 'sigma': getattr(components[j], 'sigma', None),
                 'tau': getattr(components[j], 'tau', None),
                 'area': components[j].area}
                for j in range(n_fitted) if j not in matched_fitted]

    return matches, misses, phantoms


# =========================================================================== #
#  Pipeline evaluation
# =========================================================================== #

def evaluate_pipeline(num_chromatograms=200, seed=7777,
                      weights_path=None, max_rt_distance=0.5,
                      splitting_method='emg',
                      pipeline_kwargs=None, chromatogram_kwargs=None,
                      verbose=True):
    """
    Run the production deconvolution pipeline on N synthetic chromatograms
    and compute evaluation metrics.

    Parameters
    ----------
    num_chromatograms : int
    seed : int
    weights_path : str or None
    max_rt_distance : float
    splitting_method : str
        'emg' or 'geometric' — passed to run_deconvolution_pipeline().
    pipeline_kwargs : dict or None
        Extra kwargs for run_deconvolution_pipeline().
    chromatogram_kwargs : dict or None
        Override kwargs for SyntheticChromatogramGenerator.generate().
    verbose : bool

    Returns
    -------
    EvalReport
    """
    gen = SyntheticChromatogramGenerator(seed=seed)

    chrom_kw = dict(
        num_peaks=(5, 20),
        merge_probability=0.5,
        snr=None,
        baseline_type='random',
    )
    if chromatogram_kwargs:
        chrom_kw.update(chromatogram_kwargs)

    pipe_kw = dict(
        splitting_method=splitting_method,
        heatmap_distance=10,
    )
    if pipeline_kwargs:
        pipe_kw.update(pipeline_kwargs)
    if weights_path:
        pipe_kw['weights_path'] = weights_path

    evaluations = []

    for ci in range(num_chromatograms):
        result = gen.generate(**chrom_kw)
        time_axis = result['x']
        signal = result['corrected_y']
        true_peaks = result['true_peaks']

        # On first iteration, clear model cache so weights_path is respected
        if ci == 0:
            import logic.deconvolution as _deconv
            _deconv._model_cache.clear()
            _deconv._model_instance = None

        try:
            deconv = run_deconvolution_pipeline(time_axis, signal, **pipe_kw)
            components = deconv.components
        except Exception as e:
            if verbose:
                print(f"  Chromatogram {ci}: pipeline error: {e}")
            components = []

        matches, misses, phantoms = match_peaks(
            true_peaks, components, max_rt_distance
        )

        ev = ChromatogramEval(
            chromatogram_index=ci,
            n_true=len(true_peaks),
            n_fitted=len(components),
            matches=matches,
            misses=misses,
            phantoms=phantoms,
            seed=seed,
            time_axis=time_axis,
            signal=signal,
            true_peaks=true_peaks,
            components=components,
        )
        evaluations.append(ev)

        if verbose and (ci + 1) % 20 == 0:
            print(f"  Evaluated {ci + 1}/{num_chromatograms} chromatograms")

    # Aggregate
    report = _build_report(evaluations)
    return report


def _build_report(evaluations):
    """Aggregate per-chromatogram evaluations into an EvalReport."""
    all_rt_errors = []
    all_sigma_ratios = []
    all_tau_ratios = []
    all_area_ratios = []
    total_true = 0
    total_fitted = 0
    total_matched = 0
    total_misses = 0
    total_phantoms = 0
    n_fat = 0
    n_narrow = 0
    n_rt_drift = 0

    for ev in evaluations:
        total_true += ev.n_true
        total_fitted += ev.n_fitted
        total_matched += len(ev.matches)
        total_misses += len(ev.misses)
        total_phantoms += len(ev.phantoms)

        for m in ev.matches:
            if m.rt_error is not None:
                all_rt_errors.append(m.rt_error)
            if m.sigma_ratio is not None:
                all_sigma_ratios.append(m.sigma_ratio)
                if m.sigma_ratio > 2.0:
                    n_fat += 1
                elif m.sigma_ratio < 0.5:
                    n_narrow += 1
            if m.tau_ratio is not None:
                all_tau_ratios.append(m.tau_ratio)
            if m.area_ratio is not None:
                all_area_ratios.append(m.area_ratio)
            if m.rt_error is not None and abs(m.rt_error) > 0.1:
                n_rt_drift += 1

    precision = total_matched / total_fitted if total_fitted > 0 else 0.0
    recall = total_matched / total_true if total_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return EvalReport(
        evaluations=evaluations,
        total_true=total_true,
        total_fitted=total_fitted,
        total_matched=total_matched,
        total_misses=total_misses,
        total_phantoms=total_phantoms,
        precision=precision,
        recall=recall,
        f1=f1,
        rt_errors=np.array(all_rt_errors) if all_rt_errors else np.array([]),
        sigma_ratios=np.array(all_sigma_ratios) if all_sigma_ratios else np.array([]),
        tau_ratios=np.array(all_tau_ratios) if all_tau_ratios else np.array([]),
        area_ratios=np.array(all_area_ratios) if all_area_ratios else np.array([]),
        n_fat=n_fat,
        n_narrow=n_narrow,
        n_rt_drift=n_rt_drift,
    )


# =========================================================================== #
#  Reporting
# =========================================================================== #

def print_report(report):
    """Print a summary of the evaluation report."""
    print('=' * 60)
    print('  DECONVOLUTION PIPELINE EVALUATION')
    print('=' * 60)
    print(f'  Chromatograms evaluated: {len(report.evaluations)}')
    print(f'  Total true peaks:       {report.total_true}')
    print(f'  Total fitted components:{report.total_fitted}')
    print()
    print(f'  Matched:    {report.total_matched:5d}')
    print(f'  Misses:     {report.total_misses:5d}  (true peaks not found)')
    print(f'  Phantoms:   {report.total_phantoms:5d}  (false positives)')
    print()
    print(f'  Precision:  {report.precision:.4f}')
    print(f'  Recall:     {report.recall:.4f}')
    print(f'  F1 Score:   {report.f1:.4f}')
    print()

    if len(report.rt_errors) > 0:
        print('  RT Error (min):')
        print(f'    Mean:   {np.mean(report.rt_errors):+.4f}')
        print(f'    Median: {np.median(report.rt_errors):+.4f}')
        print(f'    Std:    {np.std(report.rt_errors):.4f}')
        print(f'    |err| > 0.1 min: {report.n_rt_drift}')
        print()

    if len(report.sigma_ratios) > 0:
        print(f'  Sigma ratio (fitted/true):')
        print(f'    Median: {np.median(report.sigma_ratios):.3f}')
        print(f'    IQR:    [{np.percentile(report.sigma_ratios, 25):.3f}, '
              f'{np.percentile(report.sigma_ratios, 75):.3f}]')
        print(f'    Fat (>2x):    {report.n_fat}')
        print(f'    Narrow (<0.5x): {report.n_narrow}')
        print()

    if len(report.tau_ratios) > 0:
        print(f'  Tau ratio (fitted/true):')
        print(f'    Median: {np.median(report.tau_ratios):.3f}')
        print(f'    IQR:    [{np.percentile(report.tau_ratios, 25):.3f}, '
              f'{np.percentile(report.tau_ratios, 75):.3f}]')
        print()

    if len(report.area_ratios) > 0:
        print(f'  Area ratio (fitted/true):')
        print(f'    Median: {np.median(report.area_ratios):.3f}')
        print(f'    IQR:    [{np.percentile(report.area_ratios, 25):.3f}, '
              f'{np.percentile(report.area_ratios, 75):.3f}]')

    print('=' * 60)

    # Per-chromatogram peak count accuracy
    correct = sum(1 for ev in report.evaluations
                  if ev.n_true == ev.n_fitted)
    over = sum(1 for ev in report.evaluations
               if ev.n_fitted > ev.n_true)
    under = sum(1 for ev in report.evaluations
                if ev.n_fitted < ev.n_true)
    n = len(report.evaluations)
    print(f'\n  Peak count accuracy: {correct}/{n} exact '
          f'({100*correct/n:.1f}%)')
    print(f'  Over-predicted: {over}  Under-predicted: {under}')


# =========================================================================== #
#  Visualization
# =========================================================================== #

def plot_error_distributions(report, engine='matplotlib'):
    """4-panel histogram: RT error, sigma ratio, tau ratio, area ratio."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # RT error
    ax = axes[0, 0]
    if len(report.rt_errors) > 0:
        ax.hist(report.rt_errors, bins=50, color='#1f77b4', edgecolor='white',
                alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('RT Error (min)')
    ax.set_ylabel('Count')
    ax.set_title('RT Error Distribution')

    # Sigma ratio
    ax = axes[0, 1]
    if len(report.sigma_ratios) > 0:
        ax.hist(report.sigma_ratios, bins=50, color='#ff7f0e', edgecolor='white',
                alpha=0.8, range=(0, min(5, report.sigma_ratios.max() + 0.5)))
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1)
        ax.axvline(0.5, color='orange', linestyle=':', linewidth=1, label='narrow')
        ax.axvline(2.0, color='orange', linestyle=':', linewidth=1, label='fat')
    ax.set_xlabel('Sigma Ratio (fitted/true)')
    ax.set_ylabel('Count')
    ax.set_title('Sigma Ratio Distribution')
    ax.legend(fontsize=8)

    # Tau ratio
    ax = axes[1, 0]
    if len(report.tau_ratios) > 0:
        ax.hist(report.tau_ratios, bins=50, color='#2ca02c', edgecolor='white',
                alpha=0.8, range=(0, min(5, report.tau_ratios.max() + 0.5)))
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Tau Ratio (fitted/true)')
    ax.set_ylabel('Count')
    ax.set_title('Tau Ratio Distribution')

    # Area ratio
    ax = axes[1, 1]
    if len(report.area_ratios) > 0:
        ax.hist(report.area_ratios, bins=50, color='#d62728', edgecolor='white',
                alpha=0.8, range=(0, min(5, report.area_ratios.max() + 0.5)))
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Area Ratio (fitted/true)')
    ax.set_ylabel('Count')
    ax.set_title('Area Ratio Distribution')

    fig.suptitle(f'Error Distributions (N={len(report.evaluations)} chromatograms, '
                 f'{report.total_matched} matched peaks)', fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig


def plot_failure_breakdown(report, engine='matplotlib'):
    """Stacked bar chart of failure modes."""
    import matplotlib.pyplot as plt

    categories = ['Correct\nMatches', 'Fat\n(sigma>2x)', 'Narrow\n(sigma<0.5x)',
                  'RT Drift\n(>0.1 min)', 'Misses', 'Phantoms']
    n_correct = report.total_matched - report.n_fat - report.n_narrow - report.n_rt_drift
    # Avoid double-counting: a match can be both fat AND rt_drift
    # For simplicity, show raw counts (they may overlap)
    values = [max(0, n_correct), report.n_fat, report.n_narrow,
              report.n_rt_drift, report.total_misses, report.total_phantoms]
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728', '#7f7f7f', '#e377c2']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor='white')

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Count')
    ax.set_title(f'Failure Breakdown — {report.total_true} true peaks, '
                 f'{report.total_fitted} fitted')
    fig.tight_layout()
    plt.show()
    return fig


def plot_worst_cases(report, n=6, engine='matplotlib'):
    """Visualize the N worst chromatograms (by number of errors)."""
    import matplotlib.pyplot as plt

    # Score each chromatogram by total errors
    scored = []
    for ev in report.evaluations:
        n_errors = len(ev.misses) + len(ev.phantoms)
        for m in ev.matches:
            if m.sigma_ratio is not None and (m.sigma_ratio > 2 or m.sigma_ratio < 0.5):
                n_errors += 1
            if m.rt_error is not None and abs(m.rt_error) > 0.1:
                n_errors += 1
        scored.append((n_errors, ev))

    scored.sort(key=lambda x: x[0], reverse=True)
    worst = scored[:n]

    if not worst:
        print("No chromatograms to display.")
        return None

    ncols = min(3, len(worst))
    nrows = (len(worst) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)

    for k, (n_errors, ev) in enumerate(worst):
        ax = axes[k // ncols][k % ncols]

        if ev.time_axis is not None and ev.signal is not None:
            ax.plot(ev.time_axis, ev.signal, color='#333333', linewidth=0.8,
                    label='Signal')

        # True peak markers
        if ev.true_peaks:
            for tp in ev.true_peaks:
                ax.axvline(tp['rt'], color='green', alpha=0.3, linewidth=0.8)
            true_rts = [tp['rt'] for tp in ev.true_peaks]
            true_heights = [ev.signal[tp['apex_index']] for tp in ev.true_peaks]
            ax.plot(true_rts, true_heights, '^', color='green', markersize=7,
                    label=f'True ({ev.n_true})')

        # Fitted components
        if ev.components:
            for comp in ev.components:
                if hasattr(comp, 't_curve'):
                    # EMGComponent
                    ax.plot(comp.t_curve, comp.y_curve, linewidth=0.8,
                            linestyle='--', alpha=0.7)
                elif hasattr(comp, 'start_index') and ev.time_axis is not None:
                    # DeconvComponent
                    si = max(0, comp.start_index)
                    ei = min(len(ev.time_axis) - 1, comp.end_index)
                    ax.fill_between(
                        ev.time_axis[si:ei + 1], 0,
                        ev.signal[si:ei + 1],
                        alpha=0.2,
                    )
            fitted_rts = [c.retention_time for c in ev.components]
            fitted_heights = [c.peak_height for c in ev.components]
            ax.plot(fitted_rts, fitted_heights, 'rx', markersize=8,
                    markeredgewidth=2, label=f'Fitted ({ev.n_fitted})')

        ax.set_title(f'#{ev.chromatogram_index} — {n_errors} errors '
                     f'(T={ev.n_true}, F={ev.n_fitted})',
                     fontsize=9, color='#d62728')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('RT (min)', fontsize=8)

    # Hide unused subplots
    for k in range(len(worst), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(f'Worst {len(worst)} Chromatograms', fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig
