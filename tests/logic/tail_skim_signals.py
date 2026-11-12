"""Synthetic chromatograms for the tail-skim tests.

Every component has sigma = 0.008 min. The sum is noised (0.2 % of the parent's maximum by default) and
smoothed with a Savitzky-Golay filter of about one second, as ChromaKit smooths real data.
"""
import numpy as np
from scipy.integrate import simpson
from scipy.signal import savgol_filter
from scipy.stats import exponnorm, norm

SIGMA = 0.008          # min, every component
PARENT_MU = 0.2        # min, parent apex of the EMG itself
END_SIGMAS = 2.8       # a rider's true end sits this many sigma past its apex
T_MAX = 1.2            # min, length of the trace


def emg(t, area, mu, sigma, tau):
    """Exponentially modified Gaussian with the given area."""
    return area * exponnorm.pdf(t, tau / sigma, loc=mu, scale=sigma)


def _time_axis(dt_s):
    return np.arange(0, T_MAX, dt_s / 60)


def _smooth(x, total, parent, noise, seed):
    rng = np.random.default_rng(seed)
    return savgol_filter(total + rng.normal(0, noise * parent.max(), x.size), 21, 3)


def emg_pair(parent_tau, child_tau, ratio, sep, dt_s=0.05, noise=0.002, seed=0):
    """An EMG parent at 0.2 min and an EMG child `sep` min later, scaled so h_p / h_c = ratio."""
    x = _time_axis(dt_s)
    parent = emg(x, 1.0, PARENT_MU, SIGMA, parent_tau)
    child = emg(x, 1.0, PARENT_MU + sep, SIGMA, child_tau)
    child *= (parent.max() / ratio) / child.max()
    y = _smooth(x, parent + child, parent, noise, seed)
    apex = int(np.argmax(parent))
    child_true_apex = int(np.argmax(child))
    # lowest smoothed point between the two true apexes
    valley = apex + int(np.argmin(y[apex:child_true_apex + 1]))
    child_apex = valley + int(np.argmax(y[valley:valley + 200]))
    return dict(x=x, y=y, parent=parent, child=child,
                true_parent_area=simpson(parent, x=x), true_child_area=simpson(child, x=x),
                apex=apex, child_apex=child_apex, valley=valley,
                true_end=x[child_apex] + END_SIGMAS * SIGMA)


def rider_on_tail(h, d_tau, parent_tau=0.06, dt_s=0.05, noise=0.002, seed=0):
    """A Gaussian rider `d_tau` parent-taus past the parent apex, `h` times the parent tail under it."""
    x = _time_axis(dt_s)
    parent = emg(x, 1.0, PARENT_MU, SIGMA, parent_tau)
    mu = PARENT_MU + d_tau * parent_tau
    height = h * emg(mu, 1.0, PARENT_MU, SIGMA, parent_tau)
    child = height * SIGMA * np.sqrt(2 * np.pi) * norm.pdf(x, mu, SIGMA)
    y = _smooth(x, parent + child, parent, noise, seed)
    child_apex = int(np.searchsorted(x, mu))
    return dict(x=x, y=y, parent=parent, child=child,
                true_parent_area=simpson(parent, x=x), true_child_area=simpson(child, x=x),
                apex=int(np.argmax(parent)), child_apex=child_apex,
                valley=int(np.searchsorted(x, mu - END_SIGMAS * SIGMA)),
                true_end=mu + END_SIGMAS * SIGMA)


def processed_from(pair, extra_peaks=(), rider=None, windows=None):
    """A `processed_data` dict shaped like `ChromatogramProcessor.process()` output.

    Peaks are the parent apex, the child apex and any `extra_peaks` (apex times in min), in time order.
    `corrected_y` is the pair's signal and `baseline_y` is zero.
    """
    x, y = pair['x'], pair['y']
    idx = sorted({pair['apex'], pair['child_apex'],
                  *(int(np.searchsorted(x, t)) for t in extra_peaks)})
    meta = [dict(index=i, x=float(x[i]), y=float(y[i]), is_shoulder=False, type='peak') for i in idx]
    return {
        'x': x, 'original_y': y.copy(), 'smoothed_y': y.copy(),
        'baseline_y': np.zeros_like(y), 'corrected_y': y.copy(),
        'peaks_x': x[idx], 'peaks_y': y[idx], 'peak_metadata': meta,
        'range_filters': [], 'range_excluded_peaks_x': np.array([]),
        'peak_windows': windows, 'outside_windows': 'ignore', 'rider': rider,
    }
