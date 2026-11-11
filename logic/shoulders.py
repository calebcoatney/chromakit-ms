"""Shoulder and rider detection: where a detected peak hides a second one.

Two kinds of second peak are found:
- **shoulder**: merged with its parent, no valley between them, still clearly two;
- **rider**: a small peak on the long decaying tail of a big one.

Method (scored in ``tools/shoulder_eval.py``):

1. **Candidates from curvature.** Each detected peak's d2y is taken with a
   Savitzky-Golay window of ~0.6x that peak's own FWHM, so narrow and broad
   peaks are treated alike. Every d2y dip other than the parent's own, whose
   prominence clears ``k_prom`` x the d2y noise, is a candidate. The test is
   prominence rather than d2y < 0, because a shoulder on its parent's convex
   flank often has d2y >= 0 at its centre.
2. **Each candidate must earn its place in a fit.** The parent (plus any
   detected neighbour inside its bases) is fitted with EMGs, then refitted
   with the candidate as one more EMG. The candidate is kept only if the
   extra component is significant (F-test against the larger model's own
   residual, ``alpha``) and stands ``min_snr`` x noise tall. Curvature alone
   cannot separate a real shoulder from the dip between a tail and a rider's
   curvature lobe; the fit can.
3. **Split from the fit.** A shoulder splits from its parent where the two
   fitted components cross; a rider starts where its component first reaches
   2 % of its height.

Pure numpy/scipy; no Qt. The processor calls :func:`detect_shoulders`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_coeffs, savgol_filter
from scipy.special import erfc, erfcx
from scipy.stats import f as f_dist

#: Candidate threshold (x d2y-dip noise) at sensitivity 1 and at 10.
K_PROM_RANGE = (6.0, 2.0)
DEFAULT_SENSITIVITY = 5
#: A peak's extent for candidates and fitting, in its FWHMs (before, after the apex).
EXTENT_FWHM = (5.0, 10.0)
#: Detected neighbours fitted alongside a peak, nearest first.
MAX_NEIGHBOURS = 4


@dataclass(frozen=True)
class EMG:
    """Exponentially modified Gaussian, parameterised by area."""
    area: float
    mu: float
    sigma: float
    tau: float

    def __call__(self, t):
        return emg(np.asarray(t, dtype=float), self.area, self.mu, self.sigma, self.tau)


@dataclass(frozen=True)
class ShoulderCall:
    """A second peak found on the flank or tail of a detected peak.

    Indices are into the ``x`` passed to :func:`detect_shoulders`.
    ``split_index`` is where the child's integration should start on the
    parent's side: the fitted components' crossover for a shoulder, the
    rider's start for a rider. ``parent`` and ``child`` are the fitted
    components, e.g. for a skim baseline under a rider.
    """
    parent_index: int
    apex_index: int
    split_index: int
    outer_index: int          # where the child's fitted component falls to 2 % on the far side
    kind: str                 # 'shoulder' | 'rider'
    side: str                 # 'before' | 'after' the parent
    parent: EMG
    child: EMG
    f_stat: float


def k_prom_for(sensitivity: float) -> float:
    """Sensitivity 1..10 -> candidate threshold, 1 strictest."""
    s = float(np.clip(sensitivity, 1, 10))
    hi, lo = K_PROM_RANGE
    return hi - (s - 1) * (hi - lo) / 9.0


def emg(t, area, mu, sigma, tau):
    """Area-normalised EMG. Stable on both sides: the erfcx form where z >= 0,
    the exp*erfc form where z < 0 (there erfcx(z) overflows to inf*0)."""
    u = (t - mu) / sigma
    z = (sigma / tau - u) / np.sqrt(2)
    out = np.empty_like(u)
    pos = z >= 0
    out[pos] = np.exp(-0.5 * u[pos] ** 2) * erfcx(z[pos])
    neg = ~pos
    out[neg] = np.exp(0.5 * (sigma / tau) ** 2 - (t[neg] - mu) / tau) * erfc(z[neg])
    return area / (2 * tau) * out


def _multi_emg(t, *p):
    out = np.zeros_like(t)
    for i in range(0, len(p), 4):
        out += emg(t, *p[i:i + 4])
    return out


def noise_sd(y) -> float:
    """White-noise SD from the second difference (robust to peaks)."""
    d2 = np.diff(np.asarray(y, dtype=float), 2)
    d2 = d2[np.isfinite(d2)]
    if len(d2) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6))


def _tau_guess(t, y, mu, h, w):
    """EMG tau from the peak's asymmetry at 10 % height (tail/front - 1 ~ tau/sigma)."""
    above = np.flatnonzero(y >= 0.1 * h)
    if len(above) < 3 or mu - t[above[0]] <= 0:
        return 0.3 * w / 2.355
    ratio = (t[above[-1]] - mu) / (mu - t[above[0]])
    return float(np.clip(ratio - 1.0, 0.05, 15.0)) * w / 2.355


def _fit(t, y, seeds):
    """Least-squares fit of one EMG per seed.

    seeds: (apex_rt, height, fwhm, mu_slack, is_peak). Fitting happens in units
    where the first seed's FWHM and height are 1, so every parameter is O(1);
    in raw minutes and counts the solver stalls. A detected peak's mu may sit
    well left of its apex when it tails, so its lower bound gets 3x the slack.
    Returns (params in original units, rss), or (None, inf) if the fit failed.
    """
    t0, ws, hs = seeds[0][0], seeds[0][2], max(seeds[0][1], 1e-12)
    tt, yy = (t - t0) / ws, y / hs
    p0, lo, hi = [], [], []
    for mu, h, w, slack, is_peak in seeds:
        mu_n, w_n, h_n, sl_n = (mu - t0) / ws, w / ws, h / hs, slack / ws
        s = w_n / 2.355
        tau = _tau_guess(tt, yy, mu_n, h_n, w_n) if is_peak else 0.3 * s
        tau = min(max(tau, 0.02 * s), 19 * s)
        p0 += [h_n * w_n * 1.064, mu_n, s, tau]
        lo += [0.0, mu_n - (3 * sl_n if is_peak else sl_n), 0.1 * s, 0.01 * s]
        hi += [np.inf, mu_n + sl_n, 1.5 * s, 20 * s]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, _ = curve_fit(_multi_emg, tt, yy, p0=p0, bounds=(lo, hi), x_scale='jac', max_nfev=4000)
    except (RuntimeError, ValueError):
        return None, np.inf
    rss = float(np.sum((yy - _multi_emg(tt, *p)) ** 2)) * hs ** 2
    out = np.array(p, dtype=float)
    out[0::4] *= hs * ws
    out[1::4] = out[1::4] * ws + t0
    out[2::4] *= ws
    out[3::4] *= ws
    return out, rss


def _classify(parent: EMG, child: EMG, parent_apex_rt, child_apex_rt, rider_height_ratio):
    """'rider' when the child rides the parent's tail: after the parent, on
    its convex tail (the parent's own d2y > 0 there, past its inflection),
    with the parent's signal under the child's apex at least
    `rider_height_ratio` x the child's height. Otherwise 'shoulder'."""
    if child_apex_rt <= parent_apex_rt:
        return 'shoulder'
    h = 1e-3 * parent.sigma
    t = np.array([child_apex_rt - h, child_apex_rt, child_apex_rt + h])
    py = parent(t)
    convex = py[0] - 2 * py[1] + py[2] > 0
    child_h = float(child(t[1:2])[0])
    return 'rider' if convex and py[1] >= rider_height_ratio * child_h else 'shoulder'


def detect_shoulders(x, y, peak_indices, *, sensitivity: float = DEFAULT_SENSITIVITY,
                     alpha: float = 1e-4, min_snr: float = 3.0, k_win: float = 0.6,
                     rider_height_ratio: float = 1.0,
                     noise: float | None = None) -> list[ShoulderCall]:
    """Find shoulders and riders on the peaks at ``peak_indices``.

    Args:
        x, y: retention time and baseline-corrected signal (finite; extend
            masked regions before calling).
        peak_indices: apexes of the already-detected peaks.
        sensitivity: 1 (strictest) .. 10; sets the curvature candidate threshold.
        alpha: significance an extra fitted component must reach to be kept.
        min_snr: smallest child (fitted height / noise SD) that is reported.
        k_win: Savitzky-Golay window for d2y, in multiples of the peak's FWHM.
        rider_height_ratio: see :func:`_classify`.
        noise: the signal's noise SD, if the caller knows it better than
            :func:`noise_sd` on ``y`` (e.g. ``y`` has filled-in masked regions).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    peaks = np.asarray(sorted(int(p) for p in peak_indices), dtype=int)
    if len(peaks) == 0 or len(y) < 7:
        return []
    sd = noise_sd(y) if noise is None else float(noise)
    if sd <= 0:
        return []
    dx = float(np.median(np.diff(x)))
    k_prom = k_prom_for(sensitivity)
    w, _, left_ips, _ = peak_widths(y, peaks, rel_height=0.5)
    # Twice the leading half-width tracks the Gaussian width and ignores the
    # tail, so a tailing parent doesn't get a window too wide for its riders.
    w_lead = np.minimum(w, 2.0 * (peaks - left_ips))
    _, lb, rb = peak_prominences(y, peaks)
    # A small peak's prominence bases can run across much of a busy run; keep
    # each peak's extent local (room after the apex for a tail and its riders).
    lb = np.maximum(lb, (peaks - EXTENT_FWHM[0] * w).astype(int))
    rb = np.minimum(rb, (peaks + EXTENT_FWHM[1] * w).astype(int))
    calls: list[ShoulderCall] = []
    poly = 3

    for j, p in enumerate(peaks):
        win = max(poly + 2, int(round(k_win * w_lead[j])) | 1)
        d2_noise = sd * float(np.sqrt(np.sum(savgol_coeffs(win, poly, deriv=2, delta=dx) ** 2)))
        # candidates live between this peak's neighbours, inside its bases
        lo = max(lb[j], peaks[j - 1] if j else 0)
        hi = min(rb[j], peaks[j + 1] if j + 1 < len(peaks) else len(y) - 1)
        a, b = max(0, lo - win), min(len(y), hi + win)
        if b - a <= win:
            continue
        d2 = savgol_filter(y[a:b], win, poly, deriv=2, delta=dx)
        mins, props = find_peaks(-d2, prominence=k_prom * d2_noise)
        mins = mins + a
        own = mins[np.argmin(np.abs(mins - p))] if len(mins) else p
        guard = max(2, win // 2)
        cands = [(int(m), pr) for m, pr in zip(mins, props['prominences'])
                 if m != own and lo + guard < m < hi - guard and y[m] >= min_snr * sd]
        if not cands:
            continue

        # Fit window: this peak's bases, widened to the whole of any detected
        # neighbour whose apex lies inside them. A half-covered neighbour fits
        # badly, and then any extra component "improves" the fit.
        fw = w[j] * dx
        seeds = [(x[p], y[p], fw, 0.3 * fw, True)]
        fa, fb = lb[j], rb[j] + 1
        inside = [i2 for i2, q in enumerate(peaks) if q != p and lb[j] <= q <= rb[j]]
        for i2 in sorted(inside, key=lambda i: abs(peaks[i] - p))[:MAX_NEIGHBOURS]:
            q = peaks[i2]
            seeds.append((x[q], y[q], w[i2] * dx, 0.3 * w[i2] * dx, True))
            fa, fb = min(fa, lb[i2]), max(fb, rb[i2] + 1)
        t, yy = x[fa:fb], y[fa:fb]
        base_p, base_rss = _fit(t, yy, seeds)
        if base_p is None:
            continue

        for m, _ in sorted(cands, key=lambda c: -c[1]):
            h0 = max(y[m] - float(_multi_emg(np.array([x[m]]), *base_p)[0]), sd)
            trial = seeds + [(x[m], h0, 0.8 * fw, 0.3 * fw, False)]
            p2, rss2 = _fit(t, yy, trial)
            if p2 is None or rss2 >= base_rss:
                continue
            dof = max(len(t) - 4 * len(trial), 1)
            f_stat = ((base_rss - rss2) / 4) / (rss2 / dof) if rss2 > 0 else np.inf
            if f_stat < f_dist.ppf(1 - alpha, 4, dof):
                continue
            child = EMG(*p2[-4:])
            cy = child(t)
            if cy.max() < min_snr * sd:
                continue
            seeds, base_p, base_rss = trial, p2, rss2
            parent = EMG(*p2[:4])
            apex = fa + int(np.argmax(cy))
            kind = _classify(parent, child, x[p], x[apex], rider_height_ratio)
            split = _split_index(t, fa, parent(t), cy, p, apex, kind)
            outer = _outer_index(cy, fa, apex, after=apex > p)
            calls.append(ShoulderCall(int(p), int(apex), int(split), int(outer), kind,
                                      'after' if apex > p else 'before', parent, child, float(f_stat)))

    # a shoulder between two peaks can be claimed from both sides: keep one
    kept: list[ShoulderCall] = []
    tol = 0.3 * float(np.median(w))
    for c in calls:
        if any(abs(c.apex_index - k.apex_index) < tol for k in kept) or \
                any(abs(c.apex_index - q) < tol for q in peaks):
            continue
        kept.append(c)
    return kept


def _split_index(t, fa, parent_y, child_y, p, apex, kind):
    lo, hi = sorted((p - fa, apex - fa))
    if kind == 'rider':
        # rider start: last point on the parent's side below 2 % of its height
        side = np.arange(lo, hi + 1)
        below = side[child_y[side] < 0.02 * child_y.max()]
        return fa + int(below[-1] if len(below) else lo)
    seg = slice(lo, hi + 1)
    return fa + lo + int(np.argmin(np.abs(parent_y[seg] - child_y[seg])))


def _outer_index(child_y, fa, apex, after):
    rel = apex - fa
    idx = np.flatnonzero(child_y < 0.02 * child_y.max())
    far = idx[idx > rel] if after else idx[idx < rel]
    if len(far) == 0:
        return fa + (len(child_y) - 1 if after else 0)
    return fa + int(far[0] if after else far[-1])
