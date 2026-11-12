"""Skim geometry for tail skimming: a small rider on a bigger peak's decaying tail.

Pure functions on arrays (numpy and scipy only). They measure whether the parent's back side is a tail
(`tail_bend`), where the rider ends (`rider_end`) and what baseline the rider is integrated above
(`exp_skim`, `tangent_skim`). Indices are positions in `x` / `y`; `y` is the smoothed,
baseline-corrected signal.
"""
import logging
import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

logger = logging.getLogger(__name__)

# Tail-bend band on the parent's back side (spec section 1, test 4).
BEND_L1_VALLEY_FACTOR = 1.5    # L1 = max(1.5 * h_v, BEND_L1_PARENT_FRAC * h_p)
BEND_L1_PARENT_FRAC = 0.05
BEND_L2_SPAN = 4.0             # L2 = min(4 * L1, BEND_L2_PARENT_CAP * h_p)
BEND_L2_PARENT_CAP = 0.8
BEND_MIN_BAND_RATIO = 1.5      # fail when L2 < 1.5 * L1
BEND_MIN_POINTS = 6            # fail with fewer points in the band

# Exponential skim fit windows (spec section 4).
SKIM_MIN_UPSTREAM_POINTS = 8
SKIM_MIN_DOWNSTREAM_POINTS = 3
SKIM_FIT_MAX_EVALS = 5000
SKIM_MIN_RATE = 1e-6           # k > 0 bound, in 1/min
_TINY = 1e-12                  # floor for log of a non-positive signal value


@dataclass
class SkimCurve:
    """A skim baseline over the rider span `[a, b]`.

    `s` is clipped to the signal. `model` is 'exponential', 'exponential_2pt' or 'tangent'. `tail`
    evaluates the parent-tail model at any times (used to ask whether a tail is still real further on).
    """
    s: np.ndarray
    model: str
    tail: Callable[[np.ndarray], np.ndarray]


def tail_bend(x: np.ndarray, y: np.ndarray, apex: int, valley: int) -> float:
    """How much the parent's fall speeds up between the apex and the valley, on a log scale.

    Fits ln(y) against x on the upper and lower halves of the first contiguous run of back-side points
    with L1 <= y <= L2, and returns |lower slope| / |upper slope|. A tail gives about 1 or less, a
    symmetric flank more than 1.1. NaN when the band cannot be measured.
    """
    h_p, h_v = y[apex], y[valley]
    l1 = max(BEND_L1_VALLEY_FACTOR * h_v, BEND_L1_PARENT_FRAC * h_p)
    l2 = min(BEND_L2_SPAN * l1, BEND_L2_PARENT_CAP * h_p)
    if l2 <= l1 * BEND_MIN_BAND_RATIO:
        return np.nan
    seg = np.arange(apex, valley + 1)
    sel = seg[(y[seg] <= l2) & (y[seg] >= l1)]
    breaks = np.flatnonzero(np.diff(sel) > 1)
    if breaks.size:
        sel = sel[:breaks[0] + 1]
    if len(sel) < BEND_MIN_POINTS:
        return np.nan
    half = len(sel) // 2
    upper, lower = sel[:half], sel[half:]
    slope_upper = np.polyfit(x[upper], np.log(y[upper]), 1)[0]
    slope_lower = np.polyfit(x[lower], np.log(y[lower]), 1)[0]
    return slope_lower / slope_upper if slope_upper < 0 else np.nan


def rider_end(x: np.ndarray, y: np.ndarray, a: int, c: int, cap: int) -> int:
    """Where the signal rejoins the parent's tail.

    The index in `(c, cap]` with the most negative chord slope from `(x[a], y[a])`: a line from the
    rider's start to there touches the tail beyond the rider without cutting it. Returns `cap` when the
    range is empty.
    """
    t = np.arange(c + 1, cap + 1)
    if len(t) == 0:
        return cap
    slopes = (y[t] - y[a]) / (x[t] - x[a])
    return int(t[np.argmin(slopes)])


def _tail_points(a: int, b: int, apex: int, cap: int,
                 skip: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Parent-tail indices before `a` and from `b`, each over the rider's span, outside `skip` spans."""
    width = max(b - a, SKIM_MIN_UPSTREAM_POINTS)
    free = lambda j: not any(s <= j <= e for s, e in skip)
    upstream = [j for j in range(max(apex + 1, a - width), a + 1) if free(j)]
    downstream = [j for j in range(b, min(cap, b + width) + 1) if free(j)]
    return np.array(upstream, dtype=int), np.array(downstream, dtype=int)


def _fit_decay_rate(x, y, pre, post, a, b):
    """Decay rate k of C + A*exp(-k*t) fitted to the tail points on both sides, or None when it fails."""
    idx = np.r_[pre, post]
    t0 = x[a]
    model = lambda t, amp, k, offset: offset + amp * np.exp(-k * t)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', (OptimizeWarning, np.exceptions.RankWarning))
            k0 = max(1e-3, -np.polyfit(x[pre] - t0, np.log(np.clip(y[pre], _TINY, None)), 1)[0])
            popt, _ = curve_fit(
                model, x[idx] - t0, y[idx],
                p0=[y[a], k0, min(y[post].min(), y[b]) * 0.5],
                bounds=([0, SKIM_MIN_RATE, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=SKIM_FIT_MAX_EVALS)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        logger.debug("tail skim: exponential fit failed (%s)", exc)
        return None
    k = popt[1]
    return float(k) if np.isfinite(k) else None


def exp_skim(x: np.ndarray, y: np.ndarray, a: int, b: int, apex: int, cap: int,
             skip: List[Tuple[int, int]]) -> SkimCurve:
    """Exponential skim anchored at both ends of the rider, with its decay taken from the tail around it.

    Fits C + A*exp(-k*(t - t_a)) to parent-tail points upstream of `a` and downstream of `b` (points in
    the `skip` spans excluded), keeps k, then solves A and C so the curve passes through
    `(x[a], y[a])` and `(x[b], y[b])`. Falls back to a two-point exponential without offset when the fit
    fails, there are too few points or `y[a] <= y[b]`, and then to the tangent when `y[a] > y[b] > 0`
    does not hold.
    """
    pre, post = _tail_points(a, b, apex, cap, skip)
    t0, ya, yb = x[a], y[a], y[b]
    dt_span = x[b] - t0
    tt = x[a:b + 1] - t0

    k = None
    if len(pre) >= SKIM_MIN_UPSTREAM_POINTS and len(post) >= SKIM_MIN_DOWNSTREAM_POINTS:
        k = _fit_decay_rate(x, y, pre, post, a, b)

    if k is not None and ya > yb:
        amp = (ya - yb) / (1 - np.exp(-k * dt_span))
        offset = ya - amp
        s = offset + amp * np.exp(-k * tt)
        model = 'exponential'
        tail = lambda t: offset + amp * np.exp(-k * (np.asarray(t) - t0))
    elif ya > yb > 0:
        logger.debug("tail skim: using two-point exponential")
        k2 = np.log(ya / yb) / dt_span
        s = ya * np.exp(-k2 * tt)
        model = 'exponential_2pt'
        tail = lambda t: ya * np.exp(-k2 * (np.asarray(t) - t0))
    else:
        logger.debug("tail skim: using tangent")
        return tangent_skim(x, y, a, b)
    return SkimCurve(np.minimum(s, y[a:b + 1]), model, tail)


def tangent_skim(x: np.ndarray, y: np.ndarray, a: int, b: int) -> SkimCurve:
    """Straight line from `(x[a], y[a])` to `(x[b], y[b])`, clipped to the signal."""
    xa, xb, ya, yb = x[a], x[b], y[a], y[b]
    s = np.interp(x[a:b + 1], [xa, xb], [ya, yb])
    tail = lambda t: np.interp(t, [xa, xb], [ya, yb], right=yb)
    return SkimCurve(np.minimum(s, y[a:b + 1]), 'tangent', tail)
