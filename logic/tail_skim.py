"""Tail skimming: a small rider on a bigger peak's decaying tail.

Pure functions on arrays (numpy and scipy only). Geometry: whether the parent's back side is a tail
(`tail_bend`), where the rider ends (`rider_end`) and what baseline the rider is integrated above
(`exp_skim`, `tangent_skim`). Intake: `classify_riders` decides, peak by peak, split or skim.
Indices are positions in `x` / `y`; `y` is the smoothed, baseline-corrected signal.
"""
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from logic.windows import window_at

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


# ---------------------------------------------------------------------------------------------------
# Intake: split or skim (spec section 1)
# ---------------------------------------------------------------------------------------------------

# Mirrors logic.method.RiderParams (a test keeps the two equal; this module must not import method).
RIDER_DEFAULTS = {
    'rider_integration': 'auto',
    'skim_model': 'exponential',
    'skim_height_ratio': 2.0,
    'skim_max_valley_frac': 0.5,
    'skim_valley_ratio': 20.0,
    'skim_tail_bend': 1.10,
    'skim_chain_min_tail': 0.15,
    'skim_flag_band': 0.05,
}
PARENT_FLOOR_FRAC = 0.0025     # signal must stay above this fraction of h_p between parent and child
NO_NEXT_CAP_MIN = 2.0          # min; how far past the last apex a rider's cap may be searched


@dataclass
class RiderCall:
    """The split-or-skim decision for one child peak.

    `child`, `parent`, `a`, `cap` are indices into `x`. `parent` may be a range-excluded apex.
    `ratios` holds hp_hc, hv_hp, hc_hv and tail_bend. `failed` lists the tests that failed, drawn from
    front, height, position, valley, tail, excluded. `tail` is the exponential parent-tail model
    (skim decisions only), used to ask whether a later child still sits on the same tail.
    """
    child: int
    parent: int
    a: int
    cap: int
    decision: Literal['skim', 'split']
    source: Literal['auto', 'forced']
    skim_model: str
    ratios: dict
    failed: List[str]
    near_threshold: bool
    tail: Optional[Callable[[np.ndarray], np.ndarray]] = field(default=None, repr=False, compare=False)


def rider_settings_at(defaults: Optional[dict], windows: List[dict], rt: float) -> dict:
    """Rider settings governing a child with apex time `rt`.

    `defaults is None` means the method has no rider block (legacy): always split. Otherwise the
    non-None fields of the holding window's `rider` overrides are merged over the defaults.
    """
    if defaults is None:
        return {**RIDER_DEFAULTS, 'rider_integration': 'split'}
    out = {**RIDER_DEFAULTS, **defaults}
    window = window_at(windows or [], rt)
    overrides = (window or {}).get('rider') or {}
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _thresholds(cfg: dict, ratios: dict) -> List[Tuple[str, float, float, bool]]:
    """(test name, measured value, threshold, passes) for the four tests of the rule."""
    bend = ratios['tail_bend']
    return [
        ('height', ratios['hp_hc'], cfg['skim_height_ratio'], ratios['hp_hc'] >= cfg['skim_height_ratio']),
        ('position', ratios['hv_hp'], cfg['skim_max_valley_frac'], ratios['hv_hp'] <= cfg['skim_max_valley_frac']),
        ('valley', ratios['hc_hv'], cfg['skim_valley_ratio'], ratios['hc_hv'] <= cfg['skim_valley_ratio']),
        ('tail', bend, cfg['skim_tail_bend'], bool(np.isfinite(bend) and bend <= cfg['skim_tail_bend'])),
    ]


def _near_threshold(tests, band: float) -> bool:
    """True when moving one measured value by `band` (relative) across its threshold flips split/skim."""
    for i, (_, value, threshold, _) in enumerate(tests):
        if not np.isfinite(value) or abs(value / threshold - 1) > band:
            continue
        if all(ok for j, (_, _, _, ok) in enumerate(tests) if j != i):
            return True
    return False


def classify_riders(x: np.ndarray, y: np.ndarray, apexes: np.ndarray, metadata: List[dict],
                    defaults: Optional[dict], windows: List[dict], negative: set,
                    group_spans: List[Tuple[int, int]]) -> Dict[int, RiderCall]:
    """Decide split or skim for every child peak, in RT order. Keyed by the child's apex index.

    `apexes` are sorted indices (kept peaks plus range-excluded apexes) and `metadata[i]` belongs to
    `apexes[i]`. `negative` holds apex indices of negative peaks; `group_spans` are `(start, end)` index
    spans of peak groups. Pairs that fail the always-required checks (parent after child or not taller,
    tail back to baseline between them) get no call.
    """
    calls: Dict[int, RiderCall] = {}
    spans: Dict[int, Tuple[int, int]] = {}     # skimmed rider apex -> (a, b), excluded from later tail fits
    apexes = [int(i) for i in apexes]

    for k in range(1, len(apexes)):
        q, c = apexes[k - 1], apexes[k]
        meta = metadata[k] or {}
        cfg = rider_settings_at(defaults, windows, float(x[c]))
        h_c = float(y[c])

        # 1. parent
        if meta.get('parent_index') is not None:
            p = int(meta['parent_index'])
        else:
            p = q
            prev = calls.get(q)
            if prev is not None and prev.decision == 'skim' and prev.tail is not None:
                if float(prev.tail(x[c])) >= cfg['skim_chain_min_tail'] * h_c:
                    p = prev.parent

        # 2. valley
        if meta.get('is_shoulder') and meta.get('split_index') is not None:
            v = int(meta['split_index'])
        elif meta.get('is_shoulder') and meta.get('left_bound') is not None:
            v = int(meta['left_bound'])
        else:
            v = q + int(np.argmin(y[q:c + 1]))
        h_v, h_p = float(y[v]), float(y[p])

        # 3. always required: no call when the pair is not a parent/rider pair at all
        if not (p < c and h_p > h_c and np.all(y[p:c + 1] > PARENT_FLOOR_FRAC * h_p)):
            continue
        nxt = apexes[k + 1] if k + 1 < len(apexes) else None
        if nxt is not None:
            cap = c + int(np.argmin(y[c:nxt + 1]))
        else:
            end = min(len(y) - 1, int(np.searchsorted(x, x[c] + NO_NEXT_CAP_MIN)))
            cap = c + int(np.argmin(y[c:end + 1]))
        front = nxt is not None and y[nxt] > h_c and y[cap] > h_v
        excluded = (c in negative or p in negative
                    or any(s <= c and e >= p for s, e in group_spans))
        ratios = {'hp_hc': h_p / h_c, 'hv_hp': h_v / h_p, 'hc_hv': h_c / h_v,
                  'tail_bend': tail_bend(x, y, p, v)}
        tests = _thresholds(cfg, ratios)

        # 4. mode
        mode = cfg['rider_integration']
        always = (['front'] if front else []) + (['excluded'] if excluded else [])
        if mode == 'split':
            failed, skim, source = [], False, 'forced'
        elif mode == 'skim':
            failed, skim, source = always, not always, 'forced'
        else:
            failed = (['front'] if front else []) + [n for n, _, _, ok in tests if not ok] \
                + (['excluded'] if excluded else [])
            skim, source = not failed, 'auto'
        if excluded and mode != 'split':
            logger.warning("tail skim: peak at %.3f min not skimmed (negative peak or peak-group overlap)", x[c])

        # 5. near threshold (auto only; front and excluded are hard stops)
        near = mode == 'auto' and not always and _near_threshold(tests, cfg['skim_flag_band'])

        call = RiderCall(child=c, parent=p, a=v, cap=cap, decision='skim' if skim else 'split',
                         source=source, skim_model=cfg['skim_model'], ratios=ratios, failed=failed,
                         near_threshold=near)
        # 6. skim: the chain test always reads an exponential parent-tail model
        if skim:
            b = rider_end(x, y, v, c, cap)
            skip = [spans[r] for r, rc in calls.items() if rc.decision == 'skim' and rc.parent == p and r in spans]
            call.tail = exp_skim(x, y, v, b, p, cap, skip).tail
            spans[c] = (v, b)
        calls[c] = call
    return calls
