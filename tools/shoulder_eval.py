"""Evaluation harness for shoulder and rider detection.

Two kinds of second component are scored:
- **shoulder**: two peaks merged with no valley between them, still clearly two;
- **rider**: a small peak on the long decaying tail of a big one.
Pairs that do have a valley ("resolved") are scored too, so a detector that
breaks ordinary valley splitting shows up.

A detector is any callable ``detector(x, y) -> list[Report]``: every apex it
reports (ordinary peaks and shoulders alike) with the split it would integrate
at. Scoring matches reports to the known components by apex position, so a
correct rejection (nothing extra reported on a single peak) counts in the
detector's favour and an extra report counts against it.

Synthetic cases: ``synthetic_cases()``. Real cases: ``load_labelled_runs()``
reads label JSON files kept outside the repo (``~/chromakit-worktrees/_data``).

Run: ``python -m tools.shoulder_eval --out <dir>`` from the repo root.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import exponnorm

DT_MIN = 1.0 / 1200.0  # 20 Hz sampling, in minutes
PARENT_HEIGHT = 1.0e5


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Component:
    mu: float
    sigma: float
    tau: float
    height: float
    role: str            # 'parent' or 'child'
    kind: str = ''       # child only: 'shoulder' | 'resolved' | 'rider'
    apex: float = 0.0    # filled by build_case
    fwhm: float = 0.0


@dataclass
class Case:
    name: str
    x: np.ndarray
    y: np.ndarray                 # noisy signal, baseline 0
    y_clean: np.ndarray
    components: list[Component]
    noise_sd: float
    meta: dict = field(default_factory=dict)
    true_split: float | None = None   # child only


@dataclass
class Report:
    rt: float
    kind: str = 'peak'                # 'peak' | 'shoulder' | 'rider'
    split_rt: float | None = None     # boundary to the neighbour it was split from


# --------------------------------------------------------------------------
# Synthetic generation
# --------------------------------------------------------------------------

def emg_unit(t, mu, sigma, tau):
    """EMG (Gaussian when tau == 0) scaled to unit height."""
    if tau <= 0:
        y = np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    else:
        y = np.nan_to_num(exponnorm.pdf(t, tau / sigma, loc=mu, scale=sigma))
    m = y.max()
    return y / m if m > 0 else y


def _shape_stats(sigma, tau):
    """Apex offset from mu and FWHM of an EMG, measured on a fine grid."""
    t = np.linspace(-8 * sigma, 8 * sigma + 12 * tau, 20001)
    y = emg_unit(t, 0.0, sigma, tau)
    above = t[y >= 0.5]
    return t[np.argmax(y)], above[-1] - above[0]


def build_case(name, comps, noise_sd, rng, meta, dt=DT_MIN):
    for c in comps:
        off, fwhm = _shape_stats(c.sigma, c.tau)
        c.apex, c.fwhm = c.mu + off, fwhm
    lo = min(c.apex - 12 * c.fwhm for c in comps)
    hi = max(c.apex + 12 * c.fwhm + 10 * c.tau for c in comps)
    x = np.arange(lo, hi, dt)
    curves = [c.height * emg_unit(x, c.mu, c.sigma, c.tau) for c in comps]
    y_clean = np.sum(curves, axis=0)
    y = y_clean + rng.normal(0.0, noise_sd, len(x))
    case = Case(name, x, y, y_clean, comps, noise_sd, meta)
    children = [i for i, c in enumerate(comps) if c.role == 'child']
    if children:
        ci = children[0]
        p, c = comps[0], comps[ci]
        if c.kind != 'rider':
            c.kind = 'resolved' if _has_valley(x, y_clean, p.apex, c.apex) else 'shoulder'
        case.true_split = _true_split(x, curves[0], curves[ci], p, c)
    return case


def _has_valley(x, y, a, b):
    lo, hi = sorted((a, b))
    seg = y[(x >= lo) & (x <= hi)]
    if len(seg) < 3:
        return False
    # a real local minimum strictly inside, deeper than float noise
    return seg.min() < min(seg[0], seg[-1]) * (1 - 1e-6)


def _true_split(x, parent_curve, child_curve, p, c):
    """Shoulder/resolved: where the two components cross between the apexes.
    Rider: the rider's start, where it first reaches 2 % of its height."""
    if c.kind == 'rider':
        side = (x < c.apex) if c.apex > p.apex else (x > c.apex)
        idx = np.flatnonzero(side & (child_curve < 0.02 * c.height))
        if len(idx) == 0:
            return None
        return float(x[idx[-1]] if c.apex > p.apex else x[idx[0]])
    lo, hi = sorted((p.apex, c.apex))
    m = (x >= lo) & (x <= hi)
    d = np.abs(parent_curve[m] - child_curve[m])
    return float(x[m][np.argmin(d)])


def _place_pair(rs, sigma, tau, side):
    """mu of a child of the same shape whose apex sits at resolution `rs`."""
    _, fwhm = _shape_stats(sigma, tau)
    delta = rs * (2 * fwhm) / 1.18
    return 0.0 + side * delta   # same shape -> same apex offset, so mu shifts by delta


def _snr_bin(v):
    """The child's own height over the noise SD, binned."""
    for edge in (3, 10, 30, 100):
        if v < edge:
            return f'<{edge}'
    return '>=100'


def synthetic_cases(seed=0, quick=False):
    """The fixed synthetic grid. Every case is a short segment, baseline 0."""
    rng = np.random.default_rng(seed)
    sig_pts = [6, 12, 25]
    snrs = [1000, 200, 50]
    reps = 1 if quick else 2
    cases = []

    # shoulder / resolved pairs
    rs_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
    ratios = [1.0, 0.5, 0.25, 0.1]
    taus = [0.0, 1.0, 2.0]          # tau / sigma
    if quick:
        sig_pts, rs_grid, ratios = [12], [0.5, 0.8, 1.2], [1.0, 0.25]
    for sp, rs, hr, tr, snr, side, rep in itertools.product(
            sig_pts, rs_grid, ratios, taus, snrs, (+1, -1), range(reps)):
        s = sp * DT_MIN
        tau = tr * s
        mu_c = _place_pair(rs, s, tau, side)
        comps = [Component(0.0, s, tau, PARENT_HEIGHT, 'parent'),
                 Component(mu_c, s, tau, hr * PARENT_HEIGHT, 'child')]
        meta = dict(family='pair', sigma_pts=sp, rs=rs, ratio=hr, tau_ratio=tr,
                    snr=snr, side='after' if side > 0 else 'before', child_snr=_snr_bin(hr * snr))
        cases.append(build_case(f'pair-{len(cases)}', comps, PARENT_HEIGHT / snr, rng, meta))

    # riders on a long tail
    r_taus = [3.0, 6.0]
    r_pos = [3.0, 5.0, 8.0]          # rider apex distance after parent apex, in parent FWHM
    r_ratio = [0.01, 0.03, 0.1]
    for sp, tr, pos, hr, snr, rep in itertools.product(
            sig_pts, r_taus, r_pos, r_ratio, snrs, range(reps)):
        s = sp * DT_MIN
        tau = tr * s
        off, fwhm = _shape_stats(s, tau)
        apex_r = off + pos * fwhm
        comps = [Component(0.0, s, tau, PARENT_HEIGHT, 'parent'),
                 Component(apex_r, s, 0.0, hr * PARENT_HEIGHT, 'child', kind='rider')]
        meta = dict(family='rider', sigma_pts=sp, tau_ratio=tr, pos_fwhm=pos,
                    ratio=hr, snr=snr, side='after', child_snr=_snr_bin(hr * snr))
        cases.append(build_case(f'rider-{len(cases)}', comps, PARENT_HEIGHT / snr, rng, meta))

    # negatives: single peaks, tailing from none to heavy
    for sp, tr, snr, rep in itertools.product(
            sig_pts, [0.0, 0.5, 1.0, 2.0, 4.0, 6.0], snrs, range(reps * 4)):
        s = sp * DT_MIN
        comps = [Component(0.0, s, tr * s, PARENT_HEIGHT, 'parent')]
        meta = dict(family='single', sigma_pts=sp, tau_ratio=tr, snr=snr)
        cases.append(build_case(f'single-{len(cases)}', comps, PARENT_HEIGHT / snr, rng, meta))
    return cases


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class CaseScore:
    name: str
    meta: dict
    child_kind: str | None   # None for a single peak
    child_found: bool
    parent_found: bool
    false_reports: int
    split_err_fwhm: float | None   # |split - true| / child FWHM, matched child only
    found_as: str | None


def _match(reports, apex, tol, used):
    best = None
    for ri, r in enumerate(reports):
        d = abs(r.rt - apex)
        if ri not in used and d <= tol and (best is None or d < best[0]):
            best = (d, ri)
    return None if best is None else best[1]


def score_case(case: Case, reports: list[Report]) -> CaseScore:
    """Match the child first, tightly (it must not be confused with the parent),
    then the parent within half its FWHM. A merged pair's signal maximum can sit
    between the two apexes; that is an RT shift of the parent, not a false split."""
    comps = case.components
    used_r, matched = set(), {}
    if len(comps) > 1:
        gap = abs(comps[1].apex - comps[0].apex)
        ri = _match(reports, comps[1].apex, min(0.5 * comps[1].fwhm, 0.45 * gap), used_r)
        if ri is not None:
            matched[1] = ri
            used_r.add(ri)
    ri = _match(reports, comps[0].apex, 0.5 * comps[0].fwhm, used_r)
    if ri is not None:
        matched[0] = ri
        used_r.add(ri)
    child_i = 1 if len(comps) > 1 else None
    child_found = child_i is not None and child_i in matched
    split_err, found_as = None, None
    if child_found:
        r = reports[matched[child_i]]
        found_as = r.kind
        split = r.split_rt
        if split is None and 0 in matched:
            split = _valley_between(case.x, case.y, reports[matched[0]].rt, r.rt)
        if split is not None and case.true_split is not None:
            split_err = abs(split - case.true_split) / comps[child_i].fwhm
    return CaseScore(
        name=case.name, meta=case.meta,
        child_kind=comps[child_i].kind if child_i is not None else None,
        child_found=child_found, parent_found=0 in matched,
        false_reports=len(reports) - len(used_r),
        split_err_fwhm=split_err, found_as=found_as)


def _valley_between(x, y, a, b):
    lo, hi = sorted((a, b))
    m = (x >= lo) & (x <= hi)
    if not m.any():
        return None
    return float(x[m][np.argmin(y[m])])


def summarize(scores: Iterable[CaseScore]) -> dict:
    """Precision/recall of the second component, by kind, plus false splits."""
    scores = list(scores)
    out = {}
    tp = sum(s.child_found for s in scores)
    fn = sum(1 for s in scores if s.child_kind and not s.child_found)
    fp = sum(s.false_reports for s in scores)
    out['all'] = _pr(tp, fp, fn)
    for kind in ('shoulder', 'resolved', 'rider'):
        sub = [s for s in scores if s.child_kind == kind]
        if sub:
            k_tp = sum(s.child_found for s in sub)
            errs = [s.split_err_fwhm for s in sub if s.split_err_fwhm is not None]
            out[kind] = dict(n=len(sub), recall=k_tp / len(sub),
                             false_reports=sum(s.false_reports for s in sub),
                             split_err_median_fwhm=float(np.median(errs)) if errs else None)
    singles = [s for s in scores if s.child_kind is None]
    if singles:
        out['single'] = dict(n=len(singles),
                             false_split_rate=sum(s.false_reports > 0 for s in singles) / len(singles))
    out['parent_miss'] = sum(not s.parent_found for s in scores)
    return out


def _pr(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float('nan')
    r = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def breakdown(scores, key, kind):
    """Recall of `kind` children and false reports, grouped by meta[key]."""
    rows = {}
    for s in scores:
        if s.child_kind != kind:
            continue
        v = s.meta.get(key)
        r = rows.setdefault(v, [0, 0, 0])
        r[0] += 1
        r[1] += s.child_found
        r[2] += s.false_reports
    return {k: dict(n=n, recall=f / n, false_reports=fr) for k, (n, f, fr) in sorted(rows.items())}


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

def _load_processor(processor_file=None):
    """ChromatogramProcessor from this checkout, or from a saved copy of an
    older logic/processor.py (e.g. `git show main:logic/processor.py`)."""
    if processor_file is None:
        from logic.processor import ChromatogramProcessor
        return ChromatogramProcessor()
    import importlib.util
    spec = importlib.util.spec_from_file_location('legacy_processor', processor_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ChromatogramProcessor()


def reports_from_processed(x, out) -> list[Report]:
    """Reports from ChromatogramProcessor.process output."""
    reps = []
    parents = [p['index'] for p in out['peak_metadata'] if not p.get('is_shoulder')]
    for m in out['peak_metadata']:
        if not m.get('is_shoulder'):
            reps.append(Report(float(m['x']), 'peak'))
            continue
        split = m.get('split_index')
        if split is None and parents:
            # legacy metadata: the inner bound is the one facing the nearest peak
            nearest = min(parents, key=lambda p: abs(p - m['index']))
            split = m.get('left_bound') if nearest < m['index'] else m.get('right_bound')
        reps.append(Report(float(m['x']), m.get('shoulder_kind', 'shoulder'),
                           float(x[split]) if split is not None else None))
    return reps


def processor_detector(sensitivity=8, prominence_snr=10.0, processor_file=None):
    """The app's pipeline: ChromatogramProcessor.process with shoulders on.

    Smoothing and baseline correction are off so only detection is scored.
    Peak prominence is `prominence_snr` x the true noise SD, the way a user
    sets an absolute prominence for their detector's noise.
    """
    proc = _load_processor(processor_file)

    def detect(x, y, noise_sd):
        params = {
            'smoothing': {'enabled': False},
            'baseline': {'enabled': False, 'method': 'asls', 'lambda': 1e6},
            'peaks': {'enabled': True, 'min_prominence': max(prominence_snr * noise_sd, 1.0001)},
            # window_length/polyorder/apex_distance: read only by the legacy detector
            'shoulders': {'enabled': True, 'sensitivity': sensitivity, 'window_length': 41,
                          'polyorder': 3, 'apex_distance': 10},
        }
        return reports_from_processed(x, proc.process(x, y, params))
    return detect


def run(detector, cases):
    return [score_case(c, detector(c.x, c.y, c.noise_sd)) for c in cases]


# --------------------------------------------------------------------------
# Real runs, scored against labels from tools/shoulder_label_sheet.py
# --------------------------------------------------------------------------

def score_labelled(labels_path, sensitivity=5, processor_file=None, tol_fwhm=0.5):
    """Precision/recall of shoulder+rider calls against Caleb's labels.

    Each run is processed with its method (or ChromaMethod defaults) and
    shoulders on. Only reviewed panels count. A call or label matches within
    `tol_fwhm` x the FWHM of the nearest detected peak. An ordinary peak at a
    labelled spot also counts as found (it was split, just with a valley).
    Kind (shoulder vs rider) is reported separately, not required to match.
    """
    from scipy.signal import peak_widths
    from tools.shoulder_label_sheet import load_run
    proc = _load_processor(processor_file)
    doc = json.loads(Path(labels_path).read_text())
    tp = fp = fn = kind_ok = 0
    per_run = []
    for run_ in doc['runs']:
        x, y, params = load_run(run_['path'], run_.get('method'), detector=run_.get('detector'))
        params['shoulders'] = {'enabled': True, 'sensitivity': sensitivity, 'window_length': 41,
                               'polyorder': 3, 'apex_distance': 10}
        out = proc.process(x, y, params)
        reps = reports_from_processed(x, out)
        peaks = [int(np.argmin(np.abs(x - r.rt))) for r in reps if r.kind == 'peak']
        widths = peak_widths(out['corrected_y'], peaks, rel_height=0.5)[0] * np.median(np.diff(x)) if peaks else []

        def tol_at(rt):
            if not len(widths):
                return 0.02
            i = int(np.argmin([abs(x[p] - rt) for p in peaks]))
            return tol_fwhm * widths[i]

        def in_panels(rt):
            return any(lo <= rt <= hi for lo, hi in run_['panels'])

        calls = [r for r in reps if r.kind != 'peak' and in_panels(r.rt)]
        labels = list(run_['labels'])
        used = set()
        r_tp = r_fn = 0
        for lab in labels:
            cand = [(abs(r.rt - lab['rt']), i) for i, r in enumerate(calls)
                    if i not in used and abs(r.rt - lab['rt']) <= tol_at(lab['rt'])]
            if cand:
                _, i = min(cand)
                used.add(i)
                r_tp += 1
                kind_ok += calls[i].kind == lab['kind']
            elif any(r.kind == 'peak' and abs(r.rt - lab['rt']) <= tol_at(lab['rt']) for r in reps):
                r_tp += 1
            else:
                r_fn += 1
        r_fp = len(calls) - len(used)
        tp, fp, fn = tp + r_tp, fp + r_fp, fn + r_fn
        per_run.append(dict(run=run_['name'], labels=len(labels), calls=len(calls), tp=r_tp, fp=r_fp, fn=r_fn,
                            false_calls=[round(r.rt, 4) for i, r in enumerate(calls) if i not in used]))
    return dict(**_pr(tp, fp, fn), kind_agreement=kind_ok, runs=per_run)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _fmt(d):
    return json.dumps(d, indent=2, default=lambda o: None if o != o else o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=None, help='write JSON results here')
    ap.add_argument('--quick', action='store_true', help='small synthetic grid')
    ap.add_argument('--sensitivity', type=int, nargs='*', default=[5],
                    help="this checkout's detector at these sensitivities")
    ap.add_argument('--legacy', type=Path, default=None,
                    help='also score an older logic/processor.py at --legacy-sensitivity')
    ap.add_argument('--legacy-sensitivity', type=int, nargs='*', default=[8])
    ap.add_argument('--labels', type=Path, default=None, help='labels.json from the label sheet')
    args = ap.parse_args(argv)

    detectors = {f'new_s{s}': (s, None) for s in args.sensitivity}
    if args.legacy:
        detectors.update({f'legacy_s{s}': (s, args.legacy) for s in args.legacy_sensitivity})
    cases = synthetic_cases(quick=args.quick)
    results = {}
    for name, (sens, pfile) in detectors.items():
        scores = run(processor_detector(sensitivity=sens, processor_file=pfile), cases)
        res = dict(summary=summarize(scores),
                   **{f'shoulder_by_{k}': breakdown(scores, k, 'shoulder')
                      for k in ('rs', 'ratio', 'snr', 'child_snr', 'sigma_pts', 'tau_ratio')},
                   resolved_by_rs=breakdown(scores, 'rs', 'resolved'),
                   rider_by_ratio=breakdown(scores, 'ratio', 'rider'),
                   rider_by_snr=breakdown(scores, 'snr', 'rider'),
                   rider_by_child_snr=breakdown(scores, 'child_snr', 'rider'))
        if args.labels:
            res['real'] = score_labelled(args.labels, sens, pfile)
        results[name] = res
        s = res['summary']
        print(f"{name:12s} P={s['all']['precision']:.3f} R={s['all']['recall']:.3f} "
              f"shoulderR={s['shoulder']['recall']:.2f} riderR={s['rider']['recall']:.2f} "
              f"single_false_split={s['single']['false_split_rate']:.3f}"
              + (f" | real P={res['real']['precision']:.3f} R={res['real']['recall']:.3f}" if args.labels else ''),
              flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'results.json').write_text(_fmt(results))


if __name__ == '__main__':
    main()
