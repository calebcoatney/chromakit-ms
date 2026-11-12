"""Synthetic cases that pin the processor's output before piecewise methods.

``tests/data/piecewise_parity.json`` was generated from ``main`` at the branch
point of feat/piecewise-methods-phase1. Methods without windows must keep
reproducing it. Regenerate only on purpose:

    python -m tests.logic.piecewise_parity_cases --write
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

from logic.processor import ChromatogramProcessor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "piecewise_parity.json"


def _gauss(x, c, s, a):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def _base_params(**peaks):
    params = copy.deepcopy(ChromatogramProcessor().default_params)
    params['baseline']['enabled'] = False
    params['smoothing']['enabled'] = False
    params['peaks'].update({'min_prominence': 0.02, 'min_width': 0})
    params['peaks'].update(peaks)
    return params


def _trace(peaks, noise=0.0, n=6001, span=(0.0, 10.0), seed=0):
    x = np.linspace(span[0], span[1], n)
    y = sum(_gauss(x, c, s, a) for c, s, a in peaks)
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, n)
    return x, y


_FIVE = [(2.0, 0.08, 100), (3.5, 0.08, 60), (5.8, 0.08, 80), (7.0, 0.08, 40), (8.5, 0.08, 70)]


def plain():
    x, y = _trace(_FIVE, noise=0.05)
    return x, y, _base_params(), None


def range_single():
    x, y = _trace(_FIVE, noise=0.05)
    return x, y, _base_params(range_filters=[[3.0, 7.5]]), None


def range_touching():
    # Apexes at 2, 3.5, 5.8, 7, 8.5: none sits on the shared edge at 5.0.
    x, y = _trace(_FIVE, noise=0.05)
    return x, y, _base_params(range_filters=[[2, 5], [5, 8]]), None


def break_points_ms():
    x, y = _trace(_FIVE, noise=0.05)
    params = _base_params()
    params['baseline'].update({
        'enabled': True,
        'method': 'asls',
        'lambda': 1e5,
        'break_points': [{'time': 3.3, 'tolerance': 0.1}, {'time': 5.65, 'tolerance': 0.1}],
    })
    return x, y, params, (1.0, 9.0)


def shoulders_on():
    # A shoulder 0.12 min after a main peak, plus an isolated peak.
    x, y = _trace([(3.0, 0.08, 100), (3.2, 0.06, 35), (6.0, 0.08, 60)], noise=0.02)
    params = _base_params()
    params['shoulders'] = {'enabled': True, 'window_length': 41, 'polyorder': 3,
                           'sensitivity': 8, 'apex_distance': 10}
    return x, y, params, None


def negative_with_range():
    x, y = _trace(_FIVE + [(4.5, 0.08, -50)], noise=0.05)
    params = _base_params(range_filters=[[3.0, 7.5]])
    params['negative_peaks'] = {'enabled': True, 'min_prominence': 0.02}
    return x, y, params, None


def whittaker_noisy():
    x, y = _trace(_FIVE, noise=2.0)
    params = _base_params()
    params['smoothing'].update({'enabled': True, 'method': 'whittaker', 'lambda': 50.0, 'diff_order': 2})
    return x, y, params, None


def anh_like():
    x = np.linspace(0, 10, 6001)
    big = _gauss(x, 4.0, 0.05, 1000)
    tail = np.where(x > 4.0, 1000 * 0.1 * np.exp(-(x - 4.0) / 0.4), 0.0)
    small = sum(_gauss(x, c, 0.03, 15) for c in (4.3, 4.5, 4.65, 4.8, 5.0))
    y = big + tail + small + np.random.default_rng(0).normal(0, 0.1, x.size)
    params = _base_params(range_filters=[[4.55, 5.1]], min_prominence=5.0)
    params['baseline'].update({'enabled': True, 'method': 'asls', 'lambda': 1e5,
                               'break_points': [{'time': 4.55, 'tolerance': 0.1}]})
    return x, y, params, None


CASES = {f.__name__: f for f in (
    plain, range_single, range_touching, break_points_ms,
    shoulders_on, negative_with_range, whittaker_noisy, anh_like,
)}


def run_case(name):
    x, y, params, ms_range = CASES[name]()
    proc = ChromatogramProcessor()
    processed = proc.process(x, y, params, ms_range=ms_range)
    result = proc.integrate_peaks(processed)
    return [
        {'rt': float(p.retention_time), 'area': float(p.area),
         'start': float(p.start_time), 'end': float(p.end_time)}
        for p in result['peaks']
    ]


def write():
    data = {name: run_case(name) for name in CASES}
    empty = [n for n, v in data.items() if not v]
    if empty:
        raise SystemExit(f"cases with no peaks: {empty}")
    DATA_PATH.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    for n, v in data.items():
        print(n, len(v), [round(p['rt'], 3) for p in v])


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("pass --write to regenerate tests/data/piecewise_parity.json")
    write()
