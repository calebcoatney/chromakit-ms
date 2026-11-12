import numpy as np
import pytest
from scipy.integrate import simpson

from logic.tail_skim import SkimCurve, exp_skim, rider_end, tail_bend, tangent_skim
from tests.logic.tail_skim_signals import emg_pair, processed_from, rider_on_tail


def _skim_args(p):
    x, y = p['x'], p['y']
    a = p['valley']
    b = rider_end(x, y, a, p['child_apex'], len(x) - 1)
    return x, y, a, b


def test_tail_bend_separates_symmetric_from_tailing():
    sym = emg_pair(parent_tau=0.002, child_tau=0.002, ratio=4, sep=0.05)
    tail = emg_pair(parent_tau=0.03, child_tau=0.002, ratio=4, sep=0.08)
    assert tail_bend(tail['x'], tail['y'], tail['apex'], tail['valley']) <= 1.05
    assert tail_bend(sym['x'], sym['y'], sym['apex'], sym['valley']) > 1.10


@pytest.mark.parametrize('kind, kw', [
    ('tail', dict(parent_tau=0.03, child_tau=0.002, ratio=4, sep=0.08)),
    ('sym', dict(parent_tau=0.002, child_tau=0.002, ratio=4, sep=0.05)),
])
def test_tail_bend_measured_on_a_coarse_grid(kind, kw):
    # ChromaKit resamples every run to 10,000 points, so a narrow parent's fall can cross the band in
    # three or four samples. The bend must still be measured, and match the bend at the full rate.
    fine = emg_pair(**kw)
    step = 6  # 0.05 s -> 0.3 s per sample
    x, y = fine['x'][::step], fine['y'][::step]
    lo = max(0, fine['apex'] // step - 2)
    apex = lo + int(np.argmax(y[lo:lo + 5]))
    valley = apex + int(np.argmin(y[apex:fine['child_apex'] // step + 1]))
    coarse = tail_bend(x, y, apex, valley)
    full = tail_bend(fine['x'], fine['y'], fine['apex'], fine['valley'])
    assert np.isfinite(coarse)
    assert abs(coarse - full) < 0.05
    assert (coarse <= 1.10) == (kind == 'tail')


def test_tail_bend_on_samples_when_the_back_side_has_a_gap():
    # An MS-gated (NaN) sample is not rebuilt across: the bend comes from the samples, as before.
    p = emg_pair(parent_tau=0.03, child_tau=0.002, ratio=4, sep=0.08)
    y = p['y'].copy()
    y[p['valley'] - 1] = np.nan
    assert np.isfinite(tail_bend(p['x'], y, p['apex'], p['valley']))


def test_tail_bend_nan_when_band_cannot_be_measured():
    x = np.linspace(0, 1, 50)
    y = np.exp(-x)
    assert np.isnan(tail_bend(x, y, 0, 49))


def test_rider_end_within_0p01_min_of_true_end():
    p = rider_on_tail(h=1.0, d_tau=3.0)
    b = rider_end(p['x'], p['y'], p['valley'], p['child_apex'], len(p['x']) - 1)
    assert abs(p['x'][b] - p['true_end']) < 0.01


def test_rider_end_returns_cap_when_range_empty():
    x = np.linspace(0, 1, 20)
    assert rider_end(x, np.exp(-x), 2, 10, 10) == 10


def test_exp_skim_passes_through_both_ends_and_stays_below_signal():
    p = rider_on_tail(h=1.0, d_tau=3.0)
    x, y, a, b = _skim_args(p)
    sc = exp_skim(x, y, a, b, p['apex'], len(x) - 1, [])
    assert isinstance(sc, SkimCurve)
    assert sc.model == 'exponential'
    assert sc.s.shape == (b - a + 1,)
    assert sc.s[0] == pytest.approx(y[a])
    assert sc.s[-1] == pytest.approx(y[b])
    assert np.all(sc.s <= y[a:b + 1] + 1e-12)
    assert np.all(np.isfinite(sc.tail(x[a:b + 1])))


@pytest.mark.parametrize('h, d', [(1.0, 1.5), (1.0, 3.0), (3.0, 1.5), (3.0, 3.0)])
def test_exp_skim_gaussian_rider_area_within_10pct(h, d):
    p = rider_on_tail(h=h, d_tau=d)
    x, y, a, b = _skim_args(p)
    exp = exp_skim(x, y, a, b, p['apex'], len(x) - 1, [])
    tan = tangent_skim(x, y, a, b)
    seg = slice(a, b + 1)
    area_exp = simpson(y[seg] - exp.s, x=x[seg])
    area_tan = simpson(y[seg] - tan.s, x=x[seg])
    assert abs(area_exp - p['true_child_area']) / p['true_child_area'] < 0.10
    assert area_tan < area_exp


def test_exp_skim_falls_back_to_2pt_then_tangent():
    x = np.arange(200) * 0.001
    y = np.exp(-x * 8) + 0.01
    # rising tail: y_a <= y_b, no decay to follow
    rising = np.linspace(1.0, 2.0, 200)
    assert exp_skim(x, rising, 80, 100, 0, 199, []).model == 'tangent'
    # too few downstream points (cap == b)
    sc = exp_skim(x, y, 80, 100, 0, 100, [])
    assert sc.model == 'exponential_2pt'
    assert sc.s[0] == pytest.approx(y[80])
    assert sc.s[-1] == pytest.approx(y[100])


def test_exp_skim_skips_points_inside_other_rider_spans():
    p = rider_on_tail(h=1.0, d_tau=3.0)
    x, y, a, b = _skim_args(p)
    sc = exp_skim(x, y, a, b, p['apex'], len(x) - 1, [(a - 5, a - 1)])
    assert sc.model in ('exponential', 'exponential_2pt')


def test_tangent_skim_is_clipped_chord():
    p = rider_on_tail(h=1.0, d_tau=3.0)
    x, y, a, b = _skim_args(p)
    sc = tangent_skim(x, y, a, b)
    assert sc.model == 'tangent'
    assert sc.s[0] == pytest.approx(y[a]) and sc.s[-1] == pytest.approx(y[b])
    assert np.all(sc.s <= y[a:b + 1] + 1e-12)


def test_processed_from_has_process_shape():
    pair = emg_pair(parent_tau=0.03, child_tau=0.002, ratio=4, sep=0.08)
    pd = processed_from(pair, extra_peaks=(0.6,))
    assert len(pd['peaks_x']) == 3
    assert np.all(pd['baseline_y'] == 0)
    assert pd['corrected_y'].shape == pair['x'].shape
