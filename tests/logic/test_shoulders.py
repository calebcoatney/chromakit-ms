"""logic.shoulders: shoulder and rider detection on known EMG mixtures."""
import numpy as np
import pytest
from scipy.signal import find_peaks
from scipy.stats import exponnorm

from logic.shoulders import EMG, detect_shoulders, emg, k_prom_for, noise_sd

DT = 1.0 / 1200.0          # 20 Hz, minutes
SIGMA = 12 * DT


def _peak(x, mu, h, sigma=SIGMA, tau=0.0):
    if tau <= 0:
        y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    else:
        y = exponnorm.pdf(x, tau / sigma, loc=mu, scale=sigma)
    return h * y / y.max()


def _run(components, noise=50.0, seed=0, **kw):
    x = np.arange(-0.15, 0.35, DT)
    y = sum(_peak(x, *c) for c in components)
    y = y + np.random.default_rng(seed).normal(0, noise, len(x))
    peaks, _ = find_peaks(y, prominence=10 * noise)
    return x, y, peaks, detect_shoulders(x, y, peaks, **kw)


def test_emg_matches_scipy_and_stays_finite_on_a_long_tail():
    t = np.linspace(-0.05, 0.5, 4001)
    ours = emg(t, 1.0, 0.0, 0.002, 0.05)
    ref = exponnorm.pdf(t, 0.05 / 0.002, loc=0.0, scale=0.002)
    assert np.all(np.isfinite(ours))
    np.testing.assert_allclose(ours, ref, atol=1e-9)


def test_noise_sd_ignores_peaks():
    x = np.arange(0, 1, DT)
    y = _peak(x, 0.5, 1e5) + np.random.default_rng(1).normal(0, 100, len(x))
    assert noise_sd(y) == pytest.approx(100, rel=0.1)


def test_sensitivity_maps_monotonically():
    assert k_prom_for(1) > k_prom_for(5) > k_prom_for(10)


def test_finds_a_shoulder_with_no_valley_and_splits_between_the_apexes():
    gap = 2.6 * SIGMA                                    # Rs ~0.65
    x, y, peaks, calls = _run([(0.0, 1e5), (gap, 3e4)])
    assert len(peaks) == 1                               # no valley: find_peaks sees one peak
    assert len(calls) == 1
    c = calls[0]
    assert c.kind == 'shoulder' and c.side == 'after'
    assert abs(x[c.apex_index] - gap) < 0.5 * SIGMA
    assert 0.0 < x[c.split_index] < gap
    assert c.outer_index > c.apex_index


def test_shoulder_before_the_parent():
    x, y, peaks, calls = _run([(0.0, 1e5), (-2.6 * SIGMA, 4e4)])
    assert [c.side for c in calls] == ['before']


@pytest.mark.parametrize('tau_ratio', [0.0, 2.0, 6.0])
def test_a_single_tailing_peak_is_not_split(tau_ratio):
    for seed in range(5):
        _, _, _, calls = _run([(0.0, 1e5, SIGMA, tau_ratio * SIGMA)], seed=seed)
        assert calls == []


def test_a_resolved_pair_gets_no_extra_shoulder():
    x, y, peaks, calls = _run([(0.0, 1e5), (6 * SIGMA, 5e4)])
    assert len(peaks) == 2
    assert calls == []


def test_a_peak_on_a_long_tail_is_a_rider_that_starts_before_its_apex():
    parent = (0.0, 1e5, SIGMA, 6 * SIGMA)
    rider_rt = 0.08
    x = np.arange(-0.15, 0.35, DT)
    y = _peak(x, *parent) + _peak(x, rider_rt, 3e3) + np.random.default_rng(0).normal(0, 20, len(x))
    peaks, _ = find_peaks(y, prominence=200)
    big = peaks[np.argmax(y[peaks])]
    calls = detect_shoulders(x, y, [big])
    assert len(calls) == 1
    c = calls[0]
    assert c.kind == 'rider'
    assert abs(x[c.apex_index] - rider_rt) < SIGMA
    assert x[c.split_index] < x[c.apex_index]
    # the parent's fitted tail is what a skim would follow under the rider
    assert isinstance(c.parent, EMG) and c.parent(np.array([rider_rt]))[0] > 3e3


def test_no_peaks_means_no_calls():
    x = np.arange(0, 1, DT)
    assert detect_shoulders(x, np.zeros_like(x), []) == []


def test_processor_marks_the_shoulder_and_integration_splits_the_area():
    from logic.processor import ChromatogramProcessor
    x = np.arange(-0.15, 0.35, DT)
    gap = 2.6 * SIGMA
    y = _peak(x, 0.0, 1e5) + _peak(x, gap, 3e4) + np.random.default_rng(3).normal(0, 50, len(x))
    params = {
        'smoothing': {'enabled': False},
        'baseline': {'enabled': False, 'method': 'asls', 'lambda': 1e6},
        'peaks': {'enabled': True, 'min_prominence': 500},
        'shoulders': {'enabled': True, 'sensitivity': 5},
    }
    proc = ChromatogramProcessor()
    out = proc.process(x, y, params)
    sh = [m for m in out['peak_metadata'] if m['is_shoulder']]
    assert len(sh) == 1
    m = sh[0]
    assert m['shoulder_kind'] == 'shoulder'
    assert m['left_bound'] == m['split_index'] < m['index'] < m['right_bound']
    result = proc.integrate_peaks(out)
    areas = sorted(p.area for p in result['peaks'])
    # true areas are in the height ratio 3:10 (same width)
    assert len(areas) == 2 and 0.2 < areas[0] / areas[1] < 0.45
    # the fitted components ride along as plain floats (JSON-safe)
    assert set(m['parent_fit']) == {'area', 'mu', 'sigma', 'tau'}
    assert all(isinstance(v, float) for v in m['child_fit'].values())


def test_processor_estimates_noise_from_real_samples_not_the_masked_fill(monkeypatch):
    """A NaN (MS-masked) stretch is filled flat before detection. Flat fill has
    no second difference, so estimating noise over it reads ~3x too low."""
    import logic.shoulders
    from logic.processor import ChromatogramProcessor
    seen = {}
    real = logic.shoulders.detect_shoulders

    def spy(*a, **kw):
        seen['noise'] = kw.get('noise')
        return real(*a, **kw)
    monkeypatch.setattr(logic.shoulders, 'detect_shoulders', spy)
    x = np.arange(-0.6, 0.35, DT)
    y = _peak(x, 0.0, 1e5) + _peak(x, 2.6 * SIGMA, 3e4) + np.random.default_rng(3).normal(0, 50, len(x))
    y[x < -0.2] = np.nan                       # ~40 % of the run masked
    params = {
        'smoothing': {'enabled': False},
        'baseline': {'enabled': False, 'method': 'asls', 'lambda': 1e6},
        'peaks': {'enabled': True, 'min_prominence': 500},
        'shoulders': {'enabled': True, 'sensitivity': 5},
    }
    out = ChromatogramProcessor().process(x, y, params)
    assert 40 < seen['noise'] < 65            # true SD 50; flat fill would read ~3x low
    assert sum(m['is_shoulder'] for m in out['peak_metadata']) == 1
