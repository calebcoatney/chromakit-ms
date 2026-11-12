"""Per-window detection settings, window membership and the per-peak clamp."""
import copy

import numpy as np
import pytest

from logic.processor import ChromatogramProcessor
from tests.logic import piecewise_parity_cases as cases


def win(wid, start, end, **sections):
    w = {'id': wid, 'label': wid, 'start': start, 'end': end, 'smoothing': {},
         'peaks': {}, 'shoulders': {}, 'negative_peaks': {}, 'events': []}
    w.update(sections)
    return w


def _gauss(x, c, s, a):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def _params(windows, outside='ignore', **peaks):
    params = copy.deepcopy(ChromatogramProcessor().default_params)
    params['baseline']['enabled'] = False
    params['smoothing']['enabled'] = False
    params['peaks'].update({'min_prominence': 0.05, 'min_width': 0})
    params['peaks'].update(peaks)
    params['peak_windows'] = windows
    params['outside_windows'] = outside
    return params


def _run(x, y, params, ms_range=None):
    proc = ChromatogramProcessor()
    processed = proc.process(x, y, params, ms_range=ms_range)
    return processed, proc.integrate_peaks(processed)['peaks']


def _rts(peaks):
    return [round(p.retention_time, 2) for p in peaks]


def _anh():
    x, y, params, _ = cases.anh_like()
    params['peaks']['min_prominence'] = 0.05
    params['peaks'].pop('range_filters')
    return x, y, params


def test_window_prominence_is_relative_to_window_range():
    x, y, params = _anh()
    params['peak_windows'] = []
    plain, _ = _run(x, y, params)
    in_window = (plain['peaks_x'] >= 4.55) & (plain['peaks_x'] <= 5.1)
    assert not in_window.any(), "globals must miss the cluster peaks in the window"

    params['peak_windows'] = [win('w1', 4.55, 5.1, peaks={'min_prominence': 0.05})]
    over, _ = _run(x, y, params)
    assert len(over['peaks_x']) >= 3
    assert np.all((over['peaks_x'] >= 4.55) & (over['peaks_x'] <= 5.1))

    params['peak_windows'] = [win('w1', 4.55, 5.1)]
    inherit, _ = _run(x, y, params)
    inside = plain['peaks_x'][(plain['peaks_x'] >= 4.55) & (plain['peaks_x'] <= 5.1)]
    np.testing.assert_array_equal(inherit['peaks_x'], inside)


@pytest.fixture
def pair():
    x = np.linspace(0, 10, 6001)
    # Two overlapping peaks (valley well above baseline) and one isolated peak.
    y = _gauss(x, 4.0, 0.08, 100) + _gauss(x, 4.35, 0.08, 80) + _gauss(x, 7.0, 0.08, 50)
    return x, y


def test_ignore_reports_only_window_peaks_and_keeps_valley_neighbour(pair):
    x, y = pair
    _, unfiltered = _run(x, y, _params([]))
    valley = {round(p.retention_time, 2): p for p in unfiltered}[4.35].start_time

    for start in (4.1, valley + 0.02):
        params = _params([win('w1', start, 5.0, peaks={'min_prominence': 0.01})])
        processed, peaks = _run(x, y, params)
        assert _rts(peaks) == [4.35]
        assert 4.0 in np.round(processed['range_excluded_peaks_x'], 2)
        assert 7.0 in np.round(processed['range_excluded_peaks_x'], 2)
        assert peaks[0].start_time == pytest.approx(
            x[np.flatnonzero(x >= max(valley, start))[0]])
        assert peaks[0].end_time <= 5.0


@pytest.fixture
def mixed():
    x = np.linspace(0, 10, 6001)
    y = (_gauss(x, 2.0, 0.08, 100) + _gauss(x, 3.0, 0.08, 3)
         + _gauss(x, 6.0, 0.08, 100) + _gauss(x, 6.5, 0.08, 3))
    return x, y


def test_global_mode_keeps_outside_peaks_with_global_settings(mixed):
    x, y = mixed
    w = win('w1', 5.5, 7.0, peaks={'min_prominence': 0.01})
    _, peaks = _run(x, y, _params([w], outside='global'))
    # 3.0 is too small for the global 5 % threshold; 6.5 clears the window's 1 %.
    assert _rts(peaks) == [2.0, 6.0, 6.5]

    _, peaks = _run(x, y, _params([w], outside='ignore'))
    assert _rts(peaks) == [6.0, 6.5]


def test_global_mode_clamps_only_shared_edges():
    x = np.linspace(0, 10, 6001)
    y = (_gauss(x, 3.1, 0.08, 100) + _gauss(x, 4.9, 0.08, 100)
         + _gauss(x, 6.0, 0.08, 100) + _gauss(x, 8.5, 0.3, 100))
    windows = [win('w1', 3.0, 5.0), win('w2', 5.0, 7.0), win('w3', 8.0, 9.0)]
    processed, peaks = _run(x, y, _params(windows, outside='global'))
    by_rt = {round(p.retention_time, 1): p for p in peaks}
    assert sorted(by_rt) == [3.1, 4.9, 6.0, 8.5]
    assert by_rt[3.1].start_time < 3.0          # outer edge of w1: no wall
    assert by_rt[4.9].end_time <= 5.0           # shared edge w1|w2: clamped
    assert by_rt[8.5].start_time < 8.0 and by_rt[8.5].end_time > 9.0
    clamps = {round(m['x'], 1): m['clamp'] for m in processed['peak_metadata']}
    assert clamps == {3.1: [None, 5.0], 4.9: [None, 5.0], 6.0: [5.0, None],
                      8.5: [None, None]}

    processed, peaks = _run(x, y, _params(windows, outside='ignore'))
    by_rt = {round(p.retention_time, 1): p for p in peaks}
    assert by_rt[3.1].start_time >= 3.0
    assert 8.0 <= by_rt[8.5].start_time and by_rt[8.5].end_time <= 9.0


def test_window_id_and_clamp_in_metadata(mixed):
    x, y = mixed
    windows = [win('w1', 1.5, 2.5), win('late', 5.5, 7.0, peaks={'min_prominence': 0.01})]
    processed, peaks = _run(x, y, _params(windows))
    meta = processed['peak_metadata']
    assert [(round(m['x'], 1), m['window_id'], m['clamp']) for m in meta] == [
        (2.0, 'w1', [1.5, 2.5]), (6.0, 'late', [5.5, 7.0]), (6.5, 'late', [5.5, 7.0])]
    assert [p.window_id for p in peaks] == ['w1', 'late', 'late']
    assert [w['id'] for w in processed['peak_windows']] == ['w1', 'late']
    assert processed['outside_windows'] == 'ignore'
    assert processed['range_filters'] == [[1.5, 2.5], [5.5, 7.0]]

    processed, _ = _run(x, y, _params(windows[1:], outside='global'))
    meta = {round(m['x'], 1): m for m in processed['peak_metadata']}
    assert meta[2.0]['window_id'] is None and meta[2.0]['clamp'] == [None, None]
    assert processed['range_filters'] == []


def test_negative_peaks_follow_outside_mode():
    x = np.linspace(0, 10, 6001)
    y = (_gauss(x, 2.0, 0.08, 100) - _gauss(x, 3.0, 0.08, 50)
         + _gauss(x, 6.0, 0.08, 100) - _gauss(x, 6.5, 0.08, 50))
    windows = [win('w1', 5.5, 7.0)]
    params = _params(windows)
    params['negative_peaks'] = {'enabled': True, 'min_prominence': 0.05}

    processed, peaks = _run(x, y, params)
    assert _rts(peaks) == [6.0, 6.5]
    assert {2.0, 3.0} <= set(np.round(processed['range_excluded_peaks_x'], 2))

    params['outside_windows'] = 'global'
    processed, peaks = _run(x, y, params)
    assert _rts(peaks) == [2.0, 3.0, 6.0, 6.5]
    neg = {round(m['x'], 1): m for m in processed['peak_metadata'] if m.get('is_negative')}
    assert neg[3.0]['window_id'] is None and neg[6.5]['window_id'] == 'w1'


def test_no_spurious_peak_at_window_edge():
    x = np.linspace(0, 10, 6001)
    rng = np.random.default_rng(3)
    y = _gauss(x, 5.0, 0.1, 100) + _gauss(x, 6.0, 0.05, 20) + rng.normal(0, 0.5, x.size)
    edge = 5.0 - 0.1 * np.sqrt(2 * np.log(2))   # half height of the 5.0 flank
    params = _params([win('w1', edge, 7.0, peaks={'min_prominence': 0.01})])
    params['smoothing'].update({'enabled': True, 'method': 'savgol',
                                'savgol_window': 21, 'savgol_polyorder': 2})
    processed, _ = _run(x, y, params)
    edge_idx = np.searchsorted(x, edge)
    kept_idx = np.searchsorted(x, processed['peaks_x'])
    assert np.round(processed['peaks_x'], 1).tolist().count(5.0) == 1
    assert 6.0 in np.round(processed['peaks_x'], 1)
    assert not np.any(np.abs(kept_idx - edge_idx) <= 3), processed['peaks_x']


def test_two_windows_disjoint_settings():
    x = np.linspace(0, 10, 6001)
    y = (_gauss(x, 2.0, 0.01, 100) + _gauss(x, 3.0, 0.1, 100)
         + _gauss(x, 6.0, 0.01, 100) + _gauss(x, 7.0, 0.1, 100))
    # FWHM is about 14 samples for the narrow peaks and 141 for the wide ones.
    windows = [win('narrow_out', 1.0, 4.0, peaks={'min_width': 50}),
               win('keep_all', 5.0, 8.0, peaks={'min_width': 2})]
    _, peaks = _run(x, y, _params(windows, min_width=200))
    assert _rts(peaks) == [3.0, 6.0, 7.0]


def test_legacy_processed_without_clamp_uses_range_filters(pair):
    x, y = pair
    proc = ChromatogramProcessor()
    processed = proc.process(x, y, _params([win('w1', 6.9, 7.03)]))
    for m in processed['peak_metadata']:
        m.pop('clamp')
    peaks = proc.integrate_peaks(processed)['peaks']
    assert _rts(peaks) == [7.0]
    assert peaks[0].start_time >= 6.9 and peaks[0].end_time <= 7.03


def test_window_pass_ms_excluded_apex_outside_window_is_not_a_neighbour():
    # Baseline off, so the pre-MS region keeps its real signal. The 2.97 bump
    # is below the global threshold but clears the window's 1 % override; it
    # lies before MS-on and outside the window, so it must not bound 3.3.
    x = np.linspace(0, 10, 6001)
    y = (_gauss(x, 1.0, 0.08, 100) + _gauss(x, 2.97, 0.03, 4)
         + _gauss(x, 3.3, 0.1, 100))
    w = win('w1', 3.1, 5.0, peaks={'min_prominence': 0.01})
    params = _params([w], outside='global')

    processed, peaks = _run(x, y, params, ms_range=(3.0, 10.0))
    excluded = np.round(processed['range_excluded_peaks_x'], 2)
    assert 1.0 in excluded and 2.97 not in excluded
    kept = {round(p.retention_time, 2): p for p in peaks}
    assert sorted(kept) == [3.3]

    # Without ms_range nothing is MS-excluded: the 1.0 peak is a kept global
    # neighbour and the 2.97 bump is dropped by the window pass, as above.
    _, ref = _run(x, y, params)
    ref = {round(p.retention_time, 2): p for p in ref}
    assert kept[3.3].start_time == pytest.approx(ref[3.3].start_time)


@pytest.mark.parametrize('global_smoothing', [
    {'enabled': False},
    {'enabled': True, 'method': 'savgol', 'savgol_window': 5, 'savgol_polyorder': 2},
])
def test_window_smoothing_reaches_negative_detection(global_smoothing):
    x = np.linspace(0, 10, 6001)
    rng = np.random.default_rng(7)
    y = (_gauss(x, 2.0, 0.1, 100) + _gauss(x, 6.0, 0.1, 100)
         - _gauss(x, 6.6, 0.1, 50) + rng.normal(0, 1.0, x.size))
    heavy = {'enabled': True, 'method': 'savgol', 'savgol_window': 101, 'savgol_polyorder': 2}
    params = _params([win('w1', 5.5, 7.5, smoothing=heavy)])
    params['smoothing'].update(global_smoothing)
    # Absolute prominence 3: raw noise (sd 1) has dips that deep, the heavily
    # smoothed trace does not.
    params['negative_peaks'] = {'enabled': True, 'min_prominence': 3.0}
    processed, _ = _run(x, y, params)
    neg = [round(m['x'], 1) for m in processed['peak_metadata'] if m.get('is_negative')]
    assert neg == [6.6]
