import pytest

from logic import windows as W


def win(i, s, e, **kw):
    d = {'id': i, 'label': i, 'start': s, 'end': e, 'smoothing': {}, 'peaks': {},
         'shoulders': {}, 'negative_peaks': {}, 'events': []}
    d.update(kw)
    return d


def test_validate_rejects_overlap():
    with pytest.raises(ValueError) as e:
        W.validate_windows([win('w1', 1, 5), win('w2', 4, 8)])
    assert 'w1' in str(e.value) and 'w2' in str(e.value)


def test_validate_allows_touching():
    W.validate_windows([win('w1', 1, 5), win('w2', 5, 8)])


def test_validate_rejects_bad_span_and_duplicate_ids():
    with pytest.raises(ValueError, match='w1'):
        W.validate_windows([win('w1', 5, 5)])
    with pytest.raises(ValueError, match='w1'):
        W.validate_windows([win('w1', 1, 2), win('w1', 3, 4)])


def test_from_range_filters_merges_only_overlaps():
    wins, msgs = W.windows_from_range_filters([[1, 5], [4, 8], [8, 9]])
    assert [(w['start'], w['end']) for w in wins] == [(1, 8), (8, 9)]
    assert [w['id'] for w in wins] == ['w1', 'w2']
    assert len(msgs) == 1
    assert W.windows_from_range_filters(None) == ([], [])
    assert W.windows_from_range_filters([]) == ([], [])
    wins, _ = W.windows_from_range_filters([[8, 9], [1, 2], [3]])
    assert [(w['start'], w['end']) for w in wins] == [(1, 2), (8, 9)]


def test_resolve_prefers_peak_windows_key_even_if_empty():
    params = {'peaks': {'range_filters': [[1, 2]]}, 'peak_windows': []}
    assert W.resolve_windows(params) == ([], 'ignore')
    wins, outside = W.resolve_windows({'peaks': {'range_filters': [[1, 2]]}})
    assert len(wins) == 1 and outside == 'ignore'
    wins, _ = W.resolve_windows({'peaks': {}, 'peak_windows': [win('b', 5, 6), win('a', 1, 2)]})
    assert [w['id'] for w in wins] == ['a', 'b']


def test_resolve_rejects_bad_outside():
    with pytest.raises(ValueError, match='outside'):
        W.resolve_windows({'peaks': {}, 'outside_windows': 'bogus'})


def test_window_at_inclusive_and_later_wins():
    ws = [win('w1', 1, 5), win('w2', 5, 8)]
    assert W.window_at(ws, 5.0)['id'] == 'w2'
    assert W.window_at(ws, 8.0)['id'] == 'w2'
    assert W.window_at(ws, 1.0)['id'] == 'w1'
    assert W.window_at(ws, 8.01) is None


def test_window_at_descending_axis():
    ws = [win('w1', 1, 5), win('w2', 5, 8)]
    for rt, exp in [(5.0, 'w2'), (8.0, 'w2'), (8.01, None), (3.0, 'w1')]:
        got = W.window_at(ws, rt)
        assert (got['id'] if got else None) == exp
    assert W.window_at(list(reversed(ws)), 5.0)['id'] == 'w2'


def test_apex_clamp_ignore_vs_global():
    iso = [win('w1', 1, 5)]
    assert W.apex_clamp(iso, 3.0, 'ignore') == (1, 5)
    assert W.apex_clamp(iso, 3.0, 'global') == (None, None)
    assert W.apex_clamp(iso, 9.0, 'ignore') == (None, None)
    tw = [win('w1', 1, 5), win('w2', 5, 8)]
    assert W.apex_clamp(tw, 3.0, 'global') == (None, 5)
    assert W.apex_clamp(tw, 6.0, 'global') == (5, None)


def test_merge_overrides_skips_none_and_merges_fastchrom():
    base = {'method': 'asls', 'lambda': 1e5, 'fastchrom': {'half_window': 10, 'smooth_half_window': 5}}
    over = {'method': None, 'lambda': 1e3, 'fastchrom': {'half_window': 20, 'smooth_half_window': None}}
    out = W.merge_overrides(base, over)
    assert out == {'method': 'asls', 'lambda': 1e3,
                   'fastchrom': {'half_window': 20, 'smooth_half_window': 5}}
    assert base['lambda'] == 1e5 and base['fastchrom']['half_window'] == 10
    assert W.merge_overrides(base, None) == base
    out['fastchrom']['half_window'] = 99
    assert base['fastchrom']['half_window'] == 10


def test_window_params_only_touches_detection_sections():
    params = {'smoothing': {'enabled': True, 'lambda': 1e4}, 'baseline': {'lambda': 1e5},
              'peaks': {'min_prominence': 0.1}, 'shoulders': {}, 'negative_peaks': {}}
    w = win('w1', 1, 2, smoothing={'lambda': 5.0}, peaks={'min_prominence': 0.5},
            baseline={'lambda': 1.0})
    out = W.window_params(params, w)
    assert out['smoothing']['lambda'] == 5.0 and out['smoothing']['enabled'] is True
    assert out['peaks']['min_prominence'] == 0.5
    assert out['baseline'] == {'lambda': 1e5}
    assert params['smoothing']['lambda'] == 1e4


def test_overrides_any():
    assert not W.overrides_any(win('w', 1, 2, smoothing={'lambda': None}))
    assert W.overrides_any(win('w', 1, 2, peaks={'min_width': 0.1}))
    assert not W.overrides_any(win('w', 1, 2, peaks={'min_width': 0.1}), sections=('smoothing',))


def test_resolve_break_points_anchor_sets_time():
    ws = [win('w1', 12.55, 13.10)]
    bps = [{'time': 0.0, 'anchor': {'window_id': 'w1', 'edge': 'start'}},
           {'time': 20.0, 'tolerance': 0.3}, 15.0]
    out = W.resolve_break_points(ws, bps)
    assert [b['time'] for b in out] == [12.55, 15.0, 20.0]
    assert out[0]['anchor'] == {'window_id': 'w1', 'edge': 'start'}
    assert out[2]['tolerance'] == 0.3
    assert set(out[1]) == {'time', 'tolerance', 'overrides', 'anchor'}
    end = W.resolve_break_points(ws, [{'time': 0, 'anchor': {'window_id': 'w1', 'edge': 'end'}}])
    assert end[0]['time'] == 13.10


def test_resolve_break_points_dangling_raises():
    with pytest.raises(ValueError) as e:
        W.resolve_break_points([], [{'time': 7.5, 'anchor': {'window_id': 'w9', 'edge': 'end'}}])
    assert 'w9' in str(e.value) and '7.5' in str(e.value)


def test_resolve_break_points_duplicate_raises():
    with pytest.raises(ValueError, match='5.0'):
        W.resolve_break_points([], [5.0, {'time': 5.0}])


def test_section_overrides_latest_at_or_before():
    bps = [{'time': 1, 'overrides': {'lambda': 1e3}}, {'time': 2, 'overrides': {}}]
    idx = [100, 300]
    assert W.section_overrides(50, idx, bps) == {}
    assert W.section_overrides(100, idx, bps) == {'lambda': 1e3}
    assert W.section_overrides(299, idx, bps) == {'lambda': 1e3}
    assert W.section_overrides(300, idx, bps) == {}


def test_resolve_windows_does_not_share_caller_subdicts():
    src = [win('w1', 1, 2, peaks={'min_width': 0.1}, events=[{'t': 1}])]
    wins, _ = W.resolve_windows({'peaks': {}, 'peak_windows': src})
    wins[0]['peaks']['min_width'] = 9
    wins[0]['events'].append({'t': 2})
    assert src[0]['peaks'] == {'min_width': 0.1}
    assert src[0]['events'] == [{'t': 1}]


def test_break_point_overrides_do_not_share_caller_dicts():
    bps = [{'time': 1.0, 'overrides': {'fastchrom': {'half_window': 10}}}]
    out = W.resolve_break_points([], bps)
    out[0]['overrides']['fastchrom']['half_window'] = 99
    assert bps[0]['overrides']['fastchrom']['half_window'] == 10
    sec = W.section_overrides(0, [0], out)
    sec['fastchrom']['half_window'] = 7
    assert out[0]['overrides']['fastchrom']['half_window'] == 99


# ---- smoothing stitch ----
import numpy as np


def _traces():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 1001)
    base = np.sin(x) + 0.0 * x
    own = base + 0.5 + rng.normal(0, 0.01, x.size)
    return x, base, own


def test_stitch_interior_equals_own_trace():
    x, base, own = _traces()
    w = win('w1', 3, 7)
    out = W.stitch_smoothed(x, base, [(w, own, 20, 20)])
    inside = (x >= 3) & (x <= 7)
    core = inside & (x > 3.3) & (x < 6.7)
    assert np.array_equal(out[core], own[core])
    assert np.array_equal(out[~inside], base[~inside])


def test_stitch_no_step_at_edge():
    x, base, own = _traces()
    k = 20
    w = win('w1', 3, 7)
    out = W.stitch_smoothed(x, base, [(w, own, k, k)])
    idx = np.flatnonzero((x >= 3) & (x <= 7))
    for edge in (idx[0], idx[-1] + 1):
        seg = out[edge - k:edge + k]
        assert np.abs(np.diff(out[edge - 1:edge + 1])).max() <= np.abs(np.diff(seg)).max()
    # unblended it would step by ~0.5; blended the edge jump is small
    assert abs(out[idx[0]] - out[idx[0] - 1]) < 0.1


def test_blend_descending_axis():
    x, base, own = _traces()
    xd = x[::-1].copy()
    w = win('w1', 3, 7)
    asc = W.stitch_smoothed(x, base, [(w, own, 15, 25)])
    desc = W.stitch_smoothed(xd, base[::-1], [(w, own[::-1], 15, 25)])
    assert np.allclose(desc[::-1], asc)


def test_blend_points_rules():
    sav21 = {'enabled': True, 'method': 'savgol', 'savgol_window': 21}
    whit = {'enabled': True, 'method': 'whittaker'}
    off = {'enabled': False}
    assert W.blend_points(sav21, whit, 1000) == 21
    assert W.blend_points(off, off, 1000) == 0
    assert W.blend_points(off, whit, 1000) == 11
    assert W.blend_points(sav21, whit, 40) == 10


def test_stitch_touching_windows_blend_only_in_later_window():
    x = np.linspace(0, 10, 1001)
    base = np.zeros_like(x)
    A, B, k = np.full_like(x, 1.0), np.full_like(x, 3.0), 10
    w1, w2 = win('w1', 2, 5), win('w2', 5, 8)
    out = W.stitch_smoothed(x, base, [(w1, A, 10, 10), (w2, B, 10, 10)])
    edge = int(np.flatnonzero(x >= 5)[0])
    assert np.abs(np.diff(out[edge - k:edge + k + 1])).max() <= abs(3.0 - 1.0) / (k + 1) + 1e-9
    earlier = (x >= 2) & (x < 5)
    # earlier window's end is pure A right up to the shared edge
    assert np.all(out[earlier & (x > 2.2)] == 1.0)
    assert out[edge - 1] == 1.0


# ---- rider block in process() result ----
def test_process_returns_rider_block():
    import copy
    from logic.processor import ChromatogramProcessor
    proc = ChromatogramProcessor()
    x = np.linspace(0, 10, 1001)
    y = 100 * np.exp(-0.5 * ((x - 5.0) / 0.1) ** 2)
    params = copy.deepcopy(proc.default_params)
    params['baseline']['enabled'] = False
    params['smoothing']['enabled'] = False
    assert proc.process(x, y, copy.deepcopy(params))['rider'] is None
    params['rider'] = {'rider_integration': 'auto'}
    out = proc.process(x, y, params)
    assert out['rider']['rider_integration'] == 'auto'
