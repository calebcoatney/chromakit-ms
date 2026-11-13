"""logic.segments: per-segment editor records <-> peak_windows + baseline.break_points."""
import copy
from types import SimpleNamespace

import pytest

from logic import segments as S
from logic.method import ChromaMethod


def params_of(windows=(), breaks=None, **kw):
    """GUI params of a valid v2 method (built through ChromaMethod so they are normalised)."""
    d = {'name': 't', 'signal_type': 'gc', 'peak_windows': list(windows), **kw}
    if breaks is not None:
        d['baseline'] = {'break_points': breaks}
    return ChromaMethod.model_validate(d).to_gui_params()


def norm(params):
    return ChromaMethod.from_gui_params(params, 't', 'gc').to_gui_params()


def bps(params):
    return params['baseline']['break_points'] or []


def stripped(bp):
    return {k: v for k, v in bp['overrides'].items() if v is not None}


def anchored(params, wid, edge):
    return [bp for bp in bps(params)
            if bp.get('anchor') and bp['anchor'] == {'window_id': wid, 'edge': edge}]


def seg(params, wid):
    return next(r for r in S.segments_view(params)['segments'] if r['id'] == wid)


SPEC_WINDOW = {'id': 'w1', 'label': 'late cluster', 'start': 12.55, 'end': 13.10,
               'peaks': {'min_prominence': 0.02, 'min_width': 3}}
SPEC_BREAKS = [
    {'time': 12.55, 'anchor': {'window_id': 'w1', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
    {'time': 12.85, 'overrides': {'lambda': 1e4}},
    {'time': 13.10, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {}},
]


def test_view_of_spec_example():
    p = params_of([SPEC_WINDOW], SPEC_BREAKS)
    before = copy.deepcopy(p)
    view = S.segments_view(p)
    assert p == before
    assert view['whole_run_breaks'] == []
    [rec] = view['segments']
    assert rec['id'] == 'w1' and rec['label'] == 'late cluster'
    assert (rec['start'], rec['end']) == (12.55, 13.10)
    assert rec['peaks'] == {'min_prominence': 0.02, 'min_width': 3}
    assert rec['smoothing'] == {} and rec['rider'] == {}
    assert rec['fresh_baseline'] is True
    assert rec['baseline'] == {'lambda': 1e3}
    assert rec['partway'] == [{'time': 12.85, 'overrides': {'lambda': 1e4}}]
    assert rec['after']['mode'] == 'resume'


def test_apply_roundtrip_identity():
    p = params_of([SPEC_WINDOW], SPEC_BREAKS)
    before = copy.deepcopy(p)
    out = S.apply_segment(p, S.segments_view(p)['segments'][0])
    assert p == before
    n_in, n_out = norm(p), norm(out)
    assert n_out['peak_windows'] == n_in['peak_windows']
    assert n_out['baseline']['break_points'] == n_in['baseline']['break_points']


def test_partway_inherits_segment_method():
    p = params_of([{'id': 'w1', 'start': 12.0, 'end': 13.0}])
    rec = seg(p, 'w1')
    rec['fresh_baseline'] = True
    rec['baseline'] = {'method': 'asls', 'lambda': 1e3}
    rec['partway'] = [{'time': 12.5, 'overrides': {'lambda': 1e4}}]
    out = S.apply_segment(p, rec)
    [mid] = [bp for bp in bps(out) if bp['time'] == 12.5]
    assert stripped(mid) == {'method': 'asls', 'lambda': 1e4}
    assert seg(out, 'w1')['partway'] == [{'time': 12.5, 'overrides': {'lambda': 1e4}}]
    norm(out)  # still a valid method


def test_end_break_resumes_whole_run():
    p = params_of(breaks=[{'time': 4.01, 'overrides': {'method': 'arpls', 'lambda': 1e5}}])
    rec = S.new_segment(p, 12.0, 13.0)
    rec['fresh_baseline'] = True
    rec['baseline'] = {'lambda': 1e3}
    out = S.apply_segment(p, rec)
    [end] = anchored(out, rec['id'], 'end')
    assert end['time'] == 13.0
    assert stripped(end) == {'method': 'arpls', 'lambda': 1e5}
    assert seg(out, rec['id'])['after']['mode'] == 'resume'

    [whole] = S.segments_view(out)['whole_run_breaks']
    whole = dict(whole, overrides={'lambda': 1e6})
    out2 = S.set_whole_run_breaks(out, [whole])
    [end2] = anchored(out2, rec['id'], 'end')
    assert stripped(end2) == {'lambda': 1e6}
    assert seg(out2, rec['id'])['after']['mode'] == 'resume'
    norm(out2)


def test_touching_segments_end_break_tracks_neighbour():
    p = params_of()
    a = S.new_segment(p, 1.0, 2.0)
    a['fresh_baseline'], a['baseline'] = True, {'lambda': 1e3}
    p1 = S.apply_segment(p, a)
    assert len(anchored(p1, a['id'], 'end')) == 1
    b = S.new_segment(p1, 2.0, 3.0)
    assert b['id'] != a['id']
    b['fresh_baseline'], b['baseline'] = True, {'lambda': 1e5}
    p2 = S.apply_segment(p1, b)
    assert anchored(p2, a['id'], 'end') == []
    assert len(anchored(p2, b['id'], 'start')) == 1
    norm(p2)

    b2 = seg(p2, b['id'])
    b2['fresh_baseline'], b2['baseline'] = False, {}
    p3 = S.apply_segment(p2, b2)
    [a_end] = anchored(p3, a['id'], 'end')
    assert a_end['time'] == 2.0
    assert anchored(p3, b['id'], 'start') == [] and anchored(p3, b['id'], 'end') == []
    norm(p3)


def test_foreign_breaks_survive_other_edits():
    p = params_of(
        [{'id': 'w1', 'start': 2.0, 'end': 3.0}, {'id': 'w2', 'start': 5.0, 'end': 6.0}],
        [{'time': 2.0, 'overrides': {'lambda': 1e3}},
         {'time': 3.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {'lambda': 1e7}}])
    foreign = copy.deepcopy(bps(p))
    rec = seg(p, 'w2')
    rec['peaks'] = {'min_prominence': 0.5}
    out = S.apply_segment(p, rec)
    for bp in foreign:
        assert bp in bps(out)
    assert len(bps(out)) == 2
    w1 = seg(out, 'w1')
    assert w1['fresh_baseline'] is True and w1['baseline'] == {'lambda': 1e3}
    assert w1['after'] == {'mode': 'custom', 'overrides': {'lambda': 1e7}}
    assert seg(out, 'w2')['peaks'] == {'min_prominence': 0.5}


def test_custom_after_kept_until_reset():
    p = params_of(
        [{'id': 'w1', 'start': 2.0, 'end': 3.0}],
        [{'time': 2.0, 'anchor': {'window_id': 'w1', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
         {'time': 3.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {'lambda': 1e7}}])
    rec = seg(p, 'w1')
    rec['peaks'] = {'min_width': 4}
    out = S.apply_segment(p, rec)
    assert seg(out, 'w1')['after'] == {'mode': 'custom', 'overrides': {'lambda': 1e7}}
    rec = seg(out, 'w1')
    rec['after'] = {'mode': 'resume', 'overrides': {}}
    out2 = S.apply_segment(out, rec)
    [end] = anchored(out2, 'w1', 'end')
    assert stripped(end) == {}
    assert seg(out2, 'w1')['after']['mode'] == 'resume'


def test_remove_keeps_unanchored_start_break():
    p = params_of(
        [{'id': 'w1', 'start': 2.0, 'end': 3.0}],
        [{'time': 2.0, 'overrides': {'lambda': 1e3}},
         {'time': 2.5, 'overrides': {'lambda': 1e4}},
         {'time': 3.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {}}])
    before = copy.deepcopy(p)
    out = S.remove_segment(p, 'w1')
    assert p == before
    assert out['peak_windows'] == []
    assert [(bp['time'], stripped(bp), bp['anchor']) for bp in bps(out)] == [
        (2.0, {'lambda': 1e3}, None)]
    assert out['peaks']['range_filters'] == []
    assert S.segments_view(out)['whole_run_breaks'][0]['time'] == 2.0
    norm(out)


def test_remove_drops_owned_breaks():
    p = params_of(breaks=[{'time': 1.0, 'overrides': {'lambda': 1e5}}])
    rec = S.new_segment(p, 2.0, 3.0)
    rec['fresh_baseline'], rec['baseline'] = True, {'lambda': 1e3}
    rec['partway'] = [{'time': 2.5, 'overrides': {'lambda': 1e4}}]
    p1 = S.apply_segment(p, rec)
    assert len(bps(p1)) == 4
    out = S.remove_segment(p1, rec['id'])
    assert out['peak_windows'] == []
    assert [bp['time'] for bp in bps(out)] == [1.0]
    with pytest.raises(ValueError, match='nope'):
        S.remove_segment(p1, 'nope')


def test_partway_outside_rejected():
    p = params_of([SPEC_WINDOW], SPEC_BREAKS)
    rec = seg(p, 'w1')
    rec['partway'].append({'time': 13.10, 'overrides': {'lambda': 1e5}})
    with pytest.raises(ValueError, match='13.1'):
        S.apply_segment(p, rec)
    rec = seg(p, 'w1')
    rec['partway'] = [{'time': 12.55, 'overrides': {}}]
    with pytest.raises(ValueError, match='12.55'):
        S.apply_segment(p, rec)


def test_overlap_rejected():
    p = params_of([{'id': 'w1', 'label': 'early', 'start': 1.0, 'end': 2.0}])
    rec = S.new_segment(p, 1.5, 3.0)
    rec['label'] = 'late'
    with pytest.raises(ValueError, match='early'):
        S.apply_segment(p, rec)


def test_bad_span_rejected():
    p = params_of()
    rec = S.new_segment(p, 3.0, 3.0)
    rec['label'] = 'flat'
    with pytest.raises(ValueError, match='flat'):
        S.apply_segment(p, rec)


def test_duplicate_break_rejected():
    p = params_of([SPEC_WINDOW], SPEC_BREAKS)
    rec = seg(p, 'w1')
    rec['partway'].append({'time': 12.85, 'overrides': {'lambda': 1e5}})
    with pytest.raises(ValueError, match='12.85'):
        S.apply_segment(p, rec)


def test_new_segment_ids():
    p = params_of([{'id': 'w1', 'start': 1.0, 'end': 2.0}, {'id': 'w3', 'start': 3.0, 'end': 4.0}])
    rec = S.new_segment(p, 5.0, 6.0)
    assert rec['id'] == 'w2'
    assert (rec['start'], rec['end']) == (5.0, 6.0)
    assert rec['fresh_baseline'] is False and rec['partway'] == []
    assert rec['after']['mode'] == 'none'
    assert all(rec[s] == {} for s in S.SEGMENT_SECTIONS)


def test_apply_updates_range_filter_mirror():
    p = params_of()
    out = S.apply_segment(p, S.new_segment(p, 2.0, 3.0))
    assert out['peaks']['range_filters'] == [[2.0, 3.0]]
    g = params_of(outside_windows='global')
    assert S.apply_segment(g, S.new_segment(g, 2.0, 3.0))['peaks']['range_filters'] == []


def test_resume_overrides():
    p = params_of([{'id': 'w1', 'start': 5.0, 'end': 6.0}],
                  [{'time': 1.0, 'overrides': {'lambda': 1e5}},
                   {'time': 5.0, 'overrides': {'lambda': 1e3}}])
    assert S.resume_overrides(p, 0.5) == {}
    assert S.resume_overrides(p, 1.0) == {'lambda': 1e5}
    # the break at 5.0 is w1's fresh start, not a whole-run break
    assert S.resume_overrides(p, 5.5) == {'lambda': 1e5}


def test_summarize_spec_example():
    p = params_of([SPEC_WINDOW], SPEC_BREAKS)
    assert S.summarize(seg(p, 'w1')) == 'λ 1e3→1e4 · prom 0.02 · width 3'


def test_summarize_empty():
    assert S.summarize(S.new_segment(params_of(), 1.0, 2.0)) == '—'


def test_breaks_inside_peaks():
    peaks = [SimpleNamespace(start_time=12.80, end_time=12.90, retention_time=12.85)]
    msgs = S.breaks_inside_peaks(peaks, [12.85])
    assert len(msgs) == 1 and '12.85' in msgs[0]
    assert S.breaks_inside_peaks(peaks, [12.80]) == []


def test_v1_method_view_is_whole_run_only():
    p = ChromaMethod.model_validate({
        'name': 'tcd', 'version': '1', 'signal_type': 'gc',
        'baseline': {'break_points': [{'time': 4.01, 'tolerance': 0.1}]},
    }).to_gui_params()
    view = S.segments_view(p)
    assert view['segments'] == []
    assert [bp['time'] for bp in view['whole_run_breaks']] == [4.01]


# ── controller rulings R5-R7 ───────────────────────────────────────────────────

def test_whole_run_break_at_segment_end_is_not_duplicated():
    p = params_of(breaks=[{'time': 2.0, 'overrides': {'lambda': 1e5}}])
    whole = copy.deepcopy(bps(p)[0])
    rec = S.new_segment(p, 1.0, 2.0)
    rec['fresh_baseline'], rec['baseline'] = True, {'lambda': 1e3}
    out = S.apply_segment(p, rec)
    assert anchored(out, rec['id'], 'end') == []
    assert whole in bps(out)
    assert S.segments_view(out)['whole_run_breaks'] == [whole]
    norm(out)


def test_remove_touching_neighbour_with_unanchored_start():
    p = params_of(
        [{'id': 'A', 'start': 1.0, 'end': 2.0}, {'id': 'B', 'start': 2.0, 'end': 3.0}],
        [{'time': 1.0, 'anchor': {'window_id': 'A', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
         {'time': 2.0, 'overrides': {'lambda': 1e5}}])
    b_start = copy.deepcopy(bps(p)[1])
    out = S.remove_segment(p, 'B')
    assert [w['id'] for w in out['peak_windows']] == ['A']
    assert anchored(out, 'A', 'end') == []
    assert b_start in bps(out)
    assert S.segments_view(out)['whole_run_breaks'] == [b_start]
    norm(out)


def test_neighbour_fresh_start_drops_custom_end(caplog):
    p = params_of(
        [{'id': 'A', 'start': 1.0, 'end': 2.0}, {'id': 'B', 'start': 2.0, 'end': 3.0}],
        [{'time': 1.0, 'anchor': {'window_id': 'A', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
         {'time': 2.0, 'anchor': {'window_id': 'A', 'edge': 'end'}, 'overrides': {'lambda': 1e7}}])
    assert seg(p, 'A')['after']['mode'] == 'custom'
    rec = seg(p, 'B')
    rec['fresh_baseline'], rec['baseline'] = True, {'lambda': 1e4}
    with caplog.at_level('INFO', logger='logic.segments'):
        out = S.apply_segment(p, rec)
    assert anchored(out, 'A', 'end') == []
    assert [stripped(bp) for bp in anchored(out, 'B', 'start')] == [{'lambda': 1e4}]
    assert any('2' in r.getMessage() and 'dropped' in r.getMessage() for r in caplog.records)
    norm(out)


def test_edit_other_segment_leaves_foreign_segment_without_end_break():
    p = params_of(
        [{'id': 'w1', 'start': 2.0, 'end': 3.0}, {'id': 'w2', 'start': 5.0, 'end': 6.0}],
        [{'time': 2.0, 'anchor': {'window_id': 'w1', 'edge': 'start'}, 'overrides': {'lambda': 1e3}}])
    w1_breaks = copy.deepcopy(bps(p))
    rec = seg(p, 'w2')
    rec['peaks'] = {'min_width': 5}
    out = S.apply_segment(p, rec)
    assert bps(out) == w1_breaks
    assert seg(out, 'w1')['after']['mode'] == 'none'
    # a fresh baseline on w2 does not touch w1 either
    rec['fresh_baseline'], rec['baseline'] = True, {'lambda': 1e4}
    out2 = S.apply_segment(p, rec)
    assert [bp for bp in bps(out2) if bp.get('anchor', {}) and bp['anchor']['window_id'] == 'w1'] \
        == w1_breaks
    assert seg(out2, 'w1')['after']['mode'] == 'none'


def test_after_none_kept_when_segment_already_had_a_baseline():
    p = params_of(
        [{'id': 'w1', 'start': 2.0, 'end': 3.0}],
        [{'time': 2.0, 'anchor': {'window_id': 'w1', 'edge': 'start'}, 'overrides': {'lambda': 1e3}}])
    rec = seg(p, 'w1')
    rec['partway'] = [{'time': 2.5, 'overrides': {'lambda': 1e4}}]
    out = S.apply_segment(p, rec)
    assert anchored(out, 'w1', 'end') == []
    rec = seg(out, 'w1')
    rec['after'] = {'mode': 'resume', 'overrides': {}}
    assert len(anchored(S.apply_segment(out, rec), 'w1', 'end')) == 1


def test_set_whole_run_breaks_updates_only_resume_end_breaks():
    p = params_of(
        [{'id': 'A', 'start': 2.0, 'end': 3.0}, {'id': 'B', 'start': 5.0, 'end': 6.0},
         {'id': 'C', 'start': 8.0, 'end': 9.0}],
        [{'time': 1.0, 'overrides': {'lambda': 1e5}},
         {'time': 2.0, 'anchor': {'window_id': 'A', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
         {'time': 3.0, 'anchor': {'window_id': 'A', 'edge': 'end'}, 'overrides': {'lambda': 1e5}},
         {'time': 5.0, 'anchor': {'window_id': 'B', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
         {'time': 6.0, 'anchor': {'window_id': 'B', 'edge': 'end'}, 'overrides': {'lambda': 1e7}},
         {'time': 8.0, 'anchor': {'window_id': 'C', 'edge': 'start'}, 'overrides': {'lambda': 1e3}}])
    custom = copy.deepcopy(anchored(p, 'B', 'end'))
    out = S.set_whole_run_breaks(p, [{'time': 1.0, 'tolerance': 0.1,
                                      'overrides': {'lambda': 1e6}, 'anchor': None}])
    assert [stripped(bp) for bp in anchored(out, 'A', 'end')] == [{'lambda': 1e6}]
    assert anchored(out, 'B', 'end') == custom
    assert anchored(out, 'C', 'end') == []
    assert len(bps(out)) == len(bps(p))
