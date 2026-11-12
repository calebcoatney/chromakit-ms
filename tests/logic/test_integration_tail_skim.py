"""Tail skim wired into Integrator.integrate(): rider and parent bounds, areas, fields, shading arrays."""
import logging

import numpy as np
import pytest

from logic.integration import Integrator
from logic.tail_skim import classify_riders, exp_skim, rider_end
from tests.logic import (test_integrate_verbose_spectral, test_integration_area_factor,
                         test_integration_nan_baseline, test_range_filter_bounds)
from tests.logic.tail_skim_signals import emg_pair, processed_from, rider_on_tail
from tests.logic.test_tail_skim_intake import _two_riders

AUTO = {'rider_integration': 'auto'}
SKIM = {'rider_integration': 'skim'}


def integrate(processed, **kw):
    return Integrator.integrate(processed, verbose=False, **kw)


def at(result, x, idx):
    """The reported peak whose apex is x[idx]."""
    (peak,) = [p for p in result['peaks'] if p.retention_time == x[idx]]
    return peak


def summary(result):
    return ([p.area for p in result['peaks']], result['integration_bounds'], result['integrated_areas'])


def two_riders():
    pair, c2 = _two_riders(1.0, 3.0, 1.0, 4.5)
    x = pair['x']
    return pair, c2, processed_from(pair, extra_peaks=[x[c2]], rider=AUTO)


def skim_call(pair, defaults=AUTO):
    """The rider call and its skim curve, worked out directly from tail_skim."""
    x, y = pair['x'], pair['y']
    calls = classify_riders(x, y, np.array([pair['apex'], pair['child_apex']]), [{}, {}], defaults, [],
                            set(), [])
    call = calls[pair['child_apex']]
    b = rider_end(x, y, call.a, call.child, call.cap)
    return call, b, exp_skim(x, y, call.a, b, call.parent, call.cap, [])


# --- untouched paths --------------------------------------------------------------------------------

def test_spectral_profile_untouched():
    _, processed, profile = test_integrate_verbose_spectral._ftir_processed()
    plain = dict(processed)
    plain.pop('rider', None)
    forced = dict(processed, rider=dict(SKIM))
    a = integrate(plain, profile=profile)
    b = integrate(forced, profile=profile)
    assert [p.area for p in b['peaks']] == [p.area for p in a['peaks']]
    assert b['integration_bounds'] == a['integration_bounds']


def _main_fixtures():
    proc, x, y, params = test_range_filter_bounds.build_setup()
    yield test_integration_area_factor._simple_processed()
    yield test_integration_nan_baseline._build_processed(include_nan=False)
    yield test_integration_nan_baseline._build_processed(include_nan=True)
    params['peaks']['range_filters'] = [[4.1, 5.0]]
    yield proc.process(x, y, params)
    params['peaks']['range_filters'] = []
    yield proc.process(x, y, params)


@pytest.mark.parametrize('processed', list(_main_fixtures()),
                         ids=['area_factor', 'no_nan', 'nan_baseline', 'range_filtered', 'range_unfiltered'])
def test_split_mode_matches_main(processed):
    popped = dict(processed)
    popped.pop('rider', None)
    expected = summary(integrate(popped))
    for rider in (None, {'rider_integration': 'split'}):
        run = dict(processed)
        if rider is None:
            run.pop('rider', None)
        else:
            run['rider'] = rider
        result = integrate(run)
        assert summary(result) == expected
        assert all(p.skim_parent is None and p.skim_children is None for p in result['peaks'])
        assert all(p.skim_decision is None and p.skim_near_threshold is None for p in result['peaks'])
        assert all(top is y for top, y in zip(result['shade_top_peaks'], result['y_peaks']))
        assert all(not f.any() for f in result['shade_floor_peaks'])


# --- one rider --------------------------------------------------------------------------------------

def test_rider_and_parent_areas():
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    result = integrate(processed_from(pair, rider=AUTO))
    parent, rider = at(result, x, pair['apex']), at(result, x, pair['child_apex'])
    assert rider.skim_parent == parent.peak_number
    assert rider.area == pytest.approx(pair['true_child_area'], rel=0.10)
    assert parent.area == pytest.approx(pair['true_parent_area'], rel=0.01)
    assert parent.area + rider.area == pytest.approx(parent.area_before_skim, rel=1e-12)
    assert parent.skimmed_area == pytest.approx(rider.area)
    assert parent.skim_children == [rider.peak_number]
    assert rider.skim_model == 'exponential' and rider.skim_source == 'auto'
    assert result['integrated_areas'] == [p.area for p in result['peaks']]


def test_rider_bounds_and_parent_extended():
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    call, b, _ = skim_call(pair)
    result = integrate(processed_from(pair, rider=AUTO))
    parent, rider = at(result, x, pair['apex']), at(result, x, pair['child_apex'])
    assert (rider.start_time, rider.end_time) == (x[call.a], x[b])
    assert parent.end_time >= x[b]


def test_two_riders_both_subtracted():
    pair, c2, processed = two_riders()
    x = pair['x']
    result = integrate(processed)
    parent = at(result, x, pair['apex'])
    r1, r2 = at(result, x, pair['child_apex']), at(result, x, c2)
    assert r1.skim_parent == r2.skim_parent == parent.peak_number
    assert parent.skimmed_area == pytest.approx(r1.area + r2.area)
    assert parent.skim_children == [r1.peak_number, r2.peak_number]
    assert parent.area + r1.area + r2.area == pytest.approx(parent.area_before_skim, rel=1e-12)
    assert r1.end_time <= r2.start_time


def test_area_factor_applies_to_all():
    _, _, processed = two_riders()
    one = integrate(processed)
    two = integrate(processed, area_factor=2.0)
    for p, q in zip(one['peaks'], two['peaks']):
        assert q.area == pytest.approx(2 * p.area, rel=1e-12)
    assert two['peaks'][0].area_before_skim == pytest.approx(2 * one['peaks'][0].area_before_skim)


# --- fall back to split -----------------------------------------------------------------------------

@pytest.mark.parametrize('bound', ['a', 'b'])
def test_nan_at_rider_bound_splits(caplog, bound):
    """NaN at the rider's start (a user split placed there) or at its end: split, with a warning."""
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    call, b, _ = skim_call(pair, SKIM)
    nan_at = call.a if bound == 'a' else b
    runs = []
    for rider in (SKIM, None):
        processed = processed_from(pair, rider=rider)
        processed['corrected_y'][nan_at] = np.nan
        if bound == 'a':
            processed['peak_metadata'][1].update(is_shoulder=True, split_index=int(call.a),
                                                 parent_index=int(pair['apex']))
        runs.append(processed)
    with caplog.at_level(logging.WARNING):
        result = integrate(runs[0])
    rider = at(result, x, pair['child_apex'])
    assert rider.skim_parent is None and rider.skim_decision['decision'] == 'split'
    # Split exactly as main does. Main's own split walk crosses a lone NaN sample inside the span, so
    # an area can be NaN there too (the brief expected it finite; reported as a concern).
    np.testing.assert_array_equal([p.area for p in result['peaks']],
                                  [p.area for p in integrate(runs[1])['peaks']])
    assert any(f"{x[pair['child_apex']]:.3f}" in r.message and 'not finite' in r.message
               for r in caplog.records)


def test_nan_between_apexes_still_skimmed():
    """A NaN sample between parent and rider, away from a and b, is not taken as the valley."""
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    call, b, _ = skim_call(pair)
    processed = processed_from(pair, rider=AUTO)
    nan_at = (pair['apex'] + call.a) // 2
    processed['corrected_y'][nan_at] = np.nan
    result = integrate(processed)
    rider = at(result, x, pair['child_apex'])
    assert rider.skim_parent == at(result, x, pair['apex']).peak_number
    assert (rider.start_index, rider.end_index) == (call.a, b)
    assert rider.area == pytest.approx(pair['true_child_area'], rel=0.10)


def test_shoulder_at_previous_apex_without_rider_key():
    """Legacy run: shoulder metadata the skim intake would reject is never checked in split mode."""
    pair = rider_on_tail(1.0, 3.0)
    processed = processed_from(pair)
    processed.pop('rider')
    processed['peak_metadata'][1].update(is_shoulder=True, left_bound=int(pair['apex']),
                                         right_bound=int(pair['child_apex'] + 20))
    result = integrate(processed)
    assert len(result['peaks']) == 2
    assert all(p.skim_decision is None for p in result['peaks'])


def test_peak_group_overlap_splits(caplog):
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    groups = [[x[pair['apex']] - 0.01, x[pair['child_apex']] + 0.01]]
    main = integrate(processed_from(pair), peak_groups=groups)
    with caplog.at_level(logging.WARNING):
        result = integrate(processed_from(pair, rider=AUTO), peak_groups=groups)
    assert all(p.skim_parent is None and p.skim_children is None for p in result['peaks'])
    assert any('peak-group' in r.message for r in caplog.records)
    assert summary(result) == summary(main)


def test_group_on_parent_edge_splits(caplog):
    """A group overlapping only the parent's computed bounds (not [parent, cap]) still splits."""
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    start = integrate(processed_from(pair, rider=AUTO))['peaks'][0].start_time
    groups = [[start - 0.02, start + 0.001]]
    main = integrate(processed_from(pair), peak_groups=groups)
    with caplog.at_level(logging.WARNING):
        result = integrate(processed_from(pair, rider=AUTO), peak_groups=groups)
    assert all(p.skim_parent is None and p.skim_children is None for p in result['peaks'])
    assert at(result, x, pair['child_apex']).skim_decision['decision'] == 'split'
    assert any('peak-group' in r.message and f"{x[pair['child_apex']]:.3f}" in r.message
               for r in caplog.records)
    assert summary(result) == summary(main)


def test_rider_of_a_rider_splits(caplog):
    pair, c2 = _two_riders(1.0, 3.0, 12.0, 5.0)
    x = pair['x']
    processed = processed_from(pair, extra_peaks=[x[c2]], rider=SKIM)
    with caplog.at_level(logging.WARNING):
        result = integrate(processed)
    first, second = at(result, x, pair['child_apex']), at(result, x, c2)
    assert first.skim_parent is not None
    assert second.skim_parent is None and second.skim_decision['decision'] == 'split'
    assert any(f"{x[c2]:.3f}" in r.message for r in caplog.records)


# --- across window edges ----------------------------------------------------------------------------

def test_parent_outside_window_rider_skimmed():
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    processed = processed_from(pair, rider=AUTO)
    processed.update(peaks_x=x[[pair['child_apex']]], peaks_y=pair['y'][[pair['child_apex']]],
                     peak_metadata=processed['peak_metadata'][1:],
                     range_excluded_peaks_x=x[[pair['apex']]])
    result = integrate(processed)
    (rider,) = result['peaks']
    assert rider.area == pytest.approx(pair['true_child_area'], rel=0.10)
    assert rider.skim_model == 'exponential' and rider.skim_parent is None
    assert rider.skim_decision['parent'] is None


def test_rider_outside_window_still_subtracted():
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    expected_rider_area = at(integrate(processed_from(pair, rider=AUTO)), x, pair['child_apex']).area
    processed = processed_from(pair, rider=AUTO)
    processed.update(peaks_x=x[[pair['apex']]], peaks_y=pair['y'][[pair['apex']]],
                     peak_metadata=processed['peak_metadata'][:1],
                     range_excluded_peaks_x=x[[pair['child_apex']]])
    (parent,) = integrate(processed)['peaks']
    assert parent.area == pytest.approx(parent.area_before_skim - expected_rider_area, rel=1e-6)
    assert parent.skimmed_area == pytest.approx(expected_rider_area)
    assert parent.skim_children == []


def test_parent_clamped_before_rider_starts():
    """Parent's window ends before its rider's span; the rider is outside every window."""
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    call, b, _ = skim_call(pair)
    processed = processed_from(pair, rider=AUTO)
    processed['peak_metadata'][0]['clamp'] = (None, float(x[call.a - 10]))
    processed.update(peaks_x=x[[pair['apex']]], peaks_y=pair['y'][[pair['apex']]],
                     peak_metadata=processed['peak_metadata'][:1],
                     range_excluded_peaks_x=x[[pair['child_apex']]])
    result = integrate(processed)
    (parent,) = result['peaks']
    assert parent.end_index == call.a - 10
    np.testing.assert_array_equal(result['shade_top_peaks'][0],
                                  pair['y'][parent.start_index:parent.end_index + 1])


# --- small things -----------------------------------------------------------------------------------

def test_zero_area_rider_kept_with_issue(caplog):
    pair = rider_on_tail(0.0, 3.0, noise=0.0)
    x = pair['x']
    with caplog.at_level(logging.WARNING):
        result = integrate(processed_from(pair, rider={**SKIM, 'skim_model': 'tangent'}))
    rider = at(result, x, pair['child_apex'])
    assert rider.skim_model == 'tangent'
    assert rider.area == 0
    assert "Skim: no area above parent tail" in rider.quality_issues
    assert any('no area above' in r.message for r in caplog.records)


def test_shading_arrays():
    pair = rider_on_tail(1.0, 3.0)
    x, y = pair['x'], pair['y']
    call, b, curve = skim_call(pair)
    result = integrate(processed_from(pair, rider=AUTO))
    k_parent = result['peaks'].index(at(result, x, pair['apex']))
    k_rider = result['peaks'].index(at(result, x, pair['child_apex']))
    tops, floors = result['shade_top_peaks'], result['shade_floor_peaks']
    assert [len(t) for t in tops] == [len(f) for f in floors] == [len(xp) for xp in result['x_peaks']]
    np.testing.assert_allclose(floors[k_rider], curve.s)
    np.testing.assert_array_equal(tops[k_rider], y[call.a:b + 1])
    assert not floors[k_parent].any()
    start = result['peaks'][k_parent].start_index
    top = tops[k_parent]
    span = slice(call.a - start, b - start + 1)
    np.testing.assert_allclose(top[span], curve.s)
    outside = np.ones(len(top), bool)
    outside[span] = False
    np.testing.assert_array_equal(top[outside], y[start:start + len(top)][outside])


def test_decision_fields_on_split_child():
    pair = emg_pair(0.002, 0.002, 4.0, 0.035)
    x = pair['x']
    result = integrate(processed_from(pair, rider=AUTO))
    parent, child = at(result, x, pair['apex']), at(result, x, pair['child_apex'])
    d = child.skim_decision
    assert d['decision'] == 'split' and 'tail' in d['failed']
    assert d['parent'] == parent.peak_number
    assert set(d) == {'decision', 'parent', 'hp_hc', 'hv_hp', 'hc_hv', 'tail_bend', 'failed'}
    assert child.skim_near_threshold in (True, False)
    assert child.skim_parent is None and parent.skim_children is None
