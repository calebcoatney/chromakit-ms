"""Split-or-skim intake: the four tests, parent chain, front child, forced modes, window override, flag."""
import numpy as np
import pytest
from scipy.stats import norm

from logic.method import RiderParams
from logic.tail_skim import RIDER_DEFAULTS, classify_riders, rider_settings_at
from tests.logic.tail_skim_signals import (PARENT_MU, SIGMA, _smooth, _time_axis, emg, emg_pair,
                                           rider_on_tail)

AUTO = {'rider_integration': 'auto'}


def run(pair, defaults=AUTO, windows=None, metas=None, extra_apexes=(), negative=(), groups=()):
    apexes = np.array(sorted({pair['apex'], pair['child_apex'], *extra_apexes}))
    metadata = metas or [{} for _ in apexes]
    return classify_riders(pair['x'], pair['y'], apexes, metadata, defaults, windows or [],
                           set(negative), list(groups))


def only_call(pair, **kw):
    calls = run(pair, **kw)
    assert list(calls) == [pair['child_apex']]
    return calls[pair['child_apex']]


def test_rider_defaults_match_method_model():
    assert RIDER_DEFAULTS == RiderParams().model_dump()


def test_rider_settings_none_defaults_is_split():
    assert rider_settings_at(None, [], 1.0) == {**RIDER_DEFAULTS, 'rider_integration': 'split'}


def test_rider_settings_window_without_rider_key_is_empty():
    w = [{'id': 'w1', 'start': 0.0, 'end': 2.0}]
    assert rider_settings_at(AUTO, w, 1.0) == {**RIDER_DEFAULTS, 'rider_integration': 'auto'}


def test_anh_like_rider_is_skimmed():
    pair = rider_on_tail(1.0, 3.0)
    call = only_call(pair)
    assert call.decision == 'skim' and call.source == 'auto' and call.failed == []
    assert call.parent == pair['apex'] and call.tail is not None
    print(call.ratios)


def test_user_split_parent_index_is_used():
    pair = rider_on_tail(1.0, 3.0)
    meta = [{}, {'is_shoulder': True, 'source': 'user', 'split_index': pair['valley'],
                 'parent_index': pair['apex']}]
    call = only_call(pair, metas=meta)
    assert call.parent == pair['apex'] and call.a == pair['valley']


def test_symmetric_pair_split_by_tail_test():
    pair = emg_pair(0.002, 0.002, 4.0, 0.05)
    call = only_call(pair)
    print(call.ratios)
    assert call.decision == 'split' and 'tail' in call.failed


def test_height_ratio_below_2_splits():
    pair = emg_pair(0.06, 0.06, 2.2, 0.12)
    call = only_call(pair)
    print(call.ratios)
    assert call.decision == 'split' and call.failed == ['height']


def test_valley_above_half_parent_splits():
    pair = emg_pair(0.06, 0.06, 4.0, 0.03)
    call = only_call(pair)
    print(call.ratios)
    assert call.ratios['hv_hp'] > 0.5
    assert call.decision == 'split' and 'position' in call.failed


def test_nearly_resolved_pair_splits_on_valley_ratio():
    pair = emg_pair(0.03, 0.01, 3.0, 0.2, noise=0.0002)
    call = only_call(pair)
    print(call.ratios)
    assert call.ratios['hc_hv'] > 20
    assert call.decision == 'split' and 'valley' in call.failed


def _with_front_peak(pair, mult=2.0, gap=0.03):
    """The pair's signal plus a taller peak `gap` min after the child; returns (pair, front apex)."""
    x = pair['x']
    h_c = pair['y'][pair['child_apex']]
    mu = x[pair['child_apex']] + gap
    third = mult * h_c * SIGMA * np.sqrt(2 * np.pi) * norm.pdf(x, mu, SIGMA) / (SIGMA * np.sqrt(2 * np.pi) * norm.pdf(mu, mu, SIGMA))
    out = dict(pair, y=pair['y'] + third)
    n = int(np.searchsorted(x, mu))
    return out, n


def test_front_child_always_split():
    pair, n = _with_front_peak(rider_on_tail(1.0, 3.0))
    for mode in ('auto', 'skim'):
        call = only_call_with_next(pair, n, {'rider_integration': mode})
        print(mode, call.ratios)
        assert call.decision == 'split' and 'front' in call.failed


def only_call_with_next(pair, n, defaults):
    calls = run(pair, defaults=defaults, extra_apexes=[n])
    return calls[pair['child_apex']]


def _two_riders(h1, d1, h2, d2, parent_tau=0.06):
    """A parent with two Gaussian riders on its tail; returns a pair-shaped dict plus both child apexes."""
    x = _time_axis(0.05)
    parent = emg(x, 1.0, PARENT_MU, SIGMA, parent_tau)
    total = parent.copy()
    apexes = []
    for h, d in ((h1, d1), (h2, d2)):
        mu = PARENT_MU + d * parent_tau
        height = h * emg(mu, 1.0, PARENT_MU, SIGMA, parent_tau)
        total = total + height * np.exp(-0.5 * ((x - mu) / SIGMA) ** 2)
        apexes.append(int(np.searchsorted(x, mu)))
    y = _smooth(x, total, parent, 0.002, 0)
    return dict(x=x, y=y, apex=int(np.argmax(parent)), child_apex=apexes[0]), apexes[1]


def test_chain_keeps_parent_while_tail_real():
    pair, c2 = _two_riders(1.0, 3.0, 1.0, 4.5)
    calls = run(pair, extra_apexes=[c2])
    first, second = calls[pair['child_apex']], calls[c2]
    print(first.ratios, second.ratios, first.decision, second.decision)
    assert first.decision == 'skim'
    assert second.parent == pair['apex']


def test_chain_ends_when_tail_dies():
    pair, c2 = _two_riders(1.0, 3.0, 12.0, 5.0)
    calls = run(pair, extra_apexes=[c2])
    first = calls[pair['child_apex']]
    second = calls[c2]
    print(first.decision, float(first.tail(pair['x'][c2])), pair['y'][c2])
    assert first.decision == 'skim'
    assert second.parent == pair['child_apex']


def test_forced_skim_skips_thresholds():
    pair = emg_pair(0.06, 0.06, 2.2, 0.12)
    call = only_call(pair, defaults={'rider_integration': 'skim'})
    assert call.decision == 'skim' and call.source == 'forced'


def test_split_mode_never_skims():
    pair = rider_on_tail(1.0, 3.0)
    call = only_call(pair, defaults={'rider_integration': 'split'})
    assert call.decision == 'split' and call.source == 'forced' and call.tail is None


def test_window_override_beats_default():
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    windows = [{'id': 'w1', 'start': float(x[0]), 'end': float(x[-1]),
                'rider': {'rider_integration': 'auto'}}]
    assert only_call(pair, defaults={'rider_integration': 'split'}).decision == 'split'
    assert only_call(pair, defaults={'rider_integration': 'split'}, windows=windows).decision == 'skim'


def test_near_threshold_flag():
    pair = emg_pair(0.06, 0.06, 3.0, 0.12)
    near = only_call(pair)
    print(near.ratios, near.decision, near.failed)
    assert near.ratios['hp_hc'] == pytest.approx(2.05, rel=0.03)
    assert near.near_threshold is True
    far = only_call(emg_pair(0.06, 0.06, 2.2, 0.12))
    assert far.near_threshold is False


def test_no_rider_key_means_split():
    pair = rider_on_tail(1.0, 3.0)
    call = only_call(pair, defaults=None)
    assert call.decision == 'split'
    assert not any(c.decision == 'skim' for c in run(pair, defaults=None).values())


def test_negative_or_group_overlap_is_excluded():
    pair = rider_on_tail(1.0, 3.0)
    for kw in ({'negative': [pair['child_apex']]},
               {'groups': [(pair['apex'] - 2, pair['child_apex'] + 2)]}):
        call = only_call(pair, **kw)
        assert call.decision == 'split' and 'excluded' in call.failed


def test_child_taller_than_parent_gets_no_call():
    pair = emg_pair(0.06, 0.06, 0.5, 0.08)
    assert run(pair) == {}
