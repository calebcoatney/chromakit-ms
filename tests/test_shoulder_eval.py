"""The shoulder eval harness scores what it claims to score."""
import numpy as np

from tools.shoulder_eval import (
    Component, Report, build_case, score_case, summarize, _place_pair, DT_MIN, PARENT_HEIGHT,
)
from tools.shoulder_label_sheet import panels_for_run


def _pair(rs, ratio, sigma_pts=12, noise=0.0, tau_ratio=0.0):
    s = sigma_pts * DT_MIN
    comps = [Component(0.0, s, tau_ratio * s, PARENT_HEIGHT, 'parent'),
             Component(_place_pair(rs, s, tau_ratio * s, +1), s, tau_ratio * s, ratio * PARENT_HEIGHT, 'child')]
    return build_case('t', comps, noise, np.random.default_rng(0), {})


def _single():
    s = 12 * DT_MIN
    return build_case('s', [Component(0.0, s, 0.0, PARENT_HEIGHT, 'parent')], 0.0,
                      np.random.default_rng(0), {})


def test_low_resolution_pair_is_a_shoulder_high_resolution_is_resolved():
    assert _pair(0.5, 0.5).components[1].kind == 'shoulder'
    assert _pair(1.2, 1.0).components[1].kind == 'resolved'


def test_true_split_lies_between_the_apexes():
    c = _pair(0.7, 0.5)
    p, ch = c.components
    assert p.apex < c.true_split < ch.apex


def test_perfect_reports_score_a_hit_with_no_false_reports():
    c = _pair(0.6, 0.5)
    p, ch = c.components
    s = score_case(c, [Report(p.apex), Report(ch.apex, 'shoulder', c.true_split)])
    assert s.child_found and s.parent_found
    assert s.false_reports == 0
    assert s.split_err_fwhm == 0.0


def test_extra_report_on_a_single_peak_is_a_false_split():
    c = _single()
    p = c.components[0]
    s = score_case(c, [Report(p.apex), Report(p.apex + 2 * p.fwhm, 'shoulder')])
    assert s.false_reports == 1
    assert summarize([s])['single']['false_split_rate'] == 1.0


def test_rejecting_a_single_peak_correctly_is_a_success():
    c = _single()
    s = score_case(c, [Report(c.components[0].apex)])
    assert s.false_reports == 0 and s.parent_found


def test_merged_maximum_between_apexes_counts_as_the_parent_not_a_false_split():
    c = _pair(0.5, 1.0)
    p, ch = c.components
    mid = 0.5 * (p.apex + ch.apex)
    s = score_case(c, [Report(mid), Report(ch.apex, 'shoulder')])
    assert s.child_found and s.parent_found and s.false_reports == 0


def test_child_report_near_the_parent_does_not_count_as_the_child():
    c = _pair(0.6, 0.5)
    p, ch = c.components
    s = score_case(c, [Report(p.apex), Report(p.apex + 0.1 * (ch.apex - p.apex), 'shoulder')])
    assert not s.child_found
    assert s.false_reports == 1


def test_label_sheet_merges_overlapping_peak_windows():
    x = np.arange(0, 2, DT_MIN)
    y = sum(h * np.exp(-0.5 * ((x - mu) / 0.01) ** 2) for mu, h in [(0.5, 1e5), (0.53, 5e4), (1.5, 1e5)])
    y = y + np.random.default_rng(1).normal(0, 10, len(x))
    spans, _ = panels_for_run(x, y)
    assert len(spans) == 2
    assert x[spans[0][0]] < 0.5 and x[spans[0][1]] > 0.53
