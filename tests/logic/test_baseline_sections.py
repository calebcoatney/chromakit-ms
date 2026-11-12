"""Per-section baseline settings carried by break-point overrides."""
import logging

import numpy as np
import pytest

from logic.processor import ChromatogramProcessor


@pytest.fixture
def trace():
    x = np.linspace(0.0, 10.0, 2000)
    drift = 0.8 * x + 0.05 * x ** 2
    tail = 60.0 * np.exp(-((x - 4.0) ** 2) / 0.05) * (1 + 0.6 * (x > 4.0))
    small = (3.0 * np.exp(-((x - 2.0) ** 2) / 0.01)
             + 2.0 * np.exp(-((x - 7.0) ** 2) / 0.01))
    return x, drift + tail + small


def _params(break_points, enabled=True, lam=1e6):
    return {
        "smoothing": {"enabled": False, "method": "whittaker", "lambda": 0.1, "diff_order": 1},
        "baseline": {"enabled": enabled, "method": "arpls", "lambda": lam,
                     "break_points": break_points},
        "peaks": {"enabled": False},
    }


def _run(trace, break_points, ms_range=None, **kw):
    x, y = trace
    return ChromatogramProcessor().process(
        x, y, params=_params(break_points, **kw), ms_range=ms_range)


def test_section_lambda_override_changes_only_its_section(trace):
    x, _ = trace
    plain = _run(trace, [4.55, 5.1])
    over = _run(trace, [
        {"time": 4.55, "overrides": {"lambda": 1e3}},
        {"time": 5.1},
    ])
    before, mid, after = x < 4.55, (x >= 4.55) & (x < 5.1), x >= 5.1
    assert np.allclose(over["baseline_y"][before], plain["baseline_y"][before])
    assert np.allclose(over["baseline_y"][after], plain["baseline_y"][after])
    assert not np.allclose(over["baseline_y"][mid], plain["baseline_y"][mid])


def test_section_disabled_gives_zero_baseline(trace):
    x, y = trace
    out = _run(trace, [{"time": 5.0, "overrides": {"enabled": False}}])
    sec = x >= 5.0
    assert np.all(out["baseline_y"][sec] == 0.0)
    np.testing.assert_allclose(out["corrected_y"][sec], y[sec])
    assert np.any(out["baseline_y"][~sec] != 0.0)


def test_section_enabled_override_runs_when_global_disabled(trace):
    x, y = trace
    out = _run(trace, [{"time": 5.0, "overrides": {"enabled": True}}], enabled=False)
    sec = x >= 5.0
    assert np.all(out["baseline_y"][~sec] == 0.0)
    assert np.any(out["baseline_y"][sec] != 0.0)


def test_section_after_ms_edge_inherits_previous_break(trace):
    x, _ = trace
    bps = [{"time": 1.0, "overrides": {"lambda": 1e2}}]
    ms = (3.0, 10.0)
    plain = _run(trace, [1.0], ms_range=ms)
    over = _run(trace, bps, ms_range=ms)
    fit = x >= 3.0
    assert not np.allclose(over["baseline_y"][fit], plain["baseline_y"][fit])
    # Same as a global lambda of 1e2 over the whole MS-on region.
    ref = _run(trace, [1.0], ms_range=ms, lam=1e2)
    np.testing.assert_allclose(over["baseline_y"][fit], ref["baseline_y"][fit])


def test_dropped_break_overrides_do_not_apply(trace, caplog):
    x, _ = trace
    ms = (0.0, 8.0)
    bps = [{"time": 3.0, "overrides": {"lambda": 1e3}},
           {"time": 9.0, "overrides": {"lambda": 1e2}}]
    with caplog.at_level(logging.WARNING, logger="logic.processor"):
        out = _run(trace, bps, ms_range=ms)
    assert any("dropped" in r.message for r in caplog.records)
    ref = _run(trace, [{"time": 3.0, "overrides": {"lambda": 1e3}}], ms_range=ms)
    on = x <= 8.0
    np.testing.assert_allclose(out["baseline_y"][on], ref["baseline_y"][on])
    assert np.all(np.isnan(out["baseline_y"][x > 8.05]))
