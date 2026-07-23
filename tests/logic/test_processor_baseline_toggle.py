import numpy as np

from logic.processor import ChromatogramProcessor


def _params(baseline_enabled):
    return {
        "smoothing": {"enabled": False, "method": "whittaker", "lambda": 0.1, "diff_order": 1},
        "baseline": {"enabled": baseline_enabled, "method": "arpls", "lambda": 1e4},
        "peaks": {"enabled": False, "min_prominence": 0.02},
    }


def test_baseline_disabled_leaves_signal_uncorrected():
    x = np.linspace(0, 10, 500)
    y = 5.0 + np.exp(-((x - 5.0) ** 2) / 0.5)  # offset baseline of 5
    proc = ChromatogramProcessor()
    out = proc.process(x, y, params=_params(baseline_enabled=False))
    np.testing.assert_allclose(out["corrected_y"], y, rtol=1e-9)
    np.testing.assert_allclose(out["baseline_y"], np.zeros_like(y))


def test_baseline_enabled_still_corrects():
    x = np.linspace(0, 10, 500)
    y = 5.0 + np.exp(-((x - 5.0) ** 2) / 0.5)
    proc = ChromatogramProcessor()
    out = proc.process(x, y, params=_params(baseline_enabled=True))
    assert out["corrected_y"][0] < y[0]
