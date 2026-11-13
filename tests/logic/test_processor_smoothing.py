"""Whittaker smoothing must run on every process() call, whatever ran before.

pybaselines binds a Baseline() fitter to the first array length it sees; a fitter
shared across calls made smoothing fail (and fall back to the raw signal) after a
trace of another length or a baseline split by break points.
"""
import logging

import numpy as np

import logic.processor as processor_mod
from logic.processor import ChromatogramProcessor


def _trace(n):
    rng = np.random.default_rng(n)
    x = np.linspace(0.0, 10.0, n)
    y = (100 * np.exp(-((x - 4.0) / 0.05) ** 2) + 50 * np.exp(-((x - 7.0) / 0.05) ** 2)
         + rng.normal(0, 1.0, n) + 5.0)
    return x, y


def _params(break_points=None):
    return {
        "smoothing": {"enabled": True, "method": "whittaker", "median_enabled": False,
                      "lambda": 1.0, "diff_order": 1},
        "baseline": {"enabled": True, "method": "arpls", "lambda": 1e5,
                     "break_points": break_points or []},
        "peaks": {"enabled": True, "min_prominence": 0.05, "peak_width": 5},
    }


def _capture_detection_input(proc, monkeypatch):
    """Record (corrected_y, smoothed_y) handed to the detector on each call."""
    seen = []
    detect = proc._detect_peaks_and_shoulders_derivative

    def spy(x, y, *args, **kwargs):
        seen.append((np.copy(y), kwargs.get("smoothed_y")))
        return detect(x, y, *args, **kwargs)

    monkeypatch.setattr(proc, "_detect_peaks_and_shoulders_derivative", spy)
    return seen


def test_second_trace_of_another_length_is_smoothed():
    proc = ChromatogramProcessor()
    proc.process(*_trace(1000), params=_params())
    x, y = _trace(1500)
    out = proc.process(x, y, params=_params())
    fresh = ChromatogramProcessor().process(x, y, params=_params())
    assert not np.allclose(out["smoothed_y"], y)
    np.testing.assert_allclose(out["smoothed_y"], fresh["smoothed_y"])
    np.testing.assert_allclose(out["peaks_x"], fresh["peaks_x"])


def test_reprocess_with_break_points_is_smoothed():
    x, y = _trace(1200)
    params = _params(break_points=[{"time": 5.5}])
    proc = ChromatogramProcessor()
    first = proc.process(x, y, params=params)
    again = proc.process(x, y, params=params)
    assert not np.allclose(again["smoothed_y"], y)
    np.testing.assert_allclose(again["smoothed_y"], first["smoothed_y"])


def test_detection_signal_is_smoothed_after_break_points(monkeypatch):
    x, y = _trace(1200)
    proc = ChromatogramProcessor()
    seen = _capture_detection_input(proc, monkeypatch)
    proc.process(x, y, params=_params(break_points=[{"time": 5.5}]))
    corrected, smoothed = seen[0]
    assert smoothed is not None
    assert not np.allclose(smoothed, corrected)


def test_detection_signal_is_smoothed_with_ms_range_mask(monkeypatch):
    x, y = _trace(1200)
    proc = ChromatogramProcessor()
    seen = _capture_detection_input(proc, monkeypatch)
    proc.process(x, y, params=_params(), ms_range=(1.0, 9.0))
    corrected, smoothed = seen[0]
    masked = np.isnan(corrected)
    assert masked.any()  # the pre-MS region is NaN in the corrected trace
    np.testing.assert_array_equal(np.isnan(smoothed), masked)
    assert not np.allclose(smoothed[~masked], corrected[~masked])


def test_smoothing_failure_is_logged_and_returns_input(monkeypatch, caplog):
    class Broken:
        def asls(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(processor_mod, "Baseline", Broken)
    y = np.arange(50, dtype=float)
    with caplog.at_level(logging.ERROR, logger="logic.processor"):
        out = ChromatogramProcessor()._apply_smoothing(
            np.copy(y), {"method": "whittaker", "median_enabled": True, "median_kernel": 5})
    np.testing.assert_array_equal(out, y)
    assert any(r.levelno == logging.ERROR and "Smoothing failed" in r.getMessage()
               for r in caplog.records)
