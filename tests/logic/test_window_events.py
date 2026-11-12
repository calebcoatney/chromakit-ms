import numpy as np
import pytest
from pydantic import BaseModel

from logic import window_events
from logic.integration import Integrator
from logic.window_events import apply_window_events, validate_event


def _processed(with_windows=True):
    x = np.linspace(0, 20, 2001)
    y = np.zeros_like(x)
    centers = [4.0, 6.0, 14.0, 16.0]
    for c in centers:
        y += 100.0 * np.exp(-((x - c) ** 2) / (2 * 0.1 ** 2))
    idx = [int(np.argmin(np.abs(x - c))) for c in centers]
    meta = [{"index": i, "left_base": i - 40, "right_base": i + 40,
             "window_id": "w1" if c < 10 else "w2"}
            for i, c in zip(idx, centers)]
    d = {
        "x": x, "original_y": y, "corrected_y": y,
        "baseline_y": np.zeros_like(y),
        "peaks_x": x[idx], "peaks_y": y[idx],
        "peak_metadata": meta,
    }
    if with_windows:
        d["peak_windows"] = [
            {"id": "w1", "start": 0.0, "end": 10.0, "events": [{"type": "tag"}]},
            {"id": "w2", "start": 10.0, "end": 20.0, "events": []},
        ]
    return d


def test_unknown_event_raises_with_version_hint():
    with pytest.raises(ValueError, match="not supported by this ChromaKit version"):
        validate_event({"type": "nope"})


def test_missing_type_raises():
    with pytest.raises(ValueError, match="type"):
        validate_event({})


class _Tag(BaseModel):
    type: str = "tag"


def _tag(event, window, peaks, processed):
    for p in peaks:
        p.tagged = True
    return peaks


def test_registered_event_dispatches_to_window_peaks_only(monkeypatch):
    monkeypatch.setitem(window_events.EVENT_TYPES, "tag", _Tag)
    monkeypatch.setitem(window_events.EVENT_HANDLERS, "tag", _tag)
    assert validate_event({"type": "tag"}) == {"type": "tag"}
    res = Integrator.integrate(_processed(), verbose=False)
    peaks = res["peaks"]
    assert [getattr(p, "window_id", None) for p in peaks] == ["w1", "w1", "w2", "w2"]
    assert [getattr(p, "tagged", False) for p in peaks] == [True, True, False, False]


def test_handler_changing_peak_count_raises(monkeypatch):
    monkeypatch.setitem(window_events.EVENT_TYPES, "tag", _Tag)
    monkeypatch.setitem(window_events.EVENT_HANDLERS, "tag", lambda e, w, p, d: p[:1])
    with pytest.raises(ValueError, match="peak count"):
        Integrator.integrate(_processed(), verbose=False)


def test_no_windows_is_identity():
    peaks = [object(), object()]
    assert apply_window_events(peaks, {}) is peaks
    res = Integrator.integrate(_processed(with_windows=False), verbose=False)
    assert len(res["peaks"]) == 4
