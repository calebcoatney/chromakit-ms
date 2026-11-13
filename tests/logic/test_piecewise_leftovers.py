"""Regression tests for phase-1 leftovers: descending-axis order, window_id default, event count guard."""
import copy

import numpy as np
import pytest
from pydantic import BaseModel

from logic import window_events
from logic.integration import Integrator
from logic.processor import ChromatogramProcessor
from logic.window_events import apply_window_events


def _gauss(x, c, s, a):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def _params(windows=None):
    params = copy.deepcopy(ChromatogramProcessor().default_params)
    params['baseline']['enabled'] = False
    params['smoothing']['enabled'] = False
    params['peaks'].update({'min_prominence': 0.05, 'min_width': 0})
    if windows is not None:
        params['peak_windows'] = windows
        params['outside_windows'] = 'ignore'
    return params


def test_descending_axis_windowed_order_matches_unwindowed():
    x = np.linspace(4000, 400, 4001)  # descending, FTIR-style
    y = sum(_gauss(x, c, 30, a) for c, a in ((3000, 100), (2000, 60), (1000, 80)))
    window = {'id': 'w1', 'label': 'all', 'start': 400, 'end': 4000, 'smoothing': {},
              'peaks': {}, 'shoulders': {}, 'negative_peaks': {}, 'events': []}

    proc = ChromatogramProcessor()
    plain = proc.process(x, y, _params())
    windowed = proc.process(x, y, _params([window]))

    assert len(plain['peaks_x']) == 3
    assert list(plain['peaks_x']) == sorted(plain['peaks_x'], reverse=True)
    np.testing.assert_array_equal(windowed['peaks_x'], plain['peaks_x'])

    r_plain = Integrator.integrate(plain, verbose=False)
    r_win = Integrator.integrate(windowed, verbose=False)
    assert [(p.peak_number, p.retention_time) for p in r_win['peaks']] == \
           [(p.peak_number, p.retention_time) for p in r_plain['peaks']]


@pytest.mark.parametrize("metadata", [None, "short"])
def test_missing_metadata_peak_gets_window_id_none(metadata):
    x = np.linspace(0, 10, 2001)
    y = sum(_gauss(x, c, 0.1, 100) for c in (2.0, 5.0, 8.0))
    idx = [int(np.argmin(np.abs(x - c))) for c in (2.0, 5.0, 8.0)]
    data = {'x': x, 'original_y': y, 'corrected_y': y, 'baseline_y': np.zeros_like(y),
            'peaks_x': x[idx], 'peaks_y': y[idx]}
    if metadata == "short":
        data['peak_metadata'] = [{'index': idx[0], 'window_id': 'w1'}]
    peaks = Integrator.integrate(data, verbose=False)['peaks']
    assert len(peaks) == 3
    assert all(hasattr(p, 'window_id') for p in peaks)
    expected = ['w1', None, None] if metadata == "short" else [None] * 3
    assert [p.window_id for p in peaks] == expected


def test_event_handler_changing_count_raises(monkeypatch):
    class Drop(BaseModel):
        type: str = 'drop'

    monkeypatch.setitem(window_events.EVENT_TYPES, 'drop', Drop)
    monkeypatch.setitem(window_events.EVENT_HANDLERS, 'drop',
                        lambda event, window, peaks, data: peaks[:-1])

    class P:
        window_id = 'w1'

    data = {'peak_windows': [{'id': 'w1', 'events': [{'type': 'drop'}]}]}
    with pytest.raises(ValueError, match='changed the peak count'):
        apply_window_events([P(), P()], data)
