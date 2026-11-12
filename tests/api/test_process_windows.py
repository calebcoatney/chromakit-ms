"""/api/process accepts peak windows and break overrides."""
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.utils import convert_params_for_processor
from api.models import ProcessingParams
from logic.method import ChromaMethod
from logic.processor import ChromatogramProcessor
from tests.logic.piecewise_parity_cases import anh_like

client = TestClient(app)


def _request(params_overrides, min_prominence=50.0):
    x, y, params, _ = anh_like()
    body_params = {
        'smoothing': {'enabled': False},
        'baseline': {'enabled': False},
        'peaks': {'enabled': True, 'min_prominence': min_prominence, 'min_width': 0},
    }
    body_params.update(params_overrides)
    return {'x': x.tolist(), 'y': y.tolist(), 'params': body_params}


def _rts(resp):
    assert resp.status_code == 200, resp.text
    return [m['retention_time'] if 'retention_time' in m else None
            for m in resp.json()['peak_metadata']], resp.json()['peaks_x']


def test_process_endpoint_applies_window_override():
    plain = client.post('/api/process', json=_request({}))
    _, plain_x = _rts(plain)
    assert not [t for t in plain_x if 4.2 < t < 5.1]  # cluster hidden by global prominence

    windowed = client.post('/api/process', json=_request({
        'outside_windows': 'global',
        'peak_windows': [{'id': 'w1', 'start': 4.2, 'end': 5.1,
                          'peaks': {'min_prominence': 5.0}}],
    }))
    _, win_x = _rts(windowed)
    assert len([t for t in win_x if 4.2 < t < 5.1]) >= 4
    assert any(abs(t - 4.0) < 0.05 for t in win_x)  # big peak kept (outside=global)


def test_process_endpoint_range_filters_only_still_filters():
    """No peak_windows in the request: peaks.range_filters keeps its meaning."""
    resp = client.post('/api/process', json=_request(
        {'peaks': {'enabled': True, 'min_prominence': 5.0, 'min_width': 0,
                   'range_filters': [[4.55, 5.1]]}}))
    _, xs = _rts(resp)
    assert xs and all(4.55 <= t <= 5.1 for t in xs)


def test_convert_normalises_lambda_in_windows_and_breaks():
    params = ProcessingParams(**{
        'peak_windows': [{'id': 'w1', 'start': 1, 'end': 2,
                          'smoothing': {'lambda': 123.0}}],
        'baseline': {'break_points': [{'time': 3.0, 'overrides': {'lambda': 7.0}}]},
    }).model_dump()  # no by_alias: keys arrive as 'lambda_'
    out = convert_params_for_processor(params)
    assert out['peak_windows'][0]['smoothing']['lambda'] == 123.0
    assert 'lambda_' not in out['peak_windows'][0]['smoothing']
    assert out['baseline']['break_points'][0]['overrides']['lambda'] == 7.0
    assert 'lambda_' not in out['baseline']['break_points'][0]['overrides']


def test_empty_windows_defer_to_range_filters_key_dropped():
    params = ProcessingParams(peaks={'range_filters': [[1.0, 2.0]]}).model_dump(by_alias=True)
    out = convert_params_for_processor(params)
    assert 'peak_windows' not in out


def test_method_params_roundtrip_through_convert_and_process():
    x, y, _, _ = anh_like()
    method = ChromaMethod(
        name='m', signal_type='gc',
        smoothing={'enabled': False}, baseline={'enabled': False},
        peaks={'enabled': True, 'min_prominence': 50.0, 'min_width': 0},
        outside_windows='global',
        peak_windows=[{'id': 'w1', 'start': 4.2, 'end': 5.1,
                       'peaks': {'min_prominence': 5.0}}])
    params = convert_params_for_processor(method.to_processor_params())
    processed = ChromatogramProcessor().process(x, y, params)
    assert len([t for t in processed['peaks_x'] if 4.2 < t < 5.1]) >= 4
