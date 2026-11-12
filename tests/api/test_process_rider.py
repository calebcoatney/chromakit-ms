"""/api/process returns the rider settings, so /api/process -> /api/integrate can skim."""
from fastapi.testclient import TestClient

from api.main import app
from tests.logic.tail_skim_signals import rider_on_tail

client = TestClient(app)


def _process(rider=None):
    pair = rider_on_tail(1.0, 3.0)
    params = {'smoothing': {'enabled': False}, 'baseline': {'enabled': False},
              'peaks': {'enabled': True, 'min_prominence': 0.01, 'min_width': 0}}
    if rider is not None:
        params['rider'] = rider
    resp = client.post('/api/process', json={'x': pair['x'].tolist(), 'y': pair['y'].tolist(),
                                             'params': params})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_process_response_carries_rider_and_integrate_skims():
    processed = _process({'rider_integration': 'auto'})
    assert processed['rider']['rider_integration'] == 'auto'
    assert len(processed['peaks_x']) == 2
    resp = client.post('/api/integrate', json={'processed_data': processed})
    assert resp.status_code == 200, resp.text
    parent, rider = resp.json()['peaks']
    assert rider['skim_parent'] == parent['peak_number']
    assert parent['skim_children'] == [rider['peak_number']]


def test_process_response_rider_default_is_split():
    processed = _process()
    assert processed['rider']['rider_integration'] == 'split'
    peaks = client.post('/api/integrate', json={'processed_data': processed}).json()['peaks']
    assert all('skim_parent' not in p for p in peaks)
