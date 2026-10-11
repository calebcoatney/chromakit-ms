"""Tests for POST /api/spectral-deconvolution."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _peak_dict(rt=2.5, peak_number=1):
    return {
        'compound_id': 'unknown',
        'peak_number': peak_number,
        'retention_time': rt,
        'integrator': 'BB',
        'width': 0.05,
        'area': 1000.0,
        'start_time': rt - 0.025,
        'end_time': rt + 0.025,
        'is_shoulder': False,
        'is_negative': False,
        'is_convoluted': False,
        'is_saturated': False,
        'is_grouped': False,
        'quality_issues': [],
    }


def test_deconvolution_returns_enriched_peaks(client, tmp_path):
    """The endpoint runs the pure deconv runner and returns serialized peaks."""
    fake_data_dir = tmp_path / "fake.D"
    fake_data_dir.mkdir()

    def fake_run(peaks, *args, **kwargs):
        # Pretend one component was found and assigned to the first peak
        import numpy as np
        peaks[0].deconvolved_spectrum = {
            'mz': np.array([50.0, 73.0]),
            'intensities': np.array([100.0, 200.0]),
        }
        peaks[0].deconvolution_component_count = 1
        return peaks

    with patch('api.main.run_spectral_deconvolution', side_effect=fake_run):
        response = client.post(
            '/api/spectral-deconvolution',
            json={
                'peaks': [_peak_dict(rt=2.5)],
                'ms_data_path': str(fake_data_dir),
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body['peaks']) == 1
    assert body['components_total'] >= 0
    assert body['components_assigned'] >= 0


def test_deconvolution_404_when_path_missing(client):
    response = client.post(
        '/api/spectral-deconvolution',
        json={
            'peaks': [_peak_dict()],
            'ms_data_path': '/no/such/data.D',
        },
    )
    assert response.status_code == 404


def test_deconvolution_422_on_empty_peaks(client, tmp_path):
    fake_data_dir = tmp_path / "fake.D"
    fake_data_dir.mkdir()
    response = client.post(
        '/api/spectral-deconvolution',
        json={'peaks': [], 'ms_data_path': str(fake_data_dir)},
    )
    assert response.status_code == 422


def test_deconvolution_422_on_old_adap_param_name(client, tmp_path):
    fake_data_dir = tmp_path / "fake.D"
    fake_data_dir.mkdir()
    with patch('api.main.run_spectral_deconvolution') as run:
        response = client.post('/api/spectral-deconvolution', json={
            'peaks': [_peak_dict()], 'ms_data_path': str(fake_data_dir),
            'deconv_params': {'shape_sim_threshold': 30.0},
        })
    assert response.status_code == 422
    detail = response.json()['detail']
    assert 'shape_sim_threshold' in detail and 'valid fields' in detail
    run.assert_not_called()


def test_deconvolution_passes_new_params(client, tmp_path):
    from logic.correlation_grouping import CorrelationDeconvParams
    fake_data_dir = tmp_path / "fake.D"
    fake_data_dir.mkdir()
    seen = {}

    def fake_run(peaks, *args, **kwargs):
        seen['dp'] = kwargs['deconv_params']
        return peaks

    with patch('api.main.run_spectral_deconvolution', side_effect=fake_run):
        response = client.post('/api/spectral-deconvolution', json={
            'peaks': [_peak_dict()], 'ms_data_path': str(fake_data_dir),
            'deconv_params': {'min_r': 0.7, 'min_ions': 3},
        })
    assert response.status_code == 200
    assert isinstance(seen['dp'], CorrelationDeconvParams)
    assert seen['dp'].min_r == 0.7 and seen['dp'].min_ions == 3


def _post_with_params(client, tmp_path, **params):
    fake_data_dir = tmp_path / "fake.D"
    fake_data_dir.mkdir(exist_ok=True)
    body = {'peaks': [_peak_dict()], 'ms_data_path': str(fake_data_dir)}
    body.update(params)
    return client.post('/api/spectral-deconvolution', json=body)


def test_deconvolution_422_on_wrong_float_type_names_the_field(client, tmp_path):
    with patch('api.main.run_spectral_deconvolution') as run:
        response = _post_with_params(client, tmp_path, deconv_params={'min_r': 'high'})
    assert response.status_code == 422
    assert 'min_r' in response.json()['detail']
    run.assert_not_called()


def test_deconvolution_422_on_string_excluded_mz_list(client, tmp_path):
    """The QSettings string form '73, 147' is not a JSON list of numbers."""
    with patch('api.main.run_spectral_deconvolution') as run:
        response = _post_with_params(client, tmp_path,
                                     deconv_params={'excluded_mz': '73, 147'})
    assert response.status_code == 422
    assert 'excluded_mz' in response.json()['detail']
    run.assert_not_called()


@pytest.mark.parametrize('params, field_name', [
    ({'min_points': '4'}, 'min_points'),            # int field given a string
    ({'min_points': 4.5}, 'min_points'),            # int field given a float
    ({'min_ions': True}, 'min_ions'),               # bool is not an int
    ({'min_r': True}, 'min_r'),                     # bool is not a float
    ({'min_peak_height': None}, 'min_peak_height'), # null for a non-optional field
    ({'excluded_mz': None}, 'excluded_mz'),
    ({'excluded_mz': [73, 'x']}, 'excluded_mz'),    # list with a non-number
    ({'excluded_mz': [73, True]}, 'excluded_mz'),
])
def test_deconvolution_422_on_other_bad_deconv_values(client, tmp_path, params, field_name):
    with patch('api.main.run_spectral_deconvolution') as run:
        response = _post_with_params(client, tmp_path, deconv_params=params)
    assert response.status_code == 422
    assert field_name in response.json()['detail']
    run.assert_not_called()


def test_deconvolution_422_on_bad_grouping_value(client, tmp_path):
    with patch('api.main.run_spectral_deconvolution') as run:
        response = _post_with_params(client, tmp_path,
                                     grouping_params={'max_window_peaks': '10'})
    assert response.status_code == 422
    detail = response.json()['detail']
    assert 'max_window_peaks' in detail and 'grouping_params' in detail
    run.assert_not_called()


def test_deconvolution_good_typed_request_still_passes(client, tmp_path):
    """ints for float fields, a number list, and null gap_tolerance are all valid."""
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    seen = {}

    def fake_run(peaks, *args, **kwargs):
        seen['dp'] = kwargs['deconv_params']
        seen['gp'] = kwargs['grouping_params']
        return peaks

    with patch('api.main.run_spectral_deconvolution', side_effect=fake_run):
        response = _post_with_params(
            client, tmp_path,
            deconv_params={'min_r': 0.7, 'min_peak_height': 500, 'min_points': 3,
                           'excluded_mz': [73, 147.0]},
            grouping_params={'gap_tolerance': None, 'max_window_peaks': 5,
                             'rt_match_tolerance': 0.1},
        )
    assert response.status_code == 200
    assert isinstance(seen['dp'], CorrelationDeconvParams)
    assert seen['dp'].excluded_mz == [73, 147.0] and seen['dp'].min_points == 3
    assert isinstance(seen['gp'], WindowGroupingParams)
    assert seen['gp'].gap_tolerance is None and seen['gp'].max_window_peaks == 5
