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
