"""Tests for POST /api/ms/batch-search."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api import ms_toolkit_singleton as singleton
from logic.ms_search_core import BatchSearchSummary


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_singleton():
    singleton._toolkit = None
    singleton._loaded_paths = None
    yield
    singleton._toolkit = None
    singleton._loaded_paths = None


def _peak_dict(rt=2.5, peak_number=1):
    return {
        'compound_id': 'unknown', 'peak_number': peak_number,
        'retention_time': rt, 'integrator': 'BB', 'width': 0.05,
        'area': 1000.0, 'start_time': rt - 0.025, 'end_time': rt + 0.025,
        'is_shoulder': False, 'is_negative': False, 'is_convoluted': False,
        'is_saturated': False, 'is_grouped': False, 'quality_issues': [],
    }


def test_batch_search_409_when_library_not_loaded(client, tmp_path):
    fake_dir = tmp_path / "x.D"
    fake_dir.mkdir()
    response = client.post(
        '/api/ms/batch-search',
        json={
            'peaks': [_peak_dict()],
            'data_directory': str(fake_dir),
            'options': {'search_method': 'vector'},
        },
    )
    assert response.status_code == 409


def test_batch_search_returns_enriched_peaks_and_summary(client, tmp_path):
    fake_dir = tmp_path / "x.D"
    fake_dir.mkdir()

    # Load a fake singleton
    singleton._toolkit = MagicMock()
    singleton._loaded_paths = {'library_path': '/x'}

    def fake_search(ms_toolkit, peaks, data_directory, options, **kwargs):
        # Pretend one peak matched
        peaks[0].compound_id = "Hexane"
        peaks[0].Compound_ID = "Hexane"
        peaks[0].Qual = 0.91
        peaks[0].casno = "110-54-3"
        return BatchSearchSummary(
            total_peaks=len(peaks),
            successful_matches=1,
            saturated_peaks=0,
            cancelled=False,
            errors=[],
        )

    with patch('api.main.run_batch_search', side_effect=fake_search):
        response = client.post(
            '/api/ms/batch-search',
            json={
                'peaks': [_peak_dict()],
                'data_directory': str(fake_dir),
                'options': {'search_method': 'vector', 'top_n': 5},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body['total_peaks'] == 1
    assert body['successful_matches'] == 1
    assert body['peaks'][0]['compound_id'] == "Hexane"
    assert body['peaks'][0]['Qual'] == 0.91


def test_batch_search_404_when_data_directory_missing(client):
    singleton._toolkit = MagicMock()
    singleton._loaded_paths = {'library_path': '/x'}
    response = client.post(
        '/api/ms/batch-search',
        json={
            'peaks': [_peak_dict()],
            'data_directory': '/no/such/dir.D',
            'options': {'search_method': 'vector'},
        },
    )
    assert response.status_code == 404


def test_batch_search_422_on_empty_peaks(client, tmp_path):
    fake_dir = tmp_path / "x.D"
    fake_dir.mkdir()
    singleton._toolkit = MagicMock()
    response = client.post(
        '/api/ms/batch-search',
        json={
            'peaks': [],
            'data_directory': str(fake_dir),
            'options': {'search_method': 'vector'},
        },
    )
    assert response.status_code == 422


def test_batch_search_per_peak_errors_in_response_not_http(client, tmp_path):
    """A per-peak search error is in the response body, not an HTTP 500."""
    fake_dir = tmp_path / "x.D"
    fake_dir.mkdir()
    singleton._toolkit = MagicMock()

    def fake_search(ms_toolkit, peaks, *args, **kwargs):
        return BatchSearchSummary(
            total_peaks=len(peaks),
            successful_matches=0,
            saturated_peaks=0,
            cancelled=False,
            errors=[(0, "toolkit returned None for spectrum")],
        )

    with patch('api.main.run_batch_search', side_effect=fake_search):
        response = client.post(
            '/api/ms/batch-search',
            json={
                'peaks': [_peak_dict()],
                'data_directory': str(fake_dir),
                'options': {'search_method': 'vector'},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body['errors']) == 1
    assert body['errors'][0]['peak_index'] == 0
