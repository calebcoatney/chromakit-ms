"""Tests for POST /api/quantitate."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api import ms_toolkit_singleton as singleton
from logic.quantitation_runner import (
    QuantitationSummary, CompoundMetadata,
)


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


def _peak_dict(compound_id="Hexane", area=1000.0, peak_number=1):
    return {
        'compound_id': compound_id, 'peak_number': peak_number,
        'retention_time': 2.5, 'integrator': 'BB', 'width': 0.05,
        'area': area, 'start_time': 2.45, 'end_time': 2.55,
        'is_shoulder': False, 'is_negative': False, 'is_convoluted': False,
        'is_saturated': False, 'is_grouped': False, 'quality_issues': [],
    }


def _is_payload():
    return {
        'compound_name': 'Decane',
        'volume_uL': 1.0,
        'density_g_mL': 0.73,
        'molecular_weight': 142.28,
        'formula': 'C10H22',
    }


def test_quantitate_409_when_library_not_loaded(client):
    response = client.post(
        '/api/quantitate',
        json={'peaks': [_peak_dict()], 'internal_standard': _is_payload()},
    )
    assert response.status_code == 409


def test_quantitate_returns_summary_when_is_found(client):
    singleton._toolkit = MagicMock()

    def fake_run(peaks, internal_standard, sample, compound_lookup):
        peaks[0].mol_C = 1e-7
        peaks[0].num_carbons = 6
        return QuantitationSummary(
            internal_standard_peak_index=0,
            response_factor=2e13,
            peaks_quantitated=1,
            warnings=[],
        )

    with patch('api.main.run_quantitation', side_effect=fake_run):
        response = client.post(
            '/api/quantitate',
            json={
                'peaks': [_peak_dict(compound_id="Decane")],
                'internal_standard': _is_payload(),
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body['internal_standard_peak_index'] == 0
    assert body['response_factor'] == 2e13
    assert body['peaks_quantitated'] == 1


def test_quantitate_returns_200_with_minus_one_when_is_missing(client):
    """IS-not-found is 200 with internal_standard_peak_index=-1 and a warning."""
    singleton._toolkit = MagicMock()

    def fake_run(peaks, internal_standard, sample, compound_lookup):
        return QuantitationSummary(
            internal_standard_peak_index=-1,
            warnings=["Internal standard 'Decane' not found in peaks"],
        )

    with patch('api.main.run_quantitation', side_effect=fake_run):
        response = client.post(
            '/api/quantitate',
            json={
                'peaks': [_peak_dict(compound_id="Benzene")],
                'internal_standard': _is_payload(),
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body['internal_standard_peak_index'] == -1
    assert body['peaks_quantitated'] == 0
    assert len(body['warnings']) >= 1


def test_quantitate_422_when_is_missing_required_fields(client):
    """Malformed internal_standard -> 422."""
    singleton._toolkit = MagicMock()
    response = client.post(
        '/api/quantitate',
        json={
            'peaks': [_peak_dict()],
            'internal_standard': {'compound_name': 'Decane'},  # missing volume etc.
        },
    )
    assert response.status_code == 422
