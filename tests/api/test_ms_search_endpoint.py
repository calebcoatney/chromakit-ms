"""Tests for POST /api/ms/search."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api import ms_toolkit_singleton as singleton


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


def test_ms_search_409_when_library_not_loaded(client):
    """Without a loaded toolkit, the endpoint returns 409 with a clear message."""
    response = client.post(
        '/api/ms/search',
        json={'spectrum': {'mz': [50.0], 'intensities': [1000.0]}},
    )
    assert response.status_code == 409
    assert 'library' in response.json()['detail'].lower()


def test_ms_search_returns_top_n_results(client):
    """A successful search returns hits in MSSearchResponse shape."""
    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = [
        ("Hexane", 0.93),
        ("Pentane", 0.51),
    ]
    compound = MagicMock()
    compound.casno = "110543"
    ms_toolkit.library = {"Hexane": compound}
    singleton._toolkit = ms_toolkit
    singleton._loaded_paths = {'library_path': '/x'}

    response = client.post(
        '/api/ms/search',
        json={
            'spectrum': {'mz': [50.0, 73.0], 'intensities': [1000.0, 500.0]},
            'options': {'search_method': 'vector'},
            'top_n': 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body['results']) == 2
    assert body['results'][0]['name'] == 'Hexane'
    assert body['results'][0]['score'] == 0.93
    assert body['results'][0]['casno'] == '000110-54-3'  # format_casno padding
    assert body['results'][1]['casno'] is None  # Pentane not in fake library
    assert 'elapsed_seconds' in body


def test_ms_search_422_on_empty_spectrum(client):
    """An empty mz array returns 422."""
    singleton._toolkit = MagicMock()
    response = client.post(
        '/api/ms/search',
        json={'spectrum': {'mz': [], 'intensities': []}},
    )
    assert response.status_code == 422


def test_ms_search_422_on_mismatched_mz_intensity_lengths(client):
    """mz and intensities of different lengths returns 422."""
    singleton._toolkit = MagicMock()
    response = client.post(
        '/api/ms/search',
        json={'spectrum': {'mz': [50.0, 73.0], 'intensities': [1000.0]}},
    )
    assert response.status_code == 422
    assert 'equal length' in response.json()['detail'].lower()


def test_ms_search_applies_mz_shift_to_toolkit(client):
    """The mz_shift field mutates ms_toolkit.mz_shift before the search."""
    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = []
    ms_toolkit.library = {}
    singleton._toolkit = ms_toolkit

    response = client.post(
        '/api/ms/search',
        json={
            'spectrum': {'mz': [50.0], 'intensities': [1000.0]},
            'mz_shift': 2,
        },
    )

    assert response.status_code == 200
    assert ms_toolkit.mz_shift == 2


def test_ms_search_absorbs_5_tuple_probability_shape(client):
    """When toolkit returns 5-tuples (compute_probability=True), the endpoint still works."""
    ms_toolkit = MagicMock()
    # Simulate compute_probability=True returning 5-tuples
    ms_toolkit.search_vector.return_value = [
        ("Hexane", 0.93, 0.87, 0.05, 0.02),  # (name, score, prob, p_low, p_high)
    ]
    ms_toolkit.library = {}
    singleton._toolkit = ms_toolkit

    response = client.post(
        '/api/ms/search',
        json={'spectrum': {'mz': [50.0], 'intensities': [1000.0]}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['results'][0]['name'] == 'Hexane'
    assert body['results'][0]['score'] == 0.93
    # Probability fields are silently dropped — that's the documented behavior.


def test_ms_search_top_n_propagates_to_options(client):
    """The top_n convenience field flows into options['top_n']."""
    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = []
    ms_toolkit.library = {}
    singleton._toolkit = ms_toolkit

    response = client.post(
        '/api/ms/search',
        json={
            'spectrum': {'mz': [50.0], 'intensities': [1000.0]},
            'top_n': 10,
        },
    )

    assert response.status_code == 200
    # Inspect the call args to confirm top_n was passed
    call_kwargs = ms_toolkit.search_vector.call_args.kwargs
    assert call_kwargs.get('top_n') == 10


def test_ms_search_explicit_options_top_n_wins_over_convenience_field(client):
    """When options['top_n'] is set, it takes precedence over the top-level top_n."""
    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = []
    ms_toolkit.library = {}
    singleton._toolkit = ms_toolkit

    response = client.post(
        '/api/ms/search',
        json={
            'spectrum': {'mz': [50.0], 'intensities': [1000.0]},
            'options': {'top_n': 3},
            'top_n': 10,  # should be ignored because options['top_n'] is set
        },
    )

    assert response.status_code == 200
    call_kwargs = ms_toolkit.search_vector.call_args.kwargs
    assert call_kwargs.get('top_n') == 3, (
        f"options['top_n']=3 must win over top-level top_n=10, got {call_kwargs.get('top_n')}"
    )
