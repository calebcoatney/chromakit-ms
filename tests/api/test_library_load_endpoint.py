"""Tests for POST /api/ms/library/load."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import patch
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


def test_library_load_returns_200_and_summary(client):
    """A successful load returns the LibraryLoadResponse shape."""
    fake_summary = {
        'status': 'loaded',
        'compound_count': 99,
        'library_path': '/some/lib.json',
        'preselector_loaded': False,
        'w2v_loaded': False,
        'elapsed_seconds': 1.23,
    }
    with patch.object(singleton, 'load_library', return_value=fake_summary):
        response = client.post(
            '/api/ms/library/load',
            json={'library_path': '/some/lib.json'},
        )
    assert response.status_code == 200
    body = response.json()
    assert body['compound_count'] == 99
    assert body['status'] == 'loaded'


def test_library_load_404_when_path_missing(client):
    """FileNotFoundError from the singleton becomes HTTP 404."""
    with patch.object(singleton, 'load_library',
                       side_effect=FileNotFoundError("nope")):
        response = client.post(
            '/api/ms/library/load',
            json={'library_path': '/nope.json'},
        )
    assert response.status_code == 404


def test_library_load_500_on_unexpected_error(client):
    """Any other exception becomes HTTP 500."""
    with patch.object(singleton, 'load_library',
                       side_effect=RuntimeError("toolkit not installed")):
        response = client.post(
            '/api/ms/library/load',
            json={'library_path': '/x.json'},
        )
    assert response.status_code == 500
    assert "toolkit not installed" in response.json()['detail']
