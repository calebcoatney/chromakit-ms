"""Tests for the /api/export `output_path` override.

Uses FastAPI's TestClient — no live server needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_export_writes_to_custom_output_path(client, tmp_path):
    """When output_path is set, the JSON is written there."""
    out = tmp_path / "sweep_iter_42" / "result.json"
    # parent does not exist yet — handler must create it
    payload = {
        "peaks": [
            {"compound_id": "Hexane", "peak_number": 1, "retention_time": 2.5,
             "integrator": "BB", "width": 0.05, "area": 1000.0,
             "start_time": 2.45, "end_time": 2.55, "Qual": 0.91}
        ],
        "file_path": "/does/not/matter/when/output_path/is/set.D",
        "format": "json",
        "output_path": str(out),
    }
    response = client.post("/api/export", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "exported"
    assert body["output_file"] == str(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["peaks"][0]["compound_id"] == "Hexane"


def test_export_rejects_non_json_format(client, tmp_path):
    """Format other than 'json' returns 400 (existing behavior preserved)."""
    payload = {
        "peaks": [],
        "file_path": "/x.D",
        "format": "csv",
        "output_path": str(tmp_path / "x.csv"),
    }
    response = client.post("/api/export", json=payload)
    assert response.status_code == 400
