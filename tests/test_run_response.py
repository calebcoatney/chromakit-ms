"""Tests for api/models.py — RunResponse Pydantic shape.

Added as part of the cross-repo spectro-bridge payload-enrichment slice
(spec lives in the vault at
docs/superpowers/specs/2026-06-04-spectro-bridge-payload-enrichment-design.md).
This test pins the contract the spectro_bridge depends on.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from pydantic import ValidationError

from api.models import RunResponse


def _base_kwargs() -> dict:
    """Minimal valid RunResponse kwargs, missing only `version`."""
    return dict(
        status="complete",
        data_path="/tmp/sample.D",
        method="ir_nanoparticle",
        signal_type="ftir",
        peak_count=0,
        peaks=[],
        output_files=["/tmp/out.json"],
    )


def test_run_response_requires_version():
    """RunResponse must reject construction without an explicit `version`."""
    with pytest.raises(ValidationError) as excinfo:
        RunResponse(**_base_kwargs())
    # The error must mention the missing `version` field.
    assert "version" in str(excinfo.value)


def test_run_response_accepts_string_version():
    """`version` is a required field that accepts any string value, e.g. `"1"`."""
    resp = RunResponse(version="1", **_base_kwargs())
    assert resp.version == "1"


def test_run_response_round_trip_with_version():
    """`version` survives a model_dump → model_validate round-trip."""
    resp = RunResponse(version="2", **_base_kwargs())
    dumped = resp.model_dump()
    assert dumped["version"] == "2"
    restored = RunResponse.model_validate(dumped)
    assert restored.version == "2"


def test_run_response_version_appears_in_json():
    """`version` is included in the JSON serialization sent over the wire."""
    resp = RunResponse(version="3", **_base_kwargs())
    payload = resp.model_dump_json()
    assert '"version":"3"' in payload or '"version": "3"' in payload


def test_run_request_write_output_defaults_to_true():
    """write_output defaults to True to preserve back-compat with spectro_bridge."""
    from api.models import RunRequest
    req = RunRequest(data_path='/tmp/x.D', method_path='/tmp/m.chromethod')
    assert req.write_output is True


def test_run_request_write_output_accepts_false():
    """write_output can be set to False."""
    from api.models import RunRequest
    req = RunRequest(
        data_path='/tmp/x.D',
        method_path='/tmp/m.chromethod',
        write_output=False,
    )
    assert req.write_output is False


def test_run_endpoint_write_output_false_does_not_write_json(tmp_path, monkeypatch):
    """When write_output=False, export_integration_results_to_json is not called."""
    from unittest.mock import patch, MagicMock
    from fastapi.testclient import TestClient
    from api.main import app, data_handler

    client = TestClient(app)

    # Create a fake .D directory and method file
    fake_dir = tmp_path / "sample.D"
    fake_dir.mkdir()
    fake_method = tmp_path / "test.chromethod"
    fake_method.write_text('{"name": "t", "version": "1", "signal_type": "gc", "chemstation_area_factor": 0.0784, "smoothing": {"enabled": false}, "baseline": {"method": "asls"}, "peaks": {"min_height": 1.0}, "deconvolution": {"enabled": false}, "negative_peaks": {"enabled": false}, "shoulders": {"enabled": false}, "integration": {"peak_groups": []}}')

    fake_data = {
        'chromatogram': {'x': [0.0, 1.0], 'y': [100.0, 200.0]},
        'tic': {'x': [], 'y': []},
        'metadata': {'filename': 'sample.D'},
    }

    with patch.object(data_handler, 'load_data_directory', return_value=fake_data), \
         patch.object(data_handler, 'current_detector', 'FID1A'), \
         patch('api.main.processor.process', return_value={'x': [], 'corrected_y': []}), \
         patch('api.main.processor.integrate_peaks', return_value={'peaks': []}), \
         patch('api.main.export_integration_results_to_json') as mock_export, \
         patch('api.main._resolve_export_context', return_value=({}, '/fake/out.json')):
        response = client.post(
            '/api/run',
            json={
                'data_path': str(fake_dir),
                'method_path': str(fake_method),
                'write_output': False,
            },
        )

    assert response.status_code == 200
    mock_export.assert_not_called()
    assert response.json()['output_files'] == []


def test_run_endpoint_write_output_true_preserves_existing_behavior(tmp_path):
    """When write_output=True (default), JSON is still written. Regression for spectro_bridge."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from api.main import app, data_handler

    client = TestClient(app)

    fake_dir = tmp_path / "sample.D"
    fake_dir.mkdir()
    fake_method = tmp_path / "test.chromethod"
    fake_method.write_text('{"name": "t", "version": "1", "signal_type": "gc", "chemstation_area_factor": 0.0784, "smoothing": {"enabled": false}, "baseline": {"method": "asls"}, "peaks": {"min_height": 1.0}, "deconvolution": {"enabled": false}, "negative_peaks": {"enabled": false}, "shoulders": {"enabled": false}, "integration": {"peak_groups": []}}')

    fake_data = {
        'chromatogram': {'x': [0.0, 1.0], 'y': [100.0, 200.0]},
        'tic': {'x': [], 'y': []},
        'metadata': {'filename': 'sample.D'},
    }

    with patch.object(data_handler, 'load_data_directory', return_value=fake_data), \
         patch.object(data_handler, 'current_detector', 'FID1A'), \
         patch('api.main.processor.process', return_value={'x': [], 'corrected_y': []}), \
         patch('api.main.processor.integrate_peaks', return_value={'peaks': []}), \
         patch('api.main.export_integration_results_to_json') as mock_export, \
         patch('api.main._resolve_export_context', return_value=({}, '/fake/out.json')):
        response = client.post(
            '/api/run',
            json={
                'data_path': str(fake_dir),
                'method_path': str(fake_method),
                # no write_output → default True
            },
        )

    assert response.status_code == 200
    mock_export.assert_called_once()
    assert response.json()['output_files'] == ['/fake/out.json']
