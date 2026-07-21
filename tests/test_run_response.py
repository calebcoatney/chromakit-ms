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


# ─── .C folder (FTIR / UV-Vis) ingestion through /api/run ─────────────
#
# The HMI spectro_bridge packs a .C folder and POSTs its path to /api/run.
# ChromaKit has a full .C ingestion path (CFolder.open().load_signal()) wired
# into the GUI and AutomationWorker, but historically NOT into /api/run, which
# hard-rejected any non-.D path. These tests exercise the real loader (no
# mocking of data loading) so a raw ReactIR-style FTIR CSV can be processed
# end-to-end over HTTP — the path the bridge actually depends on.


def _write_ftir_c_folder(tmp_path):
    """Build a real .C FTIR folder from a 2-column headerless wavenumber,absorbance CSV.

    Mimics a Mettler Toledo ReactIR auto-export. Returns the .C folder path.
    """
    import numpy as np
    from logic.loaders.reactir_parser import parse_reactir_csv

    # A synthetic IR spectrum: flat baseline with one clear Gaussian absorbance band.
    wavenumbers = np.arange(4000.0, 400.0, -2.0)  # descending, like real ReactIR
    band = 5.0 * np.exp(-((wavenumbers - 1988.0) ** 2) / (2.0 * 15.0 ** 2))
    absorbance = 0.1 + band

    csv_path = tmp_path / "MoCFlow_2026-07-21_14-30-00_Spectrum.csv"
    with open(csv_path, "w") as f:
        for wn, ab in zip(wavenumbers, absorbance):
            f.write(f"{wn},{ab}\n")

    cf = parse_reactir_csv(str(csv_path))
    return cf.path


def _write_ftir_method(tmp_path):
    """A minimal ftir .chromethod that detects peaks (no MS, no deconvolution)."""
    method_path = tmp_path / "ir_nanoparticle.chromethod"
    method_path.write_text(
        '{"name": "ir_nanoparticle", "version": "1", "signal_type": "ftir", '
        '"chemstation_area_factor": 1.0, '
        '"smoothing": {"enabled": false}, '
        '"baseline": {"method": "asls"}, '
        '"peaks": {"enabled": true, "min_prominence": 0.5, "peak_prominence": 0.5, "min_height": 0.5}, '
        '"deconvolution": {"enabled": false}, '
        '"negative_peaks": {"enabled": false}, '
        '"shoulders": {"enabled": false}, '
        '"integration": {"peak_groups": []}}'
    )
    return str(method_path)


def test_run_endpoint_ingests_ftir_c_folder(tmp_path):
    """POSTing a real .C FTIR folder to /api/run processes it (does NOT 404).

    Exercises the true loader path — CFolder.open().load_signal() via CSVLoader —
    with no mocking. This is the exact call the HMI spectro_bridge makes.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    c_folder = _write_ftir_c_folder(tmp_path)
    method_path = _write_ftir_method(tmp_path)

    response = client.post(
        '/api/run',
        json={
            'data_path': c_folder,
            'method_path': method_path,
            'write_output': False,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['status'] == 'complete'
    assert body['signal_type'] == 'ftir'
    # The synthetic band at ~1988 cm-1 should be detected as at least one peak.
    assert body['peak_count'] >= 1


def test_run_endpoint_ftir_c_folder_yields_spectral_features(tmp_path):
    """FTIR .C data must be integrated with the profile's SpectralFeature class.

    The ftir signal profile declares feature_class=SpectralFeature. /api/run must
    thread the .C folder's profile into processing/integration so IR bands come
    back as spectral features (position on the wavenumber axis, absorbance/
    band_assignment fields) — NOT chromatography peaks keyed by retention_time.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    c_folder = _write_ftir_c_folder(tmp_path)
    method_path = _write_ftir_method(tmp_path)

    response = client.post(
        '/api/run',
        json={
            'data_path': c_folder,
            'method_path': method_path,
            'write_output': False,
        },
    )

    assert response.status_code == 200, response.text
    peaks = response.json()['peaks']
    assert peaks, "expected at least one detected band"
    peak = peaks[0]
    # SpectralFeature.as_dict emits 'position' + spectroscopy fields, not 'retention_time'.
    assert 'position' in peak, f"expected SpectralFeature fields, got keys: {list(peak.keys())}"
    assert 'retention_time' not in peak, (
        f"got ChromatographicPeak (retention_time) instead of SpectralFeature: {list(peak.keys())}"
    )
    assert 'absorbance' in peak
    assert 'band_assignment' in peak
    # The detected band should sit on the wavenumber axis near the synthetic 1988 band.
    assert peak['position_units'] == 'Wavenumber (cm⁻¹)'
