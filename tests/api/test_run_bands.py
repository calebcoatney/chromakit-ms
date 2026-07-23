import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _make_ftir_c_folder(tmp):
    """Create a minimal FTIR .C folder with one spectrum via the real CFolder API.

    CFolder.create(csv, "ftir") moves a source CSV into <stem>.C/data/ and writes
    manifest.json. The CSVLoader defaults (has_header=True, x/y columns) are used
    at load time, so the source CSV must carry an "x,y" header.
    """
    from logic.c_folder import CFolder
    x = np.linspace(2200, 1800, 401)  # descending wavenumbers (FTIR)
    y = 1.5 * np.exp(-((x - 1987.0) ** 2) / (2 * 4.0 ** 2))  # band at ~1987
    csv = tmp / "spectrum.csv"
    csv.write_text("x,y\n" + "\n".join(f"{xi},{yi}" for xi, yi in zip(x, y)))
    folder = CFolder.create(str(csv), "ftir", sample_id="TEST-BANDS")
    return folder.path


def test_run_with_bands_returns_named_features(tmp_path):
    data_c = _make_ftir_c_folder(tmp_path)
    method = {
        "name": "ir_bands",
        "signal_type": "ftir",
        "baseline": {"enabled": False},
        "peaks": {"enabled": False, "min_prominence": 0.02},
        "bands": [
            {"name": "precursor_CO", "x_min": 1970, "x_max": 2005},
            {"name": "np_broad", "x_min": 800, "x_max": 900},  # empty window
        ],
    }
    method_path = tmp_path / "ir.chromethod"
    method_path.write_text(json.dumps(method))

    resp = client.post("/api/run", json={
        "method_path": str(method_path),
        "data_path": str(data_c),
        "write_output": False,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["peak_count"] == 2
    names = [p["band_assignment"] for p in body["peaks"]]
    assert names == ["precursor_CO", "np_broad"]
    areas = {p["band_assignment"]: p["area"] for p in body["peaks"]}
    assert areas["precursor_CO"] > 0
    assert areas["np_broad"] == 0.0
