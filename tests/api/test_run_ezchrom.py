"""POST /api/run on a .C converted from an EZChrom .dat (synthetic, reader patched)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from logic.loaders import ezchrom_parser as ez
from logic.method import ChromaMethod
from tests.logic.test_ezchrom_parser import _streams


@pytest.fixture
def c_folder(tmp_path, monkeypatch):
    n = 3000                                   # 30 s at 100 Hz
    t_min = np.arange(n) * 0.01 / 60
    def peak(rt):                              # 1 mV Gaussian
        return 1e5 * np.exp(-((t_min - rt) ** 2) / (2 * 0.004 ** 2))
    # the two channels peak at different times, so the result shows which was read
    streams = _streams(n=n, ch1_counts=1000.0 + peak(0.10), ch2_counts=1000.0 + peak(0.25))
    monkeypatch.setattr(ez, "read_ezchrom_dat", lambda path: ez.read_ezchrom_streams(streams))
    rslt = tmp_path / "2025-01-01 20-00-00 (GMT -07-00)TEST-001.rslt"
    rslt.mkdir()
    dat = rslt / "long_001.dat"
    dat.write_bytes(b"placeholder")
    return ez.convert_dat(str(dat), str(tmp_path / "out"))


def _method(tmp_path, detector):
    m = ChromaMethod(name="microgc", signal_type="gc", detector=detector,
                     peaks={"enabled": True, "min_prominence": 0.1})
    path = tmp_path / "m.chromethod"
    m.to_file(str(path))
    return str(path)


def test_run_picks_the_method_detector(c_folder, tmp_path):
    resp = TestClient(app).post("/api/run", json={
        "data_path": c_folder, "method_path": _method(tmp_path, "Ch2_5CB"), "write_output": False})
    assert resp.status_code == 200, resp.text
    biggest = max(resp.json()["peaks"], key=lambda p: p["area"])
    assert biggest["retention_time"] == pytest.approx(0.25, abs=0.002)


def test_run_unknown_detector_is_an_error_not_a_fallback(c_folder, tmp_path):
    resp = TestClient(app).post("/api/run", json={
        "data_path": c_folder, "method_path": _method(tmp_path, "FID1A"), "write_output": False})
    assert resp.status_code == 404
    assert "FID1A" in resp.json()["detail"]
