# tests/api/test_render_endpoint.py
"""The read-only render route.

Bad REQUESTS are HTTP errors. Bad DATA is a 200 with the failure named in the
body — a sample that exists but was never analysed is a fact, not an error.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

RECORDS = Path(__file__).parent.parent / "data" / "records"


@pytest.fixture
def client():
    return TestClient(app)


def _record_doc():
    manifest = json.loads((RECORDS / "MANIFEST.json").read_text())
    return json.loads((RECORDS / manifest[0]["fixture"]).read_text())


def _sample_with_record(tmp_path) -> tuple[str, dict]:
    doc = _record_doc()
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    (sample / f"SAMPLE - {doc['detector']}.json").write_text(json.dumps(doc))
    return str(sample), doc


def test_missing_path_is_404(client):
    r = client.get("/api/render", params={"path": "/no/such/sample.D",
                                          "detector": "FID1A"})
    assert r.status_code == 404


def test_a_path_that_is_neither_d_nor_c_is_400(client, tmp_path):
    plain = tmp_path / "plain_folder"
    plain.mkdir()
    r = client.get("/api/render", params={"path": str(plain), "detector": "FID1A"})
    assert r.status_code == 400


def test_max_points_below_two_is_422(client, tmp_path):
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample,
                                          "detector": doc["detector"],
                                          "max_points": 1})
    assert r.status_code == 422


def test_max_points_above_the_cap_is_422(client, tmp_path):
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample,
                                          "detector": doc["detector"],
                                          "max_points": 20001})
    assert r.status_code == 422


def test_unanalysed_sample_is_200_with_a_named_error_not_a_404(client, tmp_path):
    """A 404 here leaves a widget unable to tell 'no such injection' from
    'nothing has run yet'."""
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    r = client.get("/api/render", params={"path": str(sample), "detector": "FID1A"})
    assert r.status_code == 200
    channel = r.json()["channel"]
    assert channel["peaks"] == []
    assert "FID1A" in channel["error"]


def test_unreadable_signal_still_returns_the_recorded_peaks(client, tmp_path):
    """The .D here holds a record but no .ch, so the trace fails and the peaks
    must survive."""
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert r.status_code == 200
    channel = r.json()["channel"]
    assert channel["trace"] is None
    assert len(channel["peaks"]) == len(doc["peaks"])
    assert channel["error"]


def test_corrupt_record_is_500_naming_the_path(client, tmp_path):
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    (sample / "SAMPLE - FID1A.json").write_text("{ not json")
    r = client.get("/api/render", params={"path": str(sample), "detector": "FID1A"})
    assert r.status_code == 500
    assert "SAMPLE - FID1A.json" in r.json()["detail"]


def test_channel_block_has_exactly_the_contract_b_keys(client, tmp_path):
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert set(r.json()["channel"]) == {
        "method", "trace", "peaks", "closure_percent", "error"}


def test_peaks_have_exactly_the_contract_b_peak_keys(client, tmp_path):
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    peak = r.json()["channel"]["peaks"][0]
    assert set(peak) == {"compound", "rt_apex", "rt_start", "rt_end",
                         "area", "raw_amount", "mol_percent"}


def test_closure_percent_is_null(client, tmp_path):
    """Recorded compositions are already normalized to 100, so the gap is not
    recoverable and 100.0 would be fabricated."""
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert r.json()["channel"]["closure_percent"] is None


def test_method_block_is_labelled_as_the_acquisition_method(client, tmp_path):
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    method = r.json()["channel"]["method"]
    assert method["source"] == "agilent_acquisition"
    assert method["name"] == doc.get("method")


def test_sample_block_carries_chromakit_context(client, tmp_path):
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    block = r.json()["sample"]
    assert block["detector"] == doc["detector"]
    assert block["record_path"].endswith(".json")
    assert "signal_factor" in block


def test_the_route_does_not_touch_the_shared_data_handler(client, tmp_path):
    """The module-level singleton's current_detector is overwritten by every
    load; sharing it would let two tabs corrupt each other."""
    from api.main import data_handler
    before = data_handler.current_detector
    sample, doc = _sample_with_record(tmp_path)
    client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert data_handler.current_detector == before
