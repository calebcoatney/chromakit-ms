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

import logic.render_trace as render_trace
from api.main import app

RECORDS = Path(__file__).parent.parent / "data" / "records"


@pytest.fixture
def client():
    return TestClient(app)


def _record_doc():
    manifest = json.loads((RECORDS / "MANIFEST.json").read_text())
    return json.loads((RECORDS / manifest[0]["fixture"]).read_text())


def _write_sample(tmp_path, doc) -> str:
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    (sample / f"SAMPLE - {doc['detector']}.json").write_text(json.dumps(doc))
    return str(sample)


def _sample_with_record(tmp_path) -> tuple[str, dict]:
    doc = _record_doc()
    return _write_sample(tmp_path, doc), doc


def _sample_with_fixture(tmp_path, fixture_name) -> tuple[str, dict]:
    doc = json.loads((RECORDS / fixture_name).read_text())
    return _write_sample(tmp_path, doc), doc


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


def test_unanalysed_sample_still_draws_a_populated_trace(client, tmp_path, monkeypatch):
    """The whole point of the 200-with-error design: an unanalysed sample
    still DRAWS, because the drawable signal and the recorded numbers are
    independent sources. This is the endpoint's primary path, and until this
    test existed, `RenderTrace(**trace.__dict__)` — the one line that puts
    drawable data on the wire — was never executed by any test in this file:
    every other test runs in a state where `trace is None`. It is a
    name-coupled reflection bridge, so a field rename in `logic.render_trace
    .Trace` would raise a ValidationError -> unhandled 500 here with nothing
    else in the repo noticing.
    """
    xs = [i * 0.1 for i in range(50)]
    ys = [float(i % 7) for i in range(50)]

    def fake_read_signal(sample_path, detector=None):
        return xs, ys, detector or "FID1A"

    monkeypatch.setattr(render_trace, "read_signal", fake_read_signal)

    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    r = client.get("/api/render", params={"path": str(sample), "detector": "FID1A"})
    assert r.status_code == 200
    channel = r.json()["channel"]
    assert channel["peaks"] == []
    assert channel["error"]
    trace = channel["trace"]
    assert trace is not None
    assert trace["points"] > 0
    assert len(trace["rt_min"]) == len(trace["plot_y"]) == trace["points"]


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


def test_signal_path_falls_back_to_the_requested_path_when_source_d_is_gone(
        client, tmp_path, monkeypatch):
    """An archival record's `source_d` can name storage that has since moved.
    The .D actually named in THIS request may still hold a perfectly readable
    signal, and that must not be abandoned in favor of a dead recorded path —
    precisely the records this feature exists to serve. `schema01.json` (the
    default fixture used elsewhere in this file) has no `source_d` at all, so
    this fallback branch is otherwise never exercised."""
    doc = dict(_record_doc())
    doc["source_d"] = str(tmp_path / "nowhere" / "GONE.D")  # does not exist
    sample = _write_sample(tmp_path, doc)

    xs = [i * 0.1 for i in range(20)]
    ys = [float(i) for i in range(20)]
    seen_paths = []

    def fake_read_signal(sample_path, detector=None):
        seen_paths.append(sample_path)
        return xs, ys, detector or doc["detector"]

    monkeypatch.setattr(render_trace, "read_signal", fake_read_signal)

    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert r.status_code == 200
    body = r.json()
    # The requested path was used for the signal, not the dead source_d.
    assert body["sample"]["signal_path"] == sample
    assert body["channel"]["trace"] is not None
    assert seen_paths == [sample]


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

    # Values, not just key names: pins the RecordPeak -> RenderPeak mapping
    # seam. Before this, a bare `RenderPeak()` (every field at its None
    # default), or `rt_start`/`rt_end` swapped, or `area` wired into
    # `raw_amount`, would all still pass every assertion in this file.
    doc_peak = doc["peaks"][0]
    assert peak["compound"] == doc_peak["Compound ID"]
    assert peak["rt_apex"] == doc_peak["retention_time"]
    assert peak["rt_start"] == doc_peak["start_time"]
    assert peak["rt_end"] == doc_peak["end_time"]
    assert peak["area"] == doc_peak["area"]
    assert peak["rt_start"] < peak["rt_apex"] < peak["rt_end"]


def test_closure_percent_is_null(client, tmp_path):
    """Recorded compositions are already normalized to 100, so the gap is not
    recoverable and 100.0 would be fabricated."""
    sample, doc = _sample_with_record(tmp_path)
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert r.json()["channel"]["closure_percent"] is None


def test_method_block_is_labelled_as_the_acquisition_method(client, tmp_path):
    """schema03.json (method='TC DEF_GC') is used here rather than the file's
    default fixture: schema01.json's `method` is `""`, which would let
    `method["name"] == doc.get("method")` pass as `"" == ""` regardless of
    whether the route ever reads the field."""
    sample, doc = _sample_with_fixture(tmp_path, "schema03.json")
    assert doc["method"], "fixture must carry a non-empty method for this test to mean anything"
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    method = r.json()["channel"]["method"]
    assert method["source"] == "agilent_acquisition"
    assert method["name"] == doc["method"] == "TC DEF_GC"


def test_method_source_is_null_when_the_recorded_method_is_empty(client, tmp_path):
    """An empty `method: ""` is not a recorded method. Gating `source` on
    `name is not None` would treat "" as non-null and falsely claim
    Agilent-acquisition provenance for nothing -- and this isn't an edge
    case: 3 of 11 distilled real records (schema01, schema02, schema05)
    carry an empty method string. `name` itself still reports the recorded
    (empty) value verbatim; only `source` -- the provenance claim -- is
    null."""
    sample, doc = _sample_with_fixture(tmp_path, "schema02.json")
    assert doc["method"] == "", "fixture must carry an empty method for this test to mean anything"
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    method = r.json()["channel"]["method"]
    assert method["name"] == ""
    assert method["source"] is None


def test_method_block_has_a_stable_shape_when_unanalysed(client, tmp_path):
    """A consumer doing `channel['method']['source']` must not KeyError on
    the common, not-yet-analysed path -- 442 of 453 real records carry no
    quantitation at all, so a `method` that changes shape between "has a
    record" and "doesn't" breaks on the majority of real samples. `source` is
    null (not fabricated as "agilent_acquisition") because there is no
    recorded name to attribute to Agilent."""
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    r = client.get("/api/render", params={"path": str(sample), "detector": "FID1A"})
    method = r.json()["channel"]["method"]
    assert set(method) == {"name", "source"}
    assert method["name"] is None
    assert method["source"] is None


def test_sample_block_carries_chromakit_context(client, tmp_path):
    """schema02.json is used here (rather than the file's default fixture,
    schema01.json, whose `scaling` block is absent) because it carries a
    real, non-null `signal_factor` (7680.0). Asserting against a fixture
    whose expected value is None would still pass for a route that never
    read `record.signal_factor` at all -- only a non-null expected value
    proves the field is actually propagated, not merely present under
    `response_model`'s always-serialize-every-field default."""
    sample, doc = _sample_with_fixture(tmp_path, "schema02.json")
    r = client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    block = r.json()["sample"]
    assert block["detector"] == doc["detector"]
    assert block["record_path"].endswith(".json")

    expected_signal_factor = doc["processing_parameters"]["scaling"]["signal_factor"]
    assert expected_signal_factor == 7680.0, "fixture must carry a non-null signal_factor"
    assert block["signal_factor"] == expected_signal_factor


# --- trailing separator (FIX 2 regression) -----------------------------
#
# A bare tmp_path .D with no real .ch files returns [] from
# available_detectors() whether or not the path carries a trailing
# separator, so it can't tell the pre-fix and post-fix behavior apart.
# This needs a sample with real, readable detector channels -- guarded the
# same way as tests/logic/test_render_trace_read.py's real-sample sweep.
#
# Point it elsewhere with:  CHROMAKIT_RENDER_SAMPLE=/some/path pytest ...

DEFAULT_REAL_SAMPLE = (
    Path.home() / "Library/CloudStorage/OneDrive-NREL/GC-MS code development"
    / "Hyewon's data" / "260224 GC1 R1 Run 311 outlet 25.D"
)


def _real_sample_path() -> Path:
    override = os.environ.get("CHROMAKIT_RENDER_SAMPLE")
    return Path(override) if override else DEFAULT_REAL_SAMPLE


REAL_SAMPLE = _real_sample_path()


@pytest.mark.skipif(not REAL_SAMPLE.is_dir(), reason="real render sample not present on this machine")
def test_trailing_separator_gives_the_same_result_as_without_one_no_detector(client):
    """Before the fix: no trailing slash auto-detects a channel and returns
    200 with a trace; a trailing slash makes `rb.read` fail inside
    `available_detectors`, so no detector is found and the route 400s."""
    path = str(REAL_SAMPLE)
    r_plain = client.get("/api/render", params={"path": path})
    r_slash = client.get("/api/render", params={"path": path + "/"})

    assert r_plain.status_code == r_slash.status_code == 200
    assert r_plain.json() == r_slash.json()


@pytest.mark.skipif(not REAL_SAMPLE.is_dir(), reason="real render sample not present on this machine")
def test_trailing_separator_gives_the_same_result_as_without_one_explicit_detector(client):
    """Before the fix: with a detector given, no trailing slash returns 200
    with a populated trace; a trailing slash makes DataHandler's own '.D'
    suffix check reject the path, so the route still 200s (bad DATA, not a
    bad ARGUMENT) but with trace=None -- a silent quality regression with no
    error naming the real cause."""
    path = str(REAL_SAMPLE)
    r_plain = client.get("/api/render", params={"path": path, "detector": "FID1A"})
    r_slash = client.get("/api/render", params={"path": path + "/", "detector": "FID1A"})

    assert r_plain.status_code == r_slash.status_code == 200
    assert r_plain.json() == r_slash.json()
    assert r_plain.json()["channel"]["trace"] is not None


def test_the_route_does_not_touch_the_shared_data_handler(client, tmp_path, monkeypatch):
    """The module-level singleton's current_detector is overwritten by every
    load; sharing it would let two tabs corrupt each other. A sentinel that
    this fixture's detector could never coincidentally equal is required
    here: `DataHandler.current_detector` defaults to `"Unknown"`, which is
    also this file's default fixture's `detector` value, so comparing
    before/after against that shared default could never observe a
    regression that reused the singleton."""
    from api.main import data_handler
    monkeypatch.setattr(data_handler, "current_detector", "SENTINEL-XYZ")
    sample, doc = _sample_with_record(tmp_path)
    client.get("/api/render", params={"path": sample, "detector": doc["detector"]})
    assert data_handler.current_detector == "SENTINEL-XYZ"
