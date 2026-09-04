# tests/logic/test_result_record_locate.py
"""Finding the record, and reading its sample-level half."""
import json
from pathlib import Path

import pytest

from logic.result_record import (
    RecordParseError, ResultRecord, find_record, read_record,
)

RECORDS = Path(__file__).parent.parent / "data" / "records"


def _fixture(index: int = 0) -> dict:
    manifest = json.loads((RECORDS / "MANIFEST.json").read_text())
    return json.loads((RECORDS / manifest[index]["fixture"]).read_text())


def _make_d(tmp_path, doc, filename=None) -> Path:
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir(parents=True, exist_ok=True)
    name = filename or f"SAMPLE - {doc['detector']}.json"
    (sample / name).write_text(json.dumps(doc))
    return sample


def test_finds_the_record_by_detector_field_not_filename(tmp_path):
    """Filenames drift; sample_id is sometimes 'SIG15397.D' and sometimes not."""
    doc = _fixture()
    sample = _make_d(tmp_path, doc, filename="wholly-unrelated-name.json")
    assert find_record(str(sample), doc["detector"]) == str(
        sample / "wholly-unrelated-name.json")


def test_ignores_a_record_for_a_different_detector(tmp_path):
    doc = _fixture()
    sample = _make_d(tmp_path, doc)
    assert find_record(str(sample), "NOSUCH9Z") is None


def test_canonical_filename_beats_a_sidecar(tmp_path):
    """The sidecar's suffix is chosen to sort BEFORE '.json' alphabetically
    ('0' < 'j'), so a naive 'first sorted match wins' implementation would
    pick the sidecar. Only genuine canonical-priority logic passes this."""
    doc = _fixture()
    detector = doc["detector"]
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    canonical = sample / f"SAMPLE - {detector}.json"
    sidecar = sample / f"SAMPLE - {detector}.0-backup.json"
    canonical.write_text(json.dumps(doc))
    sidecar.write_text(json.dumps(doc))
    assert find_record(str(sample), detector) == str(canonical)


def test_looks_in_results_subdir_for_a_c_folder(tmp_path):
    doc = _fixture()
    sample = tmp_path / "SAMPLE.C"
    (sample / "results").mkdir(parents=True)
    target = sample / "results" / f"SAMPLE - {doc['detector']}.json"
    target.write_text(json.dumps(doc))
    assert find_record(str(sample), doc["detector"]) == str(target)


def test_read_record_returns_none_when_nothing_matches(tmp_path):
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    assert read_record(str(sample), "FID1A") is None


def test_read_record_returns_none_when_peaks_are_absent(tmp_path):
    sample = _make_d(tmp_path, {"detector": "FID1A", "sample_id": "x"})
    assert read_record(str(sample), "FID1A") is None


def test_malformed_json_raises_rather_than_degrading(tmp_path):
    """A record that will not parse is a broken file or a bug. Rendering the
    sample as 'not analysed' would hide it indefinitely."""
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    (sample / "SAMPLE - FID1A.json").write_text("{ this is not json")
    with pytest.raises(RecordParseError):
        read_record(str(sample), "FID1A")


def test_unrelated_malformed_json_is_skipped_beside_a_valid_record(tmp_path):
    """The inverse of the case above: a corrupt file that is NOT named as
    this detector's canonical record must be skipped silently, not raise,
    even though a real record for the same detector sits right beside it."""
    doc = _fixture()
    detector = doc["detector"]
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir()
    canonical = sample / f"SAMPLE - {detector}.json"
    canonical.write_text(json.dumps(doc))
    (sample / "notes.json").write_text("{ this is not json")
    assert find_record(str(sample), detector) == str(canonical)


def test_reads_the_sample_level_fields(tmp_path):
    doc = _fixture()
    sample = _make_d(tmp_path, doc)
    record = read_record(str(sample), doc["detector"])
    assert isinstance(record, ResultRecord)
    assert record.detector == doc["detector"]
    assert record.sample_id == doc.get("sample_id")
    assert record.acquisition_method == doc.get("method")
    assert record.record_path.endswith(".json")


def test_signal_path_follows_source_d_when_present(tmp_path):
    doc = dict(_fixture())
    elsewhere = tmp_path / "elsewhere" / "REAL.D"
    elsewhere.mkdir(parents=True)
    doc["source_d"] = str(elsewhere)
    sample = _make_d(tmp_path, doc)
    assert read_record(str(sample), doc["detector"]).signal_path == str(elsewhere)


def test_signal_path_falls_back_to_the_sample_when_source_d_is_absent(tmp_path):
    doc = {k: v for k, v in _fixture().items() if k != "source_d"}
    sample = _make_d(tmp_path, doc)
    assert read_record(str(sample), doc["detector"]).signal_path == str(sample)


def test_signal_factor_comes_from_processing_parameters(tmp_path):
    doc = dict(_fixture())
    doc["processing_parameters"] = {"scaling": {"signal_factor": 7680.0,
                                                "area_factor": 0.0784}}
    sample = _make_d(tmp_path, doc)
    assert read_record(str(sample), doc["detector"]).signal_factor == 7680.0


def test_signal_factor_is_none_when_unrecorded(tmp_path):
    doc = {k: v for k, v in _fixture().items() if k != "processing_parameters"}
    sample = _make_d(tmp_path, doc)
    assert read_record(str(sample), doc["detector"]).signal_factor is None
