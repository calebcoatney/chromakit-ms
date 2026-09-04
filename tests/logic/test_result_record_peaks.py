# tests/logic/test_result_record_peaks.py
"""Nine on-disk schemas, one output shape — and nothing invented on the way."""
import json
import math
from pathlib import Path

from logic.result_record import RecordPeak, normalize_peak, read_record

RECORDS = Path(__file__).parent.parent / "data" / "records"


def _manifest():
    return json.loads((RECORDS / "MANIFEST.json").read_text())


def _make_d(tmp_path, doc):
    sample = tmp_path / "SAMPLE.D"
    sample.mkdir(parents=True, exist_ok=True)
    (sample / f"SAMPLE - {doc['detector']}.json").write_text(json.dumps(doc))
    return sample


def test_old_schema_compound_id_with_a_space_still_names_the_compound():
    """REGRESSION: a forwarder reading only `compound_id` returns a valid 200
    with every peak from 33 older records silently unnamed."""
    peak = normalize_peak({"Compound ID": "Methane (CH4)", "retention_time": 8.24})
    assert peak.compound == "Methane (CH4)"


def test_modern_compound_id_names_the_compound():
    assert normalize_peak({"compound_id": "Hydrogen"}).compound == "Hydrogen"


def test_unknown_labels_are_kept_not_dropped():
    """An unnamed peak is exactly what a chromatogram reviewer is looking for."""
    assert normalize_peak({"compound_id": "Unknown (8.241)"}).compound == "Unknown (8.241)"


def test_bounds_and_area_map_across():
    peak = normalize_peak({"retention_time": 8.24, "start_time": 8.18,
                           "end_time": 8.35, "area": 2596704.4})
    assert peak.rt_apex == 8.24
    assert peak.rt_start == 8.18
    assert peak.rt_end == 8.35
    assert peak.area == 2596704.4


def test_missing_bounds_are_none_never_the_apex():
    peak = normalize_peak({"retention_time": 8.24})
    assert peak.rt_start is None
    assert peak.rt_end is None


def test_unusable_first_alias_does_not_fall_through_to_a_second():
    """A present-but-unusable `retention_time` must stop the search, not fall
    through to the also-present `rt_apex` alias -- the peak said which field
    it meant, and falling through would answer with another field's value."""
    peak = normalize_peak({"retention_time": "13.31", "rt_apex": 8.24})
    assert peak.rt_apex is None


def test_compound_id_takes_precedence_over_compound_id_with_a_space():
    peak = normalize_peak({"compound_id": "Hydrogen", "Compound ID": "Methane (CH4)"})
    assert peak.compound == "Hydrogen"


def test_rf_table_vocabulary_populates_raw_amount_and_mol_percent():
    """`mol_percent` and `composition_percent` are DIFFERENT values here on
    purpose: an implementation that read mol_percent from the wrong key would
    still pass if the two happened to match, as they did in the original
    fixture value for this test."""
    peak = normalize_peak({"compound_id": "Hydrogen", "raw_amount": 0.749,
                           "mol_percent": 88.955, "composition_percent": 91.2})
    assert peak.raw_amount == 0.749
    assert peak.mol_percent == 88.955
    assert peak.extras["composition_percent"] == 91.2


def test_polyarc_vocabulary_populates_mol_percent_and_keeps_its_own_fields():
    """Flattening a mass basis against a mol% basis produces a number that
    means nothing, so the extras are carried through untouched."""
    peak = normalize_peak({"compound_id": "2-Hexanone", "mol_percent": 4.2,
                           "mass_mg": 0.345, "mol_C_percent": 12.01,
                           "num_carbons": 6})
    assert peak.mol_percent == 4.2
    assert peak.raw_amount is None
    assert peak.extras["mass_mg"] == 0.345
    assert peak.extras["mol_C_percent"] == 12.01
    assert peak.extras["num_carbons"] == 6


def test_quality_fields_survive_in_extras():
    peak = normalize_peak({"compound_id": "x", "Qual": 0.51, "casno": "064110-71-0",
                           "is_saturated": True, "quality_issues": []})
    assert peak.extras["Qual"] == 0.51
    assert peak.extras["casno"] == "064110-71-0"
    assert peak.extras["is_saturated"] is True


def test_a_string_that_parses_as_a_float_becomes_none():
    """float('13.31') is indistinguishable from a measured retention time."""
    assert normalize_peak({"retention_time": "13.31"}).rt_apex is None


def test_non_finite_numbers_become_none():
    assert normalize_peak({"area": math.nan}).area is None
    assert normalize_peak({"area": math.inf}).area is None


def test_booleans_are_not_numbers():
    assert normalize_peak({"area": True}).area is None


def test_every_fixture_schema_yields_usable_peaks(tmp_path):
    """All nine schemas, end to end."""
    for entry in _manifest():
        doc = json.loads((RECORDS / entry["fixture"]).read_text())
        sample = _make_d(tmp_path / entry["fixture"].replace(".json", ""), doc)
        record = read_record(str(sample), doc["detector"])
        assert record is not None, entry["fixture"]
        assert record.peaks, entry["fixture"]
        assert all(isinstance(p, RecordPeak) for p in record.peaks)
        assert any(p.rt_apex is not None for p in record.peaks), entry["fixture"]


def test_no_fixture_loses_every_compound_name(tmp_path):
    """The silent-blanking failure, checked against the real corpus."""
    for entry in _manifest():
        doc = json.loads((RECORDS / entry["fixture"]).read_text())
        has_name = any(p.get("compound_id") or p.get("Compound ID") for p in doc["peaks"])
        if not has_name:
            continue
        sample = _make_d(tmp_path / ("n_" + entry["fixture"].replace(".json", "")), doc)
        record = read_record(str(sample), doc["detector"])
        assert any(p.compound for p in record.peaks), entry["fixture"]
