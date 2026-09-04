# tests/logic/test_render_fixtures.py
"""The fixture corpus must actually exercise the schema variance it exists for."""
import json
from pathlib import Path

RECORDS = Path(__file__).parent.parent / "data" / "records"
PARITY = Path(__file__).parent.parent / "data" / "envelope_parity.json"


def _manifest():
    return json.loads((RECORDS / "MANIFEST.json").read_text())


def test_fixtures_cover_both_compound_key_spellings():
    keys = {k for entry in _manifest() for k in entry["peak_keys"]}
    assert "compound_id" in keys
    assert "Compound ID" in keys


def test_fixtures_cover_both_quant_vocabularies():
    keys = {k for entry in _manifest() for k in entry["peak_keys"]}
    assert "raw_amount" in keys, "RF-table vocabulary missing"
    assert "mol_C_percent" in keys, "Polyarc vocabulary missing"


def test_fixtures_cover_the_source_d_variant():
    docs = [json.loads((RECORDS / e["fixture"]).read_text()) for e in _manifest()]
    assert any("source_d" in d for d in docs)


def test_fixtures_include_a_record_without_processing_parameters():
    docs = [json.loads((RECORDS / e["fixture"]).read_text()) for e in _manifest()]
    assert any("processing_parameters" not in d for d in docs)


def test_every_fixture_is_trimmed_and_parseable():
    for entry in _manifest():
        doc = json.loads((RECORDS / entry["fixture"]).read_text())
        assert 1 <= len(doc["peaks"]) <= 3
        assert doc.get("detector")


def test_parity_fixture_has_cases():
    doc = json.loads(PARITY.read_text())
    assert doc["source"] == "rapids_bridge.chromatogram._envelope"
    assert len(doc["cases"]) == 5
