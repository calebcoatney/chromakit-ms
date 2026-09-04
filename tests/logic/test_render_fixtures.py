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


def _walk_strings(obj):
    """Yield every string value found anywhere inside a nested dict/list."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, str):
        yield obj


def _looks_like_absolute_path(value: str) -> bool:
    """True if `value` looks like it leaked an absolute filesystem path."""
    if "OneDrive" in value or "/Users/" in value:
        return True
    # A string that starts with "/" and has further path segments looks like
    # an absolute filesystem path rather than ordinary text (a bare leading
    # slash isn't, by itself, enough to flag — e.g. "1/2").
    return value.startswith("/") and value.count("/") >= 2


def test_no_absolute_paths_leak_into_fixtures():
    """Regression guard for the scrub_paths fix: no fixture or the manifest
    may carry an absolute filesystem path, in any field, ever. This is what
    actually enforces the "basename only" provenance rule — the generator's
    scrub_paths() is only as good as this test verifying its output.
    """
    offenders = []
    manifest_raw = json.loads((RECORDS / "MANIFEST.json").read_text())
    for s in _walk_strings(manifest_raw):
        if _looks_like_absolute_path(s):
            offenders.append(("MANIFEST.json", s))
    for entry in _manifest():
        doc = json.loads((RECORDS / entry["fixture"]).read_text())
        for s in _walk_strings(doc):
            if _looks_like_absolute_path(s):
                offenders.append((entry["fixture"], s))
    assert not offenders, f"absolute path(s) leaked into fixtures: {offenders}"


def test_scrub_paths_does_not_truncate_unrelated_strings():
    """scrub_paths() must basename only the known path-bearing fields, and
    must leave every other string untouched — including a longer string that
    happens to contain a path as a substring (e.g. a provenance note).
    """
    from tools.make_render_fixtures import scrub_paths, PATH_BEARING_FIELDS

    assert "source_d" in PATH_BEARING_FIELDS
    leaky_path = "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS/x.D"
    note = f"Reprocessed after recalibration; see prior run at {leaky_path} for context."
    doc = {"comment": note, "source_d": leaky_path}

    scrubbed = scrub_paths(doc)

    assert scrubbed["comment"] == note, "non-path-bearing field must not be altered"
    assert scrubbed["source_d"] == "x.D"
