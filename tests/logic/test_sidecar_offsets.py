"""Tests for logic/sidecar_offsets.py -- per-file MS time offset sidecar."""
import json
import time

import pytest

from logic.sidecar_offsets import load_offset, save_offset


def test_load_returns_none_when_sidecar_missing(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    assert load_offset("/some/file.D", sidecar_path=sidecar) is None


def test_load_returns_none_when_key_missing(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    sidecar.write_text(json.dumps({"/other/file.D": {
        "offset_min": -0.05, "timestamp": 0.0, "source": "manual"
    }}))
    assert load_offset("/some/file.D", sidecar_path=sidecar) is None


def test_save_then_load_round_trip(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/sample.D", -0.048, source="manual", sidecar_path=sidecar)
    entry = load_offset("/abs/sample.D", sidecar_path=sidecar)
    assert entry is not None
    assert entry.offset_min == pytest.approx(-0.048)
    assert entry.source == "manual"
    assert entry.timestamp > 0.0


def test_save_overwrites_existing_entry(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/sample.D", -0.048, source="manual", sidecar_path=sidecar)
    time.sleep(0.01)
    save_offset("/abs/sample.D", 0.012, source="auto", sidecar_path=sidecar)
    entry = load_offset("/abs/sample.D", sidecar_path=sidecar)
    assert entry.offset_min == pytest.approx(0.012)
    assert entry.source == "auto"


def test_save_preserves_other_entries(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/a.D", 0.01, source="manual", sidecar_path=sidecar)
    save_offset("/abs/b.D", 0.02, source="auto", sidecar_path=sidecar)
    a = load_offset("/abs/a.D", sidecar_path=sidecar)
    b = load_offset("/abs/b.D", sidecar_path=sidecar)
    assert a.offset_min == pytest.approx(0.01)
    assert b.offset_min == pytest.approx(0.02)


def test_load_tolerates_malformed_json(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    sidecar.write_text("{not valid json")
    # Must not raise; return None so the app keeps working.
    assert load_offset("/abs/a.D", sidecar_path=sidecar) is None


def test_save_rejects_invalid_source(tmp_path):
    sidecar = tmp_path / "ms_time_offsets.json"
    with pytest.raises(ValueError):
        save_offset("/abs/a.D", 0.01, source="bogus", sidecar_path=sidecar)
