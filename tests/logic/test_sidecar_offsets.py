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


def test_save_offsets_batch_writes_all_entries(tmp_path):
    """save_offsets_batch persists the same offset for every path."""
    from logic.sidecar_offsets import save_offsets_batch, load_offset
    sidecar = tmp_path / "ms_time_offsets.json"
    paths = ["/abs/a.D", "/abs/b.D", "/abs/c.D"]
    save_offsets_batch(paths, offset_min=-0.048, source="manual", sidecar_path=sidecar)
    for p in paths:
        entry = load_offset(p, sidecar_path=sidecar)
        assert entry is not None
        assert entry.offset_min == -0.048
        assert entry.source == "manual"


def test_save_offsets_batch_uses_single_timestamp(tmp_path):
    """All entries in a batch share one timestamp (witnesses single write)."""
    from logic.sidecar_offsets import save_offsets_batch, load_offset
    sidecar = tmp_path / "ms_time_offsets.json"
    paths = ["/abs/a.D", "/abs/b.D", "/abs/c.D"]
    save_offsets_batch(paths, offset_min=0.01, source="auto", sidecar_path=sidecar)
    timestamps = {load_offset(p, sidecar_path=sidecar).timestamp for p in paths}
    assert len(timestamps) == 1, f"Expected one shared timestamp, got {timestamps}"


def test_save_offsets_batch_preserves_unrelated_entries(tmp_path):
    """Batch save doesn't delete entries for paths not in the batch."""
    from logic.sidecar_offsets import save_offset, save_offsets_batch, load_offset
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/keep.D", 0.005, source="manual", sidecar_path=sidecar)
    save_offsets_batch(["/abs/new1.D", "/abs/new2.D"], offset_min=0.02,
                       source="manual", sidecar_path=sidecar)
    # Unrelated entry survived
    keep = load_offset("/abs/keep.D", sidecar_path=sidecar)
    assert keep is not None
    assert keep.offset_min == 0.005
    # New entries present
    assert load_offset("/abs/new1.D", sidecar_path=sidecar) is not None
    assert load_offset("/abs/new2.D", sidecar_path=sidecar) is not None


def test_save_offsets_batch_overwrites_existing_entries(tmp_path):
    """When a path already has an entry, batch save replaces it."""
    from logic.sidecar_offsets import save_offset, save_offsets_batch, load_offset
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/a.D", 0.005, source="auto", sidecar_path=sidecar)
    save_offsets_batch(["/abs/a.D", "/abs/b.D"], offset_min=-0.030,
                       source="manual", sidecar_path=sidecar)
    a = load_offset("/abs/a.D", sidecar_path=sidecar)
    assert a is not None
    assert a.offset_min == -0.030
    assert a.source == "manual"


def test_save_offsets_batch_empty_list_is_noop(tmp_path):
    """Empty data_paths list does not create or modify the sidecar."""
    from logic.sidecar_offsets import save_offsets_batch
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offsets_batch([], offset_min=0.01, source="manual", sidecar_path=sidecar)
    assert not sidecar.exists()


def test_save_offsets_batch_rejects_invalid_source(tmp_path):
    """source not in VALID_SOURCES raises ValueError before any write."""
    from logic.sidecar_offsets import save_offsets_batch
    sidecar = tmp_path / "ms_time_offsets.json"
    with pytest.raises(ValueError):
        save_offsets_batch(["/abs/a.D"], offset_min=0.01, source="bogus",
                           sidecar_path=sidecar)
    assert not sidecar.exists()
