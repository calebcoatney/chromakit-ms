"""Reading the signal. Bad files degrade to (None, reason); they never raise
and never return a plausible wrong trace."""
import os
from pathlib import Path

import pytest

from logic.render_trace import available_detectors, read_signal, read_trace


def test_unreadable_path_degrades_with_a_reason(tmp_path):
    fake = tmp_path / "nothing.D"
    fake.mkdir()
    trace, reason = read_trace(str(fake), detector="FID1A")
    assert trace is None
    assert reason and "FID1A" in reason


def test_available_detectors_on_a_missing_path_is_empty(tmp_path):
    assert available_detectors(str(tmp_path / "absent.D")) == []


def test_read_signal_on_a_missing_path_is_none(tmp_path):
    assert read_signal(str(tmp_path / "absent.D"), detector="FID1A") is None


def test_bad_max_points_still_raises_through_read_trace(tmp_path):
    fake = tmp_path / "nothing.D"
    fake.mkdir()
    with pytest.raises(ValueError):
        read_trace(str(fake), detector="FID1A", max_points=1)


# --- .C branch dispatch -----------------------------------------------------
#
# A negative test is enough to prove the .C branch is reached at all and
# degrades like the .D branch rather than raising: CFolder.open() raises
# FileNotFoundError when manifest.json is absent, which read_trace/
# available_detectors must catch and turn into (None, reason) / [].

def test_c_folder_without_a_manifest_degrades_with_a_reason(tmp_path):
    fake = tmp_path / "SAMPLE.C"
    fake.mkdir()
    trace, reason = read_trace(str(fake), detector="FID1A")
    assert trace is None
    assert reason


def test_available_detectors_on_a_bare_c_folder_is_empty(tmp_path):
    fake = tmp_path / "SAMPLE.C"
    fake.mkdir()
    assert available_detectors(str(fake)) == []


# --- opt-in happy-path sweep against a real sample --------------------------
#
# Guarded by skipif so the suite still passes on a machine without the
# sample, following the same pattern as
# tests/logic/test_record_corpus_sweep.py's real-corpus sweep.
#
# Point it elsewhere with:  CHROMAKIT_RENDER_SAMPLE=/some/path pytest ...

DEFAULT_SAMPLE = (
    Path.home() / "Library/CloudStorage/OneDrive-NREL/GC-MS code development"
    / "Hyewon's data" / "260224 GC1 R1 Run 311 outlet 25.D"
)


def _sample_path() -> Path:
    override = os.environ.get("CHROMAKIT_RENDER_SAMPLE")
    return Path(override) if override else DEFAULT_SAMPLE


SAMPLE = _sample_path()


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="real render sample not present on this machine")
def test_real_sample_reads_a_decimated_trace():
    detectors = available_detectors(str(SAMPLE))
    assert "FID1A" in detectors
    assert "TCD3C" in detectors

    trace, error = read_trace(str(SAMPLE), detector="FID1A")
    assert error is None
    assert trace is not None
    assert trace.points <= 2000
    assert trace.decimation > 1
    assert all(b > a for a, b in zip(trace.rt_min, trace.rt_min[1:]))
    assert len(trace.rt_min) == len(trace.plot_y) == trace.points
