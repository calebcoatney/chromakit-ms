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


# --- detector mismatch --------------------------------------------------
#
# DataHandler._get_chromatogram_data silently falls back through a fixed
# alternates list (['FID1A', 'TCD2B', 'FID2B', 'TCD1A', 'TCD1B']) when the
# requested .ch is absent, updating self.current_detector rather than
# raising. Simulated here with a fake load_data_directory so this is a
# deterministic unit test independent of the real sample used below.

def test_detector_mismatch_degrades_without_drawing_another_channel(tmp_path, monkeypatch):
    import logic.data_handler as data_handler_module

    def fake_load(self, file_path, detector=None):
        # Mirrors the real fallback: resolves to a DIFFERENT channel than
        # what was requested, silently, exactly like the real loader does.
        self.current_detector = "FID1A"
        return {
            "chromatogram": {"x": [0.0, 0.1, 0.2], "y": [1.0, 2.0, 3.0]},
            "tic": {"x": [], "y": []},
            "metadata": {},
        }

    monkeypatch.setattr(data_handler_module.DataHandler, "load_data_directory", fake_load)

    fake = tmp_path / "SAMPLE.D"
    fake.mkdir()

    signal = read_signal(str(fake), detector="TCD1A")
    assert signal is None, "a resolved detector that != the requested one must degrade to None"

    trace, reason = read_trace(str(fake), detector="TCD1A")
    assert trace is None
    assert reason and "TCD1A" in reason and "not present" in reason


def test_detector_names_matching_case_insensitively_with_ch_suffix_are_not_a_mismatch(
        tmp_path, monkeypatch):
    """FID1A, fid1a and FID1A.ch all name the same channel elsewhere in this
    codebase; the mismatch check must not fire a false positive on any of
    those spelling differences."""
    import logic.data_handler as data_handler_module

    def fake_load(self, file_path, detector=None):
        self.current_detector = "FID1A"          # loader's own canonical spelling
        return {
            "chromatogram": {"x": [0.0, 0.1, 0.2], "y": [1.0, 2.0, 3.0]},
            "tic": {"x": [], "y": []},
            "metadata": {},
        }

    monkeypatch.setattr(data_handler_module.DataHandler, "load_data_directory", fake_load)

    fake = tmp_path / "SAMPLE.D"
    fake.mkdir()

    for requested in ("fid1a", "FID1A.ch", "  FID1A  "):
        signal = read_signal(str(fake), detector=requested)
        assert signal is not None, f"{requested!r} should match resolved 'FID1A'"


# --- shared path normalizer (FIX 2) -----------------------------------------
#
# _is_c_folder used to strip only os.sep while is_supported_sample_path
# stripped both "/" and "\\", so a trailing backslash on a .C path was
# recognised by one and not the other -- wrongly dispatched to the .D branch
# on a POSIX dev machine too (os.sep == "/" there), not only on the Windows
# GC box. Deliberately independent of the real machine's OS: it pins that
# both functions now go through the one shared normalizer, so they cannot
# drift again the way they already have once.

def test_c_folder_dispatch_and_is_supported_agree_on_a_trailing_backslash():
    from logic.render_trace import _is_c_folder, is_supported_sample_path

    path = "some" + os.sep + "SAMPLE.C" + "\\"
    assert _is_c_folder(path) is True
    assert is_supported_sample_path(path) is True


def test_c_folder_dispatch_and_is_supported_agree_on_a_trailing_forward_slash():
    from logic.render_trace import _is_c_folder, is_supported_sample_path

    path = "some" + os.sep + "SAMPLE.C" + "/"
    assert _is_c_folder(path) is True
    assert is_supported_sample_path(path) is True


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


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="real render sample not present on this machine")
def test_real_sample_requesting_a_detector_it_lacks_does_not_draw_another_channel():
    """This sample only has FID1A and TCD3C. Before the fix, DataHandler's
    silent fallback through ['FID1A', 'TCD2B', 'FID2B', 'TCD1A', 'TCD1B']
    meant a TCD1A request here quietly returned FID1A's own trace with
    error=None -- a confidently wrong picture, not a missing one."""
    detectors = available_detectors(str(SAMPLE))
    assert "TCD1A" not in detectors

    signal = read_signal(str(SAMPLE), detector="TCD1A")
    assert signal is None, "must not silently hand back another channel's signal"

    trace, error = read_trace(str(SAMPLE), detector="TCD1A")
    assert trace is None
    assert error is not None
    assert "TCD1A" in error
    assert "not present" in error
