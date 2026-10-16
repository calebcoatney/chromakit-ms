import pytest

from logic.gcxgc_processor import GCxGCProcessor
from logic.gcxgc_run import GCxGCRun
from util import sepsolve_synth as synth


def _scene():
    return synth.Scene(peaks=[
        synth.Peak(t1_min=2.0, t2_s=1.0, height=10.0, base_mz=57),
        synth.Peak(t1_min=2.0, t2_s=2.5, height=1.0, base_mz=91),
        synth.Peak(t1_min=4.0, t2_s=1.6, height=3.0, base_mz=105),
    ], duration_min=6.0, offset_a=-4.0)


def _peaks(tmp_path, signal="fid", baseline=False):
    run = GCxGCRun.from_c_folder(synth.build_run(tmp_path, _scene()))
    run.set_offset("fixed", -4.0)
    proc = GCxGCProcessor(run.frame(signal), run, use_dbc=True, apex_window=10)
    if baseline:
        proc.apply_baseline(lam=1e5)
    return run, proc, proc.detect_peaks(min_prominence=50.0, rt2_grouping_tolerance=2, min_sub_peaks=2)


def _match(found, t1, t2):
    return min(found, key=lambda p: (p.rt1 - t1) ** 2 + ((p.rt2 - t2) / 60) ** 2)


def test_detects_planted_peaks_at_their_coordinates(tmp_path):
    _, _, found = _peaks(tmp_path)
    assert len(found) == 3
    for t1, t2 in ((2.0, 1.0), (2.0, 2.5), (4.0, 1.6)):
        p = _match(found, t1, t2)
        assert abs(p.rt1 - t1) < 4.0 / 60 and p.rt2 == pytest.approx(t2, abs=0.04)
        assert p.detection_signal == "fid" and p.volume > 0 and p.apex_raw_index is not None


def test_small_peak_gets_its_own_spectrum(tmp_path):
    _, proc, found = _peaks(tmp_path)
    small = _match(found, 2.0, 2.5)
    spec = dict(proc.extract_spectrum(small))
    assert max(spec, key=spec.get) == 91
    assert small.spectrum_source == "dbc" and small.apex_found


def test_detection_on_dbc_tic(tmp_path):
    _, proc, found = _peaks(tmp_path, signal="dbc_tic")
    assert len(found) == 3 and {p.detection_signal for p in found} == {"dbc_tic"}
    spec = dict(proc.extract_spectrum(_match(found, 4.0, 1.6)))
    assert max(spec, key=spec.get) == 105


def test_baseline_then_detect(tmp_path):
    _, _, found = _peaks(tmp_path, baseline=True)
    assert len(found) == 3


def test_flat_signal_has_no_peaks(tmp_path):
    run = GCxGCRun.from_c_folder(synth.build_run(tmp_path, synth.Scene(duration_min=2.0)))
    assert GCxGCProcessor(run.frame("fid"), run).detect_peaks(min_prominence=50.0) == []


def test_peak_as_dict_has_provenance_keys(tmp_path):
    _, proc, found = _peaks(tmp_path)
    proc.extract_spectrum(found[0])
    d = found[0].as_dict()
    for key in ("rt1", "rt2", "volume", "detection_signal", "apex_raw_index", "spectrum_scan",
                "spectrum_source", "apex_found", "apex_distance_scans", "saturated", "top_hits", "casno"):
        assert key in d
