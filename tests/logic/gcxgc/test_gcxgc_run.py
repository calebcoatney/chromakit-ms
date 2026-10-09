import numpy as np
import pytest

from logic.gcxgc_run import GCxGCRun, MissingModulationPeriod, OffsetModel
from tests.logic.gcxgc import synth


def _run(tmp_path, scene, **kw):
    return GCxGCRun.from_c_folder(synth.build_run(tmp_path, scene, **kw))


def _apex(frame, peak, half_rows=2, half_cols=6):
    """(row, col) of the frame maximum near a planted peak's coordinates."""
    m0 = int(np.argmin(abs(frame.t1_min - peak.t1_min)))
    j0 = int(round(peak.t2_s * frame.hz))
    r0, c0 = max(0, m0 - half_rows), max(0, j0 - half_cols)
    win = frame.data[r0:m0 + half_rows + 1, c0:j0 + half_cols + 1]
    r, c = np.unravel_index(int(np.argmax(win)), win.shape)
    return r0 + int(r), c0 + int(c)


def test_layout_comes_from_hdr_and_chromspace_view(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    assert (run.modulation_period_s, run.phase_shift_s) == (4.0, -1.8)
    assert run.provenance == {"modulation_period_s": "hdr", "phase_shift_s": "lsc_view"}
    assert (run.fid_hz, run.tof_hz) == (50.0, 50.0)
    assert run.sample_id == "Synth"


def test_period_falls_back_to_chromspace_view(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0), insight=False)
    assert run.modulation_period_s == 4.0
    assert run.provenance["modulation_period_s"] == "lsc_view"


def test_missing_period_raises(tmp_path):
    c = synth.build_run(tmp_path, synth.Scene(duration_min=2.0), insight=False, view=False)
    with pytest.raises(MissingModulationPeriod):
        GCxGCRun.from_c_folder(c)
    run = GCxGCRun.from_c_folder(c, modulation_period_s=4.0)
    assert run.provenance == {"modulation_period_s": "user", "phase_shift_s": "default"}
    assert run.phase_shift_s == 0.0


def test_user_layout_overrides_and_reverts(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    run.set_layout(modulation_period_s=2.0, phase_shift_s=0.5)
    assert (run.modulation_period_s, run.phase_shift_s) == (2.0, 0.5)
    assert run.provenance == {"modulation_period_s": "user", "phase_shift_s": "user"}
    run.set_layout(modulation_period_s=None, phase_shift_s=None)
    assert (run.modulation_period_s, run.phase_shift_s) == (4.0, -1.8)
    assert run.provenance == {"modulation_period_s": "hdr", "phase_shift_s": "lsc_view"}
    assert run.file_layout["modulation_period_s"] == (4.0, "hdr")


def test_period_must_be_whole_samples(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    with pytest.raises(ValueError, match="whole number of samples"):
        run.set_layout(modulation_period_s=4.01)
    assert run.modulation_period_s == 4.0


def test_equivalent_phases_cut_the_same_place(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    run.set_layout(phase_shift_s=-1.8)
    a = run.frame("fid")
    run.set_layout(phase_shift_s=2.2)
    b = run.frame("fid")
    assert a.cut == b.cut == 90
    np.testing.assert_array_equal(a.data, b.data)


def test_frame_is_shift_then_reshape(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    f = run.frame("fid")
    assert (f.N, f.cut, f.signal) == (200, 90, "fid")
    np.testing.assert_array_equal(f.data.ravel(), run.fid[90:90 + f.data.size])
    np.testing.assert_allclose(f.t1_min[:2], [90 / 50 / 60, 290 / 50 / 60])
    np.testing.assert_allclose(f.t2_s[[0, 1, -1]], [0.0, 0.02, 3.98])


def test_peaks_land_at_planted_coordinates_on_both_sides_of_the_cut(tmp_path):
    early = synth.Peak(t1_min=3.0, t2_s=1.0, height=1.0, base_mz=57)
    late = synth.Peak(t1_min=3.0, t2_s=3.0, height=1.0, base_mz=91)   # crosses a raw row boundary
    run = _run(tmp_path, synth.Scene(peaks=[early, late], duration_min=6.0))
    f = run.frame("fid")
    (r1, c1), (r2, c2) = _apex(f, early), _apex(f, late)
    assert r1 == r2                       # the old in-row np.roll put the late peak one row off
    assert abs(f.t1_min[r1] - 3.0) <= 4.0 / 60 / 2 + 1e-9
    assert (c1, c2) == (50, 150)


def test_raw_index_maps_roundtrip(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    for i in (90, 91, 289, 290, 5000):
        m, j = run.from_raw("fid", i)
        assert run.to_raw("fid", m, j) == i
    assert run.from_raw("fid", 90) == (0, 0)
    assert run.from_raw("fid", 10) == (-1, 120)
    assert run.raw_to_time("fid", 100) == pytest.approx(2.0)


def _offset_scene(**kw):
    peaks = [synth.Peak(t1_min=float(t), t2_s=0.6 + (k % 5) * 0.6, height=1.0 + (k % 3),
                        base_mz=57 + 14 * (k % 4))
             for k, t in enumerate(np.arange(0.5, 9.6, 0.35))]
    return synth.Scene(peaks=peaks, duration_min=10.0, **kw)


def test_offset_estimate_recovers_planted_drift(tmp_path):
    run = _run(tmp_path, _offset_scene(offset_a=-5.0, offset_b=0.3))
    model = run.estimate_offset(window_min=1.0, max_lag=20, min_r=0.8, min_windows=5)
    assert model.ok and model.n_windows >= 5
    for t in (1.0, 5.0, 9.0):
        assert model.at(t) == pytest.approx(-5.0 + 0.3 * t, abs=0.5)


def test_offset_estimate_gives_up_without_enough_windows(tmp_path):
    run = _run(tmp_path, synth.Scene(peaks=[synth.Peak(1.0, 1.0, 1.0)], duration_min=2.0))
    model = run.estimate_offset(window_min=1.0, min_windows=5)
    assert not model.ok and (model.a, model.b) == (0.0, 0.0)


def test_fixed_offset(tmp_path):
    run = _run(tmp_path, synth.Scene(duration_min=2.0))
    run.set_offset("fixed", -3.0)
    assert run.offset() == OffsetModel(a=-3.0, b=0.0, ok=True, n_windows=0, mode="fixed")
    assert run.fid_to_tof(1000) == pytest.approx(997.0)


def test_fid_to_tof_with_unequal_rates(tmp_path):
    peak = synth.Peak(t1_min=2.0, t2_s=1.5, height=1.0)
    run = _run(tmp_path, synth.Scene(peaks=[peak], duration_min=4.0, tof_hz=100.0, offset_a=-4.0))
    assert run.tof_hz == 100.0
    run.set_offset("fixed", -4.0)
    i_fid, i_tof = int(np.argmax(run.fid)), int(np.argmax(run.tic))
    assert run.fid_to_tof(i_fid) == pytest.approx(i_tof, abs=1.5)


def test_spectrum_at_takes_the_peaks_own_apex_not_a_bigger_neighbour(tmp_path):
    big = synth.Peak(t1_min=3.0, t2_s=1.0, height=10.0, base_mz=57)
    small = synth.Peak(t1_min=3.0, t2_s=2.5, height=1.0, base_mz=91)
    run = _run(tmp_path, synth.Scene(peaks=[big, small], duration_min=6.0, offset_a=-4.0))
    run.set_offset("fixed", -4.0)
    f = run.frame("fid")
    r, c = _apex(f, small)
    res = run.spectrum_at(run.to_raw("fid", r, c), "fid", source="dbc", window=10)
    assert res.mz[int(np.argmax(res.intensity))] == 91
    assert res.apex_found and abs(res.apex_distance_scans) <= 1 and res.source == "dbc"


def test_spectrum_at_run_edges_does_not_crash(tmp_path):
    run = _run(tmp_path, synth.Scene(peaks=[synth.Peak(1.0, 1.0, 1.0)], duration_min=2.0))
    run.set_offset("fixed", 0.0)
    last = len(run.fid) - 1
    for i in (0, last):
        res = run.spectrum_at(i, "fid", source="lsc", window=10)
        assert 0 <= res.scan < run.lsc.n_scans
    assert run.spectrum_at(last + 5_000, "fid", source="lsc").scan == run.lsc.n_scans - 1


def test_spectrum_at_flags_a_peak_the_tof_never_saw(tmp_path):
    fid_only = synth.Peak(t1_min=2.0, t2_s=1.0, height=1.0, in_tof=False)
    run = _run(tmp_path, synth.Scene(peaks=[fid_only], duration_min=4.0))
    run.set_offset("fixed", 0.0)
    r, c = _apex(run.frame("fid"), fid_only)
    i = run.to_raw("fid", r, c)
    res = run.spectrum_at(i, "fid", source="lsc", window=10)
    assert not res.apex_found and res.scan == int(round(run.fid_to_tof(i)))


def _blind_peak_result(tmp_path, seed, noise=5.0):
    """spectrum_at for a FID-only peak on a raw TIC with Gaussian noise; also the FID-mapped scan."""
    fid_only = synth.Peak(t1_min=2.0, t2_s=1.0, height=1.0, in_tof=False)
    scene = synth.Scene(peaks=[fid_only], duration_min=4.0, noise=noise, seed=seed)
    run = _run(tmp_path / f"s{seed}", scene)
    run.set_offset("fixed", 0.0)
    r, c = _apex(run.frame("fid"), fid_only)
    i = run.to_raw("fid", r, c)
    return run.spectrum_at(i, "fid", source="lsc", window=10), int(round(run.fid_to_tof(i)))


def test_tof_blind_peak_on_a_noisy_tic_is_not_an_apex(tmp_path):
    # Noise only (sigma 5 on a 50 background). The window maximum is interior ~90 % of the time,
    # so only the prominence test can reject it. A 3-sigma rule passes pure noise a few percent of
    # the time per trial, so across 20 fixed seeds we allow at most one such rare false positive.
    results = [_blind_peak_result(tmp_path, seed) for seed in range(20)]
    found = [res.apex_found for res, _ in results]
    assert sum(found) <= 1, f"apex_found True for seeds {[s for s, f in enumerate(found) if f]}"
    for res, mapped in results:
        if not res.apex_found:
            assert res.scan == mapped and res.apex_distance_scans == 0


def test_real_peak_on_a_noisy_tic_is_still_an_apex(tmp_path):
    peak = synth.Peak(t1_min=2.0, t2_s=1.0, height=1.0, base_mz=91)
    for seed in range(5):
        scene = synth.Scene(peaks=[peak], duration_min=4.0, noise=5.0, seed=seed)
        run = _run(tmp_path / f"s{seed}", scene)
        run.set_offset("fixed", 0.0)
        r, c = _apex(run.frame("fid"), peak)
        res = run.spectrum_at(run.to_raw("fid", r, c), "fid", source="lsc", window=10)
        assert res.apex_found and abs(res.apex_distance_scans) <= 10
        assert res.mz[int(np.argmax(res.intensity))] == 91


def test_saturated_apex_is_replaced_by_its_neighbours(tmp_path):
    scene = synth.Scene(peaks=[synth.Peak(t1_min=2.0, t2_s=1.0, height=1.0)], duration_min=4.0)
    apex = int(np.argmax([s[1].sum() for s in scene.spectra(background=False)]))
    run = _run(tmp_path, scene, saturate={apex})
    res = run.spectrum_at(apex, "tic", source="dbc", window=0)
    assert res.saturated and res.scan == apex
    before, after = run.dbc_lsc.spectrum(apex - 1), run.dbc_lsc.spectrum(apex + 1)
    np.testing.assert_allclose(res.intensity, (before[1] + after[1]) / 2)


def test_dbc_only_folder_loads(tmp_path):
    peak = synth.Peak(t1_min=2.0, t2_s=1.0, height=1.0, base_mz=91)
    run = _run(tmp_path, synth.Scene(peaks=[peak], duration_min=4.0), raw=False)
    assert run.lsc is None and run.available_signals == ("fid", "dbc_tic")
    with pytest.raises(ValueError):
        run.frame("tic")
    run.set_offset("fixed", 0.0)
    res = run.spectrum_at(int(np.argmax(run.fid)), "fid", source="lsc")   # asked for raw; only DBC exists
    assert res.source == "dbc" and res.mz[int(np.argmax(res.intensity))] == 91


def test_empty_dbc_scan_falls_back_to_raw(tmp_path):
    run = _run(tmp_path, synth.Scene(peaks=[synth.Peak(1.0, 1.0, 1.0)], duration_min=2.0))
    res = run.spectrum_at(0, "tic", source="dbc", window=0)    # scan 0: no peak, DBC empty
    assert res.source == "lsc" and res.mz.tolist() == [40]


def _quantised_jittered_times(n, rate_hz=50.0, t0=1_700_000_000.0, jitter_s=3e-4, seed=1):
    """Scan times like the real .lsc's: +-0.3 ms jitter, then 32.32 fixed point with the low 15 bits
    zero (7.63 us steps), so the median step lands one quantum off 1 / rate_hz."""
    t = t0 + np.arange(n) / rate_hz + np.random.default_rng(seed).uniform(-jitter_s, jitter_s, n)
    return np.round(t * 2 ** 17) / 2 ** 17


def test_tof_rate_from_timestamps_is_the_mean_step(tmp_path, caplog):
    import logging
    from logic.loaders.sepsolve_format import LscFile
    scene = synth.Scene(duration_min=4.0)
    c = synth.build_run(tmp_path, scene)
    data = c / "data"
    synth.write_hdr(data / "Synth.HDR", tof_scanset_us=None)          # no "scanset period"
    ts = _quantised_jittered_times(scene.n_tof())
    view = {"pm": scene.pm_s, "phase": scene.phase_s}
    synth.write_lsc(data / "Synth.lsc", scene.spectra(background=True), view=view, timestamps=ts)
    synth.write_lsc(data / "Synth_0.5_secs.dbc.lsc", scene.spectra(background=False), view=view,
                    timestamps=ts)
    with LscFile(data / "Synth.lsc") as lsc:                            # the trap the median fell into
        assert round(1.0 / float(np.median(np.diff(lsc.timestamps))), 3) != 50.0
    with caplog.at_level(logging.INFO, logger="logic.gcxgc_run"):
        run = GCxGCRun.from_c_folder(c)                                # a 4 s period loads
    assert run.tof_hz == 50.0
    assert run.modulation_period_s == 4.0 and run.frame("tic").N == 200
    assert "timestamps" in caplog.text

@pytest.mark.realdata
def test_real_8260_run_layout_and_offset(real_gcxgc_run):
    run = GCxGCRun.from_c_folder(real_gcxgc_run)
    assert (run.modulation_period_s, run.phase_shift_s) == (4.0, -1.8)
    assert run.provenance == {"modulation_period_s": "hdr", "phase_shift_s": "lsc_view"}
    assert run.fid_hz == 50.0 and run.tof_hz == pytest.approx(50.0)
    assert len(run.fid) == 486_300 and len(run.tic) == len(run.dbc_tic) == 486_308
    model = run.estimate_offset()
    assert model.ok
    assert -7.5 <= model.at(5) <= -5.0 and -3.0 <= model.at(75) <= -0.5
