import logging

import numpy as np
import pytest

from logic.loaders.sepsolve_format import (
    LscFile, SepSolveFormatError, nominal_bin, parse_hdr_text, read_hdr, read_rsd,
)
from util import sepsolve_synth as synth


def test_hdr_reads_insight_period_not_calibrant_pulse(tmp_path):
    synth.write_hdr(tmp_path / "a.HDR", pm_s=4.0, fid_hz=50, tof_scanset_us=20000, method="M1")
    info = read_hdr(tmp_path / "a.HDR")
    assert info.modulation_period_s == 4.0        # not the 0.5 s calibrant pulse
    assert info.fid_rate_hz == 50.0                # not signal #2's decoy Data Rate=10
    assert info.tof_rate_hz == pytest.approx(50.0)
    assert info.method_name == "M1"


def test_hdr_without_modulator_has_no_period():
    info = parse_hdr_text(synth.hdr_text(insight=False))
    assert info.modulation_period_s is None
    assert info.fid_rate_hz == 50.0


def test_hdr_tof_rate_follows_scanset_period():
    assert parse_hdr_text(synth.hdr_text(tof_scanset_us=10000)).tof_rate_hz == pytest.approx(100.0)


def test_rsd_roundtrip(tmp_path):
    sig = np.random.default_rng(0).normal(4e4, 50, 1234)
    synth.write_rsd(tmp_path / "a.rsd", sig, rate_hz=50, block=97)
    out, info = read_rsd(tmp_path / "a.rsd")
    np.testing.assert_array_equal(out, sig)
    assert (info.rate_hz, info.n_points, info.n_blocks) == (50.0, 1234, 13)


def test_rsd_reads_values_the_old_signature_hack_missed(tmp_path):
    # The old loader found the first block by the bytes E1 40 (first value in [32768, 65536)).
    sig = np.array([1.0, 2.0, 3.0])
    synth.write_rsd(tmp_path / "a.rsd", sig)
    np.testing.assert_array_equal(read_rsd(tmp_path / "a.rsd")[0], sig)


def test_rsd_corrupt_block_raises(tmp_path):
    offsets = synth.write_rsd(tmp_path / "a.rsd", np.arange(10.0), block=4)
    raw = bytearray((tmp_path / "a.rsd").read_bytes())
    raw[offsets[1]] ^= 0xFF
    (tmp_path / "a.rsd").write_bytes(bytes(raw))
    with pytest.raises(SepSolveFormatError):
        read_rsd(tmp_path / "a.rsd")


def _three_scans():
    return [(np.array([40.0, 57.02]), np.array([10.0, 100.0])),
            (np.array([]), np.array([])),
            (np.array([43.99, 91.06, 106.08]), np.array([5.0, 50.0, 25.0]))]


def test_lsc_roundtrip(tmp_path):
    p = tmp_path / "a.lsc"
    synth.write_lsc(p, _three_scans(), rate_hz=50.0,
                    flags=[synth.FLAG_NORMAL, synth.FLAG_NORMAL, synth.FLAG_SATURATED],
                    view={"pm": 4.0, "phase": -1.8})
    with LscFile(p) as lsc:
        assert lsc.n_scans == 3
        np.testing.assert_allclose(np.diff(lsc.timestamps), 0.02, atol=1e-6)
        mz, inten = lsc.raw_spectrum(2)
        np.testing.assert_allclose(mz, [43.99, 91.06, 106.08])
        np.testing.assert_array_equal(inten, [5.0, 50.0, 25.0])
        np.testing.assert_allclose(lsc.tic(), [110.0, 0.0, 80.0])
        assert [lsc.is_saturated(k) for k in range(3)] == [False, False, True]
        assert (lsc.view.modulation_period_s, lsc.view.phase_shift_s, lsc.view.cut_interval_s) == (4.0, -1.8, 4.0)
        assert len(lsc.spectrum(1)[0]) == 0


def test_nominal_bins_match_chromspace_asymmetry():
    mz, inten = nominal_bin(np.array([56.49, 56.5, 57.49, 57.5]), np.array([1.0, 2.0, 4.0, 8.0]))
    assert mz.tolist() == [56, 57, 58]
    assert inten.tolist() == [1.0, 6.0, 8.0]


def test_lsc_without_directory_walks_blocks(tmp_path, caplog):
    p = tmp_path / "a.lsc"
    synth.write_lsc(p, _three_scans(), with_directory=False)
    with caplog.at_level(logging.WARNING):
        with LscFile(p) as lsc:
            assert lsc.n_scans == 3
            np.testing.assert_allclose(lsc.tic(), [110.0, 0.0, 80.0])
    assert "DI02" in caplog.text


def test_lsc_without_view_has_none(tmp_path):
    synth.write_lsc(tmp_path / "a.lsc", _three_scans(), view=None)
    with LscFile(tmp_path / "a.lsc") as lsc:
        assert lsc.view.modulation_period_s is None and lsc.view.phase_shift_s is None


def test_lsc_bad_magic_raises(tmp_path):
    p = tmp_path / "a.lsc"
    p.write_bytes(b"GARBAGE!" * 4)
    with pytest.raises(SepSolveFormatError):
        LscFile(p)


def test_lsc_scan_index_out_of_range(tmp_path):
    synth.write_lsc(tmp_path / "a.lsc", _three_scans())
    with LscFile(tmp_path / "a.lsc") as lsc, pytest.raises(IndexError):
        lsc.raw_spectrum(3)


def test_build_run_folder_is_readable(tmp_path):
    scene = synth.Scene(peaks=[synth.Peak(t1_min=2.0, t2_s=1.0, height=1.0)], duration_min=3.0)
    c = synth.build_run(tmp_path, scene, name="S1")
    d = c / "data"
    fid, _ = read_rsd(d / "S1.rsd")
    assert len(fid) == 3 * 60 * 50
    with LscFile(d / "S1.lsc") as raw, LscFile(d / "S1_0.5_secs.dbc.lsc") as dbc:
        assert raw.n_scans == dbc.n_scans == 9000
        np.testing.assert_array_equal(raw.timestamps, dbc.timestamps)
        assert raw.tic().min() == pytest.approx(50.0)     # background ion only in the raw file
        assert dbc.tic().min() == 0.0


@pytest.mark.realdata
def test_real_8260_readers(real_gcxgc_run):
    d = real_gcxgc_run / "data"
    name = real_gcxgc_run.name[:-2]          # "8260-110 SAF"
    info = read_hdr(d / f"{name}.HDR")
    assert (info.modulation_period_s, info.fid_rate_hz) == (4.0, 50.0)
    assert info.tof_rate_hz == pytest.approx(50.0)
    fid, rinfo = read_rsd(d / f"{name}.rsd")
    assert rinfo.n_points == 486_300 and fid[0] == pytest.approx(35253.884)
    assert float(fid.sum()) == pytest.approx(19028897141.064, rel=1e-12)
    with LscFile(d / f"{name}.lsc") as lsc, LscFile(d / f"{name}_0.5_secs.dbc.lsc") as dbc:
        assert lsc.n_scans == dbc.n_scans == 486_308
        tic = lsc.tic()
        assert tic[0] == pytest.approx(97.5) and tic[100_000] == pytest.approx(38783.05)
        assert float(tic.sum()) == pytest.approx(26179856692.3, rel=1e-12)
        assert float(dbc.tic().sum()) == pytest.approx(14895036443.0, rel=1e-12)
        assert (lsc.view.modulation_period_s, lsc.view.phase_shift_s) == (4.0, -1.8)
        # Real timestamps are quantized (low 15 bits of the 32.32 value are zero, 7.6 us steps)
        # and jitter by ~0.3 ms, so the median step lands one quantum off 20 ms (0.0199966 s).
        # The mean step is the 50 Hz rate: 0.01999999990 s.
        ts = lsc.timestamps
        assert (ts[-1] - ts[0]) / (len(ts) - 1) == pytest.approx(0.02, abs=1e-6)
        assert int(np.count_nonzero(lsc.flags & 0xF0000000)) == 122


# ── fix round 1: strict structure checks ────────────────────────────────────

def _sd04_offsets(path):
    with LscFile(path) as lsc:
        return [int(o) for o in lsc.index["SD04"]]


def _write_three(path, **kw):
    synth.write_lsc(path, _three_scans(), **kw)
    return path


@pytest.mark.parametrize("di_off", [10**9, 16, 17])
def test_lsc_bad_directory_pointer_raises_not_walks(tmp_path, caplog, di_off):
    # A non-zero DI02 pointer past EOF (or not at a DI02 chunk) is damage, never "no directory".
    p = _write_three(tmp_path / "a.lsc")
    raw = bytearray(p.read_bytes())
    raw[8:16] = di_off.to_bytes(8, "little")
    p.write_bytes(bytes(raw))
    with caplog.at_level(logging.WARNING), pytest.raises(SepSolveFormatError):
        LscFile(p)
    assert "walking" not in caplog.text


@pytest.mark.parametrize("keep", ["after_header", "mid_scans", "last_scan_start", "before_directory"])
def test_lsc_truncated_with_directory_raises(tmp_path, keep):
    # Truncation at chunk boundaries (or inside the 24-byte head) must not open with fewer scans.
    p = _write_three(tmp_path / "a.lsc")
    raw = p.read_bytes()
    offs = _sd04_offsets(p)
    di_off = int.from_bytes(raw[8:16], "little")
    cut = {"after_header": 16, "mid_scans": offs[1], "last_scan_start": offs[2],
           "before_directory": di_off}[keep]
    (tmp_path / "t.lsc").write_bytes(raw[:cut])
    with pytest.raises(SepSolveFormatError):
        LscFile(tmp_path / "t.lsc")


@pytest.mark.parametrize("keep", [0, 4, 15])
def test_lsc_truncated_inside_ft04_header_raises(tmp_path, keep):
    p = _write_three(tmp_path / "a.lsc")
    (tmp_path / "t.lsc").write_bytes(p.read_bytes()[:keep])
    with pytest.raises(SepSolveFormatError):
        LscFile(tmp_path / "t.lsc")


def test_lsc_directoryless_truncated_mid_chunk_raises(tmp_path):
    p = _write_three(tmp_path / "a.lsc", with_directory=False)
    raw = p.read_bytes()
    (tmp_path / "t.lsc").write_bytes(raw[:-5])           # cuts into the last (MD01) chunk
    with pytest.raises(SepSolveFormatError):
        LscFile(tmp_path / "t.lsc")


@pytest.mark.parametrize("extra", [1, 4, 7])
def test_lsc_directoryless_trailing_partial_header_raises(tmp_path, extra):
    p = _write_three(tmp_path / "a.lsc", with_directory=False)
    (tmp_path / "t.lsc").write_bytes(p.read_bytes() + b"\x00" * extra)
    with pytest.raises(SepSolveFormatError, match="trailing"):
        LscFile(tmp_path / "t.lsc")


def _hdr_with(old, new):
    text = synth.hdr_text()
    assert old in text
    return text.replace(old, new, 1)


@pytest.mark.parametrize("old,new,needle", [
    ("Data Rate=50", "Data Rate=abc", "FID Data Rate"),
    ("scanset period=20000", "scanset period=xx", "scanset period"),
    ("Valve1 Change0=1,4,300,999.00", "Valve1 Change0=1,zz,3", "modulation period"),
])
def test_hdr_non_numeric_value_raises_format_error(tmp_path, old, new, needle):
    text = _hdr_with(old, new)
    with pytest.raises(SepSolveFormatError, match=needle) as exc:
        parse_hdr_text(text)
    assert exc.value.path == "<hdr text>" and exc.value.offset == 0
    assert "line " in str(exc.value)
    p = tmp_path / "bad.HDR"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(SepSolveFormatError, match=needle) as exc:
        read_hdr(p)
    assert exc.value.path == str(p)                      # read_hdr names the file
    assert str(p) in str(exc.value)


def test_hdr_non_positive_scanset_period_is_still_just_missing():
    assert parse_hdr_text(synth.hdr_text(tof_scanset_us=0)).tof_rate_hz is None


def test_rsd_truncated_at_block_boundary_raises(tmp_path):
    p = tmp_path / "a.rsd"
    offsets = synth.write_rsd(p, np.arange(20.0), block=4)          # 5 blocks + terminator
    raw = p.read_bytes()
    assert read_rsd(p)[1].n_blocks == 5
    # Cut at the start of every block (header-only file included) and just before the terminator.
    for cut in offsets + [len(raw) - 8]:
        (tmp_path / "t.rsd").write_bytes(raw[:cut])
        with pytest.raises(SepSolveFormatError):
            read_rsd(tmp_path / "t.rsd")


# ── Final review, Minor 8: a variable modulation period is reported ─────────

def test_variable_modulation_period_warns_and_uses_change0(caplog):
    text = synth.hdr_text(pm_s=4.0).replace("Valve1 Changes=1", "Valve1 Changes=2").replace(
        "Valve1 Change0=1,4,300,999.00", "Valve1 Change0=1,4,300,20.00\r\nValve1 Change1=1,6,300,999.00")
    with caplog.at_level(logging.WARNING, logger="logic.loaders.sepsolve_format"):
        info = parse_hdr_text(text, "var.HDR")
    assert info.modulation_period_s == 4.0
    assert "var.HDR" in caplog.text and "2 modulation period" in caplog.text and "Change0" in caplog.text


def test_single_modulation_period_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="logic.loaders.sepsolve_format"):
        parse_hdr_text(synth.hdr_text(pm_s=4.0))
    assert caplog.text == ""
