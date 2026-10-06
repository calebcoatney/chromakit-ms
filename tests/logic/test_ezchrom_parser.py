"""Tests for the OpenLab EZChrom .dat reader/converter (Agilent 490 Micro GC).

Everything except the last section runs on synthetic bytes.  The real-file
section reads a local folder of lab data and skips when it is absent; no lab
data or values derived from it are stored in this repo.
"""
import glob
import hashlib
import json
import os
import re
import struct

import numpy as np
import pytest

from logic.c_folder import CFolder
from logic.loaders import ezchrom_parser as ez
from logic.loaders.ezchrom_loader import EZChromLoader


# ── synthetic EZChrom pieces ──────────────────────────────────────────────────

def _cstring(text: str) -> bytes:
    raw = text.encode("utf-16-le")
    n = len(text)
    length = bytes([n]) if n < 0xFF else b"\xff" + struct.pack("<H", n)
    return b"\xff\xfe\xff" + length + raw


def _trace_bytes(counts, interval=0.01, scale=1e-5, type_code=5, n_header=None):
    counts = np.asarray(counts, dtype="<f8")
    n = len(counts) if n_header is None else n_header
    header = struct.pack("<5i", type_code, n, n, 1, 0)
    trailer = struct.pack("<diiff8xd", 1.0, 0, 2, interval, scale, 1.0)
    assert len(trailer) == 40
    return header + counts.tobytes() + trailer


def _dth_bytes(entries):
    """Detector Trace Handler: entries of (number, description, unit, interval, scale)."""
    out = b"\x01\x00\x00\x00\x02\x00\x00\x00" + b"\x00" * 23
    for number, desc, unit, interval, scale in entries:
        out += _cstring(number) + b"\x01\x00\x00\x00" + _cstring(desc)
        out += struct.pack("<f", interval) + _cstring(unit) + struct.pack("<f", scale)
        out += struct.pack("<ff", 1.0, 180.0) + b"\x00" * 12
    return out


def _chrom_header(oadate, strings):
    return struct.pack("<4i", 1006, 1001, 1, 0) + struct.pack("<d", oadate) + b"".join(
        _cstring(s) for s in strings)


# 2025-01-02 03:04:05 UTC as an OLE automation date, nudged by float noise
_OADATE = 45659 + (3 * 3600 + 4 * 60 + 5) / 86400 - 1e-10

CH1 = "Channel 1, 10m PPQ Heated Injector, Backflush"
CH2 = "Channel 2, 8m 5CB Heated Injector"


def _streams(n=50, ch1_counts=None, ch2_counts=None):
    ch1 = np.arange(n, dtype=float) * 100 + 800000 if ch1_counts is None else ch1_counts
    ch2 = np.arange(n, dtype=float) * -3 + 50 if ch2_counts is None else ch2_counts
    return {
        "Chrom Header": _chrom_header(_OADATE, [
            r"C:\Methods\Test.met", "7", "Version A.04.07", "SYSTEM (SYSTEM)", "TEST uGC",
            "", r"C:\Seq\Test.seq"]),
        "Detector Trace Handler": _dth_bytes([
            ("0", CH1, "mV", 0.01, 1e-5),
            ("100", CH2, "mV", 0.01, 1e-5),
        ]),
        "Contents": b"\x02\x00\x00\x00" * 4 + _cstring(r"C:\Result\X.rslt\long_007.dat"),
        # deliberately out of numeric order: the reader sorts by trace number
        "Detector Data/Detector 100 Trace": _trace_bytes(ch2),
        "Detector Data/Detector 0 Trace": _trace_bytes(ch1),
    }


# ── decode_trace ──────────────────────────────────────────────────────────────

def test_decode_trace_round_trip():
    counts = [0.0, 1.0, -5.0, 1.2e8]
    block = ez.decode_trace(_trace_bytes(counts))
    np.testing.assert_array_equal(block.counts, counts)
    assert block.interval_s == 0.01          # f32 read as the decimal it encodes
    assert block.y_scale == 1e-5
    assert block.header == (5, 4, 4, 1, 0)


@pytest.mark.parametrize("kwargs, match", [
    ({"type_code": 3}, "type code"),
    ({"n_header": 5}, "bytes"),
    ({"interval": 0.0}, "interval"),
    ({"scale": -1.0}, "scale"),
])
def test_decode_trace_refuses_unvalidated_layouts(kwargs, match):
    with pytest.raises(ez.EZChromFormatError, match=match):
        ez.decode_trace(_trace_bytes([1.0, 2.0, 3.0, 4.0], **kwargs))


def test_decode_trace_too_short():
    with pytest.raises(ez.EZChromFormatError):
        ez.decode_trace(b"\x05\x00\x00\x00" * 3)


def test_read_cstrings_short_and_long():
    long_text = "x" * 300
    raw = b"junk" + _cstring("ab") + b"\x00\x01" + _cstring(long_text) + _cstring("")
    got = [s for _, _, s in ez.read_cstrings(raw)]
    assert got == ["ab", long_text, ""]


def test_channel_naming():
    assert ez.channel_detector_id(CH1, "0") == "Ch1_PPQ"
    assert ez.channel_detector_id(CH2, "100") == "Ch2_5CB"
    assert ez.channel_detector_id("Front TCD", "3") == "Det3"
    assert ez.channel_column(CH1) == "10m PPQ"
    assert ez.channel_column("Channel 3, 10m MS5A Backflush") == "10m MS5A"
    assert ez.channel_column("Front TCD") is None


# ── read_ezchrom_streams ──────────────────────────────────────────────────────

def test_read_streams_channels_and_axes():
    inj = ez.read_ezchrom_streams(_streams(n=50), version="A.04.07")
    assert [c.detector for c in inj.channels] == ["Ch1_PPQ", "Ch2_5CB"]
    assert [c.trace_number for c in inj.channels] == ["0", "100"]
    ch1 = inj.channels[0]
    assert ch1.description == CH1 and ch1.column == "10m PPQ" and ch1.y_unit == "mV"
    assert ch1.n_points == 50 and ch1.sampling_rate_hz == pytest.approx(100.0)
    np.testing.assert_allclose(ch1.time_min(), np.arange(50) * 0.01 / 60)
    np.testing.assert_allclose(ch1.signal(), (np.arange(50) * 100 + 800000) * 1e-5)
    assert inj.warnings == []


def test_read_streams_header_metadata():
    inj = ez.read_ezchrom_streams(_streams(), version="A.04.07")
    assert inj.injection_time_utc.isoformat() == "2025-01-02T03:04:05+00:00"
    assert inj.sample_name == "7"
    assert inj.method_path == r"C:\Methods\Test.met"
    assert inj.operator == "SYSTEM (SYSTEM)"
    assert inj.instrument_name == "TEST uGC"
    assert inj.sequence_file == r"C:\Seq\Test.seq"
    assert inj.software_version == "A.04.07"
    assert inj.original_result_path == r"C:\Result\X.rslt\long_007.dat"


def test_channel_selection_never_falls_back():
    inj = ez.read_ezchrom_streams(_streams())
    assert inj.channel(None).detector == "Ch1_PPQ"
    for name in ["Ch2_5CB", "ch2_5cb", "100", "100_1", CH2, "Detector 100 Trace"]:
        assert inj.channel(name).detector == "Ch2_5CB", name
    assert inj.channel("0_0").detector == "Ch1_PPQ"
    with pytest.raises(KeyError):
        inj.channel("FID1A")


def test_trailer_and_handler_must_agree():
    s = _streams()
    s["Detector Trace Handler"] = _dth_bytes([("0", CH1, "mV", 0.02, 1e-5), ("100", CH2, "mV", 0.01, 1e-5)])
    with pytest.raises(ez.EZChromFormatError, match="interval_s"):
        ez.read_ezchrom_streams(s)


def test_no_traces_is_an_error():
    s = _streams()
    del s["Detector Data/Detector 0 Trace"], s["Detector Data/Detector 100 Trace"]
    with pytest.raises(ez.EZChromFormatError):
        ez.read_ezchrom_streams(s)


def test_parse_rslt_name():
    seq, tz = ez.parse_rslt_name("2025-05-27 17-27-27 (GMT -06-00)8566-043.rslt")
    assert seq == "8566-043"
    assert tz.utcoffset(None).total_seconds() == -6 * 3600
    assert ez.parse_rslt_name("whatever.rslt") == (None, None)


# ── conversion + loading (reader patched with synthetic streams) ──────────────

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture
def fake_rslt(tmp_path, monkeypatch):
    """A result-set folder holding one placeholder .dat; the OLE reader is patched."""
    rslt = tmp_path / "2025-01-01 20-00-00 (GMT -07-00)TEST-001.rslt"
    rslt.mkdir()
    dat = rslt / "long_007.dat"
    dat.write_bytes(b"placeholder for an OLE file " * 10)
    streams = _streams(n=30)
    monkeypatch.setattr(ez, "read_ezchrom_dat",
                        lambda path: ez.read_ezchrom_streams(streams, version="A.04.07"))
    return rslt, dat


def test_convert_dat_writes_c_folder_and_leaves_source(fake_rslt, tmp_path):
    rslt, dat = fake_rslt
    before = (_sha(dat), os.stat(dat).st_mtime_ns, sorted(os.listdir(rslt)))
    out = tmp_path / "converted"

    c_path = ez.convert_dat(str(dat), str(out))

    assert os.path.basename(c_path) == "TEST-001_long_007.C"
    assert (_sha(dat), os.stat(dat).st_mtime_ns, sorted(os.listdir(rslt))) == before
    assert _sha(os.path.join(c_path, "data", "long_007.dat")) == before[0]
    assert os.path.isdir(os.path.join(c_path, "results"))
    assert not os.path.exists(c_path + ".partial")

    m = CFolder.open(c_path).get_manifest()
    assert m["signal_type"] == "gc" and m["source_format"] == "ezchrom_dat"
    assert m["instrument"] == "Agilent 490 Micro GC" and m["instrument_name"] == "TEST uGC"
    assert m["created"] == "2025-01-01T20:04:05-07:00"
    assert m["injection_time_utc"] == "2025-01-02T03:04:05Z"
    assert m["sample_id"] == "TEST-001_long_007" and m["sample_name"] == "7"
    assert m["sequence_id"] == "TEST-001"
    assert m["y_units"] == "mV" and m["sampling_rate_hz"] == 100.0
    assert m["detectors"] == ["Ch1_PPQ", "Ch2_5CB"]
    assert [c["column"] for c in m["channels"]] == ["10m PPQ", "8m 5CB"]
    assert m["source_path"] == str(dat) and m["source_sha256"] == before[0]
    assert m["conversion_warnings"] == []


def test_convert_dat_refuses_output_inside_source(fake_rslt):
    rslt, dat = fake_rslt
    with pytest.raises(ValueError):
        ez.convert_dat(str(dat), str(rslt / "out"))


def test_lone_dat_may_convert_beside_itself(tmp_path, monkeypatch):
    dat = tmp_path / "run.dat"
    dat.write_bytes(b"placeholder")
    monkeypatch.setattr(ez, "read_ezchrom_dat", lambda path: ez.read_ezchrom_streams(_streams()))
    c_path = ez.convert_dat(str(dat), str(tmp_path))
    assert os.path.basename(c_path) == "run.C" and dat.exists()


def test_convert_dat_keeps_existing_c_folder(fake_rslt, tmp_path):
    rslt, dat = fake_rslt
    c_path = ez.convert_dat(str(dat), str(tmp_path / "out"))
    marker = os.path.join(c_path, "results", "keep.txt")
    open(marker, "w").write("x")
    assert ez.convert_dat(str(dat), str(tmp_path / "out")) == c_path
    assert os.path.exists(marker)
    ez.convert_dat(str(dat), str(tmp_path / "out"), overwrite=True)
    assert not os.path.exists(marker)


def test_cfolder_dispatches_on_source_format(fake_rslt, tmp_path):
    rslt, dat = fake_rslt
    cf = CFolder.open(ez.convert_dat(str(dat), str(tmp_path / "out")))
    assert cf.profile.name == "gc"
    assert cf.get_available_detectors() == ["Ch1_PPQ", "Ch2_5CB"]

    d1 = cf.load_signal()
    assert d1["metadata"]["detector"] == "Ch1_PPQ"
    assert d1["metadata"]["has_ms_data"] is False
    np.testing.assert_allclose(d1["y"], (np.arange(30) * 100 + 800000) * 1e-5)

    d2 = cf.load_signal(signal_factor=2.0, detector="Ch2_5CB")
    assert d2["metadata"]["detector"] == "Ch2_5CB"
    np.testing.assert_allclose(d2["y"], (np.arange(30) * -3 + 50) * 1e-5 * 2.0)
    np.testing.assert_allclose(d2["x"], np.arange(30) * 0.01 / 60)

    with pytest.raises(ValueError, match="FID1A"):
        cf.load_signal(detector="FID1A")


def test_convert_rslt_skips_non_dat(fake_rslt, tmp_path):
    rslt, dat = fake_rslt
    (rslt / "notes.pdf").write_bytes(b"%PDF")
    created = ez.convert_rslt(str(rslt), str(tmp_path / "out"))
    assert [os.path.basename(c) for c in created] == ["TEST-001_long_007.C"]


def test_loader_without_dat_raises(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        EZChromLoader().load(str(tmp_path))
    assert EZChromLoader().get_available_detectors(str(tmp_path)) == []


# ── real files (skipped unless the lab folder is present) ─────────────────────

_REAL_DIR = os.environ.get(
    "CHROMAKIT_EZCHROM_TEST_DIR",
    os.path.expanduser("~/Library/CloudStorage/OneDrive-NLR/GC-MS/251 micro gc data"),
)
_REAL_RSLTS = sorted(glob.glob(os.path.join(_REAL_DIR, "*.rslt")))
_REAL_DATS = [p for p in (os.path.join(r, "long_001.dat") for r in _REAL_RSLTS) if os.path.isfile(p)]
real = pytest.mark.skipif(not _REAL_DATS, reason="EZChrom example data not available")


@real
def test_real_file_decodes_and_matches_vendor_levels():
    dat = _REAL_DATS[-1]
    pytest.importorskip("olefile")
    inj = ez.read_ezchrom_dat(dat)
    assert len(inj.channels) >= 1
    for ch in inj.channels:
        assert ch.sampling_rate_hz > 0 and ch.n_points > 1
        assert np.all(ch.counts == np.round(ch.counts))
    acam = ez.map_acaml_by_data_file(os.path.dirname(dat)).get("long_001.dat")
    if acam is None:
        pytest.skip("no ACAML result file next to the .dat")
    # Every vendor peak's LevelStart/LevelEnd is the decoded sample at
    # BaselineStart/BaselineEnd (values read at test time, not stored here).
    import xml.etree.ElementTree as ET
    ns = "{urn:schemas-agilent-com:acaml15}"
    root = ET.parse(acam).getroot()
    md = root.find(f"{ns}Doc/{ns}Content/{ns}Injections/{ns}MeasData")
    sig_trace = {s.get("id"): s.findtext(f"{ns}TraceID") for s in md.findall(f"{ns}Signal")}
    checked = 0
    for sr in root.iter(f"{ns}SignalResult"):
        ch = inj.channel(sig_trace[sr.find(f"{ns}Signal_ID").get("id")])
        for pk in sr.findall(f"{ns}Peak"):
            for t_tag, lvl_tag in (("BaselineStart", "LevelStart"), ("BaselineEnd", "LevelEnd")):
                t, lvl = pk.find(f"{ns}{t_tag}"), pk.find(f"{ns}{lvl_tag}")
                if t is None or lvl is None:
                    continue
                i = int(round(float(t.get("val")) * 60 / ch.interval_s))
                assert ch.counts[i] == float(lvl.get("val"))
                checked += 1
    assert checked > 0


@real
def test_real_file_converts_without_touching_source(tmp_path):
    dat = _REAL_DATS[-1]
    pytest.importorskip("olefile")
    rslt = os.path.dirname(dat)
    before = {n: os.stat(os.path.join(rslt, n)).st_mtime_ns for n in os.listdir(rslt)}
    sha_before = _sha(dat)

    (c_path,) = ez.parse_ezchrom(dat, str(tmp_path))

    assert _sha(dat) == sha_before
    assert {n: os.stat(os.path.join(rslt, n)).st_mtime_ns for n in os.listdir(rslt)} == before
    m = CFolder.open(c_path).get_manifest()
    assert m["source_sha256"] == sha_before
    assert m["conversion_warnings"] == []
    cf = CFolder.open(c_path)
    for det in cf.get_available_detectors():
        d = cf.load_signal(detector=det)
        assert d["metadata"]["detector"] == det and len(d["x"]) == len(d["y"])
