import csv
import json

import pytest

from logic.gcxgc_export import CHROMSPACE_COLUMNS, export_gcxgc_results, safe_name
from logic.gcxgc_peak import GCxGC2DPeak
from logic.gcxgc_run import OffsetModel


class FakeRun:
    sample_id = "8260-110 SAF"
    modulation_period_s, phase_shift_s = 4.0, -1.8
    provenance = {"modulation_period_s": "hdr", "phase_shift_s": "lsc_view"}
    fid_hz = tof_hz = 50.0
    paths = {"hdr": "/x/a.HDR", "rsd": "/x/a.rsd", "lsc": "/x/a.lsc", "dbc_lsc": None}

    def offset(self):
        return OffsetModel(a=-6.0, b=0.05, ok=True, n_windows=12)


def _peak(n, rt1, rt2, vol, signal="fid", name=None):
    p = GCxGC2DPeak(peak_number=n, rt1=rt1, rt2=rt2, volume=vol, n_sub_peaks=3, mod_start=1,
                    mod_end=3, apex_mod=2, start_time=rt1 - 0.05, end_time=rt1 + 0.05)
    p.detection_signal, p.compound_name = signal, name
    return p


SPECTRA = {1: [(57, 100.0), (71, 40.0)], 2: [(91, 100.0)]}


def _spectrum(p):
    p.spectrum_scan, p.spectrum_source = 1000 + p.peak_number, "dbc"
    return SPECTRA.get(p.peak_number)


def _export(tmp_path, peaks):
    return export_gcxgc_results(peaks, FakeRun(), tmp_path / "results",
                                settings={"pm": None, "detection_signal": "fid"},
                                get_spectrum=_spectrum, library="NIST14.json", search_method="vector")


def test_csv_has_chromspace_layout(tmp_path):
    peaks = [_peak(2, 17.2, 1.12, 300.0, name="o-Xylene"), _peak(1, 14.3, 0.44, 100.0),
             _peak(3, 20.0, 2.0, 600.0, signal="dbc_tic")]
    paths = _export(tmp_path, peaks)
    rows = list(csv.reader(open(paths.csv, encoding="utf-8")))
    assert rows[0] == CHROMSPACE_COLUMNS
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]
    r2 = rows[2]
    assert r2[1] == "TIC [Front Signal]" and r2[2] == "o-Xylene" and r2[3] == r2[4] == ""
    assert float(r2[5]) == pytest.approx(17.2) and float(r2[6]) == pytest.approx(1.12 / 60)
    assert rows[3][1] == "DBC TIC"
    assert sum(float(r[8]) for r in rows[1:]) == pytest.approx(100.0)
    assert float(rows[3][8]) == pytest.approx(60.0)


def test_json_record(tmp_path):
    rec = json.load(open(_export(tmp_path, [_peak(1, 14.3, 0.44, 100.0)]).json))
    assert rec["schema"] == "chromakit.gcxgc.v1" and rec["sample_id"] == "8260-110 SAF"
    assert rec["run"]["provenance"]["phase_shift_s"] == "lsc_view"
    assert rec["run"]["offset"]["ok"] is True
    assert rec["run"]["files"] == {"hdr": "a.HDR", "rsd": "a.rsd", "lsc": "a.lsc"}
    assert rec["library"] == "NIST14.json" and rec["volume_units"]
    p = rec["peaks"][0]
    assert p["area_percent"] == pytest.approx(100.0) and p["spectrum_scan"] == 1001


def test_msp_reads_back_through_ms_toolkit(tmp_path, monkeypatch):
    io = pytest.importorskip("ms_toolkit.io")
    from ms_toolkit.models import Compound
    # ms_toolkit's tic_scaling returns numpy scalars, which orjson 3.11 refuses to serialise inside
    # parse_core (true for any text MSP). Coerce to float so the parse itself is what is under test.
    scale = io.tic_scaling
    monkeypatch.setattr(io, "tic_scaling", lambda spec: [(float(mz), float(v)) for mz, v in scale(spec)])
    paths = _export(tmp_path, [_peak(1, 14.3, 0.44, 100.0, name="Heptane"),
                               _peak(2, 17.2, 1.12, 300.0), _peak(3, 20.0, 2.0, 50.0)])
    lib = io.parse_core(file_path=str(paths.msp), load_cache=False, quiet=True)
    comps = {k: v if isinstance(v, Compound) else Compound.from_json(v) for k, v in lib.items()}
    assert set(comps) == {"Peak 1 - Heptane", "Peak 2"}       # peak 3 had no spectrum
    assert len(comps["Peak 1 - Heptane"].spectrum) == 2


def test_zero_peaks_still_writes_files(tmp_path):
    paths = _export(tmp_path, [])
    assert list(csv.reader(open(paths.csv, encoding="utf-8"))) == [CHROMSPACE_COLUMNS]
    assert paths.msp.read_text() == "" and json.load(open(paths.json))["peaks"] == []
    assert not list((tmp_path / "results").glob("*.tmp"))


def test_safe_name():
    assert safe_name('a/b:c*?') == "a_b_c__"
    assert safe_name("  ") == "sample"


def _read_mgf(path):
    """Minimal MGF reader for tests: list of (headers dict, [(mz, intensity)])."""
    entries, headers, peaks, inside = [], {}, [], False
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if line == "BEGIN IONS":
            headers, peaks, inside = {}, [], True
        elif line == "END IONS":
            entries.append((headers, peaks))
            inside = False
        elif inside and "=" in line and not line[0].isdigit():
            key, _, val = line.partition("=")
            headers[key] = val
        elif inside and line:
            mz, inten = line.split()
            peaks.append((float(mz), float(inten)))
    return entries


def test_mgf_has_one_entry_per_spectrum_and_no_retention_time(tmp_path):
    paths = _export(tmp_path, [_peak(1, 14.3, 0.44, 100.0, name="Heptane"),
                               _peak(2, 17.2, 1.12, 300.0), _peak(3, 20.0, 2.0, 50.0)])
    assert paths.mgf.name == "8260-110 SAF - GCxGC spectra.mgf"
    entries = _read_mgf(paths.mgf)
    assert [h["FEATURE_ID"] for h, _ in entries] == ["1", "2"]          # peak 3 had no spectrum
    h1, p1 = entries[0]
    assert h1["TITLE"] == "Peak 1 - Heptane" and h1["SCANS"] == "1"
    assert h1["PEPMASS"] == "0" and h1["MSLEVEL"] == "1"
    assert p1 == [(57.0, 100.0), (71.0, 40.0)]
    text = paths.mgf.read_text()
    assert "RTINSECONDS" not in text and "RTINMINUTES" not in text
    assert text.count("BEGIN IONS") == text.count("END IONS") == 2


def test_zero_peaks_writes_an_empty_mgf(tmp_path):
    paths = _export(tmp_path, [])
    assert paths.mgf.read_text() == ""
