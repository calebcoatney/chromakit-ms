import csv
import json
import re

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


def test_mgf_follows_mzmine_gnps_gc_export(tmp_path):
    paths = _export(tmp_path, [_peak(1, 14.3, 0.44, 100.0, name="Heptane"),
                               _peak(2, 17.2, 1.12, 300.0), _peak(3, 20.0, 2.0, 50.0)])
    assert paths.mgf.name == "8260-110 SAF - GCxGC spectra.mgf"
    entries = _read_mgf(paths.mgf)
    assert [h["FEATURE_ID"] for h, _ in entries] == ["1", "2"]          # peak 3 had no spectrum
    h1, p1 = entries[0]
    assert list(h1) == ["TITLE", "FEATURE_ID", "PEPMASS", "RTINSECONDS", "SCANS", "MSLEVEL",
                        "CHARGE", "Num peaks"]
    assert h1["TITLE"] == "Peak 1 - Heptane" and h1["SCANS"] == "1"
    assert h1["PEPMASS"] == "57"                                       # base peak m/z
    assert h1["RTINSECONDS"] == "858.00"                                # 14.3 min
    assert h1["MSLEVEL"] == "2" and h1["CHARGE"] == "1+" and h1["Num peaks"] == "2"
    assert p1 == [(57.0, 100.0), (71.0, 40.0)]


def test_min_snr_gates_mgf_and_msp_and_pepmass_follows_the_gate(tmp_path):
    noise = [(m, 1.0 + (m % 3)) for m in range(30, 130)]
    spectra = {1: sorted(dict(noise + [(149, 1000.0), (223, 60.0)]).items())}

    def spectrum(p):
        p.spectrum_scan, p.spectrum_source = 1001, "dbc"
        return spectra.get(p.peak_number)

    paths = export_gcxgc_results([_peak(1, 14.3, 0.44, 100.0)], FakeRun(), tmp_path / "r",
                                 settings={"detection_signal": "fid"}, get_spectrum=spectrum,
                                 min_snr=5.0)
    (h, ions), = _read_mgf(paths.mgf)
    assert [mz for mz, _ in ions] == [149.0, 223.0]
    assert h["PEPMASS"] == "149" and h["Num peaks"] == "2"
    assert "Num Peaks: 2" in paths.msp.read_text()


def test_a_feature_left_with_only_its_base_peak_is_still_exported(tmp_path):
    spectra = {1: [(91, 4.5)] + [(m, 4.0) for m in range(30, 60)]}

    def spectrum(p):
        p.spectrum_scan, p.spectrum_source = 1001, "dbc"
        return spectra.get(p.peak_number)

    paths = export_gcxgc_results([_peak(1, 14.3, 0.44, 100.0)], FakeRun(), tmp_path / "r",
                                 settings={"detection_signal": "dbc_tic"}, get_spectrum=spectrum,
                                 min_snr=50.0)
    (h, ions), = _read_mgf(paths.mgf)
    assert ions == [(91.0, 4.5)] and h["PEPMASS"] == "91" and h["Num peaks"] == "1"
    rows = list(csv.reader(open(paths.gnps_quant, encoding="utf-8")))
    assert rows[1][:2] == ["1", "91"]


def test_gnps_quant_csv(tmp_path):
    paths = _export(tmp_path, [_peak(1, 14.3, 0.44, 100.0), _peak(2, 17.2, 1.12, 300.0),
                               _peak(3, 20.0, 2.0, 50.0)])
    assert paths.gnps_quant.name == "8260-110 SAF - GNPS quant.csv"
    rows = list(csv.reader(open(paths.gnps_quant, encoding="utf-8")))
    assert rows[0] == ["row ID", "row m/z", "row retention time", "8260-110 SAF Peak area"]
    assert [r[0] for r in rows[1:]] == ["1", "2"]                       # only peaks in the MGF
    assert rows[1][1] == "57" and float(rows[1][2]) == pytest.approx(14.3)
    assert float(rows[2][3]) == pytest.approx(300.0)


def test_gnps_quant_header_survives_commas_and_quotes(tmp_path):
    class Run(FakeRun):
        sample_id = 'G2, rep "b"'

    paths = export_gcxgc_results([_peak(1, 14.3, 0.44, 100.0)], Run(), tmp_path / "r",
                                 settings={"detection_signal": "fid"}, get_spectrum=_spectrum)
    header = next(csv.reader(open(paths.gnps_quant, encoding="utf-8")))
    assert header[3] == 'G2, rep "b" Peak area' and len(header) == 4


def test_offset_is_null_when_detected_on_a_tic(tmp_path):
    class NoFitRun(FakeRun):
        def offset(self):
            raise AssertionError("the FID->TOF offset must not be fitted for TIC detection")

    paths = export_gcxgc_results([_peak(1, 14.3, 0.44, 100.0, signal="dbc_tic")], NoFitRun(),
                                 tmp_path / "r", settings={"detection_signal": "dbc_tic"},
                                 get_spectrum=_spectrum)
    assert json.load(open(paths.json))["run"]["offset"] is None


def test_export_works_without_ms_toolkit(tmp_path, monkeypatch):
    import importlib
    import sys
    monkeypatch.setitem(sys.modules, "ms_toolkit", None)
    import logic.gcxgc_export as ge
    importlib.reload(ge)
    paths = ge.export_gcxgc_results([_peak(1, 14.3, 0.44, 100.0)], FakeRun(), tmp_path / "r",
                                    settings={"detection_signal": "fid"}, get_spectrum=_spectrum)
    assert paths.mgf.exists() and paths.gnps_quant.exists()


def test_zero_peaks_writes_an_empty_mgf_and_a_header_only_quant(tmp_path):
    paths = _export(tmp_path, [])
    assert paths.mgf.read_text() == ""
    assert list(csv.reader(open(paths.gnps_quant, encoding="utf-8"))) == [
        ["row ID", "row m/z", "row retention time", "8260-110 SAF Peak area"]]


# ── Final review, Important 3: the JSON names the ChromaKit commit ──────────

def _no_distribution(monkeypatch):
    import importlib.metadata as md

    def missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "distribution", missing)


def test_commit_falls_back_to_git_in_the_package_folder(monkeypatch):
    import pathlib
    import subprocess
    from logic.gcxgc_export import _chromakit_commit
    _no_distribution(monkeypatch)
    repo = pathlib.Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    commit = _chromakit_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", commit) and commit == head


def test_commit_prefers_pep610_vcs_info(monkeypatch):
    import importlib.metadata as md
    from logic.gcxgc_export import _chromakit_commit

    class Dist:
        def read_text(self, name):
            return json.dumps({"url": "https://github.com/x/chromakit-ms",
                               "vcs_info": {"vcs": "git", "commit_id": "0123456789abcdef0123456789abcdef01234567"}})

    monkeypatch.setattr(md, "distribution", lambda name: Dist())
    assert _chromakit_commit() == "0123456789abcdef0123456789abcdef01234567"


def test_commit_is_empty_without_git(monkeypatch, tmp_path):
    import subprocess
    from logic.gcxgc_export import _chromakit_commit
    _no_distribution(monkeypatch)

    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert _chromakit_commit() == ""
    rec = json.load(open(_export(tmp_path, [_peak(1, 14.3, 0.44, 100.0)]).json))
    assert rec["chromakit_commit"] == "" and "chromakit_version" in rec


def test_json_record_names_the_commit(tmp_path):
    rec = json.load(open(_export(tmp_path, [_peak(1, 14.3, 0.44, 100.0)]).json))
    assert re.fullmatch(r"(?:[0-9a-f]{7,40})?", rec["chromakit_commit"])
    assert rec["chromakit_commit"]                      # this worktree is a git checkout


# ── Final review, Minor 6: intensities are fixed-point in MSP and MGF ───────

def test_intensity_text_is_fixed_point():
    from logic.gcxgc_export import _intensity
    assert [_intensity(v) for v in (4.825e6, 100.0, 40.0, 0.123456, 1234567.891, 0.0)] == [
        "4825000", "100", "40", "0.1235", "1234567.891", "0"]


def test_large_intensities_never_use_exponents(tmp_path, monkeypatch):
    spectra = {1: [(57, 4.825e6), (71, 1.5e7), (85, 0.25)]}
    monkeypatch.setitem(SPECTRA, 1, spectra[1])
    paths = _export(tmp_path, [_peak(1, 14.3, 0.44, 100.0)])
    for path in (paths.msp, paths.mgf):
        lines = [ln for ln in path.read_text().splitlines() if ln[:1].isdigit()]
        assert lines == ["57 4825000", "71 15000000", "85 0.25"], path.name
