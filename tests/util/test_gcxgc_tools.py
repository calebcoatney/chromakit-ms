import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NIST_MSP = """Name: Benzene
Formula: C6H6
MW: 78
CAS#: 71-43-2; NIST#: 12345
Synon: Cyclohexatriene
Num Peaks: 4
51 120; 52 40; 77 150;
78 999;

Name: Toluene
Formula: C7H8
MW: 92
CAS#: 108-88-3; NIST#: 2
Num Peaks: 2
91 999; 92 600;
"""


def test_nist_msp_converts_to_one_pair_per_line(tmp_path, monkeypatch):
    conv = _load("nist_text_for_mstoolkit")
    lines = conv.convert_lines(NIST_MSP.splitlines())
    assert "CASNO: 71-43-2" in lines and "51 120" in lines and "78 999" in lines
    out = tmp_path / "lib.txt"
    out.write_text("\n".join(lines) + "\n")
    io = pytest.importorskip("ms_toolkit.io")
    from ms_toolkit.models import Compound
    # Upstream ms-toolkit bug: tic_scaling returns numpy scalars, which orjson >= 3.10 refuses to
    # serialise inside parse_core (true for any text library). Coerce to float so the parse itself
    # is what is under test. Drop this shim once ms-toolkit's tic_scaling returns Python floats.
    scale = io.tic_scaling
    monkeypatch.setattr(io, "tic_scaling", lambda spec: [(float(mz), float(v)) for mz, v in scale(spec)])
    lib = io.parse_core(file_path=str(out), load_cache=False, quiet=True)
    comps = {k: v if isinstance(v, Compound) else Compound.from_json(v) for k, v in lib.items()}
    assert set(comps) == {"Benzene", "Toluene"}
    assert len(comps["Benzene"].spectrum) == 4 and comps["Toluene"].formula == "C7H8"


def test_compare_chromspace_matches_and_summarizes(tmp_path):
    cmp = _load("compare_chromspace")
    cs = tmp_path / "cs.csv"
    cs.write_text(
        "Peak #,Source,Compound,Synonyms,Library RI,1tR,2tR,Area,Area %\n"
        "1,TIC [Front Signal],o-Xylene,,,17.20,0.0187,1000,50\n"
        "2,TIC [Front Signal],,,,21.96,0.0107,1000,50\n"
        "Group 1,C10 dicycloalkane,,,,,,2000,100\n")
    ck = tmp_path / "peaks.json"
    ck.write_text(json.dumps({"peaks": [
        {"peak_number": 1, "rt1": 17.21, "rt2": 1.12, "volume": 2000.0, "compound_name": "o-Xylene"},
        {"peak_number": 2, "rt1": 21.95, "rt2": 0.64, "volume": 3000.0, "compound_name": "Cyclohexane, propyl-"},
        {"peak_number": 3, "rt1": 40.0, "rt2": 2.0, "volume": 10.0, "compound_name": None}]}))
    a, b = cmp.read_chromspace(cs), cmp.read_chromakit(ck)
    assert len(a) == 2 and len(b) == 3
    pairs = cmp.match(a, b, t1_tol=0.1, t2_tol=0.1)
    assert sorted(pairs) == [(0, 0), (1, 1)]
    s = cmp.summarize(a, b, pairs)
    assert s["matched"] == 2 and s["volume_ratio_median"] == pytest.approx(2.5)
    assert (s["id_pairs"], s["id_agree"]) == (1, 1)
    assert (s["chromspace_unmatched"], s["chromakit_unmatched"]) == (0, 1)


def test_compare_chromspace_reports_unmatched_on_both_sides(tmp_path):
    cmp = _load("compare_chromspace")
    cs = [{"peak": "1", "compound": "", "t1_min": 10.0, "t2_s": 1.0, "area": 5.0},
          {"peak": "2", "compound": "", "t1_min": 30.0, "t2_s": 1.0, "area": 5.0}]
    ck = [{"peak": "1", "compound": "", "t1_min": 10.0, "t2_s": 1.0, "area": 5.0},
          {"peak": "2", "compound": "", "t1_min": 50.0, "t2_s": 1.0, "area": 5.0},
          {"peak": "3", "compound": "", "t1_min": 60.0, "t2_s": 1.0, "area": 5.0}]
    pairs = cmp.match(cs, ck, t1_tol=0.1, t2_tol=0.1)
    s = cmp.summarize(cs, ck, pairs)
    assert (s["matched"], s["chromspace_unmatched"], s["chromakit_unmatched"]) == (1, 1, 2)


def test_compare_chromspace_warns_about_skipped_and_empty_input(tmp_path, capsys):
    cmp = _load("compare_chromspace")
    good = tmp_path / "good.csv"
    good.write_text(
        "Peak #,Source,Compound,Synonyms,Library RI,1tR,2tR,Area,Area %\n"
        "1,TIC,o-Xylene,,,17.20,0.0187,1000,50\n"
        "Group 1,C10 dicycloalkane,,,,,,2000,100\n")
    assert len(cmp.read_chromspace(good)) == 1
    assert capsys.readouterr().err == ""            # header, group rows and a clean parse are silent
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "Peak #,Source,Compound,Synonyms,Library RI,1tR,2tR,Area,Area %\n"
        "1,TIC,o-Xylene,,,17.20,0.0187,1000,50\n"
        "2,TIC,,,,n/a,0.0107,1000,50\n"
        "3,TIC,,,,21.96\n")
    assert len(cmp.read_chromspace(bad)) == 1
    assert "2 ChromSpace data row(s) skipped" in capsys.readouterr().err
    empty = tmp_path / "empty.csv"
    empty.write_text("Peak #,Source,Compound,Synonyms,Library RI,1tR,2tR,Area,Area %\n")
    assert cmp.read_chromspace(empty) == []
    assert "no peak rows" in capsys.readouterr().err
