import datetime
import json
import os

import openpyxl
import pytest

from logic import gc12_workbook as wb
from logic.gc12_workbook import collect_injections, pair_injections, write_paired_workbook

T0 = datetime.datetime(2026, 11, 3, 10, 15)
PEAKS = [
    {"peak_number": 1, "retention_time": 1.5, "integrator": "Integrator",
     "width": 0.1, "area": 100.0, "start_time": 1.4, "end_time": 1.6,
     "compound_id": "Methane"},
    {"peak_number": 2, "retention_time": 5.234, "integrator": "Integrator",
     "width": 0.2, "area": 55.5, "start_time": 5.1, "end_time": 5.4,
     "compound_id": "Unknown (5.234)"},
]
GC1_DETS = ["FID1A", "TCD3C"]
GC2_DETS = ["FID1A", "FID2B"]


def make_c(root, inst, name, dets, sample=None, peaks=PEAKS, write_json=True):
    c = root / inst / f"{name}.C"
    (c / "data" / f"{name}.D").mkdir(parents=True)
    (c / "results").mkdir()
    sample = sample or name
    (c / "manifest.json").write_text(json.dumps({"sample_id": sample}))
    if write_json:
        for d in dets:
            (c / "results" / f"{sample} - {d}.json").write_text(json.dumps({
                "signal": f"Signal: {sample}\\{d}.ch", "sample_id": sample,
                "timestamp": "ignored", "peaks": peaks}))
    return str(c)


@pytest.fixture(autouse=True)
def fake_time(monkeypatch):
    monkeypatch.setattr(wb, "acquisition_time", lambda d, det=None: T0)


def build(tmp_path, records_for=None, peaks=PEAKS, extra=()):
    gc1 = tmp_path / "GC1"
    gc2 = tmp_path / "GC2"
    gc1.mkdir(); gc2.mkdir()
    for n in ["S005"] + list(extra):
        make_c(tmp_path, "GC1", n, GC1_DETS, peaks=peaks)
    make_c(tmp_path, "GC2", "S005", GC2_DETS, peaks=peaks)
    rec = {}
    for inst, dets in (("GC1", GC1_DETS), ("GC2", GC2_DETS)):
        for item in sorted(os.listdir(tmp_path / inst)):
            for d in dets:
                rec[(os.path.join(str(tmp_path / inst), item), d)] = (True, "")
    if records_for:
        records_for(rec)
    i1 = collect_injections(str(gc1), "GC1", GC1_DETS, rec)
    i2 = collect_injections(str(gc2), "GC2", GC2_DETS, rec)
    return i1, i2, rec


def write(tmp_path, i1, i2):
    path = str(tmp_path / "out.xlsx")
    summary = write_paired_workbook(pair_injections(i1, i2), path)
    return openpyxl.load_workbook(path), summary


def test_paired_workbook_layout(tmp_path):
    i1, i2, _ = build(tmp_path, extra=["S006"])
    make_c(tmp_path, "GC2", "S006", GC2_DETS)
    rec = {(os.path.join(str(tmp_path / "GC2"), f"{n}.C"), d): (True, "")
           for n in ("S005", "S006") for d in GC2_DETS}
    i2 = collect_injections(str(tmp_path / "GC2"), "GC2", GC2_DETS, rec)
    rec1 = {(os.path.join(str(tmp_path / "GC1"), f"{n}.C"), d): (True, "")
            for n in ("S005", "S006") for d in GC1_DETS}
    i1 = collect_injections(str(tmp_path / "GC1"), "GC1", GC1_DETS, rec1)
    # distinct times so pairing is unambiguous
    for k, inj in enumerate(i1):
        inj.acquired = T0 + datetime.timedelta(minutes=20 * k)
    for k, inj in enumerate(i2):
        inj.acquired = T0 + datetime.timedelta(minutes=20 * k)
    book, summary = write(tmp_path, i1, i2)
    assert book.sheetnames == ["GC1 raw formatting", "GC2 raw formatting", "Pairing"]
    g1, g2 = book["GC1 raw formatting"], book["GC2 raw formatting"]
    assert g1["A1"].value == "Signal: S005\\FID1A.ch"
    assert g1["I1"].value.endswith("TCD3C.ch")
    assert g1["A2"].value == "S005"
    assert isinstance(g1["A3"].value, datetime.datetime)
    assert g1["A3"].number_format == "m/dd/yy h:mm"
    assert g1["C4"].value == "Integrator"
    assert g1["B5"].value == 1.5
    assert g2["A3"].value in (None, "") and g2["I3"].value in (None, "")
    assert g2["I1"].value.endswith("FID2B.ch")
    # block 1: rows 1-3 header, 4 column headers, 5-6 peaks; blank row 7; block 2 at 8
    assert g1["A6"].value == 2 and g1["A7"].value is None
    assert g1["A8"].value.startswith("Signal:")
    assert g1.max_column <= 15 and g2.max_column <= 15


def test_unidentified_peaks_are_written(tmp_path):
    i1, i2, _ = build(tmp_path)
    book, _ = write(tmp_path, i1, i2)
    g1 = book["GC1 raw formatting"]
    assert g1["B6"].value == 5.234


def test_failed_channel_ignores_stale_json(tmp_path):
    def rec(r):
        for (c, d) in list(r):
            if c.startswith(str(tmp_path / "GC1")) and d == "TCD3C":
                r[(c, d)] = (False, "Failed to integrate")
    i1, i2, _ = build(tmp_path, rec)
    book, _ = write(tmp_path, i1, i2)
    g1 = book["GC1 raw formatting"]
    assert g1["I5"].value == "FAILED: Failed to integrate"
    assert g1["I1"].value == "Signal: S005\\TCD3C.ch" and g1["I2"].value == "S005"
    assert isinstance(g1["I3"].value, datetime.datetime)
    for r in g1.iter_rows(min_col=9, max_col=15):
        for c in r:
            assert c.value not in (1.5, 5.234, 100.0, 55.5)
    assert i1[0].channels[1].status == "failed"
    assert i1[0].channels[1].json_path is None


def test_unrecorded_channel_is_not_processed(tmp_path):
    def rec(r):
        r.pop((os.path.join(str(tmp_path / "GC1"), "S005.C"), "TCD3C"))
    i1, i2, _ = build(tmp_path, rec)
    book, _ = write(tmp_path, i1, i2)
    assert book["GC1 raw formatting"]["I5"].value.startswith("NOT PROCESSED")
    assert book["GC1 raw formatting"]["I5"].value == "NOT PROCESSED: no result from this job"


def test_missing_channel_from_worker_skip(tmp_path):
    def rec(r):
        r[(os.path.join(str(tmp_path / "GC2"), "S005.C"), "FID2B")] = (
            False, "Detector FID2B not found in S.C (available: FID1A) - skipping")
    i1, i2, _ = build(tmp_path, rec)
    book, _ = write(tmp_path, i1, i2)
    assert book["GC2 raw formatting"]["I5"].value == "MISSING: channel FID2B not in this injection"


def test_ok_without_result_file_is_failed(tmp_path):
    i1, i2, rec = build(tmp_path)
    os.remove(tmp_path / "GC1" / "S005.C" / "results" / "S005 - TCD3C.json")
    i1 = collect_injections(str(tmp_path / "GC1"), "GC1", GC1_DETS, rec)
    assert (i1[0].channels[1].status, i1[0].channels[1].message) == ("failed", "no result file")


def test_zero_peaks_row(tmp_path):
    i1, i2, _ = build(tmp_path, peaks=[])
    book, _ = write(tmp_path, i1, i2)
    assert book["GC1 raw formatting"]["A5"].value == "NO PEAKS FOUND"


def test_missing_date_writes_marker_on_gc1(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "acquisition_time", lambda d, det=None: None)
    i1, i2, _ = build(tmp_path)
    book, _ = write(tmp_path, i1, i2)
    assert book["GC1 raw formatting"]["A3"].value == "NO ACQUISITION TIME"
    assert book["GC2 raw formatting"]["A3"].value in (None, "")


def test_unpaired_gets_full_placeholder_block(tmp_path):
    i1, i2, _ = build(tmp_path, extra=["S099"])
    i1[1].acquired = T0 + datetime.timedelta(hours=2)
    book, summary = write(tmp_path, i1, i2)
    g2 = book["GC2 raw formatting"]
    assert g2["A8"].value == "Signal: (none)" and g2["I8"].value == "Signal: (none)"
    assert g2["A9"].value == "NO GC2 INJECTION PAIRED WITH S099"
    assert g2["A10"].value in (None, "")
    assert g2["A11"].value == "Peak #" and g2["I11"].value == "Peak #"
    assert g2["A12"].value == "NO INJECTION" and g2["I12"].value == "NO INJECTION"
    pairing = book["Pairing"]
    notes = [pairing.cell(row=r, column=9).value for r in range(2, 4)]
    assert any("no GC2 injection within 5 min" in (n or "") for n in notes)
    assert summary.gc1_unpaired == 1 and summary.gc2_unpaired == 0


def test_column_a_contiguous_in_every_block(tmp_path):
    def rec(r):
        r[(os.path.join(str(tmp_path / "GC1"), "S005.C"), "TCD3C")] = (False, "boom")
    i1, i2, _ = build(tmp_path, rec, extra=["S099"])
    i1[1].acquired = T0 + datetime.timedelta(hours=2)
    book, _ = write(tmp_path, i1, i2)
    for name, ts_blank in (("GC1 raw formatting", False), ("GC2 raw formatting", True)):
        ws = book[name]
        rows = [r for r in range(1, ws.max_row + 1)]
        starts = [r for r in rows if str(ws.cell(r, 1).value or "").startswith("Signal:")]
        assert len(starts) == 2
        for s in starts:
            r = s + 1
            first = r + 1 if ts_blank else r  # GC2's timestamp row is blank
            if ts_blank:
                assert ws.cell(first, 1).value in (None, "")
                r = first + 1
            else:
                r += 2
            assert ws.cell(s + 1, 1).value not in (None, "")
            # sample row .. (timestamp row on GC1) .. headers .. status/peaks
            while r <= ws.max_row and ws.cell(r, 1).value not in (None, ""):
                r += 1
            # the run ended: the rest of the block must not have more column-A cells
            nxt = [x for x in range(r, ws.max_row + 1) if ws.cell(x, 1).value not in (None, "")]
            assert not nxt or nxt[0] in starts, f"gap in column A on {name} after row {r}"
            assert r - 1 > s + 3  # at least one row under the column headers


def test_summary_message():
    s = wb.WorkbookSummary(pairs=1, gc1_unpaired=0, gc2_unpaired=0, channel_problems=0)
    assert s.message() == ("1 pair, 0 GC1 injections unpaired, 0 GC2 injections unpaired, "
                           "0 channel problems — see Pairing sheet")
    s = wb.WorkbookSummary(pairs=48, gc1_unpaired=1, gc2_unpaired=0, channel_problems=2)
    assert s.message() == ("48 pairs, 1 GC1 injection unpaired, 0 GC2 injections unpaired, "
                           "2 channel problems — see Pairing sheet")
