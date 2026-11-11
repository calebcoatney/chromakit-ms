from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from logic import gc12_workbook as gw
from logic.gc12_workbook import Injection, acquisition_time, injection_counter, pair_injections

T0 = datetime(2026, 5, 21, 14, 36)


def inj(instrument, acquired, counter, sample=None):
    return Injection(
        c_path=f"/x/{instrument}.C", instrument=instrument, d_name=None,
        sample=sample or f"{instrument}-{counter}", acquired=acquired,
        counter=counter, channels=[],
    )


def test_injection_counter():
    assert injection_counter("260521 GC1 R1 run 319 outlet 005.D") == 5
    assert injection_counter("vs01260521 GC2 R1 run 319 outlet 005") == 5
    assert injection_counter("no number here") is None


def test_acquisition_time_reads_datafile_date(monkeypatch):
    fake = SimpleNamespace(
        datafiles=[SimpleNamespace(name="FID1A.ch", metadata={"date": "21 May 26  2:36 pm"})],
        metadata={},
    )
    monkeypatch.setattr(gw.rb, "read", lambda p: fake)
    assert acquisition_time("x.D", "FID1A") == datetime(2026, 5, 21, 14, 36)


def test_acquisition_time_prefers_matching_detector_then_dir_metadata(monkeypatch):
    fake = SimpleNamespace(
        datafiles=[
            SimpleNamespace(name="FID2B.ch", metadata={"date": "21 May 26  3:00 pm"}),
            SimpleNamespace(name="FID1A.ch", metadata={"date": "21 May 26  2:36 pm"}),
        ],
        metadata={"date": "1 Jan 20  1:00 am"},
    )
    monkeypatch.setattr(gw.rb, "read", lambda p: fake)
    assert acquisition_time("x.D", "FID1A") == datetime(2026, 5, 21, 14, 36)
    fake.datafiles = []
    monkeypatch.setattr(gw.rb, "read", lambda p: fake)
    assert acquisition_time("x.D") == datetime(2020, 1, 1, 1, 0)


def test_acquisition_time_none_when_no_date(monkeypatch):
    fake = SimpleNamespace(datafiles=[SimpleNamespace(name="FID1A.ch", metadata={})], metadata={})
    monkeypatch.setattr(gw.rb, "read", lambda p: fake)
    assert acquisition_time("x.D", "FID1A") is None

    def boom(p):
        raise OSError("nope")

    monkeypatch.setattr(gw.rb, "read", boom)
    assert acquisition_time("x.D") is None


def test_pairs_by_nearest_time_within_window():
    t2 = T0 + timedelta(minutes=44)
    gc1 = [inj("GC1", T0, 5), inj("GC1", t2, 6)]
    gc2 = [inj("GC2", T0 + timedelta(seconds=9), 5), inj("GC2", t2, 6)]
    pairs = pair_injections(gc1, gc2)
    assert len(pairs) == 2
    assert pairs[0].dt_s == 9.0
    assert pairs[0].counter_match is True
    assert pairs[0].notes == []


def test_gap_over_window_is_unpaired():
    pairs = pair_injections([inj("GC1", T0, 5)], [inj("GC2", T0 + timedelta(minutes=6), 5)])
    assert len(pairs) == 2
    single1 = next(p for p in pairs if p.gc1)
    single2 = next(p for p in pairs if p.gc2)
    assert single1.gc2 is None and "no GC2 injection within 5 min" in single1.notes
    assert single2.gc1 is None and "no GC1 injection within 5 min" in single2.notes


def test_nearest_wins_when_two_candidates():
    a = inj("GC1", T0, 5)
    b = inj("GC1", T0 + timedelta(minutes=2), 6)
    g = inj("GC2", T0 + timedelta(minutes=1, seconds=50), 6)
    pairs = pair_injections([a, b], [g])
    paired = next(p for p in pairs if p.gc2)
    assert paired.gc1 is b
    single = next(p for p in pairs if p.gc2 is None)
    assert single.gc1 is a


def test_counter_mismatch_is_warning_not_refusal():
    pairs = pair_injections([inj("GC1", T0, 5)], [inj("GC2", T0, 6)])
    assert len(pairs) == 1
    assert pairs[0].counter_match is False
    assert "injection counters differ (5 vs 6)" in pairs[0].notes


def test_untimed_injection_pairs_by_counter():
    pairs = pair_injections([inj("GC1", T0, 5)], [inj("GC2", None, 5)])
    assert len(pairs) == 1
    assert pairs[0].dt_s is None
    assert "paired by injection counter only (no acquisition time)" in pairs[0].notes


def test_untimed_with_ambiguous_counter_stays_single():
    pairs = pair_injections([inj("GC1", None, 5), inj("GC1", None, 5)], [inj("GC2", None, 5)])
    assert all(p.gc1 is None or p.gc2 is None for p in pairs)
    assert len(pairs) == 3


def test_pairs_sorted_by_time():
    late = inj("GC1", T0 + timedelta(hours=1), 7)
    early = inj("GC2", T0, 5)
    untimed = inj("GC1", None, 99)
    pairs = pair_injections([untimed, late], [early])
    keys = [(p.gc1 or p.gc2) for p in pairs]
    assert keys == [early, late, untimed]
