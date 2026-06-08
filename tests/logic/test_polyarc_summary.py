"""Unit tests for logic/polyarc_summary.py."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from logic.polyarc_calibration import Calibration
from logic.polyarc_library import PolyarcLibrary
from logic.polyarc_quantitation import SampleInputs
from logic.polyarc_summary import BatchSummary, UnmatchedPeak, summarize_batch


FIXTURE_DIR = Path(__file__).parent.parent / 'data'


def _load_minimal_library() -> PolyarcLibrary:
    return PolyarcLibrary.from_csv(FIXTURE_DIR / 'compounds_minimal.csv')


def _load_minimal_calibration() -> Calibration:
    return Calibration.from_csv(FIXTURE_DIR / 'calibration_minimal.csv')


def test_unmatched_peak_is_frozen():
    peak = UnmatchedPeak(
        sample_id='S1', peak_number=3, retention_time=12.5,
        compound_id='Foo', casno='000123-45-6', area=1000.0,
        reason='no_cas_match',
    )
    with pytest.raises(FrozenInstanceError):
        peak.sample_id = 'S2'  # type: ignore[misc]


def test_batch_summary_is_frozen():
    summary = BatchSummary(
        per_sample_peaks={}, per_sample_group_totals={},
        per_sample_inputs={}, unmatched=[], match_stats={},
    )
    with pytest.raises(FrozenInstanceError):
        summary.unmatched = []  # type: ignore[misc]


def test_unmatched_peak_valid_reasons():
    # Document the allowed reasons; not strictly validated at runtime
    # but tests should fail loudly if we typo one anywhere.
    for reason in ('no_cas_match', 'sentinel_cas', 'malformed_cas', 'no_record'):
        UnmatchedPeak(
            sample_id='S1', peak_number=1, retention_time=0.0,
            compound_id='', casno='', area=0.0, reason=reason,
        )


def test_summarize_batch_classifies_unmatched_reasons():
    library = _load_minimal_library()
    calibration = _load_minimal_calibration()
    json_paths = {'S1': FIXTURE_DIR / 'summary_minimal_sample1.json'}
    weights = {'S1': SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)}

    summary = summarize_batch(json_paths, weights, library, calibration)

    reasons = {(u.peak_number, u.reason) for u in summary.unmatched if u.sample_id == 'S1'}
    assert (4, 'no_cas_match') in reasons
    assert (5, 'sentinel_cas') in reasons
    assert (6, 'malformed_cas') in reasons
    # Peaks 1, 2, 3 are matched — should NOT appear in unmatched.
    assert not any(u.peak_number in (1, 2, 3) for u in summary.unmatched if u.sample_id == 'S1')


def test_summarize_batch_returns_peakresults_for_matched():
    library = _load_minimal_library()
    calibration = _load_minimal_calibration()
    json_paths = {'S1': FIXTURE_DIR / 'summary_minimal_sample1.json'}
    weights = {'S1': SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)}

    summary = summarize_batch(json_paths, weights, library, calibration)

    peaks = summary.per_sample_peaks['S1']
    assert len(peaks) == 6  # all peaks, matched or not
    matched = [p for p in peaks if p.matched]
    assert len(matched) == 3
    assert {p.casno for p in matched} == {'000064-19-7', '000108-95-2', '000498-07-7'}


def test_summarize_batch_group_rollup_sums_wt_pct():
    library = _load_minimal_library()
    calibration = _load_minimal_calibration()
    json_paths = {'S2': FIXTURE_DIR / 'summary_minimal_sample2.json'}
    weights = {'S2': SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)}

    summary = summarize_batch(json_paths, weights, library, calibration)

    totals = summary.per_sample_group_totals['S2']
    # Sample 2 has Nonane (Alkane/n-Alkane), Toluene (Aromatic/BTX),
    # Phenol, 2-methyl- (Oxygenate/Phenols/Methylphenol)
    assert 'Alkane' in totals
    assert 'n-Alkane' in totals
    assert 'Aromatic' in totals
    assert 'BTX' in totals
    assert 'Oxygenate' in totals
    assert 'Phenols' in totals
    assert 'Methylphenol' in totals
    assert 'Total Mass % Accounted' in totals
    # Total should equal sum of the three matched peaks' wt_pct.
    matched_wt = sum(p.wt_pct for p in summary.per_sample_peaks['S2'] if p.matched)
    assert totals['Total Mass % Accounted'] == pytest.approx(matched_wt, rel=1e-9)


def test_summarize_batch_match_stats():
    library = _load_minimal_library()
    calibration = _load_minimal_calibration()
    json_paths = {'S1': FIXTURE_DIR / 'summary_minimal_sample1.json'}
    weights = {'S1': SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)}

    summary = summarize_batch(json_paths, weights, library, calibration)

    stats = summary.match_stats['S1']
    assert stats['count_total'] == 6.0
    assert stats['count_matched'] == 3.0
    # Matched peaks: 20M + 10M + 5M = 35M out of 35M + 1M + 0.8M + 0.5M = 37.3M
    expected_area_pct = (20000000 + 10000000 + 5000000) / (20000000 + 10000000 + 5000000 + 1000000 + 800000 + 500000) * 100
    assert stats['area_matched_pct'] == pytest.approx(expected_area_pct, rel=1e-6)


def test_summarize_batch_raises_on_id_mismatch():
    library = _load_minimal_library()
    calibration = _load_minimal_calibration()
    json_paths = {'S1': FIXTURE_DIR / 'summary_minimal_sample1.json'}
    weights = {'S2': SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)}

    with pytest.raises(KeyError, match=r"S1.*S2|S2.*S1"):
        summarize_batch(json_paths, weights, library, calibration)
