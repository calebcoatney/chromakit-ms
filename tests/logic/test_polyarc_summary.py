"""Unit tests for logic/polyarc_summary.py."""
from __future__ import annotations

from pathlib import Path

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
    try:
        peak.sample_id = 'S2'  # type: ignore[misc]
    except Exception as e:
        assert 'frozen' in str(e).lower() or 'attribute' in str(e).lower() or 'cannot assign' in str(e).lower()
    else:
        assert False, 'UnmatchedPeak must be frozen'


def test_batch_summary_is_frozen():
    summary = BatchSummary(
        per_sample_peaks={}, per_sample_group_totals={},
        per_sample_inputs={}, unmatched=[], match_stats={},
    )
    try:
        summary.unmatched = []  # type: ignore[misc]
    except Exception as e:
        assert 'frozen' in str(e).lower() or 'attribute' in str(e).lower() or 'cannot assign' in str(e).lower()
    else:
        assert False, 'BatchSummary must be frozen'


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
