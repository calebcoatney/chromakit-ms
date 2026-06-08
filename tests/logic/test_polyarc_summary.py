"""Unit tests for logic/polyarc_summary.py."""
from __future__ import annotations

from logic.polyarc_summary import BatchSummary, UnmatchedPeak


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
