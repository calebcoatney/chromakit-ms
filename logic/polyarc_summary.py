"""
Polyarc batch summary: roll up per-sample peak results into Kelly-style group totals.

Pure Python (stdlib + openpyxl). No imports from `ui/` or `api/`.

See docs/superpowers/specs/2026-06-08-polyarc-batch-summary-and-validation-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from logic.polyarc_quantitation import PeakResult, SampleInputs


@dataclass(frozen=True)
class UnmatchedPeak:
    """A peak that could not be quantitated.

    reason ∈ {'no_cas_match', 'sentinel_cas', 'malformed_cas', 'no_record'}.
    """
    sample_id: str
    peak_number: int
    retention_time: float
    compound_id: str
    casno: str
    area: float
    reason: str


@dataclass(frozen=True)
class BatchSummary:
    """Aggregated quantitation results for a batch of samples."""
    per_sample_peaks: dict[str, list[PeakResult]]
    per_sample_group_totals: dict[str, dict[str, float]]
    per_sample_inputs: dict[str, SampleInputs]
    unmatched: list[UnmatchedPeak]
    match_stats: dict[str, dict[str, float]]
