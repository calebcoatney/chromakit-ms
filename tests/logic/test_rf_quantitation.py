"""Tests for logic/rf_quantitation.py — RF-table external-standard quant."""
import pytest

from logic.rf_quantitation import quantitate_rf, RFQuantSummary
from logic.method import RFTableEntry
from logic.integration import ChromatographicPeak


def _peak(compound_id, area, peak_number=1, rt=1.0):
    return ChromatographicPeak(
        compound_id=compound_id, peak_number=peak_number, retention_time=rt,
        integrator="BB", width=0.1, area=area, start_time=rt - 0.05, end_time=rt + 0.05,
    )


def _rf():
    return [
        RFTableEntry(compound="Hydrogen", response_factor=100.0),
        RFTableEntry(compound="Carbon monoxide", response_factor=200.0),
    ]


def test_normalized_mol_percent():
    # H2: 1000/100 = 10 raw; CO: 2000/200 = 10 raw; total 20 → 50/50
    peaks = [_peak("Hydrogen", 1000.0, 1), _peak("Carbon monoxide", 2000.0, 2)]
    summary = quantitate_rf(peaks, _rf(), normalize=True)
    assert peaks[0].mol_percent == pytest.approx(50.0)
    assert peaks[1].mol_percent == pytest.approx(50.0)
    assert peaks[0].raw_amount == pytest.approx(10.0)
    assert summary.peaks_quantitated == 2
    assert summary.normalized is True
    assert summary.total_raw_amount == pytest.approx(20.0)


def test_unnormalized_returns_raw_amounts():
    peaks = [_peak("Hydrogen", 1000.0, 1)]
    summary = quantitate_rf(peaks, _rf(), normalize=False)
    assert peaks[0].raw_amount == pytest.approx(10.0)
    assert peaks[0].mol_percent is None      # not computed when not normalizing
    assert summary.normalized is False


def test_compound_not_in_rf_table_is_flagged():
    peaks = [_peak("Argon", 500.0, 1)]     # not in RF table
    summary = quantitate_rf(peaks, _rf(), normalize=True)
    assert peaks[0].mol_percent is None
    assert len(summary.skipped_no_rf) == 1
    assert "Argon" in summary.skipped_no_rf[0]


def test_unassigned_peak_is_flagged():
    peaks = [_peak("Unknown (1.234)", 500.0, 1)]
    summary = quantitate_rf(peaks, _rf(), normalize=True)
    assert len(summary.skipped_unassigned) == 1


def test_empty_inputs_clean_summary():
    summary = quantitate_rf([], _rf(), normalize=True)
    assert summary.peaks_quantitated == 0
    assert summary.peaks_total == 0


def test_zero_total_adds_warning_no_crash():
    # area 0 → raw 0 → total 0; normalization must not divide by zero
    peaks = [_peak("Hydrogen", 0.0, 1)]
    summary = quantitate_rf(peaks, _rf(), normalize=True)
    assert peaks[0].mol_percent == pytest.approx(0.0)
    assert any("zero" in w.lower() for w in summary.warnings)
