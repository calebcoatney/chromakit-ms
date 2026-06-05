"""Tests for logic/polyarc_quantitation.py — pure quantitation function."""
from pathlib import Path

import pytest

from logic.polyarc_calibration import AnchorPoint, Calibration
from logic.polyarc_library import PolyarcLibrary
from logic.polyarc_quantitation import (
    PeakResult,
    SampleInputs,
    quantitate,
)

DATA = Path(__file__).parent.parent / 'data'
COMPOUNDS_MIN = DATA / 'compounds_minimal.csv'
CAL_MIN = DATA / 'calibration_minimal.csv'
CAL_FPO11 = DATA / 'calibration_fpo11_sample1.csv'

# Production data (golden test reads the full 2703-row library)
REPO_ROOT = Path(__file__).parent.parent.parent
COMPOUNDS_PROD = REPO_ROOT / 'data' / 'polyarc' / 'compounds.csv'


# ─── SampleInputs ────────────────────────────────────────────────────────────

def test_sample_inputs_dilution_factor():
    sample = SampleInputs(sample_mass_g=0.0631, solvent_mass_g=0.727)
    expected = (0.0631 + 0.727) / 0.0631
    assert sample.dilution_factor == pytest.approx(expected)


def test_sample_inputs_zero_mass_raises():
    with pytest.raises(ValueError, match='sample_mass_g'):
        SampleInputs(sample_mass_g=0.0, solvent_mass_g=0.5)


def test_sample_inputs_negative_mass_raises():
    with pytest.raises(ValueError, match='sample_mass_g'):
        SampleInputs(sample_mass_g=-0.01, solvent_mass_g=0.5)


def test_sample_inputs_negative_solvent_raises():
    with pytest.raises(ValueError, match='solvent_mass_g'):
        SampleInputs(sample_mass_g=0.0631, solvent_mass_g=-0.1)


# ─── quantitate() — anchor validation ─────────────────────────────────────────

def test_missing_anchor_raises_before_any_peak_processed():
    """Library has levoglucosan-anchored entries but calibration lacks levoglucosan."""
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)  # has Levoglucosan row
    # Construct an in-memory nonane-only calibration (no fixture file is
    # nonane-only because CAL_FPO11 mirrors the production library, which
    # requires both anchors).
    nonane_only = Calibration({
        'nonane': AnchorPoint(
            name='nonane', cas='000111-84-2', C=9, MW=128.259,
            known_wt_pct=0.0440433, area=28848616.5,
            run_date='2026-03-25', instrument_id='test-rig',
            notes='nonane-only fixture for fail-fast test',
        ),
    })
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    with pytest.raises(ValueError, match='levoglucosan'):
        quantitate(peaks=[], sample=sample, library=lib, calibration=nonane_only)


def test_no_anchor_validation_when_library_only_uses_present_anchors():
    """All compounds anchor to nonane; calibration has nonane → should not raise."""
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)  # has both anchors
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    # Should not raise even with empty peaks
    results = quantitate(peaks=[], sample=sample, library=lib, calibration=cal)
    assert results == []


# ─── quantitate() — single-peak behavior ──────────────────────────────────────

def test_unmatched_peak_returns_matched_false():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [{'compound_id': 'Bogus', 'casno': '999999-99-9', 'area': 1000.0,
              'retention_time': 10.0, 'Qual': 0.5}]
    results = quantitate(peaks, sample, lib, cal)
    assert len(results) == 1
    r = results[0]
    assert r.matched is False
    assert r.record is None
    assert r.RF is None
    assert r.wt_pct is None
    assert r.mol_C is None
    assert r.mol is None
    assert r.mass_mg is None
    # Passthrough fields preserved
    assert r.compound_id == 'Bogus'
    assert r.retention_time == 10.0


def test_zero_area_peak_returns_zero_wt_pct_not_none():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 0.0, 'retention_time': 13.5, 'Qual': 0.95}]
    results = quantitate(peaks, sample, lib, cal)
    r = results[0]
    assert r.matched is True
    assert r.wt_pct == 0.0
    assert r.mass_mg == 0.0


def test_negative_area_peak_clamps_to_zero_wt_pct():
    """Per spec §5: a peak with zero or negative area is matched but
    contributes nothing. Chromakit's below-baseline peaks (is_negative=True)
    can emit area < 0; clamping prevents them from silently reducing
    group-rollup totals."""
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': -500.0, 'retention_time': 13.5, 'Qual': 0.95}]
    results = quantitate(peaks, sample, lib, cal)
    r = results[0]
    assert r.matched is True
    assert r.wt_pct == 0.0
    assert r.mass_mg == 0.0
    assert r.mol == 0.0
    assert r.mol_C == 0.0
    # The raw input area is preserved for diagnostics
    assert r.area == -500.0


def test_input_peak_extra_fields_preserved_in_extra_dict():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 100.0, 'retention_time': 13.5, 'Qual': 0.95,
              'start_time': 13.3, 'end_time': 13.7, 'is_shoulder': False,
              'peak_number': 42}]
    results = quantitate(peaks, sample, lib, cal)
    r = results[0]
    assert r.extra['start_time'] == 13.3
    assert r.extra['end_time'] == 13.7
    assert r.extra['is_shoulder'] is False
    assert r.extra['peak_number'] == 42


def test_results_preserve_input_order():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [
        {'compound_id': 'Acetic acid', 'casno': '000064-19-7',
         'area': 100.0, 'retention_time': 13.5, 'Qual': 0.95},
        {'compound_id': 'Toluene', 'casno': '000108-88-3',
         'area': 200.0, 'retention_time': 16.1, 'Qual': 0.99},
        {'compound_id': 'Phenol', 'casno': '000108-95-2',
         'area': 300.0, 'retention_time': 40.0, 'Qual': 0.97},
    ]
    results = quantitate(peaks, sample, lib, cal)
    assert [r.compound_id for r in results] == ['Acetic acid', 'Toluene', 'Phenol']


# ─── quantitate() — anchor routing ────────────────────────────────────────────

def test_nonane_anchored_compound_uses_nonane_anchor():
    """Acetic acid (nonane-anchored) RF must use nonane calibration values."""
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 1.0, 'retention_time': 13.5, 'Qual': 0.95}]
    results = quantitate(peaks, sample, lib, cal)
    nonane = cal['nonane']
    rec = lib.lookup(casno='000064-19-7', name=None)
    # Verify RF formula uses nonane constants
    expected_RF = (nonane.known_wt_pct / nonane.area) * (nonane.C / rec.C) * (rec.MW / nonane.MW)
    assert results[0].RF == pytest.approx(expected_RF, rel=1e-12)


def test_sugar_compound_uses_levoglucosan_anchor():
    """Levoglucosan itself (levoglucosan-anchored) RF must use levoglucosan calibration."""
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.1, solvent_mass_g=0.9)
    peaks = [{'compound_id': 'Levoglucosan', 'casno': '000498-07-7',
              'area': 1.0, 'retention_time': 67.5, 'Qual': 0.99}]
    results = quantitate(peaks, sample, lib, cal)
    lev = cal['levoglucosan']
    rec = lib.lookup(casno='000498-07-7', name=None)
    expected_RF = (lev.known_wt_pct / lev.area) * (lev.C / rec.C) * (rec.MW / lev.MW)
    assert results[0].RF == pytest.approx(expected_RF, rel=1e-12)


# ─── quantitate() — derived field consistency ─────────────────────────────────

def test_mass_mg_consistent_with_wt_pct():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.0631, solvent_mass_g=0.727)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 1.0e6, 'retention_time': 13.5, 'Qual': 0.95}]
    results = quantitate(peaks, sample, lib, cal)
    r = results[0]
    # mass_mg = wt_pct / 100 * sample_mass_g * 1000
    expected_mass_mg = r.wt_pct / 100.0 * sample.sample_mass_g * 1000.0
    assert r.mass_mg == pytest.approx(expected_mass_mg, rel=1e-12)


def test_mol_consistent_with_mass_and_mw():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.0631, solvent_mass_g=0.727)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 1.0e6, 'retention_time': 13.5, 'Qual': 0.95}]
    results = quantitate(peaks, sample, lib, cal)
    r = results[0]
    # mol = mass_g / MW = (mass_mg / 1000) / MW
    expected_mol = r.mass_mg / 1000.0 / r.record.MW
    assert r.mol == pytest.approx(expected_mol, rel=1e-12)


def test_mol_C_equals_mol_times_carbon_count():
    lib = PolyarcLibrary.from_csv(COMPOUNDS_MIN)
    cal = Calibration.from_csv(CAL_MIN)
    sample = SampleInputs(sample_mass_g=0.0631, solvent_mass_g=0.727)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 1.0e6, 'retention_time': 13.5, 'Qual': 0.95}]
    results = quantitate(peaks, sample, lib, cal)
    r = results[0]
    assert r.mol_C == pytest.approx(r.mol * r.record.C, rel=1e-12)


# ─── Golden test: reproduce Kelly's Sample 1 row 12 ───────────────────────────

def test_acetic_acid_matches_kelly_to_5_sig_figs():
    """Pinned regression test against Kelly's Sample 1 Acetic acid row.

    Input values come from `GCMS-Polyarc_Alder FPO-11_12_March 2026.xlsx`,
    Sample 1 sheet, row 12, and the FPO-11 calibration anchors. Expected
    wt_pct was verified by re-evaluating the spreadsheet's `data_only=True`
    formulas (E12 * F12 * D$4 = 20810783 * 3.216674380337399e-09 * 12.521394611727416).

    Tolerance is `rel=1e-5` rather than `rel=1e-6` because PolyarcLibrary
    recomputes MW from atom counts (60.0518 for acetic acid) while Kelly's
    spreadsheet stores MW rounded to 3 decimals (60.052). This 3.3e-6
    rounding propagates linearly through the wt% formula. The recomputed
    MW is the single source of truth per design §3.5; the spreadsheet's
    MW column is informational.
    """
    lib = PolyarcLibrary.from_csv(COMPOUNDS_PROD)
    cal = Calibration.from_csv(CAL_FPO11)
    sample = SampleInputs(sample_mass_g=0.0631, solvent_mass_g=0.727)
    peaks = [{'compound_id': 'Acetic acid', 'casno': '000064-19-7',
              'area': 20810783.0, 'retention_time': 13.592, 'Qual': 91.0}]
    results = quantitate(peaks, sample, lib, cal)
    assert results[0].matched
    assert results[0].wt_pct == pytest.approx(0.838201, rel=1e-5)
