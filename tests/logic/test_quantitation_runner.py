"""Tests for logic/quantitation_runner.run_quantitation.

The runner orchestrates QuantitationCalculator: find IS peak, compute
response factor, quantitate each peak, compute composition + carbon balance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from logic.integration import ChromatographicPeak
from logic.quantitation_runner import (
    InternalStandardSpec,
    SampleSpec,
    CompoundMetadata,
    QuantitationSummary,
    run_quantitation,
)


def _make_peak(compound_id: str, area: float, rt: float = 1.0,
                peak_number: int = 1) -> ChromatographicPeak:
    peak = ChromatographicPeak(
        compound_id=compound_id,
        peak_number=peak_number,
        retention_time=rt,
        integrator="BB",
        width=0.05,
        area=area,
        start_time=rt - 0.025,
        end_time=rt + 0.025,
    )
    peak.Compound_ID = compound_id
    return peak


# Compound metadata for the test compounds (formula, MW in g/mol)
_LOOKUP_TABLE = {
    "Decane": CompoundMetadata(formula="C10H22", molecular_weight=142.28),
    "Hexane": CompoundMetadata(formula="C6H14", molecular_weight=86.18),
    "Benzene": CompoundMetadata(formula="C6H6", molecular_weight=78.11),
    "Unknown": CompoundMetadata(formula=None, molecular_weight=None),
}


def _lookup(compound_name: str) -> CompoundMetadata:
    return _LOOKUP_TABLE.get(compound_name, CompoundMetadata(None, None))


def _is_spec() -> InternalStandardSpec:
    """Decane as IS: 1 µL × 0.73 g/mL × 1 mol / 142.28 g × 10 carbons ≈ 5.13e-5 mol C."""
    return InternalStandardSpec(
        compound_name="Decane",
        volume_uL=1.0,
        density_g_mL=0.73,
        molecular_weight=142.28,
        formula="C10H22",
    )


def test_basic_quantitation_finds_is_and_computes_rf():
    """IS is found; response factor is is.area / mol_C_IS; analytes get mol_C."""
    is_peak = _make_peak("Decane", area=1.0e6, rt=5.0, peak_number=2)
    benzene = _make_peak("Benzene", area=5.0e5, rt=3.0, peak_number=1)
    peaks = [benzene, is_peak]

    summary = run_quantitation(
        peaks=peaks,
        internal_standard=_is_spec(),
        sample=SampleSpec(),
        compound_lookup=_lookup,
    )

    assert summary.internal_standard_peak_index == 1
    assert summary.response_factor is not None
    assert summary.response_factor > 0
    # IS mol_C ~ 5.13e-5; RF ~ 1e6 / 5.13e-5 ~ 1.95e10
    assert summary.response_factor == pytest.approx(1e6 / 5.13e-5, rel=0.01)
    # Benzene was quantitated (1 analyte + IS counted separately)
    assert summary.peaks_quantitated == 1
    assert benzene.mol_C is not None and benzene.mol_C > 0
    assert benzene.mol_C_percent is not None  # composition assigned for analyte
    assert is_peak.mol_C is not None  # IS also gets mol_C set
    assert is_peak.mol_C_percent is None  # but IS gets None for percentages


def test_is_not_found_returns_minus_one_and_warning():
    """When IS compound isn't in peaks, IS-index = -1 and a warning is emitted."""
    benzene = _make_peak("Benzene", area=5.0e5)
    summary = run_quantitation(
        peaks=[benzene],
        internal_standard=_is_spec(),
        sample=SampleSpec(),
        compound_lookup=_lookup,
    )
    assert summary.internal_standard_peak_index == -1
    assert summary.response_factor is None
    assert summary.peaks_quantitated == 0
    assert any("Decane" in w or "internal standard" in w.lower()
               for w in summary.warnings)
    # Benzene was not quantitated because no RF could be computed
    assert benzene.mol_C is None


def test_compound_without_metadata_is_skipped_with_warning():
    """A peak whose compound has no formula/MW in the library is skipped."""
    is_peak = _make_peak("Decane", area=1.0e6, rt=5.0, peak_number=2)
    unknown = _make_peak("Unknown", area=2.0e5, rt=3.0, peak_number=1)
    peaks = [unknown, is_peak]

    summary = run_quantitation(
        peaks=peaks,
        internal_standard=_is_spec(),
        sample=SampleSpec(),
        compound_lookup=_lookup,
    )
    # Unknown skipped; IS quantitated (and not counted in peaks_quantitated)
    assert summary.peaks_quantitated == 0
    assert unknown.mol_C is None
    assert any("Unknown" in w for w in summary.warnings)


def test_composition_percentages_sum_to_one_hundred_excluding_is():
    """mol_C_percent across analytes (NOT including IS) sums to ~100%."""
    is_peak = _make_peak("Decane", area=1.0e6, rt=5.0, peak_number=3)
    hexane = _make_peak("Hexane", area=3.0e5, rt=2.5, peak_number=1)
    benzene = _make_peak("Benzene", area=5.0e5, rt=3.0, peak_number=2)
    peaks = [hexane, benzene, is_peak]

    summary = run_quantitation(
        peaks=peaks,
        internal_standard=_is_spec(),
        sample=SampleSpec(),
        compound_lookup=_lookup,
    )
    analyte_sum = sum(
        p.mol_C_percent for p in [hexane, benzene] if p.mol_C_percent is not None
    )
    assert analyte_sum == pytest.approx(100.0, abs=0.01)
    assert is_peak.mol_C_percent is None


def test_carbon_balance_computed_when_sample_provided():
    """When SampleSpec has volume + density, carbon balance is computed."""
    is_peak = _make_peak("Decane", area=1.0e6, rt=5.0, peak_number=2)
    benzene = _make_peak("Benzene", area=5.0e5, rt=3.0, peak_number=1)
    peaks = [benzene, is_peak]

    summary = run_quantitation(
        peaks=peaks,
        internal_standard=_is_spec(),
        sample=SampleSpec(volume_uL=10.0, density_g_mL=0.88),
        compound_lookup=_lookup,
    )
    assert summary.sample_mass_mg == pytest.approx(10.0 * 1e-3 * 0.88 * 1000, rel=0.001)
    assert summary.carbon_balance_percent is not None
    assert summary.total_analyte_mass_mg is not None


def test_carbon_balance_none_when_no_sample_info():
    """Without sample volume/density, carbon_balance_percent is None."""
    is_peak = _make_peak("Decane", area=1.0e6, rt=5.0, peak_number=2)
    benzene = _make_peak("Benzene", area=5.0e5, rt=3.0, peak_number=1)
    peaks = [benzene, is_peak]

    summary = run_quantitation(
        peaks=peaks,
        internal_standard=_is_spec(),
        sample=SampleSpec(),
        compound_lookup=_lookup,
    )
    assert summary.sample_mass_mg is None
    assert summary.carbon_balance_percent is None


def test_is_match_is_case_insensitive_and_trims():
    """IS compound name matching ignores case and surrounding whitespace."""
    is_peak = _make_peak("decane", area=1.0e6)  # lowercase
    summary = run_quantitation(
        peaks=[is_peak],
        internal_standard=InternalStandardSpec(
            compound_name="  DECANE  ",  # whitespace + uppercase
            volume_uL=1.0,
            density_g_mL=0.73,
            molecular_weight=142.28,
            formula="C10H22",
        ),
        compound_lookup=_lookup,
        sample=SampleSpec(),
    )
    assert summary.internal_standard_peak_index == 0
