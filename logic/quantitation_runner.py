"""Pure orchestration for Polyarc + Internal Standard quantitation.

Extracted from ui/app.py::_perform_quantitation (lines 3319-3478) so the
GUI and the API can call identical quantitation logic.

The math primitives in logic/quantitation.py remain unchanged — this
module is the loop that wires them together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from logic.quantitation import QuantitationCalculator


@dataclass
class InternalStandardSpec:
    """Internal standard inputs (the user fills these via the GUI form)."""
    compound_name: str
    volume_uL: float
    density_g_mL: float
    molecular_weight: float
    formula: str


@dataclass
class SampleSpec:
    """Sample inputs (optional — without these, no carbon balance)."""
    volume_uL: Optional[float] = None
    density_g_mL: Optional[float] = None


@dataclass
class CompoundMetadata:
    """Per-compound metadata as resolved by the caller's lookup function."""
    formula: Optional[str]
    molecular_weight: Optional[float]


@dataclass
class QuantitationSummary:
    """Outcome of a quantitation run."""
    internal_standard_peak_index: int = -1
    response_factor: Optional[float] = None
    peaks_quantitated: int = 0  # excludes IS
    total_analyte_mass_mg: Optional[float] = None
    sample_mass_mg: Optional[float] = None
    carbon_balance_percent: Optional[float] = None
    warnings: list = field(default_factory=list)


def lookup_compound_metadata(ms_toolkit, compound_name: str) -> CompoundMetadata:
    """Resolve compound formula + MW from an MSToolkit's loaded library.

    Returns CompoundMetadata(None, None) if the compound isn't in the library
    or the library isn't loaded.
    """
    try:
        if hasattr(ms_toolkit, 'library') and compound_name in ms_toolkit.library:
            compound = ms_toolkit.library[compound_name]
            formula = getattr(compound, 'formula', None)
            mw = getattr(compound, 'mw', None)
            return CompoundMetadata(formula=formula, molecular_weight=mw)
    except Exception:
        pass
    return CompoundMetadata(formula=None, molecular_weight=None)


def _is_match(compound_id: Optional[str], target: str) -> bool:
    """Case-insensitive, whitespace-trimmed compound-name match."""
    if not compound_id:
        return False
    return compound_id.strip().lower() == target.strip().lower()


def _peak_compound_id(peak) -> Optional[str]:
    """Return the peak's compound identity — prefer Compound_ID over compound_id."""
    return getattr(peak, 'Compound_ID', None) or getattr(peak, 'compound_id', None)


def run_quantitation(
    peaks: list,
    internal_standard: InternalStandardSpec,
    sample: SampleSpec,
    compound_lookup: Callable[[str], CompoundMetadata],
) -> QuantitationSummary:
    """Run Polyarc + IS quantitation on a list of peaks, mutating them in place.

    Side effects on each peak (when quantitated):
        mol_C, num_carbons, mol, mass_mg, mol_C_percent, mol_percent, wt_percent

    IS peak gets mol_C / num_carbons / mol / mass_mg set, but the three
    percentage fields stay None (the IS is excluded from composition by design).

    Args:
        peaks: List of ChromatographicPeak objects (mutated in place).
        internal_standard: IS inputs (compound name, volume, density, MW, formula).
        sample: Sample inputs; if volume/density are None, no carbon balance.
        compound_lookup: Callable(compound_name) -> CompoundMetadata. The API
            endpoint passes partial(lookup_compound_metadata, ms_toolkit);
            the GUI passes its own lookup helper; tests pass a dict-backed one.

    Returns:
        QuantitationSummary with IS peak index, RF, counts, totals, warnings.
    """
    summary = QuantitationSummary()
    calc = QuantitationCalculator()

    # 1. Find IS peak by compound name
    is_index = -1
    is_peak = None
    for i, peak in enumerate(peaks):
        if _is_match(_peak_compound_id(peak), internal_standard.compound_name):
            is_index = i
            is_peak = peak
            break

    if is_peak is None:
        summary.warnings.append(
            f"Internal standard '{internal_standard.compound_name}' "
            "not found in peaks; quantitation aborted."
        )
        return summary

    summary.internal_standard_peak_index = is_index

    # 2. Compute mol C of IS
    mol_C_IS = calc.calculate_mol_C_internal_standard(
        volume_uL=internal_standard.volume_uL,
        density_g_mL=internal_standard.density_g_mL,
        molecular_weight=internal_standard.molecular_weight,
        formula=internal_standard.formula,
    )
    if not mol_C_IS:
        summary.warnings.append(
            "Could not compute mol C of internal standard "
            "(check volume / density / MW / formula)."
        )
        return summary

    # 3. Compute response factor
    rf = calc.calculate_response_factor(is_peak.area, mol_C_IS)
    if not rf:
        summary.warnings.append("Response factor computation failed.")
        return summary
    summary.response_factor = rf

    # 4. Quantitate each peak with a known compound_id
    analyte_results = []  # parallel list of quant dicts for non-IS peaks
    analyte_peaks = []    # parallel list of peak refs

    for i, peak in enumerate(peaks):
        compound_id = _peak_compound_id(peak)
        if not compound_id:
            continue

        is_internal = _is_match(compound_id, internal_standard.compound_name)

        meta = compound_lookup(compound_id)
        if not meta.formula or not meta.molecular_weight:
            summary.warnings.append(
                f"Skipping {compound_id}: missing formula or molecular weight."
            )
            continue

        result = calc.quantitate_peak(
            peak_area=peak.area,
            response_factor=rf,
            formula=meta.formula,
            molecular_weight=meta.molecular_weight,
        )
        if not result:
            summary.warnings.append(f"Skipping {compound_id}: calculator returned None.")
            continue

        peak.mol_C = result['mol_C']
        peak.num_carbons = result['num_carbons']
        peak.mol = result['mol']
        peak.mass_mg = result['mass_mg']

        if not is_internal:
            analyte_results.append(result)
            analyte_peaks.append(peak)
            summary.peaks_quantitated += 1

    # 5. Composition percentages (excluding IS)
    calc.calculate_composition(analyte_results)
    for peak, result in zip(analyte_peaks, analyte_results):
        peak.mol_C_percent = result.get('mol_C_percent')
        peak.mol_percent = result.get('mol_percent')
        peak.wt_percent = result.get('wt_percent')

    # IS-matching peaks stay None for percentages (excluded from composition)
    for peak in peaks:
        if _is_match(_peak_compound_id(peak), internal_standard.compound_name):
            peak.mol_C_percent = None
            peak.mol_percent = None
            peak.wt_percent = None

    # 6. Carbon balance (when sample volume + density given)
    if sample.volume_uL and sample.density_g_mL:
        sample_mass = calc.calculate_sample_mass(sample.volume_uL, sample.density_g_mL)
        total_analyte_mass = sum(r['mass_mg'] for r in analyte_results)
        c_balance = calc.calculate_carbon_balance(total_analyte_mass, sample_mass)
        summary.sample_mass_mg = sample_mass
        summary.total_analyte_mass_mg = total_analyte_mass
        summary.carbon_balance_percent = c_balance

    return summary
