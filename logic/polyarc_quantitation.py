"""
Polyarc library-broadcast quantitation.

Pure function `quantitate(peaks, sample, library, calibration) -> list[PeakResult]`.
Faithfully reproduces the math in Kelly Orton's GCMS-Polyarc spreadsheet
(see docs/superpowers/specs/2026-06-05-polyarc-quantitator-design.md).

The response factor for a compound is single-point per-carbon FID response
broadcast from an anchor standard:

    RF = (anchor.known_wt_pct / anchor.area)
         * (anchor.C / compound.C)
         * (compound.MW / anchor.MW)

Per-sample wt% applies the dilution factor of the original sample in solvent:

    wt_pct = peak.area * RF * dilution_factor

This module contains no file I/O, no GUI dependency, no global state.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from logic.polyarc_calibration import Calibration
from logic.polyarc_library import CompoundRecord, PolyarcLibrary


@dataclass(frozen=True)
class SampleInputs:
    """Sample mass and dilution-solvent mass for one injection.

    `dilution_factor` corresponds to Kelly's `Sample!D$4 = SUM(D2:D3)/D2`.
    """
    sample_mass_g: float
    solvent_mass_g: float

    def __post_init__(self) -> None:
        if self.sample_mass_g <= 0:
            raise ValueError(
                f'sample_mass_g must be > 0; got {self.sample_mass_g!r}'
            )
        if self.solvent_mass_g < 0:
            raise ValueError(
                f'solvent_mass_g must be >= 0; got {self.solvent_mass_g!r}'
            )

    @property
    def dilution_factor(self) -> float:
        return (self.sample_mass_g + self.solvent_mass_g) / self.sample_mass_g


@dataclass(frozen=True)
class PeakResult:
    """Quantitation result for a single peak. Frozen to prevent mutation."""
    # Pass-through fields from input peak
    compound_id: str
    casno: str
    retention_time: float
    area: float
    Qual: float

    # Library match
    matched: bool
    record: CompoundRecord | None

    # Quantitation outputs (None if unmatched)
    RF: float | None
    wt_pct: float | None
    mol_C: float | None
    mol: float | None
    mass_mg: float | None

    # Arbitrary additional input fields preserved verbatim
    extra: dict = field(default_factory=dict)


# Input peak fields that we promote to first-class fields on PeakResult.
# All other keys land in `extra`.
_PROMOTED_KEYS = frozenset({'compound_id', 'casno', 'retention_time', 'area', 'Qual'})


def quantitate(
    peaks: list[dict],
    sample: SampleInputs,
    library: PolyarcLibrary,
    calibration: Calibration,
) -> list[PeakResult]:
    """Compute per-peak wt% in the original sample.

    Args:
        peaks: list of dicts with at minimum {compound_id, casno, area}.
               Additional fields are passed through in `PeakResult.extra`.
        sample: sample mass (g) and solvent mass (g).
        library: indexed compound library.
        calibration: anchor calibration points.

    Returns:
        list[PeakResult], one per input peak, in input order. Unmatched peaks
        have matched=False and None for all quantitation fields.

    Raises:
        ValueError: if any library compound references an anchor not present
            in calibration. Raised once, before any peak is processed, with
            a count of affected compounds per missing anchor.
    """
    # ── Fail-fast anchor validation ───────────────────────────────────────────
    available = calibration.available_anchors()
    missing_counts: Counter[str] = Counter()
    for rec in library.records:
        if rec.anchor and rec.anchor not in available:
            missing_counts[rec.anchor] += 1
    if missing_counts:
        details = ', '.join(
            f'{anchor!r} ({n} compounds)' for anchor, n in missing_counts.items()
        )
        raise ValueError(
            f'Calibration is missing anchors referenced by the library: {details}. '
            f'Available anchors: {sorted(available)}.'
        )

    df = sample.dilution_factor
    results: list[PeakResult] = []
    for peak in peaks:
        compound_id = peak.get('compound_id', '')
        casno = peak.get('casno', '')
        area = float(peak.get('area', 0.0))
        retention_time = float(peak.get('retention_time', 0.0))
        qual = float(peak.get('Qual', 0.0))
        extra = {k: v for k, v in peak.items() if k not in _PROMOTED_KEYS}

        record = library.lookup(casno=casno, name=compound_id)
        if record is None or record.C == 0:
            results.append(PeakResult(
                compound_id=compound_id,
                casno=casno,
                retention_time=retention_time,
                area=area,
                Qual=qual,
                matched=False,
                record=None,
                RF=None, wt_pct=None, mol_C=None, mol=None, mass_mg=None,
                extra=extra,
            ))
            continue

        anchor = calibration[record.anchor]
        RF = ((anchor.known_wt_pct / anchor.area)
              * (anchor.C / record.C)
              * (record.MW / anchor.MW))
        wt_pct = area * RF * df
        mass_g = wt_pct / 100.0 * sample.sample_mass_g
        mass_mg = mass_g * 1000.0
        mol = mass_g / record.MW
        mol_C = mol * record.C

        results.append(PeakResult(
            compound_id=compound_id,
            casno=casno,
            retention_time=retention_time,
            area=area,
            Qual=qual,
            matched=True,
            record=record,
            RF=RF,
            wt_pct=wt_pct,
            mol_C=mol_C,
            mol=mol,
            mass_mg=mass_mg,
            extra=extra,
        ))

    return results
