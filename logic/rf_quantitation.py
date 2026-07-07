"""RF-table external-standard quantitation.

Third quant strategy (distinct from Polyarc and internal-standard). Computes
mol% = area / RF per compound, normalized across reported species. No library,
no anchors, no carbon math, no MS.

Assumes peaks are already RT-assigned (peak.compound_id set upstream by
logic/rt_matching). Single responsibility: area -> amount -> normalize.

See docs/superpowers/specs/2026-07-02-method-embedded-quant-phase-1a-design.md.
Layer rule: logic/ must NOT import from ui/ or api/.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import re


# RF denominator code → normalized-composition output label (None = unspecified/unknown).
RF_UNITS = {
    "area_per_mol":      "mol%",
    "area_per_mol_pct":  "mol%",
    "area_per_molC_pct": "molC%",
    "area_per_wt_pct":   "wt%",
    "unspecified":       None,
}
# Human-readable labels for the GUI dropdown (code → display text).
RF_UNIT_LABELS = {
    "area_per_mol":      "area/mol",
    "area_per_mol_pct":  "area/mol%",
    "area_per_molC_pct": "area/molC%",
    "area_per_wt_pct":   "area/wt%",
    "unspecified":       "unspecified",
}
# Codes for which mol_percent is written (backward compat).
_MOL_BASIS = {"area_per_mol", "area_per_mol_pct"}


@dataclass
class RFQuantSummary:
    strategy: str = "rf_table"
    peaks_total: int = 0
    peaks_quantitated: int = 0
    normalized: bool = True
    total_raw_amount: Optional[float] = None
    skipped_unassigned: list = field(default_factory=list)   # peak RT / index strings
    skipped_no_rf: list = field(default_factory=list)         # "compound — reason" strings
    warnings: list = field(default_factory=list)
    rf_unit: str = "unspecified"
    composition_basis: Optional[str] = None


_UNKNOWN_RT = re.compile(r"Unknown \(\d+\.\d+\)")


def _is_unassigned(compound_id) -> bool:
    if not compound_id:
        return True
    s = str(compound_id).strip()
    if not s or s.lower() == "unknown":
        return True
    if _UNKNOWN_RT.fullmatch(s):
        return True
    return False


def quantitate_rf(
    peaks: list,
    rf_table: "List",                 # list[RFTableEntry]
    rf_unit: str = "unspecified",
    normalize: bool = True,
) -> RFQuantSummary:
    summary = RFQuantSummary(normalized=normalize, peaks_total=len(peaks))
    summary.rf_unit = rf_unit
    summary.composition_basis = RF_UNITS.get(rf_unit)
    write_mol_percent = (rf_unit in _MOL_BASIS) or (rf_unit == "unspecified")
    rf_lookup = {e.compound: e.response_factor for e in rf_table}

    quantitated = []  # (peak, raw_amount)
    for peak in peaks:
        cid = getattr(peak, "compound_id", None)
        if _is_unassigned(cid):
            summary.skipped_unassigned.append(f"{peak.retention_time:.3f}")
            continue
        rf = rf_lookup.get(cid)
        if rf is None:
            summary.skipped_no_rf.append(f"{cid} — not in RF table")
            continue
        raw = peak.area / rf if rf != 0 else 0.0
        peak.raw_amount = raw
        quantitated.append((peak, raw))

    summary.peaks_quantitated = len(quantitated)
    total = float(sum(raw for _, raw in quantitated))
    summary.total_raw_amount = total

    if rf_unit == "unspecified" and summary.peaks_quantitated > 0:
        summary.warnings.append(
            "RF unit unspecified — composition basis unknown; set an RF unit for a labeled result."
        )

    # Loud, alarmable signal when nothing was quantitated but peaks were present
    # (real-time feeds want a flagged-empty result, not a silent zero — spec §3e).
    if not quantitated and (summary.skipped_unassigned or summary.skipped_no_rf):
        summary.warnings.append(
            f"No peaks quantitated: {len(summary.skipped_unassigned)} unassigned, "
            f"{len(summary.skipped_no_rf)} without an RF-table entry."
        )

    if normalize:
        if total == 0:
            if quantitated:
                summary.warnings.append(
                    "Total raw amount is zero; mol% set to 0 for all peaks."
                )
            for peak, _ in quantitated:
                peak.composition_percent = 0.0
                if write_mol_percent:
                    peak.mol_percent = 0.0
        else:
            for peak, raw in quantitated:
                pct = 100.0 * raw / total
                peak.composition_percent = pct
                if write_mol_percent:
                    peak.mol_percent = pct

    return summary
