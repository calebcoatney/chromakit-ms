"""
Polyarc compound library.

Read-only indexed view of `data/polyarc/compounds.csv`. Provides lookup by
CAS (zero-padded) or compound name (exact, then case-insensitive). Each
record carries the anchor standard whose response factor it inherits.

See docs/superpowers/specs/2026-06-05-polyarc-quantitator-design.md.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Atomic weights used in Kelly's spreadsheet (Compounds!K formula).
_ATOMIC_WEIGHT_C = 12.0107
_ATOMIC_WEIGHT_H = 1.0079
_ATOMIC_WEIGHT_O = 15.9994
_ATOMIC_WEIGHT_S = 32.065
_ATOMIC_WEIGHT_N = 14.0067

# Sentinel CAS used in Kelly's library for compounds with unknown CAS.
# Indexing these would silently misroute any peak whose ms-toolkit search
# returns "0" / "0-00-0" to a single arbitrary compound.
_SENTINEL_CAS = '000000-00-0'


@dataclass(frozen=True)
class CompoundRecord:
    """A single library compound. Frozen to prevent accidental downstream mutation."""
    compound: str
    cas: str
    group1: str
    group2: str
    group3: str
    C: int
    H: int
    O: int
    S: int
    N: int
    MW: float
    anchor: str


class PolyarcLibrary:
    """Read-only indexed view of compounds.csv."""

    def __init__(self, records: list[CompoundRecord]):
        self.records = records
        self._by_cas: dict[str, CompoundRecord] = {}
        self._by_name: dict[str, CompoundRecord] = {}
        self._by_name_ci: dict[str, CompoundRecord] = {}
        for r in records:
            if r.cas:
                padded = self._pad_cas(r.cas)
                if padded == _SENTINEL_CAS:
                    # Skip CAS indexing — these rows are findable by name only.
                    pass
                else:
                    if padded in self._by_cas:
                        existing = self._by_cas[padded]
                        if existing.compound != r.compound:
                            logger.warning(
                                'Duplicate CAS %s in library: %r (kept) and %r (dropped). '
                                'Last-write-wins; both rows remain in records.',
                                padded, existing.compound, r.compound,
                            )
                    self._by_cas[padded] = r
            self._by_name[r.compound] = r
            self._by_name_ci[r.compound.lower()] = r

    @classmethod
    def from_csv(cls, path: str | Path) -> 'PolyarcLibrary':
        path = Path(path)
        records: list[CompoundRecord] = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                C = cls._parse_int(row.get('C'))
                H = cls._parse_int(row.get('H'))
                O = cls._parse_int(row.get('O'))
                S = cls._parse_int(row.get('S'))
                N = cls._parse_int(row.get('N'))
                # Recompute MW from atom counts (CSV's MW is informational)
                MW = (C * _ATOMIC_WEIGHT_C + H * _ATOMIC_WEIGHT_H
                      + O * _ATOMIC_WEIGHT_O + S * _ATOMIC_WEIGHT_S
                      + N * _ATOMIC_WEIGHT_N)
                if C == 0:
                    logger.warning(
                        'Library compound %r has C=0; quantitation will skip it.',
                        row.get('compound'),
                    )
                records.append(CompoundRecord(
                    compound=row['compound'],
                    cas=row.get('cas', ''),
                    group1=row.get('group1', ''),
                    group2=row.get('group2', ''),
                    group3=row.get('group3', ''),
                    C=C, H=H, O=O, S=S, N=N,
                    MW=MW,
                    anchor=row.get('anchor', ''),
                ))
        return cls(records)

    def lookup(self, casno: str | None, name: str | None) -> CompoundRecord | None:
        """CAS first (zero-padded), then exact name, then case-insensitive name.

        Returns None for unknown lookups; empty/None inputs are treated as misses.
        """
        if casno:
            padded = self._pad_cas(casno)
            if padded in self._by_cas:
                return self._by_cas[padded]
        if name:
            if name in self._by_name:
                return self._by_name[name]
            lower = name.lower()
            if lower in self._by_name_ci:
                return self._by_name_ci[lower]
        return None

    @staticmethod
    def _pad_cas(cas: str) -> str:
        """Normalize CAS to '00nnnn-nn-n' (6-digit first segment) for matching.

        Inputs that don't parse as three integer segments pass through unchanged.
        """
        if not cas:
            return ''
        cas = cas.strip()
        parts = cas.split('-')
        if len(parts) == 3:
            try:
                return f'{int(parts[0]):06d}-{parts[1]}-{parts[2]}'
            except ValueError:
                return cas
        return cas

    @staticmethod
    def _parse_int(value: object) -> int:
        """Parse a CSV cell to int; empty/None → 0.

        Raises ValueError on non-empty values that don't parse as int
        (e.g. '1.5', 'abc'). Atom counts must be integers — silent
        truncation would corrupt RF calculations.
        """
        if value is None or value == '':
            return 0
        return int(value)
