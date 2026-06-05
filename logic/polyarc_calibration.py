"""
Polyarc anchor calibration.

Read-only mapping {anchor_name -> AnchorPoint}. Each AnchorPoint carries the
known wt% and measured FID area for an anchor compound at a single calibration
point (typically the high end of the curve, where Kelly's `Q2`/`Q6` lives).

See docs/superpowers/specs/2026-06-05-polyarc-quantitator-design.md.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnchorPoint:
    """One anchor's calibration measurement. Frozen to prevent mutation."""
    name: str
    cas: str
    C: int
    MW: float
    known_wt_pct: float
    area: float
    run_date: str
    instrument_id: str
    notes: str = ''


class Calibration:
    """Read-only mapping {anchor_name: AnchorPoint}."""

    def __init__(self, anchors: dict[str, AnchorPoint]):
        self._anchors = dict(anchors)

    @classmethod
    def from_csv(cls, path: str | Path) -> 'Calibration':
        path = Path(path)
        anchors: dict[str, AnchorPoint] = {}
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                area = float(row['area'])
                if area <= 0:
                    raise ValueError(
                        f'Calibration row for anchor {row["anchor"]!r} has '
                        f'non-positive area {area!r}; calibration is invalid.'
                    )
                wt_pct = float(row['known_wt_pct'])
                if wt_pct < 0:
                    raise ValueError(
                        f'Calibration row for anchor {row["anchor"]!r} has '
                        f'negative known_wt_pct {wt_pct!r}.'
                    )
                anchors[row['anchor']] = AnchorPoint(
                    name=row['anchor'],
                    cas=row.get('cas', ''),
                    C=int(row['C']),
                    MW=float(row['MW']),
                    known_wt_pct=wt_pct,
                    area=area,
                    run_date=row.get('run_date', ''),
                    instrument_id=row.get('instrument_id', ''),
                    notes=row.get('notes', ''),
                )
        return cls(anchors)

    def __getitem__(self, anchor: str) -> AnchorPoint:
        """Raises KeyError if anchor is missing."""
        return self._anchors[anchor]

    def available_anchors(self) -> set[str]:
        return set(self._anchors.keys())
