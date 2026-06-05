"""
Polyarc anchor calibration.

Read-only mapping {anchor_name -> AnchorPoint}. Each AnchorPoint carries the
known wt% and measured FID area for an anchor compound at a single calibration
point (typically the high end of the curve, where Kelly's `Q2`/`Q6` lives).

See docs/superpowers/specs/2026-06-05-polyarc-quantitator-design.md.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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
                anchor_name = row.get('anchor', '').strip()
                if not anchor_name:
                    raise ValueError(
                        f'Calibration row missing required "anchor" column. Row: {dict(row)!r}'
                    )

                try:
                    C = int(row['C'])
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f'Calibration row for anchor {anchor_name!r}: invalid or missing C={row.get("C")!r}'
                    ) from e

                try:
                    MW = float(row['MW'])
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f'Calibration row for anchor {anchor_name!r}: invalid or missing MW={row.get("MW")!r}'
                    ) from e

                try:
                    area = float(row['area'])
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f'Calibration row for anchor {anchor_name!r}: invalid or missing area={row.get("area")!r}'
                    ) from e
                if area <= 0:
                    raise ValueError(
                        f'Calibration row for anchor {anchor_name!r} has non-positive area {area!r}; '
                        f'calibration is invalid.'
                    )

                try:
                    wt_pct = float(row['known_wt_pct'])
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f'Calibration row for anchor {anchor_name!r}: invalid or missing '
                        f'known_wt_pct={row.get("known_wt_pct")!r}'
                    ) from e
                if wt_pct < 0:
                    raise ValueError(
                        f'Calibration row for anchor {anchor_name!r} has negative known_wt_pct {wt_pct!r}.'
                    )

                if anchor_name in anchors:
                    existing = anchors[anchor_name]
                    logger.warning(
                        'Duplicate anchor %r in calibration: previous values '
                        '(area=%g, known_wt_pct=%g) overwritten by new values '
                        '(area=%g, known_wt_pct=%g). Last-write-wins.',
                        anchor_name, existing.area, existing.known_wt_pct, area, wt_pct,
                    )

                anchors[anchor_name] = AnchorPoint(
                    name=anchor_name,
                    cas=row.get('cas', ''),
                    C=C,
                    MW=MW,
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
