from __future__ import annotations
from typing import List


class Feature:
    """Base class for all detected signal features (peaks, bands, etc.).

    Uses a plain __init__ (not @dataclass) so that ChromatographicPeak can
    subclass it cleanly with its own __init__ without dataclass field-introspection
    complications.
    """

    def __init__(
        self,
        feature_id: int,
        position: float,
        position_units: str,
        area: float,
        width: float,
        start: float,
        end: float,
        start_index: int,
        end_index: int,
        is_shoulder: bool = False,
        is_negative: bool = False,
        quality_issues: List[str] = None,
    ):
        self.feature_id = feature_id
        self._position = position        # backing store; access via .position property
        self.position_units = position_units
        self.area = area
        self.width = width
        self.start = start
        self.end = end
        self.start_index = start_index
        self.end_index = end_index
        self.is_shoulder = is_shoulder
        self.is_negative = is_negative
        self.quality_issues = quality_issues if quality_issues is not None else []

    @property
    def position(self) -> float:
        """Generic x-axis position. ChromatographicPeak overrides this to return retention_time."""
        return self._position


class SpectralFeature(Feature):
    """Feature for spectroscopic signals (FTIR, UV-Vis)."""

    def __init__(
        self,
        feature_id: int,
        position: float,
        position_units: str,
        area: float,
        width: float,
        start: float,
        end: float,
        start_index: int,
        end_index: int,
        is_shoulder: bool = False,
        is_negative: bool = False,
        quality_issues: List[str] = None,
        band_assignment: str = "",
        absorbance: float = 0.0,
        transmittance: float = 0.0,
    ):
        super().__init__(
            feature_id=feature_id, position=position, position_units=position_units,
            area=area, width=width, start=start, end=end,
            start_index=start_index, end_index=end_index,
            is_shoulder=is_shoulder, is_negative=is_negative, quality_issues=quality_issues,
        )
        self.band_assignment = band_assignment
        self.absorbance = absorbance
        self.transmittance = transmittance

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "position": self.position,
            "position_units": self.position_units,
            "area": self.area,
            "width": self.width,
            "start": self.start,
            "end": self.end,
            "is_shoulder": self.is_shoulder,
            "is_negative": self.is_negative,
            "band_assignment": self.band_assignment,
            "absorbance": self.absorbance,
            "transmittance": self.transmittance,
            "quality_issues": self.quality_issues,
        }

    def as_row(self) -> list:
        return [
            round(self.position, 2),
            round(self.area, 1),
            round(self.width, 2),
            round(self.start, 2),
            round(self.end, 2),
            self.band_assignment,
            round(self.absorbance, 4),
        ]
