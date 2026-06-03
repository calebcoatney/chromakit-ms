"""Method file format for ChromaKit processing pipelines.

A ChromaMethod is a named, persisted snapshot of all processing parameters.
It is the single source of truth for parameter models — imported by api/ and
read/written by the GUI's Save/Load Method buttons.

File format: JSON with .chromethod extension.

Layer rule: this module is in logic/ and must NOT import from api/ or ui/.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Processing Parameter Sub-Models ────────────────────────────────────────────
# These are the canonical definitions. api/models.py imports from here.


class SmoothingParams(BaseModel):
    enabled: bool = False
    method: str = Field(default="whittaker", description="'whittaker' or 'savgol'")
    median_enabled: bool = Field(default=False, description="Apply median pre-filter")
    median_kernel: int = Field(default=5, ge=3, description="Median filter kernel (odd)")
    lambda_: float = Field(default=1e-1, alias="lambda", description="Whittaker lambda")
    diff_order: int = Field(default=1, ge=1, le=2, description="Whittaker difference order")
    savgol_window: int = Field(default=3, ge=3, description="Savitzky-Golay window (odd)")
    savgol_polyorder: int = Field(default=1, ge=1, description="Savitzky-Golay poly order")

    model_config = {"populate_by_name": True}


class BreakPoint(BaseModel):
    time: float = Field(..., description="Break point time in minutes")
    tolerance: float = Field(default=0.1, description="Tolerance window around break point")


class FastchromParams(BaseModel):
    half_window: Optional[int] = None
    smooth_half_window: Optional[int] = None


class BaselineParams(BaseModel):
    show_corrected: bool = False
    method: str = Field(
        default="arpls",
        description="asls|arpls|airpls|imodpoly|modpoly|snip|mixture_model|irsqr|fastchrom",
    )
    lambda_: float = Field(default=1e4, alias="lambda")
    asymmetry: float = 0.01
    baseline_offset: float = Field(default=0.0)
    break_points: Optional[List[BreakPoint]] = Field(default=None)
    fastchrom: Optional[FastchromParams] = Field(default=None)

    model_config = {"populate_by_name": True}


class PeakParams(BaseModel):
    enabled: bool = False
    mode: str = Field(default="classical", description="'classical' or 'deconvolution'")
    window_length: int = 41
    polyorder: int = 3
    peak_prominence: float = 0.05
    peak_width: int = 5
    min_prominence: Optional[float] = Field(default=1e5)
    min_height: Optional[float] = 0.0
    min_width: Optional[float] = 0.0
    range_filters: Optional[List[List[float]]] = Field(default=None)


class DeconvolutionParams(BaseModel):
    splitting_method: str = Field(default="geometric", description="'geometric' or 'emg'")
    windows: Optional[List[List[float]]] = Field(default=None)
    heatmap_threshold: float = 0.36
    pre_fit_signal_threshold: float = 0.001
    min_area_frac: float = 0.15
    valley_threshold_frac: float = 0.48
    mu_bound_factor: float = 0.68
    fat_threshold_frac: float = 0.44
    dedup_sigma_factor: float = 1.32
    dedup_rt_tolerance: float = 0.005


class NegativePeakParams(BaseModel):
    enabled: bool = False
    min_prominence: float = 1e5


class ShoulderParams(BaseModel):
    enabled: bool = False
    window_length: int = 41
    polyorder: int = 3
    sensitivity: int = Field(default=8, ge=1, le=10, description="Detection sensitivity 1-10")
    apex_distance: int = 10


class IntegrationSubParams(BaseModel):
    peak_groups: Optional[List[List[float]]] = Field(
        default=None,
        description="[start, end] time windows for peak grouping",
    )


# ── ChromaMethod ────────────────────────────────────────────────────────────────

_METADATA_FIELDS = frozenset({
    "name", "version", "signal_type", "created_at",
    "chemstation_area_factor",
})


class ChromaMethod(BaseModel):
    """Named snapshot of all ChromaKit processing parameters.

    Usage:
        ChromaMethod.from_file("run.chromethod")   # load from disk
        method.to_file("run.chromethod")            # save to disk
        method.to_processor_params()                # dict for convert_params_for_processor()
        ChromaMethod.from_gui_params(params, ...)   # build from ParametersFrame.current_params
        method.to_gui_params()                      # restore to ParametersFrame.current_params
    """

    name: str
    version: str = "1"
    signal_type: str = Field(..., description="Registered SignalProfileRegistry name")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    smoothing: SmoothingParams = Field(default_factory=SmoothingParams)
    baseline: BaselineParams = Field(default_factory=BaselineParams)
    peaks: PeakParams = Field(default_factory=PeakParams)
    deconvolution: DeconvolutionParams = Field(default_factory=DeconvolutionParams)
    negative_peaks: NegativePeakParams = Field(default_factory=NegativePeakParams)
    shoulders: ShoulderParams = Field(default_factory=ShoulderParams)
    integration: IntegrationSubParams = Field(default_factory=IntegrationSubParams)
    chemstation_area_factor: float = Field(
        default=0.0784,
        description="Chemstation area conversion factor applied during integration",
    )

    @field_validator("signal_type")
    @classmethod
    def _validate_signal_type(cls, v: str) -> str:
        from logic.signal_profiles import SignalProfileRegistry
        try:
            SignalProfileRegistry.get(v)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return v

    @classmethod
    def from_file(cls, path: str | Path) -> "ChromaMethod":
        """Load a .chromethod JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def to_file(self, path: str | Path) -> None:
        """Write this method to a .chromethod JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2, by_alias=True))

    def to_processor_params(self) -> dict:
        """Return a params dict ready for convert_params_for_processor().

        Excludes all method metadata (name, signal_type, etc.) and
        serializes lambda fields using their 'lambda' alias so the
        processor receives the expected key names.
        """
        return self.model_dump(by_alias=True, exclude=_METADATA_FIELDS)

    @classmethod
    def from_gui_params(
        cls,
        params: dict,
        name: str,
        signal_type: str,
        chemstation_area_factor: float = 0.0784,
    ) -> "ChromaMethod":
        """Build a ChromaMethod from ParametersFrame.current_params.

        The GUI stores deconvolution params under the key 'peak_splitting'.
        This method renames that key to 'deconvolution' for the method schema.
        """
        d = dict(params)
        d["deconvolution"] = d.pop("peak_splitting", d.get("deconvolution", {}))
        return cls(
            name=name,
            signal_type=signal_type,
            chemstation_area_factor=chemstation_area_factor,
            **d,
        )

    def to_gui_params(self) -> dict:
        """Return a dict compatible with ParametersFrame.current_params.

        Renames 'deconvolution' back to 'peak_splitting' for GUI compatibility.
        Excludes all method metadata fields.
        """
        d = self.model_dump(by_alias=True, exclude=_METADATA_FIELDS)
        d["peak_splitting"] = d.pop("deconvolution", {})
        return d
