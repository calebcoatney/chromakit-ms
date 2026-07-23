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

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from logic.rf_quantitation import RF_UNITS


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
    enabled: bool = Field(default=True, description="Run baseline correction. False integrates raw signal.")
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
    min_prominence: float = Field(default=1e5, description="Prominence threshold. Must be non-null; values <=1 are treated as a fraction of signal range.")

    @field_validator("min_prominence")
    @classmethod
    def _min_prominence_not_null(cls, v):
        if v is None:
            raise ValueError(
                "min_prominence must not be null; use a fractional value "
                "(e.g. 0.02) for spectroscopy or a large value for chromatography."
            )
        return v
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


class RTTableEntry(BaseModel):
    compound: str
    start: float          # RT window start (min)
    apex: float           # expected apex (min)
    end: float            # RT window end (min)


class RFTableEntry(BaseModel):
    compound: str
    response_factor: float    # response factor; output basis is set by the method's rf_unit


class BandWindow(BaseModel):
    """A named fixed x-window for spectroscopy band integration.

    x_min/x_max are in the signal profile's native x-units (cm-1 for FTIR,
    nm for UV-Vis). Bounds are stored ascending regardless of axis direction.
    """
    name: str = Field(..., description="Band name -> SpectralFeature.band_assignment")
    x_min: float = Field(..., description="Window lower bound (native x-units)")
    x_max: float = Field(..., description="Window upper bound (native x-units)")

    @field_validator("x_max")
    @classmethod
    def _check_bounds(cls, v: float, info) -> float:
        x_min = info.data.get("x_min")
        if x_min is not None and v <= x_min:
            raise ValueError(f"x_max ({v}) must be greater than x_min ({x_min})")
        return v


class RTMatchingWeights(BaseModel):
    start: float = 0.25
    apex: float = 0.50
    end: float = 0.25


class RTMatchingParams(BaseModel):
    matching_mode: int = Field(
        default=0, ge=0, le=2,
        description="0=Simple Window, 1=Closest Apex, 2=Weighted Distance",
    )
    tolerance: float = 0.1          # min; Closest Apex mode
    window_expansion: float = 0.0   # min; Simple Window mode
    weights: RTMatchingWeights = Field(default_factory=RTMatchingWeights)
    allow_duplicates: bool = True
    high_priority: bool = False


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
    rt_table: List[RTTableEntry] = Field(default_factory=list)
    rf_table: List[RFTableEntry] = Field(default_factory=list)
    bands: List[BandWindow] = Field(
        default_factory=list,
        description="Fixed-window bands for spectroscopy integration. When "
                    "non-empty, band integration replaces peak detection.",
    )
    rt_matching: RTMatchingParams = Field(default_factory=RTMatchingParams)
    quant_strategy: Optional[str] = Field(
        default=None,
        description="Quantitation strategy: None | 'rf_table' | 'internal_standard'. "
                    "Phase 1a implements 'rf_table' only.",
    )
    rf_unit: str = Field(
        default="unspecified",
        description="RF response-factor unit code (see logic.rf_quantitation.RF_UNITS)",
    )
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

    @field_validator("rf_unit")
    @classmethod
    def _validate_rf_unit(cls, v: str) -> str:
        if v not in RF_UNITS:
            raise ValueError(f"Unknown rf_unit '{v}'; expected one of {sorted(RF_UNITS)}")
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

    def rt_table_as_dataframe(self) -> "pd.DataFrame":
        """Return the embedded RT table as a DataFrame with the GUI's column
        names (Compound, Start, Apex, End) that logic/rt_matching expects.

        The empty frame is constructed with explicit dtypes so the numeric
        columns are float64 on both the empty and populated paths (downstream
        RT matching does float comparisons on Start/Apex/End)."""
        if not self.rt_table:
            return pd.DataFrame({
                "Compound": pd.Series([], dtype="object"),
                "Start": pd.Series([], dtype="float64"),
                "Apex": pd.Series([], dtype="float64"),
                "End": pd.Series([], dtype="float64"),
            })
        return pd.DataFrame(
            [[e.compound, e.start, e.apex, e.end] for e in self.rt_table],
            columns=["Compound", "Start", "Apex", "End"],
        )

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
