"""Pydantic models for API request/response validation.

Mirrors the processing parameters from ui/frames/parameters.py and
the data structures from logic/.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ─── Request Models ───────────────────────────────────────────────────

class LoadFileRequest(BaseModel):
    """Request to load a .D file."""
    file_path: str = Field(..., description="Full path to .D directory")
    detector: Optional[str] = Field(None, description="Specific detector to use (e.g. 'FID1A'). Auto-detects if omitted.")


class SmoothingParams(BaseModel):
    """Smoothing parameters — mirrors logic/processor defaults."""
    enabled: bool = False
    method: str = Field(default="whittaker", description="'whittaker' or 'savgol'")
    median_enabled: bool = Field(default=False, description="Apply median pre-filter for spike removal")
    median_kernel: int = Field(default=5, ge=3, description="Median filter kernel size (odd)")
    lambda_: float = Field(default=1e-1, alias="lambda", description="Whittaker smoothing lambda")
    diff_order: int = Field(default=1, ge=1, le=2, description="Whittaker difference order")
    savgol_window: int = Field(default=3, ge=3, description="Savitzky-Golay window length (odd)")
    savgol_polyorder: int = Field(default=1, ge=1, description="Savitzky-Golay polynomial order")

    class Config:
        populate_by_name = True


class BreakPoint(BaseModel):
    """Signal break point for segmented baseline fitting."""
    time: float = Field(..., description="Break point time in minutes")
    tolerance: float = Field(default=0.1, description="Tolerance window around break point")


class FastchromParams(BaseModel):
    """FastChrom baseline method parameters."""
    half_window: Optional[int] = None
    smooth_half_window: Optional[int] = None


class BaselineParams(BaseModel):
    """Baseline correction parameters."""
    show_corrected: bool = False
    method: str = Field(default="arpls", description="asls|arpls|airpls|imodpoly|modpoly|snip|mixture_model|irsqr|fastchrom")
    lambda_: float = Field(default=1e4, alias="lambda")
    asymmetry: float = 0.01
    align_tic: bool = Field(default=False, description="Align MS TIC to FID time axis")
    break_points: Optional[List[BreakPoint]] = Field(default=None, description="Break points for segmented baseline")
    fastchrom: Optional[FastchromParams] = None

    class Config:
        populate_by_name = True


class PeakParams(BaseModel):
    """Peak detection parameters."""
    enabled: bool = False
    mode: str = Field(default="classical", description="'classical' or 'deconvolution'")
    window_length: int = 41
    polyorder: int = 3
    peak_prominence: float = 0.05
    peak_width: int = 5
    min_prominence: Optional[float] = Field(default=1e5, description="Minimum peak prominence")
    min_height: Optional[float] = Field(default=0.0)
    min_width: Optional[float] = Field(default=0.0)
    range_filters: Optional[List[List[float]]] = Field(default=None, description="List of [start, end] time ranges")


class DeconvolutionParams(BaseModel):
    """Deconvolution (peak splitting) parameters."""
    splitting_method: str = Field(default="geometric", description="'geometric' or 'emg'")
    windows: Optional[List[List[float]]] = Field(default=None, description="[start, end] windows; empty = all peaks")
    heatmap_threshold: float = 0.36
    pre_fit_signal_threshold: float = 0.001
    min_area_frac: float = 0.15
    valley_threshold_frac: float = 0.48
    mu_bound_factor: float = 0.68
    fat_threshold_frac: float = 0.44
    dedup_sigma_factor: float = 1.32
    dedup_rt_tolerance: float = 0.005


class NegativePeakParams(BaseModel):
    """Negative peak detection parameters."""
    enabled: bool = False
    min_prominence: float = 1e5


class ShoulderParams(BaseModel):
    """Shoulder detection parameters."""
    enabled: bool = False
    window_length: int = 41
    polyorder: int = 3
    sensitivity: int = Field(default=8, ge=1, le=10, description="Detection sensitivity 1-10")
    apex_distance: int = 10


class IntegrationSubParams(BaseModel):
    """Integration-specific sub-parameters (peak grouping)."""
    peak_groups: Optional[List[List[float]]] = Field(default=None, description="[start, end] time windows for grouping")


class ProcessingParams(BaseModel):
    """Complete processing parameters — mirrors ParametersFrame.current_params."""
    smoothing: SmoothingParams = Field(default_factory=SmoothingParams)
    baseline: BaselineParams = Field(default_factory=BaselineParams)
    peaks: PeakParams = Field(default_factory=PeakParams)
    deconvolution: DeconvolutionParams = Field(default_factory=DeconvolutionParams)
    negative_peaks: NegativePeakParams = Field(default_factory=NegativePeakParams)
    shoulders: ShoulderParams = Field(default_factory=ShoulderParams)
    integration: IntegrationSubParams = Field(default_factory=IntegrationSubParams)


class ProcessRequest(BaseModel):
    """Request to process chromatogram data."""
    x: List[float] = Field(..., description="Time values")
    y: List[float] = Field(..., description="Intensity values")
    params: ProcessingParams = Field(default_factory=ProcessingParams)
    ms_range: Optional[List[float]] = Field(None, description="[min, max] time range for MS data")


class IntegrateRequest(BaseModel):
    """Request to integrate peaks."""
    processed_data: Dict[str, Any] = Field(..., description="Processed chromatogram data")
    rt_table: Optional[Dict[str, Any]] = Field(None, description="Retention time table")
    chemstation_area_factor: float = 0.0784
    peak_groups: Optional[List[List[float]]] = Field(None, description="[start, end] time windows for peak grouping")


class SpectrumRequest(BaseModel):
    """Request to extract a mass spectrum."""
    file_path: str = Field(..., description="Path to .D directory")
    retention_time: float = Field(..., description="Retention time in minutes")
    options: Optional[Dict[str, Any]] = None


class MSSearchRequest(BaseModel):
    """Request to search a mass spectrum against the library."""
    spectrum: Dict[str, Any] = Field(..., description="Spectrum with 'mz' and 'intensities' arrays")
    options: Optional[Dict[str, Any]] = None


class BatchMSSearchRequest(BaseModel):
    """Request for batch MS library search."""
    file_path: str
    peak_indices: Optional[List[int]] = Field(None, description="Specific peak indices; None = all peaks")
    options: Optional[Dict[str, Any]] = None


class RTTableLoadRequest(BaseModel):
    """Request to load an RT table from file."""
    file_path: str


class RTTableMatchRequest(BaseModel):
    """Request to match peaks against an RT table."""
    peaks: List[Dict[str, Any]]
    settings: Dict[str, Any]


class QuantitateRequest(BaseModel):
    """Request to quantitate peaks."""
    peaks: List[Dict[str, Any]]
    settings: Dict[str, Any]


class ExportRequest(BaseModel):
    """Request to export results."""
    peaks: List[Dict[str, Any]]
    file_path: str
    format: str = Field(default="json", description="'json', 'csv', or 'xlsx'")
    options: Optional[Dict[str, Any]] = None


class AlignTICRequest(BaseModel):
    """Request to align TIC to FID."""
    fid_time: List[float]
    fid_signal: List[float]
    tic_time: List[float]
    tic_signal: List[float]


class AssignmentRequest(BaseModel):
    """Request to save a manual assignment."""
    retention_time: float
    compound_name: str
    spectrum: Optional[Dict[str, Any]] = None


class ScalingFactorsRequest(BaseModel):
    """Request to set scaling factors."""
    signal_factor: float = 1.0
    area_factor: float = 1.0


# ─── Response Models ─────────────────────────────────────────────────

class FileEntry(BaseModel):
    """File or directory entry."""
    name: str
    path: str
    type: str
    format: Optional[str] = None


class BrowseResponse(BaseModel):
    """Response from browsing a directory."""
    current_path: str
    parent_path: Optional[str]
    entries: List[FileEntry]


class ChromatogramData(BaseModel):
    """Chromatogram data."""
    x: List[float]
    y: List[float]


class TICData(BaseModel):
    """TIC data."""
    x: List[float]
    y: List[float]


class LoadFileResponse(BaseModel):
    """Response from loading a file."""
    chromatogram: ChromatogramData
    tic: TICData
    has_ms: bool
    metadata: Dict[str, Any]
    available_detectors: List[str] = Field(default_factory=list)
    current_detector: str = "Unknown"


class ProcessResponse(BaseModel):
    """Response from processing chromatogram."""
    x: List[float]
    original_y: List[float]
    smoothed_y: List[float]
    baseline_y: List[float]
    corrected_y: List[float]
    peaks_x: List[float]
    peaks_y: List[float]
    peak_metadata: List[Dict[str, Any]]


class IntegrateResponse(BaseModel):
    """Response from peak integration."""
    peaks: List[Dict[str, Any]]
    retention_times: List[float]
    integrated_areas: List[float]
    total_peaks: int


class SpectrumResponse(BaseModel):
    """Response from spectrum extraction."""
    rt: float
    mz: List[float]
    intensities: List[float]


class AlignTICResponse(BaseModel):
    """Response from TIC alignment."""
    aligned_time: List[float]
    aligned_signal: List[float]
    lag_seconds: float


class NavigationResponse(BaseModel):
    """Response from file navigation."""
    file_path: Optional[str]
    available_count: int
    current_index: int
