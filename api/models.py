"""Pydantic models for API request/response validation.

Mirrors the processing parameters from ui/frames/parameters.py and
the data structures from logic/.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# ── Processing param models — canonical definitions live in logic/method.py ──
from logic.method import (
    SmoothingParams,
    BreakPoint,
    FastchromParams,
    BaselineParams,
    PeakParams,
    DeconvolutionParams,
    NegativePeakParams,
    ShoulderParams,
    IntegrationSubParams,
)


# ─── Request Models ───────────────────────────────────────────────────

class LoadFileRequest(BaseModel):
    """Request to load a .D file."""
    file_path: str = Field(..., description="Full path to .D directory")
    detector: Optional[str] = Field(None, description="Specific detector to use (e.g. 'FID1A'). Auto-detects if omitted.")


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


class NavigationResponse(BaseModel):
    """Response from file navigation."""
    file_path: Optional[str]
    available_count: int
    current_index: int


class RunRequest(BaseModel):
    """Request to run the full Phase 1 pipeline: load → process → integrate → export JSON."""
    data_path: str = Field(..., description="Path to Agilent .D or .C directory")
    method_path: str = Field(..., description="Path to .chromethod file")
    detector: Optional[str] = Field(
        None,
        description="Detector to use (e.g. 'FID1A'). Auto-detected if omitted.",
    )


class RunResponse(BaseModel):
    """Response from POST /api/run."""
    status: str = Field(..., description="'complete' or 'error'")
    data_path: str
    method: str = Field(..., description="Method name from the .chromethod file")
    version: str = Field(..., description="Method version string from the .chromethod file (e.g. '1')")
    signal_type: str
    peak_count: int
    peaks: List[Dict[str, Any]]
    output_files: List[str] = Field(
        ..., description="Absolute paths of JSON files written to disk"
    )
