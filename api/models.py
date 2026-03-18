"""Pydantic models for API request/response validation."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# Request Models
class BrowseRequest(BaseModel):
    """Request to browse a directory."""
    path: str = Field(default=".", description="Directory path to browse")


class LoadFileRequest(BaseModel):
    """Request to load a .D file."""
    file_path: str = Field(..., description="Full path to .D directory")


class SmoothingParams(BaseModel):
    """Smoothing parameters."""
    enabled: bool = False
    method: str = Field(default="whittaker", description="Smoothing method: 'whittaker' or 'savgol'")
    median_enabled: bool = Field(default=False, description="Apply median pre-filter for spike removal")
    median_kernel: int = Field(default=5, ge=3, description="Median filter kernel size (odd)")
    lambda_: float = Field(default=1e-1, alias="lambda")
    diff_order: int = Field(default=1, ge=1, le=2, description="Whittaker difference order: 1=slope, 2=curvature")
    savgol_window: int = Field(default=3, ge=3, description="Savitzky-Golay window length (odd)")
    savgol_polyorder: int = Field(default=1, ge=1, description="Savitzky-Golay polynomial order")

    class Config:
        populate_by_name = True


class BreakPoint(BaseModel):
    """A signal break point for segmented baseline fitting."""
    time: float = Field(..., description="Break point time in minutes")
    tolerance: float = Field(default=0.1, description="Tolerance window around break point")


class BaselineParams(BaseModel):
    """Baseline correction parameters."""
    show_corrected: bool = False
    method: str = "asls"
    lambda_: float = Field(default=1e6, alias="lambda")
    asymmetry: float = 0.01
    break_points: Optional[List[BreakPoint]] = Field(None, description="Signal break points for segmented baseline fitting")
    
    class Config:
        populate_by_name = True  # Allow both 'lambda' and 'lambda_'


class PeakParams(BaseModel):
    """Peak detection parameters."""
    enabled: bool = True
    window_length: int = 41
    polyorder: int = 3
    peak_prominence: float = 0.05
    peak_width: int = 5
    shoulder_height_factor: Optional[float] = Field(None, description="Deprecated: use sensitivity instead. Maps 0.01-0.10 to SNR threshold.")
    shoulder_sensitivity: Optional[int] = Field(8, ge=1, le=10, description="Detection sensitivity 1-10 (higher = more detections)")
    apex_shoulder_distance: int = 10
    min_prominence: Optional[float] = None
    min_width: Optional[int] = None
    range_filters: Optional[List[List[float]]] = Field(None, description="List of [start, end] time ranges to keep peaks within")


class ProcessingParams(BaseModel):
    """Complete processing parameters."""
    smoothing: SmoothingParams = SmoothingParams()
    baseline: BaselineParams = BaselineParams()
    peaks: PeakParams = PeakParams()


class ProcessRequest(BaseModel):
    """Request to process chromatogram data."""
    x: List[float] = Field(..., description="Time values")
    y: List[float] = Field(..., description="Intensity values")
    params: ProcessingParams = ProcessingParams()
    ms_range: Optional[List[float]] = Field(None, description="Optional [min, max] time range for MS data")


class IntegrateRequest(BaseModel):
    """Request to integrate peaks."""
    processed_data: Dict[str, Any] = Field(..., description="Processed chromatogram data")
    rt_table: Optional[Dict[str, Any]] = Field(None, description="Optional retention time table")
    chemstation_area_factor: float = 0.0784
    peak_groups: Optional[List[List[float]]] = Field(None, description="List of [start, end] time windows for peak grouping")


# Response Models
class FileEntry(BaseModel):
    """Represents a file or directory entry."""
    name: str
    path: str
    type: str  # "file" or "directory"
    format: Optional[str] = None  # "agilent_d" for .D files


class BrowseResponse(BaseModel):
    """Response from browsing a directory."""
    current_path: str
    parent_path: Optional[str]
    entries: List[FileEntry]


class ChromatogramData(BaseModel):
    """Chromatogram data structure."""
    x: List[float]
    y: List[float]


class TICData(BaseModel):
    """Total Ion Chromatogram data structure."""
    x: List[float]
    y: List[float]


class LoadFileResponse(BaseModel):
    """Response from loading a file."""
    chromatogram: ChromatogramData
    tic: TICData
    has_ms: bool
    metadata: Dict[str, Any]


class PeakMetadata(BaseModel):
    """Peak metadata structure."""
    index: int
    x: float
    y: float
    is_shoulder: bool
    type: str


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


class PeakResult(BaseModel):
    """Individual peak integration result."""
    compound_id: str
    peak_number: int
    retention_time: float
    width: float
    area: float
    start_time: float
    end_time: float


class IntegrateResponse(BaseModel):
    """Response from peak integration."""
    peaks: List[Dict[str, Any]]
    retention_times: List[float]
    integrated_areas: List[float]
    total_peaks: int
