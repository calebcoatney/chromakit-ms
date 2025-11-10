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
    median_filter: Dict[str, int] = {"kernel_size": 5}
    savgol_filter: Dict[str, int] = {"window_length": 11, "polyorder": 3}


class BaselineParams(BaseModel):
    """Baseline correction parameters."""
    show_corrected: bool = False
    method: str = "asls"
    lambda_: float = Field(default=1e6, alias="lambda")
    asymmetry: float = 0.01
    
    class Config:
        populate_by_name = True  # Allow both 'lambda' and 'lambda_'


class PeakParams(BaseModel):
    """Peak detection parameters."""
    enabled: bool = True
    window_length: int = 41
    polyorder: int = 3
    peak_prominence: float = 0.05
    peak_width: int = 5
    shoulder_height_factor: float = 0.02
    apex_shoulder_distance: int = 10
    min_prominence: Optional[float] = None
    min_width: Optional[int] = None


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
