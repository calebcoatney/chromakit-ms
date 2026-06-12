"""Pydantic models for API request/response validation.

Mirrors the processing parameters from ui/frames/parameters.py and
the data structures from logic/.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

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
    """Request to search a single mass spectrum against the loaded MS library.

    Mirrors MSBatchSearchRequest's options shape but takes a spectrum
    directly instead of a peak + data_directory. Use this when the agent
    already has a spectrum in hand and wants to test it across several
    search configurations without re-running batch.

    Note: this redefines the unused stub previously at api/models.py:65-68.
    No other module imports it (verified via grep), so the redefinition
    is safe.
    """
    spectrum: Dict[str, Any] = Field(
        ...,
        description="Spectrum with 'mz' and 'intensities' arrays of equal length",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Search options matching MSBatchSearchRequest.options shape. "
            "Reads 'search_method' (vector|w2v|hybrid), 'top_n', 'similarity', "
            "'weighting', 'unmatched', 'top_k_clusters', 'intensity_power', "
            "'hybrid_method'. See do_single_search in logic/ms_search_core.py."
        ),
    )
    mz_shift: int = Field(
        default=0,
        description=(
            "Integer m/z shift applied to the toolkit's library spectra "
            "before searching. Always applied (even at the default of 0), "
            "overwriting any prior toolkit state, to guarantee deterministic "
            "results across sequential requests on the shared singleton. "
            "Default 0 (no shift)."
        ),
    )
    top_n: int = Field(
        default=5,
        description="Convenience override of options['top_n']. Default 5.",
    )


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
    """Request to run Polyarc + IS quantitation against a list of peaks.

    `internal_standard` must contain:
      compound_name, volume_uL, density_g_mL, molecular_weight, formula
    `sample` (optional) contains: volume_uL, density_g_mL.
    """
    peaks: List[Dict[str, Any]]
    internal_standard: Dict[str, Any]
    sample: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    """Request to export results.

    When `output_path` is None, the writer resolves the destination from
    `file_path` via _resolve_export_context (writes inside the .D folder
    or inside <.C>/results/). When `output_path` is set, the writer
    creates parent dirs and writes there — used by R&D agents that want
    distinct outputs per sweep iteration.
    """
    peaks: List[Dict[str, Any]]
    file_path: str
    format: str = Field(default="json", description="'json', 'csv', or 'xlsx'")
    options: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = Field(
        default=None,
        description="If set, write the export file to this absolute path instead of the default location",
    )


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


# ─── R&D Experimentation Endpoints (Phase 1) ──────────────────────────

class LibraryLoadRequest(BaseModel):
    """Request to load the MS library + optional models into the API singleton."""
    library_path: str = Field(..., description="Path to a library file (.txt or .json)")
    cache_path: Optional[str] = Field(None, description="Path to a JSON cache (faster path)")
    preselector_path: Optional[str] = Field(None, description="Path to a .pkl preselector model")
    w2v_path: Optional[str] = Field(None, description="Path to a Word2Vec .model file")


class LibraryLoadResponse(BaseModel):
    """Response from POST /api/ms/library/load."""
    status: str
    compound_count: int
    library_path: str
    preselector_loaded: bool
    w2v_loaded: bool
    elapsed_seconds: float


class SpectralDeconvolutionRequest(BaseModel):
    """Request to run ADAP-GC spectral deconvolution on integrated peaks."""
    peaks: List[Dict[str, Any]] = Field(..., description="Integrated peaks (Peak.as_dict shape)")
    ms_data_path: str = Field(..., description="Path to the .D or .C directory")
    deconv_params: Optional[Dict[str, Any]] = Field(
        None, description="Serialized DeconvolutionParams; None = defaults"
    )
    grouping_params: Optional[Dict[str, Any]] = Field(
        None, description="Serialized WindowGroupingParams; None = defaults"
    )
    ms_time_offset: float = 0.0


class SpectralDeconvolutionResponse(BaseModel):
    """Response from POST /api/spectral-deconvolution."""
    peaks: List[Dict[str, Any]] = Field(
        ..., description="Same peaks with deconvolved_spectrum populated"
    )
    components_total: int
    components_assigned: int
    elapsed_seconds: float


class MSBatchSearchRequest(BaseModel):
    """Request for batch MS library search across a list of peaks."""
    peaks: List[Dict[str, Any]] = Field(..., description="Integrated peaks (with or without deconvolved_spectrum)")
    data_directory: str = Field(..., description="Path to .D or .C (used by SpectrumExtractor)")
    options: Dict[str, Any] = Field(
        ...,
        description=(
            "Full search options. Defaults to mimic GUI: "
            "{'search_method': 'vector', 'top_n': 5, 'similarity': 'composite', "
            "'weighting': 'NIST_GC', 'unmatched': 'keep_all', 'top_k_clusters': 1, "
            "'extraction_method': 'apex', 'range_points': 5, 'range_time': 0.05, "
            "'tic_weight': True, 'subtract_enabled': True, 'subtraction_method': 'min_tic', "
            "'subtract_weight': 0.1, 'intensity_power': 0.6}. "
            "See ui/frames/ms.py:51-67 for the full default set. "
            "Note: 'mz_shift' is also read here for backward compatibility, but the "
            "top-level mz_shift field takes precedence if both are set."
        ),
    )
    respect_manual_assignments: bool = True
    ms_time_offset: float = Field(
        default=0.0,
        description=(
            "Constant shift (minutes) applied to MS retention times "
            "when extracting spectra. Mirrors the same field on "
            "/api/spectral-deconvolution. Default 0.0 (no shift)."
        ),
    )
    mz_shift: int = Field(
        default=0,
        description=(
            "Integer m/z shift applied to the toolkit's library spectra "
            "before searching. Overrides options['mz_shift'] if both set. "
            "Always applied (even at the default of 0), overwriting any "
            "prior toolkit state, to guarantee deterministic results across "
            "sequential requests on the shared singleton. Default 0 (no shift)."
        ),
    )


class MSBatchSearchResponse(BaseModel):
    """Response from POST /api/ms/batch-search."""
    peaks: List[Dict[str, Any]]
    total_peaks: int
    successful_matches: int
    saturated_peaks: int
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-peak errors as [{'peak_index': int, 'message': str}]",
    )
    elapsed_seconds: float


class MSSearchHit(BaseModel):
    """One library match returned by /api/ms/search."""
    name: str = Field(..., description="Compound name from the library")
    score: float = Field(..., description="Similarity score (higher = better)")
    casno: Optional[str] = Field(
        default=None,
        description=(
            "CAS Registry Number, formatted NNNNNN-NN-N. None if the "
            "compound is not in the loaded library or has no CAS number "
            "(empty strings from the library are normalized to None for "
            "a clean contract)."
        ),
    )

    @field_validator('casno', mode='before')
    @classmethod
    def _normalize_empty_casno(cls, v):
        """Normalize empty strings from format_casno() to None.

        logic.ms_search_core.format_casno returns "" for empty/non-string
        input. Mapping that to None here keeps the agent-facing contract
        simple: callers see either a formatted CAS string or None.
        """
        if v == "":
            return None
        return v


class MSSearchResponse(BaseModel):
    """Response from POST /api/ms/search."""
    results: List[MSSearchHit] = Field(
        ...,
        description="Top-N matches in descending score order. May be empty.",
    )
    elapsed_seconds: float


class QuantitateResponse(BaseModel):
    """Response from POST /api/quantitate."""
    peaks: List[Dict[str, Any]]
    response_factor: Optional[float]
    internal_standard_compound: str
    internal_standard_peak_index: int = Field(
        ..., description="-1 if the IS compound was not found in `peaks`"
    )
    peaks_quantitated: int = Field(
        ..., description="Number of analyte peaks (excludes IS) successfully quantitated"
    )
    sample_mass_mg: Optional[float]
    total_analyte_mass_mg: Optional[float]
    carbon_balance_percent: Optional[float]
    warnings: List[str] = Field(default_factory=list)
