"""
FastAPI backend for ChromaKit-MS.

Thin wrapper around the logic/ layer — all core processing is delegated
to existing ChromaKit-MS modules with zero modifications.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Add parent directory to path to import logic modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.data_handler import DataHandler
from logic.processor import ChromatogramProcessor
from logic.spectrum_extractor import SpectrumExtractor
from logic.spectral_deconv_runner import (
    run_spectral_deconvolution,
    WindowGroupingParams,
)
from logic.spectral_deconvolution import DeconvolutionParams
from logic.integration import ChromatographicPeak
from logic.ms_search_core import run_batch_search, BatchSearchSummary
from logic.quantitation_runner import (
    run_quantitation, lookup_compound_metadata,
    InternalStandardSpec, SampleSpec,
)
from functools import partial
from api.models import (
    BrowseResponse, FileEntry,
    LoadFileRequest, LoadFileResponse, ChromatogramData, TICData,
    ProcessRequest, ProcessResponse,
    IntegrateRequest, IntegrateResponse,
    SpectrumRequest, SpectrumResponse,
    AssignmentRequest, ScalingFactorsRequest,
    NavigationResponse,
    ExportRequest,
    RunRequest, RunResponse,
    LibraryLoadRequest, LibraryLoadResponse,
    SpectralDeconvolutionRequest, SpectralDeconvolutionResponse,
    MSBatchSearchRequest, MSBatchSearchResponse,
    QuantitateRequest, QuantitateResponse,
)
from api.utils import serialize_numpy, convert_params_for_processor


# ─── App Setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="ChromaKit-MS API",
    description="REST API for GC-MS data processing and analysis",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances (session-scoped; fine for single-user local use)
data_handler = DataHandler()
processor = ChromatogramProcessor()
spectrum_extractor = SpectrumExtractor()


# ─── Health ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "data_handler": "ready", "processor": "ready"}


@app.get("/")
async def root():
    """Root endpoint — API information."""
    return {
        "name": "ChromaKit-MS API",
        "version": "2.0.0",
        "description": "REST API for GC-MS chromatogram processing",
    }


# ─── File Browsing ────────────────────────────────────────────────────

@app.get("/api/browse", response_model=BrowseResponse)
async def browse_directory(path: str = Query(default=".", description="Directory path to browse")):
    """Browse a directory on the server for .D files."""
    try:
        browse_path = Path(path).resolve()
        if not browse_path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        if not browse_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

        parent_path = str(browse_path.parent) if browse_path.parent != browse_path else None
        entries = []
        try:
            for item in sorted(browse_path.iterdir()):
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    if item.name.endswith('.D'):
                        entries.append(FileEntry(name=item.name, path=str(item), type="file", format="agilent_d"))
                    else:
                        entries.append(FileEntry(name=item.name, path=str(item), type="directory"))
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

        return BrowseResponse(current_path=str(browse_path), parent_path=parent_path, entries=entries)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── File Loading & Navigation ────────────────────────────────────────

@app.post("/api/load", response_model=LoadFileResponse)
async def load_file(request: LoadFileRequest):
    """Load a .D file and return chromatogram, TIC, and detector info."""
    try:
        file_path = Path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        if not str(file_path).endswith('.D'):
            raise HTTPException(status_code=400, detail="File must be an Agilent .D directory")

        data = data_handler.load_data_directory(str(file_path), detector=request.detector)
        chromatogram_data = serialize_numpy(data['chromatogram'])
        tic_data = serialize_numpy(data['tic'])
        detectors = data_handler.get_available_detectors()

        return LoadFileResponse(
            chromatogram=ChromatogramData(**chromatogram_data),
            tic=TICData(**tic_data),
            has_ms=len(tic_data['x']) > 0,
            metadata=data['metadata'],
            available_detectors=detectors,
            current_detector=data_handler.current_detector,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")


@app.get("/api/detectors")
async def get_detectors(path: str = Query(..., description="Path to .D directory")):
    """List available detectors in a .D file."""
    try:
        detectors = data_handler.get_available_detectors(data_dir_path=path)
        return {"detectors": detectors, "current": data_handler.current_detector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/navigate/next", response_model=NavigationResponse)
async def navigate_next():
    """Navigate to the next .D directory."""
    result = data_handler.navigate_to_next()
    if result is None:
        return NavigationResponse(file_path=None, available_count=len(data_handler.available_directories), current_index=data_handler.current_index)
    return NavigationResponse(file_path=result, available_count=len(data_handler.available_directories), current_index=data_handler.current_index)


@app.get("/api/navigate/previous", response_model=NavigationResponse)
async def navigate_previous():
    """Navigate to the previous .D directory."""
    result = data_handler.navigate_to_previous()
    if result is None:
        return NavigationResponse(file_path=None, available_count=len(data_handler.available_directories), current_index=data_handler.current_index)
    return NavigationResponse(file_path=result, available_count=len(data_handler.available_directories), current_index=data_handler.current_index)


# ─── Processing ───────────────────────────────────────────────────────

@app.post("/api/process", response_model=ProcessResponse)
async def process_chromatogram(request: ProcessRequest):
    """Process chromatogram with smoothing, baseline correction, and peak detection."""
    try:
        x = np.array(request.x)
        y = np.array(request.y)
        params = convert_params_for_processor(request.params.model_dump(by_alias=True))
        ms_range = tuple(request.ms_range) if request.ms_range else None

        result = processor.process(x, y, params=params, ms_range=ms_range)
        serialized = serialize_numpy(result)
        return ProcessResponse(**serialized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")


# ─── Integration ──────────────────────────────────────────────────────

@app.post("/api/integrate", response_model=IntegrateResponse)
async def integrate_peaks(request: IntegrateRequest):
    """Integrate detected peaks."""
    try:
        processed_data = {}
        for key, value in request.processed_data.items():
            processed_data[key] = np.array(value) if isinstance(value, list) else value

        result = processor.integrate_peaks(
            processed_data=processed_data,
            rt_table=request.rt_table,
            chemstation_area_factor=request.chemstation_area_factor,
            peak_groups=request.peak_groups,
        )

        serialized = serialize_numpy(result)
        peaks_dicts = []
        for peak in result.get('peaks', []):
            d = peak.as_dict() if hasattr(peak, 'as_dict') else peak
            peaks_dicts.append(serialize_numpy(d))

        return IntegrateResponse(
            peaks=peaks_dicts,
            retention_times=serialized.get('retention_times', []),
            integrated_areas=serialized.get('integrated_areas', []),
            total_peaks=len(peaks_dicts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error integrating: {str(e)}")


# ─── Spectrum Extraction ─────────────────────────────────────────────

@app.post("/api/spectrum", response_model=SpectrumResponse)
async def extract_spectrum(request: SpectrumRequest):
    """Extract a mass spectrum at a given retention time."""
    try:
        result = spectrum_extractor.extract_at_rt(
            data_directory=request.file_path,
            retention_time=request.retention_time,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Could not extract spectrum at given RT")

        return SpectrumResponse(
            rt=float(result['rt']),
            mz=serialize_numpy(result['mz']),
            intensities=serialize_numpy(result['intensities']),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Spectral Deconvolution ──────────────────────────────────────────

@app.post("/api/spectral-deconvolution", response_model=SpectralDeconvolutionResponse)
async def spectral_deconvolution(request: SpectralDeconvolutionRequest):
    """Run ADAP-GC spectral deconvolution on a list of integrated peaks.

    Peaks come in as dicts (Peak.as_dict shape); we re-hydrate them via
    Peak.from_dict, call the pure runner, and serialize back.
    """
    import time
    if not request.peaks:
        raise HTTPException(status_code=422, detail="`peaks` must be non-empty")
    if not Path(request.ms_data_path).exists():
        raise HTTPException(
            status_code=404, detail=f"ms_data_path not found: {request.ms_data_path}"
        )

    try:
        peaks = [ChromatographicPeak.from_dict(p) for p in request.peaks]
    except KeyError as e:
        raise HTTPException(
            status_code=422, detail=f"Malformed peak missing required field: {e}"
        )

    deconv_params = (
        DeconvolutionParams(**request.deconv_params)
        if request.deconv_params else DeconvolutionParams()
    )
    grouping_params = (
        WindowGroupingParams(**request.grouping_params)
        if request.grouping_params else WindowGroupingParams()
    )

    start_ts = time.time()
    try:
        run_spectral_deconvolution(
            peaks=peaks,
            ms_data_path=request.ms_data_path,
            deconv_params=deconv_params,
            grouping_params=grouping_params,
            ms_time_offset=request.ms_time_offset,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deconvolution error: {e}")
    elapsed = round(time.time() - start_ts, 3)

    # Serialize peaks back; also attach deconvolved_spectrum for the response
    out_peaks = []
    components_assigned = 0
    for p in peaks:
        d = p.as_dict()
        if p.deconvolved_spectrum is not None:
            components_assigned += 1
            d['deconvolved_spectrum'] = {
                'mz': p.deconvolved_spectrum['mz'].tolist(),
                'intensities': p.deconvolved_spectrum['intensities'].tolist(),
            }
        if p.deconvolution_component_count is not None:
            d['deconvolution_component_count'] = p.deconvolution_component_count
        out_peaks.append(d)

    components_total = sum(
        (p.deconvolution_component_count or 0) for p in peaks
    )

    return SpectralDeconvolutionResponse(
        peaks=out_peaks,
        components_total=components_total,
        components_assigned=components_assigned,
        elapsed_seconds=elapsed,
    )


# ─── Manual Assignments ──────────────────────────────────────────────

OVERRIDES_PATH = Path(__file__).parent.parent / "overrides" / "manual_assignments.json"


@app.get("/api/assignments")
async def get_assignments():
    """Get all manual RT assignments."""
    if not OVERRIDES_PATH.exists():
        return {"assignments": {}}
    try:
        with open(OVERRIDES_PATH, 'r') as f:
            return {"assignments": json.load(f)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assignments")
async def save_assignment(request: AssignmentRequest):
    """Save or update a manual RT assignment."""
    try:
        assignments = {}
        if OVERRIDES_PATH.exists():
            with open(OVERRIDES_PATH, 'r') as f:
                assignments = json.load(f)

        key = f"{request.retention_time:.4f}"
        entry = {"compound_name": request.compound_name}
        if request.spectrum:
            entry["spectrum"] = request.spectrum
        assignments[key] = entry

        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OVERRIDES_PATH, 'w') as f:
            json.dump(assignments, f, indent=2)

        return {"status": "saved", "key": key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Scaling Factors ──────────────────────────────────────────────────

@app.post("/api/scaling")
async def set_scaling_factors(request: ScalingFactorsRequest):
    """Set signal and area scaling factors."""
    data_handler.signal_factor = request.signal_factor
    return {"signal_factor": request.signal_factor, "area_factor": request.area_factor}


@app.get("/api/scaling")
async def get_scaling_factors():
    """Get current scaling factors."""
    return {"signal_factor": data_handler.signal_factor, "area_factor": 1.0}


# ─── MS Library Lifecycle ────────────────────────────────────────────

@app.post("/api/ms/library/load", response_model=LibraryLoadResponse)
async def ms_library_load(request: LibraryLoadRequest):
    """Load (or reload) the MS library + optional preselector/w2v models.

    The singleton is process-wide; subsequent calls swap the library in place.
    """
    from api import ms_toolkit_singleton
    try:
        summary = ms_toolkit_singleton.load_library(
            library_path=request.library_path,
            cache_path=request.cache_path,
            preselector_path=request.preselector_path,
            w2v_path=request.w2v_path,
        )
        return LibraryLoadResponse(**summary)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── MS Library Search (stub — requires ms-toolkit-nrel) ─────────────

@app.post("/api/ms/search")
async def ms_search(request: dict):
    """Search a mass spectrum against the MS library. Requires ms-toolkit-nrel."""
    try:
        from ms_toolkit import MSToolkit
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="MS library search requires ms-toolkit-nrel. Install with: pip install ms-toolkit-nrel"
        )
    # TODO: Implement once ms-toolkit is available
    raise HTTPException(status_code=501, detail="MS search endpoint not yet implemented")


@app.post("/api/ms/batch-search", response_model=MSBatchSearchResponse)
async def ms_batch_search(request: MSBatchSearchRequest):
    """Batch MS library search across a list of peaks.

    Requires `/api/ms/library/load` to have run first. Returns enriched
    peaks plus a summary. Per-peak errors go in `errors[]`, not HTTP errors.
    """
    import time
    from api import ms_toolkit_singleton

    if not request.peaks:
        raise HTTPException(status_code=422, detail="`peaks` must be non-empty")

    if not Path(request.data_directory).exists():
        raise HTTPException(
            status_code=404,
            detail=f"data_directory not found: {request.data_directory}",
        )

    try:
        ms_toolkit = ms_toolkit_singleton.get_toolkit()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    try:
        peaks = [ChromatographicPeak.from_dict(p) for p in request.peaks]
    except KeyError as e:
        raise HTTPException(
            status_code=422, detail=f"Malformed peak missing required field: {e}"
        )

    start_ts = time.time()
    try:
        summary = run_batch_search(
            ms_toolkit=ms_toolkit,
            peaks=peaks,
            data_directory=request.data_directory,
            options=request.options,
            respect_manual_assignments=request.respect_manual_assignments,
            log_callback=print,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch search error: {e}")
    elapsed = round(time.time() - start_ts, 3)

    return MSBatchSearchResponse(
        peaks=[p.as_dict() for p in peaks],
        total_peaks=summary.total_peaks,
        successful_matches=summary.successful_matches,
        saturated_peaks=summary.saturated_peaks,
        errors=[{"peak_index": i, "message": msg} for (i, msg) in summary.errors],
        elapsed_seconds=elapsed,
    )


# ─── Quantitation ────────────────────────────────────────────────────

@app.post("/api/quantitate", response_model=QuantitateResponse)
async def quantitate(request: QuantitateRequest):
    """Quantitate peaks using Polyarc + Internal Standard method.

    Requires `/api/ms/library/load` to have run first (needed for per-compound
    formula + MW lookup). IS-not-found and zero-quantitated both return 200
    with explicit fields/warnings, not HTTP errors.
    """
    from api import ms_toolkit_singleton

    if not request.peaks:
        raise HTTPException(status_code=422, detail="`peaks` must be non-empty")

    # Validate and build typed IS spec
    is_dict = request.internal_standard
    required = ('compound_name', 'volume_uL', 'density_g_mL', 'molecular_weight', 'formula')
    missing = [k for k in required if k not in is_dict or is_dict[k] in (None, "")]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"internal_standard missing required fields: {missing}",
        )

    try:
        is_spec = InternalStandardSpec(
            compound_name=str(is_dict['compound_name']),
            volume_uL=float(is_dict['volume_uL']),
            density_g_mL=float(is_dict['density_g_mL']),
            molecular_weight=float(is_dict['molecular_weight']),
            formula=str(is_dict['formula']),
        )
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422, detail=f"internal_standard value invalid: {e}"
        )

    sample_dict = request.sample or {}
    sample_spec = SampleSpec(
        volume_uL=sample_dict.get('volume_uL'),
        density_g_mL=sample_dict.get('density_g_mL'),
    )

    # Library is required for compound metadata lookup
    try:
        ms_toolkit = ms_toolkit_singleton.get_toolkit()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Re-hydrate peaks
    try:
        peaks = [ChromatographicPeak.from_dict(p) for p in request.peaks]
    except KeyError as e:
        raise HTTPException(
            status_code=422, detail=f"Malformed peak missing required field: {e}"
        )

    compound_lookup = partial(lookup_compound_metadata, ms_toolkit)

    try:
        summary = run_quantitation(
            peaks=peaks,
            internal_standard=is_spec,
            sample=sample_spec,
            compound_lookup=compound_lookup,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantitation error: {e}")

    return QuantitateResponse(
        peaks=[p.as_dict() for p in peaks],
        response_factor=summary.response_factor,
        internal_standard_compound=is_spec.compound_name,
        internal_standard_peak_index=summary.internal_standard_peak_index,
        peaks_quantitated=summary.peaks_quantitated,
        sample_mass_mg=summary.sample_mass_mg,
        total_analyte_mass_mg=summary.total_analyte_mass_mg,
        carbon_balance_percent=summary.carbon_balance_percent,
        warnings=summary.warnings,
    )


# ─── Export ───────────────────────────────────────────────────────────

@app.post("/api/export")
async def export_results(request: ExportRequest):
    """Export serialized peak results to a JSON file.

    When `output_path` is set, write there (parent dirs created if missing).
    When `output_path` is None, fall back to _resolve_export_context for
    backward compatibility with the bridge contract.

    Only 'json' format is supported in Phase 1.
    """
    import json as _json
    from logic.json_exporter import _resolve_export_context

    try:
        if request.format != "json":
            raise HTTPException(
                status_code=400,
                detail=f"Format '{request.format}' not supported. Only 'json' is available.",
            )

        if request.output_path:
            # Custom destination — caller controls path
            output_path = request.output_path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Minimal metadata when caller bypasses _resolve_export_context
            metadata = {}
        else:
            # Default — resolve based on .D / .C structure
            metadata, output_path = _resolve_export_context(
                request.file_path, data_handler.current_detector
            )

        result_data = {
            **metadata,
            "peaks": request.peaks,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(result_data, f, indent=4)

        return {"status": "exported", "output_file": output_path}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


# ─── Full Pipeline Run ────────────────────────────────────────────────

@app.post("/api/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest):
    """Run the full Phase 1 pipeline in one call: load → process → integrate → export JSON.

    Requires a .chromethod file (save one from the GUI via Load/Save Method buttons).
    Writes a JSON result file alongside the data file and returns the peak table
    plus the output file path.
    """
    from logic.method import ChromaMethod
    from logic.json_exporter import export_integration_results_to_json, _resolve_export_context

    try:
        # 1. Load method
        try:
            method = ChromaMethod.from_file(request.method_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, detail=f"Method file not found: {request.method_path}"
            )

        # 2. Load data
        try:
            data = data_handler.load_data_directory(
                request.data_path, detector=request.detector
            )
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))

        x = np.array(data["chromatogram"]["x"])
        y = np.array(data["chromatogram"]["y"])

        # 3. Process (smoothing + baseline + peak detection)
        raw_params = method.to_processor_params()
        proc_params = convert_params_for_processor(raw_params)
        processed = processor.process(x, y, params=proc_params)

        # 4. Integrate
        integrated = processor.integrate_peaks(
            processed_data=processed,
            rt_table=None,
            chemstation_area_factor=method.chemstation_area_factor,
            peak_groups=method.integration.peak_groups or [],
        )
        peaks = integrated.get("peaks", [])

        # 5. Export JSON (writes alongside data file, same as GUI behavior)
        export_integration_results_to_json(
            peaks=peaks,
            d_path=request.data_path,
            detector=data_handler.current_detector,
            processing_params=raw_params,
            ms_time_offset=float(getattr(data_handler, 'ms_time_offset', 0.0)),
        )

        # 6. Determine output file path
        _, output_file = _resolve_export_context(
            request.data_path, data_handler.current_detector
        )

        # 7. Serialize peaks for response
        peaks_dicts = []
        for peak in peaks:
            d = peak.as_dict() if hasattr(peak, "as_dict") else peak
            peaks_dicts.append(serialize_numpy(d))

        return RunResponse(
            status="complete",
            data_path=request.data_path,
            method=method.name,
            version=method.version,
            signal_type=method.signal_type,
            peak_count=len(peaks_dicts),
            peaks=peaks_dicts,
            output_files=[str(output_file)],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ─── MS Baseline Correction (stub) ───────────────────────────────────

@app.post("/api/ms/baseline")
async def ms_baseline_correction(request: dict):
    """Apply baseline correction to MS data."""
    raise HTTPException(status_code=501, detail="MS baseline correction not yet implemented")


# ─── RT Table (stub) ─────────────────────────────────────────────────

@app.post("/api/rt-table/load")
async def load_rt_table(request: dict):
    """Load an RT table from file."""
    raise HTTPException(status_code=501, detail="RT table loading not yet implemented")


@app.post("/api/rt-table/match")
async def match_rt_table(request: dict):
    """Match peaks against an RT table."""
    raise HTTPException(status_code=501, detail="RT table matching not yet implemented")


# ─── Batch Processing (stub) ─────────────────────────────────────────

@app.post("/api/batch")
async def batch_process(request: dict):
    """Batch process multiple .D directories."""
    raise HTTPException(status_code=501, detail="Batch processing not yet implemented")


# ─── Server Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("Starting ChromaKit-MS API server...")
    print("API documentation available at: http://127.0.0.1:8000/docs")

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
