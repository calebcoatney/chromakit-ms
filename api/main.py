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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Add parent directory to path to import logic modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.data_handler import DataHandler
from logic.processor import ChromatogramProcessor
from logic.spectrum_extractor import SpectrumExtractor
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


@app.post("/api/ms/batch-search")
async def ms_batch_search(request: dict):
    """Batch MS library search with progress. Requires ms-toolkit-nrel."""
    raise HTTPException(status_code=501, detail="Batch MS search not yet implemented")


# ─── Quantitation (stub) ─────────────────────────────────────────────

@app.post("/api/quantitate")
async def quantitate(request: dict):
    """Quantitate peaks using Polyarc + internal standard method."""
    try:
        from logic.quantitation import QuantitationCalculator
        # TODO: Wire up QuantitationCalculator with request data
        raise HTTPException(status_code=501, detail="Quantitation endpoint not yet implemented")
    except ImportError:
        raise HTTPException(status_code=500, detail="Quantitation module not found")


# ─── Export ───────────────────────────────────────────────────────────

@app.post("/api/export")
async def export_results(request: ExportRequest):
    """Export serialized peak results to a JSON file.

    Accepts peak dicts from a prior /api/integrate call.
    Writes to the same location as the GUI exporter (inside the .D directory,
    or inside {.C}/results/ for .C folders).

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
    import numpy as np
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
