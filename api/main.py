"""
FastAPI backend for ChromaKit-MS.

This API provides endpoints for:
- Browsing .D files on the server
- Loading GC-MS data from .D files
- Processing chromatograms (smoothing, baseline, peak detection)
- Integrating peaks

All core logic is delegated to existing ChromaKit-MS modules with zero modifications.
"""
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Add parent directory to path to import logic modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from logic.data_handler import DataHandler
from logic.processor import ChromatogramProcessor
from api.models import (
    BrowseResponse, FileEntry, LoadFileRequest, LoadFileResponse,
    ProcessRequest, ProcessResponse, IntegrateRequest, IntegrateResponse,
    ChromatogramData, TICData
)
from api.utils import serialize_numpy, convert_params_for_processor


# Initialize FastAPI app
app = FastAPI(
    title="ChromaKit-MS API",
    description="REST API for GC-MS data processing and analysis",
    version="1.0.0"
)

# Add CORS middleware to allow web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
data_handler = DataHandler()
processor = ChromatogramProcessor()


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "ChromaKit-MS API",
        "version": "1.0.0",
        "description": "REST API for GC-MS chromatogram processing",
        "endpoints": {
            "browse": "/api/browse",
            "load": "/api/load",
            "process": "/api/process",
            "integrate": "/api/integrate"
        }
    }


@app.get("/api/browse", response_model=BrowseResponse)
async def browse_directory(path: str = Query(default=".", description="Directory path to browse")):
    """
    Browse a directory on the server for .D files.
    
    Returns a list of subdirectories and .D files in the specified path.
    
    Args:
        path: Directory path to browse (default: current directory)
        
    Returns:
        BrowseResponse with current path and list of entries
    """
    try:
        # Resolve and validate path
        browse_path = Path(path).resolve()
        
        if not browse_path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        
        if not browse_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
        
        # Get parent path
        parent_path = str(browse_path.parent) if browse_path.parent != browse_path else None
        
        # List entries
        entries = []
        
        try:
            for item in sorted(browse_path.iterdir()):
                if item.is_dir():
                    if item.name.endswith('.D'):
                        # It's a .D directory - a loadable file
                        entries.append(FileEntry(
                            name=item.name,
                            path=str(item),
                            type="file",
                            format="agilent_d"
                        ))
                    else:
                        # It's a regular directory - browsable
                        entries.append(FileEntry(
                            name=item.name,
                            path=str(item),
                            type="directory"
                        ))
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied: {path}")
        
        return BrowseResponse(
            current_path=str(browse_path),
            parent_path=parent_path,
            entries=entries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}")


@app.post("/api/load", response_model=LoadFileResponse)
async def load_file(request: LoadFileRequest):
    """
    Load a .D file and return chromatogram and TIC data.
    
    Uses the existing DataHandler to read Agilent .D files via rainbow-api.
    
    Args:
        request: LoadFileRequest with file_path
        
    Returns:
        LoadFileResponse with chromatogram, TIC, and metadata
    """
    try:
        # Validate path
        file_path = Path(request.file_path)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        if not str(file_path).endswith('.D'):
            raise HTTPException(status_code=400, detail=f"File must be an Agilent .D directory")
        
        # Load data using existing DataHandler
        data = data_handler.load_data_directory(str(file_path))
        
        # Serialize numpy arrays to lists
        chromatogram_data = serialize_numpy(data['chromatogram'])
        tic_data = serialize_numpy(data['tic'])
        
        return LoadFileResponse(
            chromatogram=ChromatogramData(**chromatogram_data),
            tic=TICData(**tic_data),
            has_ms=len(tic_data['x']) > 0,
            metadata=data['metadata']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")


@app.post("/api/process", response_model=ProcessResponse)
async def process_chromatogram(request: ProcessRequest):
    """
    Process chromatogram data with smoothing, baseline correction, and peak detection.
    
    Uses the existing ChromatogramProcessor with user-specified parameters.
    
    Args:
        request: ProcessRequest with data and processing parameters
        
    Returns:
        ProcessResponse with processed data, baseline, and detected peaks
    """
    try:
        # Convert input data to numpy arrays
        x = np.array(request.x)
        y = np.array(request.y)
        
        # Convert parameters to processor format
        params = convert_params_for_processor(request.params.model_dump())
        
        # Convert ms_range to tuple if provided
        ms_range = tuple(request.ms_range) if request.ms_range else None
        
        # Process using existing ChromatogramProcessor
        result = processor.process(x, y, params=params, ms_range=ms_range)
        
        # Serialize numpy arrays
        serialized_result = serialize_numpy(result)
        
        return ProcessResponse(**serialized_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chromatogram: {str(e)}")


@app.post("/api/integrate", response_model=IntegrateResponse)
async def integrate_peaks(request: IntegrateRequest):
    """
    Integrate detected peaks in processed chromatogram.
    
    Uses the existing ChromatogramProcessor.integrate_peaks() method.
    
    Args:
        request: IntegrateRequest with processed data and optional RT table
        
    Returns:
        IntegrateResponse with integration results
    """
    try:
        # Convert processed_data lists back to numpy arrays
        processed_data = {}
        for key, value in request.processed_data.items():
            if isinstance(value, list):
                processed_data[key] = np.array(value)
            else:
                processed_data[key] = value
        
        # Integrate using existing processor
        result = processor.integrate_peaks(
            processed_data=processed_data,
            rt_table=request.rt_table,
            chemstation_area_factor=request.chemstation_area_factor
        )
        
        # Serialize results
        serialized_result = serialize_numpy(result)
        
        # Extract key fields for response
        return IntegrateResponse(
            peaks=[peak.as_dict for peak in result.get('peaks', [])] if 'peaks' in result else [],
            retention_times=serialized_result.get('retention_times', []),
            integrated_areas=serialized_result.get('integrated_areas', []),
            total_peaks=len(result.get('peaks', []))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error integrating peaks: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "data_handler": "ready",
        "processor": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting ChromaKit-MS API server...")
    print("API documentation available at: http://127.0.0.1:8000/docs")
    print("Alternative docs at: http://127.0.0.1:8000/redoc")
    
    uvicorn.run(
        "api.main:app",  # Use module path when running from project root
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info"
    )
