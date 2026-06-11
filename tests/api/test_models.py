"""Tests for new Pydantic models in api/models.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from pydantic import ValidationError

from api.models import (
    LibraryLoadRequest, LibraryLoadResponse,
    SpectralDeconvolutionRequest, SpectralDeconvolutionResponse,
    MSBatchSearchRequest, MSBatchSearchResponse,
    QuantitateRequest, QuantitateResponse,
    ExportRequest,
)


def test_library_load_request_minimal():
    req = LibraryLoadRequest(library_path="/path/to/lib.json")
    assert req.library_path == "/path/to/lib.json"
    assert req.cache_path is None
    assert req.preselector_path is None
    assert req.w2v_path is None


def test_library_load_request_requires_library_path():
    with pytest.raises(ValidationError):
        LibraryLoadRequest()


def test_library_load_response_full():
    resp = LibraryLoadResponse(
        status="loaded",
        compound_count=234567,
        library_path="/lib.json",
        preselector_loaded=True,
        w2v_loaded=False,
        elapsed_seconds=12.5,
    )
    assert resp.compound_count == 234567


def test_spectral_deconvolution_request_defaults():
    req = SpectralDeconvolutionRequest(
        peaks=[{"compound_id": "x", "peak_number": 1, "retention_time": 1.0,
                "integrator": "BB", "width": 0.05, "area": 100.0,
                "start_time": 0.975, "end_time": 1.025}],
        ms_data_path="/data.D",
    )
    assert req.ms_time_offset == 0.0
    assert req.deconv_params is None
    assert req.grouping_params is None


def test_ms_batch_search_request_respects_default():
    req = MSBatchSearchRequest(
        peaks=[],
        data_directory="/data.D",
        options={"search_method": "vector"},
    )
    assert req.respect_manual_assignments is True


def test_ms_batch_search_response_errors_are_dicts():
    """The errors field should accept list of {peak_index, message} dicts."""
    resp = MSBatchSearchResponse(
        peaks=[],
        total_peaks=10,
        successful_matches=8,
        saturated_peaks=1,
        errors=[{"peak_index": 3, "message": "Search blew up"}],
        elapsed_seconds=4.2,
    )
    assert resp.errors[0]["peak_index"] == 3


def test_quantitate_request_minimal():
    req = QuantitateRequest(
        peaks=[],
        internal_standard={
            "compound_name": "Decane",
            "volume_uL": 1.0,
            "density_g_mL": 0.73,
            "molecular_weight": 142.28,
            "formula": "C10H22",
        },
    )
    assert req.sample is None  # optional


def test_quantitate_response_handles_missing_is():
    """IS-not-found is expressed as internal_standard_peak_index = -1."""
    resp = QuantitateResponse(
        peaks=[],
        response_factor=None,
        internal_standard_compound="Decane",
        internal_standard_peak_index=-1,
        peaks_quantitated=0,
        sample_mass_mg=None,
        total_analyte_mass_mg=None,
        carbon_balance_percent=None,
        warnings=["Internal standard 'Decane' not found in peaks"],
    )
    assert resp.internal_standard_peak_index == -1


def test_export_request_has_output_path_field():
    """ExportRequest gained an optional output_path field."""
    req = ExportRequest(peaks=[], file_path="/data.D", output_path="/tmp/custom.json")
    assert req.output_path == "/tmp/custom.json"


def test_export_request_output_path_defaults_none():
    req = ExportRequest(peaks=[], file_path="/data.D")
    assert req.output_path is None
