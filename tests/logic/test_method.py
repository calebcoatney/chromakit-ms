"""Tests for logic/method.py — ChromaMethod and param models."""
import json
import tempfile
from pathlib import Path
import pytest

# These imports will fail until logic/method.py is created — that's expected.
from logic.method import (
    ChromaMethod,
    SmoothingParams,
    BaselineParams,
    PeakParams,
    DeconvolutionParams,
    NegativePeakParams,
    ShoulderParams,
    IntegrationSubParams,
)

# Minimal GUI params dict (mirrors ParametersFrame.current_params)
_GUI_PARAMS = {
    "smoothing": {
        "enabled": False, "method": "whittaker", "median_enabled": False,
        "median_kernel": 5, "lambda": 0.1, "diff_order": 1,
        "savgol_window": 3, "savgol_polyorder": 1,
    },
    "baseline": {
        "show_corrected": False, "method": "arpls", "lambda": 1e4,
        "asymmetry": 0.01, "baseline_offset": 0.0, "align_tic": False,
        "break_points": [], "fastchrom": {"half_window": None, "smooth_half_window": None},
    },
    "peaks": {
        "enabled": False, "mode": "classical", "min_prominence": 1e5,
        "min_height": 0.0, "min_width": 0.0, "range_filters": [],
    },
    "peak_splitting": {
        "splitting_method": "geometric", "windows": [],
        "heatmap_threshold": 0.36, "pre_fit_signal_threshold": 0.001,
        "min_area_frac": 0.15, "valley_threshold_frac": 0.48,
        "mu_bound_factor": 0.68, "fat_threshold_frac": 0.44,
        "dedup_sigma_factor": 1.32, "dedup_rt_tolerance": 0.005,
    },
    "negative_peaks": {"enabled": False, "min_prominence": 1e5},
    "shoulders": {
        "enabled": False, "window_length": 41, "polyorder": 3,
        "sensitivity": 8, "apex_distance": 10,
    },
    "integration": {"peak_groups": []},
}


def test_creates_with_defaults():
    m = ChromaMethod(name="Test", signal_type="gc")
    assert m.name == "Test"
    assert m.signal_type == "gc"
    assert m.version == "1"
    assert m.chemstation_area_factor == pytest.approx(0.0784)


def test_invalid_signal_type_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChromaMethod(name="Bad", signal_type="nonexistent_instrument")


def test_round_trip_to_from_file():
    import os
    m = ChromaMethod(name="CO2 Hydro GC", signal_type="gc")
    m.smoothing.enabled = True
    m.baseline.method = "snip"
    m.chemstation_area_factor = 0.05

    with tempfile.NamedTemporaryFile(suffix=".chromethod", delete=False, mode="w") as f:
        path = f.name

    try:
        m.to_file(path)
        loaded = ChromaMethod.from_file(path)

        assert loaded.name == "CO2 Hydro GC"
        assert loaded.signal_type == "gc"
        assert loaded.smoothing.enabled is True
        assert loaded.baseline.method == "snip"
        assert loaded.chemstation_area_factor == pytest.approx(0.05)
    finally:
        os.unlink(path)


def test_to_processor_params_excludes_metadata():
    m = ChromaMethod(name="Test", signal_type="gc")
    p = m.to_processor_params()
    for key in ("name", "signal_type", "created_at", "version",
                "chemstation_area_factor", "export_output_dir"):
        assert key not in p, f"metadata key '{key}' should not be in processor params"
    for key in ("smoothing", "baseline", "peaks", "deconvolution",
                "negative_peaks", "shoulders", "integration"):
        assert key in p, f"param key '{key}' should be in processor params"


def test_to_processor_params_lambda_alias():
    """Lambda must serialize as 'lambda' (the alias), not 'lambda_'."""
    m = ChromaMethod(name="Test", signal_type="gc")
    m.smoothing.lambda_ = 0.5
    m.baseline.lambda_ = 1e5
    p = m.to_processor_params()
    assert "lambda" in p["smoothing"], "smoothing lambda key should be 'lambda'"
    assert "lambda_" not in p["smoothing"]
    assert "lambda" in p["baseline"], "baseline lambda key should be 'lambda'"


def test_from_gui_params_renames_peak_splitting():
    m = ChromaMethod.from_gui_params(_GUI_PARAMS, name="Test", signal_type="gc")
    assert m.deconvolution.splitting_method == "geometric"
    assert m.smoothing.enabled is False
    assert m.peaks.min_prominence == pytest.approx(1e5)


def test_to_gui_params_renames_deconvolution():
    m = ChromaMethod(name="Test", signal_type="gc")
    m.deconvolution.splitting_method = "emg"
    gui = m.to_gui_params()
    assert "peak_splitting" in gui, "GUI expects 'peak_splitting', not 'deconvolution'"
    assert "deconvolution" not in gui
    assert gui["peak_splitting"]["splitting_method"] == "emg"


def test_from_gui_to_gui_roundtrip():
    m = ChromaMethod.from_gui_params(_GUI_PARAMS, name="Test", signal_type="gc")
    result = m.to_gui_params()
    assert result["smoothing"]["enabled"] is False
    assert result["smoothing"]["method"] == "whittaker"
    assert result["baseline"]["method"] == "arpls"
    assert result["peaks"]["min_prominence"] == pytest.approx(1e5)
    assert "peak_splitting" in result
    assert result["peak_splitting"]["splitting_method"] == "geometric"
