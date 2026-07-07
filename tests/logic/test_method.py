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
    RTTableEntry,
    RFTableEntry,
    RTMatchingParams,
    RTMatchingWeights,
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
        "asymmetry": 0.01, "baseline_offset": 0.0,
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


def test_embedded_tables_round_trip():
    import os
    from logic.method import ChromaMethod, RTTableEntry, RFTableEntry
    m = ChromaMethod(name="RAPIDS MeOH", signal_type="gc", quant_strategy="rf_table")
    m.rt_table = [
        RTTableEntry(compound="Hydrogen", start=1.0, apex=1.1, end=1.2),
        RTTableEntry(compound="Carbon monoxide", start=2.0, apex=2.1, end=2.2),
    ]
    m.rf_table = [
        RFTableEntry(compound="Hydrogen", response_factor=402304.0),
        RFTableEntry(compound="Carbon monoxide", response_factor=209181.0),
    ]
    m.rt_matching.matching_mode = 1
    m.rt_matching.tolerance = 0.05
    m.rt_matching.high_priority = True
    m.rt_matching.weights.apex = 0.2   # non-default nested-nested value

    with tempfile.NamedTemporaryFile(suffix=".chromethod", delete=False, mode="w") as f:
        path = f.name
    try:
        m.to_file(path)
        loaded = ChromaMethod.from_file(path)
        assert loaded.quant_strategy == "rf_table"
        assert len(loaded.rt_table) == 2
        assert loaded.rt_table[1].compound == "Carbon monoxide"
        assert loaded.rt_table[1].apex == 2.1
        assert len(loaded.rf_table) == 2
        assert loaded.rf_table[0].response_factor == 402304.0
        assert loaded.rt_matching.matching_mode == 1
        assert loaded.rt_matching.tolerance == 0.05
        assert loaded.rt_matching.high_priority is True
        assert loaded.rt_matching.weights.apex == 0.2   # nested weights round-trip
    finally:
        os.unlink(path)


def test_legacy_method_loads_with_empty_quant_fields():
    """Backward-compat: a method JSON with no RT/RF/rt_matching/quant_strategy
    loads with safe defaults and does not error."""
    import os
    from logic.method import ChromaMethod
    legacy_json = (
        '{"name": "Legacy", "version": "1", "signal_type": "gc", '
        '"chemstation_area_factor": 0.0784}'
    )
    with tempfile.NamedTemporaryFile(suffix=".chromethod", delete=False, mode="w") as f:
        f.write(legacy_json)
        path = f.name
    try:
        loaded = ChromaMethod.from_file(path)
        assert loaded.name == "Legacy"
        assert loaded.quant_strategy is None
        assert loaded.rt_table == []
        assert loaded.rf_table == []
        assert loaded.rt_matching.matching_mode == 0
        assert loaded.rt_matching.tolerance == 0.1
        assert loaded.rt_matching.window_expansion == 0.0
        assert loaded.rt_matching.allow_duplicates is True
        assert loaded.rt_matching.high_priority is False
        assert loaded.rt_matching.weights.apex == 0.50
    finally:
        os.unlink(path)


def test_rt_table_as_dataframe_shape():
    from logic.method import ChromaMethod, RTTableEntry
    m = ChromaMethod(name="X", signal_type="gc")
    m.rt_table = [RTTableEntry(compound="CO", start=2.0, apex=2.1, end=2.2)]
    df = m.rt_table_as_dataframe()
    assert list(df.columns) == ["Compound", "Start", "Apex", "End"]
    assert df.iloc[0]["Compound"] == "CO"
    assert df.iloc[0]["Apex"] == 2.1


def test_rt_table_as_dataframe_empty():
    from logic.method import ChromaMethod
    m = ChromaMethod(name="X", signal_type="gc")
    df = m.rt_table_as_dataframe()
    assert list(df.columns) == ["Compound", "Start", "Apex", "End"]
    assert len(df) == 0
    # Numeric columns must be float64 on the empty path too (matches populated path),
    # so downstream float comparisons in RT matching behave consistently.
    assert str(df["Start"].dtype) == "float64"
    assert str(df["Apex"].dtype) == "float64"
    assert str(df["End"].dtype) == "float64"


def test_matching_mode_rejects_out_of_range():
    from logic.method import RTMatchingParams
    from pydantic import ValidationError
    RTMatchingParams(matching_mode=2)   # valid boundary
    with pytest.raises(ValidationError):
        RTMatchingParams(matching_mode=3)   # invalid
    with pytest.raises(ValidationError):
        RTMatchingParams(matching_mode=-1)  # invalid


def test_rf_unit_defaults_unspecified():
    from logic.method import ChromaMethod
    m = ChromaMethod(name="M", signal_type="gc")
    assert m.rf_unit == "unspecified"


def test_rf_unit_roundtrips(tmp_path):
    from logic.method import ChromaMethod
    p = tmp_path / "m.chromethod"
    ChromaMethod(name="M", signal_type="gc", rf_unit="area_per_wt_pct").to_file(p)
    loaded = ChromaMethod.from_file(p)
    assert loaded.rf_unit == "area_per_wt_pct"


def test_rf_unit_rejects_invalid():
    import pytest
    from pydantic import ValidationError
    from logic.method import ChromaMethod
    with pytest.raises(ValidationError):
        ChromaMethod(name="M", signal_type="gc", rf_unit="area/wt%")


def test_legacy_method_json_loads_with_unspecified(tmp_path):
    from logic.method import ChromaMethod
    p = tmp_path / "legacy.chromethod"
    p.write_text(
        '{"name": "Legacy", "signal_type": "gc", "quant_strategy": "rf_table", '
        '"rf_table": [{"compound": "Hydrogen", "response_factor": 1000.0}]}'
    )
    loaded = ChromaMethod.from_file(p)
    assert loaded.rf_unit == "unspecified"
