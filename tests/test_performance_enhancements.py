
import os
import json
import pytest
import numpy as np
from logic.integration import ChromatographicPeak
from logic.feature import Feature, SpectralFeature
from logic.json_exporter import export_integration_results_to_json, _serialize_peak
from logic.csv_exporter import export_results_to_csv

def test_feature_as_dict():
    f = Feature(
        feature_id=1, position=10.0, position_units="min", 
        area=100.0, width=1.0, start=9.5, end=10.5,
        start_index=95, end_index=105
    )
    d = f.as_dict()
    assert d["feature_id"] == 1
    assert d["position"] == 10.0
    assert d["area"] == 100.0
    assert "quality_issues" in d

def test_chromatographic_peak_as_dict():
    peak = ChromatographicPeak(
        compound_id="P1", peak_number=1, retention_time=1.23,
        integrator="py", width=0.1, area=500.0,
        start_time=1.1, end_time=1.3
    )
    peak.Qual = 95.5
    peak.casno = "123-45-6"
    peak.mol_C = 0.001
    
    d = peak.as_dict()
    assert d["compound_id"] == "P1"
    assert d["Qual"] == 95.5
    assert d["casno"] == "123-45-6"
    assert d["mol_C"] == 0.001
    assert "start_index" not in d  # Should be omitted in new version

def test_serialize_peak_with_numpy_types():
    peak = ChromatographicPeak(
        compound_id="P1", peak_number=1, retention_time=1.23,
        integrator="py", width=0.1, area=500.0,
        start_time=1.1, end_time=1.3
    )
    # Simulate numpy types which often come from data processing
    peak.area = np.float64(500.0)
    peak.peak_number = np.int64(1)
    
    serialized = _serialize_peak(peak)
    assert isinstance(serialized["area"], float)
    assert isinstance(serialized["peak_number"], int)

def test_json_export_integration(tmp_path):
    peak = ChromatographicPeak(
        compound_id="P1", peak_number=1, retention_time=1.23,
        integrator="py", width=0.1, area=500.0,
        start_time=1.1, end_time=1.3
    )
    peak.Qual = 90
    
    d_path = str(tmp_path)
    export_integration_results_to_json(
        [peak], d_path, detector="FID1",
        ms_time_offset=-0.048,
        ms_time_offset_source="manual",
    )
    
    # Find the JSON file - name depends on d_path basename
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    json_file = json_files[0]
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    assert len(data["peaks"]) == 1
    # Note: ChromatographicPeak.as_dict uses 'compound_id' but JSON exporter
    # might have used 'Compound ID' in previous versions. 
    # Let's see what as_dict() actually returns for P1.
    assert data["peaks"][0]["compound_id"] == "P1"
    assert data["peaks"][0]["Qual"] == 90.0
    assert data["ms_time_offset"] == pytest.approx(-0.048)
    assert data["ms_time_offset_source"] == "manual"

def test_csv_export_integration(tmp_path):
    peak = ChromatographicPeak(
        compound_id="P1", peak_number=1, retention_time=1.23,
        integrator="py", width=0.1, area=500.0,
        start_time=1.1, end_time=1.3
    )
    
    csv_path = str(tmp_path / "test.csv")
    success = export_results_to_csv([peak], csv_path)
    assert success
    assert os.path.exists(csv_path)
    
    with open(csv_path, "r") as f:
        content = f.read()
    assert "compound_id,peak_number,retention_time" in content
    assert "P1,1,1.23" in content
