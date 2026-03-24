import pytest
import os
import numpy as np
from logic.loaders.csv_loader import CSVLoader
from logic.json_exporter import metadata_from_manifest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_c_folder_with_csv(tmp_path, x_col, y_col, rows):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "signal.csv"
    lines = [f"{x_col},{y_col}"]
    for x, y in rows:
        lines.append(f"{x},{y}")
    csv_path.write_text("\n".join(lines))
    return str(tmp_path)


# ── CSVLoader ─────────────────────────────────────────────────────────────────

def test_csv_loader_basic(tmp_path):
    rows = [(100.0, 0.5), (200.0, 1.2), (300.0, 0.8)]
    c_path = _make_c_folder_with_csv(tmp_path, "wavenumber", "absorbance", rows)

    loader = CSVLoader(x_column="wavenumber", y_column="absorbance")
    result = loader.load(c_path)

    assert "x" in result and "y" in result and "metadata" in result
    np.testing.assert_array_almost_equal(result["x"], [100.0, 200.0, 300.0])
    np.testing.assert_array_almost_equal(result["y"], [0.5, 1.2, 0.8])


def test_csv_loader_guaranteed_metadata_keys(tmp_path):
    rows = [(1.0, 2.0)]
    c_path = _make_c_folder_with_csv(tmp_path, "wn", "abs", rows)
    result = CSVLoader(x_column="wn", y_column="abs").load(c_path)

    assert result["metadata"]["has_ms_data"] is False
    assert "filename" in result["metadata"]


def test_csv_loader_missing_column_raises(tmp_path):
    rows = [(1.0, 2.0)]
    c_path = _make_c_folder_with_csv(tmp_path, "wn", "abs", rows)
    loader = CSVLoader(x_column="wavelength", y_column="abs")  # wrong x col
    with pytest.raises((KeyError, ValueError)):
        loader.load(c_path)


def test_csv_loader_no_csv_raises(tmp_path):
    (tmp_path / "data").mkdir()
    loader = CSVLoader(x_column="x", y_column="y")
    with pytest.raises(FileNotFoundError):
        loader.load(str(tmp_path))


# ── metadata_from_manifest ────────────────────────────────────────────────────

def test_metadata_from_manifest_basic():
    manifest = {
        "signal_type": "ftir",
        "source_format": "csv",
        "sample_id": "RXN-01",
        "instrument": "ReactIR",
        "created": "2026-03-24T10:00:00",
    }
    result = metadata_from_manifest(manifest)
    assert result["signal_type"] == "ftir"
    assert result["sample_id"] == "RXN-01"
    assert result["instrument"] == "ReactIR"


def test_metadata_from_manifest_missing_keys():
    """Missing optional keys return empty string, not KeyError."""
    result = metadata_from_manifest({"signal_type": "gc"})
    assert result["sample_id"] == ""
    assert result["instrument"] == ""
