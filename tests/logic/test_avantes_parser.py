"""Tests for the Avantes UV-Vis CSV parser."""
import numpy as np
import pandas as pd
import pytest
from logic.loaders.avantes_parser import parse_avantes_uvvis
from logic.c_folder import CFolder


def _make_avantes_csv(path, n_spectra=3, n_wavelengths=5):
    """Build a minimal Avantes-format CSV at *path*.

    Header rows (0-4) are followed by spectral data rows.
    Intensity for spectrum i at wavelength index w = float(w + i * 0.1).
    """
    rows = []
    rows.append(["Integration Time [msec]"] + [100.0] * n_spectra)
    rows.append(["Number of Averages"] + [300] * n_spectra)
    rows.append(["Wavelength [nm]"] + ["A.U."] * n_spectra)
    rows.append(["Timestamp"] + [i * 30000 for i in range(n_spectra)])
    rows.append(["Date/Time"] + [f"26/02/2026 12:2{i}:00" for i in range(n_spectra)])
    for w in range(n_wavelengths):
        rows.append([400.0 + w * 10.0] + [float(w + i * 0.1) for i in range(n_spectra)])
    pd.DataFrame(rows).to_csv(path, index=False, header=False)


def test_parse_avantes_creates_one_folder_per_spectrum(tmp_path):
    csv_path = tmp_path / "avantes_run.csv"
    _make_avantes_csv(str(csv_path), n_spectra=3)

    results = parse_avantes_uvvis(str(csv_path), output_dir=str(tmp_path))

    assert len(results) == 3
    assert (tmp_path / "avantes_run_000.C").is_dir()
    assert (tmp_path / "avantes_run_001.C").is_dir()
    assert (tmp_path / "avantes_run_002.C").is_dir()


def test_parse_avantes_manifest_metadata(tmp_path):
    csv_path = tmp_path / "avantes_run.csv"
    _make_avantes_csv(str(csv_path), n_spectra=2)

    results = parse_avantes_uvvis(str(csv_path), output_dir=str(tmp_path))

    manifest = CFolder.open(results[0]).get_manifest()
    assert manifest["signal_type"] == "uvvis"
    assert manifest["instrument"] == "Avantes"
    assert manifest["integration_time_ms"] == 100.0
    assert manifest["n_averages"] == 300
    assert manifest["sample_timestamp"] == "2026-02-26T12:20:00"
    assert manifest["csv_columns"] == {"x_column": 0, "y_column": 1, "has_header": False}


def test_parse_avantes_timestamps_per_spectrum(tmp_path):
    csv_path = tmp_path / "avantes_run.csv"
    _make_avantes_csv(str(csv_path), n_spectra=3)

    results = parse_avantes_uvvis(str(csv_path), output_dir=str(tmp_path))

    assert CFolder.open(results[0]).get_manifest()["sample_timestamp"] == "2026-02-26T12:20:00"
    assert CFolder.open(results[1]).get_manifest()["sample_timestamp"] == "2026-02-26T12:21:00"
    assert CFolder.open(results[2]).get_manifest()["sample_timestamp"] == "2026-02-26T12:22:00"


def test_parse_avantes_data_content(tmp_path):
    csv_path = tmp_path / "avantes_run.csv"
    _make_avantes_csv(str(csv_path), n_spectra=2, n_wavelengths=5)

    results = parse_avantes_uvvis(str(csv_path), output_dir=str(tmp_path))

    signal = CFolder.open(results[0]).load_signal()
    np.testing.assert_allclose(signal["x"], [400.0, 410.0, 420.0, 430.0, 440.0])
    np.testing.assert_allclose(signal["y"], [0.0, 1.0, 2.0, 3.0, 4.0])

    signal2 = CFolder.open(results[1]).load_signal()
    np.testing.assert_allclose(signal2["y"], [0.1, 1.1, 2.1, 3.1, 4.1], atol=1e-9)


def test_parse_avantes_default_output_dir(tmp_path):
    csv_path = tmp_path / "avantes_run.csv"
    _make_avantes_csv(str(csv_path), n_spectra=2)

    results = parse_avantes_uvvis(str(csv_path))  # no output_dir

    assert (tmp_path / "avantes_run_000.C").is_dir()
    assert (tmp_path / "avantes_run_001.C").is_dir()
    assert len(results) == 2


def test_parse_avantes_skips_existing_folders(tmp_path):
    csv_path = tmp_path / "avantes_run.csv"
    _make_avantes_csv(str(csv_path), n_spectra=2)

    # First call creates the folders
    parse_avantes_uvvis(str(csv_path), output_dir=str(tmp_path))

    # Second call with a fresh copy of the source — should not raise
    csv_path2 = tmp_path / "avantes_run_copy.csv"
    _make_avantes_csv(str(csv_path2), n_spectra=2)

    # Rename source to collide with existing folders
    import shutil
    csv_path3 = tmp_path / "avantes_run_again.csv"
    shutil.copy(str(csv_path2), str(csv_path3))

    # Directly test that FileExistsError is handled: create the first target folder manually
    import os
    collision = tmp_path / "avantes_run_again_000.C"
    os.makedirs(collision / "data")
    (collision / "manifest.json").write_text('{"signal_type":"uvvis","source_format":"csv"}')

    results = parse_avantes_uvvis(str(csv_path3), output_dir=str(tmp_path))
    # Should return 2 paths despite first folder already existing
    assert len(results) == 2
