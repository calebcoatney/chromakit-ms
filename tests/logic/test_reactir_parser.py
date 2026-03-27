"""Tests for the Mettler Toledo ReactIR CSV parser."""
import numpy as np
import pandas as pd
import pytest
from logic.loaders.reactir_parser import parse_reactir_csv
from logic.c_folder import CFolder


def _make_reactir_csv(path):
    """Two-column headerless CSV: wavenumber, absorbance."""
    df = pd.DataFrame({0: [1000.0, 1001.0, 1002.0], 1: [0.10, 0.20, 0.30]})
    df.to_csv(path, index=False, header=False)


def test_parse_reactir_csv_creates_c_folder(tmp_path):
    csv_path = tmp_path / "reactir_sample.csv"
    _make_reactir_csv(str(csv_path))

    result = parse_reactir_csv(str(csv_path))

    assert (tmp_path / "reactir_sample.C").is_dir()
    assert isinstance(result, CFolder)


def test_parse_reactir_csv_manifest(tmp_path):
    csv_path = tmp_path / "reactir_sample.csv"
    _make_reactir_csv(str(csv_path))

    result = parse_reactir_csv(str(csv_path))
    manifest = result.get_manifest()

    assert manifest["signal_type"] == "ftir"
    assert manifest["instrument"] == "Mettler Toledo ReactIR"
    assert manifest["csv_columns"]["x_column"] == 0
    assert manifest["csv_columns"]["y_column"] == 1
    assert manifest["csv_columns"]["has_header"] is False


def test_parse_reactir_csv_data_loadable(tmp_path):
    csv_path = tmp_path / "reactir_sample.csv"
    _make_reactir_csv(str(csv_path))

    result = parse_reactir_csv(str(csv_path))
    signal = result.load_signal()

    np.testing.assert_allclose(signal["x"], [1000.0, 1001.0, 1002.0])
    np.testing.assert_allclose(signal["y"], [0.10, 0.20, 0.30])


def test_parse_reactir_csv_moves_source(tmp_path):
    csv_path = tmp_path / "reactir_sample.csv"
    _make_reactir_csv(str(csv_path))

    parse_reactir_csv(str(csv_path))

    assert not csv_path.exists()
    assert (tmp_path / "reactir_sample.C" / "data" / "reactir_sample.csv").exists()
