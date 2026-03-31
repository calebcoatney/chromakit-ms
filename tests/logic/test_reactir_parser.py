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


def test_parse_reactir_csv_timestamp_from_filename(tmp_path):
    """Timestamp embedded in the filename is written to manifest['created']."""
    filename = "260324_MoCO6_2026-03-24_12-27-11_Spectrum.csv"
    csv_path = tmp_path / filename
    _make_reactir_csv(str(csv_path))

    result = parse_reactir_csv(str(csv_path))
    manifest = result.get_manifest()

    assert manifest["created"] == "2026-03-24T12:27:11"


def test_parse_reactir_csv_explicit_timestamp_overrides_filename(tmp_path):
    """Explicit sample_timestamp kwarg takes priority over any filename pattern."""
    filename = "260324_MoCO6_2026-03-24_12-27-11_Spectrum.csv"
    csv_path = tmp_path / filename
    _make_reactir_csv(str(csv_path))

    result = parse_reactir_csv(str(csv_path), sample_timestamp="2099-01-01T00:00:00")
    manifest = result.get_manifest()

    assert manifest["created"] == "2099-01-01T00:00:00"


def test_parse_reactir_csv_fallback_to_now_when_no_timestamp(tmp_path):
    """Files without a timestamp pattern fall back to current wall-clock time."""
    import datetime, re
    csv_path = tmp_path / "reactir_no_ts.csv"
    _make_reactir_csv(str(csv_path))

    before = datetime.datetime.now()
    result = parse_reactir_csv(str(csv_path))
    after = datetime.datetime.now()

    created = datetime.datetime.fromisoformat(result.get_manifest()["created"])
    assert before <= created <= after
