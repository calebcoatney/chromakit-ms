"""Tests for RT-assign + RF-quant wired into POST /api/run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app
from logic.method import ChromaMethod, RTTableEntry, RFTableEntry
from logic.integration import ChromatographicPeak


@pytest.fixture
def client():
    return TestClient(app)


def _method_file(tmp_path, quant_strategy=None, with_tables=False, rf_unit=None):
    kw = {"name": "RAPIDS", "signal_type": "gc", "quant_strategy": quant_strategy}
    if rf_unit is not None:
        kw["rf_unit"] = rf_unit
    m = ChromaMethod(**kw)
    if with_tables:
        m.rt_table = [
            RTTableEntry(compound="Hydrogen", start=1.0, apex=1.1, end=1.2),
            RTTableEntry(compound="Carbon monoxide", start=2.0, apex=2.1, end=2.2),
        ]
        m.rf_table = [
            RFTableEntry(compound="Hydrogen", response_factor=100.0),
            RFTableEntry(compound="Carbon monoxide", response_factor=200.0),
        ]
    path = tmp_path / "m.chromethod"
    m.to_file(str(path))
    return str(path)


def _fake_peaks():
    return [
        ChromatographicPeak("Unknown (1.100)", 1, 1.1, "BB", 0.1, 1000.0, 1.05, 1.15),
        ChromatographicPeak("Unknown (2.100)", 2, 2.1, "BB", 0.1, 2000.0, 2.05, 2.15),
    ]


def _fake_data():
    return {"chromatogram": {"x": list(np.linspace(0, 5, 100)),
                             "y": list(np.zeros(100))}}


def test_run_with_rf_strategy_quantitates(client, tmp_path):
    method_path = _method_file(tmp_path, quant_strategy="rf_table", with_tables=True)
    with patch("api.main.data_handler.load_data_directory", return_value=_fake_data()), \
         patch("api.main.data_handler.current_detector", "FID1A", create=True), \
         patch("api.main.processor.process", return_value={}), \
         patch("api.main.processor.integrate_peaks",
               return_value={"peaks": _fake_peaks()}):
        resp = client.post("/api/run", json={
            "data_path": "/fake.D", "method_path": method_path, "write_output": False,
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quantitation"] is not None
    assert body["quantitation"]["strategy"] == "rf_table"
    assert body["quantitation"]["peaks_quantitated"] == 2
    mol = sorted(p["mol_percent"] for p in body["peaks"])
    assert mol == pytest.approx([50.0, 50.0])


def test_run_legacy_method_no_quant_block(client, tmp_path):
    method_path = _method_file(tmp_path, quant_strategy=None, with_tables=False)
    with patch("api.main.data_handler.load_data_directory", return_value=_fake_data()), \
         patch("api.main.data_handler.current_detector", "FID1A", create=True), \
         patch("api.main.processor.process", return_value={}), \
         patch("api.main.processor.integrate_peaks",
               return_value={"peaks": _fake_peaks()}):
        resp = client.post("/api/run", json={
            "data_path": "/fake.D", "method_path": method_path, "write_output": False,
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quantitation"] is None       # backward-compat: no quant ran


def test_run_nothing_quantitated_returns_200_with_warnings(client, tmp_path):
    # RF strategy is on, but the RT table windows don't cover the fake peaks'
    # RTs (1.1, 2.1), so nothing gets assigned/quantitated. Spec §3e: return 200
    # with a loud warning, not an error.
    m = ChromaMethod(name="RAPIDS", signal_type="gc", quant_strategy="rf_table")
    m.rt_table = [RTTableEntry(compound="Methane", start=8.0, apex=8.1, end=8.2)]
    m.rf_table = [RFTableEntry(compound="Methane", response_factor=100.0)]
    method_path = str(tmp_path / "m.chromethod")
    m.to_file(method_path)

    with patch("api.main.data_handler.load_data_directory", return_value=_fake_data()), \
         patch("api.main.data_handler.current_detector", "FID1A", create=True), \
         patch("api.main.processor.process", return_value={}), \
         patch("api.main.processor.integrate_peaks",
               return_value={"peaks": _fake_peaks()}):
        resp = client.post("/api/run", json={
            "data_path": "/fake.D", "method_path": method_path, "write_output": False,
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quantitation"] is not None
    assert body["quantitation"]["peaks_quantitated"] == 0
    assert len(body["quantitation"]["warnings"]) >= 1


def test_run_with_rf_unit_exposes_basis_and_composition(client, tmp_path):
    # RF strategy with a real unit → the summary must carry composition_basis
    # ("wt%") and rf_unit, and each quantitated peak carries composition_percent
    # so RAPIDS/Aspen consumes an unambiguous feed.
    method_path = _method_file(
        tmp_path, quant_strategy="rf_table", with_tables=True,
        rf_unit="area_per_wt_pct",
    )
    with patch("api.main.data_handler.load_data_directory", return_value=_fake_data()), \
         patch("api.main.data_handler.current_detector", "FID1A", create=True), \
         patch("api.main.processor.process", return_value={}), \
         patch("api.main.processor.integrate_peaks",
               return_value={"peaks": _fake_peaks()}):
        resp = client.post("/api/run", json={
            "data_path": "/fake.D", "method_path": method_path, "write_output": False,
        })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["quantitation"]["composition_basis"] == "wt%"
    assert body["quantitation"]["rf_unit"] == "area_per_wt_pct"
    assert any("composition_percent" in pk for pk in body["peaks"])


def test_run_legacy_rf_unit_unspecified_null_basis_with_warning(client, tmp_path):
    # RF strategy runs and quantitates, but no rf_unit was set (defaults to
    # "unspecified") → composition_basis is null and a loud "unspecified"
    # warning is surfaced.
    method_path = _method_file(
        tmp_path, quant_strategy="rf_table", with_tables=True,
    )
    with patch("api.main.data_handler.load_data_directory", return_value=_fake_data()), \
         patch("api.main.data_handler.current_detector", "FID1A", create=True), \
         patch("api.main.processor.process", return_value={}), \
         patch("api.main.processor.integrate_peaks",
               return_value={"peaks": _fake_peaks()}):
        resp = client.post("/api/run", json={
            "data_path": "/fake.D", "method_path": method_path, "write_output": False,
        })
    assert resp.status_code == 200, resp.text
    body_legacy = resp.json()
    assert body_legacy["quantitation"]["composition_basis"] is None
    assert any("unspecified" in w.lower() for w in body_legacy["quantitation"]["warnings"])
