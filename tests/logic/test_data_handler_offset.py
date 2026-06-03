"""Tests for DataHandler.ms_time_offset integration."""
import inspect

import pytest

from logic.data_handler import DataHandler
from logic.sidecar_offsets import save_offset


def test_default_offset_is_zero():
    dh = DataHandler()
    assert dh.ms_time_offset == 0.0


def test_load_ms_time_offset_from_sidecar(tmp_path, monkeypatch):
    sidecar = tmp_path / "ms_time_offsets.json"
    save_offset("/abs/sample.D", -0.048, source="manual", sidecar_path=sidecar)
    dh = DataHandler()
    monkeypatch.setattr("logic.data_handler.DEFAULT_OFFSET_SIDECAR", sidecar)
    dh.apply_offset_from_sidecar("/abs/sample.D")
    assert dh.ms_time_offset == pytest.approx(-0.048)


def test_apply_offset_from_sidecar_when_missing_resets_to_zero(tmp_path, monkeypatch):
    sidecar = tmp_path / "ms_time_offsets.json"
    dh = DataHandler()
    dh.ms_time_offset = -0.048
    monkeypatch.setattr("logic.data_handler.DEFAULT_OFFSET_SIDECAR", sidecar)
    dh.apply_offset_from_sidecar("/abs/nonexistent.D")
    assert dh.ms_time_offset == 0.0


def test_extract_spectrum_at_rt_signature_drops_aligned_tic_data():
    """The old vestigial `aligned_tic_data` kwarg must be removed."""
    sig = inspect.signature(DataHandler.extract_spectrum_at_rt)
    assert "aligned_tic_data" not in sig.parameters
