"""Tests for api/models.py — RunResponse Pydantic shape.

Added as part of the cross-repo spectro-bridge payload-enrichment slice
(spec lives in the vault at
docs/superpowers/specs/2026-06-04-spectro-bridge-payload-enrichment-design.md).
This test pins the contract the spectro_bridge depends on.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from pydantic import ValidationError

from api.models import RunResponse


def _base_kwargs() -> dict:
    """Minimal valid RunResponse kwargs, missing only `version`."""
    return dict(
        status="complete",
        data_path="/tmp/sample.D",
        method="ir_nanoparticle",
        signal_type="ftir",
        peak_count=0,
        peaks=[],
        output_files=["/tmp/out.json"],
    )


def test_run_response_requires_version():
    """RunResponse must reject construction without an explicit `version`."""
    with pytest.raises(ValidationError) as excinfo:
        RunResponse(**_base_kwargs())
    # The error must mention the missing `version` field.
    assert "version" in str(excinfo.value)


def test_run_response_accepts_string_version():
    """`version` is a required field that accepts any string value, e.g. `"1"`."""
    resp = RunResponse(version="1", **_base_kwargs())
    assert resp.version == "1"


def test_run_response_round_trip_with_version():
    """`version` survives a model_dump → model_validate round-trip."""
    resp = RunResponse(version="2", **_base_kwargs())
    dumped = resp.model_dump()
    assert dumped["version"] == "2"
    restored = RunResponse.model_validate(dumped)
    assert restored.version == "2"


def test_run_response_version_appears_in_json():
    """`version` is included in the JSON serialization sent over the wire."""
    resp = RunResponse(version="3", **_base_kwargs())
    payload = resp.model_dump_json()
    assert '"version":"3"' in payload or '"version": "3"' in payload
