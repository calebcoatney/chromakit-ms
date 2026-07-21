"""Regression: verbose integration must not assume chromatography-only fields.

integrate(..., verbose=True) prints an integration summary. Its tabulate branch
uses peak.as_row() (polymorphic), but the ImportError fallback historically
printed peak.peak_number / peak.retention_time directly — attributes that only
ChromatographicPeak has. When integrating FTIR/UV-Vis data (SpectralFeature)
and tabulate is not installed, that fallback raised AttributeError.

This bit the real /api/run FTIR path on synthon (no tabulate installed) once the
signal profile was correctly threaded through, so bands became SpectralFeature.
"""
import sys
import os
import builtins

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from logic.processor import ChromatogramProcessor
from logic.signal_profiles import SignalProfileRegistry
from logic.method import ChromaMethod
from api.utils import convert_params_for_processor


def _ftir_processed():
    """Process a synthetic multi-band IR spectrum with the ftir profile."""
    profile = SignalProfileRegistry.get("ftir")
    # Descending wavenumber axis like a real ReactIR export, with a few bands.
    x = np.arange(4000.0, 400.0, -2.0)
    y = np.full_like(x, 0.1)
    for center in (2900.0, 1988.0, 1450.0, 1050.0):
        y = y + 4.0 * np.exp(-((x - center) ** 2) / (2.0 * 12.0 ** 2))

    proc = ChromatogramProcessor()
    method = ChromaMethod(
        name="ir", version="1", signal_type="ftir",
        chemstation_area_factor=1.0,
        smoothing={"enabled": True, "window_length": 11, "polyorder": 3},
        baseline={"method": "asls"},
        peaks={"enabled": True, "min_prominence": 0.01,
               "peak_prominence": 0.01, "min_height": 0.02},
        deconvolution={"enabled": False},
        negative_peaks={"enabled": False},
        shoulders={"enabled": False},
    )
    params = convert_params_for_processor(method.to_processor_params())
    processed = proc.process(x, y, params=params, profile=profile)
    return proc, processed, profile


def test_verbose_integrate_ftir_without_tabulate(monkeypatch):
    """integrate(verbose=True) on SpectralFeature peaks must not crash when
    tabulate is unavailable (forces the plain-text fallback path)."""
    proc, processed, profile = _ftir_processed()

    # Force the ImportError fallback branch regardless of whether tabulate is installed.
    real_import = builtins.__import__

    def _no_tabulate(name, *args, **kwargs):
        if name == "tabulate":
            raise ImportError("forced: tabulate unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tabulate)

    # Must not raise AttributeError on SpectralFeature (no peak_number/retention_time).
    result = proc.integrate_peaks(
        processed_data=processed,
        rt_table=None,
        chemstation_area_factor=1.0,
        peak_groups=[],
        profile=profile,
    )
    peaks = result.get("peaks", [])
    assert peaks, "expected at least one detected band"
    # Confirm we really exercised the SpectralFeature path.
    assert type(peaks[0]).__name__ == "SpectralFeature"
