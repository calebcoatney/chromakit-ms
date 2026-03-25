# tests/test_chromatic_peak_fields.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logic.integration import ChromatographicPeak


def test_deconvolved_spectrum_defaults_to_none():
    peak = ChromatographicPeak(
        compound_id='test', peak_number=1, retention_time=1.0,
        integrator='test', width=0.1, area=1000.0,
        start_time=0.95, end_time=1.05
    )
    assert peak.deconvolved_spectrum is None


def test_deconvolution_component_count_defaults_to_none():
    peak = ChromatographicPeak(
        compound_id='test', peak_number=1, retention_time=1.0,
        integrator='test', width=0.1, area=1000.0,
        start_time=0.95, end_time=1.05
    )
    assert peak.deconvolution_component_count is None
