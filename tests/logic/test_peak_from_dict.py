"""Tests for ChromatographicPeak.from_dict() — inverse of as_dict().

Round-trip tests are the cheapest insurance against as_dict/from_dict drift.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from logic.integration import ChromatographicPeak, Peak


def _make_peak() -> ChromatographicPeak:
    """Build a fully-populated peak — all fields set, MS + quant + deconv enriched."""
    peak = ChromatographicPeak(
        compound_id="benzene",
        peak_number=3,
        retention_time=4.521,
        integrator="BB",
        width=0.082,
        area=12345.6,
        start_time=4.480,
        end_time=4.562,
        start_index=448,
        end_index=456,
    )
    # MS match fields (set by BatchSearchWorker)
    peak.Compound_ID = "Benzene"
    peak.Qual = 0.91
    peak.casno = "00071-43-2"
    peak.compound_name = "Benzene"
    peak.match_score = 0.91
    # Quality flags
    peak.is_shoulder = False
    peak.is_negative = False
    peak.is_convoluted = True
    peak.is_saturated = False
    peak.is_grouped = False
    peak.quality_issues = ["convoluted_with_neighbor"]
    peak.asymmetry = 1.12
    peak.spectral_coherence = 0.85
    peak.saturation_level = None
    peak.grouped_peak_count = None
    # Quantitation
    peak.mol_C = 1.23e-7
    peak.num_carbons = 6
    peak.mol = 2.05e-8
    peak.mass_mg = 0.0016
    peak.mol_C_percent = 12.3
    peak.mol_percent = 8.4
    peak.wt_percent = 9.1
    # Deconvolved spectrum
    peak.deconvolved_spectrum = {
        'mz': np.array([50.0, 51.0, 77.0, 78.0]),
        'intensities': np.array([100.0, 80.0, 1000.0, 250.0]),
    }
    peak.deconvolution_component_count = 1
    return peak


def test_peak_alias_is_chromatographic_peak():
    """`Peak` and `ChromatographicPeak` are the same class."""
    assert Peak is ChromatographicPeak


def test_from_dict_round_trip_preserves_all_fields():
    """as_dict() -> from_dict() preserves every field that as_dict() emits."""
    original = _make_peak()
    d = original.as_dict()
    restored = ChromatographicPeak.from_dict(d)

    # Always-present fields
    assert restored.compound_id == "Benzene"  # as_dict prefers Compound_ID
    assert restored.peak_number == original.peak_number
    assert restored.retention_time == original.retention_time
    assert restored.integrator == original.integrator
    assert restored.width == original.width
    assert restored.area == original.area
    assert restored.start_time == original.start_time
    assert restored.end_time == original.end_time
    # Quality flags
    assert restored.is_shoulder is False
    assert restored.is_negative is False
    assert restored.is_convoluted is True
    assert restored.is_saturated is False
    assert restored.is_grouped is False
    assert restored.quality_issues == ["convoluted_with_neighbor"]
    # MS match
    assert restored.Qual == 0.91
    assert restored.casno == "00071-43-2"
    assert restored.compound_name == "Benzene"
    assert restored.match_score == 0.91
    # Quality detail
    assert restored.asymmetry == 1.12
    assert restored.spectral_coherence == 0.85
    # Quant
    assert restored.mol_C == 1.23e-7
    assert restored.num_carbons == 6
    assert restored.mol == 2.05e-8
    assert restored.mass_mg == 0.0016
    assert restored.mol_C_percent == 12.3
    assert restored.mol_percent == 8.4
    assert restored.wt_percent == 9.1


def test_from_dict_restores_deconvolved_spectrum_as_arrays():
    """deconvolved_spectrum round-trips as numpy arrays, not lists.

    Downstream consumers (logic/spectral_deconv_runner._assign_components_to_peaks)
    require np.ndarray, not list.
    """
    original = _make_peak()
    # When as_dict() is called, deconvolved_spectrum may be serialized as lists
    # downstream (e.g. through JSON). Simulate that round-trip:
    import json
    d = original.as_dict()
    # The spectrum is not actually emitted by as_dict (Peak.as_dict drops it).
    # So we attach it manually as a dict-of-lists, which is how an API payload
    # would carry it.
    d['deconvolved_spectrum'] = {
        'mz': [50.0, 51.0, 77.0, 78.0],
        'intensities': [100.0, 80.0, 1000.0, 250.0],
    }
    d_via_json = json.loads(json.dumps(d))

    restored = ChromatographicPeak.from_dict(d_via_json)

    assert isinstance(restored.deconvolved_spectrum['mz'], np.ndarray)
    assert isinstance(restored.deconvolved_spectrum['intensities'], np.ndarray)
    assert restored.deconvolved_spectrum['mz'].tolist() == [50.0, 51.0, 77.0, 78.0]
    assert restored.deconvolved_spectrum['intensities'].tolist() == [100.0, 80.0, 1000.0, 250.0]


def test_from_dict_tolerates_minimal_dict():
    """Partial dicts (missing optional fields) round-trip with sensible defaults."""
    minimal = {
        'compound_id': 'unknown',
        'peak_number': 1,
        'retention_time': 2.5,
        'integrator': 'BB',
        'width': 0.05,
        'area': 1000.0,
        'start_time': 2.45,
        'end_time': 2.55,
    }
    peak = ChromatographicPeak.from_dict(minimal)
    assert peak.compound_id == 'unknown'
    assert peak.retention_time == 2.5
    # Missing optional fields take their __init__ defaults
    assert peak.is_saturated is False
    assert peak.mol_C is None
    assert peak.compound_name is None
    assert peak.deconvolved_spectrum is None


def test_from_dict_handles_compound_id_from_either_key():
    """as_dict emits 'compound_id' (prefers Compound_ID if set). from_dict reads it back."""
    d = {
        'compound_id': 'Hexane',
        'peak_number': 2,
        'retention_time': 3.0,
        'integrator': 'BB',
        'width': 0.05,
        'area': 500.0,
        'start_time': 2.95,
        'end_time': 3.05,
    }
    peak = ChromatographicPeak.from_dict(d)
    assert peak.compound_id == 'Hexane'
