"""Tests for logic/rt_matching.py — extracted 3-mode RT matcher."""
import pandas as pd
import pytest

from logic.rt_matching import lookup_compound_by_rt, apply_rt_matching
from logic.method import RTMatchingParams, RTMatchingWeights
from logic.integration import ChromatographicPeak


def _df():
    return pd.DataFrame(
        [
            ["Hydrogen", 1.0, 1.1, 1.2],
            ["Carbon monoxide", 2.0, 2.1, 2.2],
            ["Methanol", 3.0, 3.1, 3.2],
        ],
        columns=["Compound", "Start", "Apex", "End"],
    )


# ---- Simple Window (mode 0) ----
def test_simple_window_inside():
    p = RTMatchingParams(matching_mode=0)
    assert lookup_compound_by_rt(2.1, _df(), p) == "Carbon monoxide"

def test_simple_window_outside_returns_none():
    p = RTMatchingParams(matching_mode=0)
    assert lookup_compound_by_rt(5.0, _df(), p) is None

def test_simple_window_expansion_widens():
    p = RTMatchingParams(matching_mode=0, window_expansion=0.5)
    assert lookup_compound_by_rt(2.5, _df(), p) == "Carbon monoxide"

def test_simple_window_overlap_prefers_narrowest():
    df = pd.DataFrame(
        [["Wide", 1.0, 2.0, 3.0], ["Narrow", 1.9, 2.0, 2.1]],
        columns=["Compound", "Start", "Apex", "End"],
    )
    p = RTMatchingParams(matching_mode=0)
    assert lookup_compound_by_rt(2.0, df, p) == "Narrow"


# ---- Closest Apex (mode 1) ----
def test_closest_apex_within_tolerance():
    p = RTMatchingParams(matching_mode=1, tolerance=0.1)
    assert lookup_compound_by_rt(2.13, _df(), p) == "Carbon monoxide"

def test_closest_apex_outside_tolerance_none():
    p = RTMatchingParams(matching_mode=1, tolerance=0.05)
    assert lookup_compound_by_rt(2.5, _df(), p) is None


# ---- Weighted Distance (mode 2) ----
def test_weighted_distance_inside_window():
    p = RTMatchingParams(matching_mode=2)
    assert lookup_compound_by_rt(2.1, _df(), p) == "Carbon monoxide"

def test_weighted_distance_far_outside_none():
    p = RTMatchingParams(matching_mode=2)
    assert lookup_compound_by_rt(50.0, _df(), p) is None


# ---- apply_rt_matching (Task 4) ----
def _peak(rt, compound_id="Unknown", peak_number=1):
    return ChromatographicPeak(
        compound_id=compound_id, peak_number=peak_number, retention_time=rt,
        integrator="BB", width=0.1, area=1000.0, start_time=rt - 0.05, end_time=rt + 0.05,
    )


def test_apply_assigns_compound_id():
    peaks = [_peak(2.1)]
    apply_rt_matching(peaks, _df(), RTMatchingParams(matching_mode=0))
    assert peaks[0].compound_id == "Carbon monoxide"
    assert peaks[0].rt_assignment is True
    assert peaks[0].rt_assignment_source == "RT"


def test_high_priority_overrides_existing():
    peaks = [_peak(2.1, compound_id="SomeMSHit")]
    peaks[0].Qual = 88.0   # simulate a prior MS-search match score
    p = RTMatchingParams(matching_mode=0, high_priority=True)
    apply_rt_matching(peaks, _df(), p)
    assert peaks[0].compound_id == "Carbon monoxide"
    assert peaks[0].rt_assignment_source == "RT (priority)"
    assert peaks[0].Qual is None   # MS match score cleared on RT override


def test_low_priority_preserves_existing_nonunknown():
    peaks = [_peak(2.1, compound_id="SomeMSHit")]
    p = RTMatchingParams(matching_mode=0, high_priority=False)
    apply_rt_matching(peaks, _df(), p)
    assert peaks[0].compound_id == "SomeMSHit"   # not overridden
    assert getattr(peaks[0], "rt_match_available", None) == "Carbon monoxide"


def test_low_priority_fills_unknown():
    peaks = [_peak(2.1, compound_id="Unknown")]
    p = RTMatchingParams(matching_mode=0, high_priority=False)
    apply_rt_matching(peaks, _df(), p)
    assert peaks[0].compound_id == "Carbon monoxide"


def test_no_match_leaves_peak_untouched():
    peaks = [_peak(9.9, compound_id="Unknown")]   # far outside table
    apply_rt_matching(peaks, _df(), RTMatchingParams(matching_mode=0))
    assert peaks[0].compound_id == "Unknown"
    assert getattr(peaks[0], "rt_assignment", False) is False


def test_apply_rt_matching_skips_spectral_features():
    import pandas as pd
    from logic.rt_matching import apply_rt_matching
    from logic.method import RTMatchingParams
    from logic.feature import SpectralFeature

    sf = SpectralFeature(
        feature_id=1, position=1987.0, position_units="cm-1",
        area=10.0, width=5.0, start=1980.0, end=1994.0,
        start_index=0, end_index=10,
    )
    rt_df = pd.DataFrame(
        [["precursor", 1.0, 1.1, 1.2]],
        columns=["Compound", "Start", "Apex", "End"],
    )
    # Must NOT raise AttributeError on peak.retention_time
    apply_rt_matching([sf], rt_df, RTMatchingParams())
    assert not hasattr(sf, "compound_id") or getattr(sf, "compound_id", None) is None
