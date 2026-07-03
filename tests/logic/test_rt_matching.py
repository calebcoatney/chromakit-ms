"""Tests for logic/rt_matching.py — extracted 3-mode RT matcher."""
import pandas as pd
import pytest

from logic.rt_matching import lookup_compound_by_rt
from logic.method import RTMatchingParams, RTMatchingWeights


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
