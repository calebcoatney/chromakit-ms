"""RT-table compound matching — pure logic extracted from ui/frames/rt_table.py.

Three modes ported verbatim from the GUI's lookup methods so headless
assignments (api/) match the GUI exactly. No Qt. DataFrame + params in,
compound-name-or-None out.

Layer rule: logic/ must NOT import from ui/ or api/.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from logic.method import RTMatchingParams


def lookup_compound_by_rt(
    retention_time: float,
    rt_table: pd.DataFrame,           # columns: Compound, Start, Apex, End
    params: RTMatchingParams,
) -> Optional[str]:
    """Return the matched compound name for one RT, or None."""
    if rt_table is None or len(rt_table) == 0:
        return None
    mode = params.matching_mode
    if mode == 0:
        return _lookup_simple_window(retention_time, rt_table, params.window_expansion)
    elif mode == 1:
        return _lookup_closest_apex(retention_time, rt_table, params.tolerance)
    elif mode == 2:
        return _lookup_weighted_distance(retention_time, rt_table, params.weights)
    return None


def _lookup_simple_window(retention_time, rt_table, expansion):
    matches = rt_table[
        (rt_table["Start"] - expansion <= retention_time)
        & (retention_time <= rt_table["End"] + expansion)
    ]
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches.iloc[0]["Compound"]
    else:
        widths = matches["End"] - matches["Start"]
        best = matches.loc[widths.idxmin()]
        return best["Compound"]


def _lookup_closest_apex(retention_time, rt_table, tolerance):
    distances = np.abs(rt_table["Apex"] - retention_time)
    within = distances <= tolerance
    if not within.any():
        return None
    closest_idx = distances[within].idxmin()
    return rt_table.loc[closest_idx, "Compound"]


def _lookup_weighted_distance(retention_time, rt_table, weights):
    w = {"start": weights.start, "apex": weights.apex, "end": weights.end}
    rt_range = rt_table["End"].max() - rt_table["Start"].min()
    left_bound = rt_table["Start"].min()
    right_bound = rt_table["End"].max()
    compound_windows = rt_table["End"] - rt_table["Start"]
    avg_window_width = compound_windows.mean()
    boundary_tolerance = min(max(avg_window_width, rt_range * 0.05), 1.0)
    if (retention_time < (left_bound - boundary_tolerance)
            or retention_time > (right_bound + boundary_tolerance)):
        return None
    distances = []
    for _, row in rt_table.iterrows():
        start_dist = abs(row["Start"] - retention_time)
        apex_dist = abs(row["Apex"] - retention_time)
        end_dist = abs(row["End"] - retention_time)
        distances.append(w["start"] * start_dist + w["apex"] * apex_dist + w["end"] * end_dist)
    min_idx = int(np.argmin(distances))
    best = rt_table.iloc[min_idx]
    window_width = best["End"] - best["Start"]
    max_dist_from_window = window_width * 0.75
    dist_from_window = 0.0
    if retention_time < best["Start"]:
        dist_from_window = best["Start"] - retention_time
    elif retention_time > best["End"]:
        dist_from_window = retention_time - best["End"]
    if dist_from_window <= max_dist_from_window:
        return best["Compound"]
    return None


def apply_rt_matching(
    peaks: list,
    rt_table: pd.DataFrame,
    params: RTMatchingParams,
) -> None:
    """Assign peak.compound_id (and peak.Compound_ID when present) in place.

    Mirrors ui/app.py::_apply_rt_matching_to_peaks verbatim: high-priority
    overrides any existing assignment; low-priority fills only Unknown/None;
    sets rt_assignment / rt_assignment_source, clears Qual/casno/CAS_Number;
    matched-but-not-applied peaks record rt_match_available. No dedup — the GUI
    method performs none, and allow_duplicates is not consumed by matching.
    """
    if rt_table is None or len(rt_table) == 0:
        return

    for peak in peaks:
        rt_compound = lookup_compound_by_rt(peak.retention_time, rt_table, params)
        if not rt_compound:
            continue

        should_apply = False
        assignment_source = None
        if params.high_priority:
            should_apply = True
            assignment_source = "RT (priority)"
        else:
            current = getattr(peak, "compound_id", "Unknown")
            unknown_str = f"Unknown ({peak.retention_time:.3f})"
            if current in ("Unknown", unknown_str, None):
                should_apply = True
                assignment_source = "RT"

        if should_apply:
            peak.compound_id = rt_compound
            if hasattr(peak, "Compound_ID"):
                peak.Compound_ID = rt_compound
            peak.rt_assignment = True
            peak.rt_assignment_source = assignment_source
            peak.Qual = None
            if hasattr(peak, "casno"):
                peak.casno = None
            if hasattr(peak, "CAS_Number"):
                peak.CAS_Number = None
        else:
            peak.rt_match_available = rt_compound
