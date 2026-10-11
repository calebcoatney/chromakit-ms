# Portions of this file are ported from mzmine (https://github.com/mzmine/mzmine):
#   io.github.mzmine.modules.dataprocessing.featdet_spectraldeconvolutiongc.rtgroupingandsharecorrelation.RtGroupingAndShapeCorrelationAlgorithm
#   io.github.mzmine.modules.dataprocessing.featdet_spectraldeconvolutiongc.SpectralDeconvolutionUtils
#
# Copyright (c) 2004-2025 The mzmine Development Team
# Copyright (c) 2004-2024 The MZmine Development Team
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
"""GC-EI pseudo spectra by RT grouping and peak-shape correlation.

Python port of mzmine 4's "RT grouping and shape correlation" spectral
deconvolution. Ion peaks are taken strongest first; each joins the
best-correlated group whose representative (highest ion) lies within the RT
tolerance, if Pearson r >= min_r; an ion with candidate groups but no good
correlation is discarded (as in mzmine), one with no candidate starts a group.
A component's spectrum is each member ion's peak height.

Deliberate changes from mzmine: r needs >= 3 shared scans (else 0), and an
excluded m/z value m excludes [m - 0.5, m + 0.5] (mzmine widens m to
[m, m + 1], which at unit mass would also exclude m + 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from logic.ion_peak_resolver import IonPeak, resolve_ion_peaks

EXCLUDED_MZ_TOLERANCE = 0.5


@dataclass
class CorrelationDeconvParams:
    """Settings for ion peak detection and correlation grouping."""
    noise_percentile: float = 0.85    # whole-run percentile below which points are ignored
    search_range: float = 0.03        # min; a dip lowest within ± this splits two peaks
    min_peak_height: float = 1000.0   # counts
    min_top_edge_ratio: float = 1.7
    min_points: int = 4
    rt_tolerance: float = 0.04        # min; |apex RT - representative apex RT|
    min_r: float = 0.80               # Pearson r to join a group
    min_ions: int = 4                 # smaller groups are dropped
    excluded_mz: list = field(default_factory=list)  # can't be a group's lead ion


@dataclass
class IonGroup:
    members: list           # IonPeak; members[0] is the representative (highest)
    r: list                 # r of members[i] to members[0]; r[0] == 1.0
    lead: IonPeak           # highest member not excluded (else members[0])


@dataclass
class RejectedIon:
    peak: IonPeak
    reason: str             # "shape" or "small_group"


@dataclass(eq=False)   # identity equality: default __eq__ compares ndarray fields and raises
class DeconvolutedComponent:
    """One deconvolved analyte: RT, spectrum {m/z: height} and its lead-ion profile."""
    rt: float
    spectrum: dict
    model_peak_mz: float
    model_peak_rt_array: np.ndarray
    model_peak_intensity_array: np.ndarray


def pearson_r(a: IonPeak, b: IonPeak) -> float:
    """Pearson r of two ion peaks over the scans both cover (calculateCorrelation)."""
    lo = max(a.start_idx, b.start_idx)
    hi = min(a.end_idx, b.end_idx)
    if hi - lo + 1 < 3:
        return 0.0
    pa = np.asarray(a.intensity_array[lo - a.start_idx:hi - a.start_idx + 1], dtype=float)
    pb = np.asarray(b.intensity_array[lo - b.start_idx:hi - b.start_idx + 1], dtype=float)
    da = pa - pa.mean()
    db = pb - pb.mean()
    den = math.sqrt(float((da * da).sum())) * math.sqrt(float((db * db).sum()))
    return 0.0 if den == 0.0 else float((da * db).sum() / den)


def _is_excluded(mz: float, excluded) -> bool:
    return any(abs(mz - float(e)) <= EXCLUDED_MZ_TOLERANCE for e in excluded)


def group_ion_peaks(ion_peaks: list, params: CorrelationDeconvParams):
    """Port of RtGroupingAndShapeCorrelationAlgorithm.groupFeatures (+ getMainFeature)."""
    order = sorted(ion_peaks, key=lambda p: -p.height)      # stable for ties
    member_lists: list[list] = []
    r_lists: list[list] = []
    rejected: list[RejectedIon] = []
    for f in order:
        best_k, best_r, has_candidate = None, -math.inf, False
        for k, members in enumerate(member_lists):
            if abs(members[0].rt_apex - f.rt_apex) <= params.rt_tolerance:
                has_candidate = True
                r = pearson_r(members[0], f)
                if r > best_r and r >= params.min_r:
                    best_k, best_r = k, r
        if not has_candidate:
            member_lists.append([f])
            r_lists.append([1.0])
        elif best_k is None:
            rejected.append(RejectedIon(f, "shape"))
        else:
            member_lists[best_k].append(f)
            r_lists[best_k].append(best_r)

    min_ions = max(1, int(params.min_ions))
    groups: list[IonGroup] = []
    for members, rs in zip(member_lists, r_lists):
        if len(members) < min_ions:
            rejected.extend(RejectedIon(m, "small_group") for m in members)
            continue
        lead = next((m for m in members if not _is_excluded(m.mz, params.excluded_mz)),
                    members[0])
        groups.append(IonGroup(members=members, r=rs, lead=lead))
    return groups, rejected


def build_components(groups: list) -> list:
    """Port of SpectralDeconvolutionUtils.generatePseudoSpectra: spectrum = member heights."""
    comps = []
    for g in groups:
        spectrum: dict = {}
        for m in g.members:
            key = float(m.mz)
            spectrum[key] = max(spectrum.get(key, 0.0), float(m.height))
        comps.append(DeconvolutedComponent(
            rt=float(g.lead.rt_apex), spectrum=spectrum, model_peak_mz=float(g.lead.mz),
            model_peak_rt_array=g.lead.rt_array, model_peak_intensity_array=g.lead.intensity_array,
        ))
    return comps


def deconvolve_window(rt_window, ms_slice, mz_axis, noise_levels, params: CorrelationDeconvParams,
                      return_intermediates: bool = False):
    """Resolve every ion in one window, group them, and build components.

    rt_window: window RTs (min, MS time offset already applied); ms_slice:
    [window scans x m/z]; mz_axis: m/z of each column (ms.ylabels);
    noise_levels: whole-run noise level per column.
    """
    rt_window = np.asarray(rt_window, dtype=float)
    ms_slice = np.asarray(ms_slice, dtype=float)
    ion_peaks: list[IonPeak] = []
    for j in range(ms_slice.shape[1]):
        trace = ms_slice[:, j]
        if trace.size == 0 or trace.max() < params.min_peak_height:
            continue
        ion_peaks.extend(resolve_ion_peaks(
            rt_window, trace, float(mz_axis[j]), float(noise_levels[j]),
            search_range=params.search_range, min_peak_height=params.min_peak_height,
            min_top_edge_ratio=params.min_top_edge_ratio, min_points=params.min_points,
        ))
    groups, rejected = group_ion_peaks(ion_peaks, params)
    components = build_components(groups)
    if return_intermediates:
        return components, {"ion_peaks": ion_peaks, "groups": groups, "rejected": rejected}
    return components
