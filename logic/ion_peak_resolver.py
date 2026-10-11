# Portions of this file are ported from mzmine (https://github.com/mzmine/mzmine):
#   io.github.mzmine.modules.dataprocessing.featdet_chromatogramdeconvolution.minimumsearch.MinimumSearchFeatureResolver
#   io.github.mzmine.util.MathUtils (calcQuantile, calcQuantileSorted)
#
# Copyright (c) 2004-2024 The mzmine Development Team
# Copyright (c) 2004-2026 The mzmine Development Team
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
"""Baseline-to-baseline ion peak detection (local minimum search).

Python port of mzmine's local minimum resolver, used by ChromaKit's spectral
deconvolution. One deliberate change: the noise level is passed in, computed by
the caller over the whole run (``whole_run_noise_levels``). mzmine takes the
quantile of the trace being resolved; ChromaKit resolves short windows around
FID peaks, where that quantile would cut into real peaks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# mzmine resolver settings ChromaKit fixes at values that filter nothing.
MIN_RELATIVE_HEIGHT = 0.0
PEAK_DURATION = (0.0, 10.0)   # minutes, inclusive


@dataclass(eq=False)   # identity equality: default __eq__ compares ndarray fields and raises
class IonPeak:
    """One ion's chromatographic peak inside a deconvolution window."""
    mz: float
    start_idx: int              # inclusive index into the window's scan axis
    end_idx: int                # inclusive
    apex_idx: int               # window index of the (first) highest point
    rt_apex: float              # minutes
    height: float               # highest original intensity in [start_idx, end_idx]
    rt_array: np.ndarray        # window RTs, start_idx..end_idx
    intensity_array: np.ndarray # original intensities, start_idx..end_idx


def mzmine_quantile(values, q: float) -> float:
    """mzmine MathUtils.calcQuantile: mean of the order statistics at floor/ceil of (n-1)q."""
    v = np.asarray(values, dtype=float).ravel()
    n = v.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(v[0])
    q = min(max(float(q), 0.0), 1.0)
    s = np.sort(v)
    i1 = int(math.floor((n - 1) * q))
    i2 = int(math.ceil((n - 1) * q))
    return float((s[i1] + s[i2]) / 2.0)


def whole_run_noise_levels(data, q: float) -> np.ndarray:
    """``mzmine_quantile`` of every column of ``data`` [scans x m/z], over all scans."""
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ValueError("data must be 2-D [scans x m/z]")
    n = d.shape[0]
    if n == 0:
        return np.zeros(d.shape[1])
    if n == 1:
        return d[0].copy()
    q = min(max(float(q), 0.0), 1.0)
    s = np.sort(d, axis=0)
    i1 = int(math.floor((n - 1) * q))
    i2 = int(math.ceil((n - 1) * q))
    return (s[i1] + s[i2]) / 2.0


def resolve_ion_peaks(x, y, mz: float, noise_level: float, *, search_range: float,
                      min_peak_height: float, min_top_edge_ratio: float,
                      min_points: int) -> list[IonPeak]:
    """Split one ion's window trace into baseline-to-baseline peaks.

    Port of MinimumSearchFeatureResolver.resolve: points below ``noise_level``
    are zeroed, then a region grows from two consecutive non-zero points until
    the trace hits zero/ends, or reaches a point that is the minimum within
    ±``search_range`` and lies at least ``min_top_edge_ratio`` below the region
    top. Accepted regions keep the original intensities.
    """
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(y, dtype=float)
    n = x.size
    if y0.size != n:
        raise ValueError("x and y must have the same length")
    if n < 3:
        return []
    yz = np.where(y0 < noise_level, 0.0, y0)
    min_height = max(min_peak_height, MIN_RELATIVE_HEIGHT * float(yz.max()))
    last = n - 1
    bounds: list[tuple[int, int]] = []

    def finalize(cs: int, ce: int, height: float) -> int:
        # tryToFinalizePeak + checkPeakShape + adjustStartAndEnd
        n_points = ce - cs + 1
        duration = x[ce] - x[cs]
        if (n_points >= min_points and height >= min_height
                and height >= yz[cs] * min_top_edge_ratio
                and height >= yz[ce] * min_top_edge_ratio
                and PEAK_DURATION[0] <= duration <= PEAK_DURATION[1]):
            start, end = cs, ce
            if yz[cs] != 0.0 and yz[max(cs - 1, 0)] == 0.0:
                start = cs - 1
            if yz[ce] != 0.0 and yz[min(ce + 1, last)] == 0.0:
                end = ce + 1
            bounds.append((start, end))
        return ce - 1

    cs = 0
    while cs < last - 2:
        if yz[cs] == 0.0 or yz[cs + 1] == 0.0:
            cs += 1
            continue
        height = yz[cs]
        next_cs = cs
        ce = cs + 1
        while ce < n:
            height = max(height, yz[ce])
            if ce == last or yz[ce + 1] == 0.0:
                next_cs = finalize(cs, ce, height)
                break
            if x[ce] - x[cs] >= search_range:
                lo, hi = x[ce] - search_range, x[ce] + search_range
                is_min = True
                i = ce - 1
                while i > 0 and lo <= x[i] <= hi:
                    if yz[i] < yz[ce]:
                        is_min = False
                        break
                    i -= 1
                if is_min:
                    i = ce + 1
                    while i < n and lo <= x[i] <= hi:
                        if yz[i] < yz[ce]:
                            is_min = False
                            break
                        i += 1
                if is_min and height >= yz[ce] * min_top_edge_ratio:
                    next_cs = finalize(cs, ce, height)
                    break
            ce += 1
        cs = next_cs + 1

    peaks: list[IonPeak] = []
    for start, end in bounds:
        seg = y0[start:end + 1]
        k = int(np.argmax(seg))
        peaks.append(IonPeak(
            mz=float(mz), start_idx=start, end_idx=end, apex_idx=start + k,
            rt_apex=float(x[start + k]), height=float(seg[k]),
            rt_array=x[start:end + 1].copy(), intensity_array=seg.copy(),
        ))
    return peaks
