"""Tests for logic.correlation_grouping (port of mzmine
RtGroupingAndShapeCorrelationAlgorithm.groupFeatures / calculateCorrelation and
SpectralDeconvolutionUtils.generatePseudoSpectra / getMainFeature)."""
import copy

import numpy as np
import pytest

from logic.ion_peak_resolver import IonPeak, whole_run_noise_levels
from logic.correlation_grouping import (
    CorrelationDeconvParams, DeconvolutedComponent,
    pearson_r, group_ion_peaks, build_components, deconvolve_window,
)

X = np.round(np.arange(0.0, 2.0, 0.005), 6)


def ion(mz, center, height, sigma=0.015, profile=None):
    y = height * np.exp(-0.5 * ((X - center) / sigma) ** 2) if profile is None else profile
    idx = np.nonzero(y >= 0.01 * y.max())[0]
    s, e = int(idx[0]), int(idx[-1])
    seg = y[s:e + 1]
    k = int(np.argmax(seg))
    return IonPeak(mz=float(mz), start_idx=s, end_idx=e, apex_idx=s + k,
                   rt_apex=float(X[s + k]), height=float(seg[k]),
                   rt_array=X[s:e + 1].copy(), intensity_array=seg.copy())


def test_defaults_match_spec():
    p = CorrelationDeconvParams()
    assert (p.noise_percentile, p.search_range, p.min_peak_height, p.min_top_edge_ratio,
            p.min_points, p.rt_tolerance, p.min_r, p.min_ions, p.excluded_mz) == \
        (0.85, 0.05, 1000.0, 1.7, 4, 0.04, 0.80, 4, [])


def test_coeluting_same_shape_one_group():
    peaks = [ion(43, 1.0, 5000), ion(57, 1.0, 10000), ion(85, 1.0, 2000), ion(71, 1.0, 3000)]
    groups, rejected = group_ion_peaks(peaks, CorrelationDeconvParams())
    assert rejected == []
    assert len(groups) == 1
    g = groups[0]
    assert [m.mz for m in g.members] == [57.0, 43.0, 71.0, 85.0]   # descending height
    assert g.r[0] == 1.0
    assert all(r == pytest.approx(1.0, abs=1e-9) for r in g.r)
    assert g.lead.mz == 57.0


def test_different_shape_rejected_not_new_group():
    peaks = [ion(57, 1.0, 10000), ion(43, 1.0, 5000), ion(71, 1.0, 3000), ion(85, 1.0, 2000),
             ion(99, 1.03, 2500)]                    # within RT tolerance, offset shape
    groups, rejected = group_ion_peaks(peaks, CorrelationDeconvParams())
    assert len(groups) == 1
    assert [r.peak.mz for r in rejected] == [99.0]
    assert rejected[0].reason == "shape"


def test_outside_rt_tolerance_starts_new_group():
    peaks = [ion(57, 1.0, 10000), ion(43, 1.2, 8000)]
    groups, _ = group_ion_peaks(peaks, CorrelationDeconvParams(min_ions=1))
    assert sorted(g.lead.mz for g in groups) == [43.0, 57.0]


def test_small_group_rejected():
    peaks = [ion(57, 1.0, 10000), ion(43, 1.0, 5000)]
    groups, rejected = group_ion_peaks(peaks, CorrelationDeconvParams(min_ions=3))
    assert groups == []
    assert sorted((r.peak.mz, r.reason) for r in rejected) == [(43.0, "small_group"), (57.0, "small_group")]


def test_min_ions_clamped_to_one():
    groups, _ = group_ion_peaks([ion(57, 1.0, 10000)], CorrelationDeconvParams(min_ions=0))
    assert len(groups) == 1


def test_excluded_mz_lead_selection():
    peaks = [ion(73, 1.0, 10000), ion(57, 1.0, 8000), ion(43, 1.0, 5000), ion(85, 1.0, 2000)]
    params = CorrelationDeconvParams(excluded_mz=[73.0])
    groups, _ = group_ion_peaks(peaks, params)
    g = groups[0]
    assert g.members[0].mz == 73.0       # representative is still the highest
    assert g.lead.mz == 57.0             # lead skips excluded m/z (±0.5)
    comp = build_components(groups)[0]
    assert comp.model_peak_mz == 57.0
    assert comp.rt == g.lead.rt_apex
    assert 73.0 in comp.spectrum         # excluded ions stay in the spectrum


def test_excluded_mz_tolerance_is_half_a_mass_unit():
    peaks = [ion(73, 1.0, 10000), ion(57, 1.0, 8000), ion(43, 1.0, 5000), ion(85, 1.0, 2000)]
    # 73.4 is 0.4 from 73.0: it excludes 73, so the lead falls to 57
    groups, _ = group_ion_peaks(peaks, CorrelationDeconvParams(excluded_mz=[73.4]))
    assert groups[0].lead.mz == 57.0
    # 73.6 (0.6 away) and 74.0 (1.0 away) do not exclude 73, so the lead stays 73
    for excluded in ([73.6], [74.0]):
        groups, _ = group_ion_peaks(peaks, CorrelationDeconvParams(excluded_mz=excluded))
        assert groups[0].lead.mz == 73.0


def test_component_equality_does_not_compare_arrays():
    groups, _ = group_ion_peaks([ion(57, 1.0, 10000), ion(43, 1.0, 5000)],
                                CorrelationDeconvParams(min_ions=1))
    comp = build_components(groups)[0]
    assert comp == comp
    assert comp != copy.deepcopy(comp)


def test_two_candidate_groups_higher_r_wins():
    a = ion(100, 1.00, 10000, sigma=0.03)
    b = ion(200, 1.07, 9000, sigma=0.03)
    f = ion(300, 1.05, 1000, sigma=0.045)   # r(a,f)=0.084 and r(b,f)=0.760: b must win
    params = CorrelationDeconvParams(rt_tolerance=0.065, min_r=0.0, min_ions=1)
    groups, _ = group_ion_peaks([a, b, f], params)
    by_rep = {g.members[0].mz: [m.mz for m in g.members] for g in groups}
    assert by_rep == {100.0: [100.0], 200.0: [200.0, 300.0]}
    g_b = next(g for g in groups if g.members[0].mz == 200.0)
    assert g_b.r[0] == 1.0
    assert g_b.r[1] == pytest.approx(0.760, abs=0.01)


def test_pearson_needs_three_shared_points():
    a = ion(57, 1.0, 10000)
    b = IonPeak(mz=43.0, start_idx=a.end_idx - 1, end_idx=a.end_idx + 20, apex_idx=a.end_idx,
                rt_apex=float(X[a.end_idx]), height=1.0,
                rt_array=X[a.end_idx - 1:a.end_idx + 21].copy(),
                intensity_array=np.linspace(1.0, 0.1, 22))
    assert pearson_r(a, b) == 0.0


def test_flat_profile_r_zero():
    a = ion(57, 1.0, 10000)
    flat = IonPeak(mz=43.0, start_idx=a.start_idx, end_idx=a.end_idx, apex_idx=a.start_idx,
                   rt_apex=a.rt_apex, height=500.0, rt_array=a.rt_array.copy(),
                   intensity_array=np.full(a.intensity_array.size, 500.0))
    assert pearson_r(a, flat) == 0.0


def test_duplicate_mz_in_group_keeps_max_height():
    peaks = [ion(57, 1.0, 10000), ion(43, 1.0, 3000), ion(43, 1.005, 2000)]
    groups, _ = group_ion_peaks(peaks, CorrelationDeconvParams(min_ions=1))
    comp = build_components(groups)[0]
    assert comp.spectrum[43.0] == pytest.approx(3000.0)


def _matrix(mz_axis, compounds, floor_seed=0):
    rng = np.random.default_rng(floor_seed)
    data = rng.uniform(0.0, 30.0, (X.size, len(mz_axis)))
    col = {m: j for j, m in enumerate(mz_axis)}
    for center, ions in compounds:
        for m, h in ions.items():
            data[:, col[m]] += h * np.exp(-0.5 * ((X - center) / 0.015) ** 2)
    return data


def test_deconvolve_window_two_compounds():
    mz_axis = np.array([29.0, 43.0, 57.0, 71.0, 85.0, 99.0])
    data = _matrix(mz_axis, [(0.8, {43: 8000, 57: 20000, 71: 6000, 85: 5000}),
                             (1.4, {29: 7000, 43: 9000, 57: 15000, 99: 6000})])
    params = CorrelationDeconvParams(search_range=0.03)
    noise = whole_run_noise_levels(data, params.noise_percentile)
    comps, inter = deconvolve_window(X, data, mz_axis, noise, params, return_intermediates=True)
    assert set(inter) == {"ion_peaks", "groups", "rejected"}
    assert len(comps) == 2
    comps = sorted(comps, key=lambda c: c.rt)
    assert abs(comps[0].rt - 0.8) < 0.01 and abs(comps[1].rt - 1.4) < 0.01
    assert set(comps[0].spectrum) == {43.0, 57.0, 71.0, 85.0}
    assert set(comps[1].spectrum) == {29.0, 43.0, 57.0, 99.0}
    assert isinstance(comps[0], DeconvolutedComponent)


def test_deconvolve_window_uses_mz_axis_values():
    # sparse, non-contiguous axis: column index + 1 would be wrong
    mz_axis = np.array([31.0, 44.0, 91.0, 105.0, 120.0])
    data = _matrix(mz_axis, [(1.0, {31: 5000, 44: 9000, 91: 20000, 105: 7000})])
    params = CorrelationDeconvParams(search_range=0.03)
    noise = whole_run_noise_levels(data, params.noise_percentile)
    comps = deconvolve_window(X, data, mz_axis, noise, params)
    assert len(comps) == 1
    assert set(comps[0].spectrum) == {31.0, 44.0, 91.0, 105.0}
    assert comps[0].model_peak_mz == 91.0


def test_deconvolve_window_empty_when_nothing_above_height():
    mz_axis = np.array([43.0, 57.0])
    data = np.random.default_rng(1).uniform(0.0, 30.0, (X.size, 2))
    params = CorrelationDeconvParams()
    noise = whole_run_noise_levels(data, params.noise_percentile)
    assert deconvolve_window(X, data, mz_axis, noise, params) == []
