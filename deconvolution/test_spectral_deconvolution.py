# deconvolution/test_spectral_deconvolution.py
"""Tests for ADAP-GC 3.2 spectral deconvolution module."""
import numpy as np
import pytest
from spectral_deconvolution import (
    EICPeak, DeconvolutedComponent, DeconvolutionParams,
    sharpness_yang, is_shared, shape_similarity_angle,
    _merge_peaks, _cluster_by_rt, _cluster_by_shape,
    _filter_peaks, _find_model_peak, _build_components,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_gaussian_peak(rt_center=5.0, width=0.5, height=1000.0,
                       mz=100.0, n_points=50) -> EICPeak:
    """Create a synthetic Gaussian EIC peak."""
    rts = np.linspace(rt_center - 2 * width, rt_center + 2 * width, n_points)
    ints = height * np.exp(-0.5 * ((rts - rt_center) / (width / 2)) ** 2)
    apex_idx = int(np.argmax(ints))
    return EICPeak(
        rt_apex=rt_center,
        mz=mz,
        rt_array=rts,
        intensity_array=ints,
        left_boundary_idx=0,
        right_boundary_idx=n_points - 1,
        apex_idx=apex_idx,
    )


class TestSharpnessYang:
    def test_gaussian_peak_is_sharp(self):
        # A symmetric Gaussian should score well above 10
        peak = make_gaussian_peak(rt_center=5.0, width=0.3, height=1000.0, n_points=50)
        score = sharpness_yang(peak.rt_array, peak.intensity_array,
                               peak.left_boundary_idx, peak.right_boundary_idx)
        assert score > 10.0

    def test_flat_signal_returns_negative_one(self):
        rt = np.linspace(0, 1, 20)
        ints = np.ones(20) * 500.0
        assert sharpness_yang(rt, ints, 0, 19) == -1.0

    def test_one_sided_returns_median_not_negative_one(self):
        # All right-side points below p25 → only left side has slopes
        # Construct: apex at index 10, steeply rising left, flat right (below p25)
        rt = np.linspace(0, 1, 21)
        ints = np.zeros(21)
        ints[0] = 10.0   # left boundary
        ints[10] = 1000.0  # apex
        ints[20] = 10.0  # right boundary (right side all flat, below p25=257.5)
        # Only left side will have points above p25=0.25*(1000-10)+10=257.5
        # Fill left side with rising slope
        for i in range(1, 10):
            ints[i] = 10.0 + (i / 10.0) * 990.0
        # Right side: all at 10 (below p25) except boundary
        score = sharpness_yang(rt, ints, 0, 20)
        # Left side has slopes; right side empty → returns median_left (not -1.0)
        assert score != -1.0
        assert score > 0.0


class TestIsShared:
    def test_clean_symmetric_peak_not_shared(self):
        # Symmetric Gaussian with low boundaries → not shared
        peak = make_gaussian_peak(rt_center=5.0, width=0.3, height=1000.0, n_points=50)
        sliced = peak.intensity_array[peak.left_boundary_idx:peak.right_boundary_idx + 1]
        assert is_shared(sliced, 0.3, 0.3) is False

    def test_high_left_boundary_is_shared(self):
        # Left boundary = 50% of apex → edge_to_height ratio exceeded
        ints = np.array([500.0, 600.0, 800.0, 1000.0, 700.0, 400.0, 50.0])
        assert is_shared(ints, 0.3, 0.3) is True

    def test_bimodal_is_shared(self):
        # Two clear peaks → multiple local maxima
        ints = np.array([10.0, 500.0, 200.0, 600.0, 10.0])
        assert is_shared(ints, 0.3, 0.3) is True

    def test_high_delta_is_shared(self):
        # Left=400, right=50, apex=1000 → |400-50|/1000 = 0.35 > 0.3
        ints = np.array([400.0, 700.0, 1000.0, 600.0, 50.0])
        assert is_shared(ints, 0.3, 0.3) is True


class TestShapeSimilarityAngle:
    def test_identical_peaks_angle_near_zero(self):
        peak = make_gaussian_peak(rt_center=5.0, mz=100.0)
        angle = shape_similarity_angle(peak, peak)
        assert angle < 1.0  # degrees

    def test_angle_always_in_valid_range(self):
        peak_a = make_gaussian_peak(rt_center=4.0, mz=100.0)
        peak_b = make_gaussian_peak(rt_center=6.0, mz=200.0)
        angle = shape_similarity_angle(peak_a, peak_b)
        assert 0.0 <= angle <= 90.0

    def test_very_different_shapes_large_angle(self):
        # peak_a: early sharp spike; peak_b: late sharp spike on shared RT range
        rts = np.linspace(0, 10, 100)
        ints_a = np.zeros(100)
        ints_a[10] = 1000.0  # spike near start
        ints_b = np.zeros(100)
        ints_b[90] = 1000.0  # spike near end
        peak_a = EICPeak(rt_apex=rts[10], mz=100.0, rt_array=rts,
                         intensity_array=ints_a,
                         left_boundary_idx=0, right_boundary_idx=99, apex_idx=10)
        peak_b = EICPeak(rt_apex=rts[90], mz=200.0, rt_array=rts,
                         intensity_array=ints_b,
                         left_boundary_idx=0, right_boundary_idx=99, apex_idx=90)
        angle = shape_similarity_angle(peak_a, peak_b)
        # Both on the same RT grid with non-overlapping spikes → near 90°
        assert angle > 45.0


class TestMergePeaks:
    def _make_adjacent_same_mz_peaks(self):
        """Two peaks at m/z=100 that are close enough to merge."""
        # Peak 1: rt 1.0–1.4, apex at 1.2
        rts1 = np.linspace(1.0, 1.4, 20)
        ints1 = 1000.0 * np.exp(-0.5 * ((rts1 - 1.2) / 0.08) ** 2)
        p1 = EICPeak(rt_apex=1.2, mz=100.0, rt_array=rts1, intensity_array=ints1,
                     left_boundary_idx=0, right_boundary_idx=19, apex_idx=int(np.argmax(ints1)))
        # Peak 2: rt 1.3–1.7, apex at 1.5 (overlapping window with p1)
        rts2 = np.linspace(1.3, 1.7, 20)
        ints2 = 800.0 * np.exp(-0.5 * ((rts2 - 1.5) / 0.08) ** 2)
        p2 = EICPeak(rt_apex=1.5, mz=100.0, rt_array=rts2, intensity_array=ints2,
                     left_boundary_idx=0, right_boundary_idx=19, apex_idx=int(np.argmax(ints2)))
        return p1, p2

    def test_adjacent_same_mz_merged(self):
        p1, p2 = self._make_adjacent_same_mz_peaks()
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert len(merged) == 1
        # Merged rt_array should span from p1 start to p2 end
        assert merged[0].rt_array.min() <= 1.0 + 1e-9
        assert merged[0].rt_array.max() >= 1.7 - 1e-9

    def test_non_overlapping_same_mz_not_merged(self):
        p1 = make_gaussian_peak(rt_center=1.0, width=0.1, mz=100.0, n_points=20)
        p2 = make_gaussian_peak(rt_center=5.0, width=0.1, mz=100.0, n_points=20)
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert len(merged) == 2

    def test_different_mz_not_merged(self):
        p1 = make_gaussian_peak(rt_center=2.0, mz=100.0, n_points=20)
        p2 = make_gaussian_peak(rt_center=2.0, mz=200.0, n_points=20)
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert len(merged) == 2

    def test_merged_apex_from_highest_intensity_peak(self):
        p1, p2 = self._make_adjacent_same_mz_peaks()
        # p1 apex intensity ~1000, p2 ~800; merged should use p1's apex_intensity
        merged = _merge_peaks([p1, p2], 0.3, 0.3)
        assert merged[0].apex_intensity == pytest.approx(
            max(p1.intensity_array.max(), p2.intensity_array.max()), rel=0.01
        )


class TestClusterByRT:
    def test_two_well_separated_groups(self):
        peaks = [
            make_gaussian_peak(rt_center=1.000, height=500.0, mz=float(i))
            for i in range(100, 103)
        ] + [
            make_gaussian_peak(rt_center=5.000, height=500.0, mz=float(i))
            for i in range(200, 203)
        ]
        clusters = _cluster_by_rt(peaks, eps=0.01, min_samples=2, min_intensity=100.0)
        assert len(clusters) == 2
        # Clusters should be sorted by mean RT
        mean_rts = [np.mean([p.rt_apex for p in c]) for c in clusters]
        assert mean_rts[0] < mean_rts[1]

    def test_low_intensity_cluster_dropped(self):
        high = [make_gaussian_peak(rt_center=1.0, height=500.0, mz=float(i)) for i in range(2)]
        low = [make_gaussian_peak(rt_center=5.0, height=50.0, mz=float(i)) for i in range(2)]
        clusters = _cluster_by_rt(high + low, eps=0.01, min_samples=2, min_intensity=100.0)
        assert len(clusters) == 1

    def test_isolated_peak_dropped_as_noise(self):
        peaks = [make_gaussian_peak(rt_center=1.0, height=500.0, mz=100.0)]
        clusters = _cluster_by_rt(peaks, eps=0.01, min_samples=2, min_intensity=100.0)
        assert len(clusters) == 0

    def test_empty_input(self):
        assert _cluster_by_rt([], eps=0.01, min_samples=2, min_intensity=100.0) == []


class TestClusterByShape:
    def test_identical_shapes_in_one_cluster(self):
        peaks = [make_gaussian_peak(rt_center=5.0, mz=float(m)) for m in [100, 200, 300]]
        clusters = _cluster_by_shape(peaks, threshold=30.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_different_shapes_split_into_clusters(self):
        rts = np.linspace(0, 10, 100)
        ints_early = np.zeros(100); ints_early[10] = 1000.0
        ints_late = np.zeros(100); ints_late[90] = 1000.0

        def make_spike(ints, mz):
            return EICPeak(rt_apex=rts[np.argmax(ints)], mz=mz,
                           rt_array=rts, intensity_array=ints,
                           left_boundary_idx=0, right_boundary_idx=99,
                           apex_idx=int(np.argmax(ints)))

        early = [make_spike(ints_early.copy(), float(m)) for m in [100, 101, 102]]
        late = [make_spike(ints_late.copy(), float(m)) for m in [200, 201, 202]]
        clusters = _cluster_by_shape(early + late, threshold=30.0)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [3, 3]

    def test_single_peak_returns_one_cluster(self):
        peak = make_gaussian_peak()
        clusters = _cluster_by_shape([peak], threshold=30.0)
        assert len(clusters) == 1
        assert clusters[0][0] is peak


class TestFilterPeaks:
    def test_sharp_clean_peak_passes(self):
        peak = make_gaussian_peak(rt_center=5.0, width=0.2, height=1000.0, n_points=50)
        params = DeconvolutionParams(min_model_peak_sharpness=1.0)
        result = _filter_peaks([peak], params)
        assert len(result) == 1

    def test_low_sharpness_removed(self):
        rt = np.linspace(0, 1, 20)
        ints = np.ones(20) * 500.0
        ints[10] = 510.0  # tiny apex
        peak = EICPeak(rt_apex=0.5, mz=100.0, rt_array=rt, intensity_array=ints,
                       left_boundary_idx=0, right_boundary_idx=19, apex_idx=10)
        params = DeconvolutionParams(min_model_peak_sharpness=10.0)
        result = _filter_peaks([peak], params)
        assert len(result) == 0

    def test_excluded_mz_removed(self):
        peak = make_gaussian_peak(mz=73.0, width=0.2, height=1000.0, n_points=50)
        params = DeconvolutionParams(
            min_model_peak_sharpness=1.0,
            excluded_mz=[73.0], excluded_mz_tolerance=0.5
        )
        result = _filter_peaks([peak], params)
        assert len(result) == 0

    def test_shared_peak_removed_when_enabled(self):
        rt = np.linspace(0, 1, 20)
        ints = np.zeros(20)
        ints[0] = 10.0; ints[5] = 500.0; ints[10] = 200.0; ints[15] = 600.0; ints[19] = 10.0
        peak = EICPeak(rt_apex=0.789, mz=100.0, rt_array=rt, intensity_array=ints,
                       left_boundary_idx=0, right_boundary_idx=19, apex_idx=15)
        params = DeconvolutionParams(use_is_shared=True, min_model_peak_sharpness=0.0)
        result = _filter_peaks([peak], params)
        assert len(result) == 0


class TestFindModelPeak:
    def test_picks_sharpest_peak(self):
        sharp = make_gaussian_peak(rt_center=5.0, width=0.1, height=1000.0, n_points=50)
        broad = make_gaussian_peak(rt_center=5.0, width=0.5, height=1000.0, n_points=50)
        result = _find_model_peak([broad, sharp], 'sharpness')
        assert result is sharp

    def test_picks_highest_intensity(self):
        low = make_gaussian_peak(height=500.0, mz=100.0)
        high = make_gaussian_peak(height=2000.0, mz=200.0)
        result = _find_model_peak([low, high], 'intensity')
        assert result is high

    def test_returns_none_for_empty_list(self):
        assert _find_model_peak([], 'sharpness') is None


class TestBuildComponents:
    def _make_other_peak(self, source_peak: EICPeak):
        from spectral_deconvolution import _PeakData
        return _PeakData(
            source=source_peak,
            left_peak_rt=float(source_peak.rt_array[source_peak.left_boundary_idx]),
            right_peak_rt=float(source_peak.rt_array[source_peak.right_boundary_idx]),
            rt_array=source_peak.rt_array.copy(),
            intensity_array=source_peak.intensity_array.copy(),
            apex_intensity=float(source_peak.intensity_array[source_peak.apex_idx]),
        )

    def test_single_model_peak_has_mz_in_spectrum(self):
        model = make_gaussian_peak(rt_center=5.0, mz=100.0, height=1000.0, n_points=50)
        other_src = make_gaussian_peak(rt_center=5.0, mz=200.0, height=800.0, n_points=50)
        other = self._make_other_peak(other_src)
        components = _build_components([model], [other])
        assert len(components) == 1
        assert 200.0 in components[0].spectrum
        assert components[0].spectrum[200.0] > 0.0

    def test_model_apex_outside_other_boundary_gives_zero(self):
        # Other peak: RT 1.0–2.0; model peak: RT=5.0 (outside)
        model = make_gaussian_peak(rt_center=5.0, mz=100.0)
        other_src = make_gaussian_peak(rt_center=1.5, mz=200.0)
        other = self._make_other_peak(other_src)
        components = _build_components([model], [other])
        # model.rt_apex=5.0 is outside other's boundary [~1.0, ~2.0]
        assert len(components) == 1
        assert components[0].spectrum.get(200.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_two_models_separated_contributions(self):
        model1 = make_gaussian_peak(rt_center=2.0, mz=100.0, height=1000.0, n_points=30)
        model2 = make_gaussian_peak(rt_center=8.0, mz=200.0, height=1000.0, n_points=30)

        rts_other = np.linspace(1.5, 2.5, 30)
        ints_other = 500.0 * np.exp(-0.5 * ((rts_other - 2.0) / 0.2) ** 2)
        other_src = EICPeak(rt_apex=2.0, mz=300.0, rt_array=rts_other,
                            intensity_array=ints_other, left_boundary_idx=0,
                            right_boundary_idx=29, apex_idx=int(np.argmax(ints_other)))
        from spectral_deconvolution import _PeakData
        other = _PeakData(source=other_src, left_peak_rt=1.5, right_peak_rt=2.5,
                          rt_array=rts_other, intensity_array=ints_other,
                          apex_intensity=float(ints_other.max()))

        components = _build_components([model1, model2], [other])
        assert len(components) == 2
        comp1 = next(c for c in components if c.model_peak_mz == 100.0)
        comp2 = next(c for c in components if c.model_peak_mz == 200.0)
        assert comp1.spectrum.get(300.0, 0.0) > 0.0
        assert comp2.spectrum.get(300.0, 0.0) == pytest.approx(0.0, abs=1e-6)
