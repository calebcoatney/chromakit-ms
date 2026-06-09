"""Tests for logic/processor._build_baseline_segments and _BaselineSegment.

The helper decides which slices of the FID get fed to pybaselines (fit=True)
and which get filled with NaN (fit=False), based on the MS-on time window
and optional manual break points.
"""
import numpy as np
import pytest

from logic.processor import _BaselineSegment, _build_baseline_segments


@pytest.fixture
def x_axis():
    """0.0 to 9.99 minutes, 0.01-min spacing → 1000 points."""
    return np.linspace(0.0, 9.99, 1000)


def test_no_ms_range_no_breakpoints_returns_single_fit_segment(x_axis):
    segs = _build_baseline_segments(x_axis, ms_range=None, break_points=None)
    assert len(segs) == 1
    assert segs[0] == _BaselineSegment(start=0, end=1000, fit=True)


def test_no_ms_range_empty_breakpoints_returns_single_fit_segment(x_axis):
    segs = _build_baseline_segments(x_axis, ms_range=None, break_points=[])
    assert len(segs) == 1
    assert segs[0].fit is True
    assert segs[0].start == 0
    assert segs[0].end == 1000


def test_ms_range_after_start_creates_leading_no_fit_segment(x_axis):
    # MS turns on at t=6.3 min → index ~630
    segs = _build_baseline_segments(x_axis, ms_range=(6.3, 9.99), break_points=None)
    assert len(segs) == 2
    assert segs[0].fit is False
    assert segs[0].start == 0
    # Boundary index: argmin(|x - 6.3|) — accept ±1 for float rounding
    assert 629 <= segs[0].end <= 631
    assert segs[1].fit is True
    assert segs[1].start == segs[0].end
    assert segs[1].end == 1000


def test_ms_range_before_end_creates_trailing_no_fit_segment(x_axis):
    segs = _build_baseline_segments(x_axis, ms_range=(0.0, 8.0), break_points=None)
    assert len(segs) == 2
    assert segs[0].fit is True
    assert segs[0].start == 0
    assert segs[1].fit is False
    assert segs[1].end == 1000


def test_ms_range_carves_both_ends(x_axis):
    segs = _build_baseline_segments(x_axis, ms_range=(2.0, 8.0), break_points=None)
    assert len(segs) == 3
    assert [s.fit for s in segs] == [False, True, False]
    # Continuity: each segment's end == next segment's start
    assert segs[0].end == segs[1].start
    assert segs[1].end == segs[2].start
    # Full coverage
    assert segs[0].start == 0
    assert segs[-1].end == 1000


def test_ms_range_fully_contains_x_no_carveout(x_axis):
    """If ms_range fully spans the FID, no fit=False segments are created."""
    segs = _build_baseline_segments(x_axis, ms_range=(-1.0, 100.0), break_points=None)
    assert len(segs) == 1
    assert segs[0].fit is True


def test_ms_range_starts_exactly_at_x_zero_no_leading_carveout(x_axis):
    """When ms_range[0] aligns with x[0], no leading fit=False segment."""
    segs = _build_baseline_segments(x_axis, ms_range=(0.0, 9.99), break_points=None)
    fit_flags = [s.fit for s in segs]
    assert False not in fit_flags


def test_break_point_inside_ms_window_subdivides_fit_segment(x_axis):
    # MS on from 6.3 to end; break point at t=8.0
    segs = _build_baseline_segments(
        x_axis, ms_range=(6.3, 9.99), break_points=[{'time': 8.0}]
    )
    # Expect: [fit=False (0..~630), fit=True (~630..~800), fit=True (~800..1000)]
    assert len(segs) == 3
    assert segs[0].fit is False
    assert segs[1].fit is True
    assert segs[2].fit is True
    # Break point boundary
    assert 799 <= segs[1].end <= 801
    assert segs[1].end == segs[2].start


def test_break_point_inside_masked_region_is_dropped(x_axis):
    """Break points before ms_range[0] should be silently dropped with a warning."""
    segs = _build_baseline_segments(
        x_axis, ms_range=(6.3, 9.99), break_points=[{'time': 2.0}]
    )
    # Expect: [fit=False (0..~630), fit=True (~630..1000)] — break point ignored
    assert len(segs) == 2
    assert segs[0].fit is False
    assert segs[1].fit is True
    assert segs[1].end == 1000


def test_break_point_as_float_not_dict(x_axis):
    """Break points can be plain floats, not just dicts."""
    segs = _build_baseline_segments(
        x_axis, ms_range=None, break_points=[5.0]
    )
    assert len(segs) == 2
    assert segs[0].fit is True
    assert segs[1].fit is True
    assert 499 <= segs[0].end <= 501


def test_short_fit_segment_demoted_to_no_fit(x_axis):
    """A fit=True segment shorter than MIN_SEGMENT_LEN=10 gets demoted to fit=False."""
    # ms_range from 9.95 to 9.99 → only ~5 points
    segs = _build_baseline_segments(x_axis, ms_range=(9.95, 9.99), break_points=None)
    # Whatever segments are produced, none with fit=True should be < 10 long
    for seg in segs:
        if seg.fit:
            assert (seg.end - seg.start) >= 10


def test_segments_cover_full_range_with_no_overlap(x_axis):
    """Segments must tile [0, len(x)) exactly — no gaps, no overlaps."""
    segs = _build_baseline_segments(
        x_axis, ms_range=(2.0, 8.0), break_points=[{'time': 5.0}]
    )
    assert segs[0].start == 0
    assert segs[-1].end == 1000
    for prev, curr in zip(segs, segs[1:]):
        assert prev.end == curr.start


def test_baseline_segment_dataclass_fields():
    """_BaselineSegment has start, end, fit fields."""
    seg = _BaselineSegment(start=0, end=10, fit=True)
    assert seg.start == 0
    assert seg.end == 10
    assert seg.fit is True


def test_swapped_ms_range_raises_value_error(x_axis):
    """A swapped (t_hi, t_lo) ms_range should raise ValueError, not silently produce overlapping segments."""
    with pytest.raises(ValueError, match="t_lo <= t_hi"):
        _build_baseline_segments(x_axis, ms_range=(8.0, 2.0), break_points=None)


def test_empty_x_returns_empty_list():
    """Empty input should return an empty segment list, not raise."""
    result = _build_baseline_segments(np.array([]), ms_range=None, break_points=None)
    assert result == []


def test_apply_baseline_correction_with_ms_range_produces_nan_in_masked_region():
    """When ms_range is provided, the baseline outside it should be NaN."""
    from logic.processor import ChromatogramProcessor
    
    processor = ChromatogramProcessor()
    x = np.linspace(0.0, 9.99, 1000)
    # Signal with a noisy baseline around 1.0 — pybaselines should be able to fit something.
    rng = np.random.default_rng(seed=42)
    y = rng.normal(loc=1.0, scale=0.05, size=1000)
    
    bl, corr = processor._apply_baseline_correction(
        x, y, method="arpls", lam=1e4, ms_range=(5.0, 9.99)
    )
    
    # Output must be full-length
    assert len(bl) == len(y)
    assert len(corr) == len(y)
    
    # Pre-MS region (t < 5.0) should be entirely NaN
    pre_ms_idx = int(np.argmin(np.abs(x - 5.0)))
    assert np.all(np.isnan(bl[:pre_ms_idx])), "pre-MS baseline should be NaN"
    assert np.all(np.isnan(corr[:pre_ms_idx])), "pre-MS corrected should be NaN"
    
    # MS-on region should be entirely finite
    assert np.all(np.isfinite(bl[pre_ms_idx:])), "MS-on baseline should be finite"
    assert np.all(np.isfinite(corr[pre_ms_idx:])), "MS-on corrected should be finite"


def test_apply_baseline_correction_no_ms_range_no_nan():
    """When ms_range is None (current default), no NaN should appear (backward compat)."""
    from logic.processor import ChromatogramProcessor
    
    processor = ChromatogramProcessor()
    x = np.linspace(0.0, 9.99, 1000)
    rng = np.random.default_rng(seed=42)
    y = rng.normal(loc=1.0, scale=0.05, size=1000)
    
    bl, corr = processor._apply_baseline_correction(x, y, method="arpls", lam=1e4)
    
    assert len(bl) == len(y)
    assert np.all(np.isfinite(bl)), "no NaN should appear when ms_range is not provided"
    assert np.all(np.isfinite(corr))


def test_apply_baseline_correction_ms_range_plus_break_points():
    """ms_range carves out the masked region; break_points then subdivide the fit region."""
    from logic.processor import ChromatogramProcessor
    
    processor = ChromatogramProcessor()
    x = np.linspace(0.0, 9.99, 1000)
    rng = np.random.default_rng(seed=42)
    y = rng.normal(loc=1.0, scale=0.05, size=1000)
    
    bl, corr = processor._apply_baseline_correction(
        x, y, method="arpls", lam=1e4,
        ms_range=(2.0, 8.0),
        break_points=[{'time': 5.0}],
    )
    
    # Output must be full-length
    assert len(bl) == len(y)
    
    # Pre-MS region (t < 2.0) should be NaN
    pre_ms_idx = int(np.argmin(np.abs(x - 2.0)))
    assert np.all(np.isnan(bl[:pre_ms_idx])), "pre-MS baseline should be NaN"
    
    # Post-MS region (t > 8.0) should be NaN
    post_ms_idx = int(np.argmin(np.abs(x - 8.0))) + 1
    assert np.all(np.isnan(bl[post_ms_idx:])), "post-MS baseline should be NaN"
    
    # MS-on region should be entirely finite (both halves of the break-point split)
    assert np.all(np.isfinite(bl[pre_ms_idx:post_ms_idx])), "MS-on region should be finite"
