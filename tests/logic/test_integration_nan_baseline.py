"""Regression test for the double-baseline-subtraction bug in Integrator.integrate().

Before the MS-gated baseline merge (ff8d82f, June 2026), baseline_y and
corrected_y contained no NaN. The integrator used np.array_equal() to
detect "is integration_signal already baseline-corrected?" and that worked.

After the merge, baseline_y[masked_region] = NaN and corrected_y[masked_region]
also = NaN. np.array_equal returns False for arrays containing NaN (NaN != NaN
per IEEE 754), so the integrator silently took the wrong branch and
subtracted the baseline TWICE:

    y_peak_corrected = corrected_y - baseline_y
                     = (original_y - baseline_y) - baseline_y
                     = original_y - 2 * baseline_y

Then `area = abs(simpson(y_peak_corrected, x=x_peak))` reports a positive
number that is dominated by 2 * baseline integral, not the actual peak area.

The fix is to compare arrays with `equal_nan=True`, or use `is` identity
check (since integration_signal is set from processed_data['corrected_y']).
"""
import numpy as np
import pytest

from logic.integration import Integrator


def _build_processed(include_nan: bool, peak_height: float = 100.0):
    """Build a synthetic processed_data dict with one Gaussian peak on a
    flat baseline.

    Args:
        include_nan: If True, mask the first 100 points of baseline_y and
            corrected_y with NaN (simulates MS-gated baseline behavior).
        peak_height: Apex height of the Gaussian peak (above baseline).

    Returns:
        processed_data dict matching ChromatogramProcessor.process() output.
    """
    # 1000 points spanning 0-10 min (dt = 10 ms)
    x = np.linspace(0.0, 10.0, 1000)

    # Flat baseline at 50, with a Gaussian peak centered at t=5.0 (sigma 0.05 min)
    baseline = np.full_like(x, 50.0)
    peak = peak_height * np.exp(-0.5 * ((x - 5.0) / 0.05) ** 2)
    original_y = baseline + peak

    corrected_y = original_y - baseline  # i.e. just the peak

    baseline_y = baseline.copy()
    corrected_y = corrected_y.copy()

    if include_nan:
        # Mask first 100 points (pre-MS region in real data)
        baseline_y[:100] = np.nan
        corrected_y[:100] = np.nan

    # Detected peak at the Gaussian apex
    apex_idx = int(np.argmax(corrected_y[100:]) + 100) if include_nan else int(np.argmax(corrected_y))
    peaks_x = np.array([x[apex_idx]])
    peaks_y = np.array([corrected_y[apex_idx]])

    return {
        'x': x,
        'original_y': original_y,
        'smoothed_y': original_y.copy(),
        'baseline_y': baseline_y,
        'corrected_y': corrected_y,
        'peaks_x': peaks_x,
        'peaks_y': peaks_y,
        'peak_metadata': [
            {'index': apex_idx, 'x': float(peaks_x[0]),
             'y': float(peaks_y[0]), 'is_shoulder': False}
        ],
    }


def _expected_gaussian_area(height: float, sigma: float) -> float:
    """Closed-form area of a Gaussian: A = height * sigma * sqrt(2*pi)."""
    return height * sigma * np.sqrt(2.0 * np.pi)


def test_integration_correct_when_corrected_y_has_no_nan():
    """Baseline case: no NaN in arrays. Integrator should give roughly the
    Gaussian area."""
    processed = _build_processed(include_nan=False, peak_height=100.0)
    result = Integrator.integrate(processed, chemstation_area_factor=1.0,
                                  verbose=False)
    peaks = result['peaks']
    assert len(peaks) == 1

    expected = _expected_gaussian_area(100.0, 0.05)  # ≈ 12.53
    # The integrator walks bounds with a 0.25% threshold, so it won't capture
    # 100% of the Gaussian area — allow 10% tolerance below.
    assert peaks[0].area == pytest.approx(expected, rel=0.1), (
        f"expected ≈ {expected:.2f}, got {peaks[0].area:.2f}"
    )


def test_integration_correct_when_corrected_y_has_nan():
    """Regression test for double-subtraction bug.

    Bug: np.array_equal(corrected_y, corrected_y) returns False when the
    arrays contain NaN, so the integrator subtracts the baseline a second
    time. Reported area then = abs(simpson(original_y - 2*baseline_y, x))
    which is dominated by the baseline (50 * peak_width * 2) ≈ tens or hundreds
    of times the true peak area.

    With the fix, the integrator should give the same area whether or not
    the arrays contain NaN outside the peak window.
    """
    expected_no_nan = Integrator.integrate(
        _build_processed(include_nan=False, peak_height=100.0),
        chemstation_area_factor=1.0,
        verbose=False,
    )['peaks'][0].area

    result_with_nan = Integrator.integrate(
        _build_processed(include_nan=True, peak_height=100.0),
        chemstation_area_factor=1.0,
        verbose=False,
    )
    peaks = result_with_nan['peaks']
    assert len(peaks) == 1

    # The peak is at t=5.0, the NaN mask is at t<1.0 — they don't overlap.
    # The reported area must match the no-NaN case to within numerical noise.
    assert peaks[0].area == pytest.approx(expected_no_nan, rel=0.01), (
        f"NaN in baseline_y (outside peak window) corrupted integration: "
        f"expected ≈ {expected_no_nan:.4f}, got {peaks[0].area:.4f}. "
        f"This means the integrator subtracted the baseline twice."
    )


def test_integration_small_peak_on_high_baseline_with_nan():
    """The specific scenario from the bug report: a small peak (apex ~40)
    sitting on a high baseline (~125) where NaN masking exists earlier.

    Without the fix, peak.area is dominated by 2*baseline_integral over the
    peak window, NOT the actual peak signal.
    """
    # Build a high-baseline scenario
    x = np.linspace(0.0, 10.0, 1000)
    baseline = np.full_like(x, 125.0)  # high baseline like FID detector
    peak = 40.0 * np.exp(-0.5 * ((x - 5.0) / 0.03) ** 2)  # small sharp peak
    original_y = baseline + peak
    corrected_y = original_y - baseline

    baseline_y = baseline.copy()
    baseline_y[:100] = np.nan  # MS-off region masked
    corrected_y_masked = corrected_y.copy()
    corrected_y_masked[:100] = np.nan

    apex_idx = int(np.argmax(corrected_y_masked[100:]) + 100)
    processed = {
        'x': x,
        'original_y': original_y,
        'smoothed_y': original_y.copy(),
        'baseline_y': baseline_y,
        'corrected_y': corrected_y_masked,
        'peaks_x': np.array([x[apex_idx]]),
        'peaks_y': np.array([corrected_y_masked[apex_idx]]),
        'peak_metadata': [
            {'index': apex_idx, 'x': float(x[apex_idx]),
             'y': float(corrected_y_masked[apex_idx]),
             'is_shoulder': False}
        ],
    }

    result = Integrator.integrate(processed, chemstation_area_factor=1.0,
                                  verbose=False)
    peaks = result['peaks']
    assert len(peaks) == 1

    expected = _expected_gaussian_area(40.0, 0.03)  # ≈ 3.01
    # Without the fix, area would be on the order of (2 * 125 * peak_width)
    # ≈ tens to hundreds; with the fix, area should be ~3.
    assert peaks[0].area < 10.0, (
        f"Area = {peaks[0].area:.2f} — likely dominated by 2*baseline integral. "
        f"Expected ~{expected:.2f} (the actual Gaussian area)."
    )
    assert peaks[0].area == pytest.approx(expected, rel=0.2), (
        f"Expected ≈ {expected:.2f}, got {peaks[0].area:.2f}"
    )
