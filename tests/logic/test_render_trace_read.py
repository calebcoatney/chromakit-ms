"""Reading the signal. Bad files degrade to (None, reason); they never raise
and never return a plausible wrong trace."""
import pytest

from logic.render_trace import available_detectors, read_signal, read_trace


def test_unreadable_path_degrades_with_a_reason(tmp_path):
    fake = tmp_path / "nothing.D"
    fake.mkdir()
    trace, reason = read_trace(str(fake), detector="FID1A")
    assert trace is None
    assert reason and "FID1A" in reason


def test_available_detectors_on_a_missing_path_is_empty(tmp_path):
    assert available_detectors(str(tmp_path / "absent.D")) == []


def test_read_signal_on_a_missing_path_is_none(tmp_path):
    assert read_signal(str(tmp_path / "absent.D"), detector="FID1A") is None


def test_bad_max_points_still_raises_through_read_trace(tmp_path):
    fake = tmp_path / "nothing.D"
    fake.mkdir()
    with pytest.raises(ValueError):
        read_trace(str(fake), detector="FID1A", max_points=1)
