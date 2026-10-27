from logic.no_peaks_message import no_peaks_message


def test_no_method_loaded_says_to_load_one():
    msg = no_peaks_message(min_prominence=100000.0, signal_range=3141.5, method_loaded=False)
    assert "no method is loaded" in msg
    assert "Load Method" in msg
    assert "100000" in msg
    assert "3142" in msg


def test_threshold_above_signal_with_method_points_at_peak_settings():
    msg = no_peaks_message(min_prominence=100000.0, signal_range=3141.5, method_loaded=True)
    assert "no method is loaded" not in msg
    assert "above the signal range" in msg
    assert "100000" in msg


def test_threshold_within_signal_gives_plain_no_peaks():
    msg = no_peaks_message(min_prominence=500.0, signal_range=3141.5, method_loaded=True)
    assert "above the signal range" not in msg
    assert msg.startswith("No peaks found")


def test_relative_prominence_is_not_called_above_the_signal():
    # min_prominence <= 1 is a fraction of the signal range, so it can never exceed it
    msg = no_peaks_message(min_prominence=0.05, signal_range=3141.5, method_loaded=True)
    assert "above the signal range" not in msg
