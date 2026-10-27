"""Plain-language reason for an integration that found no peaks."""
from __future__ import annotations


def no_peaks_message(min_prominence: float, signal_range: float, method_loaded: bool) -> str:
    """Explain why integration found no peaks.

    min_prominence > 1 is in signal units; <= 1 is a fraction of the signal range
    (Processor.integrate's convention), so only an absolute value can exceed the signal.
    """
    above = min_prominence > 1 and min_prominence > signal_range
    if not method_loaded:
        return (f"No peaks found: no method is loaded, so the default settings are in use "
                f"(min prominence {min_prominence:.0f} against a signal range of {signal_range:.0f}). "
                f"Load a method (Load Method…) before integrating.")
    if above:
        return (f"No peaks found: min prominence {min_prominence:.0f} is above the signal range "
                f"of {signal_range:.0f}. Check the method's peak settings and signal scaling.")
    return "No peaks found with the current peak settings."
