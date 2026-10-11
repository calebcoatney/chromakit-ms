"""Load/save spectral-deconvolution settings (QSettings group ``ms_spectral_deconv/``).

Qt-free: works with any object offering ``value(key, default, type)`` and
``setValue(key, value)`` (QSettings in the app, a dict-backed fake in tests).
Old ADAP-era keys in the group are ignored.
"""
from __future__ import annotations

import math
import re

from logic.correlation_grouping import CorrelationDeconvParams
from logic.spectral_deconv_runner import WindowGroupingParams

PREFIX = "ms_spectral_deconv/"
_MZ_SEPARATORS = re.compile(r"[,\s]+")

_DECONV_FIELDS = (  # (field, type)
    ("noise_percentile", float),
    ("search_range", float),
    ("min_peak_height", float),
    ("min_top_edge_ratio", float),
    ("min_points", int),
    ("rt_tolerance", float),
    ("min_r", float),
    ("min_ions", int),
)


def parse_mz_list(text: str, strict: bool = False) -> list[float]:
    """Parse '73, 147 207' into [73.0, 147.0, 207.0]; commas and/or whitespace separate values.

    A token that is not a finite number above 0 is bad: strict=True raises
    ValueError naming it, otherwise it is skipped.
    """
    values: list[float] = []
    for token in _MZ_SEPARATORS.split(text or ""):
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            problem = f"'{token}' is not a number"
        else:
            if math.isfinite(value) and value > 0:
                values.append(value)
                continue
            problem = (f"'{token}' is not a finite number" if not math.isfinite(value)
                       else f"'{token}' is not above 0")
        if strict:
            raise ValueError(problem)
    return values


def load_spectral_deconv_settings(s):
    defaults = CorrelationDeconvParams()
    kwargs = {name: s.value(PREFIX + name, getattr(defaults, name), typ)
              for name, typ in _DECONV_FIELDS}
    kwargs["excluded_mz"] = parse_mz_list(s.value(PREFIX + "excluded_mz", "", str))
    dp = CorrelationDeconvParams(**kwargs)

    g_def = WindowGroupingParams()
    raw_gap = s.value(PREFIX + "gap_tolerance", 0.0, float)
    gp = WindowGroupingParams(
        gap_tolerance=None if raw_gap == 0.0 else raw_gap,
        padding_fraction=s.value(PREFIX + "padding_fraction", g_def.padding_fraction, float),
        rt_match_tolerance=s.value(PREFIX + "rt_match_tolerance", g_def.rt_match_tolerance, float),
        max_window_peaks=s.value(PREFIX + "max_window_peaks", g_def.max_window_peaks, int),
    )
    return dp, gp


def save_spectral_deconv_settings(s, dp: CorrelationDeconvParams, gp: WindowGroupingParams) -> None:
    for name, _ in _DECONV_FIELDS:
        s.setValue(PREFIX + name, getattr(dp, name))
    s.setValue(PREFIX + "excluded_mz", ", ".join(str(float(m)) for m in dp.excluded_mz))
    s.setValue(PREFIX + "gap_tolerance", 0.0 if gp.gap_tolerance is None else gp.gap_tolerance)
    s.setValue(PREFIX + "padding_fraction", gp.padding_fraction)
    s.setValue(PREFIX + "rt_match_tolerance", gp.rt_match_tolerance)
    s.setValue(PREFIX + "max_window_peaks", gp.max_window_peaks)
