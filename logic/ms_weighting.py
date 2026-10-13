"""Weighting schemes for MS library search: names, labels, storage, default.

ms-toolkit's search_vector / search_hybrid take a weighting-scheme *name*, and
only the keys of ``ms_toolkit.preprocessing.weighting_schemes`` are real names.
Any other string quietly searches unweighted, so ChromaKit offers exactly those
keys and stores the name itself.

This module has no Qt dependency. The settings helpers take any QSettings-like
object (``value`` / ``setValue`` / ``remove``).

Stored format:
  * ``ms_search/weighting_scheme``: the scheme name, e.g. ``"NIST11"``.
  * ``ms_search/weighting``: the integer index older builds wrote
    (0 = None, 1 = "NIST", 2 = NIST_GC). It is still read when the name key is
    absent, and removed the next time settings are saved.
"""
from __future__ import annotations

from functools import lru_cache

# What a fresh install, Restore Defaults and every caller that is not told
# otherwise searches with. NIST11 beat NIST_GC on the NIST replicate benchmark
# (top-1 81.7 % vs 75.3 %) and on real FPO samples.
DEFAULT_WEIGHTING = "NIST11"

WEIGHTING_KEY = "ms_search/weighting_scheme"
LEGACY_WEIGHTING_KEY = "ms_search/weighting"

# The old dropdown was ["None", "NIST", "NIST_GC"]. "NIST" is not an ms-toolkit
# scheme (it searched unweighted); NIST11 is the NIST preset that exists.
LEGACY_INDEX_TO_SCHEME = {0: "None", 1: "NIST11", 2: "NIST_GC"}

_TERM_SEPARATOR = " \u00b7 "

# Order the dropdown shows them in; schemes ms-toolkit adds later follow.
_DISPLAY_ORDER = ("NIST11", "NIST_GC", "MassBank", "SQRT", "None")

# Used only when ms-toolkit cannot be imported. A test keeps it equal to the
# real dict. Each value is (m/z exponent, intensity exponent).
_FALLBACK_SCHEMES = {
    "None": (0, 1),
    "SQRT": (0, 0.5),
    "MassBank": (2, 0.5),
    "NIST11": (1.3, 0.53),
    "NIST_GC": (3, 0.6),
}


@lru_cache(maxsize=1)
def weighting_scheme_table() -> dict:
    """ms-toolkit's ``weighting_schemes`` ({name: (mz_exponent, intensity_exponent)})."""
    try:
        from ms_toolkit.preprocessing import weighting_schemes
    except ImportError:
        return dict(_FALLBACK_SCHEMES)
    return dict(weighting_schemes)


def weighting_scheme_names() -> list:
    """Every scheme name ChromaKit offers, default first."""
    table = weighting_scheme_table()
    ordered = [name for name in _DISPLAY_ORDER if name in table]
    return ordered + [name for name in table if name not in ordered]


def weighting_label(name: str) -> str:
    """Dropdown text that says the exponents, e.g. ``NIST11 (m/z^1.3 · I^0.53)``."""
    a, b = weighting_scheme_table()[name]
    if not a and b == 1:
        return f"{name} (unweighted)"
    terms = ([f"m/z^{a:g}"] if a else []) + [f"I^{b:g}"]
    return f"{name} ({' · '.join(terms)})"


def _legacy_index(raw):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def read_weighting_scheme(settings) -> str:
    """The saved scheme name; the old integer index and the default as fallbacks."""
    table = weighting_scheme_table()
    stored = settings.value(WEIGHTING_KEY, None)
    if isinstance(stored, str) and stored in table:
        return stored
    scheme = LEGACY_INDEX_TO_SCHEME.get(_legacy_index(settings.value(LEGACY_WEIGHTING_KEY, None)))
    if scheme in table:
        return scheme
    return DEFAULT_WEIGHTING


def write_weighting_scheme(settings, name: str) -> None:
    """Save the scheme as a name and retire the old integer key."""
    if name not in weighting_scheme_table():
        raise ValueError(f"{name!r} is not an ms-toolkit weighting scheme")
    settings.setValue(WEIGHTING_KEY, name)
    settings.remove(LEGACY_WEIGHTING_KEY)
