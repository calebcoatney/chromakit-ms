"""A decimated signal for DRAWING, and nothing else.

The envelope math here is a deliberate copy of
`rapids_bridge/chromatogram.py::_envelope`, pinned by
`tests/logic/test_render_trace_envelope.py::test_matches_rapids_bridge_envelope_exactly`.
The two repos run in separate conda environments on purpose, so they share a
checked-in fixture rather than an import.

Plot the result. Do NOT integrate it, measure a baseline from it, or compute a
peak area from it — the recorded results are the source for all three.
"""
from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MAX_POINTS = 2000
MAX_POINTS_CAP = 20000


@dataclass(frozen=True)
class Trace:
    """`rt_min[i]` is the retention time OF `plot_y[i]`, exactly — both are real
    samples lifted from the file.

    `plot_y` is named for what it is safe to do with it: it is a min/max
    envelope, two samples per bin, so consecutive points are not evenly spaced
    in time and the point count bears no fixed relation to the acquisition rate.
    """

    rt_min: list[float]
    plot_y: list[float]
    points: int
    decimation: int


def _finite_floats(values) -> list[float] | None:
    """Plain finite Python floats, or None if any value is not one.

    Whitelisted on numbers.Real: every numpy scalar registers, while str and
    bytes do not — and float() would turn "13.31" into a convincing intensity.
    """
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def envelope(xs: list[float], ys: list[float], width: int) -> tuple[list[float], list[float]]:
    """Per bin of `width` samples, the lowest and highest signal, each still
    carrying its own real retention time.

    Not striding: striding decides which samples survive by position, so a peak
    apex survives only by luck and both its drawn height and drawn time come out
    slightly wrong in a way nothing downstream can detect.
    """
    out_x: list[float] = []
    out_y: list[float] = []
    for start in range(0, len(ys), width):
        stop = min(start + width, len(ys))
        lo = hi = start
        for i in range(start + 1, stop):
            if ys[i] < ys[lo]:
                lo = i
            elif ys[i] > ys[hi]:
                hi = i
        for i in sorted({lo, hi}):
            out_x.append(xs[i])
            out_y.append(ys[i])
    return out_x, out_y


def _validate_max_points(max_points: int) -> None:
    """Shared by `build_trace` and `read_trace` so the rule cannot drift
    between the two copies that used to exist here.

    Raises:
        ValueError: max_points is not an int in [2, MAX_POINTS_CAP]. Two is the
            floor because a bin emits a minimum and a maximum.
    """
    if isinstance(max_points, bool) or not isinstance(max_points, int) \
            or max_points < 2 or max_points > MAX_POINTS_CAP:
        raise ValueError(
            f"max_points must be an int in [2, {MAX_POINTS_CAP}], got {max_points!r}"
        )


def build_trace(xs, ys, max_points: int = DEFAULT_MAX_POINTS) -> Trace | None:
    """Validate and decimate. Bad DATA returns None; a bad ARGUMENT raises.

    Raises:
        ValueError: max_points is not an int in [2, MAX_POINTS_CAP]. Two is the
            floor because a bin emits a minimum and a maximum.
    """
    _validate_max_points(max_points)

    xs = list(xs)
    ys = list(ys)

    # rainbow rebuilds the time axis arithmetically, so it is not structurally
    # prevented from ending one sample past the data. One trailing time with no
    # intensity is dropped. The tolerance is one-sided: a surplus INTENSITY is a
    # real sample with no retention time, and pairing them would shift the trace.
    surplus = len(xs) - len(ys)
    if surplus < 0 or surplus > 1:
        return None
    keep = len(ys)
    if keep == 0:
        return None
    xs = xs[:keep]

    # Validate the RAW arrays, not just the samples decimation happens to keep.
    all_xs = _finite_floats(xs)
    all_ys = _finite_floats(ys)
    if all_xs is None or all_ys is None:
        return None

    if any(b <= a for a, b in zip(all_xs, all_xs[1:])):
        return None

    if keep <= max_points:
        return Trace(rt_min=all_xs, plot_y=all_ys, points=keep, decimation=1)

    bins = max(1, max_points // 2)          # two points per bin
    width = max(2, -(-keep // bins))        # ceil division
    out_x, out_y = envelope(all_xs, all_ys, width)
    return Trace(rt_min=out_x, plot_y=out_y, points=len(out_x), decimation=width)


# --- signal reading -------------------------------------------------------
#
# ChromaKit opens the signal its own way rather than reusing rapids-bridge's
# reader, because that reader goes straight to rainbow's `acq.datafiles` and has
# no .C support at all — and `source_d` records live in .C trees. The envelope
# above is shared; the file-opening legitimately is not.
#
# signal_factor is pinned to 1.0 everywhere here: the wire carries RAW COUNTS
# and the factor travels alongside as a number. Baking it in would make the same
# sample yield different values depending on which repo asked.


def _normalize_sample_path(path: str) -> str:
    """Strip a trailing path separator so the suffix checks below agree with
    each other and with the actual directory name on disk.

    Shared by `_is_c_folder` and `is_supported_sample_path`, which used to
    strip differently -- `os.sep` only vs. `/` and `\\` -- and so disagreed
    on a `.C` path with a trailing `/` on Windows (`os.sep == "\\"`), sending
    it down the `.D` branch. Stripping both slash characters regardless of
    platform is deliberate: paths reaching this module can originate from a
    web client, not just the local OS.
    """
    return path.rstrip("/\\")


def _is_c_folder(sample_path: str) -> bool:
    return _normalize_sample_path(sample_path).endswith(".C")


def is_supported_sample_path(path: str) -> bool:
    """True if `path` names an Agilent .D directory or a ChromaKit .C folder.

    The one place a caller outside this module should ask "is this a
    renderable sample path" — a public counterpart to the same suffix rule
    `_is_c_folder` here and `result_record._candidate_dir` each already
    encode inline for their own narrower purposes. Deliberately does not
    replace either: this only centralizes the rule for new callers.
    """
    trimmed = _normalize_sample_path(path)
    return trimmed.endswith(".D") or trimmed.endswith(".C")


def available_detectors(sample_path: str) -> list[str]:
    """Detector channel names in a .D or .C sample. Empty list on any failure."""
    try:
        if _is_c_folder(sample_path):
            from logic.c_folder import CFolder
            return list(CFolder.open(sample_path).get_available_detectors())
        from logic.data_handler import DataHandler
        return list(DataHandler().get_available_detectors(sample_path))
    except Exception:
        logger.warning(
            "available_detectors failed for %r", sample_path, exc_info=True
        )
        return []


def _same_detector(requested: str, resolved: str) -> bool:
    """Case-insensitive match, tolerant of an optional `.ch` suffix on either
    side -- `FID1A`, `fid1a` and `FID1A.ch` all name the same channel
    elsewhere in this codebase (see e.g. DataHandler.get_available_detectors,
    which strips `.ch`)."""
    def key(name: str) -> str:
        name = name.strip()
        if name.lower().endswith(".ch"):
            name = name[:-3]
        return name.lower()
    return key(requested) == key(resolved)


def canonical_detector(requested: str, available: list[str]) -> str:
    """`requested`, rewritten to match `available`'s own spelling when one of
    its entries names the same channel (case-insensitive, `.ch`-tolerant, via
    `_same_detector`) -- so a caller-supplied `fid1a` or `TCD3C.ch` becomes
    the sample's own `FID1A` or `TCD3C` before it is used for anything.

    Returns `requested` unchanged when `available` is empty or nothing in it
    matches -- an unknown or unresolvable name is not this function's problem
    to raise about; the existing read_record/read_trace degrade paths handle
    it downstream exactly as they do today.
    """
    for candidate in available:
        if _same_detector(requested, candidate):
            return candidate
    return requested


def read_signal(sample_path: str, detector: str | None = None) -> tuple[list[float], list[float], str] | None:
    """(xs, ys, resolved_detector) in raw counts, or None if unreadable.

    A fresh DataHandler per call: the module-level singleton in api/main.py has a
    mutable current_detector that every load overwrites, so sharing it would let
    two concurrent requests corrupt each other's detector invisibly.

    Both DataHandler and the .C AgilentLoader silently fall back to a
    different channel when the requested `.ch` is absent, rather than
    raising -- so a caller-supplied `detector` is checked against what was
    actually resolved, and a mismatch degrades to None here rather than
    handing back another channel's trace under the requested detector's
    name. This is the one guarantee this module's docstring makes: never
    return a plausible wrong trace.
    """
    try:
        if _is_c_folder(sample_path):
            from logic.c_folder import CFolder
            folder = CFolder.open(sample_path)
            data = folder.load_signal(signal_factor=1.0, detector=detector)
            resolved = (data.get("metadata") or {}).get("detector") or detector or "Unknown"
            resolved = str(resolved)
            xs, ys = list(data["x"]), list(data["y"])
        else:
            from logic.data_handler import DataHandler
            handler = DataHandler()
            handler.signal_factor = 1.0
            data = handler.load_data_directory(sample_path, detector=detector)
            chrom = data["chromatogram"]
            resolved = str(handler.current_detector)
            xs, ys = list(chrom["x"]), list(chrom["y"])

        if detector is not None and not _same_detector(detector, resolved):
            logger.warning(
                "read_signal: requested detector %r not present in %r "
                "(loader resolved to %r instead) -- refusing to return "
                "another channel's trace", detector, sample_path, resolved,
            )
            return None

        return xs, ys, resolved
    except Exception:
        logger.warning(
            "read_signal failed for %r (detector=%r)", sample_path, detector,
            exc_info=True,
        )
        return None


def read_trace(sample_path: str, detector: str | None = None,
               max_points: int = DEFAULT_MAX_POINTS) -> tuple[Trace | None, str | None]:
    """(Trace, None) on success, (None, reason) on any data failure.

    The argument check runs first and raises, so a mis-read max_points does not
    masquerade as an unreadable instrument file.
    """
    _validate_max_points(max_points)

    signal = read_signal(sample_path, detector=detector)
    if signal is None:
        if detector:
            return None, f"detector {detector} is not present in this sample"
        return None, "no readable detector signal in this sample"
    xs, ys, _ = signal
    trace = build_trace(xs, ys, max_points=max_points)
    if trace is None:
        return None, f"signal for detector {detector or '(auto)'} failed validation"
    return trace, None
