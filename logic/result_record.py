# logic/result_record.py
"""Read an already-recorded ChromaKit result JSON and normalize it.

Nine distinct peak-key schemas exist in the wild (surveyed across 453 records
on 2026-09-04). This module turns any of them into one shape. It recomputes
nothing: every number it returns was written by the run that produced it.

The "missing stays missing" discipline is adopted from
`rapids_bridge/artefact.py` so the two implementations can be argued against
each other -- a bound the record does not have is None, never the apex.
"""
from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class RecordParseError(Exception):
    """The record exists but is not readable JSON."""


@dataclass(frozen=True)
class ResultRecord:
    sample_id: Optional[str]
    timestamp: Optional[str]
    detector: Optional[str]
    acquisition_method: Optional[str]
    record_path: str
    signal_path: str
    signal_factor: Optional[float]
    processing_parameters: Optional[dict]
    quantitation: Optional[dict]
    peaks: list[RecordPeak] = field(default_factory=list)


_START_KEYS = ("rt_start", "start_time", "start")
_END_KEYS = ("rt_end", "end_time", "end")
_APEX_KEYS = ("retention_time", "rt_apex", "apex")

# Consumed into named fields; everything else rides along in `extras`.
_MAPPED_KEYS = frozenset({
    "compound_id", "Compound ID", "retention_time", "rt_apex", "apex",
    "rt_start", "start_time", "start", "rt_end", "end_time", "end",
    "area", "raw_amount", "mol_percent",
})


@dataclass(frozen=True)
class RecordPeak:
    compound: Optional[str]
    rt_apex: Optional[float]
    rt_start: Optional[float]
    rt_end: Optional[float]
    area: Optional[float]
    raw_amount: Optional[float]
    mol_percent: Optional[float]
    extras: dict = field(default_factory=dict)


def _number(value: Any) -> Optional[float]:
    """A plain finite float, or None.

    Whitelisted on numbers.Real, which admits every numpy scalar and rejects
    str/bytes -- float("13.31") would otherwise become a convincing retention
    time. bool is excluded because True is Real and 1.0 mol% is not what a flag
    meant.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _first(peak: dict, keys) -> Optional[float]:
    """The first of `keys` present with a usable number.

    A key that is present but unusable stops the search: the peak said which
    field it meant, and falling through would answer with another field.
    """
    for key in keys:
        if peak.get(key) is not None:
            return _number(peak[key])
    return None


def normalize_peak(peak: dict) -> RecordPeak:
    """One on-disk peak, in any of the nine schemas, as one shape."""
    if not isinstance(peak, dict):
        peak = {}
    compound = peak.get("compound_id")
    if compound is None:
        compound = peak.get("Compound ID")     # 33 older records spell it this way
    return RecordPeak(
        compound=None if compound is None else str(compound),
        rt_apex=_first(peak, _APEX_KEYS),
        rt_start=_first(peak, _START_KEYS),
        rt_end=_first(peak, _END_KEYS),
        area=_first(peak, ("area",)),
        raw_amount=_first(peak, ("raw_amount",)),
        mol_percent=_first(peak, ("mol_percent",)),
        extras={k: v for k, v in peak.items() if k not in _MAPPED_KEYS},
    )


def _candidate_dir(sample_path: str) -> Path:
    path = Path(sample_path)
    return path / "results" if str(path).rstrip(os.sep).endswith(".C") else path


def find_record(sample_path: str, detector: str) -> Optional[str]:
    """The record file for this detector, or None.

    Selects by reading each candidate's `detector` field rather than by
    reconstructing the filename: filenames drift, and reconstructing a .D's
    canonical name would need a rainbow read purely to learn `notebook`.

    Raises:
        RecordParseError: a file named `<name> - <detector>.json` exists but is
            not valid JSON.
    """
    directory = _candidate_dir(sample_path)
    if not directory.is_dir():
        return None

    suffix = f" - {detector}.json"
    canonical: list[Path] = []
    others: list[Path] = []

    for candidate in sorted(directory.glob("*.json")):
        named_for_us = candidate.name.endswith(suffix)
        try:
            doc = json.loads(candidate.read_text())
        except Exception as exc:
            if named_for_us:
                # A file named as THIS detector's record that will not parse is
                # corrupt. Skipping it would report the sample as "not analysed"
                # and hide the broken file indefinitely.
                raise RecordParseError(
                    f"could not parse record {candidate}: {exc}") from exc
            continue          # an unrelated neighbour we cannot read is not ours
        if isinstance(doc, dict) and doc.get("detector") == detector:
            (canonical if named_for_us else others).append(candidate)

    # A canonical `<name> - <detector>.json` beats a sidecar such as
    # `<name> - <detector>.pre-fix-20260612.json`, which shares the directory.
    if canonical:
        return str(canonical[0])
    if others:
        return str(max(others, key=lambda m: m.stat().st_mtime))
    return None


def _signal_factor(doc: dict) -> Optional[float]:
    processing_parameters = doc.get("processing_parameters")
    if not isinstance(processing_parameters, dict):
        return None
    scaling = processing_parameters.get("scaling")
    if not isinstance(scaling, dict):
        return None
    return _number(scaling.get("signal_factor"))


def read_record(sample_path: str, detector: str) -> Optional[ResultRecord]:
    """The normalized record, or None if this detector has no usable one.

    Raises:
        RecordParseError: the record file exists but is not valid JSON.
    """
    record_path = find_record(sample_path, detector)
    if record_path is None:
        return None
    try:
        doc = json.loads(Path(record_path).read_text())
    except Exception as exc:
        raise RecordParseError(f"could not parse record {record_path}: {exc}") from exc

    peaks = doc.get("peaks")
    if not isinstance(peaks, list) or not peaks:
        return None

    source_d = doc.get("source_d")
    signal_path = str(source_d) if source_d else str(sample_path)

    return ResultRecord(
        sample_id=doc.get("sample_id"),
        timestamp=doc.get("timestamp"),
        detector=doc.get("detector"),
        acquisition_method=doc.get("method"),
        record_path=record_path,
        signal_path=signal_path,
        signal_factor=_signal_factor(doc),
        processing_parameters=doc.get("processing_parameters"),
        quantitation=doc.get("quantitation"),
        peaks=[normalize_peak(p) for p in peaks if isinstance(p, dict)],
    )
