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
    peaks: list = field(default_factory=list)


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


def _as_number(value: Any) -> Optional[float]:
    """Whitelist on numbers.Real: admits numpy scalars, rejects str/bytes.

    bool is excluded because True/False are instances of numbers.Real.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    return None


def _signal_factor(doc: dict) -> Optional[float]:
    processing_parameters = doc.get("processing_parameters")
    if not isinstance(processing_parameters, dict):
        return None
    scaling = processing_parameters.get("scaling")
    if not isinstance(scaling, dict):
        return None
    return _as_number(scaling.get("signal_factor"))


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
        peaks=[],                          # populated in Task 5
    )
