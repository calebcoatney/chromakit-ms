"""Per-file MS time offset sidecar.

Stores user-applied MS time offsets in `overrides/ms_time_offsets.json` keyed
by absolute `.D` directory path. Format::

    {
      "/abs/path/to/sample.D": {
        "offset_min": -0.048,
        "timestamp": 1735000000.0,
        "source": "manual"   // or "auto"
      },
      ...
    }
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

DEFAULT_SIDECAR_PATH = Path("overrides") / "ms_time_offsets.json"
VALID_SOURCES = ("manual", "auto")
Source = Literal["manual", "auto"]


@dataclass(frozen=True)
class OffsetEntry:
    offset_min: float
    timestamp: float
    source: Source


def _resolve_path(sidecar_path: Optional[Path]) -> Path:
    return Path(sidecar_path) if sidecar_path is not None else DEFAULT_SIDECAR_PATH


def _read_sidecar(path: Path) -> dict:
    """Read the sidecar JSON; return {} if missing or unparseable."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _validate_source(source: str) -> None:
    """Raise ValueError if source is not in VALID_SOURCES."""
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")


def load_offset(data_path: str, sidecar_path: Optional[Path] = None) -> Optional[OffsetEntry]:
    """Return the saved offset for `data_path`, or None if absent / unreadable."""
    raw = _read_sidecar(_resolve_path(sidecar_path))
    entry = raw.get(str(data_path))
    if not isinstance(entry, dict):
        return None
    try:
        return OffsetEntry(
            offset_min=float(entry["offset_min"]),
            timestamp=float(entry.get("timestamp", 0.0)),
            source=entry.get("source", "manual"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def save_offset(
    data_path: str,
    offset_min: float,
    source: Source,
    sidecar_path: Optional[Path] = None,
) -> None:
    """Persist `offset_min` for `data_path` in the sidecar (creating it if needed)."""
    _validate_source(source)
    path = _resolve_path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_sidecar(path)
    data[str(data_path)] = {
        "offset_min": float(offset_min),
        "timestamp": time.time(),
        "source": source,
    }
    path.write_text(json.dumps(data, indent=2))


def save_offsets_batch(
    data_paths: list[str],
    offset_min: float,
    source: Source,
    sidecar_path: Optional[Path] = None,
) -> None:
    """Persist the same `offset_min` for every path in `data_paths`.

    Equivalent to looping `save_offset` per path, but reads + writes the
    sidecar JSON file exactly once. All entries share one timestamp.

    Args:
        data_paths: List of absolute `.D` directory paths.
        offset_min: Offset in minutes to apply to every path.
        source: Provenance tag ('manual' or 'auto'); must match VALID_SOURCES.
        sidecar_path: Override default sidecar location (mainly for tests).

    Raises:
        ValueError: source is not in VALID_SOURCES.

    No-op when data_paths is empty.
    """
    _validate_source(source)
    if not data_paths:
        return
    path = _resolve_path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_sidecar(path)
    now = time.time()
    for p in data_paths:
        data[str(p)] = {
            "offset_min": float(offset_min),
            "timestamp": now,
            "source": source,
        }
    path.write_text(json.dumps(data, indent=2))


def load_offsets_for_paths(
    data_paths: list[str],
    sidecar_path: Optional[Path] = None,
) -> dict[str, OffsetEntry]:
    """Return dict mapping data_path -> OffsetEntry for paths that have
    a saved offset. Paths with no saved offset are absent from the result.

    Reads the sidecar JSON exactly once (vs. N reads for per-path lookup).
    Returns empty dict if the sidecar file is missing or corrupt.

    Args:
        data_paths: List of absolute `.D` directory paths to look up.
        sidecar_path: Override default sidecar location (mainly for tests).
    """
    path = _resolve_path(sidecar_path)
    raw = _read_sidecar(path)
    result = {}
    for p in data_paths:
        entry = raw.get(str(p))
        if not isinstance(entry, dict):
            continue
        try:
            result[str(p)] = OffsetEntry(
                offset_min=float(entry["offset_min"]),
                timestamp=float(entry.get("timestamp", 0.0)),
                source=entry.get("source", "manual"),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result
