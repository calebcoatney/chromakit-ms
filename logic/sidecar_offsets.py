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


def load_offset(data_path: str, sidecar_path: Optional[Path] = None) -> Optional[OffsetEntry]:
    """Return the saved offset for `data_path`, or None if absent / unreadable."""
    path = _resolve_path(sidecar_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
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
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
    path = _resolve_path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    data[str(data_path)] = {
        "offset_min": float(offset_min),
        "timestamp": time.time(),
        "source": source,
    }
    path.write_text(json.dumps(data, indent=2))


def save_offsets_batch(
    data_paths: list,
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

    Raises:
        ValueError: source is not in VALID_SOURCES.

    No-op when data_paths is empty.
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")
    if not data_paths:
        return
    path = _resolve_path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, json.JSONDecodeError):
            data = {}
    now = time.time()
    for p in data_paths:
        data[str(p)] = {
            "offset_min": float(offset_min),
            "timestamp": now,
            "source": source,
        }
    path.write_text(json.dumps(data, indent=2))
