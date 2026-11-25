"""Detector positions: a GC batch row names its Front or Back detector instead of a channel.

A sample's channels are taken in the order the data file lists them: the first is
the Front detector and the second the Back detector, whatever they are called
(FID1A, TCD3C, ...). This lets one row match each GC's channel without exact names.
"""
from __future__ import annotations

FRONT = "front"
BACK = "back"
POSITIONS = (FRONT, BACK)
LABELS = {FRONT: "Front Detector", BACK: "Back Detector"}


def is_position(name) -> bool:
    return isinstance(name, str) and name.strip().lower() in POSITIONS


def label(name: str) -> str:
    """'Front Detector' / 'Back Detector' for a position, else the name unchanged."""
    return LABELS[name.strip().lower()] if is_position(name) else name


def channel_for(position: str, available) -> str | None:
    """The sample's channel at *position*, or None when it has too few channels."""
    index = POSITIONS.index(position.strip().lower())
    available = list(available or [])
    return available[index] if len(available) > index else None
