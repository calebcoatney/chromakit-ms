"""GC1&2 joint-injection workbook: injection records and GC1/GC2 pairing.

Pure logic, no Qt. Injections from the two instruments are paired by
acquisition time (nearest within ``MAX_PAIR_GAP_S``); the injection counter
in the .D name is a cross-check, and the fallback for undated injections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

import rainbow as rb

from logic.paste_blocks import parse_timestamp

MAX_PAIR_GAP_S = 300

_COUNTER_RE = re.compile(r"(\d+)(?:\.D)?\s*$", re.IGNORECASE)


@dataclass
class ChannelResult:
    detector: str
    status: str  # "ok" | "failed" | "missing" | "not_processed"
    message: str = ""
    json_path: str | None = None


@dataclass
class Injection:
    c_path: str
    instrument: str  # "GC1" | "GC2"
    d_name: str | None
    sample: str
    acquired: datetime | None
    counter: int | None
    channels: list[ChannelResult] = field(default_factory=list)


@dataclass
class Pair:
    gc1: Injection | None
    gc2: Injection | None
    dt_s: float | None
    counter_match: bool | None
    notes: list[str]


def injection_counter(name: str) -> int | None:
    """Trailing injection number of a .D / sample name, or None."""
    m = _COUNTER_RE.search(name)
    return int(m.group(1)) if m else None


def acquisition_time(d_path: str, detector: str | None = None) -> datetime | None:
    """Acquisition time from the .D's own metadata; None when unknown.

    Never falls back to the current time or file mtime.
    """
    try:
        data_dir = rb.read(d_path)
        files = list(data_dir.datafiles)
        ordered = []
        if detector:
            ordered += [f for f in files if f.name.startswith(detector)]
        ordered += [f for f in files if f not in ordered]
        sources = [f.metadata for f in ordered] + [data_dir.metadata]
        for md in sources:
            date = (md or {}).get("date")
            if date:
                parsed = parse_timestamp(str(date))
                if parsed is not None:
                    return parsed
    except Exception:
        return None
    return None


def _counters_match(a: Injection, b: Injection) -> bool | None:
    if a.counter is None or b.counter is None:
        return None
    return a.counter == b.counter


def _make_pair(a: Injection, b: Injection, dt_s: float | None, notes: list[str]) -> Pair:
    match = _counters_match(a, b)
    notes = list(notes)
    if match is False:
        notes.append(f"injection counters differ ({a.counter} vs {b.counter})")
    return Pair(a, b, dt_s, match, notes)


def pair_injections(
    gc1: list[Injection], gc2: list[Injection], max_gap_s: float = MAX_PAIR_GAP_S
) -> list[Pair]:
    free1 = set(range(len(gc1)))
    free2 = set(range(len(gc2)))
    pairs: list[Pair] = []

    # 1. nearest acquisition time within the window
    candidates = sorted(
        (abs((a.acquired - b.acquired).total_seconds()), i, j)
        for i, a in enumerate(gc1) if a.acquired is not None
        for j, b in enumerate(gc2) if b.acquired is not None
        if abs((a.acquired - b.acquired).total_seconds()) <= max_gap_s
    )
    for dt, i, j in candidates:
        if i in free1 and j in free2:
            free1.discard(i)
            free2.discard(j)
            pairs.append(_make_pair(gc1[i], gc2[j], dt, []))

    # 2. leftovers where a side is undated: pair on a unique shared counter
    for i in sorted(free1):
        a = gc1[i]
        if a.counter is None:
            continue
        mates = [j for j in free2 if gc2[j].counter == a.counter]
        rivals = [k for k in free1 if gc1[k].counter == a.counter]
        j = mates[0] if len(mates) == 1 else None
        if j is None or len(rivals) != 1:
            continue
        if a.acquired is not None and gc2[j].acquired is not None:
            continue  # both timed but outside the window: stays unpaired
        free1.discard(i)
        free2.discard(j)
        pairs.append(_make_pair(a, gc2[j], None,
                                ["paired by injection counter only (no acquisition time)"]))

    # 3. singles
    for i in free1:
        pairs.append(Pair(gc1[i], None, None, None, ["no GC2 injection within 5 min"]))
    for j in free2:
        pairs.append(Pair(None, gc2[j], None, None, ["no GC1 injection within 5 min"]))

    # 4. order by time, untimed last
    def key(p: Pair):
        t = (p.gc1.acquired if p.gc1 else None) or (p.gc2.acquired if p.gc2 else None)
        sample = (p.gc1 or p.gc2).sample
        return (t is None, t or datetime.min, sample)

    return sorted(pairs, key=key)
