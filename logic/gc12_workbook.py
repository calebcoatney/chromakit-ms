"""GC1&2 joint-injection workbook: injection records and GC1/GC2 pairing.

Pure logic, no Qt. Injections from the two instruments are paired by
acquisition time (nearest within ``MAX_PAIR_GAP_S``); the injection counter
in the .D name is a cross-check, and the fallback for undated injections.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import openpyxl
import rainbow as rb
from openpyxl.styles import Font

from logic.paste_blocks import (
    GC12_PASTE_FORMAT_CONFIG,
    channel_width,
    parse_timestamp,
    write_channel_block,
)

logger = logging.getLogger(__name__)

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
    except Exception as exc:
        logger.warning("Could not read acquisition time from %s: %s", d_path, exc)
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

    # 3. singles: say why (outside the window, or never dated so only the counter could pair it)
    minutes = f"{max_gap_s / 60:g} min"

    def single_note(inj: Injection, other: str) -> str:
        if inj.acquired is None:
            return "no acquisition time; no unique injection-counter match"
        return f"no {other} injection within {minutes}"

    for i in free1:
        pairs.append(Pair(gc1[i], None, None, None, [single_note(gc1[i], "GC2")]))
    for j in free2:
        pairs.append(Pair(None, gc2[j], None, None, [single_note(gc2[j], "GC1")]))

    # 4. order by time, untimed last
    def key(p: Pair):
        t = (p.gc1.acquired if p.gc1 else None) or (p.gc2.acquired if p.gc2 else None)
        sample = (p.gc1 or p.gc2).sample
        return (t is None, t or datetime.min, sample)

    return sorted(pairs, key=key)


# ---------------------------------------------------------------------------
# Collecting injections from a processed instrument folder
# ---------------------------------------------------------------------------

RunRecords = dict[tuple[str, str], tuple[bool, str]]  # (c_path, detector) -> (success, message)


def _channel_result(c_path: str, sample: str, detector: str, records: RunRecords) -> ChannelResult:
    record = records.get((c_path, detector))
    if record is None:
        return ChannelResult(detector, "not_processed", "no result from this job")
    ok, message = record
    if not ok:
        if message.startswith("Detector ") and "not found" in message:
            return ChannelResult(detector, "missing", message)
        return ChannelResult(detector, "failed", message)
    # Only a recorded success may use a JSON on disk; a stale file never counts.
    json_path = os.path.join(c_path, "results", f"{sample} - {detector}.json")
    if not os.path.isfile(json_path):
        return ChannelResult(detector, "failed", "no result file")
    return ChannelResult(detector, "ok", message, json_path)


def collect_injections(directory: str, instrument: str, detectors: list[str],
                       records: RunRecords) -> list[Injection]:
    """One Injection per .C folder of *directory* (same listing rule as AutomationWorker.run)."""
    injections = []
    for item in sorted(os.listdir(directory)):
        c_path = os.path.join(directory, item)
        if not (os.path.isdir(c_path) and item.endswith(".C")):
            continue
        stem = item[:-2]
        sample = stem
        manifest = os.path.join(c_path, "manifest.json")
        if os.path.isfile(manifest):
            with open(manifest, encoding="utf-8") as f:
                sample = json.load(f).get("sample_id") or stem
        d_dirs = sorted(d for d in glob.glob(os.path.join(glob.escape(c_path), "data", "*.D"))
                        if os.path.isdir(d))
        d_name = os.path.basename(d_dirs[0]) if d_dirs else None
        acquired = acquisition_time(d_dirs[0], detectors[0]) if d_dirs and detectors else None
        injections.append(Injection(
            c_path=c_path, instrument=instrument, d_name=d_name, sample=sample,
            acquired=acquired, counter=injection_counter(d_name or stem),
            channels=[_channel_result(c_path, sample, d, records) for d in detectors],
        ))
    return injections


# ---------------------------------------------------------------------------
# Writing the paired workbook
# ---------------------------------------------------------------------------

_NO_TIME = "NO ACQUISITION TIME"
_CFG = GC12_PASTE_FORMAT_CONFIG
_STRIDE = channel_width(_CFG) + 1
_MAX_CHANNELS = 2  # a third block would spill past column O, over the workbook's matching table
_PAIRING_HEADER = ["Pair", "GC1 injection", "GC1 time", "GC2 injection", "GC2 time",
                   "Δt (s)", "Counter match", "Channels OK", "Notes"]


@dataclass
class WorkbookSummary:
    pairs: int
    gc1_unpaired: int
    gc2_unpaired: int
    channel_problems: int

    def message(self) -> str:
        def n(count, noun):
            return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
        return (f"{n(self.pairs, 'pair')}, {n(self.gc1_unpaired, 'GC1 injection')} unpaired, "
                f"{n(self.gc2_unpaired, 'GC2 injection')} unpaired, "
                f"{n(self.channel_problems, 'channel problem')} — see Pairing sheet")


def _status_line(ch: ChannelResult, peak_count: int | None = None) -> str | None:
    if ch.status == "failed":
        return f"FAILED: {ch.message}"
    if ch.status == "missing":
        return f"MISSING: channel {ch.detector} not in this injection"
    if ch.status == "not_processed":
        return f"NOT PROCESSED: {ch.message or 'no result from this job'}"
    return "NO PEAKS FOUND" if peak_count == 0 else None


def _timestamp_value(inj: Injection | None, sheet_is_gc1: bool):
    if not sheet_is_gc1:
        return ""
    return inj.acquired if inj is not None and inj.acquired is not None else _NO_TIME


def _read_json(ch: ChannelResult) -> tuple[dict | None, ChannelResult]:
    """JSON data for an ok channel; a file that cannot be read turns it into a failure."""
    try:
        with open(ch.json_path, encoding="utf-8") as f:
            return json.load(f), ch
    except (OSError, ValueError) as exc:
        return None, ChannelResult(ch.detector, "failed", f"unreadable result file ({exc})")


def _write_status_row(ws, row: int, col: int, text: str) -> int:
    ws.cell(row=row, column=col, value=text)
    return row + 1


def _write_injection_block(ws, row: int, inj: Injection, sheet_is_gc1: bool) -> int:
    ts = _timestamp_value(inj, sheet_is_gc1)
    next_rows = []
    for k, ch in enumerate(inj.channels):
        col = 1 + k * _STRIDE
        data, ch = (_read_json(ch) if ch.status == "ok" else (None, ch))
        if data is None:
            data = {"signal": f"Signal: {inj.sample}\\{ch.detector}.ch",
                    "sample_id": inj.sample, "peaks": []}
        data = {**data, "timestamp": ts}
        nxt, _ = write_channel_block(ws, row, col, data, _CFG,
                                     skip_unidentified=False, skip_unquantitated=False)
        line = _status_line(ch, len(data["peaks"]))
        if line:
            nxt = _write_status_row(ws, nxt, col, line)
        next_rows.append(nxt)
    return _finish_block(ws, row, next_rows, sheet_is_gc1)


def _write_placeholder_block(ws, row: int, other: Injection, sheet_is_gc1: bool, n_channels: int) -> int:
    absent = "GC1" if sheet_is_gc1 else "GC2"
    next_rows = []
    for k in range(n_channels):
        col = 1 + k * _STRIDE
        data = {"signal": "Signal: (none)",
                "sample_id": f"NO {absent} INJECTION PAIRED WITH {other.sample}",
                "timestamp": _timestamp_value(None, sheet_is_gc1), "peaks": []}
        nxt, _ = write_channel_block(ws, row, col, data, _CFG)
        next_rows.append(_write_status_row(ws, nxt, col, "NO INJECTION"))
    return _finish_block(ws, row, next_rows, sheet_is_gc1)


def _finish_block(ws, row: int, next_rows: list[int], sheet_is_gc1: bool) -> int:
    if not sheet_is_gc1:
        # GC2's timestamp row (third header row) must be truly empty for the macro.
        for k in range(len(next_rows)):
            ws.cell(row=row + 2, column=1 + k * _STRIDE).value = None
    return max(next_rows) + 1  # one blank row between blocks


def _count_ok(inj: Injection | None) -> tuple[int, int]:
    if inj is None:
        return 0, 0
    return sum(c.status == "ok" for c in inj.channels), len(inj.channels)


def _write_pairing_sheet(ws, pairs: list[Pair]) -> None:
    ws.append(_PAIRING_HEADER)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for i, p in enumerate(pairs, start=1):
        ok1, n1 = _count_ok(p.gc1)
        ok2, n2 = _count_ok(p.gc2)
        match = "" if p.counter_match is None else ("yes" if p.counter_match else "no")
        ws.append([i,
                   p.gc1.sample if p.gc1 else "", p.gc1.acquired if p.gc1 else None,
                   p.gc2.sample if p.gc2 else "", p.gc2.acquired if p.gc2 else None,
                   None if p.dt_s is None else round(p.dt_s, 1), match,
                   f"{ok1 + ok2}/{n1 + n2}", "; ".join(p.notes)])
        for col in (3, 5):
            ws.cell(row=i + 1, column=col).number_format = _CFG["timestamp_format"]


def write_paired_workbook(pairs: list[Pair], path: str) -> WorkbookSummary:
    """Write the GC1/GC2 paste sheets and the Pairing sheet; injection k is the same on both."""
    book = openpyxl.Workbook()
    ws1 = book.active
    ws1.title = "GC1 raw formatting"
    ws2 = book.create_sheet("GC2 raw formatting")
    pairing = book.create_sheet("Pairing")

    # shortcut: a side with no injections at all gets 1 placeholder channel column, upgrade if the detector count is ever needed there
    for p in pairs:
        for inj in (p.gc1, p.gc2):
            if inj and len(inj.channels) > _MAX_CHANNELS:
                raise ValueError(
                    f"{inj.instrument} has {len(inj.channels)} channels for injection "
                    f"{inj.sample}; the paste layout holds at most {_MAX_CHANNELS} per instrument")
    n1 = max((len(p.gc1.channels) for p in pairs if p.gc1), default=1)
    n2 = max((len(p.gc2.channels) for p in pairs if p.gc2), default=1)
    row1 = row2 = 1
    for p in pairs:
        if p.gc1:
            row1 = _write_injection_block(ws1, row1, p.gc1, True)
        else:
            row1 = _write_placeholder_block(ws1, row1, p.gc2, True, n1)
        if p.gc2:
            row2 = _write_injection_block(ws2, row2, p.gc2, False)
        else:
            row2 = _write_placeholder_block(ws2, row2, p.gc1, False, n2)
    _write_pairing_sheet(pairing, pairs)
    book.save(path)

    problems = sum(c.status != "ok" for p in pairs for inj in (p.gc1, p.gc2) if inj
                   for c in inj.channels)
    return WorkbookSummary(
        pairs=sum(1 for p in pairs if p.gc1 and p.gc2),
        gc1_unpaired=sum(1 for p in pairs if p.gc1 and not p.gc2),
        gc2_unpaired=sum(1 for p in pairs if p.gc2 and not p.gc1),
        channel_problems=problems,
    )
