"""Readers for SepSolve/Markes GCxGC raw files: .HDR, .rsd, .lsc and .dbc.lsc.

Layouts were audited from the bytes of a real run on 2026-10-09 (design spec
2026-10-09-gcxgc-usable-design.md, Appendix A). Readers are pure: no Qt, no app state.
"""
from __future__ import annotations

import logging
import mmap
import re
import struct
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: SD04 flag bits seen only on near-saturated (0x2...) and saturated (0x3...) scans.
SATURATION_MASK = 0xF0000000

_REC = np.dtype([("reserved", "<u4"), ("mz1000", "<u4"), ("inten", "<f8")])
_SD04_HEADER = 24          # tag(4) size(4) u64 timestamp u32 flags u32 count
_FIXED_POINT = float(2 ** 32)
_CHUNK = 65536


class SepSolveFormatError(ValueError):
    """A SepSolve file does not have the structure the readers expect."""

    def __init__(self, path, offset: int, message: str):
        super().__init__(f"{path} (byte {offset}): {message}")
        self.path = str(path)
        self.offset = offset


# ── .HDR ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HdrInfo:
    modulation_period_s: float | None
    fid_rate_hz: float | None
    tof_rate_hz: float | None
    method_name: str | None


def parse_hdr_text(text: str) -> HdrInfo:
    """Pull the fields ChromaKit needs out of HDR text. Missing fields are None."""
    pm = fid = tof = name = None
    section = None
    in_fid_signal = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section, in_fid_signal = line[1:-1], False
            continue
        if line.lstrip(";").startswith("_Signal"):
            in_fid_signal = False
            continue
        if line.startswith(";") or "=" not in line:
            continue
        key, _, val = (s.strip() for s in line.partition("="))
        if section == "INSIGHT Modulator" and pm is None and re.fullmatch(r"Valve\d+ Change0", key):
            fields = val.split(",")
            if len(fields) >= 2:
                pm = float(fields[1])          # PM, flush ms, hold min (field 1 unknown)
        elif key == "Details":
            in_fid_signal = "(FID)" in val
        elif key == "Data Rate" and in_fid_signal and fid is None:
            fid = float(val)
        elif key == "scanset period" and tof is None and float(val) > 0:
            tof = 1e6 / float(val)             # µs per TOF spectrum
        elif section == "TOF-DS Method" and key == "Name" and name is None:
            name = val
    return HdrInfo(pm, fid, tof, name)


def read_hdr(path) -> HdrInfo:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_hdr_text(f.read())


# ── .rsd ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RsdInfo:
    rate_hz: float
    n_points: int
    n_blocks: int


def _varint(b: bytes, p: int, path) -> tuple[int, int]:
    value = shift = 0
    while True:
        if p >= len(b):
            raise SepSolveFormatError(path, p, "string length runs past end of file")
        byte = b[p]
        p += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, p


def _dotnet_str(b: bytes, p: int, path) -> tuple[str, int]:
    n, p = _varint(b, p, path)
    if p + n > len(b):
        raise SepSolveFormatError(path, p, f"string of {n} bytes runs past end of file")
    return b[p:p + n].decode("utf-8", "replace"), p + n


def read_rsd(path) -> tuple[np.ndarray, RsdInfo]:
    """Read the FID trace from an .rsd file by walking its block structure."""
    with open(path, "rb") as f:
        b = f.read()
    if len(b) < 4:
        raise SepSolveFormatError(path, 0, "file too short for an .rsd header")
    (chan_off,) = struct.unpack_from("<I", b, 0)
    if not 4 <= chan_off < len(b):
        raise SepSolveFormatError(path, 0, f"channel header offset {chan_off} is outside the file")
    p = chan_off + 7                        # 7 bytes of unknown meaning
    _, p = _dotnet_str(b, p, path)          # channel name, e.g. "Front Signal"
    if p + 22 > len(b):
        raise SepSolveFormatError(path, p, "channel header truncated")
    p += 8                                  # f64, 0.0 in every file seen (start offset?)
    (rate,) = struct.unpack_from("<i", b, p)
    p += 4 + 2 + 8                          # rate, u16 (2 seen), .NET DateTime
    for _ in range(4):                      # original path, source id, description, "0"
        _, p = _dotnet_str(b, p, path)
    if rate <= 0:
        raise SepSolveFormatError(path, chan_off, f"non-positive data rate {rate}")
    chunks = []
    while p < len(b):
        if p + 8 > len(b):
            raise SepSolveFormatError(path, p, "truncated block header")
        (selfptr,) = struct.unpack_from("<Q", b, p)
        if selfptr == 0:                    # u64 0 terminator
            break
        if selfptr != p + 8 or p + 12 > len(b):
            raise SepSolveFormatError(path, p, f"block self-pointer {selfptr} != {p + 8}")
        (n,) = struct.unpack_from("<I", b, p + 8)
        end = p + 12 + 8 * n
        if end > len(b):
            raise SepSolveFormatError(path, p, f"block of {n} values runs past end of file")
        chunks.append(np.frombuffer(b, "<f8", count=n, offset=p + 12))
        p = end
    signal = np.concatenate(chunks).astype(np.float64) if chunks else np.empty(0)
    return signal, RsdInfo(float(rate), len(signal), len(chunks))


# ── .lsc / .dbc.lsc ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LscView:
    """ChromSpace's saved GCxGC view, from the PD01 text ([BenchTOF File] section)."""
    modulation_period_s: float | None = None
    phase_shift_s: float | None = None
    cut_interval_s: float | None = None


def parse_view(text: str) -> LscView:
    def num(pattern):
        m = re.search(pattern, text, re.M)
        return float(m.group(1)) if m else None
    return LscView(
        modulation_period_s=num(r"^Modulation period=[^\r\n]*?Time:([-+0-9.eE]+)"),
        phase_shift_s=num(r"^Phase shift=([-+0-9.eE]+)"),
        cut_interval_s=num(r"^Cut interval=([-+0-9.eE]+)"),
    )


def nominal_bin(mz, intensity) -> tuple[np.ndarray, np.ndarray]:
    """Sum intensities into unit-mass bins [m - 0.5, m + 0.5), as ChromSpace does."""
    mz = np.asarray(mz, float)
    if len(mz) == 0:
        return np.empty(0, dtype=int), np.empty(0)
    u, inv = np.unique(np.floor(mz + 0.5).astype(int), return_inverse=True)
    return u, np.bincount(inv, weights=np.asarray(intensity, float))


class LscFile:
    """Random access to a Markes line-spectrum (.lsc / .dbc.lsc) file."""

    def __init__(self, path):
        self.path = str(path)
        self._f = open(path, "rb")
        try:
            self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as exc:
            self._f.close()
            raise SepSolveFormatError(path, 0, f"cannot map file ({exc})") from exc
        self._tic = None
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    # -- structure -------------------------------------------------------

    def _parse(self) -> None:
        mm = self._mm
        if len(mm) < 16:
            raise SepSolveFormatError(self.path, 0, "file too short for an FT04 header")
        tag, size, di_off = struct.unpack_from("<4sIQ", mm, 0)
        if tag != b"FT04" or size != 16:
            raise SepSolveFormatError(self.path, 0, f"expected FT04 header, found {tag!r} (size {size})")
        if 0 < di_off < len(mm):
            self.index = self._read_directory(di_off)
        else:
            log.warning("%s: no DI02 scan directory; walking blocks sequentially", self.path)
            self.index = self._walk_chunks()
        self._offs = self.index.get("SD04", np.empty(0, np.uint64)).astype(np.int64)
        self.n_scans = len(self._offs)
        if self.n_scans and int(self._offs.max()) + _SD04_HEADER > len(mm):
            raise SepSolveFormatError(self.path, int(self._offs.max()), "scan offset beyond end of file")
        head = np.empty((self.n_scans, _SD04_HEADER), np.uint8)
        buf = np.frombuffer(mm, np.uint8)
        cols = np.arange(_SD04_HEADER)
        for s in range(0, self.n_scans, _CHUNK):
            head[s:s + _CHUNK] = buf[self._offs[s:s + _CHUNK, None] + cols]
        del buf
        bad = np.flatnonzero((head[:, :4] != np.frombuffer(b"SD04", np.uint8)).any(axis=1))
        if len(bad):
            raise SepSolveFormatError(self.path, int(self._offs[bad[0]]), "directory entry does not point at an SD04 block")
        sizes = head[:, 4:8].copy().view("<u4").ravel()
        self.timestamps = head[:, 8:16].copy().view("<u8").ravel() / _FIXED_POINT
        self.flags = head[:, 16:20].copy().view("<u4").ravel()
        self.counts = head[:, 20:24].copy().view("<u4").ravel()
        wrong = np.flatnonzero(sizes.astype(np.int64) != 28 + 16 * self.counts.astype(np.int64))
        if len(wrong):
            raise SepSolveFormatError(self.path, int(self._offs[wrong[0]]), "SD04 block size does not match its peak count")
        self.view = parse_view(self.text_chunk("PD01") or "")

    def _read_directory(self, di_off: int) -> dict:
        mm = self._mm
        if di_off + 8 > len(mm):
            raise SepSolveFormatError(self.path, di_off, "DI02 offset beyond end of file")
        tag, size = struct.unpack_from("<4sI", mm, di_off)
        if tag != b"DI02" or di_off + size > len(mm):
            raise SepSolveFormatError(self.path, di_off, f"expected DI02 directory, found {tag!r}")
        index, p, end = {}, di_off + 8, di_off + size
        while p < end:
            t, n = struct.unpack_from("<4sI", mm, p)
            p += 8
            if p + 8 * n > end:
                raise SepSolveFormatError(self.path, p, f"directory entry {t!r} runs past the directory")
            index[t.decode("ascii", "replace")] = np.frombuffer(mm, "<u8", count=n, offset=p).copy()
            p += 8 * n
        return index

    def _walk_chunks(self) -> dict:
        mm, p, found = self._mm, 16, {}
        while p + 8 <= len(mm):
            tag, size = struct.unpack_from("<4sI", mm, p)
            if size < 8 or p + size > len(mm):
                raise SepSolveFormatError(self.path, p, f"chunk {tag!r} has impossible size {size}")
            key = tag.decode("ascii", "replace")
            if key == "DI02":
                break
            found.setdefault(key, []).append(p)
            p += size
        return {k: np.asarray(v, np.uint64) for k, v in found.items()}

    def text_chunk(self, tag: str) -> str | None:
        offs = self.index.get(tag)
        if offs is None or len(offs) == 0 or int(offs[0]) == 0:
            return None
        o = int(offs[0])
        t, size = struct.unpack_from("<4sI", self._mm, o)
        if t.decode("ascii", "replace") != tag:
            raise SepSolveFormatError(self.path, o, f"directory says {tag} but found {t!r}")
        return bytes(self._mm[o + 8:o + size]).decode("utf-8", "replace")

    # -- scans -----------------------------------------------------------

    def _check(self, k: int) -> int:
        k = int(k)
        if not 0 <= k < self.n_scans:
            raise IndexError(f"scan {k} outside 0..{self.n_scans - 1} in {self.path}")
        return k

    def raw_spectrum(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = self._check(k)
        n = int(self.counts[k])
        if n == 0:
            return np.empty(0), np.empty(0)
        rec = np.frombuffer(self._mm, _REC, count=n, offset=int(self._offs[k]) + _SD04_HEADER)
        return rec["mz1000"] / 1000.0, rec["inten"].copy()

    def spectrum(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        return nominal_bin(*self.raw_spectrum(k))

    def tic(self) -> np.ndarray:
        if self._tic is None:
            out = np.zeros(self.n_scans)
            mm, offs, counts = self._mm, self._offs, self.counts
            for k in range(self.n_scans):
                n = int(counts[k])
                if n:
                    out[k] = np.frombuffer(mm, _REC, count=n, offset=int(offs[k]) + _SD04_HEADER)["inten"].sum()
            self._tic = out
        return self._tic

    def is_saturated(self, k: int) -> bool:
        return bool(int(self.flags[self._check(k)]) & SATURATION_MASK)

    # -- lifetime --------------------------------------------------------

    def close(self) -> None:
        mm, f = getattr(self, "_mm", None), getattr(self, "_f", None)
        self._mm = None
        if mm is not None:
            mm.close()
        if f is not None:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
