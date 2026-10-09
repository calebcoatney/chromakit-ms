"""Tiny SepSolve/Markes GCxGC files for tests (layouts per spec Appendix A).

Writers: write_hdr, write_rsd, write_lsc. Scene builder: Peak + Scene + build_run plant
Gaussian GCxGC peaks with known coordinates and spectra into a .C folder, using the
spec's phase convention (a peak at display 2tR tau sits tau seconds after its cycle's
injection, and cycle m's injection is at cut_s + m*PM, cut_s = (-phase) mod PM).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REC = np.dtype([("reserved", "<u4"), ("mz1000", "<u4"), ("inten", "<f8")])
FLAG_NORMAL = 0x20
FLAG_SATURATED = 0x30000020


def dotnet_str(s: str) -> bytes:
    data = s.encode("utf-8")
    n, out = len(data), bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out) + data


def hdr_text(*, pm_s=4.0, fid_hz=50.0, tof_scanset_us=20000, method="Synthetic", insight=True) -> str:
    lines = ["; TOF MS Header File", "[TOF-DS Method]", f"Name={method}", ""]
    if insight:
        lines += ["[INSIGHT Modulator]", "Name=INSIGHT Modulator", "Valve1Type=Backflush",
                  "Valve1 Changes=1", f"Valve1 Change0=1,{pm_s:g},300,999.00", ""]
    lines += ['[GC 1 (GC): "Agilent 8890"]', "_Signals", "_Signal #1:  Front Signal",
              "Description=Front Signal", "Details=Front Signal (FID)", "Save=On",
              f"Data Rate={fid_hz:g}", "_Signal #2:", "Description=None", "Data Rate=10", "",
              "[BenchTOF Acquisition]", "scans per scanset=194",
              f"scanset period={tof_scanset_us}", "",
              # Decoys: the inactive TOF calibrant block the old loader misread as the period.
              "[Mass Calibration Settings 70eV]", "cal modulation period=   10.000000",
              "cal modulation pulse=    0.500000", "cal modulation active=0", ""]
    return "\r\n".join(lines)


def write_hdr(path, **kw) -> None:
    Path(path).write_text(hdr_text(**kw), encoding="utf-8")


def write_rsd(path, signal, *, rate_hz=50, block=97) -> list[int]:
    """Write an .rsd; return the byte offset of each data block."""
    body = bytearray(0x100)
    body += dotnet_str("[TOF-DS Method]\r\nName=Synthetic\r\n")
    chan_off = len(body)
    body += bytes([0x04, 0x1C, 0x09, 0x14, 0x11, 0x02, 0x00])
    body += dotnet_str("Front Signal")
    body += struct.pack("<d", 0.0) + struct.pack("<i", int(rate_hz))
    body += struct.pack("<H", 2) + struct.pack("<Q", 0)
    for s in ("C:\\data\\synthetic.rsd", "GC.0.0", "Front Signal", "0"):
        body += dotnet_str(s)
    sig = np.asarray(signal, "<f8")
    offsets = []
    for i in range(0, len(sig), block):
        chunk = sig[i:i + block]
        p = len(body)
        offsets.append(p)
        body += struct.pack("<QI", p + 8, len(chunk)) + chunk.tobytes()
    body += struct.pack("<Q", 0)
    struct.pack_into("<I", body, 0, chan_off)
    Path(path).write_bytes(bytes(body))
    return offsets


def write_lsc(path, spectra, *, rate_hz=50.0, t0_unix=1_700_000_000.0, flags=None,
              view=None, with_directory=True) -> None:
    """spectra: list of (mz array, intensity array), one per scan."""
    out = bytearray(struct.pack("<4sIQ", b"FT04", 16, 0))
    offsets = {"PD01": [], "SM01": [], "SD04": [], "MD01": []}

    def chunk(tag, payload):
        offsets[tag].append(len(out))
        out.extend(struct.pack("<4sI", tag.encode(), 8 + len(payload)) + payload)

    pd = "[TOF-DS Method]\r\nName=Synthetic\r\n"
    if view:
        pd += ("\r\n[BenchTOF File]\r\n"
               f"Modulation period=From:0,To:0,Time:{view['pm']:g};\r\n"
               f"Phase shift={view['phase']:g}\r\nTime shift=0\r\n"
               f"Cut interval={view.get('cut', view['pm']):g}\r\n")
    chunk("PD01", pd.encode())
    chunk("SM01", b"\r\n[Start of measurement]\r\n")
    flags = [FLAG_NORMAL] * len(spectra) if flags is None else flags
    for k, (mz, inten) in enumerate(spectra):
        mz, inten = np.asarray(mz, float), np.asarray(inten, float)
        rec = np.zeros(len(mz), dtype=REC)
        rec["mz1000"] = np.round(mz * 1000).astype(np.uint32)
        rec["inten"] = inten
        ts = int(round((t0_unix + k / rate_hz) * 2 ** 32))
        chunk("SD04", struct.pack("<QII", ts, flags[k], len(mz)) + rec.tobytes() + struct.pack("<I", 0))
    chunk("MD01", b"; TOF MS Header File\r\n")
    if with_directory:
        di_off = len(out)
        body = bytearray()
        for tag in ("PD01", "SM01", "SD04", "TS01", "ST01", "MD01"):
            offs = offsets.get(tag) or [0]
            body += struct.pack("<4sI", tag.encode(), len(offs)) + np.asarray(offs, "<u8").tobytes()
        out += struct.pack("<4sI", b"DI02", 8 + len(body)) + body
        struct.pack_into("<Q", out, 8, di_off)
    Path(path).write_bytes(bytes(out))


@dataclass
class Peak:
    t1_min: float
    t2_s: float
    height: float
    base_mz: int = 57
    sigma1_s: float = 6.0
    sigma2_s: float = 0.06
    in_fid: bool = True
    in_tof: bool = True


def signature(base_mz: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([base_mz - 14, base_mz, base_mz + 15], float),
            np.array([0.30, 1.00, 0.15]))


@dataclass
class Scene:
    peaks: list[Peak] = field(default_factory=list)
    pm_s: float = 4.0
    phase_s: float = -1.8
    fid_hz: float = 50.0
    tof_hz: float = 50.0
    duration_min: float = 10.0
    offset_a: float = 0.0          # TOF scans; TOF apex index = FID-time index + offset
    offset_b: float = 0.0          # TOF scans per minute
    fid_baseline: float = 100.0
    background_mz: float = 40.0    # constant ion in the raw .lsc only (DBC removes it)
    background: float = 50.0
    scale: float = 1000.0
    noise: float = 0.0             # sigma of Gaussian noise on the raw .lsc background ion (0 = none)
    seed: int = 0                  # RNG seed for that noise

    def weights(self, t_s: np.ndarray, attr: str) -> np.ndarray:
        """Per-peak signal weight at physical times t_s; shape (n_peaks, len(t_s))."""
        cut = (-self.phase_s) % self.pm_s
        t_inj = cut + np.floor((t_s - cut) / self.pm_s) * self.pm_s
        tau = t_s - t_inj
        rows = []
        for p in self.peaks:
            if not getattr(p, attr):
                rows.append(np.zeros_like(t_s))
                continue
            a = np.exp(-0.5 * ((t_inj - p.t1_min * 60.0) / p.sigma1_s) ** 2)
            g = np.exp(-0.5 * ((tau - p.t2_s) / p.sigma2_s) ** 2)
            rows.append(p.height * a * g)
        return np.array(rows) if rows else np.zeros((0, len(t_s)))

    def n_fid(self) -> int:
        return int(round(self.duration_min * 60 * self.fid_hz))

    def n_tof(self) -> int:
        return int(round(self.duration_min * 60 * self.tof_hz))

    def fid(self) -> np.ndarray:
        t = np.arange(self.n_fid()) / self.fid_hz
        return self.fid_baseline + self.scale * self.weights(t, "in_fid").sum(axis=0)

    def tof_times(self) -> np.ndarray:
        t_k = np.arange(self.n_tof()) / self.tof_hz
        return t_k - (self.offset_a + self.offset_b * t_k / 60.0) / self.tof_hz

    def spectra(self, *, background: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
        w = self.weights(self.tof_times(), "in_tof")
        sigs = [signature(p.base_mz) for p in self.peaks]
        bg = np.full(self.n_tof(), self.background)
        if background and self.noise > 0:
            bg = np.clip(bg + np.random.default_rng(self.seed).normal(0.0, self.noise, self.n_tof()), 0.0, None)
        out = []
        for k in range(self.n_tof()):
            mz_parts, in_parts = [], []
            if background:
                mz_parts.append(np.array([self.background_mz]))
                in_parts.append(np.array([bg[k]]))
            for i, (mz, rel) in enumerate(sigs):
                if w[i, k] > 1e-6:
                    mz_parts.append(mz)
                    in_parts.append(self.scale * w[i, k] * rel)
            if mz_parts:
                mz = np.concatenate(mz_parts)
                u, inv = np.unique(mz, return_inverse=True)
                out.append((u, np.bincount(inv, weights=np.concatenate(in_parts))))
            else:
                out.append((np.empty(0), np.empty(0)))
        return out


def build_run(root, scene: Scene, name: str = "Synth", *, raw=True, dbc=True, view=True,
              insight=True, saturate=frozenset(), with_directory=True) -> Path:
    c = Path(root) / f"{name}.C"
    data = c / "data"
    data.mkdir(parents=True)
    (c / "results").mkdir()
    (c / "manifest.json").write_text(json.dumps(
        {"signal_type": "gcxgc", "source_format": "sepsolve", "sample_id": name}))
    write_hdr(data / f"{name}.HDR", pm_s=scene.pm_s, fid_hz=scene.fid_hz,
              tof_scanset_us=int(round(1e6 / scene.tof_hz)), insight=insight)
    write_rsd(data / f"{name}.rsd", scene.fid(), rate_hz=scene.fid_hz)
    v = {"pm": scene.pm_s, "phase": scene.phase_s} if view else None
    flags = [FLAG_SATURATED if k in saturate else FLAG_NORMAL for k in range(scene.n_tof())]
    if raw:
        write_lsc(data / f"{name}.lsc", scene.spectra(background=True), rate_hz=scene.tof_hz,
                  flags=flags, view=v, with_directory=with_directory)
    if dbc:
        write_lsc(data / f"{name}_0.5_secs.dbc.lsc", scene.spectra(background=False),
                  rate_hz=scene.tof_hz, flags=flags, view=v, with_directory=with_directory)
    return c
