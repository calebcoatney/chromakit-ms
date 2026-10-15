"""One SepSolve GCxGC acquisition and all of its coordinate math.

Coordinates (spec 2026-10-09, "Phase convention"): for a trace sampled at hz with
N = PM*hz samples per modulation, cut = round(-phase*hz) mod N; display row m starts at
raw index cut + m*N, so displayed 2tR = (t_in_modulation + phase) mod PM. FID samples map
to TOF scans through time, both detector rates and a fitted FID->TOF offset in TOF scans;
the two detectors are never assumed to share a rate.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from logic.loaders.sepsolve_format import LscFile, nominal_bin, read_hdr, read_rsd

log = logging.getLogger(__name__)

_KEEP = object()


class MissingModulationPeriod(ValueError):
    """No modulation period from the user, the HDR, or ChromSpace's saved view."""


@dataclass(frozen=True)
class OffsetModel:
    """FID->TOF offset in TOF scans: offset(t) = a + b * t, t in minutes."""
    a: float = 0.0
    b: float = 0.0
    ok: bool = False
    n_windows: int = 0
    mode: str = "auto"

    def at(self, t_min: float) -> float:
        return self.a + self.b * t_min


@dataclass(frozen=True)
class Frame2D:
    data: np.ndarray        # (n_rows, N), a copy
    t1_min: np.ndarray      # (n_rows,) injection time of each row, minutes
    t2_s: np.ndarray        # (N,) second-dimension time, seconds
    hz: float
    cut: int
    N: int
    signal: str


@dataclass(frozen=True)
class SpectrumResult:
    mz: np.ndarray          # nominal m/z (int)
    intensity: np.ndarray
    scan: int               # TOF scan the spectrum is centred on
    source: str             # "dbc" | "lsc"
    apex_found: bool
    apex_distance_scans: int
    saturated: bool


def _rate_from_timestamps(lsc: LscFile) -> float:
    """Scan rate from the mean timestamp step, rounded to 3 decimals.

    Not the median: real timestamps are quantised to 7.63 us and jitter by ~0.3 ms, so the median
    step sits a quantum off (50.008 Hz on a 50 Hz run, which rejects every whole-second period).
    """
    ts = lsc.timestamps
    span = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
    if not span > 0:
        raise ValueError(f"{lsc.path}: {len(ts)} scan timestamps cannot give a TOF rate "
                         "and the HDR has no scanset period")
    return round((len(ts) - 1) / span, 3)


def _pick(names, pred, what, folder):
    hits = sorted(n for n in names if pred(n.lower()))
    if len(hits) > 1:
        log.warning("%s: %d %s candidates (%s); using %s", folder, len(hits), what,
                    ", ".join(hits), hits[0])
    return hits[0] if hits else None


class GCxGCRun:
    """A loaded GCxGC run. Build it with GCxGCRun.from_c_folder()."""

    def __init__(self, *, fid, fid_hz, tic, dbc_tic, tof_hz, file_period, file_phase,
                 lsc=None, dbc_lsc=None, sample_id="", paths=None, c_path=None,
                 user_period_s=None, user_phase_s=None):
        self.fid = np.asarray(fid, dtype=float)
        self.fid_hz = float(fid_hz)
        self.tic = tic
        self.dbc_tic = dbc_tic
        self.tof_hz = float(tof_hz)
        self.lsc = lsc
        self.dbc_lsc = dbc_lsc
        self.sample_id = sample_id
        self.paths = dict(paths or {})
        self.c_path = None if c_path is None else os.path.abspath(str(c_path))   # the run's .C folder
        self._file = {"modulation_period_s": file_period, "phase_shift_s": file_phase}
        self._user = {"modulation_period_s": None, "phase_shift_s": None}
        self._offset = OffsetModel()
        self._offset_mode = "auto"
        self._offset_estimated = False
        self.set_layout(modulation_period_s=user_period_s, phase_shift_s=user_phase_s)

    # ── construction ───────────────────────────────────────────────────

    @classmethod
    def from_c_folder(cls, c_path, *, modulation_period_s=None, phase_shift_s=None) -> "GCxGCRun":
        data_dir = os.path.join(str(c_path), "data")
        # Hidden files out: macOS writes a ._<name> AppleDouble sidecar beside every file on
        # exFAT/FAT drives, and "._" sorts ahead of the real file.
        names = [n for n in os.listdir(data_dir) if not n.startswith(".")]
        hdr = _pick(names, lambda n: n.endswith(".hdr"), ".HDR", data_dir)
        rsd = _pick(names, lambda n: n.endswith(".rsd"), ".rsd", data_dir)
        raw = _pick(names, lambda n: n.endswith(".lsc") and not n.endswith(".dbc.lsc"), ".lsc", data_dir)
        dbc = _pick(names, lambda n: n.endswith(".dbc.lsc"), ".dbc.lsc", data_dir)
        if hdr is None or rsd is None or (raw is None and dbc is None):
            raise FileNotFoundError(f"{data_dir} needs a .HDR, a .rsd and a .lsc or .dbc.lsc")
        full = {k: os.path.join(data_dir, v) if v else None
                for k, v in {"hdr": hdr, "rsd": rsd, "lsc": raw, "dbc_lsc": dbc}.items()}
        hdr_info = read_hdr(full["hdr"])
        fid, rsd_info = read_rsd(full["rsd"])
        lsc = dbc_lsc = None
        try:
            lsc = LscFile(full["lsc"]) if raw else None
            dbc_lsc = LscFile(full["dbc_lsc"]) if dbc else None
            ref = lsc or dbc_lsc
            if hdr_info.fid_rate_hz and hdr_info.fid_rate_hz != rsd_info.rate_hz:
                log.warning("%s: HDR FID rate %g Hz differs from the .rsd's %g Hz; using the .rsd",
                            c_path, hdr_info.fid_rate_hz, rsd_info.rate_hz)
            tof_hz = hdr_info.tof_rate_hz
            if not tof_hz:
                tof_hz = _rate_from_timestamps(ref)
                log.info("%s: no TOF rate (scanset period) in the HDR; %g Hz from the .lsc scan "
                         "timestamps (mean step)", c_path, tof_hz)
            if hdr_info.modulation_period_s is not None:
                file_period = (hdr_info.modulation_period_s, "hdr")
            elif ref.view.modulation_period_s is not None:
                file_period = (ref.view.modulation_period_s, "lsc_view")
            else:
                file_period = (None, None)
            if file_period[0] is None and modulation_period_s is None:
                raise MissingModulationPeriod(
                    f"{c_path}: no modulation period in the HDR or ChromSpace's saved view; enter it")
            file_phase = (ref.view.phase_shift_s, "lsc_view") if ref.view.phase_shift_s is not None else (None, None)
            return cls(
                fid=fid, fid_hz=rsd_info.rate_hz,
                tic=lsc.tic() if lsc else None, dbc_tic=dbc_lsc.tic() if dbc_lsc else None,
                tof_hz=tof_hz, file_period=file_period, file_phase=file_phase,
                lsc=lsc, dbc_lsc=dbc_lsc,
                sample_id=os.path.splitext(os.path.basename(full["rsd"]))[0], paths=full,
                c_path=c_path, user_period_s=modulation_period_s, user_phase_s=phase_shift_s,
            )
        except Exception:
            for f in (lsc, dbc_lsc):
                if f is not None:
                    f.close()
            raise

    def close(self) -> None:
        for f in (self.lsc, self.dbc_lsc):
            if f is not None:
                f.close()

    # ── layout ─────────────────────────────────────────────────────────

    @property
    def modulation_period_s(self) -> float:
        v = self._user["modulation_period_s"]
        return v if v is not None else self._file["modulation_period_s"][0]

    @property
    def phase_shift_s(self) -> float:
        v = self._user["phase_shift_s"]
        if v is not None:
            return v
        f = self._file["phase_shift_s"][0]
        return f if f is not None else 0.0

    @property
    def file_layout(self) -> dict:
        return dict(self._file)

    @property
    def provenance(self) -> dict:
        out = {}
        for key in ("modulation_period_s", "phase_shift_s"):
            value, source = self._file[key]
            out[key] = "user" if self._user[key] is not None else (source if value is not None else "default")
        return out

    def set_layout(self, *, modulation_period_s=_KEEP, phase_shift_s=_KEEP) -> None:
        """Override period/phase. None reverts a value to the file's."""
        pm = self._user["modulation_period_s"] if modulation_period_s is _KEEP else modulation_period_s
        phase = self._user["phase_shift_s"] if phase_shift_s is _KEEP else phase_shift_s
        effective = pm if pm is not None else self._file["modulation_period_s"][0]
        if effective is None:
            raise MissingModulationPeriod("no modulation period in the files; enter one")
        self._check_period(float(effective))
        self._user["modulation_period_s"] = None if pm is None else float(pm)
        self._user["phase_shift_s"] = None if phase is None else float(phase)

    def _check_period(self, pm: float) -> None:
        if not pm > 0:
            raise ValueError(f"modulation period must be positive, got {pm:g} s")
        for name, hz in (("FID", self.fid_hz), ("TOF", self.tof_hz)):
            n = pm * hz
            if abs(n - round(n)) > 1e-6:
                raise ValueError(f"modulation period {pm:g} s is not a whole number of samples "
                                 f"at the {name} rate of {hz:g} Hz ({n:.3f})")

    # ── signals and frames ─────────────────────────────────────────────

    @property
    def available_signals(self) -> tuple:
        return tuple(name for name, arr in (("fid", self.fid), ("tic", self.tic), ("dbc_tic", self.dbc_tic))
                     if arr is not None)

    def signal(self, name: str):
        if name == "fid":
            return self.fid, self.fid_hz
        if name not in ("tic", "dbc_tic"):
            raise ValueError(f"unknown signal {name!r}")
        arr = self.tic if name == "tic" else self.dbc_tic
        if arr is None:
            raise ValueError(f"{name} is not available for {self.sample_id}")
        return arr, self.tof_hz

    def _n(self, hz: float) -> int:
        return int(round(self.modulation_period_s * hz))

    def _cut(self, hz: float) -> int:
        return int(round(-self.phase_shift_s * hz)) % self._n(hz)

    def frame(self, signal: str) -> Frame2D:
        x, hz = self.signal(signal)
        n_per, cut = self._n(hz), self._cut(hz)
        rows = max(0, (len(x) - cut) // n_per)
        data = np.array(x[cut:cut + rows * n_per], dtype=float).reshape(rows, n_per)
        t1 = (cut + np.arange(rows) * n_per) / hz / 60.0
        t2 = np.arange(n_per) / hz
        return Frame2D(data, t1, t2, hz, cut, n_per, signal)

    def to_raw(self, signal: str, m: int, j: int) -> int:
        _, hz = self.signal(signal)
        return self._cut(hz) + int(m) * self._n(hz) + int(j)

    def from_raw(self, signal: str, i: int) -> tuple:
        _, hz = self.signal(signal)
        m, j = divmod(int(i) - self._cut(hz), self._n(hz))
        return m, j

    def raw_to_time(self, signal: str, i: int) -> float:
        _, hz = self.signal(signal)
        return float(i) / hz

    # ── FID -> TOF offset ──────────────────────────────────────────────

    def set_offset(self, mode: str, value: float = 0.0) -> None:
        if mode == "fixed":
            self._offset = OffsetModel(a=float(value), b=0.0, ok=True, n_windows=0, mode="fixed")
            self._offset_mode, self._offset_estimated = "fixed", True
        elif mode == "auto":
            if self._offset_mode != "auto":
                self._offset_mode, self._offset_estimated = "auto", False
        else:
            raise ValueError(f"offset mode must be 'auto' or 'fixed', got {mode!r}")

    def offset(self) -> OffsetModel:
        if self._offset_mode == "auto" and not self._offset_estimated:
            self._offset = self.estimate_offset()
            self._offset_estimated = True
        return self._offset

    def estimate_offset(self, window_min=6.0, max_lag=20, min_r=0.8, min_windows=5) -> OffsetModel:
        """Windowed FID/TIC cross-correlation, then a robust linear fit of lag vs time."""
        tic = self.tic if self.tic is not None else self.dbc_tic
        t_tof = np.arange(len(tic)) / self.tof_hz
        fid = np.interp(t_tof, np.arange(len(self.fid)) / self.fid_hz, self.fid)
        width = int(round(window_min * 60 * self.tof_hz))
        centers, lags = [], []
        for s in range(max_lag, len(tic) - width - max_lag, width):
            a = fid[s:s + width] - fid[s:s + width].mean()
            sa = a.std()
            if sa == 0:
                continue
            rs = np.zeros(2 * max_lag + 1)
            for li, lag in enumerate(range(-max_lag, max_lag + 1)):
                b = tic[s + lag:s + lag + width]
                b = b - b.mean()
                sb = b.std()
                rs[li] = 0.0 if sb == 0 else float(np.mean(a * b) / (sa * sb))
            k = int(np.argmax(rs))
            if rs[k] < min_r:
                continue
            frac = 0.0
            if 0 < k < len(rs) - 1:
                den = rs[k - 1] - 2 * rs[k] + rs[k + 1]
                if den != 0:
                    frac = 0.5 * (rs[k - 1] - rs[k + 1]) / den
            lags.append(k - max_lag + frac)
            centers.append((s + width / 2) / self.tof_hz / 60.0)
        if len(lags) < min_windows:
            log.warning("%s: FID/TOF offset fit used %d windows (< %d); assuming no offset",
                        self.sample_id, len(lags), min_windows)
            return OffsetModel(ok=False, n_windows=len(lags))
        from scipy.stats import theilslopes
        slope, intercept, _, _ = theilslopes(lags, centers)
        return OffsetModel(a=float(intercept), b=float(slope), ok=True, n_windows=len(lags))

    def fid_to_tof(self, i_fid) -> float:
        t = float(i_fid) / self.fid_hz
        return t * self.tof_hz + self.offset().at(t / 60.0)

    # ── spectra ────────────────────────────────────────────────────────

    def spectrum_at(self, apex_raw_index: int, detection_signal: str = "fid", *,
                    source: str = "dbc", window: int = 10) -> SpectrumResult:
        lsc, used = (self.dbc_lsc, "dbc") if source == "dbc" else (self.lsc, "lsc")
        if lsc is None:
            lsc, used = (self.lsc, "lsc") if self.lsc is not None else (self.dbc_lsc, "dbc")
        tic = self.dbc_tic if used == "dbc" else self.tic
        n = lsc.n_scans
        direct = detection_signal != "fid"     # detected on a TOF TIC: the apex IS the scan
        if direct:
            center, window = int(apex_raw_index), 0
        else:
            center = int(round(self.fid_to_tof(apex_raw_index)))
        center = min(max(center, 0), n - 1)
        lo, hi = max(0, center - window), min(n, center + window + 1)
        k = lo + int(np.argmax(tic[lo:hi]))
        # An FID peak needs a prominent interior TOF maximum; a window of 0 searched nothing.
        apex_found = direct or (lo < k < hi - 1 and self._is_prominent(tic, k, center, window))
        if not apex_found:
            k = center
        mz, inten = self._mean_spectrum(lsc, k)
        if used == "dbc" and len(mz) == 0 and self.lsc is not None:
            lsc, used = self.lsc, "lsc"
            mz, inten = self._mean_spectrum(lsc, k)
        return SpectrumResult(mz, inten, k, used, apex_found, k - center, lsc.is_saturated(k))

    @staticmethod
    def _is_prominent(tic: np.ndarray, k: int, center: int, window: int) -> bool:
        """Is tic[k] a real peak: more than 3 robust sigma above its neighbourhood's median?

        The neighbourhood is [center - 4*window, center + 4*window] clipped to the trace; sigma is
        1.4826 * MAD. A perfectly flat neighbourhood (sigma 0) needs only a strict rise above it.
        """
        ctx = tic[max(0, center - 4 * window):min(len(tic), center + 4 * window + 1)]
        med = float(np.median(ctx))
        sigma = 1.4826 * float(np.median(np.abs(ctx - med)))
        return bool(tic[k] - med > 3.0 * sigma)

    @staticmethod
    def _mean_spectrum(lsc: LscFile, k: int):
        scans = [s for s in (k - 1, k, k + 1) if 0 <= s < lsc.n_scans]
        use = [s for s in scans if not lsc.is_saturated(s)] or [k]
        parts = [lsc.raw_spectrum(s) for s in use]
        mz, inten = nominal_bin(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        return mz, inten / len(use)
