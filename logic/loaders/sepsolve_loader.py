from __future__ import annotations
import struct
import numpy as np
from logic.loaders.base import DataLoader


# ── Private parsers ────────────────────────────────────────────────────────────

def _reshape_to_2d(signal_1d: np.ndarray, scans_per_mod: int) -> np.ndarray:
    """Reshape a 1D signal into a 2D array of shape (n_mods, scans_per_mod).

    Trailing points that don't fill a complete modulation are discarded.
    """
    n_mods = len(signal_1d) // scans_per_mod
    return signal_1d[:n_mods * scans_per_mod].reshape(n_mods, scans_per_mod)


def _find_sd04_offset(path: str) -> int:
    """Scan file to find byte offset of first SD04 block tag.

    More robust than hardcoding an offset since .dbc.lsc header size may differ
    from the raw .lsc header.
    """
    with open(path, 'rb') as f:
        chunk = f.read(131072)  # 128 KB covers any plausible header
    idx = chunk.find(b'SD04')
    if idx == -1:
        raise ValueError(f"No SD04 blocks found in first 128KB of {path}")
    return idx


def extract_spectrum(lsc_path: str, scan_index: int) -> list[tuple[int, float]] | None:
    """Extract a single unit-mass-binned spectrum from a .lsc or .dbc.lsc file.

    Args:
        lsc_path: path to .lsc or .dbc.lsc file
        scan_index: 0-based SD04 scan index (the same index as the TIC sequence)

    Returns:
        List of (m/z, intensity) tuples at unit-mass resolution, or None if scan not found.

    SD04 peak record layout (16 bytes):
        bytes 0-4:  unknown
        bytes 4-8:  m/z * 1000 as u32
        bytes 8-16: intensity as float64
    """
    offset = _find_sd04_offset(lsc_path)
    with open(lsc_path, 'rb') as f:
        f.seek(offset)
        for idx in range(scan_index + 1):
            tag = f.read(4)
            if tag != b'SD04':
                return None
            size = struct.unpack('<I', f.read(4))[0]
            data = f.read(size - 8)
            if idx == scan_index:
                peak_count = struct.unpack('<I', data[12:16])[0]
                mz_raw: list[float] = []
                intensities: list[float] = []
                for i in range(peak_count):
                    rec = data[16 + i * 16: 16 + (i + 1) * 16]
                    mz_raw.append(struct.unpack('<I', rec[4:8])[0] / 1000.0)
                    intensities.append(struct.unpack('<d', rec[8:16])[0])
                if not mz_raw:
                    return []
                mz_arr = np.round(np.array(mz_raw)).astype(int)
                int_arr = np.array(intensities)
                unique_mz = np.unique(mz_arr)
                return [(int(m), float(np.sum(int_arr[mz_arr == m]))) for m in unique_mz]
    return None


class SepSolveLoader(DataLoader):
    """Loads a SepSolve GCxGC .C folder into a GCxGCRun (see logic/gcxgc_run.py)."""

    def load(self, c_folder_path: str, modulation_period_s: float | None = None) -> dict:
        from logic.gcxgc_run import GCxGCRun
        run = GCxGCRun.from_c_folder(c_folder_path, modulation_period_s=modulation_period_s)
        frame = run.frame("fid")
        return {
            "x": frame.t1_min,
            "y": frame.data.ravel(),
            "metadata": {
                "is_gcxgc": True,
                "has_ms_data": True,
                "gcxgc_run": run,
                "fid_1d": run.fid,
                "tic_1d": run.tic if run.tic is not None else run.dbc_tic,
                "dbc_tic_1d": run.dbc_tic,
                "fid_2d": frame.data,
                "pm": run.modulation_period_s,
                "hz": run.fid_hz,
                "lsc_path": run.lsc.path if run.lsc else None,
                "dbc_lsc_path": run.dbc_lsc.path if run.dbc_lsc else None,
                "filename": run.sample_id,
            },
        }
