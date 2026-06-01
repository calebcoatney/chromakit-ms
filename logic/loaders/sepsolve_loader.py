from __future__ import annotations
import os
import struct
import numpy as np
from logic.loaders.base import DataLoader


# ── Private parsers ────────────────────────────────────────────────────────────

def _parse_hdr(hdr_path: str) -> tuple[float, float]:
    """Parse acquisition parameters from SepSolve .HDR file.

    Scans for 'Data Rate' (Hz) and 'cal modulation pulse' (PM in seconds).

    Returns:
        (hz, pm): acquisition rate and modulation period
    Raises:
        ValueError: if either field is not found
    """
    hz = pm = None
    with open(hdr_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if '=' not in line or line.startswith(';'):
                continue
            key, _, val = line.partition('=')
            key = key.strip()
            val = val.strip()
            if key == 'Data Rate':
                hz = float(val)
            elif key == 'cal modulation pulse':
                pm = float(val)
    if hz is None or pm is None:
        raise ValueError(
            f"Could not parse acquisition parameters from {hdr_path}. "
            f"Expected 'Data Rate' and 'cal modulation pulse'. Got: hz={hz}, pm={pm}"
        )
    return hz, pm


def _parse_rsd(rsd_path: str) -> np.ndarray:
    """Extract FID signal from SepSolve .rsd file.

    Block structure: 12-byte header (u32, u32, count) followed by count float64 values.
    Locates the first block via the b'\\xe1\\x40' signature (end of first float64).

    Returns:
        1D float64 array of FID signal values
    """
    fid: list[float] = []
    with open(rsd_path, 'rb') as f:
        content = f.read()

    first_e140 = content.find(b'\xe1\x40')
    if first_e140 == -1:
        raise ValueError(f"No FID data signature (\\xe1\\x40) found in {rsd_path}")

    pos = first_e140 - 6 - 12  # back up to start of 12-byte header
    while pos + 12 <= len(content):
        u1, u2, count = struct.unpack('<III', content[pos:pos + 12])
        if count > 2000:
            pos += 1
            continue
        pos += 12
        data_size = count * 8
        if pos + data_size > len(content):
            break
        fid.extend(struct.unpack('<' + 'd' * count, content[pos:pos + data_size]))
        pos += data_size

    return np.array(fid)


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


def _parse_lsc_tic(lsc_path: str) -> np.ndarray:
    """Extract TIC from a .lsc or .dbc.lsc file.

    Reads SD04 blocks sequentially. Each block's TIC value is the sum of
    intensity fields across all peaks in that scan.

    SD04 block layout (after the 8-byte tag+size header):
        bytes 12-16: peak_count (u32)
        then peak_count * 16-byte records:
            bytes +8 to +16: intensity (float64)
    """
    offset = _find_sd04_offset(lsc_path)
    tic: list[float] = []
    with open(lsc_path, 'rb') as f:
        f.seek(offset)
        while True:
            tag = f.read(4)
            if tag != b'SD04':
                break
            size_data = f.read(4)
            if not size_data:
                break
            size = struct.unpack('<I', size_data)[0]
            data = f.read(size - 8)
            if len(data) < size - 8:
                break
            peak_count = struct.unpack('<I', data[12:16])[0]
            scan_tic = sum(
                struct.unpack('<d', data[16 + i * 16 + 8: 16 + i * 16 + 16])[0]
                for i in range(peak_count)
            )
            tic.append(scan_tic)
    return np.array(tic, dtype=np.float64)


def extract_spectrum(lsc_path: str, scan_index: int) -> list[tuple[int, float]] | None:
    """Extract a single unit-mass-binned spectrum from a .lsc or .dbc.lsc file.

    Args:
        lsc_path: path to .lsc or .dbc.lsc file
        scan_index: 0-based index into the TIC sequence (same index used by _parse_lsc_tic)

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
    """Loads SepSolve GCxGC data from a .C folder containing .rsd, .lsc, and .HDR files."""

    def load(self, c_folder_path: str) -> dict:
        data_dir = os.path.join(c_folder_path, "data")
        hdr_path = self._find_file(data_dir, ".HDR")
        rsd_path = self._find_file(data_dir, ".rsd")
        lsc_path = self._find_lsc_file(data_dir)

        hz, pm = _parse_hdr(hdr_path)
        scans_per_mod = int(pm * hz)

        fid_1d = _parse_rsd(rsd_path)
        fid_2d = _reshape_to_2d(fid_1d, scans_per_mod)

        tic_1d = _parse_lsc_tic(lsc_path)
        tic_2d = _reshape_to_2d(tic_1d, scans_per_mod)

        n_mods = fid_2d.shape[0]
        t1_axis = np.arange(n_mods) * pm / 60.0

        # Probe for an optional .dbc.lsc (dynamic background corrected) file.
        # Both .lsc and .dbc.lsc share the same scan rate; .dbc.lsc simply has
        # fewer peaks per scan (background noise removed by SepSolve's DBC algorithm).
        dbc_lsc_path = None
        dbc_tic_1d = None
        try:
            dbc_lsc_path = self._find_file(data_dir, ".dbc.lsc")
            dbc_tic_1d = _parse_lsc_tic(dbc_lsc_path)
        except FileNotFoundError:
            pass

        return {
            "x": t1_axis,
            "y": fid_1d[:n_mods * scans_per_mod],
            "metadata": {
                "is_gcxgc": True,
                "has_ms_data": True,
                "fid_1d": fid_1d,
                "tic_1d": tic_1d,
                "fid_2d": fid_2d,
                "tic_2d": tic_2d,
                "pm": pm,
                "hz": hz,
                "lsc_path": lsc_path,
                "dbc_lsc_path": dbc_lsc_path,   # None if not present
                "dbc_tic_1d": dbc_tic_1d,        # None if no .dbc.lsc
                "filename": os.path.splitext(os.path.basename(hdr_path))[0],
            },
        }

    @staticmethod
    def _find_lsc_file(directory: str) -> str:
        """Return the plain .lsc file, excluding .dbc.lsc background-corrected files."""
        for fname in os.listdir(directory):
            fl = fname.lower()
            if fl.endswith('.lsc') and not fl.endswith('.dbc.lsc'):
                return os.path.join(directory, fname)
        raise FileNotFoundError(f"No plain .lsc file (excluding .dbc.lsc) found in {directory}")

    @staticmethod
    def _find_file(directory: str, suffix: str) -> str:
        """Return path of the first file in directory ending with suffix (case-insensitive)."""
        suffix_lower = suffix.lower()
        for fname in os.listdir(directory):
            if fname.lower().endswith(suffix_lower):
                return os.path.join(directory, fname)
        raise FileNotFoundError(f"No *{suffix} file found in {directory}")
