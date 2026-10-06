"""Loads an OpenLab EZChrom ``.dat`` (Agilent 490 Micro GC) from inside a .C folder."""
from __future__ import annotations

import glob
import os

from logic.loaders.base import DataLoader


class EZChromLoader(DataLoader):
    """Reads ``<c_folder>/data/*.dat`` written by :mod:`logic.loaders.ezchrom_parser`.

    Returns x in minutes and y in the trace's unit (mV for the 490) times
    *signal_factor*.  Channels are picked by detector name, like the channels
    of a multi-detector Agilent ``.D``; an unknown name raises instead of
    quietly loading another channel.
    """

    def __init__(self, signal_factor: float = 1.0):
        self.signal_factor = signal_factor if signal_factor else 1.0

    def get_available_detectors(self, c_folder_path: str) -> list:
        from logic.loaders.ezchrom_parser import read_ezchrom_dat
        dat = self._find_dat(c_folder_path)
        if dat is None:
            return []
        return [c.detector for c in read_ezchrom_dat(dat).channels]

    def load(self, c_folder_path: str, detector: str = None) -> dict:
        from logic.loaders.ezchrom_parser import read_ezchrom_dat
        dat = self._find_dat(c_folder_path)
        if dat is None:
            raise FileNotFoundError(f"No .dat file found inside {os.path.join(c_folder_path, 'data')}")
        inj = read_ezchrom_dat(dat)
        try:
            ch = inj.channel(detector)
        except KeyError as e:
            raise ValueError(str(e.args[0]) if e.args else str(e)) from None
        sample_id = os.path.splitext(os.path.basename(os.path.normpath(c_folder_path)))[0]
        return {
            "x": ch.time_min(),
            "y": ch.signal() * self.signal_factor,
            "metadata": {
                "has_ms_data": False,
                "filename": os.path.basename(dat),
                "detector": ch.detector,
                "channel_name": ch.description,
                "column": ch.column,
                "y_units": ch.y_unit,
                "sampling_rate_hz": ch.sampling_rate_hz,
                "sample_id": sample_id,
                "sample_name": inj.sample_name,
                "injection_time_utc": inj.injection_time_utc.isoformat() if inj.injection_time_utc else None,
                "available_detectors": [c.detector for c in inj.channels],
            },
        }

    @staticmethod
    def _find_dat(c_folder_path: str) -> str | None:
        matches = sorted(glob.glob(os.path.join(c_folder_path, "data", "*.dat")))
        return matches[0] if matches else None
