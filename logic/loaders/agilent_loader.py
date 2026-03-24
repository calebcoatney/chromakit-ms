from __future__ import annotations
import os
from logic.loaders.base import DataLoader
from logic.data_handler import DataHandler


class AgilentLoader(DataLoader):
    """Loads Agilent .D data from inside a .C folder.

    Wraps DataHandler so all rainbow/detector-detection logic is preserved.
    The .D folder must be the first directory inside <c_folder>/data/.
    """

    def __init__(self, signal_factor: float = 1.0):
        self._handler = DataHandler()
        self._handler.signal_factor = signal_factor

    def load(self, c_folder_path: str) -> dict:
        data_dir = os.path.join(c_folder_path, "data")
        d_path = self._find_d_folder(data_dir)
        if d_path is None:
            raise FileNotFoundError(f"No .D folder found inside {data_dir}")

        result = self._handler.load_data_directory(d_path)
        chrom = result["chromatogram"]
        tic = result["tic"]
        has_ms = self._handler.has_ms_data

        metadata = {
            "has_ms_data": has_ms,
            "tic_x": tic["x"] if has_ms else None,
            "tic_y": tic["y"] if has_ms else None,
            "detector": self._handler.current_detector,
            "sample_id": result["metadata"].get("filename", ""),
            "filename": result["metadata"].get("filename", ""),
            "d_path": d_path,
        }

        return {"x": chrom["x"], "y": chrom["y"], "metadata": metadata}

    @staticmethod
    def _find_d_folder(data_dir: str) -> str | None:
        if not os.path.isdir(data_dir):
            return None
        for item in os.listdir(data_dir):
            if item.endswith(".D") and os.path.isdir(os.path.join(data_dir, item)):
                return os.path.join(data_dir, item)
        return None
