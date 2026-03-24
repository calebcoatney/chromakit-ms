from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from logic.loaders.base import DataLoader


class CSVLoader(DataLoader):
    """Loads signal data from a CSV file inside a .C folder's data/ directory.

    Column names are provided at construction time, read from manifest.json
    by CFolder before instantiating this loader.
    """

    def __init__(self, x_column: str, y_column: str):
        self.x_column = x_column
        self.y_column = y_column

    def load(self, c_folder_path: str) -> dict:
        data_dir = os.path.join(c_folder_path, "data")
        csv_path = self._find_csv(data_dir)
        if csv_path is None:
            raise FileNotFoundError(f"No CSV file found inside {data_dir}")

        df = pd.read_csv(csv_path)
        if self.x_column not in df.columns:
            raise KeyError(
                f"Column '{self.x_column}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )
        if self.y_column not in df.columns:
            raise KeyError(
                f"Column '{self.y_column}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )

        return {
            "x": df[self.x_column].to_numpy(dtype=float),
            "y": df[self.y_column].to_numpy(dtype=float),
            "metadata": {
                "has_ms_data": False,
                "filename": os.path.basename(csv_path),
            },
        }

    @staticmethod
    def _find_csv(data_dir: str) -> str | None:
        if not os.path.isdir(data_dir):
            return None
        matches = glob.glob(os.path.join(data_dir, "*.csv"))
        return matches[0] if matches else None
