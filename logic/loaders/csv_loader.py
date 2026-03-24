from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from logic.loaders.base import DataLoader


class CSVLoader(DataLoader):
    """Loads signal data from a CSV file inside a .C folder's data/ directory.

    Supports both named columns and index-based selection for headerless files.
    Column references are provided at construction time, read from manifest.json
    by CFolder before instantiating this loader.
    """

    def __init__(self, x_column: str | int = 0, y_column: str | int = 1,
                 has_header: bool = True):
        self.x_column = x_column
        self.y_column = y_column
        self.has_header = has_header

    def load(self, c_folder_path: str) -> dict:
        data_dir = os.path.join(c_folder_path, "data")
        csv_path = self._find_csv(data_dir)
        if csv_path is None:
            raise FileNotFoundError(f"No CSV file found inside {data_dir}")

        header = 0 if self.has_header else None
        df = pd.read_csv(csv_path, header=header)

        # Resolve columns — accept names (str) or indices (int)
        x_col = self._resolve_column(df, self.x_column, "x", csv_path)
        y_col = self._resolve_column(df, self.y_column, "y", csv_path)

        return {
            "x": df[x_col].to_numpy(dtype=float),
            "y": df[y_col].to_numpy(dtype=float),
            "metadata": {
                "has_ms_data": False,
                "filename": os.path.basename(csv_path),
            },
        }

    @staticmethod
    def _resolve_column(df: pd.DataFrame, col, axis_label: str, csv_path: str):
        """Accept a column name or integer index; return the actual column key."""
        if isinstance(col, int):
            if col < 0 or col >= len(df.columns):
                raise KeyError(
                    f"{axis_label} column index {col} out of range for {csv_path} "
                    f"({len(df.columns)} columns)"
                )
            return df.columns[col]
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )
        return col

    @staticmethod
    def _find_csv(data_dir: str) -> str | None:
        if not os.path.isdir(data_dir):
            return None
        matches = glob.glob(os.path.join(data_dir, "*.csv"))
        return matches[0] if matches else None
