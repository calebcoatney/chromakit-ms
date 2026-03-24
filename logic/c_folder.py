from __future__ import annotations
import os
import json
import shutil
import datetime
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from logic.feature import Feature
    from logic.signal_profiles import SignalProfile


class CFolder:
    """Manages a .C sample folder on disk.

    Layout:
      manifest.json   — signal_type, source_format, sample metadata
      data/           — raw source (CSV file or .D folder)
      results/        — features.json and features.csv
    """

    def __init__(self, path: str):
        self.path = path
        self._manifest: dict | None = None

    @staticmethod
    def create(source_path: str, signal_type: str, **metadata) -> "CFolder":
        """Create a new .C folder by copying source_path into data/.

        source_path may be a file (CSV) or a directory (.D folder).
        The original is not modified. On any error the partial .C folder is removed.
        """
        base = os.path.splitext(os.path.basename(source_path))[0]
        parent = os.path.dirname(os.path.abspath(source_path))
        c_path = os.path.join(parent, base + ".C")

        if os.path.exists(c_path):
            raise FileExistsError(f".C folder already exists: {c_path}")

        try:
            os.makedirs(os.path.join(c_path, "data"))
            os.makedirs(os.path.join(c_path, "results"))

            dest = os.path.join(c_path, "data", os.path.basename(source_path))
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest)
            else:
                shutil.copy2(source_path, dest)

            source_format = (
                "agilent_d" if os.path.isdir(source_path) and source_path.endswith(".D")
                else os.path.splitext(source_path)[1].lstrip(".") or "unknown"
            )

            manifest = {
                "signal_type": signal_type,
                "source_format": source_format,
                "created": datetime.datetime.now().isoformat(),
                **metadata,
            }
            with open(os.path.join(c_path, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

        except Exception:
            if os.path.exists(c_path):
                shutil.rmtree(c_path, ignore_errors=True)
            raise

        folder = CFolder(c_path)
        folder._manifest = manifest
        return folder

    @staticmethod
    def open(path: str) -> "CFolder":
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"manifest.json not found in {path} — is this a valid .C folder?"
            )
        return CFolder(path)

    def get_manifest(self) -> dict:
        if self._manifest is None:
            with open(os.path.join(self.path, "manifest.json")) as f:
                self._manifest = json.load(f)
        return self._manifest

    @property
    def profile(self) -> "SignalProfile":
        from logic.signal_profiles import SignalProfileRegistry
        return SignalProfileRegistry.get(self.get_manifest()["signal_type"])

    def load_signal(self) -> dict:
        """Load raw signal using the profile's loader.

        CSVLoader needs column names from the manifest; all other loaders
        are instantiated with no arguments.
        """
        profile = self.profile
        manifest = self.get_manifest()

        from logic.loaders.csv_loader import CSVLoader
        if issubclass(profile.loader_class, CSVLoader):
            csv_cols = manifest.get("csv_columns", {})
            loader = profile.loader_class(
                x_column=csv_cols.get("x_column", "x"),
                y_column=csv_cols.get("y_column", "y"),
            )
        else:
            loader = profile.loader_class()

        return loader.load(self.path)

    def save_results(self, features: List["Feature"], processing_metadata: dict) -> None:
        """Write features to results/features.json and results/features.csv."""
        import pandas as pd

        results_dir = os.path.join(self.path, "results")
        os.makedirs(results_dir, exist_ok=True)

        feature_dicts = [f.as_dict() for f in features]
        payload = {
            "manifest": self.get_manifest(),
            "processing_metadata": processing_metadata,
            "features": feature_dicts,
        }
        with open(os.path.join(results_dir, "features.json"), "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)

        if feature_dicts:
            pd.DataFrame(feature_dicts).to_csv(
                os.path.join(results_dir, "features.csv"), index=False
            )


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
