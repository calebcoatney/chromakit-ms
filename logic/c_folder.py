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
        """Create a new .C folder by moving source_path into data/.

        source_path may be a file (CSV) or a directory (.D folder).
        The source is moved, not copied. On any error the partial .C folder
        is removed and the source is restored to its original location.
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
            shutil.move(source_path, dest)

            source_format = (
                "agilent_d" if os.path.isdir(dest) and dest.endswith(".D")
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
            # Restore source if it was already moved into the .C folder
            dest = os.path.join(c_path, "data", os.path.basename(source_path))
            if os.path.exists(dest) and not os.path.exists(source_path):
                shutil.move(dest, source_path)
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

    def get_available_detectors(self) -> list:
        """Return detector channel names if the underlying source supports them."""
        from logic.loaders.agilent_loader import AgilentLoader
        if issubclass(self.profile.loader_class, AgilentLoader):
            loader = AgilentLoader()
            return loader.get_available_detectors(self.path)
        return []

    def load_signal(self, signal_factor: float = 1.0, detector: str = None) -> dict:
        """Load raw signal using the profile's loader.

        CSVLoader needs column names from the manifest; all other loaders
        are instantiated with no arguments. AgilentLoader accepts signal_factor
        and an optional detector channel name.
        """
        profile = self.profile
        manifest = self.get_manifest()

        from logic.loaders.csv_loader import CSVLoader
        from logic.loaders.agilent_loader import AgilentLoader
        if issubclass(profile.loader_class, CSVLoader):
            csv_cols = manifest.get("csv_columns", {})
            has_header = csv_cols.get("has_header", True)
            loader = profile.loader_class(
                x_column=csv_cols.get("x_column", 0 if not has_header else "x"),
                y_column=csv_cols.get("y_column", 1 if not has_header else "y"),
                has_header=has_header,
            )
            return loader.load(self.path)
        elif issubclass(profile.loader_class, AgilentLoader):
            loader = profile.loader_class(signal_factor=signal_factor)
            return loader.load(self.path, detector=detector)
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

    def extract(self, delete_wrapper: bool = False) -> str:
        """Move the data source back out of this .C folder to its parent directory.

        Returns the restored path. If *delete_wrapper* is True the .C folder
        is removed after extraction (results/ will be lost).
        """
        data_dir = os.path.join(self.path, "data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"No data/ directory in {self.path}")

        items = [i for i in os.listdir(data_dir) if not i.startswith(".")]
        if len(items) != 1:
            raise RuntimeError(
                f"Expected exactly one item in data/, found {len(items)}: {items}"
            )

        source_name = items[0]
        src = os.path.join(data_dir, source_name)
        dest = os.path.join(os.path.dirname(self.path), source_name)
        if os.path.exists(dest):
            raise FileExistsError(f"Cannot extract — {dest} already exists")

        shutil.move(src, dest)

        if delete_wrapper:
            shutil.rmtree(self.path, ignore_errors=True)

        return dest


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
