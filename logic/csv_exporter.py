"""
CSV export functionality for integration results.

Uses Feature.as_dict() so it works for any feature type (ChromatographicPeak,
SpectralFeature, etc.) without any type-specific branching.
"""

import csv
import os
from typing import Any, List


def export_results_to_csv(features: List[Any], filepath: str) -> bool:
    """Write integration results to a CSV file.

    Column headers and row values are derived from Feature.as_dict() so the
    output automatically reflects the correct fields for GC, GC-MS, FTIR,
    UV-Vis, or any future feature type.

    Args:
        features: List of Feature subclass instances (ChromatographicPeak,
                  SpectralFeature, etc.)
        filepath:  Absolute path for the output .csv file.

    Returns:
        True on success, False on error.
    """
    if not features:
        return False

    try:
        rows = [f.as_dict() for f in features]
        headers = list(rows[0].keys())

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, mode="w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        return True

    except Exception as exc:
        print(f"CSV export error: {exc}")
        return False
