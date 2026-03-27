"""Parser for Mettler Toledo ReactIR CSV files."""
from __future__ import annotations
from logic.c_folder import CFolder


def parse_reactir_csv(csv_path: str) -> CFolder:
    """Wrap a Mettler Toledo ReactIR CSV into a .C folder.

    ReactIR format: two-column headerless CSV.
      Column 0: wavenumber (cm⁻¹)
      Column 1: absorbance (A.U.)

    The source file is moved into the .C folder's data/ directory.
    Returns the created CFolder.
    """
    return CFolder.create(
        csv_path,
        "ftir",
        instrument="Mettler Toledo ReactIR",
        csv_columns={"x_column": 0, "y_column": 1, "has_header": False},
    )
