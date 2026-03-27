"""Parser for Avantes UV-Vis time-series CSV files."""
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from logic.c_folder import CFolder


def parse_avantes_uvvis(csv_path: str, output_dir: str = None) -> list[str]:
    """Parse an Avantes UV-Vis CSV into one .C folder per spectrum column.

    Avantes format:
      Row 0: Integration Time [msec]  — global value in col 1
      Row 1: Number of Averages       — global value in col 1
      Row 2: Column labels            — skipped
      Row 3: Timestamps (ms)          — skipped
      Row 4: Date/Time strings        — DD/MM/YYYY HH:MM:SS, one per column
      Row 5+: spectral data           — col 0 = wavelength (nm),
                                        cols 1..N = absorbance (A.U.)

    Each spectrum column i produces:
      {output_dir}/{basename}_{i:03d}.C/
        manifest.json  — signal_type, instrument, timestamps, integration params
        data/{basename}_{i:03d}.csv  — two-column headerless CSV

    output_dir defaults to the directory containing csv_path.
    Returns the list of created (or pre-existing) .C folder paths.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_path))

    df = pd.read_csv(csv_path, header=None, dtype=str)

    integration_time_ms = float(df.iloc[0, 1])
    n_averages = int(float(df.iloc[1, 1]))
    datetime_strings = df.iloc[4, 1:].tolist()

    data = df.iloc[5:].reset_index(drop=True).astype(float)
    wavelengths = data.iloc[:, 0].to_numpy()
    n_spectra = len(datetime_strings)
    basename = os.path.splitext(os.path.basename(csv_path))[0]

    created = []
    for i in range(n_spectra):
        intensities = data.iloc[:, i + 1].to_numpy()

        try:
            dt = datetime.strptime(datetime_strings[i].strip(), "%d/%m/%Y %H:%M:%S")
            sample_timestamp = dt.isoformat()
        except (ValueError, AttributeError):
            sample_timestamp = None

        temp_csv = os.path.join(output_dir, f"{basename}_{i:03d}.csv")
        pd.DataFrame({0: wavelengths, 1: intensities}).to_csv(
            temp_csv, index=False, header=False
        )

        kwargs: dict = {
            "instrument": "Avantes",
            "integration_time_ms": integration_time_ms,
            "n_averages": n_averages,
            "csv_columns": {"x_column": 0, "y_column": 1, "has_header": False},
        }
        if sample_timestamp:
            kwargs["sample_timestamp"] = sample_timestamp

        c_path = os.path.join(output_dir, f"{basename}_{i:03d}.C")
        try:
            cf = CFolder.create(temp_csv, "uvvis", **kwargs)
            created.append(cf.path)
        except FileExistsError:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            created.append(c_path)
        except Exception:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            raise

    return created
