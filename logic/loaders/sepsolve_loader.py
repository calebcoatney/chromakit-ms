from __future__ import annotations
from logic.loaders.base import DataLoader


class SepSolveLoader(DataLoader):
    """Loads a SepSolve GCxGC .C folder into a GCxGCRun (see logic/gcxgc_run.py)."""

    def load(self, c_folder_path: str, modulation_period_s: float | None = None) -> dict:
        from logic.gcxgc_run import GCxGCRun
        run = GCxGCRun.from_c_folder(c_folder_path, modulation_period_s=modulation_period_s)
        frame = run.frame("fid")
        return {
            "x": frame.t1_min,
            "y": frame.data.ravel(),
            "metadata": {
                "is_gcxgc": True,
                "has_ms_data": True,
                "gcxgc_run": run,
                "fid_1d": run.fid,
                "tic_1d": run.tic if run.tic is not None else run.dbc_tic,
                "dbc_tic_1d": run.dbc_tic,
                "fid_2d": frame.data,
                "pm": run.modulation_period_s,
                "hz": run.fid_hz,
                "lsc_path": run.lsc.path if run.lsc else None,
                "dbc_lsc_path": run.dbc_lsc.path if run.dbc_lsc else None,
                "filename": run.sample_id,
            },
        }
