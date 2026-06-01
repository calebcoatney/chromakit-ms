from __future__ import annotations
from logic.feature import Feature
from logic.signal_profiles import _update_gcxgc_profile


class GCxGC2DPeak(Feature):
    """A compound peak in a GCxGC-TOFMS/FID chromatogram.

    rt1: 1st dimension retention time (minutes) — volatility axis
    rt2: 2nd dimension retention time (seconds) — polarity axis, apex within modulation
    volume: sum of per-modulation FID slice trapezoid areas, used for quantitation.
            Mapped to Feature.area so QuantitationCalculator works without modification.
    """

    def __init__(
        self,
        peak_number: int,
        rt1: float,
        rt2: float,
        volume: float,
        n_sub_peaks: int,
        mod_start: int,
        mod_end: int,
        apex_mod: int,
        start_time: float,
        end_time: float,
    ):
        super().__init__(
            feature_id=peak_number,
            position=rt1,
            position_units="min",
            area=volume,
            width=end_time - start_time,
            start=start_time,
            end=end_time,
            start_index=mod_start,
            end_index=mod_end,
        )
        self.peak_number = peak_number
        self.rt1 = rt1
        self.rt2 = rt2
        self.volume = volume
        self.n_sub_peaks = n_sub_peaks
        self.mod_start = mod_start
        self.mod_end = mod_end
        self.apex_mod = apex_mod  # modulation index with highest FID intensity

        # MS identification
        self.compound_name = None
        self.match_score = None
        self.casno = None

        # Quantitation (same fields as ChromatographicPeak)
        self.mol_C = None
        self.mol_C_percent = None
        self.num_carbons = None
        self.mol = None
        self.mass_mg = None
        self.mol_percent = None
        self.wt_percent = None

    def as_dict(self) -> dict:
        return {
            "peak_number": self.peak_number,
            "rt1": self.rt1,
            "rt2": self.rt2,
            "volume": self.volume,
            "area": self.area,   # alias for volume; included for compat
            "n_sub_peaks": self.n_sub_peaks,
            "mod_start": self.mod_start,
            "mod_end": self.mod_end,
            "start": self.start,
            "end": self.end,
            "compound_name": self.compound_name,
            "match_score": self.match_score,
            "casno": self.casno,
            "mol_C": self.mol_C,
            "mol_C_percent": self.mol_C_percent,
            "num_carbons": self.num_carbons,
            "mol": self.mol,
            "mass_mg": self.mass_mg,
            "mol_percent": self.mol_percent,
            "wt_percent": self.wt_percent,
        }


# Register GCxGC2DPeak with the signal profile (mirrors integration.py pattern)
_update_gcxgc_profile(GCxGC2DPeak)
