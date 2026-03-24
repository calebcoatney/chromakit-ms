from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from logic.feature import Feature
    from logic.loaders.base import DataLoader


class PipelineStage(str, Enum):
    SMOOTHING    = "smoothing"
    BASELINE     = "baseline"
    PEAKS        = "peaks"
    MS_SEARCH    = "ms_search"
    QUANTITATION = "quantitation"


@dataclass
class SignalProfile:
    name: str
    display_name: str
    feature_class: Type["Feature"]   # may be None during bootstrapping (see _register_builtin_profiles)
    loader_class: Type["DataLoader"]
    x_label: str
    y_label: str
    pipeline_stages: List[PipelineStage]
    ui_mode: str        # "chromatography" or "spectroscopy"
    default_params: dict


class SignalProfileRegistry:
    _profiles: dict[str, SignalProfile] = {}

    @classmethod
    def register(cls, profile: SignalProfile) -> None:
        if profile.name in cls._profiles:
            raise ValueError(f"Profile '{profile.name}' already registered")
        valid = set(PipelineStage)
        for stage in profile.pipeline_stages:
            if stage not in valid:
                raise ValueError(
                    f"invalid pipeline stage '{stage}' in profile '{profile.name}'. "
                    f"Valid stages: {[s.value for s in PipelineStage]}"
                )
        cls._profiles[profile.name] = profile

    @classmethod
    def get(cls, name: str) -> SignalProfile:
        if name not in cls._profiles:
            raise KeyError(f"No signal profile registered for '{name}'")
        return cls._profiles[name]

    @classmethod
    def list_profiles(cls) -> List[str]:
        return list(cls._profiles.keys())


# ── Built-in profile registrations ────────────────────────────────────────────
# Runs at module import time. Loaders and feature classes are imported lazily
# inside the function to avoid circular imports. feature_class for gc/gcms is
# set to None here and updated in _update_chromatographic_profiles() which is
# called by logic/integration.py after ChromatographicPeak is defined.

def _register_builtin_profiles() -> None:
    from logic.loaders.agilent_loader import AgilentLoader
    from logic.loaders.csv_loader import CSVLoader
    from logic.feature import SpectralFeature

    SignalProfileRegistry.register(SignalProfile(
        name="gcms",
        display_name="GC-MS",
        feature_class=None,  # set by _update_chromatographic_profiles() in integration.py
        loader_class=AgilentLoader,
        x_label="Retention Time (min)",
        y_label="Intensity",
        pipeline_stages=[
            PipelineStage.SMOOTHING, PipelineStage.BASELINE,
            PipelineStage.PEAKS, PipelineStage.MS_SEARCH, PipelineStage.QUANTITATION,
        ],
        ui_mode="chromatography",
        default_params={},
    ))

    SignalProfileRegistry.register(SignalProfile(
        name="gc",
        display_name="GC",
        feature_class=None,  # set by _update_chromatographic_profiles()
        loader_class=AgilentLoader,
        x_label="Retention Time (min)",
        y_label="Intensity",
        pipeline_stages=[
            PipelineStage.SMOOTHING, PipelineStage.BASELINE,
            PipelineStage.PEAKS, PipelineStage.QUANTITATION,
        ],
        ui_mode="chromatography",
        default_params={},
    ))

    SignalProfileRegistry.register(SignalProfile(
        name="ftir",
        display_name="FTIR",
        feature_class=SpectralFeature,
        loader_class=CSVLoader,
        x_label="Wavenumber (cm⁻¹)",
        y_label="Absorbance",
        pipeline_stages=[
            PipelineStage.SMOOTHING, PipelineStage.BASELINE, PipelineStage.PEAKS,
        ],
        ui_mode="spectroscopy",
        default_params={},
    ))

    SignalProfileRegistry.register(SignalProfile(
        name="uvvis",
        display_name="UV-Vis",
        feature_class=SpectralFeature,
        loader_class=CSVLoader,
        x_label="Wavelength (nm)",
        y_label="Absorbance",
        pipeline_stages=[
            PipelineStage.SMOOTHING, PipelineStage.BASELINE, PipelineStage.PEAKS,
        ],
        ui_mode="spectroscopy",
        default_params={},
    ))


def _update_chromatographic_profiles(feature_class) -> None:
    """Called by logic/integration.py after ChromatographicPeak is defined.

    Sets feature_class on the gc and gcms profiles, completing their registration.
    """
    for name in ("gc", "gcms"):
        if name in SignalProfileRegistry._profiles:
            SignalProfileRegistry._profiles[name].feature_class = feature_class


_register_builtin_profiles()
