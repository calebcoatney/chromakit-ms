import pytest
from logic.signal_profiles import PipelineStage, SignalProfile, SignalProfileRegistry
from logic.feature import Feature, SpectralFeature


class _DummyLoader:
    def load(self, path):
        return {}


class _DummyFeature(Feature):
    def __init__(self):
        super().__init__(
            feature_id=0, position=0.0, position_units="",
            area=0.0, width=0.0, start=0.0, end=0.0,
            start_index=0, end_index=0
        )


def _make_profile(name="test", stages=None):
    return SignalProfile(
        name=name,
        display_name="Test",
        feature_class=_DummyFeature,
        loader_class=_DummyLoader,
        x_label="X",
        y_label="Y",
        pipeline_stages=stages or [PipelineStage.SMOOTHING, PipelineStage.BASELINE, PipelineStage.PEAKS],
        ui_mode="spectroscopy",
        default_params={},
    )


def test_pipeline_stage_values():
    assert PipelineStage.SMOOTHING == "smoothing"
    assert PipelineStage.MS_SEARCH == "ms_search"


def test_register_and_get(isolated_registry):
    p = _make_profile("myprofile")
    SignalProfileRegistry.register(p)
    assert SignalProfileRegistry.get("myprofile") is p


def test_get_unknown_raises(isolated_registry):
    with pytest.raises(KeyError):
        SignalProfileRegistry.get("nonexistent")


def test_list_profiles(isolated_registry):
    SignalProfileRegistry.register(_make_profile("a"))
    SignalProfileRegistry.register(_make_profile("b"))
    assert set(SignalProfileRegistry.list_profiles()) == {"a", "b"}


def test_duplicate_registration_raises(isolated_registry):
    SignalProfileRegistry.register(_make_profile("dup"))
    with pytest.raises(ValueError, match="already registered"):
        SignalProfileRegistry.register(_make_profile("dup"))


def test_invalid_stage_raises(isolated_registry):
    # Construct outside the raises block — dataclass accepts any list at construction.
    # The ValueError is raised by register(), not by the dataclass itself.
    bad = _make_profile("bad", stages=["not_a_stage"])  # type: ignore
    with pytest.raises(ValueError, match="invalid pipeline stage"):
        SignalProfileRegistry.register(bad)


def test_builtin_profiles_registered():
    """After normal import, all four built-in profiles are available."""
    profiles = SignalProfileRegistry.list_profiles()
    for name in ("gcms", "gc", "ftir", "uvvis"):
        assert name in profiles, f"Built-in profile '{name}' not registered"
