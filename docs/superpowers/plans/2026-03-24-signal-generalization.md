# Signal Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ChromaKit-MS to process FTIR and UV-Vis signals alongside GC/GC-MS by introducing a `.C` container format, a `SignalProfile` registry, typed `Feature` classes, and a `DataLoader` abstraction — without breaking any existing GC/GC-MS behavior.

**Architecture:** New signal types are registered as `SignalProfile` objects that bundle their loader, feature class, pipeline stages, and UI labels. Data is accessed through `.C` folders that contain the raw source file plus a `manifest.json` declaring the signal type. The existing `ChromatogramProcessor` is unchanged except for one new optional `profile` param used for stage gating and default parameters.

**Tech Stack:** Python 3.10+, PySide6, numpy, pandas, rainbow (Agilent reader), pytest

---

## File Map

| Status | Path | What it does |
|--------|------|-------------|
| CREATE | `logic/feature.py` | `Feature` base class (plain `__init__`) + `SpectralFeature` subclass |
| CREATE | `logic/signal_profiles.py` | `PipelineStage` enum, `SignalProfile` dataclass, `SignalProfileRegistry`, built-in profile registrations |
| CREATE | `logic/loaders/__init__.py` | Package init |
| CREATE | `logic/loaders/base.py` | `DataLoader` ABC |
| CREATE | `logic/loaders/agilent_loader.py` | `AgilentLoader` wrapping existing `DataHandler` |
| CREATE | `logic/loaders/csv_loader.py` | `CSVLoader` for FTIR/UV-Vis CSV files |
| CREATE | `logic/c_folder.py` | `CFolder` manager — create, open, load, save |
| MODIFY | `logic/integration.py` | Rename `Peak` → `ChromatographicPeak(Feature)`, keep `Peak` alias, convert `as_dict`/`as_row` from `@property` to methods |
| MODIFY | `logic/processor.py` | Accept optional `profile` param, use `profile.feature_class` for instantiation |
| MODIFY | `logic/automation_worker.py` | Scan `.C` folders instead of `.D` folders; emit migration prompt if none found |
| MODIFY | `logic/json_exporter.py` | Add `metadata_from_manifest()` for `.C`-based workflows |
| MODIFY | `ui/frames/tree.py` | Filter `*.C`, route double-click to `CFolder.open()`; emit `d_folder_opened` signal if `.D` double-clicked |
| MODIFY | `ui/app.py` | Add `set_mode(ui_mode)`, call on load; wire `CFolder`-based loading and migration dialog |
| MODIFY | `ui/frames/plot_frame.py` | Accept `set_axis_labels(x_label, y_label)` call from app |
| MODIFY | `ui/frames/rt_table_frame.py` | Accept `set_column_labels(position_label)` call from app |
| CREATE | `ui/dialogs/c_folder_migration_dialog.py` | Migration dialog: detect `.D` folders, confirm, wrap in `.C` |
| CREATE | `tests/__init__.py` | Test package |
| CREATE | `tests/conftest.py` | Pytest fixtures — restores built-in profiles after each test |
| CREATE | `tests/logic/__init__.py` | Logic test package |
| CREATE | `tests/logic/test_feature.py` | Tests for `Feature` and `SpectralFeature` |
| CREATE | `tests/logic/test_signal_profiles.py` | Tests for `SignalProfile` and `SignalProfileRegistry` |
| CREATE | `tests/logic/test_loaders.py` | Tests for `CSVLoader` and `metadata_from_manifest` |
| CREATE | `tests/logic/test_c_folder.py` | Tests for `CFolder` create/open/load/save |

---

## Task 1: `Feature` Base Class

**Files:**
- Create: `logic/feature.py`
- Create: `tests/__init__.py`
- Create: `tests/logic/__init__.py`
- Create: `tests/logic/test_feature.py`

> **Note:** `Feature` uses a plain `__init__`, not `@dataclass`. Mixing a `@dataclass` parent with a plain-`__init__` child (`ChromatographicPeak`) would break `dataclasses.fields()` introspection and make the `_position` field name leak into the public API. Plain classes are consistent with the existing `Peak` pattern.

- [ ] **Step 1.1: Create test directory structure**

```bash
mkdir -p tests/logic
touch tests/__init__.py tests/logic/__init__.py
```

- [ ] **Step 1.2: Write failing tests for `Feature`**

Create `tests/logic/test_feature.py`:

```python
import pytest
from logic.feature import Feature, SpectralFeature


def test_feature_position_property():
    f = Feature(
        feature_id=1, position=1.23, position_units="min",
        area=100.0, width=0.05, start=1.20, end=1.26,
        start_index=10, end_index=20
    )
    assert f.position == 1.23


def test_feature_defaults():
    f = Feature(
        feature_id=1, position=1.0, position_units="min",
        area=50.0, width=0.02, start=0.99, end=1.01,
        start_index=5, end_index=10
    )
    assert f.is_shoulder is False
    assert f.is_negative is False
    assert f.quality_issues == []


def test_spectral_feature_inherits_feature():
    sf = SpectralFeature(
        feature_id=2, position=1600.0, position_units="cm⁻¹",
        area=250.0, width=10.0, start=1595.0, end=1605.0,
        start_index=100, end_index=110
    )
    assert isinstance(sf, Feature)
    assert sf.position == 1600.0
    assert sf.band_assignment == ""
    assert sf.absorbance == 0.0
    assert sf.transmittance == 0.0


def test_spectral_feature_fields():
    sf = SpectralFeature(
        feature_id=3, position=1720.0, position_units="cm⁻¹",
        area=300.0, width=15.0, start=1712.0, end=1728.0,
        start_index=200, end_index=215,
        band_assignment="C=O stretch", absorbance=0.85
    )
    assert sf.band_assignment == "C=O stretch"
    assert sf.absorbance == 0.85


def test_spectral_feature_as_dict():
    sf = SpectralFeature(
        feature_id=4, position=1600.0, position_units="cm⁻¹",
        area=100.0, width=8.0, start=1596.0, end=1604.0,
        start_index=50, end_index=58,
        band_assignment="aromatic C=C"
    )
    d = sf.as_dict()
    assert d["position"] == 1600.0
    assert d["position_units"] == "cm⁻¹"
    assert d["band_assignment"] == "aromatic C=C"
    assert "area" in d


def test_spectral_feature_as_row():
    sf = SpectralFeature(
        feature_id=5, position=1601.5678, position_units="cm⁻¹",
        area=100.0, width=8.0, start=1597.0, end=1606.0,
        start_index=60, end_index=70
    )
    row = sf.as_row()
    assert isinstance(row, list)
    assert row[0] == round(1601.5678, 2)
```

- [ ] **Step 1.3: Run tests to confirm they fail**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_feature.py -v
```

Expected: `ModuleNotFoundError: No module named 'logic.feature'`

- [ ] **Step 1.4: Implement `logic/feature.py`**

```python
from __future__ import annotations
from typing import List


class Feature:
    """Base class for all detected signal features (peaks, bands, etc.).

    Uses a plain __init__ (not @dataclass) so that ChromatographicPeak can
    subclass it cleanly with its own __init__ without dataclass field-introspection
    complications.
    """

    def __init__(
        self,
        feature_id: int,
        position: float,
        position_units: str,
        area: float,
        width: float,
        start: float,
        end: float,
        start_index: int,
        end_index: int,
        is_shoulder: bool = False,
        is_negative: bool = False,
        quality_issues: List[str] = None,
    ):
        self.feature_id = feature_id
        self._position = position        # backing store; access via .position property
        self.position_units = position_units
        self.area = area
        self.width = width
        self.start = start
        self.end = end
        self.start_index = start_index
        self.end_index = end_index
        self.is_shoulder = is_shoulder
        self.is_negative = is_negative
        self.quality_issues = quality_issues if quality_issues is not None else []

    @property
    def position(self) -> float:
        """Generic x-axis position. ChromatographicPeak overrides this to return retention_time."""
        return self._position


class SpectralFeature(Feature):
    """Feature for spectroscopic signals (FTIR, UV-Vis)."""

    def __init__(
        self,
        feature_id: int,
        position: float,
        position_units: str,
        area: float,
        width: float,
        start: float,
        end: float,
        start_index: int,
        end_index: int,
        is_shoulder: bool = False,
        is_negative: bool = False,
        quality_issues: List[str] = None,
        band_assignment: str = "",
        absorbance: float = 0.0,
        transmittance: float = 0.0,
    ):
        super().__init__(
            feature_id=feature_id, position=position, position_units=position_units,
            area=area, width=width, start=start, end=end,
            start_index=start_index, end_index=end_index,
            is_shoulder=is_shoulder, is_negative=is_negative, quality_issues=quality_issues,
        )
        self.band_assignment = band_assignment
        self.absorbance = absorbance
        self.transmittance = transmittance

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "position": self.position,
            "position_units": self.position_units,
            "area": self.area,
            "width": self.width,
            "start": self.start,
            "end": self.end,
            "is_shoulder": self.is_shoulder,
            "is_negative": self.is_negative,
            "band_assignment": self.band_assignment,
            "absorbance": self.absorbance,
            "transmittance": self.transmittance,
            "quality_issues": self.quality_issues,
        }

    def as_row(self) -> list:
        return [
            round(self.position, 2),
            round(self.area, 1),
            round(self.width, 2),
            round(self.start, 2),
            round(self.end, 2),
            self.band_assignment,
            round(self.absorbance, 4),
        ]
```

- [ ] **Step 1.5: Run tests to confirm they pass**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_feature.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 1.6: Commit**

```bash
git add logic/feature.py tests/__init__.py tests/logic/__init__.py tests/logic/test_feature.py
git commit -m "feat: add Feature base class and SpectralFeature"
```

---

## Task 2: `SignalProfile` Registry + `conftest.py`

**Files:**
- Create: `logic/signal_profiles.py`
- Create: `tests/conftest.py`
- Create: `tests/logic/test_signal_profiles.py`

> **Note:** `SignalProfileRegistry._profiles` is a class-level dict populated at module import time by `_register_builtin_profiles()`. Tests that call `_profiles.clear()` must restore built-ins afterwards. The `conftest.py` fixture handles this automatically for all tests.

- [ ] **Step 2.1: Write failing tests**

Create `tests/logic/test_signal_profiles.py`:

```python
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
```

- [ ] **Step 2.2: Create `tests/conftest.py` with the `isolated_registry` fixture**

```python
"""Shared pytest fixtures for ChromaKit test suite."""
import pytest
from logic.signal_profiles import SignalProfileRegistry


@pytest.fixture
def isolated_registry():
    """Clear the profile registry for this test, then restore built-ins afterwards."""
    saved = dict(SignalProfileRegistry._profiles)
    SignalProfileRegistry._profiles.clear()
    yield
    SignalProfileRegistry._profiles.clear()
    SignalProfileRegistry._profiles.update(saved)
```

- [ ] **Step 2.3: Run tests to confirm they fail**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_signal_profiles.py -v
```

Expected: `ModuleNotFoundError: No module named 'logic.signal_profiles'`

- [ ] **Step 2.4: Implement `logic/signal_profiles.py`**

```python
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
```

- [ ] **Step 2.5: Run tests to confirm they pass**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_signal_profiles.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 2.6: Commit**

```bash
git add logic/signal_profiles.py tests/conftest.py tests/logic/test_signal_profiles.py
git commit -m "feat: add SignalProfile registry with PipelineStage enum and conftest fixture"
```

---

## Task 3: `DataLoader` ABC + `AgilentLoader`

**Files:**
- Create: `logic/loaders/__init__.py`
- Create: `logic/loaders/base.py`
- Create: `logic/loaders/agilent_loader.py`

No automated tests for `AgilentLoader` — it requires real `.D` files. Integration-tested via the UI in Task 9.

- [ ] **Step 3.1: Create the loader package and ABC**

Create `logic/loaders/__init__.py` (empty).

Create `logic/loaders/base.py`:

```python
from abc import ABC, abstractmethod


class DataLoader(ABC):
    """Abstract base for all signal source loaders.

    All loaders return the same dict shape:

        {
            'x': np.ndarray,
            'y': np.ndarray,
            'metadata': {
                'has_ms_data': bool,   # guaranteed key
                'filename': str,       # guaranteed key
                # all other keys are loader-specific; access with .get()
            }
        }
    """

    @abstractmethod
    def load(self, c_folder_path: str) -> dict:
        """Load signal data from the given .C folder path."""
```

- [ ] **Step 3.2: Implement `AgilentLoader`**

Create `logic/loaders/agilent_loader.py`:

```python
from __future__ import annotations
import os
from logic.loaders.base import DataLoader
from logic.data_handler import DataHandler


class AgilentLoader(DataLoader):
    """Loads Agilent .D data from inside a .C folder.

    Wraps DataHandler so all rainbow/detector-detection logic is preserved.
    The .D folder must be the first directory inside <c_folder>/data/.
    """

    def __init__(self):
        self._handler = DataHandler()

    def load(self, c_folder_path: str) -> dict:
        data_dir = os.path.join(c_folder_path, "data")
        d_path = self._find_d_folder(data_dir)
        if d_path is None:
            raise FileNotFoundError(f"No .D folder found inside {data_dir}")

        result = self._handler.load_data_directory(d_path)
        chrom = result["chromatogram"]
        tic = result["tic"]
        has_ms = self._handler.has_ms_data

        metadata = {
            "has_ms_data": has_ms,
            "tic_x": tic["x"] if has_ms else None,
            "tic_y": tic["y"] if has_ms else None,
            "detector": self._handler.current_detector,
            "sample_id": result["metadata"].get("filename", ""),
            "filename": result["metadata"].get("filename", ""),
            "d_path": d_path,
        }

        return {"x": chrom["x"], "y": chrom["y"], "metadata": metadata}

    @staticmethod
    def _find_d_folder(data_dir: str) -> str | None:
        if not os.path.isdir(data_dir):
            return None
        for item in os.listdir(data_dir):
            if item.endswith(".D") and os.path.isdir(os.path.join(data_dir, item)):
                return os.path.join(data_dir, item)
        return None
```

- [ ] **Step 3.3: Smoke test imports cleanly**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "from logic.loaders.agilent_loader import AgilentLoader; print('OK')"
```

Expected: `OK`

- [ ] **Step 3.4: Commit**

```bash
git add logic/loaders/__init__.py logic/loaders/base.py logic/loaders/agilent_loader.py
git commit -m "feat: add DataLoader ABC and AgilentLoader"
```

---

## Task 4: `CSVLoader` + Built-in Profile Registrations

**Files:**
- Create: `logic/loaders/csv_loader.py`
- Modify: `logic/signal_profiles.py` (built-in registrations already added in Task 2)
- Create: `tests/logic/test_loaders.py`

- [ ] **Step 4.1: Write failing tests for `CSVLoader`**

Create `tests/logic/test_loaders.py`:

```python
import pytest
import os
import numpy as np
from logic.loaders.csv_loader import CSVLoader
from logic.json_exporter import metadata_from_manifest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_c_folder_with_csv(tmp_path, x_col, y_col, rows):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "signal.csv"
    lines = [f"{x_col},{y_col}"]
    for x, y in rows:
        lines.append(f"{x},{y}")
    csv_path.write_text("\n".join(lines))
    return str(tmp_path)


# ── CSVLoader ─────────────────────────────────────────────────────────────────

def test_csv_loader_basic(tmp_path):
    rows = [(100.0, 0.5), (200.0, 1.2), (300.0, 0.8)]
    c_path = _make_c_folder_with_csv(tmp_path, "wavenumber", "absorbance", rows)

    loader = CSVLoader(x_column="wavenumber", y_column="absorbance")
    result = loader.load(c_path)

    assert "x" in result and "y" in result and "metadata" in result
    np.testing.assert_array_almost_equal(result["x"], [100.0, 200.0, 300.0])
    np.testing.assert_array_almost_equal(result["y"], [0.5, 1.2, 0.8])


def test_csv_loader_guaranteed_metadata_keys(tmp_path):
    rows = [(1.0, 2.0)]
    c_path = _make_c_folder_with_csv(tmp_path, "wn", "abs", rows)
    result = CSVLoader(x_column="wn", y_column="abs").load(c_path)

    assert result["metadata"]["has_ms_data"] is False
    assert "filename" in result["metadata"]


def test_csv_loader_missing_column_raises(tmp_path):
    rows = [(1.0, 2.0)]
    c_path = _make_c_folder_with_csv(tmp_path, "wn", "abs", rows)
    loader = CSVLoader(x_column="wavelength", y_column="abs")  # wrong x col
    with pytest.raises((KeyError, ValueError)):
        loader.load(c_path)


def test_csv_loader_no_csv_raises(tmp_path):
    (tmp_path / "data").mkdir()
    loader = CSVLoader(x_column="x", y_column="y")
    with pytest.raises(FileNotFoundError):
        loader.load(str(tmp_path))


# ── metadata_from_manifest ────────────────────────────────────────────────────

def test_metadata_from_manifest_basic():
    manifest = {
        "signal_type": "ftir",
        "source_format": "csv",
        "sample_id": "RXN-01",
        "instrument": "ReactIR",
        "created": "2026-03-24T10:00:00",
    }
    result = metadata_from_manifest(manifest)
    assert result["signal_type"] == "ftir"
    assert result["sample_id"] == "RXN-01"
    assert result["instrument"] == "ReactIR"


def test_metadata_from_manifest_missing_keys():
    """Missing optional keys return empty string, not KeyError."""
    result = metadata_from_manifest({"signal_type": "gc"})
    assert result["sample_id"] == ""
    assert result["instrument"] == ""
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_loaders.py -v
```

Expected: `ModuleNotFoundError: No module named 'logic.loaders.csv_loader'`

- [ ] **Step 4.3: Implement `logic/loaders/csv_loader.py`**

```python
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd
from logic.loaders.base import DataLoader


class CSVLoader(DataLoader):
    """Loads signal data from a CSV file inside a .C folder's data/ directory.

    Column names are provided at construction time, read from manifest.json
    by CFolder before instantiating this loader.
    """

    def __init__(self, x_column: str, y_column: str):
        self.x_column = x_column
        self.y_column = y_column

    def load(self, c_folder_path: str) -> dict:
        data_dir = os.path.join(c_folder_path, "data")
        csv_path = self._find_csv(data_dir)
        if csv_path is None:
            raise FileNotFoundError(f"No CSV file found inside {data_dir}")

        df = pd.read_csv(csv_path)
        if self.x_column not in df.columns:
            raise KeyError(
                f"Column '{self.x_column}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )
        if self.y_column not in df.columns:
            raise KeyError(
                f"Column '{self.y_column}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )

        return {
            "x": df[self.x_column].to_numpy(dtype=float),
            "y": df[self.y_column].to_numpy(dtype=float),
            "metadata": {
                "has_ms_data": False,
                "filename": os.path.basename(csv_path),
            },
        }

    @staticmethod
    def _find_csv(data_dir: str) -> str | None:
        if not os.path.isdir(data_dir):
            return None
        matches = glob.glob(os.path.join(data_dir, "*.csv"))
        return matches[0] if matches else None
```

- [ ] **Step 4.4: Add `metadata_from_manifest()` to `logic/json_exporter.py`**

Append to the bottom of `logic/json_exporter.py` (keep existing functions intact):

```python
def metadata_from_manifest(manifest: dict) -> dict:
    """Build export metadata dict from a .C folder manifest.

    For .C-based workflows this replaces scrape_metadata_from_d_directory().
    The existing scrape function remains for any legacy call sites.
    """
    return {
        "sample_id":    manifest.get("sample_id", ""),
        "instrument":   manifest.get("instrument", ""),
        "signal_type":  manifest.get("signal_type", ""),
        "created":      manifest.get("created", ""),
        "source_format": manifest.get("source_format", ""),
    }
```

- [ ] **Step 4.5: Run tests to confirm they pass**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_loaders.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 4.6: Verify built-in profiles register without error**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "from logic.signal_profiles import SignalProfileRegistry; print(SignalProfileRegistry.list_profiles())"
```

Expected: `['gcms', 'gc', 'ftir', 'uvvis']`

- [ ] **Step 4.7: Commit**

```bash
git add logic/loaders/csv_loader.py logic/json_exporter.py tests/logic/test_loaders.py
git commit -m "feat: add CSVLoader, metadata_from_manifest, built-in profile registrations"
```

---

## Task 5: `CFolder` Manager

**Files:**
- Create: `logic/c_folder.py`
- Create: `tests/logic/test_c_folder.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/logic/test_c_folder.py`:

```python
import pytest
import os
import json
import shutil
from logic.c_folder import CFolder
from logic.feature import SpectralFeature


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_csv(tmp_path, name="signal.csv"):
    csv = tmp_path / name
    csv.write_text("wavenumber,absorbance\n" + "\n".join(f"{w},{w*0.001}" for w in range(1000, 1010)))
    return str(csv)


# ── CFolder.create ────────────────────────────────────────────────────────────

def test_create_makes_c_folder_structure(tmp_path):
    csv = _make_csv(tmp_path)
    folder = CFolder.create(csv, "ftir", sample_id="TEST-01")

    assert os.path.isdir(folder.path)
    assert os.path.isfile(os.path.join(folder.path, "manifest.json"))
    assert os.path.isdir(os.path.join(folder.path, "data"))
    assert os.path.isdir(os.path.join(folder.path, "results"))


def test_create_writes_manifest(tmp_path):
    csv = _make_csv(tmp_path)
    folder = CFolder.create(csv, "ftir", sample_id="TEST-02", instrument="ReactIR")

    manifest = folder.get_manifest()
    assert manifest["signal_type"] == "ftir"
    assert manifest["sample_id"] == "TEST-02"
    assert manifest["instrument"] == "ReactIR"
    assert "created" in manifest


def test_create_copies_source_not_moves(tmp_path):
    csv = _make_csv(tmp_path)
    CFolder.create(csv, "ftir")
    # Original CSV should still exist at its original path
    assert os.path.isfile(csv)


def test_create_atomic_on_error(tmp_path, monkeypatch):
    """If copytree/copy2 fails, the partially created .C folder is removed."""
    # Use a directory source so copytree is called
    d_dir = tmp_path / "MySample.D"
    d_dir.mkdir()
    (d_dir / "data.ch").write_text("fake")

    import shutil as _shutil
    original_copytree = _shutil.copytree

    def failing_copytree(src, dst, **kwargs):
        raise OSError("simulated disk error")

    monkeypatch.setattr("logic.c_folder.shutil.copytree", failing_copytree)

    with pytest.raises(OSError, match="simulated disk error"):
        CFolder.create(str(d_dir), "gcms")

    # No .C folder should remain
    c_path = str(tmp_path / "MySample.C")
    assert not os.path.exists(c_path)


# ── CFolder.open ──────────────────────────────────────────────────────────────

def test_open_existing_c_folder(tmp_path):
    csv = _make_csv(tmp_path)
    created = CFolder.create(csv, "ftir")
    reopened = CFolder.open(created.path)
    assert reopened.get_manifest()["signal_type"] == "ftir"


def test_open_missing_manifest_raises(tmp_path):
    c_path = str(tmp_path / "Bad.C")
    os.makedirs(c_path)
    with pytest.raises(FileNotFoundError):
        CFolder.open(c_path)


# ── CFolder.profile ───────────────────────────────────────────────────────────

def test_profile_returns_signal_profile(tmp_path):
    from logic.signal_profiles import SignalProfile
    csv = _make_csv(tmp_path)
    folder = CFolder.create(csv, "ftir")
    assert isinstance(folder.profile, SignalProfile)
    assert folder.profile.name == "ftir"


# ── CFolder.save_results ──────────────────────────────────────────────────────

def test_save_results_writes_files(tmp_path):
    csv = _make_csv(tmp_path)
    folder = CFolder.create(csv, "ftir")
    features = [
        SpectralFeature(
            feature_id=1, position=1600.0, position_units="cm⁻¹",
            area=100.0, width=10.0, start=1595.0, end=1605.0,
            start_index=5, end_index=15, band_assignment="C=C"
        )
    ]
    folder.save_results(features, processing_metadata={"method": "asls"})

    results_dir = os.path.join(folder.path, "results")
    assert os.path.isfile(os.path.join(results_dir, "features.json"))
    assert os.path.isfile(os.path.join(results_dir, "features.csv"))


def test_save_results_json_content(tmp_path):
    csv = _make_csv(tmp_path)
    folder = CFolder.create(csv, "ftir")
    features = [
        SpectralFeature(
            feature_id=1, position=1720.0, position_units="cm⁻¹",
            area=200.0, width=12.0, start=1714.0, end=1726.0,
            start_index=20, end_index=32
        )
    ]
    folder.save_results(features, processing_metadata={})

    with open(os.path.join(folder.path, "results", "features.json")) as f:
        data = json.load(f)

    assert len(data["features"]) == 1
    assert data["features"][0]["position"] == 1720.0
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_c_folder.py -v
```

Expected: `ModuleNotFoundError: No module named 'logic.c_folder'`

- [ ] **Step 5.3: Implement `logic/c_folder.py`**

```python
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
```

- [ ] **Step 5.4: Run tests to confirm they pass**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_c_folder.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5.5: Run full test suite**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5.6: Commit**

```bash
git add logic/c_folder.py tests/logic/test_c_folder.py
git commit -m "feat: add CFolder manager"
```

---

## Task 6: Refactor `Peak` → `ChromatographicPeak(Feature)`

This is the highest-risk task. The goals are: (1) zero behavior change for all existing GC/GC-MS code, (2) `ChromatographicPeak` is a proper `Feature` subclass, (3) `as_dict` and `as_row` become regular methods (they are currently `@property` — `CFolder.save_results` calls `f.as_dict()` with parentheses).

**Files:**
- Modify: `logic/integration.py`
- Modify: `logic/signal_profiles.py` (call `_update_chromatographic_profiles`)

- [ ] **Step 6.1: Read `logic/integration.py` in full before editing**

Use the Read tool to read the entire file. Note every call site of `as_dict` and `as_row` in the rest of the codebase before changing anything.

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && grep -rn "\.as_dict\b\|\.as_row\b" --include="*.py" .
```

Any call site using `peak.as_dict` (no parens, property access) must be updated to `peak.as_dict()` after this task.

- [ ] **Step 6.2: Rename `Peak` to `ChromatographicPeak` and subclass `Feature`**

At the top of `logic/integration.py`, add:

```python
from logic.feature import Feature
```

Change the class definition:

```python
class ChromatographicPeak(Feature):
    """Chromatographic peak. Subclass of Feature with RT, MS, and quantitation fields.

    All original field names are preserved (retention_time, start_time, etc.).
    The base Feature.position property is overridden to return retention_time.
    """

    def __init__(self, compound_id, peak_number, retention_time,
                 integrator, width, area, start_time, end_time,
                 start_index=None, end_index=None):
        # Initialize Feature base with generic fields
        super().__init__(
            feature_id=peak_number,
            position=retention_time,
            position_units="min",
            area=area,
            width=width,
            start=start_time,
            end=end_time,
            start_index=start_index or 0,
            end_index=end_index or 0,
        )
        # Chromatography-specific fields — names unchanged from original Peak
        self.compound_id = compound_id
        self.peak_number = peak_number
        self.retention_time = retention_time
        self.integrator = integrator
        self.start_time = start_time
        self.end_time = end_time
        # ... remainder of existing __init__ body unchanged ...
```

Override `position` to stay in sync with `retention_time`:

```python
    @property
    def position(self) -> float:
        """Returns retention_time, satisfying the Feature.position interface."""
        return self.retention_time
```

- [ ] **Step 6.3: Convert `as_dict` and `as_row` from `@property` to regular methods**

Remove the `@property` decorator from both `as_dict` and `as_row`. They become:

```python
def as_row(self):
    """Return peak data as a list for table display."""
    ...

def as_dict(self):
    """Return a dict representation of the peak."""
    ...
```

- [ ] **Step 6.4: Update all call sites that used property syntax**

For every file found in Step 6.1 that uses `peak.as_dict` or `peak.as_row` **without** parentheses, add parentheses. Common locations: `ui/frames/rt_table_frame.py`, `logic/json_exporter.py`, `api/` routes.

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && grep -rn "\.as_dict\b\|\.as_row\b" --include="*.py" . | grep -v "_test\|test_\|\.pyc"
```

For each hit, verify whether it uses property or method syntax and update as needed.

- [ ] **Step 6.5: Add `Peak` alias and call `_update_chromatographic_profiles`**

At the bottom of the `ChromatographicPeak` class definition (after the class body):

```python
Peak = ChromatographicPeak  # backward-compatibility alias — no call sites need to change

# Register ChromatographicPeak as the feature_class for gc/gcms profiles
from logic.signal_profiles import _update_chromatographic_profiles
_update_chromatographic_profiles(ChromatographicPeak)
```

- [ ] **Step 6.6: Smoke test — existing Peak API still works**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "
from logic.integration import Peak
p = Peak('A', 1, 5.23, 'auto', 0.05, 100.0, 5.20, 5.26)
print('retention_time:', p.retention_time)
print('position:', p.position)
print('as_dict keys:', list(p.as_dict().keys())[:5])
print('as_row len:', len(p.as_row()))
"
```

Expected: `retention_time: 5.23`, `position: 5.23`, no errors.

- [ ] **Step 6.7: Verify gcms profile now has feature_class set**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "from logic.signal_profiles import SignalProfileRegistry; p = SignalProfileRegistry.get('gcms'); print(p.feature_class)"
```

Expected: `<class 'logic.integration.ChromatographicPeak'>`

- [ ] **Step 6.8: Run full test suite**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6.9: Commit**

```bash
git add logic/integration.py logic/signal_profiles.py
git commit -m "feat: refactor Peak → ChromatographicPeak(Feature), convert as_dict/as_row to methods"
```

---

## Task 7: `ChromatogramProcessor` Profile Param

**Files:**
- Modify: `logic/processor.py`

- [ ] **Step 7.1: Check existing `process()` and `integrate_peaks()` signatures**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && grep -n "def process\|def integrate_peaks" logic/processor.py
```

- [ ] **Step 7.2: Add `profile` param to `process()`**

Change signature from:
```python
def process(self, x, y, params=None, ms_range=None):
```
to:
```python
def process(self, x, y, params=None, ms_range=None, profile=None):
```

At the top of the method body, after `params` defaults are applied, add:

```python
if profile is not None and profile.default_params:
    merged = dict(profile.default_params)
    if params:
        merged.update(params)
    params = merged
```

- [ ] **Step 7.3: Add `profile` param to `integrate_peaks()` and use `profile.feature_class`**

Change signature to accept `profile=None`. Find every `Peak(...)` instantiation inside the method body and replace with:

```python
feature_cls = profile.feature_class if (profile is not None and profile.feature_class is not None) else Peak
# then: feature_cls(...) instead of Peak(...)
```

The `profile.feature_class is not None` guard prevents a cryptic `TypeError` if someone calls this before Task 6 completes the `_update_chromatographic_profiles()` call.

- [ ] **Step 7.4: Smoke test — existing processor still works**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "
import numpy as np
from logic.processor import ChromatogramProcessor
p = ChromatogramProcessor()
x = np.linspace(0, 10, 1000)
y = np.exp(-((x - 5)**2) / 0.1)
result = p.process(x, y)
print('process OK, keys:', list(result.keys()))
"
```

Expected: `process OK, keys: [...]` with no errors.

- [ ] **Step 7.5: Commit**

```bash
git add logic/processor.py
git commit -m "feat: add optional profile param to ChromatogramProcessor"
```

---

## Task 8: Update `automation_worker.py`

**Files:**
- Modify: `logic/automation_worker.py`

> **Note:** The `metadata_from_manifest()` function was already added to `json_exporter.py` in Task 4.

- [ ] **Step 8.1: Read `AutomationWorker.run()` in full**

Use the Read tool on `logic/automation_worker.py`. Understand exactly how each `.D` file is processed and what app methods are called.

- [ ] **Step 8.2: Update the directory scan from `.D` to `.C`**

In `run()`, change:

```python
# OLD
for item in os.listdir(self.directory_path):
    item_path = os.path.join(self.directory_path, item)
    if os.path.isdir(item_path) and item.endswith('.D'):
        d_files.append(item_path)
```

to:

```python
# NEW
c_files = []
for item in sorted(os.listdir(self.directory_path)):
    item_path = os.path.join(self.directory_path, item)
    if os.path.isdir(item_path) and item.endswith('.C'):
        c_files.append(item_path)
```

Update all variable names `d_files` → `c_files` and log messages accordingly.

- [ ] **Step 8.3: Add fallback when no `.C` folders are found**

After building `c_files`, add:

```python
if not c_files:
    self.signals.log_message.emit(
        "No .C folders found in directory. "
        "Use File > Migrate .D Folders to convert existing Agilent data."
    )
    self.signals.finished.emit()
    return
```

- [ ] **Step 8.4: Update the per-file call to use `CFolder.open()`**

Add at the top of the file:

```python
from logic.c_folder import CFolder
```

For each `file_path` in `c_files`, replace the existing `DataHandler`-based load call with `CFolder.open(file_path)`. The exact app method call depends on what you read in Step 8.1 — adapt accordingly, routing through the profile-aware loading path.

- [ ] **Step 8.5: Smoke test imports**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "from logic.automation_worker import AutomationWorker; print('OK')"
```

Expected: `OK`

- [ ] **Step 8.6: Commit**

```bash
git add logic/automation_worker.py
git commit -m "feat: update AutomationWorker to scan .C folders with migration fallback"
```

---

## Task 9: Update `FileTreeFrame` for `.C` Folders

**Files:**
- Modify: `ui/frames/tree.py`

> **Note:** Changing the name filter from `*.D` to `*.C` hides all `.D` folders from the tree. Users who have not migrated will see an empty tree. The `d_folder_opened` signal handles the case where a user navigates to a raw `.D` directory and double-clicks it directly (`.D` folders will still appear when the tree root is set to a directory that contains them as non-filtered items, or when the user navigates up). If this is too disruptive, a future improvement could show `.D` folders in a grayed state — that is out of scope here.

- [ ] **Step 9.1: Change the name filter**

```python
self.model.setNameFilters(["*.C"])
```

- [ ] **Step 9.2: Add `d_folder_opened` signal and update `on_item_double_clicked`**

Add to the class signals:

```python
file_selected = Signal(str)    # emitted for .C folders
d_folder_opened = Signal(str)  # emitted if user double-clicks a bare .D folder
```

Update `on_item_double_clicked`:

```python
def on_item_double_clicked(self, index):
    path = self.model.filePath(index)
    if os.path.isdir(path):
        if path.endswith('.C'):
            self.file_selected.emit(path)
        elif path.endswith('.D'):
            self.d_folder_opened.emit(path)
```

- [ ] **Step 9.3: Manual smoke test**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python main.py
```

Navigate to a directory containing `.C` folders. Verify only `.C` folders are shown.

- [ ] **Step 9.4: Commit**

```bash
git add ui/frames/tree.py
git commit -m "feat: update FileTreeFrame to show .C folders and emit d_folder_opened"
```

---

## Task 10: App `set_mode()` + Frame Label Updates

**Files:**
- Modify: `ui/frames/plot_frame.py`
- Modify: `ui/frames/rt_table_frame.py`
- Modify: `ui/app.py`

- [ ] **Step 10.1: Read all three files before editing**

Use the Read tool on each file. Understand: how axis labels are currently set in `PlotFrame`, how column headers are set in `RTTableFrame`, and how `file_selected` is currently handled in `ChromaKitApp`.

- [ ] **Step 10.2: Add `set_axis_labels()` to `PlotFrame`**

```python
def set_axis_labels(self, x_label: str, y_label: str) -> None:
    """Update axis labels to match the active signal profile."""
    self.x_label = x_label
    self.y_label = y_label
    if hasattr(self, 'ax') and self.ax is not None:
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.canvas.draw_idle()
```

- [ ] **Step 10.3: Add `set_column_labels()` to `RTTableFrame`**

Read the file to determine the widget type and how headers are set, then implement:

```python
def set_column_labels(self, position_label: str) -> None:
    """Relabel the position column header (Retention Time → position_label)."""
    # Implementation depends on QTableWidget vs QTableView — read file first
    pass
```

- [ ] **Step 10.4: Add `set_mode()` to `ChromaKitApp`**

```python
def set_mode(self, ui_mode: str, profile=None) -> None:
    is_gcms = (profile is not None and profile.name == "gcms")

    if hasattr(self, 'ms_frame'):
        self.ms_frame.setVisible(is_gcms)
    if hasattr(self, 'quantitation_frame'):
        self.quantitation_frame.setVisible(is_gcms)

    if profile is not None:
        if hasattr(self, 'plot_frame'):
            self.plot_frame.set_axis_labels(profile.x_label, profile.y_label)
        if hasattr(self, 'rt_table_frame'):
            self.rt_table_frame.set_column_labels(profile.x_label)
```

- [ ] **Step 10.5: Update the `file_selected` slot to use `CFolder`**

Find the slot wired to `file_tree_frame.file_selected`. Replace `DataHandler.load_data_directory()` with:

```python
def on_file_selected(self, path: str) -> None:
    from logic.c_folder import CFolder
    self.current_folder = CFolder.open(path)
    profile = self.current_folder.profile
    self.set_mode(profile.ui_mode, profile)
    signal_data = self.current_folder.load_signal()
    # pass signal_data['x'] and signal_data['y'] to the rest of the loading pipeline
    # pass signal_data['metadata'] where TIC/MS data used to come from DataHandler
```

Wire the `d_folder_opened` signal:

```python
self.file_tree_frame.d_folder_opened.connect(self.on_d_folder_opened)

def on_d_folder_opened(self, d_path: str) -> None:
    from ui.dialogs.c_folder_migration_dialog import CFolderMigrationDialog
    dialog = CFolderMigrationDialog([d_path], parent=self)
    dialog.exec()
```

- [ ] **Step 10.6: Manual smoke test — load a `.C` folder end-to-end**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python main.py
```

Wrap an existing `.D` folder into a `.C` folder manually (or use the migration dialog), then load it. Verify axis labels, frame visibility, and that the signal plots correctly.

- [ ] **Step 10.7: Commit**

```bash
git add ui/app.py ui/frames/plot_frame.py ui/frames/rt_table_frame.py
git commit -m "feat: add set_mode() and CFolder-based loading to app"
```

---

## Task 11: Migration Dialog

**Files:**
- Create: `ui/dialogs/c_folder_migration_dialog.py`

- [ ] **Step 11.1: Implement `CFolderMigrationDialog`**

```python
"""Migration dialog: wrap Agilent .D folders inside .C containers."""
from __future__ import annotations
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox,
    QProgressBar, QDialogButtonBox, QMessageBox
)
from logic.c_folder import CFolder


class CFolderMigrationDialog(QDialog):
    """Shows detected .D folders and wraps them in .C containers on confirmation.

    Each row shows: source .D path, destination .C path, signal type selector.
    Originals are copied, not moved — they remain at their original paths.
    """

    SIGNAL_TYPES = ["gcms", "gc"]

    def __init__(self, d_paths: list[str], parent=None):
        super().__init__(parent)
        self.d_paths = d_paths
        self.setWindowTitle("Migrate .D Folders to .C Format")
        self.setMinimumWidth(720)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "ChromaKit detected Agilent .D folders without .C wrappers.\n"
            "Select folders to migrate. Originals will not be deleted."
        ))

        self.table = QTableWidget(len(self.d_paths), 3)
        self.table.setHorizontalHeaderLabels(["Source (.D)", "Destination (.C)", "Signal Type"])
        self.table.setColumnWidth(0, 280)
        self.table.setColumnWidth(1, 280)

        self._combos: list[QComboBox] = []
        for row, d_path in enumerate(self.d_paths):
            base = os.path.splitext(os.path.basename(d_path))[0]
            c_path = os.path.join(os.path.dirname(d_path), base + ".C")
            self.table.setItem(row, 0, QTableWidgetItem(d_path))
            self.table.setItem(row, 1, QTableWidgetItem(c_path))
            combo = QComboBox()
            combo.addItems(self.SIGNAL_TYPES)
            self.table.setCellWidget(row, 2, combo)
            self._combos.append(combo)

        layout.addWidget(self.table)
        layout.addWidget(QLabel(
            "⚠ External tools referencing the original .D path will need to be "
            "updated if you later remove the originals."
        ))

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._run_migration)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _run_migration(self):
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.d_paths))
        errors = []

        for row, d_path in enumerate(self.d_paths):
            signal_type = self._combos[row].currentText()
            try:
                CFolder.create(d_path, signal_type)
            except FileExistsError:
                pass  # already migrated
            except Exception as e:
                errors.append(f"{os.path.basename(d_path)}: {e}")
            self.progress.setValue(row + 1)

        if errors:
            QMessageBox.warning(
                self, "Migration Errors",
                "Some folders could not be migrated:\n\n" + "\n".join(errors)
            )
        self.accept()
```

- [ ] **Step 11.2: Smoke test imports**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -c "from ui.dialogs.c_folder_migration_dialog import CFolderMigrationDialog; print('OK')"
```

Expected: `OK`

- [ ] **Step 11.3: Manual smoke test — trigger migration from the app**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python main.py
```

Double-click a bare `.D` folder in the file tree. Verify dialog appears, shows correct source/destination paths, and creates the `.C` folder on confirmation.

- [ ] **Step 11.4: Run full test suite**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 11.5: Commit**

```bash
git add ui/dialogs/c_folder_migration_dialog.py
git commit -m "feat: add CFolderMigrationDialog for .D → .C migration"
```

---

## Final Integration Check

- [ ] **Load an existing Agilent dataset end-to-end**

  1. Use migration dialog (or python shell) to wrap a real `.D` folder in a `.C` container
  2. Load it from the file tree
  3. Process → integrate → export
  4. Verify `results/features.json` is written inside the `.C` folder
  5. Verify GC-MS behavior is identical to before this feature branch

- [ ] **Load a minimal FTIR CSV end-to-end**

  Create a CSV file:
  ```
  wavenumber,absorbance
  4000,0.01
  3500,0.05
  3000,0.80
  2900,1.20
  1720,0.95
  1600,0.40
  ```

  From a Python shell:
  ```python
  from logic.c_folder import CFolder
  CFolder.create("path/to/spectrum.csv", "ftir",
                 csv_columns={"x_column": "wavenumber", "y_column": "absorbance"},
                 instrument="Mettler Toledo ReactIR")
  ```

  Load `spectrum.C` in the app. Verify: x-axis reads "Wavenumber (cm⁻¹)", MS and quantitation frames are hidden, processing completes without error.

- [ ] **Final commit**

```bash
git add .
git commit -m "feat: signal generalization complete — .C format, Feature hierarchy, SignalProfile registry"
```
