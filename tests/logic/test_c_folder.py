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
