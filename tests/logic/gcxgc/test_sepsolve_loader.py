from logic.c_folder import CFolder
from logic.gcxgc_run import GCxGCRun
from logic.loaders.sepsolve_loader import SepSolveLoader
from util import sepsolve_synth as synth


def test_loader_returns_run_and_correct_layout(tmp_path):
    c = synth.build_run(tmp_path, synth.Scene(peaks=[synth.Peak(1.0, 1.0, 1.0)], duration_min=2.0))
    meta = SepSolveLoader().load(str(c))["metadata"]
    assert meta["is_gcxgc"] and meta["has_ms_data"]
    assert (meta["pm"], meta["hz"]) == (4.0, 50.0)
    assert isinstance(meta["gcxgc_run"], GCxGCRun)
    assert meta["fid_2d"].shape[1] == 200
    assert meta["lsc_path"].endswith("Synth.lsc") and meta["dbc_lsc_path"].endswith(".dbc.lsc")
    assert meta["filename"] == "Synth"


def test_loader_through_cfolder(tmp_path):
    c = synth.build_run(tmp_path, synth.Scene(duration_min=2.0))
    assert CFolder.open(str(c)).load_signal()["metadata"]["is_gcxgc"]


def test_loader_passes_user_period(tmp_path):
    c = synth.build_run(tmp_path, synth.Scene(duration_min=2.0), insight=False, view=False)
    assert SepSolveLoader().load(str(c), modulation_period_s=4.0)["metadata"]["pm"] == 4.0
