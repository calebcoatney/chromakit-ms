import json

from logic.method import ChromaMethod, GCxGCParams


def test_gcxgc_params_roundtrip(tmp_path):
    m = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(
        modulation_period_s=4.0, phase_shift_s=-1.8, detection_signal="dbc_tic",
        apex_window_scans=6, offset_mode="fixed", offset_scans=-5.5))
    m.to_file(tmp_path / "g.chromethod")
    assert ChromaMethod.from_file(tmp_path / "g.chromethod").gcxgc == m.gcxgc


def test_old_method_without_gcxgc_gets_defaults(tmp_path):
    p = tmp_path / "old.chromethod"
    ChromaMethod(name="old", signal_type="gc").to_file(p)
    raw = json.loads(p.read_text())
    raw.pop("gcxgc")
    p.write_text(json.dumps(raw))
    assert ChromaMethod.from_file(p).gcxgc == GCxGCParams()


def test_gui_params_carry_gcxgc():
    m = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(min_modulations=3))
    again = ChromaMethod.from_gui_params(m.to_gui_params(), name="g", signal_type="gcxgc")
    assert again.gcxgc.min_modulations == 3


def test_defaults_turn_processing_on():
    g = GCxGCParams()
    assert g.baseline_enabled and g.peaks_enabled and g.modulation_period_s is None


def test_band_and_solvent_settings_default_and_round_trip(tmp_path):
    assert GCxGCParams().remove_bands is True and GCxGCParams().solvent_end_min is None
    m = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(remove_bands=False, solvent_end_min=7.0))
    m.to_file(tmp_path / "g.chromethod")
    again = ChromaMethod.from_file(tmp_path / "g.chromethod").gcxgc
    assert again.remove_bands is False and again.solvent_end_min == 7.0
