import json

from logic.method import ChromaMethod, GCxGCParams


def test_export_min_snr_defaults_to_5_and_round_trips(tmp_path):
    assert GCxGCParams().export_min_snr == 5.0
    m = ChromaMethod(name="t", signal_type="gcxgc", gcxgc=GCxGCParams(export_min_snr=3.5))
    m.to_file(tmp_path / "m.chromethod")
    assert ChromaMethod.from_file(tmp_path / "m.chromethod").gcxgc.export_min_snr == 3.5


def test_old_method_without_the_field_loads_with_5(tmp_path):
    m = ChromaMethod(name="old", signal_type="gcxgc")
    data = json.loads(m.model_dump_json())
    del data["gcxgc"]["export_min_snr"]
    p = tmp_path / "old.chromethod"
    p.write_text(json.dumps(data))
    assert ChromaMethod.from_file(p).gcxgc.export_min_snr == 5.0
