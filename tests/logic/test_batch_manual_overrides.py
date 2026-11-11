"""Manual assignment overrides in batch: opt-in, and never applied on RT alone when a spectrum can't be compared."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import logic.automation_worker as aw
from logic.automation_worker import AutomationWorker


def _worker(tmp_path, monkeypatch, overrides, batch_options, peaks):
    odir = tmp_path / "overrides"
    odir.mkdir()
    (odir / "manual_assignments.json").write_text(json.dumps(overrides))
    monkeypatch.setattr("util.app_paths.writable_dir", lambda name: str(odir))
    app = SimpleNamespace(
        batch_options=batch_options,
        integrated_peaks=peaks,
        data_handler=SimpleNamespace(current_directory_path=str(tmp_path)),
    )
    worker = AutomationWorker(app, str(tmp_path))
    logs = []
    worker.signals.log_message.connect(logs.append)
    return worker, logs


def _peak(rt=5.0):
    return SimpleNamespace(position=rt, compound_id="Unknown", manual_assignment=False)


def _entry(with_spectrum):
    e = {"compound_name": "Testane", "retention_time": 5.0}
    if with_spectrum:
        e["spectrum"] = {"mz": [41, 43], "intensities": [100, 50]}
    return {"5.0": e}


def test_overrides_not_applied_when_option_off(tmp_path, monkeypatch):
    peak = _peak()
    worker, _ = _worker(tmp_path, monkeypatch, _entry(False), {}, [peak])
    worker._apply_manual_overrides()
    assert peak.compound_id == "Unknown"
    assert not peak.manual_assignment


def test_option_on_but_spectrum_comparison_raises_skips_and_logs(tmp_path, monkeypatch):
    peak = _peak()
    worker, logs = _worker(tmp_path, monkeypatch, _entry(True),
                           {"apply_manual_overrides": True}, [peak])

    def boom(*a, **k):
        raise RuntimeError("no MS data")
    monkeypatch.setattr("logic.batch_search.extract_peak_spectrum", boom)
    worker._apply_manual_overrides()
    assert peak.compound_id == "Unknown"
    assert any("Testane" in m and "5.000" in m and "no MS data" in m for m in logs)


def test_option_on_empty_spectrum_skips_and_logs(tmp_path, monkeypatch):
    peak = _peak()
    worker, logs = _worker(tmp_path, monkeypatch, _entry(True),
                           {"apply_manual_overrides": True}, [peak])
    monkeypatch.setattr("logic.batch_search.extract_peak_spectrum", lambda *a, **k: None)
    worker._apply_manual_overrides()
    assert peak.compound_id == "Unknown"
    assert any("Testane" in m and "skipped" in m.lower() for m in logs)


def test_option_on_entry_without_spectrum_applies_by_rt(tmp_path, monkeypatch):
    peak = _peak()
    worker, _ = _worker(tmp_path, monkeypatch, _entry(False),
                        {"apply_manual_overrides": True}, [peak])
    worker._apply_manual_overrides()
    assert peak.compound_id == "Testane"
    assert peak.manual_assignment is True


def test_dialog_checkbox_off_by_default_and_flows_into_options(qtbot, tmp_path):
    from ui.dialogs.batch_job_dialog import BatchJobDialog
    dlg = BatchJobDialog()
    qtbot.addWidget(dlg)
    assert not dlg.manual_overrides_check.isChecked()
    assert dlg.manual_overrides_check.toolTip()
    dlg.directories = [str(tmp_path)]
    dlg.manual_overrides_check.setChecked(True)
    with qtbot.waitSignal(dlg.start_batch) as blocker:
        dlg.on_start_processing()
    assert blocker.args[1]['apply_manual_overrides'] is True


def _stub_similarity(monkeypatch, value):
    """Stand in for ms_toolkit.similarity so the threshold is tested without ms_toolkit."""
    seen = []

    def dot_product_similarity(s1, s2):
        seen.append((s1, s2))
        return value

    monkeypatch.setitem(sys.modules, "ms_toolkit.similarity",
                        ModuleType("ms_toolkit.similarity"))
    sys.modules["ms_toolkit.similarity"].dot_product_similarity = dot_product_similarity
    monkeypatch.setattr("logic.batch_search.extract_peak_spectrum",
                        lambda *a, **k: {"mz": [41, 43], "intensities": [90, 60]})
    return seen


@pytest.mark.parametrize("similarity", [0.7, 0.95])
def test_option_on_similar_spectrum_applies(tmp_path, monkeypatch, similarity):
    peak = _peak()
    worker, _ = _worker(tmp_path, monkeypatch, _entry(True),
                        {"apply_manual_overrides": True}, [peak])
    seen = _stub_similarity(monkeypatch, similarity)
    worker._apply_manual_overrides()
    assert seen == [([(41, 90), (43, 60)], [(41, 100), (43, 50)])]
    assert peak.compound_id == "Testane"
    assert peak.manual_assignment is True


def test_option_on_dissimilar_spectrum_skips_and_logs(tmp_path, monkeypatch):
    peak = _peak()
    worker, logs = _worker(tmp_path, monkeypatch, _entry(True),
                           {"apply_manual_overrides": True}, [peak])
    _stub_similarity(monkeypatch, 0.69)
    worker._apply_manual_overrides()
    assert peak.compound_id == "Unknown"
    assert not peak.manual_assignment
    assert any("Testane" in m and "5.000" in m and "0.69 is below 0.7" in m for m in logs)
