"""A batch's manual-overrides opt-in must not carry into a later Automation run."""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pytestqt")

from ui.app import ChromaKitApp


class _IdlePool:
    """Stands in for the automation QThreadPool: never runs the worker."""

    def start(self, worker):
        pass


def test_automation_after_batch_does_not_apply_overrides(qtbot, tmp_path, monkeypatch):
    window = ChromaKitApp()
    qtbot.addWidget(window)

    odir = tmp_path / "overrides"
    odir.mkdir()
    (odir / "manual_assignments.json").write_text(
        json.dumps({"5.0": {"compound_name": "Testane", "retention_time": 5.0}}))
    monkeypatch.setattr("util.app_paths.writable_dir", lambda name: str(odir))
    monkeypatch.setattr("ui.app.QThreadPool", _IdlePool)

    # what start_batch_processing leaves behind after a batch the user opted into
    window.batch_options = {"apply_manual_overrides": True, "ms_search": False}
    sample = tmp_path / "run" / "sample.D"
    sample.mkdir(parents=True)
    window.data_handler.current_directory_path = str(sample)

    window.start_automation()
    qtbot.addWidget(window.automation_dialog)

    peak = SimpleNamespace(position=5.0, compound_id="Unknown", manual_assignment=False)
    window.integrated_peaks = [peak]
    window.automation_worker._apply_manual_overrides()

    assert peak.compound_id == "Unknown"
    assert not peak.manual_assignment
    assert window.batch_options["ms_search"] is False      # other batch options untouched
