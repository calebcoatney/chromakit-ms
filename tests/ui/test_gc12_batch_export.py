import pytest
pytest.importorskip('pytestqt')

from logic import gc12_workbook
from ui.app import ChromaKitApp

_REAL_WARN = ChromaKitApp._warn_workbook_failed


class _Dialog:
    def __init__(self):
        self.lines = []
        self.directory_status = {}

    def update_directory_status(self, *a, **k):
        pass

    def update_overall_progress(self, *a, **k):
        pass

    def update_file_progress(self, *a, **k):
        pass

    def add_log_message(self, msg):
        self.lines.append(msg)


class _Summary:
    def message(self):
        return "1 pair, summary text"


@pytest.fixture(autouse=True)
def warned(monkeypatch):
    """Never open a real message box; record what would have been shown."""
    calls = []
    monkeypatch.setattr(ChromaKitApp, "_warn_workbook_failed",
                        lambda self, path, error: calls.append((path, str(error))))
    return calls


def _make(qtbot, options, cancelled=False):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.batch_dialog = _Dialog()
    app.batch_options = options
    app._batch_cancelled = cancelled
    app._gc12_records = {("/a/x.C", "FID1A"): (True, "ok")}
    return app


GC12_OPTIONS = {
    'gc12_workbook': '/out/paired.xlsx',
    'gc12_rows': [
        {'instrument': 'GC1', 'detector': 'FID1A', 'method_path': 'm1'},
        {'instrument': 'GC1', 'detector': 'FID2B', 'method_path': 'm1'},
        {'instrument': 'GC2', 'detector': 'FID1A', 'method_path': 'm2'},
    ],
    'directory_instruments': {'/d/one': 'GC1', '/d/two': 'GC2', '/d/three': ''},
}


def test_cancelled_job_writes_no_workbook(qtbot, monkeypatch):
    app = _make(qtbot, dict(GC12_OPTIONS), cancelled=True)
    monkeypatch.setattr(gc12_workbook, "write_paired_workbook",
                        lambda *a, **k: pytest.fail("must not write"))
    app._finish_gc12_workbook()
    assert app.batch_dialog.lines == ["Batch cancelled - GC1&2 workbook not written"]


def test_no_workbook_requested_returns_none(qtbot):
    app = _make(qtbot, {'gc12_workbook': None})
    assert app._finish_gc12_workbook() is None
    assert app.batch_dialog.lines == []


def test_workbook_written_with_records_and_logged(qtbot, monkeypatch):
    app = _make(qtbot, dict(GC12_OPTIONS))
    collected = []
    written = []

    def fake_collect(directory, instrument, detectors, records):
        collected.append((directory, instrument, detectors, records))
        return [f"{instrument}:{directory}"]

    def fake_pair(gc1, gc2):
        return ("pairs", gc1, gc2)

    def fake_write(pairs, path):
        written.append((pairs, path))
        return _Summary()

    monkeypatch.setattr(gc12_workbook, "collect_injections", fake_collect)
    monkeypatch.setattr(gc12_workbook, "pair_injections", fake_pair)
    monkeypatch.setattr(gc12_workbook, "write_paired_workbook", fake_write)

    line = app._finish_gc12_workbook()

    assert [(d, i, det) for d, i, det, _ in collected] == [
        ('/d/one', 'GC1', ['FID1A', 'FID2B']),
        ('/d/two', 'GC2', ['FID1A']),
    ]
    assert all(r is app._gc12_records for *_, r in collected)
    assert written == [(("pairs", ["GC1:/d/one"], ["GC2:/d/two"]), '/out/paired.xlsx')]
    expected = "GC1&2 workbook written: /out/paired.xlsx (1 pair, summary text)"
    assert line == expected
    assert app.batch_dialog.lines == [expected]


def test_export_error_is_logged_not_raised(qtbot, monkeypatch):
    app = _make(qtbot, dict(GC12_OPTIONS))

    def boom(*a, **k):
        raise ValueError("bad manifest")

    monkeypatch.setattr(gc12_workbook, "collect_injections", boom)
    app._finish_gc12_workbook()
    assert app.batch_dialog.lines == ["ERROR writing GC1&2 workbook: bad manifest"]
    assert "bad manifest" in app.status_bar.currentMessage()


def test_write_failure_warns_with_path_and_error(qtbot, monkeypatch, warned):
    app = _make(qtbot, dict(GC12_OPTIONS))
    monkeypatch.setattr(gc12_workbook, "collect_injections", lambda *a, **k: [])
    monkeypatch.setattr(gc12_workbook, "pair_injections", lambda *a: [])

    def locked(*a, **k):
        raise PermissionError("file is open in Excel")

    monkeypatch.setattr(gc12_workbook, "write_paired_workbook", locked)
    app._finish_gc12_workbook()
    assert warned == [("/out/paired.xlsx", "file is open in Excel")]


def test_warning_dialog_names_path_and_says_json_saved(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    app = ChromaKitApp()
    qtbot.addWidget(app)
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: shown.append(text))
    _REAL_WARN(app, "/out/paired.xlsx", PermissionError("locked"))
    assert "/out/paired.xlsx" in shown[0] and "locked" in shown[0]
    assert "per-channel JSON results are saved" in shown[0]


def test_run_completed_signal_populates_records(qtbot, monkeypatch):
    from PySide6.QtCore import QThreadPool

    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.batch_dialog = _Dialog()
    app.batch_directories = ["/d/one"]
    started = []

    class _Pool:
        def start(self, worker):
            started.append(worker)

    monkeypatch.setattr(QThreadPool, "globalInstance", staticmethod(lambda: _Pool()))
    app.process_next_batch_directory()
    assert len(started) == 1
    started[0].signals.run_completed.emit("/d/one/S1.C", "FID1A", False, "result JSON not saved")
    assert app._gc12_records == {("/d/one/S1.C", "FID1A"): (False, "result JSON not saved")}


def test_batch_state_initialised_before_any_job(qtbot):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    assert app._gc12_records == {} and app._batch_cancelled is False
