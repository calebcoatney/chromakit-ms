"""Batch processing per detector: run setup, detector choice, and the worker loop."""
import json
import os
from types import SimpleNamespace

import pytest

import logic.automation_worker as aw
from logic.automation_worker import AutomationWorker, build_detector_runs, resolve_detector
from logic.method import ChromaMethod


def _method_file(tmp_path, name, detector=None):
    path = tmp_path / f"{name}.chromethod"
    ChromaMethod(name=name, signal_type="gc", detector=detector).to_file(path)
    return str(path)


# --- build_detector_runs ----------------------------------------------------

def test_no_detector_runs_gives_single_default_run():
    runs = build_detector_runs({})
    assert runs == [{'detector': None, 'method': None, 'method_path': None, 'explicit': False}]


def test_detector_runs_load_their_methods(tmp_path):
    m1 = _method_file(tmp_path, "front")
    m2 = _method_file(tmp_path, "back")
    runs = build_detector_runs({'detector_runs': [
        {'detector': 'FID1A', 'method_path': m1},
        {'detector': 'FID2B', 'method_path': m2},
    ]})
    assert [r['detector'] for r in runs] == ['FID1A', 'FID2B']
    assert [r['method'].name for r in runs] == ['front', 'back']
    assert all(r['explicit'] for r in runs)


def test_blank_detector_falls_back_to_method_detector(tmp_path):
    m = _method_file(tmp_path, "back", detector="FID2B")
    runs = build_detector_runs({'detector_runs': [{'detector': '', 'method_path': m}]})
    assert runs[0]['detector'] == 'FID2B'


def test_detector_without_method_uses_current_settings():
    runs = build_detector_runs({'detector_runs': [{'detector': 'FID2B', 'method_path': None}]})
    assert runs[0]['detector'] == 'FID2B'
    assert runs[0]['method'] is None


def test_missing_detector_name_raises(tmp_path):
    m = _method_file(tmp_path, "nodet")
    with pytest.raises(ValueError):
        build_detector_runs({'detector_runs': [{'detector': '', 'method_path': m}]})


def test_duplicate_detector_raises():
    with pytest.raises(ValueError):
        build_detector_runs({'detector_runs': [
            {'detector': 'FID1A'}, {'detector': 'FID1A'}]})


# --- resolve_detector -------------------------------------------------------

@pytest.mark.parametrize("requested, available, explicit, expected", [
    (None, ['FID1A', 'FID2B'], False, (None, True)),
    ('FID2B', ['FID1A', 'FID2B'], False, ('FID2B', True)),
    ('FID2B', ['FID1A', 'FID2B'], True, ('FID2B', True)),
    ('FID2B', ['FID1A'], False, (None, True)),     # selection missing: auto-detect
    ('FID2B', ['FID1A'], True, (None, False)),     # explicit run missing: skip
    ('FID2B', [], True, ('FID2B', True)),          # channels unknown: trust request
])
def test_resolve_detector(requested, available, explicit, expected):
    assert resolve_detector(requested, available, explicit) == expected


# --- worker loop ------------------------------------------------------------

def _make_c_folders(base, names):
    for name in names:
        c = base / f"{name}.C"
        c.mkdir()
        (c / "manifest.json").write_text(json.dumps({"signal_type": "gc"}))


def _run_worker(tmp_path, monkeypatch, batch_options, selected_detector,
                available=('FID1A', 'FID2B')):
    _make_c_folders(tmp_path, ["S1", "S2"])
    monkeypatch.setattr(aw.CFolder, "get_available_detectors", lambda self: list(available))
    app = SimpleNamespace(batch_options=batch_options,
                          data_handler=SimpleNamespace(current_detector=selected_detector))
    worker = AutomationWorker(app, str(tmp_path))
    calls, completed = [], []
    worker.signals.file_completed.connect(lambda f, ok, msg: completed.append((f, ok)))
    monkeypatch.setattr(
        worker, "_process_file",
        lambda file_path, label, signal_type, stages, detector=None, method=None:
            calls.append((os.path.basename(file_path), label, detector,
                          method.name if method else None)))
    worker.run()
    return worker, calls, completed


def test_default_batch_uses_selected_detector_for_c_folders(tmp_path, monkeypatch):
    worker, calls, _ = _run_worker(tmp_path, monkeypatch, {}, selected_detector='FID2B')
    assert calls == [('S1.C', 'S1.C', 'FID2B', None), ('S2.C', 'S2.C', 'FID2B', None)]
    assert worker.total_files == 2


def test_default_batch_without_selection_auto_detects(tmp_path, monkeypatch):
    _, calls, _ = _run_worker(tmp_path, monkeypatch, {}, selected_detector='Unknown')
    assert [c[2] for c in calls] == [None, None]


def test_each_file_processed_once_per_detector_with_its_method(tmp_path, monkeypatch):
    methods = tmp_path / "methods"
    methods.mkdir()
    m1 = _method_file(methods, "front")
    m2 = _method_file(methods, "back")
    opts = {'detector_runs': [{'detector': 'FID1A', 'method_path': m1},
                              {'detector': 'FID2B', 'method_path': m2}]}
    worker, calls, _ = _run_worker(tmp_path, monkeypatch, opts, selected_detector='FID1A')
    assert calls == [
        ('S1.C', 'S1.C [FID1A]', 'FID1A', 'front'),
        ('S1.C', 'S1.C [FID2B]', 'FID2B', 'back'),
        ('S2.C', 'S2.C [FID1A]', 'FID1A', 'front'),
        ('S2.C', 'S2.C [FID2B]', 'FID2B', 'back'),
    ]
    assert worker.total_files == 4


def test_detector_missing_from_file_is_skipped_and_reported(tmp_path, monkeypatch):
    opts = {'detector_runs': [{'detector': 'FID1A'}, {'detector': 'TCD3C'}]}
    _, calls, completed = _run_worker(tmp_path, monkeypatch, opts, selected_detector=None)
    assert [c[2] for c in calls] == ['FID1A', 'FID1A']
    assert ('S1.C [TCD3C]', False) in completed
    assert ('S2.C [TCD3C]', False) in completed


def test_load_file_applies_method_then_loads_requested_detector(monkeypatch):
    events = []

    def on_file_selected(path, batch_mode=False, detector=None):
        events.append(('load', path, batch_mode, detector))

    app = SimpleNamespace(
        status_bar=SimpleNamespace(currentMessage=lambda: ""),
        on_file_selected=on_file_selected,
        apply_batch_method=lambda m: events.append(('method', m.name)),
        parameters_frame=SimpleNamespace(get_parameters=lambda: {'peaks': {}}),
    )
    monkeypatch.setattr(aw.main_thread_dispatcher, "run_on_main_thread", lambda fn: fn())
    worker = AutomationWorker(app, "unused")
    method = ChromaMethod(name="back", signal_type="gc")
    assert worker._load_file("X.C", detector="FID2B", method=method) is True
    assert events == [('method', 'back'), ('load', 'X.C', True, 'FID2B')]


def test_detector_run_without_method_gets_users_method_back(tmp_path, monkeypatch):
    """A run with a blank method field must not inherit the previous run's method."""
    methods = tmp_path / "methods"
    methods.mkdir()
    m1 = _method_file(methods, "front")
    _make_c_folders(tmp_path, ["S1"])
    monkeypatch.setattr(aw.CFolder, "get_available_detectors", lambda self: ['FID1A', 'FID2B'])
    monkeypatch.setattr(aw.main_thread_dispatcher, "run_on_main_thread", lambda fn: fn())
    applied = []
    app = SimpleNamespace(
        batch_options={'detector_runs': [{'detector': 'FID1A', 'method_path': m1},
                                         {'detector': 'FID2B', 'method_path': None}]},
        data_handler=SimpleNamespace(current_detector='FID1A'),
        status_bar=SimpleNamespace(currentMessage=lambda: ""),
        on_file_selected=lambda *a, **k: None,
        apply_batch_method=lambda m: applied.append(m.name if m else None),
        parameters_frame=SimpleNamespace(get_parameters=lambda: {'peaks': {}}),
    )
    worker = AutomationWorker(app, str(tmp_path))
    monkeypatch.setattr(worker, "_process_and_integrate", lambda: False)  # stop after load
    worker.run()
    # None asks the app for the user's own method back.
    assert applied == ['front', None]
