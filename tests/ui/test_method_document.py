import pytest
pytest.importorskip('pytestqt')

from ui.app import ChromaKitApp


def _make(qtbot):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    return app


def test_boots_with_untitled_method(qtbot):
    app = _make(qtbot)
    assert app.current_method is not None
    assert app.current_method.name == "Untitled"
    assert app.current_method.signal_type == "gc"
    assert app.current_method_path is None
    assert app._method_dirty is False


def test_title_shows_untitled(qtbot):
    app = _make(qtbot)
    assert "Untitled" in app.windowTitle()
    assert "*" not in app.windowTitle()


def test_mark_dirty_updates_title(qtbot):
    app = _make(qtbot)
    app._mark_dirty(True)
    assert app._method_dirty is True
    assert "Untitled*" in app.windowTitle()


def test_apply_method_to_frames_guards_writeback(qtbot):
    app = _make(qtbot)
    # Programmatic population must not flip dirty.
    app._method_dirty = False
    app._apply_method_to_frames()
    assert app._method_dirty is False


def test_param_edit_preserves_method_metadata(qtbot):
    from logic.method import ChromaMethod
    app = _make(qtbot)
    original = ChromaMethod(name="Loaded", signal_type="gc", version="9")
    fixed_created = original.created_at  # capture the (fresh) creation time
    app.current_method = original
    app.current_method_path = None
    app._method_dirty = False
    # Simulate a user parameter edit: drive the write-back slot directly with
    # the frame's current params (this is what parameters_changed carries).
    app._loading_method = False
    app._on_params_writeback(app.parameters_frame.current_params)
    assert app.current_method.version == "9"
    assert app.current_method.created_at == fixed_created
    assert app.current_method.name == "Loaded"          # identity preserved
    assert app.current_method.signal_type == "gc"
    assert app._method_dirty is True                     # edit still marks dirty


def test_rf_tab_present(qtbot):
    app = _make(qtbot)
    assert app.rf_table_frame is not None
    titles = [app.right_tabs.tabText(i) for i in range(app.right_tabs.count())]
    assert "RF Table" in titles


def test_rt_edit_writes_back_and_dirties(qtbot):
    app = _make(qtbot)
    app._method_dirty = False
    app.rt_table_frame.rt_table.set_rows([
        {"Compound": "Methane", "Start": 1.0, "Apex": 1.1, "End": 1.2},
    ])
    app.rt_table_frame.rt_table.table_edited.emit()  # simulate a user edit notification
    assert any(e.compound == "Methane" for e in app.current_method.rt_table)
    assert app._method_dirty is True


def test_rf_edit_writes_back(qtbot):
    app = _make(qtbot)
    app.rf_table_frame.add_entry("Hydrogen", 402304.0)
    assert any(e.compound == "Hydrogen" for e in app.current_method.rf_table)
    assert app._method_dirty is True


def test_strategy_change_writes_back(qtbot):
    app = _make(qtbot)
    app.quantitation_frame.select_strategy("rf_table")
    assert app.current_method.quant_strategy == "rf_table"
    assert app._method_dirty is True


def test_save_as_writes_and_clears_dirty(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    app._mark_dirty(True)
    out = tmp_path / "mymethod.chromethod"
    monkeypatch.setattr(
        "ui.app.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), "ChromaKit Method (*.chromethod)"),
    )
    app.save_method_as()
    assert out.exists()
    assert app.current_method_path == out or str(app.current_method_path) == str(out)
    assert app.current_method.name == "mymethod"
    assert app._method_dirty is False


def test_load_repopulates_and_clears_dirty(qtbot, tmp_path, monkeypatch):
    from logic.method import ChromaMethod, RFTableEntry
    src = tmp_path / "loaded.chromethod"
    ChromaMethod(
        name="loaded", signal_type="gc", quant_strategy="rf_table",
        rf_table=[RFTableEntry(compound="Hydrogen", response_factor=402304.0)],
    ).to_file(src)

    app = _make(qtbot)
    app._mark_dirty(True)
    monkeypatch.setattr(
        "ui.app.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "ChromaKit Method (*.chromethod)"),
    )
    # Avoid the dirty prompt blocking the test:
    monkeypatch.setattr(app, "_maybe_prompt_save", lambda: True)
    app.load_method()
    assert app.current_method.name == "loaded"
    assert app.current_method.quant_strategy == "rf_table"
    assert app._method_dirty is False
    # Frames were repopulated:
    assert app.quantitation_frame.current_strategy() == "rf_table"
    assert any(e.compound == "Hydrogen" for e in app.current_method.rf_table)


def test_load_and_resave_peak_ranges_and_grouping(qtbot, tmp_path, monkeypatch):
    from logic.method import ChromaMethod, IntegrationSubParams, PeakParams

    src = tmp_path / "ranges.chromethod"
    ChromaMethod(
        name="ranges",
        signal_type="gc",
        peaks=PeakParams(range_filters=[[1.25, 2.5], [3.5, 4.75]]),
        integration=IntegrationSubParams(peak_groups=[[5.0, 6.0]]),
    ).to_file(src)

    app = _make(qtbot)
    monkeypatch.setattr(
        "ui.app.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "ChromaKit Method (*.chromethod)"),
    )
    monkeypatch.setattr(app, "_maybe_prompt_save", lambda: True)
    app.load_method()

    parameters = app.parameters_frame
    assert parameters.range_filter_table.rowCount() == 2
    assert parameters.peak_group_table.rowCount() == 1
    assert parameters.range_filter_table.cellWidget(1, 0).value() == 3.5
    assert parameters.range_filter_table.cellWidget(1, 1).value() == 4.75
    assert parameters.peak_group_table.cellWidget(0, 0).value() == 5.0
    assert parameters.peak_group_table.cellWidget(0, 1).value() == 6.0

    app._on_params_writeback(parameters.current_params)
    saved_path = tmp_path / "resaved.chromethod"
    app.current_method_path = saved_path
    app.save_method()
    saved = ChromaMethod.from_file(saved_path)
    assert saved.peaks.range_filters == [[1.25, 2.5], [3.5, 4.75]]
    assert saved.integration.peak_groups == [[5.0, 6.0]]


def test_manual_assign_gate_uses_frame_is_enabled(qtbot, monkeypatch):
    """Manual RT-assign must gate on rt_table_frame.is_enabled(), not rt_settings.

    Fail-first trick: seed rt_settings={'enabled': True} so the OLD gate would
    PROCEED to the confirmation modal. With is_enabled()==False, the NEW gate
    must BAIL before any dialog. lookup is stubbed to return a compound so the
    OLD path would otherwise reach the modal (it wouldn't bail on empty lookup).
    """
    app = _make(qtbot)

    class _Peak:
        retention_time = 1.1
        compound_id = "Unknown"
        peak_number = 1

    peak = _Peak()
    app.integrated_peaks = [peak]

    # OLD gate would have proceeded on this; prove the NEW gate is what stops us.
    app.rt_settings = {"enabled": True, "rt_table": None}

    # Frame reports NOT enabled → manual assign must bail before any dialog.
    monkeypatch.setattr(app.rt_table_frame, "is_enabled", lambda: False)

    # Guarantee the OLD path would reach the modal (non-None lookup result).
    monkeypatch.setattr(
        "logic.rt_matching.lookup_compound_by_rt",
        lambda *a, **k: "Methane",
    )

    # If the gate fails to bail, exec() is called → flip the flag (and don't
    # actually pop a modal, which would hang the headless test).
    called = {"exec": False}
    from PySide6.QtWidgets import QMessageBox

    def _fake_exec(self):
        called["exec"] = True
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec)

    app.on_rt_assignment_requested(0)

    # New gate bailed: no confirmation modal, peak unchanged, status shown.
    assert called["exec"] is False
    assert peak.compound_id == "Unknown"
    assert "not enabled" in app.status_bar.currentMessage()


class _FakePeak:
    def __init__(self, compound_id, area, rt):
        self.compound_id = compound_id
        self.area = area
        self.retention_time = rt
        self.mol_percent = None
        self.raw_amount = None


def test_rf_dispatch_runs_and_updates_summary(qtbot):
    from logic.method import RFTableEntry
    app = _make(qtbot)
    app.current_method.quant_strategy = "rf_table"
    app.current_method.rf_table = [
        RFTableEntry(compound="Hydrogen", response_factor=1000.0),
        RFTableEntry(compound="Carbon monoxide", response_factor=2000.0),
    ]
    app.integrated_peaks = [
        _FakePeak("Hydrogen", 1000.0, 1.0),
        _FakePeak("Carbon monoxide", 2000.0, 2.0),
    ]
    app.run_quantitation_for_current_strategy()
    # 1000/1000 = 1.0 ; 2000/2000 = 1.0 ; each normalized to 50%
    assert abs(app.integrated_peaks[0].mol_percent - 50.0) < 1e-6
    assert "2 / 2" in app.quantitation_frame.rf_quantitated_label.text()


def test_none_strategy_is_noop(qtbot):
    app = _make(qtbot)
    app.current_method.quant_strategy = None
    app.integrated_peaks = [_FakePeak("Hydrogen", 1.0, 1.0)]
    app.run_quantitation_for_current_strategy()  # must not raise
    assert app.integrated_peaks[0].mol_percent is None


def test_rf_unit_writes_back_and_dirties(qtbot):
    app = _make(qtbot)
    app._method_dirty = False
    app.rf_table_frame.select_rf_unit("area_per_wt_pct")
    assert app.current_method.rf_unit == "area_per_wt_pct"
    assert app._method_dirty is True


def test_scaling_change_writes_method_and_marks_dirty(qtbot):
    app = _make(qtbot)
    app._method_dirty = False
    app._on_scaling_factors_changed(7700.0, 600.0)
    assert app.current_method.signal_factor == 7700.0
    assert app.current_method.area_factor == 600.0
    assert app.signal_factor == 7700.0
    assert app.area_factor == 600.0
    assert app.data_handler.signal_factor == 7700.0
    assert app._method_dirty is True


def test_scaling_change_stores_one_as_none(qtbot):
    """A factor of 1.0 means 'no scaling' and is stored as None on the method."""
    app = _make(qtbot)
    app._on_scaling_factors_changed(1.0, 1.0)
    assert app.current_method.signal_factor is None
    assert app.current_method.area_factor is None


def test_load_method_syncs_scaling_to_session(qtbot, tmp_path):
    from logic.method import ChromaMethod
    app = _make(qtbot)
    m = ChromaMethod(name="Scaled", signal_type="gc",
                     signal_factor=7700.0, area_factor=600.0)
    path = tmp_path / "scaled.chromethod"
    m.to_file(path)
    app.current_method = ChromaMethod.from_file(str(path))
    app._sync_scaling_from_method()
    assert app.signal_factor == 7700.0
    assert app.area_factor == 600.0
    assert app.data_handler.signal_factor == 7700.0


def test_params_writeback_preserves_scaling(qtbot):
    """Editing parameters must NOT wipe the method's scaling/detector fields."""
    from logic.method import ChromaMethod
    app = _make(qtbot)
    app.current_method = ChromaMethod(name="X", signal_type="gc",
                                      signal_factor=7700.0, area_factor=600.0,
                                      detector="TCD3C")
    app._method_dirty = False
    app._loading_method = False
    app._on_params_writeback(app.parameters_frame.current_params)
    assert app.current_method.signal_factor == 7700.0
    assert app.current_method.area_factor == 600.0
    assert app.current_method.detector == "TCD3C"


def test_rf_dispatch_honours_rf_normalize_false(qtbot):
    from logic.method import RFTableEntry
    app = _make(qtbot)
    app.current_method.quant_strategy = "rf_table"
    app.current_method.rf_normalize = False
    app.current_method.rf_table = [RFTableEntry(compound="Hydrogen", response_factor=1000.0)]
    app.integrated_peaks = [_FakePeak("Hydrogen", 3000.0, 1.0)]
    app.run_quantitation_for_current_strategy()
    assert app.integrated_peaks[0].raw_amount == 3.0
    assert app.integrated_peaks[0].mol_percent is None


def test_rf_dispatch_passes_method_rf_unit(qtbot):
    from logic.method import RFTableEntry
    app = _make(qtbot)
    app.current_method.quant_strategy = "rf_table"
    app.current_method.rf_unit = "area_per_wt_pct"
    app.current_method.rf_table = [RFTableEntry(compound="Hydrogen", response_factor=1000.0)]
    app.integrated_peaks = [_FakePeak("Hydrogen", 3000.0, 1.0)]
    app.run_quantitation_for_current_strategy()
    # wt% basis: composition_percent is written, mol_percent is not
    assert app.integrated_peaks[0].mol_percent is None
    assert getattr(app.integrated_peaks[0], "composition_percent", None) == 100.0


def test_params_writeback_preserves_rf_unit_and_normalize(qtbot):
    from logic.method import ChromaMethod
    app = _make(qtbot)
    app.current_method = ChromaMethod(name="X", signal_type="gc",
                                      rf_unit="area_per_wt_pct", rf_normalize=False)
    app._loading_method = False
    app._on_params_writeback(app.parameters_frame.current_params)
    assert app.current_method.rf_unit == "area_per_wt_pct"
    assert app.current_method.rf_normalize is False


def test_apply_batch_method_none_restores_users_method(qtbot):
    from logic.method import ChromaMethod
    app = _make(qtbot)
    app.current_method = ChromaMethod(name="mine", signal_type="gc")
    app._save_method_state_for_batch()
    app.apply_batch_method(ChromaMethod(name="front", signal_type="gc"))
    assert app.current_method.name == "front"
    app.apply_batch_method(None)
    assert app.current_method.name == "mine"


# ── Loading a method re-applies it to the open file ─────────────────────────

def _load_with(app, monkeypatch, tmp_path, **method_kwargs):
    from logic.method import ChromaMethod
    path = tmp_path / "m.chromethod"
    ChromaMethod(name="applied", signal_type="gc", **method_kwargs).to_file(path)
    monkeypatch.setattr("ui.app.QFileDialog.getOpenFileName", lambda *a, **k: (str(path), ""))
    monkeypatch.setattr(app, "_maybe_prompt_save", lambda: True)
    app.load_method()


def _open_file(app, monkeypatch, detectors=("FID1A", "FID2B")):
    import numpy as np
    app.current_directory_path = "/data/sample1.D"
    app.current_cf = None
    app.current_x = np.linspace(0, 10, 50)
    app.current_y = np.zeros(50)
    app.data_handler.current_detector = "FID1A"
    monkeypatch.setattr(app.data_handler, "get_available_detectors", lambda: list(detectors))
    calls = {"process": [], "reload": []}
    monkeypatch.setattr(app, "process_and_display", lambda *a, **k: calls["process"].append(app.parameters_frame.get_parameters()))
    monkeypatch.setattr(app, "on_file_selected", lambda path, **k: calls["reload"].append((path, k)))
    return calls


def test_load_method_reprocesses_open_file_with_new_params(qtbot, tmp_path, monkeypatch):
    from logic.method import PeakParams
    app = _make(qtbot)
    calls = _open_file(app, monkeypatch)
    _load_with(app, monkeypatch, tmp_path, peaks=PeakParams(range_filters=[[1.0, 2.0]]))
    assert calls["reload"] == []
    assert len(calls["process"]) == 1
    assert calls["process"][0]["peaks"]["range_filters"] == [[1.0, 2.0]]
    assert "applied" in app.status_bar.currentMessage() and "sample1.D" in app.status_bar.currentMessage()


def test_load_method_without_open_file_only_fills_frames(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    calls = {"process": [], "reload": []}
    monkeypatch.setattr(app, "process_and_display", lambda *a, **k: calls["process"].append(1))
    monkeypatch.setattr(app, "on_file_selected", lambda *a, **k: calls["reload"].append(1))
    _load_with(app, monkeypatch, tmp_path)
    assert calls == {"process": [], "reload": []}


def test_load_method_with_new_signal_factor_reloads_once(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    calls = _open_file(app, monkeypatch)
    _load_with(app, monkeypatch, tmp_path, signal_factor=2.0)
    assert calls["reload"] == [("/data/sample1.D", {"detector": None})]
    assert calls["process"] == []


def test_load_method_detector_present_reloads_with_it(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    calls = _open_file(app, monkeypatch)
    _load_with(app, monkeypatch, tmp_path, detector="FID2B")
    assert calls["reload"] == [("/data/sample1.D", {"detector": "FID2B"})]
    assert calls["process"] == []


def test_load_method_detector_absent_keeps_channel_and_says_so(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    calls = _open_file(app, monkeypatch)
    _load_with(app, monkeypatch, tmp_path, detector="TCD3C")
    assert calls["reload"] == []
    assert len(calls["process"]) == 1
    assert "TCD3C not in this file; showing FID1A" in app.status_bar.currentMessage()


def test_method_label_tracks_name_dirty_and_path(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    assert app.method_label.text() == "Method: Untitled"
    assert app.method_label.toolTip() == "Not saved"
    assert app.method_label.accessibleName()
    app._mark_dirty(True)
    assert app.method_label.text() == "Method: Untitled*"
    _load_with(app, monkeypatch, tmp_path)
    assert app.method_label.text() == "Method: applied"
    assert app.method_label.toolTip() == str(tmp_path / "m.chromethod")


def test_save_reports_current_method(qtbot, tmp_path, monkeypatch):
    app = _make(qtbot)
    target = tmp_path / "saved.chromethod"
    monkeypatch.setattr("ui.app.QFileDialog.getSaveFileName", lambda *a, **k: (str(target), ""))
    app.save_method_as()
    assert "saved" in app.status_bar.currentMessage() and "current method" in app.status_bar.currentMessage()
