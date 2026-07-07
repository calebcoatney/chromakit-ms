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
