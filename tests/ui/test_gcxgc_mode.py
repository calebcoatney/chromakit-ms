import json
from pathlib import Path

import pytest

pytest.importorskip("pytestqt")

from logic.ms_search_core import format_casno
from tests.logic.gcxgc import synth
from ui.app import ChromaKitApp


def _scene():
    return synth.Scene(peaks=[
        synth.Peak(t1_min=2.0, t2_s=1.0, height=10.0, base_mz=57),
        synth.Peak(t1_min=2.0, t2_s=2.5, height=1.0, base_mz=91),
        synth.Peak(t1_min=4.0, t2_s=1.6, height=3.0, base_mz=105),
    ], duration_min=6.0)


@pytest.fixture
def gcxgc_app(qtbot, tmp_path):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    c = synth.build_run(tmp_path, _scene())
    app.on_file_selected(str(c))
    assert app._gcxgc_mode
    pf = app.parameters_frame
    pf.gcxgc_baseline_enabled.setChecked(True)
    pf.gcxgc_peaks_enabled.setChecked(True)
    pf.gcxgc_prominence_entry.setText("50")
    return app, c


def test_layout_fields_show_file_values_and_provenance(gcxgc_app):
    app, _ = gcxgc_app
    pf = app.parameters_frame
    assert pf.gcxgc_pm_entry.text() == "" and pf.gcxgc_pm_entry.placeholderText() == "4"
    assert pf.gcxgc_phase_entry.text() == "" and pf.gcxgc_phase_entry.placeholderText() == "-1.8"
    assert pf.gcxgc_pm_source.text() == "from file (HDR)"
    assert pf.gcxgc_phase_source.text() == "ChromSpace saved view"


def test_integrate_finds_peaks_and_click_shows_spectrum(gcxgc_app):
    app, _ = gcxgc_app
    app.button_frame.integrate_button.click()
    assert len(app._gcxgc_peaks) == 3
    p = max(app._gcxgc_peaks, key=lambda q: q.volume)
    app.plot_frame.peak_selected.emit(p)
    assert p.spectrum_scan is not None


def test_bad_period_is_reported_not_crashed(gcxgc_app, monkeypatch):
    app, _ = gcxgc_app
    warned = []
    monkeypatch.setattr("ui.app.QMessageBox.warning", lambda *a, **k: warned.append(a))
    app.parameters_frame.gcxgc_pm_entry.setText("4.01")
    app.button_frame.integrate_button.click()
    assert warned and "whole number" in str(warned[0][2])
    assert app._gcxgc_run.modulation_period_s == 4.0          # the rejected period left the layout alone


def test_missing_period_prompts(qtbot, tmp_path, monkeypatch):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    c = synth.build_run(tmp_path, synth.Scene(duration_min=2.0), insight=False, view=False)
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getDouble", lambda *a, **k: (4.0, True))
    app.on_file_selected(str(c))
    assert app._gcxgc_mode and app._gcxgc_run.modulation_period_s == 4.0


def test_batch_processing_is_refused_in_gcxgc_mode(gcxgc_app, monkeypatch):
    app, _ = gcxgc_app
    shown = []
    monkeypatch.setattr("ui.app.QMessageBox.information", lambda *a, **k: shown.append(a))
    app.on_automation_clicked()
    assert shown and "not supported" in str(shown[0][2])


def test_ms_search_all_keeps_top_hits(gcxgc_app):
    app, _ = gcxgc_app
    app.button_frame.integrate_button.click()

    class Comp:
        casno = "71432"

    class FakeToolkit:
        library = {"Benzene": Comp()}
        mz_shift = 0

        def search_vector(self, spectrum, top_n=1, **kw):
            return [("Benzene", 0.91), ("Toluene", 0.80)][:top_n]

        search_w2v = search_hybrid = search_vector

    app.ms_frame.ms_toolkit = FakeToolkit()
    app.ms_frame.library_loaded = True
    app.ms_frame.search_options = {"search_method": "vector", "top_n": 2}
    app.run_batch_ms_search()
    p = app._gcxgc_peaks[0]
    assert p.top_hits == [("Benzene", 0.91, format_casno("71432")), ("Toluene", 0.80, None)]
    assert (p.compound_name, p.casno) == ("Benzene", format_casno("71432"))


def test_export_button_writes_all_result_files(gcxgc_app):
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    for f in (c / "results").glob("*"):
        f.unlink()                                    # clear any auto-export
    app.button_frame.export_button.click()
    assert sorted(f.name for f in (c / "results").iterdir()) == [
        "Synth - GCxGC peaks.csv", "Synth - GCxGC peaks.json",
        "Synth - GCxGC spectra.mgf", "Synth - GCxGC spectra.msp", "Synth - GNPS quant.csv"]


def test_numeric_item_sorts_by_value(qtbot):
    from ui.widgets.numeric_item import NumericItem
    items = [NumericItem(v, str(v)) for v in (10, 9, 100)]
    assert [i.text() for i in sorted(items)] == ["9", "10", "100"]


def test_auto_export_follows_integrate_and_ms_search(gcxgc_app, monkeypatch):
    app, c = gcxgc_app
    monkeypatch.setattr(app.export_manager, "should_export_json", lambda trigger: True)
    csv_path = c / "results" / "Synth - GCxGC peaks.csv"
    app.button_frame.integrate_button.click()
    assert csv_path.exists() and "Benzene" not in csv_path.read_text()
    csv_path.unlink()

    class FakeToolkit:
        library = {}
        mz_shift = 0

        def search_vector(self, spectrum, top_n=1, **kw):
            return [("Benzene", 0.91)][:top_n]

        search_w2v = search_hybrid = search_vector

    app.ms_frame.ms_toolkit = FakeToolkit()
    app.ms_frame.library_loaded = True
    app.ms_frame.search_options = {"search_method": "vector", "top_n": 1}
    app.run_batch_ms_search()
    assert "Benzene" in csv_path.read_text()


def _clear_results(c):
    for f in (c / "results").glob("*"):
        f.unlink()


def _record(c):
    return json.load(open(c / "results" / "Synth - GCxGC peaks.json"))


def _fake_search(app):
    class FakeToolkit:
        library = {}
        mz_shift = 0

        def search_vector(self, spectrum, top_n=1, **kw):
            return [("Benzene", 0.91)][:top_n]

        search_w2v = search_hybrid = search_vector

    class Settings:                       # never write the real ms/library_path
        def value(self, key, default=None, **kw):
            return "/libs/NIST14.json" if key == "ms/library_path" else default

    app.ms_frame.settings = Settings()
    app.ms_frame.ms_toolkit = FakeToolkit()
    app.ms_frame.library_loaded = True
    app.ms_frame.search_options = {"search_method": "vector", "top_n": 1}


def test_export_records_the_settings_that_produced_the_peaks(gcxgc_app):
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    app.parameters_frame.gcxgc_prominence_entry.setText("9999")   # edited, never re-integrated
    _clear_results(c)
    app.button_frame.export_button.click()
    settings = _record(c)["settings"]
    assert settings["min_prominence"] == 50.0 and settings["detection_signal"] == "fid"


def test_search_provenance_is_recorded_only_after_a_search(gcxgc_app):
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    _clear_results(c)
    app.button_frame.export_button.click()
    rec = _record(c)
    assert rec["search_method"] is None and rec["library"] is None and rec["search_complete"] is None

    _fake_search(app)
    app.run_batch_ms_search()
    _clear_results(c)
    app.button_frame.export_button.click()
    rec = _record(c)
    assert rec["search_method"] == "vector" and rec["library"] == "NIST14.json"
    assert rec["search_complete"] is True

    app.button_frame.integrate_button.click()            # a new peak set has had no search
    _clear_results(c)
    app.button_frame.export_button.click()
    rec = _record(c)
    assert rec["search_method"] is None and rec["library"] is None


def test_layout_change_without_detection_drops_stale_peaks(gcxgc_app):
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks and not app.view_results_button.isHidden()
    _clear_results(c)
    pf = app.parameters_frame
    pf.gcxgc_baseline_enabled.setChecked(False)
    pf.gcxgc_peaks_enabled.setChecked(False)
    pf.gcxgc_phase_entry.setText("-1.0")
    app.button_frame.integrate_button.click()
    assert app._gcxgc_run.phase_shift_s == -1.0           # the new layout was applied ...
    assert app._gcxgc_peaks == [] and app._gcxgc_proc is None      # ... so the old peaks are gone
    assert app.integrated_peaks == [] and app.plot_frame._gcxgc_peaks == []
    assert app.view_results_button.isHidden()
    app.button_frame.export_button.click()
    assert list((c / "results").glob("*")) == []
    assert "Run Integrate" in app.status_bar.currentMessage()


def test_processing_failure_drops_stale_peaks(gcxgc_app, monkeypatch):
    app, _ = gcxgc_app
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks

    def boom(self, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("logic.gcxgc_processor.GCxGCProcessor.detect_peaks", boom)
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks == [] and app._gcxgc_proc is None
    assert "boom" in app.status_bar.currentMessage()


def test_auto_export_failure_is_a_status_message_not_a_dialog(gcxgc_app, monkeypatch):
    app, _ = gcxgc_app
    monkeypatch.setattr(app.export_manager, "should_export_json", lambda trigger: True)
    shown = []
    monkeypatch.setattr("ui.app.QMessageBox.critical", lambda *a, **k: shown.append(a))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("logic.gcxgc_export.export_gcxgc_results", boom)
    app.button_frame.integrate_button.click()
    assert not shown and "disk full" in app.status_bar.currentMessage()
    app.button_frame.export_button.click()               # the Export button does warn
    assert shown and "disk full" in str(shown[0][2])



def test_user_phase_marks_provenance(gcxgc_app):
    app, _ = gcxgc_app
    pf = app.parameters_frame
    pf.gcxgc_phase_entry.setText("2.2")
    app.button_frame.integrate_button.click()
    assert app._gcxgc_run.provenance["phase_shift_s"] == "user"
    assert pf.gcxgc_phase_source.text() == "your value"


def test_tic_display_toggle_redraws_without_reprocessing(gcxgc_app, monkeypatch):
    app, _ = gcxgc_app
    pf = app.parameters_frame
    app.button_frame.integrate_button.click()
    calls = {"process": 0, "render": 0}
    monkeypatch.setattr(app, "_run_gcxgc_processing",
                        lambda: calls.__setitem__("process", calls["process"] + 1))
    real_render = app.plot_frame.render_gcxgc

    def counting_render(*a, **k):
        calls["render"] += 1
        return real_render(*a, **k)

    monkeypatch.setattr(app.plot_frame, "render_gcxgc", counting_render)
    pf.gcxgc_tic_dbc.setChecked(True)
    pf.gcxgc_tic_raw.setChecked(True)
    assert calls == {"process": 0, "render": 2}


def test_gcxgc_widgets_write_into_method(gcxgc_app):
    app, _ = gcxgc_app
    app.parameters_frame.gcxgc_min_sub_peaks_spin.setValue(4)
    assert app.parameters_frame.current_params["gcxgc"]["min_modulations"] == 4
    assert app.current_method.gcxgc.min_modulations == 4


def test_apply_method_restores_gcxgc_widgets(qtbot):
    from logic.method import ChromaMethod, GCxGCParams
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_method = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(
        modulation_period_s=6.0, detection_signal="dbc_tic", apex_window_scans=7))
    app._apply_method_to_frames()
    pf = app.parameters_frame
    assert pf.gcxgc_pm_entry.text() == "6"
    assert pf.gcxgc_signal_combo.currentData() == "dbc_tic"
    assert pf.gcxgc_apex_window_spin.value() == 7


def _show_parameters_tab(app):
    """Show the window on the Parameters tab and process events WITHOUT running deferred deletes
    (qtbot.wait would run them), as a script or a test that only pumps processEvents() does."""
    from PySide6.QtWidgets import QApplication
    app.resize(1700, 1050)
    app.show()
    app.right_tabs.setCurrentWidget(app.parameters_frame)
    QApplication.processEvents()


def _assert_only_gcxgc_groups_visible(pf):
    """Every visible QGroupBox anywhere under the parameters frame (not just params_layout)."""
    from PySide6.QtWidgets import QGroupBox
    assert pf.isVisible()
    visible = [g for g in pf.findChildren(QGroupBox) if g.isVisible()]
    expected = [pf.section_groups[k] for k in ("gcxgc_layout", "gcxgc_baseline", "gcxgc_peaks")]
    assert set(visible) == set(expected), [g.title() for g in visible]


def test_gcxgc_mode_hides_every_1d_group(gcxgc_app):
    app, _ = gcxgc_app
    pf = app.parameters_frame
    pf.set_section_visibility({"peaks": True, "baseline": True, "smoothing": True})
    for key, group in pf.section_groups.items():
        assert group.isVisibleTo(pf) == key.startswith("gcxgc"), key
    groups = set(pf.section_groups.values())
    for i in range(pf.params_layout.count()):
        w = pf.params_layout.itemAt(i).widget()
        if w is not None and w not in groups and w.isVisibleTo(pf):
            assert w.text() == "Integration Parameters", w     # the panel title is the only extra
    # Task 8 walkthrough: the groups the startup rebuild replaced stayed visible outside params_layout.
    _show_parameters_tab(app)
    _assert_only_gcxgc_groups_visible(pf)


def test_shown_panel_shows_only_gcxgc_groups_after_load_method(qtbot, tmp_path):
    """Load Method rebuilds the panel again while a GCxGC run is shown: the replaced GCxGC and 1D
    groups must not linger on screen."""
    from PySide6.QtWidgets import QApplication
    from logic.method import ChromaMethod, GCxGCParams
    app = ChromaKitApp()
    qtbot.addWidget(app)
    _show_parameters_tab(app)
    app.on_file_selected(str(synth.build_run(tmp_path, synth.Scene(duration_min=2.0))))
    app.right_tabs.setCurrentWidget(app.parameters_frame)
    QApplication.processEvents()
    _assert_only_gcxgc_groups_visible(app.parameters_frame)
    app.current_method = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(phase_shift_s=2.2))
    app._apply_method_to_frames()
    QApplication.processEvents()
    _assert_only_gcxgc_groups_visible(app.parameters_frame)


# ── Carried from the Task 3 and Task 4 reviews ──────────────────────────────

_FILE_PROVENANCE = {"modulation_period_s": "hdr", "phase_shift_s": "lsc_view"}


def test_second_run_gets_its_own_file_layout_and_no_processing_on_load(gcxgc_app, monkeypatch):
    """Run A shows the DBC TIC with blank layout fields; loading run B (no .dbc.lsc, its own
    file phase) gives B its own file layout and provenance, and runs no processing mid-load."""
    app, c = gcxgc_app
    pf = app.parameters_frame
    pf.gcxgc_tic_dbc.setChecked(True)
    assert pf.gcxgc_pm_entry.text() == "" and pf.gcxgc_phase_entry.text() == ""
    b = synth.build_run(c.parent, synth.Scene(phase_s=-1.0, duration_min=2.0), name="SynthB", dbc=False)
    processed = []
    monkeypatch.setattr(app, "_run_gcxgc_processing", lambda: processed.append(True))
    app.on_file_selected(str(b))
    run = app._gcxgc_run
    assert run.sample_id == "SynthB"
    assert (run.modulation_period_s, run.phase_shift_s) == (4.0, -1.0)
    assert run.provenance == _FILE_PROVENANCE
    assert processed == []
    assert pf.gcxgc_tic_raw.isChecked() and not pf.gcxgc_tic_dbc.isEnabled()
    assert pf.gcxgc_phase_entry.placeholderText() == "-1"


def test_integrate_with_blank_layout_keeps_file_provenance(gcxgc_app):
    app, _ = gcxgc_app
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks
    assert app._gcxgc_run.provenance == _FILE_PROVENANCE
    assert app.parameters_frame.gcxgc_pm_source.text() == "from file (HDR)"


def test_loading_a_new_run_drops_the_previous_runs_results(gcxgc_app):
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks and not app.view_results_button.isHidden()
    b = synth.build_run(c.parent, synth.Scene(duration_min=2.0), name="SynthB")
    run_a = app._gcxgc_run
    app.on_file_selected(str(b))
    assert app._gcxgc_run.sample_id == "SynthB"
    assert app._gcxgc_peaks == [] and app._gcxgc_proc is None
    assert app.integrated_peaks == [] and app.plot_frame._gcxgc_peaks == []
    assert app.view_results_button.isHidden()
    assert _run_files_closed(run_a) and not _run_files_closed(app._gcxgc_run)


# ── Panel consistency the brief's wiring depends on ─────────────────────────

def test_load_method_in_gcxgc_mode_keeps_the_panel_and_run_info(gcxgc_app):
    """Load Method rebuilds every widget: the 1D groups must stay hidden and the run's file
    values, provenance and DBC availability must come back."""
    from logic.method import ChromaMethod, GCxGCParams
    app, _ = gcxgc_app
    app.current_method = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(phase_shift_s=2.2))
    app._apply_method_to_frames()
    pf = app.parameters_frame
    for key, group in pf.section_groups.items():
        assert group.isVisibleTo(pf) == key.startswith("gcxgc"), key
    assert pf.gcxgc_phase_entry.text() == "2.2"
    assert pf.gcxgc_pm_entry.placeholderText() == "4"
    assert pf.gcxgc_pm_source.text() == "from file (HDR)"
    assert pf.gcxgc_tic_dbc.isEnabled()


def test_gcxgc_edit_does_not_trigger_1d_reprocessing(gcxgc_app, monkeypatch):
    """GCxGC edits write into the method; they must not run the 1D pipeline or tell the
    user to load data."""
    app, _ = gcxgc_app
    ran = []
    monkeypatch.setattr(app, "process_and_display", lambda *a, **k: ran.append(True))
    app.status_bar.showMessage("")
    app.parameters_frame.gcxgc_min_sub_peaks_spin.setValue(4)
    assert ran == []
    assert "Load data first" not in app.status_bar.currentMessage()


def test_offset_mode_reaches_the_run(gcxgc_app):
    app, _ = gcxgc_app
    pf = app.parameters_frame
    pf.gcxgc_offset_mode_combo.setCurrentIndex(pf.gcxgc_offset_mode_combo.findData("fixed"))
    assert pf.gcxgc_offset_spin.isEnabled()
    pf.gcxgc_offset_spin.setValue(-5.5)
    app.button_frame.integrate_button.click()
    off = app._gcxgc_run.offset()
    assert (off.mode, off.a, off.b, off.ok) == ("fixed", -5.5, 0.0, True)
    assert "offset fit failed" not in app.status_bar.currentMessage()
    assert app.current_method.gcxgc.offset_mode == "fixed" and app.current_method.gcxgc.offset_scans == -5.5


def test_failed_offset_fit_is_reported_after_integrate(gcxgc_app):
    app, _ = gcxgc_app                 # a 6-min run is shorter than one 6-min fit window: the fit fails
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks and not app._gcxgc_run.offset().ok
    assert "offset fit failed" in app.status_bar.currentMessage()


def test_export_settings_record_the_layout_the_run_used(gcxgc_app):
    app, c = gcxgc_app                 # blank period/phase fields: the snapshot names the file values
    app.button_frame.integrate_button.click()
    _clear_results(c)
    app.button_frame.export_button.click()
    rec = _record(c)
    assert (rec["settings"]["pm"], rec["settings"]["phase_shift"]) == (4.0, -1.8)
    assert (rec["run"]["modulation_period_s"], rec["run"]["phase_shift_s"]) == (4.0, -1.8)


def test_method_layout_that_does_not_fit_the_run_is_ignored_on_load(qtbot, tmp_path):
    from logic.method import ChromaMethod, GCxGCParams
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_method = ChromaMethod(name="g", signal_type="gcxgc",
                                      gcxgc=GCxGCParams(modulation_period_s=4.01))
    app._apply_method_to_frames()
    app.on_file_selected(str(synth.build_run(tmp_path, synth.Scene(duration_min=2.0))))
    run = app._gcxgc_run
    assert run.modulation_period_s == 4.0 and run.provenance == _FILE_PROVENANCE
    assert "layout from the method ignored" in app.status_bar.currentMessage()
    assert app.parameters_frame.gcxgc_pm_source.text() == "from file (HDR)"


def _assert_dbc_tic_kept_but_disabled(app):
    pf = app.parameters_frame
    combo = pf.gcxgc_signal_combo
    assert app.current_method.gcxgc.detection_signal == "dbc_tic"
    assert pf.current_params["gcxgc"]["detection_signal"] == "dbc_tic"
    assert combo.currentData() == "dbc_tic"
    assert not combo.model().item(combo.findData("dbc_tic")).isEnabled()


def test_no_dbc_run_keeps_the_methods_detection_signal_and_detects_on_fid(qtbot, tmp_path):
    """Review fix 1 (file-load path): a run without the method's detection signal leaves the
    method alone (no write-back, not dirty); Integrate detects on FID for this run and says so."""
    from logic.method import ChromaMethod, GCxGCParams
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_method = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(detection_signal="dbc_tic"))
    app._apply_method_to_frames()
    pf = app.parameters_frame
    emitted = []
    pf.parameters_changed.connect(lambda _p: emitted.append(True))
    dirty_before = app._method_dirty
    app.on_file_selected(str(synth.build_run(tmp_path, _scene(), name="NoDbc", dbc=False)))
    _assert_dbc_tic_kept_but_disabled(app)
    assert app._method_dirty == dirty_before and emitted == []

    pf.gcxgc_prominence_entry.setText("50")
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks and all(p.detection_signal == "fid" for p in app._gcxgc_peaks)
    assert "DBC TIC not available for this run; detected on FID" in app.status_bar.currentMessage()
    assert app._gcxgc_export_settings["detection_signal"] == "fid"
    _assert_dbc_tic_kept_but_disabled(app)
    assert emitted == []


def test_load_method_on_a_no_dbc_run_keeps_the_methods_detection_signal(qtbot, tmp_path, monkeypatch):
    """Review fix 1 (Load Method path): same outcome as the file-load path, and a later unrelated
    GCxGC edit does not write FID into the method either."""
    from logic.method import ChromaMethod, GCxGCParams
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.on_file_selected(str(synth.build_run(tmp_path, synth.Scene(duration_min=2.0), name="NoDbc", dbc=False)))
    path = tmp_path / "dbc_tic.chromethod"
    ChromaMethod(name="dbc_tic", signal_type="gcxgc", gcxgc=GCxGCParams(detection_signal="dbc_tic")).to_file(path)
    app._method_dirty = False
    monkeypatch.setattr("ui.app.QFileDialog.getOpenFileName", lambda *a, **k: (str(path), ""))
    app.load_method()
    assert app.current_method.name == "dbc_tic"
    _assert_dbc_tic_kept_but_disabled(app)
    assert app._method_dirty is False
    app.parameters_frame.gcxgc_min_sub_peaks_spin.setValue(4)          # an unrelated edit
    assert app.current_method.gcxgc.min_modulations == 4
    _assert_dbc_tic_kept_but_disabled(app)


def test_no_dbc_run_does_not_rewrite_the_methods_dbc_choice(qtbot, tmp_path):
    """Controller ruling (T5): a run without a .dbc.lsc disables "Use DBC spectra" but leaves it
    checked, so loading it neither rewrites use_dbc_spectra nor dirties the method."""
    from logic.method import ChromaMethod, GCxGCParams
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_method = ChromaMethod(name="g", signal_type="gcxgc", gcxgc=GCxGCParams(use_dbc_spectra=True))
    app._apply_method_to_frames()
    pf = app.parameters_frame
    emitted = []
    pf.parameters_changed.connect(lambda _p: emitted.append(True))
    dirty_before = app._method_dirty
    app.on_file_selected(str(synth.build_run(tmp_path, synth.Scene(duration_min=2.0), name="NoDbc", dbc=False)))
    assert app._gcxgc_run.dbc_lsc is None
    assert app.current_method.gcxgc.use_dbc_spectra is True
    assert pf.current_params["gcxgc"]["use_dbc_spectra"] is True
    assert pf.gcxgc_use_dbc.isChecked() and not pf.gcxgc_use_dbc.isEnabled()
    assert app._method_dirty == dirty_before and emitted == []
    app.on_file_selected(str(synth.build_run(tmp_path, synth.Scene(duration_min=2.0), name="WithDbc")))
    assert pf.gcxgc_use_dbc.isEnabled() and pf.gcxgc_use_dbc.isChecked()
    assert pf.gcxgc_tic_dbc.isEnabled()


# ── Review fix 2: a cleared numeric field reaches the method (real key events) ──

def _type_and_return(qtbot, entry, text):
    from PySide6.QtCore import Qt
    entry.selectAll()
    qtbot.keyClick(entry, Qt.Key_Backspace)          # clears the field; a no-op when it is empty
    if text:
        qtbot.keyClicks(entry, text)
    qtbot.keyClick(entry, Qt.Key_Return)


@pytest.mark.parametrize("entry_name, field, typed, value", [
    ("gcxgc_pm_entry", "modulation_period_s", "4", 4.0),
    ("gcxgc_phase_entry", "phase_shift_s", "2.2", 2.2),
    ("gcxgc_prominence_entry", "min_prominence", "75", 75.0),
])
def test_typing_then_clearing_a_numeric_field_reaches_the_method(gcxgc_app, qtbot, entry_name, field,
                                                                 typed, value):
    app, _ = gcxgc_app
    entry = getattr(app.parameters_frame, entry_name)
    app._method_dirty = False
    _type_and_return(qtbot, entry, typed)
    assert getattr(app.current_method.gcxgc, field) == value and app._method_dirty
    _type_and_return(qtbot, entry, "")
    assert entry.text() == ""
    assert getattr(app.current_method.gcxgc, field) is None
    assert app.parameters_frame.current_params["gcxgc"][field] is None


def test_clearing_the_phase_returns_the_run_to_the_file_phase(gcxgc_app, qtbot):
    app, _ = gcxgc_app
    entry = app.parameters_frame.gcxgc_phase_entry
    app._method_dirty = False
    _type_and_return(qtbot, entry, "2.2")
    assert app.current_method.gcxgc.phase_shift_s == 2.2 and app._method_dirty
    app.button_frame.integrate_button.click()
    assert app._gcxgc_run.provenance["phase_shift_s"] == "user"
    _type_and_return(qtbot, entry, "")
    assert app.current_method.gcxgc.phase_shift_s is None
    app.button_frame.integrate_button.click()
    assert app._gcxgc_run.provenance["phase_shift_s"] == "lsc_view"
    assert app._gcxgc_run.phase_shift_s == -1.8


def test_return_on_an_untouched_blank_field_does_not_dirty_the_method(qtbot, tmp_path):
    from PySide6.QtCore import Qt
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.on_file_selected(str(synth.build_run(tmp_path, synth.Scene(duration_min=2.0))))
    app._method_dirty = False
    qtbot.keyClick(app.parameters_frame.gcxgc_pm_entry, Qt.Key_Return)
    assert app._method_dirty is False


# ── Final review, Important 1: a failed load never leaves the previous run's state behind ──

def _run_files_closed(run):
    return all(f is None or f._mm is None for f in (run.lsc, run.dbc_lsc))


def _integrated_run_a(app, tmp_path, monkeypatch):
    """Load run A, Integrate with auto-export on, and return (A, its GCxGCRun) with A's results/ empty."""
    monkeypatch.setattr(app.export_manager, "should_export_json", lambda trigger: True)
    a = synth.build_run(tmp_path / "a", _scene())
    app.on_file_selected(str(a))
    app.parameters_frame.gcxgc_prominence_entry.setText("50")
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks
    _clear_results(a)
    return a, app._gcxgc_run


def _assert_run_dropped(app, old_run):
    assert app._gcxgc_run is None and not app._gcxgc_mode
    assert app._gcxgc_peaks == [] and app._gcxgc_proc is None
    assert app.plot_frame._gcxgc_peaks == []
    assert not hasattr(app.plot_frame, "_gcxgc_ax_fid")          # A's heatmaps are off the screen
    assert _run_files_closed(old_run)


def _assert_nothing_written(app, *folders):
    app.button_frame.export_button.click()
    app.button_frame.integrate_button.click()
    for folder in folders:
        assert list((folder / "results").iterdir()) == [], folder


def test_export_writes_into_the_runs_own_folder(gcxgc_app):
    from logic.c_folder import CFolder
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    _clear_results(c)
    other = synth.build_run(c.parent / "other", synth.Scene(duration_min=2.0), name="Other")
    app.current_cf = CFolder.open(str(other))          # the app's notion of "current" moved on
    app.current_directory_path = str(other)
    app.button_frame.export_button.click()
    assert list((other / "results").iterdir()) == []
    assert len(list((c / "results").iterdir())) == 5


def test_failed_load_drops_the_previous_run(qtbot, tmp_path, monkeypatch):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    monkeypatch.setattr("ui.app.QMessageBox.critical", lambda *a, **k: None)
    a, run_a = _integrated_run_a(app, tmp_path, monkeypatch)
    b = synth.build_run(tmp_path / "b", _scene())
    lsc = b / "data" / "Synth.lsc"
    lsc.write_bytes(lsc.read_bytes()[: lsc.stat().st_size // 2])      # a damaged copy
    app.on_file_selected(str(b))
    assert "Error loading file" in app.status_bar.currentMessage()
    _assert_run_dropped(app, run_a)
    _assert_nothing_written(app, a, b)


def test_cancelled_period_prompt_drops_the_previous_run(qtbot, tmp_path, monkeypatch):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    a, run_a = _integrated_run_a(app, tmp_path, monkeypatch)
    b = synth.build_run(tmp_path / "b", _scene(), insight=False, view=False)     # no period anywhere
    monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getDouble", lambda *a, **k: (4.0, False))
    app.on_file_selected(str(b))
    assert "Load cancelled" in app.status_bar.currentMessage()
    _assert_run_dropped(app, run_a)
    _assert_nothing_written(app, a, b)


# ── Follow-up: a dropped GCxGC run (or a failed load) leaves no 1D sample to integrate ──

def _make_1d_sample(folder):
    """A 1D chromatogram as a CSV-backed .C (FTIR profile, read by the real CSV loader)."""
    import numpy as np
    from logic.c_folder import CFolder
    x = np.linspace(0, 10, 4000)
    y = sum(h * np.exp(-((x - c) / 0.05) ** 2)
            for c, h in ((2, 5e5), (4, 8e5), (6, 3e5), (8, 6e5))) + 10
    folder.mkdir(parents=True, exist_ok=True)
    csv = folder / "one_d.csv"
    csv.write_text("x,y\n" + "\n".join(f"{a},{b}" for a, b in zip(x, y)))
    return Path(CFolder.create(str(csv), "ftir").path)


def _load_1d_sample(app, folder, monkeypatch):
    """Load a 1D sample through the real path with peak detection on and auto-export on."""
    monkeypatch.setattr(app.export_manager, "should_export_json", lambda trigger: True)
    monkeypatch.setattr(app.export_manager, "should_export_csv", lambda trigger: True)
    monkeypatch.setattr("ui.app.QMessageBox.critical", lambda *a, **k: pytest.fail(f"load failed: {a[2]}"))
    c = _make_1d_sample(folder)
    app.on_file_selected(str(c))
    assert not app._gcxgc_mode and app.current_processed is not None
    app.parameters_frame.peaks_enabled.setChecked(True)
    return c


@pytest.mark.parametrize("how", ["failed", "cancelled"])
def test_dropped_gcxgc_run_leaves_no_1d_sample_to_integrate(qtbot, tmp_path, monkeypatch, how):
    """1D sample X, GCxGC run A, then a load of B that fails or is cancelled: Integrate must not
    process X's stale peaks and export them into A's (or B's) folder."""
    app = ChromaKitApp()
    qtbot.addWidget(app)
    x = _load_1d_sample(app, tmp_path / "x", monkeypatch)
    monkeypatch.setattr("ui.app.QMessageBox.critical", lambda *a, **k: None)     # B's failure dialog
    monkeypatch.setattr("ui.app.QMessageBox.warning", lambda *a, **k: None)
    a = synth.build_run(tmp_path / "a", _scene())
    app.on_file_selected(str(a))
    assert app._gcxgc_mode
    if how == "failed":
        b = synth.build_run(tmp_path / "b", _scene())
        lsc = b / "data" / "Synth.lsc"
        lsc.write_bytes(lsc.read_bytes()[: lsc.stat().st_size // 2])      # a damaged copy
    else:
        b = synth.build_run(tmp_path / "b", _scene(), insight=False, view=False)     # no period anywhere
        monkeypatch.setattr("PySide6.QtWidgets.QInputDialog.getDouble", lambda *a, **k: (4.0, False))
    app.on_file_selected(str(b))
    assert app._gcxgc_run is None and not app._gcxgc_mode
    for folder in (a, b, x):
        _clear_results(folder)
    app.button_frame.integrate_button.click()
    assert "No data available for integration" in app.status_bar.currentMessage()
    assert app.current_processed is None
    for folder in (a, b, x):
        assert list((folder / "results").iterdir()) == [], folder


def test_failed_1d_load_leaves_no_stale_sample_to_integrate(qtbot, tmp_path, monkeypatch):
    """A 1D load that fails has already discarded the previous sample's x/y; Integrate must not
    still run on that sample's processed data."""
    app = ChromaKitApp()
    qtbot.addWidget(app)
    x = _load_1d_sample(app, tmp_path / "x", monkeypatch)
    monkeypatch.setattr("ui.app.QMessageBox.critical", lambda *a, **k: None)     # the bad file's dialog
    bad = _make_1d_sample(tmp_path / "bad")
    (bad / "data" / "one_d.csv").write_text("not,a\nchromatogram\n")     # no x / y columns
    app.on_file_selected(str(bad))
    assert "Error loading file" in app.status_bar.currentMessage()
    _clear_results(x)
    app.button_frame.integrate_button.click()
    assert "No data available for integration" in app.status_bar.currentMessage()
    assert list((x / "results").iterdir()) == [] and list((bad / "results").iterdir()) == []


# ── Follow-up: a 1D sample loaded after a GCxGC run gets the 1D plot and its clicks back ──

def test_loading_a_1d_sample_after_a_gcxgc_run_returns_clicks_to_the_1d_plot(gcxgc_app, tmp_path, monkeypatch):
    """The GCxGC click handler must not swallow every click once a 1D sample is on screen."""
    from types import SimpleNamespace
    app, _ = gcxgc_app
    pf = app.plot_frame
    assert hasattr(pf, "_gcxgc_ax_fid") and hasattr(pf, "_gcxgc_ax_tic")
    _load_1d_sample(app, tmp_path / "x", monkeypatch)             # fails the test if the load errors
    assert not hasattr(pf, "_gcxgc_ax_fid") and not hasattr(pf, "_gcxgc_ax_tic")
    assert pf._gcxgc_peaks == []
    assert pf.chromatogram_ax in pf.figure.axes and len(pf.chromatogram_ax.lines) > 0   # the 1D plot survived
    app.button_frame.integrate_button.click()
    assert pf.integrated_peaks
    gcxgc_clicks, menus = [], []
    monkeypatch.setattr(pf, "_on_gcxgc_click", lambda event: gcxgc_clicks.append(event))
    monkeypatch.setattr(pf, "_show_peak_context_menu", lambda i, event: menus.append(i))
    peak = pf.integrated_peaks[0]
    rt = getattr(peak, "retention_time", None) or peak.position
    pf._on_plot_click(SimpleNamespace(inaxes=pf.chromatogram_ax, button=3, xdata=rt, ydata=1.0))
    assert gcxgc_clicks == [] and menus == [0]                    # a right-click on a peak opens its menu
    pf._on_plot_click(SimpleNamespace(inaxes=pf.chromatogram_ax, button=1, xdata=rt, ydata=1.0))
    assert gcxgc_clicks == [] and pf._last_click_x == rt           # a left-click is recorded for point selection


# ── Final review, Minors 4, 5, 7 and 10 ─────────────────────────────────────

def _toolkit(*hits):
    class FakeToolkit:
        library = {}
        mz_shift = 0

        def search_vector(self, spectrum, top_n=1, **kw):
            return list(hits)[:top_n]

        search_w2v = search_hybrid = search_vector

    return FakeToolkit()


def test_clicking_a_searched_peak_shows_its_hits(gcxgc_app):
    app, _ = gcxgc_app
    app.button_frame.integrate_button.click()
    _fake_search(app)
    app.ms_frame.ms_toolkit = _toolkit(("Benzene", 0.91), ("Toluene", 0.80))
    app.ms_frame.search_options = {"search_method": "vector", "top_n": 2}
    app.run_batch_ms_search()
    first, second = app._gcxgc_peaks[:2]
    app.plot_frame.peak_selected.emit(first)
    assert "Benzene (0.910)" in app.status_bar.currentMessage()
    tree = app.ms_frame.ms_search_tree
    assert [(tree.topLevelItem(i).text(0), tree.topLevelItem(i).text(1))
            for i in range(tree.topLevelItemCount())] == [("Benzene", "0.910"), ("Toluene", "0.800")]
    second.top_hits, second.compound_name, second.match_score = [], None, None
    app.plot_frame.peak_selected.emit(second)               # no hits: the previous peak's table goes
    assert tree.topLevelItemCount() == 0 and "Benzene" not in app.status_bar.currentMessage()


def _cancel_after(monkeypatch, n_peaks):
    """Make MS Search All's progress dialog report Cancel once n_peaks peaks were searched."""
    calls = {"n": 0}

    def was_canceled(self):
        calls["n"] += 1
        return calls["n"] > n_peaks

    monkeypatch.setattr("PySide6.QtWidgets.QProgressDialog.wasCanceled", was_canceled)


def test_ms_search_all_starts_from_no_ids_and_marks_a_cancelled_search(gcxgc_app, monkeypatch):
    app, c = gcxgc_app
    monkeypatch.setattr(app.export_manager, "should_export_json", lambda trigger: False)
    app.button_frame.integrate_button.click()
    _fake_search(app)
    app.run_batch_ms_search()                                         # search 1: complete
    assert all(p.compound_name == "Benzene" for p in app._gcxgc_peaks)
    assert app._gcxgc_search_info["complete"] is True
    app.ms_frame.ms_toolkit = _toolkit(("Toluene", 0.7))
    app.ms_frame.search_options = {"search_method": "w2v", "top_n": 1}
    _cancel_after(monkeypatch, 1)
    app.run_batch_ms_search()                                         # search 2: cancelled after peak 1
    first, *rest = app._gcxgc_peaks
    assert (first.compound_name, first.top_hits[0][0]) == ("Toluene", "Toluene")
    for p in rest:                                                    # no leftovers from search 1
        assert (p.compound_name, p.match_score, p.casno, p.top_hits) == (None, None, None, [])
    assert app._gcxgc_search_info == {"library": "NIST14.json", "search_method": "w2v", "complete": False}
    assert "cancelled" in app.status_bar.currentMessage()
    _clear_results(c)
    app.button_frame.export_button.click()
    rec = _record(c)
    assert (rec["search_method"], rec["search_complete"]) == ("w2v", False)


def test_spectrum_window_cannot_be_zero(gcxgc_app):
    import pydantic
    from logic.method import GCxGCParams
    app, _ = gcxgc_app
    assert app.parameters_frame.gcxgc_apex_window_spin.minimum() == 1
    with pytest.raises(pydantic.ValidationError):
        GCxGCParams(apex_window_scans=0)


def test_batch_process_directories_is_refused_in_gcxgc_mode(gcxgc_app, monkeypatch):
    from PySide6.QtGui import QAction
    app, _ = gcxgc_app
    shown, opened = [], []
    monkeypatch.setattr("ui.app.QMessageBox.information", lambda *a, **k: shown.append(a))
    monkeypatch.setattr("ui.dialogs.batch_job_dialog.BatchJobDialog.exec", lambda self: opened.append(True))
    action = next(a for a in app.findChildren(QAction) if a.text() == "Batch Process Directories...")
    action.trigger()
    assert shown and "not supported" in str(shown[0][2]) and opened == []


def test_a_load_that_fails_after_reading_closes_the_new_run(gcxgc_app, monkeypatch):
    from logic.gcxgc_run import GCxGCRun
    app, c = gcxgc_app
    run_a = app._gcxgc_run
    monkeypatch.setattr("ui.app.QMessageBox.critical", lambda *a, **k: None)
    made, real = [], GCxGCRun.from_c_folder.__func__

    def recording(cls, *a, **k):
        made.append(real(cls, *a, **k))
        return made[-1]

    def boom(profile):
        raise RuntimeError("boom")

    monkeypatch.setattr(GCxGCRun, "from_c_folder", classmethod(recording))
    monkeypatch.setattr(app, "set_mode", boom)             # fails after the run's files are open
    app.on_file_selected(str(synth.build_run(c.parent / "b", synth.Scene(duration_min=2.0), name="SynthB")))
    assert "boom" in app.status_bar.currentMessage()
    _assert_run_dropped(app, run_a)
    assert len(made) == 1 and _run_files_closed(made[0])


def test_export_min_snr_reaches_the_method_and_the_export(gcxgc_app, monkeypatch):
    app, _ = gcxgc_app
    pf = app.parameters_frame
    assert pf.gcxgc_export_snr_spin.value() == 5.0
    pf.gcxgc_export_snr_spin.setValue(2.5)
    assert app.current_method.gcxgc.export_min_snr == 2.5
    app.button_frame.integrate_button.click()
    seen = {}
    import logic.gcxgc_export as ge
    real = ge.export_gcxgc_results

    def spy(*a, **k):
        seen["min_snr"] = k.get("min_snr")
        return real(*a, **k)

    monkeypatch.setattr(ge, "export_gcxgc_results", spy)
    app._export_gcxgc(interactive=True)
    assert seen["min_snr"] == 2.5
    assert "GNPS quant.csv" in app.status_bar.currentMessage()
