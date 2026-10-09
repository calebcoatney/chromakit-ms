import json

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


def test_export_button_writes_three_files(gcxgc_app):
    app, c = gcxgc_app
    app.button_frame.integrate_button.click()
    for f in (c / "results").glob("*"):
        f.unlink()                                    # clear any auto-export
    app.button_frame.export_button.click()
    assert sorted(f.name for f in (c / "results").iterdir()) == [
        "Synth - GCxGC peaks.csv", "Synth - GCxGC peaks.json", "Synth - GCxGC spectra.msp"]


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
    assert rec["search_method"] is None and rec["library"] is None

    _fake_search(app)
    app.run_batch_ms_search()
    _clear_results(c)
    app.button_frame.export_button.click()
    rec = _record(c)
    assert rec["search_method"] == "vector" and rec["library"] == "NIST14.json"

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
    app.on_file_selected(str(b))
    assert app._gcxgc_run.sample_id == "SynthB"
    assert app._gcxgc_peaks == [] and app._gcxgc_proc is None
    assert app.integrated_peaks == [] and app.plot_frame._gcxgc_peaks == []
    assert app.view_results_button.isHidden()


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


def test_unavailable_detection_signal_is_reported_not_raised(qtbot, tmp_path):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.on_file_selected(str(synth.build_run(tmp_path, _scene(), dbc=False)))
    pf = app.parameters_frame
    combo = pf.gcxgc_signal_combo
    assert not combo.model().item(combo.findData("dbc_tic")).isEnabled()
    combo.setCurrentIndex(combo.findData("dbc_tic"))       # only reachable programmatically
    app.button_frame.integrate_button.click()
    assert app._gcxgc_peaks == [] and app._gcxgc_proc is None
    assert "processing error" in app.status_bar.currentMessage()
    assert combo.currentData() == "fid"                     # the run info put the combo back
