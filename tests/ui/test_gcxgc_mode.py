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


def test_load_prefills_layout_from_file(gcxgc_app):
    app, _ = gcxgc_app
    assert app.parameters_frame.gcxgc_pm_entry.text() == "4"
    assert app.parameters_frame.gcxgc_phase_entry.text() == "-1.8"


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
