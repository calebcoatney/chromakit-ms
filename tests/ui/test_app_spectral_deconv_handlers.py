"""ChromaKitApp's spectral deconvolution handlers, run on a stand-in ``self``.

None of these build ChromaKitApp: its constructor reads Caleb's real QSettings.
The handlers are called unbound on a SimpleNamespace that carries only what
each one touches. The one test that needs a real Inspector builds it on a temp
INI, and sidecar offset writes are patched out.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox

from ui.app import ChromaKitApp
from tests.ui.dialogs.test_spectral_deconv_inspector import _fake_rainbow, _peak

TRACEBACK = ('Traceback (most recent call last):\n  File "x.py", line 1, in run\n'
             'ValueError: boom\n')


def _pk(deconvolved):
    return SimpleNamespace(deconvolved_spectrum={'mz': [1.0], 'intensities': [1.0]}
                           if deconvolved else None)


def _fake_app(peaks=(), inspector=None):
    return SimpleNamespace(
        integrated_peaks=list(peaks), status_bar=MagicMock(), ms_frame=MagicMock(),
        _refresh_current_peak_ms_display=MagicMock(), _deconv_inspector=inspector,
        _deconvolve_ms_run=False)


# ── #4: one message per outcome; a cancelled run is not "complete" ─────────────

@pytest.mark.parametrize("flags, cancelled, message", [
    ([True, False, False], True,
     "Spectral deconvolution cancelled: 1 of 3 peaks have a deconvolved spectrum; the rest use "
     "the fallback spectrum. Click Deconvolve MS to run the whole file."),
    ([True, True, True], False,
     "Spectral deconvolution complete: all 3 peaks have a deconvolved spectrum. Click MS Search "
     "All to search with them."),
    ([True, False, False], False,
     "Spectral deconvolution complete: 1 of 3 peaks have a deconvolved spectrum; the other 2 use "
     "the fallback spectrum (MS Options → Spectrum Extraction). Click MS Search All to search "
     "with them."),
    ([False, False, False], False,
     "Spectral deconvolution complete: no peak got a deconvolved spectrum, so all 3 use the "
     "fallback spectrum. Click Inspect on the MS tab to check the settings."),
])
def test_finished_message_per_outcome(flags, cancelled, message):
    app = _fake_app([_pk(f) for f in flags])
    ChromaKitApp._on_deconvolve_ms_finished(app, None, cancelled=cancelled)
    app.status_bar.showMessage.assert_called_once_with(message)


def _dispatch(monkeypatch, start):
    """Run a dispatcher with the progress dialog and thread pool stubbed; return its worker."""
    import ui.app as app_mod
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    started = []
    monkeypatch.setattr(app_mod, "QProgressDialog", MagicMock())
    monkeypatch.setattr(app_mod, "QThreadPool", SimpleNamespace(
        globalInstance=lambda: SimpleNamespace(start=started.append)))
    app = _fake_app([_peak(1.0, 0.96, 1.04)], inspector=MagicMock())
    app.data_handler = SimpleNamespace(current_directory_path="/x.D", ms_time_offset=0.0)
    app._get_ms_data_path = lambda: "/x.D"
    app._get_spectral_deconv_params = lambda: (CorrelationDeconvParams(), WindowGroupingParams())
    app._on_deconvolve_ms_finished = MagicMock()
    app._on_deconvolve_ms_error = MagicMock()
    start(app)
    assert len(started) == 1
    return app, started[0]


@pytest.mark.parametrize("start", [
    lambda app: ChromaKitApp._on_deconvolve_ms_clicked(app),
    lambda app: ChromaKitApp._on_inspector_rerun_requested(app, *app._get_spectral_deconv_params()),
], ids=["deconvolve-ms-button", "inspector-rerun"])
@pytest.mark.parametrize("cancel", [True, False])
def test_both_call_sites_say_whether_the_run_was_cancelled(monkeypatch, start, cancel):
    app, worker = _dispatch(monkeypatch, start)
    worker.cancelled = cancel
    worker.signals.finished.emit()
    assert app._on_deconvolve_ms_finished.call_count == 1
    assert app._on_deconvolve_ms_finished.call_args.kwargs == {"cancelled": cancel}


# ── #15: a failed rerun re-enables the Inspector; no raw traceback ─────────────

def test_failed_rerun_reenables_the_inspector_and_shows_a_plain_message(monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda parent, title, text, *a, **k: shown.append((title, text))))
    inspector = MagicMock()
    inspector.isVisible.return_value = True
    app = _fake_app(inspector=inspector)
    ChromaKitApp._on_deconvolve_ms_error(app, TRACEBACK, MagicMock())
    inspector.set_controls_enabled.assert_called_with(True)
    assert len(shown) == 1
    title, text = shown[0]
    assert "ValueError: boom" in text and "Traceback" not in text and 'File "x.py"' not in text
    assert "Traceback" in capsys.readouterr().out


# ── #16: no library: the Inspector says so instead of "searching…" ─────────────

@pytest.mark.parametrize("toolkit, loaded", [(None, False), (object(), False)])
def test_cluster_search_without_a_library_sets_the_inspector_status(monkeypatch, toolkit, loaded):
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    inspector = MagicMock()
    app = _fake_app(inspector=inspector)
    app.ms_frame = SimpleNamespace(ms_toolkit=toolkit, library_loaded=loaded)
    ChromaKitApp._on_cluster_search_requested(app, {57.0: 100.0}, 1.0)
    inspector.show_status.assert_called_once_with("Load an MS library on the MS tab to search groups.")


# ── #12: Save Offset saves the shown settings, and reruns exactly those ────────

@pytest.fixture
def inspector(qtbot, tmp_path, monkeypatch):
    _fake_rainbow(monkeypatch)
    import ui.dialogs.spectral_deconv_inspector as mod
    path = str(tmp_path / "t.ini")
    monkeypatch.setattr(mod, "QSettings", lambda *a, **k: QSettings(path, QSettings.IniFormat))
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    dlg = mod.SpectralDeconvInspectorDialog(
        peaks=[_peak(1.0, 0.96, 1.04)], ms_data_path="/fake.D",
        deconv_params=CorrelationDeconvParams(search_range=0.03),
        grouping_params=WindowGroupingParams(gap_tolerance=0.1))
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: dlg._last_result is not None, timeout=10000)
    dlg._ini_path = path
    return dlg


@pytest.fixture
def offset_app(inspector, monkeypatch):
    import logic.sidecar_offsets as so
    offsets = []
    monkeypatch.setattr(so, "save_offset", lambda path, off, **k: offsets.append((path, off)))
    monkeypatch.setattr(so, "save_offsets_batch", lambda **k: offsets.extend(
        (p, k["offset_min"]) for p in k["data_paths"]))
    monkeypatch.setattr(so, "load_offsets_for_paths", lambda paths, **k: {})
    app = _fake_app(inspector=inspector)
    app.data_handler = SimpleNamespace(current_directory_path="/a.D", ms_time_offset=0.0,
                                       available_directories=["/a.D", "/b.D"])
    app.plot_frame = MagicMock()
    app._on_inspector_rerun_requested = MagicMock()
    app._on_inspector_apply_offset_requested = (
        lambda off, src: ChromaKitApp._on_inspector_apply_offset_requested(app, off, src))
    app._offsets = offsets
    return app


def _saved(inspector):
    from logic.spectral_deconv_settings import load_spectral_deconv_settings
    s = QSettings(inspector._ini_path, QSettings.IniFormat)
    return load_spectral_deconv_settings(s) if s.allKeys() else None


def test_save_offset_saves_the_shown_settings_and_reruns_them(offset_app, inspector):
    inspector._min_r_spin.setValue(0.9)
    inspector._excluded_mz_edit.setText("73 147")
    ChromaKitApp._on_inspector_apply_offset_requested(offset_app, 0.01, "manual")
    assert offset_app._offsets == [("/a.D", 0.01)]
    saved = _saved(inspector)
    assert saved is not None and saved[0].min_r == pytest.approx(0.9)
    assert saved[0].excluded_mz == [73.0, 147.0]
    offset_app._on_inspector_rerun_requested.assert_called_once()
    assert tuple(offset_app._on_inspector_rerun_requested.call_args.args) == saved


def _fake_batch_dialog(monkeypatch, accepted):
    import ui.dialogs.batch_offset_dialog as bod
    from PySide6.QtWidgets import QDialog

    class FakeBatch:
        def __init__(self, siblings, **k):
            self._siblings = siblings

        def exec(self):
            return QDialog.Accepted if accepted else QDialog.Rejected

        def selected_paths(self):
            return list(self._siblings)
    monkeypatch.setattr(bod, "BatchOffsetDialog", FakeBatch)


def test_save_offset_to_folder_saves_settings_only_once_accepted(offset_app, inspector, monkeypatch):
    inspector._min_r_spin.setValue(0.9)
    _fake_batch_dialog(monkeypatch, accepted=False)
    ChromaKitApp._on_inspector_apply_offset_to_folder_requested(offset_app, 0.01, "manual")
    assert _saved(inspector) is None and offset_app._offsets == []
    offset_app._on_inspector_rerun_requested.assert_not_called()

    _fake_batch_dialog(monkeypatch, accepted=True)
    ChromaKitApp._on_inspector_apply_offset_to_folder_requested(offset_app, 0.01, "manual")
    saved = _saved(inspector)
    assert saved is not None and saved[0].min_r == pytest.approx(0.9)
    assert tuple(offset_app._on_inspector_rerun_requested.call_args.args) == saved


def test_no_siblings_message_names_save_offset(offset_app, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda parent, title, text, *a, **k: shown.append(text)))
    offset_app.data_handler.available_directories = ["/a.D"]
    ChromaKitApp._on_inspector_apply_offset_to_folder_requested(offset_app, 0.01, "manual")
    assert shown and "Save Offset" in shown[0] and "'Apply'" not in shown[0]


# ── #5, #21: tooltips say what the option does ─────────────────────────────────

def test_batch_deconvolution_tooltip(qtbot):
    from ui.dialogs.batch_job_dialog import BatchJobDialog
    dlg = BatchJobDialog()
    qtbot.addWidget(dlg)
    assert dlg.spectral_deconv_check.toolTip() == (
        "<p>Before MS library search, give each FID peak a cleaner mass spectrum built from the "
        "ions that elute with it, using the settings in MS Options → Spectral Deconvolution. "
        "Compounds closer than about 1.5 peak widths usually still come out as one spectrum. "
        "Ignored for non-GC-MS data.</p>")


def test_deconvolve_ms_button_tooltip(qtbot):
    from ui.frames.buttons import ButtonFrame
    frame = ButtonFrame()
    qtbot.addWidget(frame)
    assert frame.deconvolve_ms_button.toolTip() == (
        "<p>Give each integrated FID peak a cleaner mass spectrum, built from the ions that "
        "elute with it. Settings: MS Options → Spectral Deconvolution.</p>")
