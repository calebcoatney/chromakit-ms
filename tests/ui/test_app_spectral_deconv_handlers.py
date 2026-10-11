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
