"""Converter auto-selects the saved format preset that matches the JSONs' detector."""
import json

import pytest
pytest.importorskip('pytestqt')

from PySide6.QtCore import QSettings

import util.json_to_xlsx as jx


@pytest.fixture
def make_dialog(qtbot, tmp_path, monkeypatch):
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        jx, "QSettings", lambda *a, **k: QSettings(ini, QSettings.IniFormat))

    def make(presets, initial_directory=None):
        QSettings(ini, QSettings.IniFormat).setValue(
            "xlsx_format_presets", json.dumps(presets))
        d = jx.JsonToExcelConverter(initial_directory=initial_directory)
        qtbot.addWidget(d)
        return d
    return make


def _results(tmp_path, *names):
    folder = tmp_path / "S1.C" / "results"
    folder.mkdir(parents=True)
    for i, det in enumerate(names):
        (folder / f"S1 - {det}.json").write_text(json.dumps({"detector": det}))
    return str(tmp_path)


def _preset(show_method):
    cfg = json.loads(json.dumps(jx.DEFAULT_FORMAT_CONFIG))
    for hf in cfg['header_fields']:
        if hf['key'] == 'method':
            hf['enabled'] = show_method
    return cfg


PRESETS = {"Gas FID1A": _preset(True), "Gas FID2B": _preset(False)}


def test_matching_preset_is_selected_and_applied(make_dialog, tmp_path):
    d = make_dialog(PRESETS, _results(tmp_path, "FID2B"))
    assert d.preset_combo.currentText() == "Gas FID2B"
    assert not any(h['enabled'] for h in d.get_current_format_config()['header_fields']
                   if h['key'] == 'method')


def test_match_is_case_insensitive_and_uses_signal_fallback(make_dialog, tmp_path):
    folder = tmp_path / "S1.C" / "results"
    folder.mkdir(parents=True)
    (folder / "a.json").write_text(json.dumps({"signal": "Signal: S1\\fid1a.ch"}))
    d = make_dialog(PRESETS, str(tmp_path))
    assert d.preset_combo.currentText() == "Gas FID1A"


def test_no_match_leaves_selection(make_dialog, tmp_path):
    d = make_dialog(PRESETS, _results(tmp_path, "TCD3C"))
    assert d.preset_combo.currentText() == "Gas FID1A"  # untouched first item
    assert d._preset_chosen_by_user is False


def test_mixed_detectors_are_ambiguous(make_dialog, tmp_path):
    d = make_dialog(PRESETS, _results(tmp_path, "FID1A", "FID2B"))
    assert d.preset_combo.currentText() == "Gas FID1A"
    assert not any("selected for detector" in line
                   for line in d.log_text.toPlainText().splitlines())


def test_user_choice_is_not_overridden(make_dialog, tmp_path):
    d = make_dialog(PRESETS, None)
    d.preset_combo.activated.emit(0)  # user picks "Gas FID1A"
    d._auto_select_preset_for_directory(_results(tmp_path, "FID2B"))
    assert d.preset_combo.currentText() == "Gas FID1A"


def test_programmatic_selection_is_not_manual(make_dialog, tmp_path):
    d = make_dialog(PRESETS, _results(tmp_path, "FID2B"))
    assert d._preset_chosen_by_user is False
