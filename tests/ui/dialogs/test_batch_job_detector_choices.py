"""BatchJobDialog: each detector row offers the channels of its own instrument's folders."""
import json
import os

import pytest

pytest.importorskip('pytestqt')

from PySide6.QtWidgets import QFileDialog

from logic.c_folder import CFolder
from ui.dialogs import batch_job_dialog
from ui.dialogs.batch_job_dialog import BatchJobDialog

CHANNELS = {"GC1": ["FID1A", "TCD3C"], "GC2": ["FID1A", "FID2B"]}


@pytest.fixture
def reads(monkeypatch):
    """Fake the sample reader (no Agilent data in the repo); returns the paths it was asked for."""
    calls = []

    def channels(self):
        calls.append(self.path)
        parent = os.path.basename(os.path.dirname(str(self.path)))
        return next((list(v) for k, v in CHANNELS.items() if k in parent), [])

    monkeypatch.setattr(CFolder, "get_available_detectors", channels)
    return calls


@pytest.fixture
def dialog(qtbot, monkeypatch, reads):
    class NoSettings:
        def value(self, *a, **k):
            return None

        def setValue(self, *a):
            pass

    monkeypatch.setattr(batch_job_dialog, "_settings", NoSettings)
    d = BatchJobDialog()
    qtbot.addWidget(d)
    return d


def _folder(tmp_path, name):
    c = tmp_path / name / "S001.C"
    c.mkdir(parents=True)
    (c / "manifest.json").write_text(json.dumps({"signal_type": "gc", "source_format": "agilent_d"}))
    return str(tmp_path / name)


def _add(dialog, monkeypatch, path):
    """Add a folder through the dialog's own handler.

    Folders are auto-tagged by path, and tmp_path holds the test name: keep "gc1"/"gc2" out of it.
    """
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: path)
    dialog.add_directory()


def _tag(dialog, row, instrument):
    combo = dialog.instrument_combos[row]
    combo.setCurrentIndex(combo.findData(instrument))


def _offered(dialog, row):
    combo = dialog.detector_rows[row][0]
    return [combo.itemText(i) for i in range(combo.count())]


def _text(dialog, row):
    return dialog.detector_rows[row][0].currentText()


def test_each_row_offers_its_own_instruments_channels(dialog, tmp_path, monkeypatch):
    _add(dialog, monkeypatch, _folder(tmp_path, "GC1"))
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    _tag(dialog, 0, "GC1")
    _tag(dialog, 2, "GC2")
    assert _offered(dialog, 0) == ["FID1A", "TCD3C"]
    assert _offered(dialog, 2) == ["FID1A", "FID2B"]


def test_rows_tagged_before_folders_are_added_get_their_channels(dialog, tmp_path, monkeypatch):
    _tag(dialog, 2, "GC2")
    _add(dialog, monkeypatch, _folder(tmp_path, "GC1"))
    assert _offered(dialog, 2) == []
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    assert _offered(dialog, 2) == ["FID1A", "FID2B"]


def test_retagging_a_folder_updates_the_rows(dialog, tmp_path, monkeypatch):
    plain = _folder(tmp_path, "GC1 GC2 mixed name")  # names both, so it is not auto-tagged
    _add(dialog, monkeypatch, plain)
    _tag(dialog, 2, "GC1")
    assert _offered(dialog, 2) == []
    dialog.set_directory_instrument(plain, "GC1")
    assert _offered(dialog, 2) == ["FID1A", "TCD3C"]


def test_channel_the_instrument_lacks_is_cleared_when_the_row_is_tagged(dialog, tmp_path, monkeypatch):
    # Adding the GC1 folder fills row 2 with TCD3C; tagging that row GC2 must not keep it.
    _add(dialog, monkeypatch, _folder(tmp_path, "GC1"))
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    assert _text(dialog, 1) == "TCD3C"
    _tag(dialog, 1, "GC2")
    assert _text(dialog, 1) == ""


def test_channel_the_instrument_has_is_kept_when_the_row_is_tagged(dialog, tmp_path, monkeypatch):
    _add(dialog, monkeypatch, _folder(tmp_path, "GC1"))
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    assert _text(dialog, 0) == "FID1A"
    _tag(dialog, 0, "GC2")
    assert _text(dialog, 0) == "FID1A"


def test_typed_channel_is_kept_when_the_row_is_tagged(dialog, tmp_path, monkeypatch):
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    dialog.detector_rows[1][0].setCurrentText("AUX9Z")
    _tag(dialog, 1, "GC2")
    assert _text(dialog, 1) == "AUX9Z"


def test_loaded_setup_is_not_changed_by_adding_folders(dialog, tmp_path, monkeypatch):
    rows = [{"instrument": "GC1", "detector": "FID1A", "method_path": None},
            {"instrument": "GC1", "detector": "TCD3C", "method_path": None},
            {"instrument": "GC2", "detector": "FID1A", "method_path": None},
            {"instrument": "GC2", "detector": "FID2B", "method_path": None}]
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps({"version": 1, "rows": rows}))
    dialog.load_setup(str(setup))
    for name in ("GC1", "GC2"):
        _add(dialog, monkeypatch, _folder(tmp_path, name))
    assert dialog.get_setup_rows() == rows


def test_empty_tagged_row_is_not_filled_in(dialog, tmp_path, monkeypatch):
    # A blank detector means "use the method's own detector"; a guess would override it.
    _tag(dialog, 0, "GC2")
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    assert _text(dialog, 0) == ""


def test_untagged_rows_still_use_the_first_folder(dialog, tmp_path, monkeypatch):
    _add(dialog, monkeypatch, _folder(tmp_path, "GC1"))
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    assert _offered(dialog, 3) == ["FID1A", "TCD3C"]
    assert [_text(dialog, i) for i in range(4)] == ["FID1A", "TCD3C", "", ""]


def test_each_folder_is_read_once(dialog, tmp_path, monkeypatch, reads):
    _add(dialog, monkeypatch, _folder(tmp_path, "GC1"))
    _add(dialog, monkeypatch, _folder(tmp_path, "GC2"))
    for i, tag in enumerate(("GC1", "GC1", "GC2", "GC2")):
        _tag(dialog, i, tag)
    assert len(reads) == len(set(reads)) == 2


def test_removed_folder_is_read_again_when_re_added(dialog, tmp_path, monkeypatch):
    _tag(dialog, 0, "GC1")
    path = _folder(tmp_path, "GC1")
    _add(dialog, monkeypatch, path)
    assert _offered(dialog, 0) == ["FID1A", "TCD3C"]
    dialog.directory_list.item(0).setSelected(True)
    dialog.remove_selected_directories()
    assert _offered(dialog, 0) == []
    monkeypatch.setitem(CHANNELS, "GC1", ["FID1A"])
    _add(dialog, monkeypatch, path)
    assert _offered(dialog, 0) == ["FID1A"]
