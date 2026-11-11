"""BatchJobDialog: detector/method rows become options['detector_runs']."""
import json

import pytest

pytest.importorskip('pytestqt')

from PySide6.QtWidgets import QWidget

from logic.method import ChromaMethod
from ui.dialogs import batch_job_dialog
from ui.dialogs.batch_job_dialog import BatchJobDialog


@pytest.fixture
def dialog(qtbot):
    d = BatchJobDialog()
    qtbot.addWidget(d)
    return d


def _fill(dialog, row, detector, method_path=""):
    combo, edit = dialog.detector_rows[row]
    combo.setCurrentText(detector)
    edit.setText(method_path)


def test_unchecked_gives_no_detector_runs(dialog):
    _fill(dialog, 0, "FID1A")
    assert dialog.get_detector_runs() == []


def test_two_detectors_with_methods(dialog, tmp_path):
    m1, m2 = tmp_path / "a.chromethod", tmp_path / "b.chromethod"
    ChromaMethod(name="a", signal_type="gc").to_file(m1)
    ChromaMethod(name="b", signal_type="gc").to_file(m2)
    dialog.per_detector_check.setChecked(True)
    _fill(dialog, 0, "FID1A", str(m1))
    _fill(dialog, 1, "FID2B", str(m2))
    assert dialog.get_detector_runs() == [
        {'instrument': '', 'detector': 'FID1A', 'method_path': str(m1)},
        {'instrument': '', 'detector': 'FID2B', 'method_path': str(m2)},
    ]


def test_blank_second_row_is_ignored(dialog):
    dialog.per_detector_check.setChecked(True)
    _fill(dialog, 0, "FID2B")
    _fill(dialog, 1, "")
    assert dialog.get_detector_runs() == [{'instrument': '', 'detector': 'FID2B', 'method_path': None}]


def test_detector_taken_from_method_when_blank(dialog, tmp_path):
    m = tmp_path / "b.chromethod"
    ChromaMethod(name="b", signal_type="gc", detector="FID2B").to_file(m)
    dialog.per_detector_check.setChecked(True)
    _fill(dialog, 0, "", str(m))
    _fill(dialog, 1, "")
    assert dialog.get_detector_runs() == [{'instrument': '', 'detector': 'FID2B', 'method_path': str(m)}]


@pytest.mark.parametrize("rows", [
    [("FID1A", "missing.chromethod"), ("", "")],
    [("FID1A", ""), ("FID1A", "")],
    [("", ""), ("", "")],
])
def test_invalid_rows_raise(dialog, rows):
    dialog.per_detector_check.setChecked(True)
    for i, (det, path) in enumerate(rows):
        _fill(dialog, i, det, path)
    with pytest.raises(ValueError):
        dialog.get_detector_runs()


def _dialog_with_main(qtbot, path, detector):
    main = QWidget()
    qtbot.addWidget(main)
    main.current_method_path = path
    main.current_method = ChromaMethod(name="m", signal_type="gc", detector=detector)
    d = BatchJobDialog(main)
    d._main = main  # parent must outlive the dialog's use
    return d


def test_first_row_prefilled_from_saved_method(qtbot, tmp_path):
    p = tmp_path / "m.chromethod"
    d = _dialog_with_main(qtbot, p, "FID2B")
    combo, edit = d.detector_rows[0]
    assert combo.currentText() == "FID2B"
    assert edit.text() == str(p)
    assert not d.per_detector_check.isChecked()
    assert d.detector_rows[1][0].currentText() == ""


def test_no_prefill_without_saved_path_or_detector(qtbot, tmp_path):
    for path, det in [(None, "FID2B"), (tmp_path / "m.chromethod", None)]:
        d = _dialog_with_main(qtbot, path, det)
        assert d.detector_rows[0][0].currentText() == ""
        assert d.detector_rows[0][1].text() == ""


# ---- instrument tags, four rows, setup file, workbook ----

def _row(dialog, i, instrument, detector, method_path=""):
    combo = dialog.instrument_combos[i]
    combo.setCurrentIndex(combo.findData(instrument))
    _fill(dialog, i, detector, method_path)


def _folder(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return str(d)


def _add(dialog, path, tag=""):
    dialog._append_directory(path)
    dialog.set_directory_instrument(path, tag)


def test_four_rows(dialog):
    assert batch_job_dialog.MAX_DETECTOR_RUNS == 4
    assert len(dialog.detector_rows) == len(dialog.instrument_combos) == 4
    assert [dialog.instrument_combos[0].itemData(i) for i in range(3)] == ["", "GC1", "GC2"]


def test_fid1a_on_both_instruments_is_valid(dialog, tmp_path):
    _add(dialog, _folder(tmp_path, "a"), "GC1")
    _add(dialog, _folder(tmp_path, "b"), "GC2")
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    _row(dialog, 1, "GC2", "FID1A")
    assert dialog.get_detector_runs() == [
        {'instrument': 'GC1', 'detector': 'FID1A', 'method_path': None},
        {'instrument': 'GC2', 'detector': 'FID1A', 'method_path': None},
    ]


def test_duplicate_instrument_detector_raises(dialog, tmp_path):
    _add(dialog, _folder(tmp_path, "a"), "GC1")
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    _row(dialog, 1, "GC1", "FID1A")
    with pytest.raises(ValueError, match="more than once"):
        dialog.get_detector_runs()


def test_guess_instrument_from_path(dialog, tmp_path):
    assert dialog.guess_instrument(_folder(tmp_path, "run gc1 mix")) == "GC1"
    assert dialog.guess_instrument(_folder(tmp_path, "GC2")) == "GC2"
    assert dialog.guess_instrument(_folder(tmp_path, "GC1 vs GC2")) == ""
    assert dialog.guess_instrument(_folder(tmp_path, "plain")) == ""
    child = tmp_path / "from_child"
    (child / "GC2_sample1.C").mkdir(parents=True)
    assert dialog.guess_instrument(str(child)) == "GC2"
    assert dialog.guess_instrument(str(tmp_path / "does not exist")) == ""


def test_setup_round_trip(dialog, tmp_path):
    m = tmp_path / "a.chromethod"
    ChromaMethod(name="a", signal_type="gc").to_file(m)
    _row(dialog, 0, "GC1", "FID1A", str(m))
    _row(dialog, 1, "GC2", "FID1A")
    _row(dialog, 3, "", "TCD2B")
    before = dialog.get_setup_rows()
    assert len(before) == 3
    path = tmp_path / "setup.json"
    dialog.save_setup(str(path))
    saved = json.loads(path.read_text())
    assert saved["version"] == 1 and saved["rows"] == before
    for i in range(4):
        _row(dialog, i, "", "")
    assert dialog.get_setup_rows() == []
    dialog.load_setup(str(path))
    assert dialog.get_setup_rows() == before


def test_load_setup_rejects_unknown_version(dialog, tmp_path):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps({"version": 2, "rows": []}))
    with pytest.raises(ValueError, match="version"):
        dialog.load_setup(str(path))


def test_tagged_folder_without_row_is_refused(dialog, tmp_path):
    _add(dialog, _folder(tmp_path, "g1"), "GC1")
    _add(dialog, _folder(tmp_path, "g2"), "GC2")
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    with pytest.raises(ValueError) as e:
        dialog.get_detector_runs()
    assert str(e.value) == "Folder g2 is tagged GC2 but no detector row is for GC2."


def test_row_whose_tag_no_folder_has_is_refused(dialog, tmp_path):
    _add(dialog, _folder(tmp_path, "g1"), "GC1")
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    _row(dialog, 1, "GC2", "FID1A")
    with pytest.raises(ValueError, match="GC2"):
        dialog.get_detector_runs()


def test_workbook_field_enabled_only_in_gc12_mode(dialog, tmp_path):
    assert not dialog.workbook_edit.isEnabled()
    path = _folder(tmp_path, "plain")
    _add(dialog, path)
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    assert not dialog.workbook_edit.isEnabled()  # no folder tagged
    dialog.set_directory_instrument(path, "GC1")
    assert dialog.workbook_edit.isEnabled()
    _row(dialog, 0, "", "FID1A")
    assert not dialog.workbook_edit.isEnabled()  # no row has an instrument
    _row(dialog, 0, "GC1", "FID1A")
    dialog.per_detector_check.setChecked(False)
    assert not dialog.workbook_edit.isEnabled()


def test_workbook_prefilled_when_both_tags_present(dialog, tmp_path):
    g1 = _folder(tmp_path, "run1")
    _add(dialog, g1, "GC1")
    assert dialog.workbook_edit.text() == ""
    _add(dialog, _folder(tmp_path, "run2"), "GC2")
    assert dialog.workbook_edit.text() == str(tmp_path / "run1 GC1&2.xlsx")


def test_workbook_needs_both_instruments(dialog, tmp_path):
    _add(dialog, _folder(tmp_path, "g1"), "GC1")
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    dialog.workbook_edit.setText(str(tmp_path / "out.xlsx"))
    with pytest.raises(ValueError, match="GC2"):
        dialog.get_workbook_path()
    _add(dialog, _folder(tmp_path, "g2"), "GC2")
    assert dialog.get_workbook_path() == str(tmp_path / "out.xlsx")


def test_workbook_needs_json_export(qtbot, tmp_path):
    main = QWidget()
    qtbot.addWidget(main)

    class _Export:
        def should_export_json(self, trigger):
            assert trigger == 'batch'
            return False

    main.export_manager = _Export()
    d = BatchJobDialog(main)
    qtbot.addWidget(d)
    d._main = main
    _add(d, _folder(tmp_path, "g1"), "GC1")
    _add(d, _folder(tmp_path, "g2"), "GC2")
    d.per_detector_check.setChecked(True)
    _row(d, 0, "GC1", "FID1A")
    d.workbook_edit.setText(str(tmp_path / "out.xlsx"))
    with pytest.raises(ValueError) as e:
        d.get_workbook_path()
    assert str(e.value) == ("Turn on JSON export during batch (Export Settings) "
                            "to write the GC1&2 workbook.")


def test_options_carry_directory_instruments_and_workbook(dialog, tmp_path):
    g1, g2 = _folder(tmp_path, "g1"), _folder(tmp_path, "g2")
    _add(dialog, g1, "GC1")
    _add(dialog, g2, "GC2")
    out = str(tmp_path / "out.xlsx")
    dialog.workbook_edit.setText(out)
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "GC1", "FID1A")
    _row(dialog, 1, "GC2", "FID1A")
    got = []
    dialog.start_batch.connect(lambda dirs, opts: got.append((dirs, opts)))
    dialog.on_start_processing()
    dirs, opts = got[0]
    assert dirs == [g1, g2]
    assert opts['directory_instruments'] == {g1: "GC1", g2: "GC2"}
    assert opts['gc12_workbook'] == out
    assert opts['detector_runs'] == opts['gc12_rows'] == [
        {'instrument': 'GC1', 'detector': 'FID1A', 'method_path': None},
        {'instrument': 'GC2', 'detector': 'FID1A', 'method_path': None},
    ]


def test_untagged_job_options_unchanged(dialog, tmp_path):
    g = _folder(tmp_path, "plain")
    _add(dialog, g)
    got = []
    dialog.start_batch.connect(lambda dirs, opts: got.append(opts))
    dialog.on_start_processing()
    assert got[0]['directory_instruments'] == {g: ""}
    assert got[0]['gc12_workbook'] is None
    assert got[0]['detector_runs'] == got[0]['gc12_rows'] == []


def test_list_item_shows_tag_and_removal_uses_path(dialog, tmp_path):
    g = _folder(tmp_path, "g1")
    _add(dialog, g, "GC1")
    assert dialog.directory_list.item(0).text() == f"GC1 \u2014 {g}"
    dialog.directory_list.item(0).setSelected(True)
    dialog.remove_selected_directories()
    assert dialog.directories == [] and dialog.directory_instruments == {}


def test_instrument_combo_tags_selected_folders(dialog, tmp_path):
    g = _folder(tmp_path, "g1")
    _add(dialog, g)
    dialog.directory_list.item(0).setSelected(True)
    dialog.instrument_set_combo.setCurrentIndex(2)
    dialog.instrument_set_combo.activated.emit(2)
    assert dialog.directory_instruments[g] == "GC2"


def test_setup_remembered_between_dialogs(qtbot, tmp_path):
    d1 = BatchJobDialog()
    qtbot.addWidget(d1)
    g1, g2 = _folder(tmp_path, "g1"), _folder(tmp_path, "g2")
    _add(d1, g1, "GC1")
    _add(d1, g2, "GC2")
    d1.workbook_edit.setText(str(tmp_path / "o.xlsx"))
    d1.per_detector_check.setChecked(True)
    _row(d1, 0, "GC1", "FID1A")
    _row(d1, 1, "GC2", "FID1A")
    d1.on_start_processing()
    d2 = BatchJobDialog()
    qtbot.addWidget(d2)
    assert d2.get_setup_rows() == d1.get_setup_rows()
    assert not d2.per_detector_check.isChecked()


def test_remembered_setup_beats_main_window_prefill(qtbot, tmp_path):
    batch_job_dialog._settings().setValue(
        "batch/detector_setup",
        json.dumps({"version": 1, "rows": [
            {"instrument": "GC2", "detector": "FID1A", "method_path": None}]}))
    d = _dialog_with_main(qtbot, tmp_path / "m.chromethod", "FID2B")
    assert d.get_setup_rows() == [
        {"instrument": "GC2", "detector": "FID1A", "method_path": None}]


def test_corrupt_remembered_setup_is_ignored(qtbot):
    batch_job_dialog._settings().setValue("batch/detector_setup", "{not json")
    d = BatchJobDialog()
    qtbot.addWidget(d)
    assert d.get_setup_rows() == []


# ---- auto-tagged folders must not change a non-GC1&2 job ----

def _autotagged(dialog, tmp_path):
    path = _folder(tmp_path, "260224 GC1 R1 Run 311")
    dialog._append_directory(path)
    assert dialog.directory_instruments[path] == "GC1"
    return path


def _start(dialog):
    got = []
    dialog.start_batch.connect(lambda dirs, opts: got.append(opts))
    dialog.on_start_processing()
    return got[0]


def test_autotagged_folder_with_box_unchecked_is_a_plain_job(dialog, tmp_path):
    path = _autotagged(dialog, tmp_path)
    assert dialog.get_detector_runs() == []
    opts = _start(dialog)
    assert opts['detector_runs'] == []
    assert opts['directory_instruments'] == {path: ""}
    assert opts['gc12_workbook'] is None


def test_autotagged_folder_with_blank_instrument_rows_keeps_pr5_behaviour(dialog, tmp_path):
    path = _autotagged(dialog, tmp_path)
    dialog.per_detector_check.setChecked(True)
    _row(dialog, 0, "", "FID1A")
    _row(dialog, 1, "", "FID2B")
    expected = [{'instrument': '', 'detector': 'FID1A', 'method_path': None},
                {'instrument': '', 'detector': 'FID2B', 'method_path': None}]
    assert dialog.get_detector_runs() == expected
    opts = _start(dialog)
    assert opts['detector_runs'] == expected
    assert opts['directory_instruments'] == {path: ""}
    assert opts['gc12_workbook'] is None


def test_filled_rows_are_not_used_when_box_unchecked(dialog, tmp_path):
    _autotagged(dialog, tmp_path)
    _row(dialog, 0, "GC1", "FID1A")
    assert not dialog.per_detector_check.isChecked()
    assert dialog.get_detector_runs() == []
    assert _start(dialog)['detector_runs'] == []
