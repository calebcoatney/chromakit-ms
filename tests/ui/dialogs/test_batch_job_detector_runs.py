"""BatchJobDialog: detector/method rows become options['detector_runs']."""
import pytest

pytest.importorskip('pytestqt')

from logic.method import ChromaMethod
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
        {'detector': 'FID1A', 'method_path': str(m1)},
        {'detector': 'FID2B', 'method_path': str(m2)},
    ]


def test_blank_second_row_is_ignored(dialog):
    dialog.per_detector_check.setChecked(True)
    _fill(dialog, 0, "FID2B")
    _fill(dialog, 1, "")
    assert dialog.get_detector_runs() == [{'detector': 'FID2B', 'method_path': None}]


def test_detector_taken_from_method_when_blank(dialog, tmp_path):
    m = tmp_path / "b.chromethod"
    ChromaMethod(name="b", signal_type="gc", detector="FID2B").to_file(m)
    dialog.per_detector_check.setChecked(True)
    _fill(dialog, 0, "", str(m))
    _fill(dialog, 1, "")
    assert dialog.get_detector_runs() == [{'detector': 'FID2B', 'method_path': str(m)}]


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
    from PySide6.QtWidgets import QWidget
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
