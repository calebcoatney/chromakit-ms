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
