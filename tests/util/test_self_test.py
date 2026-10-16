import pytest

pytest.importorskip("pytestqt")

from util.self_test import run_self_test


def test_self_test_passes_in_a_folder_with_a_space(qtbot, tmp_path):
    out = tmp_path / "self test"
    assert run_self_test(out) == 0
    lines = (out / "self-test.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "OK"
    assert any(line.startswith("peaks:") for line in lines)


def test_self_test_reports_a_failure(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("util.self_test._check_export", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run_self_test(tmp_path) == 1
    text = (tmp_path / "self-test.txt").read_text(encoding="utf-8")
    assert text.startswith("FAIL") and "boom" in text
