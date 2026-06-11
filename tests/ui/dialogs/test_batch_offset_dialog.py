"""Smoke tests for ui/dialogs/batch_offset_dialog.BatchOffsetDialog.

Uses pytest-qt. Skipped on environments where pytest-qt isn't installed.
"""
import sys
import os

# Skip whole file if pytest-qt isn't available (e.g., headless CI envs)
import pytest
pytest.importorskip('pytestqt')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from PySide6.QtCore import Qt

from logic.sidecar_offsets import OffsetEntry
from ui.dialogs.batch_offset_dialog import BatchOffsetDialog


def _make_dialog(qtbot, offset_min=0.0234, with_existing=False,
                 siblings=None, current="/abs/sample_01.D"):
    siblings = siblings or [
        "/abs/sample_01.D",
        "/abs/sample_02.D",
        "/abs/sample_03.D",
        "/abs/sample_04.D",
    ]
    existing = {}
    if with_existing:
        existing = {
            "/abs/sample_02.D": OffsetEntry(
                offset_min=-0.012, timestamp=1.0, source="manual"
            ),
            "/abs/sample_04.D": OffsetEntry(
                offset_min=0.045, timestamp=2.0, source="auto"
            ),
        }
    dlg = BatchOffsetDialog(
        offset_min=offset_min,
        siblings=siblings,
        current_path=current,
        existing_offsets=existing,
    )
    qtbot.addWidget(dlg)
    return dlg


def test_dialog_shows_all_siblings_with_correct_default_checks(qtbot):
    """Open dialog; verify all sibling rows checked, current row disabled+checked."""
    dlg = _make_dialog(qtbot)
    assert len(dlg._row_widgets) == 4
    for path, cb in dlg._row_widgets:
        assert cb.isChecked(), f"{path} should be checked by default"
    # Current row is disabled
    current_cb = next(cb for path, cb in dlg._row_widgets
                      if path == "/abs/sample_01.D")
    assert not current_cb.isEnabled()
    # Sibling rows are enabled
    sib_cb = next(cb for path, cb in dlg._row_widgets
                  if path == "/abs/sample_02.D")
    assert sib_cb.isEnabled()


def test_dialog_shows_existing_offset_marker_on_conflict_rows(qtbot):
    """When a sibling has an existing entry, its label includes the warning text."""
    dlg = _make_dialog(qtbot, with_existing=True)
    sample02_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_02.D")
    # Existing offset is -0.012; the label should include "already -0.0120 min"
    assert "already" in sample02_cb.text()
    assert "-0.0120" in sample02_cb.text()
    # Sample 03 has no existing entry; no warning text
    sample03_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_03.D")
    assert "already" not in sample03_cb.text()


def test_dialog_select_none_unchecks_all_except_current(qtbot):
    """[Select none] button leaves current checked but unchecks siblings."""
    dlg = _make_dialog(qtbot)
    dlg._on_select_none()
    current_cb = next(cb for path, cb in dlg._row_widgets
                      if path == "/abs/sample_01.D")
    assert current_cb.isChecked(), "Current row must stay checked"
    sib_cb = next(cb for path, cb in dlg._row_widgets
                  if path == "/abs/sample_02.D")
    assert not sib_cb.isChecked()


def test_dialog_select_non_conflicts_skips_existing_entries(qtbot):
    """[Select non-conflicts] unchecks rows that already have an entry."""
    dlg = _make_dialog(qtbot, with_existing=True)
    dlg._on_select_non_conflicts()
    # sample_02 had existing -> unchecked
    sample02_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_02.D")
    assert not sample02_cb.isChecked()
    # sample_04 had existing -> unchecked
    sample04_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_04.D")
    assert not sample04_cb.isChecked()
    # sample_03 no existing -> checked
    sample03_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_03.D")
    assert sample03_cb.isChecked()
    # Current row stays checked (disabled)
    current_cb = next(cb for path, cb in dlg._row_widgets
                      if path == "/abs/sample_01.D")
    assert current_cb.isChecked()


def test_dialog_selected_paths_returns_only_checked(qtbot):
    """selected_paths() returns the checked rows (including current)."""
    dlg = _make_dialog(qtbot)
    # Uncheck sample_03
    sample03_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_03.D")
    sample03_cb.setChecked(False)
    selected = dlg.selected_paths()
    assert "/abs/sample_01.D" in selected  # current
    assert "/abs/sample_02.D" in selected
    assert "/abs/sample_03.D" not in selected
    assert "/abs/sample_04.D" in selected
    assert len(selected) == 3


def test_dialog_apply_disabled_when_only_current_selected(qtbot):
    """Apply button greyed when no siblings are checked."""
    dlg = _make_dialog(qtbot)
    dlg._on_select_none()  # uncheck all siblings; current stays
    assert not dlg._apply_btn.isEnabled()
    # Re-check one sibling -> Apply enables
    sample02_cb = next(cb for path, cb in dlg._row_widgets
                       if path == "/abs/sample_02.D")
    sample02_cb.setChecked(True)
    assert dlg._apply_btn.isEnabled()


def test_dialog_returns_accepted_on_apply(qtbot):
    """Clicking Apply triggers QDialog.Accepted."""
    from PySide6.QtWidgets import QDialog
    dlg = _make_dialog(qtbot)
    # Programmatically click Apply (avoids relying on actual mouse events)
    dlg._apply_btn.click()
    assert dlg.result() == QDialog.Accepted
