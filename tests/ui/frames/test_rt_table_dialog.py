"""Tests for ui/frames/rt_table.AddToRTTableDialog NIST-backed selection."""
import pytest
pytest.importorskip('pytestqt')

from PySide6.QtWidgets import QDialog, QDialogButtonBox

from ui.frames.rt_table import AddToRTTableDialog


PEAK_DATA = {
    'start_time': 2.0,
    'retention_time': 2.1,
    'end_time': 2.2,
    'peak_number': 1,
}

LIBRARY = ["Propanoic acid", "Nonane", "Hexanoic acid", "Decane"]


def _get_ok_button(dlg):
    """Return the Save / OK button from the dialog's QDialogButtonBox."""
    box = dlg.findChild(QDialogButtonBox)
    return box.button(QDialogButtonBox.Ok)


class TestStrictNistSelection:
    def test_ok_disabled_on_open_with_library(self, qtbot):
        """When NIST library is available, OK starts disabled."""
        dlg = AddToRTTableDialog(
            peak_data=PEAK_DATA,
            library_compounds=LIBRARY,
        )
        qtbot.addWidget(dlg)
        ok = _get_ok_button(dlg)
        assert ok.isEnabled() is False

    def test_typing_alone_does_not_enable_ok(self, qtbot):
        dlg = AddToRTTableDialog(
            peak_data=PEAK_DATA,
            library_compounds=LIBRARY,
        )
        qtbot.addWidget(dlg)
        dlg.compound_name_edit.setText("Propan")
        # Trigger the filter timer manually
        if hasattr(dlg, 'filter_timer'):
            dlg.filter_timer.stop()
            dlg.filter_compounds()
        ok = _get_ok_button(dlg)
        assert ok.isEnabled() is False

    def test_selecting_from_list_enables_ok(self, qtbot):
        dlg = AddToRTTableDialog(
            peak_data=PEAK_DATA,
            library_compounds=LIBRARY,
        )
        qtbot.addWidget(dlg)
        dlg.compound_name_edit.setText("propan")
        dlg.filter_compounds()
        # Simulate clicking the first item
        item = dlg.results_list.item(0)
        assert item is not None
        dlg.on_item_selected(item)
        ok = _get_ok_button(dlg)
        assert ok.isEnabled() is True
        assert dlg.selected_compound == "Propanoic acid"

    def test_exact_match_in_different_case_promotes_on_save(self, qtbot):
        """Typing 'PROPANOIC ACID' and clicking save should auto-promote."""
        dlg = AddToRTTableDialog(
            peak_data=PEAK_DATA,
            library_compounds=LIBRARY,
        )
        qtbot.addWidget(dlg)
        dlg.compound_name_edit.setText("PROPANOIC ACID")
        # accept() should auto-promote and accept
        dlg.accept()
        assert dlg.selected_compound == "Propanoic acid"
        assert dlg.result() == QDialog.Accepted

    def test_no_library_disables_input(self, qtbot):
        """When NIST library is not loaded, the dialog is read-only."""
        dlg = AddToRTTableDialog(
            peak_data=PEAK_DATA,
            library_compounds=[],
        )
        qtbot.addWidget(dlg)
        assert dlg.compound_name_edit.isEnabled() is False
        ok = _get_ok_button(dlg)
        assert ok.isEnabled() is False
