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


import pandas as pd
from ui.frames.rt_table import RTTableFrame


class TestRTTableFrameValidation:
    def test_mark_library_mismatches_identifies_missing_names(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        frame.rt_table_data = pd.DataFrame({
            'Compound': ['Propanoic acid', 'Nonsense Compound', 'Nonane'],
            'Start': [2.0, 3.0, 4.0],
            'Apex': [2.1, 3.1, 4.1],
            'End': [2.2, 3.2, 4.2],
        })
        frame._library_compounds = ['Propanoic acid', 'Nonane', 'Decane']
        frame._mark_library_mismatches()
        assert frame._library_mismatches == {'Nonsense Compound'}

    def test_mark_library_mismatches_empty_when_library_unloaded(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        frame.rt_table_data = pd.DataFrame({
            'Compound': ['Anything'],
            'Start': [1.0], 'Apex': [1.1], 'End': [1.2],
        })
        frame._library_compounds = []
        frame._mark_library_mismatches()
        assert frame._library_mismatches == set()

    def test_set_library_compounds_triggers_revalidation(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        frame.rt_table_data = pd.DataFrame({
            'Compound': ['Foo', 'Bar'],
            'Start': [1.0, 2.0], 'Apex': [1.1, 2.1], 'End': [1.2, 2.2],
        })
        # Initially no library
        frame.set_library_compounds([])
        assert frame._library_mismatches == set()
        # Set a library missing 'Bar'
        frame.set_library_compounds(['Foo'])
        assert frame._library_mismatches == {'Bar'}
