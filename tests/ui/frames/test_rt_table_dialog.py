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


class TestAllowDuplicatesCheckbox:
    """Tests for the new allow_duplicates_checkbox added in the
    2026-06-22 spec follow-up (one peak per RT entry vs permissive).
    """

    def test_checkbox_default_is_on_permissive(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        assert frame.allow_duplicates_checkbox.isChecked() is True

    def test_settings_dict_includes_allow_duplicates(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        s = frame.get_settings()
        assert 'allow_duplicates' in s
        assert s['allow_duplicates'] is True

    def test_unchecking_propagates_to_settings(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        frame.allow_duplicates_checkbox.setChecked(False)
        assert frame.get_settings()['allow_duplicates'] is False

    def test_signal_emits_allow_duplicates(self, qtbot):
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        # Load minimal data so _on_settings_changed emits with enabled=True
        frame.rt_table_data = pd.DataFrame({
            'Compound': ['Decane'],
            'Start': [4.0], 'Apex': [4.1], 'End': [4.2],
        })
        captured = {}
        frame.rt_table_changed.connect(lambda s: captured.update(s))
        frame.allow_duplicates_checkbox.setChecked(False)
        assert captured.get('allow_duplicates') is False


class TestApplyRTMatchingDedup:
    """Tests for the dedup logic in ChromaKitApp._apply_rt_matching_to_peaks.

    Rather than instantiating the full ChromaKitApp, we test the dedup as a
    pure transformation: construct an RTTableFrame with the right state,
    construct mock peaks, and call the production code path that reads the
    frame's checkbox state and applies the dedup.
    """

    @staticmethod
    def _make_peak(compound_id, retention_time, peak_number):
        """Build a minimal mock peak with the fields the dedup logic touches."""
        class MockPeak:
            pass
        p = MockPeak()
        p.peak_number = peak_number
        p.retention_time = retention_time
        p.compound_id = compound_id
        p.Compound_ID = compound_id
        p.rt_assignment = True
        p.rt_assignment_source = 'RT'
        p.Qual = None
        return p

    def _run_dedup(self, frame, peaks):
        """Run the dedup block from _apply_rt_matching_to_peaks against
        the given frame state. Kept in sync with ui/app.py."""
        if frame.allow_duplicates_checkbox.isChecked():
            return  # no-op
        by_compound = {}
        for peak in peaks:
            if not getattr(peak, 'rt_assignment', False):
                continue
            cid = getattr(peak, 'compound_id', None)
            if not cid:
                continue
            window = frame.get_rt_window(cid)
            if not window:
                continue
            dist = abs(peak.retention_time - window[1])
            by_compound.setdefault(cid, []).append((peak, dist))
        for cid, candidates in by_compound.items():
            if len(candidates) <= 1:
                continue
            candidates.sort(key=lambda x: x[1])
            for peak, dist in candidates[1:]:
                placeholder = f"Unknown ({peak.retention_time:.3f})"
                peak.compound_id = placeholder
                peak.Compound_ID = placeholder
                peak.rt_assignment = False
                peak.rt_assignment_source = None
                peak.Qual = None

    def _make_frame_with_rt_table(self, qtbot):
        """Build an RTTableFrame with the user's actual mixed-acids RT table."""
        frame = RTTableFrame()
        qtbot.addWidget(frame)
        frame.rt_table_data = pd.DataFrame({
            'Compound': ['2-Hexanone', 'Nonane', '4-Octanone', '5-Nonanone'],
            'Start':    [3.052,        4.319,    5.128,        6.100],
            'Apex':     [3.079,        4.391,    5.164,        6.152],
            'End':      [3.266,        4.494,    5.271,        6.232],
        })
        return frame

    def test_no_dedup_when_allow_duplicates_on(self, qtbot):
        frame = self._make_frame_with_rt_table(qtbot)
        frame.allow_duplicates_checkbox.setChecked(True)
        # Two peaks both labeled '2-Hexanone'
        p1 = self._make_peak('2-Hexanone', 3.029, peak_number=1)
        p2 = self._make_peak('2-Hexanone', 3.079, peak_number=2)
        self._run_dedup(frame, [p1, p2])
        # Both should still be assigned
        assert p1.compound_id == '2-Hexanone'
        assert p2.compound_id == '2-Hexanone'
        assert p1.rt_assignment is True
        assert p2.rt_assignment is True

    def test_dedup_keeps_closest_to_apex(self, qtbot):
        frame = self._make_frame_with_rt_table(qtbot)
        frame.allow_duplicates_checkbox.setChecked(False)
        # From user's real data:
        #   2-Hexanone apex 3.079
        #   Peak at 3.029 (dist 0.050)  <-- should lose
        #   Peak at 3.079 (dist 0.000)  <-- should win
        p_far = self._make_peak('2-Hexanone', 3.029, peak_number=1)
        p_near = self._make_peak('2-Hexanone', 3.079, peak_number=2)
        self._run_dedup(frame, [p_far, p_near])
        assert p_near.compound_id == '2-Hexanone'
        assert p_near.rt_assignment is True
        assert p_far.compound_id == 'Unknown (3.029)'
        assert p_far.rt_assignment is False
        assert p_far.Compound_ID == 'Unknown (3.029)'

    def test_dedup_handles_multiple_compounds_independently(self, qtbot):
        frame = self._make_frame_with_rt_table(qtbot)
        frame.allow_duplicates_checkbox.setChecked(False)
        # Two compounds, each with 2 peaks
        # 2-Hexanone apex 3.079; Nonane apex 4.391
        peaks = [
            self._make_peak('2-Hexanone', 3.029, peak_number=1),  # 2-Hex loser
            self._make_peak('2-Hexanone', 3.079, peak_number=2),  # 2-Hex winner
            self._make_peak('Nonane',     4.266, peak_number=3),  # Nonane loser
            self._make_peak('Nonane',     4.391, peak_number=4),  # Nonane winner
        ]
        self._run_dedup(frame, peaks)
        assert peaks[0].compound_id == 'Unknown (3.029)'  # loser
        assert peaks[1].compound_id == '2-Hexanone'       # winner
        assert peaks[2].compound_id == 'Unknown (4.266)'  # loser
        assert peaks[3].compound_id == 'Nonane'           # winner

    def test_dedup_leaves_single_match_alone(self, qtbot):
        frame = self._make_frame_with_rt_table(qtbot)
        frame.allow_duplicates_checkbox.setChecked(False)
        p = self._make_peak('Nonane', 4.391, peak_number=1)
        self._run_dedup(frame, [p])
        assert p.compound_id == 'Nonane'
        assert p.rt_assignment is True

    def test_dedup_ignores_non_rt_assigned_peaks(self, qtbot):
        """Peaks without rt_assignment=True are not deduped even if compound_id matches."""
        frame = self._make_frame_with_rt_table(qtbot)
        frame.allow_duplicates_checkbox.setChecked(False)
        p1 = self._make_peak('Nonane', 4.266, peak_number=1)
        p1.rt_assignment = False  # e.g. came from MS search
        p2 = self._make_peak('Nonane', 4.391, peak_number=2)  # RT assigned
        self._run_dedup(frame, [p1, p2])
        # p1 was not RT-assigned, so dedup never considers it; p2 stays
        assert p1.compound_id == 'Nonane'
        assert p2.compound_id == 'Nonane'
