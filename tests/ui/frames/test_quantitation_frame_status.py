"""Tests for QuantitationFrame status panel + stale-state tracking."""
import pytest
pytest.importorskip('pytestqt')

from logic.quantitation_runner import QuantitationSummary
from ui.frames.quantitation import QuantitationFrame


def _basic_summary(**overrides):
    """Build a QuantitationSummary with sensible defaults; override any field."""
    s = QuantitationSummary()
    s.peaks_total = overrides.get('peaks_total', 14)
    s.peaks_assigned = overrides.get('peaks_assigned', 12)
    s.peaks_quantitated = overrides.get('peaks_quantitated', 11)
    s.peaks_skipped_unassigned = overrides.get('peaks_skipped_unassigned', 2)
    s.peaks_skipped_no_metadata = overrides.get('peaks_skipped_no_metadata', [])
    s.peaks_skipped_other = overrides.get('peaks_skipped_other', [])
    s.response_factor = overrides.get('response_factor', 1.95e10)
    s.carbon_balance_percent = overrides.get('carbon_balance_percent', 95.3)
    return s


class TestStatusPanel:
    def test_clear_status_shows_no_run_message(self, qtbot):
        frame = QuantitationFrame()
        qtbot.addWidget(frame)
        frame.clear_status()
        assert "No quantitation run" in frame.status_summary_label.text()
        # Stale flag should not be visible after clear
        assert frame.stale_label.isVisible() is False

    def test_update_status_populates_counts(self, qtbot):
        frame = QuantitationFrame()
        qtbot.addWidget(frame)
        frame.show()
        summary = _basic_summary()
        frame.update_status(summary)
        text = frame.status_summary_label.text()
        assert "14" in text  # peaks_total
        assert "12" in text  # peaks_assigned
        assert "11" in text  # peaks_quantitated

    def test_update_status_lists_skipped_no_metadata(self, qtbot):
        frame = QuantitationFrame()
        qtbot.addWidget(frame)
        frame.show()
        summary = _basic_summary(
            peaks_skipped_no_metadata=["Polyglycerol", "Mysterium"]
        )
        frame.update_status(summary)
        # The skipped-peaks list widget should contain both names
        items = [frame.skipped_list.item(i).text()
                 for i in range(frame.skipped_list.count())]
        joined = " ".join(items)
        assert "Polyglycerol" in joined
        assert "Mysterium" in joined

    def test_update_status_hides_skipped_section_when_empty(self, qtbot):
        frame = QuantitationFrame()
        qtbot.addWidget(frame)
        frame.show()
        summary = _basic_summary(peaks_skipped_no_metadata=[])
        frame.update_status(summary)
        assert frame.skipped_list.isVisible() is False


class TestStaleTracking:
    def test_stale_flag_appears_after_input_change(self, qtbot):
        frame = QuantitationFrame()
        qtbot.addWidget(frame)
        frame.show()
        # Enable + set baseline state
        frame.enable_checkbox.setChecked(True)
        frame.compound_edit.setText("Nonane")
        frame.formula_edit.setText("C9H20")
        frame.mw_edit.setText("128.259")
        frame.density_edit.setText("0.718")
        frame.volume_is_edit.setText("2")
        summary = _basic_summary()
        frame.update_status(summary)
        assert frame.stale_label.isVisible() is False

        # Change an input
        frame.volume_is_edit.setText("3")
        assert frame.stale_label.isVisible() is True

    def test_update_status_clears_stale_flag(self, qtbot):
        frame = QuantitationFrame()
        qtbot.addWidget(frame)
        frame.show()
        frame.enable_checkbox.setChecked(True)
        frame.compound_edit.setText("Nonane")
        frame.formula_edit.setText("C9H20")
        frame.mw_edit.setText("128.259")
        frame.density_edit.setText("0.718")
        frame.volume_is_edit.setText("2")
        # Set status, change input, then re-update — stale should clear
        frame.update_status(_basic_summary())
        frame.volume_is_edit.setText("3")
        assert frame.stale_label.isVisible() is True
        frame.update_status(_basic_summary())
        assert frame.stale_label.isVisible() is False
