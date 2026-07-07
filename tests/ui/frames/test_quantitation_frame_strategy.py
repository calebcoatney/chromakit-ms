import pytest
pytest.importorskip('pytestqt')

from logic.method import ChromaMethod
from ui.frames.quantitation import QuantitationFrame


def _make(qtbot):
    frame = QuantitationFrame()
    qtbot.addWidget(frame)
    frame.show()
    return frame


def test_combo_enabled_with_three_strategies(qtbot):
    frame = _make(qtbot)
    assert frame.method_combo.isEnabled()
    items = [frame.method_combo.itemText(i) for i in range(frame.method_combo.count())]
    assert items == ["None", "Internal Standard (Polyarc)", "RF Table"]


def test_selecting_rf_hides_is_groups(qtbot):
    frame = _make(qtbot)
    frame.select_strategy("rf_table")
    assert frame.is_group.isVisible() is False
    assert frame.sample_group.isVisible() is False


def test_selecting_is_shows_is_groups(qtbot):
    frame = _make(qtbot)
    frame.select_strategy("internal_standard")
    assert frame.is_group.isVisible() is True
    assert frame.sample_group.isVisible() is True


def test_apply_method_sets_combo(qtbot):
    frame = _make(qtbot)
    frame.apply_method(ChromaMethod(name="M", signal_type="gc", quant_strategy="rf_table"))
    assert frame.current_strategy() == "rf_table"


def test_apply_method_none(qtbot):
    frame = _make(qtbot)
    frame.apply_method(ChromaMethod(name="M", signal_type="gc", quant_strategy=None))
    assert frame.current_strategy() is None


def test_combo_change_emits_and_reports_strategy(qtbot):
    frame = _make(qtbot)
    fired = []
    frame.quantitation_changed.connect(lambda: fired.append(True))
    frame.select_strategy("rf_table")
    assert frame.current_strategy() == "rf_table"
    assert fired  # at least one change emitted
