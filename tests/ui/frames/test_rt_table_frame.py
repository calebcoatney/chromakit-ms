import pytest
pytest.importorskip('pytestqt')

from logic.method import (
    ChromaMethod, RTTableEntry, RTMatchingParams, RTMatchingWeights,
)
from ui.frames.rt_table import RTTableFrame


def _make(qtbot):
    frame = RTTableFrame()
    qtbot.addWidget(frame)
    return frame


def _method_with_rt():
    return ChromaMethod(
        name="M", signal_type="gc",
        rt_table=[
            RTTableEntry(compound="Methane", start=1.0, apex=1.1, end=1.2),
            RTTableEntry(compound="Ethane", start=2.0, apex=2.1, end=2.2),
        ],
        rt_matching=RTMatchingParams(
            matching_mode=1, tolerance=0.15, window_expansion=0.05,
            weights=RTMatchingWeights(start=0.3, apex=0.4, end=0.3),
            high_priority=True,
        ),
    )


def test_apply_method_populates_table(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())
    df = frame.rt_table.get_dataframe()
    assert list(df.columns) == ["Compound", "Start", "Apex", "End"]
    assert list(df["Compound"]) == ["Methane", "Ethane"]


def test_apply_method_sets_matching_widgets(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())
    assert frame.matching_mode_combo.currentIndex() == 1
    assert abs(frame.tolerance_spin.value() - 0.15) < 1e-9
    assert abs(frame.window_expansion_spin.value() - 0.05) < 1e-9
    assert frame.high_priority_checkbox.isChecked() is True


def test_get_rt_entries_and_matching_params(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())
    entries = frame.get_rt_entries()
    assert [e.compound for e in entries] == ["Methane", "Ethane"]
    params = frame.get_matching_params()
    assert params.matching_mode == 1
    assert params.high_priority is True


def test_apply_method_does_not_emit_rt_table_changed(qtbot):
    frame = _make(qtbot)
    fired = []
    frame.rt_table_changed.connect(lambda *a: fired.append(True))
    frame.apply_method(_method_with_rt())
    assert fired == []  # programmatic load must be silent (no feedback-loop trigger)
