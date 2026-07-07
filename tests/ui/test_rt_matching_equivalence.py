import pytest
pytest.importorskip('pytestqt')

import pandas as pd
from logic.method import RTMatchingParams, RTMatchingWeights
from logic.rt_matching import lookup_compound_by_rt as logic_lookup
from ui.frames.rt_table import RTTableFrame


RT_DF = pd.DataFrame({
    "Compound": ["Methane", "Ethane", "Propane"],
    "Start":    [1.00, 2.00, 3.00],
    "Apex":     [1.10, 2.10, 3.10],
    "End":      [1.20, 2.20, 3.20],
})

# RTs chosen to cover: inside a window, near an apex, and outside all windows.
PROBES = [1.10, 1.15, 2.05, 2.50, 3.10, 5.00]


def _frame_with_data(qtbot, mode: int):
    frame = RTTableFrame()
    qtbot.addWidget(frame)
    frame.rt_table_data = RT_DF.copy()
    frame.enable_checkbox.setChecked(True)
    frame.matching_mode_combo.setCurrentIndex(mode)
    return frame


def _params_for(mode: int, frame: RTTableFrame) -> RTMatchingParams:
    return RTMatchingParams(
        matching_mode=mode,
        tolerance=frame.tolerance_spin.value(),
        window_expansion=frame.window_expansion_spin.value(),
        weights=RTMatchingWeights(**getattr(frame, "normalized_weights",
                                            {"start": 0.25, "apex": 0.50, "end": 0.25})),
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_gui_matcher_matches_logic(qtbot, mode):
    frame = _frame_with_data(qtbot, mode)
    params = _params_for(mode, frame)
    for rt in PROBES:
        gui_result = frame.lookup_compound_by_rt(rt)
        logic_result = logic_lookup(rt, RT_DF, params)
        assert gui_result == logic_result, f"mode={mode} rt={rt}: {gui_result!r} != {logic_result!r}"
