import pytest
pytest.importorskip('pytestqt')

import pandas as pd
from logic.method import RTMatchingParams
from logic.rt_matching import lookup_compound_by_rt as logic_lookup
from ui.frames.rt_table import RTTableFrame


RT_DF = pd.DataFrame({
    "Compound": ["Methane", "Ethane", "Propane"],
    "Start":    [1.00, 2.00, 3.00],
    "Apex":     [1.10, 2.10, 3.10],
    "End":      [1.20, 2.20, 3.20],
})


def test_gui_private_matcher_removed(qtbot):
    frame = RTTableFrame()
    qtbot.addWidget(frame)
    assert not hasattr(frame, "lookup_compound_by_rt")
    assert not hasattr(frame, "_lookup_simple_window")
    assert not hasattr(frame, "_lookup_closest_apex")
    assert not hasattr(frame, "_lookup_weighted_distance")


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_logic_lookup_stable(qtbot, mode):
    params = RTMatchingParams(matching_mode=mode)
    # Inside Methane window and outside-all are stable anchors across modes.
    assert logic_lookup(1.10, RT_DF, params) == "Methane"
    assert logic_lookup(9.99, RT_DF, params) is None
