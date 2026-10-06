"""ChromaKitApp._apply_rt_matching_to_peaks: shared logic, dedup via the method."""
import pytest
pytest.importorskip('pytestqt')

from ui.app import ChromaKitApp
from logic.integration import ChromatographicPeak
from logic.method import RTMatchingParams, RTTableEntry


def _peak(rt, n):
    return ChromatographicPeak(compound_id=f"Unknown ({rt:.3f})", peak_number=n, retention_time=rt,
                               integrator="py", width=0.01, area=1.0, start_time=rt - 0.005,
                               end_time=rt + 0.005)


@pytest.mark.parametrize("allow_duplicates, expected", [
    (True, ["Methane", "Methane"]),
    (False, ["Unknown (1.020)", "Methane"]),
])
def test_gui_rt_matching_follows_allow_duplicates(qtbot, monkeypatch, allow_duplicates, expected):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_method.rt_table = [RTTableEntry(compound="Methane", start=1.0, apex=1.1, end=1.2)]
    app.current_method.rt_matching = RTMatchingParams(allow_duplicates=allow_duplicates)
    monkeypatch.setattr(app.rt_table_frame, "is_enabled", lambda: True)
    peaks = [_peak(1.02, 1), _peak(1.10, 2)]
    app._apply_rt_matching_to_peaks(peaks)   # used to raise NameError (rt_table_frame)
    assert [p.compound_id for p in peaks] == expected
