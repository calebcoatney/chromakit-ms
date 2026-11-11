"""Every ParametersFrame control must survive a .chromethod save and load.

Anh (2026-11-10 demo): signal break points and peak range filters were not in
the reloaded method. The values reached the file, but the tables were rebuilt
empty on load, and the next table edit overwrote the loaded list with what the
(empty) table held. The same applied to several other controls.
"""
import pytest

pytest.importorskip('pytestqt')

from logic.method import BreakPoint, ChromaMethod, FastchromParams
from ui.app import ChromaKitApp
from ui.frames.parameters import ParametersFrame


def _app(qtbot):
    app = ChromaKitApp()
    qtbot.addWidget(app)
    # No file loaded. on_parameters_changed reads these before any load sets them.
    app.current_x = app.current_y = None
    return app


def _table_rows(table):
    return [[table.cellWidget(r, c).value() for c in (0, 1)] for r in range(table.rowCount())]


def test_break_points_and_range_filters_survive_save_and_load(qtbot, tmp_path, monkeypatch):
    first = _app(qtbot)
    pf = first.parameters_frame
    pf._add_range_filter_row()
    pf.range_filter_table.cellWidget(0, 0).setValue(2.5)
    pf.range_filter_table.cellWidget(0, 1).setValue(9.0)
    pf._add_break_point_row()
    pf.break_point_table.cellWidget(0, 0).setValue(4.2)
    path = tmp_path / "anh.chromethod"
    first.current_method.to_file(path)

    second = _app(qtbot)
    monkeypatch.setattr(
        "ui.app.QFileDialog.getOpenFileName", lambda *_a, **_k: (str(path), ""))
    second.load_method()
    pf2 = second.parameters_frame

    assert _table_rows(pf2.range_filter_table) == [[2.5, 9.0]]
    assert _table_rows(pf2.break_point_table) == [[4.2, pytest.approx(0.1)]]

    # Editing the table after load keeps the loaded row.
    pf2._add_range_filter_row()
    assert second.current_method.peaks.range_filters[0] == [2.5, 9.0]
    assert second.current_method.baseline.break_points[0].time == pytest.approx(4.2)


def test_added_row_left_at_defaults_reaches_method(qtbot):
    app = _app(qtbot)
    app.parameters_frame._add_range_filter_row()
    app.parameters_frame._add_peak_group_row()
    assert app.current_method.peaks.range_filters == [[0.0, 60.0]]
    assert app.current_method.integration.peak_groups == [[0.0, 60.0]]


def _method():
    m = ChromaMethod(name="t", signal_type="gc")
    m.smoothing.method = "savgol"
    m.smoothing.diff_order = 2
    m.baseline.baseline_offset = 1.5
    m.baseline.fastchrom = FastchromParams(half_window=7, smooth_half_window=9)
    m.baseline.break_points = [BreakPoint(time=3.0, tolerance=0.2), BreakPoint(time=6.0, tolerance=0.05)]
    m.peaks.mode = "peak_splitting"
    m.peaks.range_filters = [[1.0, 2.0], [5.0, 8.0]]
    m.deconvolution.splitting_method = "emg"
    m.deconvolution.windows = [[3.0, 4.0]]
    m.integration.peak_groups = [[7.0, 7.5]]
    return m


@pytest.mark.parametrize("read, expected", [
    (lambda f: f.smooth_method_combo.currentData(), "savgol"),
    (lambda f: f.smooth_diff_order_combo.currentData(), 2),
    (lambda f: float(f.baseline_offset_input.text()), 1.5),
    (lambda f: (f.fc_half_window.value(), f.fc_smooth_half_window.value()), (7, 9)),
    (lambda f: _table_rows(f.break_point_table), [[3.0, 0.2], [6.0, 0.05]]),
    (lambda f: f.peak_mode_combo.currentData(), "peak_splitting"),
    (lambda f: _table_rows(f.range_filter_table), [[1.0, 2.0], [5.0, 8.0]]),
    (lambda f: f.splitting_method_combo.currentData(), "emg"),
    (lambda f: _table_rows(f.peak_splitting_windows_table), [[3.0, 4.0]]),
    (lambda f: _table_rows(f.peak_group_table), [[7.0, 7.5]]),
], ids=["smoothing.method", "smoothing.diff_order", "baseline.baseline_offset",
        "baseline.fastchrom", "baseline.break_points", "peaks.mode",
        "peaks.range_filters", "deconvolution.splitting_method",
        "deconvolution.windows", "integration.peak_groups"])
def test_apply_method_restores_control(qtbot, read, expected):
    frame = ParametersFrame()
    qtbot.addWidget(frame)
    frame.apply_method(_method())
    assert read(frame) == expected


def test_apply_method_does_not_rewrite_params(qtbot):
    """Restoring controls must not fire handlers that change the loaded values."""
    method = _method()
    frame = ParametersFrame()
    qtbot.addWidget(frame)
    frame.apply_method(method)
    assert frame.current_params == method.to_gui_params()
