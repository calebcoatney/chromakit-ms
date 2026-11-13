"""Mouse and keyboard segment editing on the chromatogram plot (design 2026-11-12 §4.3)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('pytestqt')

from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
from PySide6.QtWidgets import QMenu  # noqa: E402

OPEN_W1 = {'id': 'w1', 'label': 'early', 'start': 3.0, 'end': 6.0, 'partway': []}
CHANGE_TEXT = 'Change the baseline here'


def _fake_menu(choose=None, seen=None):
    """A QMenu whose exec records the action texts and returns the one titled ``choose``.

    PySide6 ignores a class-level patch of QMenu.exec, so tests swap plot.py's QMenu for this.
    """
    class FakeMenu(QMenu):
        def exec(self, *args, **kwargs):
            if seen is None:
                raise AssertionError("a real modal QMenu was opened")
            actions = [a for a in self.actions() if not a.isSeparator()]
            seen.append([a.text() for a in actions])
            return next((a for a in actions if a.text() == choose), None)
        exec_ = exec
    return FakeMenu


@pytest.fixture(autouse=True)
def _no_modal_menus(monkeypatch):
    """A real QMenu.exec blocks offscreen forever; tests that expect a menu use the menus fixture."""
    monkeypatch.setattr('ui.frames.plot.QMenu', _fake_menu())


@pytest.fixture
def menus(monkeypatch):
    """menus(text) makes every plot menu pick ``text``; returns the list of menus shown."""
    def choose(text):
        seen = []
        monkeypatch.setattr('ui.frames.plot.QMenu', _fake_menu(text, seen))
        return seen
    return choose


def _data():
    x = np.linspace(0.0, 20.0, 4001)
    corrected = 50.0 * np.exp(-((x - 4.5) ** 2) / (2 * 0.05 ** 2))
    baseline = np.full_like(x, 100.0)
    return {
        'x': x, 'original_y': corrected + baseline, 'smoothed_y': corrected + baseline,
        'baseline_y': baseline, 'corrected_y': corrected, 'peaks_x': np.array([4.5]),
        'peaks_y': np.array([50.0]), 'peak_metadata': [{'x': 4.5, 'y': 50.0, 'type': 'peak'}],
        'range_excluded_peaks_x': [],
    }


def _plot(qtbot, invert_x=False):
    from ui.frames.plot import PlotFrame
    plot = PlotFrame()
    qtbot.addWidget(plot)
    if invert_x:
        plot.set_axis_labels('Wavenumber (cm⁻¹)', 'Absorbance', invert_x=True)
    plot.plot_chromatogram(_data(), new_file=True)
    plot.chromatogram_ax.set_xlim(20.0, 0.0) if invert_x else plot.chromatogram_ax.set_xlim(0.0, 20.0)
    plot.canvas.draw()
    return plot


def _px(plot, x):
    ax = plot.chromatogram_ax
    y_mid = sum(ax.get_ylim()) / 2
    return ax.transData.transform((x, y_mid))


def _mouse(plot, name, x, button=1, shift=False, px_x=None):
    """A matplotlib mouse event at data x (or display px_x) mid-height on the chromatogram."""
    px, py = _px(plot, x)
    if px_x is not None:
        px = px_x
    event = MouseEvent(name, plot.canvas, px, py, button=button,
                       modifiers=frozenset({'shift'}) if shift else frozenset())
    plot.canvas.callbacks.process(name, event)
    return event


def _drag(plot, x0, x1, shift=True, button=1):
    _mouse(plot, 'button_press_event', x0, button=button, shift=shift)
    _mouse(plot, 'motion_notify_event', x1, button=button, shift=shift)
    _mouse(plot, 'button_release_event', x1, button=button, shift=shift)


def _key(plot, key):
    event = KeyEvent('key_press_event', plot.canvas, key)
    plot.canvas.callbacks.process('key_press_event', event)


def _record(signal):
    got = []
    signal.connect(lambda *args: got.append(args))
    return got


def _role(plot, role):
    return [a for a in plot._segment_artists if getattr(a, 'segment_role', None) == role]


# ── shift-drag ────────────────────────────────────────────────────────────────

def test_shift_drag_emits_span(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    _mouse(plot, 'button_press_event', 8.0, shift=True)
    _mouse(plot, 'motion_notify_event', 10.0, shift=True)
    (temp,) = [p for p in plot.chromatogram_ax.patches if getattr(p, 'segment_role', None) == 'drag_span']
    assert temp.get_alpha() == pytest.approx(0.2)
    _mouse(plot, 'button_release_event', 10.0, shift=True)
    assert len(spans) == 1
    assert spans[0] == (pytest.approx(8.0, abs=0.05), pytest.approx(10.0, abs=0.05))
    assert temp.axes is None   # the temporary span is gone


def test_shift_drag_right_to_left_sorted(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    _drag(plot, 10.0, 8.0)
    assert len(spans) == 1
    start, end = spans[0]
    assert start < end
    assert (start, end) == (pytest.approx(8.0, abs=0.05), pytest.approx(10.0, abs=0.05))


def test_shift_drag_on_descending_axis_sorted(qtbot):
    plot = _plot(qtbot, invert_x=True)
    spans = _record(plot.segment_span_requested)
    _drag(plot, 8.0, 10.0)
    assert spans == [(pytest.approx(8.0, abs=0.05), pytest.approx(10.0, abs=0.05))]


def test_plain_drag_does_not_emit(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    _drag(plot, 8.0, 10.0, shift=False)
    assert spans == []
    assert not [p for p in plot.chromatogram_ax.patches if getattr(p, 'segment_role', None) == 'drag_span']


def test_narrow_shift_drag_does_not_emit(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    px0 = _px(plot, 8.0)[0]
    _mouse(plot, 'button_press_event', 8.0, shift=True)
    _mouse(plot, 'button_release_event', 8.0, shift=True, px_x=px0 + 2)
    assert spans == []


def test_shift_drag_ignored_while_toolbar_pans(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    plot.toolbar.pan()
    try:
        assert plot.toolbar.mode
        _drag(plot, 8.0, 10.0)
    finally:
        plot.toolbar.pan()
    assert spans == []


def test_plain_click_still_marks_the_point(qtbot):
    plot = _plot(qtbot)
    _mouse(plot, 'button_press_event', 9.0)
    _mouse(plot, 'button_release_event', 9.0)
    marks = [ln for ln in plot.chromatogram_ax.get_lines() if ln.get_label() == '_selected_point']
    assert len(marks) == 1


# ── right-click ───────────────────────────────────────────────────────────────

def test_right_click_inside_open_segment_offers_change(qtbot, menus):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    changes = _record(plot.baseline_change_requested)
    seen = menus(CHANGE_TEXT)
    _mouse(plot, 'button_press_event', 5.2, button=3)
    _mouse(plot, 'button_release_event', 5.2, button=3)
    assert seen == [[CHANGE_TEXT]]
    assert changes == [(pytest.approx(5.2, abs=0.05),)]


def test_right_click_outside_open_segment_has_no_menu(qtbot, menus):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    changes = _record(plot.baseline_change_requested)
    seen = menus(CHANGE_TEXT)
    _mouse(plot, 'button_press_event', 9.0, button=3)
    _mouse(plot, 'button_release_event', 9.0, button=3)
    assert seen == [] and changes == []


def test_right_click_without_open_segment_has_no_menu(qtbot, menus):
    plot = _plot(qtbot)
    seen = menus(CHANGE_TEXT)
    _mouse(plot, 'button_press_event', 5.2, button=3)
    assert seen == []


def _peak(rt):
    return SimpleNamespace(retention_time=rt, start_time=rt - 0.1, end_time=rt + 0.1,
                           peak_number=1)


def test_right_click_on_peak_inside_open_segment_adds_change(qtbot, menus):
    plot = _plot(qtbot)
    plot.integrated_peaks = [_peak(4.5)]
    plot.set_segment_overlay([], [], OPEN_W1)
    changes = _record(plot.baseline_change_requested)
    seen = menus(CHANGE_TEXT)
    _mouse(plot, 'button_press_event', 4.52, button=3)
    assert len(seen) == 1
    assert 'Search MS Library' in seen[0] and seen[0][-1] == CHANGE_TEXT
    assert changes == [(pytest.approx(4.52, abs=0.05),)]


def test_peak_menu_without_open_segment_has_no_change(qtbot, menus):
    plot = _plot(qtbot)
    plot.integrated_peaks = [_peak(4.5)]
    seen = menus(CHANGE_TEXT)
    _mouse(plot, 'button_press_event', 4.52, button=3)
    assert len(seen) == 1 and CHANGE_TEXT not in seen[0]


# ── edge drag ─────────────────────────────────────────────────────────────────

def _edge_line(plot, x):
    (line,) = [ln for ln in _role(plot, 'edge') if ln.get_xdata()[0] == pytest.approx(x, abs=0.05)]
    return line


def test_edge_drag_emits_edge_moved(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    line = _edge_line(plot, 6.0)
    _mouse(plot, 'button_press_event', 6.05)
    _mouse(plot, 'motion_notify_event', 7.0)
    assert line.get_xdata()[0] == pytest.approx(7.0, abs=0.05)   # follows the mouse live
    assert moved == []                                           # nothing until release
    _mouse(plot, 'button_release_event', 7.0)
    assert moved == [('end', pytest.approx(7.0, abs=0.05))]


def test_edge_drag_start_edge(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    _drag(plot, 3.0, 2.0, shift=False)
    assert moved == [('start', pytest.approx(2.0, abs=0.05))]


def test_edge_drag_cannot_cross_the_other_edge(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    _drag(plot, 6.0, 1.0, shift=False)
    assert len(moved) == 1
    edge, t = moved[0]
    assert edge == 'end'
    gap_px = _px(plot, t)[0] - _px(plot, 3.0)[0]
    assert gap_px == pytest.approx(3.0, abs=0.01)


def test_edge_drag_on_descending_axis(qtbot):
    plot = _plot(qtbot, invert_x=True)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    _drag(plot, 6.0, 1.0, shift=False)
    edge, t = moved[0]
    assert edge == 'end'
    assert abs(_px(plot, t)[0] - _px(plot, 3.0)[0]) == pytest.approx(3.0, abs=0.01)


def test_press_away_from_edges_is_a_plain_click(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    _drag(plot, 5.0, 5.5, shift=False)
    assert moved == []


# ── Esc ───────────────────────────────────────────────────────────────────────

def test_escape_emits(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    escapes = _record(plot.segment_escape)
    _key(plot, 'escape')
    assert escapes == [()]


def test_escape_without_open_segment_does_nothing(qtbot):
    plot = _plot(qtbot)
    escapes = _record(plot.segment_escape)
    _key(plot, 'escape')
    assert escapes == []


def test_canvas_takes_focus_on_click(qtbot):
    from PySide6.QtCore import Qt
    plot = _plot(qtbot)
    assert plot.canvas.focusPolicy() & Qt.FocusPolicy.ClickFocus


# ── app routing ───────────────────────────────────────────────────────────────

def _app(qtbot):
    from ui.app import ChromaKitApp
    app = ChromaKitApp()
    qtbot.addWidget(app)
    return app


def test_span_opens_new_segment(qtbot):
    app = _app(qtbot)
    app.plot_frame.segment_span_requested.emit(8.0, 10.0)
    editor = app.parameters_frame.segment_editor
    assert editor is not None
    assert (editor.record()['start'], editor.record()['end']) == (8.0, 10.0)
    assert app.parameters_frame.has_open_segment()


def test_span_refused_while_a_segment_is_open(qtbot):
    app = _app(qtbot)
    app.plot_frame.segment_span_requested.emit(8.0, 10.0)
    editor = app.parameters_frame.segment_editor
    app.plot_frame.segment_span_requested.emit(12.0, 14.0)
    assert app.parameters_frame.segment_editor is editor
    assert (editor.record()['start'], editor.record()['end']) == (8.0, 10.0)
    assert app.status_bar.currentMessage() == 'Save or cancel the open segment first'


def test_span_reshows_a_hidden_segments_panel(qtbot):
    app = _app(qtbot)
    pf = app.parameters_frame
    pf.set_section_visibility({'segments': False})
    assert pf.segments_panel.isHidden()
    app.plot_frame.segment_span_requested.emit(8.0, 10.0)
    assert not pf.segments_panel.isHidden()
    assert pf.segment_editor is not None


def test_plot_edits_route_to_the_open_segment(qtbot):
    app = _app(qtbot)
    pf = app.parameters_frame
    app.plot_frame.segment_span_requested.emit(8.0, 10.0)
    app.plot_frame.baseline_change_requested.emit(9.0)
    assert [row['time'] for row in pf.segment_editor.record()['partway']] == [9.0]
    app.plot_frame.segment_edge_moved.emit('end', 11.0)
    assert pf.segment_editor.record()['end'] == 11.0
    app.plot_frame.segment_escape.emit()
    assert pf.segment_editor is None


# ── fix round 1 ───────────────────────────────────────────────────────────────

def test_right_click_in_toolbar_mode_has_no_baseline_menu(qtbot, menus):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    seen = menus(CHANGE_TEXT)
    plot.toolbar.pan()
    try:
        _mouse(plot, 'button_press_event', 5.2, button=3)
        _mouse(plot, 'button_release_event', 5.2, button=3)
    finally:
        plot.toolbar.pan()
    assert seen == []        # the toolbar keeps its right-drag (pan scale / zoom out)


def test_peak_menu_in_toolbar_mode_is_unchanged(qtbot, menus):
    plot = _plot(qtbot)
    plot.integrated_peaks = [_peak(4.5)]
    plot.set_segment_overlay([], [], OPEN_W1)
    seen = menus(CHANGE_TEXT)
    plot.toolbar.zoom()
    try:
        _mouse(plot, 'button_press_event', 4.52, button=3)
    finally:
        plot.toolbar.zoom()
    assert len(seen) == 1 and CHANGE_TEXT not in seen[0]


def test_edge_press_on_a_peak_with_wobble_selects_the_peak(qtbot, monkeypatch):
    plot = _plot(qtbot)
    plot.integrated_peaks = [_peak(6.0)]
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    selected = []
    monkeypatch.setattr(plot, '_highlight_selected_peak', selected.append)
    px = _px(plot, 6.0)[0]
    line = _edge_line(plot, 6.0)
    _mouse(plot, 'button_press_event', 6.0)
    _mouse(plot, 'motion_notify_event', 6.0, px_x=px + 1)
    assert line.get_xdata()[0] == 6.0          # wobble does not move the edge
    _mouse(plot, 'button_release_event', 6.0, px_x=px + 1)
    assert moved == []
    assert selected == [plot.integrated_peaks[0]]


def test_edge_press_ignored_in_toolbar_mode(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    line = _edge_line(plot, 6.0)
    plot.toolbar.pan()
    try:
        _drag(plot, 6.0, 7.0, shift=False)
    finally:
        plot.toolbar.pan()
    assert moved == [] and line.get_xdata()[0] == 6.0


def _ratio(monkeypatch, plot, ratio):
    monkeypatch.setattr(type(plot.canvas), 'device_pixel_ratio', property(lambda self: ratio))


def test_thresholds_scale_with_device_pixel_ratio(qtbot, monkeypatch):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    _ratio(monkeypatch, plot, 2)
    spans = _record(plot.segment_span_requested)
    moved = _record(plot.segment_edge_moved)

    px = _px(plot, 10.0)[0]          # a 5 physical px shift-drag is 2.5 logical px: a click
    _mouse(plot, 'button_press_event', 10.0, shift=True)
    _mouse(plot, 'button_release_event', 10.0, shift=True, px_x=px + 5)
    assert spans == []
    _mouse(plot, 'button_press_event', 10.0, shift=True)
    _mouse(plot, 'button_release_event', 10.0, shift=True, px_x=px + 7)
    assert len(spans) == 1

    edge_px = _px(plot, 6.0)[0]      # 10 physical px from the edge is 5 logical px: grabbed
    _mouse(plot, 'button_press_event', 6.0, px_x=edge_px + 10)
    _mouse(plot, 'button_release_event', 6.0, px_x=edge_px + 15)   # 5 physical px: still a click
    assert moved == []
    _mouse(plot, 'button_press_event', 6.0, px_x=edge_px + 10)
    _mouse(plot, 'button_release_event', 6.0, px_x=edge_px + 40)
    assert len(moved) == 1 and moved[0][0] == 'end'


def test_edge_clamp_gap_scales_with_device_pixel_ratio(qtbot, monkeypatch):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    _ratio(monkeypatch, plot, 2)
    moved = _record(plot.segment_edge_moved)
    _drag(plot, 6.0, 1.0, shift=False)
    assert _px(plot, moved[0][1])[0] - _px(plot, 3.0)[0] == pytest.approx(6.0, abs=0.01)


def test_emitted_times_are_rounded_to_4_decimals(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    _drag(plot, 8.01234567, 9.98765432)
    start, end = spans[0]
    assert start == round(start, 4) and end == round(end, 4) and start < end

    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    _drag(plot, 6.0, 7.01234567, shift=False)
    (edge, t), = moved
    assert t == round(t, 4)


def test_round_past_keeps_strict_order():
    from ui.frames.plot import PlotFrame
    assert PlotFrame._round_past(3.00004, 3.00006, below=True) == 3.0
    assert PlotFrame._round_past(3.00005, 3.00006, below=True) < 3.00006
    assert PlotFrame._round_past(3.00007, 3.00006, below=False) > 3.00006
    assert PlotFrame._round_past(5.00001, 5.0, below=False) == 5.0001


def test_escape_during_shift_drag_removes_the_span(qtbot):
    plot = _plot(qtbot)
    spans = _record(plot.segment_span_requested)
    _mouse(plot, 'button_press_event', 8.0, shift=True)
    _mouse(plot, 'motion_notify_event', 10.0, shift=True)
    assert [p for p in plot.chromatogram_ax.patches if getattr(p, 'segment_role', None) == 'drag_span']
    _key(plot, 'escape')
    assert not [p for p in plot.chromatogram_ax.patches if getattr(p, 'segment_role', None) == 'drag_span']
    _mouse(plot, 'button_release_event', 10.0, shift=True)
    assert spans == []


def test_escape_during_edge_drag_restores_the_edge(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([], [], OPEN_W1)
    moved = _record(plot.segment_edge_moved)
    escapes = _record(plot.segment_escape)
    line = _edge_line(plot, 6.0)
    _mouse(plot, 'button_press_event', 6.0)
    _mouse(plot, 'motion_notify_event', 7.0)
    assert line.get_xdata()[0] == pytest.approx(7.0, abs=0.05)
    _key(plot, 'escape')
    assert line.get_xdata()[0] == 6.0
    assert escapes == [()]
    _mouse(plot, 'button_release_event', 7.0)
    assert moved == []
