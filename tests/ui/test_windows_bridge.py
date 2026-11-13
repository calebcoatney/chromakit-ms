"""Segments and the whole-run break table edit peak_windows and break_points in place,
so per-window and per-break overrides loaded from a method survive."""
import pytest

pytest.importorskip('pytestqt')

from logic.method import ChromaMethod
from ui.frames.parameters import ParametersFrame


def _frame(qtbot, windows=(), breaks=()):
    method = ChromaMethod.model_validate({
        'name': 't', 'signal_type': 'gc',
        'peak_windows': list(windows),
        'baseline': {'break_points': list(breaks)},
    })
    frame = ParametersFrame()
    qtbot.addWidget(frame)
    frame.apply_method(method)
    return frame


W1 = {'id': 'w1', 'label': 'Light', 'start': 1.0, 'end': 4.0,
      'peaks': {'min_prominence': 0.02}}
W2 = {'id': 'w2', 'start': 6.0, 'end': 9.0}


def _win(frame, wid):
    return next(w for w in frame.current_params['peak_windows'] if w['id'] == wid)


def _move(frame, wid, start, end):
    frame.open_segment(wid)
    frame.segment_editor.set_span(start, end)
    return frame.commit_open_segment()


def test_segment_span_edit_keeps_window_overrides(qtbot):
    f = _frame(qtbot, [W1, W2])
    assert _move(f, 'w1', 1.0, 5.0)
    w1 = _win(f, 'w1')
    assert w1['end'] == 5.0
    assert w1['peaks']['min_prominence'] == 0.02
    assert w1['label'] == 'Light'
    assert f.current_params['peaks']['range_filters'] == [[1.0, 5.0], [6.0, 9.0]]


def test_add_and_delete_segments_map_to_windows(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.segments_panel.delete_requested.emit('w1')
    assert [w['id'] for w in f.current_params['peak_windows']] == ['w2']
    f.open_segment(None, (10.0, 12.0))
    assert f.commit_open_segment()
    ids = [w['id'] for w in f.current_params['peak_windows']]
    assert ids == ['w2', 'w1']  # smallest unused n
    assert _win(f, 'w1')['start'] == 10.0 and _win(f, 'w1')['label'] == ''
    assert f.current_params['peaks']['range_filters'] == [[6.0, 9.0], [10.0, 12.0]]


def test_overlapping_edit_rejected_and_flagged(qtbot):
    f = _frame(qtbot, [W1, W2])
    assert not _move(f, 'w1', 1.0, 7.0)
    assert _win(f, 'w1')['end'] == 4.0
    msg = f.segment_editor.error_label.text()
    assert 'Light' in msg and 'w2' in msg
    # fixing the edit clears the flag
    f.segment_editor.set_span(1.0, 5.0)
    assert f.segment_editor.error_label.isHidden()
    assert f.commit_open_segment()
    assert _win(f, 'w1')['end'] == 5.0


def test_break_row_edit_keeps_overrides(qtbot):
    end = {'time': 4.0, 'tolerance': 0.1, 'overrides': {'lambda': 123.0},
           'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [end, {'time': 5.0, 'overrides': {'lambda': 55.0}}])
    assert f.break_point_table.rowCount() == 1  # only the whole-run break
    f.break_point_table.cellWidget(0, 0).setValue(5.5)
    f.break_point_table.cellWidget(0, 1).setValue(0.3)
    bps = f.current_params['baseline']['break_points']
    assert bps[0]['time'] == 4.0 and bps[0]['overrides']['lambda'] == 123.0
    assert bps[0]['anchor'] == {'window_id': 'w1', 'edge': 'end'}
    assert bps[1]['time'] == 5.5 and bps[1]['tolerance'] == 0.3
    assert bps[1]['overrides']['lambda'] == 55.0


def test_delete_segment_drops_its_anchored_breaks(qtbot):
    start = {'time': 1.0, 'overrides': {'lambda': 1e3}, 'anchor': {'window_id': 'w1', 'edge': 'start'}}
    end = {'time': 4.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [start, end, {'time': 5.0}])
    f.segments_panel.delete_requested.emit('w1')
    assert [b['time'] for b in f.current_params['baseline']['break_points']] == [5.0]
    assert f.break_point_table.rowCount() == 1


def test_overrides_note_shown_only_with_overrides(qtbot):
    f = _frame(qtbot, [W1])
    assert any(n.isVisibleTo(f) for n in f._overrides_notes if not n.isHidden())
    plain = _frame(qtbot, [W2])
    assert all(n.isHidden() for n in plain._overrides_notes if n.parent() is not None
               and plain.isAncestorOf(n))


def test_anchored_break_follows_window_span_edit(qtbot):
    start = {'time': 1.0, 'overrides': {'lambda': 123.0}, 'anchor': {'window_id': 'w1', 'edge': 'start'}}
    end = {'time': 4.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [start, end])
    assert _move(f, 'w1', 1.0, 5.0)
    bps = f.current_params['baseline']['break_points']
    assert [(b['time'], b['anchor']['edge']) for b in bps] == [(1.0, 'start'), (5.0, 'end')]
    assert _move(f, 'w1', 0.5, 5.0)  # a start edge change leaves the end break alone
    bps = f.current_params['baseline']['break_points']
    assert [(b['time'], b['anchor']['edge']) for b in bps] == [(0.5, 'start'), (5.0, 'end')]
    assert bps[0]['overrides']['lambda'] == 123.0


def test_duplicate_break_edit_rejected_and_flagged(qtbot):
    f = _frame(qtbot, breaks=[{'time': 3.0}, {'time': 6.0}])
    emitted = []
    f.parameters_changed.connect(emitted.append)
    f.break_point_table.cellWidget(1, 0).setValue(3.0)
    assert [b['time'] for b in f.current_params['baseline']['break_points']] == [3.0, 6.0]
    assert not emitted
    for r in (0, 1):
        spin = f.break_point_table.cellWidget(r, 0)
        assert 'red' in spin.styleSheet() and 'Duplicate' in spin.toolTip()
    f.break_point_table.cellWidget(1, 0).setValue(7.0)
    assert f.break_point_table.cellWidget(1, 0).styleSheet() == ''
    assert f.current_params['baseline']['break_points'][1]['time'] == 7.0


def test_break_on_segment_end_break_rejected_and_flagged(qtbot):
    end = {'time': 4.0, 'overrides': {'lambda': 9.0}, 'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1], [end, {'time': 6.0}])
    spin = f.break_point_table.cellWidget(0, 0)
    spin.setValue(4.0)
    assert 'red' in spin.styleSheet() and 'Duplicate' in spin.toolTip()
    assert [b['time'] for b in f.current_params['baseline']['break_points']] == [4.0, 6.0]


def test_new_segment_defaults_after_last_window(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.segments_panel.add_button.click()
    assert f.commit_open_segment()
    assert [w['end'] for w in f.current_params['peak_windows']] == [4.0, 9.0, 10.0]
    assert _win(f, 'w3')['label'] == ''


def test_new_segment_defaults_to_full_run_when_no_windows(qtbot):
    f = _frame(qtbot)
    f.segments_panel.add_button.click()
    assert (f.segment_editor.from_edit.value(), f.segment_editor.to_edit.value()) == (0.0, 60.0)
