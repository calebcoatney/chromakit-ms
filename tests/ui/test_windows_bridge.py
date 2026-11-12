"""The Peak Range Filters and Signal Break Points tables edit peak_windows and
break_points in place, so per-window overrides loaded from a method survive."""
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


def test_range_row_edit_keeps_window_overrides(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.range_filter_table.cellWidget(0, 1).setValue(5.0)
    w1 = _win(f, 'w1')
    assert w1['end'] == 5.0
    assert w1['peaks']['min_prominence'] == 0.02
    assert w1['label'] == 'Light'
    assert f.current_params['peaks']['range_filters'] == [[1.0, 5.0], [6.0, 9.0]]


def test_add_and_delete_rows_map_to_windows(qtbot):
    f = _frame(qtbot, [W1, W2])
    f._remove_range_filter_row(0)
    assert [w['id'] for w in f.current_params['peak_windows']] == ['w2']
    f._add_range_filter_row()
    f.range_filter_table.cellWidget(1, 0).setValue(10.0)
    f.range_filter_table.cellWidget(1, 1).setValue(12.0)
    ids = [w['id'] for w in f.current_params['peak_windows']]
    assert ids == ['w2', 'w1']  # smallest unused n
    assert _win(f, 'w1')['start'] == 10.0
    assert f.current_params['peaks']['range_filters'] == [[6.0, 9.0], [10.0, 12.0]]


def test_overlapping_edit_rejected_and_flagged(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.range_filter_table.cellWidget(0, 1).setValue(7.0)
    assert _win(f, 'w1')['end'] == 4.0
    spin = f.range_filter_table.cellWidget(0, 1)
    assert 'red' in spin.styleSheet()
    assert 'w1' in spin.toolTip() and 'w2' in spin.toolTip()
    # fixing the edit clears the flag
    spin.setValue(5.0)
    assert spin.styleSheet() == '' and spin.toolTip() == ''
    assert _win(f, 'w1')['end'] == 5.0


def test_break_row_edit_keeps_overrides_and_anchor(qtbot):
    bp = {'time': 4.0, 'tolerance': 0.1, 'overrides': {'lambda': 123.0},
          'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [bp, {'time': 7.0, 'overrides': {'lambda': 55.0}}])
    f.break_point_table.cellWidget(1, 0).setValue(7.5)
    f.break_point_table.cellWidget(0, 1).setValue(0.3)
    bps = f.current_params['baseline']['break_points']
    assert bps[0]['tolerance'] == 0.3
    assert bps[0]['overrides']['lambda'] == 123.0
    assert bps[0]['anchor'] == {'window_id': 'w1', 'edge': 'end'}
    assert bps[1]['time'] == 7.5 and bps[1]['overrides']['lambda'] == 55.0


def test_anchored_break_time_disabled(qtbot):
    bp = {'time': 4.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [bp, {'time': 7.0}])
    anchored = f.break_point_table.cellWidget(0, 0)
    assert not anchored.isEnabled()
    assert anchored.toolTip() == "Follows window 'Light' end"
    assert f.break_point_table.cellWidget(1, 0).isEnabled()


def test_delete_window_clears_anchor_keeps_time(qtbot, caplog):
    bp = {'time': 4.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [bp])
    with caplog.at_level('WARNING'):
        f._remove_range_filter_row(0)
    out = f.current_params['baseline']['break_points'][0]
    assert out['anchor'] is None and out['time'] == 4.0
    assert f.break_point_table.cellWidget(0, 0).isEnabled()
    assert 'Light' in caplog.text
    assert not f.windows_status.isHidden()


def test_overrides_note_shown_only_with_overrides(qtbot):
    f = _frame(qtbot, [W1])
    assert any(n.isVisibleTo(f) for n in f._overrides_notes if not n.isHidden())
    plain = _frame(qtbot, [W2])
    assert all(n.isHidden() for n in plain._overrides_notes if n.parent() is not None
               and plain.isAncestorOf(n))


def test_anchored_break_follows_window_span_edit(qtbot):
    bp = {'time': 4.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}}
    f = _frame(qtbot, [W1, W2], [bp])
    f.range_filter_table.cellWidget(0, 1).setValue(5.0)
    assert f.current_params['baseline']['break_points'][0]['time'] == 5.0
    assert f.break_point_table.cellWidget(0, 0).value() == 5.0
    f.range_filter_table.cellWidget(0, 0).setValue(0.5)  # start edge change leaves an end anchor alone
    assert f.current_params['baseline']['break_points'][0]['time'] == 5.0


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
