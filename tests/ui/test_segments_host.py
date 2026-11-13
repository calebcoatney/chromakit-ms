"""ParametersFrame hosts the Segments table and the segment editor (design 2026-11-12 §4.2)."""
import copy

import pytest

pytest.importorskip('pytestqt')

from logic import segments as S
from logic.method import ChromaMethod
from ui.frames.parameters import ParametersFrame

W1 = {'id': 'w1', 'label': 'Light', 'start': 1.0, 'end': 4.0,
      'peaks': {'min_prominence': 0.02}}
W2 = {'id': 'w2', 'start': 6.0, 'end': 9.0}
SPEC_WINDOW = {'id': 'w1', 'label': 'late cluster', 'start': 12.55, 'end': 13.10,
               'peaks': {'min_prominence': 0.02, 'min_width': 3}}
SPEC_BREAKS = [
    {'time': 12.55, 'anchor': {'window_id': 'w1', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
    {'time': 12.85, 'overrides': {'lambda': 1e4}},
    {'time': 13.10, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {}},
]


def method_of(windows=(), breaks=None, **kw):
    d = {'name': 't', 'signal_type': 'gc', 'peak_windows': list(windows), **kw}
    if breaks is not None:
        d['baseline'] = {'break_points': list(breaks)}
    return ChromaMethod.model_validate(d)


def _frame(qtbot, windows=(), breaks=None, **kw):
    frame = ParametersFrame()
    qtbot.addWidget(frame)
    frame.apply_method(method_of(windows, breaks, **kw))
    return frame


def type_into(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _win(frame, wid):
    return next(w for w in frame.current_params['peak_windows'] if w['id'] == wid)


def _setting_keys(frame):
    return [k for k in frame.section_groups if k != 'segments' and not k.startswith('gcxgc')]


def test_range_filter_group_gone(qtbot):
    f = _frame(qtbot, [W1])
    assert 'range_filters' not in f.section_groups
    assert not hasattr(f, 'range_filter_table')
    assert f.section_groups['segments'] is f.segments_panel


def test_panel_lists_segments(qtbot):
    f = _frame(qtbot, [W1, W2])
    table = f.segments_panel.table
    assert table.rowCount() == 3
    assert [table.item(r, 0).text() for r in range(3)] == ['Whole run (defaults)', 'Light', 'w2']


def test_open_hides_groups_and_shows_editor(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.set_section_visibility({'peak_grouping': False})
    before = {k: g.isHidden() for k, g in f.section_groups.items()}
    states = []
    f.segment_state_changed.connect(states.append)
    f.open_segment('w1')
    assert f.segment_editor is not None and not f.segment_editor.isHidden()
    for key in _setting_keys(f):
        assert f.section_groups[key].isHidden(), key
    for key, g in f.section_groups.items():
        if key.startswith('gcxgc') or key == 'segments':
            assert g.isHidden() == before[key], key
    assert not f.segments_panel.table.isEnabled()
    assert states == [{'id': 'w1', 'label': 'Light', 'start': 1.0, 'end': 4.0, 'partway': []}]
    # visibility asked for while the editor is open applies when it closes
    f.set_section_visibility({'shoulders': False})
    assert f.section_groups['shoulders'].isHidden()
    f.cancel_open_segment()
    assert f.segment_editor is None
    assert states[-1] is None
    before['shoulders'] = True
    assert {k: g.isHidden() for k, g in f.section_groups.items()} == before
    assert f.segments_panel.table.isEnabled()


def test_whole_run_row_closes_editor_with_cancel(qtbot):
    f = _frame(qtbot, [W1, W2])
    snapshot = copy.deepcopy(f.current_params)
    f.open_segment('w1')
    type_into(f.segment_editor.groups['peaks'].fields['min_prominence'].edit, '0.5')
    assert _win(f, 'w1')['peaks']['min_prominence'] == 0.5
    f.segments_panel.open_requested.emit('')
    assert f.segment_editor is None
    assert f.current_params == snapshot


def test_preview_applies_and_emits(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.open_segment('w1')
    states = []
    f.segment_state_changed.connect(states.append)
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        type_into(f.segment_editor.groups['peaks'].fields['min_prominence'].edit, '0.5')
    assert _win(f, 'w1')['peaks']['min_prominence'] == 0.5
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        f.segment_set_edge('end', 5.0)
    assert _win(f, 'w1')['end'] == 5.0
    assert f.current_params['peaks']['range_filters'] == [[1.0, 5.0], [6.0, 9.0]]
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        f.segment_add_partway(2.0)
    assert states[-1] == {'id': 'w1', 'label': 'Light', 'start': 1.0, 'end': 5.0, 'partway': [2.0]}
    assert f.segment_editor is not None  # previews never close the editor


def test_preview_error_shows_in_editor_and_keeps_last_valid(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.open_segment('w1')
    f.segment_set_edge('end', 5.0)
    emitted = []
    f.parameters_changed.connect(emitted.append)
    states = []
    f.segment_state_changed.connect(states.append)
    f.segment_set_edge('end', 7.0)  # overlaps w2
    assert not emitted
    assert _win(f, 'w1')['end'] == 5.0
    ed = f.segment_editor
    assert not ed.error_label.isHidden() and 'overlaps' in ed.error_label.text()
    assert states[-1]['end'] == 7.0  # the draft, for the plot
    f.segment_set_edge('end', 5.5)
    assert ed.error_label.isHidden()
    assert _win(f, 'w1')['end'] == 5.5 and len(emitted) == 1


def test_cancel_restores_snapshot_exactly(qtbot):
    f = _frame(qtbot, [SPEC_WINDOW], SPEC_BREAKS)
    snapshot = copy.deepcopy(f.current_params)
    f.open_segment('w1')
    ed = f.segment_editor
    ed.set_span(12.5, 13.2)
    ed.add_partway(13.0)
    prom = ed.groups['peaks'].fields['min_prominence']
    type_into(prom.edit, '0.7')
    assert _win(f, 'w1')['peaks']['min_prominence'] == 0.7
    type_into(prom.edit, '0.02')
    assert f.current_params != snapshot
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        ed.cancel_button.click()
    assert f.segment_editor is None
    assert f.current_params == snapshot


def test_new_segment_cancel_leaves_nothing(qtbot):
    f = _frame(qtbot, [W1, W2])
    snapshot = copy.deepcopy(f.current_params)
    f.open_segment(None, (12.0, 10.0))  # a right-to-left drag
    ed = f.segment_editor
    assert (ed.from_edit.value(), ed.to_edit.value()) == (10.0, 12.0)
    assert f.current_params == snapshot  # not in the method until a preview or Save
    assert ed.delete_button.isHidden()
    type_into(ed.groups['peaks'].fields['min_prominence'].edit, '0.5')
    assert [w['id'] for w in f.current_params['peak_windows']] == ['w1', 'w2', 'w3']
    f.cancel_open_segment()
    assert f.current_params == snapshot


def test_new_segment_save_adds_it(qtbot):
    f = _frame(qtbot, [W2])
    f.open_segment(None, (1.0, 2.0))
    assert f.commit_open_segment() is True
    assert f.segment_editor is None
    assert [(w['id'], w['start'], w['end']) for w in f.current_params['peak_windows']] == [
        ('w1', 1.0, 2.0), ('w2', 6.0, 9.0)]
    assert f.segments_panel.table.rowCount() == 3


def test_save_invalid_keeps_editor_open(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.open_segment('w1')
    f.segment_set_edge('end', 7.0)  # overlaps w2
    ed = f.segment_editor
    ed.save_button.click()
    assert f.segment_editor is ed
    assert 'overlaps' in ed.error_label.text()
    assert f.commit_open_segment() is False
    assert f.segment_editor is ed
    assert _win(f, 'w1')['end'] == 4.0
    f.segment_set_edge('end', 5.0)
    assert f.commit_open_segment() is True
    assert f.segment_editor is None
    assert _win(f, 'w1')['end'] == 5.0


def test_commit_with_bad_text_keeps_editor_open(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.open_segment('w1')
    type_into(f.segment_editor.groups['peaks'].fields['min_width'].edit, 'abc')
    assert f.commit_open_segment() is False
    assert f.segment_editor.error_label.text() == 'Fix the fields marked in red before saving'


def test_commit_with_nothing_open_is_true(qtbot):
    f = _frame(qtbot, [W1])
    assert f.commit_open_segment() is True


def test_editor_delete_removes_segment(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.open_segment('w1')
    type_into(f.segment_editor.groups['peaks'].fields['min_prominence'].edit, '0.5')
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        f.segment_editor.delete_button.click()
    assert f.segment_editor is None
    assert [w['id'] for w in f.current_params['peak_windows']] == ['w2']
    assert f.current_params['peaks']['range_filters'] == [[6.0, 9.0]]
    assert f.segments_panel.table.rowCount() == 2
    assert not any(f.section_groups[k].isHidden() for k in ('smoothing', 'peaks'))


def test_panel_delete_removes_selected_segment(qtbot):
    f = _frame(qtbot, [W1, W2])
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        f.segments_panel.delete_requested.emit('w2')
    assert [w['id'] for w in f.current_params['peak_windows']] == ['w1']
    assert f.segments_panel.table.rowCount() == 2


def test_add_button_opens_new_segment_after_last(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.segments_panel.add_button.click()
    ed = f.segment_editor
    assert (ed.from_edit.value(), ed.to_edit.value()) == (9.0, 10.0)
    empty = _frame(qtbot)
    empty.segments_panel.add_button.click()
    assert (empty.segment_editor.from_edit.value(), empty.segment_editor.to_edit.value()) == (0.0, 60.0)


def test_apply_method_closes_open_segment(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.open_segment('w1')
    type_into(f.segment_editor.groups['peaks'].fields['min_prominence'].edit, '0.5')
    states, emitted = [], []
    f.segment_state_changed.connect(states.append)
    f.parameters_changed.connect(emitted.append)
    other = method_of([{'id': 'w1', 'start': 20.0, 'end': 21.0}])
    f.apply_method(other)
    assert f.segment_editor is None
    assert states == [None]
    assert not emitted
    assert f.current_params == other.to_gui_params()
    assert f.segments_panel.table.isEnabled()
    for key in _setting_keys(f):
        assert not f.section_groups[key].isHidden(), key


def test_gcxgc_switch_closes_open_segment(qtbot):
    f = _frame(qtbot, [W1, W2])
    snapshot = copy.deepcopy(f.current_params)
    f.open_segment('w1')
    type_into(f.segment_editor.groups['peaks'].fields['min_prominence'].edit, '0.5')
    f.set_mode('gcxgc')
    assert f.segment_editor is None
    assert f.current_params == snapshot
    assert f.section_groups['segments'].isHidden()
    for key in _setting_keys(f):
        assert f.section_groups[key].isHidden(), key
    f.set_mode('gc')
    for key in _setting_keys(f):
        assert not f.section_groups[key].isHidden(), key


def test_set_mode_1d_while_open_keeps_groups_hidden(qtbot):
    f = _frame(qtbot, [W1])
    f.open_segment('w1')
    f.set_mode('gc')
    for key in _setting_keys(f):
        assert f.section_groups[key].isHidden(), key
    assert not f.section_groups['segments'].isHidden()
    f.cancel_open_segment()
    for key in _setting_keys(f):
        assert not f.section_groups[key].isHidden(), key


def test_open_cancel_is_identity_on_foreign_method(qtbot):
    """Review Focus 1: an unanchored break at a window start and a custom end break."""
    f = _frame(qtbot,
               [{'id': 'w1', 'start': 2.0, 'end': 3.0}, {'id': 'w2', 'start': 5.0, 'end': 6.0}],
               [{'time': 2.0, 'overrides': {'lambda': 1e3}},
                {'time': 3.0, 'anchor': {'window_id': 'w1', 'edge': 'end'},
                 'overrides': {'lambda': 1e7}}])
    snapshot = copy.deepcopy(f.current_params)
    foreign = copy.deepcopy(f.current_params['baseline']['break_points'])
    f.open_segment('w1')
    assert f.current_params == snapshot
    f.cancel_open_segment()
    assert f.current_params == snapshot
    f.open_segment('w2')
    type_into(f.segment_editor.groups['peaks'].fields['min_prominence'].edit, '0.5')
    assert f.current_params['baseline']['break_points'] == foreign
    f.cancel_open_segment()
    assert f.current_params == snapshot


def test_whole_run_break_table_lists_only_outside_breaks(qtbot):
    f = _frame(qtbot, [SPEC_WINDOW],
               SPEC_BREAKS + [{'time': 20.0, 'tolerance': 0.2, 'overrides': {'lambda': 55.0}}])
    table = f.break_point_table
    assert table.rowCount() == 1
    assert table.cellWidget(0, 0).value() == 20.0
    owned = [bp for bp in copy.deepcopy(f.current_params['baseline']['break_points'])
             if bp['time'] != 20.0]
    assert len(owned) == 3
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        table.cellWidget(0, 0).setValue(21.0)
    bps = f.current_params['baseline']['break_points']
    whole = [bp for bp in bps if bp['time'] == 21.0]
    assert whole and whole[0]['overrides']['lambda'] == 55.0 and whole[0]['tolerance'] == 0.2
    assert [bp for bp in bps if bp['time'] != 21.0] == owned


def test_whole_run_break_inside_segment_rejected(qtbot):
    f = _frame(qtbot, [W1], [{'time': 6.0}])
    emitted = []
    f.parameters_changed.connect(emitted.append)
    spin = f.break_point_table.cellWidget(0, 0)
    spin.setValue(2.0)
    assert not emitted
    assert [b['time'] for b in f.current_params['baseline']['break_points']] == [6.0]
    assert 'red' in spin.styleSheet() and 'Light' in spin.toolTip()
    spin.setValue(5.0)
    assert spin.styleSheet() == ''
    assert [b['time'] for b in f.current_params['baseline']['break_points']] == [5.0]


def test_segment_save_over_whole_run_break_refused(qtbot):
    """R21: a new segment may not take over a whole-run baseline change."""
    f = _frame(qtbot, [W1], [{'time': 6.0}])
    f.open_segment(None, (5.0, 7.0))
    assert f.commit_open_segment() is False
    assert ('Whole-run baseline change at 6 min is inside segment'
            in f.segment_editor.error_label.text())
    f.segment_set_edge('end', 5.5)
    assert f.commit_open_segment() is True
    assert [w['end'] for w in f.current_params['peak_windows']] == [4.0, 5.5]
    assert f.break_point_table.rowCount() == 1   # still a whole-run change


def test_outside_combo_sets_mode_and_mirror(qtbot):
    f = _frame(qtbot, [W1, W2])
    combo = f.segments_panel.outside_combo
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        combo.setCurrentIndex(combo.findData('global'))
    assert f.current_params['outside_windows'] == 'global'
    assert f.current_params['peaks']['range_filters'] == []
    with qtbot.waitSignal(f.parameters_changed, timeout=500):
        combo.setCurrentIndex(combo.findData('ignore'))
    assert f.current_params['outside_windows'] == 'ignore'
    assert f.current_params['peaks']['range_filters'] == [[1.0, 4.0], [6.0, 9.0]]


def test_outside_combo_shows_loaded_mode(qtbot):
    f = _frame(qtbot, [W1], outside_windows='global')
    assert f.segments_panel.outside_combo.currentData() == 'global'


def test_set_segment_warnings_marks_rows_and_editor(qtbot):
    f = _frame(qtbot, [W1, W2])
    f.set_segment_warnings([(7.0, 'Baseline change at 7 falls inside a peak')])
    assert f.segments_panel.table.item(2, 0).text() == '⚠ w2'
    f.open_segment('w2')
    assert 'falls inside a peak' in f.segment_editor.warning_label.text()
    f.set_segment_warnings([])
    assert f.segment_editor.warning_label.isHidden()
    assert f.segments_panel.table.item(2, 0).text() == 'w2'


def test_from_across_whole_run_break_updates_resume(qtbot):
    """R12: moving From across a whole-run break changes what partway rows inherit."""
    f = _frame(qtbot, [{'id': 'w1', 'start': 5.0, 'end': 8.0}],
               [{'time': 4.0, 'overrides': {'lambda': 1e3}}])
    f.open_segment('w1')
    f.segment_add_partway(6.0)
    row = f.segment_editor.partway.rows[0]
    assert row.fields['lambda']._inherited == 1e3
    f.segment_set_edge('start', 3.0)
    assert row.fields['lambda']._inherited == f.current_params['baseline']['lambda']


def test_segments_help_icons(qtbot):
    from PySide6.QtWidgets import QLabel
    f = _frame(qtbot, [W1])
    labels = f.segments_panel.findChildren(QLabel)
    assert any(lbl.property('_has_help') and 'Outside segments' in lbl.text() for lbl in labels)
    assert f.break_point_table.parent().findChild(QLabel).text().startswith('Baseline changes (whole run)')


# ── app ───────────────────────────────────────────────────────────────────────

def _app(qtbot):
    from ui.app import ChromaKitApp
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_x = app.current_y = None
    return app


def _stub_save_as(monkeypatch, out):
    monkeypatch.setattr("ui.app.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(out), "ChromaKit Method (*.chromethod)"))


def test_save_method_commits_open_segment(qtbot, tmp_path, monkeypatch):
    app = _app(qtbot)
    pf = app.parameters_frame
    pf.open_segment(None, (2.0, 3.0))
    out = tmp_path / "seg.chromethod"
    _stub_save_as(monkeypatch, out)
    app.save_method()
    assert pf.segment_editor is None
    saved = ChromaMethod.from_file(out)
    assert [(w.start, w.end) for w in saved.peak_windows] == [(2.0, 3.0)]


def test_save_method_aborts_when_segment_invalid(qtbot, tmp_path, monkeypatch):
    app = _app(qtbot)
    pf = app.parameters_frame
    pf.open_segment(None, (2.0, 3.0))
    assert pf.commit_open_segment()
    pf.open_segment(None, (5.0, 6.0))
    pf.segment_set_edge('start', 2.5)  # overlaps the first
    out = tmp_path / "bad.chromethod"
    _stub_save_as(monkeypatch, out)
    app.save_method()
    assert not out.exists()
    assert pf.segment_editor is not None
    assert 'segment' in app.status_bar.currentMessage().lower()


def test_whole_run_row_summary_follows_whole_run_edits(qtbot):
    f = _frame(qtbot, [W1])
    cell = lambda: f.segments_panel.table.item(0, 3).text()
    assert 'λ 10000' in cell()
    f.lam_spinbox.setValue(6)
    assert 'λ 1e+06' in cell()


def test_range_filters_without_windows_warn(qtbot, caplog):
    m = ChromaMethod(name='t', signal_type='gc')
    m.peaks.range_filters = [[1.0, 2.0]]  # no file load gives this: _migrate_v1 builds windows
    f = ParametersFrame()
    qtbot.addWidget(f)
    with caplog.at_level('WARNING', logger='ui.frames.parameters'):
        f.apply_method(m)
    assert 'range_filters' in caplog.text and 'no peak_windows' in caplog.text


def test_panel_refreshes_once_per_segment_change(qtbot, monkeypatch):
    f = _frame(qtbot, [W1, W2])
    calls = []
    real = f.segments_panel.set_view
    monkeypatch.setattr(f.segments_panel, 'set_view', lambda *a: (calls.append(1), real(*a)))
    f.open_segment('w1')
    f.segment_set_edge('end', 5.0)
    assert len(calls) == 1
    f.commit_open_segment()
    assert len(calls) == 2
    f.segments_panel.delete_requested.emit('w2')
    assert len(calls) == 3
    combo = f.segments_panel.outside_combo
    combo.setCurrentIndex(combo.findData('global'))
    assert len(calls) == 4


# ── final fix wave #1: an out-of-range override blocks preview and Save ─────────

def test_out_of_range_override_blocks_preview_and_save(qtbot):
    f = _frame(qtbot, [W1, W2])
    before = copy.deepcopy(f.current_params)
    emitted = []
    f.parameters_changed.connect(emitted.append)
    f.open_segment('w1')
    ed = f.segment_editor
    type_into(ed.groups['shoulders'].fields['sensitivity'].edit, '20')
    assert not emitted
    assert f.current_params == before
    assert not ed.error_label.isHidden()
    assert 'sensitivity' in ed.error_label.text() and '1 to 10' in ed.error_label.text()
    ed.save_button.click()
    assert f.segment_editor is ed
    assert f.commit_open_segment() is False
    assert f.segment_editor is ed and f.current_params == before
    assert 'sensitivity' in ed.error_label.text()
    type_into(ed.groups['shoulders'].fields['sensitivity'].edit, '7')
    assert ed.error_label.isHidden()
    assert _win(f, 'w1')['shoulders']['sensitivity'] == 7
    assert f.commit_open_segment() is True


def _method_slice(m):
    return ([w.model_dump(by_alias=True) for w in m.peak_windows],
            [b.model_dump(by_alias=True) for b in m.baseline.break_points or []])


def _assert_method_follows_gui(app):
    pf = app.parameters_frame
    rebuilt = ChromaMethod.from_gui_params(pf.current_params, name=app.current_method.name,
                                           signal_type=app.current_method.signal_type)
    assert _method_slice(rebuilt) == _method_slice(app.current_method)


def test_current_method_matches_gui_after_segment_save_and_cancel(qtbot):
    app = _app(qtbot)
    pf = app.parameters_frame
    pf.open_segment(None, (2.0, 3.0))
    ed = pf.segment_editor
    ed.fresh_check.setChecked(True)
    type_into(ed.groups['baseline'].fields['lambda'].edit, '1e3')
    type_into(ed.groups['shoulders'].fields['sensitivity'].edit, '20')   # refused
    type_into(ed.groups['shoulders'].fields['sensitivity'].edit, '4')
    assert pf.commit_open_segment()
    _assert_method_follows_gui(app)
    assert [w.shoulders.sensitivity for w in app.current_method.peak_windows] == [4]
    pf.open_segment('w1')
    pf.segment_set_edge('end', 3.5)
    pf.segment_add_partway(3.0)
    type_into(pf.segment_editor.groups['shoulders'].fields['sensitivity'].edit, '20')
    pf.cancel_open_segment()
    _assert_method_follows_gui(app)
    assert [(w.start, w.end) for w in app.current_method.peak_windows] == [(2.0, 3.0)]


def test_params_writeback_failure_is_loud(qtbot, caplog):
    app = _app(qtbot)
    before = app.current_method.model_dump()
    params = copy.deepcopy(app.parameters_frame.current_params)
    params['peak_windows'] = [{'id': 'w1', 'label': '', 'start': 2.0, 'end': 3.0,
                               'shoulders': {'sensitivity': 20}}]
    with caplog.at_level('ERROR', logger='ui.app'):
        app._on_params_writeback(params)     # must not raise out of the slot
    assert app.current_method.model_dump() == before
    assert 'sensitivity' in caplog.text
    assert 'not' in app.status_bar.currentMessage().lower()


# ── final fix wave #2 (R22): no batch or automation while a segment is open ─────

OPEN_FIRST = 'Save or cancel the open segment first'


def _app_with_open_draft(qtbot, tmp_path):
    app = _app(qtbot)
    app.data_handler.current_directory_path = str(tmp_path / 'run.D')
    pf = app.parameters_frame
    pf.open_segment(None, (2.0, 3.0))
    pf.segment_set_edge('end', 4.0)      # previewed: the draft is in current_method
    return app


def test_batch_refused_while_segment_open(qtbot, tmp_path, monkeypatch):
    app = _app_with_open_draft(qtbot, tmp_path)
    monkeypatch.setattr(app, 'process_next_batch_directory',
                        lambda: pytest.fail('batch started with a segment open'))
    app.start_batch_processing([str(tmp_path / 'a.D')], {'detector_runs': [{'detector': 'FID'}]})
    assert app.status_bar.currentMessage() == OPEN_FIRST
    assert getattr(app, '_batch_saved_method_state', None) is None
    assert getattr(app, 'batch_dialog', None) is None
    assert app.parameters_frame.segment_editor is not None


def test_batch_dialog_refused_while_segment_open(qtbot, tmp_path, monkeypatch):
    app = _app_with_open_draft(qtbot, tmp_path)
    monkeypatch.setattr('ui.dialogs.batch_job_dialog.BatchJobDialog',
                        lambda *a, **k: pytest.fail('batch dialog shown with a segment open'))
    app.show_batch_queue_dialog()
    assert app.status_bar.currentMessage() == OPEN_FIRST


@pytest.mark.parametrize('entry', ['start_automation', 'on_automation_clicked'])
def test_automation_refused_while_segment_open(qtbot, tmp_path, monkeypatch, entry):
    app = _app_with_open_draft(qtbot, tmp_path)
    monkeypatch.setattr('ui.app.AutomationWorker',
                        lambda *a, **k: pytest.fail('automation started with a segment open'))
    getattr(app, entry)()
    assert app.status_bar.currentMessage() == OPEN_FIRST
    assert app.parameters_frame.segment_editor is not None


# ── final fix wave #4: Cancel with nothing previewed changes nothing ────────────

def test_cancel_without_preview_emits_nothing(qtbot):
    f = _frame(qtbot, [W1, W2])
    snapshot = copy.deepcopy(f.current_params)
    emitted = []
    f.parameters_changed.connect(emitted.append)
    f.open_segment('w1')
    f.cancel_open_segment()
    f.open_segment('w1')
    f.segment_set_edge('end', 7.0)       # refused (overlaps w2): nothing applied
    f.cancel_open_segment()
    assert emitted == [] and f.current_params == snapshot
    assert f.segments_panel.table.isEnabled()
    f.open_segment('w1')
    f.segment_set_edge('end', 5.0)       # applied
    f.cancel_open_segment()
    assert len(emitted) == 2 and f.current_params == snapshot


def test_cancel_without_edits_leaves_method_clean(qtbot):
    app = _app(qtbot)
    app._mark_dirty(False)
    pf = app.parameters_frame
    pf.open_segment(None, (2.0, 3.0))
    pf.cancel_open_segment()
    assert app._method_dirty is False
