"""ui.frames.segments: the Segments table and the segment editor (design §4.2)."""
import copy

import pytest
from PySide6.QtCore import Qt
pytest.importorskip('pytestqt')

from logic import segments as S
from logic.method import ChromaMethod
from ui.frames.segments import SegmentEditor, SegmentsPanel

SPEC_WINDOW = {'id': 'w1', 'label': 'late cluster', 'start': 12.55, 'end': 13.10,
               'peaks': {'min_prominence': 0.02, 'min_width': 3}}
SPEC_BREAKS = [
    {'time': 12.55, 'anchor': {'window_id': 'w1', 'edge': 'start'}, 'overrides': {'lambda': 1e3}},
    {'time': 12.85, 'overrides': {'lambda': 1e4}},
    {'time': 13.10, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {}},
]
SECTIONS = ('baseline',) + S.SEGMENT_SECTIONS


def params_of(windows=(), breaks=None):
    d = {'name': 't', 'signal_type': 'gc', 'peak_windows': list(windows)}
    if breaks is not None:
        d['baseline'] = {'break_points': breaks}
    return ChromaMethod.model_validate(d).to_gui_params()


@pytest.fixture
def spec():
    p = params_of([SPEC_WINDOW], SPEC_BREAKS)
    return p, S.segments_view(p)['segments'][0]


def make_editor(qtbot, params, record, deconvolution=False, is_new=False, resume=None):
    inherited = {s: copy.deepcopy(params[s]) for s in SECTIONS}
    if resume is None:
        resume = S.resume_overrides(params, record['start'])
    ed = SegmentEditor(record, inherited, resume, deconvolution, is_new)
    qtbot.addWidget(ed)
    return ed


def type_into(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


# ── SegmentsPanel ─────────────────────────────────────────────────────────────

def make_panel(qtbot, params):
    panel = SegmentsPanel()
    qtbot.addWidget(panel)
    panel.set_view(S.segments_view(params), 'asls λ 1e6 · prom 0.05', 'ignore')
    return panel


def test_panel_rows(qtbot, spec):
    p, rec = spec
    panel = make_panel(qtbot, p)
    t = panel.table
    assert t.rowCount() == 2
    assert t.item(0, 0).text() == 'Whole run (defaults)'
    assert t.item(0, 3).text() == 'asls λ 1e6 · prom 0.05'
    assert t.item(1, 0).text() == 'late cluster'
    assert (t.item(1, 1).text(), t.item(1, 2).text()) == ('12.55', '13.1')
    assert t.item(1, 3).text() == S.summarize(rec) == 'λ 1e3→1e4 · prom 0.02 · width 3'


def test_panel_set_editing_disables_table_and_buttons(qtbot, spec):
    panel = make_panel(qtbot, spec[0])
    panel.set_editing(True)
    assert not panel.table.isEnabled()
    assert not panel.add_button.isEnabled()
    assert not panel.delete_button.isEnabled()
    panel.set_editing(False)
    assert panel.table.isEnabled() and panel.add_button.isEnabled()


def test_panel_click_row_emits_open(qtbot, spec):
    panel = make_panel(qtbot, spec[0])
    with qtbot.waitSignal(panel.open_requested, timeout=500) as blocker:
        panel.table.cellClicked.emit(1, 0)
    assert blocker.args == ['w1']
    with qtbot.waitSignal(panel.open_requested, timeout=500) as blocker:
        panel.table.cellClicked.emit(0, 2)
    assert blocker.args == ['']


def test_panel_add_and_delete(qtbot, spec):
    panel = make_panel(qtbot, spec[0])
    with qtbot.waitSignal(panel.add_requested, timeout=500):
        qtbot.mouseClick(panel.add_button, Qt.LeftButton)
    panel.table.selectRow(0)
    assert not panel.delete_button.isEnabled()  # Whole run can't be deleted
    panel.table.selectRow(1)
    assert panel.delete_button.isEnabled()
    with qtbot.waitSignal(panel.delete_requested, timeout=500) as blocker:
        qtbot.mouseClick(panel.delete_button, Qt.LeftButton)
    assert blocker.args == ['w1']


def test_panel_outside_combo(qtbot, spec):
    panel = SegmentsPanel()
    qtbot.addWidget(panel)
    with qtbot.assertNotEmitted(panel.outside_changed):
        panel.set_view(S.segments_view(spec[0]), '', 'global')
    assert panel.outside_combo.currentData() == 'global'
    panel.set_view(S.segments_view(spec[0]), '', 'ignore')
    with qtbot.waitSignal(panel.outside_changed, timeout=500) as blocker:
        panel.outside_combo.setCurrentIndex(panel.outside_combo.findData('global'))
    assert blocker.args == ['global']


def test_panel_row_warning_tooltip_survives_set_view(qtbot, spec):
    panel = make_panel(qtbot, spec[0])
    panel.set_row_warning('w1', 'Baseline change at 12.85 falls inside a peak')
    assert panel.table.item(1, 0).toolTip() == 'Baseline change at 12.85 falls inside a peak'
    panel.set_view(S.segments_view(spec[0]), '', 'ignore')
    assert panel.table.item(1, 3).toolTip() == 'Baseline change at 12.85 falls inside a peak'
    panel.set_row_warning('w1', None)
    assert panel.table.item(1, 0).toolTip() == ''


# ── SegmentEditor ─────────────────────────────────────────────────────────────

def test_editor_roundtrips_spec_record(qtbot, spec):
    p, rec = spec
    before = copy.deepcopy(rec)
    ed = make_editor(qtbot, p, rec)
    assert rec == before
    assert ed.record() == rec
    via_editor = S.apply_segment(p, ed.record())
    via_view = S.apply_segment(p, S.segments_view(p)['segments'][0])
    assert via_editor['peak_windows'] == via_view['peak_windows']
    assert via_editor['baseline']['break_points'] == via_view['baseline']['break_points']


def test_editor_no_preview_during_construction(qtbot, spec, monkeypatch):
    p, rec = spec
    calls = []  # every preview goes through _preview; a signal can't be connected before __init__
    real = SegmentEditor._preview
    monkeypatch.setattr(SegmentEditor, '_preview', lambda self: (calls.append(1), real(self)))
    rec['partway'].append({'time': 12.95, 'overrides': {'baseline_offset': 1.0}})
    rec['after'] = {'mode': 'custom', 'overrides': {'lambda': 1e5}}
    ed = make_editor(qtbot, p, rec, deconvolution=True, is_new=True)
    assert calls == []
    type_into(ed.groups['peaks'].fields['min_height'].edit, '2')
    assert calls == [1]


def test_editor_typing_prominence_previews(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    field = ed.groups['peaks'].fields['min_prominence']
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        type_into(field.edit, '0.5')
    assert blocker.args[0]['peaks']['min_prominence'] == 0.5
    assert ed.record()['peaks'] == {'min_prominence': 0.5, 'min_width': 3}


def test_editor_untick_fresh_disables_baseline(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert ed.fresh_check.isChecked()
    assert ed.fresh_check.text() == 'Start a fresh baseline at 12.55'
    lam = ed.groups['baseline'].fields['lambda']
    assert lam.isEnabled()
    with qtbot.waitSignal(ed.preview_changed, timeout=500):
        ed.fresh_check.setChecked(False)
    assert not lam.isEnabled()
    r = ed.record()
    assert r['fresh_baseline'] is False and r['baseline'] == {}
    ed.fresh_check.setChecked(True)  # values come back
    assert ed.record()['baseline'] == {'lambda': 1e3}


def test_editor_baseline_count_includes_partway_rows(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert ed.groups['baseline'].count_label.text() == '2 changes'
    ed.fresh_check.setChecked(False)
    assert ed.groups['baseline'].count_label.text() == '1 change'


def test_editor_partway_rows(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert len(ed.partway.rows) == 1
    row = ed.partway.rows[0]
    assert set(row.fields) == {'enabled', 'method', 'lambda', 'baseline_offset'}
    assert row.fields['lambda'].value() == 1e4
    assert row.fields['lambda'].tag_label.text() == 'same as segment'
    assert row.fields['method'].combo.itemText(0).startswith('same as segment')
    # inherited = the segment's effective baseline: own values over the whole-run baseline
    assert row.fields['lambda'].edit.placeholderText() == '1000'
    assert row.fields['baseline_offset'].edit.placeholderText() == f"{p['baseline']['baseline_offset']:g}"
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        ed.add_partway(12.9)
    assert len(ed.partway.rows) == 2
    assert blocker.args[0]['partway'][1] == {'time': 12.9, 'overrides': {}}
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        qtbot.mouseClick(ed.partway.rows[0].delete_button, Qt.LeftButton)
    assert blocker.args[0]['partway'] == [{'time': 12.9, 'overrides': {}}]


def test_editor_partway_inherits_resume_when_not_fresh(qtbot):
    p = params_of([{'id': 'w1', 'label': '', 'start': 5.0, 'end': 8.0}],
                  [{'time': 1.0, 'overrides': {'lambda': 1e5}},
                   {'time': 6.0, 'overrides': {'lambda': 1e5, 'baseline_offset': 2.0}}])
    rec = S.segments_view(p)['segments'][0]
    assert rec['fresh_baseline'] is False
    ed = make_editor(qtbot, p, rec)
    row = ed.partway.rows[0]
    assert row.fields['lambda'].edit.placeholderText() == '100000'
    assert row.fields['baseline_offset'].value() == 2.0
    assert ed.record() == rec


def test_editor_change_at_adds_midpoint_row(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        qtbot.mouseClick(ed.partway.add_button, Qt.LeftButton)
    assert ed.partway.add_button.text() == '+ Change at…'
    times = [r['time'] for r in blocker.args[0]['partway']]
    assert times == [pytest.approx((12.55 + 13.10) / 2), 12.85]  # sorted by time


def test_editor_partway_time_edit_previews(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        type_into(ed.partway.rows[0].time_edit, '12.95')
    assert blocker.args[0]['partway'][0]['time'] == 12.95


def test_editor_after_line(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert ed.after_label.text() == 'After 13.1 the whole-run baseline takes over again.'
    assert ed.after_reset.isHidden()

    custom = copy.deepcopy(rec)
    custom['after'] = {'mode': 'custom', 'overrides': {'lambda': 1e5}}
    ed2 = make_editor(qtbot, p, custom, resume={'lambda': 1e6})
    assert ed2.after_label.text() == 'After 13.1: custom (λ 1e5)'
    assert not ed2.after_reset.isHidden() and ed2.after_reset.text() == '↺'
    assert ed2.record()['after'] == custom['after']
    with qtbot.waitSignal(ed2.preview_changed, timeout=500) as blocker:
        qtbot.mouseClick(ed2.after_reset, Qt.LeftButton)
    assert blocker.args[0]['after'] == {'mode': 'resume', 'overrides': {'lambda': 1e6}}
    assert ed2.after_label.text() == 'After 13.1 the whole-run baseline takes over again.'
    assert ed2.after_reset.isHidden()


def test_editor_deconvolution_disables_detection_groups(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec, deconvolution=True)
    for section in ('peaks', 'shoulders', 'rider', 'negative_peaks'):
        assert not ed.groups[section].isEnabled(), section
    for section in ('baseline', 'smoothing'):
        assert ed.groups[section].isEnabled(), section
    assert not ed.deconv_note.isHidden()
    assert ed.deconv_note.text() == 'deconvolution mode: segments filter only'
    assert ed.record() == rec
    plain = make_editor(qtbot, p, rec)
    assert plain.deconv_note.isHidden()


def test_editor_save_and_cancel(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    with qtbot.waitSignal(ed.save_requested, timeout=500) as blocker:
        qtbot.mouseClick(ed.save_button, Qt.LeftButton)
    assert blocker.args == [rec]
    with qtbot.waitSignal(ed.cancel_requested, timeout=500):
        qtbot.mouseClick(ed.cancel_button, Qt.LeftButton)


def test_editor_errors_block_save_and_keep_last_valid(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    type_into(ed.groups['peaks'].fields['min_prominence'].edit, '0.5')
    with qtbot.assertNotEmitted(ed.preview_changed):
        type_into(ed.groups['peaks'].fields['min_width'].edit, 'abc')
    assert ed.has_errors()
    assert ed.record()['peaks'] == {'min_prominence': 0.5, 'min_width': 3}
    with qtbot.assertNotEmitted(ed.save_requested):
        qtbot.mouseClick(ed.save_button, Qt.LeftButton)
    assert not ed.error_label.isHidden()
    assert ed.error_label.text() == 'Fix the fields marked in red before saving'
    with qtbot.waitSignal(ed.preview_changed, timeout=500):
        type_into(ed.groups['peaks'].fields['min_width'].edit, '4')
    assert not ed.has_errors()


def test_editor_bad_span_text_is_an_error(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    with qtbot.assertNotEmitted(ed.preview_changed):
        type_into(ed.from_edit, 'x')
    assert ed.has_errors()
    assert ed.record()['start'] == 12.55
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        type_into(ed.from_edit, '12.6')
    assert blocker.args[0]['start'] == 12.6


def test_editor_set_span(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    seen = []
    ed.preview_changed.connect(seen.append)
    ed.set_span(12.5, 13.2)
    assert len(seen) == 1
    assert (seen[0]['start'], seen[0]['end']) == (12.5, 13.2)
    assert (ed.from_edit.text(), ed.to_edit.text()) == ('12.5', '13.2')
    assert ed.fresh_check.text() == 'Start a fresh baseline at 12.5'
    assert ed.after_label.text().startswith('After 13.2 ')


def test_editor_show_error_and_warnings(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert ed.error_label.isHidden() and ed.warning_label.isHidden()
    ed.show_error("Segment 'late cluster' overlaps segment 'x'")
    assert not ed.error_label.isHidden()
    ed.show_error(None)
    assert ed.error_label.isHidden()
    ed.set_warnings(['one', 'two'])
    assert not ed.warning_label.isHidden()
    assert ed.warning_label.text() == '⚠ one\n⚠ two'
    ed.set_warnings([])
    assert ed.warning_label.isHidden()


def test_editor_header_label(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert ed.title_label.text() == 'Segment: late cluster'
    assert ed.label_edit.text() == 'late cluster'
    new = S.new_segment(p, 1.0, 2.0)
    ed2 = make_editor(qtbot, p, new, is_new=True)
    assert ed2.title_label.text() == 'New segment'
    with qtbot.waitSignal(ed2.preview_changed, timeout=500) as blocker:
        type_into(ed2.label_edit, 'early lights')
    assert blocker.args[0]['label'] == 'early lights'
    assert ed2.title_label.text() == 'Segment: early lights'
    assert ed2.record() == dict(new, label='early lights')
    with qtbot.waitSignal(ed2.cancel_requested, timeout=500):
        qtbot.mouseClick(ed2.cancel_button, Qt.LeftButton)


def test_editor_hint_line(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert ed.hint_label.text() == ('Grey = same as the whole run. Type a value to change it '
                                    'for this segment; ↺ puts it back.')


def test_editor_group_order_and_titles(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    titles = [ed.groups[s].toggle_button.text()[2:] for s in ed.groups]
    assert titles == ['Baseline', 'Peak detection', 'Smoothing', 'Shoulders', 'Tail skim',
                      'Negative peaks']


def test_editor_carries_keys_it_has_no_field_for(qtbot):
    """A file may hold overrides the form doesn't show (asymmetry, skim_chain_min_tail): kept as is."""
    p = params_of([{'id': 'w1', 'label': '', 'start': 5.0, 'end': 8.0,
                    'rider': {'skim_chain_min_tail': 0.2, 'skim_model': 'tangent'}}],
                  [{'time': 5.0, 'anchor': {'window_id': 'w1', 'edge': 'start'},
                    'overrides': {'lambda': 1e3, 'asymmetry': 0.05}},
                   {'time': 6.0, 'overrides': {'lambda': 1e3, 'asymmetry': 0.1}},
                   {'time': 8.0, 'anchor': {'window_id': 'w1', 'edge': 'end'}, 'overrides': {}}])
    rec = S.segments_view(p)['segments'][0]
    assert rec['partway'][0]['overrides'] == {'asymmetry': 0.1}
    ed = make_editor(qtbot, p, rec)
    assert ed.record() == rec
    type_into(ed.groups['rider'].fields['skim_height_ratio'].edit, '3')
    assert ed.record()['rider'] == {'skim_chain_min_tail': 0.2, 'skim_model': 'tangent',
                                    'skim_height_ratio': 3.0}


def test_panel_enter_opens_row(qtbot, spec):
    panel = make_panel(qtbot, spec[0])
    with qtbot.waitSignal(panel.open_requested, timeout=500) as blocker:
        panel.table.cellActivated.emit(1, 0)
    assert blocker.args == ['w1']


def test_editor_clearing_last_error_previews_held_edits(qtbot, spec):
    """Reviewer repro: bad From, then an edit (held), then From fixed to its old value."""
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    with qtbot.assertNotEmitted(ed.preview_changed):
        type_into(ed.from_edit, 'x')
        type_into(ed.groups['peaks'].fields['min_prominence'].edit, '0.5')
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        type_into(ed.from_edit, '12.55')
    assert blocker.args[0]['peaks']['min_prominence'] == 0.5
    assert blocker.args[0]['start'] == 12.55


def test_editor_clearing_bad_text_in_inherited_field_previews(qtbot, spec):
    """An inheriting field with bad text keeps value None, so clearing it changes no value."""
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    with qtbot.assertNotEmitted(ed.preview_changed):
        type_into(ed.groups['peaks'].fields['min_height'].edit, 'abc')
        type_into(ed.groups['peaks'].fields['min_width'].edit, '4')
    seen = []
    ed.preview_changed.connect(seen.append)
    type_into(ed.groups['peaks'].fields['min_height'].edit, '')
    assert len(seen) == 1 and seen[0]['peaks']['min_width'] == 4.0


def test_editor_save_blocked_message_clears_with_errors(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    type_into(ed.to_edit, 'y')
    qtbot.mouseClick(ed.save_button, Qt.LeftButton)
    assert not ed.error_label.isHidden()
    type_into(ed.to_edit, '13.1')
    assert ed.error_label.isHidden()
    # a host's message is not the editor's to clear
    ed.show_error("Segment 'late cluster' overlaps segment 'x'")
    type_into(ed.groups['peaks'].fields['min_height'].edit, '1')
    assert not ed.error_label.isHidden()


def test_editor_delete_segment_button(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    assert not ed.delete_button.isHidden() and ed.delete_button.text() == 'Delete segment'
    with qtbot.waitSignal(ed.delete_requested, timeout=500) as blocker:
        qtbot.mouseClick(ed.delete_button, Qt.LeftButton)
    assert blocker.args == ['w1']
    new = make_editor(qtbot, p, S.new_segment(p, 1.0, 2.0), is_new=True)
    assert new.delete_button.isHidden()


def test_editor_after_line_none_with_baseline_carries_on(qtbot):
    """R11: a file whose fresh segment has no end break; its baseline runs past the end."""
    p = params_of([{'id': 'w1', 'label': '', 'start': 5.0, 'end': 8.0}],
                  [{'time': 5.0, 'overrides': {'lambda': 1e3}}])
    rec = S.segments_view(p)['segments'][0]
    assert rec['fresh_baseline'] and rec['after']['mode'] == 'none'
    ed = make_editor(qtbot, p, rec, resume={})
    assert ed.after_label.text() == "After 8 this segment's baseline carries on"
    assert not ed.after_reset.isHidden()
    assert ed.record() == rec
    with qtbot.waitSignal(ed.preview_changed, timeout=500) as blocker:
        qtbot.mouseClick(ed.after_reset, Qt.LeftButton)
    assert blocker.args[0]['after'] == {'mode': 'resume', 'overrides': {}}
    assert ed.after_label.text() == 'After 8 the whole-run baseline takes over again.'
    assert ed.after_reset.isHidden()


def test_editor_after_line_first_baseline_resumes(qtbot, spec):
    """'none' on a segment gaining its first fresh start is written as 'resume' (apply_segment)."""
    p, _rec = spec
    new = S.new_segment(p, 1.0, 2.0)
    ed = make_editor(qtbot, p, new, is_new=True)
    ed.fresh_check.setChecked(True)
    assert ed.after_label.text() == 'After 2 the whole-run baseline takes over again.'
    assert ed.after_reset.isHidden()
    ed.add_partway(1.5)
    assert ed.after_label.text() == 'After 2 the whole-run baseline takes over again.'


def test_editor_partway_sorted_by_time(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    ed.add_partway(12.6)
    assert [r['time'] for r in ed.record()['partway']] == [12.6, 12.85]


def test_editor_set_resume_updates_partway_inheritance(qtbot):
    """R12: From moved across a whole-run break; partway rows inherit the new resume value."""
    p = params_of([{'id': 'w1', 'label': '', 'start': 5.0, 'end': 8.0}],
                  [{'time': 4.0, 'overrides': {'lambda': 1e3}}])
    rec = S.segments_view(p)['segments'][0]
    ed = make_editor(qtbot, p, rec)
    ed.add_partway(6.0)
    row = ed.partway.rows[0]
    assert row.fields['lambda']._inherited == 1e3
    with qtbot.assertNotEmitted(ed.preview_changed):
        ed.set_resume({'lambda': 1e5})
    assert row.fields['lambda']._inherited == 1e5
    with qtbot.assertNotEmitted(ed.preview_changed):
        ed.set_resume({'lambda': 1e5})  # unchanged: nothing to do


# ── final fix wave #5: the editor fits a 440 px pane ────────────────────────────

PANE_WIDTH = 440


def test_editor_fits_narrow_pane_with_baseline_and_partway_open(qtbot, spec):
    from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget
    p, rec = spec
    pane = QWidget()
    qtbot.addWidget(pane)
    pane.setFixedWidth(PANE_WIDTH)
    ed = make_editor(qtbot, p, rec)
    QVBoxLayout(pane).addWidget(ed)
    ed.add_partway(13.0)
    for g in ed.groups.values():
        g.set_expanded(True)
    pane.show()
    assert ed.minimumSizeHint().width() <= PANE_WIDTH
    assert ed.sizeHint().width() <= PANE_WIDTH
    assert not ed.save_button.visibleRegion().isEmpty()
    assert not ed.cancel_button.visibleRegion().isEmpty()
    for combo in ed.findChildren(QComboBox):
        assert combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon
        assert combo.minimumContentsLength() == 12


def test_combo_fields_name_the_inherited_value_in_their_first_item(qtbot, spec):
    p, rec = spec
    ed = make_editor(qtbot, p, rec)
    method = ed.groups['baseline'].fields['method']
    assert method.tag_label is None
    assert method.combo.itemText(0).startswith('same as whole run (')
    row_method = ed.partway.rows[0].fields['method']
    assert row_method.tag_label is None and row_method.combo.itemText(0).startswith('same as segment (')
