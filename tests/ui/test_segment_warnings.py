"""In-place segment warnings (design 2026-11-12 §4.4): a baseline change inside a kept
peak (after Integrate) and the processor's baseline_notes, on the Segments rows and in
the open segment editor."""
import numpy as np
import pytest

pytest.importorskip('pytestqt')

from logic.method import ChromaMethod
from ui.app import ChromaKitApp
from ui.frames.parameters import ParametersFrame
from ui.frames.segments import WHOLE_RUN

SEG = {'id': 'w1', 'start': 3.0, 'end': 8.0}
PARTWAY = {'time': 5.0, 'overrides': {}}      # a baseline change at the peak's apex
WHOLE_RUN_ROW, SEG_ROW = 0, 1


def _method(breaks):
    return ChromaMethod.model_validate({
        'name': 't', 'signal_type': 'gc', 'peak_windows': [SEG],
        'baseline': {'break_points': list(breaks)}, 'peaks': {'enabled': True, 'min_prominence': 10.0},
    })


def _app(qtbot, monkeypatch, tmp_path, breaks=(PARTWAY,), ms_range=None):
    monkeypatch.setattr('ui.app.QMessageBox.warning',
                        lambda *a, **k: pytest.fail(f"unexpected dialog: {a[1:3]}"))
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.current_method = _method(breaks)
    app._apply_method_to_frames()
    x = np.linspace(0.0, 10.0, 10000)
    y = 1000.0 * np.exp(-0.5 * ((x - 5.0) / 0.05) ** 2) + 5.0
    app.current_x, app.current_y = x, y
    if ms_range is not None:
        app.plot_frame.tic_data = {'x': np.linspace(*ms_range, 100), 'y': np.zeros(100)}
    # Integrate exports into the open folder; keep it off the disk
    app.data_handler.current_directory_path = str(tmp_path)
    monkeypatch.setattr(app.export_manager, 'export_after_integration',
                        lambda *a, **k: {'messages': []})
    app.process_and_display(x, y)
    return app


def _row(app, row):
    item = app.parameters_frame.segments_panel.table.item(row, 0)
    return item.text(), item.toolTip()


def _reprocess(app):
    pf = app.parameters_frame
    pf.parameters_changed.emit(pf.current_params)


def test_break_inside_kept_peak_warns_after_integrate(qtbot, monkeypatch, tmp_path):
    app = _app(qtbot, monkeypatch, tmp_path)
    assert _row(app, SEG_ROW) == ('w1', '')       # no warning before Integrate
    app.on_integrate()
    assert any(p.start_time < 5.0 < p.end_time for p in app.integrated_peaks)
    text, tip = _row(app, SEG_ROW)
    assert text == '⚠ w1'
    assert 'Baseline change at 5 falls inside the peak' in tip
    assert _row(app, WHOLE_RUN_ROW)[1] == ''
    app.parameters_frame.open_segment('w1')
    label = app.parameters_frame.segment_editor.warning_label
    assert not label.isHidden() and 'Baseline change at 5 ' in label.text()


def test_inside_peak_warning_clears_when_parameters_change(qtbot, monkeypatch, tmp_path):
    app = _app(qtbot, monkeypatch, tmp_path)
    app.on_integrate()
    assert _row(app, SEG_ROW)[0] == '⚠ w1'
    _reprocess(app)      # new processing: the peaks are stale until the next Integrate
    assert _row(app, SEG_ROW) == ('w1', '')


def test_dropped_break_note_lands_on_whole_run(qtbot, monkeypatch, tmp_path):
    app = _app(qtbot, monkeypatch, tmp_path, breaks=[{'time': 9.5, 'overrides': {}}],
               ms_range=(1.0, 9.0))
    text, tip = _row(app, WHOLE_RUN_ROW)
    assert text.startswith('⚠ ')
    assert tip == 'Break at 9.50 min is outside the MS range; ignored'
    assert _row(app, SEG_ROW) == ('w1', '')
    # the cause goes away: no break, no note
    pf = app.parameters_frame
    pf.current_params['baseline']['break_points'] = []
    _reprocess(app)
    assert not _row(app, WHOLE_RUN_ROW)[0].startswith('⚠')
    assert _row(app, WHOLE_RUN_ROW)[1] == ''


def test_short_section_note_lands_on_the_segment_holding_its_start(qtbot, monkeypatch, tmp_path):
    app = _app(qtbot, monkeypatch, tmp_path,
               breaks=[{'time': 6.0, 'overrides': {}}, {'time': 6.002, 'overrides': {}}],
               ms_range=(1.0, 9.0))
    text, tip = _row(app, SEG_ROW)
    assert text == '⚠ w1' and 'too short to fit' in tip


def test_loading_a_method_clears_warnings(qtbot, monkeypatch, tmp_path):
    app = _app(qtbot, monkeypatch, tmp_path)
    app.on_integrate()
    assert _row(app, SEG_ROW)[0] == '⚠ w1'
    app.current_method = _method([PARTWAY])
    app._apply_method_to_frames()
    assert _row(app, SEG_ROW) == ('w1', '')


# ── frame level ──

def _frame(qtbot, windows):
    f = ParametersFrame()
    qtbot.addWidget(f)
    f.apply_method(ChromaMethod.model_validate(
        {'name': 't', 'signal_type': 'gc', 'peak_windows': list(windows)}))
    return f


def test_whole_run_row_takes_a_warning(qtbot):
    f = _frame(qtbot, [SEG])
    f.set_segment_warnings([(9.5, 'outside')])
    item = f.segments_panel.table.item(WHOLE_RUN_ROW, 0)
    assert item.text() == f'⚠ {WHOLE_RUN}' and item.toolTip() == 'outside'
    f.set_segment_warnings([])
    assert item.text() == WHOLE_RUN and item.toolTip() == ''


def test_new_segment_editor_shows_warnings_inside_its_draft_span(qtbot):
    f = _frame(qtbot, [SEG])
    f.set_segment_warnings([(9.5, 'late note'), (5.0, 'mid note')])
    f.open_segment(None, (9.0, 10.0))       # new, unsaved: not in peak_windows yet
    label = f.segment_editor.warning_label
    assert not label.isHidden()
    assert 'late note' in label.text() and 'mid note' not in label.text()


# ── final fix wave #7: breaks the processor didn't use never warn inside a peak ─

@pytest.mark.parametrize('break_time, ms_range', [
    (5.03, (1.0, 5.02)),    # after the MS end: dropped
    (4.97, (4.98, 9.0)),    # before the MS start: doesn't split the baseline
])
def test_unused_break_inside_peak_does_not_warn(qtbot, monkeypatch, tmp_path, break_time, ms_range):
    app = _app(qtbot, monkeypatch, tmp_path, breaks=[{'time': break_time, 'overrides': {}}],
               ms_range=ms_range)
    app.on_integrate()
    # Integration clips this synthetic peak to the MS range; a peak whose bounds reach past
    # it (as other integration settings can give) is what put the break inside it
    peak = app.integrated_peaks[0]
    peak.start_time, peak.end_time = 4.8, 5.2
    app.on_peaks_integrated(app.integrated_peaks)
    assert 'falls inside the peak' not in _row(app, SEG_ROW)[1]
    assert not any('falls inside the peak' in msg for _, msg in app._peak_warnings)


def test_break_inside_ms_range_and_peak_still_warns(qtbot, monkeypatch, tmp_path):
    app = _app(qtbot, monkeypatch, tmp_path, ms_range=(1.0, 9.0))
    app.on_integrate()
    assert 'Baseline change at 5 falls inside the peak' in _row(app, SEG_ROW)[1]
