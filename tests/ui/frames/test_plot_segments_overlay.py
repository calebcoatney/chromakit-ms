"""Segment overlay on the chromatogram plot (design 2026-11-12 §4.3)."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('pytestqt')

BASE = 100.0
W1 = {'id': 'w1', 'label': 'early', 'start': 3.0, 'end': 6.0}
W2 = {'id': 'w2', 'label': 'late cluster', 'start': 12.5, 'end': 13.1}
OPEN_W2 = {'id': 'w2', 'label': 'late cluster', 'start': 12.5, 'end': 13.1, 'partway': [12.85]}


def _gauss(x, at, height, width=0.03):
    return height * np.exp(-((x - at) ** 2) / (2 * width ** 2))


def _data(x=None):
    x = np.linspace(0.0, 20.0, 4001) if x is None else x
    corrected = (_gauss(x, 5.0, 5000.0, 0.1)
                 + _gauss(x, 12.65, 40.0) + _gauss(x, 12.95, 25.0))
    baseline = np.full_like(x, BASE)
    raw = corrected + baseline
    meta = [{'x': 5.0, 'y': 5000.0, 'type': 'peak'},
            {'x': 12.65, 'y': 40.0, 'type': 'peak'},
            {'x': 12.95, 'y': 25.0, 'type': 'peak'}]
    return {
        'x': x, 'original_y': raw, 'smoothed_y': raw, 'baseline_y': baseline,
        'corrected_y': corrected, 'peaks_x': np.array([m['x'] for m in meta]),
        'peaks_y': np.array([m['y'] for m in meta]), 'peak_metadata': meta,
        'range_excluded_peaks_x': [],
    }


def _plot(qtbot, data=None, corrected=False, invert_x=False):
    from ui.frames.plot import PlotFrame
    plot = PlotFrame()
    qtbot.addWidget(plot)
    if invert_x:
        plot.set_axis_labels('Wavenumber (cm⁻¹)', 'Absorbance', invert_x=True)
    plot.plot_chromatogram(_data() if data is None else data, show_corrected=corrected, new_file=True)
    return plot


def _role(plot, role):
    return [a for a in plot._segment_artists if getattr(a, 'segment_role', None) == role]


def _xs(rect):
    return sorted((rect.get_x(), rect.get_x() + rect.get_width()))


def _marker(plot, x):
    (line,) = [ln for ln in plot.chromatogram_ax.lines
               if getattr(ln, 'peak_x', None) == x and ln.get_marker() == '*']
    return line


def test_two_windows_two_bands(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W1, W2], [], None)
    bands = _role(plot, 'band')
    assert len(bands) == 2
    assert sorted(_xs(b) for b in bands) == [pytest.approx([3.0, 6.0]), pytest.approx([12.5, 13.1])]
    assert all(b in plot.chromatogram_ax.get_children() for b in bands)
    assert not _role(plot, 'grey')


def test_open_segment_greys_outside_and_fits_y(qtbot):
    plot = _plot(qtbot)
    full_top = plot.chromatogram_ax.get_ylim()[1]
    plot.set_segment_overlay([W1, W2], [], OPEN_W2)
    assert len(_role(plot, 'grey')) == 2
    assert len(_role(plot, 'edge')) == 2
    assert [ln.get_xdata()[0] for ln in _role(plot, 'partway')] == [12.85]
    lo, hi = plot.chromatogram_ax.get_ylim()
    assert hi < full_top / 10, "the big peak outside the segment still sets the y range"
    assert lo <= BASE < BASE + 40.0 <= hi          # the span's trace fits
    assert hi == pytest.approx(BASE + 40.0 + 0.05 * 40.0, rel=0.02)


def test_grey_covers_outside_the_span(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W2], [], OPEN_W2)
    left, right = sorted(_xs(g) for g in _role(plot, 'grey'))
    assert left[1] == pytest.approx(12.5) and left[0] <= 0.0
    assert right[0] == pytest.approx(13.1) and right[1] >= 20.0


def test_close_restores_previous_ylim(qtbot):
    plot = _plot(qtbot)
    before = plot.chromatogram_ax.get_ylim()
    plot.set_segment_overlay([W2], [], OPEN_W2)
    assert plot.chromatogram_ax.get_ylim() != before
    plot.set_segment_overlay([W2], [], None)
    assert plot.chromatogram_ax.get_ylim() == before
    assert not _role(plot, 'grey') and not _role(plot, 'edge')


def test_user_zoom_kept_until_span_changes(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W2], [], OPEN_W2)
    plot.chromatogram_ax.set_ylim(0, 7)
    plot.set_segment_overlay([W2], [], dict(OPEN_W2, partway=[12.8]))   # a preview, same span
    assert plot.chromatogram_ax.get_ylim() == (0, 7)
    plot.set_segment_overlay([W2], [], dict(OPEN_W2, end=13.0))
    assert plot.chromatogram_ax.get_ylim() != (0, 7)


def test_corrected_view_fits_corrected_trace(qtbot):
    plot = _plot(qtbot, corrected=True)
    plot.set_segment_overlay([W2], [], OPEN_W2)
    lo, hi = plot.chromatogram_ax.get_ylim()
    assert lo < 0 < 40.0 < hi < 50.0


def test_overlay_on_inverted_axis(qtbot):
    x = np.linspace(20.0, 0.0, 4001)    # high → low, as FTIR wavenumbers arrive
    plot = _plot(qtbot, _data(x), invert_x=True)
    ax = plot.chromatogram_ax
    assert ax.xaxis_inverted()
    plot.set_segment_overlay([W2], [], OPEN_W2)
    (band,) = _role(plot, 'band')
    assert _xs(band) == pytest.approx([12.5, 13.1])
    plot.canvas.draw()
    box = band.get_window_extent(plot.canvas.get_renderer())
    px = sorted(ax.transData.transform([(12.5, 0), (13.1, 0)])[:, 0])
    assert box.x0 == pytest.approx(px[0], abs=1) and box.x1 == pytest.approx(px[1], abs=1)
    lo, hi = ax.get_ylim()
    assert hi < 1000          # fitted from the span's data, whichever way x runs


def test_replot_keeps_overlay(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W1, W2], [8.0], OPEN_W2)
    fitted = plot.chromatogram_ax.get_ylim()
    plot.plot_chromatogram(_data(), new_file=False)
    children = plot.chromatogram_ax.get_children()
    assert len(_role(plot, 'band')) == 2 and len(_role(plot, 'grey')) == 2
    assert all(a in children for a in plot._segment_artists)
    assert plot.chromatogram_ax.get_ylim() == fitted
    assert _marker(plot, 5.0).get_alpha() == 0.3


def test_new_file_refits_the_open_segment(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W2], [], OPEN_W2)
    fitted = plot.chromatogram_ax.get_ylim()
    plot.plot_chromatogram(_data(), new_file=True)
    assert plot.chromatogram_ax.get_ylim() == pytest.approx(fitted)
    plot.set_segment_overlay([W2], [], None)
    assert plot.chromatogram_ax.get_ylim()[1] > 5000      # back to the new file's full view


def test_excluded_peaks_hollow_markers(qtbot):
    data = _data()
    data['range_excluded_peaks_x'] = [5.0, 12.65]
    plot = _plot(qtbot, data)
    plot.set_segment_overlay([], [], None)
    (line,) = _role(plot, 'excluded')
    assert line.get_marker() == 'o'
    assert line.get_markerfacecolor() == 'none'
    assert list(line.get_xdata()) == [5.0, 12.65]
    np.testing.assert_allclose(line.get_ydata(), np.interp([5.0, 12.65], data['x'], data['smoothed_y']))


def test_excluded_markers_follow_corrected_view(qtbot):
    data = _data()
    data['range_excluded_peaks_x'] = [12.65]
    plot = _plot(qtbot, data, corrected=True)
    plot.set_segment_overlay([], [], None)
    (line,) = _role(plot, 'excluded')
    assert line.get_ydata()[0] == pytest.approx(40.0, rel=1e-3)


def test_whole_run_breaks_dashed(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W2], [8.0, 15.0], None)
    lines = _role(plot, 'break')
    assert [ln.get_xdata()[0] for ln in lines] == [8.0, 15.0]
    assert all(ln.get_linestyle() == '--' for ln in lines)


def test_markers_outside_fade_and_come_back(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W2], [], OPEN_W2)
    assert _marker(plot, 5.0).get_alpha() == 0.3
    assert _marker(plot, 12.65).get_alpha() == 0.9
    plot.set_segment_overlay([W2], [], None)
    assert _marker(plot, 5.0).get_alpha() == 0.9


def _integration(data):
    x = data['x']
    peaks, xs, ys, bases = [], [], [], []
    for rt, half in ((5.0, 0.4), (12.65, 0.08)):
        m = (x >= rt - half) & (x <= rt + half)
        peaks.append(SimpleNamespace(retention_time=rt, position=rt, peak_number=len(peaks) + 1))
        xs.append(x[m])
        ys.append(data['corrected_y'][m])
        bases.append(data['baseline_y'][m])
    return {'peaks': peaks, 'x_peaks': xs, 'y_peaks': ys, 'baseline_peaks': bases}


def test_integration_shading_outside_fades(qtbot):
    data = _data()
    plot = _plot(qtbot, data)
    plot.set_segment_overlay([W2], [], OPEN_W2)
    plot.shade_integration_areas(_integration(data))
    shades = {c.peak_number: c for c in plot.chromatogram_ax.collections if hasattr(c, 'peak_number')}
    assert shades[1].get_alpha() == 0.3
    assert shades[2].get_alpha() == 0.4
    plot.set_segment_overlay([W2], [], None)
    assert shades[1].get_alpha() == 0.4


def test_new_segment_draws_its_band(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W1], [], {'id': 'w9', 'label': '', 'start': 9.0, 'end': 10.0,
                                        'partway': []})
    assert sorted(b.get_x() for b in _role(plot, 'band')) == [3.0, 9.0]


def test_open_band_replaces_its_window_band(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W2], [], dict(OPEN_W2, start=12.4))   # draft moved its From
    (band,) = _role(plot, 'band')
    assert band.get_x() == 12.4
    (label,) = _role(plot, 'label')
    assert label.get_text() == 'late cluster (editing)'


def test_label_omitted_when_band_too_narrow(qtbot):
    plot = _plot(qtbot)
    plot.set_segment_overlay([W1, {'id': 'w3', 'label': 'early', 'start': 9.0, 'end': 9.01}], [], None)
    plot.canvas.draw()
    renderer = plot.canvas.get_renderer()
    fits = {round(t.span[0], 2): t.fits(renderer) for t in _role(plot, 'label')}
    assert fits == {3.0: True, 9.0: False}


def test_nothing_enters_the_legend(qtbot):
    data = _data()
    data['range_excluded_peaks_x'] = [5.0]
    plot = _plot(qtbot, data)
    plot.set_segment_overlay([W1, W2], [8.0], OPEN_W2)
    assert all(a.get_label().startswith('_') for a in plot._segment_artists)
    handles, _ = plot.chromatogram_ax.get_legend_handles_labels()
    assert not set(handles) & set(plot._segment_artists)


def test_gcxgc_overlay_is_noop(qtbot):
    plot = _plot(qtbot)
    plot._gcxgc_axes = []
    plot.set_segment_overlay([W1, W2], [8.0], OPEN_W2)
    assert plot._segment_artists == []


# ── app wiring ────────────────────────────────────────────────────────────────

def test_app_passes_overlay_on_state_and_after_processing(qtbot, monkeypatch):
    from logic.method import ChromaMethod
    from ui.app import ChromaKitApp
    app = ChromaKitApp()
    qtbot.addWidget(app)
    app.parameters_frame.apply_method(ChromaMethod.model_validate({
        'name': 't', 'signal_type': 'gc', 'peak_windows': [W2],
        'baseline': {'break_points': [{'time': 8.0, 'overrides': {'lambda': 1e4}}]}}))
    calls = []
    monkeypatch.setattr(app.plot_frame, 'set_segment_overlay', lambda *a: calls.append(a))

    app.parameters_frame.segment_state_changed.emit(OPEN_W2)
    windows, breaks, open_segment = calls[-1]
    assert [w['id'] for w in windows] == ['w2'] and breaks == [8.0] and open_segment == OPEN_W2

    data = _data()
    app.process_and_display(data['x'], data['original_y'], new_file=True)
    assert calls[-1][2] == OPEN_W2           # a reprocess keeps the open segment

    app.parameters_frame.segment_state_changed.emit(None)
    assert calls[-1][2] is None
