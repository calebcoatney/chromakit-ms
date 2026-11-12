"""Tail skim display: rider shading floor, dashed skim line, peak-table quality text."""
from types import SimpleNamespace

import numpy as np
import pytest

from logic.integration import Integrator
from tests.logic.tail_skim_signals import processed_from, rider_on_tail

OFFSET = 50.0


def _scenario(drop_shade_keys=False):
    pair = rider_on_tail(1.0, 3.0)
    processed = processed_from(pair, rider={'rider_integration': 'auto'})
    processed['original_y'] = processed['original_y'] + OFFSET
    processed['smoothed_y'] = processed['smoothed_y'] + OFFSET
    processed['baseline_y'] = np.full_like(pair['x'], OFFSET)
    result = Integrator.integrate(processed, verbose=False)
    if drop_shade_keys:
        result.pop('shade_top_peaks')
        result.pop('shade_floor_peaks')
    return processed, result


def _plot(qtbot, corrected, **kw):
    from ui.frames.plot import PlotFrame
    processed, result = _scenario(**kw)
    plot = PlotFrame()
    qtbot.addWidget(plot)
    plot.plot_chromatogram(processed, show_corrected=corrected, new_file=True)
    plot.shade_integration_areas(result)
    return plot, result


def _rider_index(result):
    return [i for i, p in enumerate(result['peaks']) if p.skim_parent is not None][0]


@pytest.mark.parametrize('corrected', [False, True])
def test_rider_gets_dashed_skim_line(qtbot, corrected):
    plot, _ = _plot(qtbot, corrected)
    lines = [ln for ln in plot.chromatogram_ax.lines if getattr(ln, 'is_skim_line', False)]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == '--'


@pytest.mark.parametrize('corrected', [False, True])
def test_rider_shading_floor_is_skim_curve(qtbot, corrected):
    plot, result = _plot(qtbot, corrected)
    i = _rider_index(result)
    (coll,) = [c for c in plot.chromatogram_ax.collections if getattr(c, 'peak_number', None) == i + 1]
    verts = coll.get_paths()[0].vertices
    floor = np.asarray(result['shade_floor_peaks'][i]) + (0.0 if corrected else OFFSET)
    xs = np.asarray(result['x_peaks'][i])
    for x, f in zip(xs, floor):
        col = verts[np.isclose(verts[:, 0], x)]
        assert col[:, 1].min() == pytest.approx(f, abs=1e-9)


def test_results_without_shade_keys_still_shade(qtbot):
    plot, result = _plot(qtbot, True, drop_shade_keys=True)
    assert len([c for c in plot.chromatogram_ax.collections if hasattr(c, 'peak_number')]) == len(result['peaks'])


def _auto_peaks(*pair_args):
    from tests.logic.tail_skim_signals import emg_pair
    pair = emg_pair(*pair_args)
    return Integrator.integrate(processed_from(pair, rider={'rider_integration': 'auto'}),
                                verbose=False)['peaks']


def test_quality_text():
    from ui.app import _quality_text
    _, result = _scenario()
    rider = result['peaks'][_rider_index(result)]
    parent = [p for p in result['peaks'] if p.skim_children][0]
    assert _quality_text(rider)[0] == f"Skimmed from #{parent.peak_number}"
    assert _quality_text(parent)[0] == f"Riders: #{rider.peak_number}"
    # real children the rule split: skim_parent is None, skim_decision says split
    split = _auto_peaks(0.002, 0.002, 4.0, 0.035)[-1]
    assert split.skim_parent is None and split.skim_decision['decision'] == 'split'
    assert _quality_text(split)[0] == "Split (rule)"
    near = _auto_peaks(0.06, 0.06, 2.8, 0.12)[-1]
    assert near.skim_decision['decision'] == 'split' and near.skim_near_threshold
    assert _quality_text(near)[0] == "Split (rule) · near threshold"
    assert _quality_text(SimpleNamespace())[0] == "OK"
    sat = SimpleNamespace(is_saturated=True, is_convoluted=True, skim_parent=2)
    assert _quality_text(sat)[0] == "⚠️ SATURATED"


def _one_sided(reported):
    """Rider and parent with only one of them reported (the other outside every window)."""
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    k = 1 if reported == 'rider' else 0
    other = pair['apex'] if reported == 'rider' else pair['child_apex']
    processed = processed_from(pair, rider={'rider_integration': 'auto'})
    processed.update(peaks_x=processed['peaks_x'][[k]], peaks_y=processed['peaks_y'][[k]],
                     peak_metadata=[processed['peak_metadata'][k]], range_excluded_peaks_x=x[[other]])
    (peak,) = Integrator.integrate(processed, verbose=False)['peaks']
    return processed, peak


def test_quality_text_when_one_side_not_reported():
    from ui.app import _quality_text
    _, rider = _one_sided('rider')
    assert rider.skim_model is not None and rider.skim_parent is None
    assert _quality_text(rider)[0] == "Skimmed (parent not reported)"
    _, parent = _one_sided('parent')
    assert parent.area_before_skim is not None and parent.skim_children == []
    assert _quality_text(parent)[0] == "Riders: not reported"
    near = SimpleNamespace(skim_model='exponential', skim_parent=None, skim_near_threshold=True)
    assert _quality_text(near)[0] == "Skimmed (parent not reported) · near threshold"
    assert _quality_text(SimpleNamespace(area_before_skim=1.0, skim_children=None))[0] == \
        "Riders: not reported"
    split = SimpleNamespace(skim_decision={'decision': 'split'}, area_before_skim=1.0, skim_children=[])
    assert _quality_text(split)[0] == "Split (rule)"


@pytest.mark.parametrize('corrected', [False, True])
def test_rider_with_unreported_parent_gets_dashed_skim_line(qtbot, corrected):
    from ui.frames.plot import PlotFrame
    processed, rider = _one_sided('rider')
    result = Integrator.integrate(processed, verbose=False)
    plot = PlotFrame()
    qtbot.addWidget(plot)
    plot.plot_chromatogram(processed, show_corrected=corrected, new_file=True)
    plot.shade_integration_areas(result)
    lines = [ln for ln in plot.chromatogram_ax.lines if getattr(ln, 'is_skim_line', False)]
    assert len(lines) == 1 and lines[0].get_linestyle() == '--'
