"""Toolbar Home after a replot that changes the plotted data's extent."""
import numpy as np


def _data(offset, n=200):
    x = np.linspace(0, 10, n)
    corrected = np.exp(-((x - 5) ** 2) / 0.1) * 100.0
    baseline = np.full_like(x, offset)
    raw = corrected + baseline
    return {
        'x': x, 'original_y': raw, 'smoothed_y': raw, 'baseline_y': baseline,
        'corrected_y': corrected, 'peaks_x': np.array([5.0]),
        'peaks_y': np.array([100.0]), 'peak_metadata': [],
    }


def _home_ylim(plot):
    plot.toolbar.home()
    return plot.chromatogram_ax.get_ylim()


def test_home_after_show_corrected_fits_corrected_data(qtbot):
    from ui.frames.plot import PlotFrame
    plot = PlotFrame()
    qtbot.addWidget(plot)
    data = _data(offset=5000.0)
    plot.plot_chromatogram(data, show_corrected=False, new_file=True)
    plot.toolbar.push_current()
    plot.plot_chromatogram(data, show_corrected=True, new_file=False)
    lo, hi = _home_ylim(plot)
    assert hi < 1000, f"Home restored raw-signal limits: {(lo, hi)}"
    assert lo > -1000


def test_home_after_new_file_fits_new_data(qtbot):
    from ui.frames.plot import PlotFrame
    plot = PlotFrame()
    qtbot.addWidget(plot)
    plot.plot_chromatogram(_data(offset=5000.0), new_file=True)
    plot.plot_chromatogram(_data(offset=0.0), new_file=True)
    lo, hi = _home_ylim(plot)
    assert hi < 1000, f"Home restored the previous file's limits: {(lo, hi)}"


def test_home_after_tic_axis_rebuild(qtbot):
    from ui.frames.plot import PlotFrame
    plot = PlotFrame()
    qtbot.addWidget(plot)
    data = _data(offset=5000.0)
    plot.plot_chromatogram(data, show_corrected=True, new_file=True)
    plot.set_tic_visible(False)
    plot.set_tic_visible(True)
    lo, hi = _home_ylim(plot)
    assert hi < 1000, f"{(lo, hi)}"


def test_param_update_keeps_user_zoom(qtbot):
    from ui.frames.plot import PlotFrame
    plot = PlotFrame()
    qtbot.addWidget(plot)
    data = _data(offset=5000.0)
    plot.plot_chromatogram(data, show_corrected=False, new_file=True)
    plot.chromatogram_ax.set_xlim(4, 6)
    plot.chromatogram_ax.set_ylim(5000, 5100)
    plot.plot_chromatogram(data, show_corrected=False, new_file=False)
    assert plot.chromatogram_ax.get_xlim() == (4, 6)
    assert plot.chromatogram_ax.get_ylim() == (5000, 5100)
