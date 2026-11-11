def test_zoom_out_action_expands_plot_views_and_preserves_zoom_toggle(qtbot):
    from ui.frames.plot import PlotFrame

    plot = PlotFrame()
    qtbot.addWidget(plot)
    plot.tic_ax.set_xlim(0, 10)
    plot.tic_ax.set_ylim(0, 10)
    plot.chromatogram_ax.set_ylim(-5, 5)

    actions = plot.toolbar.actions()
    zoom_action = next(action for action in actions if action.text().strip().lower() == "zoom")
    zoom_out_action = next(action for action in actions if action.text() == "Zoom Out")

    zoom_action.trigger()
    assert zoom_action.isChecked()
    zoom_out_action.trigger()

    assert plot.tic_ax.get_xlim() == (-1.0, 11.0)
    assert plot.chromatogram_ax.get_xlim() == (-1.0, 11.0)
    assert plot.tic_ax.get_ylim() == (-1.0, 11.0)
    assert plot.chromatogram_ax.get_ylim() == (-6.0, 6.0)
    assert zoom_action.isChecked()

    zoom_action.trigger()
    assert not zoom_action.isChecked()