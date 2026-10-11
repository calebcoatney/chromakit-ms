from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings

BAD_MZ = "Excluded m/z must be numbers separated by commas or spaces, e.g. 73, 147, 207 ("


def _fake_rainbow(monkeypatch):
    x = np.round(np.arange(0.0, 2.0, 0.005), 6)
    mz_axis = np.array([43.0, 57.0, 71.0, 85.0, 99.0])
    rng = np.random.default_rng(0)
    data = rng.uniform(0.0, 30.0, (x.size, mz_axis.size))
    def gauss(c, h, s=0.015):
        return h * np.exp(-0.5 * ((x - c) / s) ** 2)
    for m, h in {43: 8000, 57: 20000, 71: 6000, 85: 5000}.items():
        data[:, list(mz_axis).index(m)] += gauss(1.0, h)
    data[:, 4] += gauss(1.03, 7000)                    # m/z 99: different shape -> rejected
    ms = SimpleNamespace(xlabels=x, ylabels=mz_axis, data=data)
    data_dir = MagicMock()
    data_dir.get_file.return_value = ms
    import rainbow
    monkeypatch.setattr(rainbow, "read", lambda path, *a, **k: data_dir)


def _peak(rt, start, end):
    return SimpleNamespace(retention_time=rt, start_time=start, end_time=end,
                           deconvolved_spectrum=None, deconvolution_component_count=None)


@pytest.fixture
def dialog(qtbot, tmp_path, monkeypatch):
    _fake_rainbow(monkeypatch)
    import ui.dialogs.spectral_deconv_inspector as mod
    path = str(tmp_path / "t.ini")
    monkeypatch.setattr(mod, "QSettings", lambda *a, **k: QSettings(path, QSettings.IniFormat))
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    dlg = mod.SpectralDeconvInspectorDialog(
        peaks=[_peak(1.0, 0.96, 1.04)], ms_data_path="/fake.D",
        deconv_params=CorrelationDeconvParams(search_range=0.03),
        grouping_params=WindowGroupingParams(gap_tolerance=0.1))
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: dlg._last_result is not None, timeout=10000)
    dlg._ini_path = path
    return dlg


def test_preview_renders_groups_and_rejected(dialog):
    res = dialog._last_result
    assert len(res['groups']) == 1
    assert [r.peak.mz for r in res['rejected']] == [99.0]
    title = dialog._ax_scatter.get_title()
    assert title.startswith("Ion groups") and "1 group(s)" in title and "1 rejected" in title
    hollow = [c for c in dialog._ax_scatter.collections
              if len(c.get_facecolors()) == 0 or np.allclose(c.get_facecolors()[:, 3], 0)]
    assert hollow, "rejected ion peaks must be drawn hollow"


def test_group_spectrum_for_search(dialog):
    spec, rt = dialog._spectrum_for_group(0)
    assert set(spec) == {43.0, 57.0, 71.0, 85.0}
    assert abs(rt - 1.0) < 0.01
    assert dialog._spectrum_for_group(5) is None


def test_bottom_plot_labels_r(dialog):
    labels = [line.get_label() for line in dialog._ax_eic.get_lines()]
    assert any(lbl.startswith("57 (r=1.00)") for lbl in labels)


def test_no_adap_wording_left(dialog):
    import inspect
    import ui.dialogs.spectral_deconv_inspector as mod
    assert "ADAP" not in inspect.getsource(mod)


def _dashed_gray_lines(ax):
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == '--' and ln.get_color() == 'gray']


def test_show_rejected_toggle_rerenders_rejected_trace(dialog):
    assert not dialog._show_rejected_check.isChecked()
    assert not _dashed_gray_lines(dialog._ax_eic)
    dialog._show_rejected_check.setChecked(True)
    assert len(_dashed_gray_lines(dialog._ax_eic)) == 1   # the rejected m/z 99 trace


def test_apply_saves_through_helper_and_emits(dialog, qtbot):
    from logic.spectral_deconv_settings import load_spectral_deconv_settings
    dialog._min_r_spin.setValue(0.9)
    dialog._excluded_mz_edit.setText("73, 147")
    with qtbot.waitSignal(dialog.rerun_requested, timeout=1000) as blocker:
        dialog._on_apply_clicked()
    emitted_dp, _ = blocker.args
    assert emitted_dp.min_r == pytest.approx(0.9) and emitted_dp.excluded_mz == [73.0, 147.0]
    dp, gp = load_spectral_deconv_settings(QSettings(dialog._ini_path, QSettings.IniFormat))
    assert dp.min_r == pytest.approx(0.9) and dp.excluded_mz == [73.0, 147.0]
    assert gp.gap_tolerance == pytest.approx(0.1)


def test_invalid_excluded_mz_blocks_preview_and_apply(dialog, qtbot):
    dialog._excluded_mz_edit.setText("73, abc")
    with qtbot.assertNotEmitted(dialog.rerun_requested):
        dialog._on_apply_clicked()
    assert dialog._status_label.text().startswith(BAD_MZ)
    dialog._status_label.setText("")
    dialog._on_preview_clicked()
    assert dialog._status_label.text().startswith(BAD_MZ)
    assert dialog._preview_worker is None
    s = QSettings(dialog._ini_path, QSettings.IniFormat)
    assert s.value("ms_spectral_deconv/excluded_mz", None) is None   # nothing saved


def test_bad_excluded_mz_never_reaches_the_app(dialog, qtbot):
    dialog._excluded_mz_edit.setText("73, abc")
    dialog._offset_slider.setValue(10)        # re-windows without parsing excluded m/z
    assert dialog._current_preview_offset_min == pytest.approx(0.01)
    with qtbot.assertNotEmitted(dialog.apply_offset_requested):
        dialog._on_offset_apply_clicked()
    with qtbot.assertNotEmitted(dialog.apply_offset_to_folder_requested):
        dialog._on_offset_apply_folder_clicked()
    assert dialog._status_label.text().startswith(BAD_MZ)


def test_noise_levels_are_whole_run_and_cached(dialog):
    """The preview uses the same whole-run noise as the runner (spec 3, step 2)."""
    from logic.ion_peak_resolver import whole_run_noise_levels
    for q in (0.85, 0.5):
        got = dialog._noise_levels(q)
        np.testing.assert_array_equal(got, whole_run_noise_levels(dialog._ms_data, q))
    assert dialog._noise_levels(0.5) is dialog._noise_levels(0.5)      # cached per percentile


def test_preview_components_match_the_runner(dialog):
    """Preview and production give the same spectrum for the same window and settings."""
    from logic.spectral_deconv_runner import run_spectral_deconvolution
    dp, gp = dialog._read_params_or_report()
    twin = _peak(1.0, 0.96, 1.04)
    run_spectral_deconvolution([twin], "/fake.D", deconv_params=dp, grouping_params=gp)

    preview = dialog._last_result['components']
    assert len(preview) == 1 and twin.deconvolved_spectrum is not None
    spec = preview[0].spectrum
    np.testing.assert_array_equal(twin.deconvolved_spectrum['mz'], sorted(spec))
    np.testing.assert_allclose(twin.deconvolved_spectrum['intensities'],
                               [spec[m] for m in sorted(spec)])


def test_bad_excluded_mz_explains_and_selects_the_field(dialog):
    """Review #8: the error went to the status line, away from the field, with no fix."""
    dialog._excluded_mz_edit.setText("73 abc")
    dialog._on_preview_clicked()
    assert dialog._status_label.text() == BAD_MZ + "'abc' is not a number)."
    assert dialog.focusWidget() is dialog._excluded_mz_edit
    assert dialog._excluded_mz_edit.selectedText() == "73 abc"


def test_space_separated_excluded_mz_previews(dialog, qtbot):
    dialog._excluded_mz_edit.setText("73 147")
    dp, _ = dialog._read_params_or_report()
    assert dp.excluded_mz == [73.0, 147.0]


def test_settings_scroll_and_the_buttons_stay_pinned(dialog, qtbot):
    """Review #13: the 829 px settings panel squeezed its spin boxes at the 650 px minimum."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFrame, QScrollArea
    scrolls = dialog.findChildren(QScrollArea)
    assert len(scrolls) == 1
    scroll = scrolls[0]
    assert scroll.widgetResizable() and scroll.frameShape() == QFrame.Shape.NoFrame
    assert scroll.focusPolicy() == Qt.FocusPolicy.NoFocus
    inner = scroll.widget()
    for w in (dialog._offset_slider, dialog._gap_spin, dialog._noise_percentile_spin,
              dialog._min_r_spin, dialog._excluded_mz_edit, dialog._top_n_spin,
              dialog._show_rejected_check):
        assert inner.isAncestorOf(w)
    for btn in (dialog._preview_btn, dialog._apply_btn, dialog._close_btn):
        assert not scroll.isAncestorOf(btn)
    assert dialog._close_btn.text() == "Close"

    dialog.resize(1100, 650)
    dialog.show()
    qtbot.waitExposed(dialog)
    for spin in (dialog._gap_spin, dialog._noise_percentile_spin, dialog._min_ions_spin):
        assert spin.height() >= spin.minimumSizeHint().height()      # not squeezed
    bottom = dialog._close_btn.mapTo(dialog, dialog._close_btn.rect().bottomLeft()).y()
    assert bottom <= dialog.height()                                  # buttons on screen
    assert scroll.verticalScrollBar().maximum() > 0                   # the panel scrolls instead


def test_bad_excluded_mz_scrolls_the_field_into_view(dialog, qtbot):
    """Focus alone doesn't scroll a QScrollArea: at the minimum size the field sat below the fold."""
    from PySide6.QtWidgets import QScrollArea
    dialog.resize(1100, 650)
    dialog.show()
    qtbot.waitExposed(dialog)
    scroll = dialog.findChild(QScrollArea)
    field = dialog._excluded_mz_edit
    viewport = scroll.viewport()

    def field_in_view():
        return viewport.rect().contains(field.rect().translated(field.mapTo(viewport, field.rect().topLeft())))

    assert scroll.verticalScrollBar().value() == 0 and not field_in_view()   # below the fold
    field.setText("73 abc")
    dialog._on_preview_clicked()
    assert field_in_view()


# ── Part B of the impeccable review: status, hints, empty states, plots, buttons ──

STALE = "Settings changed — click Preview to update this window."


def _fake_ms(monkeypatch, compounds, mz_axis, x=None, seed=0):
    """Fake rainbow .D: ``compounds`` is [(apex RT, {m/z: height}), ...] on uniform 0-30 noise."""
    x = np.round(np.arange(0.0, 2.0, 0.005), 6) if x is None else x
    mz_axis = np.asarray(mz_axis, dtype=float)
    data = np.random.default_rng(seed).uniform(0.0, 30.0, (x.size, mz_axis.size))
    for c, ions in compounds:
        for m, h in ions.items():
            data[:, list(mz_axis).index(m)] += h * np.exp(-0.5 * ((x - c) / 0.015) ** 2)
    ms = SimpleNamespace(xlabels=x, ylabels=mz_axis, data=data)
    data_dir = MagicMock()
    data_dir.get_file.return_value = ms
    import rainbow
    monkeypatch.setattr(rainbow, "read", lambda path, *a, **k: data_dir)
    return ms


def _make(qtbot, tmp_path, monkeypatch, peaks, dp=None, gp=None, fake=True, **kw):
    """An Inspector on ``_fake_rainbow`` (unless ``fake=False``), QSettings on a temp INI."""
    if fake:
        _fake_rainbow(monkeypatch)
    import ui.dialogs.spectral_deconv_inspector as mod
    path = str(tmp_path / "t.ini")
    monkeypatch.setattr(mod, "QSettings", lambda *a, **k: QSettings(path, QSettings.IniFormat))
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    dlg = mod.SpectralDeconvInspectorDialog(
        peaks=peaks, ms_data_path="/fake.D",
        deconv_params=dp or CorrelationDeconvParams(search_range=0.03),
        grouping_params=gp or WindowGroupingParams(gap_tolerance=0.1), **kw)
    qtbot.addWidget(dlg)
    qtbot.waitUntil(lambda: dlg._preview_worker is None and (dlg._last_result is not None
                                                              or not dlg._windows), timeout=10000)
    dlg._ini_path = path
    return dlg


def _repreview(dlg, qtbot):
    prev = dlg._last_result
    dlg._on_preview_clicked()
    qtbot.waitUntil(lambda: dlg._preview_worker is None and dlg._last_result is not prev,
                    timeout=10000)


def _click(dlg, rt, mz):
    dlg._on_scatter_click(SimpleNamespace(inaxes=dlg._ax_scatter, xdata=rt, ydata=mz))


def _status(dlg):
    return dlg._status_label.text()


def _plot_texts(dlg):
    return [" ".join(t.get_text().split()) for t in dlg._ax_scatter.texts]


# #2: the status describes the preview, with the runner's own matching rule

def test_status_counts_the_preview_matches_not_the_last_run(dialog):
    assert dialog._peaks[0].deconvolved_spectrum is None      # no full run yet
    assert _status(dialog) == "Preview: 1 group(s); 1 of 1 FID peak(s) matched (RT 1.000 min)"


def test_status_names_the_fallback_peaks_and_follows_the_offset(qtbot, tmp_path, monkeypatch):
    from logic.spectral_deconv_runner import WindowGroupingParams
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(1.0, 0.96, 1.04), _peak(1.2, 1.16, 1.24)],
                gp=WindowGroupingParams(gap_tolerance=0.2))
    assert len(dlg._windows) == 1
    assert _status(dlg) == ("Preview: 1 group(s); 1 of 2 FID peak(s) matched (RT 1.000 min); "
                            "1 will use the fallback spectrum")
    dlg._offset_slider.setValue(100)          # +0.1 min: the group now sits 0.1 from both peaks
    assert _status(dlg) == ("Preview: 1 group(s); 0 of 2 FID peak(s) matched; "
                            "2 will use the fallback spectrum")


def test_preview_matching_reuses_the_runner_rule(dialog, monkeypatch):
    import ui.dialogs.spectral_deconv_inspector as mod
    calls = []
    real = mod._assign_components_to_peaks
    monkeypatch.setattr(mod, "_assign_components_to_peaks",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    dialog._render_plots(dialog._last_result)
    assert calls and calls[0][2] == pytest.approx(0.05)      # the RT match tolerance


# #3: hints follow the actual rejection reason

def test_no_spectra_because_groups_are_too_small(dialog, qtbot):
    dialog._min_ions_spin.setValue(8)
    _repreview(dialog, qtbot)
    assert _status(dialog) == ("No spectra in this window: every group had fewer than 8 ions "
                               "(Min ions per spectrum). Lower Min ions per spectrum.")


def test_no_spectra_because_shapes_dont_match(dialog, qtbot):
    dialog._min_r_spin.setValue(1.0)
    _repreview(dialog, qtbot)
    assert _status(dialog) == ("No spectra in this window: 4 ion peaks didn't match the shape of "
                               "a stronger ion nearby (r below 1.00). Lower Min shape correlation.")


def test_many_rejected_for_shape_hint(dialog, qtbot):
    dialog._min_r_spin.setValue(1.0)
    dialog._min_ions_spin.setValue(1)
    _repreview(dialog, qtbot)
    assert _status(dialog) == (
        "Preview: 1 group(s); 1 of 1 FID peak(s) matched (RT 1.000 min)"
        "  ⚠ 4 of 5 ion peaks rejected for shape: lower Min shape correlation, or raise Search "
        "range if one compound's traces are split into several peaks.")


def test_many_in_small_groups_hint(qtbot, tmp_path, monkeypatch):
    from logic.spectral_deconv_runner import WindowGroupingParams
    _fake_ms(monkeypatch, [(1.0, {43: 8000, 57: 20000, 71: 6000, 85: 5000}),
                           (1.15, {99: 9000, 113: 6000, 127: 4000})],
             [43, 57, 71, 85, 99, 113, 127])
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(1.0, 0.96, 1.04), _peak(1.15, 1.11, 1.19)],
                gp=WindowGroupingParams(gap_tolerance=0.2), fake=False)
    assert _status(dlg) == (
        "Preview: 1 group(s); 1 of 2 FID peak(s) matched (RT 1.000 min); 1 will use the fallback "
        "spectrum  ⚠ 3 of 7 ion peaks were in groups smaller than 4 ions: lower Min ions per "
        "spectrum.")


def test_too_many_groups_names_the_right_settings(dialog):
    dialog._MAX_RENDERABLE_CLUSTERS = 0
    dialog._render_plots(dialog._last_result)
    texts = " ".join(t.get_text() for t in dialog._ax_scatter.texts)
    assert "Too many groups to display. Raise Min ions per spectrum or Min peak height." in texts
    assert "↑" not in texts


# #9: empty and error states name the right cause

def test_window_before_ms_acquisition(qtbot, tmp_path, monkeypatch):
    _fake_ms(monkeypatch, [], [43, 57], x=np.round(np.arange(3.0, 6.0, 0.005), 6))
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(1.5, 1.46, 1.54), _peak(4.0, 3.96, 4.04)],
                fake=False)
    assert dlg._window_combo.itemText(0) == "Window 1 (no MS data; FID peak(s) at 1.500 min)"
    assert _status(dlg) == ("No MS scans cover this window. Its FID peaks may fall in the MS "
                            "solvent delay, or the MS time offset is off.")
    assert not dlg._ax_scatter.axison and not dlg._ax_eic.axison
    assert _plot_texts(dlg) == [_status(dlg)]               # wrapped, same words


def test_no_ion_peaks_names_the_threshold(dialog, qtbot):
    dialog._min_peak_height_spin.setValue(50000)
    _repreview(dialog, qtbot)
    msg = ("No ion peaks in this window: no ion rises above Min peak height (50000 counts) and "
           "its noise level. Lower Min peak height or Noise percentile.")
    assert _status(dialog) == msg
    assert not dialog._ax_scatter.axison
    assert _plot_texts(dialog) == [msg]
    dialog._min_peak_height_spin.setValue(1000)
    _repreview(dialog, qtbot)
    assert dialog._ax_scatter.axison                             # back on for a real result


def test_preview_error_shows_the_last_line_and_logs_the_traceback(dialog, qtbot, monkeypatch, capsys):
    import ui.dialogs.spectral_deconv_inspector as mod
    def boom(*a, **k):
        raise ValueError("boom")
    monkeypatch.setattr(mod, "deconvolve_window", boom)
    dialog._on_preview_clicked()
    qtbot.waitUntil(lambda: dialog._preview_worker is None, timeout=10000)
    assert _status(dialog) == "Preview failed: ValueError: boom"
    assert "Traceback" in capsys.readouterr().out


# #10: a settings change marks the preview stale; Return previews

_SETTING_ATTRS = ["_gap_spin", "_padding_spin", "_rt_match_spin", "_max_window_peaks_spin",
                  "_noise_percentile_spin", "_search_range_spin", "_min_peak_height_spin",
                  "_min_top_edge_ratio_spin", "_min_points_spin", "_rt_tolerance_spin",
                  "_min_r_spin", "_min_ions_spin", "_excluded_mz_edit", "_top_n_spin"]


@pytest.mark.parametrize("attr", _SETTING_ATTRS)
def test_settings_change_marks_the_preview_stale(dialog, qtbot, attr):
    w = getattr(dialog, attr)
    if hasattr(w, "setText"):
        w.setText("73")
    else:
        w.setValue(w.value() + w.singleStep() if w.value() < w.maximum() else w.value() - w.singleStep())
    assert _status(dialog) == STALE
    dialog._offset_slider.setValue(5)          # re-renders the old preview: still stale
    assert _status(dialog) == STALE
    _repreview(dialog, qtbot)
    assert _status(dialog) != STALE


def test_return_in_a_field_previews_and_never_clicks_auto(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton
    _fake_rainbow(monkeypatch)
    import rainbow
    ms = rainbow.read("/fake.D").get_file("data.ms")
    x, data = ms.xlabels, ms.data
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(1.0, 0.96, 1.04)], fake=False,
                fid_time=x, fid_signal=data[:, 1], tic_time=x, tic_signal=data.sum(1))
    assert dlg._offset_auto_btn.isEnabled()
    assert dlg._preview_btn.isDefault()
    assert [b.text() for b in dlg.findChildren(QPushButton)
            if b.autoDefault() and b is not dlg._preview_btn] == []
    dlg.show()
    qtbot.waitExposed(dlg)
    clicked = []
    for b in dlg.findChildren(QPushButton):
        b.clicked.connect(lambda *_, b=b: clicked.append(b.text()))
    offset = dlg._current_preview_offset_min
    prev = dlg._last_result
    dlg._min_r_spin.setFocus()
    QTest.keyClick(dlg._min_r_spin, Qt.Key_Return)
    qtbot.waitUntil(lambda: dlg._preview_worker is None and dlg._last_result is not prev,
                    timeout=10000)
    assert clicked == ["Preview"]
    assert dlg._current_preview_offset_min == offset and dlg.isVisible()


# #11: crowded windows

def test_group_colors_leave_gray_for_rejected():
    from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog
    colors = SpectralDeconvInspectorDialog._group_colors(12)
    assert not [c for c in colors.values() if c[0] == c[1] == c[2]]
    assert len({tuple(c) for c in colors.values()}) == 9


def test_each_lead_star_is_numbered(dialog):
    assert "1" in [t.get_text() for t in dialog._ax_scatter.texts]


def test_fid_rt_text_only_for_four_or_fewer_peaks(dialog, qtbot, tmp_path, monkeypatch):
    assert "1.000" in [t.get_text() for t in dialog._ax_scatter.texts]
    from logic.spectral_deconv_runner import WindowGroupingParams
    rts = [0.90, 0.95, 1.00, 1.05, 1.10]
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(rt, rt - 0.02, rt + 0.02) for rt in rts],
                gp=WindowGroupingParams(gap_tolerance=0.2))
    assert len(dlg._windows) == 1
    texts = [t.get_text() for t in dlg._ax_scatter.texts]
    assert not [t for t in texts if t in {f"{rt:.3f}" for rt in rts}]


# #12: buttons say what they save; Auto says why it is off

def test_offset_and_rerun_buttons_say_what_they_save(dialog):
    from ui.dialogs.spectral_deconv_fields import tooltip_html
    assert dialog._offset_apply_btn.text() == "Save Offset"
    assert dialog._offset_apply_folder_btn.text() == "Save Offset to Folder…"
    assert dialog._apply_btn.text() == "Save && Rerun File"
    assert dialog._apply_btn.toolTip() == tooltip_html(
        "Save these settings (Deconvolve MS, MS Options and batch use them too) and rerun "
        "deconvolution on every window of this file.")
    assert dialog._preview_btn.toolTip() == tooltip_html(
        "Rerun only the window shown, with the settings above. Nothing is saved.")
    assert dialog._offset_reset_btn.toolTip() == tooltip_html(
        "Set the preview offset to 0 min. The saved offset changes only when you click Save Offset.")
    tip = dialog._offset_apply_btn.toolTip()
    assert "settings shown here" in tip and "rerun deconvolution on the whole file" in tip


def test_auto_stays_disabled_without_fid_and_tic(dialog, qtbot):
    from ui.dialogs.spectral_deconv_fields import tooltip_html
    _repreview(dialog, qtbot)                  # set_controls_enabled(True) has run
    assert not dialog._offset_auto_btn.isEnabled()
    assert dialog._offset_auto_btn.toolTip() == tooltip_html("Needs an FID trace and a TIC.")


def test_auto_tooltip_with_fid_and_tic(qtbot, tmp_path, monkeypatch):
    from ui.dialogs.spectral_deconv_fields import tooltip_html
    x = np.round(np.arange(0.0, 2.0, 0.005), 6)
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(1.0, 0.96, 1.04)],
                fid_time=x, fid_signal=np.ones_like(x), tic_time=x, tic_signal=np.ones_like(x))
    assert dlg._offset_auto_btn.isEnabled()
    assert dlg._offset_auto_btn.toolTip() == tooltip_html(
        "Estimate the offset by aligning the TIC to the FID trace.")


def test_save_shown_settings_saves_and_returns_them(dialog):
    from logic.spectral_deconv_settings import load_spectral_deconv_settings
    dialog._min_r_spin.setValue(0.9)
    dp, gp = dialog.save_shown_settings()
    saved_dp, saved_gp = load_spectral_deconv_settings(QSettings(dialog._ini_path, QSettings.IniFormat))
    assert dp == saved_dp and gp == saved_gp and saved_dp.min_r == pytest.approx(0.9)


# #14: "Show rejected traces" says what it draws, and when there is nothing to draw

def test_rejected_traces_are_labelled(dialog):
    from ui.dialogs.spectral_deconv_fields import tooltip_html
    assert dialog._show_rejected_check.toolTip() == tooltip_html(
        "Also draw rejected ion peaks as dashed gray traces in the lower plot (only those among "
        "the Top N strongest ion peaks).")
    dialog._show_rejected_check.setChecked(True)
    assert [ln.get_label() for ln in _dashed_gray_lines(dialog._ax_eic)] == ["99 rejected"]
    assert not dialog._ax_eic.get_title().endswith("among these")


def test_no_rejected_among_the_top_n_says_so(dialog, qtbot):
    dialog._top_n_spin.setValue(1)
    _repreview(dialog, qtbot)
    dialog._show_rejected_check.setChecked(True)
    assert dialog._ax_eic.get_title().endswith(" — no rejected ion peaks among these")


# #17: selection and search copy (groups numbered from 1)

def test_group_click_and_search_copy(dialog):
    _click(dialog, 1.0, 57.0)
    assert _status(dialog) == "Group 1 selected (RT 1.000 min, 4 ions): searching the library…"
    assert "group 1 selected" in dialog._ax_scatter.get_title()
    dialog.show_search_results([("Toluene", 0.94), ("Cycloheptatriene", 0.88)], 1.0)
    assert dialog._results_label.text() == "Library matches for group 1 (RT 1.000 min)"
    assert _status(dialog) == "Group 1: 2 library match(es)"
    dialog.show_search_results([], 1.0)
    assert dialog._results_label.text() == "No library matches for group 1 (RT 1.000 min)"
    dialog.show_search_results([("Toluene", 0.94)], 1.0)
    _click(dialog, 1.0, 57.0)                 # same group again: deselect
    assert dialog._results_tree.topLevelItemCount() == 0
    assert dialog._results_label.text() == "Click a group to search"


def test_rejected_click_gives_the_reason(dialog, qtbot):
    _click(dialog, 1.03, 99.0)
    assert _status(dialog) == ("m/z 99 at 1.030 min was rejected: its shape didn't match the "
                               "stronger ion nearby (r below 0.80), so it is in no spectrum.")
    dialog._min_ions_spin.setValue(8)
    _repreview(dialog, qtbot)
    _click(dialog, 1.0, 57.0)
    assert _status(dialog) == ("m/z 57 at 1.000 min was rejected: its group had fewer than 8 "
                               "ions, so it is in no spectrum.")


def test_show_status_sets_the_status_line(dialog):
    dialog.show_status("hello")
    assert _status(dialog) == "hello"


# #18: legends

def test_top_legend_explains_every_symbol(dialog):
    legend = dialog._ax_scatter.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == [
        "group member", "★ lead ion (sets RT)", "rejected: shape", "rejected: group too small",
        "FID peak"]


def test_only_the_lead_ion_extent_is_shaded(dialog):
    assert len(dialog._ax_eic.patches) == 1


def test_many_traces_legend_lists_the_lead_ions(qtbot, tmp_path, monkeypatch):
    mzs = list(range(40, 55))                                     # 15 ions, one compound
    _fake_ms(monkeypatch, [(1.0, {m: 2000 + 1000 * i for i, m in enumerate(mzs)})], mzs)
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(1.0, 0.96, 1.04)], fake=False)
    assert len(dlg._last_result['groups'][0].members) == 15
    legend = dlg._ax_eic.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["54 (r=1.00), ★ group 1"]


# #19: the window combo lists at most four FID RTs

def test_window_label_lists_at_most_four_rts(dialog, qtbot, tmp_path, monkeypatch):
    a, b, _ = dialog._windows[0]
    assert dialog._window_combo.itemText(0) == f"Window 1 (RT {a:.2f}–{b:.2f} min, 1 FID peak: 1.000)"
    from logic.spectral_deconv_runner import WindowGroupingParams
    rts = [0.90, 0.95, 1.00, 1.05, 1.10]
    dlg = _make(qtbot, tmp_path, monkeypatch, [_peak(rt, rt - 0.02, rt + 0.02) for rt in rts],
                gp=WindowGroupingParams(gap_tolerance=0.2))
    a, b, _ = dlg._windows[0]
    assert dlg._window_combo.itemText(0) == (
        f"Window 1 (RT {a:.2f}–{b:.2f} min, 5 FID peaks: 0.900, 0.950, 1.000, 1.050, …)")
