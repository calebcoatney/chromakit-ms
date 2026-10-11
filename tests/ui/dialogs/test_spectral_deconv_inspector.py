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
