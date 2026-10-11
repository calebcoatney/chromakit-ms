import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDialog


@pytest.fixture
def ini_settings(tmp_path, monkeypatch):
    """Redirect the dialog's QSettings to a temp INI (never Caleb's real plist)."""
    path = str(tmp_path / "t.ini")
    import ui.dialogs.ms_options_dialog as mod
    monkeypatch.setattr(mod, "QSettings", lambda *a, **k: QSettings(path, QSettings.IniFormat))
    return path


def test_tab_shows_new_boxes_with_defaults(qtbot, ini_settings):
    from ui.dialogs.ms_options_dialog import MSOptionsDialog
    dlg = MSOptionsDialog()
    qtbot.addWidget(dlg)
    assert dlg.noise_percentile_spin.value() == pytest.approx(0.85)
    assert dlg.search_range_spin.value() == pytest.approx(0.03)
    assert dlg.min_peak_height_spin.value() == pytest.approx(1000.0)
    assert dlg.min_top_edge_ratio_spin.value() == pytest.approx(1.7)
    assert dlg.min_points_spin.value() == 4
    assert dlg.rt_tolerance_spin.value() == pytest.approx(0.04)
    assert dlg.min_r_spin.value() == pytest.approx(0.80)
    assert dlg.min_ions_spin.value() == 4
    assert not hasattr(dlg, "shape_sim_threshold_spin")


def test_adap_widgets_are_gone(qtbot, ini_settings):
    from ui.dialogs.ms_options_dialog import MSOptionsDialog
    dlg = MSOptionsDialog()
    qtbot.addWidget(dlg)
    for name in ("min_cluster_distance_spin", "min_cluster_size_spin",
                 "min_cluster_intensity_spin", "min_eic_prominence_spin",
                 "shape_sim_threshold_spin", "model_peak_choice_combo"):
        assert not hasattr(dlg, name), name


def test_accept_saves_through_helper(qtbot, ini_settings):
    from ui.dialogs.ms_options_dialog import MSOptionsDialog
    from logic.spectral_deconv_settings import load_spectral_deconv_settings
    dlg = MSOptionsDialog()
    qtbot.addWidget(dlg)
    dlg.min_r_spin.setValue(0.9)
    dlg.min_ions_spin.setValue(3)
    dlg.excluded_mz_edit.setText("73, 147")
    dlg.accept()
    dp, _ = load_spectral_deconv_settings(QSettings(ini_settings, QSettings.IniFormat))
    assert dp.min_r == pytest.approx(0.9) and dp.min_ions == 3 and dp.excluded_mz == [73.0, 147.0]


def test_dialog_loads_saved_values(qtbot, ini_settings):
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    from logic.spectral_deconv_settings import save_spectral_deconv_settings
    from ui.dialogs.ms_options_dialog import MSOptionsDialog
    s = QSettings(ini_settings, QSettings.IniFormat)
    save_spectral_deconv_settings(
        s,
        CorrelationDeconvParams(noise_percentile=0.7, search_range=0.08, min_peak_height=2500.0,
                                min_top_edge_ratio=2.2, min_points=6, rt_tolerance=0.02,
                                min_r=0.95, min_ions=5, excluded_mz=[73.0, 147.0]),
        WindowGroupingParams(gap_tolerance=0.3, padding_fraction=0.4,
                             rt_match_tolerance=0.07, max_window_peaks=7),
    )
    s.sync()
    dlg = MSOptionsDialog()
    qtbot.addWidget(dlg)
    assert dlg.noise_percentile_spin.value() == pytest.approx(0.7)
    assert dlg.search_range_spin.value() == pytest.approx(0.08)
    assert dlg.min_peak_height_spin.value() == pytest.approx(2500.0)
    assert dlg.min_top_edge_ratio_spin.value() == pytest.approx(2.2)
    assert dlg.min_points_spin.value() == 6
    assert dlg.rt_tolerance_spin.value() == pytest.approx(0.02)
    assert dlg.min_r_spin.value() == pytest.approx(0.95)
    assert dlg.min_ions_spin.value() == 5
    assert dlg.excluded_mz_edit.text() == "73, 147"
    assert dlg.gap_tolerance_spin.value() == pytest.approx(0.3)
    assert dlg.padding_fraction_spin.value() == pytest.approx(0.4)
    assert dlg.rt_match_tolerance_spin.value() == pytest.approx(0.07)
    assert dlg.max_window_peaks_spin.value() == 7


def test_restore_defaults_resets_deconv_widgets(qtbot, ini_settings):
    from ui.dialogs.ms_options_dialog import MSOptionsDialog
    dlg = MSOptionsDialog()
    qtbot.addWidget(dlg)
    dlg.noise_percentile_spin.setValue(0.5)
    dlg.search_range_spin.setValue(0.2)
    dlg.min_peak_height_spin.setValue(5000.0)
    dlg.min_top_edge_ratio_spin.setValue(3.0)
    dlg.min_points_spin.setValue(9)
    dlg.rt_tolerance_spin.setValue(0.2)
    dlg.min_r_spin.setValue(0.5)
    dlg.min_ions_spin.setValue(9)
    dlg.excluded_mz_edit.setText("73")
    dlg._restore_defaults()
    assert dlg.noise_percentile_spin.value() == pytest.approx(0.85)
    assert dlg.search_range_spin.value() == pytest.approx(0.03)
    assert dlg.min_peak_height_spin.value() == pytest.approx(1000.0)
    assert dlg.min_top_edge_ratio_spin.value() == pytest.approx(1.7)
    assert dlg.min_points_spin.value() == 4
    assert dlg.rt_tolerance_spin.value() == pytest.approx(0.04)
    assert dlg.min_r_spin.value() == pytest.approx(0.80)
    assert dlg.min_ions_spin.value() == 4
    assert dlg.excluded_mz_edit.text() == ""


def test_options_dialog_rejects_bad_excluded_mz(qtbot, ini_settings, monkeypatch):
    from ui.dialogs import ms_options_dialog as mod
    warned = []
    monkeypatch.setattr(mod.QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dlg = mod.MSOptionsDialog()
    qtbot.addWidget(dlg)
    dlg.excluded_mz_edit.setText("73, abc")
    dlg.accept()
    assert warned, "a warning must be shown"
    # result() is an int; compare with the enum's int value (comparing to the enum itself is vacuous)
    assert dlg.result() != QDialog.DialogCode.Accepted.value
    s = QSettings(ini_settings, QSettings.IniFormat)
    assert s.value("ms_spectral_deconv/excluded_mz", None) is None   # nothing saved
    assert s.allKeys() == []   # no tab's settings were written, not only the m/z key
