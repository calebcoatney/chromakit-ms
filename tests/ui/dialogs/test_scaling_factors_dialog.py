import pytest

pytest.importorskip("pytestqt")

from ui.dialogs.scaling_factors_dialog import ScalingFactorsDialog


def test_zero_signal_factor_coerced_to_one_on_accept(qtbot):
    dlg = ScalingFactorsDialog(None)
    qtbot.addWidget(dlg)
    dlg.signal_factor_spin.setValue(0.0)
    dlg.area_factor_spin.setValue(0.0)
    captured = {}
    dlg.factors_changed.connect(lambda s, a: captured.update(signal=s, area=a))
    dlg.accept()
    assert captured["signal"] == 1.0
    assert captured["area"] == 1.0
    assert dlg.signal_factor_spin.value() == 1.0
    assert dlg.area_factor_spin.value() == 1.0
