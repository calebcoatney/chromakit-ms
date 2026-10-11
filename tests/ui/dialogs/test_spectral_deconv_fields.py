"""MS Options and the Inspector build their deconvolution settings from one spec."""
import pytest

pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox

from tests.ui.dialogs.test_spectral_deconv_inspector import _fake_rainbow, _peak

# spec key -> (MS Options attribute, Inspector attribute); app.py and tests use these names
ATTRS = {
    "gap_tolerance": ("gap_tolerance_spin", "_gap_spin"),
    "padding_fraction": ("padding_fraction_spin", "_padding_spin"),
    "rt_match_tolerance": ("rt_match_tolerance_spin", "_rt_match_spin"),
    "max_window_peaks": ("max_window_peaks_spin", "_max_window_peaks_spin"),
    "noise_percentile": ("noise_percentile_spin", "_noise_percentile_spin"),
    "search_range": ("search_range_spin", "_search_range_spin"),
    "min_peak_height": ("min_peak_height_spin", "_min_peak_height_spin"),
    "min_top_edge_ratio": ("min_top_edge_ratio_spin", "_min_top_edge_ratio_spin"),
    "min_points": ("min_points_spin", "_min_points_spin"),
    "rt_tolerance": ("rt_tolerance_spin", "_rt_tolerance_spin"),
    "min_r": ("min_r_spin", "_min_r_spin"),
    "min_ions": ("min_ions_spin", "_min_ions_spin"),
    "excluded_mz": ("excluded_mz_edit", "_excluded_mz_edit"),
}


@pytest.fixture
def both(qtbot, tmp_path, monkeypatch):
    """(MS Options dialog, Inspector), both on a temp INI (never Caleb's real plist)."""
    import ui.dialogs.ms_options_dialog as opt_mod
    import ui.dialogs.spectral_deconv_inspector as insp_mod
    path = str(tmp_path / "t.ini")
    factory = lambda *a, **k: QSettings(path, QSettings.IniFormat)
    monkeypatch.setattr(opt_mod, "QSettings", factory)
    monkeypatch.setattr(insp_mod, "QSettings", factory)
    _fake_rainbow(monkeypatch)
    from logic.correlation_grouping import CorrelationDeconvParams
    from logic.spectral_deconv_runner import WindowGroupingParams
    opt = opt_mod.MSOptionsDialog()
    qtbot.addWidget(opt)
    insp = insp_mod.SpectralDeconvInspectorDialog(
        peaks=[_peak(1.0, 0.96, 1.04)], ms_data_path="/fake.D",
        deconv_params=CorrelationDeconvParams(), grouping_params=WindowGroupingParams())
    qtbot.addWidget(insp)
    qtbot.waitUntil(lambda: insp._last_result is not None, timeout=10000)
    return opt, insp


def _form_label(widget):
    """Text of the QFormLayout label next to ``widget``."""
    form = widget.parentWidget().layout()
    assert isinstance(form, QFormLayout)
    return form.labelForField(widget).text()


def test_spec_covers_every_setting():
    from ui.dialogs.spectral_deconv_fields import FIELDS, SECTIONS
    assert set(FIELDS) == set(ATTRS)
    assert [k for _, keys in SECTIONS for k in keys] == list(ATTRS)


@pytest.mark.parametrize("key", list(ATTRS))
def test_both_dialogs_build_the_setting_from_the_spec(both, key):
    from ui.dialogs.spectral_deconv_fields import FIELDS, tooltip_html
    spec = FIELDS[key]
    opt, insp = both
    widgets = (getattr(opt, ATTRS[key][0]), getattr(insp, ATTRS[key][1]))
    for w in widgets:
        assert w.toolTip() == tooltip_html(spec.tooltip)
        assert _form_label(w) == spec.label + ":"
        if isinstance(w, QLineEdit):
            assert w.placeholderText() == spec.placeholder
            continue
        assert isinstance(w, QDoubleSpinBox if spec.kind == "float" else QSpinBox)
        assert (w.minimum(), w.maximum()) == pytest.approx((spec.minimum, spec.maximum))
        assert w.singleStep() == pytest.approx(spec.step)
        assert w.specialValueText() == spec.special_value_text
        if isinstance(w, QDoubleSpinBox):
            assert w.decimals() == spec.decimals


def test_drifted_limits_now_use_the_options_values(both):
    """Review #7: the Inspector had gap 0-10 step 0.01 and padding 0-5 with Qt's step of 1.0."""
    opt, insp = both
    for gap in (opt.gap_tolerance_spin, insp._gap_spin):
        assert (gap.minimum(), gap.maximum(), gap.singleStep()) == pytest.approx((0.0, 5.0, 0.05))
    for pad in (opt.padding_fraction_spin, insp._padding_spin):
        assert (pad.minimum(), pad.maximum(), pad.singleStep()) == pytest.approx((0.0, 2.0, 0.1))


def test_tooltips_say_which_way_to_move_each_number():
    """Review #6: every numeric setting's tooltip says when to raise it and when to lower it."""
    from ui.dialogs.spectral_deconv_fields import FIELDS
    for key, spec in FIELDS.items():
        assert spec.tooltip, key
        if spec.kind != "mz_list":
            assert "Raise it" in spec.tooltip and "lower it" in spec.tooltip, key
    assert "representative" not in " ".join(s.tooltip for s in FIELDS.values())
    assert "commas or spaces" in FIELDS["excluded_mz"].tooltip


def test_tooltips_are_rich_text_so_they_wrap():
    from ui.dialogs.spectral_deconv_fields import tooltip_html
    assert tooltip_html("a < b") == "<p>a &lt; b</p>"


def test_boxes_are_named_ion_grouping_in_both_dialogs(both):
    """Review #22: 'Grouping' sat next to 'Peak Window Grouping'."""
    for dlg in both:
        titles = [b.title() for b in dlg.findChildren(QGroupBox)]
        assert "Ion Grouping" in titles and "Grouping" not in titles
        assert "Peak Window Grouping" in titles and "Ion Peak Detection" in titles
