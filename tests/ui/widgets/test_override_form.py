import pytest
pytest.importorskip('pytestqt')

from logic.method import (
    BaselineOverrides, NegativePeakOverrides, PeakOverrides, RiderOverrides,
    ShoulderOverrides, SmoothingOverrides,
)
from ui.widgets.override_form import FieldSpec, OverrideField, OverrideGroup, SECTION_FIELDS

MODELS = {
    'baseline': BaselineOverrides, 'smoothing': SmoothingOverrides, 'peaks': PeakOverrides,
    'shoulders': ShoulderOverrides, 'negative_peaks': NegativePeakOverrides, 'rider': RiderOverrides,
}


def _field(qtbot, spec, tag='same as whole run'):
    w = OverrideField(spec, tag)
    qtbot.addWidget(w)
    return w


FLOAT = FieldSpec('min_prominence', 'Min Prominence:', 'float')


def test_blank_float_inherits_and_shows_placeholder(qtbot):
    w = _field(qtbot, FLOAT)
    w.set_inherited(1e6)
    assert w.value() is None
    assert w.edit.placeholderText() == '1e+06'


def test_typing_sets_value_emits_once_and_shows_reset(qtbot):
    w = _field(qtbot, FLOAT)
    assert w.reset_button.isHidden()
    with qtbot.waitSignal(w.changed, timeout=500):
        w.edit.setText('1e3')
        w.edit.editingFinished.emit()
    assert w.value() == 1000.0
    assert not w.reset_button.isHidden()


def test_reset_button_clears(qtbot):
    w = _field(qtbot, FLOAT)
    w.set_value(5.0)
    assert w.value() == 5.0
    w.reset_button.click()
    assert w.value() is None
    assert w.reset_button.isHidden()
    assert w.edit.text() == ''


def test_invalid_text_is_none_and_red(qtbot):
    w = _field(qtbot, FLOAT)
    w.edit.setText('abc')
    w.edit.editingFinished.emit()
    assert w.value() is None
    assert 'red' in w.edit.styleSheet() or '#' in w.edit.styleSheet()
    w.set_value(None)
    assert w.edit.styleSheet() == ''


def test_set_error_marks_field(qtbot):
    w = _field(qtbot, FLOAT)
    w.set_error('bad')
    assert w.edit.toolTip() == 'bad'
    w.set_error(None)
    assert w.edit.styleSheet() == ''


def test_int_field(qtbot):
    w = _field(qtbot, FieldSpec('sensitivity', 'Sensitivity:', 'int'))
    w.edit.setText('7')
    w.edit.editingFinished.emit()
    assert w.value() == 7 and isinstance(w.value(), int)


def test_bool_field_first_item_is_inherit(qtbot):
    w = _field(qtbot, FieldSpec('enabled', 'Enable', 'bool'))
    w.set_inherited(True)
    assert w.combo.itemText(0) == 'same as whole run (on)'
    assert w.value() is None
    w.set_value(False)
    assert w.value() is False
    w.set_value(True)
    assert w.value() is True
    w.set_value(None)
    assert w.value() is None


def test_choice_field_and_custom_tag(qtbot):
    spec = FieldSpec('skim_model', 'Skim model:', 'choice',
                     (('Exponential', 'exponential'), ('Tangent', 'tangent')))
    w = _field(qtbot, spec, 'same as segment')
    w.set_inherited('tangent')
    assert w.combo.itemText(0) == 'same as segment (Tangent)'
    assert w.value() is None
    w.set_value('exponential')
    assert w.value() == 'exponential'


def test_combo_change_emits(qtbot):
    w = _field(qtbot, FieldSpec('enabled', 'Enable', 'bool'))
    with qtbot.waitSignal(w.changed, timeout=500):
        w.combo.setCurrentIndex(1)


def test_group_values_title_and_fastchrom_nesting(qtbot):
    g = OverrideGroup('Baseline', 'baseline', {'lambda': 1e6}, {})
    qtbot.addWidget(g)
    assert g.values() == {}
    assert 'same as whole run' in g.title()
    g.set_inherited({'lambda': 1e5, 'fastchrom': {'half_window': 5}})
    g.fields['lambda'].set_value(1e4)
    g.fields['fastchrom.half_window'].set_value(9)
    assert g.values() == {'lambda': 1e4, 'fastchrom': {'half_window': 9}}
    assert '2 changes' in g.title()
    g.fields['fastchrom.half_window'].set_value(None)
    assert '1 change' in g.title() and '1 changes' not in g.title()


def test_group_initial_values_and_signal(qtbot):
    g = OverrideGroup('Peaks', 'peaks', {'min_prominence': 1.0},
                      {'min_prominence': 3.0, 'min_width': 2.0})
    qtbot.addWidget(g)
    assert g.values() == {'min_prominence': 3.0, 'min_width': 2.0}
    with qtbot.waitSignal(g.changed, timeout=500):
        g.fields['min_width'].reset_button.click()
    assert g.values() == {'min_prominence': 3.0}


@pytest.mark.parametrize('section', sorted(MODELS))
def test_section_fields_subset_of_model(section):
    model = MODELS[section]
    aliases = {f.alias or n for n, f in model.model_fields.items()}
    keys = {s.key.split('.')[0] for s in SECTION_FIELDS[section]}
    assert keys <= aliases
    assert not keys & {'asymmetry', 'skim_chain_min_tail', 'skim_flag_band'}


def test_section_fields_cover_model():
    for section, model in MODELS.items():
        aliases = {f.alias or n for n, f in model.model_fields.items()}
        aliases -= {'asymmetry', 'skim_chain_min_tail', 'skim_flag_band'}
        assert {s.key.split('.')[0] for s in SECTION_FIELDS[section]} == aliases
