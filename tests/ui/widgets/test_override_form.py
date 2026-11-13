import pytest
from PySide6.QtCore import Qt
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
    assert 'border: 1px solid #c0392b' in w.edit.styleSheet()
    assert w.has_error()
    w.set_value(None)
    assert w.edit.styleSheet() == ''
    assert not w.has_error()


def test_set_error_marks_field(qtbot):
    w = _field(qtbot, FLOAT)
    w.set_error('bad')
    assert w.edit.toolTip() == 'bad'
    assert w.has_error()
    w.set_error(None)
    assert w.edit.styleSheet() == ''
    assert not w.has_error()


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
    assert g.count_label.text() == 'same as whole run'
    g.set_inherited({'lambda': 1e5, 'fastchrom': {'half_window': 5}})
    g.fields['lambda'].set_value(1e4)
    g.fields['fastchrom.half_window'].set_value(9)
    assert g.values() == {'lambda': 1e4, 'fastchrom': {'half_window': 9}}
    assert g.count_label.text() == '2 changes'
    g.fields['fastchrom.half_window'].set_value(None)
    assert g.count_label.text() == '1 change'


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


def test_tags_and_changed_label(qtbot):
    w = _field(qtbot, FLOAT)
    w.show()
    assert w.tag_label.text() == 'same as whole run' and w.tag_label.isVisible()
    assert not w.reset_button.isVisible()
    w.set_value(2.0)
    assert w.reset_button.text() == 'changed \u21ba'
    assert w.reset_button.isVisible() and not w.tag_label.isVisible()
    w.reset_button.click()
    assert w.tag_label.isVisible() and w.value() is None
    c = _field(qtbot, FieldSpec('enabled', 'Enable', 'bool'), 'same as segment')
    c.show()
    assert c.tag_label is None   # a combo's first item carries the tag (fits a narrow pane)
    c.set_inherited(True)
    assert c.combo.itemText(0) == 'same as segment (on)'
    c.set_value(False)
    assert c.reset_button.isVisible()


def test_combo_inherit_item_grey_italic(qtbot):
    w = _field(qtbot, FieldSpec('enabled', 'Enable', 'bool'))
    assert w.combo.itemData(0, Qt.FontRole).italic()
    assert w.combo.itemData(0, Qt.ForegroundRole).color().name() == '#888888'


def test_unchanged_text_never_reparses(qtbot):
    w = _field(qtbot, FLOAT)
    w.set_value(1234567.89)
    assert w.edit.text() == '1234567.89'
    with qtbot.assertNotEmitted(w.changed):
        w.edit.editingFinished.emit()
    assert w.value() == 1234567.89


def test_has_errors_group(qtbot):
    g = OverrideGroup('Peaks', 'peaks', {}, {})
    qtbot.addWidget(g)
    assert not g.has_errors()
    g.fields['min_width'].edit.setText('abc')
    g.fields['min_width'].edit.editingFinished.emit()
    assert g.has_errors()
    g.fields['min_width'].reset_button.click()
    assert not g.has_errors()


def test_group_header_collapse(qtbot):
    g = OverrideGroup('Peaks', 'peaks', {}, {})
    qtbot.addWidget(g)
    g.show()
    assert g.toggle_button.text().startswith('\u25b8 Peaks')
    assert not g._body.isVisible() and g._body.isEnabled()
    g.toggle_button.click()
    assert g._body.isVisible() and g.toggle_button.text().startswith('\u25be Peaks')
    assert not g.isCheckable()
