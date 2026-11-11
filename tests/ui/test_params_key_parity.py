"""ParametersFrame.current_params and ChromaMethod.to_gui_params() must list the same keys.

A processing parameter added on only one side is silently lost: a GUI-only key
never reaches the method file, and a method-only key has no control to restore.
Each section is compared on its own so a failure names the section and key.
"""
import pytest

pytest.importorskip('pytestqt')

from logic.method import ChromaMethod
from ui.frames.parameters import ParametersFrame

# Top-level method keys with no entry in current_params, on purpose: they are
# document data owned by other frames (RT/RF tables, quant, bands), not
# ParametersFrame processing parameters.
METHOD_ONLY_SECTIONS = {
    'bands', 'quant_strategy', 'rf_normalize', 'rf_table', 'rf_unit',
    'rt_matching', 'rt_table',
}

# Nested keys present only in the method schema, on purpose.
METHOD_ONLY_KEYS = {
    # Legacy savgol-style detection fields kept so old method files and API
    # payloads still validate. No code reads them and the GUI has no control.
    'peaks': {'window_length', 'polyorder', 'peak_prominence', 'peak_width'},
}

# Nested keys present only in current_params, on purpose. None today.
GUI_ONLY_KEYS = {}


@pytest.fixture(scope='module')
def key_sets(qapp):
    frame = ParametersFrame()
    gui = frame.current_params
    method = ChromaMethod(name='parity', signal_type='gc').to_gui_params()
    yield gui, method
    frame.close()
    frame.deleteLater()


def test_top_level_sections_match(key_sets):
    gui, method = key_sets
    assert set(gui) == set(method) - METHOD_ONLY_SECTIONS


SECTIONS = ['smoothing', 'baseline', 'peaks', 'peak_splitting', 'negative_peaks',
            'shoulders', 'integration', 'gcxgc']


# shortcut: compares keys one level deep (section -> key); keys inside a nested dict
# such as baseline['fastchrom'] are not checked. Recurse into nested dicts when a
# nested key goes missing on one side, or when another section gains a nested dict.
@pytest.mark.parametrize('section', SECTIONS)
def test_section_keys_match(key_sets, section):
    gui, method = key_sets
    assert isinstance(gui[section], dict) and isinstance(method[section], dict)
    gui_keys = set(gui[section]) - GUI_ONLY_KEYS.get(section, set())
    method_keys = set(method[section]) - METHOD_ONLY_KEYS.get(section, set())
    assert gui_keys == method_keys, (
        f"{section}: GUI-only {sorted(gui_keys - method_keys)}, "
        f"method-only {sorted(method_keys - gui_keys)}")


def test_every_gui_section_is_covered(key_sets):
    gui, _ = key_sets
    assert set(gui) == set(SECTIONS)
