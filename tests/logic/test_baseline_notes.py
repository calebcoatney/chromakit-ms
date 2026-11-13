"""processed['baseline_notes'] reports dropped breaks and demoted sections."""
import copy

import numpy as np

from logic.processor import ChromatogramProcessor


def _run(break_points, ms_range=(2.0, 10.0)):
    proc = ChromatogramProcessor()
    params = copy.deepcopy(proc.default_params)
    params['baseline']['enabled'] = True
    params['baseline']['break_points'] = break_points
    x = np.linspace(0.0, 12.0, 1201)  # 0.01 min spacing
    rng = np.random.default_rng(0)
    y = 1.0 + rng.normal(0.0, 0.01, x.size)
    return proc.process(x, y, params=params, ms_range=ms_range)


def test_notes_empty_by_default():
    assert _run([])['baseline_notes'] == []


def test_dropped_break_noted():
    notes = _run([11.0])['baseline_notes']
    assert len(notes) == 1
    assert "11.00" in notes[0]
    assert notes[0] == "Break at 11.00 min is outside the MS range; ignored"


def test_short_section_noted():
    # Two breaks 3 samples (0.03 min) apart leave a section below MIN_SEGMENT_LEN.
    notes = _run([5.00, 5.03])['baseline_notes']
    assert len(notes) == 1
    assert "5.00" in notes[0] and "5.03" in notes[0]
    assert notes[0] == "Baseline section 5.00–5.03 min is too short to fit; left uncorrected"


def test_note_times_parallel_to_notes():
    assert _run([])['baseline_note_times'] == []
    dropped = _run([11.0])
    assert dropped['baseline_note_times'] == [11.0]
    short = _run([5.00, 5.03])
    assert short['baseline_note_times'] == [5.0]  # the section's start
    assert len(short['baseline_note_times']) == len(short['baseline_notes'])
