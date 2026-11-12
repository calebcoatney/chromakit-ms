"""Hybrid (peak-splitting) integration renumbers and merges peaks, so its classical peaks are split."""
from types import SimpleNamespace

import numpy as np

from logic.processor import ChromatogramProcessor
from tests.logic.tail_skim_signals import processed_from, rider_on_tail


def test_hybrid_classical_peaks_integrated_in_split_mode():
    from ui.app import ChromaKitApp
    pair = rider_on_tail(1.0, 3.0)
    rider = {'rider_integration': 'skim'}
    processed = processed_from(pair, rider=rider)
    processed['fitted_curves'] = []
    deconv_peak = SimpleNamespace(peak_number=1, retention_time=0.01)
    deconv = {'peaks': [deconv_peak], 'x_peaks': [np.zeros(3)], 'y_peaks': [np.zeros(3)],
              'baseline_peaks': [np.zeros(3)], 'retention_times': [0.01], 'integrated_areas': [0.0],
              'integration_bounds': [(0.0, 0.02)], 'peaks_list': [deconv_peak]}
    app = SimpleNamespace(current_processed=processed, processor=ChromatogramProcessor(),
                          _method_area_factor=lambda: None, _integrate_peak_splitting=lambda: deconv)
    merged = ChromaKitApp._integrate_hybrid(app, {'peak_splitting': {'windows': []}}, None, None)
    classical = [p for p in merged['peaks'] if p is not deconv_peak]
    assert len(classical) == 2
    assert all(p.skim_parent is None and p.skim_children is None and p.skim_decision is None
               and p.skim_model is None for p in classical)
    assert processed['rider'] is rider
