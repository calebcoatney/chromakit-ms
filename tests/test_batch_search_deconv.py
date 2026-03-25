import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_peak(with_deconvolved=False):
    peak = MagicMock()
    peak.retention_time = 1.0
    peak.peak_number = 1
    peak.manual_assignment = False
    peak.is_saturated = False
    if with_deconvolved:
        peak.deconvolved_spectrum = {
            'mz': np.array([50.0, 73.0]),
            'intensities': np.array([1000.0, 500.0]),
        }
    else:
        peak.deconvolved_spectrum = None
    return peak


def test_uses_deconvolved_spectrum_when_available():
    """When peak.deconvolved_spectrum is set, extract_for_peak must NOT be called."""
    from logic.batch_search import BatchSearchWorker

    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = [('Compound A', 0.95)]

    peak = _make_peak(with_deconvolved=True)
    worker = BatchSearchWorker(ms_toolkit, [peak], '/fake/path.D', options={'search_method': 'vector', 'top_n': 5})

    with patch.object(worker.spectrum_extractor, 'extract_for_peak') as mock_extract:
        worker.run()
        mock_extract.assert_not_called()


def test_falls_back_to_extraction_when_no_deconvolved_spectrum():
    """When peak.deconvolved_spectrum is None, extract_for_peak must be called."""
    from logic.batch_search import BatchSearchWorker

    ms_toolkit = MagicMock()
    ms_toolkit.search_vector.return_value = [('Compound B', 0.90)]

    peak = _make_peak(with_deconvolved=False)
    worker = BatchSearchWorker(ms_toolkit, [peak], '/fake/path.D', options={'search_method': 'vector', 'top_n': 5})

    fake_spectrum = {'mz': np.array([50.0]), 'intensities': np.array([1000.0])}
    with patch.object(worker.spectrum_extractor, 'extract_for_peak', return_value=fake_spectrum):
        worker.run()
        worker.spectrum_extractor.extract_for_peak.assert_called_once()
