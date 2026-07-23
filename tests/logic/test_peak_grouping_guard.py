import numpy as np

from logic.integration import Integrator
from logic.feature import SpectralFeature


def test_peak_grouping_skips_spectral_features():
    sf = SpectralFeature(
        feature_id=1, position=1987.0, position_units="cm-1",
        area=10.0, width=5.0, start=1980.0, end=1994.0,
        start_index=0, end_index=10,
    )
    x = np.linspace(1800, 2200, 100)
    y = np.ones_like(x)
    baseline_y = np.zeros_like(x)
    # Must not raise AttributeError on peak.retention_time
    result = Integrator._apply_peak_grouping(
        [sf], [[1950.0, 2000.0]], x, y, baseline_y,
        [x], [y], [baseline_y], [1987.0], [10.0], [(1980.0, 1994.0)],
        profile=None,
    )
    # returns the peaks_list unchanged (first element of the returned tuple)
    assert result[0] == [sf]
