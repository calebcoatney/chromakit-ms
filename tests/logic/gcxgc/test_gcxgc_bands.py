"""A persistent band is not a peak, and nothing before the solvent cut-off is reported."""
from types import SimpleNamespace

import numpy as np
import pytest

from logic.gcxgc_processor import GCxGCProcessor
from logic.gcxgc_run import Frame2D

HZ, N, ROWS = 50.0, 200, 400


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _image(band=0.0, peaks=()):
    """Rows are modulations. peaks = (row, col, height); each spans a few modulations."""
    rows, cols = np.mgrid[0:ROWS, 0:N]
    rng = np.random.default_rng(0)
    data = band * _gauss(cols, 150, 2.0) * (1 + 0.02 * rng.standard_normal(ROWS))[:, None]
    for row, col, height in peaks:
        data = data + height * _gauss(rows, row, 1.2) * _gauss(cols, col, 2.0)
    return data


def _processor(data):
    frame = Frame2D(data=data, t1_min=np.arange(ROWS) * 4.0 / 60.0, t2_s=np.arange(N) / HZ,
                    hz=HZ, cut=0, N=N, signal="dbc_tic")
    run = SimpleNamespace(dbc_lsc=None, to_raw=lambda _signal, m, j: m * N + j)
    return GCxGCProcessor(frame, run)


def _detect(data, **kwargs):
    return _processor(data).detect_peaks(min_prominence=1e4, rt2_grouping_tolerance=2, min_sub_peaks=2, **kwargs)


def test_band_is_a_long_peak_when_not_removed():
    assert max(p.n_sub_peaks for p in _detect(_image(band=1e5))) > 100     # would fail if the fixture had no band


def test_band_alone_gives_no_peaks():
    assert _detect(_image(band=1e5), remove_bands=True) == []


def test_peak_on_a_band_gets_its_own_marker_at_its_apex():
    found = _detect(_image(band=1e5, peaks=[(100, 150, 2e6)]), remove_bands=True)
    assert len(found) == 1
    assert found[0].apex_mod == 100 and found[0].rt1 == pytest.approx(100 * 4.0 / 60.0, abs=4.0 / 60.0)
    assert found[0].n_sub_peaks < 15


def test_peak_beside_a_band_keeps_its_volume():
    data = _image(peaks=[(100, 60, 2e6)])
    plain = _detect(data)[0].volume
    with_band = _detect(_image(band=1e5, peaks=[(100, 60, 2e6)]), remove_bands=True)
    assert [p.apex_scan for p in with_band] == [60]
    assert with_band[0].volume == pytest.approx(plain, rel=0.01)


def test_short_peak_survives_band_removal_unchanged():
    data = _image(peaks=[(100, 60, 2e6), (300, 90, 1e6)])
    assert ([(p.apex_mod, p.apex_scan, p.volume) for p in _detect(data, remove_bands=True)]
            == [(p.apex_mod, p.apex_scan, p.volume) for p in _detect(data)])


def test_solvent_cut_off_drops_earlier_peaks_only():
    data = _image(peaks=[(30, 60, 2e6), (200, 60, 2e6)])          # 2.0 and 13.3 min
    assert [p.apex_mod for p in _detect(data, solvent_end_min=7.0)] == [200]


def test_peak_numbers_stay_consecutive_after_the_cut_off():
    data = _image(peaks=[(30, 60, 2e6), (200, 60, 2e6), (300, 60, 2e6)])
    assert [p.peak_number for p in _detect(data, solvent_end_min=7.0)] == [1, 2]


def test_solvent_cut_off_of_none_keeps_everything():
    assert len(_detect(_image(peaks=[(30, 60, 2e6), (200, 60, 2e6)]), solvent_end_min=None)) == 2
