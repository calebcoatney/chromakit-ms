"""JSON exporter records peak windows and break-point overrides."""
from logic.json_exporter import _build_processing_metadata


def _params(**extra):
    p = {'peaks': {'enabled': True, 'mode': 'classical', 'min_prominence': 0.02},
         'baseline': {'method': 'asls', 'break_points': [
             {'time': 4.55, 'tolerance': 0.1, 'overrides': {'lambda': 1e4}, 'anchor': None}]}}
    p.update(extra)
    return p


def test_exporter_records_windows():
    windows = [{'id': 'w1', 'start': 4.2, 'end': 5.1, 'peaks': {'min_prominence': 5.0}}]
    meta = _build_processing_metadata(_params(peak_windows=windows, outside_windows='global'))
    pd = meta['peak_detection']
    assert pd['peak_windows'] == windows
    assert pd['outside_windows'] == 'global'
    bp = meta['baseline']['break_points'][0]
    assert bp['overrides'] == {'lambda': 1e4} and 'anchor' in bp


def test_exporter_omits_windows_when_none():
    meta = _build_processing_metadata(_params())
    assert 'peak_windows' not in meta['peak_detection']
    assert 'outside_windows' not in meta['peak_detection']
