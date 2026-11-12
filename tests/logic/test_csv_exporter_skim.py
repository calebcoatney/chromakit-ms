"""CSV export keeps columns that only later rows carry (the optional tail-skim fields)."""
import csv

import numpy as np

from logic.csv_exporter import export_results_to_csv
from logic.integration import Integrator
from tests.logic.tail_skim_signals import SIGMA, processed_from, rider_on_tail

SKIM_COLUMNS = ['skim_model', 'skim_parent', 'skim_source', 'area_before_skim', 'skimmed_area',
                'skim_children', 'skim_decision', 'skim_near_threshold']


def test_skim_columns_kept_when_first_peak_is_plain(tmp_path):
    pair = rider_on_tail(1.0, 3.0)
    x = pair['x']
    pair['y'] = pair['y'] + 0.5 * pair['y'].max() * np.exp(-0.5 * ((x - 0.08) / SIGMA) ** 2)
    processed = processed_from(pair, extra_peaks=[0.08], rider={'rider_integration': 'auto'})
    plain, parent, rider = Integrator.integrate(processed, verbose=False)['peaks']
    assert plain.skim_decision is None and parent.skim_children == [3] and rider.skim_parent == 2

    path = tmp_path / 'peaks.csv'
    assert export_results_to_csv([plain, parent, rider], str(path))
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    header = list(rows[0])
    assert header[:len(plain.as_dict())] == list(plain.as_dict())
    assert set(SKIM_COLUMNS) <= set(header)
    assert rows[0]['skim_parent'] == '' and rows[0]['area_before_skim'] == ''
    assert rows[1]['skim_children'] == '[3]'
    assert float(rows[1]['area_before_skim']) == parent.area_before_skim
    assert rows[2]['skim_parent'] == '2' and rows[2]['skim_model'] == rider.skim_model
