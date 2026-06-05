"""Tests for logic/polyarc_calibration.py — anchor calibration."""
from pathlib import Path

import pytest

from logic.polyarc_calibration import AnchorPoint, Calibration

MINIMAL = Path(__file__).parent.parent / 'data' / 'calibration_minimal.csv'
FPO11 = Path(__file__).parent.parent / 'data' / 'calibration_fpo11_sample1.csv'


def test_load_minimal_returns_both_anchors():
    cal = Calibration.from_csv(MINIMAL)
    assert cal.available_anchors() == {'nonane', 'levoglucosan'}


def test_getitem_returns_anchor_point():
    cal = Calibration.from_csv(MINIMAL)
    nonane = cal['nonane']
    assert isinstance(nonane, AnchorPoint)
    assert nonane.name == 'nonane'
    assert nonane.cas == '000111-84-2'
    assert nonane.C == 9
    assert nonane.MW == pytest.approx(128.259)
    assert nonane.known_wt_pct == pytest.approx(0.0440433)
    assert nonane.area == pytest.approx(28848616.5)


def test_getitem_unknown_anchor_raises_keyerror():
    cal = Calibration.from_csv(MINIMAL)
    with pytest.raises(KeyError):
        _ = cal['no_such_anchor']


def test_fpo11_fixture_loads_nonane_only():
    cal = Calibration.from_csv(FPO11)
    assert cal.available_anchors() == {'nonane'}


def test_anchor_point_is_frozen():
    cal = Calibration.from_csv(MINIMAL)
    nonane = cal['nonane']
    with pytest.raises((AttributeError, TypeError)):
        nonane.area = 0.0


def test_zero_area_in_csv_raises_at_load():
    bad_csv = MINIMAL.parent / 'calibration_zero_area.csv'
    bad_csv.write_text(
        'anchor,cas,C,MW,known_wt_pct,area,run_date,instrument_id,notes\n'
        'nonane,000111-84-2,9,128.259,0.04,0,2026-03-25,test,bad\n'
    )
    try:
        with pytest.raises(ValueError, match='area'):
            Calibration.from_csv(bad_csv)
    finally:
        bad_csv.unlink()


def test_negative_known_wt_pct_raises():
    bad_csv = MINIMAL.parent / 'calibration_negative_wt.csv'
    bad_csv.write_text(
        'anchor,cas,C,MW,known_wt_pct,area,run_date,instrument_id,notes\n'
        'nonane,000111-84-2,9,128.259,-0.01,1000,2026-03-25,test,bad\n'
    )
    try:
        with pytest.raises(ValueError, match='known_wt_pct'):
            Calibration.from_csv(bad_csv)
    finally:
        bad_csv.unlink()


def test_load_from_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        Calibration.from_csv(Path('/nonexistent/calibration.csv'))
