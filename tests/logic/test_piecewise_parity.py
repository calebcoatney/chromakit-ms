"""Methods without windows must reproduce the output pinned from main."""
import json

import pytest

from tests.logic.piecewise_parity_cases import CASES, DATA_PATH, run_case

EXPECTED = json.loads(DATA_PATH.read_text())


def test_fixture_covers_all_cases():
    assert set(EXPECTED) == set(CASES)


@pytest.mark.parametrize("case", sorted(CASES))
def test_parity(case):
    got = run_case(case)
    want = EXPECTED[case]
    assert len(got) == len(want) > 0
    for g, w in zip(got, want):
        assert g['rt'] == pytest.approx(w['rt'], abs=1e-12)
        assert g['start'] == pytest.approx(w['start'], abs=1e-12)
        assert g['end'] == pytest.approx(w['end'], abs=1e-12)
        assert g['area'] == pytest.approx(w['area'], rel=1e-9)
