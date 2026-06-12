"""Tests for the new MSSearchRequest / MSSearchHit / MSSearchResponse Pydantic models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from pydantic import ValidationError


def test_ms_search_request_accepts_minimal_input():
    """MSSearchRequest accepts spectrum + uses defaults for the rest."""
    from api.models import MSSearchRequest
    req = MSSearchRequest(spectrum={'mz': [50.0, 73.0], 'intensities': [1000.0, 500.0]})
    assert req.spectrum == {'mz': [50.0, 73.0], 'intensities': [1000.0, 500.0]}
    assert req.options == {}
    assert req.mz_shift == 0
    assert req.top_n == 5


def test_ms_search_request_accepts_full_input():
    """MSSearchRequest accepts all optional fields."""
    from api.models import MSSearchRequest
    req = MSSearchRequest(
        spectrum={'mz': [50.0], 'intensities': [1000.0]},
        options={'search_method': 'w2v', 'intensity_power': 0.7},
        mz_shift=2,
        top_n=10,
    )
    assert req.options['search_method'] == 'w2v'
    assert req.mz_shift == 2
    assert req.top_n == 10


def test_ms_search_hit_required_fields():
    """MSSearchHit requires name and score; casno is optional."""
    from api.models import MSSearchHit
    hit = MSSearchHit(name='Hexane', score=0.91)
    assert hit.name == 'Hexane'
    assert hit.score == 0.91
    assert hit.casno is None

    hit2 = MSSearchHit(name='Hexane', score=0.91, casno='110-54-3')
    assert hit2.casno == '110-54-3'


def test_ms_search_hit_rejects_missing_score():
    """MSSearchHit requires a numeric score."""
    from api.models import MSSearchHit
    with pytest.raises(ValidationError):
        MSSearchHit(name='Hexane')


def test_ms_search_response_round_trips():
    """MSSearchResponse round-trips through JSON serialization."""
    from api.models import MSSearchResponse, MSSearchHit
    resp = MSSearchResponse(
        results=[MSSearchHit(name='Hexane', score=0.91, casno='110-54-3')],
        elapsed_seconds=0.042,
    )
    dumped = resp.model_dump()
    assert dumped['elapsed_seconds'] == 0.042
    assert dumped['results'][0]['name'] == 'Hexane'

    restored = MSSearchResponse.model_validate(dumped)
    assert restored.results[0].score == 0.91
