"""Tests for logic/polyarc_library.py — compound library lookup."""
from pathlib import Path

import pytest

from logic.polyarc_library import CompoundRecord, PolyarcLibrary

FIXTURE = Path(__file__).parent.parent / 'data' / 'compounds_minimal.csv'


def test_load_from_csv_returns_expected_record_count():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert len(lib._records) == 8


def test_lookup_by_zero_padded_cas():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas='000064-19-7', name=None)
    assert rec is not None
    assert rec.compound == 'Acetic acid'
    assert rec.C == 2 and rec.O == 2


def test_lookup_by_unpadded_cas():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas='64-19-7', name=None)
    assert rec is not None
    assert rec.compound == 'Acetic acid'


def test_lookup_by_exact_name():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas=None, name='Phenol')
    assert rec is not None
    assert rec.cas == '000108-95-2'


def test_lookup_case_insensitive_name():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas=None, name='ACETIC ACID')
    assert rec is not None
    assert rec.compound == 'Acetic acid'


def test_lookup_unknown_returns_none():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas='999999-99-9', name='Bogus compound')
    assert rec is None


def test_lookup_cas_takes_precedence_over_name():
    """If CAS matches one compound and name matches another, CAS wins."""
    lib = PolyarcLibrary.from_csv(FIXTURE)
    # CAS of acetic acid, name of phenol → should return acetic acid
    rec = lib.lookup(cas='000064-19-7', name='Phenol')
    assert rec.compound == 'Acetic acid'


def test_lookup_with_both_none_returns_none():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert lib.lookup(cas=None, name=None) is None


def test_lookup_with_empty_strings_returns_none():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert lib.lookup(cas='', name='') is None


def test_empty_heteroatom_cells_parsed_as_zero():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas=None, name='Nonane')
    # Nonane has no O, S, N in the CSV (empty cells)
    assert rec.O == 0
    assert rec.S == 0
    assert rec.N == 0


def test_mw_recomputed_from_atom_counts():
    """MW from CSV is informational; loader recomputes from C/H/O/S/N."""
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas=None, name='Acetic acid')
    expected = 2 * 12.0107 + 4 * 1.0079 + 2 * 15.9994
    assert rec.MW == pytest.approx(expected, rel=1e-6)


def test_anchor_field_preserved():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert lib.lookup(cas=None, name='Acetic acid').anchor == 'nonane'
    assert lib.lookup(cas=None, name='Levoglucosan').anchor == 'levoglucosan'


def test_records_are_frozen_dataclasses():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(cas=None, name='Acetic acid')
    with pytest.raises((AttributeError, TypeError)):
        rec.compound = 'changed'  # frozen=True forbids mutation


def test_load_from_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        PolyarcLibrary.from_csv(Path('/nonexistent/path/compounds.csv'))


def test_pad_cas_helper():
    """Internal helper: pad first segment to 6 digits."""
    assert PolyarcLibrary._pad_cas('64-19-7') == '000064-19-7'
    assert PolyarcLibrary._pad_cas('000064-19-7') == '000064-19-7'
    assert PolyarcLibrary._pad_cas('111-84-2') == '000111-84-2'
    assert PolyarcLibrary._pad_cas('') == ''
    assert PolyarcLibrary._pad_cas('garbage') == 'garbage'  # not a CAS, pass through
