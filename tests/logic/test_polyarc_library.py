"""Tests for logic/polyarc_library.py — compound library lookup."""
from pathlib import Path

import pytest

from logic.polyarc_library import CompoundRecord, PolyarcLibrary

FIXTURE = Path(__file__).parent.parent / 'data' / 'compounds_minimal.csv'


def test_load_from_csv_returns_expected_record_count():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert len(lib.records) == 8


def test_lookup_by_zero_padded_cas():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno='000064-19-7', name=None)
    assert rec is not None
    assert rec.compound == 'Acetic acid'
    assert rec.C == 2 and rec.O == 2


def test_lookup_by_unpadded_cas():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno='64-19-7', name=None)
    assert rec is not None
    assert rec.compound == 'Acetic acid'


def test_lookup_by_exact_name():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno=None, name='Phenol')
    assert rec is not None
    assert rec.cas == '000108-95-2'


def test_lookup_case_insensitive_name():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno=None, name='ACETIC ACID')
    assert rec is not None
    assert rec.compound == 'Acetic acid'


def test_lookup_unknown_returns_none():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno='999999-99-9', name='Bogus compound')
    assert rec is None


def test_lookup_cas_takes_precedence_over_name():
    """If CAS matches one compound and name matches another, CAS wins."""
    lib = PolyarcLibrary.from_csv(FIXTURE)
    # CAS of acetic acid, name of phenol → should return acetic acid
    rec = lib.lookup(casno='000064-19-7', name='Phenol')
    assert rec.compound == 'Acetic acid'


def test_lookup_with_both_none_returns_none():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert lib.lookup(casno=None, name=None) is None


def test_lookup_with_empty_strings_returns_none():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert lib.lookup(casno='', name='') is None


def test_empty_heteroatom_cells_parsed_as_zero():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno=None, name='Nonane')
    # Nonane has no O, S, N in the CSV (empty cells)
    assert rec.O == 0
    assert rec.S == 0
    assert rec.N == 0


def test_mw_recomputed_from_atom_counts():
    """MW from CSV is informational; loader recomputes from C/H/O/S/N."""
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno=None, name='Acetic acid')
    expected = 2 * 12.0107 + 4 * 1.0079 + 2 * 15.9994
    assert rec.MW == pytest.approx(expected, rel=1e-6)


def test_anchor_field_preserved():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    assert lib.lookup(casno=None, name='Acetic acid').anchor == 'nonane'
    assert lib.lookup(casno=None, name='Levoglucosan').anchor == 'levoglucosan'


def test_records_are_frozen_dataclasses():
    lib = PolyarcLibrary.from_csv(FIXTURE)
    rec = lib.lookup(casno=None, name='Acetic acid')
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


def test_sentinel_cas_is_not_indexed_by_cas(tmp_path):
    """Compounds with placeholder CAS '0-00-0' should not be CAS-indexed.

    They are still findable by name. This prevents silent misrouting when
    ms-toolkit returns '0' for an unknown peak.
    """
    csv_path = tmp_path / 'sentinel.csv'
    csv_path.write_text(
        'compound,cas,group1,group2,group3,C,H,O,S,N,MW,anchor\n'
        'Unknown thing A,0-00-0,Aromatic,BTX,,6,6,,,,78.11,nonane\n'
        'Unknown thing B,000000-00-0,Alkane,n-Alkane,,9,20,,,,128.255,nonane\n'
    )
    lib = PolyarcLibrary.from_csv(csv_path)
    assert len(lib.records) == 2
    # Neither row is reachable by CAS
    assert lib.lookup(casno='0-00-0', name=None) is None
    assert lib.lookup(casno='000000-00-0', name=None) is None
    # Both still reachable by exact name
    assert lib.lookup(casno=None, name='Unknown thing A') is not None
    assert lib.lookup(casno=None, name='Unknown thing B') is not None


def test_duplicate_cas_warns_when_compound_names_differ(tmp_path, caplog):
    """When two rows share a CAS but have different compound names, warn."""
    import logging
    csv_path = tmp_path / 'dup.csv'
    csv_path.write_text(
        'compound,cas,group1,group2,group3,C,H,O,S,N,MW,anchor\n'
        'Compound X,000111-11-1,Alkane,,,9,20,,,,128.255,nonane\n'
        'Compound Y,000111-11-1,Alkane,,,9,20,,,,128.255,nonane\n'
    )
    with caplog.at_level(logging.WARNING, logger='logic.polyarc_library'):
        lib = PolyarcLibrary.from_csv(csv_path)
    # Last-write-wins by CAS
    rec = lib.lookup(casno='000111-11-1', name=None)
    assert rec.compound == 'Compound Y'
    # But both records remain
    assert len(lib.records) == 2
    # Warning was emitted
    assert any('Duplicate CAS' in m for m in caplog.messages), caplog.messages


def test_duplicate_cas_does_not_warn_for_identical_compound_name(tmp_path, caplog):
    """Same name twice with same CAS (e.g. stereoisomer entries) is silent."""
    import logging
    csv_path = tmp_path / 'dup_same_name.csv'
    csv_path.write_text(
        'compound,cas,group1,group2,group3,C,H,O,S,N,MW,anchor\n'
        'Decane,000124-18-5,Alkane,n-Alkane,,10,22,,,,142.282,nonane\n'
        'Decane,000124-18-5,Alkane,n-Alkane,,10,22,,,,142.282,nonane\n'
    )
    with caplog.at_level(logging.WARNING, logger='logic.polyarc_library'):
        PolyarcLibrary.from_csv(csv_path)
    assert not any('Duplicate CAS' in m for m in caplog.messages)
