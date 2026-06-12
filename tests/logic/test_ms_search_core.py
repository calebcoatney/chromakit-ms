"""Tests for logic/ms_search_core.run_batch_search.

These mock MSToolkit and SpectrumExtractor so the tests run with no
library load or rainbow read. They assert peak mutation behavior matches
BatchSearchWorker semantics exactly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from logic.integration import ChromatographicPeak
from logic.ms_search_core import run_batch_search, BatchSearchSummary


def _make_peak(rt: float = 1.0, peak_number: int = 1,
                manual_assignment: bool = False,
                with_deconvolved: bool = False) -> ChromatographicPeak:
    peak = ChromatographicPeak(
        compound_id="unknown",
        peak_number=peak_number,
        retention_time=rt,
        integrator="BB",
        width=0.05,
        area=1000.0,
        start_time=rt - 0.025,
        end_time=rt + 0.025,
    )
    if manual_assignment:
        peak.manual_assignment = True
        peak.compound_id = "ManuallySet"
    if with_deconvolved:
        peak.deconvolved_spectrum = {
            'mz': np.array([50.0, 73.0]),
            'intensities': np.array([1000.0, 500.0]),
        }
    return peak


def _make_toolkit(top_match=("Hexane", 0.93)):
    ms = MagicMock()
    ms.search_vector.return_value = [top_match, ("Pentane", 0.51)]
    ms.search_w2v.return_value = [top_match]
    ms.search_hybrid.return_value = [top_match]
    # Library lookup for CAS
    compound = MagicMock()
    compound.casno = "110543"  # padded CAS for hexane
    ms.library = {top_match[0]: compound}
    return ms


def test_successful_match_mutates_peak_fields():
    """A matching peak gets compound_id, Compound_ID, Qual, casno set."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=True)

    summary = run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=[peak],
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
    )

    assert peak.compound_id == "Hexane"
    assert peak.Compound_ID == "Hexane"
    assert peak.Qual == pytest.approx(0.93)
    assert peak.casno is not None and len(peak.casno) > 0
    assert summary.successful_matches == 1
    assert summary.total_peaks == 1


def test_manual_assignment_skipped_when_respected():
    """Peaks with manual_assignment=True keep identity, search is not called."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(manual_assignment=True)

    run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=[peak],
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
        respect_manual_assignments=True,
    )

    ms_toolkit.search_vector.assert_not_called()
    assert peak.compound_id == "ManuallySet"


def test_manual_assignment_searched_when_not_respected():
    """When respect_manual_assignments=False, manual peaks are searched anyway."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(manual_assignment=True, with_deconvolved=True)

    run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=[peak],
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
        respect_manual_assignments=False,
    )

    ms_toolkit.search_vector.assert_called_once()
    assert peak.compound_id == "Hexane"


def test_deconvolved_spectrum_skips_extraction():
    """When peak.deconvolved_spectrum is populated, SpectrumExtractor is bypassed."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=True)

    with patch('logic.ms_search_core.SpectrumExtractor') as MockExtractor:
        run_batch_search(
            ms_toolkit=ms_toolkit,
            peaks=[peak],
            data_directory='/fake/path.D',
            options={'search_method': 'vector', 'top_n': 5},
        )
        MockExtractor.return_value.extract_for_peak.assert_not_called()


def test_no_deconvolved_falls_back_to_extraction():
    """When no deconvolved spectrum, SpectrumExtractor.extract_for_peak is called."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=False)
    fake_spectrum = {'mz': np.array([50.0, 73.0]), 'intensities': np.array([100.0, 200.0])}

    with patch('logic.ms_search_core.SpectrumExtractor') as MockExtractor:
        MockExtractor.return_value.extract_for_peak.return_value = fake_spectrum
        run_batch_search(
            ms_toolkit=ms_toolkit,
            peaks=[peak],
            data_directory='/fake/path.D',
            options={'search_method': 'vector', 'top_n': 5},
        )
        MockExtractor.return_value.extract_for_peak.assert_called_once()
    assert peak.compound_id == "Hexane"


def test_saturation_flag_propagates_from_spectrum():
    """When the extracted spectrum reports is_saturated, the flag goes onto the peak."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=False)
    saturated_spectrum = {
        'mz': np.array([50.0, 73.0]),
        'intensities': np.array([100.0, 200.0]),
        'is_saturated': True,
        'saturation_level': 1.2e9,
    }
    with patch('logic.ms_search_core.SpectrumExtractor') as MockExtractor:
        MockExtractor.return_value.extract_for_peak.return_value = saturated_spectrum
        summary = run_batch_search(
            ms_toolkit=ms_toolkit,
            peaks=[peak],
            data_directory='/fake/path.D',
            options={'search_method': 'vector', 'top_n': 5},
        )
    assert peak.is_saturated is True
    assert peak.saturation_level == 1.2e9
    assert summary.saturated_peaks == 1


def test_empty_spectrum_recorded_no_match():
    """A peak whose spectrum extraction returns empty mz is not matched."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=False)
    empty_spectrum = {'mz': np.array([]), 'intensities': np.array([])}
    with patch('logic.ms_search_core.SpectrumExtractor') as MockExtractor:
        MockExtractor.return_value.extract_for_peak.return_value = empty_spectrum
        summary = run_batch_search(
            ms_toolkit=ms_toolkit,
            peaks=[peak],
            data_directory='/fake/path.D',
            options={'search_method': 'vector', 'top_n': 5},
        )
    assert summary.successful_matches == 0
    assert peak.compound_id == "unknown"


def test_progress_callback_invoked_per_peak():
    """progress_callback gets called once per peak with (index, label, results)."""
    ms_toolkit = _make_toolkit()
    peaks = [_make_peak(rt=i * 0.5, peak_number=i, with_deconvolved=True) for i in range(3)]
    calls = []
    def on_progress(index, label, results):
        calls.append((index, label, len(results) if results else 0))

    run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=peaks,
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
        progress_callback=on_progress,
    )
    assert len(calls) == 3
    assert [c[0] for c in calls] == [0, 1, 2]


def test_should_cancel_halts_loop():
    """When should_cancel returns True, the loop breaks and summary.cancelled=True."""
    ms_toolkit = _make_toolkit()
    peaks = [_make_peak(rt=i * 0.5, peak_number=i, with_deconvolved=True) for i in range(5)]
    cancel_after = {'count': 0}
    def should_cancel():
        cancel_after['count'] += 1
        return cancel_after['count'] > 2

    summary = run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=peaks,
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
        should_cancel=should_cancel,
    )
    assert summary.cancelled is True
    assert summary.total_peaks == 5
    matched = sum(1 for p in peaks if p.compound_id == "Hexane")
    assert matched <= 2


def test_search_method_w2v_dispatches_correctly():
    """options['search_method']='w2v' selects search_w2v, not search_vector."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=True)

    run_batch_search(ms_toolkit, [peak], '/x.D', {'search_method': 'w2v', 'top_n': 5})
    ms_toolkit.search_w2v.assert_called_once()
    ms_toolkit.search_vector.assert_not_called()


def test_search_exception_recorded_per_peak_not_aborting():
    """A search-time exception on one peak is recorded; other peaks continue."""
    ms_toolkit = _make_toolkit()
    ms_toolkit.search_vector.side_effect = [
        RuntimeError("toolkit blew up"),
        [("Toluene", 0.88)],
    ]
    second_compound = MagicMock()
    second_compound.casno = "108883"
    ms_toolkit.library = {"Toluene": second_compound}

    peaks = [
        _make_peak(rt=1.0, peak_number=1, with_deconvolved=True),
        _make_peak(rt=2.0, peak_number=2, with_deconvolved=True),
    ]
    summary = run_batch_search(ms_toolkit, peaks, '/x.D', {'search_method': 'vector', 'top_n': 5})

    assert summary.successful_matches == 1
    assert len(summary.errors) == 1
    assert summary.errors[0][0] == 0
    assert peaks[0].compound_id == "unknown"
    assert peaks[1].compound_id == "Toluene"


def test_lookup_casno_returns_none_when_compound_missing_from_library():
    """If the compound is matched but not in toolkit.library (or library is empty),
    peak.casno must be None (matching original BatchSearchWorker semantics), not ''."""
    ms = _make_toolkit()
    # Override: library has only "Pentane", but search will match "Hexane"
    pentane_compound = MagicMock()
    pentane_compound.casno = "109660"
    ms.library = {"Pentane": pentane_compound}
    # Hexane is the match returned by search_vector but not in library
    peak = _make_peak(with_deconvolved=True)
    run_batch_search(
        ms_toolkit=ms,
        peaks=[peak],
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
    )
    assert peak.compound_id == "Hexane"
    assert peak.casno is None, f"expected None on library miss, got {peak.casno!r}"


def test_caller_provided_extractor_is_used():
    """When an extractor is passed in, the function uses it instead of constructing one."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak(with_deconvolved=False)
    fake_spectrum = {'mz': np.array([50.0, 73.0]), 'intensities': np.array([100.0, 200.0])}

    injected_extractor = MagicMock()
    injected_extractor.extract_for_peak.return_value = fake_spectrum

    run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=[peak],
        data_directory='/fake/path.D',
        options={'search_method': 'vector', 'top_n': 5},
        extractor=injected_extractor,
    )
    injected_extractor.extract_for_peak.assert_called_once()
    assert peak.compound_id == "Hexane"


def test_run_batch_search_passes_ms_time_offset_to_extractor():
    """The ms_time_offset kwarg must reach SpectrumExtractor.extract_for_peak."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak()  # no deconvolved spectrum → forces extractor call

    fake_extractor = MagicMock()
    fake_extractor.extract_for_peak.return_value = {
        'mz': np.array([50.0, 73.0]),
        'intensities': np.array([1000.0, 500.0]),
    }

    run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=[peak],
        data_directory="/fake/dir.D",
        options={'search_method': 'vector'},
        extractor=fake_extractor,
        ms_time_offset=0.123,
    )

    # Verify the offset was forwarded as a keyword arg
    fake_extractor.extract_for_peak.assert_called_once()
    call_kwargs = fake_extractor.extract_for_peak.call_args.kwargs
    assert call_kwargs.get('ms_time_offset') == 0.123


def test_run_batch_search_default_offset_is_zero():
    """Existing callers (GUI BatchSearchWorker) that don't pass offset must still work."""
    ms_toolkit = _make_toolkit()
    peak = _make_peak()
    fake_extractor = MagicMock()
    fake_extractor.extract_for_peak.return_value = {
        'mz': np.array([50.0]),
        'intensities': np.array([1000.0]),
    }

    # Note: no ms_time_offset kwarg passed
    run_batch_search(
        ms_toolkit=ms_toolkit,
        peaks=[peak],
        data_directory="/fake/dir.D",
        options={'search_method': 'vector'},
        extractor=fake_extractor,
    )

    call_kwargs = fake_extractor.extract_for_peak.call_args.kwargs
    assert call_kwargs.get('ms_time_offset') == 0.0


def test_do_single_search_dispatches_vector():
    """do_single_search routes to search_vector when search_method='vector'."""
    from logic.ms_search_core import do_single_search

    ms_toolkit = _make_toolkit()
    query = [(50.0, 1000.0), (73.0, 500.0)]
    result = do_single_search(ms_toolkit, query, {'search_method': 'vector', 'top_n': 3})

    ms_toolkit.search_vector.assert_called_once()
    ms_toolkit.search_w2v.assert_not_called()
    ms_toolkit.search_hybrid.assert_not_called()
    assert result == [("Hexane", 0.93), ("Pentane", 0.51)]


def test_do_single_search_dispatches_w2v():
    """do_single_search routes to search_w2v when search_method='w2v'."""
    from logic.ms_search_core import do_single_search

    ms_toolkit = _make_toolkit()
    query = [(50.0, 1000.0)]
    do_single_search(ms_toolkit, query, {'search_method': 'w2v'})

    ms_toolkit.search_w2v.assert_called_once()
    ms_toolkit.search_vector.assert_not_called()


def test_do_single_search_dispatches_hybrid():
    """do_single_search routes to search_hybrid when search_method='hybrid'."""
    from logic.ms_search_core import do_single_search

    ms_toolkit = _make_toolkit()
    query = [(50.0, 1000.0)]
    do_single_search(ms_toolkit, query, {'search_method': 'hybrid'})

    ms_toolkit.search_hybrid.assert_called_once()


def test_do_search_alias_still_works():
    """The legacy _do_search name remains available for one release cycle."""
    from logic.ms_search_core import _do_search

    ms_toolkit = _make_toolkit()
    result = _do_search(ms_toolkit, [(50.0, 1000.0)], {'search_method': 'vector'})
    assert result == [("Hexane", 0.93), ("Pentane", 0.51)]


def test_lookup_casno_alias_still_works():
    """The legacy _lookup_casno name remains available for one release cycle."""
    from logic.ms_search_core import _lookup_casno, lookup_casno
    assert _lookup_casno is lookup_casno
