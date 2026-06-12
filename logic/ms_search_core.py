"""Pure batch MS library search — no Qt dependency.

Extracted from BatchSearchWorker.run() (logic/batch_search.py:52-242) so the
GUI worker and the API endpoint share identical search logic.

Callers pass callbacks for progress / log / cancellation; the GUI wraps them
in Qt signal emitters, the API endpoint passes None or print().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from logic.spectrum_extractor import SpectrumExtractor


@dataclass
class BatchSearchSummary:
    """Outcome of a batch search run."""
    total_peaks: int
    successful_matches: int = 0
    saturated_peaks: int = 0
    cancelled: bool = False
    errors: list = field(default_factory=list)  # list[tuple[int, str]]


def format_casno(casno) -> str:
    """Format a CAS Registry Number with dashes.

    Pads to 9 digits then splits as ``NNNNNN-NN-N`` (the standard CAS format).
    Returns '' for empty / non-string input.
    """
    if not casno or not isinstance(casno, str):
        return ""
    padded = casno.zfill(9)
    return padded[:-3] + '-' + padded[-3:-1] + '-' + padded[-1:]


def _do_search(ms_toolkit, query_spectrum, options: dict) -> list:
    """Dispatch to the correct ms_toolkit search method based on options."""
    search_method = options.get('search_method', 'vector')
    if search_method == 'w2v':
        return ms_toolkit.search_w2v(
            query_spectrum,
            top_n=options.get('top_n', 5),
            intensity_power=options.get('intensity_power', 0.6),
            top_k_clusters=options.get('top_k_clusters', 1),
        )
    elif search_method == 'hybrid':
        return ms_toolkit.search_hybrid(
            query_spectrum,
            method=options.get('hybrid_method', 'auto'),
            top_n=options.get('top_n', 5),
            intensity_power=options.get('intensity_power', 0.6),
            weighting_scheme=options.get('weighting', 'NIST_GC'),
            composite=(options.get('similarity', 'composite') == 'composite'),
            unmatched_method=options.get('unmatched', 'keep_all'),
            top_k_clusters=options.get('top_k_clusters', 1),
        )
    return ms_toolkit.search_vector(
        query_spectrum,
        top_n=options.get('top_n', 5),
        composite=(options.get('similarity', 'composite') == 'composite'),
        weighting_scheme=options.get('weighting', 'NIST_GC'),
        unmatched_method=options.get('unmatched', 'keep_all'),
        top_k_clusters=options.get('top_k_clusters', 1),
    )


def _lookup_casno(ms_toolkit, compound_name: str) -> Optional[str]:
    """Look up a CAS number from the loaded library.

    Returns the formatted CAS string on a hit, or None on miss / when the
    library is unavailable. None semantics match the original
    BatchSearchWorker behavior so downstream `casno is None` checks remain
    valid.
    """
    try:
        if hasattr(ms_toolkit, 'library') and compound_name in ms_toolkit.library:
            compound = ms_toolkit.library[compound_name]
            if hasattr(compound, 'casno'):
                return format_casno(compound.casno)
    except Exception:
        pass
    return None


def run_batch_search(
    ms_toolkit,
    peaks: list,
    data_directory: str,
    options: dict,
    *,
    respect_manual_assignments: bool = True,
    progress_callback: Optional[Callable[[int, str, list], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    extractor: Optional[SpectrumExtractor] = None,
    ms_time_offset: float = 0.0,
) -> BatchSearchSummary:
    """Run MS library search across `peaks`, mutating them in place.

    For each peak:
      - If manual_assignment is set and respect_manual_assignments=True, skip search.
      - Use deconvolved_spectrum if present; otherwise extract via SpectrumExtractor.
      - Propagate saturation flags from the extracted spectrum onto the peak.
      - Search via ms_toolkit using options['search_method'] (vector|w2v|hybrid).
      - On match, set peak.compound_id, peak.Compound_ID, peak.Qual, peak.casno.
      - On per-peak error, record in summary.errors but continue with other peaks.

    Args:
        ms_toolkit: A loaded MSToolkit instance.
        peaks: List of ChromatographicPeak objects (mutated in place).
        data_directory: Path to the .D or .C directory (used by SpectrumExtractor).
        options: Search options dict; see ui/frames/ms.py:51-67 for defaults.
        respect_manual_assignments: If True, skip peaks with .manual_assignment=True.
        progress_callback: Optional callable(peak_index, label, results) per peak.
        log_callback: Optional callable(message) for human-readable progress lines.
        should_cancel: Optional callable() -> bool; loop breaks when it returns True.
        extractor: Optional pre-constructed SpectrumExtractor instance.
            When None (default), a fresh one is built internally. The GUI
            worker passes its own extractor so tests that patch the
            worker's instance continue to verify real behavior.
        ms_time_offset: Constant shift (minutes) applied to MS retention
            times when extracting spectra for each peak. Forwarded to
            SpectrumExtractor.extract_for_peak(). Default 0.0 (no shift).
            Mirrors the same parameter on /api/spectral-deconvolution.

    Returns:
        BatchSearchSummary with counts and per-peak errors.
    """
    summary = BatchSearchSummary(total_peaks=len(peaks))

    if 'mz_shift' in options:
        ms_toolkit.mz_shift = options['mz_shift']

    if extractor is None:
        extractor = SpectrumExtractor(debug=options.get('debug', False))

    def _log(msg: str):
        if log_callback is not None:
            log_callback(msg)

    _log(f"Starting batch search on {len(peaks)} peaks...")

    for i, peak in enumerate(peaks):
        if should_cancel is not None and should_cancel():
            summary.cancelled = True
            _log("Batch search cancelled by user")
            break

        # Manual assignment shortcut
        if respect_manual_assignments and getattr(peak, 'manual_assignment', False):
            compound_name = peak.compound_id
            if progress_callback is not None:
                progress_callback(i, compound_name, [(compound_name, 1.0)])
            _log(f"Peak {i+1}/{len(peaks)}: Using manual assignment '{compound_name}'")
            summary.successful_matches += 1
            continue

        # Extract or reuse deconvolved spectrum
        if getattr(peak, 'deconvolved_spectrum', None) is not None:
            spectrum = peak.deconvolved_spectrum
            if len(spectrum.get('mz', [])) == 0:
                spectrum = extractor.extract_for_peak(
                    data_directory, peak, options, ms_time_offset=ms_time_offset
                )
        else:
            spectrum = extractor.extract_for_peak(
                data_directory, peak, options, ms_time_offset=ms_time_offset
            )

        if not spectrum or 'mz' not in spectrum or 'intensities' not in spectrum or len(spectrum['mz']) == 0:
            if progress_callback is not None:
                progress_callback(i, f"No spectrum at RT {peak.retention_time:.3f}", [])
            continue

        # Saturation flag
        if spectrum.get('is_saturated'):
            peak.is_saturated = True
            peak.saturation_level = spectrum.get('saturation_level', 0)
            summary.saturated_peaks += 1
            _log(f"WARNING: Peak {peak.peak_number} at RT={peak.retention_time:.3f} shows saturation")
        else:
            peak.is_saturated = False

        # Search
        query_spectrum = list(zip(spectrum['mz'], spectrum['intensities']))
        try:
            results = _do_search(ms_toolkit, query_spectrum, options)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            summary.errors.append((i, err_msg))
            _log(f"Search error for peak {i+1} at RT={peak.retention_time:.3f}: {err_msg}")
            if progress_callback is not None:
                progress_callback(i, f"Search error at RT {peak.retention_time:.3f}", [])
            continue

        if results:
            best_name, best_score = results[0][0], results[0][1]
            peak.compound_id = best_name
            peak.Compound_ID = best_name
            peak.Qual = best_score
            peak.casno = _lookup_casno(ms_toolkit, best_name)
            if progress_callback is not None:
                progress_callback(i, best_name, results)
            summary.successful_matches += 1
            _log(f"Peak {i+1}/{len(peaks)}: {best_name} (score {best_score:.3f})")
        else:
            if progress_callback is not None:
                progress_callback(i, "No matches found", [])

    _log(
        f"Batch search {'cancelled' if summary.cancelled else 'completed'}: "
        f"{summary.successful_matches}/{summary.total_peaks} matched, "
        f"{summary.saturated_peaks} saturated, {len(summary.errors)} errors"
    )
    return summary
