from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from pybaselines import Baseline2D

from logic.gcxgc_peak import GCxGC2DPeak


class GCxGCProcessor:
    """Processes GCxGC FID+TIC data into GCxGC2DPeak objects.

    Workflow:
        1. apply_baseline()   — 2D arpls baseline correction on FID
        2. detect_peaks()     — column-by-column 1D detection + sub-peak grouping
        3. extract_spectrum() — per-peak on demand for MS search
    """

    def __init__(
        self,
        fid_2d: np.ndarray,
        tic_2d: np.ndarray,
        pm: float,
        hz: float,
        lsc_path: str,
        phase_scans: int = 0,
        dbc_lsc_path: str | None = None,
        use_dbc: bool = True,
    ):
        self.fid_2d = fid_2d
        self.tic_2d = tic_2d
        self.pm = pm
        self.hz = hz
        self.lsc_path = lsc_path
        self.phase_scans = phase_scans
        self.n_mods, self.scans_per_mod = fid_2d.shape
        self._fid_corrected: np.ndarray | None = None

        # DBC (dynamic background corrected) spectrum source.
        # Both .lsc and .dbc.lsc share the same scan rate; .dbc.lsc simply has
        # fewer peaks per scan (background noise removed by SepSolve's DBC algorithm).
        self.dbc_lsc_path = dbc_lsc_path
        # Use dbc for spectrum extraction if available and requested
        self.use_dbc = use_dbc and dbc_lsc_path is not None

    def apply_baseline(self, lam: float = 1e5, diff_order: int = 2) -> np.ndarray:
        """Fit and subtract 2D arpls baseline from FID.

        Args:
            lam: smoothing parameter (larger = smoother baseline)
            diff_order: differential order (default 2)

        Returns:
            Baseline-corrected FID array, clipped to non-negative, shape = fid_2d.shape
        """
        fitter = Baseline2D()
        baseline, _ = fitter.arpls(
            self.fid_2d, lam=lam, diff_order=diff_order, num_eigens=(5, 5)
        )
        self._fid_corrected = np.maximum(self.fid_2d - baseline, 0.0)
        return self._fid_corrected

    def detect_peaks(
        self,
        min_height: float | None = None,
        min_prominence: float | None = None,
        rt2_grouping_tolerance: int = 2,
        min_sub_peaks: int = 2,
    ) -> list[GCxGC2DPeak]:
        """Detect 2D peaks via column-by-column 1D detection and sub-peak grouping.

        Args:
            min_height: absolute FID threshold; None disables height filter
            min_prominence: scipy prominence threshold; None disables
            rt2_grouping_tolerance: max scan-index difference between adjacent
                                    sub-peaks to be grouped into the same compound
            min_sub_peaks: minimum modulation slices a compound must span

        Returns:
            List of GCxGC2DPeak objects sorted by rt1
        """
        fid = self._fid_corrected if self._fid_corrected is not None else self.fid_2d

        # Step 1: find sub-peaks in each modulation column
        # sub_peaks[mod_idx] = [(scan_idx, apex_intensity, left_base, right_base), ...]
        sub_peaks: dict[int, list[tuple[int, float, int, int]]] = {}
        for i in range(self.n_mods):
            col = fid[i]
            kwargs: dict = {}
            if min_height is not None:
                kwargs['height'] = min_height
            if min_prominence is not None:
                kwargs['prominence'] = min_prominence
            idxs, _ = find_peaks(col, **kwargs)
            if len(idxs) == 0:
                continue
            prominences, left_bases, right_bases = peak_prominences(col, idxs)
            sub_peaks[i] = [
                (int(idx), float(col[idx]), int(lb), int(rb))
                for idx, lb, rb in zip(idxs, left_bases.astype(int), right_bases.astype(int))
            ]

        # Step 2: group sub-peaks across adjacent modulations
        groups = _group_sub_peaks(sub_peaks, rt2_grouping_tolerance)

        # Step 3: filter and build peak objects
        peaks = []
        for n, group in enumerate(groups):
            if len(group) < min_sub_peaks:
                continue
            peaks.append(self._build_peak(n + 1, group, fid))

        peaks.sort(key=lambda p: p.rt1)
        # Renumber after sort
        for n, p in enumerate(peaks):
            p.peak_number = n + 1
            p.feature_id = n + 1

        return peaks

    def _build_peak(
        self,
        peak_number: int,
        group: list[tuple[int, int, float, int, int]],
        fid: np.ndarray,
    ) -> GCxGC2DPeak:
        """Construct a GCxGC2DPeak from a list of sub-peak tuples.

        group items: (mod_idx, scan_idx, apex_intensity, left_base, right_base)
        """
        mod_indices  = [item[0] for item in group]
        scan_indices = [item[1] for item in group]
        intensities  = [item[2] for item in group]

        apex_pos  = int(np.argmax(intensities))
        apex_scan = scan_indices[apex_pos]
        apex_mod_idx = mod_indices[apex_pos]

        total_intensity = sum(intensities)
        # rt1: intensity-weighted mean modulation index → minutes
        rt1 = (
            sum(m * i for m, i in zip(mod_indices, intensities))
            / total_intensity
            * self.pm / 60.0
        )
        # rt2: apex scan index → seconds
        rt2 = apex_scan / self.hz

        # volume: sum of trapezoid areas under each modulation slice
        volume = 0.0
        for mod_idx, scan_idx, apex_int, lb, rb in group:
            volume += float(np.trapz(fid[mod_idx][lb: rb + 1]))

        mod_start  = min(mod_indices)
        mod_end    = max(mod_indices)
        start_time = mod_start * self.pm / 60.0
        end_time   = mod_end   * self.pm / 60.0

        return GCxGC2DPeak(
            peak_number=peak_number,
            rt1=rt1,
            rt2=rt2,
            volume=volume,
            n_sub_peaks=len(group),
            mod_start=mod_start,
            mod_end=mod_end,
            apex_mod=apex_mod_idx,
            start_time=start_time,
            end_time=end_time,
        )

    def extract_spectrum(self, peak: GCxGC2DPeak) -> list[tuple[int, float]] | None:
        """Extract the MS spectrum for a peak.

        Uses the DBC (.dbc.lsc) file when available and ``use_dbc=True``, falling
        back to the plain .lsc otherwise.  Both files share the same scan rate and
        scan indices; the DBC file simply has fewer peaks per scan (background removed).

        Critically, the apex scan is always located using ``self.tic_2d`` (the
        phase-shifted 2D TIC array) — NOT from the DBC file directly.  Both files
        use the same scan index so the same argmax applies to either.  Using DBC
        blocks to locate the apex would introduce a second, incorrect phase correction.

        Scan index: apex_mod * scans_per_mod + real_scan_within_mod, where the phase
        shift (applied as np.roll before this processor was created) is undone to
        recover the raw file scan index.
        """
        from logic.loaders.sepsolve_loader import extract_spectrum as _extract

        apex_mod = peak.apex_mod
        if apex_mod >= self.n_mods:
            apex_mod = self.n_mods - 1

        # Locate apex scan using the phase-shifted TIC 2D (works for both files).
        apex_scan_shifted = int(np.argmax(self.tic_2d[apex_mod]))
        real_scan = (apex_scan_shifted - self.phase_scans) % self.scans_per_mod
        scan_index = apex_mod * self.scans_per_mod + real_scan

        if self.use_dbc:
            result = _extract(self.dbc_lsc_path, scan_index)
            if result:
                return result
            # Fall through to plain .lsc on failure

        return _extract(self.lsc_path, scan_index)


# ── Module-level grouping helper ───────────────────────────────────────────────

def _group_sub_peaks(
    sub_peaks: dict[int, list[tuple[int, float, int, int]]],
    tolerance: int,
) -> list[list[tuple[int, int, float, int, int]]]:
    """Group sub-peaks from adjacent modulations into compound peaks.

    Args:
        sub_peaks: {mod_idx: [(scan_idx, apex_intensity, left_base, right_base), ...]}
        tolerance: max scan-index difference to consider two sub-peaks the same compound

    Returns:
        List of groups. Each group is a list of
        (mod_idx, scan_idx, apex_intensity, left_base, right_base) tuples.
    """
    if not sub_peaks:
        return []

    active_groups: list[list] = []
    completed_groups: list[list] = []
    prev_mod = -2

    for mod_idx in sorted(sub_peaks.keys()):
        current = [
            (mod_idx, scan_idx, apex_int, lb, rb)
            for scan_idx, apex_int, lb, rb in sub_peaks[mod_idx]
        ]

        if mod_idx != prev_mod + 1:
            # Gap in modulations — close all active groups
            completed_groups.extend(active_groups)
            active_groups = [[sp] for sp in current]
            prev_mod = mod_idx
            continue

        # Greedily match current sub-peaks to active groups by closest scan_idx
        matched_groups: set[int] = set()
        matched_peaks: set[int] = set()

        for g_idx, group in enumerate(active_groups):
            last_scan = group[-1][1]  # scan_idx of last sub-peak in this group
            best_p_idx, best_dist = None, tolerance + 1
            for p_idx, sp in enumerate(current):
                if p_idx in matched_peaks:
                    continue
                dist = abs(sp[1] - last_scan)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_p_idx = p_idx
            if best_p_idx is not None:
                active_groups[g_idx].append(current[best_p_idx])
                matched_groups.add(g_idx)
                matched_peaks.add(best_p_idx)

        # Groups with no match are completed
        for g_idx, group in enumerate(active_groups):
            if g_idx not in matched_groups:
                completed_groups.append(group)

        # Keep matched groups; start new groups for unmatched peaks
        active_groups = [g for g_idx, g in enumerate(active_groups) if g_idx in matched_groups]
        for p_idx, sp in enumerate(current):
            if p_idx not in matched_peaks:
                active_groups.append([sp])

        prev_mod = mod_idx

    completed_groups.extend(active_groups)
    return completed_groups
