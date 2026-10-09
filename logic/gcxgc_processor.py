from __future__ import annotations

import numpy as np
from pybaselines import Baseline2D
from scipy.signal import find_peaks, peak_prominences

from logic.gcxgc_peak import GCxGC2DPeak
from logic.gcxgc_run import Frame2D, GCxGCRun

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class GCxGCProcessor:
    """Detects GCxGC peaks on one Frame2D of a GCxGCRun and fetches their spectra.

    Workflow: apply_baseline() (optional) -> detect_peaks() -> extract_spectrum(peak).
    """

    def __init__(self, frame: Frame2D, run: GCxGCRun, *, use_dbc: bool = True, apex_window: int = 10):
        self.frame = frame
        self.run = run
        self.n_mods, self.scans_per_mod = frame.data.shape
        self.use_dbc = use_dbc and run.dbc_lsc is not None
        self.apex_window = int(apex_window)
        self._corrected: np.ndarray | None = None

    @property
    def corrected(self) -> np.ndarray | None:
        return self._corrected

    def apply_baseline(self, lam: float = 1e5, diff_order: int = 2) -> np.ndarray:
        baseline, _ = Baseline2D().arpls(self.frame.data, lam=lam, diff_order=diff_order, num_eigens=(5, 5))
        self._corrected = np.maximum(self.frame.data - baseline, 0.0)
        return self._corrected

    def detect_peaks(self, min_height=None, min_prominence=None, rt2_grouping_tolerance=2,
                     min_sub_peaks=2) -> list[GCxGC2DPeak]:
        data = self._corrected if self._corrected is not None else self.frame.data
        sub_peaks: dict[int, list[tuple[int, float, int, int]]] = {}
        for i in range(self.n_mods):
            col = data[i]
            kwargs = {}
            if min_height is not None:
                kwargs["height"] = min_height
            if min_prominence is not None:
                kwargs["prominence"] = min_prominence
            idxs, _ = find_peaks(col, **kwargs)
            if len(idxs) == 0:
                continue
            _, left, right = peak_prominences(col, idxs)
            sub_peaks[i] = [(int(x), float(col[x]), int(lb), int(rb))
                            for x, lb, rb in zip(idxs, left.astype(int), right.astype(int))]
        groups = [g for g in _group_sub_peaks(sub_peaks, rt2_grouping_tolerance) if len(g) >= min_sub_peaks]
        peaks = [self._build_peak(n + 1, g, data) for n, g in enumerate(groups)]
        peaks.sort(key=lambda p: p.rt1)
        for n, p in enumerate(peaks):
            p.peak_number = p.feature_id = n + 1
        return peaks

    def _build_peak(self, number: int, group, data) -> GCxGC2DPeak:
        rows = np.array([g[0] for g in group])
        heights = np.array([g[2] for g in group])
        apex = int(np.argmax(heights))
        apex_row, apex_col = int(rows[apex]), int(group[apex][1])
        t1 = self.frame.t1_min
        volume = float(sum(_trapezoid(data[m][lb:rb + 1]) for m, _, _, lb, rb in group))
        peak = GCxGC2DPeak(
            peak_number=number,
            rt1=float(np.dot(t1[rows], heights) / heights.sum()),
            rt2=float(self.frame.t2_s[apex_col]),
            volume=volume,
            n_sub_peaks=len(group),
            mod_start=int(rows.min()),
            mod_end=int(rows.max()),
            apex_mod=apex_row,
            start_time=float(t1[rows.min()]),
            end_time=float(t1[rows.max()]),
        )
        peak.detection_signal = self.frame.signal
        peak.apex_scan = apex_col
        peak.apex_raw_index = self.run.to_raw(self.frame.signal, apex_row, apex_col)
        return peak

    def extract_spectrum(self, peak: GCxGC2DPeak) -> list[tuple[int, float]] | None:
        res = self.run.spectrum_at(peak.apex_raw_index, peak.detection_signal,
                                   source="dbc" if self.use_dbc else "lsc", window=self.apex_window)
        peak.spectrum_scan, peak.spectrum_source = res.scan, res.source
        peak.apex_found, peak.apex_distance_scans, peak.saturated = res.apex_found, res.apex_distance_scans, res.saturated
        if len(res.mz) == 0:
            return None
        return [(int(m), float(i)) for m, i in zip(res.mz, res.intensity)]


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
