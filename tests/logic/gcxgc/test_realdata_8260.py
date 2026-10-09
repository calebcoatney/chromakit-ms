import numpy as np
import pytest

from logic.gcxgc_processor import GCxGCProcessor
from logic.gcxgc_run import GCxGCRun


@pytest.mark.realdata
def test_real_8260_pipeline(real_gcxgc_run):
    run = GCxGCRun.from_c_folder(real_gcxgc_run)
    proc = GCxGCProcessor(run.frame("fid"), run, use_dbc=True, apex_window=10)
    proc.apply_baseline(lam=1e5, diff_order=2)
    peaks = proc.detect_peaks(min_prominence=1e4, rt2_grouping_tolerance=2, min_sub_peaks=2)
    for p in peaks:
        proc.extract_spectrum(p)

    # Task 2 made apex_found strict: a faint FID peak with no distinct TOF apex reports False.
    found = [p for p in peaks if p.apex_found]
    near = [p for p in found if abs(p.apex_distance_scans) <= 3]
    wrong = 0
    # Regression guard, not independent validation: it reuses spectrum_at's own argmax and FID->TOF
    # mapping, so a wrong mapping would pass it. The independent evidence on this run is near_apex
    # and the four library-ID anchors (task 8 walkthrough, all found).
    for p in found:
        center = int(round(run.fid_to_tof(p.apex_raw_index)))
        lo = max(0, center - 12)
        local = lo + int(np.argmax(run.dbc_tic[lo:center + 13]))
        wrong += abs(p.spectrum_scan - local) > 8
    biggest = sorted(peaks, key=lambda p: p.volume, reverse=True)[:50]
    top50_found = sum(p.apex_found for p in biggest)
    print(f"\npeaks={len(peaks)} found={len(found) / len(peaks):.3f} "
          f"near_apex={len(near) / len(found):.3f} wrong_apex={wrong / len(found):.3f} "
          f"top50_found={top50_found}/{len(biggest)} "
          f"offset=({run.offset().a:.2f} + {run.offset().b:.4f}·t) ok={run.offset().ok}")
    assert 230 <= len(peaks) <= 312               # 271 ± 15 % before the fixes
    # Known miss: peak 236 (~72.87 min, 1.32 s) has a real TOF apex but sits 2.94 sigma above its
    # neighbourhood (crowded, so the MAD is inflated), just under the 3 sigma gate.
    assert top50_found >= 49
    assert len(near) / len(found) >= 0.95
    assert wrong / len(found) <= 0.05             # 28 % (over all peaks) before the fixes
