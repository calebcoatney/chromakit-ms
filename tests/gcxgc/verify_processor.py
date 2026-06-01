import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from logic.gcxgc_peak import GCxGC2DPeak

peak = GCxGC2DPeak(
    peak_number=1,
    rt1=5.0,
    rt2=0.24,
    volume=12345.6,
    n_sub_peaks=4,
    mod_start=600,
    mod_end=603,
    start_time=5.0,
    end_time=5.033,
)

assert peak.rt1 == 5.0
assert peak.rt2 == 0.24
assert peak.volume == 12345.6
assert peak.area == 12345.6, "Feature.area must equal volume for QuantitationCalculator"
assert peak.n_sub_peaks == 4
assert peak.compound_name is None
assert peak.match_score is None

d = peak.as_dict()
assert d['rt1'] == 5.0
assert d['rt2'] == 0.24
assert d['volume'] == 12345.6
assert d['n_sub_peaks'] == 4
assert 'compound_name' in d
assert 'mol_percent' in d
print("✓ GCxGC2DPeak: construction, Feature.area mapping, and as_dict() all correct")

from logic.signal_profiles import SignalProfileRegistry
profile = SignalProfileRegistry.get('gcxgc')
assert profile.feature_class is GCxGC2DPeak, "Profile feature_class not updated"
print("✓ gcxgc profile feature_class set to GCxGC2DPeak")

import numpy as np
from logic.gcxgc_processor import GCxGCProcessor

# Synthetic FID: constant background + a few peaks
rng = np.random.default_rng(42)
n_mods, scans = 100, 25
background = np.linspace(100, 200, n_mods)[:, None] * np.ones((1, scans))
fid_2d = background + rng.normal(0, 5, (n_mods, scans))
# Add a peak at mod 50, scan 12
fid_2d[48:53, 10:15] += 500

proc = GCxGCProcessor(
    fid_2d=fid_2d,
    tic_2d=np.zeros_like(fid_2d),
    pm=0.5,
    hz=50.0,
    lsc_path="dummy.lsc",
)

corrected = proc.apply_baseline(lam=1e4)
assert corrected.shape == fid_2d.shape, "Shape must be preserved"
assert corrected.min() >= 0, "Corrected signal must be non-negative"
# Background region should be near zero after correction
bg_region = corrected[:20, :].mean()
assert bg_region < 50, f"Background not adequately corrected: mean={bg_region:.1f}"
print(f"✓ apply_baseline: shape={corrected.shape}, bg_mean={bg_region:.2f}, min={corrected.min():.2f}")

# Detection test on synthetic data
rng2 = np.random.default_rng(0)
n_mods2, scans2 = 200, 25
fid_clean = rng2.normal(0, 2, (n_mods2, scans2)).clip(0)
# Compound A: apex at mod 60, scan 8, spans mods 58-62
for m in range(58, 63):
    dist = abs(m - 60)
    fid_clean[m, 6:11] += np.array([50, 150, 300, 150, 50]) * (1 - dist * 0.15)
# Compound B: apex at mod 140, scan 18, spans mods 138-142
for m in range(138, 143):
    dist = abs(m - 140)
    fid_clean[m, 16:21] += np.array([30, 100, 200, 100, 30]) * (1 - dist * 0.15)

proc2 = GCxGCProcessor(
    fid_2d=fid_clean,
    tic_2d=np.zeros_like(fid_clean),
    pm=0.5,
    hz=50.0,
    lsc_path="dummy.lsc",
)
peaks = proc2.detect_peaks(min_height=40.0, min_prominence=30.0, min_sub_peaks=2)

assert len(peaks) >= 2, f"Expected ≥2 peaks, got {len(peaks)}"
rt1_vals = sorted([p.rt1 for p in peaks])
assert any(0.4 < rt1 < 0.6 for rt1 in rt1_vals), f"No peak near rt1=0.5 min: {rt1_vals}"
assert any(1.1 < rt1 < 1.2 for rt1 in rt1_vals), f"No peak near rt1=1.167 min: {rt1_vals}"
for p in peaks:
    assert p.volume > 0, "Volume must be positive"
    assert p.n_sub_peaks >= 2
print(f"✓ detect_peaks: found {len(peaks)} peaks with correct rt1 positions")
print(f"  rt1 values: {[round(p.rt1, 3) for p in peaks]}")
print(f"  rt2 values: {[round(p.rt2, 3) for p in peaks]}")
