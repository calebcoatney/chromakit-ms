import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

GCGC_DIR = os.path.join("example GCxGC data", "GCGC")
HDR = os.path.join(GCGC_DIR, "8260-110 SAF.HDR")
RSD = os.path.join(GCGC_DIR, "8260-110 SAF.rsd")
REF_FID_2D = os.path.join("example GCxGC data", "extracted_8260", "fid_2d.npy")

from logic.loaders.sepsolve_loader import _parse_hdr, _parse_rsd, _reshape_to_2d

# HDR
hz, pm = _parse_hdr(HDR)
assert hz == 50.0, f"Expected hz=50.0, got {hz}"
assert pm == 0.5, f"Expected pm=0.5, got {pm}"
scans_per_mod = int(pm * hz)
assert scans_per_mod == 25, f"Expected scans_per_mod=25, got {scans_per_mod}"
print(f"✓ HDR: hz={hz}, pm={pm}, scans_per_mod={scans_per_mod}")

# FID
ref = np.load(REF_FID_2D)
fid_1d = _parse_rsd(RSD)
fid_2d = _reshape_to_2d(fid_1d, scans_per_mod)
assert fid_2d.shape == ref.shape, f"Shape mismatch: {fid_2d.shape} != {ref.shape}"
np.testing.assert_allclose(fid_2d, ref, rtol=1e-5)
print(f"✓ FID: shape={fid_2d.shape}, matches reference exactly")

from logic.loaders.sepsolve_loader import _find_sd04_offset, _parse_lsc_tic, extract_spectrum

DBC_LSC = os.path.join(GCGC_DIR, "8260-110 SAF_0.5_secs.dbc.lsc")

# TIC from .dbc.lsc
tic_1d = _parse_lsc_tic(DBC_LSC)
tic_2d = _reshape_to_2d(tic_1d, scans_per_mod)
assert tic_2d.shape[1] == scans_per_mod, f"Expected {scans_per_mod} scans/mod, got {tic_2d.shape[1]}"
assert tic_2d.max() > 0, "TIC is all zeros — file parsing failed"
print(f"✓ TIC from .dbc.lsc: shape={tic_2d.shape}, max={tic_2d.max():.0f}")

# Spectrum extraction — use scan 1000 as a basic sanity check
spectrum = extract_spectrum(DBC_LSC, 1000)
assert spectrum is not None, "Spectrum extraction returned None"
assert len(spectrum) > 0, "Empty spectrum at scan 1000"
mz_vals = [m for m, _ in spectrum]
assert all(isinstance(m, int) for m in mz_vals), "m/z values should be integers"
assert max(mz_vals) > 40, f"Suspiciously low max m/z: {max(mz_vals)}"
print(f"✓ Spectrum at scan 1000: {len(spectrum)} m/z bins, max m/z={max(mz_vals)}")

import tempfile, shutil
from logic.loaders.sepsolve_loader import SepSolveLoader

# Build a minimal .C-like folder structure in a temp dir
with tempfile.TemporaryDirectory() as tmp:
    data_dir = os.path.join(tmp, 'data')
    os.makedirs(data_dir)
    # Copy example files in
    for fname in ['8260-110 SAF.HDR', '8260-110 SAF.rsd']:
        shutil.copy(os.path.join(GCGC_DIR, fname), data_dir)
    shutil.copy(
        os.path.join(GCGC_DIR, '8260-110 SAF_0.5_secs.dbc.lsc'),
        data_dir
    )

    result = SepSolveLoader().load(tmp)

    assert result['metadata']['is_gcxgc'] is True
    assert result['metadata']['has_ms_data'] is True
    assert result['metadata']['fid_2d'].shape[1] == 25
    assert result['metadata']['tic_2d'].shape[1] == 25
    assert result['metadata']['pm'] == 0.5
    assert result['metadata']['hz'] == 50.0
    assert os.path.isfile(result['metadata']['lsc_path'])
    assert len(result['x']) == result['metadata']['fid_2d'].shape[0]
    print(f"✓ SepSolveLoader.load(): fid_2d={result['metadata']['fid_2d'].shape}, "
          f"tic_2d={result['metadata']['tic_2d'].shape}")

from logic.signal_profiles import SignalProfileRegistry
profile = SignalProfileRegistry.get('gcxgc')
assert profile.ui_mode == 'gcxgc'
assert profile.loader_class is SepSolveLoader
print("✓ gcxgc signal profile registered")
