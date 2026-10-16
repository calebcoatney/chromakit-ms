"""Per-spectrum noise gate for exported mass spectra (MGF/MSP).

ChromSpace's TOF spectra carry a few hundred ions at noise level beside the real ones. The noise
floor is estimated from the spectrum itself: median and 1.4826 * MAD of its nonzero intensities.
Most ions are noise, so the median sits on the floor and the MAD is its spread.
"""
from __future__ import annotations

import numpy as np


def snr_gate(spectrum, min_snr: float) -> list[tuple[int, float]]:
    """Ions above median + min_snr * sigma, in input order; the base peak always survives.

    min_snr <= 0 returns the spectrum unchanged. A flat floor (sigma 0) keeps ions strictly above
    the median.
    """
    spec = list(spectrum or [])
    if not spec or min_snr <= 0:
        return spec
    inten = np.array([float(i) for _, i in spec])
    nz = inten[inten > 0]
    if nz.size == 0:
        return []
    med = float(np.median(nz))
    sigma = 1.4826 * float(np.median(np.abs(nz - med)))
    threshold = med + min_snr * sigma
    base = int(np.argmax(inten))
    return [(mz, i) for k, (mz, i) in enumerate(spec) if float(i) > threshold or k == base]
