"""Per-spectrum noise gate for exported mass spectra (MGF/MSP).

ChromSpace's TOF spectra carry a few hundred ions at noise level beside the real ones. The noise
floor is estimated from the spectrum itself: median and 1.4826 * MAD of its nonzero intensities.
Most ions are noise, so the median sits on the floor and the MAD is its spread.
"""
from __future__ import annotations

import numpy as np

# The noise estimate needs a noise majority; real TOF spectra carry 200-560 ions.
MIN_IONS_FOR_NOISE_ESTIMATE = 20


def snr_gate(spectrum: list[tuple[int, float]] | None, min_snr: float) -> list[tuple[int, float]]:
    """Ions above median + min_snr * sigma, in input order; the base peak always survives.

    min_snr <= 0 returns the spectrum unchanged. A flat floor (sigma 0) keeps ions strictly above
    the median. A spectrum with fewer than MIN_IONS_FOR_NOISE_ESTIMATE nonzero ions passes through
    unchanged, because its median is a real ion rather than the noise floor.
    """
    spec = list(spectrum or [])
    if not spec or min_snr <= 0:
        return spec
    inten = np.array([float(i) for _, i in spec])
    nz = inten[inten > 0]
    if nz.size == 0:
        return []
    if nz.size < MIN_IONS_FOR_NOISE_ESTIMATE:
        return spec
    med = float(np.median(nz))
    sigma = 1.4826 * float(np.median(np.abs(nz - med)))
    threshold = med + min_snr * sigma
    base = int(np.argmax(inten))
    return [(mz, i) for k, (mz, i) in enumerate(spec) if float(i) > threshold or k == base]
