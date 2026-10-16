from logic.spectrum_noise import snr_gate


def _noisy(signal):
    """~200 noise ions at 1-3 counts with real ions on top (ion: intensity)."""
    spec = [(mz, 1.0 + (mz * 7919 % 200) / 100.0) for mz in range(30, 230)]
    d = dict(spec)
    d.update(signal)
    return sorted(d.items())


def test_noise_ions_are_dropped_and_real_ions_kept():
    out = dict(snr_gate(_noisy({149: 10000.0, 223: 430.0, 278: 49.0}), 5.0))
    assert set(out) == {149, 223, 278}
    assert out[278] == 49.0


def test_base_peak_is_always_kept():
    spec = [(mz, 2.0) for mz in range(50, 75)]  # 25 nonzero ions, all at the floor
    assert snr_gate(spec, 5.0) == [(50, 2.0)]


def test_sparse_spectrum_passes_through_unchanged():
    spec = [(10, 100.0), (20, 90.0), (30, 80.0), (40, 1.0), (50, 1.0)]
    assert snr_gate(spec, 5.0) == spec


def test_only_base_peak_survives_for_a_weak_feature():
    spec = _noisy({91: 4.5})
    assert snr_gate(spec, 50.0) == [(91, 4.5)]


def test_zero_or_negative_min_snr_is_identity():
    spec = _noisy({149: 100.0})
    assert snr_gate(spec, 0) == spec
    assert snr_gate(spec, -1) == spec


def test_flat_spectrum_keeps_ions_strictly_above_the_median():
    spec = [(mz, 1.0) for mz in range(50, 74)] + [(74, 9.0)]  # 24 at the floor, one real ion
    assert snr_gate(spec, 5.0) == [(74, 9.0)]


def test_order_is_preserved_and_none_is_empty():
    spec = [(200, 900.0), (57, 1000.0)] + [(m, 1.0) for m in range(60, 120)]
    assert snr_gate(spec, 5.0) == [(200, 900.0), (57, 1000.0)]
    assert snr_gate(None, 5.0) == [] and snr_gate([], 5.0) == []
