"""Tests for logic/ms_time.py -- MS time axis shift helper."""
import numpy as np
import pytest

from logic.ms_time import shifted_xlabels


class _FakeMS:
    """Minimal stand-in for a rainbow DataFile, exposing only .xlabels."""

    def __init__(self, xlabels):
        self.xlabels = np.asarray(xlabels, dtype=float)


def test_zero_offset_returns_equal_values():
    ms = _FakeMS([1.0, 2.0, 3.0])
    out = shifted_xlabels(ms, 0.0)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


def test_positive_offset_adds_to_each_value():
    ms = _FakeMS([1.0, 2.0, 3.0])
    out = shifted_xlabels(ms, 0.05)
    np.testing.assert_allclose(out, [1.05, 2.05, 3.05])


def test_negative_offset_subtracts_from_each_value():
    ms = _FakeMS([1.0, 2.0, 3.0])
    out = shifted_xlabels(ms, -0.05)
    np.testing.assert_allclose(out, [0.95, 1.95, 2.95])


def test_does_not_mutate_source_xlabels():
    ms = _FakeMS([1.0, 2.0, 3.0])
    _ = shifted_xlabels(ms, 0.1)
    np.testing.assert_array_equal(ms.xlabels, [1.0, 2.0, 3.0])


def test_returns_ndarray_even_for_list_input():
    ms = _FakeMS([1.0, 2.0, 3.0])
    ms.xlabels = [1.0, 2.0, 3.0]  # simulate non-array attribute
    out = shifted_xlabels(ms, 0.0)
    assert isinstance(out, np.ndarray)
