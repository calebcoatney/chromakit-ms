"""The envelope is a DRAW path: every emitted point must be real, and the
apex must survive decimation. These are the invariants that make it safe to
plot and unsafe to integrate."""
import json
import math
from pathlib import Path

import pytest

from logic.render_trace import Trace, envelope, build_trace

PARITY = Path(__file__).parent.parent / "data" / "envelope_parity.json"


def test_matches_rapids_bridge_envelope_exactly():
    """Pins the math across two repos that deliberately never import each other."""
    doc = json.loads(PARITY.read_text())
    for case in doc["cases"]:
        out_x, out_y = envelope(case["xs"], case["ys"], case["width"])
        assert out_x == case["out_x"], f"x mismatch at width={case['width']}"
        assert out_y == case["out_y"], f"y mismatch at width={case['width']}"


def test_every_emitted_pair_exists_in_the_input():
    xs = [i * 0.1 for i in range(100)]
    ys = [float((i * 37) % 91) for i in range(100)]
    out_x, out_y = envelope(xs, ys, 7)
    pairs = set(zip(xs, ys))
    assert all(pair in pairs for pair in zip(out_x, out_y))


def test_apex_survives_decimation():
    xs = [i * 0.1 for i in range(1000)]
    ys = [1.0] * 1000
    ys[437] = 9999.0          # the apex
    ys[812] = -500.0          # the trough
    trace = build_trace(xs, ys, max_points=50)
    assert 9999.0 in trace.plot_y
    assert -500.0 in trace.plot_y
    assert trace.rt_min[trace.plot_y.index(9999.0)] == pytest.approx(43.7)


def test_retention_axis_stays_strictly_increasing():
    xs = [i * 0.1 for i in range(500)]
    ys = [float((i * 13) % 71) for i in range(500)]
    trace = build_trace(xs, ys, max_points=40)
    assert all(b > a for a, b in zip(trace.rt_min, trace.rt_min[1:]))


def test_output_never_exceeds_max_points():
    xs = [i * 0.1 for i in range(48997)]
    ys = [float(i % 1000) for i in range(48997)]
    trace = build_trace(xs, ys, max_points=2000)
    assert trace.points <= 2000
    assert trace.points == len(trace.rt_min) == len(trace.plot_y)


def test_decimation_is_one_exactly_when_input_fits():
    xs = [0.1, 0.2, 0.3]
    ys = [1.0, 2.0, 3.0]
    assert build_trace(xs, ys, max_points=2000).decimation == 1
    big_x = [i * 0.1 for i in range(5000)]
    big_y = [float(i) for i in range(5000)]
    assert build_trace(big_x, big_y, max_points=100).decimation > 1


def test_refuses_non_finite_values_anywhere_in_the_raw_array():
    """Validating only the survivors would return a clean-looking chromatogram
    over a damaged run."""
    xs = [i * 0.1 for i in range(1000)]
    ys = [1.0] * 1000
    ys[500] = math.nan
    assert build_trace(xs, ys, max_points=20) is None


def test_refuses_a_string_that_looks_like_a_number():
    xs = [0.1, 0.2, 0.3]
    ys = [1.0, "2.0", 3.0]
    assert build_trace(xs, ys) is None


def test_refuses_a_non_increasing_time_axis():
    """A header with end <= start plots normally with every peak at the wrong time."""
    assert build_trace([0.1, 0.1, 0.2], [1.0, 2.0, 3.0]) is None
    assert build_trace([0.3, 0.2, 0.1], [1.0, 2.0, 3.0]) is None


def test_refuses_zero_samples():
    assert build_trace([], []) is None


def test_tolerates_exactly_one_surplus_time_but_not_two():
    assert build_trace([0.1, 0.2, 0.3], [1.0, 2.0]) is not None
    assert build_trace([0.1, 0.2, 0.3, 0.4], [1.0, 2.0]) is None


def test_refuses_surplus_intensities_outright():
    """An extra intensity is a real sample with no retention time."""
    assert build_trace([0.1, 0.2], [1.0, 2.0, 3.0]) is None


def test_bad_max_points_raises_rather_than_returning_none():
    """A mis-read config must not look like an instrument fault forever."""
    xs, ys = [0.1, 0.2], [1.0, 2.0]
    for bad in (0, 1, -5, True, 2.5, "2000"):
        with pytest.raises(ValueError):
            build_trace(xs, ys, max_points=bad)
