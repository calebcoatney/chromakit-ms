"""ChromaMethod peak windows: schema, v1 migration, range-filter mirror."""
import logging

import pytest

from logic.method import ChromaMethod
from tests.logic.test_method import _GUI_PARAMS


def _v1(range_filters, **extra):
    d = {"name": "m", "version": "1", "signal_type": "gc",
         "peaks": {"range_filters": range_filters}}
    d.update(extra)
    return d


def _win(wid, start, end, **extra):
    return {"id": wid, "start": start, "end": end, **extra}


def test_v1_file_migrates_range_filters():
    m = ChromaMethod.model_validate(_v1([[2, 5], [7, 9]]))
    assert [(w.id, w.start, w.end) for w in m.peak_windows] == [("w1", 2, 5), ("w2", 7, 9)]
    for w in m.peak_windows:
        assert w.peaks.min_prominence is None and w.smoothing.enabled is None
        assert w.events == []
    assert m.outside_windows == "ignore"
    assert m.version == "2"


def test_overlapping_range_filters_merge_and_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="logic.method"):
        m = ChromaMethod.model_validate(_v1([[1, 5], [4, 8]]))
    assert [(w.start, w.end) for w in m.peak_windows] == [(1, 8)]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_v1_break_points_load_without_overrides():
    d = _v1([], baseline={"break_points": [{"time": 3.0, "tolerance": 0.2}]})
    m = ChromaMethod.model_validate(d)
    bp = m.baseline.break_points[0]
    assert bp.time == 3.0 and bp.tolerance == 0.2
    assert bp.anchor is None and bp.overrides.method is None


def test_v2_round_trip(tmp_path):
    m = ChromaMethod.model_validate({
        "name": "m", "signal_type": "gc",
        "peak_windows": [
            _win("a", 1, 4, peaks={"min_prominence": 5.0},
                 smoothing={"lambda": 0.5}),
            _win("b", 6, 9, shoulders={"enabled": True, "sensitivity": 5}),
        ],
        "baseline": {"break_points": [
            {"time": 0, "anchor": {"window_id": "a", "edge": "start"},
             "overrides": {"method": "snip"}},
            {"time": 7.5, "tolerance": 0.3, "overrides": {"lambda": 10.0}},
        ]},
    })
    path = tmp_path / "x.chromethod"
    m.to_file(path)
    m2 = ChromaMethod.from_file(path)
    assert m2.model_dump(by_alias=True) == m.model_dump(by_alias=True)
    assert m2.peak_windows[0].smoothing.lambda_ == 0.5
    assert m2.baseline.break_points[0].time == 1
    assert m2.baseline.break_points[1].overrides.lambda_ == 10.0


def test_mirror_written_under_ignore_and_empty_under_global():
    wins = [_win("a", 1, 4), _win("b", 6, 9)]
    m = ChromaMethod.model_validate({"name": "m", "signal_type": "gc", "peak_windows": wins})
    assert m.peaks.range_filters == [[1, 4], [6, 9]]
    g = ChromaMethod.model_validate({"name": "m", "signal_type": "gc", "peak_windows": wins,
                                     "outside_windows": "global"})
    assert g.peaks.range_filters == []
    assert len(g.peak_windows) == 2


def test_peak_windows_wins_over_range_filters():
    m = ChromaMethod.model_validate({
        "name": "m", "signal_type": "gc",
        "peak_windows": [_win("a", 1, 2)],
        "peaks": {"range_filters": [[10, 20]]},
    })
    assert [w.id for w in m.peak_windows] == ["a"]
    assert m.peaks.range_filters == [[1, 2]]


def test_anchor_follows_window_on_revalidate():
    m = ChromaMethod.model_validate({
        "name": "m", "signal_type": "gc",
        "peak_windows": [_win("w1", 2, 5)],
        "baseline": {"break_points": [
            {"time": 2, "anchor": {"window_id": "w1", "edge": "end"}}]},
    })
    assert m.baseline.break_points[0].time == 5
    m.peak_windows[0].start = 3
    m.peak_windows[0].end = 6
    m2 = ChromaMethod.model_validate(m.model_dump(by_alias=True))
    assert m2.baseline.break_points[0].time == 6


def test_dangling_anchor_raises():
    with pytest.raises(ValueError, match="missing window"):
        ChromaMethod.model_validate({
            "name": "m", "signal_type": "gc",
            "peak_windows": [_win("w1", 2, 5)],
            "baseline": {"break_points": [
                {"time": 2, "anchor": {"window_id": "zz", "edge": "end"}}]},
        })


def test_overlapping_windows_rejected():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate({
            "name": "m", "signal_type": "gc",
            "peak_windows": [_win("a", 1, 5), _win("b", 4, 8)],
        })


def test_from_gui_params_without_window_keys_migrates():
    params = {**_GUI_PARAMS, "peaks": {**_GUI_PARAMS["peaks"], "range_filters": [[1, 3], [5, 6]]}}
    m = ChromaMethod.from_gui_params(params, name="g", signal_type="gc")
    assert [(w.id, w.start, w.end) for w in m.peak_windows] == [("w1", 1, 3), ("w2", 5, 6)]
    assert m.version == "2"


def test_to_processor_params_carries_windows():
    m = ChromaMethod.model_validate({
        "name": "m", "signal_type": "gc",
        "peak_windows": [_win("a", 1, 4)],
        "baseline": {"break_points": [
            {"time": 2, "overrides": {"method": "snip"}}]},
    })
    p = m.to_processor_params()
    assert p["outside_windows"] == "ignore"
    assert p["peak_windows"][0]["id"] == "a"
    bp = p["baseline"]["break_points"][0]
    assert bp["overrides"]["method"] == "snip" and bp["anchor"] is None


def test_unknown_event_type_raises():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate({
            "name": "m", "signal_type": "gc",
            "peak_windows": [_win("a", 1, 4, events=[{"type": "nope"}])],
        })


def test_peaks_model_instance_migrates_via_constructor():
    from logic.method import PeakParams
    m = ChromaMethod(name="r", signal_type="gc",
                     peaks=PeakParams(range_filters=[[1.25, 2.5]]))
    assert [(w.start, w.end) for w in m.peak_windows] == [(1.25, 2.5)]
    assert m.peaks.range_filters == [[1.25, 2.5]]


def test_peaks_model_instance_migrates_via_model_validate():
    from logic.method import PeakParams
    m = ChromaMethod.model_validate({
        "name": "r", "signal_type": "gc",
        "peaks": PeakParams(range_filters=[[1.25, 2.5]])})
    assert [(w.start, w.end) for w in m.peak_windows] == [(1.25, 2.5)]
    assert m.peaks.range_filters == [[1.25, 2.5]]


def test_migration_keeps_explicit_version():
    m = ChromaMethod.model_validate({"name": "m", "signal_type": "gc", "version": "9"})
    assert m.version == "9"
    m1 = ChromaMethod.model_validate({"name": "m", "signal_type": "gc", "version": "1"})
    assert m1.version == "2"


# --- final-review fixes -----------------------------------------------------

def test_v1_duplicate_breaks_dedupe_with_warning(caplog):
    d = _v1([], baseline={"break_points": [{"time": 0.0}, {"time": 0.0}, {"time": 3.0}]})
    with caplog.at_level(logging.WARNING):
        m = ChromaMethod.model_validate(d)
    assert [b.time for b in m.baseline.break_points] == [0.0, 3.0]
    assert any("duplicate break" in r.getMessage().lower() for r in caplog.records)


def test_v1_inverted_and_zero_width_filters_dropped_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        m = ChromaMethod.model_validate(_v1([[5, 2], [3, 3], [6, 8]]))
    assert [(w.start, w.end) for w in m.peak_windows] == [(6, 8)]
    assert sum("dropped" in r.getMessage().lower() for r in caplog.records) == 2


def test_v2_inverted_window_still_raises():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate(
            {"name": "m", "version": "2", "signal_type": "gc",
             "peak_windows": [_win("a", 5, 2)]})


def test_unknown_window_key_raises():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate(
            {"name": "m", "version": "2", "signal_type": "gc",
             "peak_windows": [_win("a", 1, 2, rider={"x": 1})]})


def test_unknown_override_key_in_window_section_raises():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate(
            {"name": "m", "version": "2", "signal_type": "gc",
             "peak_windows": [_win("a", 1, 2, peaks={"min_prominance": 1})]})


def test_unknown_break_override_key_raises():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate(
            {"name": "m", "version": "2", "signal_type": "gc",
             "baseline": {"break_points": [{"time": 3.0, "overrides": {"bogus": 1}}]}})


def test_noop_min_height_and_asymmetry_round_trip():
    m = ChromaMethod.model_validate(
        {"name": "m", "version": "2", "signal_type": "gc",
         "peak_windows": [_win("a", 1, 2, peaks={"min_height": 0.5})],
         "baseline": {"break_points": [{"time": 3.0, "overrides": {"asymmetry": 0.02}}]}})
    assert m.peak_windows[0].peaks.min_height == 0.5
    assert m.baseline.break_points[0].overrides.asymmetry == 0.02
    m2 = ChromaMethod.model_validate(m.model_dump(by_alias=True))
    assert m2.peak_windows[0].peaks.min_height == 0.5


def test_window_baseline_key_raises():
    with pytest.raises(ValueError):
        ChromaMethod.model_validate(
            {"name": "m", "version": "2", "signal_type": "gc",
             "peak_windows": [_win("a", 1, 2, baseline={"method": "asls"})]})


def test_migrated_window_label_is_empty():
    m = ChromaMethod.model_validate(_v1([[2, 5]]))
    assert m.peak_windows[0].label == ""
