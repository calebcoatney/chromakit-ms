import pytest
pytest.importorskip('pytestqt')

from logic.method import (
    ChromaMethod, RTTableEntry, RTMatchingParams, RTMatchingWeights,
)
from ui.frames.rt_table import RTTableFrame


def _make(qtbot):
    frame = RTTableFrame()
    qtbot.addWidget(frame)
    return frame


def _method_with_rt():
    return ChromaMethod(
        name="M", signal_type="gc",
        rt_table=[
            RTTableEntry(compound="Methane", start=1.0, apex=1.1, end=1.2),
            RTTableEntry(compound="Ethane", start=2.0, apex=2.1, end=2.2),
        ],
        rt_matching=RTMatchingParams(
            matching_mode=1, tolerance=0.15, window_expansion=0.05,
            weights=RTMatchingWeights(start=0.3, apex=0.4, end=0.3),
            high_priority=True,
        ),
    )


def test_apply_method_populates_table(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())
    df = frame.rt_table.get_dataframe()
    assert list(df.columns) == ["Compound", "Start", "Apex", "End"]
    assert list(df["Compound"]) == ["Methane", "Ethane"]


def test_apply_method_sets_matching_widgets(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())
    assert frame.matching_mode_combo.currentIndex() == 1
    assert abs(frame.tolerance_spin.value() - 0.15) < 1e-9
    assert abs(frame.window_expansion_spin.value() - 0.05) < 1e-9
    assert frame.high_priority_checkbox.isChecked() is True


def test_get_rt_entries_and_matching_params(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())
    entries = frame.get_rt_entries()
    assert [e.compound for e in entries] == ["Methane", "Ethane"]
    params = frame.get_matching_params()
    assert params.matching_mode == 1
    assert params.high_priority is True


def test_apply_method_does_not_emit_rt_table_changed(qtbot):
    frame = _make(qtbot)
    fired = []
    frame.rt_table_changed.connect(lambda *a: fired.append(True))
    frame.apply_method(_method_with_rt())
    assert fired == []  # programmatic load must be silent (no feedback-loop trigger)


def test_import_replaces_table_and_emits(qtbot, tmp_path, monkeypatch):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())  # start with Methane/Ethane
    src = tmp_path / "rt.csv"
    src.write_text("Compound,Start,Apex,End\nBenzene,4.0,4.1,4.2\n")
    monkeypatch.setattr(
        "ui.frames.rt_table.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "CSV Files (*.csv)"),
    )
    fired = []
    frame.rt_table_changed.connect(lambda *a: fired.append(True))
    frame.import_button.click()
    entries = frame.get_rt_entries()
    assert [e.compound for e in entries] == ["Benzene"]   # replaced
    assert fired


def test_legacy_3col_csv_synthesizes_apex(qtbot, tmp_path, monkeypatch):
    frame = _make(qtbot)
    src = tmp_path / "legacy.csv"
    src.write_text("Compound,Start,End\nToluene,5.0,5.4\n")
    monkeypatch.setattr(
        "ui.frames.rt_table.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "CSV Files (*.csv)"),
    )
    frame.import_button.click()
    entries = frame.get_rt_entries()
    assert entries[0].compound == "Toluene"
    assert abs(entries[0].apex - 5.2) < 1e-9   # (5.0 + 5.4) / 2


def test_import_json_skips_blank_compound_rows(qtbot, tmp_path, monkeypatch):
    import json
    frame = _make(qtbot)
    src = tmp_path / "rt.json"
    src.write_text(json.dumps({"compounds": [
        {"name": "Benzene", "start": 4.0, "apex": 4.1, "end": 4.2},
        {"start": 5.0, "apex": 5.1, "end": 5.2},  # missing name/compound -> must be skipped
    ]}))
    monkeypatch.setattr(
        "ui.frames.rt_table.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "JSON Files (*.json)"),
    )
    frame.import_button.click()
    entries = frame.get_rt_entries()
    assert [e.compound for e in entries] == ["Benzene"]  # blank-name row dropped
    assert not (frame.rt_table_data["Compound"].astype(str).str.strip() == "").any()


def test_import_csv_skips_blank_compound_rows(qtbot, tmp_path, monkeypatch):
    frame = _make(qtbot)
    src = tmp_path / "rt.csv"
    src.write_text("Compound,Start,Apex,End\nBenzene,4.0,4.1,4.2\n,5.0,5.1,5.2\n")
    monkeypatch.setattr(
        "ui.frames.rt_table.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "CSV Files (*.csv)"),
    )
    frame.import_button.click()
    entries = frame.get_rt_entries()
    assert [e.compound for e in entries] == ["Benzene"]  # blank-name row dropped
    assert not (frame.rt_table_data["Compound"].astype(str).str.strip() == "").any()


def test_apply_method_enables_controls_when_rt_table_present(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rt())   # _method_with_rt has 2 RT entries
    # After loading a method carrying an RT table, the frame must be in the same
    # enabled state as after a file import: the enable checkbox is usable (the
    # gate is no longer stuck disabled) and the ancillary file controls are on.
    assert frame.enable_checkbox.isEnabled() is True
    assert frame.export_button.isEnabled() is True
    assert frame.clear_button.isEnabled() is True


def test_apply_method_with_empty_rt_table_leaves_controls_disabled(qtbot):
    frame = _make(qtbot)
    from logic.method import ChromaMethod
    frame.apply_method(ChromaMethod(name="M", signal_type="gc"))  # no RT entries
    # No data → controls stay disabled, mirroring the boot / clear state.
    assert frame.enable_checkbox.isEnabled() is False
    assert frame.export_button.isEnabled() is False


def test_apply_method_does_not_emit_when_enabling_controls(qtbot):
    frame = _make(qtbot)
    fired = []
    frame.rt_table_changed.connect(lambda *a: fired.append(True))
    frame.apply_method(_method_with_rt())  # enabling controls must stay silent
    assert fired == []
