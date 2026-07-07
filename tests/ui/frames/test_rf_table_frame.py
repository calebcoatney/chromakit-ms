import pytest
pytest.importorskip('pytestqt')

from logic.method import ChromaMethod, RFTableEntry
from ui.frames.rf_table import RFTableFrame


def _make(qtbot):
    frame = RFTableFrame()
    qtbot.addWidget(frame)
    return frame


def _method_with_rf():
    return ChromaMethod(
        name="M", signal_type="gc",
        rf_table=[
            RFTableEntry(compound="Hydrogen", response_factor=402304.0),
            RFTableEntry(compound="Carbon monoxide", response_factor=209181.0),
        ],
    )


def test_apply_method_populates_table(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rf())
    rows = frame.table.get_rows()
    assert rows == [
        {"Compound": "Hydrogen", "response_factor": 402304.0},
        {"Compound": "Carbon monoxide", "response_factor": 209181.0},
    ]


def test_apply_method_does_not_emit(qtbot):
    frame = _make(qtbot)
    fired = []
    frame.rf_table_changed.connect(lambda: fired.append(True))
    frame.apply_method(_method_with_rf())
    assert fired == []


def test_add_entry_emits_change(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rf())
    fired = []
    frame.rf_table_changed.connect(lambda: fired.append(True))
    frame.add_entry("Methane", 1000.0)
    assert fired == [True]


def test_direct_cell_edit_emits_change(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rf())
    fired = []
    frame.rf_table_changed.connect(lambda: fired.append(True))
    # A user editing a cell in the embedded EditableTableWidget triggers rf_table_changed
    # because RFTableFrame connects table.table_edited -> rf_table_changed.
    frame.table.table.item(0, 0).setText("Deuterium")
    assert fired == [True]


def test_get_rf_entries_returns_models(qtbot):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rf())
    entries = frame.get_rf_entries()
    assert all(isinstance(e, RFTableEntry) for e in entries)
    assert entries[0].compound == "Hydrogen"
    assert entries[0].response_factor == 402304.0


def test_export_csv_writes_rows(qtbot, tmp_path, monkeypatch):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rf())
    out = tmp_path / "rf.csv"
    monkeypatch.setattr(
        "ui.frames.rf_table.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), "CSV Files (*.csv)"),
    )
    frame.export_btn.click()
    text = out.read_text()
    assert "Hydrogen" in text and "402304" in text
    assert "Carbon monoxide" in text


def test_import_csv_replaces_and_accepts_spellings(qtbot, tmp_path, monkeypatch):
    frame = _make(qtbot)
    frame.apply_method(_method_with_rf())  # start with 2 rows
    src = tmp_path / "in.csv"
    src.write_text("compound,RF\nMethane,1000\nEthane,2000\n")
    monkeypatch.setattr(
        "ui.frames.rf_table.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(src), "CSV Files (*.csv)"),
    )
    fired = []
    frame.rf_table_changed.connect(lambda: fired.append(True))
    frame.import_btn.click()
    entries = frame.get_rf_entries()
    assert [e.compound for e in entries] == ["Methane", "Ethane"]  # replaced
    assert entries[0].response_factor == 1000.0
    assert fired == [True]
