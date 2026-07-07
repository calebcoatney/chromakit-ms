import pytest
pytest.importorskip('pytestqt')

import pandas as pd
from ui.widgets.editable_table import EditableTableWidget, ColumnSpec


RF_COLUMNS = [
    ColumnSpec(key="Compound", header="Compound", dtype="str"),
    ColumnSpec(key="response_factor", header="Response Factor", dtype="float", default=0.0),
]


def _make(qtbot, columns=RF_COLUMNS):
    w = EditableTableWidget(columns)
    qtbot.addWidget(w)
    return w


def test_set_and_get_rows_roundtrip(qtbot):
    w = _make(qtbot)
    w.set_rows([
        {"Compound": "Hydrogen", "response_factor": 402304.0},
        {"Compound": "Carbon monoxide", "response_factor": 209181.0},
    ])
    rows = w.get_rows()
    assert rows == [
        {"Compound": "Hydrogen", "response_factor": 402304.0},
        {"Compound": "Carbon monoxide", "response_factor": 209181.0},
    ]


def test_get_dataframe_uses_column_keys(qtbot):
    w = _make(qtbot)
    w.set_rows([{"Compound": "Methane", "response_factor": 1000.0}])
    df = w.get_dataframe()
    assert list(df.columns) == ["Compound", "response_factor"]
    assert df.iloc[0]["Compound"] == "Methane"
    assert float(df.iloc[0]["response_factor"]) == 1000.0


def test_set_dataframe_populates(qtbot):
    w = _make(qtbot)
    df = pd.DataFrame({"Compound": ["Ethane"], "response_factor": [2000.0]})
    w.set_dataframe(df)
    assert w.get_rows() == [{"Compound": "Ethane", "response_factor": 2000.0}]
