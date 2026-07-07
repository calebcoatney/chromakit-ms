"""Reusable editable table widget over a typed column spec.

Pure ui/ widget (no logic/ or api/ imports beyond pandas). Used by the RT and
RF table frames. Emits table_edited on cell commit / add row / delete row;
consumers pull current state via get_rows()/get_dataframe().
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QAbstractItemView,
)


@dataclass
class ColumnSpec:
    key: str                 # short/DataFrame name, e.g. "Start"
    header: str              # display header, e.g. "Start RT"
    dtype: str               # "str" | "float"
    default: object = None


class EditableTableWidget(QWidget):
    table_edited = Signal()

    def __init__(self, columns: List[ColumnSpec], parent=None):
        super().__init__(parent)
        self._columns = list(columns)
        self._populating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels([c.header for c in self._columns])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        btn_bar = QHBoxLayout()
        self.add_btn = QPushButton("Add Row")
        self.delete_btn = QPushButton("Delete Row")
        self.add_btn.clicked.connect(self._on_add_row)
        self.delete_btn.clicked.connect(self._on_delete_row)
        btn_bar.addWidget(self.add_btn)
        btn_bar.addWidget(self.delete_btn)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)

    # population (guarded, no emit)
    def set_rows(self, rows: List[dict]) -> None:
        self._populating = True
        try:
            self.table.setRowCount(0)
            for row in rows:
                self._append_row_items(row)
        finally:
            self._populating = False

    def set_dataframe(self, df: "pd.DataFrame") -> None:
        rows = [
            {c.key: r.get(c.key, c.default) for c in self._columns}
            for r in df.to_dict("records")
        ]
        self.set_rows(rows)

    # read
    def get_rows(self) -> List[dict]:
        rows = []
        for r in range(self.table.rowCount()):
            row = {}
            for c_idx, c in enumerate(self._columns):
                item = self.table.item(r, c_idx)
                text = item.text() if item is not None else ""
                row[c.key] = self._coerce(text, c)
            rows.append(row)
        return rows

    def get_dataframe(self) -> "pd.DataFrame":
        rows = self.get_rows()
        if not rows:
            return pd.DataFrame({c.key: [] for c in self._columns})
        return pd.DataFrame(rows, columns=[c.key for c in self._columns])

    def set_column_header(self, key: str, text: str) -> None:
        """Retitle the visible header of the column identified by ColumnSpec.key."""
        for idx, col in enumerate(self._columns):
            if col.key == key:
                self.table.setHorizontalHeaderItem(idx, QTableWidgetItem(text))
                return

    # internals
    def _coerce(self, text: str, col: ColumnSpec):
        if col.dtype == "float":
            try:
                return float(text)
            except (ValueError, TypeError):
                return col.default if col.default is not None else 0.0
        return text

    def _append_row_items(self, row: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c_idx, c in enumerate(self._columns):
            val = row.get(c.key, c.default)
            item = QTableWidgetItem("" if val is None else str(val))
            self.table.setItem(r, c_idx, item)

    def _on_add_row(self) -> None:
        default_row = {c.key: c.default for c in self._columns}
        self._populating = True
        try:
            self._append_row_items(default_row)
        finally:
            self._populating = False
        self.table_edited.emit()

    def _on_delete_row(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            self.table.removeRow(r)
        self.table_edited.emit()

    def _on_item_changed(self, item) -> None:
        if self._populating:
            return
        self.table_edited.emit()
