"""RF (response-factor) table frame — edits the current method's rf_table.

A {compound: response_factor} table used by the 'rf_table' quant strategy.
Pure ui/ widget. Emits rf_table_changed on any edit; the app pulls
get_rf_entries() into current_method.rf_table.
"""
from __future__ import annotations
from typing import List

import csv

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox,
)

from logic.method import ChromaMethod, RFTableEntry
from ui.widgets.editable_table import EditableTableWidget, ColumnSpec


RF_COLUMNS = [
    ColumnSpec(key="Compound", header="Compound", dtype="str", default=""),
    ColumnSpec(key="response_factor", header="Response Factor", dtype="float", default=0.0),
]


class RFTableFrame(QWidget):
    rf_table_changed = Signal()

    _COMPOUND_KEYS = ("Compound", "compound", "name", "Name")
    _RF_KEYS = ("Response Factor", "response_factor", "RF", "rf")

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.active_badge = QLabel("Active quant strategy")
        self.active_badge.setStyleSheet("color: #0a7d00; font-weight: bold;")
        self.active_badge.setVisible(False)
        layout.addWidget(self.active_badge)

        self.table = EditableTableWidget(RF_COLUMNS)
        self.table.table_edited.connect(self.rf_table_changed.emit)
        layout.addWidget(self.table)

        file_bar = QHBoxLayout()
        self.import_btn = QPushButton("Import RF Table\u2026")
        self.export_btn = QPushButton("Export RF Table\u2026")
        file_bar.addWidget(self.import_btn)
        file_bar.addWidget(self.export_btn)
        file_bar.addStretch()
        layout.addLayout(file_bar)
        layout.addStretch()

        self.import_btn.clicked.connect(self._on_import)
        self.export_btn.clicked.connect(self._on_export)

    def apply_method(self, method: ChromaMethod) -> None:
        rows = [
            {"Compound": e.compound, "response_factor": e.response_factor}
            for e in method.rf_table
        ]
        self.table.set_rows(rows)   # guarded — no emit

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import RF Table", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            rows = []
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for raw in reader:
                    name = next((raw[k] for k in self._COMPOUND_KEYS if k in raw and raw[k]), None)
                    rf = next((raw[k] for k in self._RF_KEYS if k in raw and raw[k] not in (None, "")), None)
                    if name is None or rf is None:
                        continue
                    rows.append({"Compound": str(name).strip(), "response_factor": float(rf)})
        except Exception as e:
            QMessageBox.critical(self, "Import RF Table Failed", str(e))
            return
        self.table.set_rows(rows)   # replace
        self.rf_table_changed.emit()

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export RF Table", "rf_table.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Compound", "Response Factor"])
                for e in self.get_rf_entries():
                    writer.writerow([e.compound, e.response_factor])
        except Exception as e:
            QMessageBox.critical(self, "Export RF Table Failed", str(e))

    def add_entry(self, compound: str, response_factor: float) -> None:
        rows = self.table.get_rows()
        rows.append({"Compound": compound, "response_factor": response_factor})
        self.table.set_rows(rows)
        self.rf_table_changed.emit()

    def get_rf_entries(self) -> List[RFTableEntry]:
        entries = []
        for row in self.table.get_rows():
            name = str(row.get("Compound", "")).strip()
            if not name:
                continue
            entries.append(
                RFTableEntry(compound=name, response_factor=float(row.get("response_factor", 0.0)))
            )
        return entries

    def set_active(self, active: bool) -> None:
        self.active_badge.setVisible(active)
