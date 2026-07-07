"""RF (response-factor) table frame — edits the current method's rf_table.

A {compound: response_factor} table used by the 'rf_table' quant strategy.
Pure ui/ widget. Emits rf_table_changed on any edit; the app pulls
get_rf_entries() into current_method.rf_table.
"""
from __future__ import annotations
from typing import List

import csv
import io

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QMessageBox,
)

from logic.method import ChromaMethod, RFTableEntry
from logic.rf_quantitation import RF_UNITS, RF_UNIT_LABELS
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
        self._applying = False
        layout = QVBoxLayout(self)

        self.active_badge = QLabel("Active quant strategy")
        self.active_badge.setStyleSheet("color: #0a7d00; font-weight: bold;")
        self.active_badge.setVisible(False)
        layout.addWidget(self.active_badge)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("RF unit:"))
        self.unit_combo = QComboBox()
        for code in ("area_per_mol", "area_per_mol_pct", "area_per_molC_pct",
                     "area_per_wt_pct", "unspecified"):
            self.unit_combo.addItem(RF_UNIT_LABELS[code], code)   # text, data=code
        self.unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        unit_row.addWidget(self.unit_combo)
        unit_row.addStretch()
        layout.addLayout(unit_row)

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

        # Initial state: default to the method default ("unspecified") with a
        # plain header. Guard the selection so the combo's currentIndexChanged
        # cannot emit rf_table_changed during construction.
        self._applying = True
        try:
            self.select_rf_unit("unspecified")
        finally:
            self._applying = False
        self._update_rf_header()

    def apply_method(self, method: ChromaMethod) -> None:
        self._applying = True
        try:
            rows = [
                {"Compound": e.compound, "response_factor": e.response_factor}
                for e in method.rf_table
            ]
            self.table.set_rows(rows)   # guarded — no emit
            self.select_rf_unit(method.rf_unit)
            self._update_rf_header()
        finally:
            self._applying = False

    def get_rf_unit(self) -> str:
        return self.unit_combo.currentData()

    def select_rf_unit(self, code: str) -> None:
        idx = self.unit_combo.findData(code)
        if idx >= 0:
            self.unit_combo.setCurrentIndex(idx)

    def _on_unit_changed(self, _idx):
        self._update_rf_header()
        if not self._applying:
            self.rf_table_changed.emit()

    def _update_rf_header(self):
        code = self.get_rf_unit()
        if code == "unspecified":
            self.table.set_column_header("response_factor", "Response Factor")
        else:
            self.table.set_column_header(
                "response_factor", f"Response Factor ({RF_UNIT_LABELS[code]})"
            )

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import RF Table", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            raw_text = open(path, newline="", encoding="utf-8").read()
            lines = raw_text.splitlines()
            unit_code = None
            if lines and lines[0].strip().lower().startswith("# rf_unit:"):
                unit_code = lines[0].split(":", 1)[1].strip()
                lines = lines[1:]
            rows = []
            reader = csv.DictReader(io.StringIO("\n".join(lines)))
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
        if unit_code is not None and unit_code in RF_UNITS:
            self._applying = True
            try:
                self.select_rf_unit(unit_code)
                self._update_rf_header()
            finally:
                self._applying = False
        self.rf_table_changed.emit()

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export RF Table", "rf_table.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(f"# rf_unit: {self.get_rf_unit()}\n")
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
