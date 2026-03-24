"""Migration dialog: wrap Agilent .D folders inside .C containers."""
from __future__ import annotations
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox,
    QProgressBar, QDialogButtonBox, QMessageBox
)
from logic.c_folder import CFolder


def _detect_signal_type(d_path: str) -> str:
    """Return 'gcms' if the .D folder contains MS data, otherwise 'gc'."""
    return "gcms" if os.path.isfile(os.path.join(d_path, "data.ms")) else "gc"


class CFolderMigrationDialog(QDialog):
    """Shows detected .D folders and wraps them in .C containers on confirmation.

    Each row shows: source .D path, destination .C path, auto-detected signal type.
    The .D folder is moved (not copied) into the .C container.
    """

    SIGNAL_TYPES = ["GC-MS", "GC"]
    _SIGNAL_KEY = {"GC-MS": "gcms", "GC": "gc"}

    def __init__(self, d_paths: list[str], parent=None):
        super().__init__(parent)
        self.d_paths = d_paths
        self.setWindowTitle("Migrate .D Folders to .C Format")
        self.setMinimumWidth(720)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "ChromaKit detected Agilent .D folders without .C wrappers.\n"
            "Select the signal type for each folder, then click OK to migrate.\n"
            "The .D folders will be moved inside the new .C containers."
        ))

        self.table = QTableWidget(len(self.d_paths), 3)
        self.table.setHorizontalHeaderLabels(["Source (.D)", "Destination (.C)", "Signal Type"])
        self.table.setColumnWidth(0, 280)
        self.table.setColumnWidth(1, 280)

        self._combos: list[QComboBox] = []
        for row, d_path in enumerate(self.d_paths):
            base = os.path.splitext(os.path.basename(d_path))[0]
            c_path = os.path.join(os.path.dirname(d_path), base + ".C")
            self.table.setItem(row, 0, QTableWidgetItem(d_path))
            self.table.setItem(row, 1, QTableWidgetItem(c_path))
            combo = QComboBox()
            combo.addItems(self.SIGNAL_TYPES)
            # Auto-detect: set to GC-MS if data.ms exists, otherwise GC
            detected = _detect_signal_type(d_path)
            combo.setCurrentText("GC-MS" if detected == "gcms" else "GC")
            self.table.setCellWidget(row, 2, combo)
            self._combos.append(combo)

        layout.addWidget(self.table)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._run_migration)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _run_migration(self):
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.d_paths))
        errors = []

        for row, d_path in enumerate(self.d_paths):
            display_type = self._combos[row].currentText()
            signal_type = self._SIGNAL_KEY[display_type]
            try:
                CFolder.create(d_path, signal_type)
            except FileExistsError:
                pass  # already migrated
            except Exception as e:
                errors.append(f"{os.path.basename(d_path)}: {e}")
            self.progress.setValue(row + 1)

        if errors:
            QMessageBox.warning(
                self, "Migration Errors",
                "Some folders could not be migrated:\n\n" + "\n".join(errors)
            )
        self.accept()
