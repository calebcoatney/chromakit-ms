"""Migration dialog: wrap Agilent .D folders inside .C containers."""
from __future__ import annotations
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox,
    QProgressBar, QDialogButtonBox, QMessageBox
)
from logic.c_folder import CFolder


class CFolderMigrationDialog(QDialog):
    """Shows detected .D folders and wraps them in .C containers on confirmation.

    Each row shows: source .D path, destination .C path, signal type selector.
    Originals are copied, not moved — they remain at their original paths.
    """

    SIGNAL_TYPES = ["gcms", "gc"]

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
            "Select folders to migrate. Originals will not be deleted."
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
            self.table.setCellWidget(row, 2, combo)
            self._combos.append(combo)

        layout.addWidget(self.table)
        layout.addWidget(QLabel(
            "⚠ External tools referencing the original .D path will need to be "
            "updated if you later remove the originals."
        ))

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
            signal_type = self._combos[row].currentText()
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
