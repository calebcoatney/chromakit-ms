"""Batch tools for converting files to/from .C container format."""
from __future__ import annotations
import os
import re
import pandas as pd
from datetime import datetime
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QComboBox,
    QProgressBar, QDialogButtonBox, QMessageBox, QCheckBox,
    QGroupBox, QHeaderView, QSplitter, QWidget
)
from PySide6.QtCore import Qt
from logic.c_folder import CFolder
from logic.loaders.reactir_parser import parse_reactir_csv
from logic.loaders.avantes_parser import parse_avantes_uvvis


def _detect_signal_type(d_path: str) -> str:
    """Return 'gcms' if the .D folder contains MS data, otherwise 'gc'."""
    return "gcms" if os.path.isfile(os.path.join(d_path, "data.ms")) else "gc"


_DISPLAY_TO_KEY = {"GC-MS": "gcms", "GC": "gc", "FTIR": "ftir", "UV-Vis": "uvvis"}
_KEY_TO_DISPLAY = {v: k for k, v in _DISPLAY_TO_KEY.items()}

_TIMESTAMP_PATTERNS = [
    (r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", "%Y-%m-%d_%H-%M-%S"),
    (r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}", "%Y-%m-%d_%H.%M.%S"),
    (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
]


def _extract_timestamp(filename: str) -> str | None:
    for pattern, fmt in _TIMESTAMP_PATTERNS:
        m = re.search(pattern, filename)
        if m:
            try:
                return datetime.strptime(m.group(), fmt).isoformat()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Convert to .C
# ---------------------------------------------------------------------------

class BatchConvertDialog(QDialog):
    """Batch-wrap .D folders or CSV files into .C containers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Convert to .C")
        self.setMinimumSize(820, 600)
        self._source_paths: list[str] = []
        self._row_types: list[str] = []   # "d" | "reactir" | "avantes" | "generic"
        self._has_generic_csvs = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Source selection buttons ---
        btn_row = QHBoxLayout()
        self._add_d_btn = QPushButton("Add .D Folders…")
        self._add_d_btn.clicked.connect(self._add_d_folders)
        self._add_reactir_btn = QPushButton("Add ReactIR Files…")
        self._add_reactir_btn.clicked.connect(self._add_reactir_files)
        self._add_avantes_btn = QPushButton("Add Avantes UV-Vis File…")
        self._add_avantes_btn.clicked.connect(self._add_avantes_file)
        self._add_csv_btn = QPushButton("Add CSV Files…")
        self._add_csv_btn.clicked.connect(self._add_csv_files)
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self._add_d_btn)
        btn_row.addWidget(self._add_reactir_btn)
        btn_row.addWidget(self._add_avantes_btn)
        btn_row.addWidget(self._add_csv_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        # --- Generic CSV settings (shown only when generic CSV files are loaded) ---
        self._csv_group = QGroupBox("CSV Settings (applied to all generic CSV files)")
        csv_layout = QHBoxLayout(self._csv_group)

        csv_layout.addWidget(QLabel("Signal Type:"))
        self._csv_type_combo = QComboBox()
        self._csv_type_combo.addItems(["FTIR", "UV-Vis", "GC", "GC-MS"])
        csv_layout.addWidget(self._csv_type_combo)

        self._header_check = QCheckBox("Has header row")
        self._header_check.setChecked(False)
        csv_layout.addWidget(self._header_check)

        csv_layout.addStretch()
        self._csv_group.setVisible(False)
        layout.addWidget(self._csv_group)

        # --- Timestamp extraction ---
        self._ts_group = QGroupBox("Timestamp Extraction")
        ts_layout = QVBoxLayout(self._ts_group)
        ts_top = QHBoxLayout()
        self._ts_check = QCheckBox("Extract timestamp from filename")
        self._ts_check.toggled.connect(self._on_timestamp_toggled)
        ts_top.addWidget(self._ts_check)
        ts_top.addStretch()
        ts_layout.addLayout(ts_top)

        self._ts_hint = QLabel(
            "Detected pattern: YYYY-MM-DD_HH-MM-SS  (e.g. 2026-02-20_17-17-45)"
        )
        self._ts_hint.setStyleSheet("color: gray; font-style: italic;")
        self._ts_hint.setVisible(False)
        ts_layout.addWidget(self._ts_hint)
        layout.addWidget(self._ts_group)

        # --- Splitter: file list + preview ---
        splitter = QSplitter(Qt.Vertical)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Source", "Destination (.C)", "Type", "Timestamp"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnHidden(3, True)
        splitter.addWidget(self.table)

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_label = QLabel("CSV Preview (first 5 rows of first file)")
        self._preview_label.setVisible(False)
        preview_layout.addWidget(self._preview_label)
        self._preview_table = QTableWidget()
        self._preview_table.setMaximumHeight(160)
        self._preview_table.setVisible(False)
        preview_layout.addWidget(self._preview_table)
        splitter.addWidget(preview_widget)

        layout.addWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Convert")
        buttons.accepted.connect(self._run)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # --- Timestamp toggle ---

    def _on_timestamp_toggled(self, checked: bool):
        self.table.setColumnHidden(3, not checked)
        self._ts_hint.setVisible(checked)
        if checked:
            self._refresh_timestamps()

    def _refresh_timestamps(self):
        for row in range(self.table.rowCount()):
            if self._row_types[row] == "avantes":
                continue  # Populated from file header at add time
            name = os.path.basename(self._source_paths[row])
            ts = _extract_timestamp(name)
            item = QTableWidgetItem(ts if ts else "—")
            if not ts:
                item.setForeground(Qt.gray)
            self.table.setItem(row, 3, item)

    # --- Add helpers ---

    def _add_d_folders(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select directory containing .D folders"
        )
        if not dir_path:
            return
        d_paths = sorted(
            os.path.join(dir_path, d) for d in os.listdir(dir_path)
            if d.endswith(".D") and os.path.isdir(os.path.join(dir_path, d))
        )
        for p in d_paths:
            if p not in self._source_paths:
                detected = _detect_signal_type(p)
                self._add_row(p, "d", display_type=_KEY_TO_DISPLAY.get(detected, "GC"))

    def _add_reactir_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ReactIR CSV files", "", "CSV files (*.csv)"
        )
        for p in paths:
            if p not in self._source_paths:
                self._add_row(p, "reactir", display_type="ReactIR (FTIR)")

    def _add_avantes_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Avantes UV-Vis CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            df = pd.read_csv(path, header=None, dtype=str)
            datetime_strings = df.iloc[4, 1:].tolist()
        except Exception as e:
            QMessageBox.critical(self, "Parse Error",
                                 f"Could not parse Avantes file:\n{e}")
            return

        basename = os.path.splitext(os.path.basename(path))[0]
        for i, dt_str in enumerate(datetime_strings):
            try:
                ts = datetime.strptime(dt_str.strip(), "%d/%m/%Y %H:%M:%S").isoformat()
            except (ValueError, AttributeError):
                ts = None
            dest_name = f"{basename}_{i:03d}.C"
            self._add_row(path, "avantes", dest_name=dest_name,
                          display_type="Avantes UV-Vis", ts=ts)

    def _add_csv_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV files", "", "CSV files (*.csv)"
        )
        if not paths:
            return

        first_new = True
        for p in sorted(paths):
            if p not in self._source_paths:
                self._add_row(p, "generic", display_type="CSV")
                if first_new and not self._has_generic_csvs:
                    self._show_preview(p)
                    first_new = False

        self._has_generic_csvs = True
        self._csv_group.setVisible(True)

    def _add_row(self, path: str, row_type: str, dest_name: str = None,
                 display_type: str = None, ts: str = None):
        self._source_paths.append(path)
        self._row_types.append(row_type)
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))
        self.table.item(row, 0).setToolTip(path)

        if dest_name is None:
            base = os.path.splitext(os.path.basename(path))[0]
            dest_name = base + ".C"
        parent = os.path.dirname(os.path.abspath(path))
        self.table.setItem(row, 1, QTableWidgetItem(dest_name))
        self.table.item(row, 1).setToolTip(os.path.join(parent, dest_name))

        self.table.setItem(row, 2, QTableWidgetItem(display_type or row_type))

        if ts is None:
            ts = _extract_timestamp(os.path.basename(path))
        ts_item = QTableWidgetItem(ts if ts else "—")
        if not ts:
            ts_item.setForeground(Qt.gray)
        self.table.setItem(row, 3, ts_item)

    def _show_preview(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path, header=None, nrows=5)
            self._preview_table.setRowCount(len(df))
            self._preview_table.setColumnCount(len(df.columns))
            self._preview_table.setHorizontalHeaderLabels(
                [f"Col {i}" for i in range(len(df.columns))]
            )
            for r in range(len(df)):
                for c in range(len(df.columns)):
                    self._preview_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r, c])))
            self._preview_label.setVisible(True)
            self._preview_table.setVisible(True)
            self._preview_label.setText(f"Preview: {os.path.basename(csv_path)} (first 5 rows)")
        except Exception as e:
            self._preview_label.setText(f"Preview error: {e}")
            self._preview_label.setVisible(True)

    def _clear_all(self):
        self.table.setRowCount(0)
        self._source_paths.clear()
        self._row_types.clear()
        self._has_generic_csvs = False
        self._csv_group.setVisible(False)
        self._preview_table.setVisible(False)
        self._preview_label.setVisible(False)

    # --- Run ---

    def _run(self):
        if self.table.rowCount() == 0:
            return
        self.progress.setVisible(True)
        self.progress.setMaximum(self.table.rowCount())
        errors = []
        extract_ts = self._ts_check.isChecked()
        has_header = self._header_check.isChecked()
        csv_signal_type = _DISPLAY_TO_KEY[self._csv_type_combo.currentText()]
        avantes_processed: set[str] = set()

        for row in range(self.table.rowCount()):
            path = self._source_paths[row]
            row_type = self._row_types[row]

            try:
                if row_type == "d":
                    type_text = self.table.item(row, 2).text()
                    signal_type = _DISPLAY_TO_KEY.get(type_text, "gc")
                    kwargs: dict = {}
                    if extract_ts:
                        ts = _extract_timestamp(os.path.basename(path))
                        if ts:
                            kwargs["sample_timestamp"] = ts
                    CFolder.create(path, signal_type, **kwargs)

                elif row_type == "reactir":
                    parse_reactir_csv(path)

                elif row_type == "avantes":
                    if path not in avantes_processed:
                        avantes_processed.add(path)
                        parse_avantes_uvvis(
                            path,
                            output_dir=os.path.dirname(os.path.abspath(path))
                        )

                elif row_type == "generic":
                    kwargs = {
                        "csv_columns": {
                            "x_column": 0,
                            "y_column": 1,
                            "has_header": has_header,
                        }
                    }
                    if extract_ts:
                        ts = _extract_timestamp(os.path.basename(path))
                        if ts:
                            kwargs["sample_timestamp"] = ts
                    CFolder.create(path, csv_signal_type, **kwargs)

            except FileExistsError:
                pass
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

            self.progress.setValue(row + 1)

        if errors:
            QMessageBox.warning(
                self, "Conversion Errors",
                "Some files could not be converted:\n\n" + "\n".join(errors)
            )
        else:
            QMessageBox.information(
                self, "Done",
                f"Successfully converted {self.table.rowCount()} item(s) to .C format."
            )
        self.accept()


# ---------------------------------------------------------------------------
# Extract from .C  ← leave everything from here down UNCHANGED
# ---------------------------------------------------------------------------

class BatchExtractDialog(QDialog):
    """Batch-extract data from .C containers back to standalone files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extract from .C Containers")
        self.setMinimumSize(680, 420)
        self._c_paths: list[str] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Select .C folders to extract. The original data files will be\n"
            "moved back to the parent directory."
        ))

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Select Directory…")
        self._add_btn.clicked.connect(self._select_dir)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self._add_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([".C Folder", "Data Inside", "Signal Type"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)

        self._delete_check = QCheckBox("Delete .C wrapper after extraction (results/ will be lost)")
        layout.addWidget(self._delete_check)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Extract")
        buttons.accepted.connect(self._run)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select directory containing .C folders"
        )
        if not dir_path:
            return
        c_paths = sorted(
            os.path.join(dir_path, d) for d in os.listdir(dir_path)
            if d.endswith(".C") and os.path.isdir(os.path.join(dir_path, d))
        )
        for cp in c_paths:
            if cp in self._c_paths:
                continue
            try:
                cf = CFolder.open(cp)
                manifest = cf.get_manifest()
            except Exception:
                continue

            self._c_paths.append(cp)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(cp))

            data_dir = os.path.join(cp, "data")
            items = [i for i in os.listdir(data_dir) if not i.startswith(".")] if os.path.isdir(data_dir) else []
            self.table.setItem(row, 1, QTableWidgetItem(items[0] if items else "?"))

            sig = manifest.get("signal_type", "?")
            self.table.setItem(row, 2, QTableWidgetItem(
                _KEY_TO_DISPLAY.get(sig, sig)
            ))

    def _clear(self):
        self.table.setRowCount(0)
        self._c_paths.clear()

    def _run(self):
        if not self._c_paths:
            return
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self._c_paths))
        delete = self._delete_check.isChecked()
        errors = []

        for i, cp in enumerate(self._c_paths):
            try:
                cf = CFolder.open(cp)
                cf.extract(delete_wrapper=delete)
            except Exception as e:
                errors.append(f"{os.path.basename(cp)}: {e}")
            self.progress.setValue(i + 1)

        if errors:
            QMessageBox.warning(
                self, "Extraction Errors",
                "Some folders could not be extracted:\n\n" + "\n".join(errors)
            )
        else:
            QMessageBox.information(
                self, "Done",
                f"Successfully extracted {len(self._c_paths)} item(s)."
            )
        self.accept()
