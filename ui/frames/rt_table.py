from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QCheckBox, QMessageBox, QGroupBox, QFormLayout,
    QHeaderView, QSpinBox, QDoubleSpinBox, QFrame, QComboBox, QSlider, QDialog,
    QDialogButtonBox, QLineEdit, QListWidget
)
from PySide6.QtCore import Qt, Signal
import pandas as pd
import json

from logic.method import ChromaMethod, RTTableEntry, RTMatchingParams, RTMatchingWeights
from ui.widgets.editable_table import EditableTableWidget, ColumnSpec


GCXGC_COLUMN_HEADERS = [
    'Peak #', '1D RT (min)', '2D RT (s)', 'Volume',
    'Compound', 'Score', 'CAS#', 'mol%', 'wt%',
]


class AddToRTTableDialog(QDialog):
    """Dialog for adding a peak to the RT table."""
    
    def __init__(self, parent=None, peak_data=None, library_compounds=None):
        super().__init__(parent)
        self.peak_data = peak_data
        self.library_compounds = library_compounds or []
        self.selected_compound = None
        self.setWindowTitle("Add Peak to RT Table")
        self.setModal(True)
        self.resize(500, 400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Peak information display
        info_group = QGroupBox("Peak Information")
        info_layout = QFormLayout(info_group)
        
        if peak_data:
            info_layout.addRow("Start RT:", QLabel(f"{peak_data['start_time']:.3f} min"))
            info_layout.addRow("Apex RT:", QLabel(f"{peak_data['retention_time']:.3f} min"))
            info_layout.addRow("End RT:", QLabel(f"{peak_data['end_time']:.3f} min"))
            if 'peak_number' in peak_data:
                info_layout.addRow("Peak Number:", QLabel(str(peak_data['peak_number'])))
        
        layout.addWidget(info_group)
        
        # Compound name input with autocomplete
        compound_group = QGroupBox("Compound Identification")
        compound_layout = QVBoxLayout(compound_group)
        
        # Search input
        input_layout = QFormLayout()
        self.compound_name_edit = QLineEdit()
        if self.library_compounds:
            self.compound_name_edit.setPlaceholderText("Enter compound name or start typing to search...")
        else:
            self.compound_name_edit.setPlaceholderText("Enter compound name...")
        input_layout.addRow("Compound Name:", self.compound_name_edit)
        compound_layout.addLayout(input_layout)
        
        # Results list for autocomplete (initially hidden)
        if self.library_compounds:
            self.results_list = QListWidget()
            self.results_list.itemClicked.connect(self.on_item_selected)
            self.results_list.setMaximumHeight(120)
            self.results_list.hide()  # Initially hidden
            compound_layout.addWidget(self.results_list)
            
            # Set up autocomplete functionality
            self._setup_autocomplete()
        
        layout.addWidget(compound_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Set focus to compound name input
        self.compound_name_edit.setFocus()
    
    def _setup_autocomplete(self):
        """Set up autocomplete functionality."""
        from PySide6.QtCore import QTimer
        
        # Create a timer for delayed filtering
        self.filter_timer = QTimer()
        self.filter_timer.setSingleShot(True)
        self.filter_timer.timeout.connect(self.filter_compounds)
        
        # Connect text changes to delayed filtering
        self.compound_name_edit.textChanged.connect(self.on_text_changed)
    
    def on_text_changed(self):
        """Handle text changes with delayed filtering."""
        if not hasattr(self, 'filter_timer'):
            return
            
        self.filter_timer.stop()
        if len(self.compound_name_edit.text()) >= 3:  # Start filtering after 3 characters
            self.filter_timer.start(300)  # 300ms delay
        else:
            if hasattr(self, 'results_list'):
                self.results_list.clear()
                self.results_list.hide()
    
    def filter_compounds(self):
        """Filter compounds based on search text."""
        if not self.library_compounds or not hasattr(self, 'results_list'):
            return
            
        text = self.compound_name_edit.text().lower().strip()
        if len(text) < 3:
            self.results_list.clear()
            self.results_list.hide()
            return
            
        # Filter compounds that contain the search text
        matching_compounds = [comp for comp in self.library_compounds 
                             if text in comp.lower()]
        
        # Update results list
        self.results_list.clear()
        if matching_compounds:
            self.results_list.addItems(matching_compounds[:25])  # Limit to 25 matches
            self.results_list.show()
            
            # Auto-select if exact match
            if text in [comp.lower() for comp in self.library_compounds]:
                self.selected_compound = next(comp for comp in self.library_compounds if comp.lower() == text)
        else:
            self.results_list.hide()
    
    def on_item_selected(self, item):
        """Handle selection from the results list."""
        if hasattr(self, 'results_list'):
            selected_text = item.text()
            self.compound_name_edit.setText(selected_text)
            self.selected_compound = selected_text
            self.results_list.hide()
    
    def get_compound_name(self):
        """Get the entered compound name."""
        return self.compound_name_edit.text().strip()


class RTTableFrame(QWidget):
    """Frame for loading and managing retention time tables for compound identification."""
    
    # Signal emitted when RT table settings change
    rt_table_changed = Signal(dict)  # Emits RT table data and settings
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(350)
        
        # Guard against re-entrant settings emits while apply_method syncs widgets
        self._applying = False
        
        # RT table data
        self.rt_table_data = None
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        
        # Initialize UI components
        self._init_file_controls()
        self._init_table_widget()
        self._init_settings()
        
        # Add stretch at the end
        self.layout.addStretch()
    
    def _init_file_controls(self):
        """Initialize file loading controls."""
        file_group = QGroupBox("RT Table File")
        file_layout = QVBoxLayout(file_group)
        
        # File selection controls
        file_controls = QHBoxLayout()
        
        self.import_button = QPushButton("Import RT Table\u2026")
        self.import_button.clicked.connect(self._import_rt_table)
        file_controls.addWidget(self.import_button)
        
        self.clear_button = QPushButton("Clear Table")
        self.clear_button.clicked.connect(self._clear_rt_table)
        self.clear_button.setEnabled(False)
        file_controls.addWidget(self.clear_button)
        
        file_layout.addLayout(file_controls)
        
        # Export controls
        save_controls = QHBoxLayout()
        
        self.export_button = QPushButton("Export RT Table\u2026")
        self.export_button.clicked.connect(self._export_rt_table)
        self.export_button.setEnabled(False)
        save_controls.addWidget(self.export_button)
        
        file_layout.addLayout(save_controls)
        
        # File info label
        self.file_info_label = QLabel("No RT table loaded")
        self.file_info_label.setStyleSheet("color: #666; font-size: 10px;")
        file_layout.addWidget(self.file_info_label)
        
        self.layout.addWidget(file_group)
    
    def _init_table_widget(self):
        """Initialize the table widgets for displaying RT data.

        Two widgets share this group box:
          * ``self.rt_table`` — an editable RT grid (Compound/Start/Apex/End)
            that mirrors ``self.rt_table_data`` and is the normal RT view.
          * ``self.table_widget`` — a plain ``QTableWidget`` used ONLY for the
            transient 9-column GCxGC results view (populate_gcxgc). It is hidden
            in the normal RT workflow so the two views never corrupt each other.
        """
        table_group = QGroupBox("RT Table Contents")
        table_layout = QVBoxLayout(table_group)

        # Editable RT grid — the normal RT view; kept consistent with rt_table_data.
        RT_COLUMNS = [
            ColumnSpec(key="Compound", header="Compound", dtype="str", default=""),
            ColumnSpec(key="Start", header="Start RT", dtype="float", default=0.0),
            ColumnSpec(key="Apex", header="Apex RT", dtype="float", default=0.0),
            ColumnSpec(key="End", header="End RT", dtype="float", default=0.0),
        ]
        self.rt_table = EditableTableWidget(RT_COLUMNS)
        self.rt_table.setMaximumHeight(240)
        self.rt_table.table_edited.connect(self._on_table_edited)

        # Configure the inner table's header sizing (Compound stretches).
        rt_header = self.rt_table.table.horizontalHeader()
        rt_header.setStretchLastSection(False)
        rt_header.setSectionResizeMode(0, QHeaderView.Stretch)              # Compound
        rt_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)     # Start RT
        rt_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)     # Apex RT
        rt_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)     # End RT

        table_layout.addWidget(self.rt_table)

        # Transient GCxGC results view — dedicated QTableWidget, hidden by default.
        self.table_widget = QTableWidget()
        self.table_widget.setMaximumHeight(200)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setColumnCount(len(GCXGC_COLUMN_HEADERS))
        self.table_widget.setHorizontalHeaderLabels(GCXGC_COLUMN_HEADERS)
        self.table_widget.hide()

        table_layout.addWidget(self.table_widget)
        self.layout.addWidget(table_group)
    
    def set_column_labels(self, position_label: str) -> None:
        """Update table headers to match the active signal profile (e.g., 'Wavenumber')."""
        labels = ["Compound", f"Start {position_label}", f"Apex {position_label}", f"End {position_label}"]
        self.rt_table.table.setHorizontalHeaderLabels(labels)
    
    def populate_gcxgc(self, peaks: list) -> None:
        """Display GCxGC2DPeak results in the table.

        Swaps the normal editable RT grid out for a dedicated flat peak results
        table. Does not modify rt_table_data or affect compound matching.
        """
        from logic.gcxgc_peak import GCxGC2DPeak

        # Swap views: hide the editable RT grid, show the GCxGC results widget.
        self.rt_table.hide()
        self.table_widget.show()

        self.table_widget.clearContents()
        self.table_widget.setColumnCount(len(GCXGC_COLUMN_HEADERS))
        self.table_widget.setHorizontalHeaderLabels(GCXGC_COLUMN_HEADERS)
        self.table_widget.setRowCount(len(peaks))

        for row, p in enumerate(peaks):
            if not isinstance(p, GCxGC2DPeak):
                continue
            values = [
                str(p.peak_number),
                f"{p.rt1:.4f}",
                f"{p.rt2:.4f}",
                f"{p.volume:.2f}",
                p.compound_name or '',
                f"{p.match_score:.4f}" if p.match_score is not None else '',
                p.casno or '',
                f"{p.mol_percent:.4f}" if p.mol_percent is not None else '',
                f"{p.wt_percent:.4f}" if p.wt_percent is not None else '',
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                self.table_widget.setItem(row, col, item)

        # Resize columns to content
        self.table_widget.resizeColumnsToContents()
    
    def _init_settings(self):
        """Initialize RT matching settings."""
        settings_group = QGroupBox("RT Matching Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Enable RT matching checkbox
        self.enable_checkbox = QCheckBox("Enable RT Table Matching")
        self.enable_checkbox.toggled.connect(self._on_settings_changed)
        settings_layout.addRow(self.enable_checkbox)
        
        # Priority setting
        self.high_priority_checkbox = QCheckBox("High Priority (Override MS assignments)")
        self.high_priority_checkbox.setToolTip(
            "When enabled, RT assignments will override existing MS library assignments.\n"
            "When disabled, RT assignments only apply to unidentified peaks."
        )
        self.high_priority_checkbox.toggled.connect(self._on_settings_changed)
        settings_layout.addRow(self.high_priority_checkbox)
        
        # RT Matching Mode Selection
        self.matching_mode_combo = QComboBox()
        self.matching_mode_combo.addItems([
            "Simple Window Matching",
            "Closest Apex RT Matching", 
            "Weighted Distance Matching"
        ])
        self.matching_mode_combo.setCurrentIndex(0)  # Default to legacy mode
        self.matching_mode_combo.setToolTip(
            "Select RT matching strategy:\n"
            "• Simple Window: Traditional start/end window matching\n"
            "• Closest Apex: Match to closest apex RT within tolerance\n"
            "• Weighted Distance: Use weighted distance considering all three points"
        )
        self.matching_mode_combo.currentIndexChanged.connect(self._on_matching_mode_changed)
        settings_layout.addRow("Matching Mode:", self.matching_mode_combo)
        
        # Tolerance setting (for closest apex mode)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.01, 5.0)
        self.tolerance_spin.setValue(0.1)
        self.tolerance_spin.setSingleStep(0.01)
        self.tolerance_spin.setSuffix(" min")
        self.tolerance_spin.setToolTip("Maximum allowed difference for closest apex RT matching")
        self.tolerance_spin.valueChanged.connect(self._on_settings_changed)
        settings_layout.addRow("Apex Tolerance:", self.tolerance_spin)
        
        # Weighted Distance Settings Group
        self.weight_group = QGroupBox("Weighted Distance Parameters")
        weight_layout = QFormLayout(self.weight_group)
        
        # Weight sliders with labels
        self.start_weight_slider = QSlider(Qt.Horizontal)
        self.start_weight_slider.setRange(0, 100)
        self.start_weight_slider.setValue(25)  # Default 0.25
        self.start_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.start_weight_slider.valueChanged.connect(self._on_weight_changed)
        self.start_weight_label = QLabel("0.25")
        start_weight_layout = QHBoxLayout()
        start_weight_layout.addWidget(self.start_weight_slider)
        start_weight_layout.addWidget(self.start_weight_label)
        weight_layout.addRow("Start RT Weight:", start_weight_layout)
        
        self.apex_weight_slider = QSlider(Qt.Horizontal)
        self.apex_weight_slider.setRange(0, 100)
        self.apex_weight_slider.setValue(50)  # Default 0.50
        self.apex_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.apex_weight_slider.valueChanged.connect(self._on_weight_changed)
        self.apex_weight_label = QLabel("0.50")
        apex_weight_layout = QHBoxLayout()
        apex_weight_layout.addWidget(self.apex_weight_slider)
        apex_weight_layout.addWidget(self.apex_weight_label)
        weight_layout.addRow("Apex RT Weight:", apex_weight_layout)
        
        self.end_weight_slider = QSlider(Qt.Horizontal)
        self.end_weight_slider.setRange(0, 100)
        self.end_weight_slider.setValue(25)  # Default 0.25
        self.end_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.end_weight_slider.valueChanged.connect(self._on_weight_changed)
        self.end_weight_label = QLabel("0.25")
        end_weight_layout = QHBoxLayout()
        end_weight_layout.addWidget(self.end_weight_slider)
        end_weight_layout.addWidget(self.end_weight_label)
        weight_layout.addRow("End RT Weight:", end_weight_layout)
        
        settings_layout.addRow(self.weight_group)
        
        # Window expansion controls (legacy support)
        self.window_expansion_spin = QDoubleSpinBox()
        self.window_expansion_spin.setRange(0.0, 2.0)
        self.window_expansion_spin.setValue(0.0)
        self.window_expansion_spin.setSingleStep(0.1)
        self.window_expansion_spin.setSuffix(" min")
        self.window_expansion_spin.setToolTip("Additional time window to expand RT matching (for simple window mode)")
        self.window_expansion_spin.valueChanged.connect(self._on_settings_changed)
        settings_layout.addRow("Window Expansion:", self.window_expansion_spin)
        
        # Status label
        self.status_label = QLabel("RT matching disabled")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        settings_layout.addRow(self.status_label)
        
        self.layout.addWidget(settings_group)
        
        # Initially disable settings until RT table is loaded
        self._set_settings_enabled(False)
        
        # Initialize weight display and visibility
        self._on_weight_changed()
        self._on_matching_mode_changed()
    
    def _set_settings_enabled(self, enabled):
        """Enable or disable the settings controls."""
        self.enable_checkbox.setEnabled(enabled)
        self.high_priority_checkbox.setEnabled(enabled and self.enable_checkbox.isChecked())
        self.matching_mode_combo.setEnabled(enabled and self.enable_checkbox.isChecked())
        self.tolerance_spin.setEnabled(enabled and self.enable_checkbox.isChecked())
        self.weight_group.setEnabled(enabled and self.enable_checkbox.isChecked())
        self.window_expansion_spin.setEnabled(enabled and self.enable_checkbox.isChecked())
    
    def _on_matching_mode_changed(self):
        """Handle matching mode selection changes."""
        if not hasattr(self, 'matching_mode_combo'):
            return
            
        mode = self.matching_mode_combo.currentIndex()
        
        # Show/hide relevant controls based on mode
        if mode == 0:  # Simple Window Matching
            self.tolerance_spin.setVisible(False)
            self.weight_group.setVisible(False)
            self.window_expansion_spin.setVisible(True)
        elif mode == 1:  # Closest Apex RT Matching
            self.tolerance_spin.setVisible(True)
            self.weight_group.setVisible(False)
            self.window_expansion_spin.setVisible(False)
        elif mode == 2:  # Weighted Distance Matching
            self.tolerance_spin.setVisible(False)
            self.weight_group.setVisible(True)
            self.window_expansion_spin.setVisible(False)
        
        self._on_settings_changed()
    
    def _on_weight_changed(self):
        """Handle weight slider changes and update labels."""
        if not hasattr(self, 'start_weight_slider'):
            return
            
        # Update weight labels
        start_weight = self.start_weight_slider.value() / 100.0
        apex_weight = self.apex_weight_slider.value() / 100.0
        end_weight = self.end_weight_slider.value() / 100.0
        
        self.start_weight_label.setText(f"{start_weight:.2f}")
        self.apex_weight_label.setText(f"{apex_weight:.2f}")
        self.end_weight_label.setText(f"{end_weight:.2f}")
        
        # Normalize weights to sum to 1.0
        total_weight = start_weight + apex_weight + end_weight
        if total_weight > 0:
            self.normalized_weights = {
                'start': start_weight / total_weight,
                'apex': apex_weight / total_weight,
                'end': end_weight / total_weight
            }
        else:
            # Fallback to equal weights
            self.normalized_weights = {'start': 0.33, 'apex': 0.34, 'end': 0.33}
        
        self._on_settings_changed()
    
    def _import_rt_table(self):
        """Import an RT table from CSV or JSON, replacing the current grid.

        Parses the file into a validated DataFrame (via ``_parse_csv`` /
        ``_parse_json``), pushes it into the editable RT grid, and emits
        ``rt_table_changed``. Importing REPLACES the method's current RT table;
        the app writes the emitted table back into ``current_method``.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import RT Table", "",
            "RT Table Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        try:
            if file_path.lower().endswith(".json"):
                df = self._parse_json(file_path)
            else:
                df = self._parse_csv(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Import RT Table Failed", str(e))
            return

        # Replace the editable grid (guarded — no table_edited emit).
        self.table_widget.hide()
        self.rt_table.show()
        self.rt_table.set_dataframe(df)
        self.rt_table_data = self.rt_table.get_dataframe()

        # Enable controls now that we have data.
        self._set_settings_enabled(True)
        self.clear_button.setEnabled(True)
        self.export_button.setEnabled(True)

        self._update_file_info()      # refresh "(N compounds)" label
        self._on_settings_changed()   # emits rt_table_changed

    def _parse_csv(self, file_path):
        """Parse a CSV RT table into a validated DataFrame (import adapter).

        Supports the legacy 3-column format (Compound/Start/End) by synthesizing
        Apex = (Start + End) / 2. Raises ValueError on invalid data.
        """
        df = pd.read_csv(file_path)
        legacy = list(df.columns) == ["Compound", "Start", "End"]
        if legacy:
            df["Apex"] = (df["Start"] + df["End"]) / 2.0
            df = df[["Compound", "Start", "Apex", "End"]]
        elif not all(col in df.columns for col in ["Compound", "Start", "Apex", "End"]):
            raise ValueError(
                "CSV file must contain columns: Compound, Start, Apex, End\n"
                "Or legacy format: Compound, Start, End\n"
                f"Found columns: {', '.join(df.columns)}"
            )
        self._validate_rt_data(df, legacy)
        # Drop rows with no compound name (NaN/blank) so blank identities never
        # reach rt_table_data (and thus the matcher). Mirrors _parse_json.
        df = df[df["Compound"].notna() & (df["Compound"].astype(str).str.strip() != "")]
        return df

    def _parse_json(self, file_path):
        """Parse a JSON RT table into a validated DataFrame (import adapter).

        Supports both new (name/start_rt/apex_rt/end_rt) and legacy
        (compound/start/apex/end) field names, synthesizing Apex when absent.
        Raises ValueError on invalid data.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for c in data.get("compounds", []):
            name = c.get("name", c.get("compound"))
            start = c.get("start_rt", c.get("start"))
            end = c.get("end_rt", c.get("end"))
            apex = c.get("apex_rt", c.get("apex"))
            if apex is None and start is not None and end is not None:
                apex = (start + end) / 2.0
            if name is None or str(name).strip() == "":
                continue
            rows.append({"Compound": name, "Start": start, "Apex": apex, "End": end})
        df = pd.DataFrame(rows, columns=["Compound", "Start", "Apex", "End"])
        self._validate_rt_data(df, False)
        return df

    def _validate_rt_data(self, df, legacy_format):
        """Validate RT table data in place; raise ValueError on invalid data."""
        # Validate data types
        try:
            df['Start'] = pd.to_numeric(df['Start'])
            df['Apex'] = pd.to_numeric(df['Apex'])
            df['End'] = pd.to_numeric(df['End'])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Start, Apex, and End columns must contain numeric values.\nError: {str(e)}"
            )

        # Validate RT windows and apex positions
        invalid_windows = df[df['Start'] >= df['End']]
        if not invalid_windows.empty:
            raise ValueError(
                f"Found {len(invalid_windows)} compounds where Start RT >= End RT.\n"
                "Please fix these entries in the file."
            )

        # Validate apex positions (should be between start and end)
        invalid_apex = df[(df['Apex'] < df['Start']) | (df['Apex'] > df['End'])]
        if not invalid_apex.empty:
            raise ValueError(
                f"Found {len(invalid_apex)} compounds where Apex RT is outside the Start-End window.\n"
                "Apex RT should be between Start RT and End RT."
            )

        return df

    def _clear_rt_table(self):
        """Clear the loaded RT table."""
        self.rt_table_data = None
        
        # Clear UI
        self.table_widget.setRowCount(0)
        self.table_widget.hide()
        self.rt_table.set_rows([])   # guarded — no emit
        self.rt_table.show()
        self.file_info_label.setText("No RT table loaded")
        self.status_label.setText("RT matching disabled")
        
        # Disable controls
        self._set_settings_enabled(False)
        self.enable_checkbox.setChecked(False)
        self.clear_button.setEnabled(False)
        self.export_button.setEnabled(False)
        
        # Emit settings change
        self._on_settings_changed()
    
    def _export_rt_table(self):
        """Export the current RT grid to a chosen CSV or JSON path."""
        df = self.rt_table.get_dataframe()
        if df is None or len(df) == 0:
            QMessageBox.warning(self, "No Data", "No RT table data to export.")
            return
        
        # File dialog with format options
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setDefaultSuffix("csv")
        file_dialog.setNameFilters([
            "CSV Files (*.csv)",
            "JSON Files (*.json)",
            "All Files (*)"
        ])
        
        if file_dialog.exec() == QFileDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            selected_filter = file_dialog.selectedNameFilter()
            
            try:
                if "JSON" in selected_filter or file_path.lower().endswith('.json'):
                    self._write_json(df, file_path)
                else:
                    self._write_csv(df, file_path)
                
                QMessageBox.information(
                    self, "Exported",
                    f"RT table exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export RT table:\n{str(e)}")
    
    def _write_csv(self, df, file_path):
        """Write the given RT DataFrame to CSV."""
        df.to_csv(file_path, index=False)
    
    def _write_json(self, df, file_path):
        """Write the given RT DataFrame to JSON."""
        data = {
            'format': 'ChromaKit-MS RT Table',
            'version': '1.0',
            'created': pd.Timestamp.now().isoformat(),
            'compounds': []
        }
        
        for _, row in df.iterrows():
            compound = {
                'name': row['Compound'],
                'start_rt': float(row['Start']),
                'apex_rt': float(row['Apex']),
                'end_rt': float(row['End'])
            }
            data['compounds'].append(compound)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _populate_table(self):
        """Populate the editable RT grid from ``self.rt_table_data``.

        Keeps the editable widget (``self.rt_table``) in sync with the backing
        DataFrame (``self.rt_table_data``), which remains the source of truth for
        compound lookups. Swaps back from any transient GCxGC view.
        """
        if self.rt_table_data is None:
            return

        # Ensure the normal RT view is shown (in case a GCxGC view was active).
        self.table_widget.hide()
        self.rt_table.show()

        # set_dataframe is guarded internally (no table_edited emit).
        self.rt_table.set_dataframe(self.rt_table_data)
    
    def _update_file_info(self):
        """Update the info label with the current in-memory RT table status.

        The frame no longer owns a file path or dirty flag (the method's own
        dirty flag supersedes them), so this reports only the compound count of
        the current grid.
        """
        if self.rt_table_data is not None and len(self.rt_table_data) > 0:
            count = len(self.rt_table_data)
            self.file_info_label.setText(f"RT table ({count} compounds)")
            self.file_info_label.setStyleSheet("color: #666; font-size: 10px;")
        else:
            self.file_info_label.setText("No RT table loaded")
            self.file_info_label.setStyleSheet("color: #666; font-size: 10px;")
    
    def _on_settings_changed(self):
        """Handle changes to RT matching settings."""
        if getattr(self, "_applying", False):
            return
        enabled = self.enable_checkbox.isChecked() and self.rt_table_data is not None
        
        # Update dependent controls
        self.high_priority_checkbox.setEnabled(enabled)
        self.matching_mode_combo.setEnabled(enabled)
        self.tolerance_spin.setEnabled(enabled)
        self.weight_group.setEnabled(enabled)
        self.window_expansion_spin.setEnabled(enabled)
        
        # Update status based on mode and settings
        if enabled:
            priority = "high" if self.high_priority_checkbox.isChecked() else "low"
            count = len(self.rt_table_data)
            
            mode_names = ["Simple Window", "Closest Apex", "Weighted Distance"]
            mode = mode_names[self.matching_mode_combo.currentIndex()]
            
            self.status_label.setText(f"RT matching enabled ({mode}, {priority} priority, {count} compounds)")
        else:
            self.status_label.setText("RT matching disabled")
        
        # Emit settings change signal
        settings = {
            'enabled': enabled,
            'high_priority': self.high_priority_checkbox.isChecked(),
            'matching_mode': self.matching_mode_combo.currentIndex(),
            'tolerance': self.tolerance_spin.value(),
            'weights': getattr(self, 'normalized_weights', {'start': 0.25, 'apex': 0.50, 'end': 0.25}),
            'window_expansion': self.window_expansion_spin.value(),
            'rt_table': self.rt_table_data,
        }
        
        self.rt_table_changed.emit(settings)
    
    def add_peak_to_rt_table(self, peak_data):
        """Add a peak to the RT table with user input for compound name."""
        if self.rt_table_data is None:
            # Create new RT table if none exists
            self.rt_table_data = pd.DataFrame(columns=['Compound', 'Start', 'Apex', 'End'])
        
        # Get library compounds for autocomplete if available
        library_compounds = []
        try:
            # Try to get library compounds from parent app
            parent_app = self.parent()
            while parent_app and not hasattr(parent_app, 'ms_frame'):
                parent_app = parent_app.parent()
            
            if parent_app and hasattr(parent_app, 'ms_frame') and hasattr(parent_app.ms_frame, 'library_compounds'):
                library_compounds = parent_app.ms_frame.library_compounds
        except Exception:
            # If we can't get compounds, just continue without autocomplete
            pass
        
        # Show dialog to get compound name
        dialog = AddToRTTableDialog(self, peak_data, library_compounds)
        if dialog.exec() == QDialog.Accepted:
            compound_name = dialog.get_compound_name()
            
            if not compound_name:
                QMessageBox.warning(self, "Invalid Input", "Please enter a compound name.")
                return False
            
            # Check if compound already exists
            if compound_name in self.rt_table_data['Compound'].values:
                reply = QMessageBox.question(
                    self, "Compound Exists", 
                    f"Compound '{compound_name}' already exists in the RT table.\n"
                    "Do you want to update it with the new RT values?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Update existing entry
                    idx = self.rt_table_data[self.rt_table_data['Compound'] == compound_name].index[0]
                    self.rt_table_data.loc[idx, 'Start'] = peak_data['start_time']
                    self.rt_table_data.loc[idx, 'Apex'] = peak_data['retention_time']
                    self.rt_table_data.loc[idx, 'End'] = peak_data['end_time']
                else:
                    return False
            else:
                # Add new entry
                new_row = pd.DataFrame({
                    'Compound': [compound_name],
                    'Start': [peak_data['start_time']],
                    'Apex': [peak_data['retention_time']],
                    'End': [peak_data['end_time']]
                })
                self.rt_table_data = pd.concat([self.rt_table_data, new_row], ignore_index=True)
            
            # Sort by Start RT
            self.rt_table_data = self.rt_table_data.sort_values('Start').reset_index(drop=True)
            
            # Update UI
            self._populate_table()
            self._update_file_info()
            
            # Enable settings if this was the first entry
            if len(self.rt_table_data) == 1:
                self._set_settings_enabled(True)
                self.export_button.setEnabled(True)
            
            # Emit settings change
            self._on_settings_changed()
            
            QMessageBox.information(
                self, "Peak Added", 
                f"Peak for '{compound_name}' has been added to the RT table.\n"
                f"RT: {peak_data['retention_time']:.3f} min"
            )
            
            return True
        
        return False
    
    def get_rt_window(self, compound_name):
        """Get the RT window for a specific compound."""
        if self.rt_table_data is None:
            return None
        
        matches = self.rt_table_data[self.rt_table_data['Compound'] == compound_name]
        if len(matches) == 0:
            return None
        
        match = matches.iloc[0]
        return (match['Start'], match['Apex'], match['End'])
    
    def get_all_compounds(self):
        """Get list of all compounds in the RT table."""
        if self.rt_table_data is None:
            return []
        
        return self.rt_table_data['Compound'].tolist()
    
    def is_enabled(self):
        """Check if RT matching is currently enabled."""
        return (self.enable_checkbox.isChecked() and 
                self.rt_table_data is not None)
    
    def get_settings(self):
        """Get current RT matching settings."""
        return {
            'enabled': self.is_enabled(),
            'high_priority': self.high_priority_checkbox.isChecked(),
            'matching_mode': self.matching_mode_combo.currentIndex(),
            'tolerance': self.tolerance_spin.value(),
            'weights': getattr(self, 'normalized_weights', {'start': 0.25, 'apex': 0.50, 'end': 0.25}),
            'window_expansion': self.window_expansion_spin.value(),
            'rt_table': self.rt_table_data,
        }

    # ── Method sync surface (Phase 1b) ──────────────────────────────────────────

    def apply_method(self, method: ChromaMethod) -> None:
        """Populate the RT grid and matching widgets from a ChromaMethod.

        Guarded by ``self._applying`` so the widget updates below do not emit
        settings-changed churn while syncing.
        """
        self._applying = True
        try:
            rows = [
                {"Compound": e.compound, "Start": e.start, "Apex": e.apex, "End": e.end}
                for e in method.rt_table
            ]
            self.rt_table.set_rows(rows)          # guarded — no emit
            self.rt_table_data = self.rt_table.get_dataframe()
            p = method.rt_matching
            self.matching_mode_combo.setCurrentIndex(p.matching_mode)
            self.tolerance_spin.setValue(p.tolerance)
            self.window_expansion_spin.setValue(p.window_expansion)
            self.high_priority_checkbox.setChecked(p.high_priority)
            self.normalized_weights = {
                "start": p.weights.start, "apex": p.weights.apex, "end": p.weights.end,
            }
            # Refresh control-enable state to match the file-import path: when the
            # method carries an RT table, the enable checkbox + file controls become
            # usable; with no data they stay disabled (boot/clear state). These are
            # pure setEnabled() calls (no signals), so no rt_table_changed is emitted.
            has_data = self.rt_table_data is not None and len(self.rt_table_data) > 0
            self._set_settings_enabled(has_data)
            self.clear_button.setEnabled(has_data)
            self.export_button.setEnabled(has_data)
            self._update_file_info()
        finally:
            self._applying = False

    def get_rt_entries(self):
        """Read the editable RT grid back as a list of RTTableEntry models."""
        entries = []
        for row in self.rt_table.get_rows():
            name = str(row.get("Compound", "")).strip()
            if not name:
                continue
            entries.append(RTTableEntry(
                compound=name,
                start=float(row.get("Start", 0.0)),
                apex=float(row.get("Apex", 0.0)),
                end=float(row.get("End", 0.0)),
            ))
        return entries

    def get_matching_params(self) -> RTMatchingParams:
        """Read the matching widgets back as an RTMatchingParams model."""
        w = getattr(self, "normalized_weights", {"start": 0.25, "apex": 0.50, "end": 0.25})
        return RTMatchingParams(
            matching_mode=self.matching_mode_combo.currentIndex(),
            tolerance=self.tolerance_spin.value(),
            window_expansion=self.window_expansion_spin.value(),
            weights=RTMatchingWeights(**w),
            high_priority=self.high_priority_checkbox.isChecked(),
        )

    def _on_table_edited(self):
        """Keep ``rt_table_data`` in sync when the user edits the grid."""
        self.rt_table_data = self.rt_table.get_dataframe()
        if not getattr(self, "_applying", False):
            self._on_settings_changed()
