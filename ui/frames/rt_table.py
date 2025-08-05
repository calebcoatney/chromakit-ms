from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QCheckBox, QMessageBox, QGroupBox, QFormLayout,
    QHeaderView, QSpinBox, QDoubleSpinBox, QFrame, QComboBox, QSlider, QDialog,
    QDialogButtonBox, QLineEdit, QListWidget
)
from PySide6.QtCore import Qt, Signal
import pandas as pd
import numpy as np
import os
import json


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
        
        # RT table data
        self.rt_table_data = None
        self.rt_table_file = None
        
        # Change tracking
        self.is_modified = False
        self.original_data = None  # Store original data for comparison
        
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
        
        self.load_button = QPushButton("Load CSV File...")
        self.load_button.clicked.connect(self._load_rt_table)
        file_controls.addWidget(self.load_button)
        
        self.clear_button = QPushButton("Clear Table")
        self.clear_button.clicked.connect(self._clear_rt_table)
        self.clear_button.setEnabled(False)
        file_controls.addWidget(self.clear_button)
        
        file_layout.addLayout(file_controls)
        
        # Save/Export controls
        save_controls = QHBoxLayout()
        
        self.save_button = QPushButton("Save...")
        self.save_button.clicked.connect(self._save_rt_table)
        self.save_button.setEnabled(False)
        save_controls.addWidget(self.save_button)
        
        self.export_button = QPushButton("Export As...")
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
        """Initialize the table widget for displaying RT data."""
        table_group = QGroupBox("RT Table Contents")
        table_layout = QVBoxLayout(table_group)
        
        self.table_widget = QTableWidget()
        self.table_widget.setMaximumHeight(200)
        self.table_widget.setAlternatingRowColors(True)
        
        # Set headers - now includes Apex RT column
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Compound", "Start RT", "Apex RT", "End RT"])
        
        # Configure table
        header = self.table_widget.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Compound column stretches
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Start RT
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Apex RT
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # End RT
        
        table_layout.addWidget(self.table_widget)
        self.layout.addWidget(table_group)
    
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
    
    def _load_rt_table(self):
        """Load RT table from CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load RT Table", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check for legacy format (without Apex RT column)
            legacy_format = False
            required_columns_legacy = ['Compound', 'Start', 'End']
            required_columns_new = ['Compound', 'Start', 'Apex', 'End']
            
            # Check if this is the new format with Apex column
            if all(col in df.columns for col in required_columns_new):
                # New format with Apex RT
                pass
            elif all(col in df.columns for col in required_columns_legacy):
                # Legacy format - need to add Apex column
                legacy_format = True
                df['Apex'] = (df['Start'] + df['End']) / 2.0
                # Reorder columns to put Apex between Start and End
                df = df[['Compound', 'Start', 'Apex', 'End']]
                
                # Show user notification about automatic apex calculation
                QMessageBox.information(
                    self, "Legacy RT Table Format Detected", 
                    "This RT table uses the legacy format without an 'Apex RT' column.\n\n"
                    "Apex RT values have been automatically calculated by averaging Start and End RT values.\n"
                    "You may want to review and manually adjust these apex values for better accuracy."
                )
            else:
                QMessageBox.warning(
                    self, "Invalid Format", 
                    f"CSV file must contain columns: {', '.join(required_columns_new)}\n"
                    f"Or legacy format: {', '.join(required_columns_legacy)}\n"
                    f"Found columns: {', '.join(df.columns)}"
                )
                return
            
            # Validate data types
            try:
                df['Start'] = pd.to_numeric(df['Start'])
                df['Apex'] = pd.to_numeric(df['Apex'])
                df['End'] = pd.to_numeric(df['End'])
            except ValueError as e:
                QMessageBox.warning(
                    self, "Invalid Data", 
                    f"Start, Apex, and End columns must contain numeric values.\nError: {str(e)}"
                )
                return
            
            # Validate RT windows and apex positions
            invalid_windows = df[df['Start'] >= df['End']]
            if not invalid_windows.empty:
                QMessageBox.warning(
                    self, "Invalid RT Windows", 
                    f"Found {len(invalid_windows)} compounds where Start RT >= End RT.\n"
                    "Please fix these entries in the CSV file."
                )
                return
            
            # Validate apex positions (should be between start and end)
            invalid_apex = df[(df['Apex'] < df['Start']) | (df['Apex'] > df['End'])]
            if not invalid_apex.empty:
                QMessageBox.warning(
                    self, "Invalid Apex Positions", 
                    f"Found {len(invalid_apex)} compounds where Apex RT is outside the Start-End window.\n"
                    "Apex RT should be between Start RT and End RT."
                )
                return
            
            # Store the data
            self.rt_table_data = df
            self.rt_table_file = file_path
            
            # Initialize change tracking
            self.original_data = df.copy()
            self.is_modified = legacy_format  # Mark as modified if we auto-calculated apex values
            
            # Update UI
            self._populate_table()
            self._update_file_info()
            self._set_settings_enabled(True)
            self._on_settings_changed()
            
            # Enable buttons
            self.clear_button.setEnabled(True)
            self.save_button.setEnabled(self.is_modified)  # Enable save if modified
            self.export_button.setEnabled(True)
            
            # Show success message
            apex_note = " (with auto-calculated apex values)" if legacy_format else ""
            QMessageBox.information(
                self, "RT Table Loaded", 
                f"Successfully loaded {len(df)} compounds from RT table{apex_note}."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading RT Table", 
                f"Failed to load RT table:\n{str(e)}"
            )
    
    def _clear_rt_table(self):
        """Clear the loaded RT table."""
        self.rt_table_data = None
        self.rt_table_file = None
        
        # Reset change tracking
        self.is_modified = False
        self.original_data = None
        
        # Clear UI
        self.table_widget.setRowCount(0)
        self.file_info_label.setText("No RT table loaded")
        self.status_label.setText("RT matching disabled")
        
        # Disable controls
        self._set_settings_enabled(False)
        self.enable_checkbox.setChecked(False)
        self.clear_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)
        
        # Emit settings change
        self._on_settings_changed()
    
    def _save_rt_table(self):
        """Save the current RT table to its original file or prompt for new location."""
        if self.rt_table_data is None or len(self.rt_table_data) == 0:
            QMessageBox.warning(self, "No Data", "No RT table data to save.")
            return
        
        # If we have an original file, save there, otherwise prompt for location
        if self.rt_table_file:
            try:
                self.rt_table_data.to_csv(self.rt_table_file, index=False)
                self._mark_as_saved()
                QMessageBox.information(self, "Saved", f"RT table saved to:\n{self.rt_table_file}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save RT table:\n{str(e)}")
        else:
            # No original file, prompt for save location
            self._export_rt_table()
    
    def _export_rt_table(self):
        """Export RT table to a new file with format selection."""
        if self.rt_table_data is None or len(self.rt_table_data) == 0:
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
                    # Export as JSON
                    self._export_as_json(file_path)
                else:
                    # Export as CSV (default)
                    self._export_as_csv(file_path)
                    
                QMessageBox.information(self, "Exported", f"RT table exported to:\n{file_path}")
                
                # If this is the first save and we don't have an original file, set it as the current file
                if not self.rt_table_file:
                    self.rt_table_file = file_path
                    self._mark_as_saved()
                    
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export RT table:\n{str(e)}")
    
    def _export_as_csv(self, file_path):
        """Export RT table data as CSV."""
        self.rt_table_data.to_csv(file_path, index=False)
    
    def _export_as_json(self, file_path):
        """Export RT table data as JSON."""
        # Convert DataFrame to dictionary format
        data = {
            'format': 'ChromaKit-MS RT Table',
            'version': '1.0',
            'created': pd.Timestamp.now().isoformat(),
            'compounds': []
        }
        
        for _, row in self.rt_table_data.iterrows():
            compound = {
                'name': row['Compound'],
                'start_rt': float(row['Start']),
                'apex_rt': float(row['Apex']),
                'end_rt': float(row['End'])
            }
            data['compounds'].append(compound)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _mark_as_modified(self):
        """Mark the RT table as modified."""
        if not self.is_modified:
            self.is_modified = True
            self._update_file_info()
            self.save_button.setEnabled(True)
    
    def _mark_as_saved(self):
        """Mark the RT table as saved (no unsaved changes)."""
        self.is_modified = False
        self.original_data = self.rt_table_data.copy() if self.rt_table_data is not None else None
        self._update_file_info()
    
    def _populate_table(self):
        """Populate the table widget with RT data."""
        if self.rt_table_data is None:
            return
        
        df = self.rt_table_data
        self.table_widget.setRowCount(len(df))
        
        for i, row in df.iterrows():
            # Check if this row is new (not in original data)
            is_new_row = False
            if self.original_data is not None:
                # Check if this compound exists in original data with same RT values
                original_matches = self.original_data[self.original_data['Compound'] == row['Compound']]
                if original_matches.empty:
                    is_new_row = True
                else:
                    # Check if RT values have changed
                    original_row = original_matches.iloc[0]
                    if (abs(original_row['Start'] - row['Start']) > 0.001 or 
                        abs(original_row['Apex'] - row['Apex']) > 0.001 or 
                        abs(original_row['End'] - row['End']) > 0.001):
                        is_new_row = True
            
            # Create styling for modified items
            from PySide6.QtGui import QBrush, QColor, QFont
            modified_brush = QBrush(QColor("#CC6600"))  # Orange color
            normal_brush = QBrush(QColor("#000000"))    # Black color
            
            # Font for modified items
            modified_font = QFont()
            modified_font.setBold(True)
            normal_font = QFont()
            
            # Compound name
            compound_item = QTableWidgetItem(str(row['Compound']))
            if is_new_row:
                compound_item.setForeground(modified_brush)
                compound_item.setFont(modified_font)
            else:
                compound_item.setForeground(normal_brush)
                compound_item.setFont(normal_font)
            self.table_widget.setItem(i, 0, compound_item)
            
            # Start RT
            start_item = QTableWidgetItem(f"{row['Start']:.3f}")
            start_item.setTextAlignment(Qt.AlignCenter)
            if is_new_row:
                start_item.setForeground(modified_brush)
                start_item.setFont(modified_font)
            else:
                start_item.setForeground(normal_brush)
                start_item.setFont(normal_font)
            self.table_widget.setItem(i, 1, start_item)
            
            # Apex RT
            apex_item = QTableWidgetItem(f"{row['Apex']:.3f}")
            apex_item.setTextAlignment(Qt.AlignCenter)
            if is_new_row:
                apex_item.setForeground(modified_brush)
                apex_item.setFont(modified_font)
            else:
                apex_item.setForeground(normal_brush)
                apex_item.setFont(normal_font)
            self.table_widget.setItem(i, 2, apex_item)
            
            # End RT
            end_item = QTableWidgetItem(f"{row['End']:.3f}")
            end_item.setTextAlignment(Qt.AlignCenter)
            if is_new_row:
                end_item.setForeground(modified_brush)
                end_item.setFont(modified_font)
            else:
                end_item.setForeground(normal_brush)
                end_item.setFont(normal_font)
            self.table_widget.setItem(i, 3, end_item)
    
    def _update_file_info(self):
        """Update the file info label with current status."""
        if self.rt_table_file:
            filename = os.path.basename(self.rt_table_file)
            count = len(self.rt_table_data) if self.rt_table_data is not None else 0
            if self.is_modified:
                self.file_info_label.setText(f"File: {filename} ({count} compounds) *")
                self.file_info_label.setStyleSheet("color: #CC6600; font-size: 10px; font-style: italic;")
            else:
                self.file_info_label.setText(f"File: {filename} ({count} compounds)")
                self.file_info_label.setStyleSheet("color: #666; font-size: 10px;")
        elif self.rt_table_data is not None:
            count = len(self.rt_table_data)
            if self.is_modified:
                self.file_info_label.setText(f"Unsaved RT table ({count} compounds) *")
                self.file_info_label.setStyleSheet("color: #CC6600; font-size: 10px; font-style: italic;")
            else:
                self.file_info_label.setText(f"New RT table ({count} compounds)")
                self.file_info_label.setStyleSheet("color: #666; font-size: 10px;")
        else:
            self.file_info_label.setText("No RT table loaded")
            self.file_info_label.setStyleSheet("color: #666; font-size: 10px;")
    
    def _on_settings_changed(self):
        """Handle changes to RT matching settings."""
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
            'file_path': self.rt_table_file
        }
        
        self.rt_table_changed.emit(settings)
    
    def lookup_compound_by_rt(self, retention_time):
        """Look up compound name by retention time using the selected matching strategy."""
        if self.rt_table_data is None or not self.enable_checkbox.isChecked():
            return None
        
        mode = self.matching_mode_combo.currentIndex()
        
        if mode == 0:  # Simple Window Matching (legacy mode)
            return self._lookup_simple_window(retention_time)
        elif mode == 1:  # Closest Apex RT Matching
            return self._lookup_closest_apex(retention_time)
        elif mode == 2:  # Weighted Distance Matching
            return self._lookup_weighted_distance(retention_time)
        
        return None
    
    def _lookup_simple_window(self, retention_time):
        """Simple window matching (legacy method)."""
        # Add window expansion if specified
        expansion = self.window_expansion_spin.value()
        
        # Find compounds where retention time falls within the window
        matches = self.rt_table_data[
            (self.rt_table_data['Start'] - expansion <= retention_time) &
            (retention_time <= self.rt_table_data['End'] + expansion)
        ]
        
        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches.iloc[0]['Compound']
        else:
            # Multiple matches - choose the one with the narrowest window
            matches['window_size'] = matches['End'] - matches['Start']
            best_match = matches.loc[matches['window_size'].idxmin()]
            return best_match['Compound']
    
    def _lookup_closest_apex(self, retention_time):
        """Closest apex RT matching with tolerance."""
        tolerance = self.tolerance_spin.value()
        
        # Calculate distance from each apex RT
        distances = np.abs(self.rt_table_data['Apex'] - retention_time)
        
        # Find matches within tolerance
        within_tolerance = distances <= tolerance
        
        if not within_tolerance.any():
            return None
        
        # Get the closest match within tolerance
        closest_idx = distances[within_tolerance].idxmin()
        return self.rt_table_data.loc[closest_idx, 'Compound']
    
    def _lookup_weighted_distance(self, retention_time):
        """Weighted distance-based matching using start, apex, and end RTs."""
        if not hasattr(self, 'normalized_weights'):
            # Fallback to default weights if not initialized
            self.normalized_weights = {'start': 0.25, 'apex': 0.50, 'end': 0.25}
        
        weights = self.normalized_weights
        
        # Calculate weighted distances for each compound
        distances = []
        for _, row in self.rt_table_data.iterrows():
            start_dist = abs(row['Start'] - retention_time)
            apex_dist = abs(row['Apex'] - retention_time)
            end_dist = abs(row['End'] - retention_time)
            
            # Weighted distance calculation
            weighted_dist = (weights['start'] * start_dist + 
                           weights['apex'] * apex_dist + 
                           weights['end'] * end_dist)
            distances.append(weighted_dist)
        
        # Find the compound with minimum weighted distance
        min_dist_idx = np.argmin(distances)
        min_distance = distances[min_dist_idx]
        
        # Only return a match if the weighted distance is reasonable
        # Use a dynamic threshold based on the RT range in the table
        rt_range = self.rt_table_data['End'].max() - self.rt_table_data['Start'].min()
        max_reasonable_distance = rt_range * 0.05  # 5% of total RT range
        
        if min_distance <= max_reasonable_distance:
            return self.rt_table_data.iloc[min_dist_idx]['Compound']
        
        return None
    
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
            
            # Mark as modified
            self._mark_as_modified()
            
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
            'file_path': self.rt_table_file
        }
