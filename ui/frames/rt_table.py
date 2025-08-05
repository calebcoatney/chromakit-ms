from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QCheckBox, QMessageBox, QGroupBox, QFormLayout,
    QHeaderView, QSpinBox, QDoubleSpinBox, QFrame
)
from PySide6.QtCore import Qt, Signal
import pandas as pd
import numpy as np
import os


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
        
        # Set headers
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Compound", "Start RT", "End RT"])
        
        # Configure table
        header = self.table_widget.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Compound column stretches
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Start RT
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # End RT
        
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
            "When enabled, RT assignments will override existing MS library assignments.\\n"
            "When disabled, RT assignments only apply to unidentified peaks."
        )
        self.high_priority_checkbox.toggled.connect(self._on_settings_changed)
        settings_layout.addRow(self.high_priority_checkbox)
        
        # Window expansion controls (for future use)
        self.window_expansion_spin = QDoubleSpinBox()
        self.window_expansion_spin.setRange(0.0, 2.0)
        self.window_expansion_spin.setValue(0.0)
        self.window_expansion_spin.setSingleStep(0.1)
        self.window_expansion_spin.setSuffix(" min")
        self.window_expansion_spin.setToolTip("Additional time window to expand RT matching (experimental)")
        self.window_expansion_spin.valueChanged.connect(self._on_settings_changed)
        settings_layout.addRow("Window Expansion:", self.window_expansion_spin)
        
        # Status label
        self.status_label = QLabel("RT matching disabled")
        self.status_label.setStyleSheet("color: #666; font-size: 10px;")
        settings_layout.addRow(self.status_label)
        
        self.layout.addWidget(settings_group)
        
        # Initially disable settings until RT table is loaded
        self._set_settings_enabled(False)
    
    def _set_settings_enabled(self, enabled):
        """Enable or disable the settings controls."""
        self.enable_checkbox.setEnabled(enabled)
        self.high_priority_checkbox.setEnabled(enabled and self.enable_checkbox.isChecked())
        self.window_expansion_spin.setEnabled(enabled and self.enable_checkbox.isChecked())
    
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
            
            # Validate required columns
            required_columns = ['Compound', 'Start', 'End']
            if not all(col in df.columns for col in required_columns):
                QMessageBox.warning(
                    self, "Invalid Format", 
                    f"CSV file must contain columns: {', '.join(required_columns)}\\n"
                    f"Found columns: {', '.join(df.columns)}"
                )
                return
            
            # Validate data types
            try:
                df['Start'] = pd.to_numeric(df['Start'])
                df['End'] = pd.to_numeric(df['End'])
            except ValueError as e:
                QMessageBox.warning(
                    self, "Invalid Data", 
                    f"Start and End columns must contain numeric values.\\nError: {str(e)}"
                )
                return
            
            # Validate RT windows
            invalid_windows = df[df['Start'] >= df['End']]
            if not invalid_windows.empty:
                QMessageBox.warning(
                    self, "Invalid RT Windows", 
                    f"Found {len(invalid_windows)} compounds where Start RT >= End RT.\\n"
                    "Please fix these entries in the CSV file."
                )
                return
            
            # Store the data
            self.rt_table_data = df
            self.rt_table_file = file_path
            
            # Update UI
            self._populate_table()
            self._update_file_info()
            self._set_settings_enabled(True)
            self._on_settings_changed()
            
            self.clear_button.setEnabled(True)
            
            QMessageBox.information(
                self, "RT Table Loaded", 
                f"Successfully loaded {len(df)} compounds from RT table."
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading RT Table", 
                f"Failed to load RT table:\\n{str(e)}"
            )
    
    def _clear_rt_table(self):
        """Clear the loaded RT table."""
        self.rt_table_data = None
        self.rt_table_file = None
        
        # Clear UI
        self.table_widget.setRowCount(0)
        self.file_info_label.setText("No RT table loaded")
        self.status_label.setText("RT matching disabled")
        
        # Disable controls
        self._set_settings_enabled(False)
        self.enable_checkbox.setChecked(False)
        self.clear_button.setEnabled(False)
        
        # Emit settings change
        self._on_settings_changed()
    
    def _populate_table(self):
        """Populate the table widget with RT data."""
        if self.rt_table_data is None:
            return
        
        df = self.rt_table_data
        self.table_widget.setRowCount(len(df))
        
        for i, row in df.iterrows():
            # Compound name
            self.table_widget.setItem(i, 0, QTableWidgetItem(str(row['Compound'])))
            
            # Start RT
            start_item = QTableWidgetItem(f"{row['Start']:.3f}")
            start_item.setTextAlignment(Qt.AlignCenter)
            self.table_widget.setItem(i, 1, start_item)
            
            # End RT
            end_item = QTableWidgetItem(f"{row['End']:.3f}")
            end_item.setTextAlignment(Qt.AlignCenter)
            self.table_widget.setItem(i, 2, end_item)
    
    def _update_file_info(self):
        """Update the file info label."""
        if self.rt_table_file:
            filename = os.path.basename(self.rt_table_file)
            count = len(self.rt_table_data)
            self.file_info_label.setText(f"Loaded: {filename} ({count} compounds)")
        else:
            self.file_info_label.setText("No RT table loaded")
    
    def _on_settings_changed(self):
        """Handle changes to RT matching settings."""
        enabled = self.enable_checkbox.isChecked() and self.rt_table_data is not None
        
        # Update dependent controls
        self.high_priority_checkbox.setEnabled(enabled)
        self.window_expansion_spin.setEnabled(enabled)
        
        # Update status
        if enabled:
            priority = "high" if self.high_priority_checkbox.isChecked() else "low"
            count = len(self.rt_table_data)
            self.status_label.setText(f"RT matching enabled ({priority} priority, {count} compounds)")
        else:
            self.status_label.setText("RT matching disabled")
        
        # Emit settings change signal
        settings = {
            'enabled': enabled,
            'high_priority': self.high_priority_checkbox.isChecked(),
            'window_expansion': self.window_expansion_spin.value(),
            'rt_table': self.rt_table_data,
            'file_path': self.rt_table_file
        }
        
        self.rt_table_changed.emit(settings)
    
    def lookup_compound_by_rt(self, retention_time):
        """Look up compound name by retention time using strict window matching."""
        if self.rt_table_data is None or not self.enable_checkbox.isChecked():
            return None
        
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
    
    def get_rt_window(self, compound_name):
        """Get the RT window for a specific compound."""
        if self.rt_table_data is None:
            return None
        
        matches = self.rt_table_data[self.rt_table_data['Compound'] == compound_name]
        if len(matches) == 0:
            return None
        
        match = matches.iloc[0]
        return (match['Start'], match['End'])
    
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
            'window_expansion': self.window_expansion_spin.value(),
            'rt_table': self.rt_table_data,
            'file_path': self.rt_table_file
        }
