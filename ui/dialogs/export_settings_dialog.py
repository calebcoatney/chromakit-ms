"""
Export Settings Dialog for controlling automatic export behavior.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QPushButton, QLabel, QComboBox
)
from PySide6.QtCore import Qt, QSettings


class ExportSettingsDialog(QDialog):
    """Dialog for configuring automatic export settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Settings")
        self.resize(400, 300)
        
        # Access export manager through parent if available
        self.export_manager = getattr(parent, 'export_manager', None)
        
        self.setup_ui()
        self.load_current_settings()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Initialize settings
        self.settings = QSettings("CalebCoatney", "ChromaKit")
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        
        # Create export triggers group
        self._create_triggers_group()
        
        # Create export formats group
        self._create_formats_group()
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create restore defaults button
        self.restore_button = QPushButton("Restore Defaults")
        self.restore_button.clicked.connect(self._restore_defaults)
        button_layout.addWidget(self.restore_button)
        
        # Add spacer
        button_layout.addStretch()
        
        # Create OK and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.layout.addLayout(button_layout)
    
    def _create_triggers_group(self):
        """Create the export triggers group."""
        group = QGroupBox("Automatic Export Triggers")
        layout = QVBoxLayout(group)
        
        # After integration
        self.export_after_integration = QCheckBox("After Peak Integration")
        self.export_after_integration.setToolTip("Automatically export results when peak integration completes")
        layout.addWidget(self.export_after_integration)
        
        # After MS search
        self.export_after_ms_search = QCheckBox("After MS Library Search")
        self.export_after_ms_search.setToolTip("Automatically export results when MS library search completes")
        layout.addWidget(self.export_after_ms_search)
        
        # After manual assignment
        self.export_after_assignment = QCheckBox("After Manual Peak Assignment")
        self.export_after_assignment.setToolTip("Automatically export results when peak assignments are manually changed")
        layout.addWidget(self.export_after_assignment)
        
        # During batch processing
        self.export_during_batch = QCheckBox("During Batch Processing")
        self.export_during_batch.setToolTip("Automatically export results during batch/automation workflows")
        layout.addWidget(self.export_during_batch)
        
        self.layout.addWidget(group)
    
    def _create_formats_group(self):
        """Create the export formats group."""
        group = QGroupBox("Export Formats")
        layout = QVBoxLayout(group)
        
        # JSON export
        self.export_json = QCheckBox("JSON Format")
        self.export_json.setToolTip("Export comprehensive results with full metadata to JSON files")
        layout.addWidget(self.export_json)
        
        # CSV export
        self.export_csv = QCheckBox("CSV Format (RESULTS.CSV)")
        self.export_csv.setToolTip("Export results to CSV format compatible with GCMS software")
        layout.addWidget(self.export_csv)
        
        # JSON filename format
        json_layout = QHBoxLayout()
        json_layout.addWidget(QLabel("JSON Filename:"))
        self.json_filename_format = QComboBox()
        self.json_filename_format.addItems([
            "{notebook} - {detector}.json",
            "{sample_id}_results.json",
            "integration_results.json"
        ])
        self.json_filename_format.setToolTip("Choose the naming format for JSON export files")
        json_layout.addWidget(self.json_filename_format)
        layout.addLayout(json_layout)
        
        # CSV filename format
        csv_layout = QHBoxLayout()
        csv_layout.addWidget(QLabel("CSV Filename:"))
        self.csv_filename_format = QComboBox()
        self.csv_filename_format.addItems([
            "RESULTS.CSV",
            "{notebook}_results.csv",
            "{sample_id}_results.csv"
        ])
        self.csv_filename_format.setToolTip("Choose the naming format for CSV export files")
        csv_layout.addWidget(self.csv_filename_format)
        layout.addLayout(csv_layout)
        
        self.layout.addWidget(group)
    
    def load_current_settings(self):
        """Load current settings into the UI elements."""
        # Export triggers (defaults to current behavior)
        self.export_after_integration.setChecked(
            self.settings.value("export/after_integration", True, type=bool)
        )
        self.export_after_ms_search.setChecked(
            self.settings.value("export/after_ms_search", True, type=bool)
        )
        self.export_after_assignment.setChecked(
            self.settings.value("export/after_assignment", True, type=bool)
        )
        self.export_during_batch.setChecked(
            self.settings.value("export/during_batch", True, type=bool)
        )
        
        # Export formats (defaults to current behavior)
        self.export_json.setChecked(
            self.settings.value("export/json_enabled", True, type=bool)
        )
        self.export_csv.setChecked(
            self.settings.value("export/csv_enabled", True, type=bool)
        )
        
        # Filename formats
        json_format = self.settings.value("export/json_filename_format", "{notebook} - {detector}.json")
        json_index = self.json_filename_format.findText(json_format)
        if json_index >= 0:
            self.json_filename_format.setCurrentIndex(json_index)
        
        csv_format = self.settings.value("export/csv_filename_format", "RESULTS.CSV")
        csv_index = self.csv_filename_format.findText(csv_format)
        if csv_index >= 0:
            self.csv_filename_format.setCurrentIndex(csv_index)
    
    def _save_settings(self):
        """Save settings to QSettings."""
        # Export triggers
        self.settings.setValue("export/after_integration", self.export_after_integration.isChecked())
        self.settings.setValue("export/after_ms_search", self.export_after_ms_search.isChecked())
        self.settings.setValue("export/after_assignment", self.export_after_assignment.isChecked())
        self.settings.setValue("export/during_batch", self.export_during_batch.isChecked())
        
        # Export formats
        self.settings.setValue("export/json_enabled", self.export_json.isChecked())
        self.settings.setValue("export/csv_enabled", self.export_csv.isChecked())
        
        # Filename formats
        self.settings.setValue("export/json_filename_format", self.json_filename_format.currentText())
        self.settings.setValue("export/csv_filename_format", self.csv_filename_format.currentText())
    
    def _restore_defaults(self):
        """Restore default settings."""
        # Export triggers - match current behavior
        self.export_after_integration.setChecked(True)   # JSON: YES, CSV: NO (default)
        self.export_after_ms_search.setChecked(True)     # JSON: YES, CSV: YES (current)
        self.export_after_assignment.setChecked(True)    # JSON: YES, CSV: NO (default)
        self.export_during_batch.setChecked(True)        # JSON: YES, CSV: YES (current)
        
        # Export formats - JSON always, CSV selectively
        self.export_json.setChecked(True)
        self.export_csv.setChecked(True)
        
        # Filename formats (use current defaults)
        self.json_filename_format.setCurrentText("{notebook} - {detector}.json")
        self.csv_filename_format.setCurrentText("RESULTS.CSV")
    
    def accept(self):
        """Save settings and close dialog."""
        self._save_settings()
        super().accept()
    
    def get_settings(self):
        """Get current settings as a dictionary."""
        return {
            'triggers': {
                'after_integration': self.export_after_integration.isChecked(),
                'after_ms_search': self.export_after_ms_search.isChecked(),
                'after_assignment': self.export_after_assignment.isChecked(),
                'during_batch': self.export_during_batch.isChecked()
            },
            'formats': {
                'json_enabled': self.export_json.isChecked(),
                'csv_enabled': self.export_csv.isChecked()
            },
            'filenames': {
                'json_format': self.json_filename_format.currentText(),
                'csv_format': self.csv_filename_format.currentText()
            }
        }
