from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QDialogButtonBox
)

class DetectorSelectionDialog(QDialog):
    """Dialog for selecting which detector channel to use."""
    
    def __init__(self, parent=None, detectors=None, current_detector=None):
        super().__init__(parent)
        self.setWindowTitle("Select Detector Channel")
        self.resize(300, 150)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add info label
        info_label = QLabel("Select which detector channel to display:")
        layout.addWidget(info_label)
        
        # Create detector dropdown
        self.detector_combo = QComboBox()
        if detectors:
            self.detector_combo.addItems(detectors)
            if current_detector and current_detector in detectors:
                index = detectors.index(current_detector)
                self.detector_combo.setCurrentIndex(index)
        layout.addWidget(self.detector_combo)
        
        # Add note about reloading
        note_label = QLabel("Note: Changing the detector will reload the current data.")
        note_label.setStyleSheet("color: #666;")
        layout.addWidget(note_label)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_selected_detector(self):
        """Return the selected detector."""
        return self.detector_combo.currentText()