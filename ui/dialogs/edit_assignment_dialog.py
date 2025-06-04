from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QListWidget, QCompleter,
                             QCheckBox, QGroupBox, QProgressDialog, QMessageBox)
from PySide6.QtCore import Qt, Signal, QStringListModel, QTimer
import numpy as np
import os
import re
import rainbow as rb

class EditAssignmentDialog(QDialog):
    """Dialog for editing peak compound assignments."""
    
    # Add signal for cross-file application
    apply_to_files_requested = Signal(str, float, float, object)  # compound_name, RT, tolerance, spectrum
    
    def __init__(self, parent=None, peak=None, library_compounds=None, current_directory=None, spectrum=None, all_files=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Compound Assignment")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.peak = peak
        self.library_compounds = library_compounds or []
        self.selected_compound = None
        self.current_directory = current_directory
        self.spectrum = spectrum  # The spectrum of the current peak
        self.all_files = all_files or []  # List of processed files
        
        # Setup UI
        self.setup_ui()
        
        # Start with the current assignment
        if hasattr(peak, 'compound_id') and peak.compound_id:
            self.search_edit.setText(peak.compound_id)
            
        # Create a timer for delayed filtering
        self.filter_timer = QTimer()
        self.filter_timer.setSingleShot(True)
        self.filter_timer.timeout.connect(self.filter_compounds)
        
        # Connect search edit text changes to delayed filtering
        self.search_edit.textChanged.connect(self.on_text_changed)
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Peak information section
        peak_info = QLabel(f"Editing Peak {self.peak.peak_number} (RT: {self.peak.retention_time:.3f})")
        peak_info.setStyleSheet("font-weight: bold;")
        layout.addWidget(peak_info)
        
        # Current assignment
        current_assignment = QLabel(f"Current Assignment: {self.peak.compound_id if hasattr(self.peak, 'compound_id') else 'Unknown'}")
        layout.addWidget(current_assignment)
        
        # Search input
        search_layout = QHBoxLayout()
        search_label = QLabel("Compound Name:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Start typing to search (min 5 characters)")
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
        
        # Results list
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_item_selected)
        layout.addWidget(self.results_list)
        
        # Cross-file application group
        if self.current_directory and self.all_files:
            cross_file_group = QGroupBox("Apply to Other Files")
            cross_file_layout = QVBoxLayout(cross_file_group)
            
            # Add checkbox to enable cross-file application
            self.cross_file_checkbox = QCheckBox("Apply this assignment to matching peaks in other files")
            self.cross_file_checkbox.setChecked(False)
            cross_file_layout.addWidget(self.cross_file_checkbox)
            
            # RT tolerance input
            rt_layout = QHBoxLayout()
            rt_label = QLabel("RT Tolerance (min):")
            self.rt_tolerance_edit = QLineEdit("0.05")  # Default 0.05 min (3 seconds)
            self.rt_tolerance_edit.setMaximumWidth(60)
            rt_layout.addWidget(rt_label)
            rt_layout.addWidget(self.rt_tolerance_edit)
            rt_layout.addStretch()
            cross_file_layout.addLayout(rt_layout)
            
            # Spectral similarity threshold
            similarity_layout = QHBoxLayout()
            similarity_label = QLabel("Spectral Similarity Threshold:")
            self.similarity_edit = QLineEdit("0.7")  # Default 0.7
            self.similarity_edit.setMaximumWidth(60)
            similarity_layout.addWidget(similarity_label)
            similarity_layout.addWidget(self.similarity_edit)
            similarity_layout.addStretch()
            cross_file_layout.addLayout(similarity_layout)
            
            # Add explanation text
            info_label = QLabel(
                "This will find peaks with similar retention time and mass spectrum in other processed files "
                "and apply this compound assignment to them."
            )
            info_label.setWordWrap(True)
            info_label.setStyleSheet("color: #666;")
            cross_file_layout.addWidget(info_label)
            
            # Add the group to main layout
            layout.addWidget(cross_file_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        self.save_button.setEnabled(False)  # Disabled until selection
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
    
    def on_text_changed(self, text):
        """Handle text changes in the search edit."""
        # If text is empty, clear results
        if not text:
            self.results_list.clear()
            self.save_button.setEnabled(False)
            return
            
        # If text is too short, don't filter yet - INCREASED TO 5 CHARACTERS
        if len(text) < 5:
            self.results_list.clear()
            return
            
        # Start timer to delay filtering (300ms)
        self.filter_timer.start(300)
    
    def filter_compounds(self):
        """Filter compounds based on search text."""
        text = self.search_edit.text().lower()
        
        # If text is empty or too short, clear results
        if not text or len(text) < 5:  # Increased minimum to 5 characters
            self.results_list.clear()
            return
            
        # Filter compounds that contain the search text
        matching_compounds = [comp for comp in self.library_compounds 
                             if text in comp.lower()]
        
        # Update results list - limit to fewer items for better performance
        self.results_list.clear()
        self.results_list.addItems(matching_compounds[:50])  # Reduced from 100 to 50 matches
        
        # Enable save button if we have an exact match or only one result
        if text in self.library_compounds:
            self.save_button.setEnabled(True)
            self.selected_compound = text
        elif len(matching_compounds) == 1:
            self.save_button.setEnabled(True)
            self.selected_compound = matching_compounds[0]
    
    def on_item_selected(self, item):
        """Handle selection from the results list."""
        selected_text = item.text()
        self.search_edit.setText(selected_text)
        self.selected_compound = selected_text
        self.save_button.setEnabled(True)
    
    def get_selected_compound(self):
        """Return the selected compound."""
        return self.selected_compound or self.search_edit.text()
    
    def should_apply_to_files(self):
        """Check if assignment should be applied to other files."""
        if not hasattr(self, 'cross_file_checkbox'):
            return False
        return self.cross_file_checkbox.isChecked()
    
    def get_rt_tolerance(self):
        """Get the RT tolerance value."""
        if not hasattr(self, 'rt_tolerance_edit'):
            return 0.05
        
        try:
            return float(self.rt_tolerance_edit.text())
        except ValueError:
            return 0.05
    
    def get_similarity_threshold(self):
        """Get the spectral similarity threshold."""
        if not hasattr(self, 'similarity_edit'):
            return 0.7
        
        try:
            return float(self.similarity_edit.text())
        except ValueError:
            return 0.7
    
    def accept(self):
        """Handle dialog acceptance with cross-file application."""
        # If cross-file application is enabled and we have a spectrum
        if self.should_apply_to_files() and self.spectrum is not None:
            # Emit signal to request cross-file application
            compound_name = self.get_selected_compound()
            rt = self.peak.retention_time
            tolerance = self.get_rt_tolerance()
            self.apply_to_files_requested.emit(compound_name, rt, tolerance, self.spectrum)
        
        # Continue with standard acceptance
        super().accept()