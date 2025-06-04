from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QListWidget, QFileDialog, QMenu, QAbstractItemView,
                              QCheckBox, QGroupBox, QFormLayout, QSpinBox, QMessageBox)
from PySide6.QtCore import Qt, Signal, QSize

class BatchJobDialog(QDialog):
    """Dialog for setting up a batch processing job."""
    
    # Signal emitted when user clicks "Start Processing"
    start_batch = Signal(list, dict)  # directories, options
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Setup")
        self.setMinimumSize(600, 400)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Initialize directories list
        self.directories = []
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Select one or more directories containing GC-MS data to process.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Directory list
        self.directory_list = QListWidget()
        self.directory_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.directory_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.directory_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.directory_list)
        
        # Directory buttons
        dir_button_layout = QHBoxLayout()
        
        add_dir_button = QPushButton("Add Directory...")
        add_dir_button.clicked.connect(self.add_directory)
        dir_button_layout.addWidget(add_dir_button)
        
        remove_dir_button = QPushButton("Remove Selected")
        remove_dir_button.clicked.connect(self.remove_selected_directories)
        dir_button_layout.addWidget(remove_dir_button)
        
        dir_button_layout.addStretch()
        layout.addLayout(dir_button_layout)
        
        # Processing options group
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout(options_group)
        
        # Integration options 
        self.integration_check = QCheckBox("Perform peak integration")
        self.integration_check.setChecked(True)
        options_layout.addWidget(self.integration_check)
        
        # MS search options
        self.ms_search_check = QCheckBox("Perform MS library search")
        self.ms_search_check.setChecked(True)
        options_layout.addWidget(self.ms_search_check)
        
        # Add options group to main layout
        layout.addWidget(options_group)
        
        # File handling options group
        file_group = QGroupBox("File Handling Options")
        file_layout = QVBoxLayout(file_group)
        
        # Save results checkbox
        self.save_results_check = QCheckBox("Save integration results to JSON")
        self.save_results_check.setChecked(True)
        file_layout.addWidget(self.save_results_check)
        
        # Export CSV checkbox
        self.export_csv_check = QCheckBox("Export results to CSV")
        self.export_csv_check.setChecked(True)
        file_layout.addWidget(self.export_csv_check)
        
        # Overwrite existing files checkbox
        self.overwrite_check = QCheckBox("Overwrite existing result files")
        self.overwrite_check.setChecked(True)
        file_layout.addWidget(self.overwrite_check)
        
        # Add file group to main layout
        layout.addWidget(file_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # Start button - disabled until at least one directory is added
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.on_start_processing)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        layout.addLayout(button_layout)
    
    def add_directory(self):
        """Add a directory to the list."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", "", QFileDialog.ShowDirsOnly
        )
        
        if directory:
            # Check if directory is already in the list
            if directory in self.directories:
                QMessageBox.warning(
                    self, 
                    "Directory Already Added", 
                    f"The directory {directory} is already in the list."
                )
                return
            
            # Check if it contains .D directories
            import os
            has_data_dirs = False
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and item.endswith('.D'):
                    has_data_dirs = True
                    break
            
            if not has_data_dirs:
                result = QMessageBox.question(
                    self,
                    "No Data Directories Found",
                    f"The directory {directory} does not contain any .D subdirectories. Add it anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if result != QMessageBox.Yes:
                    return
            
            # Add to the list
            self.directories.append(directory)
            self.directory_list.addItem(directory)
            
            # Enable start button if we have directories
            self.start_button.setEnabled(True)
    
    def remove_selected_directories(self):
        """Remove selected directories from the list."""
        selected_items = self.directory_list.selectedItems()
        
        if not selected_items:
            return
        
        # Get list of directories to remove
        to_remove = [item.text() for item in selected_items]
        
        # Remove from our list
        for directory in to_remove:
            self.directories.remove(directory)
        
        # Remove from the widget
        for item in selected_items:
            row = self.directory_list.row(item)
            self.directory_list.takeItem(row)
        
        # Disable start button if no directories left
        self.start_button.setEnabled(len(self.directories) > 0)
    
    def show_context_menu(self, position):
        """Show context menu for the directory list."""
        menu = QMenu()
        
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(self.remove_selected_directories)
        
        menu.exec_(self.directory_list.mapToGlobal(position))
    
    def on_start_processing(self):
        """Handle start processing button click."""
        if not self.directories:
            QMessageBox.warning(
                self, 
                "No Directories", 
                "Please add at least one directory to process."
            )
            return
        
        # Gather options
        options = {
            'integration': self.integration_check.isChecked(),
            'ms_search': self.ms_search_check.isChecked(),
            # Add the new options
            'save_results': self.save_results_check.isChecked(),
            'export_csv': self.export_csv_check.isChecked(),
            'overwrite_existing': self.overwrite_check.isChecked()
        }
        
        # Emit signal with directories and options
        self.start_batch.emit(self.directories, options)
        
        # Close this dialog
        self.accept()