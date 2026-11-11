import os

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QListWidget, QFileDialog, QMenu, QAbstractItemView,
                              QCheckBox, QGroupBox, QFormLayout, QSpinBox, QMessageBox,
                              QComboBox, QLineEdit, QWidget)
from PySide6.QtCore import Qt, Signal, QSize

MAX_DETECTOR_RUNS = 2

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
        instructions = QLabel("Select one or more directories containing .C sample folders to process.")
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

        # Spectral deconvolution (correlation grouping, after mzmine)
        self.spectral_deconv_check = QCheckBox("Run spectral deconvolution before MS search (GC-MS only)")
        self.spectral_deconv_check.setToolTip(
            "<p>Before MS library search, give each FID peak a cleaner mass spectrum built from the "
            "ions that elute with it, using the settings in MS Options → Spectral Deconvolution. "
            "Compounds closer than about 1.5 peak widths usually still come out as one spectrum. "
            "Ignored for non-GC-MS data.</p>"
        )
        self.spectral_deconv_check.setChecked(True)
        options_layout.addWidget(self.spectral_deconv_check)

        # Timing option
        self.time_steps_check = QCheckBox("Time processing steps")
        self.time_steps_check.setChecked(False)
        options_layout.addWidget(self.time_steps_check)

        # Speed option
        self.skip_ui_check = QCheckBox("Skip UI updates during batch (faster)")
        self.skip_ui_check.setToolTip(
            "Skips chromatogram re-rendering between files. "
            "Uncheck to watch the display update as files are processed."
        )
        self.skip_ui_check.setChecked(True)
        options_layout.addWidget(self.skip_ui_check)

        # Manual overrides (opt-in; deliberately not remembered between runs)
        self.manual_overrides_check = QCheckBox("Apply manual assignment overrides (manual_assignments.json)")
        self.manual_overrides_check.setToolTip(
            "<p>Re-assign peaks using the saved manual assignments: a peak within 0.05 min of a saved "
            "retention time takes that compound name. Where a saved assignment has a stored mass "
            "spectrum, the peak's spectrum must match it (similarity 0.7 or higher); if it can't be "
            "compared, for example GC-only data, that assignment is skipped. Off by default.</p>"
        )
        self.manual_overrides_check.setChecked(False)
        options_layout.addWidget(self.manual_overrides_check)

        # Add options group to main layout
        layout.addWidget(options_group)

        layout.addWidget(self._build_detector_group())
        
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
    
    def _build_detector_group(self):
        """Per-detector runs: each detector channel processed with its own method."""
        group = QGroupBox("Detectors and Methods")
        group_layout = QVBoxLayout(group)

        self.per_detector_check = QCheckBox("Process each detector below with its own method")
        self.per_detector_check.setToolTip(
            "Every sample is processed once per detector listed here, using that "
            "detector's method. Results are saved per detector. When unchecked, "
            "the detector selected in Settings → Select Detector Channel and the "
            "current method are used."
        )
        group_layout.addWidget(self.per_detector_check)

        self.detector_rows = []
        rows_widget = QWidget()
        form = QFormLayout(rows_widget)
        form.setContentsMargins(0, 0, 0, 0)
        for n in range(1, MAX_DETECTOR_RUNS + 1):
            detector_combo = QComboBox()
            detector_combo.setEditable(True)
            detector_combo.setMinimumWidth(110)
            detector_combo.setToolTip("Detector channel name, e.g. FID1A")
            method_edit = QLineEdit()
            method_edit.setPlaceholderText("Method file (.chromethod); blank = current settings")
            browse = QPushButton("Browse...")
            browse.clicked.connect(
                lambda _=False, c=detector_combo, e=method_edit: self._browse_method(c, e))
            row = QHBoxLayout()
            row.addWidget(detector_combo)
            row.addWidget(method_edit, 1)
            row.addWidget(browse)
            form.addRow(f"Detector {n}:", row)
            self.detector_rows.append((detector_combo, method_edit))
        group_layout.addWidget(rows_widget)

        rows_widget.setEnabled(False)
        self.per_detector_check.toggled.connect(rows_widget.setEnabled)
        self._refresh_detector_choices()
        self._prefill_first_row_from_main_window()
        return group

    def _prefill_first_row_from_main_window(self):
        """Offer the saved method's detector and path in the first row.

        Only fills an empty row and never ticks per_detector_check.
        """
        parent = self.parent()
        path = getattr(parent, "current_method_path", None)
        method = getattr(parent, "current_method", None)
        detector = getattr(method, "detector", None)
        if not path or not detector:
            return
        combo, method_edit = self.detector_rows[0]
        if combo.currentText().strip() or method_edit.text().strip():
            return
        combo.setCurrentText(detector)
        method_edit.setText(str(path))

    def _browse_method(self, detector_combo, method_edit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Method", "", "ChromaKit Method (*.chromethod)")
        if not path:
            return
        method_edit.setText(path)
        if not detector_combo.currentText().strip():
            try:
                from logic.method import ChromaMethod
                detector = ChromaMethod.from_file(path).detector
            except Exception:
                detector = None
            if detector:
                detector_combo.setCurrentText(detector)

    def _refresh_detector_choices(self):
        """Offer the detector channels found in the first sample of the added directories."""
        detectors = []
        for directory in self.directories:
            try:
                names = sorted(os.listdir(directory))
            except OSError:
                continue
            for name in names:
                path = os.path.join(directory, name)
                if not (os.path.isdir(path) and name.endswith('.C')):
                    continue
                try:
                    from logic.c_folder import CFolder
                    detectors = list(CFolder.open(path).get_available_detectors() or [])
                except Exception:
                    detectors = []
                break
            if detectors:
                break
        for i, (combo, _) in enumerate(self.detector_rows):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(detectors)
            if current:
                combo.setCurrentText(current)
            elif i < len(detectors):
                combo.setCurrentIndex(i)
            else:
                combo.setCurrentText("")
            combo.blockSignals(False)

    def get_detector_runs(self):
        """Return [{'detector', 'method_path'}] for filled-in rows (empty when unchecked).

        Raises ValueError describing the first problem found.
        """
        if not self.per_detector_check.isChecked():
            return []
        from logic.method import ChromaMethod
        runs = []
        for n, (combo, method_edit) in enumerate(self.detector_rows, start=1):
            detector = combo.currentText().strip()
            method_path = method_edit.text().strip()
            if not detector and not method_path:
                continue
            if method_path:
                if not os.path.isfile(method_path):
                    raise ValueError(f"Detector {n}: method file not found:\n{method_path}")
                try:
                    method = ChromaMethod.from_file(method_path)
                except Exception as e:
                    raise ValueError(f"Detector {n}: could not read method:\n{e}")
                detector = detector or (method.detector or "")
            if not detector:
                raise ValueError(f"Detector {n}: enter a detector channel name.")
            if any(r['detector'] == detector for r in runs):
                raise ValueError(f"Detector {detector} is listed more than once.")
            runs.append({'detector': detector, 'method_path': method_path or None})
        if not runs:
            raise ValueError("Enter at least one detector, or uncheck "
                             "\"Process each detector below with its own method\".")
        return runs

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
            
            # Check if it contains .C directories
            import os
            has_data_dirs = False
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and item.endswith('.C'):
                    has_data_dirs = True
                    break
            
            if not has_data_dirs:
                result = QMessageBox.question(
                    self,
                    "No .C Folders Found",
                    f"The directory {directory} does not contain any .C sample folders. Add it anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if result != QMessageBox.Yes:
                    return
            
            # Add to the list
            self.directories.append(directory)
            self.directory_list.addItem(directory)
            self._refresh_detector_choices()
            
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
        
        try:
            detector_runs = self.get_detector_runs()
        except ValueError as e:
            QMessageBox.warning(self, "Detectors and Methods", str(e))
            return

        # Gather options
        options = {
            'integration': self.integration_check.isChecked(),
            'ms_search': self.ms_search_check.isChecked(),
            'spectral_deconv': self.spectral_deconv_check.isChecked(),
            'save_results': self.save_results_check.isChecked(),
            'export_csv': self.export_csv_check.isChecked(),
            'overwrite_existing': self.overwrite_check.isChecked(),
            'time_steps': self.time_steps_check.isChecked(),
            'skip_ui_updates': self.skip_ui_check.isChecked(),
            'apply_manual_overrides': self.manual_overrides_check.isChecked(),
            'detector_runs': detector_runs,
        }
        
        # Emit signal with directories and options
        self.start_batch.emit(self.directories, options)
        
        # Close this dialog
        self.accept()