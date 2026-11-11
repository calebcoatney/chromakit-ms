import json
import logging
import os

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QListWidget, QFileDialog, QMenu, QAbstractItemView,
                              QCheckBox, QGroupBox, QFormLayout, QSpinBox, QMessageBox,
                              QComboBox, QLineEdit, QWidget, QListWidgetItem)
from PySide6.QtCore import Qt, Signal, QSize, QSettings

logger = logging.getLogger(__name__)

MAX_DETECTOR_RUNS = 4
INSTRUMENT_TAGS = ("", "GC1", "GC2")
SETUP_VERSION = 1
SETUP_SETTINGS_KEY = "batch/detector_setup"


def _settings():
    """The app's QSettings; tests monkeypatch this so they never touch real settings."""
    return QSettings("CalebCoatney", "ChromaKit")


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
        self.directory_instruments = {}  # folder path (as in self.directories) -> "GC1"/"GC2"/""
        
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

        instrument_label = QLabel("Instrument:")
        dir_button_layout.addWidget(instrument_label)
        self.instrument_set_combo = QComboBox()
        self._fill_instrument_combo(self.instrument_set_combo)
        self.instrument_set_combo.setAccessibleName("Instrument for selected folders")
        self.instrument_set_combo.setToolTip(
            "Tag the selected folders as GC1 or GC2. Tagged folders are processed with "
            "that instrument's detector rows and can be merged into one GC1&2 workbook.")
        instrument_label.setBuddy(self.instrument_set_combo)
        self.instrument_set_combo.activated.connect(self._apply_instrument_to_selection)
        dir_button_layout.addWidget(self.instrument_set_combo)
        
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

        layout.addLayout(self._build_workbook_row())
        
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
    
    @staticmethod
    def _fill_instrument_combo(combo):
        for tag in INSTRUMENT_TAGS:
            combo.addItem(tag or "(none)", tag)

    def _build_workbook_row(self):
        row = QHBoxLayout()
        label = QLabel("Output workbook:")
        self.workbook_edit = QLineEdit()
        self.workbook_edit.setPlaceholderText("GC1&2 paste workbook (.xlsx); needs a GC1 and a GC2 folder")
        self.workbook_edit.setAccessibleName("Output workbook")
        self.workbook_edit.setToolTip(
            "Where to write the paired GC1 and GC2 results. Needs a tagged folder, a ticked "
            "\"Process each detector below\" box with at least one detector row that has an "
            "instrument, and JSON export during batch.")
        label.setBuddy(self.workbook_edit)
        self.workbook_browse = QPushButton("Browse...")
        self.workbook_browse.setAccessibleName("Browse for output workbook")
        self.workbook_browse.clicked.connect(self._browse_workbook)
        row.addWidget(label)
        row.addWidget(self.workbook_edit, 1)
        row.addWidget(self.workbook_browse)
        self.per_detector_check.toggled.connect(self._update_workbook_state)
        for inst_combo, (combo, method_edit) in zip(self.instrument_combos, self.detector_rows):
            inst_combo.currentIndexChanged.connect(self._update_workbook_state)
            combo.currentTextChanged.connect(self._update_workbook_state)
            method_edit.textChanged.connect(self._update_workbook_state)
        self._update_workbook_state()
        return row

    def _browse_workbook(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Output Workbook", self.workbook_edit.text(), "Excel workbook (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.workbook_edit.setText(path)

    def _update_workbook_state(self):
        """Enable the workbook field when a folder is tagged; prefill it when both tags exist."""
        tagged = (any(self.directory_instruments.get(d) for d in self.directories)
                  and self._gc12_mode())
        self.workbook_edit.setEnabled(tagged)
        self.workbook_browse.setEnabled(tagged)
        if self.workbook_edit.text().strip():
            return
        first = {}
        for d in self.directories:
            tag = self.directory_instruments.get(d, "")
            if tag and tag not in first:
                first[tag] = d
        if "GC1" in first and "GC2" in first:
            gc1 = first["GC1"]
            name = os.path.basename(gc1.rstrip("/\\"))
            parent = os.path.dirname(gc1.rstrip("/\\"))
            self.workbook_edit.setText(os.path.join(parent, f"{name} GC1&2.xlsx"))

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
        self.instrument_combos = []
        rows_widget = QWidget()
        form = QFormLayout(rows_widget)
        form.setContentsMargins(0, 0, 0, 0)
        for n in range(1, MAX_DETECTOR_RUNS + 1):
            instrument_combo = QComboBox()
            self._fill_instrument_combo(instrument_combo)
            instrument_combo.setAccessibleName(f"Instrument for detector {n}")
            instrument_combo.setToolTip("GC1 or GC2; blank for a folder with no instrument tag")
            detector_combo = QComboBox()
            detector_combo.setAccessibleName(f"Detector {n} channel")
            detector_combo.setEditable(True)
            detector_combo.setMinimumWidth(110)
            detector_combo.setToolTip("Detector channel name, e.g. FID1A")
            method_edit = QLineEdit()
            method_edit.setAccessibleName(f"Detector {n} method file")
            method_edit.setPlaceholderText("Method file (.chromethod); blank = current settings")
            browse = QPushButton("Browse...")
            browse.setAccessibleName(f"Browse for detector {n} method")
            browse.clicked.connect(
                lambda _=False, c=detector_combo, e=method_edit: self._browse_method(c, e))
            row = QHBoxLayout()
            row.addWidget(instrument_combo)
            row.addWidget(detector_combo)
            row.addWidget(method_edit, 1)
            row.addWidget(browse)
            form.addRow(f"Detector {n}:", row)
            self.detector_rows.append((detector_combo, method_edit))
            self.instrument_combos.append(instrument_combo)
        group_layout.addWidget(rows_widget)

        setup_row = QHBoxLayout()
        save_setup = QPushButton("Save setup...")
        save_setup.setToolTip("Save these rows (instrument, detector, method) to a file")
        save_setup.clicked.connect(self._save_setup_dialog)
        load_setup = QPushButton("Load setup...")
        load_setup.setToolTip("Fill these rows from a saved setup file")
        load_setup.clicked.connect(self._load_setup_dialog)
        setup_row.addWidget(save_setup)
        setup_row.addWidget(load_setup)
        setup_row.addStretch()
        group_layout.addLayout(setup_row)

        rows_widget.setEnabled(False)
        self.per_detector_check.toggled.connect(rows_widget.setEnabled)
        self._refresh_detector_choices()
        self._restore_remembered_setup()
        self._prefill_first_row_from_main_window()
        return group

    # ---- setup rows: remembered, saved and loaded ----

    def get_setup_rows(self):
        """Raw rows as typed: [{'instrument', 'detector', 'method_path'}], blanks skipped."""
        rows = []
        for inst_combo, (combo, method_edit) in zip(self.instrument_combos, self.detector_rows):
            detector = combo.currentText().strip()
            method_path = method_edit.text().strip()
            if not detector and not method_path:
                continue
            rows.append({'instrument': inst_combo.currentData() or "",
                         'detector': detector,
                         'method_path': method_path or None})
        return rows

    @staticmethod
    def _parse_setup(data):
        """Validate decoded setup JSON and return its rows; raises ValueError."""
        if not isinstance(data, dict) or data.get("version") != SETUP_VERSION:
            version = data.get("version") if isinstance(data, dict) else None
            raise ValueError(f"Unsupported setup version: {version!r} (expected {SETUP_VERSION}).")
        rows = data.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Setup file has no list of rows.")
        if len(rows) > MAX_DETECTOR_RUNS:
            raise ValueError(f"Setup has {len(rows)} rows; at most {MAX_DETECTOR_RUNS} are supported.")
        parsed = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Setup row is not an object.")
            instrument = row.get("instrument", "")
            if instrument not in INSTRUMENT_TAGS:
                raise ValueError(f"Setup row has unknown instrument {instrument!r}.")
            parsed.append({'instrument': instrument,
                           'detector': str(row.get("detector") or ""),
                           'method_path': row.get("method_path") or None})
        return parsed

    def _apply_rows(self, rows):
        for i, (inst_combo, (combo, method_edit)) in enumerate(
                zip(self.instrument_combos, self.detector_rows)):
            row = rows[i] if i < len(rows) else {'instrument': "", 'detector': "", 'method_path': None}
            inst_combo.setCurrentIndex(max(0, inst_combo.findData(row['instrument'])))
            combo.setCurrentText(row['detector'])
            method_edit.setText(row['method_path'] or "")

    def _setup_payload(self):
        return {"version": SETUP_VERSION, "rows": self.get_setup_rows()}

    def save_setup(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._setup_payload(), f, indent=2)

    def load_setup(self, path):
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Setup file is not valid JSON: {e}")
        self._apply_rows(self._parse_setup(data))

    def _save_setup_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Setup", "", "Detector setup (*.json)")
        if not path:
            return
        try:
            self.save_setup(path)
        except OSError as e:
            QMessageBox.warning(self, "Save Setup", f"Could not save the setup:\n{e}")

    def _load_setup_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Setup", "", "Detector setup (*.json)")
        if not path:
            return
        try:
            self.load_setup(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Load Setup", f"Could not load the setup:\n{e}")
            return
        self.per_detector_check.setChecked(True)

    def _restore_remembered_setup(self):
        raw = _settings().value(SETUP_SETTINGS_KEY)
        if not raw:
            return
        try:
            self._apply_rows(self._parse_setup(json.loads(raw)))
        except (ValueError, TypeError) as e:
            logger.warning("Ignoring remembered batch detector setup: %s", e)

    def _remember_setup(self):
        if self.per_detector_check.isChecked() and self.get_setup_rows():
            _settings().setValue(SETUP_SETTINGS_KEY, json.dumps(self._setup_payload()))

    # ---- instrument tags on folders ----

    @staticmethod
    def guess_instrument(path):
        """'GC1' or 'GC2' when the folder path, else its first .C child's name, names exactly one."""
        def one_of(text):
            text = text.upper()
            found = [t for t in ("GC1", "GC2") if t in text]
            return found[0] if len(found) == 1 else ""
        tag = one_of(path)
        if tag:
            return tag
        try:
            names = sorted(os.listdir(path))
        except OSError:
            return ""
        for name in names:
            if name.endswith('.C') and os.path.isdir(os.path.join(path, name)):
                return one_of(name)
        return ""

    def set_directory_instrument(self, path, tag):
        if tag not in INSTRUMENT_TAGS:
            raise ValueError(f"Unknown instrument tag {tag!r}")
        if path not in self.directories:
            raise ValueError(f"Directory not in the list: {path}")
        self.directory_instruments[path] = tag
        self._refresh_item_text(path)
        self._update_workbook_state()

    def _refresh_item_text(self, path):
        tag = self.directory_instruments.get(path, "")
        for i in range(self.directory_list.count()):
            item = self.directory_list.item(i)
            if item.data(Qt.UserRole) == path:
                item.setText(f"{tag} \u2014 {path}" if tag else path)

    def _apply_instrument_to_selection(self, index):
        tag = self.instrument_set_combo.itemData(index) or ""
        for item in self.directory_list.selectedItems():
            self.set_directory_instrument(item.data(Qt.UserRole), tag)

    def _append_directory(self, directory):
        """Add a folder to the list, tagging it when its name says which instrument."""
        self.directories.append(directory)
        item = QListWidgetItem(directory)
        item.setData(Qt.UserRole, directory)
        self.directory_list.addItem(item)
        self.directory_instruments[directory] = ""
        self.set_directory_instrument(directory, self.guess_instrument(directory))

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
        """Return [{'instrument', 'detector', 'method_path'}] for filled-in rows.

        Empty when unchecked and no folder is tagged. Raises ValueError describing
        the first problem found.
        """
        if not self.per_detector_check.isChecked():
            return []
        from logic.method import ChromaMethod
        runs = []
        for n, (inst_combo, (combo, method_edit)) in enumerate(
                zip(self.instrument_combos, self.detector_rows), start=1):
            instrument = inst_combo.currentData() or ""
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
            if any(r['instrument'] == instrument and r['detector'] == detector for r in runs):
                label = f"{instrument} {detector}" if instrument else f"Detector {detector}"
                raise ValueError(f"{label} is listed more than once.")
            runs.append({'instrument': instrument, 'detector': detector,
                         'method_path': method_path or None})
        if runs and not any(r['instrument'] for r in runs):
            return runs  # PR #5 job: tags don't apply without an instrument on a row
        tagged = [(d, self.directory_instruments.get(d, "")) for d in self.directories]
        tagged = [(d, tag) for d, tag in tagged if tag]
        for directory, tag in tagged:
            if not any(r['instrument'] == tag for r in runs):
                name = os.path.basename(directory.rstrip("/\\")) or directory
                raise ValueError(f"Folder {name} is tagged {tag} but no detector row is for {tag}.")
        tags_in_use = {tag for _, tag in tagged}
        for r in runs:
            if r['instrument'] and r['instrument'] not in tags_in_use:
                raise ValueError(f"The {r['instrument']} {r['detector']} row has no folder tagged "
                                 f"{r['instrument']}.")
        if not runs:
            raise ValueError("Enter at least one detector, or uncheck "
                             "\"Process each detector below with its own method\".")
        return runs

    def _gc12_mode(self):
        """Tags take effect only when the per-detector box is ticked and a filled row has an instrument."""
        return (self.per_detector_check.isChecked()
                and any(r['instrument'] for r in self.get_setup_rows()))

    def get_workbook_path(self):
        """The GC1&2 workbook path, or None when the field is off or empty.

        Raises ValueError when the job can't produce it.
        """
        path = self.workbook_edit.text().strip()
        if not self._gc12_mode() or not self.workbook_edit.isEnabled() or not path:
            return None
        tags = {self.directory_instruments.get(d, "") for d in self.directories}
        for tag in ("GC1", "GC2"):
            if tag not in tags:
                raise ValueError(f"An output workbook needs both instruments, but no folder is "
                                 f"tagged {tag}.")
        export_manager = getattr(self.parent(), "export_manager", None)
        if export_manager is not None and not export_manager.should_export_json('batch'):
            raise ValueError("Turn on JSON export during batch (Export Settings) "
                             "to write the GC1&2 workbook.")
        return path

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
            self._append_directory(directory)
            self._refresh_detector_choices()
            
            # Enable start button if we have directories
            self.start_button.setEnabled(True)
    
    def remove_selected_directories(self):
        """Remove selected directories from the list."""
        selected_items = self.directory_list.selectedItems()
        
        if not selected_items:
            return
        
        # Get list of directories to remove
        to_remove = [item.data(Qt.UserRole) for item in selected_items]
        
        # Remove from our list
        for directory in to_remove:
            self.directories.remove(directory)
            self.directory_instruments.pop(directory, None)
        
        # Remove from the widget
        for item in selected_items:
            row = self.directory_list.row(item)
            self.directory_list.takeItem(row)
        
        # Disable start button if no directories left
        self.start_button.setEnabled(len(self.directories) > 0)
        self._update_workbook_state()
    
    def show_context_menu(self, position):
        """Show context menu for the directory list."""
        menu = QMenu()
        
        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(self.remove_selected_directories)
        
        menu.addSeparator()
        for tag in INSTRUMENT_TAGS:
            action = menu.addAction(f"Set instrument: {tag}" if tag else "Clear instrument tag")
            action.triggered.connect(
                lambda _=False, t=tag: [self.set_directory_instrument(i.data(Qt.UserRole), t)
                                        for i in self.directory_list.selectedItems()])

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
        try:
            workbook = self.get_workbook_path()
        except ValueError as e:
            QMessageBox.warning(self, "Output Workbook", str(e))
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
            'directory_instruments': {
                d: (self.directory_instruments.get(d, "") if self._gc12_mode() else "")
                for d in self.directories},
            'gc12_workbook': workbook,
            'gc12_rows': [dict(r) for r in detector_runs],
        }
        self._remember_setup()
        
        # Emit signal with directories and options
        self.start_batch.emit(self.directories, options)
        
        # Close this dialog
        self.accept()