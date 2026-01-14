# -*- coding: utf-8 -*-
"""
ChromaKit-MS JSON ↔ Excel Converter

A utility to convert between ChromaKit-MS integration results and Excel format:
- JSON → Excel: Compile integration results from multiple .D directories into a single Excel summary
- Excel → JSON: Update JSON files with compound IDs from manually annotated Excel files

Created on Thu Dec  5 11:28:40 2024
Updated on Jul 28, 2025 - Converted to PySide6 GUI and added compound ID support
Updated on Aug  6, 2025 - Added bidirectional functionality for updating JSON from Excel

@author: ccoatney
"""

import os
import sys
import json
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QCheckBox, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


class ProcessingThread(QThread):
    """Thread for processing JSON files to Excel."""
    
    progress_update = Signal(str)  # Progress message
    finished = Signal(bool, str)   # Success, message
    
    def __init__(self, directory, output_file, include_quality_data=True):
        super().__init__()
        self.directory = directory
        self.output_file = output_file
        self.include_quality_data = include_quality_data
    
    def run(self):
        """Process JSON files to Excel in background thread."""
        try:
            self.progress_update.emit("Starting processing...")
            success = self.process_json_to_excel(self.directory, self.output_file)
            
            if success:
                self.finished.emit(True, f"Successfully created Excel file: {self.output_file}")
            else:
                self.finished.emit(False, "Processing completed but no JSON files were found.")
                
        except Exception as e:
            self.finished.emit(False, f"Error during processing: {str(e)}")

    def process_json_to_excel(self, directory, output_file):
        """Process JSON files to Excel with compound IDs in leftmost column."""
        wb = Workbook()
        ws = wb.active
        ws.title = "Chromatograms"
        
        # Define styles
        header_font = Font(bold=True, size=12)
        subheader_font = Font(bold=True, size=10)
        quality_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # Light yellow
        
        starting_row = 1
        files_processed = 0
        
        self.progress_update.emit("Scanning for JSON files...")
        
        for root, _, files in os.walk(directory):
            # Filter for .json files in the current folder
            json_files = [f for f in files if f.endswith('.json')]
            
            if not json_files:
                continue  # Skip folders without .json files
            
            for file in json_files:
                json_path = os.path.join(root, file)
                self.progress_update.emit(f"Processing {file}...")
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Write sample header information
                    headers = [
                        ("Sample ID:", json_data.get("sample_id", "")),
                        ("Timestamp:", json_data.get("timestamp", "")),
                        ("Method:", json_data.get("method", "")),
                        ("Detector:", json_data.get("detector", "")),
                    ]
                    
                    current_row = starting_row
                    col_offset = 1
                    
                    # Write headers with bold formatting
                    for header, value in headers:
                        cell = ws.cell(row=current_row, column=col_offset, value=header)
                        cell.font = header_font
                        ws.cell(row=current_row, column=col_offset + 1, value=value)
                        current_row += 1
                    
                    # Optional signal and notebook fields
                    if json_data.get("signal"):
                        ws.cell(row=current_row, column=col_offset, value="Signal:")
                        ws.cell(row=current_row, column=col_offset, value="Signal:").font = header_font
                        ws.cell(row=current_row, column=col_offset + 1, value=json_data.get("signal", ""))
                        current_row += 1
                    
                    if json_data.get("notebook"):
                        ws.cell(row=current_row, column=col_offset, value="Notebook:")
                        ws.cell(row=current_row, column=col_offset, value="Notebook:").font = header_font
                        ws.cell(row=current_row, column=col_offset + 1, value=json_data.get("notebook", ""))
                        current_row += 1
                    
                    current_row += 1  # Add spacing
                    
                    # Write peaks table header with Compound ID as first column
                    peaks_headers = ["Compound ID", "Peak #", "Ret Time", "Integrator",
                                   "Width", "Area", "Start Time", "End Time"]
                    
                    # Add quality columns if enabled
                    if self.include_quality_data:
                        peaks_headers.extend(["Match Score", "CAS No", "Quality Issues"])
                    
                    for col, header in enumerate(peaks_headers, start=col_offset):
                        cell = ws.cell(row=current_row, column=col, value=header)
                        cell.font = subheader_font
                        cell.alignment = Alignment(horizontal='center')
                    
                    current_row += 1
                    
                    # Write peaks data
                    for peak in json_data.get("peaks", []):
                        # Compound ID (leftmost column)
                        compound_id = peak.get("Compound ID", "Unknown")
                        ws.cell(row=current_row, column=col_offset, value=compound_id)
                        
                        # Basic peak data
                        ws.cell(row=current_row, column=col_offset + 1,
                               value=int(peak.get("peak_number", 0)))
                        ws.cell(row=current_row, column=col_offset + 2,
                               value=round(float(peak.get("retention_time", 0.0)), 3))
                        ws.cell(row=current_row, column=col_offset + 3, 
                               value=peak.get("integrator", ""))
                        ws.cell(row=current_row, column=col_offset + 4, 
                               value=round(float(peak.get("width", 0.0)), 3))
                        ws.cell(row=current_row, column=col_offset + 5, 
                               value=round(float(peak.get("area", 0.0)), 2))
                        ws.cell(row=current_row, column=col_offset + 6, 
                               value=round(float(peak.get("start_time", 0.0)), 3))
                        ws.cell(row=current_row, column=col_offset + 7, 
                               value=round(float(peak.get("end_time", 0.0)), 3))
                        
                        # Quality data if enabled
                        if self.include_quality_data:
                            # Match score (Qual field)
                            qual_score = peak.get("Qual", "")
                            if qual_score:
                                ws.cell(row=current_row, column=col_offset + 8, 
                                       value=round(float(qual_score), 3))
                            
                            # CAS number
                            cas_no = peak.get("casno", "")
                            ws.cell(row=current_row, column=col_offset + 9, value=cas_no)
                            
                            # Quality issues
                            quality_issues = []
                            if peak.get("is_saturated"):
                                quality_issues.append("Saturated")
                            if peak.get("is_convoluted"):
                                quality_issues.append("Convoluted")
                            if peak.get("quality_issues"):
                                if isinstance(peak["quality_issues"], list):
                                    quality_issues.extend(peak["quality_issues"])
                                else:
                                    quality_issues.append(str(peak["quality_issues"]))
                            
                            quality_text = ", ".join(quality_issues) if quality_issues else ""
                            quality_cell = ws.cell(row=current_row, column=col_offset + 10, value=quality_text)
                            
                            # Highlight rows with quality issues
                            if quality_issues:
                                for col_idx in range(col_offset, col_offset + 11):
                                    ws.cell(row=current_row, column=col_idx).fill = quality_fill
                        
                        current_row += 1
                    
                    # Add spacing between files
                    starting_row = current_row + 2
                    files_processed += 1
                    
                except Exception as e:
                    self.progress_update.emit(f"Error reading {json_path}: {e}")
                    print(f"Error reading {json_path}: {e}")
        
        if files_processed > 0:
            # Auto-adjust column widths
            self.progress_update.emit("Adjusting column widths...")
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                ws.column_dimensions[column_letter].width = adjusted_width
            
            wb.save(output_file)
            self.progress_update.emit(f"Successfully processed {files_processed} files")
            return True
        else:
            return False


class UpdateJsonThread(QThread):
    """Thread for updating JSON files from Excel compound IDs."""
    
    progress_update = Signal(str)  # Progress message
    finished = Signal(bool, str)   # Success, message
    
    def __init__(self, excel_file, json_directory):
        super().__init__()
        self.excel_file = excel_file
        self.json_directory = json_directory
    
    def run(self):
        """Update JSON files with compound IDs from Excel in background thread."""
        try:
            self.progress_update.emit("Reading Excel file...")
            success, files_updated = self.update_json_from_excel(self.excel_file, self.json_directory)
            
            if success:
                self.finished.emit(True, f"Successfully updated {files_updated} JSON files with compound IDs")
            else:
                self.finished.emit(False, "No JSON files were updated. Check file formats and data structure.")
                
        except Exception as e:
            self.finished.emit(False, f"Error during update: {str(e)}")

    def update_json_from_excel(self, excel_file, json_directory):
        """Update JSON files with compound IDs from Excel file."""
        try:
            # Load the Excel workbook
            wb = load_workbook(excel_file)
            ws = wb.active
            
            # Parse the Excel file to extract compound ID mappings
            compound_mappings = {}  # {sample_id: {peak_number: compound_id}}
            current_sample_id = None
            
            self.progress_update.emit("Parsing Excel data...")
            
            for row in ws.iter_rows(values_only=True):
                if not any(row):  # Skip empty rows
                    continue
                
                # Check if this row contains sample ID
                if row[0] == "Sample ID:":
                    current_sample_id = row[1] if len(row) > 1 else None
                    if current_sample_id:
                        compound_mappings[current_sample_id] = {}
                    continue
                
                # Check if this is a peaks data row (has Compound ID in first column)
                if (current_sample_id and len(row) >= 3 and 
                    isinstance(row[1], (int, float)) and  # Peak number
                    isinstance(row[2], (int, float))):    # Retention time
                    
                    compound_id = row[0] if row[0] not in [None, "Compound ID"] else "Unknown"
                    peak_number = int(row[1])
                    
                    compound_mappings[current_sample_id][peak_number] = compound_id
            
            if not compound_mappings:
                self.progress_update.emit("No compound ID mappings found in Excel file")
                return False, 0
            
            self.progress_update.emit(f"Found mappings for {len(compound_mappings)} samples")
            
            # Update JSON files
            files_updated = 0
            
            for root, _, files in os.walk(json_directory):
                json_files = [f for f in files if f.endswith('.json')]
                
                for file in json_files:
                    json_path = os.path.join(root, file)
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        sample_id = json_data.get("sample_id")
                        if not sample_id or sample_id not in compound_mappings:
                            continue
                        
                        self.progress_update.emit(f"Updating {file}...")
                        
                        # Update compound IDs for each peak
                        updated = False
                        sample_mappings = compound_mappings[sample_id]
                        
                        for peak in json_data.get("peaks", []):
                            peak_number = peak.get("peak_number")
                            if peak_number in sample_mappings:
                                old_compound_id = peak.get("Compound ID", "Unknown")
                                new_compound_id = sample_mappings[peak_number]
                                
                                if old_compound_id != new_compound_id:
                                    peak["Compound ID"] = new_compound_id
                                    updated = True
                        
                        # Save the updated JSON file
                        if updated:
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)
                            files_updated += 1
                    
                    except Exception as e:
                        self.progress_update.emit(f"Error updating {json_path}: {e}")
                        continue
            
            return files_updated > 0, files_updated
            
        except Exception as e:
            self.progress_update.emit(f"Error reading Excel file: {e}")
            return False, 0


class JsonToExcelConverter(QDialog):
    """Dialog for JSON to Excel conversion."""
    
    def __init__(self, parent=None, initial_directory=None):
        super().__init__(parent)
        self.setWindowTitle("ChromaKit-MS: JSON ↔ Excel Converter")
        self.setModal(True)
        self.resize(700, 600)
        
        # Initialize variables for JSON to Excel
        self.input_directory = initial_directory or ""
        self.output_file = ""
        
        # Initialize variables for Excel to JSON
        self.excel_input_file = ""
        self.json_directory = initial_directory or ""
        
        self.init_ui()
        
        # Pre-populate directory labels if initial_directory was provided
        if initial_directory:
            self.input_label.setText(initial_directory)
            self.json_dir_label.setText(initial_directory)
            self.log_text.append(f"Pre-loaded directory: {initial_directory}")
            self.update_process_button()
            self.update_json_update_button()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("ChromaKit-MS JSON ↔ Excel Converter")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Convert between ChromaKit-MS JSON files and Excel format, "
            "including compound ID management."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        
        layout.addSpacing(10)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        
        # JSON to Excel tab
        self.json_to_excel_tab = self.create_json_to_excel_tab()
        self.tab_widget.addTab(self.json_to_excel_tab, "JSON → Excel")
        
        # Excel to JSON tab
        self.excel_to_json_tab = self.create_excel_to_json_tab()
        self.tab_widget.addTab(self.excel_to_json_tab, "Excel → JSON")
        
        layout.addWidget(self.tab_widget)
        
        # Shared progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Shared log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
    
    def create_json_to_excel_tab(self):
        """Create the JSON to Excel conversion tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input directory selection
        input_group = QGroupBox("Input Directory")
        input_layout = QVBoxLayout(input_group)
        
        input_button_layout = QHBoxLayout()
        self.input_label = QLabel("No directory selected")
        self.input_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.select_input_btn = QPushButton("Select Directory")
        self.select_input_btn.clicked.connect(self.select_input_directory)
        
        input_button_layout.addWidget(self.input_label, 1)
        input_button_layout.addWidget(self.select_input_btn)
        input_layout.addLayout(input_button_layout)
        
        layout.addWidget(input_group)
        
        # Output file selection
        output_group = QGroupBox("Output Excel File")
        output_layout = QVBoxLayout(output_group)
        
        output_button_layout = QHBoxLayout()
        self.output_label = QLabel("No output file selected")
        self.output_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.select_output_btn = QPushButton("Select Output File")
        self.select_output_btn.clicked.connect(self.select_output_file)
        
        output_button_layout.addWidget(self.output_label, 1)
        output_button_layout.addWidget(self.select_output_btn)
        output_layout.addLayout(output_button_layout)
        
        layout.addWidget(output_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.quality_data_checkbox = QCheckBox("Include MS search quality data (match scores, CAS numbers, quality issues)")
        self.quality_data_checkbox.setChecked(True)
        options_layout.addWidget(self.quality_data_checkbox)
        
        layout.addWidget(options_group)
        
        # Process button
        self.process_btn = QPushButton("Convert to Excel")
        self.process_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        layout.addWidget(self.process_btn)
        
        layout.addStretch()
        return tab
    
    def create_excel_to_json_tab(self):
        """Create the Excel to JSON update tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Description
        desc_label = QLabel(
            "Update JSON files with compound IDs from an Excel file that was "
            "previously exported and manually annotated with compound identifications."
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        layout.addSpacing(10)
        
        # Excel input file selection
        excel_input_group = QGroupBox("Input Excel File")
        excel_input_layout = QVBoxLayout(excel_input_group)
        
        excel_input_button_layout = QHBoxLayout()
        self.excel_input_label = QLabel("No Excel file selected")
        self.excel_input_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.select_excel_input_btn = QPushButton("Select Excel File")
        self.select_excel_input_btn.clicked.connect(self.select_excel_input_file)
        
        excel_input_button_layout.addWidget(self.excel_input_label, 1)
        excel_input_button_layout.addWidget(self.select_excel_input_btn)
        excel_input_layout.addLayout(excel_input_button_layout)
        
        layout.addWidget(excel_input_group)
        
        # JSON directory selection
        json_dir_group = QGroupBox("JSON Files Directory")
        json_dir_layout = QVBoxLayout(json_dir_group)
        
        json_dir_button_layout = QHBoxLayout()
        self.json_dir_label = QLabel("No directory selected")
        self.json_dir_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.select_json_dir_btn = QPushButton("Select Directory")
        self.select_json_dir_btn.clicked.connect(self.select_json_directory)
        
        json_dir_button_layout.addWidget(self.json_dir_label, 1)
        json_dir_button_layout.addWidget(self.select_json_dir_btn)
        json_dir_layout.addLayout(json_dir_button_layout)
        
        layout.addWidget(json_dir_group)
        
        # Update button
        self.update_json_btn = QPushButton("Update JSON Files")
        self.update_json_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.update_json_btn.clicked.connect(self.start_json_update)
        self.update_json_btn.setEnabled(False)
        layout.addWidget(self.update_json_btn)
        
        layout.addStretch()
        return tab
    
    def select_input_directory(self):
        """Select input directory containing .D folders with JSON files."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory Containing .D Folders",
            ""
        )
        
        if directory:
            self.input_directory = directory
            self.input_label.setText(directory)
            self.log_text.append(f"Selected input directory: {directory}")
            self.update_process_button()
    
    def select_output_file(self):
        """Select output Excel file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Excel File",
            "ChromaKit_Results.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            self.output_file = file_path
            self.output_label.setText(file_path)
            self.log_text.append(f"Selected output file: {file_path}")
            self.update_process_button()
    
    def update_process_button(self):
        """Enable process button if both input and output are selected."""
        self.process_btn.setEnabled(bool(self.input_directory and self.output_file))
    
    def start_processing(self):
        """Start the JSON to Excel conversion process."""
        if not self.input_directory or not self.output_file:
            QMessageBox.warning(self, "Error", "Please select both input directory and output file.")
            return
        
        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.select_input_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.quality_data_checkbox.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear log
        self.log_text.clear()
        self.log_text.append("Starting conversion process...")
        
        # Start processing thread
        self.processing_thread = ProcessingThread(
            self.input_directory,
            self.output_file,
            self.quality_data_checkbox.isChecked()
        )
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()
    
    def update_progress(self, message):
        """Update progress log."""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
        QApplication.processEvents()
    
    def processing_finished(self, success, message):
        """Handle processing completion."""
        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.select_input_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        self.quality_data_checkbox.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Show completion message
        self.log_text.append(f"Processing completed: {message}")
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Warning", message)
    
    def select_excel_input_file(self):
        """Select Excel input file for updating JSON files."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            "",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            self.excel_input_file = file_path
            self.excel_input_label.setText(file_path)
            self.log_text.append(f"Selected Excel input file: {file_path}")
            self.update_json_update_button()
    
    def select_json_directory(self):
        """Select directory containing JSON files to update."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory Containing JSON Files",
            ""
        )
        
        if directory:
            self.json_directory = directory
            self.json_dir_label.setText(directory)
            self.log_text.append(f"Selected JSON directory: {directory}")
            self.update_json_update_button()
    
    def update_json_update_button(self):
        """Enable JSON update button if both Excel file and JSON directory are selected."""
        self.update_json_btn.setEnabled(bool(self.excel_input_file and self.json_directory))
    
    def start_json_update(self):
        """Start the Excel to JSON update process."""
        if not self.excel_input_file or not self.json_directory:
            QMessageBox.warning(self, "Error", "Please select both Excel file and JSON directory.")
            return
        
        # Disable UI during processing
        self.update_json_btn.setEnabled(False)
        self.select_excel_input_btn.setEnabled(False)
        self.select_json_dir_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Clear log
        self.log_text.clear()
        self.log_text.append("Starting JSON update process...")
        
        # Start update thread
        self.update_thread = UpdateJsonThread(
            self.excel_input_file,
            self.json_directory
        )
        self.update_thread.progress_update.connect(self.update_progress)
        self.update_thread.finished.connect(self.json_update_finished)
        self.update_thread.start()
    
    def json_update_finished(self, success, message):
        """Handle JSON update completion."""
        # Re-enable UI
        self.update_json_btn.setEnabled(True)
        self.select_excel_input_btn.setEnabled(True)
        self.select_json_dir_btn.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Show completion message
        self.log_text.append(f"JSON update completed: {message}")
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Warning", message)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ChromaKit-MS JSON ↔ Excel Converter")
    app.setApplicationVersion("2.0.0")
    
    # Create and show main window
    window = JsonToExcelConverter()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
