# -*- coding: utf-8 -*-
"""
ChromaKit-MS JSON to Excel Converter

A utility to compile integration results from multiple .D directories
into a single Excel summary file.

Created on Thu Dec  5 11:28:40 2024
Updated on Jul 28, 2025 - Converted to PySide6 GUI and added compound ID support

@author: ccoatney
"""

import os
import sys
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QCheckBox
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


class JsonToExcelConverter(QMainWindow):
    """Main application window for JSON to Excel conversion."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromaKit-MS: JSON to Excel Converter")
        self.setGeometry(100, 100, 600, 500)
        
        # Initialize variables
        self.input_directory = ""
        self.output_file = ""
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("ChromaKit-MS JSON to Excel Converter")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "Convert ChromaKit-MS integration results from multiple .D directories "
            "into a single Excel summary file."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)
        
        layout.addSpacing(20)
        
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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
    
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


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("ChromaKit-MS JSON to Excel Converter")
    app.setApplicationVersion("1.1.0")
    
    # Create and show main window
    window = JsonToExcelConverter()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
