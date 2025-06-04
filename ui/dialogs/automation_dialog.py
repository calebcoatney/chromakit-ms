from PySide6.QtWidgets import (QDialog, QVBoxLayout, QProgressBar, 
                              QTextEdit, QPushButton, QHBoxLayout, 
                              QLabel, QFrame)
from PySide6.QtCore import Qt, Signal

class AutomationDialog(QDialog):
    """Dialog for showing automation progress."""
    
    cancelled = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automation Progress")
        self.setMinimumSize(600, 400)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add title
        title_label = QLabel("Processing Files")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Add file progress section
        file_frame = QFrame()
        file_frame.setFrameShape(QFrame.StyledPanel)
        file_layout = QVBoxLayout(file_frame)
        
        self.file_label = QLabel("Preparing...")
        file_layout.addWidget(self.file_label)
        
        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 100)
        file_layout.addWidget(self.file_progress)
        
        layout.addWidget(file_frame)
        
        # Add overall progress section
        overall_frame = QFrame()
        overall_frame.setFrameShape(QFrame.StyledPanel)
        overall_layout = QVBoxLayout(overall_frame)
        
        self.overall_label = QLabel("Overall Progress:")
        overall_layout.addWidget(self.overall_label)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        overall_layout.addWidget(self.overall_progress)
        
        layout.addWidget(overall_frame)
        
        # Add log section
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        
        log_label = QLabel("Processing Log:")
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_frame)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def update_file_progress(self, filename, step, percent):
        """Update the file progress."""
        # Make the label with filename more prominent
        self.file_label.setText(f"<b>Processing: {filename}</b><br>{step}")
        self.file_progress.setValue(int(percent))
    
    def update_overall_progress(self, current, total):
        """Update the overall progress."""
        percent = int((current / total) * 100) if total > 0 else 0
        self.overall_label.setText(f"Overall Progress: {current}/{total} files")
        self.overall_progress.setValue(percent)
    
    def update_overall_progress_percent(self, current, total, percent):
        """Update the overall progress with a specific percentage."""
        self.overall_label.setText(f"Overall Progress: {current+1}/{total} files ({percent}%)")
        self.overall_progress.setValue(percent)
    
    def add_log_message(self, message):
        """Add a message to the log."""
        try:
            # Get the current time
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Add timestamped message
            self.log_text.append(f"[{timestamp}] {message}")
            
            # Scroll to the bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
            
            # Force UI update - add this line
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception as e:
            print(f"Error adding log message: {e}")
    
    def update_detailed_status(self, message):
        """Update with a detailed status that doesn't affect progress bar."""
        # Update just the label without changing progress
        self.file_label.setText(f"<b>Status:</b><br>{message}")
        
        # Also add to log
        self.add_log_message(message)
    
    def on_cancel(self):
        """Handle cancel button click."""
        self.add_log_message("Cancelling... (this may take a moment)")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
        self.cancelled.emit()
        
        # If the dialog was requested to be closed via the "Close" button after completion
        if self.cancel_button.text() == "Close":
            self.close()