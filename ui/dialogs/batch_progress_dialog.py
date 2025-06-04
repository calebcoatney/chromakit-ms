from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QTreeWidget, QTreeWidgetItem, QProgressBar, QTextEdit, 
                              QSplitter, QFrame, QHeaderView, QAbstractItemView,
                              QMenu, QMessageBox, QApplication)
from PySide6.QtCore import Qt, Signal, QSize, QDateTime
from PySide6.QtGui import QAction, QIcon, QColor, QBrush

class BatchProgressDialog(QDialog):
    """Dialog for displaying batch processing progress with queue management."""
    
    cancelled = Signal()
    modify_queue = Signal(list)  # Signal with updated directory list
    
    def __init__(self, parent=None, directories=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Progress")
        self.setMinimumSize(800, 600)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Store directories and their status
        self.directories = directories or []
        self.directory_status = {}  # Store status for each directory
        
        for directory in self.directories:
            self.directory_status[directory] = {
                'status': 'queued',  # 'queued', 'processing', 'completed', 'failed', 'skipped'
                'progress': 0,
                'current_file': '',
                'files_completed': 0,
                'total_files': 0,
                'error': None
            }
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create a splitter for the main panels
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        # Top panel - Queue with status
        queue_panel = QFrame()
        queue_layout = QVBoxLayout(queue_panel)
        
        # Title
        title = QLabel("Processing Queue")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        queue_layout.addWidget(title)
        
        # Queue tree (with progress bars)
        self.queue_tree = QTreeWidget()
        self.queue_tree.setHeaderLabels(["Directory", "Status", "Progress", "Details"])
        self.queue_tree.setRootIsDecorated(False)
        self.queue_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.queue_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_tree.customContextMenuRequested.connect(self.show_context_menu)
        self.queue_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.queue_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.queue_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.queue_tree.header().setSectionResizeMode(3, QHeaderView.Stretch)
        queue_layout.addWidget(self.queue_tree)
        
        # Populate the tree
        self.populate_queue_tree()
        
        # Bottom panel - Log
        log_panel = QFrame()
        log_layout = QVBoxLayout(log_panel)
        
        log_title = QLabel("Processing Log")
        log_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        log_layout.addWidget(log_title)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Add panels to splitter
        splitter.addWidget(queue_panel)
        splitter.addWidget(log_panel)
        
        # Set initial splitter sizes (60/40 split)
        splitter.setSizes([350, 250])
        
        # Overall progress
        overall_layout = QHBoxLayout()
        
        self.overall_label = QLabel("Overall Progress:")
        overall_layout.addWidget(self.overall_label)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        overall_layout.addWidget(self.overall_progress)
        
        layout.addLayout(overall_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Add directory button
        self.add_button = QPushButton("Add Directory...")
        self.add_button.clicked.connect(self.add_directory)
        button_layout.addWidget(self.add_button)
        
        # Move buttons to front/end of queue
        button_layout.addStretch()
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Add initial log message
        self.add_log_message("Batch processing started")
    
    def populate_queue_tree(self):
        """Populate the queue tree with directories."""
        self.queue_tree.clear()
        
        for directory in self.directories:
            status = self.directory_status[directory]
            
            item = QTreeWidgetItem([
                directory,
                status['status'].title(),
                "",  # Progress bar column
                ""   # Details (populated during processing)
            ])
            
            # Set background color based on status
            color = self.get_status_color(status['status'])
            if color:
                item.setBackground(1, QBrush(color))
            
            self.queue_tree.addTopLevelItem(item)
            
            # Add progress bar to the progress column
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(status['progress'])
            progress_bar.setTextVisible(True)
            self.queue_tree.setItemWidget(item, 2, progress_bar)
    
    def get_status_color(self, status):
        """Get the color associated with a status."""
        status_colors = {
            'queued': QColor(200, 200, 200),  # Light gray
            'processing': QColor(173, 216, 230),  # Light blue
            'completed': QColor(144, 238, 144),  # Light green
            'failed': QColor(255, 182, 193),  # Light red
            'skipped': QColor(245, 222, 179)   # Light orange
        }
        return status_colors.get(status)
    
    def update_directory_status(self, directory, new_status, progress=None, details=None, error=None):
        """Update the status of a directory in the queue."""
        if directory not in self.directory_status:
            return
            
        # Update status dict
        status = self.directory_status[directory]
        if new_status:
            status['status'] = new_status
        if progress is not None:
            status['progress'] = progress
        if details:
            status['current_file'] = details
        if error:
            status['error'] = error
        
        # Find the item in the tree
        for i in range(self.queue_tree.topLevelItemCount()):
            item = self.queue_tree.topLevelItem(i)
            if item.text(0) == directory:
                # Update status text
                item.setText(1, status['status'].title())
                
                # Update color
                color = self.get_status_color(status['status'])
                if color:
                    item.setBackground(1, QBrush(color))
                
                # Update progress bar
                progress_bar = self.queue_tree.itemWidget(item, 2)
                if progress_bar and progress is not None:
                    progress_bar.setValue(progress)
                
                # Update details
                if details:
                    item.setText(3, details)
                
                break
    
    def update_overall_progress(self, completed, total, percent=None):
        """Update the overall progress."""
        if percent is None:
            percent = int((completed / total) * 100) if total > 0 else 0
            
        self.overall_label.setText(f"Overall Progress: {completed}/{total} directories")
        self.overall_progress.setValue(percent)
    
    def update_file_progress(self, directory, filename, step, percent):
        """Update progress for a specific file within a directory."""
        if directory not in self.directory_status:
            return
            
        # Update status
        status = self.directory_status[directory]
        status['progress'] = percent
        status['current_file'] = filename
        details = f"{filename}: {step} ({percent}%)"
        
        # Update UI
        for i in range(self.queue_tree.topLevelItemCount()):
            item = self.queue_tree.topLevelItem(i)
            if item.text(0) == directory:
                # Update progress bar
                progress_bar = self.queue_tree.itemWidget(item, 2)
                if progress_bar:
                    # Ensure value is set and visible
                    progress_bar.setValue(percent)
                    # Force update
                    progress_bar.repaint()
                
                # Update details column
                item.setText(3, details)
                break
        
        # Process events to ensure UI updates
        QApplication.processEvents()
    
    def add_log_message(self, message):
        """Add a message to the log."""
        try:
            # Get the current time
            timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
            
            # Add timestamped message
            self.log_text.append(f"[{timestamp}] {message}")
            
            # Scroll to the bottom
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
            
            # Force UI update
            QApplication.processEvents()
        except Exception as e:
            print(f"Error adding log message: {str(e)}")
    
    def show_context_menu(self, position):
        """Show context menu for the queue tree."""
        selected_items = self.queue_tree.selectedItems()
        
        if not selected_items:
            return
            
        menu = QMenu()
        
        # Add actions
        if len(selected_items) == 1:
            # Single item actions
            item = selected_items[0]
            directory = item.text(0)
            status = self.directory_status[directory]['status']
            
            # Show details action
            details_action = QAction("Show Details", self)
            details_action.triggered.connect(lambda: self.show_directory_details(directory))
            menu.addAction(details_action)
            
            # Actions based on status
            if status == 'queued':
                skip_action = QAction("Skip", self)
                skip_action.triggered.connect(lambda: self.skip_directory(directory))
                menu.addAction(skip_action)
                
            elif status == 'failed':
                retry_action = QAction("Retry", self)
                retry_action.triggered.connect(lambda: self.retry_directory(directory))
                menu.addAction(retry_action)
                
            menu.addSeparator()
        
        # Multi-selection actions
        remove_action = QAction("Remove from Queue", self)
        remove_action.triggered.connect(self.remove_selected)
        menu.addAction(remove_action)
        
        # Show the menu
        menu.exec_(self.queue_tree.mapToGlobal(position))
    
    def show_directory_details(self, directory):
        """Show details for a directory."""
        status = self.directory_status[directory]
        
        details = f"Directory: {directory}\n"
        details += f"Status: {status['status'].title()}\n"
        details += f"Progress: {status['progress']}%\n"
        
        if status['current_file']:
            details += f"Current File: {status['current_file']}\n"
            
        if status['files_completed'] > 0:
            details += f"Files Completed: {status['files_completed']}/{status['total_files']}\n"
            
        if status['error']:
            details += f"\nError: {status['error']}\n"
        
        QMessageBox.information(self, "Directory Details", details)
    
    def skip_directory(self, directory):
        """Mark a directory to be skipped."""
        self.update_directory_status(directory, 'skipped', 0, "Skipped by user")
        self.add_log_message(f"Directory {directory} marked to be skipped")
        
        # Update the directory list
        self.update_directory_list()
    
    def retry_directory(self, directory):
        """Mark a failed directory for retry."""
        self.update_directory_status(directory, 'queued', 0, "Queued for retry")
        self.add_log_message(f"Directory {directory} queued for retry")
        
        # Update the directory list
        self.update_directory_list()
    
    def remove_selected(self):
        """Remove selected directories from the queue."""
        selected_items = self.queue_tree.selectedItems()
        
        if not selected_items:
            return
            
        # Confirm removal
        result = QMessageBox.question(
            self,
            "Remove Directories",
            f"Remove {len(selected_items)} selected {'directory' if len(selected_items) == 1 else 'directories'} from the queue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result != QMessageBox.Yes:
            return
            
        # Get the directories to remove
        to_remove = [item.text(0) for item in selected_items]
        
        # Update the directories list
        self.directories = [d for d in self.directories if d not in to_remove]
        
        # Remove from status dict
        for directory in to_remove:
            if directory in self.directory_status:
                del self.directory_status[directory]
        
        # Repopulate the tree
        self.populate_queue_tree()
        
        # Update log
        self.add_log_message(f"Removed {len(to_remove)} directories from the queue")
        
        # Update the directory list in the worker
        self.update_directory_list()
    
    def add_directory(self):
        """Add a new directory to the queue."""
        from PySide6.QtWidgets import QFileDialog
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", "", QFileDialog.ShowDirsOnly
        )
        
        if directory:
            # Check if already in queue
            if directory in self.directories:
                QMessageBox.warning(
                    self,
                    "Directory Already in Queue",
                    f"The directory {directory} is already in the queue."
                )
                return
                
            # Check if it's a valid GC-MS data directory
            import os
            valid_dir = False
            
            for item in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, item)) and item.endswith('.D'):
                    valid_dir = True
                    break
            
            if valid_dir:
                # Add to our lists
                self.directories.append(directory)
                self.directory_status[directory] = {
                    'status': 'queued',
                    'progress': 0,
                    'current_file': '',
                    'files_completed': 0,
                    'total_files': 0,
                    'error': None
                }
                
                # Update the tree
                self.populate_queue_tree()
                
                # Update log
                self.add_log_message(f"Added directory {directory} to the queue")
                
                # Update the directory list in the worker
                self.update_directory_list()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Directory",
                    "The selected directory does not contain any .D files."
                )
    
    def update_directory_list(self):
        """Emit a signal with the updated directory list."""
        # Create a list with only 'queued' directories
        queued_dirs = [d for d in self.directories if self.directory_status[d]['status'] == 'queued']
        
        # Emit the signal
        self.modify_queue.emit(queued_dirs)
    
    def on_cancel(self):
        """Handle cancel button click."""
        result = QMessageBox.question(
            self,
            "Cancel Processing",
            "Are you sure you want to cancel the batch processing?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            self.add_log_message("Cancelling batch processing... (this may take a moment)")
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Cancelling...")
            self.cancelled.emit()
            
    def complete_processing(self):
        """Mark processing as complete."""
        self.cancel_button.setText("Close")
        self.cancel_button.setEnabled(True)
        self.add_button.setEnabled(False)
        
        # Count successes and failures
        successes = sum(1 for status in self.directory_status.values() if status['status'] == 'completed')
        failures = sum(1 for status in self.directory_status.values() if status['status'] == 'failed')
        skipped = sum(1 for status in self.directory_status.values() if status['status'] == 'skipped')
        
        # Show summary
        self.add_log_message(f"Batch processing complete: {successes} succeeded, {failures} failed, {skipped} skipped")
        
        # Show dialog
        QMessageBox.information(
            self,
            "Batch Processing Complete",
            f"Processing completed:\n\n{successes} directories succeeded\n{failures} directories failed\n{skipped} directories skipped"
        )