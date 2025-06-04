from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QFrame
from PySide6.QtCore import Signal

class ButtonFrame(QWidget):
    """Frame containing control buttons for navigation and automation."""
    
    # Define signals
    export_clicked = Signal()
    back_clicked = Signal()
    next_clicked = Signal()
    integrate_clicked = Signal()
    automation_clicked = Signal()  # Changed from automation_toggled
    batch_search_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create first row frame for main buttons
        self.row1_frame = QFrame()
        self.row1_layout = QHBoxLayout(self.row1_frame)
        self.row1_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create buttons for first row
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)  # Disabled by default
        self.export_button.clicked.connect(self.export_clicked)
        
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self.back_clicked)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_clicked)
        
        self.integrate_button = QPushButton("Integrate")
        self.integrate_button.clicked.connect(self.integrate_clicked)
        
        # Add buttons to first row
        self.row1_layout.addWidget(self.export_button)
        self.row1_layout.addWidget(self.back_button)
        self.row1_layout.addWidget(self.next_button)
        self.row1_layout.addWidget(self.integrate_button)
        self.row1_layout.addStretch(1)  # Add stretch to push buttons to the left
        
        # Create second row frame for automation controls
        self.row2_frame = QFrame()
        self.row2_layout = QHBoxLayout(self.row2_frame)
        self.row2_layout.setContentsMargins(0, 0, 0, 0)
        
        # Replace checkbox with button (CHANGE HERE)
        self.automation_button = QPushButton("Batch Process All Files")
        self.automation_button.setToolTip("Process and perform MS search on all .D files in the directory")
        self.automation_button.clicked.connect(self.automation_clicked.emit)
        self.row2_layout.addWidget(self.automation_button)
        
        # Add batch search button
        self.batch_search_button = QPushButton("MS Search All")
        self.batch_search_button.setToolTip("Search MS library for all integrated peaks")
        self.batch_search_button.clicked.connect(self.batch_search_clicked.emit)
        self.batch_search_button.setEnabled(False)  # Disabled by default
        self.row2_layout.addWidget(self.batch_search_button)
        
        # Add rows to main layout
        self.layout.addWidget(self.row1_frame)
        self.layout.addWidget(self.row2_frame)
    
    def enable_export(self, enabled=True):
        """Enable or disable the export button."""
        self.export_button.setEnabled(enabled)