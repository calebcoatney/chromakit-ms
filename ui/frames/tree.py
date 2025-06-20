from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QFileDialog,
    QPushButton, QHBoxLayout, QLabel
)
from PySide6.QtCore import QDir, QModelIndex, Signal, Qt
from PySide6.QtWidgets import QFileSystemModel
import os

class FileTreeFrame(QWidget):
    """A file browser panel that can be placed on the left side of the application."""
    
    file_selected = Signal(str)  # Signal emitted when a file is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Browser")
        self.setMinimumWidth(250)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        
        # Create a button to open a directory
        self.button_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Folder")
        self.open_button.clicked.connect(self.open_directory)
        self.button_layout.addWidget(self.open_button)
        self.path_label = QLabel("No folder selected")
        self.button_layout.addWidget(self.path_label)
        self.layout.addLayout(self.button_layout)
        
        # Set up the model
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.homePath())
        self.model.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files)
        
        # Create a name filter to show directories with .D extension
        self.model.setNameFilters(["*.D"])
        self.model.setNameFilterDisables(False)  # Hide items that don't match filter
        
        # Set up the tree view
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(QDir.homePath()))
        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(20)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.sortByColumn(0, Qt.AscendingOrder)
        
        # Only show the file name column and hide others (size, type, date modified)
        for i in range(1, self.model.columnCount()):
            self.tree_view.hideColumn(i)
            
        # Connect signals
        # self.tree_view.clicked.connect(self.on_item_clicked)  # Remove single-click loading
        self.tree_view.doubleClicked.connect(self.on_item_double_clicked)  # Add double-click loading

        self.layout.addWidget(self.tree_view)
    
    def open_directory(self):
        """Open a dialog to select a directory and set it as the root path."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", QDir.homePath(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if dir_path:
            self.set_root_path(dir_path)
    
    def set_root_path(self, path):
        """Set the root path of the tree view."""
        root_index = self.model.setRootPath(path)
        self.tree_view.setRootIndex(root_index)
        self.path_label.setText(path)
    
    def on_item_double_clicked(self, index):
        """Handle item double-click in the tree view."""
        file_path = self.model.filePath(index)

        # Check if it's a .D directory or regular file
        if os.path.isdir(file_path) and file_path.endswith('.D'):
            self.file_selected.emit(file_path)
        elif os.path.isdir(file_path) and not file_path.endswith('.D'):
            # If it's a regular directory, expand/collapse it
            if self.tree_view.isExpanded(index):
                self.tree_view.collapse(index)
            else:
                self.tree_view.expand(index)
        else:
            # For non-.D files, still emit the signal
            self.file_selected.emit(file_path)
        
    def get_selected_path(self):
        """Return the path of the selected item."""
        indexes = self.tree_view.selectedIndexes()
        if indexes:
            return self.model.filePath(indexes[0])
        return None