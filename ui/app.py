from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStatusBar, QVBoxLayout, 
    QMessageBox, QTableWidget, QTableWidgetItem, QDialog, 
    QPushButton, QFileDialog, QHeaderView, QProgressDialog, 
    QApplication, QCheckBox, QLabel, QComboBox, QDialogButtonBox,
    QTabWidget  # Add QTabWidget to imports
)
from PySide6.QtCore import Qt, Slot, QThreadPool, QTimer
from PySide6.QtGui import QIcon, QBrush, QColor
from ui.frames.tree import FileTreeFrame
from ui.frames.plot import PlotFrame
from ui.frames.parameters import ParametersFrame
from ui.frames.ms import MSFrame
from ui.frames.buttons import ButtonFrame
from ui.dialogs.automation_dialog import AutomationDialog
from ui.dialogs.export_settings_dialog import ExportSettingsDialog
from logic.automation_worker import AutomationWorker
from logic.processor import ChromatogramProcessor
from logic.batch_search import BatchSearchWorker
import numpy as np
import pandas as pd
import rainbow as rb
import sys
import os
import json
import datetime
import time
import traceback
import matplotlib

# Import the data handler directly as a required dependency
from logic.data_handler import DataHandler

class ChromaKitApp(QMainWindow):
    """Main application window for ChromaKit."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChromaKit")
        self.resize(1600, 800)
        self.setWindowIcon(QIcon(r'resources\file.ico'))
        
        # Initialize processor
        self.processor = ChromatogramProcessor()
        
        # Add flag to track if real data is loaded
        self.real_data_loaded = False
        
        # Initialize data handler
        self.data_handler = DataHandler()
        
        # Initialize export manager
        from logic.export_manager import ExportManager
        self.export_manager = ExportManager(self)
        
        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create and add file tree frame to the left side
        self.file_tree = FileTreeFrame()
        self.main_layout.addWidget(self.file_tree)
        
        # Create a container for plot and buttons
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create and add plot frame to the container
        self.plot_frame = PlotFrame()
        self.plot_layout.addWidget(self.plot_frame)
        
        # Create and add button frame below the plot
        self.button_frame = ButtonFrame()
        self.plot_layout.addWidget(self.button_frame)
        
        # Add the plot container to the main layout
        self.main_layout.addWidget(self.plot_container, 1)  # stretch factor = 1
        
        # Create and add parameters frame to the right side
        self.parameters_frame = ParametersFrame()
        
        # Create and add MS frame to the far right
        self.ms_frame = MSFrame()
        
        # Create a tab widget for the right-side panels
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.parameters_frame, "Parameters")
        self.right_tabs.addTab(self.ms_frame, "Mass Spectrometry")
        self.right_tabs.setMinimumWidth(350) # Ensure the tab widget has a reasonable minimum width
        
        # Add the tab widget to the main layout
        self.main_layout.addWidget(self.right_tabs)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Connect signals - only connecting those signals that currently exist
        self.file_tree.file_selected.connect(self.on_file_selected)
        self.plot_frame.point_selected.connect(self.on_point_selected)
        self.parameters_frame.parameters_changed.connect(self.on_parameters_changed)
        
        # Connect button frame signals
        self.button_frame.export_clicked.connect(self.on_export)
        self.button_frame.back_clicked.connect(self.on_previous_sample)
        self.button_frame.next_clicked.connect(self.on_next_sample)
        self.button_frame.integrate_clicked.connect(self.on_integrate)
        self.button_frame.automation_clicked.connect(self.on_automation_clicked)
        self.button_frame.batch_search_clicked.connect(self.run_batch_ms_search)
        
        # Add this new connection for MS spectrum viewing
        self.plot_frame.ms_spectrum_requested.connect(self.on_ms_spectrum_requested)

        # Add this connection for peak-specific spectrum extraction
        self.plot_frame.peak_spectrum_requested.connect(self.on_peak_spectrum_requested)
        
        # Add connection for MS search completion (NEW)
        if hasattr(self.ms_frame, 'search_completed'):
            self.ms_frame.search_completed.connect(self.on_ms_search_completed)
        
        # Add this new connection for MS search request
        self.plot_frame.ms_search_requested.connect(self.on_ms_search_requested)
        
        # Add this new connection for edit assignment requests
        self.plot_frame.edit_assignment_requested.connect(self.on_edit_assignment_requested)

        # Connect MS baseline correction button
        self.parameters_frame.ms_baseline_clicked.connect(self.perform_ms_baseline_correction)

        # Process menu
        process_menu = self.menuBar().addMenu("Process")

        # Add batch processing option
        batch_action = process_menu.addAction("Batch Process Directories...")
        batch_action.triggered.connect(self.show_batch_queue_dialog)
        
        # Show a simple welcome message instead of generating dummy data
        self.status_bar.showMessage("Welcome to ChromaKit. Please load a data file to begin.")
        
        # Add a Settings menu after Process menu
        settings_menu = self.menuBar().addMenu("Settings")
        
        # Add detector selection option
        detector_action = settings_menu.addAction("Select Detector Channel...")
        detector_action.triggered.connect(self.show_detector_selection_dialog)
        
        # Add export settings option
        export_settings_action = settings_menu.addAction("Export Settings...")
        export_settings_action.triggered.connect(self.show_export_settings_dialog)
        
        # Theme state
        self.current_theme = 'light'
        self.matplotlib_theme_colors = None
        self.apply_stylesheet(self.current_theme)

        # Add theme toggle to Settings menu
        self.theme_action = settings_menu.addAction("Toggle Dark/Light Mode")
        self.theme_action.triggered.connect(self.toggle_theme)

    def apply_stylesheet(self, theme):
        """Apply the QSS stylesheet with the selected theme."""
        qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                qss = f.read()
            
            # Set the theme property on the main window first
            self.setProperty("theme", theme)
            
            # Apply stylesheet to application
            self.setStyleSheet(qss)
            
            # Process special widgets first (QTreeView, etc.)
            self._apply_theme_to_special_widgets(theme)
            
            # Now update all regular widgets
            self._apply_theme_to_regular_widgets(theme)
            
            # Apply matplotlib theme and update any existing plots
            self.set_matplotlib_theme(theme)
            
            # Update the main window
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
        else:
            self.setStyleSheet("")

    def _apply_theme_to_special_widgets(self, theme):
        """Apply theme to special widget types that need custom handling."""
        from PySide6.QtWidgets import QTreeView, QTreeWidget, QHeaderView, QAbstractItemView, QTableView

        # Process tree views and similar widgets separately with special handling
        special_widgets = []
        for widget_type in (QTreeView, QTreeWidget, QHeaderView, QAbstractItemView, QTableView):
            special_widgets.extend(self.findChildren(widget_type))

        for widget in special_widgets:
            try:
                # Set theme property explicitly
                widget.setProperty("theme", theme)

                # Force style refresh - this is critical for these widget types
                widget.style().unpolish(widget)
                widget.style().polish(widget)

                # Use repaint instead of update to avoid argument errors
                widget.repaint()
                # For views, also update the viewport
                if hasattr(widget, "viewport"):
                    widget.viewport().update()
            except Exception as e:
                print(f"Error updating special widget {widget.__class__.__name__}: {e}")

    def _apply_theme_to_regular_widgets(self, theme):
        """Apply theme to all other regular widgets."""
        from PySide6.QtWidgets import QWidget, QTreeView, QTreeWidget, QHeaderView, QAbstractItemView, QTableView
        
        # Skip special widget types which were handled separately
        special_types = (QTreeView, QTreeWidget, QHeaderView, QAbstractItemView, QTableView)
        
        # Get all widgets
        all_widgets = self.findChildren(QWidget)
        
        for widget in all_widgets:
            # Skip special widgets that were already processed
            if any(isinstance(widget, special_type) for special_type in special_types):
                continue
                
            try:
                # Set theme property
                widget.setProperty("theme", theme)
                
                # Force style refresh
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
            except Exception as e:
                print(f"Error updating widget {widget.__class__.__name__}: {e}")

    def _apply_theme_to_regular_widgets(self, theme):
        """Apply theme to all other regular widgets."""
        from PySide6.QtWidgets import QWidget, QTreeView, QTreeWidget, QHeaderView, QAbstractItemView, QTableView
        
        # Skip special widget types which were handled separately
        special_types = (QTreeView, QTreeWidget, QHeaderView, QAbstractItemView, QTableView)
        
        # Get all widgets
        all_widgets = self.findChildren(QWidget)
        
        for widget in all_widgets:
            # Skip special widgets that were already processed
            if any(isinstance(widget, special_type) for special_type in special_types):
                continue
                
            try:
                # Set theme property
                widget.setProperty("theme", theme)
                
                # Force style refresh
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
            except Exception as e:
                print(f"Error updating widget {widget.__class__.__name__}: {e}")

    def set_matplotlib_theme(self, theme):
        """Set matplotlib rcParams based on the current theme."""
        if theme == "dark":
            colors = {
                'background': '#23272e',
                'axes': '#23272e',
                'edge': '#f3f3f3',
                'label': '#f3f3f3',
                'grid': '#555555',
                'ticks': '#f3f3f3',
                'line1': '#4cc2ff',
                'line2': '#ff7f50',
                'line3': '#7fff7f',
                'text': '#f3f3f3'
            }
        else:
            colors = {
                'background': '#f6f7fa',
                'axes': '#f6f7fa', 
                'edge': '#2E2E2E',
                'label': '#2E2E2E',
                'grid': '#888888',
                'ticks': '#2E2E2E',
                'line1': '#2B6A99',
                'line2': '#8B3C41',
                'line3': '#3C6E47',
                'text': '#2E2E2E'
            }
        
        # Store current theme colors for reference in plot creation
        self.matplotlib_theme_colors = colors
        
        # Update matplotlib rcParams - more comprehensive settings
        matplotlib.rcParams.update({
            'axes.facecolor': colors['axes'],
            'figure.facecolor': colors['background'],
            'axes.edgecolor': colors['edge'],
            'axes.labelcolor': colors['label'],
            'xtick.color': colors['ticks'],
            'ytick.color': colors['ticks'],
            'grid.color': colors['grid'],
            'text.color': colors['text'],
            'axes.titlecolor': colors['text'],  # Add this explicitly
            'figure.titlesize': 14,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,   # Add explicit tick label sizes
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.loc': 'best',
            'legend.edgecolor': colors['edge'],
            'legend.facecolor': colors['background'],
            'legend.labelcolor': colors['label'],
            'figure.autolayout': True,
            'figure.dpi': 100,
            'savefig.dpi': 600,
            'savefig.transparent': False,
            'axes.spines.top': True,    # Ensure spines are visible
            'axes.spines.right': True,
            'axes.spines.bottom': True,
            'axes.spines.left': True,
        })
        
        # Update existing plots
        self._update_existing_plots(colors)

    def _update_existing_plots(self, colors):
        """Update any existing matplotlib plots with the new theme."""
        # This method now simply calls the apply_theme method on each plot frame.
        if hasattr(self, 'plot_frame') and hasattr(self.plot_frame, 'apply_theme'):
            self.plot_frame.apply_theme()
        
        if hasattr(self, 'ms_frame') and hasattr(self.ms_frame, 'apply_theme'):
            self.ms_frame.apply_theme()
    
    def refresh_matplotlib_theme(self):
        """Force refresh of matplotlib theme on all plot elements."""
        # This method is now simpler and just calls _update_existing_plots
        if hasattr(self, 'matplotlib_theme_colors'):
            self._update_existing_plots(self.matplotlib_theme_colors)
    
    def toggle_theme(self):
        """Toggle between dark and light mode."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_stylesheet(self.current_theme)
        
        # Special handling for file tree after theme switch
        if hasattr(self, 'file_tree'):
            self._refresh_tree_view(self.file_tree)
        
        # Add explicit matplotlib theme refresh
        self.refresh_matplotlib_theme()
    
        self.status_bar.showMessage(f"Switched to {self.current_theme.capitalize()} Mode")

    def _refresh_tree_view(self, tree_widget):
        """Special refresh for tree views after theme change."""
        if hasattr(tree_widget, 'tree'):
            tree = tree_widget.tree
            tree.setProperty("theme", self.current_theme)
            tree.style().unpolish(tree)
            tree.style().polish(tree)
            tree.viewport().update()
            
            # Sometimes a full reset helps
            current_model = tree.model()
            tree.setModel(None)
            tree.setModel(current_model)

    def generate_sample_data(self):
        """Generate sample chromatogram data for testing."""
        # Skip if real data is already loaded
        if self.real_data_loaded:
            return
            
        # Create sample x and y data
        raw_x = np.linspace(0, 10, 1000)
        raw_y = np.sin(raw_x) * np.exp(-raw_x/5) * 100 + np.random.normal(0, 0.1, size=len(raw_x))
        
        # Add baseline
        baseline = 10 * np.sin(raw_x/3) + 5 * np.exp(-raw_x/10) + 2
        raw_y = raw_y + baseline
        
        # Interpolate to standard length for consistency with real data
        from logic.interpolation import interpolate_arrays
        interp_x, interp_y = interpolate_arrays(raw_x, raw_y, target_length=10000)
        
        # Store both original and interpolated data
        self.original_x = raw_x
        self.original_y = raw_y
        self.current_x = interp_x
        self.current_y = interp_y
        
        # Process and display with current parameters
        self.process_and_display(self.current_x, self.current_y)
        
        # Add sample MS data
        sample_results = [
            ("Compound A", 0.95),
            ("Compound B", 0.82),
            ("Compound C", 0.78)
        ]
        self.ms_frame.update_ms_results(sample_results)
        
        self.status_bar.showMessage("Generated sample data")
        
    @Slot(str)
    def on_file_selected(self, file_path, batch_mode=False):
        """Handle file selection from the tree view.
        
        Args:
            file_path: Path to the .D directory to load
            batch_mode: If True, suppress error dialogs (for batch processing)
        """
        try:
            # Clear stored peak data to prevent ghost peaks
            self.plot_frame.clear_peak_data()
            
            # Clear any stored peak data in the app
            if hasattr(self, 'integrated_peaks'):
                delattr(self, 'integrated_peaks')
            if hasattr(self, 'integration_results'):
                delattr(self, 'integration_results')
            
            # Disable batch search button until new peaks are integrated
            self.button_frame.batch_search_button.setEnabled(False)
            
            # Reset current data to prevent length mismatches
            self.current_x = None
            self.current_y = None
            
            # Normalize path to ensure consistent path formatting
            file_path = os.path.normpath(os.path.abspath(file_path))
            
            # Set current directory path
            self.current_directory_path = file_path
            
            # Show loading message
            self.status_bar.showMessage(f"Loading: {file_path}")
            
            # Check if it's a .D directory
            if not file_path.endswith('.D'):
                self.status_bar.showMessage(f"Not a valid data directory: {file_path}")
                return
                
            # Load the data
            data = self.data_handler.load_data_directory(file_path)
            
            # Process chromatogram data if available
            if 'chromatogram' in data and len(data['chromatogram']['x']) > 0:
                # Get raw data
                raw_x = np.array(data['chromatogram']['x'])
                raw_y = np.array(data['chromatogram']['y'])
                
                # Interpolate to standard length at load time
                from logic.interpolation import interpolate_arrays
                interp_x, interp_y = interpolate_arrays(raw_x, raw_y, target_length=10000)
                
                # Store both original and interpolated data
                self.original_x = raw_x
                self.original_y = raw_y
                self.current_x = interp_x
                self.current_y = interp_y
                
                original_length = len(raw_x)
                if original_length > 10000:
                    self.status_bar.showMessage(f"Interpolated {original_length} points to 10,000 for processing...")
                
                # Process with current parameters and display
                self.process_and_display(self.current_x, self.current_y)
                
                # Set flag that real data is loaded
                self.real_data_loaded = True
            
            # Check if MS data is available
            has_ms_data = self.data_handler.has_ms_data
            
            # Show or hide TIC plot based on MS data availability
            try:
                self.plot_frame.set_tic_visible(has_ms_data)
            except Exception as tic_error:
                print(f"Error setting TIC visibility: {type(tic_error).__name__}: {tic_error}")
                # Try to continue without TIC functionality
                has_ms_data = False
            
            # Enable or disable MS tab
            ms_tab_index = self.right_tabs.indexOf(self.ms_frame)
            self.right_tabs.setTabEnabled(ms_tab_index, has_ms_data)
            
            # Plot TIC only if MS data is available
            if has_ms_data and 'tic' in data and len(data['tic']['x']) > 0:
                try:
                    self.plot_frame.plot_tic(data['tic']['x'], data['tic']['y'], new_file=True)
                except Exception as plot_error:
                    # If TIC plotting fails, just log it and continue
                    print(f"Warning: Failed to plot TIC data: {plot_error}")
                    has_ms_data = False  # Treat as no MS data if plotting fails
            else:
                # Explicitly clear the TIC plot if no MS data
                try:
                    if hasattr(self.plot_frame, 'clear_tic'):
                        self.plot_frame.clear_tic()
                except Exception as clear_error:
                    print(f"Warning: Failed to clear TIC plot: {clear_error}")
                    # Continue anyway since this is not critical
            
            # Update status bar with success message
            sample_name = os.path.basename(file_path)
            
            # Show navigation info
            num_dirs = len(self.data_handler.available_directories)
            curr_idx = self.data_handler.current_index + 1  # 1-indexed for display
            
            # Add information about MS data availability to the status message
            ms_status = "with" if has_ms_data else "without"
            self.status_bar.showMessage(f"Loaded: {sample_name} ({curr_idx}/{num_dirs}) - {ms_status} MS data")
            
            # Enable export button
            self.button_frame.enable_export(True)

            # Process and display the chromatogram data
            self.process_and_display(self.current_x, self.current_y, new_file=True)
            
        except Exception as e:
            # Handle any errors during loading
            # Debug: print type and details of the exception
            print(f"DEBUG: Exception type: {type(e)}")
            print(f"DEBUG: Exception repr: {repr(e)}")
            print(f"DEBUG: Exception str: {str(e)}")
            
            # Check if this is actually an exception or something else
            if not isinstance(e, BaseException):
                print(f"WARNING: Caught object that is not an exception: {type(e)}")
                error_msg = f"Error loading file: Unexpected object caught: {str(e)}"
            else:
                error_msg = f"Error loading file: {str(e)}"
            
            self.status_bar.showMessage(error_msg)
            
            # Only show error dialog if not in batch mode
            if not batch_mode:
                QMessageBox.critical(self, "Error", f"Error loading file:\n{str(e)}")
            else:
                # In batch mode, just log the error without showing dialog
                print(f"Batch processing error: {error_msg}")
    
    @Slot(float)
    def on_point_selected(self, x_value):
        """Handle point selection from the plot."""
        self.status_bar.showMessage(f"Selected point at x={x_value}")
        # Update the RT entry in the MS frame
        self.ms_frame.rt_entry.setText(str(x_value))
        
    @Slot(float, float, float)
    def on_view_spectrum(self, rt, subtract_rt, mz_shift):
        """Handle view spectrum request."""
        self.status_bar.showMessage(f"Viewing spectrum at RT={rt}")
        
        # Generate example mass spectrum for demo purposes
        mz = np.arange(10, 150, 1)
        intensity = np.zeros_like(mz)
        
        # Create some random peaks for demonstration
        for i in range(10):
            peak_pos = np.random.randint(20, 140)
            intensity[peak_pos-10:peak_pos+10] = np.random.random() * np.exp(-(mz[peak_pos-10:peak_pos+10]-peak_pos)**2/10)
        
        # Plot the spectrum
        self.ms_frame.plot_mass_spectrum(mz, intensity, f"Sample Spectrum at RT: {rt}")
    
    @Slot(float)
    def on_ms_spectrum_requested(self, retention_time):
        """Handle request to view MS spectrum at specified retention time."""
        # First check if MS data is available
        if not hasattr(self.data_handler, 'has_ms_data') or not self.data_handler.has_ms_data:
            self.status_bar.showMessage("No MS data available for this file")
            return
        
        self.status_bar.showMessage(f"Viewing MS spectrum at RT={retention_time:.3f}")
        
        # Check if we have MS data available
        if hasattr(self, 'data_handler'):
            try:
                spectrum = self.data_handler.extract_spectrum_at_rt(retention_time)
                if spectrum and 'mz' in spectrum and 'intensities' in spectrum:
                    self.ms_frame.plot_mass_spectrum(
                        spectrum['mz'], 
                        spectrum['intensities'],
                        f"RT: {retention_time:.3f} min"
                    )
                else:
                    self.status_bar.showMessage(f"Could not extract spectrum at RT={retention_time:.3f}")
            except Exception as e:
                self.status_bar.showMessage(f"Error extracting spectrum: {str(e)}")
        else:
            # No real data available, display simulated spectrum
            self.ms_frame.set_extract_spectrum_function(None)  # Ensure no function is set
            self.ms_frame.view_spectrum_at_rt(retention_time)
    
    def on_peak_spectrum_requested(self, peak_index):
        """Handle request to extract MS spectrum for a specific peak."""
        # Check if MS data exists for this sample
        if not hasattr(self.data_handler, 'has_ms_data') or not self.data_handler.has_ms_data:
            self.status_bar.showMessage("No MS data available for this file")
            return
        # Check if we have integrated peaks
        if not hasattr(self, 'integrated_peaks') or peak_index >= len(self.integrated_peaks):
            self.status_bar.showMessage("No peak data available at that position")
            return
        
        # Get the peak
        peak = self.integrated_peaks[peak_index]
        
        # Show info in status bar
        self.status_bar.showMessage(f"Extracting mass spectrum for peak {peak.peak_number} at RT={peak.retention_time:.3f}")
        
        # Check if we have the MS toolkit
        if not hasattr(self.ms_frame, 'ms_toolkit') or not self.ms_frame.ms_toolkit:
            self.status_bar.showMessage("MS toolkit not available")
            return
            
        if not hasattr(self.data_handler, 'current_directory_path') or not self.data_handler.current_directory_path:
            self.status_bar.showMessage("No data directory selected")
            return
        
        # Get current MS search options
        if hasattr(self.ms_frame, 'search_options'):
            options = self.ms_frame.search_options
        else:
            # Default options if not configured
            options = {
                'extraction_method': 'apex',
                'range_points': 5,
                'tic_weight': True,
                'subtract_background': True,  # renamed from subtract_enabled for consistency
                'subtraction_method': 'min_tic',
                'subtract_weight': 0.1,
                'intensity_threshold': 0.01,
                'midpoint_width_percent': 20
            }
        
        try:
            # Get the mz_shift from UI and set it on the toolkit
            try:
                mz_shift = int(self.ms_frame.mz_shift_entry.text() or 0)
                self.ms_frame.ms_toolkit.mz_shift = mz_shift
                print(f"Applied m/z shift of {mz_shift} to toolkit")
            except (ValueError, AttributeError) as e:
                print(f"Error setting m/z shift: {str(e)}")
                
            # Debug print showing extraction parameters
            print(f"Extracting spectrum for peak {peak.peak_number} at RT={peak.retention_time:.3f}")
            print(f"Method: {options.get('extraction_method', 'apex')}, Range points: {options.get('range_points', 5)}")
            print(f"Start time: {peak.start_time:.3f}, End time: {peak.end_time:.3f}")
            
            # Use the data handler to extract the spectrum
            spectrum = self.data_handler.extract_spectrum_for_peak(
                peak,
                {'extraction_method': 'apex', 'debug': False}
            )
            
            if spectrum and 'mz' in spectrum and 'intensities' in spectrum:
                # Plot the spectrum in the MS frame
                self.ms_frame.plot_mass_spectrum(
                    spectrum['mz'], 
                    spectrum['intensities'],
                    f"Peak {peak.peak_number} (RT={peak.retention_time:.3f})"
                )
                
                # Also set current spectrum for possible searching with RT
                self.ms_frame.set_current_spectrum(
                    spectrum['mz'], 
                    spectrum['intensities'],
                    f"Peak {peak.peak_number} (RT={peak.retention_time:.3f})",
                    rt=peak.retention_time
                )
                
                # Explicitly update RT entry as well
                self.ms_frame.rt_entry.setText(f"{peak.retention_time:.3f}")
                
                # Update status bar
                self.status_bar.showMessage(
                    f"Extracted mass spectrum for peak {peak.peak_number} at RT={peak.retention_time:.3f}"
                )
            else:
                self.status_bar.showMessage(f"Could not extract spectrum for peak {peak.peak_number}")
                
        except Exception as e:
            self.status_bar.showMessage(f"Error extracting spectrum: {str(e)}")
            print(f"Error extracting spectrum: {str(e)}")
    
    @Slot(int)
    def on_ms_search_requested(self, peak_index):
        """Handle request to search library for specific peak."""
        # First, make sure the peak spectrum is displayed
        self.on_peak_spectrum_requested(peak_index)
        
        # Then trigger the search
        self.ms_frame._search_current_spectrum()
    
    @Slot()
    def on_ms_search_completed(self):
        """Handle MS search completion."""
        self.status_bar.showMessage("MS library search complete")
    
    def on_edit_assignment_requested(self, peak_index):
        """Handle request to edit peak compound assignment."""
        # Check if we have the required components
        if not hasattr(self, 'integrated_peaks') or peak_index >= len(self.integrated_peaks):
            self.status_bar.showMessage("No peak available at this position")
            return
        
        # Get the peak
        peak = self.integrated_peaks[peak_index]
        
        # Get library compounds if available
        library_compounds = []
        if hasattr(self.ms_frame, 'ms_toolkit') and self.ms_frame.ms_toolkit:
            if hasattr(self.ms_frame.ms_toolkit, 'library') and self.ms_frame.ms_toolkit.library:
                library_compounds = list(self.ms_frame.ms_toolkit.library.keys())
        
        # Get the spectrum for the peak if available
        spectrum = None
        if hasattr(self, 'data_handler'):
            try:
                # Get current MS search options
                if hasattr(self.ms_frame, 'search_options'):
                    options = self.ms_frame.search_options
                else:
                    options = {}
                        
                spectrum = self.data_handler.extract_spectrum_for_peak(peak, options)
            except Exception as e:
                print(f"Error extracting peak spectrum for assignment: {e}")
        
        # Get list of processed files in the directory
        all_files = []
        if hasattr(self.data_handler, 'get_processed_files'):
            try:
                all_files = self.data_handler.get_processed_files()
            except Exception as e:
                print(f"Error getting processed files: {e}")
        
        # Import the dialog
        from ui.dialogs.edit_assignment_dialog import EditAssignmentDialog
        
        # Create and show the dialog
        dialog = EditAssignmentDialog(
            self, 
            peak, 
            library_compounds, 
            self.data_handler.current_directory_path if hasattr(self.data_handler, 'current_directory_path') else None,
            spectrum,
            all_files
        )
        
        # Store reference to the dialog for later use
        self.edit_assignment_dialog = dialog
        
        # Connect the cross-file application signal
        dialog.apply_to_files_requested.connect(self.apply_assignment_to_files)
        
        result = dialog.exec()
        
        # If accepted, update the peak assignment
        if result == QDialog.Accepted:
            new_compound = dialog.get_selected_compound()
            if new_compound:
                # Update peak with new assignment
                old_compound = peak.compound_id if hasattr(peak, 'compound_id') else "Unknown"
                
                # Update all relevant fields
                peak.compound_id = new_compound
                if hasattr(peak, 'Compound_ID'):  # Update notebook-style field if present
                    peak.Compound_ID = new_compound
                    
                # Try to get CAS number if available
                casno = None
                if hasattr(self.ms_frame, 'ms_toolkit') and self.ms_frame.ms_toolkit:
                    if hasattr(self.ms_frame.ms_toolkit, 'library') and new_compound in self.ms_frame.ms_toolkit.library:
                        compound = self.ms_frame.ms_toolkit.library[new_compound]
                        if hasattr(compound, 'casno'):
                            from logic.batch_search import format_casno
                            casno = format_casno(compound.casno)
                            
                # Update CAS number if available
                if casno:
                    peak.casno = casno
                
                # Manual assignments don't have a match score
                peak.Qual = None
                
                # Update the plot to show the new assignment
                self.plot_frame.update_annotations()
                
                # Update status bar
                self.status_bar.showMessage(f"Updated assignment from '{old_compound}' to '{new_compound}'")
                
                # Update any results views
                self.update_results_view()
                
                # Update integration results file
                if hasattr(self, 'integration_results'):
                    self._auto_update_json_with_assignment(peak)
                    
                # Add to the override database
                self._add_to_override_database(peak.retention_time, new_compound, spectrum)

    @Slot(str, float, float, object)
    def apply_assignment_to_files(self, compound_name, retention_time, tolerance, spectrum):
        """Apply a compound assignment to peaks in other files based on RT and spectral similarity."""
        if not hasattr(self.data_handler, 'get_processed_files'):
            self.status_bar.showMessage("Cannot apply to other files - data handler not available")
            return
        
        # Get list of processed files
        try:
            files = self.data_handler.get_processed_files()
        except Exception as e:
            self.status_bar.showMessage(f"Error getting file list: {str(e)}")
            return
        
        # Exclude current file
        current_dir = self.data_handler.current_directory_path
        files = [f for f in files if f != current_dir]
        
        # If no other files, show message and return
        if not files:
            self.status_bar.showMessage("No other processed files available")
            QMessageBox.information(self, "No Other Files", "No other processed files were found in the directory.")
            return
        
        # Import similarity function
        try:
            from ms_toolkit.similarity import dot_product_similarity
        except ImportError:
            # Fallback implementation if ms_toolkit not available
            def dot_product_similarity(spectrum1, spectrum2, max_mz=1000, unmatched_method="keep_all"):
                """Simple cosine similarity implementation."""
                # Convert spectra to vectors
                max_mz = int(max(max(m for m, i in spectrum1), max(m for m, i in spectrum2))) + 1
                vec1 = np.zeros(max_mz)
                vec2 = np.zeros(max_mz)
                
                for m, i in spectrum1:
                    if m < max_mz:
                        vec1[int(m)] = i
                
                for m, i in spectrum2:
                    if m < max_mz:
                        vec2[int(m)] = i
                
                # Calculate cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0
                
                return dot_product / (norm1 * norm2)
        
        # Create a progress dialog
        progress = QProgressDialog("Applying assignment to other files...", "Cancel", 0, len(files), self)
        progress.setWindowTitle("Cross-File Assignment")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Process each file
        matches_found = 0
        files_processed = 0
        
        for i, file_path in enumerate(files):
            # Check for cancellation
            if progress.wasCanceled():
                break
            
            # Update progress
            progress.setValue(i)
            progress.setLabelText(f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
            QApplication.processEvents()
            
            try:
                # Load integration results from the file
                integration_json_path = os.path.join(file_path, "integration_results.json")
                if not os.path.exists(integration_json_path):
                    continue
                
                # Load the integration results
                with open(integration_json_path, 'r') as f:
                    import json
                    integration_data = json.load(f)
                
                peaks = integration_data.get('peaks', [])
                if not peaks:
                    continue
                
                # Convert to Peak objects if necessary
                from logic.integration import Peak
                peak_objects = []
                for p in peaks:
                    # Create peak object from dict
                    peak = Peak(
                        p.get('compound_id', "Unknown"),
                        p.get('peak_number', 0),
                        p.get('retention_time', 0),
                        p.get('integrator', 'py'),
                        p.get('width', 0),
                        p.get('area', 0),
                        p.get('start_time', 0),
                        p.get('end_time', 0)
                    )
                    peak_objects.append(peak)
                
                # Find peaks within RT tolerance
                matching_peaks = []
                for peak in peak_objects:
                    if abs(peak.retention_time - retention_time) <= tolerance:
                        matching_peaks.append(peak)
                
                if not matching_peaks:
                    continue
                
                # Extract spectra for potential matches and check similarity
                confirmed_matches = []
                for peak in matching_peaks:
                    try:
                        # Extract spectrum
                        spectrum = self.data_handler.extract_spectrum_for_peak(
                            peak,
                            {'extraction_method': 'apex', 'debug': False}
                        )
                        
                        if not spectrum or 'mz' not in spectrum or 'intensities' not in spectrum:
                            continue
                        
                        # Convert to tuples for similarity function
                        spectrum1 = [(m, i) for m, i in zip(spectrum['mz'], spectrum['intensities'])]
                        spectrum2 = [(m, i) for m, i in zip(spectrum['mz'], spectrum['intensities'])]
                        
                        # Calculate similarity
                        similarity = dot_product_similarity(spectrum1, spectrum2)
                        
                        # If above threshold, add to confirmed matches
                        similarity_threshold = 0.7  # Default value
                        if hasattr(self, 'edit_assignment_dialog') and hasattr(self.edit_assignment_dialog, 'get_similarity_threshold'):
                            similarity_threshold = self.edit_assignment_dialog.get_similarity_threshold()
                        
                        if similarity >= similarity_threshold:
                            confirmed_matches.append((peak, similarity))
                    except Exception as e:
                        print(f"Error processing peak in {file_path}: {e}")
                
                # Update confirmed matches
                for peak, similarity in confirmed_matches:
                    # Update peak with new assignment
                    old_compound = peak.compound_id if hasattr(peak, 'compound_id') else "Unknown"
                    
                    # Update relevant fields
                    peak.compound_id = compound_name
                    if hasattr(peak, 'Compound_ID'):
                        peak.Compound_ID = compound_name
                        
                    # Try to get CAS number if available
                    casno = None
                    if hasattr(self.ms_frame, 'ms_toolkit') and self.ms_frame.ms_toolkit:
                        if hasattr(self.ms_frame.ms_toolkit, 'library') and compound_name in self.ms_frame.ms_toolkit.library:
                            compound = self.ms_frame.ms_toolkit.library[compound_name]
                            if hasattr(compound, 'casno'):
                                from logic.batch_search import format_casno
                                casno = format_casno(compound.casno)
                                
                    # Update CAS number if available
                    if casno:
                        peak.casno = casno
                    
                    # Manual assignments don't have a match score
                    peak.Qual = None
                    
                    matches_found += 1
                
                # If matches found, save the updated integration results
                if confirmed_matches:
                    # Update the original peaks data with changes
                    for i, p in enumerate(peaks):
                        for peak, _ in confirmed_matches:
                            if p.get('peak_number') == peak.peak_number:
                                p['compound_id'] = peak.compound_id
                                if hasattr(peak, 'Compound_ID'):
                                    p['Compound ID'] = peak.compound_id
                                if hasattr(peak, 'casno') and peak.casno:
                                    p['casno'] = peak.casno
                    
                    # Save back to file
                    with open(integration_json_path, 'w') as f:
                        json.dump(integration_data, f, indent=2)
                
                files_processed += 1
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Close progress dialog
        progress.setValue(len(files))
        
        # Show summary
        message = f"Assignment applied to {matches_found} peaks in {files_processed} files."
        self.status_bar.showMessage(message)
        QMessageBox.information(self, "Cross-File Assignment Complete", message)

    def _add_to_override_database(self, retention_time, compound_name, spectrum=None):
        """Add a manual assignment to the override database for future automatic applications."""
        # Create the overrides directory if it doesn't exist
        import os
        import time
        import json
        
        overrides_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'overrides')
        os.makedirs(overrides_dir, exist_ok=True)
        
        # Create or load the overrides database
        overrides_file = os.path.join(overrides_dir, 'manual_assignments.json')
        overrides = {}
        
        if os.path.exists(overrides_file):
            try:
                with open(overrides_file, 'r') as f:
                    overrides = json.load(f)
            except Exception as e:
                print(f"Error loading override database: {e}")
        
        # Format RT as string for use as key
        rt_key = f"{retention_time:.3f}"
        
        # Add or update the entry
        overrides[rt_key] = {
            'compound_name': compound_name,
            'retention_time': retention_time,
            'timestamp': time.time()  # When the override was created
        }
        
        # Add spectrum data if available
        if spectrum and 'mz' in spectrum and 'intensities' in spectrum:
            # Store only top N peaks to keep file size reasonable
            N = 20
            pairs = list(zip(spectrum['mz'], spectrum['intensities']))
            pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by intensity
            
            top_mz = [float(m) for m, _ in pairs[:N]]
            top_intensities = [float(i) for _, i in pairs[:N]]
            
            overrides[rt_key]['spectrum'] = {
                'mz': top_mz,
                'intensities': top_intensities
            }
        
        # Save the updated database
        try:
            with open(overrides_file, 'w') as f:
                json.dump(overrides, f, indent=2)
            print(f"Added manual assignment override for RT={retention_time:.3f}: {compound_name}")
        except Exception as e:
            print(f"Error saving override database: {e}")

    def _auto_update_json_with_assignment(self, peak):
        """Update JSON file when a peak assignment is manually changed."""
        try:
            from logic.json_exporter import update_json_with_ms_search_results
            
            # Get current data directory
            if not hasattr(self, 'data_handler') or not hasattr(self.data_handler, 'current_directory_path'):
                return
                
            d_path = self.data_handler.current_directory_path
            if not d_path:
                return
                
            # Get detector name
            detector = self.data_handler.current_detector if hasattr(self.data_handler, 'current_detector') else 'Unknown'
            
            # Use export manager for assignment updates
            export_result = self.export_manager.export_after_assignment(
                self.integrated_peaks, 
                d_path, 
                detector
            )
            
            # Show appropriate status message
            success_messages = [msg for msg in export_result['messages'] if 'successfully' in msg or 'exported' in msg]
            if success_messages:
                self.status_bar.showMessage("Assignment updated. " + "; ".join(success_messages))
            elif export_result['json']:
                self.status_bar.showMessage("JSON file updated with assignment change")
            else:
                # Fallback to old method
                self._save_integration_json(self.integration_results)
                
        except Exception as e:
            print(f"Error updating JSON with assignment: {e}")
            # Fallback to old method
            try:
                self._save_integration_json(self.integration_results)
            except:
                pass

    # Button handlers - now with navigation functionality
    def on_export(self):
        """Handle export button click."""
        self.status_bar.showMessage("Export clicked")
        
    def on_previous_sample(self):
        """Handle back button click to navigate to previous sample."""
        if not hasattr(self, 'data_handler'):
            self.status_bar.showMessage("Navigation not available")
            return
            
        prev_path = self.data_handler.navigate_to_previous()
        
        if prev_path:
            self.status_bar.showMessage(f"Navigating to previous sample: {os.path.basename(prev_path)}")
            self.on_file_selected(prev_path)
        else:
            # Show current position information
            if self.data_handler.current_index == 0:
                self.status_bar.showMessage("Already at first sample")
            else:
                self.status_bar.showMessage("No previous sample available")
        
    def on_next_sample(self):
        """Handle next button click to navigate to next sample."""
        if not hasattr(self, 'data_handler'):
            self.status_bar.showMessage("Navigation not available")
            return
            
        next_path = self.data_handler.navigate_to_next()
        
        if next_path:
            self.status_bar.showMessage(f"Navigating to next sample: {os.path.basename(next_path)}")
            self.on_file_selected(next_path)
        else:
            # Show current position information
            if self.data_handler.current_index == len(self.data_handler.available_directories) - 1:
                self.status_bar.showMessage("Already at last sample")
            else:
                self.status_bar.showMessage("No next sample available")
        
    def on_integrate(self):
        """Handle integrate button click."""
        if not hasattr(self, 'current_processed') or self.current_processed is None:
            self.status_bar.showMessage("No data available for integration")
            return
        
        # Check if peak detection is enabled
        params = self.parameters_frame.get_parameters()
        if not params['peaks']['enabled']:
            self.status_bar.showMessage("Enable peak detection first to perform integration")
            QMessageBox.warning(self, "Integration Error", 
                               "You must enable peak detection first to perform integration.")
            return
        
        # Inform user that integration is in progress
        self.status_bar.showMessage("Integrating peaks...")
        
        # Get MS data if available
        ms_data = None
        if hasattr(self.data_handler, 'current_data_dir'):
            try:
                # Use data handler method instead of direct access
                ms_data = self.data_handler.get_ms_data()
                print("Retrieved MS data for peak quality assessment")
            except Exception as e:
                print(f"Could not get MS data for peak quality assessment: {str(e)}")
        
        # Get quality check options from MS frame if available
        quality_options = None
        if hasattr(self.ms_frame, 'search_options'):
            quality_options = {
                'quality_checks_enabled': self.ms_frame.search_options.get('quality_checks_enabled', False),
                'skew_check': self.ms_frame.search_options.get('skew_check', False),
                'coherence_check': self.ms_frame.search_options.get('coherence_check', False),
                'skew_threshold': self.ms_frame.search_options.get('skew_threshold', 0.5),
                'coherence_threshold': self.ms_frame.search_options.get('coherence_threshold', 0.7),
                'high_corr_threshold': self.ms_frame.search_options.get('high_corr_threshold', 0.5)
            }
        
        # Use our non-UI integration method (reuses core logic)
        integration_results = self.integrate_peaks_no_ui(ms_data=ms_data, quality_options=quality_options)
        
        # Check if integration was successful
        if not integration_results:
            self.status_bar.showMessage("No peaks found for integration")
            return
        
        # Shade the areas under the curve
        self.plot_frame.shade_integration_areas(integration_results)
        
        # Automatically save integration results using export manager
        if hasattr(self, 'data_handler') and hasattr(self.data_handler, 'current_directory_path') and self.data_handler.current_directory_path:
            try:
                # Get integration results in the right format
                peaks = integration_results.get('peaks', [])
                d_path = self.data_handler.current_directory_path
                detector = self.data_handler.current_detector if hasattr(self.data_handler, 'current_detector') else 'Unknown'
                
                # Use export manager for consistent export behavior
                export_result = self.export_manager.export_after_integration(peaks, d_path, detector)
                
                # Show export status
                success_messages = [msg for msg in export_result['messages'] if 'successfully' in msg or 'exported' in msg]
                if success_messages:
                    self.status_bar.showMessage(f"Integration complete: {len(peaks)} peaks. " + "; ".join(success_messages))
                else:
                    self.status_bar.showMessage(f"Integration complete: {len(peaks)} peaks found.")
                    
            except Exception as e:
                print(f"Warning: Export manager failed: {e}")
                # Fallback to old method
                self._save_integration_json(integration_results)
        
        # Create and show the "View Integration Results" button in the status bar area
        if hasattr(self, 'view_results_button'):
            self.view_results_button.setVisible(True)
        else:
            self.view_results_button = QPushButton("View Integration Results")
            self.view_results_button.clicked.connect(lambda: self._show_integration_results(self.integration_results))
            self.statusBar().addPermanentWidget(self.view_results_button)
        
        # Update status bar
        if not success_messages:
            self.status_bar.showMessage(f"Integration complete: {len(integration_results['peaks'])} peaks found. Click the button to view results.")
        
        # Notify peaks integrated
        self.on_peaks_integrated(integration_results['peaks'])

    def integrate_peaks_no_ui(self, ms_data=None, quality_options=None):
        """Thread-safe version of integration without UI updates."""
        try:
            # Check for valid data (no UI messaging)
            if not hasattr(self, 'current_processed') or self.current_processed is None:
                return None
            
            # Check if peak detection is enabled (no UI messaging)
            params = self.parameters_frame.get_parameters()
            if not params['peaks']['enabled']:
                return None
            
            # Perform integration - CORE FUNCTIONALITY WITHOUT UI UPDATES
            integration_results = self.processor.integrate_peaks(
                processed_data=self.current_processed,
                ms_data=ms_data,  # Pass MS data
                quality_options=quality_options  # Pass quality options
            )
            
            # Check if integration was successful
            if not integration_results['peaks']:
                return None
            
            # Store integration results
            self.integration_results = integration_results
            
            # Store peaks separately for easier access
            self.integrated_peaks = integration_results['peaks']
            
            return integration_results
            
        except Exception as e:
            print(f"Error in integrate_peaks_no_ui: {str(e)}")
            return None

    def _show_integration_results(self, integration_results):
        """Show integration results in a dialog."""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Integration Results")
        dialog.resize(900, 500)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create table
        table = QTableWidget()
        headers = ['Compound ID', 'Peak #', 'Ret Time', 'Integrator', 'Width', 'Area', 'Start Time', 'End Time', 'Quality']
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        
        # Populate table
        peaks = integration_results['peaks']
        table.setRowCount(len(peaks))
        
        for i, peak in enumerate(peaks):
            # Add regular peak data
            for j, value in enumerate(peak.as_row):
                item = QTableWidgetItem(str(value))
                table.setItem(i, j, item)
            
            # Add quality indicator in the last column
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                quality_item = QTableWidgetItem("⚠️ SATURATED")
                quality_item.setForeground(QBrush(QColor(128, 0, 128)))  # Purple text
            elif hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                quality_item = QTableWidgetItem("⚠️ Check")
                quality_item.setForeground(QBrush(QColor(255, 0, 0)))  # Red text
            else:
                quality_item = QTableWidgetItem("OK")
                quality_item.setForeground(QBrush(QColor(0, 128, 0)))  # Green text
            
            table.setItem(i, len(headers)-1, quality_item)
        
        # Adjust column widths
        header = table.horizontalHeader()
        for i in range(len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        # Add table to layout
        layout.addWidget(table)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        # Export button
        export_button = QPushButton("Export to CSV")
        export_button.clicked.connect(lambda: self._export_integration_results(integration_results))
        button_layout.addWidget(export_button)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        dialog.exec()

    def _export_integration_results(self, integration_results):
        """Export integration results to a CSV file.
        
        Args:
            integration_results: Dictionary containing integration results
        """
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Integration Results", "", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Create DataFrame from peaks
        peaks_data = [peak.as_dict for peak in integration_results['peaks']]
        df = pd.DataFrame(peaks_data)
        
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        # Show confirmation
        self.status_bar.showMessage(f"Integration results exported to {file_path}")
    
    def _save_integration_json(self, integration_results):
        """Save integration results as JSON in the data directory."""
        # Check if we have a valid data directory
        if not hasattr(self, 'current_directory_path') or not self.current_directory_path:
            print("DEBUG: No current_directory_path available")
            self.status_bar.showMessage("No data directory available for saving results")
            return False
        
        try:
            # Get current data directory
            data_dir_path = self.current_directory_path
            print(f"DEBUG: Saving integration results to {data_dir_path}")
            
            # First try to get metadata from current_sample_data if it exists
            if hasattr(self, 'current_sample_data') and self.current_sample_data is not None:
                print("DEBUG: Using metadata from current_sample_data")
                sample_id = os.path.basename(data_dir_path)
                
                # Detector might be set from data handler
                detector = getattr(self.data_handler, 'current_detector', 'Unknown')  # Use actual detector
                
                # Extract metadata from sample data
                method = self.current_sample_data.metadata.get('method', 'Unknown')
                if method and '.M' in method:
                    method = method.split('.M')[0]
                    
                notebook = self.current_sample_data.metadata.get('notebook', sample_id)
                timestamp = self.current_sample_data.metadata.get('date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                signal = f"Signal: {notebook}\\{detector}.ch"
                
            # Otherwise try to get the file metadata directly from the data directory
            elif hasattr(self, 'data_handler'):
                try:
                    # Use the current detector for metadata
                    detector = self.data_handler.current_detector
                    fid_metadata = self.data_handler.get_detector_metadata(detector)
                    print(f"DEBUG: Successfully accessed {detector}.ch metadata")
                    
                    # Extract metadata from the detector file
                    data_dir_obj = self.data_handler.current_data_dir
                    sample_id = getattr(data_dir_obj, 'name', os.path.basename(data_dir_path))
                    timestamp = fid_metadata.get('date', data_dir_obj.metadata.get('date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    method = fid_metadata.get('method', data_dir_obj.metadata.get('method', 'Unknown'))
                    if method and '.M' in method:
                        method = method.split('.M')[0]
                    
                    # Keep the detector as retrieved from data_handler (don't hardcode)
                    notebook = fid_metadata.get('notebook', data_dir_obj.metadata.get('notebook', sample_id))
                    signal = f"Signal: {notebook}\\{detector}.ch"
                    
                except Exception as e:
                    print(f"DEBUG: Failed to get FID1A.ch metadata: {str(e)}")
                    
                    # Fall back to directory metadata
                    print("DEBUG: Falling back to directory metadata")
                    data_dir_obj = self.data_handler.current_data_dir
                    sample_id = getattr(data_dir_obj, 'name', os.path.basename(data_dir_path))
                    timestamp = data_dir_obj.metadata.get('date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    method = data_dir_obj.metadata.get('method', 'Unknown')
                    if method and '.M' in method:
                        method = method.split('.M')[0]
                    
                    # Use the current detector from data handler (don't hardcode)
                    detector = self.data_handler.current_detector
                    notebook = data_dir_obj.metadata.get('notebook', sample_id)
                    signal = f"Signal: {notebook}\\{detector}.ch"
            
            print(f"DEBUG: Extracted metadata - Sample: {sample_id}, Method: {method}")
        except Exception as e:
            # Fallback in case of error
            print(f"DEBUG: Error extracting metadata: {str(e)}")
            sample_id = os.path.basename(data_dir_path)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            method = "Unknown"
            notebook = sample_id
            detector = getattr(self.data_handler, 'current_detector', 'Unknown')  # Use actual detector as fallback
            signal = f"Signal: {notebook}\\{detector}.ch"
        
        # Prepare data structure for output
        result_data = {
            'sample_id': sample_id,
            'timestamp': timestamp,
            'method': method,
            'detector': detector,
            'signal': signal,
            'notebook': notebook,
            'peaks': []
        }
        
        # Add peaks data
        peaks = integration_results.get('peaks', [])
        for peak in peaks:
            peak_data = {
                'compound_id': peak.compound_id,
                'peak_number': peak.peak_number,
                'retention_time': peak.retention_time,
                'integrator': peak.integrator,
                'width': peak.width,
                'area': peak.area,
                'start_time': peak.start_time,
                'end_time': peak.end_time
            }
            result_data['peaks'].append(peak_data)
        
        # Define the file path and save the results as JSON
        result_filename = f"{notebook} - {detector}.json"
        result_file_path = os.path.join(data_dir_path, result_filename)
        
        print(f"DEBUG: Writing to {result_file_path}")
        with open(result_file_path, 'w') as result_file:
            json.dump(result_data, result_file, indent=4)
        
        print(f"DEBUG: Successfully wrote integration results")
        self.status_bar.showMessage(f"Integration results saved to {result_filename}")
        return True


    def on_automation_clicked(self):
        """Handle automation button click."""
        # Start the automation process directly without checking enabled state
        self.start_automation()

    def start_automation(self):
        """Start the automation process."""
        # Check for current directory
        if not hasattr(self.data_handler, 'current_directory_path') or not self.data_handler.current_directory_path:
            self.status_bar.showMessage("No directory selected")
            return
        
        # Get the parent directory of the current file
        current_dir = self.data_handler.current_directory_path
        parent_dir = os.path.dirname(current_dir)
        
        # Create and show the automation dialog
        self.automation_dialog = AutomationDialog(self)
        self.automation_dialog.cancelled.connect(self.cancel_automation)
        
        # Create the worker
        self.automation_worker = AutomationWorker(self, parent_dir)
        
        # Connect signals
        self.automation_worker.signals.started.connect(
            lambda total: self.automation_dialog.update_overall_progress(0, total))
        
        self.automation_worker.signals.file_started.connect(
            lambda filename, idx, total: self.automation_dialog.update_overall_progress(idx-1, total))
        
        self.automation_worker.signals.file_progress.connect(
            lambda filename, step, percent: self.automation_dialog.update_file_progress(filename, step, percent))
        
        self.automation_worker.signals.log_message.connect(
            self.update_status_from_worker, 
            Qt.ConnectionType.QueuedConnection
        )
        
        self.automation_worker.signals.log_message.connect(
            self.automation_dialog.add_log_message,
            Qt.ConnectionType.QueuedConnection
        )
        
        self.automation_worker.signals.finished.connect(
            self.on_automation_finished)
        
        self.automation_worker.signals.error.connect(
            lambda msg: self.on_automation_error(msg))
        
        # Add this new connection for smoother overall progress
        self.automation_worker.signals.overall_progress.connect(
            lambda current, total, percent: self.automation_dialog.update_overall_progress_percent(current, total, percent)
        )
        
        # Add this connection for status bar updates
        self.automation_worker.signals.log_message.connect(self.update_status_from_worker)
        
        # Start worker in a thread
        self.automation_threadpool = QThreadPool()
        self.automation_threadpool.start(self.automation_worker)
        
        # Show the dialog
        self.automation_dialog.show()

    def cancel_automation(self):
        """Cancel the automation process."""
        if hasattr(self, 'automation_worker'):
            self.automation_worker.cancelled = True
            
            # Also cancel any inner workers
            if hasattr(self.automation_worker, 'batch_worker') and self.automation_worker.batch_worker:
                self.automation_worker.batch_worker.cancelled = True
                
            self.status_bar.showMessage("Cancelling automation...")
            
            # Update dialog if visible
            if hasattr(self, 'automation_dialog') and self.automation_dialog.isVisible():
                self.automation_dialog.add_log_message("Cancellation requested - this may take a moment to complete...")

    def on_automation_finished(self):
        """Handle automation completion."""
        # Check if automation was cancelled
        was_cancelled = False
        if hasattr(self, 'automation_worker') and self.automation_worker.cancelled:
            was_cancelled = True
            self.status_bar.showMessage("Automation cancelled")
        else:
            self.status_bar.showMessage("Automation completed")
        
        # Add appropriate message to the dialog
        if hasattr(self, 'automation_dialog') and self.automation_dialog.isVisible():
            if was_cancelled:
                self.automation_dialog.mark_cancelled()
            else:
                self.automation_dialog.mark_completed(success=True)

    def on_automation_error(self, error_message):
        """Handle automation error."""
        self.status_bar.showMessage("Automation error")
        
        # Add error message to the dialog
        if hasattr(self, 'automation_dialog') and self.automation_dialog.isVisible():
            self.automation_dialog.mark_error(error_message)

    # Add this helper method to create batch search workers
    def _create_batch_search_worker(self):
        """Create a batch search worker without showing a progress dialog."""
        # Check if we have the required components
        if not hasattr(self.ms_frame, 'ms_toolkit') or not self.ms_frame.ms_toolkit:
            return None
        
        if not self.ms_frame.library_loaded:
            return None
        
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            return None
        
        # Check if we have a data directory
        if not hasattr(self.data_handler, 'current_directory_path') or not self.data_handler.current_directory_path:
            return None
        
        # Ensure the current m/z shift from the UI is applied to the toolkit
        try:
            mz_shift = int(self.ms_frame.mz_shift_entry.text() or 0)
            self.ms_frame.ms_toolkit.mz_shift = mz_shift
        except (ValueError, AttributeError):
            mz_shift = 0
        
        # Get MS search options
        if hasattr(self.ms_frame, 'search_options'):
            search_options = self.ms_frame.search_options
        else:
            search_options = {
                'search_method': 'vector',
                'extraction_method': 'apex',
                'range_points': 5,
                'tic_weight': True,
                'subtract_enabled': True,
                'subtraction_method': 'min_tic',
                'subtract_weight': 0.1,
                'similarity': 'composite',
                'weighting': 'NIST_GC',
                'unmatched': 'keep_all',
                'intensity_power': 0.6,
                'top_n': 5
            }
        
        # Create the worker
        worker = BatchSearchWorker(
            self.ms_frame.ms_toolkit,
            self.integrated_peaks,
            self.data_handler.current_directory_path,
            options={
                **search_options,
                'mz_shift': mz_shift,
                'debug': True
            }
        )
        
        return worker


    @Slot(dict)
    def on_parameters_changed(self, params):
        """Handle parameter changes from the parameters frame"""
        print("Parameters changed signal received!")
        
        # Check if we have data to process
        if self.current_x is not None and self.current_y is not None:
            print(f"Processing data with shape: x={len(self.current_x)}, y={len(self.current_y)}")
            self.status_bar.showMessage("Processing with updated parameters...")
            
            # Process and display the data with new parameters
            self.process_and_display(self.current_x, self.current_y, new_file=False)
            
            # Update status message with details
            method_name = params['baseline']['method']
            lambda_val = params['baseline']['lambda']
            view_mode = "corrected" if params['baseline']['show_corrected'] else "raw with baseline"
            
            # Build status message
            msg_parts = []
            
            if params['smoothing']['enabled']:
                med_kernel = params['smoothing']['median_filter']['kernel_size']
                sg_window = params['smoothing']['savgol_filter']['window_length']
                msg_parts.append(f"Smoothing (med={med_kernel}, sg={sg_window})")
            
            # Always include baseline info since we're always applying baseline correction
            msg_parts.append(f"{method_name} baseline (λ={lambda_val:.0f})")
            
            # Add view mode info
            msg_parts.append(f"View: {view_mode}")
            
            if msg_parts:
                msg = "Applied: " + ", ".join(msg_parts)
            else:
                msg = "Processing complete"
                
            self.status_bar.showMessage(msg)
            print(msg)
        else:
            print("No data available to process!")
            self.status_bar.showMessage("Load data first before changing parameters")
    
    def process_and_display(self, x, y, new_file=False):
        """Process data with current parameters and update display"""
        # Validate input data
        if x is None or y is None:
            print("No data to process")
            return
            
        if len(x) != len(y):
            print(f"Data length mismatch: x={len(x)}, y={len(y)}")
            return
            
        # Data is already interpolated to standard length
        print(f"Processing data with {len(x)} points")
        
        # Get current parameters
        params = self.parameters_frame.get_parameters()
        
        # Get MS data range if available
        ms_range = None
        if hasattr(self, 'plot_frame') and hasattr(self.plot_frame, 'tic_data') and self.plot_frame.tic_data is not None:
            if 'x' in self.plot_frame.tic_data and len(self.plot_frame.tic_data['x']) > 0:
                ms_min = np.min(self.plot_frame.tic_data['x'])
                ms_max = np.max(self.plot_frame.tic_data['x'])
                ms_range = (ms_min, ms_max)
                print(f"Using MS range for peak filtering: {ms_min:.2f} to {ms_max:.2f} min")

        # Process the data (already interpolated)
        processed = self.processor.process(x, y, params, ms_range)
        
        # Update the chromatogram plot
        self.plot_frame.plot_chromatogram(
            processed,
            show_corrected=params['baseline']['show_corrected'], 
            new_file=new_file
        )
        
        # Force a repaint
        self.plot_frame.canvas.draw_idle()
        
        # Store the processed data for reference
        self.current_processed = processed

    def debug_data(self):
        """Print debug information about current data."""
        if hasattr(self, 'current_x') and hasattr(self, 'current_y'):
            if self.current_x is not None and self.current_y is not None:
                print(f"Current data info:")
                print(f"  X: type={type(self.current_x)}, length={len(self.current_x)}")
                print(f"  Y: type={type(self.current_y)}, length={len(self.current_y)}")
                
                if hasattr(self, 'current_processed'):
                    print(f"Processed data info:")
                    for key, value in self.current_processed.items():
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: type={type(value)}, length={len(value)}")
                        else:
                            print(f"  {key}: type={type(value)}")
            else:
                print("No current data available")
        else:
            print("Data attributes not initialized")

    def run_batch_ms_search(self):
        """Run MS library search on all integrated peaks."""
        # Check if we have the required components
        if not hasattr(self.ms_frame, 'ms_toolkit') or not self.ms_frame.ms_toolkit:
            self.status_bar.showMessage("MS toolkit not available")
            return
        
        if not self.ms_frame.library_loaded:
            self.status_bar.showMessage("MS library not loaded")
            QMessageBox.warning(self, "Library Not Loaded", "Please load an MS library first.")
            return
        
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            self.status_bar.showMessage("No integrated peaks available")
            QMessageBox.warning(self, "No Peaks", "Please integrate peaks first.")
            return
        
        # Check if we have a data directory
        if not hasattr(self.data_handler, 'current_directory_path') or not self.data_handler.current_directory_path:
            self.status_bar.showMessage("No data directory selected")
            return
        
        # Ensure the current m/z shift from the UI is applied to the toolkit
        try:
            mz_shift = int(self.ms_frame.mz_shift_entry.text() or 0)
            self.ms_frame.ms_toolkit.mz_shift = mz_shift
        except (ValueError, AttributeError) as e:
            self.status_bar.showMessage(f"Error setting m/z shift: {str(e)}")
            mz_shift = 0
        
        # Get MS search options from the MS frame
        if hasattr(self.ms_frame, 'search_options'):
            search_options = self.ms_frame.search_options
        else:
            search_options = {
                'search_method': 'vector',
                'extraction_method': 'apex',
                'range_points': 5,
                'tic_weight': True,
                'subtract_enabled': True,
                'subtraction_method': 'min_tic',
                'subtract_weight': 0.1,
                'similarity': 'composite',
                'weighting': 'NIST_GC',
                'unmatched': 'keep_all',
                'intensity_power': 0.6,
                'top_n': 5
            }
        
        # Create the worker instance for batch search - FIX HERE
        # Inconsistent construction of BatchSearchWorker. We have a method to do this.
        '''
        worker = BatchSearchWorker(
            self.ms_frame.ms_toolkit,
            self.integrated_peaks,
            self.data_handler.current_directory_path,
            options={
                'search_method': search_options['search_method'],
                'extraction_method': search_options['extraction_method'],
                'range_points': search_options['range_points'],
                'tic_weight': search_options['tic_weight'],
                'subtract_enabled': search_options['subtract_enabled'],
                'subtraction_method': search_options['subtraction_method'],
                'subtract_weight': search_options['subtract_weight'],
                'similarity': search_options['similarity'],
                'weighting': search_options['weighting'],
                'unmatched': search_options['unmatched'],
                'intensity_power': search_options['intensity_power'],
                'top_n': search_options['top_n'],
                'mz_shift': mz_shift,
                'debug': True
            }
        )
        '''
        # Using the method we alread have here -->
        worker = self._create_batch_search_worker()
        
        # Track the worker to be able to stop it
        self.batch_search_worker = worker
        
        # Connect the cancel button to stop the search
        progress_dialog = QProgressDialog("Searching MS library...", "Cancel", 0, len(self.integrated_peaks), self)
        progress_dialog.setWindowTitle("Batch MS Search")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        
        # CRITICAL FIX: Connect cancel button to cancellation method
        progress_dialog.canceled.connect(self._cancel_batch_search)
        
        # Connect signals
        worker.signals.started.connect(lambda total: 
            self.status_bar.showMessage(f"Starting batch search on {total} peaks..."))
        
        worker.signals.progress.connect(lambda i, name, results: 
            self._update_batch_search_progress(i, name, results, progress_dialog))
        
        worker.signals.finished.connect(lambda: 
            self._on_batch_search_finished(progress_dialog))
        
        worker.signals.error.connect(lambda msg: 
            self._on_batch_search_error(msg, progress_dialog))
        
        worker.signals.log_message.connect(
            self.update_status_from_worker, 
            Qt.ConnectionType.QueuedConnection
        )
        
        # Start worker
        QThreadPool.globalInstance().start(worker)

    def _cancel_batch_search(self):
        """Cancel the running batch search."""
        if hasattr(self, 'batch_search_worker'):
            # Set a cancel flag in the worker
            self.batch_search_worker.cancelled = True
            self.status_bar.showMessage("Batch search cancelled by user")
            
            # The finished signal will still be emitted by the worker, which will close the dialog
            # This ensures proper cleanup even when cancelled

    def _update_batch_search_progress(self, index, compound_name, results, dialog):
        """Update batch search progress."""
        dialog.setValue(index + 1)
        dialog.setLabelText(f"Found match: {compound_name}")

        # Update status bar
        self.status_bar.showMessage(f"Processing peak {index + 1}/{dialog.maximum()}: {compound_name}")

        # Also update the peak in the results table if it exists
        self._update_peak_match_in_results(index, compound_name, results[0][1])

        # Process events to keep UI responsive
        QApplication.processEvents()

    def _on_batch_search_finished(self, dialog):
        """Handle batch search completion."""
        dialog.close()
        
        # Count successful matches
        match_count = sum(1 for peak in self.integrated_peaks 
                         if hasattr(peak, 'Compound_ID') and peak.Compound_ID)
        
        total_peaks = len(self.integrated_peaks)
        
        # Check if search was cancelled
        if hasattr(self, 'batch_search_worker') and self.batch_search_worker.cancelled:
            self.status_bar.showMessage(f"Batch MS search cancelled: {match_count}/{total_peaks} peaks identified")
        else:
            self.status_bar.showMessage(f"Batch MS search completed: {match_count}/{total_peaks} peaks identified")
        
        # Update results view
        self.update_results_view()
        
        # Use a helper method to handle the JSON/CSV saving on the main thread
        if match_count > 0:
            # Save the updated integration results back to the original JSON file
            if hasattr(self.data_handler, 'current_directory_path') and self.data_handler.current_directory_path:
                # Use a main thread method to handle all this
                self._save_and_export_results_main_thread(match_count, total_peaks)
        elif not (hasattr(self, 'batch_search_worker') and self.batch_search_worker.cancelled):
            # Only show this message if not cancelled
            QMessageBox.information(
                self, "Batch Search Complete", 
                "MS library search completed but no compounds were identified. This may occur if:\n\n"
                "1. The MS data file (.ms) is not present or accessible\n"
                "2. The peaks don't correspond to MS scans\n"
                "3. The m/z shift value needs adjustment\n\n"
                "Try adjusting the m/z shift or checking the data files."
            )

    def _save_and_export_results_main_thread(self, match_count, total_peaks):
        """Handle saving and exporting results on the main thread."""
        current_dir = self.data_handler.current_directory_path
        
        # Use export manager for consistent export behavior
        try:
            detector = self.data_handler.current_detector if hasattr(self.data_handler, 'current_detector') else 'Unknown'
            # Export using export manager
            export_result = self.export_manager.export_after_ms_search(
                self.integrated_peaks, 
                current_dir, 
                detector
            )
            
            # Show appropriate status message
            if export_result['json'] or export_result['csv']:
                export_messages = [msg for msg in export_result['messages'] if 'successfully' in msg or 'exported' in msg]
                if export_messages:
                    status_msg = f"MS search completed with {match_count}/{total_peaks} peaks identified. " + "; ".join(export_messages)
                    self.status_bar.showMessage(status_msg)
                    
                    # Only show dialog if not cancelled
                    if not (hasattr(self, 'batch_search_worker') and self.batch_search_worker.cancelled):
                        result_details = "\n".join(export_result['messages'])
                        QMessageBox.information(
                            self, "Batch Search Complete", 
                            f"MS library search completed with {match_count}/{total_peaks} peaks identified.\n\n{result_details}",
                            QMessageBox.Ok
                        )
                else:
                    self.status_bar.showMessage(f"MS search completed with {match_count}/{total_peaks} peaks identified")
            else:
                self.status_bar.showMessage("MS search completed but export failed")
                
        except Exception as e:
            print(f"Error in export manager: {e}")
            # Fallback to old method
            try:
                from logic.json_exporter import update_json_with_ms_search_results
                detector = self.data_handler.current_detector if hasattr(self.data_handler, 'current_detector') else 'Unknown'
                json_success = update_json_with_ms_search_results(self.integrated_peaks, current_dir, detector)
                
                if json_success:
                    # Automatically export CSV to the same directory
                    csv_filename = os.path.join(current_dir, "RESULTS.CSV")
                    csv_success = self.export_results_csv(csv_filename)
                    
                    if csv_success:
                        self.status_bar.showMessage(f"Updated integration results and exported to {csv_filename}")
                    
                    # Only show info dialog if not cancelled
                    if not (hasattr(self, 'batch_search_worker') and self.batch_search_worker.cancelled):
                        QMessageBox.information(
                            self, "Batch Search Complete", 
                            f"MS library search completed with {match_count}/{total_peaks} peaks identified.\n\n"
                            f"Results saved to JSON and exported to {csv_filename}.",
                            QMessageBox.Ok
                        )
                else:
                    self.status_bar.showMessage("Failed to update integration results")
            except Exception as fallback_error:
                print(f"Fallback export also failed: {fallback_error}")
                self.status_bar.showMessage("Export failed")
                
        except Exception as e:
            print(f"Error updating results: {e}")
            self.status_bar.showMessage(f"Error updating results: {str(e)}")
            
            # Fallback to old method
            integration_results = {'peaks': self.integrated_peaks}
            json_success = self._save_integration_json(integration_results)
            
            if json_success:
                csv_filename = os.path.join(current_dir, "RESULTS.CSV")
                csv_success = self.export_results_csv(csv_filename)
                
                if csv_success:
                    self.status_bar.showMessage(f"Updated integration results and exported to {csv_filename}")
                
                if not (hasattr(self, 'batch_search_worker') and self.batch_search_worker.cancelled):
                    QMessageBox.information(
                        self, "Batch Search Complete", 
                        f"MS library search completed with {match_count}/{total_peaks} peaks identified.\n\n"
                        f"Results saved to JSON and exported to {csv_filename}.",
                        QMessageBox.Ok
                    )
            else:
                self.status_bar.showMessage("Failed to save updated integration results")

    def _on_batch_search_error(self, error_message, dialog):
        """Handle batch search error."""
        dialog.close()
        self.status_bar.showMessage("Error in batch MS search")
        QMessageBox.critical(self, "Batch Search Error", error_message)

    def _update_peak_match_in_results(self, peak_index, compound_name, match_score):
        """Update peak match in results table."""
        # Update the compound name and match score in results table
        if hasattr(self, 'results_table') and self.results_table:
            # Update the compound name and match score in the appropriate columns
            compound_col = 4  # Adjust based on your actual table structure
            score_col = 5     # Adjust based on your actual table structure

            # Set the values
            self.results_table.setItem(peak_index, compound_col, 
                                    QTableWidgetItem(compound_name))
            self.results_table.setItem(peak_index, score_col, 
                                    QTableWidgetItem(f"{match_score:.3f}"))

    def extract_spectrum_at_rt(self, retention_time):
        """Extract mass spectrum at the given retention time."""
        try:
            # Check if we have access to the data handler and data
            if not hasattr(self, 'data_handler') or not self.data_handler:
                return None
            
            # Get the current data directory
            current_dir = self.data_handler.current_directory_path
            if not current_dir:
                return None
            
            # Get TIC data
            tic_data = self.data_handler._get_tic_data(current_dir)
            if not tic_data or 'x' not in tic_data or 'y' not in tic_data:
                return None
            
            # Check if TIC data arrays are empty (THIS IS THE KEY FIX)
            if len(tic_data['x']) == 0 or len(tic_data['y']) == 0:
                print(f"Empty TIC data arrays for directory: {current_dir}")
                return None
            
            # Find the closest point in the TIC
            rt_index = np.argmin(np.abs(np.array(tic_data['x']) - retention_time))
            
            # Get MS data
            try:
                datadir = rb.read(current_dir)
                ms = datadir.get_file('data.ms')
                
                # Extract the spectrum at the given retention time
                spectrum = ms.data[rt_index, :].astype(float)
                
                # Create mz array
                mz_values = np.arange(len(spectrum)) + 1
                
               
                
                # Filter out low intensity values
                threshold = 0.01 * np.max(spectrum)
                mask = spectrum > threshold
                
                result = {
                    'rt': retention_time,
                    'mz': mz_values[mask],
                    'intensities': spectrum[mask]
                }
                
                return result
                
            except Exception as e:
                print(f"MS data extraction error: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error extracting spectrum: {str(e)}")
            return None

    def on_peaks_integrated(self, peaks):
        """Handle when peaks are integrated."""
        self.integrated_peaks = peaks
        
        # Update batch search button state based on peaks and library
        has_library = (hasattr(self.ms_frame, 'library_loaded') and 
                      
                      self.ms_frame.library_loaded)
        
        self.button_frame.batch_search_button.setEnabled(bool(peaks) and has_library)
        
        # Update other UI elements as needed
        self.update_results_view()

    # Export integration results to JSON file
    def export_results(self):
        """Export integration results to JSON file."""
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            QMessageBox.warning(self, "No Results", "No integrated peaks to export.")
            return
        
        # Get file path for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            import json
            
            # Create dictionary of results
            peaks_data = []
            
            for peak in self.integrated_peaks:
                peak_data = {
                    'peak_number': peak.peak_number,
                    'retention_time': peak.retention_time,
                    'start_time': peak.start_time,
                    'end_time': peak.end_time,
                    'width': peak.width,
                    'area': peak.area
                }
                
                # Add compound info if available
                if peak.compound_name:
                    peak_data['compound_id'] = peak.compound_name
                    peak_data['match_score'] = peak.match_score
                    if peak.casno:
                        peak_data['casno'] = peak.casno
                
                peaks_data.append(peak_data)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump({'peaks': peaks_data}, f, indent=2)
            
            self.status_bar.showMessage(f"Results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")

    def export_results_csv(self, filepath=None):
        """Export integration results to a CSV file matching GCMS standard format."""
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            QMessageBox.warning(self, "No Results", "No integrated peaks to export.")
            return False
        
        if filepath is None:
            # Get file path for saving
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Results CSV", "", "CSV Files (*.csv);;All Files (*.*)"
            )
        
        if not filepath:
            return False
        
        try:
            import csv
            
            # Create and write to the CSV file
            with open(filepath, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # Skip the first 9 rows by initializing with empty rows - exactly as in notebook
                for _ in range(9):
                    csv_writer.writerow([])
                
                # Write headers - exactly as in notebook
                headers = ['Library/ID', 'CAS', 'Qual', 'FID R.T.', 'FID Area']
                csv_writer.writerow(headers)
                
                # Write peak data
                for peak in self.integrated_peaks:
                    # Use the exact field names from the notebook
                    compound_id = getattr(peak, 'Compound_ID', None) or getattr(peak, 'compound_id', "Unknown")
                    casno = getattr(peak, 'casno', "")
                    qual = getattr(peak, 'Qual', "")
                    
                    # Format qual as a float with 4 decimal places if it's a number
                    if isinstance(qual, (int, float)):
                        qual = f"{qual:.4f}"
                    
                    row = [
                        compound_id,
                        casno,
                        qual,
                        f"{peak.retention_time:.3f}",
                        f"{peak.area:.1f}"
                    ]
                    csv_writer.writerow(row)
            
            self.status_bar.showMessage(f"Results exported to {filepath}")
            return True
        
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")
            return False

    def update_results_view(self):
        """Update the results view with current peak information."""
        # Check if we have integration results to display
        if not hasattr(self, 'integrated_peaks') or not self.integrated_peaks:
            return
        
        # If we have a current results table, update it
        if hasattr(self, 'results_table') and self.results_table:
            # Clear the table first
            while self.results_table.rowCount() > 0:
                self.results_table.removeRow(0)
            
            # Populate with current data
            for i, peak in enumerate(self.integrated_peaks):
                self.results_table.insertRow(i)
                
                # Basic peak info
                self.results_table.setItem(i, 0, QTableWidgetItem(str(peak.peak_number)))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{peak.retention_time:.3f}"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{peak.area:.1f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{peak.width:.3f}"))
                
                # Add compound info if available - use notebook's field names
                compound_id = getattr(peak, 'Compound_ID', None) or getattr(peak, 'compound_id', None)
                if compound_id and "Unknown" not in str(compound_id):
                    self.results_table.setItem(i, 4, QTableWidgetItem(str(compound_id)))
                    if hasattr(peak, 'Qual') and peak.Qual is not None:
                                               self.results_table.setItem(i, 5, QTableWidgetItem(f"{peak.Qual:.4f}"))
        
        # Update any visualization that might depend on peak identifications
        if hasattr(self, 'plot_frame'):
            self.plot_frame.update_annotations()

    # Update the status_from_worker method to be robust
    @Slot(str)
    def update_status_from_worker(self, message):
        """Update status bar with a message from a worker thread."""
        # Use a single-shot timer to ensure this runs on the main thread
        QTimer.singleShot(0, lambda: self.status_bar.showMessage(message))

    def show_batch_queue_dialog(self):
        """Show the batch job setup dialog before the progress dialog."""
        from ui.dialogs.batch_job_dialog import BatchJobDialog
        
        # Create and show the setup dialog first
        setup_dialog = BatchJobDialog(self)
        setup_dialog.start_batch.connect(self.start_batch_processing)
        
        # Show the dialog modally - will block until user configures and starts or cancels
        setup_dialog.exec()
    
    def start_batch_processing(self, directories, options):
        """Start batch processing with the selected directories and options."""
        if not directories:
            self.status_bar.showMessage("No directories to process")
            return
        
        print(f"Starting batch processing with {len(directories)} directories")
        
        # Now show the progress dialog to monitor the process
        from ui.dialogs.batch_progress_dialog import BatchProgressDialog
        
        # Initialize with the directories from the setup dialog
        self.batch_dialog = BatchProgressDialog(self, directories)
        
        # Connect signals
        self.batch_dialog.cancelled.connect(self.on_batch_cancelled)
        self.batch_dialog.modify_queue.connect(self.on_batch_queue_modified)
        
        # Store options for use during processing
        self.batch_options = options
        
        # CRITICAL: Set the batch_directories property
        self.batch_directories = directories.copy()
        
        # Show the dialog
        self.batch_dialog.show()
        
        # Start processing the first directory
        print(f"Calling process_next_batch_directory with {len(self.batch_directories)} dirs")
        self.process_next_batch_directory()

    def on_batch_cancelled(self):
        """Handle batch processing cancellation."""
               # Set cancelled flag on any active workers
        if hasattr(self, 'current_batch_worker') and self.current_batch_worker:
            self.current_batch_worker.cancelled = True
        
        # Clean up
        self.current_batch_directory = None
        self.batch_directories = []

    def on_batch_queue_modified(self, new_queue):
        """Handle modification of the batch queue."""
        self.batch_directories = new_queue
        
        # If we're not currently processing, start processing
        if not hasattr(self, 'current_batch_directory') or not self.current_batch_directory:
            self.process_next_batch_directory()

    def process_next_batch_directory(self):
        """Process the next directory in the batch queue."""
        # If we have no batch dialog or it's been closed, stop processing
        if not hasattr(self, 'batch_dialog') or not self.batch_dialog:
            print("No batch dialog - stopping processing")
            return
        
        # If no more directories, we're done
        if not hasattr(self, 'batch_directories') or not self.batch_directories:
            print("No more directories to process - completing")
            self.batch_dialog.complete_processing()
            return
        
        # Get the next directory
        next_dir = self.batch_directories.pop(0)
        self.current_batch_directory = next_dir
        
        print(f"Processing next directory: {next_dir}")
        
        # Update status
        self.batch_dialog.update_directory_status(
            next_dir, 'processing', 0, "Starting processing..."
        )
        
        # Create the worker with this directory
        from logic.automation_worker import AutomationWorker
        worker = AutomationWorker(self, next_dir)
        
        # Connect the worker signals directly to update the UI
        # These are the key connections to fix the progress bar issue
        worker.signals.file_progress.connect(
            lambda filename, step, percent: self.batch_dialog.update_file_progress(next_dir, filename, step, percent)
        )
        
        worker.signals.log_message.connect(
            lambda msg: self.batch_dialog.add_log_message(f"[{os.path.basename(next_dir)}] {msg}")
        )
        
        worker.signals.finished.connect(
            lambda: self.on_batch_directory_completed(next_dir, True)
        )
        
        worker.signals.error.connect(
            lambda msg: self.on_batch_directory_error(next_dir, msg)
        )
        
        # Store reference to the current worker
        self.current_batch_worker = worker
        
        # Start the worker
        print(f"Starting worker for directory: {next_dir}")
        QThreadPool.globalInstance().start(worker)
        
        # Update overall progress
        completed = len([d for d in self.batch_dialog.directory_status if 
                        self.batch_dialog.directory_status[d]['status'] in ['completed', 'failed', 'skipped']])
        total = len(self.batch_dialog.directory_status)
        self.batch_dialog.update_overall_progress(completed, total)

    def on_batch_directory_completed(self, directory, success):
        """Handle completion of a batch directory."""
        # Update status
        status = 'completed' if success else 'failed'
        details = "Processing completed successfully" if success else "Processing failed"
        
        # Log the completion status
        self.status_bar.showMessage(f"Directory {os.path.basename(directory)} {status}")
        
        # Update status in dialog
        self.batch_dialog.update_directory_status(
            directory, status, 100, details
        )
        
        # Clean up
        self.current_batch_directory = None
        self.current_batch_worker = None
        
        # Update overall progress
        completed = len([d for d in self.batch_dialog.directory_status if 
                        self.batch_dialog.directory_status[d]['status'] in ['completed', 'failed', 'skipped']])
        total = len(self.batch_dialog.directory_status)
        self.batch_dialog.update_overall_progress(completed, total)
        
        # Process next directory
        self.process_next_batch_directory()

    def on_batch_directory_error(self, directory, error):
        """Handle error in batch directory processing."""
        # Update status
        self.batch_dialog.update_directory_status(
            directory, 'failed', 0, f"Error: {error}", error
        )
        
        # Clean up
        self.current_batch_directory = None
        self.current_batch_worker = None
        
        # Update overall progress
        completed = len([d for d in self.batch_dialog.directory_status if 
                        self.batch_dialog.directory_status[d]['status'] in ['completed', 'failed', 'skipped']])
        total = len(self.batch_dialog.directories)
        self.batch_dialog.update_overall_progress(completed, total)
        
        # Process next directory
        self.process_next_batch_directory()

    def perform_ms_baseline_correction(self):
        """Apply baseline correction to all ion traces in the MS data."""
        # Check if we have MS data
        if not hasattr(self.data_handler, 'current_directory_path') or not self.data_handler.current_directory_path:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        
        try:
            # Get baseline parameters from the UI
            params = self.parameters_frame.get_parameters()
            baseline_params = params['baseline']
            
            # Debug info about baseline parameters
            print(f"MS Baseline correction with parameters: {baseline_params}")
            
            # Create progress dialog
            progress_dialog = QProgressDialog("Applying baseline correction to MS data...", "Cancel", 0, 100, self)
            progress_dialog.setWindowTitle("MS Baseline Correction")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setAutoClose(False)  # Don't auto close
            progress_dialog.setValue(0)
            progress_dialog.show()
            QApplication.processEvents()  # Ensure dialog appears
            
            # Use data handler method instead of direct access
            ms = self.data_handler.get_ms_data()
            ms_data = ms.data
            
            # Show dimensions debug info
            print(f"MS data shape: {ms_data.shape}")
            if ms_data.shape[1] == 0:
                QMessageBox.warning(self, "Error", "MS data has no m/z channels")
                progress_dialog.close()
                return
            
            # Import the worker
            from logic.ms_baseline_worker import MSBaselineCorrectionWorker
            
            # Create and configure worker
            worker = MSBaselineCorrectionWorker(ms_data, baseline_params)
            
            # Connect signals
            worker.signals.started.connect(lambda total: 
                self.status_bar.showMessage(f"Starting MS baseline correction for {total} m/z channels..."))
            
            worker.signals.progress.connect(lambda current, total: 
                self._update_ms_baseline_progress(current, total, progress_dialog))
            
            # Update finished signal connection to handle both arrays
            worker.signals.finished.connect(lambda corrected_ms, ms_baselines: 
                self._on_ms_baseline_correction_completed(ms, corrected_ms, ms_baselines, progress_dialog))
            
            worker.signals.error.connect(lambda msg: 
                self._on_ms_baseline_correction_error(msg, progress_dialog))
            
            worker.signals.log_message.connect(lambda msg: self.status_bar.showMessage(msg, 3000))
            
            # Connect cancel button
            progress_dialog.canceled.connect(lambda: setattr(worker, 'cancelled', True))
            
            # Start worker
            QThreadPool.globalInstance().start(worker)
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            error_msg = f"Failed to start MS baseline correction: {str(e)}\n\n{traceback_str}"
            print(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def _update_ms_baseline_progress(self, current, total, dialog):
        """Update MS baseline correction progress with additional logging."""
        try:
            progress_percent = int(current * 100 / total)
            dialog.setValue(progress_percent)
            
            # Only log at intervals to avoid console spam
            if current % 100 == 0 or current == total:
                print(f"MS Baseline progress: {current}/{total} ({progress_percent}%)")
            
            # Keep UI responsive
            QApplication.processEvents()
        except Exception as e:
            print(f"Error updating progress: {str(e)}")

    def _on_ms_baseline_correction_completed(self, ms_object, corrected_data, ms_baselines, dialog):
        """Handle completion of MS baseline correction."""
        dialog.close()
        
        try:
            # Store the corrected MS data and baselines as attributes of the app
            self.original_ms_data = ms_object.data.copy()
            self.corrected_ms_data = corrected_data
            self.ms_baselines = ms_baselines
            
            # Calculate corrected and uncorrected TIC
            self.original_tic_data = {
                'x': ms_object.xlabels.copy(),
                'y': np.sum(self.original_ms_data, axis=1)
            }
            
            self.corrected_tic_data = {
                'x': ms_object.xlabels.copy(),
                'y': np.sum(self.corrected_ms_data, axis=1)
            }
            
            self.tic_baseline = {
                'x': ms_object.xlabels.copy(),
                'y': np.sum(ms_baselines, axis=1)
            }
            
            # Set a flag to indicate we have corrected data
            self.has_corrected_ms_data = True
            
            # Update status
            self.status_bar.showMessage("MS baseline correction completed. Use the MS options to toggle corrected data.")
            
            # Add a toggle checkbox to the parameters frame's baseline section
            if not hasattr(self.parameters_frame, 'use_corrected_ms_data_check'):
                # Create checkbox to toggle between original and corrected data
                self.parameters_frame.use_corrected_ms_data_check = QCheckBox("Use Baseline-Corrected MS Data")
                self.parameters_frame.use_corrected_ms_data_check.setChecked(True)
                self.parameters_frame.use_corrected_ms_data_check.setToolTip("Toggle between original and baseline-corrected MS data")
                
                # Add it to the baseline section of parameters frame
                baseline_layout = self.parameters_frame.baseline_group.layout()
                if baseline_layout:
                    baseline_layout.addWidget(self.parameters_frame.use_corrected_ms_data_check)
                
                # Connect signal to update function
                self.parameters_frame.use_corrected_ms_data_check.toggled.connect(self._toggle_corrected_ms_data)
            else:
                # Just make sure it's visible and checked
                self.parameters_frame.use_corrected_ms_data_check.setVisible(True)
                self.parameters_frame.use_corrected_ms_data_check.setChecked(True)
            
            # Ask if user wants to apply the corrected data now
            reply = QMessageBox.question(self, "Apply Corrected Data", 
                "MS baseline correction completed. Would you like to use the corrected data now?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # Apply corrected data and refresh the view
                self._toggle_corrected_ms_data(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying corrected MS data: {str(e)}")

    def _toggle_corrected_ms_data(self, use_corrected):
        """Toggle between original and corrected MS data."""
        if not hasattr(self, 'has_corrected_ms_data') or not self.has_corrected_ms_data:
            print("No corrected MS data available")
            return
        
        try:
            # Use data handler method instead of direct access
            ms = self.data_handler.get_ms_data()
            
            if use_corrected:
                # Replace with corrected data
                print(f"Applying corrected MS data with shape {self.corrected_ms_data.shape}")
                ms.data = self.corrected_ms_data.copy()
                self.status_bar.showMessage("Using baseline-corrected MS data")
                
                # Update the TIC display - show both TIC and baseline
                if hasattr(self, 'corrected_tic_data') and hasattr(self.plot_frame, 'plot_tic'):
                    self.plot_frame.plot_tic(
                        self.corrected_tic_data['x'],
                        self.corrected_tic_data['y'],
                        show_baseline=True,
                        baseline_x=self.tic_baseline['x'],
                        baseline_y=self.tic_baseline['y']
                    )
            else:
                # Restore original data
                print(f"Restoring original MS data with shape {self.original_ms_data.shape}")
                ms.data = self.original_ms_data.copy()
                self.status_bar.showMessage("Using original MS data")
                
                # Also restore original TIC display without baseline
                if hasattr(self, 'original_tic_data') and hasattr(self.plot_frame, 'plot_tic'):
                    self.plot_frame.plot_tic(
                        self.original_tic_data['x'],
                        self.original_tic_data['y']
                    )
    
        except Exception as e:
            error_msg = f"Error toggling MS data: {str(e)}"
            self.status_bar.showMessage(error_msg)
            print(f"Error toggling MS data: {str(e)}")
    
    def show_detector_selection_dialog(self):
        """Show dialog for selecting which detector channel to use."""
        # Check if we have data loaded
        if not hasattr(self.data_handler, 'current_directory_path') or not self.data_handler.current_directory_path:
            QMessageBox.warning(self, "No Data Loaded", "Please load a data file first.")
            return
        
        # Get available detectors
        detectors = self.data_handler.get_available_detectors()
        if not detectors:
            QMessageBox.warning(self, "No Detectors", "No detector channels found in the current data.")
            return
        
        # Get current detector
        current_detector = self.data_handler.current_detector
        
        # Create dialog directly without importing
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Detector Channel")
        dialog.resize(300, 150)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add info label
        info_label = QLabel("Select which detector channel to display:")
        layout.addWidget(info_label)
        
        # Create detector dropdown
        detector_combo = QComboBox()
        detector_combo.addItems(detectors)
        if current_detector in detectors:
            index = detectors.index(current_detector)
            detector_combo.setCurrentIndex(index)
        layout.addWidget(detector_combo)
        
        # Add note about reloading
        note_label = QLabel("Note: Changing the detector will reload the current data.")
        note_label.setStyleSheet("color: #666;")
        layout.addWidget(note_label)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        result = dialog.exec()
        
        # If accepted, update detector and reload data
        if result == QDialog.Accepted:
            selected_detector = detector_combo.currentText()
            if selected_detector != current_detector:
                # Store existing peak data if available
                had_integrated_peaks = hasattr(self.plot_frame, 'integrated_peaks') and self.plot_frame.integrated_peaks
                integration_results = None
                if had_integrated_peaks:
                    # Save existing integration results
                    integration_results = {
                        'peaks': self.plot_frame.integrated_peaks,
                        'x_peaks': self.plot_frame.x_peaks,
                        'y_peaks': self.plot_frame.y_peaks,
                        'baseline_peaks': self.plot_frame.baseline_peaks
                    }
                
                # Update detector and reload data
                self.data_handler.current_detector = selected_detector
                self.status_bar.showMessage(f"Changed detector to {selected_detector}")
                
                # Reload current data with new detector
                if hasattr(self, 'current_directory_path') and self.current_directory_path:
                    # Don't clear peak data before loading the file
                    # (this is the key fix - we'll reapply it after loading)
                    self.on_file_selected(self.current_directory_path)
                    
                    # Reapply integration results if they existed
                    if had_integrated_peaks and integration_results:
                        self.plot_frame.shade_integration_areas(integration_results)

    def show_export_settings_dialog(self):
        """Show the export settings dialog."""
        dialog = ExportSettingsDialog(self)
        dialog.exec()