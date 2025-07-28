from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFrame, QTreeWidget, QTreeWidgetItem, QProgressBar,
    QFileDialog, QMessageBox, QApplication, QDialog
)
from PySide6.QtCore import Qt, Signal, QThread, Slot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np
import os

# Add the MSToolkit import
try:
    from ms_toolkit.api import MSToolkit
    HAS_MSTOOLKIT = True
except ImportError:
    HAS_MSTOOLKIT = False

# Add this import at the top
from ui.dialogs.ms_options_dialog import MSOptionsDialog

class MSFrame(QWidget):
    """Frame for MS library search and peak identification tools."""
    
    # Add a signal for search completion
    search_completed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(350)
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        
        # Create MS tools
        self._create_ms_tools()
        
        # Initialize MSToolkit if available
        self.ms_toolkit = None
        self.library_loaded = False
        self.models_loaded = False
        
        # Initialize default search options
        self.search_options = {
            'search_method': 'vector',
            'hybrid_method': 'auto',  # Default hybrid method
            'extraction_method': 'apex',
            'range_points': 5,
            'range_time': 0.05,
            'tic_weight': True,
            'subtract_enabled': True,
            'subtraction_method': 'min_tic',
            'subtract_weight': 0.1,
            'similarity': 'composite',
            'weighting': 'NIST_GC',
            'unmatched': 'keep_all',
            'intensity_power': 0.6,
            'top_n': 5,
            'top_k_clusters': 1  # NEW: Add cluster option
        }
        
        # Add initial library load message
        if HAS_MSTOOLKIT:
            self._init_mstoolkit()
        else:
            self._show_toolkit_missing_message()
        
        # Add this attribute to track if theme has been applied
        self.theme_applied = False

    def apply_theme(self):
        """Apply the current theme to MS plots."""
        # Check if we have a parent with theme colors
        if not hasattr(self, 'ms_fig') or not hasattr(self, 'ms_ax'):
            return
            
        # Try to get theme colors from parent
        parent = self.parent()
        while parent and not hasattr(parent, 'matplotlib_theme_colors'):
            parent = parent.parent()
            
        if not parent or not hasattr(parent, 'matplotlib_theme_colors'):
            return
            
        colors = parent.matplotlib_theme_colors
        if not colors:
            return
            
        # Apply colors to the figure and axes
        self.ms_fig.patch.set_facecolor(colors['background'])
        self.ms_ax.set_facecolor(colors['axes'])
        
        # Update spines
        for spine in self.ms_ax.spines.values():
            spine.set_color(colors['edge'])
        
        # Update ticks and labels
        self.ms_ax.tick_params(axis='both', colors=colors['ticks'], which='both',
                            labelcolor=colors['label'], reset=True)
        
        if self.ms_ax.xaxis.label:
            self.ms_ax.xaxis.label.set_color(colors['label'])
        if self.ms_ax.yaxis.label:
            self.ms_ax.yaxis.label.set_color(colors['label'])
        if self.ms_ax.title:
            self.ms_ax.title.set_color(colors['text'])
        
        # Update grid
        # self.ms_ax.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        
        # Mark theme as applied
        self.theme_applied = True
        
        # Force redraw
        self.ms_canvas.draw()
    
    def _create_ms_tools(self):
        """Create the MS library search and peak identification tools."""
        # Title label
        ms_label = QLabel("Mass Spectrometry Tools")
        ms_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.addWidget(ms_label)
        
        # Retention Time Input Frame
        rt_frame = QFrame()
        rt_layout = QVBoxLayout(rt_frame)
        
        # Retention Time Label and Entry
        rt_label = QLabel("Retention Time (RT):")
        rt_layout.addWidget(rt_label)
        
        self.rt_entry = QLineEdit()
        self.rt_entry.setPlaceholderText("Enter RT (or double right-click on plot)")
        rt_layout.addWidget(self.rt_entry)
        
        # Add m/z shift entry
        mz_shift_layout = QHBoxLayout()
        mz_shift_label = QLabel("m/z Shift:")
        mz_shift_layout.addWidget(mz_shift_label)
        
        self.mz_shift_entry = QLineEdit()
        self.mz_shift_entry.setPlaceholderText("0")
        self.mz_shift_entry.setMaximumWidth(60)
        mz_shift_layout.addWidget(self.mz_shift_entry)
        
        # Add a button to apply shift and redisplay spectrum
        redisplay_button = QPushButton("Apply")
        redisplay_button.setMaximumWidth(60)
        redisplay_button.clicked.connect(self._redisplay_with_shift)
        mz_shift_layout.addWidget(redisplay_button)
        
        # Add search button
        self.search_button = QPushButton("Search Library")
        self.search_button.setEnabled(False)  # Disabled until library is loaded
        self.search_button.clicked.connect(self._search_current_spectrum)
        mz_shift_layout.addWidget(self.search_button)
        
        # Add configure button
        self.configure_button = QPushButton("Configure...")
        self.configure_button.clicked.connect(self._configure_search_options)
        mz_shift_layout.addWidget(self.configure_button)
        
        mz_shift_layout.addStretch()
        rt_layout.addLayout(mz_shift_layout)
        
        # Instruction label
        instruction_label = QLabel("Double right-click on the plot to view spectrum")
        instruction_label.setStyleSheet("color: #666666; font-style: italic;")
        rt_layout.addWidget(instruction_label)
        
        self.layout.addWidget(rt_frame)
        
        # Embed Mass Spectrum Plot
        ms_plot_frame = QFrame()
        ms_plot_layout = QVBoxLayout(ms_plot_frame)
        
        self.ms_fig = Figure(figsize=(3, 2))
        self.ms_ax = self.ms_fig.add_subplot(111)
        self.ms_ax.set_xlabel("m/z", fontsize=8)
        self.ms_ax.set_xlim((1, 150))
        self.ms_ax.set_yticks([])
        self.ms_ax.tick_params(axis='x', top=False) # Add this line to remove top ticks
        self.ms_ax.margins(0.05)
        self.ms_canvas = FigureCanvasQTAgg(self.ms_fig)
        ms_plot_layout.addWidget(self.ms_canvas)
        
        self.layout.addWidget(ms_plot_frame)
        
        # MS Library Search Results Tree
        tree_frame = QFrame()
        tree_layout = QVBoxLayout(tree_frame)
        
        tree_label = QLabel("MS Library Search Results:")
        tree_layout.addWidget(tree_label)
        
        self.ms_search_tree = QTreeWidget()
        self.ms_search_tree.setHeaderLabels(["Compound Name", "Match Score"])
        self.ms_search_tree.setColumnWidth(0, 200)
        tree_layout.addWidget(self.ms_search_tree)
        
        self.layout.addWidget(tree_frame)
        
        # Store the current spectrum data for redisplay with shift
        self.current_spectrum = None
    
    def _redisplay_with_shift(self):
        """Redisplay the current spectrum with the applied m/z shift."""
        if not hasattr(self, 'current_spectrum') or self.current_spectrum is None:
            return
        
        # Get the new m/z shift value
        try:
            mz_shift = int(self.mz_shift_entry.text() or 0)
        except ValueError:
            mz_shift = 0
            self.mz_shift_entry.setText("0")
        
        # Update the MSToolkit's mz_shift parameter if toolkit is available
        if hasattr(self, 'ms_toolkit') and self.ms_toolkit:
            self.ms_toolkit.mz_shift = mz_shift
            print(f"Updated MSToolkit m/z shift to {mz_shift}")
        
        # Check if we have mz and intensities values to redisplay directly 
        # (without needing to go back to the RT)
        if 'mz' in self.current_spectrum and 'intensities' in self.current_spectrum:
            # Just redisplay the existing spectrum with the new shift
            mz = self.current_spectrum['mz']
            intensities = self.current_spectrum['intensities']
            title = self.current_spectrum.get('label', "Current Spectrum")
            
            # Apply shift to display only (original data is preserved)
            display_mz = mz + mz_shift
            
            # Redraw the spectrum with the shifted m/z values
            self.ms_ax.clear()
            if np.max(intensities) > 0:
                normalized_intensities = intensities / np.max(intensities)
            else:
                normalized_intensities = intensities
                
            self.ms_ax.vlines(display_mz, ymin=0, ymax=normalized_intensities, color="blue")
            self.ms_ax.set_xlabel("m/z", fontsize=8)
            
            # Set x-axis range to show relevant data
            if len(display_mz) > 0:
                min_mz = max(1, min(display_mz) - 5)
                max_mz = max(display_mz) + 5
                self.ms_ax.set_xlim((min_mz, max_mz))
            else:
                self.ms_ax.set_xlim((1 + mz_shift, 150 + mz_shift))
                
            self.ms_ax.set_yticks([])
            
            # Add appropriate title
            if 'rt' in self.current_spectrum:
                self.ms_ax.set_title(f"{title} (m/z shift: {mz_shift})", fontsize=10)
            else:
                self.ms_ax.set_title(f"{title} (m/z shift: {mz_shift})", fontsize=10)
                
            self.ms_fig.tight_layout()
            self.ms_canvas.draw()
        elif 'rt' in self.current_spectrum:
            # Use the original method if rt is available
            self.view_spectrum_at_rt(self.current_spectrum['rt'])
        else:
            # Handle the case where neither direct data nor RT is available
            self.ms_ax.clear()
            self.ms_ax.text(75, 0.5, "Cannot redisplay spectrum: no data available", 
                           horizontalalignment='center',
                           fontsize=9, color='red')
            self.ms_canvas.draw()
    
    def view_spectrum_at_rt(self, retention_time):
        """Display the mass spectrum at the given retention time.
        
        Args:
            retention_time (float): Retention time in minutes
        """
        # Import numpy at the function level to ensure it's available
        import numpy as np
        
        # Update the RT entry
        self.rt_entry.setText(f"{retention_time:.3f}")
        
        # Initialize current_spectrum if None
        if self.current_spectrum is None:
            self.current_spectrum = {}
        
        # Store the current RT for redisplay
        self.current_spectrum['rt'] = retention_time
        
        # Get m/z shift value
        try:
            mz_shift = int(self.mz_shift_entry.text() or 0)
        except ValueError:
            mz_shift = 0
            self.mz_shift_entry.setText("0")
        
        # In a real implementation, this would extract the spectrum from the MS data
        try:
            if hasattr(self, 'extract_spectrum_func') and self.extract_spectrum_func:
                # If we have a reference to a function that can extract real MS data
                spectrum = self.extract_spectrum_func(retention_time)
                mz_values = spectrum['mz']
                intensities = spectrum['intensities']
                
                # Store the extracted spectrum data
                self.current_spectrum['mz'] = mz_values
                self.current_spectrum['intensities'] = intensities
            else:
                # Generate a simulated spectrum for demonstration
                max_mz = 150
                mz_values = np.arange(max_mz) + 1
                intensities = np.zeros(max_mz)
                
                # Create some random peaks - more realistic than completely random
                for i in range(5):  # 5 major fragment groups
                    center = np.random.randint(15, max_mz - 15)  # Peak center
                    width = np.random.randint(3, 8)  # Peak width
                    
                    # Create a peak cluster with a central larger peak
                    intensities[center] = np.random.random() * 0.8 + 0.2  # Main peak
                    
                    # Add some smaller related peaks
                    for j in range(1, width):
                        if center + j < max_mz:
                            intensities[center + j] = intensities[center] * (0.6 ** j) * (np.random.random() * 0.3 + 0.7)
                        if center - j >= 0:
                            intensities[center - j] = intensities[center] * (0.6 ** j) * (np.random.random() * 0.3 + 0.7)
                
                # Store the generated spectrum data
                self.current_spectrum['mz'] = mz_values
                self.current_spectrum['intensities'] = intensities
            
            # Apply m/z shift to the display (not to the stored data)
            display_mz = mz_values + mz_shift
            
            # Normalize intensities for display
            if np.sum(intensities) > 0:
                display_intensities = intensities / np.max(intensities)
            else:
                display_intensities = intensities
            
            # Plot the spectrum
            self.ms_ax.clear()
            self.ms_ax.vlines(display_mz, ymin=0, ymax=display_intensities, color="blue")
            self.ms_ax.set_xlabel("m/z", fontsize=8)
            self.ms_ax.set_xlim((1 + mz_shift, 150 + mz_shift))
            self.ms_ax.set_yticks([])
            self.ms_ax.set_title(f"Mass Spectrum at RT: {retention_time:.3f} min", fontsize=10)
            self.ms_fig.tight_layout()
            self.ms_canvas.draw()
            
        except Exception as e:
            print(f"Error displaying spectrum: {str(e)}")
            # Show error in the plot
            self.ms_ax.clear()
            self.ms_ax.text(75, 0.5, f"Error displaying spectrum: {str(e)}", 
                           horizontalalignment='center',
                           fontsize=9, color='red')
            self.ms_canvas.draw()
    
    def plot_mass_spectrum(self, mz, intensities, title=None):
        """Plot a mass spectrum."""
        # Clear the previous plot
        self.ms_ax.clear()
        
        # Get the current m/z shift value
        try:
            mz_shift = int(self.mz_shift_entry.text() or 0)
        except ValueError:
            mz_shift = 0
            self.mz_shift_entry.setText("0")
        
        # Apply shift to display only (original data is preserved)
        display_mz = mz + mz_shift
        
        # Normalize intensities for display
        if np.max(intensities) > 0:
            normalized_intensities = intensities / np.max(intensities)
        else:
            normalized_intensities = intensities
        
        # Plot the spectrum using vlines with shifted m/z values
        self.ms_ax.vlines(display_mz, ymin=0, ymax=normalized_intensities, color="blue")
        
        # Add labels and title
        self.ms_ax.set_xlabel("m/z", fontsize=8)
        self.ms_ax.set_yticks([])
        
        # Set x-axis range to show relevant data
        if len(display_mz) > 0:
            min_mz = max(1, min(display_mz) - 5)
            max_mz = max(display_mz) + 5
            self.ms_ax.set_xlim((min_mz, max_mz))
        else:
            self.ms_ax.set_xlim((1 + mz_shift, 150 + mz_shift))
        
        # Add title if provided
        if title:
            self.ms_ax.set_title(f"{title} (m/z shift: {mz_shift})", fontsize=10)
        
        # Update the plot
        self.ms_fig.tight_layout()
        self.ms_canvas.draw()
        
        # Store the current spectrum data for potential searching
        self.current_spectrum = {
            'mz': mz,  # Store original unshifted values
            'intensities': intensities,
            'label': title or "Current Spectrum"
        }
        
        # Parse RT from title if available (like "Peak 5 (RT=4.123)")
        import re
        if title and "RT=" in title:
            rt_match = re.search(r'RT=(\d+\.\d+)', title)
            if rt_match:
                self.current_spectrum['rt'] = float(rt_match.group(1))
        
        # Enable search button if library is loaded
        if hasattr(self, 'library_loaded') and self.library_loaded and hasattr(self, 'search_button'):
            self.search_button.setEnabled(True)
        
        # Apply theme if it hasn't been applied yet
        if not getattr(self, 'theme_applied', False):
            self.apply_theme()
    
    def set_extract_spectrum_function(self, func):
        """Set a function that can extract real MS spectra.
        
        Args:
            func: A function that takes a retention time and returns a spectrum dict
                 with 'mz' and 'intensities' keys
        """
        self.extract_spectrum_func = func

    def set_current_spectrum(self, mz, intensities, label="Current Spectrum", rt=None):
        """Set the current spectrum for display and potential searching."""
        self.current_spectrum = {
            'mz': mz,
            'intensities': intensities,
            'label': label
        }
        
        # Store RT if provided
        if rt is not None:
            self.current_spectrum['rt'] = rt
        else:
            # Try to extract RT from label
            import re
            if label and "RT=" in label:
                rt_match = re.search(r'RT=(\d+\.\d+)', label)
                if rt_match:
                    self.current_spectrum['rt'] = float(rt_match.group(1))
        
        # Update UI to reflect the current spectrum
        if hasattr(self, 'current_spectrum_label'):
            self.current_spectrum_label.setText(f"Current: {label}")
        
        # Enable search button if library is loaded
        if hasattr(self, 'library_loaded') and self.library_loaded and hasattr(self, 'search_button'):
            self.search_button.setEnabled(True)
    
    def update_ms_results(self, results):
        """Update the MS library search results tree.
        
        Args:
            results: List of tuples (compound_name, match_score)
        """
        self.ms_search_tree.clear()
        
        # Add results to the tree
        for compound_name, match_score in results:
            item = QTreeWidgetItem()
            item.setText(0, compound_name)
            item.setText(1, f"{match_score:.3f}")
            self.ms_search_tree.addTopLevelItem(item)
    
    def _search_current_spectrum(self):
        """Search the library for the current spectrum."""
        if not self.ms_toolkit or not self.library_loaded or not self.current_spectrum:
            QMessageBox.warning(self, "Error", "Library not loaded or no spectrum available")
            return
        
        try:
            # Extract the spectrum from current_spectrum
            mz_values = self.current_spectrum.get('mz', [])
            intensities = self.current_spectrum.get('intensities', [])
            
            if len(mz_values) == 0 or len(intensities) == 0:
                QMessageBox.warning(self, "Error", "No valid spectrum data available")
                return
            
            # Read shift value from UI and update toolkit
            try:
                mz_shift = int(self.mz_shift_entry.text() or 0)
                self.ms_toolkit.mz_shift = mz_shift  # Make sure toolkit has current value
            except ValueError:
                mz_shift = 0
                self.mz_shift_entry.setText("0")
            
            # Create the query spectrum WITHOUT applying the shift
            # (the toolkit will apply it internally during search)
            query_spectrum = [(m, i) for m, i in zip(mz_values, intensities)]
            
            # Set status message
            self.status_label.setText("Searching library...")
            QApplication.processEvents()
            
            # Get options
            options = self.search_options
            
            # Search based on selected method
            try:
                if options['search_method'] == 'vector':
                    # Configure vector search parameters
                    vector_results = self.ms_toolkit.search_vector(
                        query_spectrum, 
                        top_n=options['top_n'],
                        composite=(options['similarity'] == 'composite'),
                        weighting_scheme=options['weighting'],
                        unmatched_method=options['unmatched'],
                        top_k_clusters=options.get('top_k_clusters', 1)  # This is included here
                    )
                    w2v_results = []  # No W2V results if vector search is selected
                elif options['search_method'] == 'hybrid':
                    # Configure hybrid search parameters
                    hybrid_results = self.ms_toolkit.search_hybrid(
                        query_spectrum,
                        method=options.get('hybrid_method', 'auto'),
                        top_n=options['top_n'],
                        intensity_power=options['intensity_power'],
                        weighting_scheme=options['weighting'],
                        composite=(options['similarity'] == 'composite'),
                        unmatched_method=options['unmatched'],
                        top_k_clusters=options.get('top_k_clusters', 1)
                    )
                    # For hybrid, show results as one group
                    vector_results = hybrid_results
                    w2v_results = []
                else:
                    # Configure Word2Vec search parameters
                    w2v_results = self.ms_toolkit.search_w2v(
                        query_spectrum, 
                        top_n=options['top_n'],
                        intensity_power=options['intensity_power'],
                        top_k_clusters=options.get('top_k_clusters', 1)  # This is included here
                    )
                    vector_results = []  # No vector results if W2V search is selected
                    
            except RuntimeError as e:
                if "Preselector must be loaded first" in str(e):
                    # Handle missing preselector gracefully
                    self.status_label.setText("No preselector loaded - searches will be slow")
                    self.status_label.setStyleSheet("color: #FF6600;")  # Orange warning color
                    
                    # Show warning dialog with option to continue
                    reply = QMessageBox.question(
                        self, 
                        "No Preselector Loaded",
                        "No preselector model is loaded. This will make library searches extremely slow for large libraries.\n\n"
                        "Would you like to:\n"
                        "- Continue anyway (slow search)\n"
                        "- Cancel and load a preselector first",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    
                    if reply == QMessageBox.No:
                        self.status_label.setText("Search cancelled - please load a preselector")
                        return
                    
                    # If user chooses to continue, show a different message
                    self.status_label.setText("Searching entire library (this may take a while)...")
                    QApplication.processEvents()
                    
                    # For now, we can't proceed without modifying ms-toolkit
                    # So show an error message explaining the limitation
                    QMessageBox.warning(
                        self,
                        "Search Not Possible",
                        "Cannot perform search without a preselector model.\n\n"
                        "Please:\n"
                        "1. Train a preselector using the ms-toolkit API, or\n"
                        "2. Select an existing preselector file when setting up the library"
                    )
                    self.status_label.setText("Search cancelled - preselector required")
                    self.status_label.setStyleSheet("color: #990000;")
                    return
                else:
                    # Re-raise other RuntimeErrors
                    raise
            
            # Clear previous results
            self.ms_search_tree.clear()
            
            # Add vector-based results if available
            if vector_results:
                vector_parent = QTreeWidgetItem(self.ms_search_tree)
                vector_parent.setText(0, "Vector-based Results")
                vector_parent.setExpanded(True)
                
                for compound_name, score in vector_results:
                    item = QTreeWidgetItem(vector_parent)
                    item.setText(0, compound_name)
                    item.setText(1, f"{score:.3f}")
            
            # Add Word2Vec-based results if available
            if w2v_results:
                w2v_parent = QTreeWidgetItem(self.ms_search_tree)
                w2v_parent.setText(0, "Word2Vec-based Results")
                w2v_parent.setExpanded(True)
                
                for compound_name, score in w2v_results:
                    item = QTreeWidgetItem(w2v_parent)
                    item.setText(0, compound_name)
                    item.setText(1, f"{score:.3f}")
            
            # Update status
            self.status_label.setText("Search complete")
            self.status_label.setStyleSheet("color: #009900;")
            
            # Emit the search completed signal
            self.search_completed.emit()
            
        except Exception as e:
            self.status_label.setText(f"Search error: {str(e)}")
            self.status_label.setStyleSheet("color: #990000;")
            QMessageBox.warning(self, "Search Error", str(e))
    
    def _configure_search_options(self):
        """Open the MS search options dialog."""
        dialog = MSOptionsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.search_options = dialog.get_options()
            print("MS search options updated:", self.search_options)
    
    def _init_mstoolkit(self):
        """Initialize the MSToolkit."""
        try:
            # Get initial m/z shift from UI if available
            initial_mz_shift = 0
            if hasattr(self, 'mz_shift_entry'):
                try:
                    initial_mz_shift = int(self.mz_shift_entry.text() or 0)
                except (ValueError, AttributeError):
                    pass
            
            # Create a toolkit instance with default parameters
            self.ms_toolkit = MSToolkit(mz_shift=initial_mz_shift, show_ui=False)
            
            # Add library setup button
            self.setup_library_button = QPushButton("Setup MS Library")
            self.setup_library_button.clicked.connect(self._setup_library)
            self.layout.addWidget(self.setup_library_button)
            
            # Add progress bar (initially hidden)
            self.progress_frame = QFrame()
            progress_layout = QVBoxLayout(self.progress_frame)
            
            self.progress_label = QLabel("Loading library...")
            progress_layout.addWidget(self.progress_label)
            
            self.progress_bar = QProgressBar()
            progress_layout.addWidget(self.progress_bar)
            
            self.layout.addWidget(self.progress_frame)
            self.progress_frame.hide()
            
            self.status_label = QLabel("MS Library not loaded")
            self.status_label.setStyleSheet("color: #990000;")
            self.layout.addWidget(self.status_label)
            
        except Exception as e:
            print(f"Error initializing MSToolkit: {str(e)}")
            self._show_toolkit_error(str(e))
    
    def _show_toolkit_missing_message(self):
        """Display a message when the MSToolkit is not available."""
        missing_label = QLabel("MSToolkit is not installed.")
        missing_label.setStyleSheet("color: #990000; font-weight: bold;")
        
        install_label = QLabel("Install with: pip install ms-toolkit-nrel")
        install_label.setStyleSheet("font-family: monospace;")
        
        self.layout.addWidget(missing_label)
        self.layout.addWidget(install_label)
    
    def _show_toolkit_error(self, error_message):
        """Display an error message related to the MSToolkit."""
        error_label = QLabel(f"MSToolkit Error: {error_message}")
        error_label.setStyleSheet("color: #990000;")
        error_label.setWordWrap(True)
        self.layout.addWidget(error_label)
    
    def _setup_library(self):
        """Set up the MS library and embeddings."""
        try:
            # Set default library folder path
            default_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'library')
            os.makedirs(default_library_path, exist_ok=True)
            
            # Create file dialog with improved filters for both .txt and .json
            library_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select MS Library File", 
                default_library_path,  # Start in the library folder
                "Library Files (*.txt *.json);;All Files (*.*)"  # Show txt and json by default
            )
            
            if not library_path:
                self.status_label.setText("Library setup cancelled")
                return
                
            # Configure progress display
            if hasattr(self, 'progress_frame'):
                self.progress_frame.show()
            else:
                # Add progress frame if not already added
                self.progress_frame = QFrame()
                progress_layout = QVBoxLayout(self.progress_frame)
                
                self.progress_label = QLabel("Loading library...")
                progress_layout.addWidget(self.progress_label)
                
                self.progress_bar = QProgressBar()
                progress_layout.addWidget(self.progress_bar)
                
                self.layout.addWidget(self.progress_frame)
            
            # Show progress bar
            self.progress_bar.setValue(0)
            
            # Initialize paths for models
            preselector_path = None
            w2v_path = None
            
            # First check for models in the 'models' subfolder (automatic detection)
            models_dir = os.path.join(os.path.dirname(library_path), 'models')
            if not os.path.exists(models_dir):
                models_dir = default_library_path
            
            # Always require preselector model selection
            preselector_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Preselector Model (Required)", 
                models_dir,
                "Preselector Files (*.pkl);;All Files (*.*)"
            )
            
            if not preselector_path:
                self.status_label.setText("Library setup cancelled - preselector required")
                self.progress_frame.hide()
                return
            
            self.progress_label.setText(f"Using preselector: {os.path.basename(preselector_path)}")
            
            # Always require Word2Vec model selection
            w2v_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Word2Vec Model (Required)", 
                models_dir,
                "Word2Vec Files (*.bin *.model);;All Files (*.*)"
            )
            
            if not w2v_path:
                self.status_label.setText("Library setup cancelled - Word2Vec model required")
                self.progress_frame.hide()
                return
            
            self.progress_label.setText(f"Using Word2Vec model: {os.path.basename(w2v_path)}")
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create a thread to load the library
            self.load_thread = LibraryLoadThread(
                self.ms_toolkit, 
                library_path, 
                cache_path=cache_dir,
                preselector_path=preselector_path,
                w2v_path=w2v_path
            )
            
            # Connect signals
            self.load_thread.progress_update.connect(self._update_progress)
            self.load_thread.finished.connect(self._on_library_load_finished)
            
            # Start the thread
            self.load_thread.start()
            
            # Update status
            self.status_label.setText(f"Loading library from {os.path.basename(library_path)}...")
            
        except Exception as e:
            error_message = f"Error setting up library: {str(e)}"
            self._show_toolkit_error(error_message)
    
    def _update_progress(self, message, value):
        """Update progress bar and message."""
        self.progress_label.setText(message)
        self.progress_bar.setValue(value)
        QApplication.processEvents()
    
    def _on_library_load_finished(self, success, message):
        """Handle library load completion."""
        self.progress_frame.hide()
        
        if success:
            self.library_loaded = True
            self.models_loaded = True
            self.status_label.setText(f"MS Library loaded: {message}")
            self.status_label.setStyleSheet("color: #009900; font-weight: bold;")
            
            # Enable the search button if we have it
            if hasattr(self, 'search_button'):
                self.search_button.setEnabled(True)
            
            # Emit signal for any listeners
            self.search_completed.emit()
        else:
            error_msg = f"Error: {message}"
            self.status_label.setText(error_msg)
            self.status_label.setStyleSheet("color: #990000;")
            print(error_msg)  # Also print to console for troubleshooting
            QMessageBox.warning(self, "Library Load Error", message)

class LibraryLoadThread(QThread):
    """Thread for loading MS library and models."""
    
    # Signals
    progress_update = Signal(str, int)
    finished = Signal(bool, str)
    
    def __init__(self, ms_toolkit, library_path, cache_path=None, 
                 preselector_path=None, w2v_path=None):
        super().__init__()
        self.ms_toolkit = ms_toolkit
        self.library_path = library_path
        self.cache_path = cache_path
        self.preselector_path = preselector_path
        self.w2v_path = w2v_path
    
    def run(self):
        """Run the library loading process."""
        try:
            # Simplified progress callback that only expects a single value (0.0-1.0)
            def progress_callback(value):
                # Convert to percentage (0-100)
                percent = int(value * 100)
                message = f"Loading library... {percent}%"
                self.progress_update.emit(message, percent)
            
            # Step 1: Load the library
            self.progress_update.emit("Loading library...", 0)
            
            # Determine which path to use as the main source
            file_path = None
            json_path = None
            
            if self.library_path.lower().endswith('.txt'):
                file_path = self.library_path
                json_path = self.cache_path
            elif self.library_path.lower().endswith('.json'):
                json_path = self.library_path
            
            # BYPASS THE ORIGINAL PARSER AND JUST LOAD DIRECTLY FROM JSON
            if json_path and os.path.exists(json_path):
                self.progress_update.emit(f"Loading library from {os.path.basename(json_path)}...", 10)
                
                # Direct JSON loading without using the toolkit's parse function
                import json
                from ms_toolkit.models import Compound
                
                with open(json_path, 'r') as f:
                    compounds_json = json.load(f)
                    
                total = len(compounds_json)
                library = {}
                
                for i, (name, data) in enumerate(compounds_json.items()):
                    # Update progress periodically
                    if i % max(1, int(total/100)) == 0:
                        percent = int(20 + (i/total) * 60)  # Scale from 20% to 80%
                        self.progress_update.emit(f"Processing compounds... ({i}/{total})", percent)
                    
                    # Convert JSON to Compound
                    library[name] = Compound.from_json(data)
                
                # Assign to toolkit
                self.ms_toolkit.library = library
                
            else:
                # We need to parse from text - custom implementation
                self.progress_update.emit("Error: JSON file not found", 0)
                self.finished.emit(False, "JSON file not found. Please select a valid JSON library file.")
                return
            
            # Step 2: Vectorize the library
            self.progress_update.emit("Vectorizing library...", 80)
            self.ms_toolkit.vectorize_library()
            
            # Step 3: Load preselector
            self.progress_update.emit("Setting up preselector...", 90)
            
            # First, try user-selected preselector
            if self.preselector_path and os.path.exists(self.preselector_path):
                self.ms_toolkit.load_preselector(self.preselector_path)
            else:
                # Try to find existing preselectors in library directory
                preselector_dir = os.path.dirname(self.library_path)
                possible_preselectors = [
                    os.path.join(preselector_dir, "NIST14_GMM_optimized.pkl"),
                    os.path.join(preselector_dir, "preselector.pkl"),
                    os.path.join(preselector_dir, "NIST14_CHO_GMM_preselector.pkl"),
                    os.path.join(preselector_dir, "NIST14_CHO_preselector.pkl")
                ]
                
                preselector_loaded = False
                for preselector_path in possible_preselectors:
                    if os.path.exists(preselector_path):
                        self.ms_toolkit.load_preselector(preselector_path)
                        preselector_loaded = True
                        break
                
                if not preselector_loaded:
                    # No preselector found - warn user but continue
                    import warnings
                    warning_msg = (
                        "No preselector model found! Library searches will be extremely slow. "
                        "Please train a preselector using the ms-toolkit API or select an existing one."
                    )
                    warnings.warn(warning_msg, UserWarning)
                    print(f"WARNING: {warning_msg}")
            
            # Step 4: Load or train Word2Vec model (simplified)
            self.progress_update.emit("Setting up Word2Vec model...", 95)
            if self.w2v_path and os.path.exists(self.w2v_path):
                self.ms_toolkit.load_w2v(self.w2v_path)
            else:
                # Skip training for now, it's not essential for basic functionality
                self.progress_update.emit("Skipping Word2Vec training (optional)...", 98)
            
            # Complete
            library_name = os.path.basename(self.library_path)
            self.progress_update.emit(f"Setup complete: {library_name}", 100)
            self.finished.emit(True, library_name)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)  # Print to console for debugging
            self.finished.emit(False, error_msg)