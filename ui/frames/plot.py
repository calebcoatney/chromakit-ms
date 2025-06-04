from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QMenu, QApplication
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QCursor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

class PlotFrame(QWidget):
    """Frame for displaying chromatogram and TIC plots."""
    
    # Signal to notify when a point has been selected on the plot
    point_selected = Signal(float)
    
    # New signal for MS spectrum viewing
    ms_spectrum_requested = Signal(float)
    
    # Add this new signal for peak-specific spectrum requests
    peak_spectrum_requested = Signal(int)  # Peak index for specific peak extraction
    
    # Signal for MS search requests
    ms_search_requested = Signal(int)  # Peak index for MS library search
    
    # Signal for edit assignment requests
    edit_assignment_requested = Signal(int)  # Peak index for assignment editing
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set the layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create figure and canvas
        self.figure = plt.figure(figsize=(6, 6), dpi=100)
        
        # Set up the axes - Swap positions so TIC is on top
        self.tic_ax = self.figure.add_subplot(211)  # Top plot for TIC
        self.chromatogram_ax = self.figure.add_subplot(212, sharex=self.tic_ax)  # Bottom plot for chromatogram with shared x axis
        
        # Add figure to canvas
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Add widgets to layout
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        
        # Initialize data for storage
        self.chromatogram_data = None
        self.tic_data = None
        
        # Replace existing mouse click connection with new connections
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        
        # Track click information
        self._click_time = 0
        self._last_click_x = None
        self._is_right_button = False
        
        # Set up initial empty plots
        self._setup_empty_plots()
        
    def _setup_empty_plots(self):
        """Set up initial empty plots."""
        self.tic_ax.clear()
        self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
        self.tic_ax.set_xlabel('Time (min)')
        self.tic_ax.set_ylabel('Intensity')
        self.tic_ax.grid(True, linestyle='--', alpha=0.7)
        
        self.chromatogram_ax.clear()
        self.chromatogram_ax.set_title('Processed Chromatogram')
        self.chromatogram_ax.set_xlabel('Time (min)')
        self.chromatogram_ax.set_ylabel('Intensity')
        self.chromatogram_ax.grid(True, linestyle='--', alpha=0.7)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_chromatogram(self, data, show_corrected=False):
        """Plot the chromatogram data."""
        # Clear any existing peak highlights
        self._clear_peak_highlights()
        
        # Store data
        self.chromatogram_data = data
        
        # Clear previous plot
        self.chromatogram_ax.clear()
        
        # Extract data from dictionary with different naming
        x = data['x']
        
        # Handle both old and new formats
        if 'original_y' in data and 'smoothed_y' in data:
            # New format
            original_y = data['original_y']
            smoothed_y = data['smoothed_y']
            baseline = data['baseline_y']
            corrected_y = data['corrected_y']
        elif 'y_raw' in data and 'y_corrected' in data:
            # Old format
            original_y = data.get('y', data['y_raw'])
            smoothed_y = data['y_raw']
            baseline = data['baseline_y']
            corrected_y = data['y_corrected']
        else:
            # Fallback
            original_y = data.get('y', np.array([]))
            smoothed_y = original_y
            baseline = data.get('baseline_y', np.zeros_like(original_y))
            corrected_y = original_y - baseline
        
        peaks_x = data['peaks_x']
        peaks_y = data['peaks_y']
        
        # Choose which data to display based on show_corrected flag
        if show_corrected:
            # Show corrected signal (baseline will be at zero)
            main_y = corrected_y
            baseline_to_show = np.zeros_like(baseline)
            self.chromatogram_ax.set_title('Baseline-Corrected Chromatogram')
        else:
            # Show smoothed signal with calculated baseline
            main_y = smoothed_y
            baseline_to_show = baseline
            self.chromatogram_ax.set_title('Preprocessed Chromatogram')
        
        # Plot the main chromatogram
        self.chromatogram_ax.plot(x, main_y, 'b-', linewidth=1, label='Chromatogram')
        
        # Plot baseline
        self.chromatogram_ax.plot(x, baseline_to_show, 'r--', linewidth=1, alpha=0.7, label='Baseline')
        
        # Plot peaks using orange star markers (only if not empty)
        if len(peaks_x) > 0:
            if not show_corrected:
                # Adjust peak heights to be on the raw/smoothed data
                peak_indices = np.searchsorted(x, peaks_x)
                peak_baselines = baseline[peak_indices]
                adjusted_peaks_y = peaks_y + peak_baselines
                # Use star markers with orange color and appropriate size
                self.chromatogram_ax.plot(peaks_x, adjusted_peaks_y, '*', 
                                         color='orange', markersize=6, alpha=0.8, 
                                         label='Peaks')
            else:
                self.chromatogram_ax.plot(peaks_x, peaks_y, '*', 
                                         color='orange', markersize=6, alpha=0.8, 
                                         label='Peaks')
        
        # Set labels
        self.chromatogram_ax.set_xlabel('Time (min)')
        self.chromatogram_ax.set_ylabel('Intensity')
        self.chromatogram_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        self.chromatogram_ax.legend()
        
        # Force autoscale to see all data
        if len(x) > 0 and len(main_y) > 0:
            # Set x limits to show all data
            x_min, x_max = np.min(x), np.max(x)
            x_padding = (x_max - x_min) * 0.02  # 2% padding
            self.chromatogram_ax.set_xlim(x_min - x_padding, x_max + x_padding)
            
            # Set y limits to show all data with some padding
            y_min = np.min(baseline_to_show)
            y_max = np.max(main_y) * 1.1  # Add 10% padding at the top
            self.chromatogram_ax.set_ylim(y_min, y_max)
        
        # Also update TIC x-axis to match
        if hasattr(self, 'tic_ax') and self.tic_ax is not None:
            if len(x) > 0:
                self.tic_ax.set_xlim(self.chromatogram_ax.get_xlim())
        
        # Adjust layout and draw
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_tic(self, x, y, show_baseline=False, baseline_x=None, baseline_y=None):
        """Plot the Total Ion Chromatogram (TIC) data."""
        # Store data
        self.tic_data = {'x': x, 'y': y}
        
        # Clear previous plot
        self.tic_ax.clear()
        
        # Plot the TIC
        self.tic_ax.plot(x, y, 'g-', linewidth=1)
        
        # Plot baseline if available
        if show_baseline and baseline_x is not None and baseline_y is not None:
            # Add the baseline as a red dashed line
            self.tic_ax.plot(baseline_x, baseline_y, 'r--', linewidth=1, alpha=0.7, label='Baseline')
        
        # Set labels and title
        self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
        self.tic_ax.set_xlabel('Time (min)')
        self.tic_ax.set_ylabel('Intensity')
        self.tic_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set appropriate limits to see the data
        if len(x) > 0 and len(y) > 0:
            # Set x limits to show all data
            x_min, x_max = np.min(x), np.max(x)
            x_padding = (x_max - x_min) * 0.02  # 2% padding
            self.tic_ax.set_xlim(x_min - x_padding, x_max + x_padding)
            
            # Set y limits to show all data with some padding
            if show_baseline and baseline_y is not None:
                y_min = np.min(baseline_y)
            else:
                y_min = 0
            y_max = np.max(y) * 1.1  # Add 10% padding at the top
            self.tic_ax.set_ylim(y_min, y_max)
        
        # If chromatogram plot exists, synchronize x limits
        if hasattr(self, 'chromatogram_ax') and self.chromatogram_ax is not None:
            self.chromatogram_ax.set_xlim(self.tic_ax.get_xlim())
        
        # Add legend if baseline is shown
        if show_baseline and baseline_x is not None:
            self.tic_ax.legend()
        
        # Adjust layout and draw
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _on_plot_click(self, event):
        """Handle mouse button press on the plot."""
        if event.inaxes is None:
            return
            
        # Store click information
        import time
        self._click_time = time.time()
        self._last_click_x = event.xdata
        self._is_right_button = event.button == 3  # Right button is 3 in matplotlib
        
        # Only handle right-clicks on the chromatogram axis specifically
        if self._is_right_button and event.inaxes == self.chromatogram_ax and hasattr(self, 'integrated_peaks') and self.integrated_peaks:
            # Check if the click is near any integrated peak
            found_peak = False
            
            for i, peak in enumerate(self.integrated_peaks):
                if hasattr(peak, 'retention_time'):
                    distance = abs(peak.retention_time - event.xdata)
                    
                    # If within bounds or close to apex
                    if ((hasattr(peak, 'start_time') and hasattr(peak, 'end_time') and 
                        peak.start_time <= event.xdata <= peak.end_time) or distance < 0.05):
                        # Show context menu and immediately reset click tracking to prevent
                        # further event handling for this click
                        found_peak = True
                        self._show_peak_context_menu(i, event)
                        
                        # Reset click tracking immediately to prevent further processing of this click
                        self._last_click_x = None
                        self._is_right_button = False
                        break
            
            # If no peak was found under the click, we'll return without resetting
            # letting the regular right-click behavior proceed
            if found_peak:
                return
    
    def _on_button_release(self, event):
        """Handle mouse button release on the plot."""
        if event.inaxes is None or self._last_click_x is None:
            return
            
        # Calculate time since press
        import time
        release_time = time.time()
        click_duration = release_time - self._click_time
        
        # For very quick clicks (press & release < 0.3s)
        if click_duration < 0.3:
            # Handle left click for peak highlight and marker display
            if not self._is_right_button:
                # Get the x-value at click position
                x_value = self._last_click_x
                
                # First check if we have integrated peaks and if the click is near a peak
                if hasattr(self, 'integrated_peaks') and self.integrated_peaks:
                    # Find the closest peak
                    closest_peak = None
                    min_distance = float('inf')
                    
                    for i, peak in enumerate(self.integrated_peaks):
                        # Check if click is within peak bounds or close to apex
                        if hasattr(peak, 'retention_time'):
                            distance = abs(peak.retention_time - x_value)
                            
                            # If within bounds or close to apex
                            if (hasattr(peak, 'start_time') and hasattr(peak, 'end_time') and 
                                peak.start_time <= x_value <= peak.end_time) or distance < 0.05:
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_peak = i
                    
                    # If a peak was found nearby, highlight it
                    if closest_peak is not None:
                        self._highlight_selected_peak(self.integrated_peaks[closest_peak])
                        return
                
                # If no peak was found or no integrated peaks, just add the marker line
                for ax in [self.chromatogram_ax, self.tic_ax]:
                    # Remove previous lines
                    for line in ax.get_lines():
                        if line.get_label() == '_selected_point':
                            line.remove()
                    
                    # Add new line
                    ax.axvline(x=x_value, color='r', alpha=0.5, linestyle=':', label='_selected_point')
                
                self.canvas.draw()
            
            # Handle right click for MS spectrum (simplified to always show spectrum at that point)
            elif self._is_right_button and hasattr(self, '_last_right_click_time'):
                # Check if this is a double click (< 0.5s between clicks)
                if release_time - self._last_right_click_time < 0.5:
                    clicked_rt = event.xdata
                    
                    # Clear any existing highlights
                    self._clear_peak_highlights()
                    
                    # Signal to show MS at this retention time (no special extraction)
                    self.ms_spectrum_requested.emit(clicked_rt)
                    
                    # Update RT box
                    self.point_selected.emit(clicked_rt)
        
        # Store last right click time for double-click detection
        if self._is_right_button:
            self._last_right_click_time = release_time
        
        # Reset click tracking
        self._last_click_x = None
        self._is_right_button = False

    def _clear_peak_highlights(self):
        """Clear any peak highlights."""
        # Store current axis limits
        chromatogram_xlim = self.chromatogram_ax.get_xlim()
        chromatogram_ylim = self.chromatogram_ax.get_ylim()
        tic_xlim = self.tic_ax.get_xlim()
        tic_ylim = self.tic_ax.get_ylim()
        
        for ax in [self.chromatogram_ax, self.tic_ax]:
            # Remove peak indicators
            for line in ax.get_lines():
                if line.get_label() == '_selected_peak':
                    line.remove()
            
            # Remove any annotations
            for txt in ax.texts:
                if hasattr(txt, 'peak_annotation') and txt.peak_annotation:
                    txt.remove()
        
        # Remove any highlighted fill_between
        for collection in self.chromatogram_ax.collections:
            if hasattr(collection, 'peak_highlight') and collection.peak_highlight:
                collection.remove()
        
        # Restore the axis limits
        self.chromatogram_ax.set_xlim(chromatogram_xlim)
        self.chromatogram_ax.set_ylim(chromatogram_ylim)
        self.tic_ax.set_xlim(tic_xlim)
        self.tic_ax.set_ylim(tic_ylim)
        
        self.canvas.draw_idle()

    def _highlight_selected_peak(self, peak):
        """Highlight the selected peak with enhanced information."""
        # Store current axis limits before clearing highlights
        chromatogram_xlim = self.chromatogram_ax.get_xlim()
        chromatogram_ylim = self.chromatogram_ax.get_ylim()
        tic_xlim = self.tic_ax.get_xlim()
        tic_ylim = self.tic_ax.get_ylim()
        
        # First clear any existing highlights
        self._clear_peak_highlights()
        
        # Determine text color based on peak issues (prioritize saturation over convolution)
        text_color = 'purple' if hasattr(peak, 'is_saturated') and peak.is_saturated else \
                    'red' if hasattr(peak, 'is_convoluted') and peak.is_convoluted else 'g'
        
        # Add annotation and highlight for each plot
        for ax in [self.chromatogram_ax, self.tic_ax]:
            # Create a more detailed annotation that includes compound ID if available
            annotation_text = f"Peak {peak.peak_number}\nRT: {peak.retention_time:.3f}"
            
            # Add compound ID if MS search has been performed
            if hasattr(peak, 'compound_id') and peak.compound_id and peak.compound_id != f"Unknown ({peak.retention_time:.3f})":
                annotation_text += f"\n{peak.compound_id}"
                if hasattr(peak, 'Qual') and peak.Qual is not None:
                    annotation_text += f" ({peak.Qual:.3f})"
            
            # Add saturation warning if detected
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                annotation_text += "\n⚠️ DETECTOR SATURATION"
                if hasattr(peak, 'saturation_level'):
                    annotation_text += f"\n • Max: {peak.saturation_level:.2e}"
            
            # Add quality information if available
            elif hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                annotation_text += "\n⚠️ Possible convolution"
                
                if hasattr(peak, 'quality_issues') and peak.quality_issues:
                    for issue in peak.quality_issues:
                        annotation_text += f"\n • {issue}"
                else:
                    # Fallback to raw metrics
                    if hasattr(peak, 'asymmetry') and peak.asymmetry is not None:
                        annotation_text += f"\n • Asym: {peak.asymmetry:.2f}"
                    if hasattr(peak, 'spectral_coherence') and peak.spectral_coherence is not None:
                        annotation_text += f"\n • Coh: {peak.spectral_coherence:.2f}"
            
            # Add the annotation
            text = ax.text(peak.retention_time, ax.get_ylim()[1]*0.9, 
                    annotation_text, 
                    color=text_color, fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
            
            # Tag the text object so we can identify it later
            text.peak_annotation = True
        
        # If the peak is part of integration results, highlight its area
        if hasattr(self, 'x_peaks') and hasattr(self, 'y_peaks') and hasattr(self, 'baseline_peaks'):
            for i, integrated_peak in enumerate(self.integrated_peaks):
                if integrated_peak.peak_number == peak.peak_number:
                    # Increase alpha for this peak's shading
                    # First clear any previous highlighted areas
                    for collection in self.chromatogram_ax.collections:
                        if hasattr(collection, 'peak_highlight') and collection.peak_highlight:
                            collection.remove()
                    
                    # Add a new shaded area with higher alpha
                    collection = self.chromatogram_ax.fill_between(
                        self.x_peaks[i],
                        self.y_peaks[i],
                        self.baseline_peaks[i],
                        alpha=0.8,  # Higher alpha for highlight
                        color='green',
                        label=f'_highlight_peak_{peak.peak_number}'
                    )
                    collection.peak_highlight = True
                    
                    # Send signal to update MS spectrum with this peak's data
                    self.peak_spectrum_requested.emit(integrated_peak.peak_number - 1)  # Convert to zero-based index
                    
                    # Also update RT entry box
                    self.point_selected.emit(peak.retention_time)
                    
                    break
        
        # Restore the axis limits before redrawing
        self.chromatogram_ax.set_xlim(chromatogram_xlim)
        self.chromatogram_ax.set_ylim(chromatogram_ylim)
        self.tic_ax.set_xlim(tic_xlim) 
        self.tic_ax.set_ylim(tic_ylim)
        
        # Redraw the canvas
        self.canvas.draw_idle()
    
    def shade_integration_areas(self, integration_results):
        """Shade the areas under peaks based on integration results with distinctly different colors."""
        # Check if we have the required data
        if not integration_results or not self.chromatogram_data:
            return
        
        # Store the integrated peaks and related data for highlighting
        self.integrated_peaks = integration_results.get('peaks', [])
        self.x_peaks = integration_results.get('x_peaks', [])
        self.y_peaks = integration_results.get('y_peaks', [])
        self.baseline_peaks = integration_results.get('baseline_peaks', [])
        
        # Store current limits before clearing
        xlim = self.chromatogram_ax.get_xlim() if not self.chromatogram_ax.get_autoscalex_on() else None
        ylim = self.chromatogram_ax.get_ylim() if not self.chromatogram_ax.get_autoscaley_on() else None
        
        # Clear previous plot - redraw without the peak markers
        self.chromatogram_data['peaks_x'] = np.array([])
        self.chromatogram_data['peaks_y'] = np.array([])
        self.plot_chromatogram(self.chromatogram_data, show_corrected=False)
        
        # Import matplotlib's cm and create a colormap with good differentiation
        import matplotlib.cm as cm
        
        # Get a distinct colormap from matplotlib
        cmap = cm.get_cmap('tab20')
        
        # Shade each integrated peak area with a unique color
        for i in range(len(self.x_peaks)):
            # Use prime number step to give good color separation
            color_idx = (i * 7) % 20
            
            # Get the color from the colormap
            color = cmap(color_idx / 20)
            
            # Check if peak is flagged as saturated or convoluted
            peak = self.integrated_peaks[i]
            is_saturated = hasattr(peak, 'is_saturated') and peak.is_saturated
            is_convoluted = hasattr(peak, 'is_convoluted') and peak.is_convoluted
            
            # Use different styling based on flags (prioritize saturation)
            if is_saturated:
                linewidth = 2.0
                edgecolor = 'purple'
            elif is_convoluted:
                linewidth = 1.5
                edgecolor = 'red'
            else:
                linewidth = 0
                edgecolor = None
            
            # Shade the peak area
            collection = self.chromatogram_ax.fill_between(
                self.x_peaks[i],
                self.y_peaks[i],
                self.baseline_peaks[i],
                alpha=0.4,
                color=color,
                label=f'Peak {i+1}' if i < 10 else '_nolegend_',
                edgecolor=edgecolor,
                linewidth=linewidth
            )
            # Store peak number for later identification
            collection.peak_number = i + 1
            
            # Add warning marker for saturated or convoluted peaks
            if is_saturated:
                # Use diamond marker for saturation
                self.chromatogram_ax.plot(
                    peak.retention_time,
                    self.y_peaks[i][len(self.y_peaks[i])//2],  # Approximate apex height
                    marker='D',  # Diamond marker for saturation
                    markersize=12,
                    markerfacecolor='purple',
                    markeredgecolor='black',
                    alpha=0.8,
                    linestyle='None'
                )
            elif is_convoluted:
                # Use triangle marker for convolution
                self.chromatogram_ax.plot(
                    peak.retention_time,
                    self.y_peaks[i][len(self.y_peaks[i])//2],  # Approximate apex height
                    marker='^',  # Triangle marker
                    markersize=12,
                    markerfacecolor='red',
                    markeredgecolor='black',
                    alpha=0.8,
                    linestyle='None'
                )
        
        # Restore previous limits
        if xlim is not None:
            self.chromatogram_ax.set_xlim(xlim)
        if ylim is not None:
            self.chromatogram_ax.set_ylim(ylim)
        
        # Update the plot
        self.canvas.draw_idle()
    
    def update_annotations(self):
        """Update peak annotations based on compound identifications."""
        # This will be called when compounds are identified
        # If you don't need this functionality yet, just make it a pass-through
        pass
    
    def clear_peak_data(self):
        """Clear all peak-related data when a new file is loaded."""
        # First clear any highlights
        self._clear_peak_highlights()
        
        # Clear stored peak data
        if hasattr(self, 'integrated_peaks'):
            delattr(self, 'integrated_peaks')
        if hasattr(self, 'x_peaks'):
            delattr(self, 'x_peaks')
        if hasattr(self, 'y_peaks'):
            delattr(self, 'y_peaks')
        if hasattr(self, 'baseline_peaks'):
            delattr(self, 'baseline_peaks')
    
    def _show_peak_context_menu(self, peak_index, event):
        """Show context menu for a peak."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QClipboard
        
        # Get the peak
        peak = self.integrated_peaks[peak_index]
        
        # Create context menu
        menu = QMenu()
        
        # Add actions
        view_ms_action = menu.addAction(f"View MS Spectrum for Peak {peak.peak_number}")
        view_ms_action.triggered.connect(lambda: self.peak_spectrum_requested.emit(peak_index))
        
        # Add search action if the peak has an MS spectrum
        search_ms_action = menu.addAction("Search MS Library")
        search_ms_action.triggered.connect(lambda: self._request_ms_search(peak_index))
        
        # Add edit assignment action
        edit_assignment_action = menu.addAction("Edit Compound Assignment...")
        edit_assignment_action.triggered.connect(lambda: self._edit_peak_assignment(peak_index))
        
        menu.addSeparator()
        
        # Copy peak info to clipboard
        copy_rt_action = menu.addAction(f"Copy RT: {peak.retention_time:.3f}")
        copy_rt_action.triggered.connect(lambda: QApplication.clipboard().setText(f"{peak.retention_time:.3f}"))
        
        # If compound ID is assigned, add option to copy it
        if hasattr(peak, 'compound_id') and peak.compound_id and peak.compound_id != f"Unknown ({peak.retention_time:.3f})":
            copy_name_action = menu.addAction(f"Copy Compound Name: {peak.compound_id}")
            copy_name_action.triggered.connect(lambda: QApplication.clipboard().setText(peak.compound_id))
            
            # If we have match score (Qual), show and allow copy
            if hasattr(peak, 'Qual') and peak.Qual is not None:
                copy_qual_action = menu.addAction(f"Copy Match Score: {peak.Qual:.3f}")
                copy_qual_action.triggered.connect(lambda: QApplication.clipboard().setText(f"{peak.Qual:.3f}"))
        
        # Add separator and more actions for peaks with quality issues
        if hasattr(peak, 'is_saturated') and peak.is_saturated:
            menu.addSeparator()
            saturation_action = menu.addAction("⚠️ Detector Saturation Detected")
            saturation_action.setEnabled(False)  # Just for display
        
        if hasattr(peak, 'is_convoluted') and peak.is_convoluted:
            menu.addSeparator()
            convolution_action = menu.addAction("⚠️ Possible Peak Convolution")
            convolution_action.setEnabled(False)  # Just for display
        
        # Show the menu at current cursor position instead of using matplotlib event coordinates
        cursor_pos = QCursor.pos()  # Get global cursor position
        menu.exec_(cursor_pos)      # Show menu at cursor position

    def _request_ms_search(self, peak_index):
        """Request MS search for a peak."""
        # First request spectrum
        self.peak_spectrum_requested.emit(peak_index)
        
        # We'll implement a signal for MS search in the app class
        self.ms_search_requested.emit(peak_index)
    
    def _edit_peak_assignment(self, peak_index):
        """Open dialog to edit peak compound assignment."""
        # Emit signal to request edit dialog
        self.edit_assignment_requested.emit(peak_index)