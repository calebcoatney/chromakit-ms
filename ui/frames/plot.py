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

    # Signal for RT table assignment requests
    rt_assignment_requested = Signal(int)  # Peak index for RT assignment
    
    # Signal for adding peaks to RT table
    add_to_rt_table_requested = Signal(int)  # Peak index for adding to RT table
    
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
        self.aligned_tic_data = None  # New attribute for aligned TIC data
        self.tic_alignment_info = None  # Store alignment metadata
        
        # Replace existing mouse click connection with new connections
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        
        # Track click information
        self._click_time = 0
        self._last_click_x = None
        self._is_right_button = False
        
        # Set up initial empty plots
        self._setup_empty_plots()

    def apply_theme(self):
        """Apply the current theme to the plots."""
        parent = self.parent()
        while parent and not hasattr(parent, 'matplotlib_theme_colors'):
            parent = parent.parent()
        
        if not parent or not parent.matplotlib_theme_colors:
            return

        colors = parent.matplotlib_theme_colors
        
        self.figure.patch.set_facecolor(colors['background'])
        
        for ax in [self.chromatogram_ax, self.tic_ax]:
            if ax:
                ax.set_facecolor(colors['axes'])
                for spine in ax.spines.values():
                    spine.set_color(colors['edge'])
                
                ax.tick_params(axis='both', colors=colors['ticks'], which='both', labelcolor=colors['label'])
                
                if ax.title: ax.title.set_color(colors['text'])
                if ax.xaxis.label: ax.xaxis.label.set_color(colors['label'])
                if ax.yaxis.label: ax.yaxis.label.set_color(colors['label'])
                
                ax.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])

        self.canvas.draw_idle()

    def clear_tic(self):
        """Clear the TIC plot and redraw the canvas."""
        try:
            if hasattr(self, 'tic_ax') and self.tic_ax is not None:
                # Check if the axis is still valid and part of the figure
                if self.tic_ax in self.figure.axes:
                    self.tic_ax.clear()
                    self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
                    self.tic_ax.set_xlabel('Time (min)')
                    self.tic_ax.set_ylabel('Intensity')
                    self.tic_ax.grid(True, linestyle='--', alpha=0.7)
                    self.figure.tight_layout()
                    self.canvas.draw_idle()
                else:
                    # Axis is stale - recreate it
                    print("TIC axis is stale, recreating...")
                    self._recreate_tic_axis()
            else:
                # No valid TIC axis exists
                print("No valid TIC axis found in clear_tic")
                
        except Exception as e:
            print(f"Error in clear_tic: {type(e).__name__}: {e}")
            # Try to recreate the axis structure
            try:
                self._recreate_tic_axis()
            except Exception as recreate_error:
                print(f"Failed to recreate TIC axis in clear_tic: {recreate_error}")
    
    def _recreate_tic_axis(self):
        """Helper method to recreate the TIC axis safely."""
        try:
            # Clear figure and recreate both axes
            self.figure.clear()
            self.tic_ax = self.figure.add_subplot(211)  # Top plot for TIC
            self.chromatogram_ax = self.figure.add_subplot(212, sharex=self.tic_ax)  # Bottom plot
            
            # Set basic TIC properties
            self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
            self.tic_ax.set_xlabel('Time (min)')
            self.tic_ax.set_ylabel('Intensity')
            self.tic_ax.grid(True, linestyle='--', alpha=0.7)
            
            # Restore chromatogram data if available
            if hasattr(self, 'chromatogram_data') and self.chromatogram_data is not None:
                self.plot_chromatogram(self.chromatogram_data, new_file=False)
            
            self.figure.tight_layout()
            self.canvas.draw_idle()
            print("Successfully recreated TIC axis structure")
            
        except Exception as e:
            print(f"Failed to recreate TIC axis: {e}")

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
        self.apply_theme() # Apply theme on initialization
        
    def plot_chromatogram(self, data, show_corrected=False, new_file=True):
        """
        Plot the chromatogram data.
        
        Args:
            data: Chromatogram data dictionary
            show_corrected: Whether to show baseline-corrected data
            new_file: Whether this is a new file (True) or just parameter updates (False)
        """
        # Get theme colors if available from the main app
        colors = None
        if hasattr(self.parent(), 'matplotlib_theme_colors'):
            colors = self.parent().matplotlib_theme_colors
        # Store current view limits BEFORE doing anything else
        x_was_autoscaled = self.chromatogram_ax.get_autoscalex_on()
        y_was_autoscaled = self.chromatogram_ax.get_autoscaley_on()
        
        previous_xlim = self.chromatogram_ax.get_xlim() if not x_was_autoscaled else None
        previous_ylim = self.chromatogram_ax.get_ylim() if not y_was_autoscaled else None
        
        # Clear any existing peak highlights
        self._clear_peak_highlights()
        
        # Store data
        self.chromatogram_data = data
        
        # Apply theme to ensure it's up-to-date
        self.apply_theme()
        
        # Clear previous plot
        self.chromatogram_ax.clear()
        
        # Extract data
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
        
        # Get fitted curves if available
        fitted_curves = data.get('fitted_curves', [])
        
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
        
        # Plot detected peaks and shoulders using metadata
        if 'peak_metadata' in data and len(data['peak_metadata']) > 0:
            for meta in data['peak_metadata']:
                # FIXED: Adjust marker position based on current view mode
                marker_x = meta['x']
                marker_y = meta['y']  # Original detected peak height
                
                # If showing uncorrected signal, need to add baseline value at this position
                if not show_corrected and 'baseline_y' in data:
                    # Find nearest point in the baseline array
                    idx = np.abs(data['x'] - marker_x).argmin()
                    if idx < len(data['baseline_y']):
                        # When showing uncorrected view, add baseline height to marker position
                        marker_y += data['baseline_y'][idx]
                
                if meta['type'] == 'peak':
                    self.chromatogram_ax.plot(marker_x, marker_y, marker='*', color='orange', 
                                         markersize=8, alpha=0.9, 
                                         label='Peak' if 'Peak' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None)
                elif meta['type'] == 'shoulder':
                    self.chromatogram_ax.plot(marker_x, marker_y, marker='^', color='red', 
                                         markersize=8, alpha=0.9, 
                                         label='Shoulder' if 'Shoulder' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None)
        
        # Set labels and add legend
        self.chromatogram_ax.set_xlabel('Time (min)')
        self.chromatogram_ax.set_ylabel('Intensity')
        self.chromatogram_ax.grid(True, linestyle='--', alpha=0.7)
        self.chromatogram_ax.legend()
        
        # Only calculate and apply new view limits for new files
        # or if previous limits weren't set
        if new_file or previous_xlim is None or previous_ylim is None:
            if len(x) > 0 and len(main_y) > 0:
                x_min, x_max = np.min(x), np.max(x)
                x_padding = (x_max - x_min) * 0.02  # 2% padding
                x_full_range = [x_min - x_padding, x_max + x_padding]
                
                y_min = np.min(baseline_to_show)
                y_max_raw = np.max(main_y)
                # Add 10% padding at the top - handle negative values correctly
                if y_max_raw >= 0:
                    y_max = y_max_raw * 1.1  # Positive: multiply to increase
                else:
                    y_max = y_max_raw * 0.9  # Negative: multiply by 0.9 to make less negative
                
                # Only apply MS range adjustment for new files
                if new_file and self.tic_data is not None and 'x' in self.tic_data:
                    ms_x = self.tic_data['x']
                    if len(ms_x) > 0:
                        ms_x_min, ms_x_max = np.min(ms_x), np.max(ms_x)
                        
                        # Mask FID x to MS x range
                        mask = (x >= ms_x_min) & (x <= ms_x_max)
                        if np.any(mask):
                            y_max_in_ms_range = np.max(main_y[mask])
                            # 10% padding - handle negative values correctly
                            if y_max_in_ms_range >= 0:
                                y_max = y_max_in_ms_range * 1.1
                            else:
                                y_max = y_max_in_ms_range * 0.9  # Negative: make less negative
                            print(f"Solvent delay detected at {ms_x_min:.2f} minutes, rescaling FID ylim to {y_max:.2f}")
                
                # Apply the calculated limits
                self.chromatogram_ax.set_xlim(x_full_range)
                self.chromatogram_ax.set_ylim(y_min, y_max)
                
                # If TIC plot exists, sync x limits for new files
                if new_file and hasattr(self, 'tic_ax') and self.tic_ax is not None:
                    self.tic_ax.set_xlim(x_full_range)
        else:
            # Restore previous view limits for parameter changes
            self.chromatogram_ax.set_xlim(previous_xlim)
            self.chromatogram_ax.set_ylim(previous_ylim)
        
        # Before returning, apply theme if available
        if colors:
            # Use the correct attributes for the figure and axes
            self.figure.patch.set_facecolor(colors['background'])
            
            # Apply to chromatogram axis
            self.chromatogram_ax.set_facecolor(colors['axes'])
            for spine in self.chromatogram_ax.spines.values():
                spine.set_color(colors['edge'])
            
            self.chromatogram_ax.tick_params(axis='both', colors=colors['ticks'], which='both',
                                        labelcolor=colors['label'], reset=True)
            
            if self.chromatogram_ax.xaxis.label:
                self.chromatogram_ax.xaxis.label.set_color(colors['label'])
            if self.chromatogram_ax.yaxis.label:
                self.chromatogram_ax.yaxis.label.set_color(colors['label'])
            if self.chromatogram_ax.title:
                self.chromatogram_ax.title.set_color(colors['text'])
            
            # Apply to TIC axis if it exists
            if hasattr(self, 'tic_ax') and self.tic_ax is not None:
                self.tic_ax.set_facecolor(colors['axes'])
                for spine in self.tic_ax.spines.values():
                    spine.set_color(colors['edge'])
                
                self.tic_ax.tick_params(axis='both', colors=colors['ticks'], which='both',
                                    labelcolor=colors['label'], reset=True)
                
                if self.tic_ax.xaxis.label:
                    self.tic_ax.xaxis.label.set_color(colors['label'])
                if self.tic_ax.yaxis.label:
                    self.tic_ax.yaxis.label.set_color(colors['label'])
                if self.tic_ax.title:
                    self.tic_ax.title.set_color(colors['text'])
        
      
        # Adjust layout and draw
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_tic(self, x, y, show_baseline=False, baseline_x=None, baseline_y=None, new_file=True):
        """
        Plot the Total Ion Chromatogram (TIC) data.
        
        Args:
            x, y: TIC data arrays
            show_baseline: Whether to show baseline
            baseline_x, baseline_y: Baseline data arrays
            new_file: Whether this is a new file (True) or just parameter updates (False)
        """
        # Robustly handle empty or missing data
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            self.clear_tic()
            return
        # Get theme colors if available from the main app
        colors = None
        if hasattr(self.parent(), 'matplotlib_theme_colors'):
            colors = self.parent().matplotlib_theme_colors
        
        # Store current view limits before doing anything
        x_was_autoscaled = self.tic_ax.get_autoscalex_on()
        y_was_autoscaled = self.tic_ax.get_autoscaley_on()
        
        previous_xlim = self.tic_ax.get_xlim() if not x_was_autoscaled else None
        previous_ylim = self.tic_ax.get_ylim() if not y_was_autoscaled else None
        
        # Store data
        self.tic_data = {'x': x, 'y': y}
        
        # Apply theme to ensure it's up-to-date
        self.apply_theme()

        # Clear previous plot
        self.tic_ax.clear()
        
        # Plot the TIC
        self.tic_ax.plot(x, y, 'g-', linewidth=1)
        
        # Plot baseline if available
        if show_baseline and baseline_x is not None and baseline_y is not None:
            self.tic_ax.plot(baseline_x, baseline_y, 'r--', linewidth=1, alpha=0.7, label='Baseline')
        
        # Set labels and title
        self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
        self.tic_ax.set_xlabel('Time (min)')
        self.tic_ax.set_ylabel('Intensity')
        self.tic_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only calculate and apply new view limits for new files
        # or if previous limits weren't set
        if new_file or previous_xlim is None or previous_ylim is None:
            if len(x) > 0 and len(y) > 0:
                x_min, x_max = np.min(x), np.max(x)
                x_padding = (x_max - x_min) * 0.02  # 2% padding
                x_full_range = [x_min - x_padding, x_max + x_padding]
                
                if show_baseline and baseline_y is not None:
                    y_min = np.min(baseline_y)
                else:
                    y_min = 0
                y_max_raw = np.max(y)
                # Add 10% padding at the top - handle negative values correctly
                if y_max_raw >= 0:
                    y_max = y_max_raw * 1.1  # Positive: multiply to increase
                else:
                    y_max = y_max_raw * 0.9  # Negative: multiply by 0.9 to make less negative
                
                # Apply the calculated limits
                self.tic_ax.set_xlim(x_full_range)
                self.tic_ax.set_ylim(y_min, y_max)
                
                # Sync chromatogram x axis for new files
                if new_file and hasattr(self, 'chromatogram_ax') and self.chromatogram_ax is not None:
                    self.chromatogram_ax.set_xlim(x_full_range)
        else:
            # Restore previous view limits for parameter changes
            self.tic_ax.set_xlim(previous_xlim)
            self.tic_ax.set_ylim(previous_ylim)
        
        # Add legend if baseline is shown
        if show_baseline and baseline_x is not None:
            self.tic_ax.legend()
        
        # Adjust layout and draw
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Apply theme if available
        if colors:
            self.figure.patch.set_facecolor(colors['background'])
            
            # Apply to both axes to ensure consistency
            for ax in [self.tic_ax, self.chromatogram_ax]:
                if ax is not None:
                    ax.set_facecolor(colors['axes'])
                    
                    for spine in ax.spines.values():
                        spine.set_color(colors['edge'])
                    
                    ax.tick_params(axis='both', colors=colors['ticks'], which='both',
                                labelcolor=colors['label'], reset=True)
                    
                    if ax.xaxis.label:
                        ax.xaxis.label.set_color(colors['label'])
                    if ax.yaxis.label:
                        ax.yaxis.label.set_color(colors['label'])
                    if ax.title:
                        ax.title.set_color(colors['text'])
            
            # Force complete redraw
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
            
            # Add compound ID with source information
            if hasattr(peak, 'compound_id') and peak.compound_id and peak.compound_id != f"Unknown ({peak.retention_time:.3f})":
                annotation_text += f"\n{peak.compound_id}"

                # Show assignment source and confidence
                if hasattr(peak, 'rt_assignment') and peak.rt_assignment:
                    # RT Table assignment
                    source = getattr(peak, 'rt_assignment_source', 'RT')
                    annotation_text += f" [{source}]"
                elif hasattr(peak, 'Qual') and peak.Qual is not None:
                    # MS library assignment
                    annotation_text += f" [MS: {peak.Qual:.3f}]"
                else:
                    # Manual or other assignment
                    annotation_text += " [Manual]"
            
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
                    
                    # FIXED: Check if we're showing corrected or original signal
                    showing_corrected = self.chromatogram_ax.get_title() == 'Baseline-Corrected Chromatogram'
                    
                    # Adjust the shading based on the current view mode
                    if showing_corrected:
                        # For corrected view: Use the stored y_peaks and zero baseline
                        shade_y = self.y_peaks[i]
                        shade_baseline = np.zeros_like(self.baseline_peaks[i])
                    else:
                        # For uncorrected view: Adjust y_peaks back to raw signal
                        shade_y = self.y_peaks[i] + self.baseline_peaks[i] 
                        shade_baseline = self.baseline_peaks[i]
                    
                    # Add a new shaded area with higher alpha
                    collection = self.chromatogram_ax.fill_between(
                        self.x_peaks[i],
                        shade_y,
                        shade_baseline,
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
        
        # Determine if we're showing corrected or original signal
        showing_corrected = self.chromatogram_ax.get_title() == 'Baseline-Corrected Chromatogram'
        
        # Clear previous plot - redraw without the peak markers, PRESERVING view state
        self.chromatogram_data['peaks_x'] = np.array([])
        self.chromatogram_data['peaks_y'] = np.array([])
        self.plot_chromatogram(self.chromatogram_data, show_corrected=showing_corrected, new_file=False)
        
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
            
            # Shade the peak area - adjust shading based on view mode
            if showing_corrected:
                # For corrected view: Use the stored y_peaks and zero baseline 
                shade_y = self.y_peaks[i]
                shade_baseline = np.zeros_like(self.baseline_peaks[i])
            else:
                # For uncorrected view: Adjust y_peaks back to raw signal
                shade_y = self.y_peaks[i] + self.baseline_peaks[i] 
                shade_baseline = self.baseline_peaks[i]
            
            collection = self.chromatogram_ax.fill_between(
                self.x_peaks[i],
                shade_y,
                shade_baseline,
                alpha=0.4,
                color=color,
                label=f'Peak {i+1}' if i < 10 else '_nolegend_',
                edgecolor=edgecolor,
                linewidth=linewidth
            )
            # Store peak number for later identification
            collection.peak_number = i + 1
            
            # Add warning marker for saturated or convoluted peaks - adjust position to match view
            marker_y = shade_y[len(shade_y)//2]  # Use middle point height
            
            if is_saturated:
                # Use diamond marker for saturation
                self.chromatogram_ax.plot(
                    peak.retention_time,
                    marker_y,  # Match the current view's peak height
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
                    marker_y,  # Match the current view's peak height
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
    
    def set_tic_data(self, x, y):
        """Store the TIC data without plotting it."""
        self.tic_data = {'x': x, 'y': y}

    def set_aligned_tic_data(self, aligned_time, aligned_signal, lag_seconds):
        """Store the aligned TIC data."""
        self.aligned_tic_data = {'x': aligned_time, 'y': aligned_signal}
        self.tic_alignment_info = {'lag_seconds': lag_seconds}
        print(f"Stored aligned TIC data with lag of {lag_seconds:.4f} seconds")

    def set_tic_visible(self, visible: bool):
        """Set the visibility of the TIC plot.
        
        Args:
            visible (bool): Whether to show the TIC plot
        """
        try:
            # Store current axis limits to preserve them
            chromatogram_xlim = self.chromatogram_ax.get_xlim()
            chromatogram_ylim = self.chromatogram_ax.get_ylim()
            
            # Set visibility of the TIC axis
            self.tic_ax.set_visible(visible)
            
            # When hiding TIC, make chromatogram plot take full height
            if not visible:
                # Store reference to old TIC axis before deleting
                old_tic_ax = self.tic_ax
                
                # Clear the figure completely to avoid axis corruption
                self.figure.clear()
                
                # Explicitly set the TIC axis to None to avoid stale references
                self.tic_ax = None
                
                # Create new chromatogram axis taking full space
                self.chromatogram_ax = self.figure.add_subplot(111)  # Full figure
                
                # Restore the original data
                if self.chromatogram_data is not None:
                    self.plot_chromatogram(self.chromatogram_data, new_file=False)
                
                # Restore original view limits
                self.chromatogram_ax.set_xlim(chromatogram_xlim)
                self.chromatogram_ax.set_ylim(chromatogram_ylim)
            else:
                # If we're restoring visibility, we need to reset the subplot layout
                # Use a safer check to see if we need to recreate the TIC axis
                tic_ax_exists = False
                try:
                    # Check if the axis still exists and is valid
                    tic_ax_exists = (hasattr(self, 'tic_ax') and 
                                   self.tic_ax is not None and 
                                   self.tic_ax in self.figure.axes)
                except (AttributeError, TypeError, ValueError):
                    # If any error occurs during the check, assume axis doesn't exist
                    tic_ax_exists = False
                    
                if not tic_ax_exists:
                    # Clear the figure completely to start fresh
                    self.figure.clear()
                    
                    # Set up the axes again
                    self.tic_ax = self.figure.add_subplot(211)  # Top plot for TIC
                    self.chromatogram_ax = self.figure.add_subplot(212, sharex=self.tic_ax)  # Bottom plot with shared x axis
                    
                    # Restore the plots
                    if self.chromatogram_data is not None:
                        self.plot_chromatogram(self.chromatogram_data, new_file=False)
                    if self.tic_data is not None and len(self.tic_data['x']) > 0:
                        self.plot_tic(self.tic_data['x'], self.tic_data['y'], new_file=False)
            
            # Apply theme and refresh
            self.apply_theme()
            self.figure.tight_layout()
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error in set_tic_visible: {type(e).__name__}: {e}")
            # Fallback: try to recreate everything from scratch
            try:
                self.figure.clear()
                self.tic_ax = self.figure.add_subplot(211)
                self.chromatogram_ax = self.figure.add_subplot(212, sharex=self.tic_ax)
                self.canvas.draw_idle()
            except Exception as fallback_error:
                print(f"Fallback failed in set_tic_visible: {fallback_error}")
                # Last resort: just hide the problematic axis if possible
                try:
                    if hasattr(self, 'tic_ax') and self.tic_ax is not None:
                        self.tic_ax.set_visible(False)
                        self.canvas.draw_idle()
                except:
                    pass
    
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

        # Add RT assignment action
        rt_assign_action = menu.addAction("Assign from RT Table")
        rt_assign_action.triggered.connect(lambda: self._request_rt_assignment(peak_index))
        
        # Add "Add to RT Table" action
        add_to_rt_action = menu.addAction("Add to RT Table...")
        add_to_rt_action.triggered.connect(lambda: self._add_peak_to_rt_table(peak_index))
        
        menu.addSeparator()
        
        # Add reassignment options submenu
        reassign_menu = menu.addMenu("Reassign Compound")
        
        # Reassign from MS library
        reassign_ms_action = reassign_menu.addAction("From MS Library Search")
        reassign_ms_action.triggered.connect(lambda: self._request_ms_search(peak_index))
        
        # Reassign from RT table
        reassign_rt_action = reassign_menu.addAction("From RT Table")
        reassign_rt_action.triggered.connect(lambda: self._request_rt_assignment(peak_index))
        
        # Manual reassignment
        reassign_manual_action = reassign_menu.addAction("Manual Assignment...")
        reassign_manual_action.triggered.connect(lambda: self._edit_peak_assignment(peak_index))
        
        menu.addSeparator()
        
        # Add edit assignment action (moved to separate section)
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

    def _request_rt_assignment(self, peak_index):
        """Request RT table assignment for a peak."""
        # Emit signal to request RT assignment
        self.rt_assignment_requested.emit(peak_index)
    
    def _edit_peak_assignment(self, peak_index):
        """Open dialog to edit peak compound assignment."""
        # Emit signal to request edit dialog
        self.edit_assignment_requested.emit(peak_index)
    
    def _add_peak_to_rt_table(self, peak_index):
        """Add a peak to the RT table."""
        # Emit signal to request adding peak to RT table
        self.add_to_rt_table_requested.emit(peak_index)