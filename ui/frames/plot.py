from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QMenu, QApplication
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QCursor, QAction, QIcon, QPainter, QPen, QPixmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

# Segment overlay (design 2026-11-12 §4.3). Its zorders sit above the integration
# shading (fill_between, zorder 1) and below the peak annotations (text, zorder 3).
SEGMENT_BAND_COLOR = '#3b7ddd'
SEGMENT_GREY_COLOR = '#888888'
SEGMENT_BREAK_COLOR = '#7a4c00'
SEGMENT_EXCLUDED_COLOR = '#999999'
SEGMENT_FADED_ALPHA = 0.3
SEGMENT_BAND_BOTTOM = 0.97   # bands fill the top 3 % of the axes; vertical lines stop below them
SEGMENT_MIN_WIDTH_PX = 3     # logical px: a shorter shift-drag or edge drag is a click; an edge keeps this gap
SEGMENT_EDGE_GRAB_PX = 6     # logical px: a press this close to an open segment's edge drags that edge
SEGMENT_TIME_DECIMALS = 4    # emitted span/edge times: 0.1 ms in minutes
SEGMENT_DRAG_ALPHA = 0.2     # the temporary span shown during a shift-drag
BASELINE_CHANGE_TEXT = "Change the baseline here"


class _BandLabel(Text):
    """A segment band's label, drawn only while it fits inside its band at the current zoom."""

    def __init__(self, span, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.span = span

    def fits(self, renderer) -> bool:
        px = self.axes.transData.transform([(self.span[0], 0), (self.span[1], 0)])[:, 0]
        return self.get_window_extent(renderer).width + 6 <= abs(px[1] - px[0])

    def draw(self, renderer):
        if self.get_visible() and self.fits(renderer):
            super().draw(renderer)

class PlotFrame(QWidget):
    """Frame for displaying chromatogram and TIC plots."""
    
    # Signal to notify when a point has been selected on the plot
    point_selected = Signal(float)
    
    # Signal for GCxGC peak selection
    peak_selected = Signal(object)  # emits GCxGC2DPeak on heatmap click
    
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

    # Segment editing on the chromatogram (design 2026-11-12 §4.3); app routes them to ParametersFrame
    segment_span_requested = Signal(float, float)   # shift-drag; start < end in data units
    baseline_change_requested = Signal(float)       # right-click inside the open segment
    segment_edge_moved = Signal(str, float)         # 'start' | 'end', new time, on release
    segment_escape = Signal()                       # Esc on the canvas with a segment open
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Current axis labels (updated by set_axis_labels from the active profile)
        self._x_label = 'Time (min)'
        self._y_label = 'Intensity'
        self._invert_x = False  # True for high→low x-axes (e.g. FTIR wavenumbers)
        
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
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)   # key_press_event (Esc) needs focus
        
        # Add toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._add_zoom_out_action()
        
        # Add widgets to layout
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        
        # Initialize data for storage
        self.chromatogram_data = None
        self.tic_data = None
        self.ms_time_offset: float = 0.0
        # Signal mode of the last chromatogram plot. Switching it changes the y extent
        # (raw ~ baseline height vs corrected ~ 0), so the old view limits no longer fit.
        self._show_corrected = False
        
        # Replace existing mouse click connection with new connections
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # GCxGC mode state
        self._gcxgc_peaks: list = []

        # Segment overlay (set_segment_overlay); redrawn after every plot_chromatogram
        self._segment_artists: list = []
        self._segment_windows: list = []
        self._segment_break_times: list = []
        self._open_segment = None
        self._segment_fit_key = None      # (id, start, end) the y-axis was last fitted to
        self._segment_saved_ylim = None   # the ylim to restore when the segment closes
        self._displayed_trace = None      # (x, trace, baseline) as plot_chromatogram shows them
        self._span_drag = None            # shift-drag in progress: {'x0', 'px0', 'patch'}
        self._edge_drag = None            # edge drag in progress: {'edge', 'other', 'orig', 'line', 'px0'}
        
        # Track click information
        self._click_time = 0
        self._last_click_x = None
        self._is_right_button = False
        
        # Set up initial empty plots
        self._setup_empty_plots()

    def _add_zoom_out_action(self):
        """Add a one-step zoom-out control beside Matplotlib's zoom tool."""
        icon_size = self.toolbar.iconSize()
        pixmap = QPixmap(icon_size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.toolbar.palette().color(self.toolbar.foregroundRole()), 1.8)
        painter.setPen(pen)
        center = icon_size.width() // 2 - 2
        radius = max(4, icon_size.width() // 4)
        painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
        painter.drawLine(center + radius - 1, center + radius - 1,
                         icon_size.width() - 2, icon_size.height() - 2)
        painter.drawLine(center - radius // 2, center, center + radius // 2, center)
        painter.end()

        action = QAction(QIcon(pixmap), "Zoom Out", self.toolbar)
        action.setToolTip("Zoom Out")
        action.setStatusTip("Zoom out one step")
        action.triggered.connect(self._zoom_out)

        zoom_action = next(
            (item for item in self.toolbar.actions() if item.text().strip().lower() == "zoom"),
            None,
        )
        if zoom_action:
            self.toolbar.insertAction(zoom_action, action)
        else:
            self.toolbar.addAction(action)

    def _zoom_out(self):
        """Expand each plot's current view around its center by 20 percent."""
        self.toolbar.push_current()
        seen_x_groups = set()
        for ax in self.figure.axes:
            x_group = frozenset(id(shared_ax) for shared_ax in ax.get_shared_x_axes().get_siblings(ax))
            if x_group not in seen_x_groups:
                lower, upper = ax.get_xlim()
                center = (lower + upper) / 2
                half_span = (upper - lower) * 0.6
                ax.set_xlim(center - half_span, center + half_span)
                seen_x_groups.add(x_group)

            lower, upper = ax.get_ylim()
            center = (lower + upper) / 2
            half_span = (upper - lower) * 0.6
            ax.set_ylim(center - half_span, center + half_span)

        self.canvas.draw_idle()

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

    def set_axis_labels(self, x_label: str, y_label: str, invert_x: bool = False) -> None:
        """Update axis labels and x-axis direction to match the active signal profile."""
        self._x_label = x_label
        self._y_label = y_label
        self._invert_x = invert_x
        if hasattr(self, 'chromatogram_ax') and self.chromatogram_ax is not None:
            self.chromatogram_ax.set_xlabel(self._x_label)
            self.chromatogram_ax.set_ylabel(self._y_label)
        if hasattr(self, 'tic_ax') and self.tic_ax is not None:
            self.tic_ax.set_xlabel(self._x_label)
            self.tic_ax.set_ylabel(self._y_label)
        self.canvas.draw_idle()

    def clear_tic(self):
        """Clear the TIC plot and redraw the canvas."""
        try:
            if hasattr(self, 'tic_ax') and self.tic_ax is not None:
                # Check if the axis is still valid and part of the figure
                if self.tic_ax in self.figure.axes:
                    self.tic_ax.clear()
                    self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
                    self.tic_ax.set_xlabel(self._x_label)
                    self.tic_ax.set_ylabel(self._y_label)
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
            self.tic_ax.set_xlabel(self._x_label)
            self.tic_ax.set_ylabel(self._y_label)
            self.tic_ax.grid(True, linestyle='--', alpha=0.7)
            
            # Restore chromatogram data if available
            if hasattr(self, 'chromatogram_data') and self.chromatogram_data is not None:
                self.plot_chromatogram(self.chromatogram_data, show_corrected=self._show_corrected, new_file=False)
            
            self.figure.tight_layout()
            self.canvas.draw_idle()
            print("Successfully recreated TIC axis structure")
            
        except Exception as e:
            print(f"Failed to recreate TIC axis: {e}")

    def _setup_empty_plots(self):
        """Set up initial empty plots."""
        self.tic_ax.clear()
        self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
        self.tic_ax.set_xlabel(self._x_label)
        self.tic_ax.set_ylabel(self._y_label)
        self.tic_ax.grid(True, linestyle='--', alpha=0.7)
        
        self.chromatogram_ax.clear()
        self.chromatogram_ax.set_title('Processed Chromatogram')
        self.chromatogram_ax.set_xlabel(self._x_label)
        self.chromatogram_ax.set_ylabel(self._y_label)
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
        
        # set_ylim() turns autoscale off, so a replot cannot tell "user
        # zoomed" from "we fitted it last time". Compare the signal mode explicitly: a
        # switch between raw and corrected needs a fresh full view, not the old limits.
        mode_changed = show_corrected != self._show_corrected
        self._show_corrected = show_corrected

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
        self._displayed_trace = (np.asarray(x), np.asarray(main_y), np.asarray(baseline_to_show))
        
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
                
                marker_line = None
                if meta['type'] == 'peak':
                    marker_line, = self.chromatogram_ax.plot(marker_x, marker_y, marker='*', color='orange',
                                         markersize=8, alpha=0.9,
                                         label='Peak' if 'Peak' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None)
                elif meta['type'] == 'shoulder':
                    marker_line, = self.chromatogram_ax.plot(marker_x, marker_y, marker='^', color='red',
                                         markersize=8, alpha=0.9,
                                         label='Shoulder' if 'Shoulder' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None)
                elif meta['type'] == 'negative_peak':
                    marker_line, = self.chromatogram_ax.plot(marker_x, marker_y, marker='v', color='cyan',
                                         markersize=8, alpha=0.9,
                                         label='Neg. Peak' if 'Neg. Peak' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None)
                if marker_line is not None:
                    marker_line.peak_x = marker_x   # the segment overlay fades by apex
        
        # Draw peak splitting component curves from pipeline
        if fitted_curves:
            _deconv_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
                              '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
                              '#fabebe', '#469990']
            for i, comp in enumerate(fitted_curves):
                color = _deconv_colors[i % len(_deconv_colors)]

                if hasattr(comp, 't_curve'):
                    # EMGComponent: draw dashed EMG curves
                    y_emg = comp.y_curve
                    if not show_corrected:
                        baseline_at = np.interp(comp.t_curve, data['x'], data['baseline_y'])
                        y_emg = y_emg + baseline_at
                    self.chromatogram_ax.plot(
                        comp.t_curve, y_emg, '--', color=color,
                        linewidth=1.0, alpha=0.7,
                        label='EMG' if i == 0 and 'EMG' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None
                    )
                elif hasattr(comp, 'start_index'):
                    # DeconvComponent: colored fill_between + vertical boundary lines
                    si = max(0, comp.start_index)
                    ei = min(len(data['x']) - 1, comp.end_index)
                    seg_x = data['x'][si:ei + 1]
                    seg_y = data['corrected_y'][si:ei + 1]
                    if not show_corrected:
                        seg_y = seg_y + data['baseline_y'][si:ei + 1]
                    base_y = data['baseline_y'][si:ei + 1] if not show_corrected else np.zeros(len(seg_x))
                    self.chromatogram_ax.fill_between(
                        seg_x, base_y, seg_y, color=color, alpha=0.25,
                        label='Peak Split' if i == 0 and 'Peak Split' not in self.chromatogram_ax.get_legend_handles_labels()[1] else None
                    )
                    # Vertical boundary lines at split points (skip chunk edges)
                    if comp.split_type_left != 'chunk':
                        self.chromatogram_ax.axvline(
                            comp.start_time, color=color, linewidth=0.6,
                            linestyle=':', alpha=0.5
                        )
                    if comp.split_type_right != 'chunk':
                        self.chromatogram_ax.axvline(
                            comp.end_time, color=color, linewidth=0.6,
                            linestyle=':', alpha=0.5
                        )

        # Set labels and add legend
        self.chromatogram_ax.set_xlabel(self._x_label)
        self.chromatogram_ax.set_ylabel(self._y_label)
        self.chromatogram_ax.grid(True, linestyle='--', alpha=0.7)
        self.chromatogram_ax.legend()
        
        # Only calculate and apply new view limits for new files
        # or if previous limits weren't set
        refit_view = new_file or mode_changed or previous_xlim is None or previous_ylim is None
        if refit_view:
            # The full view below replaces the one saved when the segment opened;
            # the overlay saves this one instead and fits the open segment again.
            self._segment_fit_key = None
            self._segment_saved_ylim = None
            if len(x) > 0 and len(main_y) > 0:
                x_min, x_max = np.min(x), np.max(x)
                x_padding = (x_max - x_min) * 0.02  # 2% padding
                x_full_range = [x_min - x_padding, x_max + x_padding]
                if self._invert_x:
                    x_full_range = x_full_range[::-1]
                
                # NaN-safe: baseline_to_show may contain NaN in MS-gated masked regions.
                # Use nanmin/nanmax to compute axis limits from finite values only.
                if np.any(np.isfinite(baseline_to_show)):
                    y_min = float(np.nanmin(baseline_to_show))
                else:
                    y_min = 0.0
                if np.any(np.isfinite(main_y)):
                    y_max_raw = float(np.nanmax(main_y))
                else:
                    y_max_raw = 1.0
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
                            if np.any(np.isfinite(main_y[mask])):
                                y_max_in_ms_range = float(np.nanmax(main_y[mask]))
                            else:
                                y_max_in_ms_range = y_max_raw  # Fall back to outer y_max_raw
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
            # The toolbar's view stack still holds the previous extent; Home would
            # restore it. Drop it so the fitted view above becomes Home.
            self.toolbar.update()
        else:
            # Restore previous view limits for parameter changes
            self.chromatogram_ax.set_xlim(previous_xlim)
            self.chromatogram_ax.set_ylim(previous_ylim)

        self._draw_segment_overlay()
        
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
        tic_label = 'TIC'
        if self.ms_time_offset != 0.0:
            tic_label = f"TIC (offset {self.ms_time_offset:+.4f} min)"
        self.tic_ax.plot(x, y, 'g-', linewidth=1, label=tic_label)
        
        # Plot baseline if available
        if show_baseline and baseline_x is not None and baseline_y is not None:
            self.tic_ax.plot(baseline_x, baseline_y, 'r--', linewidth=1, alpha=0.7, label='Baseline')
        
        # Set labels and title
        self.tic_ax.set_title('Total Ion Chromatogram (TIC)')
        self.tic_ax.set_xlabel(self._x_label)
        self.tic_ax.set_ylabel(self._y_label)
        self.tic_ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only calculate and apply new view limits for new files
        # or if previous limits weren't set
        if new_file or previous_xlim is None or previous_ylim is None:
            if len(x) > 0 and len(y) > 0:
                x_min, x_max = np.min(x), np.max(x)
                x_padding = (x_max - x_min) * 0.02  # 2% padding
                x_full_range = [x_min - x_padding, x_max + x_padding]
                if self._invert_x:
                    x_full_range = x_full_range[::-1]
                
                # NaN-safe: baseline_y may theoretically contain NaN (currently TIC baseline
                # doesn't run through MS-gated correction, but defensive coding makes this
                # robust to future changes).
                if show_baseline and baseline_y is not None:
                    if np.any(np.isfinite(baseline_y)):
                        y_min = float(np.nanmin(baseline_y))
                    else:
                        y_min = 0.0
                else:
                    y_min = 0
                if np.any(np.isfinite(y)):
                    y_max_raw = float(np.nanmax(y))
                else:
                    y_max_raw = 1.0
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
        
        # Add legend if baseline is shown or an offset annotation is active
        if (show_baseline and baseline_x is not None) or self.ms_time_offset != 0.0:
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
        # GCxGC mode: delegate to gcxgc click handler
        if hasattr(self, '_gcxgc_axes'):
            self._on_gcxgc_click(event)
            return
        
        if event.inaxes is None:
            return
            
        # Store click information
        import time
        self._click_time = time.time()
        self._last_click_x = event.xdata
        self._is_right_button = event.button == 3  # Right button is 3 in matplotlib

        # Shift-drag (new segment) or a press on the open segment's edge; a press that
        # never becomes a drag still ends as the plain click recorded above.
        if event.button == 1:
            self._begin_segment_drag(event)

        if not (self._is_right_button and event.inaxes == self.chromatogram_ax):
            return
        # Inside the open segment, right-click offers a baseline change (on a peak: in its menu).
        # Not while pan/zoom is active: a modal menu would swallow the toolbar's right-drag.
        change_x = (event.xdata if not self.toolbar.mode and self._in_open_segment(event.xdata)
                    else None)

        # Only handle right-clicks on the chromatogram axis specifically
        if hasattr(self, 'integrated_peaks') and self.integrated_peaks:
            # Check if the click is near any integrated peak
            found_peak = False
            
            for i, peak in enumerate(self.integrated_peaks):
                peak_pos = getattr(peak, 'retention_time', None) or peak.position
                peak_start = getattr(peak, 'start_time', None) or peak.start
                peak_end = getattr(peak, 'end_time', None) or peak.end
                distance = abs(peak_pos - event.xdata)
                bound_lo, bound_hi = min(peak_start, peak_end), max(peak_start, peak_end)
                
                if (bound_lo <= event.xdata <= bound_hi) or distance < abs(bound_hi - bound_lo) * 0.1:
                    found_peak = True
                    self._show_peak_context_menu(i, event, baseline_change_x=change_x)
                    
                    self._last_click_x = None
                    self._is_right_button = False
                    break
            
            # If no peak was found under the click, we'll return without resetting
            # letting the regular right-click behavior proceed
            if found_peak:
                return

        if change_x is not None:
            self._show_baseline_change_menu(change_x)
            self._last_click_x = None
            self._is_right_button = False
    
    def _on_button_release(self, event):
        """Handle mouse button release on the plot."""
        if self._finish_segment_drag(event):
            self._last_click_x = None
            self._is_right_button = False
            return
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
                        # Use generic Feature attrs (position/start/end) with fallback to chromatography attrs
                        peak_pos = getattr(peak, 'retention_time', None) or peak.position
                        peak_start = getattr(peak, 'start_time', None) or peak.start
                        peak_end = getattr(peak, 'end_time', None) or peak.end
                        distance = abs(peak_pos - x_value)
                        
                        # Normalize bounds for inverted axes
                        bound_lo = min(peak_start, peak_end)
                        bound_hi = max(peak_start, peak_end)
                        
                        # If within bounds or close to apex
                        if (bound_lo <= x_value <= bound_hi) or distance < abs(bound_hi - bound_lo) * 0.1:
                            if distance < min_distance:
                                min_distance = distance
                                closest_peak = i
                    
                    # If a peak was found nearby, highlight it
                    if closest_peak is not None:
                        self._highlight_selected_peak(self.integrated_peaks[closest_peak])
                        return
                
                # If no peak was found or no integrated peaks, just add the marker line
                for ax in [self.chromatogram_ax, self.tic_ax]:
                    if ax is None:
                        continue
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

    # ── segment editing gestures ─────────────────────────────────────────────

    def _in_open_segment(self, x) -> bool:
        seg = self._open_segment
        if seg is None or x is None:
            return False
        lo, hi = sorted((float(seg['start']), float(seg['end'])))
        return lo <= x <= hi

    def _data_x(self, event) -> float:
        """The chromatogram x under the mouse, also when the mouse has left the axes."""
        return float(self.chromatogram_ax.transData.inverted().transform((event.x, event.y))[0])

    def _x_px(self, x) -> float:
        ax = self.chromatogram_ax
        return float(ax.transData.transform((x, ax.get_ylim()[0]))[0])

    def _px(self, logical: float) -> float:
        """Logical px in the physical px that matplotlib's Qt events and transforms use."""
        return logical * self.canvas.device_pixel_ratio

    def _dragged(self, drag, event) -> bool:
        """True once the pointer is more than SEGMENT_MIN_WIDTH_PX from where the drag began."""
        return abs(event.x - drag['px0']) > self._px(SEGMENT_MIN_WIDTH_PX)

    def _begin_segment_drag(self, event):
        """Start an edge drag (press near an open segment's edge) or a shift-drag span."""
        if event.inaxes is not self.chromatogram_ax or self.toolbar.mode:
            return
        seg = self._open_segment
        if seg is not None:
            lo, hi = sorted((float(seg['start']), float(seg['end'])))
            near = [(abs(event.x - self._x_px(t)), edge, t, other)
                    for edge, t, other in (('start', lo, hi), ('end', hi, lo))]
            dist, edge, t, other = min(near)
            if dist <= self._px(SEGMENT_EDGE_GRAB_PX):
                line = next((ln for ln in self._segment_artists
                             if getattr(ln, 'segment_role', None) == 'edge' and ln.get_xdata()[0] == t),
                            None)
                if line is None:
                    raise RuntimeError(f"Open segment {seg['id']!r} has no edge line at {t}")
                self._edge_drag = {'edge': edge, 'other': other, 'orig': t, 'line': line,
                                   'px0': event.x}
                return
        if 'shift' in (event.modifiers or ()):
            self._span_drag = {'x0': float(event.xdata), 'px0': event.x, 'patch': None}

    def _edge_drag_x(self, drag, event) -> float:
        """The dragged edge's time, kept SEGMENT_MIN_WIDTH_PX short of the other edge."""
        ax = self.chromatogram_ax
        x0, x1 = ax.get_xlim()
        min_gap = self._px(SEGMENT_MIN_WIDTH_PX) * abs(x1 - x0) / ax.bbox.width
        x = self._data_x(event)
        if drag['edge'] == 'start':
            return min(x, drag['other'] - min_gap)
        return max(x, drag['other'] + min_gap)

    @staticmethod
    def _round_past(x: float, other: float, below: bool) -> float:
        """x rounded to SEGMENT_TIME_DECIMALS, then stepped until it stays strictly below
        (or above) ``other``; only needed when zoomed in past the rounding step."""
        step = 10.0 ** -SEGMENT_TIME_DECIMALS
        x = round(x, SEGMENT_TIME_DECIMALS)
        while (x >= other) if below else (x <= other):
            x = round(x - step if below else x + step, SEGMENT_TIME_DECIMALS)
        return x

    def _on_motion(self, event):
        if self._span_drag is None and self._edge_drag is None:
            return
        x = self._data_x(event)
        if self._span_drag is not None:
            drag = self._span_drag
            if drag['patch'] is None:
                drag['patch'] = Rectangle((drag['x0'], 0), 0, 1,
                                          transform=self.chromatogram_ax.get_xaxis_transform(),
                                          facecolor=SEGMENT_BAND_COLOR, edgecolor='none',
                                          alpha=SEGMENT_DRAG_ALPHA, zorder=2.4)
                drag['patch'].segment_role = 'drag_span'
                self.chromatogram_ax.add_artist(drag['patch'])   # leaves the data limits alone
            drag['patch'].set_width(x - drag['x0'])
        else:
            if not self._dragged(self._edge_drag, event):
                return   # a press with a little wobble is still a click (it may select a peak)
            x = self._edge_drag_x(self._edge_drag, event)
            self._edge_drag['line'].set_xdata([x, x])
        self.canvas.draw_idle()

    def _finish_segment_drag(self, event) -> bool:
        """End a drag on release; True if it emitted (so the release is not also a click)."""
        span, edge = self._span_drag, self._edge_drag
        self._cancel_segment_drag()
        if span is not None and self._dragged(span, event):
            x1 = self._data_x(event)
            lo = round(min(span['x0'], x1), SEGMENT_TIME_DECIMALS)
            self.segment_span_requested.emit(lo, self._round_past(max(span['x0'], x1), lo, below=False))
            return True
        if edge is not None and self._dragged(edge, event):
            x = self._round_past(self._edge_drag_x(edge, event), edge['other'],
                                 below=edge['edge'] == 'start')
            edge['line'].set_xdata([x, x])   # stays put until the editor's preview redraws it
            self.segment_edge_moved.emit(edge['edge'], x)
            return True
        return False

    def _cancel_segment_drag(self):
        """Drop a drag in progress: remove the temporary span, put a dragged edge back."""
        if self._span_drag is not None and self._span_drag['patch'] is not None:
            self._span_drag['patch'].remove()
            self.canvas.draw_idle()
        if self._edge_drag is not None:
            orig = self._edge_drag['orig']
            self._edge_drag['line'].set_xdata([orig, orig])
            self.canvas.draw_idle()
        self._span_drag = None
        self._edge_drag = None

    def _on_key_press(self, event):
        if event.key != 'escape':
            return
        self._cancel_segment_drag()
        if self._open_segment is not None:
            self.segment_escape.emit()

    def _show_baseline_change_menu(self, x):
        menu = QMenu(self)
        action = menu.addAction(BASELINE_CHANGE_TEXT)
        if menu.exec(QCursor.pos()) is action:
            self.baseline_change_requested.emit(float(x))

    def _clear_peak_highlights(self):
        """Clear any peak highlights."""
        # Store current axis limits (guard against None axes)
        chromatogram_xlim = self.chromatogram_ax.get_xlim() if self.chromatogram_ax is not None else None
        chromatogram_ylim = self.chromatogram_ax.get_ylim() if self.chromatogram_ax is not None else None
        tic_xlim = self.tic_ax.get_xlim() if self.tic_ax is not None else None
        tic_ylim = self.tic_ax.get_ylim() if self.tic_ax is not None else None

        for ax in [self.chromatogram_ax, self.tic_ax]:
            if ax is None:
                continue
            # Remove peak indicators
            for line in ax.get_lines():
                if line.get_label() == '_selected_peak':
                    line.remove()

            # Remove any annotations
            for txt in ax.texts:
                if hasattr(txt, 'peak_annotation') and txt.peak_annotation:
                    txt.remove()

        # Remove any highlighted fill_between
        if self.chromatogram_ax is not None:
            for collection in self.chromatogram_ax.collections:
                if hasattr(collection, 'peak_highlight') and collection.peak_highlight:
                    collection.remove()

        # Restore the axis limits
        if self.chromatogram_ax is not None and chromatogram_xlim is not None:
            self.chromatogram_ax.set_xlim(chromatogram_xlim)
        if self.chromatogram_ax is not None and chromatogram_ylim is not None:
            self.chromatogram_ax.set_ylim(chromatogram_ylim)
        if self.tic_ax is not None and tic_xlim is not None:
            self.tic_ax.set_xlim(tic_xlim)
        if self.tic_ax is not None and tic_ylim is not None:
            self.tic_ax.set_ylim(tic_ylim)

        self.canvas.draw_idle()

    def _highlight_selected_peak(self, peak):
        """Highlight the selected peak with enhanced information."""
        # Store current axis limits before clearing highlights (guard against None axes)
        chromatogram_xlim = self.chromatogram_ax.get_xlim() if self.chromatogram_ax is not None else None
        chromatogram_ylim = self.chromatogram_ax.get_ylim() if self.chromatogram_ax is not None else None
        tic_xlim = self.tic_ax.get_xlim() if self.tic_ax is not None else None
        tic_ylim = self.tic_ax.get_ylim() if self.tic_ax is not None else None
        
        # First clear any existing highlights
        self._clear_peak_highlights()
        
        # Generic Feature attrs with chromatography fallbacks
        peak_pos = getattr(peak, 'retention_time', None) or peak.position
        peak_id = getattr(peak, 'peak_number', None) or peak.feature_id
        
        # Determine text color based on peak issues (prioritize saturation over convolution)
        text_color = 'purple' if hasattr(peak, 'is_saturated') and peak.is_saturated else \
                    'red' if hasattr(peak, 'is_convoluted') and peak.is_convoluted else 'g'
        
        # Build annotation text — chromatography vs spectroscopy
        is_chromatographic = hasattr(peak, 'retention_time')
        if is_chromatographic:
            annotation_text = f"Peak {peak_id}\nRT: {peak_pos:.3f}"
            
            # Add compound ID with source information
            if hasattr(peak, 'compound_id') and peak.compound_id and peak.compound_id != f"Unknown ({peak_pos:.3f})":
                annotation_text += f"\n{peak.compound_id}"

                if hasattr(peak, 'rt_assignment') and peak.rt_assignment:
                    source = getattr(peak, 'rt_assignment_source', 'RT')
                    annotation_text += f" [{source}]"
                elif hasattr(peak, 'Qual') and peak.Qual is not None:
                    annotation_text += f" [MS: {peak.Qual:.3f}]"
                else:
                    annotation_text += " [Manual]"
            
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                annotation_text += "\n⚠️ DETECTOR SATURATION"
                if hasattr(peak, 'saturation_level'):
                    annotation_text += f"\n • Max: {peak.saturation_level:.2e}"
            elif hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                annotation_text += "\n⚠️ Possible convolution"
                if hasattr(peak, 'quality_issues') and peak.quality_issues:
                    for issue in peak.quality_issues:
                        annotation_text += f"\n • {issue}"
                else:
                    if hasattr(peak, 'asymmetry') and peak.asymmetry is not None:
                        annotation_text += f"\n • Asym: {peak.asymmetry:.2f}"
                    if hasattr(peak, 'spectral_coherence') and peak.spectral_coherence is not None:
                        annotation_text += f"\n • Coh: {peak.spectral_coherence:.2f}"
        else:
            annotation_text = f"Feature {peak_id}\n{self._x_label}: {peak_pos:.2f}"
            annotation_text += f"\nArea: {abs(peak.area):.4f}"
            if hasattr(peak, 'band_assignment') and peak.band_assignment:
                annotation_text += f"\n{peak.band_assignment}"
        
        # Add annotation and highlight for each plot
        for ax in [self.chromatogram_ax, self.tic_ax]:
            if ax is None:
                continue
            text = ax.text(peak_pos, ax.get_ylim()[1]*0.9, 
                    annotation_text, 
                    color=text_color, fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7))
            text.peak_annotation = True
        
        # If the peak is part of integration results, highlight its area
        if hasattr(self, 'x_peaks') and hasattr(self, 'y_peaks') and hasattr(self, 'baseline_peaks'):
            for i, integrated_peak in enumerate(self.integrated_peaks):
                ip_id = getattr(integrated_peak, 'peak_number', None) or integrated_peak.feature_id
                if ip_id == peak_id:
                    for collection in self.chromatogram_ax.collections:
                        if hasattr(collection, 'peak_highlight') and collection.peak_highlight:
                            collection.remove()
                    
                    showing_corrected = self.chromatogram_ax.get_title() == 'Baseline-Corrected Chromatogram'
                    
                    offset = 0.0 if showing_corrected else self.baseline_peaks[i]
                    top = getattr(self, 'shade_top_peaks', self.y_peaks)
                    floor = getattr(self, 'shade_floor_peaks', None)
                    shade_y = top[i] + offset
                    shade_baseline = (floor[i] if floor is not None else 0.0) + offset
                    
                    collection = self.chromatogram_ax.fill_between(
                        self.x_peaks[i],
                        shade_y,
                        shade_baseline,
                        alpha=0.8,
                        color='green',
                        label=f'_highlight_peak_{peak_id}'
                    )
                    collection.peak_highlight = True
                    
                    self.peak_spectrum_requested.emit(ip_id - 1)
                    self.point_selected.emit(peak_pos)
                    
                    break
        
        # Restore the axis limits before redrawing
        if self.chromatogram_ax is not None and chromatogram_xlim is not None:
            self.chromatogram_ax.set_xlim(chromatogram_xlim)
        if self.chromatogram_ax is not None and chromatogram_ylim is not None:
            self.chromatogram_ax.set_ylim(chromatogram_ylim)
        if self.tic_ax is not None and tic_xlim is not None:
            self.tic_ax.set_xlim(tic_xlim)
        if self.tic_ax is not None and tic_ylim is not None:
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
        # Older results (deconvolution, cached) lack the skim shading arrays
        self.shade_top_peaks = integration_results.get('shade_top_peaks', self.y_peaks)
        self.shade_floor_peaks = integration_results.get(
            'shade_floor_peaks', [np.zeros_like(y) for y in self.y_peaks])
        
        # Store current limits before clearing
        xlim = self.chromatogram_ax.get_xlim() if not self.chromatogram_ax.get_autoscalex_on() else None
        ylim = self.chromatogram_ax.get_ylim() if not self.chromatogram_ax.get_autoscaley_on() else None
        
        # Determine if we're showing corrected or original signal
        showing_corrected = self.chromatogram_ax.get_title() == 'Baseline-Corrected Chromatogram'
        
        # Clear previous plot - redraw without the peak markers, PRESERVING view state
        self.chromatogram_data['peaks_x'] = np.array([])
        self.chromatogram_data['peaks_y'] = np.array([])
        self.plot_chromatogram(self.chromatogram_data, show_corrected=showing_corrected, new_file=False)
        
        # Get a distinct colormap from matplotlib with good differentiation.
        # (matplotlib.cm.get_cmap was removed in 3.9+; use pyplot.get_cmap.)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab20')
        
        # Shade each integrated peak area with a unique color
        for i in range(len(self.x_peaks)):
            # Use prime number step to give good color separation
            color_idx = (i * 7) % 20
            
            # Get the color from the colormap
            color = cmap(color_idx / 20)
            
            # Check if peak is flagged as saturated, convoluted, or negative
            peak = self.integrated_peaks[i]
            is_saturated = hasattr(peak, 'is_saturated') and peak.is_saturated
            is_convoluted = hasattr(peak, 'is_convoluted') and peak.is_convoluted
            is_negative = hasattr(peak, 'is_negative') and peak.is_negative

            # Use cyan for negative peaks
            if is_negative:
                color = 'cyan'

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
            
            # Shade the peak area: top/floor are in corrected space; raw view adds the baseline back
            offset = 0.0 if showing_corrected else self.baseline_peaks[i]
            shade_y = self.shade_top_peaks[i] + offset
            shade_baseline = self.shade_floor_peaks[i] + offset

            apex_x = getattr(peak, 'retention_time', None) or peak.position
            if getattr(peak, 'skim_model', None) is not None:  # a rider, its parent reported or not
                line, = self.chromatogram_ax.plot(
                    self.x_peaks[i], shade_baseline, linestyle='--', color='crimson', linewidth=1.4)
                line.is_skim_line = True
                line.peak_x = apex_x

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
            collection.peak_x = apex_x
            
            # Add warning marker for saturated or convoluted peaks - adjust position to match view
            marker_y = shade_y[len(shade_y)//2]  # Use middle point height
            
            warning_line = None
            if is_saturated:
                warning_line, = self.chromatogram_ax.plot(
                    apex_x,
                    marker_y,
                    marker='D',
                    markersize=12,
                    markerfacecolor='purple',
                    markeredgecolor='black',
                    alpha=0.8,
                    linestyle='None'
                )
            elif is_convoluted:
                warning_line, = self.chromatogram_ax.plot(
                    apex_x,
                    marker_y,
                    marker='^',
                    markersize=12,
                    markerfacecolor='red',
                    markeredgecolor='black',
                    alpha=0.8,
                    linestyle='None'
                )
            if warning_line is not None:
                warning_line.peak_x = apex_x
        
        # Restore previous limits
        if xlim is not None:
            self.chromatogram_ax.set_xlim(xlim)
        if ylim is not None:
            self.chromatogram_ax.set_ylim(ylim)

        self._apply_segment_fade()   # the shading above is new since the overlay drew
        
        # Update the plot
        self.canvas.draw_idle()
    
    # ── segment overlay ──────────────────────────────────────────────────────

    def set_segment_overlay(self, windows: list[dict], break_times: list[float],
                            open_segment: dict | None) -> None:
        """Show the method's segments on the chromatogram; kept across later replots.

        windows: peak_windows dicts (id, label, start, end). break_times: the whole-run
        baseline breaks only; a segment's own start/end breaks are implied by its edges.
        open_segment: None, or {'id', 'label', 'start', 'end', 'partway': [t, ...]} for the
        segment open in the editor (its draft, which may not be in ``windows`` yet).
        """
        self._segment_windows = [dict(w) for w in windows]
        self._segment_break_times = [float(t) for t in break_times]
        self._open_segment = dict(open_segment) if open_segment is not None else None
        self._draw_segment_overlay()
        self.canvas.draw_idle()

    def _segment_add(self, artist, role):
        artist.segment_role = role
        artist.set_label('_segment_overlay')
        self.chromatogram_ax.add_artist(artist)   # add_artist leaves the data limits alone
        self._segment_artists.append(artist)

    def _draw_segment_overlay(self):
        """(Re)draw the overlay into chromatogram_ax from the stored windows/breaks/open segment."""
        for artist in self._segment_artists:
            if artist.axes is not None:           # ax.clear() already detached the rest
                artist.remove()
        self._segment_artists = []
        if hasattr(self, '_gcxgc_axes'):
            return
        ax = self.chromatogram_ax
        xaxis = ax.get_xaxis_transform()            # x in data, y in axes fraction
        seg = self._open_segment

        bands = [w for w in self._segment_windows if seg is None or w['id'] != seg['id']]
        for w in bands + ([seg] if seg is not None else []):
            start, end = float(w['start']), float(w['end'])
            name = w.get('label') or w['id']
            if w is seg:
                name += ' (editing)'
            self._segment_add(Rectangle((start, SEGMENT_BAND_BOTTOM), end - start,
                                        1 - SEGMENT_BAND_BOTTOM, transform=xaxis,
                                        facecolor=SEGMENT_BAND_COLOR, edgecolor='none',
                                        alpha=0.85, zorder=2.5), 'band')
            self._segment_add(_BandLabel((start, end), (start + end) / 2,
                                         (1 + SEGMENT_BAND_BOTTOM) / 2, name, transform=xaxis,
                                         color='white', fontsize=7, ha='center', va='center',
                                         zorder=2.6, clip_on=True), 'label')

        for t in self._segment_break_times:
            self._segment_add(Line2D([t, t], [0, SEGMENT_BAND_BOTTOM], transform=xaxis, color=SEGMENT_BREAK_COLOR,
                                     linestyle='--', linewidth=1.0, zorder=2.5), 'break')

        if seg is not None:
            lo, hi = sorted((float(seg['start']), float(seg['end'])))
            # Grey reaches well past the data so panning or zooming out stays covered.
            ends = [lo, hi, *ax.get_xlim()]
            if self._displayed_trace is not None and len(self._displayed_trace[0]):
                ends += [float(np.nanmin(self._displayed_trace[0])),
                         float(np.nanmax(self._displayed_trace[0]))]
            reach = 10 * (max(ends) - min(ends))
            for edge, far in ((lo, min(ends) - reach), (hi, max(ends) + reach)):
                self._segment_add(Rectangle((edge, 0), far - edge, 1, transform=xaxis,
                                            facecolor=SEGMENT_GREY_COLOR, edgecolor='none',
                                            alpha=0.18, zorder=1.5), 'grey')
                self._segment_add(Line2D([edge, edge], [0, SEGMENT_BAND_BOTTOM], transform=xaxis,
                                         color=SEGMENT_BAND_COLOR, linewidth=1.2, zorder=2.5), 'edge')
            for t in seg.get('partway') or []:
                self._segment_add(Line2D([t, t], [0, SEGMENT_BAND_BOTTOM], transform=xaxis, color=SEGMENT_BREAK_COLOR,
                                         linestyle='--', linewidth=1.0, zorder=2.5), 'partway')

        excluded = []
        if self.chromatogram_data is not None:
            excluded = self.chromatogram_data.get('range_excluded_peaks_x')
            excluded = np.asarray(excluded if excluded is not None else [], dtype=float)
        if len(excluded) and self._displayed_trace is not None:
            x, trace, _ = self._displayed_trace
            order = np.argsort(x)                 # np.interp needs ascending x (FTIR runs high→low)
            ys = np.interp(excluded, x[order], trace[order])
            self._segment_add(Line2D(excluded, ys, linestyle='none', marker='o', markersize=7,
                                     markerfacecolor='none', markeredgecolor=SEGMENT_EXCLUDED_COLOR,
                                     markeredgewidth=1.2, zorder=2.5), 'excluded')

        self._fit_segment_ylim()
        self._apply_segment_fade()

    def _fit_segment_ylim(self):
        """Fit y to the open segment when its id or span changes; restore the old ylim on close.

        A preview that keeps the span leaves the y-axis alone, so the user's zoom stands.
        """
        seg = self._open_segment
        ax = self.chromatogram_ax
        if seg is None:
            if self._segment_fit_key is not None and self._segment_saved_ylim is not None:
                ax.set_ylim(self._segment_saved_ylim)
            self._segment_fit_key = None
            self._segment_saved_ylim = None
            return
        key = (seg['id'], float(seg['start']), float(seg['end']))
        if key == self._segment_fit_key or self._displayed_trace is None:
            return
        if self._segment_fit_key is None:
            self._segment_saved_ylim = ax.get_ylim()
        self._segment_fit_key = key
        x, trace, baseline = self._displayed_trace
        lo, hi = sorted(key[1:])
        inside = (x >= lo) & (x <= hi)
        values = np.concatenate([trace[inside], baseline[inside]])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return      # the span holds no data points (beyond the run, or MS-gated): keep the view
        y0, y1 = float(values.min()), float(values.max())
        pad = 0.05 * (y1 - y0) or 0.05 * abs(y1) or 1.0
        ax.set_ylim(y0 - pad, y1 + pad)

    def _apply_segment_fade(self):
        """Peak markers and shading whose apex lies outside the open segment drop to alpha 0.3;
        everything else gets its own alpha back."""
        seg = self._open_segment
        if seg is not None:
            lo, hi = sorted((float(seg['start']), float(seg['end'])))
        for artist in self.chromatogram_ax.get_children():
            peak_x = getattr(artist, 'peak_x', None)
            if peak_x is None:
                continue
            outside = seg is not None and not (lo <= peak_x <= hi)
            if outside and not hasattr(artist, 'segment_alpha'):
                artist.segment_alpha = artist.get_alpha()
                artist.set_alpha(SEGMENT_FADED_ALPHA)
            elif not outside and hasattr(artist, 'segment_alpha'):
                artist.set_alpha(artist.segment_alpha)
                del artist.segment_alpha

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
        for name in ('shade_top_peaks', 'shade_floor_peaks'):
            if hasattr(self, name):
                delattr(self, name)
    
    def set_tic_data(self, x, y):
        """Store the TIC data without plotting it."""
        self.tic_data = {'x': x, 'y': y}

    def set_ms_time_offset(self, offset_min: float) -> None:
        """Store the active MS time offset and refresh the TIC legend."""
        self.ms_time_offset = float(offset_min)
        # tic_ax is None while a GC-only file hides the TIC; the previous file's tic_data lingers
        if self.tic_ax is not None and self.tic_data is not None and len(self.tic_data.get('x', [])) > 0:
            self.plot_tic(self.tic_data['x'], self.tic_data['y'], new_file=False)

    def set_tic_visible(self, visible: bool):
        """Set the visibility of the TIC plot.
        
        Args:
            visible (bool): Whether to show the TIC plot
        """
        try:
            # Store current axis limits to preserve them
            chromatogram_xlim = self.chromatogram_ax.get_xlim() if self.chromatogram_ax is not None else None
            chromatogram_ylim = self.chromatogram_ax.get_ylim() if self.chromatogram_ax is not None else None

            # If tic_ax is None (previously hidden), skip straight to recreation
            if self.tic_ax is None:
                if visible:
                    # Need to recreate the dual-axis layout
                    self.figure.clear()
                    self.tic_ax = self.figure.add_subplot(211)
                    self.chromatogram_ax = self.figure.add_subplot(212, sharex=self.tic_ax)

                    if self.chromatogram_data is not None:
                        self.plot_chromatogram(self.chromatogram_data, show_corrected=self._show_corrected, new_file=False)
                    if self.tic_data is not None and len(self.tic_data['x']) > 0:
                        self.plot_tic(self.tic_data['x'], self.tic_data['y'], new_file=False)

                    self.apply_theme()
                    self.figure.tight_layout()
                    self.canvas.draw_idle()
                # If not visible and already None, nothing to do
                return

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
                    self.plot_chromatogram(self.chromatogram_data, show_corrected=self._show_corrected, new_file=False)
                
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
                        self.plot_chromatogram(self.chromatogram_data, show_corrected=self._show_corrected, new_file=False)
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
    
    def _show_peak_context_menu(self, peak_index, event, baseline_change_x=None):
        """Show context menu for a peak; with baseline_change_x (a click inside the open
        segment) it also offers a baseline change there."""
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QClipboard
        
        # Get the peak
        peak = self.integrated_peaks[peak_index]
        peak_pos = getattr(peak, 'retention_time', None) or peak.position
        peak_id = getattr(peak, 'peak_number', None) or peak.feature_id
        is_chromatographic = hasattr(peak, 'retention_time')
        
        # Create context menu
        menu = QMenu()
        
        if is_chromatographic:
            view_ms_action = menu.addAction(f"View MS Spectrum for Peak {peak_id}")
            view_ms_action.triggered.connect(lambda: self.peak_spectrum_requested.emit(peak_index))
            
            search_ms_action = menu.addAction("Search MS Library")
            search_ms_action.triggered.connect(lambda: self._request_ms_search(peak_index))

            rt_assign_action = menu.addAction("Assign from RT Table")
            rt_assign_action.triggered.connect(lambda: self._request_rt_assignment(peak_index))
            
            add_to_rt_action = menu.addAction("Add to RT Table...")
            add_to_rt_action.triggered.connect(lambda: self._add_peak_to_rt_table(peak_index))
            
            menu.addSeparator()
            
            reassign_menu = menu.addMenu("Reassign Compound")
            reassign_ms_action = reassign_menu.addAction("From MS Library Search")
            reassign_ms_action.triggered.connect(lambda: self._request_ms_search(peak_index))
            reassign_rt_action = reassign_menu.addAction("From RT Table")
            reassign_rt_action.triggered.connect(lambda: self._request_rt_assignment(peak_index))
            reassign_manual_action = reassign_menu.addAction("Manual Assignment...")
            reassign_manual_action.triggered.connect(lambda: self._edit_peak_assignment(peak_index))
            
            menu.addSeparator()
            
            edit_assignment_action = menu.addAction("Edit Compound Assignment...")
            edit_assignment_action.triggered.connect(lambda: self._edit_peak_assignment(peak_index))
            
            menu.addSeparator()
        
        # Copy position to clipboard (RT for chromatography, position for spectroscopy)
        pos_label = "RT" if is_chromatographic else self._x_label
        copy_rt_action = menu.addAction(f"Copy {pos_label}: {peak_pos:.3f}")
        copy_rt_action.triggered.connect(lambda: QApplication.clipboard().setText(f"{peak_pos:.3f}"))
        
        # If compound ID is assigned, add option to copy it
        if hasattr(peak, 'compound_id') and peak.compound_id and peak.compound_id != f"Unknown ({peak_pos:.3f})":
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
        
        change_action = None
        if baseline_change_x is not None:
            menu.addSeparator()
            change_action = menu.addAction(BASELINE_CHANGE_TEXT)

        # Show the menu at current cursor position instead of using matplotlib event coordinates
        cursor_pos = QCursor.pos()  # Get global cursor position
        chosen = menu.exec(cursor_pos)      # Show menu at cursor position
        if change_action is not None and chosen is change_action:
            self.baseline_change_requested.emit(float(baseline_change_x))

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
    
    def render_gcxgc(self, fid_frame, tic_frame, *, show_fid: bool = True) -> None:
        """GCxGC heatmaps from Frame2D objects: the TIC on top and, when show_fid, the FID below.

        Linear colour scale from the 1st to the 99th percentile of each frame (a floor of zero would
        paint an uncorrected FID, baseline tens of thousands of counts, as one flat colour). X: 1tR (min) from each frame's
        row times; Y: 2tR (s), origin bottom-left.
        """
        import numpy as np

        self.figure.clear()
        self.figure.set_constrained_layout(True)
        rows = 2 if show_fid else 1
        ax_tic = self.figure.add_subplot(rows, 1, 1)

        def extent(f):
            if len(f.t1_min) == 0:
                return [0.0, 1.0, 0.0, f.N / f.hz]
            return [f.t1_min[0], f.t1_min[-1] + f.N / f.hz / 60.0, 0.0, f.N / f.hz]

        def clim(data):
            if not data.size:
                return 0.0, 1.0
            lo, hi = (float(v) for v in np.percentile(data, (1, 99)))
            if hi <= lo:                                  # flat or nearly so: fall back to the full range
                lo, hi = float(data.min()), float(data.max())
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi

        lo, hi = clim(tic_frame.data)
        im = ax_tic.imshow(tic_frame.data.T, aspect='auto', origin='lower', cmap='viridis',
                           vmin=lo, vmax=hi, extent=extent(tic_frame))
        ax_tic.set_ylabel("2D RT (s)")
        ax_tic.set_title("DBC TIC" if tic_frame.signal == "dbc_tic" else "TIC")
        axes = [ax_tic]
        if show_fid:
            ax_tic.tick_params(labelbottom=False)
            ax_fid = self.figure.add_subplot(2, 1, 2, sharex=ax_tic, sharey=ax_tic)
            lo, hi = clim(fid_frame.data)
            im = ax_fid.imshow(fid_frame.data.T, aspect='auto', origin='lower', cmap='viridis',
                               vmin=lo, vmax=hi, extent=extent(fid_frame))
            ax_fid.set_xlabel("1st Dimension RT (min)")
            ax_fid.set_ylabel("2D RT (s)")
            ax_fid.set_title("FID")
            axes.append(ax_fid)
            self._gcxgc_ax_fid = ax_fid
        else:
            ax_tic.set_xlabel("1st Dimension RT (min)")
            if hasattr(self, '_gcxgc_ax_fid'):
                delattr(self, '_gcxgc_ax_fid')
        self.figure.colorbar(im, ax=axes, label="Intensity")
        self._gcxgc_ax_tic = ax_tic
        self._gcxgc_axes = axes
        self.canvas.draw()

    def clear_gcxgc(self) -> None:
        """Take the GCxGC heatmaps and their peaks off the canvas and show the empty 1D axes."""
        for name in ('_gcxgc_ax_tic', '_gcxgc_ax_fid', '_gcxgc_axes'):
            if hasattr(self, name):
                delattr(self, name)
        self._gcxgc_peaks = []
        self.figure.clear()
        self.figure.set_layout_engine('none')       # render_gcxgc turned constrained layout on
        self.tic_ax = self.figure.add_subplot(211)
        self.chromatogram_ax = self.figure.add_subplot(212, sharex=self.tic_ax)
        self._setup_empty_plots()

    def overlay_gcxgc_peaks(self, peaks: list) -> None:
        """Overlay detected peaks as red + markers on every heatmap shown."""
        if not hasattr(self, '_gcxgc_axes'):
            return
        self._gcxgc_peaks = peaks
        rt1_vals = [p.rt1 for p in peaks]
        rt2_vals = [p.rt2 for p in peaks]
        for ax in self._gcxgc_axes:
            for artist in getattr(ax, '_gcxgc_peak_scatter', []):
                try:
                    artist.remove()
                except Exception:
                    pass
            sc = ax.scatter(rt1_vals, rt2_vals, c='red', s=15, marker='+',
                            linewidths=0.8, zorder=5)
            ax._gcxgc_peak_scatter = [sc]
        self.canvas.draw()

    def _on_gcxgc_click(self, event) -> None:
        """Map a click on either heatmap to the nearest GCxGC2DPeak."""
        ax = event.inaxes
        if ax is None or ax not in getattr(self, '_gcxgc_axes', []):
            return
        if not self._gcxgc_peaks:
            return
        clicked_rt1 = event.xdata
        clicked_rt2 = event.ydata
        if clicked_rt1 is None or clicked_rt2 is None:
            return
        rt1_range = ax.get_xlim()
        rt2_range = ax.get_ylim()
        rt1_scale = (rt1_range[1] - rt1_range[0]) or 1
        rt2_scale = (rt2_range[1] - rt2_range[0]) or 1
        best = min(
            self._gcxgc_peaks,
            key=lambda p: ((p.rt1 - clicked_rt1) / rt1_scale) ** 2
                        + ((p.rt2 - clicked_rt2) / rt2_scale) ** 2,
        )
        self.peak_selected.emit(best)
