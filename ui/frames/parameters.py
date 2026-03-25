from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QFormLayout,
    QCheckBox, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QHBoxLayout,
    QLineEdit, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
import numpy as np
import numpy as np

class ParametersFrame(QWidget):
    """Frame containing integration parameters for chromatogram processing."""
    
    # Add signal for when parameters change
    parameters_changed = Signal(dict)
    
    # Add signal for MS baseline correction button click
    ms_baseline_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        
        # Track which methods use lambda
        self.lam_methods = {"asls", "airpls", "arpls", "mixture_model", "irsqr"}
        
        # Current parameters dictionary - update the baseline section
        self.current_params = {
            'smoothing': {
                'enabled': False,
                'method': 'whittaker',
                'median_enabled': False,
                'median_kernel': 5,
                'lambda': 1e-1,
                'diff_order': 1,
                'savgol_window': 3,
                'savgol_polyorder': 1
            },
            'baseline': {
                'show_corrected': False,
                'method': 'arpls',  # Changed from 'asls' to 'arpls'
                'lambda': 1e4,      # Changed from 1e6 to 1e4
                'asymmetry': 0.01,
                'baseline_offset': 0.0,
                'align_tic': False,  # Add alignment option
                'break_points': [],  # List of {'time': float, 'tolerance': float}
                'fastchrom': {
                    'half_window': None,
                    'smooth_half_window': None,
                }
            },
            'peaks': {
                'enabled': False,
                'mode': 'classical',  # 'classical' or 'peak_splitting'
                'min_prominence': 1e5,  # Changed from 0.5 to 1e5
                'min_height': 0.0,
                'min_width': 0.0,
                'range_filters': []  # List of [start, end] time ranges
            },
            'peak_splitting': {
                'splitting_method': 'geometric',
                'windows': [],  # List of [start, end] — empty = split entire chromatogram
                # Shared parameters
                'heatmap_threshold': 0.36,
                'pre_fit_signal_threshold': 0.001,
                'min_area_frac': 0.15,
                'valley_threshold_frac': 0.48,
                # EMG-only
                'mu_bound_factor': 0.68,
                'fat_threshold_frac': 0.44,
                'dedup_sigma_factor': 1.32,
                # Geometric-only
                'dedup_rt_tolerance': 0.005,
            },
            'negative_peaks': {
                'enabled': False,
                'min_prominence': 1e5,
            },
            'shoulders': {
                'enabled': False,
                'window_length': 41,
                'polyorder': 3,
                'sensitivity': 8,
                'apex_distance': 10
            },
            'integration': {
                'peak_groups': []  # List of [start, end] time windows for grouping
            }
        }
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        
        # Create scrollable area for parameters
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout(self.params_widget)
        
        # Add title
        parameters_label = QLabel("Integration Parameters")
        parameters_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.params_layout.addWidget(parameters_label)

        # Dict to hold group box references for visibility toggling
        self.section_groups = {}
        
        # Add smoothing group
        self._init_smoothing_controls()
        
        # Add baseline group
        self._init_baseline_controls()
        
        # Add peaks group
        self._init_peaks_controls()

        # Add negative peaks group
        self._init_negative_peaks_controls()

        # Add shoulder detection group
        self._init_shoulder_controls()

        # Add range filter controls
        self._init_range_filter_controls()

        # Add peak grouping controls
        self._init_peak_grouping_controls()
        
        # Add stretch at the end to push everything to the top
        self.params_layout.addStretch()
        
        # Set the widget for the scroll area
        self.params_scroll.setWidget(self.params_widget)
        
        # Add scroll area to main layout
        self.layout.addWidget(self.params_scroll)
    
    def _init_smoothing_controls(self):
        """Initialize smoothing controls"""
        smoothing_group = QGroupBox("Signal Smoothing")
        form_layout = QFormLayout(smoothing_group)
        
        # Enable/disable checkbox
        self.smoothing_enabled = QCheckBox("Enable Smoothing")
        self.smoothing_enabled.setChecked(self.current_params['smoothing']['enabled'])
        self.smoothing_enabled.stateChanged.connect(self._on_smoothing_toggled)
        form_layout.addRow(self.smoothing_enabled)

        # Method selector
        self.smooth_method_combo = QComboBox()
        self.smooth_method_combo.addItem("Whittaker", "whittaker")
        self.smooth_method_combo.addItem("Savitzky-Golay", "savgol")
        self.smooth_method_combo.setCurrentIndex(0)
        self.smooth_method_combo.currentIndexChanged.connect(self._on_smooth_method_changed)
        form_layout.addRow("Method:", self.smooth_method_combo)

        # --- Median pre-filter (shared by both methods) ---
        self.median_enabled = QCheckBox("Median Pre-filter")
        self.median_enabled.setChecked(self.current_params['smoothing']['median_enabled'])
        self.median_enabled.stateChanged.connect(self._on_median_toggled)
        form_layout.addRow(self.median_enabled)

        self.median_kernel_container = QWidget()
        median_kernel_layout = QHBoxLayout(self.median_kernel_container)
        median_kernel_layout.setContentsMargins(0, 0, 0, 0)

        self.median_kernel_slider = QSlider(Qt.Horizontal)
        self.median_kernel_slider.setMinimum(3)
        self.median_kernel_slider.setMaximum(31)
        self.median_kernel_slider.setValue(self.current_params['smoothing']['median_kernel'])
        self.median_kernel_slider.setSingleStep(2)
        self.median_kernel_slider.setPageStep(2)
        self.median_kernel_slider.setTickInterval(2)
        self.median_kernel_slider.setTickPosition(QSlider.TicksBelow)
        self.median_kernel_slider.valueChanged.connect(self._on_median_kernel_changed)

        self.median_kernel_spinbox = QSpinBox()
        self.median_kernel_spinbox.setMinimum(3)
        self.median_kernel_spinbox.setMaximum(31)
        self.median_kernel_spinbox.setValue(self.current_params['smoothing']['median_kernel'])
        self.median_kernel_spinbox.setSingleStep(2)
        self.median_kernel_spinbox.valueChanged.connect(self.median_kernel_slider.setValue)

        median_kernel_layout.addWidget(self.median_kernel_slider, 3)
        median_kernel_layout.addWidget(self.median_kernel_spinbox, 1)

        self.median_kernel_row_label = QLabel("Median kernel:")
        form_layout.addRow(self.median_kernel_row_label, self.median_kernel_container)

        # --- Whittaker controls ---

        # Lambda (smoothing penalty) - log10 scale slider
        self.smooth_lam_container = QWidget()
        smooth_lam_layout = QHBoxLayout(self.smooth_lam_container)
        smooth_lam_layout.setContentsMargins(0, 0, 0, 0)

        self.smooth_lam_slider = QSlider(Qt.Horizontal)
        self.smooth_lam_slider.setMinimum(-3)
        self.smooth_lam_slider.setMaximum(6)
        default_log = int(round(np.log10(self.current_params['smoothing']['lambda'])))
        self.smooth_lam_slider.setValue(default_log)
        self.smooth_lam_slider.setSingleStep(1)
        self.smooth_lam_slider.setPageStep(1)
        self.smooth_lam_slider.setTickInterval(1)
        self.smooth_lam_slider.setTickPosition(QSlider.TicksBelow)
        self.smooth_lam_slider.valueChanged.connect(self._on_smooth_lambda_changed)

        self.smooth_lam_spinbox = QSpinBox()
        self.smooth_lam_spinbox.setMinimum(-3)
        self.smooth_lam_spinbox.setMaximum(6)
        self.smooth_lam_spinbox.setValue(default_log)
        self.smooth_lam_spinbox.setSingleStep(1)
        self.smooth_lam_spinbox.valueChanged.connect(self.smooth_lam_slider.setValue)

        self.smooth_lam_label = QLabel(f"\u03bb = 10^{default_log}")

        smooth_lam_layout.addWidget(self.smooth_lam_slider, 3)
        smooth_lam_layout.addWidget(self.smooth_lam_spinbox, 1)
        smooth_lam_layout.addWidget(self.smooth_lam_label, 1)

        form_layout.addRow("Smoothing \u03bb:", self.smooth_lam_container)

        # Diff order combo (1 = slope penalty, 2 = curvature penalty)
        self.smooth_diff_order_combo = QComboBox()
        self.smooth_diff_order_combo.addItem("d=1 (slope)", 1)
        self.smooth_diff_order_combo.addItem("d=2 (curvature)", 2)
        self.smooth_diff_order_combo.setCurrentIndex(0)
        self.smooth_diff_order_combo.currentIndexChanged.connect(self._on_smooth_diff_order_changed)
        self.smooth_diff_order_row_label = QLabel("Penalty order:")
        form_layout.addRow(self.smooth_diff_order_row_label, self.smooth_diff_order_combo)

        # --- Savitzky-Golay controls ---

        # Window length
        self.savgol_window_container = QWidget()
        savgol_window_layout = QHBoxLayout(self.savgol_window_container)
        savgol_window_layout.setContentsMargins(0, 0, 0, 0)

        self.savgol_window_slider = QSlider(Qt.Horizontal)
        self.savgol_window_slider.setMinimum(3)
        self.savgol_window_slider.setMaximum(51)
        self.savgol_window_slider.setValue(self.current_params['smoothing']['savgol_window'])
        self.savgol_window_slider.setSingleStep(2)
        self.savgol_window_slider.setPageStep(2)
        self.savgol_window_slider.setTickInterval(2)
        self.savgol_window_slider.setTickPosition(QSlider.TicksBelow)
        self.savgol_window_slider.valueChanged.connect(self._on_savgol_window_changed)

        self.savgol_window_spinbox = QSpinBox()
        self.savgol_window_spinbox.setMinimum(3)
        self.savgol_window_spinbox.setMaximum(51)
        self.savgol_window_spinbox.setValue(self.current_params['smoothing']['savgol_window'])
        self.savgol_window_spinbox.setSingleStep(2)
        self.savgol_window_spinbox.valueChanged.connect(self.savgol_window_slider.setValue)

        savgol_window_layout.addWidget(self.savgol_window_slider, 3)
        savgol_window_layout.addWidget(self.savgol_window_spinbox, 1)

        self.savgol_window_row_label = QLabel("Window length:")
        form_layout.addRow(self.savgol_window_row_label, self.savgol_window_container)

        # Polynomial order
        self.savgol_polyorder_container = QWidget()
        savgol_poly_layout = QHBoxLayout(self.savgol_polyorder_container)
        savgol_poly_layout.setContentsMargins(0, 0, 0, 0)

        self.savgol_polyorder_slider = QSlider(Qt.Horizontal)
        self.savgol_polyorder_slider.setMinimum(1)
        self.savgol_polyorder_slider.setMaximum(5)
        self.savgol_polyorder_slider.setValue(self.current_params['smoothing']['savgol_polyorder'])
        self.savgol_polyorder_slider.setSingleStep(1)
        self.savgol_polyorder_slider.setTickInterval(1)
        self.savgol_polyorder_slider.setTickPosition(QSlider.TicksBelow)
        self.savgol_polyorder_slider.valueChanged.connect(self._on_savgol_polyorder_changed)

        self.savgol_polyorder_spinbox = QSpinBox()
        self.savgol_polyorder_spinbox.setMinimum(1)
        self.savgol_polyorder_spinbox.setMaximum(5)
        self.savgol_polyorder_spinbox.setValue(self.current_params['smoothing']['savgol_polyorder'])
        self.savgol_polyorder_spinbox.setSingleStep(1)
        self.savgol_polyorder_spinbox.valueChanged.connect(self.savgol_polyorder_slider.setValue)

        savgol_poly_layout.addWidget(self.savgol_polyorder_slider, 3)
        savgol_poly_layout.addWidget(self.savgol_polyorder_spinbox, 1)

        self.savgol_polyorder_row_label = QLabel("Polynomial order:")
        form_layout.addRow(self.savgol_polyorder_row_label, self.savgol_polyorder_container)

        # Hide savgol controls by default (whittaker is selected)
        self._update_smoothing_method_visibility()

        # Enable/disable controls based on initial state
        self._update_smoothing_controls_state()
        
        # Add to parameters layout
        self.section_groups['smoothing'] = smoothing_group
        self.params_layout.addWidget(smoothing_group)

    def _init_baseline_controls(self):
        """Initialize baseline correction controls"""
        baseline_group = QGroupBox("Baseline Correction")
        form_layout = QFormLayout(baseline_group)
        
        # Change the checkbox label and meaning
        self.baseline_show_corrected = QCheckBox("Show Corrected Signal")
        self.baseline_show_corrected.setChecked(self.current_params['baseline']['show_corrected'])
        self.baseline_show_corrected.stateChanged.connect(self._on_baseline_display_toggled)
        form_layout.addRow(self.baseline_show_corrected)
        
        # Add explanation label
        baseline_info = QLabel("Unchecked: Show raw signal with baseline\nChecked: Show corrected signal (baseline at zero)")
        baseline_info.setStyleSheet("color: #666666; font-size: 10px;")
        form_layout.addRow("", baseline_info)
        
        # Algorithm selection dropdown - always enabled
        self.baseline_algorithm = QComboBox()
        self.baseline_algorithm.addItem("asls - Asymmetric Least Squares", "asls")
        self.baseline_algorithm.addItem("imodpoly - Improved Modified Polynomial", "imodpoly")
        self.baseline_algorithm.addItem("modpoly - Modified Polynomial", "modpoly")
        self.baseline_algorithm.addItem("snip - Statistics-sensitive Non-linear", "snip")
        self.baseline_algorithm.addItem("airpls - Adaptive Iteratively Reweighted", "airpls")
        self.baseline_algorithm.addItem("arpls - Asymmetrically Reweighted", "arpls")
        self.baseline_algorithm.addItem("mixture_model - Mixture Model", "mixture_model")
        self.baseline_algorithm.addItem("irsqr - Spline Quantile Regression", "irsqr")
        self.baseline_algorithm.addItem("fastchrom - FastChrom's Baseline", "fastchrom")
        
        # Set current method
        index = self.baseline_algorithm.findData(self.current_params['baseline']['method'])
        if index >= 0:
            self.baseline_algorithm.setCurrentIndex(index)
            
        self.baseline_algorithm.currentIndexChanged.connect(self._on_baseline_method_changed)
        form_layout.addRow("Algorithm:", self.baseline_algorithm)
        
        # Lambda parameter slider - always enabled for supported methods
        self.lam_container = QWidget()
        lam_layout = QHBoxLayout(self.lam_container)
        lam_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lam_slider = QSlider(Qt.Horizontal)
        self.lam_slider.setMinimum(2)
        self.lam_slider.setMaximum(12)
        
        # Convert lambda to slider value (log10)
        lambda_value = self.current_params['baseline']['lambda']
        lambda_exp = int(np.log10(lambda_value))
        self.lam_slider.setValue(lambda_exp)
        
        self.lam_slider.setTickPosition(QSlider.TicksBelow)
        self.lam_slider.valueChanged.connect(self._on_lambda_changed)
        
        self.lam_spinbox = QSpinBox()
        self.lam_spinbox.setMinimum(2)
        self.lam_spinbox.setMaximum(12)
        self.lam_spinbox.setValue(lambda_exp)
        self.lam_spinbox.valueChanged.connect(self.lam_slider.setValue)
        
        lam_layout.addWidget(self.lam_slider, 3)
        lam_layout.addWidget(self.lam_spinbox, 1)
        
        self.lam_label = QLabel(f"λ = 10^{lambda_exp}")
        
        form_layout.addRow("Lambda (λ):", self.lam_container)
        form_layout.addRow("", self.lam_label)
        
        # Baseline offset - shifts baseline down (positive) to increase peak areas
        self.baseline_offset_input = QLineEdit("0.0")
        self.baseline_offset_input.setToolTip(
            "Offset subtracted from the computed baseline (in signal units).\n"
            "Positive values lower the baseline, increasing peak areas.\n"
            "Useful for matching baseline positions from other integration engines.\n"
            "Accepts scientific notation (e.g., 1.5e-2)."
        )
        self.baseline_offset_input.editingFinished.connect(self._on_baseline_offset_changed)
        form_layout.addRow("Baseline Offset:", self.baseline_offset_input)
        # FastChrom-specific parameters
        self.fastchrom_container = QWidget()
        fc_layout = QFormLayout(self.fastchrom_container)
        fc_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fc_half_window = QSpinBox()
        self.fc_half_window.setMinimum(0)
        self.fc_half_window.setMaximum(500)
        self.fc_half_window.setSpecialValueText("Auto")
        self.fc_half_window.setValue(0)
        self.fc_half_window.setToolTip("Half-window for rolling std dev (~FWHM of peaks). 0 = Auto.")
        self.fc_half_window.valueChanged.connect(self._on_fastchrom_param_changed)
        fc_layout.addRow("Half Window:", self.fc_half_window)
        
        self.fc_smooth_half_window = QSpinBox()
        self.fc_smooth_half_window.setMinimum(0)
        self.fc_smooth_half_window.setMaximum(500)
        self.fc_smooth_half_window.setSpecialValueText("Auto")
        self.fc_smooth_half_window.setValue(0)
        self.fc_smooth_half_window.setToolTip("Half-window for smoothing the interpolated baseline. 0 = Auto.")
        self.fc_smooth_half_window.valueChanged.connect(self._on_fastchrom_param_changed)
        fc_layout.addRow("Smooth Half Window:", self.fc_smooth_half_window)
        
        form_layout.addRow("", self.fastchrom_container)
        # Update lambda and fastchrom control visibility based on method
        self.update_lam_visibility()
        
        # --- Advanced baseline options (hideable via visibility config) ---
        self.baseline_advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.baseline_advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        # Add MS baseline correction button
        ms_baseline_frame = QFrame()
        ms_baseline_layout = QVBoxLayout(ms_baseline_frame)
        ms_baseline_layout.setContentsMargins(0, 10, 0, 0)
        
        ms_baseline_label = QLabel("Advanced Options:")
        ms_baseline_label.setStyleSheet("font-weight: bold;")
        ms_baseline_layout.addWidget(ms_baseline_label)
        
        self.ms_baseline_button = QPushButton("Apply Full MS Baseline Correction")
        self.ms_baseline_button.setToolTip("Apply baseline correction to all m/z traces individually")
        self.ms_baseline_button.clicked.connect(self._on_ms_baseline_clicked)
        ms_baseline_layout.addWidget(self.ms_baseline_button)
        
        advanced_layout.addWidget(ms_baseline_frame)
        
        # Add TIC alignment checkbox
        self.align_tic = QCheckBox("Align TIC with FID")
        self.align_tic.setChecked(self.current_params['baseline'].get('align_tic', False))
        self.align_tic.stateChanged.connect(self._on_align_tic_toggled)
        advanced_layout.addWidget(self.align_tic)
        
        # Add explanation label
        align_info = QLabel("Corrects for small time delays between FID and MS detectors")
        align_info.setStyleSheet("color: #666666; font-size: 10px;")
        advanced_layout.addWidget(align_info)
        
        # Break point controls for segmented baseline fitting
        bp_frame = QFrame()
        bp_frame.setFrameStyle(QFrame.Box)
        bp_layout = QVBoxLayout(bp_frame)
        
        bp_label = QLabel("Signal Break Points")
        bp_label.setStyleSheet("font-weight: bold;")
        bp_layout.addWidget(bp_label)
        
        bp_info = QLabel("Split baseline fitting at valve-switch step changes.\nEach segment is fitted independently.")
        bp_info.setStyleSheet("color: #666666; font-size: 10px;")
        bp_layout.addWidget(bp_info)
        
        self.break_point_table = QTableWidget(0, 3)
        self.break_point_table.setHorizontalHeaderLabels(["Time (min)", "Tolerance", ""])
        self.break_point_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.break_point_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.break_point_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.break_point_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.break_point_table.setMaximumHeight(120)
        bp_layout.addWidget(self.break_point_table)
        
        bp_add_btn = QPushButton("+ Add Break Point")
        bp_add_btn.clicked.connect(self._add_break_point_row)
        bp_layout.addWidget(bp_add_btn)
        
        advanced_layout.addWidget(bp_frame)

        form_layout.addRow(self.baseline_advanced_container)
        
        # Add to parameters layout
        self.section_groups['baseline'] = baseline_group
        self.section_groups['baseline_advanced'] = self.baseline_advanced_container
        self.params_layout.addWidget(baseline_group)

    def _init_peaks_controls(self):
        """Initialize peak detection controls"""
        peaks_group = QGroupBox("Peak Detection")
        form_layout = QFormLayout(peaks_group)

        # Enable/disable checkbox
        self.peaks_enabled = QCheckBox("Enable Peak Detection")
        self.peaks_enabled.setChecked(self.current_params['peaks']['enabled'])
        self.peaks_enabled.stateChanged.connect(self._on_peaks_toggled)
        form_layout.addRow(self.peaks_enabled)

        # Peak detection mode combo box
        self.peak_mode_combo = QComboBox()
        self.peak_mode_combo.addItem("Classical", "classical")
        self.peak_mode_combo.addItem("Peak Splitting (U-Net)", "peak_splitting")

        # Grey out peak splitting if not available
        from logic.deconvolution import is_available as deconv_available
        if not deconv_available():
            model = self.peak_mode_combo.model()
            item = model.item(1)
            item.setEnabled(False)
            item.setToolTip("Requires PyTorch and model weights in cache/gc_heatmap_unet.pth")

        self.peak_mode_combo.currentIndexChanged.connect(self._on_peak_mode_changed)
        form_layout.addRow("Detection Mode:", self.peak_mode_combo)

        # Direct entry box for prominence with validation
        self.prominence_entry = QLineEdit()
        self.prominence_entry.setText(str(self.current_params['peaks']['min_prominence']))

        # Set up validator for decimal and scientific notation
        validator = QRegularExpressionValidator(QRegularExpression(r'^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'))
        self.prominence_entry.setValidator(validator)

        self.prominence_entry.editingFinished.connect(self._on_prominence_entry_changed)
        self.prominence_label = QLabel("Min Prominence:")
        form_layout.addRow(self.prominence_label, self.prominence_entry)

        # Help label (changes with mode)
        self.prominence_help = QLabel("Enter a number or scientific notation (e.g. 1e-3)")
        form_layout.addRow("", self.prominence_help)

        # Min width filter (samples) — filters narrow noise spikes
        self.min_width_entry = QLineEdit()
        self.min_width_entry.setText(str(int(self.current_params['peaks'].get('min_width', 0))))
        self.min_width_entry.setToolTip(
            "Minimum peak width in samples (data points).\n"
            "Noise spikes are 1–3 samples wide; real GC peaks are typically wider.\n"
            "Set to 0 to disable width filtering."
        )
        self.min_width_entry.editingFinished.connect(self._on_min_width_changed)
        form_layout.addRow("Min Width (samples):", self.min_width_entry)

        # ── Peak Splitting sub-controls (visible only in peak_splitting mode) ──

        self.peak_splitting_controls_frame = QFrame()
        peak_splitting_layout = QFormLayout(self.peak_splitting_controls_frame)
        peak_splitting_layout.setContentsMargins(0, 4, 0, 0)

        # Splitting method
        self.splitting_method_combo = QComboBox()
        self.splitting_method_combo.addItem("Geometric Splitting", "geometric")
        self.splitting_method_combo.addItem("EMG Curve Fitting", "emg")
        self.splitting_method_combo.setToolTip(
            "Geometric: fast, accurate RT & area, best for large peaks.\n"
            "EMG: better at detecting small peaks, recovers peak shape parameters."
        )
        self.splitting_method_combo.currentIndexChanged.connect(self._on_splitting_method_changed)
        peak_splitting_layout.addRow("Splitting Method:", self.splitting_method_combo)

        # Peak splitting windows table
        windows_info = QLabel("Apply peak splitting only in these time ranges.\n"
                              "Leave empty to split the entire chromatogram.")
        windows_info.setStyleSheet("color: #666666; font-size: 10px;")
        peak_splitting_layout.addRow(windows_info)

        self.peak_splitting_windows_table = QTableWidget(0, 3)
        self.peak_splitting_windows_table.setHorizontalHeaderLabels(["Start (min)", "End (min)", ""])
        self.peak_splitting_windows_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.peak_splitting_windows_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.peak_splitting_windows_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.peak_splitting_windows_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.peak_splitting_windows_table.setMaximumHeight(120)
        peak_splitting_layout.addRow(self.peak_splitting_windows_table)

        add_window_btn = QPushButton("+ Add Window")
        add_window_btn.clicked.connect(self._add_peak_splitting_window_row)
        peak_splitting_layout.addRow(add_window_btn)

        # Advanced toggle
        self.peak_splitting_advanced_toggle = QPushButton("▶ Advanced")
        self.peak_splitting_advanced_toggle.setFlat(True)
        self.peak_splitting_advanced_toggle.setStyleSheet("text-align: left; padding: 2px;")
        self.peak_splitting_advanced_toggle.clicked.connect(self._toggle_peak_splitting_advanced)
        peak_splitting_layout.addRow(self.peak_splitting_advanced_toggle)

        # Advanced frame (hidden by default)
        self.peak_splitting_advanced_frame = QFrame()
        adv_layout = QFormLayout(self.peak_splitting_advanced_frame)
        adv_layout.setContentsMargins(8, 0, 0, 0)

        # Shared params
        self.heatmap_threshold_spin = QDoubleSpinBox()
        self.heatmap_threshold_spin.setRange(0.10, 0.50)
        self.heatmap_threshold_spin.setSingleStep(0.02)
        self.heatmap_threshold_spin.setDecimals(2)
        self.heatmap_threshold_spin.setValue(self.current_params['peak_splitting']['heatmap_threshold'])
        self.heatmap_threshold_spin.setToolTip("U-Net confidence threshold for detecting peak apexes.\nLower = more sensitive, higher = fewer false positives.")
        self.heatmap_threshold_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('heatmap_threshold', v))
        adv_layout.addRow("U-Net Confidence:", self.heatmap_threshold_spin)

        self.pre_fit_signal_entry = QLineEdit()
        self.pre_fit_signal_entry.setText(str(self.current_params['peak_splitting']['pre_fit_signal_threshold']))
        self.pre_fit_signal_entry.setToolTip(
            "Skip U-Net detections where the signal is below this fraction of the peak maximum.\n"
            "Pre-fit noise gate — prevents fitting on baseline noise.\n"
            "1e-3 is a good starting point. 0 = no pre-filtering."
        )
        self.pre_fit_signal_entry.editingFinished.connect(self._on_peak_splitting_pre_fit_signal_changed)
        adv_layout.addRow("Pre-fit Signal Gate:", self.pre_fit_signal_entry)

        self.min_area_frac_spin = QDoubleSpinBox()
        self.min_area_frac_spin.setRange(0.00, 0.30)
        self.min_area_frac_spin.setSingleStep(0.01)
        self.min_area_frac_spin.setDecimals(2)
        self.min_area_frac_spin.setValue(self.current_params['peak_splitting']['min_area_frac'])
        self.min_area_frac_spin.setToolTip("Remove components with area below this fraction of the median.\nHigher = more aggressive phantom filtering.")
        self.min_area_frac_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('min_area_frac', v))
        adv_layout.addRow("Min Area Fraction:", self.min_area_frac_spin)

        self.valley_threshold_spin = QDoubleSpinBox()
        self.valley_threshold_spin.setRange(0.20, 0.80)
        self.valley_threshold_spin.setSingleStep(0.05)
        self.valley_threshold_spin.setDecimals(2)
        self.valley_threshold_spin.setValue(self.current_params['peak_splitting']['valley_threshold_frac'])
        self.valley_threshold_spin.setToolTip("Valley depth for splitting merged peaks.\nLower = split at shallower valleys, higher = require deeper valleys.")
        self.valley_threshold_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('valley_threshold_frac', v))
        adv_layout.addRow("Valley Depth:", self.valley_threshold_spin)

        # EMG-only params
        self.emg_only_label = QLabel("EMG-specific:")
        self.emg_only_label.setStyleSheet("color: #666666; font-size: 10px; font-weight: bold;")
        adv_layout.addRow(self.emg_only_label)

        self.mu_bound_spin = QDoubleSpinBox()
        self.mu_bound_spin.setRange(0.50, 3.00)
        self.mu_bound_spin.setSingleStep(0.1)
        self.mu_bound_spin.setDecimals(2)
        self.mu_bound_spin.setValue(self.current_params['peak_splitting']['mu_bound_factor'])
        self.mu_bound_spin.setToolTip("Peak position bounds as multiples of sigma.\nSmaller = tighter peak positions, larger = more flexibility.")
        self.mu_bound_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('mu_bound_factor', v))
        adv_layout.addRow("Position Bounds (x sigma):", self.mu_bound_spin)

        self.fat_threshold_spin = QDoubleSpinBox()
        self.fat_threshold_spin.setRange(0.20, 0.80)
        self.fat_threshold_spin.setSingleStep(0.05)
        self.fat_threshold_spin.setDecimals(2)
        self.fat_threshold_spin.setValue(self.current_params['peak_splitting']['fat_threshold_frac'])
        self.fat_threshold_spin.setToolTip("FWHM threshold for flagging over-wide components.\nComponents wider than this fraction of the window trigger refitting.")
        self.fat_threshold_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('fat_threshold_frac', v))
        adv_layout.addRow("Width Threshold:", self.fat_threshold_spin)

        self.dedup_sigma_spin = QDoubleSpinBox()
        self.dedup_sigma_spin.setRange(0.00, 2.00)
        self.dedup_sigma_spin.setSingleStep(0.1)
        self.dedup_sigma_spin.setDecimals(2)
        self.dedup_sigma_spin.setValue(self.current_params['peak_splitting']['dedup_sigma_factor'])
        self.dedup_sigma_spin.setToolTip("Merge components within this many sigma of each other.\n0 = no deduplication.")
        self.dedup_sigma_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('dedup_sigma_factor', v))
        adv_layout.addRow("Dedup Distance (sigma):", self.dedup_sigma_spin)

        # Geometric-only params
        self.geo_only_label = QLabel("Geometric-specific:")
        self.geo_only_label.setStyleSheet("color: #666666; font-size: 10px; font-weight: bold;")
        adv_layout.addRow(self.geo_only_label)

        self.dedup_rt_spin = QDoubleSpinBox()
        self.dedup_rt_spin.setRange(0.000, 0.100)
        self.dedup_rt_spin.setSingleStep(0.005)
        self.dedup_rt_spin.setDecimals(3)
        self.dedup_rt_spin.setValue(self.current_params['peak_splitting']['dedup_rt_tolerance'])
        self.dedup_rt_spin.setToolTip("Merge components within this RT distance (minutes).\n0 = no deduplication.")
        self.dedup_rt_spin.valueChanged.connect(lambda v: self._on_peak_splitting_param_changed('dedup_rt_tolerance', v))
        adv_layout.addRow("Dedup Distance (min):", self.dedup_rt_spin)

        # Reset button
        self.peak_splitting_reset_btn = QPushButton("Reset to Optimized Defaults")
        self.peak_splitting_reset_btn.setToolTip("Restore Optuna-optimized defaults for the current splitting method.")
        self.peak_splitting_reset_btn.clicked.connect(self._reset_peak_splitting_defaults)
        adv_layout.addRow(self.peak_splitting_reset_btn)

        self.peak_splitting_advanced_frame.setVisible(False)
        peak_splitting_layout.addRow(self.peak_splitting_advanced_frame)

        # Initially hidden (shown when peak_splitting mode is selected)
        self.peak_splitting_controls_frame.setVisible(False)
        form_layout.addRow(self.peak_splitting_controls_frame)

        # Enable/disable controls based on initial state
        self._update_peaks_controls_state()
        self._update_peak_splitting_method_visibility()

        # Add to parameters layout
        self.section_groups['peaks'] = peaks_group
        self.params_layout.addWidget(peaks_group)

    def _init_negative_peaks_controls(self):
        """Initialize negative peak detection controls"""
        neg_peaks_group = QGroupBox("Negative Peak Detection")
        form_layout = QFormLayout(neg_peaks_group)

        # Enable/disable checkbox
        self.neg_peaks_enabled = QCheckBox("Enable Negative Peak Detection")
        self.neg_peaks_enabled.setChecked(self.current_params['negative_peaks']['enabled'])
        self.neg_peaks_enabled.stateChanged.connect(self._on_neg_peaks_toggled)
        form_layout.addRow(self.neg_peaks_enabled)

        # Direct entry box for prominence with validation
        self.neg_prominence_entry = QLineEdit()
        self.neg_prominence_entry.setText(str(self.current_params['negative_peaks']['min_prominence']))

        # Set up validator for decimal and scientific notation
        validator = QRegularExpressionValidator(QRegularExpression(r'^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'))
        self.neg_prominence_entry.setValidator(validator)

        self.neg_prominence_entry.editingFinished.connect(self._on_neg_prominence_entry_changed)
        form_layout.addRow("Min Prominence:", self.neg_prominence_entry)

        # Add help label
        neg_help = QLabel("Detects peaks below the baseline (e.g. H2 on He carrier in TCD).\nEnter a number or scientific notation (e.g. 1e5).")
        neg_help.setStyleSheet("color: #666666; font-size: 10px;")
        form_layout.addRow("", neg_help)

        # Enable/disable controls based on initial state
        self._update_neg_peaks_controls_state()

        # Add to parameters layout
        self.section_groups['negative_peaks'] = neg_peaks_group
        self.params_layout.addWidget(neg_peaks_group)

    def _on_neg_peaks_toggled(self, state):
        """Handle negative peaks enable/disable toggle"""
        enabled = bool(state)
        self.current_params['negative_peaks']['enabled'] = enabled
        self._update_neg_peaks_controls_state()
        self.parameters_changed.emit(self.current_params)

    def _on_neg_prominence_entry_changed(self):
        """Handle negative peak prominence entry change with validation"""
        text = self.neg_prominence_entry.text()
        try:
            value = float(text)
            self.current_params['negative_peaks']['min_prominence'] = value

            # Provide visual feedback
            self.neg_prominence_entry.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff;")
            QTimer.singleShot(800, lambda: self.neg_prominence_entry.setStyleSheet(""))

            self.parameters_changed.emit(self.current_params)
        except ValueError:
            self.neg_prominence_entry.setStyleSheet("background-color: #ffcccc; border: 1px solid #ff9999;")
            QTimer.singleShot(800, lambda: self.neg_prominence_entry.setStyleSheet(""))

    def _update_neg_peaks_controls_state(self):
        """Update enabled state of negative peaks controls"""
        enabled = self.neg_peaks_enabled.isChecked()
        self.neg_prominence_entry.setEnabled(enabled)

    def _init_shoulder_controls(self):
        """Initialize shoulder detection controls"""
        shoulder_group = QGroupBox("Shoulder Detection")
        form_layout = QFormLayout(shoulder_group)
        
        # Enable/disable checkbox
        self.shoulders_enabled = QCheckBox("Enable Shoulder Detection")
        self.shoulders_enabled.setChecked(self.current_params['shoulders']['enabled'])
        self.shoulders_enabled.stateChanged.connect(self._on_shoulders_toggled)
        form_layout.addRow(self.shoulders_enabled)
        
        shoulder_info = QLabel("Detects overlapping peaks (shoulders) on the flanks of main peaks\nusing noise-adaptive derivative analysis.")
        shoulder_info.setStyleSheet("color: #666666; font-size: 10px;")
        form_layout.addRow("", shoulder_info)
        
        # --- Primary detection parameters ---
        detection_frame = QFrame()
        detection_frame.setFrameStyle(QFrame.Box)
        detection_layout = QFormLayout(detection_frame)
        
        # Sensitivity slider (1-10, higher = more sensitive)
        sensitivity_container = QWidget()
        sensitivity_layout = QHBoxLayout(sensitivity_container)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        
        self.shoulder_sensitivity_slider = QSlider(Qt.Horizontal)
        self.shoulder_sensitivity_slider.setMinimum(1)
        self.shoulder_sensitivity_slider.setMaximum(10)
        self.shoulder_sensitivity_slider.setValue(self.current_params['shoulders']['sensitivity'])
        self.shoulder_sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.shoulder_sensitivity_slider.setTickInterval(1)
        self.shoulder_sensitivity_slider.valueChanged.connect(self._on_shoulder_sensitivity_changed)
        
        self.shoulder_sensitivity_spinbox = QSpinBox()
        self.shoulder_sensitivity_spinbox.setMinimum(1)
        self.shoulder_sensitivity_spinbox.setMaximum(10)
        self.shoulder_sensitivity_spinbox.setValue(self.current_params['shoulders']['sensitivity'])
        self.shoulder_sensitivity_spinbox.valueChanged.connect(self.shoulder_sensitivity_slider.setValue)
        
        sensitivity_layout.addWidget(self.shoulder_sensitivity_slider, 3)
        sensitivity_layout.addWidget(self.shoulder_sensitivity_spinbox, 1)
        
        sensitivity_label = QLabel("Sensitivity:")
        sensitivity_label.setToolTip(
            "Controls how aggressively shoulders are detected.\n"
            "Higher values detect smaller/weaker shoulders but may\n"
            "introduce false positives. Lower values are more conservative.\n\n"
            "1 = Very strict (only obvious shoulders)\n"
            "5 = Moderate\n"
            "8 = Default (good balance)\n"
            "10 = Very sensitive (may detect noise features)"
        )
        detection_layout.addRow(sensitivity_label, sensitivity_container)
        
        # Min peak-shoulder distance
        distance_container = QWidget()
        distance_layout = QHBoxLayout(distance_container)
        distance_layout.setContentsMargins(0, 0, 0, 0)
        
        self.shoulder_distance_slider = QSlider(Qt.Horizontal)
        self.shoulder_distance_slider.setMinimum(5)
        self.shoulder_distance_slider.setMaximum(30)
        self.shoulder_distance_slider.setValue(self.current_params['shoulders']['apex_distance'])
        self.shoulder_distance_slider.setTickPosition(QSlider.TicksBelow)
        self.shoulder_distance_slider.valueChanged.connect(self._on_shoulder_distance_changed)
        
        self.shoulder_distance_spinbox = QSpinBox()
        self.shoulder_distance_spinbox.setMinimum(5)
        self.shoulder_distance_spinbox.setMaximum(30)
        self.shoulder_distance_spinbox.setValue(self.current_params['shoulders']['apex_distance'])
        self.shoulder_distance_spinbox.valueChanged.connect(self.shoulder_distance_slider.setValue)
        
        distance_layout.addWidget(self.shoulder_distance_slider, 3)
        distance_layout.addWidget(self.shoulder_distance_spinbox, 1)
        
        distance_label = QLabel("Min Separation:")
        distance_label.setToolTip(
            "Minimum distance (in data points) between a main peak\n"
            "apex and a shoulder candidate. Increase to avoid detecting\n"
            "features too close to the peak center."
        )
        detection_layout.addRow(distance_label, distance_container)
        
        form_layout.addRow("Detection:", detection_frame)
        
        # --- Advanced smoothing parameters (collapsible) ---
        self.shoulder_advanced_toggle = QPushButton("▶ Advanced")
        self.shoulder_advanced_toggle.setFlat(True)
        self.shoulder_advanced_toggle.setStyleSheet(
            "QPushButton { text-align: left; color: #666666; font-size: 11px; padding: 2px; }"
            "QPushButton:hover { color: #333333; }"
        )
        self.shoulder_advanced_toggle.clicked.connect(self._toggle_shoulder_advanced)
        form_layout.addRow(self.shoulder_advanced_toggle)
        
        self.shoulder_advanced_frame = QFrame()
        self.shoulder_advanced_frame.setFrameStyle(QFrame.Box)
        advanced_layout = QFormLayout(self.shoulder_advanced_frame)
        
        advanced_info = QLabel("Savitzky-Golay filter parameters for derivative computation.\nLarger window = more smoothing, less noise sensitivity.")
        advanced_info.setStyleSheet("color: #666666; font-size: 10px;")
        advanced_layout.addRow("", advanced_info)
        
        # Window length
        window_container = QWidget()
        window_layout = QHBoxLayout(window_container)
        window_layout.setContentsMargins(0, 0, 0, 0)
        
        self.shoulder_window_slider = QSlider(Qt.Horizontal)
        self.shoulder_window_slider.setMinimum(5)
        self.shoulder_window_slider.setMaximum(101)
        self.shoulder_window_slider.setValue(self.current_params['shoulders']['window_length'])
        self.shoulder_window_slider.setSingleStep(2)
        self.shoulder_window_slider.setTickPosition(QSlider.TicksBelow)
        self.shoulder_window_slider.valueChanged.connect(self._on_shoulder_window_changed)
        
        self.shoulder_window_spinbox = QSpinBox()
        self.shoulder_window_spinbox.setMinimum(5)
        self.shoulder_window_spinbox.setMaximum(101)
        self.shoulder_window_spinbox.setValue(self.current_params['shoulders']['window_length'])
        self.shoulder_window_spinbox.setSingleStep(2)
        self.shoulder_window_spinbox.valueChanged.connect(self.shoulder_window_slider.setValue)
        
        window_layout.addWidget(self.shoulder_window_slider, 3)
        window_layout.addWidget(self.shoulder_window_spinbox, 1)
        
        window_label = QLabel("Smoothing Window:")
        window_label.setToolTip("Savitzky-Golay filter window size (must be odd).\nLarger = smoother derivatives, less sensitive to narrow features.")
        advanced_layout.addRow(window_label, window_container)
        
        # Polynomial order
        polyorder_container = QWidget()
        polyorder_layout = QHBoxLayout(polyorder_container)
        polyorder_layout.setContentsMargins(0, 0, 0, 0)
        
        self.shoulder_polyorder_slider = QSlider(Qt.Horizontal)
        self.shoulder_polyorder_slider.setMinimum(1)
        self.shoulder_polyorder_slider.setMaximum(5)
        self.shoulder_polyorder_slider.setValue(self.current_params['shoulders']['polyorder'])
        self.shoulder_polyorder_slider.setTickPosition(QSlider.TicksBelow)
        self.shoulder_polyorder_slider.valueChanged.connect(self._on_shoulder_polyorder_changed)
        
        self.shoulder_polyorder_spinbox = QSpinBox()
        self.shoulder_polyorder_spinbox.setMinimum(1)
        self.shoulder_polyorder_spinbox.setMaximum(5)
        self.shoulder_polyorder_spinbox.setValue(self.current_params['shoulders']['polyorder'])
        self.shoulder_polyorder_spinbox.valueChanged.connect(self.shoulder_polyorder_slider.setValue)
        
        polyorder_layout.addWidget(self.shoulder_polyorder_slider, 3)
        polyorder_layout.addWidget(self.shoulder_polyorder_spinbox, 1)
        
        polyorder_label = QLabel("Polynomial Order:")
        polyorder_label.setToolTip("Order of the polynomial fit.\nHigher = preserves sharper features but may amplify noise.")
        advanced_layout.addRow(polyorder_label, polyorder_container)
        
        self.shoulder_advanced_frame.setVisible(False)
        form_layout.addRow(self.shoulder_advanced_frame)
        
        # Enable/disable controls based on initial state
        self._update_shoulder_controls_state()
        
        # Add to parameters layout
        self.section_groups['shoulders'] = shoulder_group
        self.params_layout.addWidget(shoulder_group)

    def _init_range_filter_controls(self):
        """Initialize peak range filter table UI."""
        group = QGroupBox("Peak Range Filters")
        layout = QVBoxLayout(group)

        info = QLabel("Only keep peaks within these time ranges.\nLeave empty to keep all peaks.")
        info.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(info)

        self.range_filter_table = QTableWidget(0, 3)
        self.range_filter_table.setHorizontalHeaderLabels(["Start Time", "End Time", ""])
        self.range_filter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.range_filter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.range_filter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.range_filter_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.range_filter_table.setMaximumHeight(150)
        layout.addWidget(self.range_filter_table)

        add_btn = QPushButton("+ Add Range")
        add_btn.clicked.connect(self._add_range_filter_row)
        layout.addWidget(add_btn)

        self.section_groups['range_filters'] = group
        self.params_layout.addWidget(group)

    def _add_range_filter_row(self):
        """Add a new row to the range filter table."""
        row = self.range_filter_table.rowCount()
        self.range_filter_table.insertRow(row)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(0, 9999)
        start_spin.valueChanged.connect(self._on_range_filters_changed)
        self.range_filter_table.setCellWidget(row, 0, start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(0, 9999)
        end_spin.setValue(60.0)
        end_spin.valueChanged.connect(self._on_range_filters_changed)
        self.range_filter_table.setCellWidget(row, 1, end_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_range_filter_row(r))
        self.range_filter_table.setCellWidget(row, 2, del_btn)

    def _remove_range_filter_row(self, row):
        """Remove a row from the range filter table."""
        self.range_filter_table.removeRow(row)
        # Reconnect delete buttons with correct row indices
        for r in range(self.range_filter_table.rowCount()):
            del_btn = self.range_filter_table.cellWidget(r, 2)
            if del_btn:
                del_btn.clicked.disconnect()
                del_btn.clicked.connect(lambda checked, r=r: self._remove_range_filter_row(r))
        self._on_range_filters_changed()

    def _on_range_filters_changed(self, _=None):
        """Collect range filter values and emit parameters_changed."""
        ranges = []
        for r in range(self.range_filter_table.rowCount()):
            start_w = self.range_filter_table.cellWidget(r, 0)
            end_w = self.range_filter_table.cellWidget(r, 1)
            if start_w and end_w:
                ranges.append([start_w.value(), end_w.value()])
        self.current_params['peaks']['range_filters'] = ranges
        self.parameters_changed.emit(self.current_params)

    def _init_peak_grouping_controls(self):
        """Initialize peak grouping table UI."""
        group = QGroupBox("Peak Grouping")
        layout = QVBoxLayout(group)

        info = QLabel("Merge peaks within each time window into a single grouped peak.\nApplied during integration.")
        info.setStyleSheet("color: #666666; font-size: 10px;")
        layout.addWidget(info)

        self.peak_group_table = QTableWidget(0, 3)
        self.peak_group_table.setHorizontalHeaderLabels(["Start Time", "End Time", ""])
        self.peak_group_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.peak_group_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.peak_group_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.peak_group_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.peak_group_table.setMaximumHeight(150)
        layout.addWidget(self.peak_group_table)

        add_btn = QPushButton("+ Add Group")
        add_btn.clicked.connect(self._add_peak_group_row)
        layout.addWidget(add_btn)

        self.section_groups['peak_grouping'] = group
        self.params_layout.addWidget(group)

    def _add_peak_group_row(self):
        """Add a new row to the peak grouping table."""
        row = self.peak_group_table.rowCount()
        self.peak_group_table.insertRow(row)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(0, 9999)
        start_spin.valueChanged.connect(self._on_peak_groups_changed)
        self.peak_group_table.setCellWidget(row, 0, start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(0, 9999)
        end_spin.setValue(60.0)
        end_spin.valueChanged.connect(self._on_peak_groups_changed)
        self.peak_group_table.setCellWidget(row, 1, end_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_peak_group_row(r))
        self.peak_group_table.setCellWidget(row, 2, del_btn)

    def _remove_peak_group_row(self, row):
        """Remove a row from the peak grouping table."""
        self.peak_group_table.removeRow(row)
        for r in range(self.peak_group_table.rowCount()):
            del_btn = self.peak_group_table.cellWidget(r, 2)
            if del_btn:
                del_btn.clicked.disconnect()
                del_btn.clicked.connect(lambda checked, r=r: self._remove_peak_group_row(r))
        self._on_peak_groups_changed()

    def _on_peak_groups_changed(self, _=None):
        """Collect peak group values and emit parameters_changed."""
        groups = []
        for r in range(self.peak_group_table.rowCount()):
            start_w = self.peak_group_table.cellWidget(r, 0)
            end_w = self.peak_group_table.cellWidget(r, 1)
            if start_w and end_w:
                groups.append([start_w.value(), end_w.value()])
        self.current_params['integration']['peak_groups'] = groups
        self.parameters_changed.emit(self.current_params)

    def _add_break_point_row(self):
        """Add a new row to the break point table."""
        row = self.break_point_table.rowCount()
        self.break_point_table.insertRow(row)

        time_spin = QDoubleSpinBox()
        time_spin.setDecimals(3)
        time_spin.setRange(0, 9999)
        time_spin.valueChanged.connect(self._on_break_points_changed)
        self.break_point_table.setCellWidget(row, 0, time_spin)

        tol_spin = QDoubleSpinBox()
        tol_spin.setDecimals(3)
        tol_spin.setRange(0, 10)
        tol_spin.setValue(0.1)
        tol_spin.valueChanged.connect(self._on_break_points_changed)
        self.break_point_table.setCellWidget(row, 1, tol_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_break_point_row(r))
        self.break_point_table.setCellWidget(row, 2, del_btn)

    def _remove_break_point_row(self, row):
        """Remove a row from the break point table."""
        self.break_point_table.removeRow(row)
        for r in range(self.break_point_table.rowCount()):
            del_btn = self.break_point_table.cellWidget(r, 2)
            if del_btn:
                del_btn.clicked.disconnect()
                del_btn.clicked.connect(lambda checked, r=r: self._remove_break_point_row(r))
        self._on_break_points_changed()

    def _on_break_points_changed(self, _=None):
        """Collect break point values and emit parameters_changed."""
        bps = []
        for r in range(self.break_point_table.rowCount()):
            time_w = self.break_point_table.cellWidget(r, 0)
            tol_w = self.break_point_table.cellWidget(r, 1)
            if time_w and tol_w:
                bps.append({'time': time_w.value(), 'tolerance': tol_w.value()})
        self.current_params['baseline']['break_points'] = bps
        self.parameters_changed.emit(self.current_params)

    def update_lam_visibility(self):
        """Show/hide lambda and fastchrom controls based on the current algorithm"""
        method = self.baseline_algorithm.currentData()
        
        # Only show lambda controls for methods that use it
        uses_lambda = method in self.lam_methods
        self.lam_container.setVisible(uses_lambda)
        self.lam_label.setVisible(uses_lambda)
        
        # Only show fastchrom controls for fastchrom
        self.fastchrom_container.setVisible(method == "fastchrom")
    
    def _update_smoothing_method_visibility(self):
        """Show/hide controls based on selected smoothing method and median toggle"""
        is_whittaker = self.smooth_method_combo.currentData() == 'whittaker'
        median_on = self.median_enabled.isChecked()
        # Median kernel (only visible when median is checked)
        self.median_kernel_container.setVisible(median_on)
        self.median_kernel_row_label.setVisible(median_on)
        # Whittaker controls
        self.smooth_lam_container.setVisible(is_whittaker)
        self.smooth_lam_label.setVisible(is_whittaker)
        self.smooth_diff_order_combo.setVisible(is_whittaker)
        self.smooth_diff_order_row_label.setVisible(is_whittaker)
        # Savgol controls
        self.savgol_window_container.setVisible(not is_whittaker)
        self.savgol_window_row_label.setVisible(not is_whittaker)
        self.savgol_polyorder_container.setVisible(not is_whittaker)
        self.savgol_polyorder_row_label.setVisible(not is_whittaker)

    def _update_smoothing_controls_state(self):
        """Update enabled state of smoothing controls"""
        enabled = self.smoothing_enabled.isChecked()
        median_enabled = enabled and self.median_enabled.isChecked()
        self.smooth_method_combo.setEnabled(enabled)
        self.median_enabled.setEnabled(enabled)
        self.median_kernel_slider.setEnabled(median_enabled)
        self.median_kernel_spinbox.setEnabled(median_enabled)
        self.smooth_lam_slider.setEnabled(enabled)
        self.smooth_lam_spinbox.setEnabled(enabled)
        self.smooth_diff_order_combo.setEnabled(enabled)
        self.savgol_window_slider.setEnabled(enabled)
        self.savgol_window_spinbox.setEnabled(enabled)
        self.savgol_polyorder_slider.setEnabled(enabled)
        self.savgol_polyorder_spinbox.setEnabled(enabled)
    
    def _update_peaks_controls_state(self):
        """Update enabled state of peaks controls"""
        enabled = self.peaks_enabled.isChecked()
        is_peak_splitting = self.current_params['peaks']['mode'] == 'peak_splitting'
        self.prominence_entry.setEnabled(enabled)
        self.peak_mode_combo.setEnabled(enabled)
        self.min_width_entry.setEnabled(enabled)
        self.peak_splitting_controls_frame.setVisible(enabled and is_peak_splitting)

    def _update_shoulder_controls_state(self):
        """Update enabled state of shoulder controls"""
        shoulder_enabled = self.shoulders_enabled.isChecked()
        peak_enabled = self.peaks_enabled.isChecked()
        is_peak_splitting = self.current_params['peaks']['mode'] == 'peak_splitting'

        # Shoulder detection requires classical mode and peak detection enabled
        overall_enabled = shoulder_enabled and peak_enabled and not is_peak_splitting

        # Disable shoulder checkbox entirely in peak_splitting mode
        self.shoulders_enabled.setEnabled(peak_enabled and not is_peak_splitting)

        # Enable/disable all shoulder detection controls
        self.shoulder_sensitivity_slider.setEnabled(overall_enabled)
        self.shoulder_sensitivity_spinbox.setEnabled(overall_enabled)
        self.shoulder_distance_slider.setEnabled(overall_enabled)
        self.shoulder_distance_spinbox.setEnabled(overall_enabled)
        self.shoulder_advanced_toggle.setEnabled(overall_enabled)
        self.shoulder_window_slider.setEnabled(overall_enabled)
        self.shoulder_window_spinbox.setEnabled(overall_enabled)
        self.shoulder_polyorder_slider.setEnabled(overall_enabled)
        self.shoulder_polyorder_spinbox.setEnabled(overall_enabled)

    def _on_smoothing_toggled(self, state):
        """Handle smoothing enable/disable toggle"""
        enabled = bool(state)
        self.current_params['smoothing']['enabled'] = enabled
        self._update_smoothing_controls_state()
        self.parameters_changed.emit(self.current_params)
    
    def _on_smooth_lambda_changed(self, value):
        """Handle smoothing lambda change (log10 scale)"""
        # Update spinbox without triggering its signal
        if self.smooth_lam_spinbox.value() != value:
            self.smooth_lam_spinbox.blockSignals(True)
            self.smooth_lam_spinbox.setValue(value)
            self.smooth_lam_spinbox.blockSignals(False)

        self.smooth_lam_label.setText(f"\u03bb = 10^{value}")
        self.current_params['smoothing']['lambda'] = 10 ** value
        self.parameters_changed.emit(self.current_params)

    def _on_smooth_diff_order_changed(self, index):
        """Handle smoothing diff order change"""
        diff_order = self.smooth_diff_order_combo.currentData()
        self.current_params['smoothing']['diff_order'] = diff_order
        self.parameters_changed.emit(self.current_params)

    def _on_median_toggled(self, state):
        """Handle median pre-filter toggle"""
        enabled = bool(state)
        self.current_params['smoothing']['median_enabled'] = enabled
        self._update_smoothing_controls_state()
        self._update_smoothing_method_visibility()
        self.parameters_changed.emit(self.current_params)

    def _on_median_kernel_changed(self, value):
        """Handle median kernel size change"""
        if value % 2 == 0:
            value += 1
            self.median_kernel_slider.blockSignals(True)
            self.median_kernel_slider.setValue(value)
            self.median_kernel_slider.blockSignals(False)
        if self.median_kernel_spinbox.value() != value:
            self.median_kernel_spinbox.blockSignals(True)
            self.median_kernel_spinbox.setValue(value)
            self.median_kernel_spinbox.blockSignals(False)
        self.current_params['smoothing']['median_kernel'] = value
        self.parameters_changed.emit(self.current_params)

    def _on_smooth_method_changed(self, index):
        """Handle smoothing method change"""
        method = self.smooth_method_combo.currentData()
        self.current_params['smoothing']['method'] = method
        self._update_smoothing_method_visibility()
        self.parameters_changed.emit(self.current_params)

    def _on_savgol_window_changed(self, value):
        """Handle Savitzky-Golay window size change"""
        if value % 2 == 0:
            value += 1
            self.savgol_window_slider.blockSignals(True)
            self.savgol_window_slider.setValue(value)
            self.savgol_window_slider.blockSignals(False)
        if self.savgol_window_spinbox.value() != value:
            self.savgol_window_spinbox.blockSignals(True)
            self.savgol_window_spinbox.setValue(value)
            self.savgol_window_spinbox.blockSignals(False)
        # Clamp polyorder below window
        if self.current_params['smoothing']['savgol_polyorder'] >= value:
            new_order = value - 1
            self.savgol_polyorder_slider.setValue(new_order)
        self.current_params['smoothing']['savgol_window'] = value
        self.parameters_changed.emit(self.current_params)

    def _on_savgol_polyorder_changed(self, value):
        """Handle Savitzky-Golay polynomial order change"""
        window = self.current_params['smoothing']['savgol_window']
        if value >= window:
            value = window - 1
            self.savgol_polyorder_slider.blockSignals(True)
            self.savgol_polyorder_slider.setValue(value)
            self.savgol_polyorder_slider.blockSignals(False)
        if self.savgol_polyorder_spinbox.value() != value:
            self.savgol_polyorder_spinbox.blockSignals(True)
            self.savgol_polyorder_spinbox.setValue(value)
            self.savgol_polyorder_spinbox.blockSignals(False)
        self.current_params['smoothing']['savgol_polyorder'] = value
        self.parameters_changed.emit(self.current_params)

    def _on_baseline_display_toggled(self, state):
        """Handle baseline display mode toggle"""
        show_corrected = bool(state)
        self.current_params['baseline']['show_corrected'] = show_corrected
        
        # Provide visual feedback
        if show_corrected:
            feedback = "Showing corrected signal (baseline at zero)"
        else:
            feedback = "Showing raw signal with baseline"
        
        print(feedback)
        
        # No need to disable/enable controls - they're always active
        self.parameters_changed.emit(self.current_params)
    
    def _on_baseline_method_changed(self, index):
        """Handle baseline method change"""
        method = self.baseline_algorithm.currentData()
        self.current_params['baseline']['method'] = method
        self.update_lam_visibility()
        self.parameters_changed.emit(self.current_params)
    
    def _on_fastchrom_param_changed(self, value):
        """Handle fastchrom parameter change"""
        hw = self.fc_half_window.value()
        shw = self.fc_smooth_half_window.value()
        self.current_params['baseline']['fastchrom'] = {
            'half_window': hw if hw > 0 else None,
            'smooth_half_window': shw if shw > 0 else None,
        }
        self.parameters_changed.emit(self.current_params)
    
    def _on_lambda_changed(self, value):
        """Handle lambda slider change"""
        # Update spinbox without triggering its signal
        if self.lam_spinbox.value() != value:
            self.lam_spinbox.blockSignals(True)
            self.lam_spinbox.setValue(value)
            self.lam_spinbox.blockSignals(False)
        
        # Convert from exponent to actual value
        lambda_value = 10 ** value
        self.current_params['baseline']['lambda'] = lambda_value
        
        # Update label
        self.lam_label.setText(f"λ = 10^{value}")
        
        # More obvious visual feedback - style the whole container
        self.lam_container.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff;")
        QTimer.singleShot(800, lambda: self.lam_container.setStyleSheet(""))
        
        # Print debug info
        print(f"Lambda changed to 10^{value} = {lambda_value}")
        
        # Emit signal
        self.parameters_changed.emit(self.current_params)
    
    def _on_baseline_offset_changed(self):
        """Handle baseline offset input change."""
        text = self.baseline_offset_input.text().strip()
        try:
            offset = float(text)
        except ValueError:
            # Revert to current value on invalid input
            offset = self.current_params['baseline'].get('baseline_offset', 0.0)
            self.baseline_offset_input.setText(str(offset))
            return
        self.current_params['baseline']['baseline_offset'] = offset
        self.parameters_changed.emit(self.current_params)
    
    def _on_peaks_toggled(self, state):
        """Handle peaks enable/disable toggle"""
        enabled = bool(state)
        self.current_params['peaks']['enabled'] = enabled
        self._update_peaks_controls_state()
        self._update_shoulder_controls_state()  # Also update shoulder controls
        self.parameters_changed.emit(self.current_params)

    def _on_peak_mode_changed(self, index):
        """Handle peak detection mode change."""
        mode = self.peak_mode_combo.currentData()
        self.current_params['peaks']['mode'] = mode

        # Set sensible defaults for each mode.  The prominence box always
        # controls classical peak detection (scipy find_peaks).  In hybrid
        # peak_splitting mode (with windows), this determines which peaks are
        # kept outside the peak splitting windows.
        if mode == 'peak_splitting':
            # Keep prominence at its classical default — it now controls
            # classical peak detection in hybrid mode (outside peak splitting windows).
            # The peak splitting pipeline internally uses min_prominence=0.
            self.prominence_label.setText("Min Prominence:")
            self.prominence_entry.setToolTip(
                'Prominence threshold for classical peak detection.\n'
                'In hybrid mode (with peak splitting windows), this controls\n'
                'peak sensitivity outside the peak splitting windows.'
            )
            self.prominence_help.setText("Controls classical peaks (outside peak splitting windows)")
        else:
            self.current_params['peaks']['min_prominence'] = 1e5
            self.prominence_entry.setText('100000.0')
            self.prominence_label.setText("Min Prominence:")
            self.prominence_entry.setToolTip(
                'Absolute prominence threshold for scipy find_peaks.'
            )
            self.prominence_help.setText("Enter a number or scientific notation (e.g. 1e-3)")

        self._update_peaks_controls_state()
        self._update_shoulder_controls_state()
        self.parameters_changed.emit(self.current_params)

    def _on_splitting_method_changed(self, index):
        """Handle splitting method change — update defaults and visibility."""
        method = self.splitting_method_combo.currentData()
        self.current_params['peak_splitting']['splitting_method'] = method
        self._update_peak_splitting_method_visibility()
        self.parameters_changed.emit(self.current_params)

    def _on_peak_splitting_param_changed(self, key, value):
        """Handle any peak splitting advanced parameter change."""
        self.current_params['peak_splitting'][key] = value
        self.parameters_changed.emit(self.current_params)

    def _on_peak_splitting_pre_fit_signal_changed(self):
        """Handle pre-fit signal gate change."""
        text = self.pre_fit_signal_entry.text()
        try:
            value = float(text)
            self.current_params['peak_splitting']['pre_fit_signal_threshold'] = max(0.0, value)
            self.pre_fit_signal_entry.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff;")
            QTimer.singleShot(800, lambda: self.pre_fit_signal_entry.setStyleSheet(""))
            self.parameters_changed.emit(self.current_params)
        except ValueError:
            self.pre_fit_signal_entry.setStyleSheet("background-color: #ffcccc; border: 1px solid #ff9999;")
            QTimer.singleShot(800, lambda: self.pre_fit_signal_entry.setStyleSheet(""))

    def _add_peak_splitting_window_row(self):
        """Add a new row to the peak splitting windows table."""
        row = self.peak_splitting_windows_table.rowCount()
        self.peak_splitting_windows_table.insertRow(row)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(0, 9999)
        start_spin.valueChanged.connect(self._on_peak_splitting_windows_changed)
        self.peak_splitting_windows_table.setCellWidget(row, 0, start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(0, 9999)
        end_spin.setValue(60.0)
        end_spin.valueChanged.connect(self._on_peak_splitting_windows_changed)
        self.peak_splitting_windows_table.setCellWidget(row, 1, end_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_peak_splitting_window_row(r))
        self.peak_splitting_windows_table.setCellWidget(row, 2, del_btn)

    def _remove_peak_splitting_window_row(self, row):
        """Remove a row from the peak splitting windows table."""
        self.peak_splitting_windows_table.removeRow(row)
        for r in range(self.peak_splitting_windows_table.rowCount()):
            del_btn = self.peak_splitting_windows_table.cellWidget(r, 2)
            if del_btn:
                del_btn.clicked.disconnect()
                del_btn.clicked.connect(lambda checked, r=r: self._remove_peak_splitting_window_row(r))
        self._on_peak_splitting_windows_changed()

    def _on_peak_splitting_windows_changed(self, _=None):
        """Collect peak splitting window values and emit parameters_changed."""
        windows = []
        for r in range(self.peak_splitting_windows_table.rowCount()):
            start_w = self.peak_splitting_windows_table.cellWidget(r, 0)
            end_w = self.peak_splitting_windows_table.cellWidget(r, 1)
            if start_w and end_w:
                windows.append([start_w.value(), end_w.value()])
        self.current_params['peak_splitting']['windows'] = windows
        self.parameters_changed.emit(self.current_params)

    def _toggle_peak_splitting_advanced(self):
        """Toggle visibility of advanced peak splitting parameters."""
        visible = not self.peak_splitting_advanced_frame.isVisible()
        self.peak_splitting_advanced_frame.setVisible(visible)
        self.peak_splitting_advanced_toggle.setText("▼ Advanced" if visible else "▶ Advanced")

    def _update_peak_splitting_method_visibility(self):
        """Show/hide method-specific params based on splitting method."""
        is_emg = self.current_params['peak_splitting']['splitting_method'] == 'emg'

        # EMG-only controls
        for w in [self.emg_only_label, self.mu_bound_spin, self.fat_threshold_spin,
                  self.dedup_sigma_spin]:
            w.setVisible(is_emg)
        # Find and toggle the QLabel row labels for EMG spinboxes
        adv_layout = self.peak_splitting_advanced_frame.layout()
        for i in range(adv_layout.rowCount()):
            label_item = adv_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = adv_layout.itemAt(i, QFormLayout.FieldRole)
            if field_item and field_item.widget() in (self.mu_bound_spin,
                                                       self.fat_threshold_spin,
                                                       self.dedup_sigma_spin):
                if label_item and label_item.widget():
                    label_item.widget().setVisible(is_emg)

        # Geometric-only controls
        for w in [self.geo_only_label, self.dedup_rt_spin]:
            w.setVisible(not is_emg)
        for i in range(adv_layout.rowCount()):
            label_item = adv_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = adv_layout.itemAt(i, QFormLayout.FieldRole)
            if field_item and field_item.widget() is self.dedup_rt_spin:
                if label_item and label_item.widget():
                    label_item.widget().setVisible(not is_emg)

    # Defaults for each method: Optuna-optimized where applicable,
    # with min_prominence=0 and pre_fit_signal_threshold=0.001 based
    # on real-data testing (high dynamic range chromatograms).
    _PEAK_SPLITTING_DEFAULTS = {
        'geometric': {
            'heatmap_threshold': 0.36,
            'pre_fit_signal_threshold': 0.001,
            'min_area_frac': 0.15,
            'valley_threshold_frac': 0.48,
            'dedup_rt_tolerance': 0.005,
            # EMG params kept at their defaults for switching
            'mu_bound_factor': 0.68,
            'fat_threshold_frac': 0.44,
            'dedup_sigma_factor': 1.32,
        },
        'emg': {
            'heatmap_threshold': 0.47,
            'pre_fit_signal_threshold': 0.001,
            'min_area_frac': 0.11,
            'valley_threshold_frac': 0.48,
            'mu_bound_factor': 0.68,
            'fat_threshold_frac': 0.44,
            'dedup_sigma_factor': 1.32,
            # Geometric params kept at their defaults for switching
            'dedup_rt_tolerance': 0.005,
        },
    }

    def _reset_peak_splitting_defaults(self):
        """Reset peak splitting parameters to Optuna-optimized defaults.

        Does not touch Min Prominence — that controls classical peak detection
        and is independent of the peak splitting tuning parameters.
        """
        method = self.current_params['peak_splitting']['splitting_method']
        defaults = self._PEAK_SPLITTING_DEFAULTS[method]

        for key, val in defaults.items():
            self.current_params['peak_splitting'][key] = val

        # Update spinboxes without emitting per-change signals
        self._sync_peak_splitting_spinboxes()
        self.parameters_changed.emit(self.current_params)

    def _sync_peak_splitting_spinboxes(self):
        """Sync all peak splitting spinbox values from current_params."""
        d = self.current_params['peak_splitting']
        for spin, key in [
            (self.heatmap_threshold_spin, 'heatmap_threshold'),
            (self.min_area_frac_spin, 'min_area_frac'),
            (self.valley_threshold_spin, 'valley_threshold_frac'),
            (self.mu_bound_spin, 'mu_bound_factor'),
            (self.fat_threshold_spin, 'fat_threshold_frac'),
            (self.dedup_sigma_spin, 'dedup_sigma_factor'),
            (self.dedup_rt_spin, 'dedup_rt_tolerance'),
        ]:
            spin.blockSignals(True)
            spin.setValue(d[key])
            spin.blockSignals(False)
        self.pre_fit_signal_entry.blockSignals(True)
        self.pre_fit_signal_entry.setText(str(d['pre_fit_signal_threshold']))
        self.pre_fit_signal_entry.blockSignals(False)

    def _on_ms_baseline_clicked(self):
        """Signal that MS baseline correction button was clicked."""
        # Emit the signal
        self.ms_baseline_clicked.emit()
    
    def get_parameters(self):
        """Get the current processing parameters"""
        return self.current_params

    def set_section_visibility(self, visibility: dict):
        """Show/hide parameter sections based on visibility dict.
        
        Args:
            visibility: Dict of {section_key: bool}
        """
        for key, visible in visibility.items():
            if key in self.section_groups:
                self.section_groups[key].setVisible(visible)

    def _on_align_tic_toggled(self, state):
        """Handle TIC alignment toggle."""
        align_tic = bool(state)
        
        # Ensure the baseline dictionary has the align_tic key
        if 'align_tic' not in self.current_params['baseline']:
            self.current_params['baseline']['align_tic'] = align_tic
        else:
            self.current_params['baseline']['align_tic'] = align_tic
        
        # Provide visual feedback
        if align_tic:
            print("TIC alignment enabled - will align MS data to FID")
        else:
            print("TIC alignment disabled")
        
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulders_toggled(self, state):
        """Handle shoulder detection enable/disable toggle"""
        enabled = bool(state)
        self.current_params['shoulders']['enabled'] = enabled
        self._update_shoulder_controls_state()
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulder_window_changed(self, value):
        """Handle shoulder detection window length change"""
        # Ensure value is odd
        if value % 2 == 0:
            value += 1
        self.shoulder_window_slider.setValue(value)
        self.shoulder_window_spinbox.setValue(value)
        self.current_params['shoulders']['window_length'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulder_polyorder_changed(self, value):
        """Handle shoulder detection polynomial order change"""
        self.shoulder_polyorder_slider.setValue(value)
        self.shoulder_polyorder_spinbox.setValue(value)
        self.current_params['shoulders']['polyorder'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulder_sensitivity_changed(self, value):
        """Handle shoulder sensitivity slider change"""
        self.shoulder_sensitivity_slider.setValue(value)
        self.shoulder_sensitivity_spinbox.setValue(value)
        self.current_params['shoulders']['sensitivity'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _toggle_shoulder_advanced(self):
        """Toggle visibility of advanced shoulder detection parameters"""
        visible = not self.shoulder_advanced_frame.isVisible()
        self.shoulder_advanced_frame.setVisible(visible)
        self.shoulder_advanced_toggle.setText("▼ Advanced" if visible else "▶ Advanced")
    
    def _on_shoulder_distance_changed(self, value):
        """Handle shoulder apex-distance change"""
        self.shoulder_distance_slider.setValue(value)
        self.shoulder_distance_spinbox.setValue(value)
        self.current_params['shoulders']['apex_distance'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _on_min_width_changed(self):
        """Handle min width entry change with validation."""
        text = self.min_width_entry.text()
        try:
            value = int(float(text))
            self.current_params['peaks']['min_width'] = max(0, value)
            self.min_width_entry.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff;")
            QTimer.singleShot(800, lambda: self.min_width_entry.setStyleSheet(""))
            self.parameters_changed.emit(self.current_params)
        except ValueError:
            self.min_width_entry.setStyleSheet("background-color: #ffcccc; border: 1px solid #ff9999;")
            QTimer.singleShot(800, lambda: self.min_width_entry.setStyleSheet(""))

    def _on_prominence_entry_changed(self):
        """Handle prominence entry change with validation"""
        text = self.prominence_entry.text()
        try:
            value = float(text)
            self.current_params['peaks']['min_prominence'] = value

            # Provide visual feedback
            self.prominence_entry.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff;")
            QTimer.singleShot(800, lambda: self.prominence_entry.setStyleSheet(""))

            self.parameters_changed.emit(self.current_params)

        except ValueError:
            self.prominence_entry.setStyleSheet("background-color: #ffcccc; border: 1px solid #ff9999;")
            QTimer.singleShot(800, lambda: self.prominence_entry.setStyleSheet(""))