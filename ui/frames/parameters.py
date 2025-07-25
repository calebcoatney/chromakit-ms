from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QFormLayout,
    QCheckBox, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QHBoxLayout,
    QLineEdit, QFrame, QPushButton
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
        self.lam_methods = {"asls", "airpls", "arpls"}
        
        # Current parameters dictionary - update the baseline section
        self.current_params = {
            'smoothing': {
                'enabled': False,
                'median_filter': {
                    'kernel_size': 9
                },
                'savgol_filter': {
                    'window_length': 15,
                    'polyorder': 2
                }
            },
            'baseline': {
                'show_corrected': False,
                'method': 'arpls',  # Changed from 'asls' to 'arpls'
                'lambda': 1e4,      # Changed from 1e6 to 1e4
                'asymmetry': 0.01,
                'align_tic': False  # Add alignment option
            },
            'peaks': {
                'enabled': False,
                'min_prominence': 1e5,  # Changed from 0.5 to 1e5
                'min_height': 0.0,
                'min_width': 0.0
            },
            'shoulders': {
                'enabled': False,
                'window_length': 41,
                'polyorder': 3,
                'height_factor': 0.02,
                'apex_distance': 10
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
        
        # Add smoothing group
        self._init_smoothing_controls()
        
        # Add baseline group
        self._init_baseline_controls()
        
        # Add peaks group
        self._init_peaks_controls()
        
        # Add shoulder detection group
        self._init_shoulder_controls()
        
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
        
        # Median filter kernel size
        median_container = QWidget()
        median_layout = QHBoxLayout(median_container)
        median_layout.setContentsMargins(0, 0, 0, 0)
        
        self.median_slider = QSlider(Qt.Horizontal)
        self.median_slider.setMinimum(3)
        self.median_slider.setMaximum(31)  # Changed from 101 to 31 as requested
        self.median_slider.setValue(self.current_params['smoothing']['median_filter']['kernel_size'])
        self.median_slider.setSingleStep(2)  # Only odd values
        self.median_slider.setPageStep(2)
        self.median_slider.setTickInterval(2)
        self.median_slider.setTickPosition(QSlider.TicksBelow)
        self.median_slider.valueChanged.connect(self._on_median_kernel_changed)
        
        self.median_spinbox = QSpinBox()
        self.median_spinbox.setMinimum(3)
        self.median_spinbox.setMaximum(31)  # Changed from 101 to 31 as requested
        self.median_spinbox.setValue(self.current_params['smoothing']['median_filter']['kernel_size'])
        self.median_spinbox.setSingleStep(2)
        self.median_spinbox.valueChanged.connect(self.median_slider.setValue)
        
        median_layout.addWidget(self.median_slider, 3)
        median_layout.addWidget(self.median_spinbox, 1)
        
        form_layout.addRow("Median Filter Size:", median_container)
        
        # Savitzky-Golay window size
        savgol_container = QWidget()
        savgol_layout = QHBoxLayout(savgol_container)
        savgol_layout.setContentsMargins(0, 0, 0, 0)
        
        self.savgol_slider = QSlider(Qt.Horizontal)
        self.savgol_slider.setMinimum(5)
        self.savgol_slider.setMaximum(51)  # Changed from 201 to 51 as requested
        self.savgol_slider.setValue(self.current_params['smoothing']['savgol_filter']['window_length'])
        self.savgol_slider.setSingleStep(2)
        self.savgol_slider.setPageStep(2)
        self.savgol_slider.setTickInterval(2)
        self.savgol_slider.setTickPosition(QSlider.TicksBelow)
        self.savgol_slider.valueChanged.connect(self._on_savgol_window_changed)
        
        self.savgol_spinbox = QSpinBox()
        self.savgol_spinbox.setMinimum(5)
        self.savgol_spinbox.setMaximum(51)  # Changed from 201 to 51 as requested
        self.savgol_spinbox.setValue(self.current_params['smoothing']['savgol_filter']['window_length'])
        self.savgol_spinbox.setSingleStep(2)
        self.savgol_spinbox.valueChanged.connect(self.savgol_slider.setValue)
        
        savgol_layout.addWidget(self.savgol_slider, 3)
        savgol_layout.addWidget(self.savgol_spinbox, 1)
        
        form_layout.addRow("Savitzky-Golay Window:", savgol_container)
        
        # Enable/disable controls based on initial state
        self._update_smoothing_controls_state()
        
        # Add to parameters layout
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
        
        # Update lambda control visibility based on method
        self.update_lam_visibility()
        
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
        
        form_layout.addRow(ms_baseline_frame)
        
        # Add TIC alignment checkbox
        self.align_tic = QCheckBox("Align TIC with FID")
        self.align_tic.setChecked(self.current_params['baseline'].get('align_tic', False))
        self.align_tic.stateChanged.connect(self._on_align_tic_toggled)
        form_layout.addRow(self.align_tic)
        
        # Add explanation label
        align_info = QLabel("Corrects for small time delays between FID and MS detectors")
        align_info.setStyleSheet("color: #666666; font-size: 10px;")
        form_layout.addRow("", align_info)
        
        # Add to parameters layout
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
        
        # Direct entry box for prominence with validation
        self.prominence_entry = QLineEdit()
        self.prominence_entry.setText(str(self.current_params['peaks']['min_prominence']))
        
        # Set up validator for decimal and scientific notation
        validator = QRegularExpressionValidator(QRegularExpression(r'^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'))
        self.prominence_entry.setValidator(validator)
        
        self.prominence_entry.editingFinished.connect(self._on_prominence_entry_changed)
        form_layout.addRow("Min Prominence:", self.prominence_entry)
        
        # Add help label
        form_layout.addRow("", QLabel("Enter a number or scientific notation (e.g. 1e-3)"))
        
        # Enable/disable controls based on initial state
        self._update_peaks_controls_state()
        
        # Add to parameters layout
        self.params_layout.addWidget(peaks_group)
    
    def _init_shoulder_controls(self):
        """Initialize shoulder detection controls"""
        shoulder_group = QGroupBox("Shoulder Detection")
        form_layout = QFormLayout(shoulder_group)
        
        # Enable/disable checkbox
        self.shoulders_enabled = QCheckBox("Enable Shoulder Detection")
        self.shoulders_enabled.setChecked(self.current_params['shoulders']['enabled'])
        self.shoulders_enabled.stateChanged.connect(self._on_shoulders_toggled)
        form_layout.addRow(self.shoulders_enabled)
        
        # Add explanation
        shoulder_info = QLabel("Uses derivative analysis to detect shoulder peaks.\nRequires Peak Detection to be enabled.")
        shoulder_info.setStyleSheet("color: #666666; font-size: 10px;")
        form_layout.addRow("", shoulder_info)
        
        # Smoothing parameters for shoulder detection
        smoothing_frame = QFrame()
        smoothing_frame.setFrameStyle(QFrame.Box)
        smoothing_layout = QFormLayout(smoothing_frame)
        
        # Window length for shoulder detection smoothing
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
        
        smoothing_layout.addRow("Smoothing Window:", window_container)
        
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
        
        smoothing_layout.addRow("Polynomial Order:", polyorder_container)
        
        form_layout.addRow("Smoothing for Detection:", smoothing_frame)
        
        # Shoulder detection parameters
        detection_frame = QFrame()
        detection_frame.setFrameStyle(QFrame.Box)
        detection_layout = QFormLayout(detection_frame)
        
        # Shoulder height factor
        height_container = QWidget()
        height_layout = QHBoxLayout(height_container)
        height_layout.setContentsMargins(0, 0, 0, 0)
        
        self.shoulder_height_slider = QSlider(Qt.Horizontal)
        self.shoulder_height_slider.setMinimum(1)
        self.shoulder_height_slider.setMaximum(10)
        height_value = int(self.current_params['shoulders']['height_factor'] * 100)
        self.shoulder_height_slider.setValue(height_value)
        self.shoulder_height_slider.setTickPosition(QSlider.TicksBelow)
        self.shoulder_height_slider.valueChanged.connect(self._on_shoulder_height_changed)
        
        self.shoulder_height_spinbox = QDoubleSpinBox()
        self.shoulder_height_spinbox.setMinimum(0.01)
        self.shoulder_height_spinbox.setMaximum(0.10)
        self.shoulder_height_spinbox.setSingleStep(0.01)
        self.shoulder_height_spinbox.setDecimals(2)
        self.shoulder_height_spinbox.setValue(self.current_params['shoulders']['height_factor'])
        self.shoulder_height_spinbox.valueChanged.connect(self._on_shoulder_height_spinbox_changed)
        
        height_container.setLayout(height_layout)
        height_layout.addWidget(self.shoulder_height_slider, 3)
        height_layout.addWidget(self.shoulder_height_spinbox, 1)
        
        detection_layout.addRow("Shoulder Sensitivity:", height_container)
        
        # Apex-shoulder distance
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
        
        distance_container.setLayout(distance_layout)
        distance_layout.addWidget(self.shoulder_distance_slider, 3)
        distance_layout.addWidget(self.shoulder_distance_spinbox, 1)
        
        detection_layout.addRow("Min Apex-Shoulder Distance:", distance_container)
        
        form_layout.addRow("Detection Parameters:", detection_frame)
        
        # Enable/disable controls based on initial state
        self._update_shoulder_controls_state()
        
        # Add to parameters layout
        self.params_layout.addWidget(shoulder_group)

    def update_lam_visibility(self):
        """Show/hide lambda controls based on the current algorithm"""
        method = self.baseline_algorithm.currentData()
        
        # Only show lambda controls for methods that use it
        uses_lambda = method in self.lam_methods
        self.lam_container.setVisible(uses_lambda)
        self.lam_label.setVisible(uses_lambda)
    
    def _update_smoothing_controls_state(self):
        """Update enabled state of smoothing controls"""
        enabled = self.smoothing_enabled.isChecked()
        self.median_slider.setEnabled(enabled)
        self.median_spinbox.setEnabled(enabled)
        self.savgol_slider.setEnabled(enabled)
        self.savgol_spinbox.setEnabled(enabled)
    
    def _update_peaks_controls_state(self):
        """Update enabled state of peaks controls"""
        enabled = self.peaks_enabled.isChecked()
        self.prominence_entry.setEnabled(enabled)
    
    def _update_shoulder_controls_state(self):
        """Update enabled state of shoulder controls"""
        shoulder_enabled = self.shoulders_enabled.isChecked()
        peak_enabled = self.peaks_enabled.isChecked()
        
        # Shoulder detection requires peak detection to be enabled
        overall_enabled = shoulder_enabled and peak_enabled
        
        # Enable shoulder checkbox only if peaks are enabled
        self.shoulders_enabled.setEnabled(peak_enabled)
        
        # Enable/disable all shoulder detection controls
        self.shoulder_window_slider.setEnabled(overall_enabled)
        self.shoulder_window_spinbox.setEnabled(overall_enabled)
        self.shoulder_polyorder_slider.setEnabled(overall_enabled)
        self.shoulder_polyorder_spinbox.setEnabled(overall_enabled)
        self.shoulder_height_slider.setEnabled(overall_enabled)
        self.shoulder_height_spinbox.setEnabled(overall_enabled)
        self.shoulder_distance_slider.setEnabled(overall_enabled)
        self.shoulder_distance_spinbox.setEnabled(overall_enabled)

    def _on_smoothing_toggled(self, state):
        """Handle smoothing enable/disable toggle"""
        enabled = bool(state)
        self.current_params['smoothing']['enabled'] = enabled
        self._update_smoothing_controls_state()
        self.parameters_changed.emit(self.current_params)
    
    def _on_median_kernel_changed(self, value):
        """Handle median kernel size change"""
        # Ensure value is odd
        if value % 2 == 0:
            value += 1
            self.median_slider.blockSignals(True)
            self.median_slider.setValue(value)
            self.median_slider.blockSignals(False)
        
        # Update spinbox without triggering its signal
        if self.median_spinbox.value() != value:
            self.median_spinbox.blockSignals(True)
            self.median_spinbox.setValue(value)
            self.median_spinbox.blockSignals(False)
        
        # Update parameters
        self.current_params['smoothing']['median_filter']['kernel_size'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _on_savgol_window_changed(self, value):
        """Handle Savitzky-Golay window size change"""
        # Ensure value is odd
        if value % 2 == 0:
            value += 1
            self.savgol_slider.blockSignals(True)
            self.savgol_slider.setValue(value)
            self.savgol_slider.blockSignals(False)
        
        # Update spinbox without triggering its signal
        if self.savgol_spinbox.value() != value:
            self.savgol_spinbox.blockSignals(True)
            self.savgol_spinbox.setValue(value)
            self.savgol_spinbox.blockSignals(False)
        
        # Update parameters
        self.current_params['smoothing']['savgol_filter']['window_length'] = value
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
    
    def _on_peaks_toggled(self, state):
        """Handle peaks enable/disable toggle"""
        enabled = bool(state)
        self.current_params['peaks']['enabled'] = enabled
        self._update_peaks_controls_state()
        self._update_shoulder_controls_state()  # Also update shoulder controls
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulders_toggled(self, state):
        """Handle shoulder detection enable/disable toggle"""
        enabled = bool(state)
        self.current_params['shoulders']['enabled'] = enabled
        self._update_shoulder_controls_state()
        self.parameters_changed.emit(self.current_params)
    
    def _on_ms_baseline_clicked(self):
        """Signal that MS baseline correction button was clicked."""
        # Emit the signal
        self.ms_baseline_clicked.emit()
    
    def get_parameters(self):
        """Get the current processing parameters"""
        return self.current_params

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
    
    def _on_shoulder_height_changed(self, value):
        """Handle shoulder height factor slider change"""
        height_factor = value / 100.0  # Convert to decimal
        self.shoulder_height_spinbox.setValue(height_factor)
        self.current_params['shoulders']['height_factor'] = height_factor
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulder_height_spinbox_changed(self, value):
        """Handle shoulder height factor spinbox change"""
        slider_value = int(value * 100)
        self.shoulder_height_slider.setValue(slider_value)
        self.current_params['shoulders']['height_factor'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulder_distance_changed(self, value):
        """Handle shoulder apex-distance change"""
        self.shoulder_distance_slider.setValue(value)
        self.shoulder_distance_spinbox.setValue(value)
        self.current_params['shoulders']['apex_distance'] = value
        self.parameters_changed.emit(self.current_params)
    
    def _on_prominence_entry_changed(self):
        """Handle prominence entry change with validation"""
        text = self.prominence_entry.text()
        try:
            # Try to convert the text to a float
            value = float(text)
            
            # Update parameters
            self.current_params['peaks']['min_prominence'] = value
            
            # Provide visual feedback
            self.prominence_entry.setStyleSheet("background-color: #e6f2ff; border: 1px solid #99ccff;")
            QTimer.singleShot(800, lambda: self.prominence_entry.setStyleSheet(""))
            
            # Emit signal for parameter change
            self.parameters_changed.emit(self.current_params)
            
        except ValueError:
            # Invalid input - show red feedback
            self.prominence_entry.setStyleSheet("background-color: #ffcccc; border: 1px solid #ff9999;")
            QTimer.singleShot(800, lambda: self.prominence_entry.setStyleSheet(""))