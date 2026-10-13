from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QFormLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QRadioButton, QButtonGroup, QGroupBox, QSlider, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSettings

from logic.ms_weighting import (
    DEFAULT_WEIGHTING, read_weighting_scheme, weighting_label,
    weighting_scheme_names, write_weighting_scheme,
)
from ui.dialogs.spectral_deconv_fields import (
    INVALID_MZ_TITLE, build_setting_boxes, invalid_mz_message,
)

class MSOptionsDialog(QDialog):
    """Dialog for configuring MS search options."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MS Search Options")
        self.setMinimumWidth(450)
        
        # Initialize settings
        self.settings = QSettings("CalebCoatney", "ChromaKit")
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_general_tab()
        self._create_extraction_tab()
        self._create_subtraction_tab()
        self._create_spectral_deconv_tab()
        self._create_algorithm_tab()
        self._create_quality_checks_tab()  # Add this line
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Create restore defaults button. Not auto-default: as the first push button it
        # would otherwise become the default, and Return in any field would restore defaults.
        self.restore_button = QPushButton("Restore Defaults")
        self.restore_button.setAutoDefault(False)
        self.restore_button.clicked.connect(self._restore_defaults)
        button_layout.addWidget(self.restore_button)
        
        # Add spacer
        button_layout.addStretch()
        
        # Create OK and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.setDefault(True)   # Return = OK (accept() still validates)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        # Add button layout
        self.layout.addLayout(button_layout)
        
        # Load saved settings
        self._load_settings()

        # Open wide enough for every tab name; the scroll-arrow tab bar caps the
        # size hint at 200 px, which hid the later tabs. Users may still shrink to 450.
        margins = self.layout.contentsMargins()
        tabs_width = (self.tab_widget.tabBar().sizeHint().width()
                      + margins.left() + margins.right() + 8)   # 8: slack for the style's tab frame
        self.resize(max(self.sizeHint().width(), tabs_width), self.sizeHint().height())
    
    def _create_general_tab(self):
        """Create the general options tab."""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Search method
        self.search_method_combo = QComboBox()
        self.search_method_combo.addItems(["Vector Search", "Word2Vec Search", "Hybrid Search"])
        self.search_method_combo.currentTextChanged.connect(self._on_search_method_changed)
        layout.addRow("Search Method:", self.search_method_combo)
        
        # Hybrid search sub-options (initially hidden)
        self.hybrid_method_combo = QComboBox()
        self.hybrid_method_combo.addItems(["Auto", "Fast", "Ensemble"])
        self.hybrid_method_combo.setToolTip("Auto: Automatic method selection based on spectrum complexity\n"
                                           "Fast: Rule-based quick selection\n"
                                           "Ensemble: Combine both vector and Word2Vec results")
        
        # Create label explicitly so we can control its visibility
        self.hybrid_method_label = QLabel("Hybrid Method:")
        layout.addRow(self.hybrid_method_label, self.hybrid_method_combo)
        
        # Initially hide hybrid options
        self.hybrid_method_combo.hide()
        self.hybrid_method_label.hide()
        
        # Add full MS baseline correction option
        self.full_ms_baseline_check = QCheckBox("Enable Full MS Baseline Correction")
        self.full_ms_baseline_check.setChecked(False)  # Disabled by default
        self.full_ms_baseline_check.setToolTip("Apply baseline correction to all m/z traces individually")
        layout.addWidget(self.full_ms_baseline_check)
        
        # Add to tab widget
        self.tab_widget.addTab(tab, "General")
    
    def _create_extraction_tab(self):
        """Create the spectrum extraction options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create extraction method group
        extraction_group = QGroupBox("Extraction Method")
        extraction_layout = QVBoxLayout(extraction_group)
        
        # Create radio buttons for extraction methods
        self.apex_radio = QRadioButton("Peak Apex (single point)")
        self.average_radio = QRadioButton("Peak Average (full bounds)")
        self.range_radio = QRadioButton("Range Around Apex")
        self.midpoint_radio = QRadioButton("Midpoint Window")  # Add new option
        
        extraction_layout.addWidget(self.apex_radio)
        extraction_layout.addWidget(self.average_radio)
        extraction_layout.addWidget(self.range_radio)
        extraction_layout.addWidget(self.midpoint_radio)  # Add to layout
        
        # Group the radio buttons
        self.extraction_group = QButtonGroup()
        self.extraction_group.addButton(self.apex_radio, 0)
        self.extraction_group.addButton(self.average_radio, 1)
        self.extraction_group.addButton(self.range_radio, 2)
        self.extraction_group.addButton(self.midpoint_radio, 3)  # Add to button group
        
        layout.addWidget(extraction_group)
        
        # Range options in a separate group that's only enabled when range is selected
        self.range_options_group = QGroupBox("Range Options")
        range_options_layout = QFormLayout(self.range_options_group)
        
        self.range_points_spin = QSpinBox()
        self.range_points_spin.setRange(1, 50)
        self.range_points_spin.setValue(5)
        range_options_layout.addRow("Points on each side:", self.range_points_spin)
        
        self.range_options_group.setLayout(range_options_layout)
        layout.addWidget(self.range_options_group)
        
        # Midpoint options group (NEW)
        self.midpoint_options_group = QGroupBox("Midpoint Window Options")
        midpoint_options_layout = QFormLayout(self.midpoint_options_group)
        
        self.midpoint_width_spin = QSpinBox()
        self.midpoint_width_spin.setRange(1, 100)
        self.midpoint_width_spin.setValue(20)
        self.midpoint_width_spin.setSuffix("%")
        self.midpoint_width_spin.setToolTip("Window width as percentage of peak width")
        midpoint_options_layout.addRow("Window width:", self.midpoint_width_spin)
        
        self.midpoint_options_group.setLayout(midpoint_options_layout)
        layout.addWidget(self.midpoint_options_group)
        
        # Connect radio buttons to enable/disable appropriate option groups
        self.range_radio.toggled.connect(self.range_options_group.setEnabled)
        self.midpoint_radio.toggled.connect(self.midpoint_options_group.setEnabled)
        
        # TIC weighting
        self.tic_weight_check = QCheckBox("Weight spectra by TIC intensity")
        self.tic_weight_check.setChecked(True)
        layout.addWidget(self.tic_weight_check)
        
        # Add to tab widget
        self.tab_widget.addTab(tab, "Spectrum Extraction (Fallback)")
        
        # Initially disable option groups if not selected
        self.range_options_group.setEnabled(self.range_radio.isChecked())
        self.midpoint_options_group.setEnabled(self.midpoint_radio.isChecked())
    
    def _create_subtraction_tab(self):
        """Create the background subtraction options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Enable background subtraction
        self.subtract_check = QCheckBox("Enable Background Subtraction")
        self.subtract_check.setChecked(True)
        layout.addWidget(self.subtract_check)
        
        # Subtraction method group
        subtraction_group = QGroupBox("Subtraction Method")
        subtraction_layout = QVBoxLayout(subtraction_group)
        
        # Create radio buttons for subtraction methods
        self.left_bound_radio = QRadioButton("Left Bound")
        self.right_bound_radio = QRadioButton("Right Bound")
        self.min_tic_radio = QRadioButton("Min TIC Point")
        self.average_bounds_radio = QRadioButton("Average Bounds")
        
        subtraction_layout.addWidget(self.left_bound_radio)
        subtraction_layout.addWidget(self.right_bound_radio)
        subtraction_layout.addWidget(self.min_tic_radio)
        subtraction_layout.addWidget(self.average_bounds_radio)
        
        # Group the radio buttons
        self.subtraction_group = QButtonGroup()
        self.subtraction_group.addButton(self.left_bound_radio, 0)
        self.subtraction_group.addButton(self.right_bound_radio, 1)
        self.subtraction_group.addButton(self.min_tic_radio, 2)
        self.subtraction_group.addButton(self.average_bounds_radio, 3)
        
        layout.addWidget(subtraction_group)
        
        # Subtraction parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        
        # Subtraction weight
        self.subtract_weight_spin = QDoubleSpinBox()
        self.subtract_weight_spin.setRange(0.01, 1.0)
        self.subtract_weight_spin.setSingleStep(0.01)
        self.subtract_weight_spin.setValue(0.1)
        params_layout.addRow("Subtraction Weight:", self.subtract_weight_spin)
        
        # Add intensity threshold control
        self.intensity_threshold_spin = QDoubleSpinBox()
        self.intensity_threshold_spin.setRange(0.001, 0.2)  # 0.1% to 20%
        self.intensity_threshold_spin.setSingleStep(0.001)
        self.intensity_threshold_spin.setDecimals(3)
        self.intensity_threshold_spin.setValue(0.01)  # Default 1%
        self.intensity_threshold_spin.setToolTip("Filter out peaks with intensity less than this percentage of the maximum")
        params_layout.addRow("Intensity Threshold (%):", self.intensity_threshold_spin)
        
        layout.addWidget(params_group)
        
        # Add to tab widget
        self.tab_widget.addTab(tab, "Background Subtraction (Fallback)")
    
    def _create_spectral_deconv_tab(self):
        """Create the Spectral Deconvolution (correlation grouping, after MZmine) options tab."""
        tab = QWidget()
        self._spectral_deconv_tab = tab
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Spectral deconvolution gives each integrated FID peak a cleaner spectrum for "
            "library search, keeping only the ions that elute together with the same peak "
            "shape. It runs when you click Deconvolve MS, and in batch runs before MS search. "
            "FID peaks without a matching spectrum use the fallback spectrum (Spectrum "
            "Extraction tab)."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        boxes, w = build_setting_boxes()
        for box in boxes:
            layout.addWidget(box)
        self.gap_tolerance_spin = w["gap_tolerance"]
        self.padding_fraction_spin = w["padding_fraction"]
        self.rt_match_tolerance_spin = w["rt_match_tolerance"]
        self.max_window_peaks_spin = w["max_window_peaks"]
        self.noise_percentile_spin = w["noise_percentile"]
        self.search_range_spin = w["search_range"]
        self.min_peak_height_spin = w["min_peak_height"]
        self.min_top_edge_ratio_spin = w["min_top_edge_ratio"]
        self.min_points_spin = w["min_points"]
        self.rt_tolerance_spin = w["rt_tolerance"]
        self.min_r_spin = w["min_r"]
        self.min_ions_spin = w["min_ions"]
        self.excluded_mz_edit = w["excluded_mz"]
        layout.addStretch()

        self.tab_widget.addTab(tab, "Spectral Deconvolution")

    def _create_algorithm_tab(self):
        """Create the algorithm options tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Vector search options
        vector_group = QGroupBox("Vector Search Options")
        vector_layout = QFormLayout(vector_group)
        
        # Similarity measure
        self.similarity_combo = QComboBox()
        self.similarity_combo.addItems(["Cosine Similarity", "Composite Similarity"])
        vector_layout.addRow("Similarity Measure:", self.similarity_combo)
        
        # Weighting scheme
        self.weighting_combo = QComboBox()
        for scheme in weighting_scheme_names():
            self.weighting_combo.addItem(weighting_label(scheme), scheme)
        self.weighting_combo.setToolTip(
            "How m/z and intensity are weighted before a spectrum is compared with the library.\n"
            "NIST11 is the default. In a library-against-library replicate test on NIST14, "
            "searching all clusters, it put the right compound first 81.7 % of the time, "
            "against 75.3 % for NIST_GC. It also scored better on real fast-pyrolysis oil "
            "samples.\n"
            "Raising Clusters to Consider to 50 can add a little accuracy, but the search "
            "takes at least 4 times as long."
        )
        vector_layout.addRow("Weighting Scheme:", self.weighting_combo)
        
        # Unmatched method
        self.unmatched_combo = QComboBox()
        self.unmatched_combo.addItems(["Keep All", "Remove All", "Keep Library", "Keep Experimental"])
        vector_layout.addRow("Unmatched Peaks:", self.unmatched_combo)
        
        # NEW: Preselector type
        self.preselector_combo = QComboBox()
        self.preselector_combo.addItems(["K-means", "GMM (Gaussian Mixture Model)"])
        self.preselector_combo.setToolTip("Algorithm used for preselecting library spectra before detailed comparison")
        vector_layout.addRow("Preselector Type:", self.preselector_combo)
        
        # NEW: Number of clusters/components to consider
        self.top_k_clusters_spin = QSpinBox()
        self.top_k_clusters_spin.setRange(1, 100)
        self.top_k_clusters_spin.setValue(10)
        self.top_k_clusters_spin.setToolTip(
            "Number of preselector clusters/components to consider during "
            "search (higher = more candidates, slightly slower but more "
            "robust against noisy or air-contaminated spectra; 1 = strictest, "
            "100 = effectively no preselector filtering)"
        )
        vector_layout.addRow("Clusters to Consider:", self.top_k_clusters_spin)
        
        layout.addWidget(vector_group)
        
        # Word2Vec options
        w2v_group = QGroupBox("Word2Vec Options")
        w2v_layout = QFormLayout(w2v_group)
        
        # Intensity power
        self.intensity_power_spin = QDoubleSpinBox()
        self.intensity_power_spin.setRange(0.1, 1.0)
        self.intensity_power_spin.setSingleStep(0.1)
        self.intensity_power_spin.setValue(0.6)
        w2v_layout.addRow("Intensity Power:", self.intensity_power_spin)
        
        layout.addWidget(w2v_group)
        
        # Top results
        results_layout = QFormLayout()
        
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 50)
        self.top_n_spin.setValue(5)
        results_layout.addRow("Number of Results:", self.top_n_spin)
        
        layout.addLayout(results_layout)
        
        # Add to tab widget
        self.tab_widget.addTab(tab, "Search Algorithm")
    
    def _create_quality_checks_tab(self):
        """Create the peak quality checks configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Enable quality checks
        self.quality_checks_enabled = QCheckBox("Enable Peak Quality Checks")
        self.quality_checks_enabled.setChecked(False)  # Disabled by default
        layout.addWidget(self.quality_checks_enabled)
        
        # Quality checks group
        checks_group = QGroupBox("Quality Checks")
        checks_layout = QVBoxLayout(checks_group)
        
        # Individual checks
        self.skew_check = QCheckBox("Check Peak Asymmetry (Skewness)")
        self.skew_check.setChecked(False)  # Disabled by default
        self.skew_check.setToolTip("Flag peaks with high asymmetry (|skew| > threshold)")
        checks_layout.addWidget(self.skew_check)
        
        self.coherence_check = QCheckBox("Check Spectral Coherence")
        self.coherence_check.setChecked(False)  # Disabled by default
        self.coherence_check.setToolTip("Flag peaks with low ion coherence (correlation < threshold)")
        checks_layout.addWidget(self.coherence_check)
        
        # Connect main checkbox to enable/disable individual checks
        self.quality_checks_enabled.toggled.connect(checks_group.setEnabled)
        
        # Initially disable the group if main check is unchecked
        checks_group.setEnabled(self.quality_checks_enabled.isChecked())
        
        layout.addWidget(checks_group)
        
        # Thresholds group
        threshold_group = QGroupBox("Thresholds")
        threshold_layout = QFormLayout(threshold_group)
        
        # Skewness threshold 
        self.skew_threshold_spin = QDoubleSpinBox()
        self.skew_threshold_spin.setRange(0.1, 2.0)
        self.skew_threshold_spin.setSingleStep(0.1)
        self.skew_threshold_spin.setValue(0.5)
        self.skew_threshold_spin.setToolTip("Peaks with |skew| > threshold will be flagged")
        threshold_layout.addRow("Skewness Threshold:", self.skew_threshold_spin)
        
        # Coherence threshold
        self.coherence_threshold_spin = QDoubleSpinBox()
        self.coherence_threshold_spin.setRange(0.5, 0.95)
        self.coherence_threshold_spin.setSingleStep(0.05)
        self.coherence_threshold_spin.setValue(0.7)
        self.coherence_threshold_spin.setToolTip("Peaks with coherence < threshold will be flagged")
        threshold_layout.addRow("Coherence Threshold:", self.coherence_threshold_spin)
        
        # High correlation percentage
        self.high_corr_threshold_spin = QDoubleSpinBox()
        self.high_corr_threshold_spin.setRange(0.1, 0.9)
        self.high_corr_threshold_spin.setSingleStep(0.1)
        self.high_corr_threshold_spin.setValue(0.5)
        self.high_corr_threshold_spin.setToolTip("Minimum percentage of correlations > 0.9")
        threshold_layout.addRow("Min % High Correlations:", self.high_corr_threshold_spin)
        
        # Add the threshold group
        layout.addWidget(threshold_group)
        
        # Connect main checkbox to enable/disable thresholds group
        self.quality_checks_enabled.toggled.connect(threshold_group.setEnabled)
        
        # Initially disable the thresholds if main check is unchecked
        threshold_group.setEnabled(self.quality_checks_enabled.isChecked())
        
        # Add to tab widget
        self.tab_widget.addTab(tab, "Quality Checks")
    
    def _on_search_method_changed(self, method_text):
        """Handle search method combo box changes."""
        # Show/hide hybrid method options based on selection
        if method_text == "Hybrid Search":
            self.hybrid_method_combo.show()
            self.hybrid_method_label.show()
        else:
            self.hybrid_method_combo.hide()
            self.hybrid_method_label.hide()
    
    def _get_search_method(self):
        """Get the current search method as a string."""
        method_index = self.search_method_combo.currentIndex()
        if method_index == 0:
            return 'vector'
        elif method_index == 1:
            return 'w2v'
        elif method_index == 2:
            return 'hybrid'
        else:
            return 'vector'  # Default fallback
    
    def _select_weighting(self, scheme):
        """Show `scheme` (an ms-toolkit weighting-scheme name) in the dropdown."""
        index = self.weighting_combo.findData(scheme)
        if index < 0:
            index = self.weighting_combo.findData(DEFAULT_WEIGHTING)
        self.weighting_combo.setCurrentIndex(index)

    def _load_settings(self):
        """Load settings from QSettings."""
        # General tab
        self.search_method_combo.setCurrentIndex(self.settings.value("ms_search/method", 0, int))
        self.hybrid_method_combo.setCurrentIndex(self.settings.value("ms_search/hybrid_method", 0, int))  # Add hybrid method setting
        self.full_ms_baseline_check.setChecked(self.settings.value("ms_search/full_ms_baseline", False, bool))  # Load new setting
        
        # Trigger the visibility update for hybrid options
        self._on_search_method_changed(self.search_method_combo.currentText())
        
        # Extraction tab
        extraction_method = self.settings.value("ms_search/extraction_method", "apex")
        if extraction_method == "apex":
            self.apex_radio.setChecked(True)
        elif extraction_method == "average":
            self.average_radio.setChecked(True)
        elif extraction_method == "range":
            self.range_radio.setChecked(True)
        elif extraction_method == "midpoint":  # Add new method
            self.midpoint_radio.setChecked(True)
        
        self.range_points_spin.setValue(self.settings.value("ms_search/range_points", 5, int))
        self.midpoint_width_spin.setValue(self.settings.value("ms_search/midpoint_width_percent", 20, int))
        self.tic_weight_check.setChecked(self.settings.value("ms_search/tic_weight", True, bool))
        
        # Subtraction tab
        self.subtract_check.setChecked(self.settings.value("ms_search/subtract_enabled", True, bool))
        subtraction_method = self.settings.value("ms_search/subtraction_method", 2, int)
        self.subtraction_group.button(subtraction_method).setChecked(True)
        self.subtract_weight_spin.setValue(self.settings.value("ms_search/subtract_weight", 0.1, float))
        self.intensity_threshold_spin.setValue(self.settings.value("ms_search/intensity_threshold", 0.01, float))
        
        # Algorithm tab
        self.similarity_combo.setCurrentIndex(self.settings.value("ms_search/similarity", 1, int))
        self._select_weighting(read_weighting_scheme(self.settings))
        self.unmatched_combo.setCurrentIndex(self.settings.value("ms_search/unmatched", 0, int))
        self.intensity_power_spin.setValue(self.settings.value("ms_search/intensity_power", 0.6, float))
        self.top_n_spin.setValue(self.settings.value("ms_search/top_n", 5, int))
        
        # NEW: Load preselector settings
        self.preselector_combo.setCurrentIndex(self.settings.value("ms_search/preselector_type", 0, int))
        self.top_k_clusters_spin.setValue(self.settings.value("ms_search/top_k_clusters", 10, int))
        
        # Quality checks tab
        self.quality_checks_enabled.setChecked(self.settings.value("ms_search/quality_checks_enabled", False, bool))
        self.skew_check.setChecked(self.settings.value("ms_search/skew_check", False, bool))
        self.coherence_check.setChecked(self.settings.value("ms_search/coherence_check", False, bool))
        self.skew_threshold_spin.setValue(self.settings.value("ms_search/skew_threshold", 0.5, float))
        self.coherence_threshold_spin.setValue(self.settings.value("ms_search/coherence_threshold", 0.7, float))
        self.high_corr_threshold_spin.setValue(self.settings.value("ms_search/high_corr_threshold", 0.5, float))

        # Spectral Deconvolution tab
        from logic.spectral_deconv_settings import load_spectral_deconv_settings
        dp, gp = load_spectral_deconv_settings(self.settings)
        self.gap_tolerance_spin.setValue(gp.gap_tolerance or 0.0)
        self.padding_fraction_spin.setValue(gp.padding_fraction)
        self.rt_match_tolerance_spin.setValue(gp.rt_match_tolerance)
        self.max_window_peaks_spin.setValue(gp.max_window_peaks)
        self.noise_percentile_spin.setValue(dp.noise_percentile)
        self.search_range_spin.setValue(dp.search_range)
        self.min_peak_height_spin.setValue(dp.min_peak_height)
        self.min_top_edge_ratio_spin.setValue(dp.min_top_edge_ratio)
        self.min_points_spin.setValue(dp.min_points)
        self.rt_tolerance_spin.setValue(dp.rt_tolerance)
        self.min_r_spin.setValue(dp.min_r)
        self.min_ions_spin.setValue(dp.min_ions)
        self.excluded_mz_edit.setText(", ".join(f"{m:g}" for m in dp.excluded_mz))

    def _save_settings(self):
        """Save settings to QSettings."""
        # General tab
        self.settings.setValue("ms_search/method", self.search_method_combo.currentIndex())
        self.settings.setValue("ms_search/hybrid_method", self.hybrid_method_combo.currentIndex())  # Save hybrid method setting
        self.settings.setValue("ms_search/full_ms_baseline", self.full_ms_baseline_check.isChecked())  # Save new setting
        
        # Extraction tab
        extraction_method = "apex"
        if self.average_radio.isChecked():
            extraction_method = "average"
        elif self.range_radio.isChecked():
            extraction_method = "range"
        elif self.midpoint_radio.isChecked():  # Add new method
            extraction_method = "midpoint"
        
        self.settings.setValue("ms_search/extraction_method", extraction_method)
        self.settings.setValue("ms_search/range_points", self.range_points_spin.value())
        self.settings.setValue("ms_search/midpoint_width_percent", self.midpoint_width_spin.value())
        self.settings.setValue("ms_search/tic_weight", self.tic_weight_check.isChecked())
        
        # Subtraction tab
        self.settings.setValue("ms_search/subtract_enabled", self.subtract_check.isChecked())
        self.settings.setValue("ms_search/subtraction_method", self.subtraction_group.checkedId())
        self.settings.setValue("ms_search/subtract_weight", self.subtract_weight_spin.value())
        self.settings.setValue("ms_search/intensity_threshold", self.intensity_threshold_spin.value())
        
        # Algorithm tab
        self.settings.setValue("ms_search/similarity", self.similarity_combo.currentIndex())
        write_weighting_scheme(self.settings, self.weighting_combo.currentData())
        self.settings.setValue("ms_search/unmatched", self.unmatched_combo.currentIndex())
        self.settings.setValue("ms_search/intensity_power", self.intensity_power_spin.value())
        self.settings.setValue("ms_search/top_n", self.top_n_spin.value())
        
        # NEW: Save preselector settings
        self.settings.setValue("ms_search/preselector_type", self.preselector_combo.currentIndex())
        self.settings.setValue("ms_search/top_k_clusters", self.top_k_clusters_spin.value())
        
        # Quality checks tab
        self.settings.setValue("ms_search/quality_checks_enabled", self.quality_checks_enabled.isChecked())
        self.settings.setValue("ms_search/skew_check", self.skew_check.isChecked())
        self.settings.setValue("ms_search/coherence_check", self.coherence_check.isChecked())
        self.settings.setValue("ms_search/skew_threshold", self.skew_threshold_spin.value())
        self.settings.setValue("ms_search/coherence_threshold", self.coherence_threshold_spin.value())
        self.settings.setValue("ms_search/high_corr_threshold", self.high_corr_threshold_spin.value())

        # Spectral Deconvolution tab
        from logic.spectral_deconv_settings import save_spectral_deconv_settings
        dp, gp = self._read_spectral_deconv_params()
        save_spectral_deconv_settings(self.settings, dp, gp)

    def _read_spectral_deconv_params(self):
        """Widget values as (CorrelationDeconvParams, WindowGroupingParams); ValueError on bad m/z."""
        from logic.correlation_grouping import CorrelationDeconvParams
        from logic.spectral_deconv_runner import WindowGroupingParams
        from logic.spectral_deconv_settings import parse_mz_list
        excluded = parse_mz_list(self.excluded_mz_edit.text(), strict=True)
        raw_gap = self.gap_tolerance_spin.value()
        gp = WindowGroupingParams(
            gap_tolerance=None if raw_gap == 0.0 else raw_gap,
            padding_fraction=self.padding_fraction_spin.value(),
            rt_match_tolerance=self.rt_match_tolerance_spin.value(),
            max_window_peaks=self.max_window_peaks_spin.value(),
        )
        dp = CorrelationDeconvParams(
            noise_percentile=self.noise_percentile_spin.value(),
            search_range=self.search_range_spin.value(),
            min_peak_height=self.min_peak_height_spin.value(),
            min_top_edge_ratio=self.min_top_edge_ratio_spin.value(),
            min_points=self.min_points_spin.value(),
            rt_tolerance=self.rt_tolerance_spin.value(),
            min_r=self.min_r_spin.value(),
            min_ions=self.min_ions_spin.value(),
            excluded_mz=excluded,
        )
        return dp, gp

    def _restore_defaults(self):
        """Restore default values."""
        # General tab
        self.search_method_combo.setCurrentIndex(0)
        self.hybrid_method_combo.setCurrentIndex(0)  # Reset to Auto
        self.full_ms_baseline_check.setChecked(False)  # Reset new setting
        
        # Trigger visibility update for hybrid options
        self._on_search_method_changed(self.search_method_combo.currentText())
        
        # Extraction tab
        self.apex_radio.setChecked(True)
        self.range_points_spin.setValue(5)
        self.midpoint_width_spin.setValue(20)
        self.tic_weight_check.setChecked(True)
        
        # Subtraction tab
        self.subtract_check.setChecked(True)
        self.min_tic_radio.setChecked(True)
        self.subtract_weight_spin.setValue(0.1)
        self.intensity_threshold_spin.setValue(0.01)  # Reset to 1%
        
        # Algorithm tab
        self.similarity_combo.setCurrentIndex(1)  # Composite
        self._select_weighting(DEFAULT_WEIGHTING)
        self.unmatched_combo.setCurrentIndex(0)   # Keep All
        self.intensity_power_spin.setValue(0.6)
        self.top_n_spin.setValue(5)
        
        # NEW: Restore preselector defaults
        self.preselector_combo.setCurrentIndex(0)  # K-means
        self.top_k_clusters_spin.setValue(10)     # 10 clusters (matches 2026-06-12 audit; 98% preselector recall in measured datasets)
        
        # Quality checks tab
        self.quality_checks_enabled.setChecked(False)
        self.skew_check.setChecked(False)
        self.coherence_check.setChecked(False)
        self.skew_threshold_spin.setValue(0.5)
        self.coherence_threshold_spin.setValue(0.7)
        self.high_corr_threshold_spin.setValue(0.5)

        # Spectral Deconvolution tab
        from logic.correlation_grouping import CorrelationDeconvParams
        from logic.spectral_deconv_runner import WindowGroupingParams
        dp, gp = CorrelationDeconvParams(), WindowGroupingParams()
        self.gap_tolerance_spin.setValue(gp.gap_tolerance or 0.0)
        self.padding_fraction_spin.setValue(gp.padding_fraction)
        self.rt_match_tolerance_spin.setValue(gp.rt_match_tolerance)
        self.max_window_peaks_spin.setValue(gp.max_window_peaks)
        self.noise_percentile_spin.setValue(dp.noise_percentile)
        self.search_range_spin.setValue(dp.search_range)
        self.min_peak_height_spin.setValue(dp.min_peak_height)
        self.min_top_edge_ratio_spin.setValue(dp.min_top_edge_ratio)
        self.min_points_spin.setValue(dp.min_points)
        self.rt_tolerance_spin.setValue(dp.rt_tolerance)
        self.min_r_spin.setValue(dp.min_r)
        self.min_ions_spin.setValue(dp.min_ions)
        self.excluded_mz_edit.clear()

    def get_options(self):
        """Get the current options as a dictionary."""
        # Determine extraction method from radio buttons
        extraction_method = "apex"
        if self.average_radio.isChecked():
            extraction_method = "average"
        elif self.range_radio.isChecked():
            extraction_method = "range"
        elif self.midpoint_radio.isChecked():  # Add new method
            extraction_method = "midpoint"
        
        options = {
            # General options
            'search_method': self._get_search_method(),
            'hybrid_method': ['auto', 'fast', 'ensemble'][self.hybrid_method_combo.currentIndex()],
            'full_ms_baseline': self.full_ms_baseline_check.isChecked(),  # Add new option
            
            # Extraction options
            'extraction_method': extraction_method,
            'range_points': self.range_points_spin.value(),
            'midpoint_width_percent': self.midpoint_width_spin.value(),  # Add new option
            'tic_weight': self.tic_weight_check.isChecked(),
            
            # Subtraction options
            'subtract_enabled': self.subtract_check.isChecked(),
            'subtraction_method': ['left', 'right', 'min_tic', 'average'][self.subtraction_group.checkedId()],
            'subtract_weight': self.subtract_weight_spin.value(),
            'intensity_threshold': self.intensity_threshold_spin.value(),
            
            # Algorithm options
            'similarity': 'composite' if self.similarity_combo.currentIndex() == 1 else 'cosine',
            'weighting': self.weighting_combo.currentData(),
            'unmatched': ['keep_all', 'remove_all', 'keep_library', 'keep_experimental'][self.unmatched_combo.currentIndex()],
            'preselector_type': 'kmeans' if self.preselector_combo.currentIndex() == 0 else 'gmm',
            'top_k_clusters': self.top_k_clusters_spin.value(),
            'intensity_power': self.intensity_power_spin.value(),
            'top_n': self.top_n_spin.value(),
            
            # Quality checks options
            'quality_checks_enabled': self.quality_checks_enabled.isChecked(),
            'skew_check': self.skew_check.isChecked(),
            'coherence_check': self.coherence_check.isChecked(),
            'skew_threshold': self.skew_threshold_spin.value(),
            'coherence_threshold': self.coherence_threshold_spin.value(),
            'high_corr_threshold': self.high_corr_threshold_spin.value()
        }
        return options
    
    def accept(self):
        """Save settings when OK is clicked; refuse to close on an invalid excluded m/z."""
        try:
            self._read_spectral_deconv_params()
        except ValueError as e:
            # Show the field behind the box; Qt returns focus (and the selection) to it after.
            self.tab_widget.setCurrentWidget(self._spectral_deconv_tab)
            self.excluded_mz_edit.setFocus()
            self.excluded_mz_edit.selectAll()
            QMessageBox.warning(self, INVALID_MZ_TITLE, invalid_mz_message(e))
            return
        self._save_settings()
        super().accept()