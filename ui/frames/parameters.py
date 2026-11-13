from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QFormLayout,
    QCheckBox, QSlider, QDoubleSpinBox, QSpinBox, QComboBox, QHBoxLayout,
    QLineEdit, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QRadioButton, QButtonGroup, QStyle, QStyleOptionGroupBox
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent, QRect
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression
import copy
import html
import logging
import numpy as np

from logic.method import GCxGCParams, RiderParams
from logic.segments import (
    apply_segment, new_segment, remove_segment, resume_overrides, segments_view,
    set_whole_run_breaks,
)
from logic.windows import _EDGE_TOL
from ui.frames.parameter_help import PARAMETER_HELP, SECTION_KEY
from ui.frames.segments import SAVE_BLOCKED, SegmentEditor, SegmentsPanel

INFO_ICON = "\u24d8"
INFO_COLOR = "#1a73e8"


def _fix_table_height(table, rows=3):
    """Size a table to show exactly `rows` rows so the panel scrolls instead of squashing it."""
    table.verticalHeader().setDefaultSectionSize(30)
    height = (table.horizontalHeader().sizeHint().height()
              + rows * table.verticalHeader().defaultSectionSize()
              + 2 * table.frameWidth())
    table.setFixedHeight(height)


logger = logging.getLogger(__name__)

OVERRIDES_NOTE = "λ etc. set per section in the method file"
# Whole-run sections a segment's fields inherit from
_INHERITED_SECTIONS = ('smoothing', 'peaks', 'shoulders', 'negative_peaks', 'rider', 'baseline')
# peaks.mode values that run deconvolution: the GUI combo writes 'peak_splitting', the
# method schema documents 'deconvolution'
_DECONVOLUTION_MODES = ('peak_splitting', 'deconvolution')


def _has_override(value):
    """True if any leaf of an overrides structure is set (not None / empty)."""
    if isinstance(value, dict):
        return any(_has_override(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value)
    return value is not None


def _is_dead(widget):
    try:
        widget.isVisible()
        return False
    except RuntimeError:  # underlying C++ object deleted by a controls rebuild
        return True


class _TitleHelpIcon(QLabel):
    """Info icon overlaid on a QGroupBox, just after its title text.

    QGroupBox titles are plain text drawn by the style, so the icon is a child
    label positioned from the style's title rect and kept there on resize.
    """

    def __init__(self, group: QGroupBox, tip: str):
        super().__init__(INFO_ICON, group)
        self._group = group
        self.setStyleSheet(f"color: {INFO_COLOR}; font-weight: bold; padding: 0 3px 0 4px;")
        # Opaque background so the group box frame line doesn't strike through the icon.
        self.setAutoFillBackground(True)
        self.setToolTip(tip)
        self.setCursor(Qt.WhatsThisCursor)
        self.adjustSize()
        group.installEventFilter(self)
        self._reposition()
        self.raise_()

    def _reposition(self):
        group = self._group
        opt = QStyleOptionGroupBox()
        opt.initFrom(group)
        opt.text = group.title()
        opt.textAlignment = group.alignment()
        opt.subControls = QStyle.SC_GroupBoxLabel | QStyle.SC_GroupBoxFrame
        opt.lineWidth = 1
        rect = group.style().subControlRect(
            QStyle.CC_GroupBox, opt, QStyle.SC_GroupBoxLabel, group
        )
        fm = group.fontMetrics()
        if not rect.isValid():
            rect = QRect(8, 0, fm.horizontalAdvance(group.title()), fm.height())
        # Under the app stylesheet on macOS the style's label rect is narrower than the title as
        # drawn (77 px for "Peak Grouping", drawn 87 px), so the icon covered its last letters.
        # Place it after whichever is wider, plus a few px because that style also indents the text.
        right = max(rect.right(), rect.left() + fm.horizontalAdvance(group.title()))
        self.move(right + 3, rect.center().y() - self.height() // 2)

    def eventFilter(self, obj, event):
        if obj is self._group and event.type() in (
            QEvent.Resize, QEvent.Show, QEvent.FontChange, QEvent.StyleChange
        ):
            self._reposition()
        return False


class ParametersFrame(QWidget):
    """Frame containing integration parameters for chromatogram processing."""
    
    # Add signal for when parameters change
    parameters_changed = Signal(dict)
    gcxgc_display_changed = Signal()      # display-only GCxGC change (TIC source): redraw, never reprocess
    # None, or {'id','label','start','end','partway':[t...]} of the open segment (draft values)
    segment_state_changed = Signal(object)
    
    # Add signal for MS baseline correction button click
    ms_baseline_clicked = Signal()

    # Delegate Save/Load Method to the app-level document owner
    save_method_requested = Signal()
    load_method_requested = Signal()
    
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
                'enabled': True,
                'show_corrected': False,
                'method': 'arpls',  # Changed from 'asls' to 'arpls'
                'lambda': 1e4,      # Changed from 1e6 to 1e4
                'asymmetry': 0.01,
                'baseline_offset': 0.0,
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
                'range_filters': []  # Mirror of peak_windows spans (kept for older code paths)
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
                'sensitivity': 5,
            },
            'rider': RiderParams().model_dump(),
            'integration': {
                'peak_groups': []  # List of [start, end] time windows for grouping
            }
        }
        self.current_params['gcxgc'] = GCxGCParams().model_dump()
        # Per-RT-window parameters, edited as segments (logic/segments.py).
        self.current_params['peak_windows'] = []
        self.current_params['outside_windows'] = 'ignore'
        self._overrides_notes = []
        self._ui_mode = None
        self.segment_editor = None     # the open SegmentEditor, if any
        self._segment_snapshot = None  # current_params when it opened; Cancel restores it
        self._segment_hidden = {}      # setting group key -> isHidden() before it opened
        self._segment_warnings = {}    # segment id -> warning lines
        self._whole_run_breaks = []    # whole-run break dicts in table row order
        # The one place the Segments table refreshes: every change emits parameters_changed
        # (the Whole run row summarises whole-run values, so any edit may change it)
        self.parameters_changed.connect(lambda _params: self._refresh_segments_panel())
        
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

        # Segments table (top of the pane)
        self._init_segments_controls()
        
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

        # Add peak grouping controls
        self._init_peak_grouping_controls()

        # Add tail skim controls
        self._init_tail_skim_controls()

        # Add GCxGC controls
        self._init_gcxgc_controls()

        self._apply_help_icons()
        
        # Add stretch at the end to push everything to the top
        self.params_layout.addStretch()
        
        # Set the widget for the scroll area
        self.params_scroll.setWidget(self.params_widget)

        # ── Method file buttons ─────────────────────────────────────────
        _method_bar = QHBoxLayout()
        self._load_method_btn = QPushButton("Load Method\u2026")
        self._save_method_btn = QPushButton("Save Method\u2026")
        _method_bar.addWidget(self._load_method_btn)
        _method_bar.addWidget(self._save_method_btn)
        self._load_method_btn.clicked.connect(self.load_method_requested.emit)
        self._save_method_btn.clicked.connect(self.save_method_requested.emit)
        self.layout.addLayout(_method_bar)
        # ───────────────────────────────────────────────────────────────

        # Add scroll area to main layout
        self.layout.addWidget(self.params_scroll)
    
    def _apply_help_icons(self):
        """Append an info icon with hover help to every documented label/checkbox.

        Help text lives in ui/frames/parameter_help.py, keyed by section and by
        the visible label text, so it survives _reinitialize_controls().
        """
        self._apply_label_help(self.section_groups)
        for key, group in self.section_groups.items():
            help_map = PARAMETER_HELP.get(key)
            if isinstance(group, QGroupBox) and help_map and help_map.get(SECTION_KEY):
                _TitleHelpIcon(group, help_map[SECTION_KEY])

    @staticmethod
    def _apply_label_help(groups):
        """Info icon and hover help on the labels/checkboxes of ``groups`` ({section key: group}).

        The segment editor's groups use it too: their labels match the main groups' labels.
        """
        icon_html = f"<span style='color:{INFO_COLOR}; font-weight:bold;'>{INFO_ICON}</span>"
        for key, group in groups.items():
            if not isinstance(group, QGroupBox):
                continue
            help_map = PARAMETER_HELP.get(key)
            if not help_map:
                continue

            for label in group.findChildren(QLabel):
                text = label.text().strip()
                tip = help_map.get(text)
                if tip is None or label.property("_has_help"):
                    continue
                label.setTextFormat(Qt.RichText)
                label.setText(f"{html.escape(text)} {icon_html}")
                label.setToolTip(tip)
                label.setProperty("_has_help", True)

            for checkbox in group.findChildren(QCheckBox):
                text = checkbox.text().strip()
                tip = help_map.get(text)
                if tip is None or checkbox.property("_has_help"):
                    continue
                checkbox.setText(f"{text}  {INFO_ICON}")
                checkbox.setToolTip(tip)
                checkbox.setProperty("_has_help", True)

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
        self.smooth_method_combo.setCurrentIndex(max(0, self.smooth_method_combo.findData(self.current_params['smoothing']['method'])))
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
        self.smooth_diff_order_combo.setCurrentIndex(max(0, self.smooth_diff_order_combo.findData(self.current_params['smoothing']['diff_order'])))
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
        
        # Master switch: off integrates the smoothed signal without baseline subtraction
        self.baseline_enabled = QCheckBox("Enable Baseline Correction")
        self.baseline_enabled.setChecked(self.current_params['baseline'].get('enabled', True))
        self.baseline_enabled.stateChanged.connect(self._on_baseline_toggled)
        form_layout.addRow(self.baseline_enabled)

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
        self.baseline_offset_input = QLineEdit(str(self.current_params['baseline'].get('baseline_offset', 0.0)))
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
        fastchrom = self.current_params['baseline'].get('fastchrom') or {}
        self.fc_half_window.setValue(fastchrom.get('half_window') or 0)
        self.fc_half_window.setToolTip("Half-window for rolling std dev (~FWHM of peaks). 0 = Auto.")
        self.fc_half_window.valueChanged.connect(self._on_fastchrom_param_changed)
        fc_layout.addRow("Half Window:", self.fc_half_window)
        
        self.fc_smooth_half_window = QSpinBox()
        self.fc_smooth_half_window.setMinimum(0)
        self.fc_smooth_half_window.setMaximum(500)
        self.fc_smooth_half_window.setSpecialValueText("Auto")
        self.fc_smooth_half_window.setValue(fastchrom.get('smooth_half_window') or 0)
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
        
        # Break point controls for segmented baseline fitting
        bp_frame = QFrame()
        self.break_point_frame = bp_frame
        bp_frame.setFrameStyle(QFrame.Box)
        bp_layout = QVBoxLayout(bp_frame)
        
        bp_label = QLabel("Baseline changes (whole run)")
        bp_label.setStyleSheet("font-weight: bold;")
        bp_layout.addWidget(bp_label)
        
        bp_info = QLabel("Restart the baseline fit at valve-switch step changes.\n"
                         "Each part is fitted on its own. Segments set theirs in the segment editor.")
        bp_info.setStyleSheet("color: #666666; font-size: 10px;")
        bp_layout.addWidget(bp_info)
        
        self.break_point_table = QTableWidget(0, 3)
        self.break_point_table.setHorizontalHeaderLabels(["Time (min)", "Tolerance", ""])
        self.break_point_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.break_point_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.break_point_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.break_point_table.setSelectionMode(QAbstractItemView.NoSelection)
        _fix_table_height(self.break_point_table)
        bp_layout.addWidget(self.break_point_table)
        
        bp_add_btn = QPushButton("+ Add Break Point")
        bp_add_btn.clicked.connect(lambda: self._add_break_point_row())
        self._rebuild_break_point_rows()
        bp_layout.addWidget(bp_add_btn)
        bp_layout.addWidget(self._make_overrides_note())
        
        advanced_layout.addWidget(bp_frame)

        form_layout.addRow(self.baseline_advanced_container)
        
        # Add to parameters layout
        self._update_baseline_controls_state()
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

        self.peak_mode_combo.setCurrentIndex(max(0, self.peak_mode_combo.findData(self.current_params['peaks'].get('mode', 'classical'))))
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
        self.splitting_method_combo.setCurrentIndex(max(0, self.splitting_method_combo.findData(self.current_params['peak_splitting'].get('splitting_method', 'geometric'))))
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
        _fix_table_height(self.peak_splitting_windows_table)
        peak_splitting_layout.addRow(self.peak_splitting_windows_table)

        add_window_btn = QPushButton("+ Add Window")
        add_window_btn.clicked.connect(lambda: self._add_peak_splitting_window_row())
        for start, end in self.current_params['peak_splitting'].get('windows') or []:
            self._add_peak_splitting_window_row(start, end, notify=False)
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
        
        shoulder_info = QLabel("Finds peaks merged into a larger one (shoulders) or riding its tail\n(riders): curvature proposes them, a peak-shape fit confirms them.")
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
            "5 = Default\n"
            "10 = Most sensitive"
        )
        detection_layout.addRow(sensitivity_label, sensitivity_container)
        
        form_layout.addRow("Detection:", detection_frame)
        
        # Enable/disable controls based on initial state
        self._update_shoulder_controls_state()
        
        # Add to parameters layout
        self.section_groups['shoulders'] = shoulder_group
        self.params_layout.addWidget(shoulder_group)

    # ── Segments (design 2026-11-12 §4.2) ─────────────────────────────────

    def _init_segments_controls(self):
        """The Segments table at the top of the pane; the segment editor opens just below it."""
        panel = SegmentsPanel()
        panel.open_requested.connect(self._on_segment_open_requested)
        panel.add_requested.connect(self._on_segment_add_requested)
        panel.delete_requested.connect(self._on_segment_delete)
        panel.outside_changed.connect(self._on_outside_changed)
        self.segments_panel = panel
        self.section_groups['segments'] = panel
        self.params_layout.addWidget(panel)
        self._refresh_segments_panel()

    def _setting_keys(self):
        """The 1D setting groups the segment editor takes the place of."""
        return [k for k in self.section_groups if k != 'segments' and not k.startswith('gcxgc')]

    def _whole_run_summary(self, view):
        def num(v):
            return f"{v:g}" if isinstance(v, (int, float)) else str(v)
        b, p = self.current_params['baseline'], self.current_params['peaks']
        parts = [f"{b.get('method')} λ {num(b.get('lambda'))}" if b.get('enabled', True)
                 else 'baseline off',
                 f"prom {num(p.get('min_prominence'))}"]
        n = len(view['whole_run_breaks'])
        if n:
            parts.append(f"{n} baseline change{'s' if n > 1 else ''}")
        return ' · '.join(parts)

    def _refresh_segments_panel(self):
        peaks = self.current_params.get('peaks') or {}
        if peaks.get('range_filters') and not self.current_params.get('peak_windows'):
            # ChromaMethod always turns range_filters into windows; params like these were
            # built some other way, and the first segment edit rewrites the mirror from windows
            logger.warning("Parameters have peaks.range_filters %s but no peak_windows; the "
                           "Segments table shows no segments and a segment edit drops them",
                           peaks['range_filters'])
        view = segments_view(self.current_params)
        self.segments_panel.set_view(view, self._whole_run_summary(view),
                                     self.current_params.get('outside_windows', 'ignore'))
        for sid, lines in self._segment_warnings.items():
            self.segments_panel.set_row_warning(sid, '\n'.join(lines))

    def _after_segment_change(self):
        """Redraw the whole-run breaks after a segment change (the Segments table follows
        parameters_changed)."""
        self._rebuild_break_point_rows()
        self._update_overrides_note()

    def _emit_segment_state(self, record):
        self.segment_state_changed.emit({
            'id': record['id'], 'label': record.get('label', ''),
            'start': record['start'], 'end': record['end'],
            'partway': [row['time'] for row in record.get('partway') or []],
        })

    def open_segment(self, segment_id, span=None):
        """Open the editor on a segment, or on a new one over ``span`` when ``segment_id`` is None.

        A new segment is not in current_params until a preview or Save applies it. A span
        given with an existing id moves that segment's edges (one preview).
        """
        if self.segment_editor is not None:
            logger.info("Segment '%s' cancelled: another segment was opened",
                        self.segment_editor.record()['id'])
            self.cancel_open_segment()
        snapshot = copy.deepcopy(self.current_params)
        if segment_id is None:
            if span is None:
                raise ValueError("A new segment needs a span")
            record, is_new = new_segment(snapshot, *sorted(float(t) for t in span)), True
        else:
            record = next((r for r in segments_view(snapshot)['segments'] if r['id'] == segment_id), None)
            if record is None:
                raise ValueError(f"No segment with id '{segment_id}'")
            is_new = False
        inherited = {s: copy.deepcopy(snapshot.get(s) or {}) for s in _INHERITED_SECTIONS}
        deconvolution = snapshot['peaks'].get('mode') in _DECONVOLUTION_MODES
        editor = SegmentEditor(record, inherited, resume_overrides(snapshot, record['start']),
                               deconvolution, is_new)
        editor.preview_changed.connect(self._on_segment_preview)
        editor.save_requested.connect(self._save_segment)
        editor.cancel_requested.connect(self.cancel_open_segment)
        editor.delete_requested.connect(self._on_segment_delete)
        editor.set_warnings(self._segment_warnings.get(record['id'], []))
        self._apply_label_help(editor.groups)

        self._segment_snapshot = snapshot
        self.segment_editor = editor
        self._segment_hidden = {}
        for key in self._setting_keys():
            group = self.section_groups[key]
            self._segment_hidden[key] = group.isHidden()
            group.hide()
        self.params_layout.insertWidget(self.params_layout.indexOf(self.segments_panel) + 1, editor)
        self.segments_panel.set_editing(True)
        self._emit_segment_state(record)
        if segment_id is not None and span is not None:
            editor.set_span(*sorted(float(t) for t in span))

    def _close_segment_editor(self):
        """Remove the editor and give the setting groups back the visibility they had."""
        editor = self.segment_editor
        self.segment_editor = None
        self._segment_snapshot = None
        self.params_layout.removeWidget(editor)
        editor.hide()
        editor.deleteLater()
        for key, hidden in self._segment_hidden.items():
            if key in self.section_groups:
                self.section_groups[key].setHidden(hidden)
        self._segment_hidden = {}
        self.segments_panel.set_editing(False)
        self.segment_state_changed.emit(None)

    def _on_segment_preview(self, record):
        """Apply the draft to the snapshot; an invalid draft shows its reason and keeps the last preview."""
        editor = self.segment_editor
        editor.set_resume(resume_overrides(self._segment_snapshot, record['start']))  # R12
        try:
            params = apply_segment(self._segment_snapshot, record)
        except ValueError as exc:
            editor.show_error(str(exc))
        else:
            editor.show_error(None)
            self.current_params = params
            self._after_segment_change()
            self.parameters_changed.emit(self.current_params)
        self._emit_segment_state(record)

    def _save_segment(self, record) -> bool:
        try:
            params = apply_segment(self._segment_snapshot, record)
        except ValueError as exc:
            self.segment_editor.show_error(str(exc))
            return False
        self.current_params = params
        self._close_segment_editor()
        self._after_segment_change()
        self.parameters_changed.emit(self.current_params)
        return True

    def commit_open_segment(self) -> bool:
        """Save the open segment. True if nothing is open or the save succeeded."""
        editor = self.segment_editor
        if editor is None:
            return True
        if editor.has_errors():
            editor.show_error(SAVE_BLOCKED)
            return False
        return self._save_segment(editor.record())

    def cancel_open_segment(self, emit=True) -> None:
        """Close the editor and restore the method as it was when the segment was opened."""
        if self.segment_editor is None:
            return
        self.current_params = self._segment_snapshot
        self._close_segment_editor()
        self._after_segment_change()
        if emit:
            self.parameters_changed.emit(self.current_params)
        else:
            self._refresh_segments_panel()

    def _require_editor(self):
        if self.segment_editor is None:
            raise RuntimeError("No segment is open")
        return self.segment_editor

    def segment_add_partway(self, time: float) -> None:
        self._require_editor().add_partway(time)

    def segment_set_edge(self, edge: str, time: float) -> None:
        editor = self._require_editor()
        if edge == 'start':
            editor.set_span(time, editor.to_edit.value())
        elif edge == 'end':
            editor.set_span(editor.from_edit.value(), time)
        else:
            raise ValueError(f"edge must be 'start' or 'end', not {edge!r}")

    def set_segment_warnings(self, lines_by_id: dict) -> None:
        """Warnings per segment id: a ⚠ on its row, and lines in the editor when it is open."""
        for sid in set(self._segment_warnings) | set(lines_by_id):
            self.segments_panel.set_row_warning(sid, '\n'.join(lines_by_id.get(sid) or []) or None)
        self._segment_warnings = {sid: list(lines) for sid, lines in lines_by_id.items() if lines}
        if self.segment_editor is not None:
            self.segment_editor.set_warnings(
                self._segment_warnings.get(self.segment_editor.record()['id'], []))

    def _on_segment_open_requested(self, segment_id):
        if segment_id:
            self.open_segment(segment_id)
        else:
            self.cancel_open_segment()  # Whole run: the setting groups

    def _on_segment_add_requested(self):
        """A new segment 1 min after the last one, or 0-60 when there is none."""
        last_end = max((w['end'] for w in self.current_params.get('peak_windows') or []), default=None)
        self.open_segment(None, (0.0, 60.0) if last_end is None else (last_end, last_end + 1.0))

    def _on_segment_delete(self, segment_id):
        """R13: no confirmation; nothing is saved until Save Method."""
        if self.segment_editor is not None:
            params = remove_segment(self._segment_snapshot, segment_id)
            self._close_segment_editor()
        else:
            params = remove_segment(self.current_params, segment_id)
        self.current_params = params
        self._after_segment_change()
        self.parameters_changed.emit(self.current_params)

    def _on_outside_changed(self, mode):
        self.current_params['outside_windows'] = mode
        self._sync_range_filter_mirror()
        self.parameters_changed.emit(self.current_params)

    def _make_overrides_note(self):
        note = QLabel(OVERRIDES_NOTE)
        note.setStyleSheet("color: #666666; font-size: 10px;")
        note.setVisible(self._params_have_overrides())
        self._overrides_notes = [n for n in self._overrides_notes if not _is_dead(n)]
        self._overrides_notes.append(note)
        return note

    def _params_have_overrides(self):
        windows = self.current_params.get('peak_windows') or []
        breaks = self.current_params.get('baseline', {}).get('break_points') or []
        return (any(_has_override({k: v for k, v in w.items()
                                   if k not in ('id', 'label', 'start', 'end')})
                    for w in windows)
                or any(_has_override(b.get('overrides')) for b in breaks))

    def _update_overrides_note(self):
        visible = self._params_have_overrides()
        self._overrides_notes = [n for n in self._overrides_notes if not _is_dead(n)]
        for n in self._overrides_notes:
            n.setVisible(visible)

    def _sync_range_filter_mirror(self):
        """peaks.range_filters mirrors the window spans under 'ignore' and is empty under 'global'."""
        windows = self.current_params['peak_windows']
        self.current_params['peaks']['range_filters'] = (
            [[w['start'], w['end']] for w in windows]
            if self.current_params.get('outside_windows', 'ignore') == 'ignore' else [])

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
        _fix_table_height(self.peak_group_table)
        layout.addWidget(self.peak_group_table)

        add_btn = QPushButton("+ Add Group")
        add_btn.clicked.connect(lambda: self._add_peak_group_row())
        layout.addWidget(add_btn)

        self.section_groups['peak_grouping'] = group
        self.params_layout.addWidget(group)
        for start, end in self.current_params.get('integration', {}).get('peak_groups') or []:
            self._add_peak_group_row(start, end, notify=False)

    _RIDER_MODES = [("Automatic", "auto"), ("Always skim", "skim"), ("Always split", "split")]
    _RIDER_MODELS = [("Exponential", "exponential"), ("Tangent", "tangent")]
    # (attribute, rider key, label, min, max, step, decimals)
    # Ranges follow logic.method.RiderParams' validation, so a loaded method shows its real value
    _RIDER_SPINS = [
        ("skim_height_ratio_spin", "skim_height_ratio", "Height ratio:", 1.0, 100.0, 0.1, 1),
        ("skim_max_valley_frac_spin", "skim_max_valley_frac", "Max valley fraction:", 0.01, 1.0, 0.05, 2),
        ("skim_valley_ratio_spin", "skim_valley_ratio", "Valley ratio:", 1.0, 1000.0, 1.0, 1),
        ("skim_tail_bend_spin", "skim_tail_bend", "Tail bend:", 0.01, 10.0, 0.01, 2),
    ]

    def _init_tail_skim_controls(self):
        """Initialize tail skim controls (small peak riding on a larger peak's tail)."""
        rider = self.current_params['rider']
        group = QGroupBox("Tail Skim")
        form = QFormLayout(group)

        info = QLabel("Integrates a small peak on a larger peak's tail above a skim\nbaseline instead of splitting it off the tail.")
        info.setStyleSheet("color: #666666; font-size: 10px;")
        form.addRow("", info)

        self.rider_mode_combo = QComboBox()
        for text, value in self._RIDER_MODES:
            self.rider_mode_combo.addItem(text, value)
        self.rider_mode_combo.setCurrentIndex(self.rider_mode_combo.findData(rider['rider_integration']))
        self.rider_mode_combo.currentIndexChanged.connect(self._on_rider_mode_changed)
        form.addRow("Mode:", self.rider_mode_combo)

        self.rider_model_combo = QComboBox()
        for text, value in self._RIDER_MODELS:
            self.rider_model_combo.addItem(text, value)
        self.rider_model_combo.setCurrentIndex(self.rider_model_combo.findData(rider['skim_model']))
        self.rider_model_combo.currentIndexChanged.connect(self._on_rider_model_changed)
        form.addRow("Skim model:", self.rider_model_combo)

        self.skim_advanced_toggle = QPushButton("\u25b6 Advanced")
        self.skim_advanced_toggle.setFlat(True)
        self.skim_advanced_toggle.setStyleSheet("text-align: left; font-weight: bold;")
        self.skim_advanced_toggle.clicked.connect(self._toggle_skim_advanced)
        form.addRow(self.skim_advanced_toggle)

        self.skim_advanced_frame = QFrame()
        self.skim_advanced_frame.setFrameStyle(QFrame.Box)
        adv = QFormLayout(self.skim_advanced_frame)
        for attr, key, label, lo, hi, step, decimals in self._RIDER_SPINS:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(rider[key])
            spin.valueChanged.connect(lambda v, k=key: self._on_rider_value_changed(k, v))
            setattr(self, attr, spin)
            adv.addRow(label, spin)
        self.skim_advanced_frame.setToolTip("Applies in Automatic mode only")
        self.skim_advanced_frame.setVisible(False)
        form.addRow(self.skim_advanced_frame)

        self._update_skim_controls_state()
        self.section_groups['tail_skim'] = group
        self.params_layout.addWidget(group)

    def _update_skim_controls_state(self):
        # Thresholds only matter when the rule decides split vs skim.
        self.skim_advanced_frame.setEnabled(self.current_params['rider']['rider_integration'] == 'auto')

    def _on_rider_mode_changed(self, _index):
        self.current_params['rider']['rider_integration'] = self.rider_mode_combo.currentData()
        self._update_skim_controls_state()
        self.parameters_changed.emit(self.current_params)

    def _on_rider_model_changed(self, _index):
        self.current_params['rider']['skim_model'] = self.rider_model_combo.currentData()
        self.parameters_changed.emit(self.current_params)

    def _on_rider_value_changed(self, key, value):
        self.current_params['rider'][key] = value
        self.parameters_changed.emit(self.current_params)

    def _toggle_skim_advanced(self):
        visible = not self.skim_advanced_frame.isVisible()
        self.skim_advanced_frame.setVisible(visible)
        self.skim_advanced_toggle.setText("\u25bc Advanced" if visible else "\u25b6 Advanced")

    def _add_peak_group_row(self, start=0.0, end=60.0, notify=True):
        """Add a new row to the peak grouping table."""
        row = self.peak_group_table.rowCount()
        self.peak_group_table.insertRow(row)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(0, 9999)
        start_spin.setValue(start)
        start_spin.valueChanged.connect(self._on_peak_groups_changed)
        self.peak_group_table.setCellWidget(row, 0, start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(0, 9999)
        end_spin.setValue(end)
        end_spin.valueChanged.connect(self._on_peak_groups_changed)
        self.peak_group_table.setCellWidget(row, 1, end_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_peak_group_row(r))
        self.peak_group_table.setCellWidget(row, 2, del_btn)
        if notify:
            self._on_peak_groups_changed()

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

    def _add_break_point_row(self, time=0.0, tolerance=0.1, notify=True):
        """Add a new row to the whole-run break table."""
        row = self.break_point_table.rowCount()
        self.break_point_table.insertRow(row)

        time_spin = QDoubleSpinBox()
        time_spin.setDecimals(3)
        time_spin.setRange(0, 9999)
        time_spin.setValue(time)
        time_spin.valueChanged.connect(self._on_break_points_changed)
        self.break_point_table.setCellWidget(row, 0, time_spin)

        tol_spin = QDoubleSpinBox()
        tol_spin.setDecimals(3)
        tol_spin.setRange(0, 10)
        tol_spin.setValue(tolerance)
        tol_spin.valueChanged.connect(self._on_break_points_changed)
        self.break_point_table.setCellWidget(row, 1, tol_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_break_point_row(r))
        self.break_point_table.setCellWidget(row, 2, del_btn)
        # A row left at its defaults is still part of the method.
        if notify:
            self._on_break_points_changed()

    def _remove_break_point_row(self, row):
        """Remove a row (and its break point) from the whole-run break table."""
        if row < len(self._whole_run_breaks):
            del self._whole_run_breaks[row]
        self.break_point_table.removeRow(row)
        for r in range(self.break_point_table.rowCount()):
            del_btn = self.break_point_table.cellWidget(r, 2)
            if del_btn:
                del_btn.clicked.disconnect()
                del_btn.clicked.connect(lambda checked, r=r: self._remove_break_point_row(r))
        self._on_break_points_changed()

    def _on_break_points_changed(self, _=None):
        """Write the table's whole-run breaks back; overrides stay as loaded.

        A duplicate time, or a time inside a segment (it would become that segment's
        baseline change), flags the rows red and keeps the last valid set.
        """
        old = self._whole_run_breaks
        table = self.break_point_table
        bps = []
        for r in range(table.rowCount()):
            time_w, tol_w = table.cellWidget(r, 0), table.cellWidget(r, 1)
            if time_w and tol_w:
                bp = copy.deepcopy(old[r]) if r < len(old) else {}
                bp['time'], bp['tolerance'] = time_w.value(), tol_w.value()
                bps.append(bp)
        for r in range(table.rowCount()):
            for c in (0, 1):
                spin = table.cellWidget(r, c)
                if spin:
                    spin.setStyleSheet("")
                    spin.setToolTip("")
        try:
            params = set_whole_run_breaks(self.current_params, bps)
            self._check_breaks_outside_segments(params, bps)
        except ValueError as exc:
            msg = str(exc)
            logger.warning("Break point edit not applied: %s", msg)
            self._flag_break_rows(bps, msg)
            return
        self.current_params = params
        self._whole_run_breaks = bps
        self.parameters_changed.emit(self.current_params)

    @staticmethod
    def _check_breaks_outside_segments(params, bps):
        """Raise if a table break landed inside a segment (or on its start) in ``params``."""
        kept = segments_view(params)['whole_run_breaks']
        if len(kept) == len(bps):
            return
        kept_times = [float(bp['time']) for bp in kept]
        for bp in bps:
            t = float(bp['time'])
            if any(abs(t - k) < _EDGE_TOL for k in kept_times):
                continue
            w = next(w for w in params['peak_windows']
                     if w['start'] - _EDGE_TOL < t < w['end'] - _EDGE_TOL)
            raise ValueError(f"Baseline change at {t:g} is inside segment "
                             f"'{w.get('label') or w['id']}'; change it in that segment")
        raise ValueError("A whole-run baseline change was taken over by a segment")

    def _flag_break_rows(self, bps, msg):
        """Red rows: the time named in ``msg``'s duplicate or segment, else every row."""
        current = self.current_params['baseline'].get('break_points') or []
        whole = segments_view(self.current_params)['whole_run_breaks']
        owned = [float(bp['time']) for bp in current if bp not in whole]
        times = [float(bp['time']) for bp in bps]
        bad = [r for r, t in enumerate(times)
               if any(abs(t - o) < _EDGE_TOL for i, o in enumerate(times) if i != r)
               or any(abs(t - o) < _EDGE_TOL for o in owned)
               or f"at {t:g} is inside" in msg]
        table = self.break_point_table
        for r in bad or range(len(bps)):
            for c in (0, 1):
                table.cellWidget(r, c).setStyleSheet("border: 1px solid red;")
                table.cellWidget(r, c).setToolTip(msg)

    def _rebuild_break_point_rows(self):
        """Redraw the whole-run break table from current_params."""
        self.break_point_table.setRowCount(0)
        self._whole_run_breaks = segments_view(self.current_params)['whole_run_breaks']
        for bp in self._whole_run_breaks:
            self._add_break_point_row(bp['time'], bp.get('tolerance', 0.1), notify=False)

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

    def _update_baseline_controls_state(self):
        """Grey out the fit settings while baseline correction is off.

        Show Corrected Signal and the MS baseline button stay live: the first is
        display-only and the second works on the m/z traces, not this fit.
        """
        enabled = self.baseline_enabled.isChecked()
        for w in (self.baseline_algorithm, self.lam_container, self.fastchrom_container,
                  self.baseline_offset_input, self.break_point_frame):
            w.setEnabled(enabled)

    def _on_baseline_toggled(self, state):
        """Handle baseline correction enable/disable toggle"""
        self.current_params['baseline']['enabled'] = bool(state)
        self._update_baseline_controls_state()
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
            self.prominence_entry.setToolTip(
                'Prominence threshold for classical peak detection.\n'
                'In hybrid mode (with peak splitting windows), this controls\n'
                'peak sensitivity outside the peak splitting windows.'
            )
            self.prominence_help.setText("Controls classical peaks (outside peak splitting windows)")
        else:
            self.current_params['peaks']['min_prominence'] = 1e5
            self.prominence_entry.setText('100000.0')
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

    def _add_peak_splitting_window_row(self, start=0.0, end=60.0, notify=True):
        """Add a new row to the peak splitting windows table."""
        row = self.peak_splitting_windows_table.rowCount()
        self.peak_splitting_windows_table.insertRow(row)

        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(3)
        start_spin.setRange(0, 9999)
        start_spin.setValue(start)
        start_spin.valueChanged.connect(self._on_peak_splitting_windows_changed)
        self.peak_splitting_windows_table.setCellWidget(row, 0, start_spin)

        end_spin = QDoubleSpinBox()
        end_spin.setDecimals(3)
        end_spin.setRange(0, 9999)
        end_spin.setValue(end)
        end_spin.valueChanged.connect(self._on_peak_splitting_windows_changed)
        self.peak_splitting_windows_table.setCellWidget(row, 1, end_spin)

        del_btn = QPushButton("✕")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda checked, r=row: self._remove_peak_splitting_window_row(r))
        self.peak_splitting_windows_table.setCellWidget(row, 2, del_btn)
        # A row left at its defaults is still part of the method.
        if notify:
            self._on_peak_splitting_windows_changed()

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
            if getattr(self, '_ui_mode', None) == 'gcxgc' and not key.startswith('gcxgc'):
                continue
            if key in self._segment_hidden:   # segment editor open: applies when it closes
                self._segment_hidden[key] = not visible
            elif key in self.section_groups:
                self.section_groups[key].setVisible(visible)

    def _on_shoulders_toggled(self, state):
        """Handle shoulder detection enable/disable toggle"""
        enabled = bool(state)
        self.current_params['shoulders']['enabled'] = enabled
        self._update_shoulder_controls_state()
        self.parameters_changed.emit(self.current_params)
    
    def _on_shoulder_sensitivity_changed(self, value):
        """Handle shoulder sensitivity slider change"""
        self.shoulder_sensitivity_slider.setValue(value)
        self.shoulder_sensitivity_spinbox.setValue(value)
        self.current_params['shoulders']['sensitivity'] = value
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

    _PROVENANCE_TEXT = {"hdr": "from file (HDR)", "lsc_view": "ChromSpace saved view",
                        "user": "your value", "default": "default"}

    def _init_gcxgc_controls(self):
        """GCxGC controls: 2D layout, 2D baseline, 2D peak detection. All bound to current_params['gcxgc']."""
        g = GCxGCParams(**self.current_params.get('gcxgc', {}))
        # Empty is Acceptable, so clearing a field + Return / focus-out emits editingFinished and stores None.
        num_validator = QRegularExpressionValidator(QRegularExpression(r'^(-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)?$'))

        # ── 2D Layout ──
        layout_group = QGroupBox("2D Layout")
        ly = QFormLayout(layout_group)
        self.gcxgc_pm_entry = QLineEdit("" if g.modulation_period_s is None else f"{g.modulation_period_s:g}")
        self.gcxgc_pm_entry.setValidator(num_validator)
        self.gcxgc_pm_entry.setToolTip("Modulation period (s). Leave blank to use the run's files.")
        self.gcxgc_pm_source = QLabel("")
        self.gcxgc_pm_source.setStyleSheet("color: #666666; font-size: 10px;")
        ly.addRow("Modulation Period (s):", self.gcxgc_pm_entry)
        ly.addRow("", self.gcxgc_pm_source)
        self.gcxgc_phase_entry = QLineEdit("" if g.phase_shift_s is None else f"{g.phase_shift_s:g}")
        self.gcxgc_phase_entry.setValidator(num_validator)
        self.gcxgc_phase_entry.setToolTip("Phase shift (s), ChromSpace's sign convention. Leave blank to use ChromSpace's saved view.")
        self.gcxgc_phase_source = QLabel("")
        self.gcxgc_phase_source.setStyleSheet("color: #666666; font-size: 10px;")
        ly.addRow("Phase Shift (s):", self.gcxgc_phase_entry)
        ly.addRow("", self.gcxgc_phase_source)
        self.gcxgc_signal_combo = QComboBox()
        for label, key in (("FID", "fid"), ("DBC TIC", "dbc_tic"), ("TIC", "tic")):
            self.gcxgc_signal_combo.addItem(label, key)
        self.gcxgc_signal_combo.setCurrentIndex(max(0, self.gcxgc_signal_combo.findData(g.detection_signal)))
        self.gcxgc_signal_combo.setToolTip("Signal used for peak detection and volumes.")
        ly.addRow("Detection Signal:", self.gcxgc_signal_combo)
        layout_group.setVisible(False)
        self.section_groups['gcxgc_layout'] = layout_group
        self.params_layout.addWidget(layout_group)

        # ── 2D Baseline Correction ──
        baseline_group = QGroupBox("2D Baseline Correction")
        bl = QFormLayout(baseline_group)
        self.gcxgc_baseline_enabled = QCheckBox("Apply 2D Baseline Correction")
        self.gcxgc_baseline_enabled.setChecked(g.baseline_enabled)
        bl.addRow(self.gcxgc_baseline_enabled)
        lam_exp = int(round(np.log10(g.baseline_lambda)))
        lam_box = QWidget()
        lam_row = QHBoxLayout(lam_box)
        lam_row.setContentsMargins(0, 0, 0, 0)
        self.gcxgc_lam_slider = QSlider(Qt.Horizontal)
        self.gcxgc_lam_slider.setRange(2, 12)
        self.gcxgc_lam_slider.setValue(lam_exp)
        self.gcxgc_lam_spinbox = QSpinBox()
        self.gcxgc_lam_spinbox.setRange(2, 12)
        self.gcxgc_lam_spinbox.setValue(lam_exp)
        self.gcxgc_lam_slider.valueChanged.connect(self.gcxgc_lam_spinbox.setValue)
        self.gcxgc_lam_spinbox.valueChanged.connect(self.gcxgc_lam_slider.setValue)
        lam_row.addWidget(self.gcxgc_lam_slider, 3)
        lam_row.addWidget(self.gcxgc_lam_spinbox, 1)
        self.gcxgc_lam_label = QLabel(f"λ = 10^{lam_exp}")
        self.gcxgc_lam_slider.valueChanged.connect(lambda v: self.gcxgc_lam_label.setText(f"λ = 10^{v}"))
        bl.addRow("Lambda (λ):", lam_box)
        bl.addRow("", self.gcxgc_lam_label)
        self.gcxgc_diff_order_spin = QSpinBox()
        self.gcxgc_diff_order_spin.setRange(1, 3)
        self.gcxgc_diff_order_spin.setValue(g.baseline_diff_order)
        bl.addRow("Diff. Order:", self.gcxgc_diff_order_spin)
        baseline_group.setVisible(False)
        self.section_groups['gcxgc_baseline'] = baseline_group
        self.params_layout.addWidget(baseline_group)

        # ── 2D Peak Detection ──
        peaks_group = QGroupBox("2D Peak Detection")
        pk = QFormLayout(peaks_group)
        self.gcxgc_peaks_enabled = QCheckBox("Enable Peak Detection")
        self.gcxgc_peaks_enabled.setChecked(g.peaks_enabled)
        pk.addRow(self.gcxgc_peaks_enabled)
        self.gcxgc_prominence_entry = QLineEdit("" if g.min_prominence is None else f"{g.min_prominence:g}")
        self.gcxgc_prominence_entry.setValidator(num_validator)
        self.gcxgc_prominence_entry.setToolTip("Minimum peak prominence in signal units (e.g. 1e4). Blank = none.")
        pk.addRow("Min Prominence:", self.gcxgc_prominence_entry)
        self.gcxgc_rt2_tolerance_spin = QSpinBox()
        self.gcxgc_rt2_tolerance_spin.setRange(1, 20)
        self.gcxgc_rt2_tolerance_spin.setValue(g.rt2_grouping_tolerance)
        self.gcxgc_rt2_tolerance_spin.setToolTip("Max 2D shift (scans) between slices of one peak in adjacent modulations.")
        pk.addRow("2D RT Tolerance (scans):", self.gcxgc_rt2_tolerance_spin)
        self.gcxgc_min_sub_peaks_spin = QSpinBox()
        self.gcxgc_min_sub_peaks_spin.setRange(1, 10)
        self.gcxgc_min_sub_peaks_spin.setValue(g.min_modulations)
        self.gcxgc_min_sub_peaks_spin.setToolTip("Minimum number of modulations a peak must span.")
        pk.addRow("Min Modulations:", self.gcxgc_min_sub_peaks_spin)
        self.gcxgc_remove_bands = QCheckBox("Ignore persistent bands (TIC only)")
        self.gcxgc_remove_bands.setChecked(g.remove_bands)
        self.gcxgc_remove_bands.setToolTip(
            "When detecting on a TIC, do not report signal that persists along the first dimension "
            "(air, column bleed) as peaks. Peak volumes are not changed. Turn off for dense samples, "
            "where real peaks can fill a 2D column for minutes. Has no effect on the FID.")
        pk.addRow(self.gcxgc_remove_bands)
        self.gcxgc_solvent_end_entry = QLineEdit("" if g.solvent_end_min is None else f"{g.solvent_end_min:g}")
        self.gcxgc_solvent_end_entry.setValidator(QRegularExpressionValidator(QRegularExpression(r'^([0-9]*\.?[0-9]+)?$')))
        self.gcxgc_solvent_end_entry.setToolTip("Ignore peaks whose apex is before this 1st-dimension time (min). Blank = none.")
        pk.addRow("Solvent Cut-off (min):", self.gcxgc_solvent_end_entry)
        self.gcxgc_use_dbc = QCheckBox("Use DBC spectra (.dbc.lsc) for MS search")
        self.gcxgc_use_dbc.setChecked(g.use_dbc_spectra)
        pk.addRow(self.gcxgc_use_dbc)
        self.gcxgc_apex_window_spin = QSpinBox()
        self.gcxgc_apex_window_spin.setRange(1, 50)     # 0 would search nothing yet flag every apex found
        self.gcxgc_apex_window_spin.setValue(g.apex_window_scans)
        self.gcxgc_apex_window_spin.setToolTip("Search ± this many TOF scans around the peak's mapped apex for its spectrum.")
        pk.addRow("Spectrum Window (scans):", self.gcxgc_apex_window_spin)
        self.gcxgc_offset_mode_combo = QComboBox()
        self.gcxgc_offset_mode_combo.addItem("Auto (fit)", "auto")
        self.gcxgc_offset_mode_combo.addItem("Fixed", "fixed")
        self.gcxgc_offset_mode_combo.setCurrentIndex(self.gcxgc_offset_mode_combo.findData(g.offset_mode))
        self.gcxgc_offset_spin = QDoubleSpinBox()
        self.gcxgc_offset_spin.setRange(-200.0, 200.0)
        self.gcxgc_offset_spin.setDecimals(2)
        self.gcxgc_offset_spin.setValue(g.offset_scans)
        self.gcxgc_offset_spin.setEnabled(g.offset_mode == "fixed")
        pk.addRow("FID→TOF Offset:", self.gcxgc_offset_mode_combo)
        pk.addRow("Offset (TOF scans):", self.gcxgc_offset_spin)
        self.gcxgc_tic_source_group = QButtonGroup(self)
        self.gcxgc_tic_raw = QRadioButton("Raw (.lsc)")
        self.gcxgc_tic_dbc = QRadioButton("Background Corrected (.dbc.lsc)")
        self.gcxgc_tic_raw.setChecked(True)
        self.gcxgc_tic_dbc.setEnabled(False)
        self.gcxgc_tic_source_group.addButton(self.gcxgc_tic_raw, 0)
        self.gcxgc_tic_source_group.addButton(self.gcxgc_tic_dbc, 1)
        self.gcxgc_tic_source_group.buttonToggled.connect(
            lambda _btn, checked: checked and self.gcxgc_display_changed.emit())
        tic_box = QWidget()
        tic_row = QHBoxLayout(tic_box)
        tic_row.setContentsMargins(0, 0, 0, 0)
        tic_row.setAlignment(Qt.AlignLeft)
        tic_row.addWidget(self.gcxgc_tic_raw)
        tic_row.addWidget(self.gcxgc_tic_dbc)
        pk.addRow("TIC Display:", tic_box)
        self.gcxgc_show_fid = QCheckBox("Show FID")
        self.gcxgc_show_fid.setToolTip("Also show the FID heatmap while detecting on a TIC. View only; "
                                       "not saved in the method.")
        self.gcxgc_show_fid.toggled.connect(lambda _c: self.gcxgc_display_changed.emit())
        pk.addRow(self.gcxgc_show_fid)
        self.gcxgc_export_snr_spin = QDoubleSpinBox()
        self.gcxgc_export_snr_spin.setRange(0.0, 100.0)
        self.gcxgc_export_snr_spin.setDecimals(1)
        self.gcxgc_export_snr_spin.setSingleStep(0.5)
        self.gcxgc_export_snr_spin.setValue(g.export_min_snr)
        self.gcxgc_export_snr_spin.setToolTip(
            "Exported MGF/MSP spectra keep only ions above this signal-to-noise, estimated from each "
            "spectrum's own noise floor. 0 = keep every ion. Does not change the spectrum shown here "
            "or library search. Read when you press Export, so it can be changed after Integrate.")
        pk.addRow("Export Min S/N:", self.gcxgc_export_snr_spin)
        peaks_group.setVisible(False)
        self.section_groups['gcxgc_peaks'] = peaks_group
        self.params_layout.addWidget(peaks_group)

        # Every edit writes back into current_params['gcxgc'] (→ method, dirty flag).
        for w in (self.gcxgc_pm_entry, self.gcxgc_phase_entry, self.gcxgc_prominence_entry,
                  self.gcxgc_solvent_end_entry):
            w.editingFinished.connect(self._on_gcxgc_widget_changed)
        for w in (self.gcxgc_baseline_enabled, self.gcxgc_peaks_enabled, self.gcxgc_use_dbc, self.gcxgc_remove_bands):
            w.toggled.connect(self._on_gcxgc_widget_changed)
        for w in (self.gcxgc_lam_slider, self.gcxgc_diff_order_spin, self.gcxgc_rt2_tolerance_spin,
                  self.gcxgc_min_sub_peaks_spin, self.gcxgc_apex_window_spin, self.gcxgc_offset_spin,
                  self.gcxgc_export_snr_spin):
            w.valueChanged.connect(self._on_gcxgc_widget_changed)
        for w in (self.gcxgc_signal_combo, self.gcxgc_offset_mode_combo):
            w.currentIndexChanged.connect(self._on_gcxgc_widget_changed)
        self.gcxgc_signal_combo.currentIndexChanged.connect(self._update_gcxgc_signal_widgets)
        self.gcxgc_offset_mode_combo.currentIndexChanged.connect(self._update_gcxgc_signal_widgets)
        self._update_gcxgc_signal_widgets()

    def _update_gcxgc_signal_widgets(self, *_):
        """Detecting on a TIC: the FID-only controls do nothing, so disable them; Show FID applies."""
        on_tic = self.gcxgc_signal_combo.currentData() in ("tic", "dbc_tic")
        self.gcxgc_show_fid.setEnabled(on_tic)
        self.gcxgc_remove_bands.setEnabled(on_tic)
        self.gcxgc_offset_mode_combo.setEnabled(not on_tic)
        self.gcxgc_offset_spin.setEnabled(
            not on_tic and self.gcxgc_offset_mode_combo.currentData() == "fixed")
        self.gcxgc_apex_window_spin.setEnabled(not on_tic)

    def _on_gcxgc_widget_changed(self, *_):
        values = self._read_gcxgc_widgets()
        if values == self.current_params.get('gcxgc'):
            return      # e.g. Return / focus-out on an unchanged field: nothing to write, nothing dirty
        self.current_params['gcxgc'] = values
        self.parameters_changed.emit(self.current_params)

    def _read_gcxgc_widgets(self) -> dict:
        def opt_float(edit):
            text = edit.text().strip()
            try:
                return float(text) if text else None
            except ValueError:
                return None
        return GCxGCParams(
            modulation_period_s=opt_float(self.gcxgc_pm_entry),
            phase_shift_s=opt_float(self.gcxgc_phase_entry),
            detection_signal=self.gcxgc_signal_combo.currentData(),
            baseline_enabled=self.gcxgc_baseline_enabled.isChecked(),
            baseline_lambda=float(10 ** self.gcxgc_lam_slider.value()),
            baseline_diff_order=self.gcxgc_diff_order_spin.value(),
            peaks_enabled=self.gcxgc_peaks_enabled.isChecked(),
            min_prominence=opt_float(self.gcxgc_prominence_entry),
            rt2_grouping_tolerance=self.gcxgc_rt2_tolerance_spin.value(),
            min_modulations=self.gcxgc_min_sub_peaks_spin.value(),
            remove_bands=self.gcxgc_remove_bands.isChecked(),
            solvent_end_min=opt_float(self.gcxgc_solvent_end_entry),
            use_dbc_spectra=self.gcxgc_use_dbc.isChecked(),
            apex_window_scans=self.gcxgc_apex_window_spin.value(),
            offset_mode=self.gcxgc_offset_mode_combo.currentData(),
            offset_scans=self.gcxgc_offset_spin.value(),
            export_min_snr=self.gcxgc_export_snr_spin.value(),
        ).model_dump()

    # ── Method file I/O ────────────────────────────────────────────────────

    def apply_method(self, method):
        """Populate parameter widgets from a ChromaMethod (no change signal).

        Rebuilds ``current_params`` from the method's processing-params slice
        and re-initializes the controls. The app guards write-back during this
        via its ``_loading_method`` flag, so no dirty flip occurs.
        """
        self.cancel_open_segment(emit=False)  # no draft leaks into the loaded method
        self.current_params = method.to_gui_params()
        self._reinitialize_controls()

    def _reinitialize_controls(self):
        """Rebuild all parameter widgets from self.current_params.

        Called after loading a method file. Because all _init_*_controls()
        methods read from self.current_params when creating their widgets,
        clearing the layout and re-running them is sufficient to restore
        the full widget state.
        """
        self.cancel_open_segment()
        # Remove all existing widgets from params_layout. Hide each one now: deleteLater() only runs
        # once control returns to an event loop, and until then a removed group stays a visible
        # child of params_widget at its old place (in GCxGC mode the 1D groups showed through).
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        # The app adds this checkbox to the baseline section after an MS baseline
        # correction; it goes with the old section, and the app recreates it on the next one.
        self.use_corrected_ms_data_check = None

        # Rebuild title
        parameters_label = QLabel("Integration Parameters")
        parameters_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.params_layout.addWidget(parameters_label)

        # Rebuild all control sections — must match the order in __init__
        self.section_groups = {}
        self._init_segments_controls()
        self._init_smoothing_controls()
        self._init_baseline_controls()
        self._init_peaks_controls()
        self._init_negative_peaks_controls()
        self._init_shoulder_controls()
        self._init_peak_grouping_controls()
        self._init_tail_skim_controls()
        self._init_gcxgc_controls()
        self._apply_help_icons()
        self.params_layout.addStretch()
        if getattr(self, '_ui_mode', None):
            self.set_mode(self._ui_mode)

    def set_mode(self, ui_mode: str) -> None:
        """Show/hide parameter sections based on the active UI mode.

        In 'gcxgc' mode: hide 1D sections, show gcxgc_layout, gcxgc_baseline and gcxgc_peaks.
        Otherwise: show 1D sections, hide gcxgc sections.
        """
        is_gcxgc = ui_mode == 'gcxgc'
        if is_gcxgc:
            self.cancel_open_segment()
        self._ui_mode = ui_mode
        for key, group in self.section_groups.items():
            visible = is_gcxgc if key.startswith('gcxgc') else not is_gcxgc
            if key in self._segment_hidden:   # segment editor open: applies when it closes
                self._segment_hidden[key] = not visible
            else:
                group.setVisible(visible)

    def get_gcxgc_params(self) -> dict:
        """GCxGC settings in the shape the app and exporter use."""
        g = self._read_gcxgc_widgets()
        prom = g['min_prominence']
        return {
            'pm': g['modulation_period_s'],
            'phase_shift': g['phase_shift_s'],
            'detection_signal': g['detection_signal'],
            'baseline_enabled': g['baseline_enabled'],
            'peaks_enabled': g['peaks_enabled'],
            'lam': g['baseline_lambda'],
            'diff_order': g['baseline_diff_order'],
            'min_height': None,
            'min_prominence': prom if prom and prom > 0 else None,
            'rt2_grouping_tolerance': g['rt2_grouping_tolerance'],
            'min_sub_peaks': g['min_modulations'],
            'remove_bands': g['remove_bands'],
            'solvent_end_min': g['solvent_end_min'],
            'use_dbc': g['use_dbc_spectra'],
            'apex_window': g['apex_window_scans'],
            'offset_mode': g['offset_mode'],
            'offset_scans': g['offset_scans'],
            'export_min_snr': g['export_min_snr'],
            'tic_source': 'dbc' if self.gcxgc_tic_dbc.isChecked() else 'lsc',
        }

    def set_gcxgc_run_info(self, run) -> None:
        """Show the run's file values as placeholders, provenance, and which signals exist.

        Signals the run lacks are disabled in the detection combo but never deselected: the
        selection is the method's choice, so a run must not rewrite it (Integrate falls back to
        FID for that run). Nothing here emits parameters_changed."""
        (fpm, _), (fph, _) = run.file_layout["modulation_period_s"], run.file_layout["phase_shift_s"]
        self.gcxgc_pm_entry.setPlaceholderText(f"{fpm:g}" if fpm is not None else "required")
        self.gcxgc_phase_entry.setPlaceholderText(f"{fph:g}" if fph is not None else "0")
        self.gcxgc_pm_source.setText(self._PROVENANCE_TEXT[run.provenance["modulation_period_s"]])
        self.gcxgc_phase_source.setText(self._PROVENANCE_TEXT[run.provenance["phase_shift_s"]])
        available = set(run.available_signals)
        combo = self.gcxgc_signal_combo
        model = combo.model()
        combo.blockSignals(True)
        try:
            for i in range(combo.count()):
                model.item(i).setEnabled(combo.itemData(i) in available)
        finally:
            combo.blockSignals(False)
        self.set_gcxgc_dbc_available(run.dbc_lsc is not None)

    def set_gcxgc_dbc_available(self, available: bool) -> None:
        """Enable or disable DBC controls based on whether a .dbc.lsc file was found.

        Disabling leaves "Use DBC spectra" checked as the method has it: loading a run never
        rewrites or dirties the method (GCxGCProcessor ignores use_dbc for a run without a
        .dbc.lsc). The TIC display falls back to Raw, which is display state only."""
        self.gcxgc_use_dbc.setEnabled(available)
        self.gcxgc_tic_dbc.setEnabled(available)
        if not available:
            self.gcxgc_use_dbc.setToolTip("No .dbc.lsc file found for this sample; spectra come from the raw .lsc.")
            self.gcxgc_tic_raw.setChecked(True)
            self.gcxgc_tic_dbc.setToolTip("No .dbc.lsc file found for this sample.")
        else:
            self.gcxgc_use_dbc.setToolTip(
                "When checked, spectrum extraction uses the dynamic background-corrected\n"
                "(.dbc.lsc) file for cleaner MS library matches.\n"
                "When unchecked, uses the plain full-rate .lsc file."
            )
            self.gcxgc_tic_dbc.setToolTip(
                "Display the background-corrected TIC heatmap from .dbc.lsc.\n"
                "Provides a cleaner view with chemical noise removed."
            )
