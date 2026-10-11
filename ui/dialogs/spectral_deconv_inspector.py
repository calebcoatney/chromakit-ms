"""Spectral deconvolution chunk inspector dialog.

Shows each peak window's ion peaks, their correlation groups and the rejected
ion peaks, plus the ion traces behind them, with parameter controls for
interactive tuning. Previews run the same ``deconvolve_window`` as the runner.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QRunnable, Slot, QThreadPool
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QSplitter, QWidget,
    QGroupBox, QFormLayout, QLabel, QSpinBox,
    QComboBox, QCheckBox, QPushButton, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSlider, QScrollArea, QFrame,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.spatial import cKDTree

from logic.correlation_grouping import CorrelationDeconvParams, deconvolve_window
from logic.ion_peak_resolver import whole_run_noise_levels
from logic.spectral_deconv_settings import save_spectral_deconv_settings, parse_mz_list
from logic.spectral_deconv_runner import WindowGroupingParams, _group_peaks_into_windows
from logic.ms_time import shifted_xlabels
from ui.dialogs.spectral_deconv_fields import build_setting_boxes, invalid_mz_message


class _PreviewWorkerSignals(QObject):
    finished = Signal(dict)   # result dict passed to _on_preview_finished
    error = Signal(str)


class _PreviewWorker(QRunnable):
    """Background worker: run deconvolve_window() for one window."""

    def __init__(self, xlabels, ms_data, mz_axis, noise_levels, w_start: float, w_end: float,
                 win_peaks: list, deconv_params: CorrelationDeconvParams, top_n: int,
                 ms_time_offset: float = 0.0):
        super().__init__()
        self._xlabels = xlabels          # already shifted by ms_time_offset
        self._ms_data = ms_data
        self._mz_axis = mz_axis
        self._noise_levels = noise_levels
        self._w_start, self._w_end = w_start, w_end
        self._win_peaks = win_peaks
        self._deconv_params = deconv_params
        self._top_n = top_n
        self._ms_time_offset = ms_time_offset
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            mask = (self._xlabels >= self._w_start) & (self._xlabels <= self._w_end)
            rt_window = self._xlabels[mask]
            ms_slice = self._ms_data[mask, :]
            base = {'win_peaks': self._win_peaks, 'top_n': self._top_n, 'w_start': self._w_start,
                    'w_end': self._w_end, 'ms_time_offset': self._ms_time_offset,
                    'rt_window': rt_window}
            if rt_window.size == 0:
                self.signals.finished.emit({**base, 'ion_peaks': [], 'groups': [], 'rejected': [],
                                            'components': [], 'top_ions': [], 'traces': {},
                                            'empty': True})
                return
            components, inter = deconvolve_window(
                rt_window, ms_slice, self._mz_axis, self._noise_levels, self._deconv_params,
                return_intermediates=True)
            col = {float(m): j for j, m in enumerate(self._mz_axis)}
            top_ions = sorted(inter['ion_peaks'], key=lambda p: -p.height)[:self._top_n]
            traces = {id(p): ms_slice[:, col[float(p.mz)]] for p in top_ions}
            self.signals.finished.emit({**base, **inter, 'components': components,
                                        'top_ions': top_ions, 'traces': traces,
                                        'empty': not inter['ion_peaks']})
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class SpectralDeconvInspectorDialog(QDialog):
    """Non-modal dialog for inspecting spectral deconvolution per window."""

    rerun_requested = Signal(object, object)  # (CorrelationDeconvParams, WindowGroupingParams)
    cluster_search_requested = Signal(object, float)  # (spectrum_dict: dict, rt: float)
    apply_offset_requested = Signal(float, str)  # (offset_min, source)
    apply_offset_to_folder_requested = Signal(float, str)  # (offset_min, source)

    def __init__(
        self,
        peaks: list,
        ms_data_path: str,
        deconv_params: CorrelationDeconvParams,
        grouping_params: WindowGroupingParams,
        initial_peak_index: int = 0,
        initial_offset_min: float = 0.0,
        fid_time=None,
        fid_signal=None,
        tic_time=None,
        tic_signal=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectral Deconvolution Inspector")
        self.setMinimumSize(1100, 650)
        self.setModal(False)

        self._peaks = peaks           # live reference — not a copy
        self._ms_data_path = ms_data_path
        self._ms = None               # opened once on first use
        self._ms_data = None          # cached [scans x m/z] array of self._ms
        self._mz_axis = None          # cached m/z of each column
        self._noise_cache: dict = {}  # noise percentile -> whole-run noise levels
        self._windows: list = []      # [(w_start, w_end, [peaks_in_window]), ...]
        self._current_window_idx: int = 0
        self._preview_worker = None
        self._current_preview_offset_min: float = float(initial_offset_min)
        self._offset_source: str = "manual"
        self._last_render_payload: dict | None = None
        self._fid_time = fid_time
        self._fid_signal = fid_signal
        self._tic_time = tic_time
        self._tic_signal = tic_signal

        # Click-to-search spatial index state
        self._last_result: dict | None = None
        self._scatter_coords: np.ndarray | None = None
        self._scatter_cluster_ids: list[int] = []
        self._scatter_tree: cKDTree | None = None
        self._selected_cluster_idx: int | None = None

        # Build UI first, then populate params from arguments
        self._build_ui()
        self._set_offset_slider_value(self._current_preview_offset_min, source="manual")
        self._load_params(deconv_params, grouping_params)
        self._rebuild_windows()

        # Navigate to the window containing initial_peak_index
        target_window = self._window_for_peak_index(initial_peak_index)
        self._navigate_to(target_window, auto_run=True)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_params_panel())
        splitter.addWidget(self._build_plot_panel())
        splitter.setSizes([320, 780])

    def _build_params_panel(self) -> QWidget:
        """Settings in a scroll area (taller than the dialog's minimum), buttons pinned below."""
        settings = QWidget()
        layout = QVBoxLayout(settings)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)

        # MS time offset group
        offset_group = QGroupBox("MS Time Offset")
        offset_layout = QVBoxLayout(offset_group)
        slider_row = QHBoxLayout()
        self._offset_slider = QSlider(Qt.Horizontal)
        self._offset_slider.setMinimum(-500)
        self._offset_slider.setMaximum(500)
        self._offset_slider.setSingleStep(1)
        self._offset_slider.setPageStep(10)
        self._offset_readout = QLabel("+0.0000 min (+0.00 s)")
        self._offset_readout.setMinimumWidth(160)
        slider_row.addWidget(self._offset_slider, 1)
        slider_row.addWidget(self._offset_readout)
        offset_layout.addLayout(slider_row)

        button_row = QHBoxLayout()
        self._offset_auto_btn = QPushButton("Auto")
        self._offset_reset_btn = QPushButton("Reset")
        self._offset_apply_btn = QPushButton("Apply")
        self._offset_apply_btn.setToolTip(
            "Apply this offset to all windows of this .D file and persist it.\n"
            "Slider drags only preview the current window."
        )
        self._offset_apply_folder_btn = QPushButton("Apply to folder...")
        self._offset_apply_folder_btn.setToolTip(
            "Open a dialog to apply this offset to multiple .D files\n"
            "in the same parent folder."
        )
        button_row.addWidget(self._offset_auto_btn)
        button_row.addWidget(self._offset_reset_btn)
        button_row.addStretch(1)
        button_row.addWidget(self._offset_apply_folder_btn)
        button_row.addWidget(self._offset_apply_btn)
        offset_layout.addLayout(button_row)
        layout.addWidget(offset_group)

        self._offset_auto_btn.setEnabled(
            self._fid_time is not None and self._fid_signal is not None
            and self._tic_time is not None and self._tic_signal is not None
        )
        self._offset_slider.valueChanged.connect(self._on_offset_slider_changed)
        self._offset_auto_btn.clicked.connect(self._on_offset_auto_clicked)
        self._offset_reset_btn.clicked.connect(self._on_offset_reset_clicked)
        self._offset_apply_btn.clicked.connect(self._on_offset_apply_clicked)
        self._offset_apply_folder_btn.clicked.connect(
            self._on_offset_apply_folder_clicked
        )

        boxes, w = build_setting_boxes()
        for box in boxes:
            layout.addWidget(box)
        self._gap_spin = w["gap_tolerance"]
        self._padding_spin = w["padding_fraction"]
        self._rt_match_spin = w["rt_match_tolerance"]
        self._max_window_peaks_spin = w["max_window_peaks"]
        self._noise_percentile_spin = w["noise_percentile"]
        self._search_range_spin = w["search_range"]
        self._min_peak_height_spin = w["min_peak_height"]
        self._min_top_edge_ratio_spin = w["min_top_edge_ratio"]
        self._min_points_spin = w["min_points"]
        self._rt_tolerance_spin = w["rt_tolerance"]
        self._min_r_spin = w["min_r"]
        self._min_ions_spin = w["min_ions"]
        self._excluded_mz_edit = w["excluded_mz"]

        # Top N traces + display options
        top_n_layout = QFormLayout()
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 200)
        self._top_n_spin.setValue(20)
        top_n_layout.addRow("Top N ion traces:", self._top_n_spin)
        layout.addLayout(top_n_layout)

        self._normalize_eic_check = QCheckBox("Normalize traces")
        self._normalize_eic_check.setToolTip(
            "Scale each ion trace to its own maximum\n"
            "so shapes can be compared regardless of intensity"
        )
        layout.addWidget(self._normalize_eic_check)

        self._show_rejected_check = QCheckBox("Show rejected traces")
        self._show_rejected_check.setToolTip(
            "Also draw the traces of rejected ion peaks (dashed gray) in the lower plot"
        )
        self._show_rejected_check.toggled.connect(self._on_show_rejected_toggled)
        layout.addWidget(self._show_rejected_check)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(settings)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFocusPolicy(Qt.NoFocus)   # no invisible tab stop; Tab still scrolls to fields
        # Scroll vertically only; keep the panel as wide as its contents need.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(settings.minimumSizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width())

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(scroll, 1)

        # Buttons, outside the scroll area so they are always visible
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        panel_layout.addWidget(self._preview_btn)

        self._apply_btn = QPushButton("Apply to All && Rerun")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        panel_layout.addWidget(self._apply_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        panel_layout.addWidget(self._close_btn)

        return panel

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Navigation row
        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.clicked.connect(self._on_prev)
        nav_layout.addWidget(self._prev_btn)

        self._window_combo = QComboBox()
        self._window_combo.setSizePolicy(
            self._window_combo.sizePolicy().horizontalPolicy(),
            self._window_combo.sizePolicy().verticalPolicy()
        )
        self._window_combo.currentIndexChanged.connect(self._on_window_selected)
        nav_layout.addWidget(self._window_combo, stretch=1)

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        nav_layout.addWidget(self._next_btn)
        layout.addLayout(nav_layout)

        # Matplotlib canvas — 2-row subplot
        self._fig = Figure(figsize=(8, 6), tight_layout=True)
        self._ax_scatter = self._fig.add_subplot(2, 1, 1)
        self._ax_eic = self._fig.add_subplot(2, 1, 2, sharex=self._ax_scatter)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.mpl_connect('button_press_event', self._on_scatter_click)
        layout.addWidget(self._canvas, stretch=1)

        # Status label
        self._status_label = QLabel("Select a window to preview.")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Inline search results panel
        self._results_label = QLabel("Click a group to search")
        self._results_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
        layout.addWidget(self._results_label)

        self._results_tree = QTreeWidget()
        self._results_tree.setHeaderLabels(["Rank", "Compound", "Score"])
        self._results_tree.setRootIsDecorated(False)
        self._results_tree.setMaximumHeight(140)
        self._results_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._results_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._results_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self._results_tree)

        return panel

    # ── Params load/save ───────────────────────────────────────────────────────

    def _load_params(self, dp: CorrelationDeconvParams, gp: WindowGroupingParams):
        self._gap_spin.setValue(gp.gap_tolerance or 0.0)
        self._padding_spin.setValue(gp.padding_fraction)
        self._rt_match_spin.setValue(gp.rt_match_tolerance)
        self._max_window_peaks_spin.setValue(gp.max_window_peaks)

        self._noise_percentile_spin.setValue(dp.noise_percentile)
        self._search_range_spin.setValue(dp.search_range)
        self._min_peak_height_spin.setValue(dp.min_peak_height)
        self._min_top_edge_ratio_spin.setValue(dp.min_top_edge_ratio)
        self._min_points_spin.setValue(dp.min_points)
        self._rt_tolerance_spin.setValue(dp.rt_tolerance)
        self._min_r_spin.setValue(dp.min_r)
        self._min_ions_spin.setValue(dp.min_ions)
        self._excluded_mz_edit.setText(", ".join(f"{m:g}" for m in dp.excluded_mz))

    def _read_grouping_params(self) -> WindowGroupingParams:
        raw_gap = self._gap_spin.value()
        return WindowGroupingParams(
            gap_tolerance=None if raw_gap == 0.0 else raw_gap,
            padding_fraction=self._padding_spin.value(),
            rt_match_tolerance=self._rt_match_spin.value(),
            max_window_peaks=self._max_window_peaks_spin.value(),
        )

    def _read_params(self) -> tuple[CorrelationDeconvParams, WindowGroupingParams]:
        """Widget values as (CorrelationDeconvParams, WindowGroupingParams); ValueError on bad m/z."""
        excluded = parse_mz_list(self._excluded_mz_edit.text(), strict=True)
        gp = self._read_grouping_params()
        dp = CorrelationDeconvParams(
            noise_percentile=self._noise_percentile_spin.value(),
            search_range=self._search_range_spin.value(),
            min_peak_height=self._min_peak_height_spin.value(),
            min_top_edge_ratio=self._min_top_edge_ratio_spin.value(),
            min_points=self._min_points_spin.value(),
            rt_tolerance=self._rt_tolerance_spin.value(),
            min_r=self._min_r_spin.value(),
            min_ions=self._min_ions_spin.value(),
            excluded_mz=excluded,
        )
        return dp, gp

    def _read_params_or_report(self):
        """``_read_params()``, or None: a bad excluded m/z is explained and its field selected."""
        try:
            return self._read_params()
        except ValueError as e:
            self._status_label.setText(invalid_mz_message(e))
            self._excluded_mz_edit.setFocus()
            self._excluded_mz_edit.selectAll()
            return None

    def read_current_params(self) -> tuple[CorrelationDeconvParams, WindowGroupingParams]:
        """Return current deconvolution/grouping params for external rerun requests."""
        return self._read_params()

    def _set_offset_slider_value(self, offset_min: float, source: str) -> None:
        clamped = max(-0.5, min(0.5, float(offset_min)))
        self._offset_source = source
        self._offset_slider.blockSignals(True)
        self._offset_slider.setValue(int(round(clamped * 1000)))
        self._offset_slider.blockSignals(False)
        self._current_preview_offset_min = clamped
        self._offset_readout.setText(f"{clamped:+.4f} min ({clamped * 60:+.2f} s)")

    def _on_offset_slider_changed(self, value: int) -> None:
        offset_min = value / 1000.0
        self._current_preview_offset_min = offset_min
        self._offset_source = "manual"
        self._offset_readout.setText(f"{offset_min:+.4f} min ({offset_min * 60:+.2f} s)")
        self._rebuild_windows()
        if self._last_render_payload is not None:
            self._render_plots(self._last_render_payload)

    def _on_offset_auto_clicked(self) -> None:
        if (self._fid_time is None or self._fid_signal is None
                or self._tic_time is None or self._tic_signal is None):
            self._status_label.setText("Auto offset unavailable: FID/TIC data not loaded.")
            return
        from logic.processor import ChromatogramProcessor
        proc = ChromatogramProcessor()
        _, _, lag_seconds = proc.align_tic_to_fid(
            self._fid_time,
            self._fid_signal,
            self._tic_time,
            self._tic_signal,
            verbose=False,
        )
        # lag_seconds > 0 means FID lags the TIC (Polyarc transit before FID).
        # shifted_xlabels ADDS the offset to MS times, so aligning MS onto the
        # FID axis requires the same sign as the lag, not its negation.
        offset_min = lag_seconds / 60.0
        self._set_offset_slider_value(offset_min, source="auto")
        self._rebuild_windows()
        if self._last_render_payload is not None:
            self._render_plots(self._last_render_payload)

    def _on_offset_reset_clicked(self) -> None:
        self._set_offset_slider_value(0.0, source="manual")
        self._rebuild_windows()
        if self._last_render_payload is not None:
            self._render_plots(self._last_render_payload)

    def _on_offset_apply_clicked(self) -> None:
        # The app reruns with read_current_params(); refuse here on a bad excluded m/z.
        if self._read_params_or_report() is None:
            return
        self.apply_offset_requested.emit(self._current_preview_offset_min, self._offset_source)

    def _on_offset_apply_folder_clicked(self) -> None:
        if self._read_params_or_report() is None:
            return
        self.apply_offset_to_folder_requested.emit(
            self._current_preview_offset_min, self._offset_source
        )

    # ── Window management ──────────────────────────────────────────────────────

    def _rebuild_windows(self):
        """Recompute window list from current peaks + grouping params."""
        import rainbow as rb
        if self._ms is None:
            data_dir = rb.read(self._ms_data_path)
            self._ms = data_dir.get_file('data.ms')
            self._ms_data = np.asarray(self._ms.data, dtype=float)
            self._mz_axis = np.asarray(self._ms.ylabels, dtype=float)
            self._noise_cache = {}

        gp = self._read_grouping_params()   # no excluded-m/z parse: windows never need it
        _xlabels = shifted_xlabels(self._ms, self._current_preview_offset_min)
        rt_min = float(_xlabels[0])
        rt_max = float(_xlabels[-1])
        self._windows = _group_peaks_into_windows(self._peaks, gp, rt_min, rt_max)

        # Clamp current index to new window count, then resync combo + nav buttons.
        # _populate_combo clears the QComboBox (resetting visual selection to index 0)
        # so we must explicitly restore the correct index afterwards.
        n = len(self._windows)
        self._current_window_idx = min(self._current_window_idx, n - 1) if n > 0 else 0
        self._populate_combo()
        self._window_combo.blockSignals(True)
        self._window_combo.setCurrentIndex(self._current_window_idx)
        self._window_combo.blockSignals(False)
        self._prev_btn.setEnabled(self._current_window_idx > 0)
        self._next_btn.setEnabled(self._current_window_idx < n - 1)

    def _noise_levels(self, q: float):
        key = round(float(q), 6)
        if key not in self._noise_cache:
            self._noise_cache[key] = whole_run_noise_levels(self._ms_data, q)
        return self._noise_cache[key]

    def _populate_combo(self):
        self._window_combo.blockSignals(True)
        self._window_combo.clear()
        for i, (w_start, w_end, win_peaks) in enumerate(self._windows):
            peak_rts = ", ".join(f"{p.retention_time:.3f}" for p in win_peaks)
            label = (
                f"Window {i+1} "
                f"(RT {w_start:.2f}–{w_end:.2f}, "
                f"{len(win_peaks)} peak(s): {peak_rts} min)"
            )
            self._window_combo.addItem(label)
        self._window_combo.blockSignals(False)

    def _window_for_peak_index(self, peak_index: int) -> int:
        """Return window index containing the peak at peak_index, or 0."""
        if not self._windows or peak_index < 0 or peak_index >= len(self._peaks):
            return 0
        target_peak = self._peaks[peak_index]
        for i, (_, _, win_peaks) in enumerate(self._windows):
            if target_peak in win_peaks:
                return i
        return 0

    def _navigate_to(self, window_idx: int, auto_run: bool = False):
        if not self._windows:
            return
        window_idx = max(0, min(window_idx, len(self._windows) - 1))
        self._current_window_idx = window_idx
        self._window_combo.blockSignals(True)
        self._window_combo.setCurrentIndex(window_idx)
        self._window_combo.blockSignals(False)
        self._prev_btn.setEnabled(window_idx > 0)
        self._next_btn.setEnabled(window_idx < len(self._windows) - 1)
        if auto_run:
            self._run_preview()

    def navigate_to_peak(self, peak_index: int):
        """Called by ChromaKitApp when user clicks a peak while dialog is open."""
        self._navigate_to(self._window_for_peak_index(peak_index), auto_run=True)

    def refresh_current_window(self):
        """Called by ChromaKitApp after a full rerun to update plots."""
        self._run_preview()

    def set_controls_enabled(self, enabled: bool):
        """Enable/disable all interactive controls (during full rerun)."""
        for w in (self._preview_btn, self._apply_btn, self._prev_btn,
                  self._next_btn, self._window_combo,
                  self._gap_spin, self._padding_spin, self._rt_match_spin,
                  self._max_window_peaks_spin,
                  self._noise_percentile_spin, self._search_range_spin,
                  self._min_peak_height_spin, self._min_top_edge_ratio_spin,
                  self._min_points_spin, self._rt_tolerance_spin, self._min_r_spin,
                  self._min_ions_spin, self._excluded_mz_edit, self._top_n_spin,
                  self._normalize_eic_check, self._show_rejected_check):
            w.setEnabled(enabled)
        for w in (self._offset_slider, self._offset_auto_btn, self._offset_reset_btn,
                  self._offset_apply_btn, self._offset_apply_folder_btn):
            w.setEnabled(enabled)

    # ── Navigation slots ───────────────────────────────────────────────────────

    def _on_prev(self):
        self._navigate_to(self._current_window_idx - 1, auto_run=True)

    def _on_next(self):
        self._navigate_to(self._current_window_idx + 1, auto_run=True)

    def _on_window_selected(self, idx: int):
        if idx != self._current_window_idx:
            self._navigate_to(idx, auto_run=True)

    # ── Preview / Apply ────────────────────────────────────────────────────────

    def _on_preview_clicked(self):
        if self._read_params_or_report() is None:
            return
        self._rebuild_windows()   # re-group in case grouping params changed
        self._run_preview()

    def _on_apply_clicked(self):
        params = self._read_params_or_report()
        if params is None:
            return
        dp, gp = params
        save_spectral_deconv_settings(QSettings("CalebCoatney", "ChromaKit"), dp, gp)
        self.rerun_requested.emit(dp, gp)

    def _on_show_rejected_toggled(self, _checked: bool):
        if self._last_render_payload is None:
            return
        self._render_plots(self._last_render_payload)
        if self._selected_cluster_idx is not None:
            self._highlight_cluster(self._selected_cluster_idx)

    def _run_preview(self):
        """Launch background worker for the current window."""
        if not self._windows:
            return
        if self._preview_worker is not None:
            return  # already running
        if self._ms is None:
            self._status_label.setText("Error: MS data file could not be opened.")
            return
        params = self._read_params_or_report()
        if params is None:
            return
        dp, _ = params

        self.set_controls_enabled(False)
        self._status_label.setText("Running deconvolution…")

        w_start, w_end, win_peaks = self._windows[self._current_window_idx]

        self._preview_worker = _PreviewWorker(
            xlabels=np.asarray(shifted_xlabels(self._ms, self._current_preview_offset_min),
                               dtype=float),
            ms_data=self._ms_data,
            mz_axis=self._mz_axis,
            noise_levels=self._noise_levels(dp.noise_percentile),
            w_start=w_start,
            w_end=w_end,
            win_peaks=win_peaks,
            deconv_params=dp,
            top_n=self._top_n_spin.value(),
            ms_time_offset=self._current_preview_offset_min,
        )
        self._preview_worker.signals.finished.connect(self._on_preview_finished)
        self._preview_worker.signals.error.connect(self._on_preview_error)
        QThreadPool.globalInstance().start(self._preview_worker)

    def _on_preview_finished(self, result: dict):
        self._preview_worker = None
        self._last_result = result
        self._last_render_payload = result
        self._selected_cluster_idx = None
        self._results_tree.clear()
        self._results_label.setText("Click a group to search")
        self.set_controls_enabled(True)
        self._render_plots(result)

    def _on_preview_error(self, msg: str):
        self._preview_worker = None
        self.set_controls_enabled(True)
        self._status_label.setText(f"Error: {msg}")

    # ── Click-to-search ────────────────────────────────────────────────────────

    def _spectrum_for_group(self, group_idx: int):
        """(spectrum, RT at the current preview offset) of group ``group_idx``, or None."""
        if self._last_result is None:
            return None
        comps = self._last_result.get('components', [])
        if not (0 <= group_idx < len(comps)) or not comps[group_idx].spectrum:
            return None
        offset = self._current_preview_offset_min - float(self._last_result.get('ms_time_offset', 0.0))
        return comps[group_idx].spectrum, comps[group_idx].rt + offset

    def _on_scatter_click(self, event):
        """Handle click on the scatter plot to select a group for search."""
        if event.inaxes is not self._ax_scatter:
            return
        if self._preview_worker is not None:
            return
        if self._scatter_tree is None or self._last_result is None:
            return

        # Convert a pixel tolerance (15 px) to data-coordinate distance
        inv = self._ax_scatter.transData.inverted()
        display_pt = self._ax_scatter.transData.transform([event.xdata, event.ydata])
        offset_pt = inv.transform([display_pt[0] + 15, display_pt[1] + 15])
        data_pt = np.array([event.xdata, event.ydata])
        tol = np.linalg.norm(offset_pt - data_pt)

        dist, idx = self._scatter_tree.query(data_pt)
        if dist > tol:
            self._clear_cluster_selection()
            return

        cluster_id = self._scatter_cluster_ids[idx]

        if cluster_id == -1:
            # Clear first: clearing re-renders, which rewrites the status line.
            self._clear_cluster_selection()
            self._status_label.setText("Rejected ion peak — not in any spectrum")
            return

        if cluster_id == self._selected_cluster_idx:
            self._clear_cluster_selection()
            return

        self._selected_cluster_idx = cluster_id
        self._highlight_cluster(cluster_id)

        found = self._spectrum_for_group(cluster_id)
        if found is None:
            self._status_label.setText(f"Group {cluster_id}: no spectrum available")
            return
        spec, rt = found
        self._status_label.setText(
            f"Group {cluster_id} selected — RT {rt:.3f} min, {len(spec)} ions — searching…"
        )
        self.cluster_search_requested.emit(spec, rt)

    def _highlight_cluster(self, cluster_idx: int):
        """Re-render the top plot with the selected group highlighted, the others dimmed."""
        if self._last_result is None:
            return
        offset = self._current_preview_offset_min - float(self._last_result.get('ms_time_offset', 0.0))
        self._ax_scatter.clear()
        self._draw_group_scatter(self._last_result, offset, selected=cluster_idx)
        self._canvas.draw()

    def _clear_cluster_selection(self):
        """Remove group highlight and restore normal render."""
        if self._selected_cluster_idx is None:
            return
        self._selected_cluster_idx = None
        if self._last_result is not None:
            self._render_plots(self._last_result)

    def show_search_results(self, results: list, rt: float):
        """Display MS library search results in the inline panel.

        Args:
            results: list of (compound_name, match_score) tuples from ms_toolkit.
            rt: Component retention time in minutes.
        """
        self._results_tree.clear()

        if not results:
            self._results_label.setText(f"No matches for component at RT = {rt:.3f} min")
            return

        self._results_label.setText(f"Search results — component at RT = {rt:.3f} min")

        for rank, (name, score) in enumerate(results, start=1):
            item = QTreeWidgetItem([str(rank), str(name), f"{score:.4f}"])
            item.setTextAlignment(0, Qt.AlignCenter)
            item.setTextAlignment(2, Qt.AlignCenter)
            self._results_tree.addTopLevelItem(item)

        self._status_label.setText(
            f"Group search complete — {len(results)} result(s) at RT = {rt:.3f} min"
        )

    # Scatter becomes unreadable beyond this many groups; show guidance instead.
    _MAX_RENDERABLE_CLUSTERS = 100

    @staticmethod
    def _group_colors(n_groups: int) -> dict:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab10')
        return {i: cmap(i % 10) for i in range(n_groups)}

    def _draw_group_scatter(self, result: dict, offset: float, selected: int | None = None):
        """Draw the top plot: FID span and lines, ion groups (lead = star), hollow rejected ions.

        With ``selected``, that group is highlighted and the others dimmed.
        """
        ax = self._ax_scatter
        groups = result['groups']
        rejected = result['rejected']
        fid_rts = [p.retention_time for p in result['win_peaks']]
        cluster_colors = self._group_colors(len(groups))
        dim = selected is not None

        # Shade FID peak span to distinguish "in-window" from padding zone
        if fid_rts:
            fid_min, fid_max = min(fid_rts), max(fid_rts)
            # Add a half-peak-width margin so isolated peaks get a visible band
            half_gap = (result['w_end'] - result['w_start']) * 0.05
            ax.axvspan(
                fid_min - half_gap, fid_max + half_gap,
                alpha=0.08, color='steelblue', zorder=1, label='FID span'
            )

        # Rejected ion peaks: hollow gray markers
        for reason, marker, label in (("shape", 'o', 'rejected: shape'),
                                      ("small_group", 's', 'rejected: small group')):
            pts = [r.peak for r in rejected if r.reason == reason]
            if pts:
                ax.scatter(
                    [p.rt_apex + offset for p in pts], [p.mz for p in pts],
                    marker=marker, facecolors='none', edgecolors='gray', s=24,
                    alpha=0.3 if dim else None, zorder=2, label=label,
                )

        # Groups: members filled, lead ion as a star
        for i, g in enumerate(groups):
            color = cluster_colors[i]
            rts = [p.rt_apex + offset for p in g.members]
            mzs = [p.mz for p in g.members]
            lead_xy = ([g.lead.rt_apex + offset], [g.lead.mz])
            if not dim:
                ax.scatter(rts, mzs, color=color, s=18, zorder=3)
                ax.scatter(*lead_xy, color=color, marker='*', s=80, zorder=4)
            elif i == selected:
                ax.scatter(rts, mzs, color=color, s=30, edgecolors='black',
                           linewidths=0.8, zorder=5)
                ax.scatter(*lead_xy, color=color, marker='*', s=120,
                           edgecolors='black', linewidths=0.8, zorder=6)
            else:
                ax.scatter(rts, mzs, color=color, s=18, alpha=0.25, edgecolors='none', zorder=3)
                ax.scatter(*lead_xy, color=color, marker='*', s=50, alpha=0.25, zorder=4)

        # FID peak RT lines with RT labels
        for rt in fid_rts:
            ax.axvline(rt, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)
            ax.text(
                rt, 1.01, f"{rt:.3f}",
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7, color='steelblue',
            )

        if rejected:
            ax.legend(fontsize=7)

        title = f"Ion groups — {len(groups)} group(s), {len(rejected)} rejected ion peak(s)"
        if dim:
            title += f" — group {selected} selected"
        ax.set_title(title)
        ax.set_ylabel("m/z")

    def _render_plots(self, result: dict):
        """Render ion groups (top) and ion traces (bottom) from a preview result."""
        self._last_render_payload = result
        offset = self._current_preview_offset_min - float(result.get('ms_time_offset', 0.0))
        self._ax_scatter.clear()
        self._ax_eic.clear()

        win_peaks = result['win_peaks']
        fid_rts = [p.retention_time for p in win_peaks]

        if result.get('empty'):
            self._ax_scatter.set_title("No ion peaks in this window")
            self._ax_eic.set_title("(check Min peak height / Noise percentile)")
            self._canvas.draw()
            self._scatter_coords = None
            self._scatter_cluster_ids = []
            self._scatter_tree = None
            self._status_label.setText(
                "No ion peaks in this window (check Min peak height / Noise percentile)"
            )
            return

        ion_peaks = result['ion_peaks']
        groups = result['groups']
        rejected = result['rejected']
        components = result['components']
        cluster_colors = self._group_colors(len(groups))
        n_groups = len(groups)

        # ── Top subplot: ion groups ────────────────────────────────────────────
        if n_groups > self._MAX_RENDERABLE_CLUSTERS:
            self._ax_scatter.text(
                0.5, 0.5,
                f"{n_groups} groups from {len(ion_peaks)} ion peaks\n\n"
                "Too many groups to display usefully.\n\n"
                "Try ↑ Min shape correlation or ↑ Min ions",
                transform=self._ax_scatter.transAxes,
                ha='center', va='center', fontsize=10, color='#cc3300',
                multialignment='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='#fff5f5', ec='#cc3300', alpha=0.9),
            )
            self._ax_scatter.set_title(
                f"Ion groups — {n_groups} group(s), {len(rejected)} rejected ion peak(s)"
                " (too many to render)"
            )
            self._ax_scatter.set_ylabel("m/z")
            self._scatter_coords = None
            self._scatter_cluster_ids = []
            self._scatter_tree = None
        else:
            self._draw_group_scatter(result, offset)

            # Build spatial index for click-to-search: rejected ions -1, members their group
            all_coords = [[r.peak.rt_apex + offset, r.peak.mz] for r in rejected]
            all_cluster_ids = [-1] * len(rejected)
            for gi, g in enumerate(groups):
                for p in g.members:
                    all_coords.append([p.rt_apex + offset, p.mz])
                    all_cluster_ids.append(gi)
            if all_coords:
                self._scatter_coords = np.array(all_coords)
                self._scatter_cluster_ids = all_cluster_ids
                self._scatter_tree = cKDTree(self._scatter_coords)
            else:
                self._scatter_coords = None
                self._scatter_cluster_ids = []
                self._scatter_tree = None

        # ── Bottom subplot: ion traces ─────────────────────────────────────────
        normalize = self._normalize_eic_check.isChecked()
        member_of = {id(m): (gi, g.r[k], m is g.lead)
                     for gi, g in enumerate(groups) for k, m in enumerate(g.members)}
        rejected_ids = {id(r.peak) for r in rejected}
        rt_window = result['rt_window'] + offset
        n_labelled = 0
        for p in result['top_ions']:
            y = np.asarray(result['traces'][id(p)], dtype=float)
            if normalize and y.max() > 0:
                y = y / y.max()
            if id(p) in member_of:
                gi, r, is_lead = member_of[id(p)]
                color = cluster_colors[gi]
                self._ax_eic.plot(rt_window, y, color=color, linewidth=2.5 if is_lead else 1.0,
                                  alpha=0.85, label=f"{p.mz:g} (r={r:.2f})")
                self._ax_eic.axvspan(p.rt_array[0] + offset, p.rt_array[-1] + offset,
                                     color=color, alpha=0.06, zorder=0)
                n_labelled += 1
            elif id(p) in rejected_ids and self._show_rejected_check.isChecked():
                self._ax_eic.plot(rt_window, y, color='gray', linestyle='--', linewidth=0.8,
                                  alpha=0.7)

        for rt in fid_rts:
            self._ax_eic.axvline(rt, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)

        shown = len(result['top_ions'])
        total = len(ion_peaks)
        self._ax_eic.set_xlabel("Retention Time (min)")
        self._ax_eic.set_ylabel("Norm. Intensity" if normalize else "Intensity")
        self._ax_eic.set_title(
            f"Ion traces (top {shown} of {total} ion peaks"
            + (", normalized)" if normalize else ")")
        )
        if 0 < n_labelled <= 12:
            self._ax_eic.legend(fontsize=7)

        self._canvas.draw()

        # ── Status line ────────────────────────────────────────────────────────
        n_comp = len(components)
        matched = [
            p for p in win_peaks
            if getattr(p, 'deconvolved_spectrum', None) is not None
        ]
        if n_comp == 0:
            status = "0 components found (all peaks filtered)"
        else:
            n_matched = len(matched)
            matched_rts = ", ".join(f"{p.retention_time:.3f}" for p in matched)
            unmatched = len(win_peaks) - n_matched
            status = (
                f"{n_comp} component(s) found — "
                f"{n_matched} matched to FID peak(s)"
                + (f" (RT {matched_rts} min)" if matched_rts else "")
                + (f" — {unmatched} unmatched" if unmatched > 0 else "")
            )

        if len(rejected) / max(len(ion_peaks), 1) > 0.4:
            status += (
                f"  ⚠ Many rejected ion peaks ({len(rejected)}/{len(ion_peaks)}) — "
                "try ↓ Min shape correlation or ↑ Search range"
            )

        self._status_label.setText(status)

    def closeEvent(self, event):
        self._ms = None  # release rainbow data and the cached arrays
        self._ms_data = None
        self._mz_axis = None
        self._noise_cache = {}
        super().closeEvent(event)
