"""Spectral deconvolution chunk inspector dialog.

Shows DBSCAN RT clustering and EIC traces for each peak window,
with parameter controls for interactive tuning.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QRunnable, Slot, QThreadPool
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QSplitter, QWidget,
    QGroupBox, QFormLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QProgressBar,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.spatial import cKDTree

from logic.spectral_deconvolution import DeconvolutionParams, deconvolve
from logic.spectral_deconv_runner import WindowGroupingParams, _group_peaks_into_windows
from logic.eic_extractor import extract_eic_peaks


class _PreviewWorkerSignals(QObject):
    finished = Signal(dict)   # result dict passed to _on_preview_finished
    error = Signal(str)


class _PreviewWorker(QRunnable):
    """Background worker: extract EICs + run deconvolve() for one window."""

    def __init__(self, ms, w_start: float, w_end: float,
                 win_peaks: list, deconv_params: DeconvolutionParams,
                 top_n: int):
        super().__init__()
        self._ms = ms
        self._w_start = w_start
        self._w_end = w_end
        self._win_peaks = win_peaks
        self._deconv_params = deconv_params
        self._top_n = top_n
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            eic_peaks = extract_eic_peaks(
                self._ms,
                t_start=self._w_start,
                t_end=self._w_end,
                min_intensity=self._deconv_params.min_cluster_intensity,
                min_prominence=self._deconv_params.min_eic_prominence,
            )

            if not eic_peaks:
                self.signals.finished.emit({
                    'eic_peaks': [],
                    'components': [],
                    'intermediates': {'rt_clusters': [], 'noise_peaks': [], 'model_peaks': []},
                    'win_peaks': self._win_peaks,
                    'top_n': self._top_n,
                    'w_start': self._w_start,
                    'w_end': self._w_end,
                    'empty': True,
                })
                return

            # Sort by descending max intensity; keep top_n for EIC plot
            eic_sorted = sorted(eic_peaks, key=lambda p: -p.intensity_array.max())
            top_eic = eic_sorted[:self._top_n]

            # Run deconvolution on all EIC peaks (not just top N) for accuracy
            components, intermediates = deconvolve(
                eic_peaks, self._deconv_params, return_intermediates=True
            )

            self.signals.finished.emit({
                'eic_peaks': eic_peaks,
                'top_eic': top_eic,
                'components': components,
                'intermediates': intermediates,
                'win_peaks': self._win_peaks,
                'top_n': self._top_n,
                'w_start': self._w_start,
                'w_end': self._w_end,
                'empty': False,
            })
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class SpectralDeconvInspectorDialog(QDialog):
    """Non-modal dialog for inspecting ADAP-GC deconvolution per window."""

    rerun_requested = Signal(object, object)  # (DeconvolutionParams, WindowGroupingParams)
    cluster_search_requested = Signal(object, float)  # (spectrum_dict: dict, rt: float)

    def __init__(
        self,
        peaks: list,
        ms_data_path: str,
        deconv_params: DeconvolutionParams,
        grouping_params: WindowGroupingParams,
        initial_peak_index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectral Deconvolution Inspector")
        self.setMinimumSize(1100, 650)
        self.setModal(False)

        self._peaks = peaks           # live reference — not a copy
        self._ms_data_path = ms_data_path
        self._ms = None               # opened once on first use
        self._windows: list = []      # [(w_start, w_end, [peaks_in_window]), ...]
        self._current_window_idx: int = 0
        self._preview_worker = None

        # Click-to-search spatial index state
        self._last_result: dict | None = None
        self._scatter_coords: np.ndarray | None = None
        self._scatter_cluster_ids: list[int] = []
        self._scatter_tree: cKDTree | None = None
        self._selected_cluster_idx: int | None = None

        # Build UI first, then populate params from arguments
        self._build_ui()
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
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        # Window Grouping group
        wg_group = QGroupBox("Peak Window Grouping")
        wg_form = QFormLayout(wg_group)

        self._gap_spin = QDoubleSpinBox()
        self._gap_spin.setRange(0.0, 10.0)
        self._gap_spin.setDecimals(3)
        self._gap_spin.setSingleStep(0.01)
        self._gap_spin.setSpecialValueText("Auto")
        self._gap_spin.setToolTip("0 = auto (0.5× median FID peak width)")
        wg_form.addRow("Gap tolerance (min):", self._gap_spin)

        self._padding_spin = QDoubleSpinBox()
        self._padding_spin.setRange(0.0, 5.0)
        self._padding_spin.setDecimals(2)
        wg_form.addRow("Padding fraction:", self._padding_spin)

        self._rt_match_spin = QDoubleSpinBox()
        self._rt_match_spin.setRange(0.001, 1.0)
        self._rt_match_spin.setDecimals(3)
        self._rt_match_spin.setSingleStep(0.005)
        self._rt_match_spin.setToolTip("Max RT gap (min) between FID peak and MS component")
        wg_form.addRow("RT match tolerance (min):", self._rt_match_spin)

        self._max_window_peaks_spin = QSpinBox()
        self._max_window_peaks_spin.setRange(0, 100)
        self._max_window_peaks_spin.setSpecialValueText("Unlimited")
        self._max_window_peaks_spin.setToolTip(
            "Force a window split when this many FID peaks accumulate.\n"
            "Splits at the largest internal RT gap. 0 = unlimited."
        )
        wg_form.addRow("Max peaks per window:", self._max_window_peaks_spin)

        layout.addWidget(wg_group)

        # ADAP-GC Parameters group
        adap_group = QGroupBox("ADAP-GC Parameters")
        adap_form = QFormLayout(adap_group)

        self._min_dist_spin = QDoubleSpinBox()
        self._min_dist_spin.setRange(0.0001, 1.0)
        self._min_dist_spin.setDecimals(4)
        self._min_dist_spin.setSingleStep(0.001)
        self._min_dist_spin.setToolTip("DBSCAN eps — max RT gap within a cluster (min)")
        adap_form.addRow("Min cluster distance (min):", self._min_dist_spin)

        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(1, 50)
        self._min_size_spin.setToolTip("DBSCAN min_samples — minimum EIC peaks per cluster")
        adap_form.addRow("Min cluster size:", self._min_size_spin)

        self._min_intensity_spin = QDoubleSpinBox()
        self._min_intensity_spin.setRange(0.0, 1e8)
        self._min_intensity_spin.setDecimals(0)
        self._min_intensity_spin.setSingleStep(100.0)
        self._min_intensity_spin.setToolTip("Drop clusters below this max intensity (counts)")
        adap_form.addRow("Min cluster intensity:", self._min_intensity_spin)

        self._prominence_spin = QDoubleSpinBox()
        self._prominence_spin.setRange(0.0, 1e8)
        self._prominence_spin.setDecimals(0)
        self._prominence_spin.setSingleStep(500.0)
        self._prominence_spin.setSpecialValueText("Off")
        self._prominence_spin.setToolTip(
            "Min prominence (counts) for EIC peak detection.\n"
            "Rejects noise bumps that don't rise meaningfully above\n"
            "their local baseline. 0 = disabled. Recommended: 1000-5000."
        )
        adap_form.addRow("Min EIC prominence:", self._prominence_spin)

        self._shape_sim_spin = QDoubleSpinBox()
        self._shape_sim_spin.setRange(0.0, 90.0)
        self._shape_sim_spin.setDecimals(1)
        self._shape_sim_spin.setToolTip("Max angle (°) between EIC shapes in the same cluster")
        adap_form.addRow("Shape similarity (°):", self._shape_sim_spin)

        self._model_peak_combo = QComboBox()
        self._model_peak_combo.addItems(["sharpness", "intensity", "mz"])
        self._model_peak_combo.setToolTip("Criterion for selecting the representative EIC peak")
        adap_form.addRow("Model peak choice:", self._model_peak_combo)

        self._excluded_mz_edit = QLineEdit()
        self._excluded_mz_edit.setPlaceholderText("e.g. 73, 147, 221")
        self._excluded_mz_edit.setToolTip("Comma-separated m/z values to exclude (e.g. TMS artifacts)")
        adap_form.addRow("Excluded m/z:", self._excluded_mz_edit)
        layout.addWidget(adap_group)

        # Top N traces + display options
        top_n_layout = QFormLayout()
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 200)
        self._top_n_spin.setValue(20)
        top_n_layout.addRow("Top N EIC traces:", self._top_n_spin)
        layout.addLayout(top_n_layout)

        self._normalize_eic_check = QCheckBox("Normalize EIC traces")
        self._normalize_eic_check.setToolTip(
            "Scale each EIC trace to its own maximum\n"
            "so shapes can be compared regardless of intensity"
        )
        layout.addWidget(self._normalize_eic_check)

        layout.addStretch()

        # Buttons
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        layout.addWidget(self._preview_btn)

        self._apply_btn = QPushButton("Apply to All && Rerun")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self._apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

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

        return panel

    # ── Params load/save ───────────────────────────────────────────────────────

    def _load_params(self, dp: DeconvolutionParams, gp: WindowGroupingParams):
        self._gap_spin.setValue(gp.gap_tolerance or 0.0)
        self._padding_spin.setValue(gp.padding_fraction)
        self._rt_match_spin.setValue(gp.rt_match_tolerance)
        self._max_window_peaks_spin.setValue(gp.max_window_peaks)

        self._min_dist_spin.setValue(dp.min_cluster_distance)
        self._min_size_spin.setValue(dp.min_cluster_size)
        self._min_intensity_spin.setValue(dp.min_cluster_intensity)
        self._prominence_spin.setValue(dp.min_eic_prominence)
        self._shape_sim_spin.setValue(dp.shape_sim_threshold)
        idx = self._model_peak_combo.findText(dp.model_peak_choice)
        if idx >= 0:
            self._model_peak_combo.setCurrentIndex(idx)
        self._excluded_mz_edit.setText(
            ", ".join(str(m) for m in dp.excluded_mz)
        )

    def _read_params(self) -> tuple[DeconvolutionParams, WindowGroupingParams]:
        raw_gap = self._gap_spin.value()
        gp = WindowGroupingParams(
            gap_tolerance=None if raw_gap == 0.0 else raw_gap,
            padding_fraction=self._padding_spin.value(),
            rt_match_tolerance=self._rt_match_spin.value(),
            max_window_peaks=self._max_window_peaks_spin.value(),
        )
        excluded = [
            float(x.strip())
            for x in self._excluded_mz_edit.text().split(",")
            if x.strip()
        ]
        dp = DeconvolutionParams(
            min_cluster_distance=self._min_dist_spin.value(),
            min_cluster_size=self._min_size_spin.value(),
            min_cluster_intensity=self._min_intensity_spin.value(),
            min_eic_prominence=self._prominence_spin.value(),
            shape_sim_threshold=self._shape_sim_spin.value(),
            model_peak_choice=self._model_peak_combo.currentText(),
            excluded_mz=excluded,
        )
        return dp, gp

    # ── Window management ──────────────────────────────────────────────────────

    def _rebuild_windows(self):
        """Recompute window list from current peaks + grouping params."""
        import rainbow as rb
        if self._ms is None:
            data_dir = rb.read(self._ms_data_path)
            self._ms = data_dir.get_file('data.ms')

        _, gp = self._read_params()
        rt_min = float(self._ms.xlabels[0])
        rt_max = float(self._ms.xlabels[-1])
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
                  self._min_dist_spin, self._min_size_spin, self._min_intensity_spin, self._prominence_spin,
                  self._shape_sim_spin, self._model_peak_combo,
                  self._excluded_mz_edit, self._top_n_spin,
                  self._normalize_eic_check):
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
        self._rebuild_windows()   # re-group in case grouping params changed
        self._run_preview()

    def _on_apply_clicked(self):
        dp, gp = self._read_params()
        from PySide6.QtCore import QSettings
        s = QSettings("CalebCoatney", "ChromaKit")
        s.setValue("ms_spectral_deconv/min_cluster_distance", dp.min_cluster_distance)
        s.setValue("ms_spectral_deconv/min_cluster_size", dp.min_cluster_size)
        s.setValue("ms_spectral_deconv/min_cluster_intensity", dp.min_cluster_intensity)
        s.setValue("ms_spectral_deconv/min_eic_prominence", dp.min_eic_prominence)
        s.setValue("ms_spectral_deconv/shape_sim_threshold", dp.shape_sim_threshold)
        s.setValue("ms_spectral_deconv/model_peak_choice", dp.model_peak_choice)
        s.setValue("ms_spectral_deconv/excluded_mz",
                   ", ".join(str(m) for m in dp.excluded_mz))
        s.setValue("ms_spectral_deconv/gap_tolerance", self._gap_spin.value())
        s.setValue("ms_spectral_deconv/padding_fraction", gp.padding_fraction)
        s.setValue("ms_spectral_deconv/rt_match_tolerance", gp.rt_match_tolerance)
        s.setValue("ms_spectral_deconv/max_window_peaks", gp.max_window_peaks)
        
        self.rerun_requested.emit(dp, gp)

    def _run_preview(self):
        """Launch background worker for the current window."""
        if not self._windows:
            return
        if self._preview_worker is not None:
            return  # already running
        if self._ms is None:
            self._status_label.setText("Error: MS data file could not be opened.")
            return

        self.set_controls_enabled(False)
        self._status_label.setText("Running deconvolution…")

        w_start, w_end, win_peaks = self._windows[self._current_window_idx]
        dp, _ = self._read_params()

        self._preview_worker = _PreviewWorker(
            ms=self._ms,
            w_start=w_start,
            w_end=w_end,
            win_peaks=win_peaks,
            deconv_params=dp,
            top_n=self._top_n_spin.value(),
        )
        self._preview_worker.signals.finished.connect(self._on_preview_finished)
        self._preview_worker.signals.error.connect(self._on_preview_error)
        QThreadPool.globalInstance().start(self._preview_worker)

    def _on_preview_finished(self, result: dict):
        self._preview_worker = None
        self._last_result = result
        self._selected_cluster_idx = None
        self.set_controls_enabled(True)
        self._render_plots(result)

    def _on_preview_error(self, msg: str):
        self._preview_worker = None
        self.set_controls_enabled(True)
        self._status_label.setText(f"Error: {msg}")

    # ── Click-to-search ────────────────────────────────────────────────────────

    def _on_scatter_click(self, event):
        """Handle click on the scatter plot to select a cluster for search."""
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
            self._status_label.setText("Noise point — not assigned to any component")
            self._clear_cluster_selection()
            return

        if cluster_id == self._selected_cluster_idx:
            self._clear_cluster_selection()
            return

        self._selected_cluster_idx = cluster_id
        self._highlight_cluster(cluster_id)

        # Find the component for this cluster via its model peak
        components = self._last_result.get('components', [])
        rt_clusters = self._last_result['intermediates']['rt_clusters']
        model_peaks = self._last_result['intermediates']['model_peaks']
        model_peak_ids = {id(mp) for mp in model_peaks}

        cluster_peaks = rt_clusters[cluster_id]
        cluster_model = None
        for peak in cluster_peaks:
            if id(peak) in model_peak_ids:
                cluster_model = peak
                break

        if cluster_model is None:
            self._status_label.setText(
                f"Cluster {cluster_id} has no model peak — cannot search"
            )
            return

        target_component = None
        for comp in components:
            if abs(comp.rt - cluster_model.rt_apex) < 1e-6:
                target_component = comp
                break

        if target_component is None or not target_component.spectrum:
            self._status_label.setText(
                f"Cluster {cluster_id}: no spectrum available"
            )
            return

        self._status_label.setText(
            f"Cluster {cluster_id} selected — RT {target_component.rt:.3f} min, "
            f"{len(target_component.spectrum)} m/z ions — searching…"
        )
        self.cluster_search_requested.emit(target_component.spectrum, target_component.rt)

    def _highlight_cluster(self, cluster_idx: int):
        """Re-render scatter with the selected cluster highlighted."""
        if self._last_result is None:
            return

        intermediates = self._last_result['intermediates']
        rt_clusters = intermediates['rt_clusters']
        noise_peaks = intermediates['noise_peaks']
        model_peaks = intermediates['model_peaks']
        model_peak_ids = {id(mp) for mp in model_peaks}
        win_peaks = self._last_result['win_peaks']
        fid_rts = [p.retention_time for p in win_peaks]

        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab10')

        self._ax_scatter.clear()

        # FID span shading
        if fid_rts:
            fid_min, fid_max = min(fid_rts), max(fid_rts)
            half_gap = (self._last_result['w_end'] - self._last_result['w_start']) * 0.05
            self._ax_scatter.axvspan(
                fid_min - half_gap, fid_max + half_gap,
                alpha=0.08, color='steelblue', zorder=1,
            )

        # Noise points — always dimmed
        if noise_peaks:
            self._ax_scatter.scatter(
                [p.rt_apex for p in noise_peaks],
                [p.mz for p in noise_peaks],
                color='gray', s=10, alpha=0.15, zorder=2,
            )

        # Clustered points — selected vs dimmed
        for ci, cluster in enumerate(rt_clusters):
            color = cmap(ci % 10)
            rts = [p.rt_apex for p in cluster]
            mzs = [p.mz for p in cluster]
            is_selected = (ci == cluster_idx)

            self._ax_scatter.scatter(
                rts, mzs,
                color=color,
                s=30 if is_selected else 18,
                alpha=1.0 if is_selected else 0.25,
                edgecolors='black' if is_selected else 'none',
                linewidths=0.8 if is_selected else 0,
                zorder=5 if is_selected else 3,
            )

            if is_selected:
                for peak in cluster:
                    if id(peak) in model_peak_ids:
                        self._ax_scatter.scatter(
                            [peak.rt_apex], [peak.mz],
                            color=color, marker='*', s=120,
                            edgecolors='black', linewidths=0.8, zorder=6,
                        )
            else:
                for peak in cluster:
                    if id(peak) in model_peak_ids:
                        self._ax_scatter.scatter(
                            [peak.rt_apex], [peak.mz],
                            color=color, marker='*', s=50, alpha=0.25, zorder=4,
                        )

        # FID peak RT lines
        for rt in fid_rts:
            self._ax_scatter.axvline(rt, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)
            self._ax_scatter.text(
                rt, 1.01, f"{rt:.3f}",
                transform=self._ax_scatter.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7, color='steelblue',
            )

        n_clusters = len(rt_clusters)
        n_noise = len(noise_peaks)
        self._ax_scatter.set_title(
            f"RT Clusters — {n_clusters} cluster(s), {n_noise} noise — "
            f"cluster {cluster_idx} selected"
        )
        self._ax_scatter.set_ylabel("m/z")
        self._canvas.draw()

    def _clear_cluster_selection(self):
        """Remove cluster highlight and restore normal render."""
        if self._selected_cluster_idx is None:
            return
        self._selected_cluster_idx = None
        if self._last_result is not None:
            self._render_plots(self._last_result)

    # Scatter becomes unreadable beyond this many clusters; show guidance instead.
    _MAX_RENDERABLE_CLUSTERS = 100

    def _render_plots(self, result: dict):
        """Render DBSCAN scatter (top) and EIC traces (bottom) from preview result."""
        self._ax_scatter.clear()
        self._ax_eic.clear()

        win_peaks = result['win_peaks']
        fid_rts = [p.retention_time for p in win_peaks]

        if result.get('empty'):
            self._ax_scatter.set_title("No EIC peaks found in this window")
            self._ax_eic.set_title("(check Min Cluster Intensity setting)")
            self._canvas.draw()
            self._scatter_coords = None
            self._scatter_cluster_ids = []
            self._scatter_tree = None
            self._status_label.setText(
                "No EIC peaks found in this window — try lowering Min Cluster Intensity."
            )
            return

        intermediates = result['intermediates']
        rt_clusters = intermediates['rt_clusters']
        noise_peaks = intermediates['noise_peaks']
        model_peaks = intermediates['model_peaks']
        top_eic = result['top_eic']
        components = result['components']

        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab10')
        cluster_colors: dict[int, tuple] = {}
        eic_to_color: dict[int, tuple] = {}

        for ci, cluster in enumerate(rt_clusters):
            color = cmap(ci % 10)
            cluster_colors[ci] = color
            for peak in cluster:
                eic_to_color[id(peak)] = color

        model_peak_ids = {id(mp) for mp in model_peaks}
        n_clusters = len(rt_clusters)
        n_noise = len(noise_peaks)

        # ── Top subplot: DBSCAN scatter ────────────────────────────────────────
        if n_clusters > self._MAX_RENDERABLE_CLUSTERS:
            self._ax_scatter.text(
                0.5, 0.5,
                f"{n_clusters} clusters from {len(result['eic_peaks'])} EIC peaks\n\n"
                "Too many clusters to display usefully.\n\n"
                "Suggestions:\n"
                "• ↑ Min Cluster Size (currently too low for this data density)\n"
                "• ↑ Min Cluster Intensity (filters weak EIC peaks)\n"
                "• ↑ Min Cluster Distance (merges nearby clusters)\n"
                "• ↓ Padding fraction (reduces EIC extraction range)",
                transform=self._ax_scatter.transAxes,
                ha='center', va='center', fontsize=10, color='#cc3300',
                multialignment='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='#fff5f5', ec='#cc3300', alpha=0.9),
            )
            self._ax_scatter.set_title(
                f"RT Clusters — {n_clusters} cluster(s), {n_noise} noise point(s) (too many to render)"
            )
            self._scatter_coords = None
            self._scatter_cluster_ids = []
            self._scatter_tree = None
        else:
            # Shade FID peak span to distinguish "in-window" from padding zone
            if fid_rts:
                fid_min, fid_max = min(fid_rts), max(fid_rts)
                # Add a half-peak-width margin so isolated peaks get a visible band
                half_gap = (result['w_end'] - result['w_start']) * 0.05
                self._ax_scatter.axvspan(
                    fid_min - half_gap, fid_max + half_gap,
                    alpha=0.08, color='steelblue', zorder=1, label='FID span'
                )

            # Noise points
            if noise_peaks:
                noise_rts = [p.rt_apex for p in noise_peaks]
                noise_mzs = [p.mz for p in noise_peaks]
                self._ax_scatter.scatter(
                    noise_rts, noise_mzs, color='gray', s=10, alpha=0.5,
                    label='Noise', zorder=2
                )

            # Clustered points
            for ci, cluster in enumerate(rt_clusters):
                color = cluster_colors[ci]
                rts = [p.rt_apex for p in cluster]
                mzs = [p.mz for p in cluster]
                self._ax_scatter.scatter(rts, mzs, color=color, s=18, zorder=3)
                for peak in cluster:
                    if id(peak) in model_peak_ids:
                        self._ax_scatter.scatter(
                            [peak.rt_apex], [peak.mz],
                            color=color, marker='*', s=80, zorder=4
                        )

            # FID peak RT lines with RT labels
            for rt in fid_rts:
                self._ax_scatter.axvline(rt, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)
                self._ax_scatter.text(
                    rt, 1.01, f"{rt:.3f}",
                    transform=self._ax_scatter.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color='steelblue',
                )

            self._ax_scatter.set_title(
                f"RT Clusters — {n_clusters} cluster(s), {n_noise} noise point(s)"
            )

            # Build spatial index for click-to-search
            all_coords = []
            all_cluster_ids = []
            for peak in noise_peaks:
                all_coords.append([peak.rt_apex, peak.mz])
                all_cluster_ids.append(-1)
            for ci, cluster in enumerate(rt_clusters):
                for peak in cluster:
                    all_coords.append([peak.rt_apex, peak.mz])
                    all_cluster_ids.append(ci)
            if all_coords:
                self._scatter_coords = np.array(all_coords)
                self._scatter_cluster_ids = all_cluster_ids
                self._scatter_tree = cKDTree(self._scatter_coords)
            else:
                self._scatter_coords = None
                self._scatter_cluster_ids = []
                self._scatter_tree = None

        self._ax_scatter.set_ylabel("m/z")

        # ── Bottom subplot: EIC traces ─────────────────────────────────────────
        normalize = self._normalize_eic_check.isChecked()
        for peak in top_eic:
            color = eic_to_color.get(id(peak), 'gray')
            lw = 2.5 if id(peak) in model_peak_ids else 1.0
            y = peak.intensity_array.astype(float)
            if normalize:
                peak_max = y.max()
                if peak_max > 0:
                    y = y / peak_max
            self._ax_eic.plot(peak.rt_array, y, color=color, linewidth=lw, alpha=0.8)

        for rt in fid_rts:
            self._ax_eic.axvline(rt, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)

        shown = len(top_eic)
        total = len(result['eic_peaks'])
        self._ax_eic.set_xlabel("Retention Time (min)")
        self._ax_eic.set_ylabel("Norm. Intensity" if normalize else "Intensity")
        self._ax_eic.set_title(
            f"EIC Traces (top {shown} of {total} shown"
            + (", normalized)" if normalize else ")")
        )

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

        # Hint when padding is pulling in far-from-peak EIC data
        total_eic = len(result['eic_peaks'])
        if total_eic > 0 and n_noise / max(total_eic, 1) > 0.4:
            status += f"  ⚠ High noise ratio ({n_noise}/{total_eic} EIC peaks) — try ↑ Min Cluster Size or ↓ Padding"

        self._status_label.setText(status)

    def closeEvent(self, event):
        self._ms = None  # release rainbow data
        super().closeEvent(event)
