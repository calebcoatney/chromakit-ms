"""QRunnable worker for spectral deconvolution.

Wraps run_spectral_deconvolution() for use on QThreadPool.
Follows the same pattern as BatchSearchWorker.
"""
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from logic.spectral_deconvolution import DeconvolutionParams
from logic.spectral_deconv_runner import run_spectral_deconvolution, WindowGroupingParams


class SpectralDeconvWorkerSignals(QObject):
    progress = Signal(int)    # 0–100 percentage
    finished = Signal()       # peaks mutated in place; caller accesses them directly
    error = Signal(str)


class SpectralDeconvWorker(QRunnable):
    """Run ADAP-GC spectral deconvolution on a background thread."""

    def __init__(
        self,
        peaks: list,
        ms_data_path: str,
        deconv_params: DeconvolutionParams | None = None,
        grouping_params: WindowGroupingParams | None = None,
    ):
        super().__init__()
        self.peaks = peaks
        self.ms_data_path = ms_data_path
        self.deconv_params = deconv_params or DeconvolutionParams()
        self.grouping_params = grouping_params or WindowGroupingParams()
        self.signals = SpectralDeconvWorkerSignals()
        self.cancelled = False

    @Slot()
    def run(self):
        try:
            run_spectral_deconvolution(
                peaks=self.peaks,
                ms_data_path=self.ms_data_path,
                deconv_params=self.deconv_params,
                grouping_params=self.grouping_params,
                progress_callback=self.signals.progress.emit,
                should_cancel=lambda: self.cancelled,
            )
            self.signals.finished.emit()

        except Exception as e:
            import traceback
            self.signals.error.emit(traceback.format_exc())
