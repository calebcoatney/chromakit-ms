"""QRunnable worker for spectral deconvolution.

Wraps run_spectral_deconvolution() for use on QThreadPool.
Follows the same pattern as BatchSearchWorker.
"""
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from logic.correlation_grouping import CorrelationDeconvParams
from logic.spectral_deconv_runner import run_spectral_deconvolution, WindowGroupingParams


class SpectralDeconvWorkerSignals(QObject):
    progress = Signal(int)    # 0–100 percentage
    finished = Signal()       # peaks mutated in place; caller accesses them directly
    error = Signal(str)


class SpectralDeconvWorker(QRunnable):
    """Run spectral deconvolution on a background thread."""

    def __init__(
        self,
        peaks: list,
        ms_data_path: str,
        deconv_params: CorrelationDeconvParams | None = None,
        grouping_params: WindowGroupingParams | None = None,
        ms_time_offset: float = 0.0,
    ):
        super().__init__()
        self.peaks = peaks
        self.ms_data_path = ms_data_path
        self.deconv_params = deconv_params or CorrelationDeconvParams()
        self.grouping_params = grouping_params or WindowGroupingParams()
        self.ms_time_offset = ms_time_offset
        self.signals = SpectralDeconvWorkerSignals()
        self.cancelled = False       # set by the UI's Cancel
        self.stopped_early = False   # the runner skipped windows because of it

    def _should_cancel(self) -> bool:
        """The runner asks before each window and breaks on True, so True means it stopped early.

        A Cancel clicked after the last window has started is never seen: that run is complete.
        """
        if self.cancelled:
            self.stopped_early = True
        return self.cancelled

    @Slot()
    def run(self):
        try:
            run_spectral_deconvolution(
                peaks=self.peaks,
                ms_data_path=self.ms_data_path,
                deconv_params=self.deconv_params,
                grouping_params=self.grouping_params,
                progress_callback=self.signals.progress.emit,
                should_cancel=self._should_cancel,
                ms_time_offset=self.ms_time_offset,
            )
            self.signals.finished.emit()

        except Exception as e:
            import traceback
            self.signals.error.emit(traceback.format_exc())
