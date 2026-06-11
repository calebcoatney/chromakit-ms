import traceback
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
# Import the spectrum extractor
from logic.spectrum_extractor import SpectrumExtractor
from logic.ms_search_core import run_batch_search
# Re-export format_casno so legacy module-level imports keep working.
# (Several call sites in ui/app.py and logic/automation_worker.py historically
# imported `format_casno` from this module; it had quietly been ClassName.format_casno
# only, breaking those imports. Now both forms work.)
from logic.ms_search_core import format_casno  # noqa: F401

class BatchSearchWorkerSignals(QObject):
    """Signals for the batch search worker."""
    started = Signal(int)  # Total number of peaks
    progress = Signal(int, str, object)  # Peak index, peak name, search results
    finished = Signal()
    error = Signal(str)
    log_message = Signal(str)  # Added signal for log messages

# Create a proxy function that uses the new extractor
def extract_peak_spectrum(data_directory, peak, **kwargs):
    """Legacy function that uses SpectrumExtractor for backward compatibility."""
    extractor = SpectrumExtractor(debug=kwargs.get('debug', False))
    return extractor.extract_for_peak(data_directory, peak, kwargs)

class BatchSearchWorker(QRunnable):
    """Worker for batch MS library search on integrated peaks."""
    
    def __init__(self, ms_toolkit, peaks, data_directory, options=None):
        """
        Initialize the worker.
                
                Args:
                    ms_toolkit: The MSToolkit instance
                    peaks: List of Peak objects
                    data_directory: Path to the data directory
                    options: Dictionary of search options
        """
        super().__init__()
        self.ms_toolkit = ms_toolkit
        self.peaks = peaks
        self.data_directory = data_directory
        self.options = options or {}
        self.signals = BatchSearchWorkerSignals()
        self.cancelled = False  # Add cancellation flag

        # Ensure the toolkit's mz_shift matches the UI value
        if 'mz_shift' in self.options:
            self.ms_toolkit.mz_shift = self.options['mz_shift']
        
        self.spectrum_extractor = SpectrumExtractor(debug=self.options.get('debug', False))
    
    @Slot()
    def run(self):
        """Run the batch search by delegating to ms_search_core.run_batch_search.

        The worker's responsibility shrinks to:
          1. Emit Qt signals (started, progress, finished, error, log_message)
             via callbacks bound to ms_search_core.
          2. Forward the cancellation flag.
          3. Translate the BatchSearchSummary into the legacy finished/error signals.
        """
        try:
            self.signals.started.emit(len(self.peaks))

            def on_progress(index, label, results):
                self.signals.progress.emit(index, label, results)

            def on_log(message):
                self.signals.log_message.emit(message)

            def should_cancel():
                return self.cancelled

            run_batch_search(
                ms_toolkit=self.ms_toolkit,
                peaks=self.peaks,
                data_directory=self.data_directory,
                options=self.options,
                respect_manual_assignments=True,
                progress_callback=on_progress,
                log_callback=on_log,
                should_cancel=should_cancel,
                extractor=self.spectrum_extractor,
            )

            self.signals.finished.emit()
        except Exception as e:
            error_msg = f"Error in batch search: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
            self.signals.log_message.emit(f"Error in batch search: {str(e)}")
            self.signals.finished.emit()
