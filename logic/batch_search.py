import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QThreadPool
import traceback
import sys
import rainbow as rb
# Import the spectrum extractor
from logic.spectrum_extractor import SpectrumExtractor

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
        self.search_completed = False  # Add search completion flag
        
        # Ensure the toolkit's mz_shift matches the UI value
        if 'mz_shift' in self.options:
            self.ms_toolkit.mz_shift = self.options['mz_shift']
        
        self.spectrum_extractor = SpectrumExtractor(debug=self.options.get('debug', False))
    
    @Slot()
    def run(self):
        """Run the batch search."""
        try:
            # Signal the start of processing
            self.signals.started.emit(len(self.peaks))
            self.signals.log_message.emit(f"Starting batch search on {len(self.peaks)} peaks...")
            
            # Track successful matches and saturated peaks
            successful_matches = 0
            saturated_peaks = 0
            
            # Get options
            options = self.options
            debug = options.get('debug', True)
            
            # Process each peak
            for i, peak in enumerate(self.peaks):
                # Check for cancellation
                if self.cancelled:
                    if debug:
                        print("Batch search cancelled by user")
                    self.signals.log_message.emit("Batch search cancelled by user")
                    break
                
                # CRITICAL FIX: Skip peaks that have been manually assigned
                if hasattr(peak, 'manual_assignment') and peak.manual_assignment:
                    # Keep the manual assignment
                    compound_name = peak.compound_id
                    self.signals.progress.emit(i, compound_name, [(compound_name, 1.0)])
                    self.signals.log_message.emit(f"Peak {i+1}/{len(self.peaks)}: Using manual assignment '{compound_name}'")
                    successful_matches += 1
                    continue
                
                # Add this: periodically log progress
                if i == 0 or i == len(self.peaks)-1 or (i+1) % 5 == 0:
                    self.signals.log_message.emit(
                        f"Processing peak {i+1}/{len(self.peaks)} at RT={peak.retention_time:.3f}"
                    )
                
                # Extract spectrum based on extraction method
                extraction_method = options.get('extraction_method', 'apex')
                
                spectrum = self.spectrum_extractor.extract_for_peak(
                    self.data_directory, 
                    peak, 
                    self.options
                )
                
                if not spectrum or 'mz' not in spectrum or 'intensities' not in spectrum:
                    # Signal progress even when spectrum extraction fails
                    self.signals.progress.emit(i, f"No spectrum at RT {peak.retention_time:.3f}", [])
                    continue
                
                # Check for saturation and store in peak object
                if 'is_saturated' in spectrum and spectrum['is_saturated']:
                    peak.is_saturated = True
                    peak.saturation_level = spectrum.get('saturation_level', 0)
                    saturated_peaks += 1
                    
                    # Log saturation warning
                    self.signals.log_message.emit(
                        f"WARNING: Peak {peak.peak_number} at RT={peak.retention_time:.3f} shows detector saturation! "
                        f"Max intensity: {peak.saturation_level:.2e}"
                    )
                else:
                    peak.is_saturated = False
                
                # Continue with search as usual...
                # Create spectrum tuples
                query_spectrum = [(m, i) for m, i in zip(spectrum['mz'], spectrum['intensities'])]
                
                # Search based on selected method
                search_method = options.get('search_method', 'vector')
                results = []
                
                if search_method == 'vector':
                    # Vector search
                    results = self.ms_toolkit.search_vector(
                        query_spectrum,
                        top_n=options.get('top_n', 5),
                        composite=(options.get('similarity', 'composite') == 'composite'),
                        weighting_scheme=options.get('weighting', 'NIST_GC'),
                        unmatched_method=options.get('unmatched', 'keep_all'),
                        top_k_clusters=options.get('top_k_clusters', 1)  # Add this parameter
                    )
                else:
                    # Word2Vec search
                    results = self.ms_toolkit.search_w2v(
                        query_spectrum,
                        top_n=options.get('top_n', 5),
                        intensity_power=options.get('intensity_power', 0.6),
                        top_k_clusters=options.get('top_k_clusters', 1)  # Add this parameter
                    )
                
                # Get best match
                if results:
                    best_match = results[0]
                    
                    # Get compound name and score
                    compound_name = best_match[0]
                    match_score = best_match[1]
                    
                    # Get CAS number if available in the library
                    casno = None
                    try:
                        # Access the library through the toolkit
                        if hasattr(self.ms_toolkit, 'library') and compound_name in self.ms_toolkit.library:
                            compound = self.ms_toolkit.library[compound_name]
                            if hasattr(compound, 'casno'):
                                casno = self.format_casno(compound.casno)
                    except Exception as e:
                        if debug:
                            print(f"Error getting CAS number for {compound_name}: {e}")
                    
                    # Update peak with match info - use exact same field names as in the notebook
                    peak.compound_id = compound_name
                    peak.Compound_ID = compound_name  # Add notebook-style field name
                    peak.casno = casno
                    peak.Qual = match_score  # Use exact same field name as notebook
                    
                    # Signal progress
                    self.signals.progress.emit(i, peak.compound_id, results)
                    self.signals.log_message.emit(f"Processed peak {i+1}/{len(self.peaks)}: {peak.compound_id}")
                    successful_matches += 1
                    
                    # After processing peak, add a bit more to the progress log for significant matches
                    if match_score > 0.7:  # Only log good matches
                        self.signals.log_message.emit(
                            f"Peak {i+1} identified as {compound_name} (score: {match_score:.3f})"
                        )
                else:
                    # Signal progress even when no matches are found
                    self.signals.progress.emit(i, "No matches found", [])
            
            # Include saturation info in final summary
            self.signals.log_message.emit(
                f"Batch search {'completed' if not self.cancelled else 'cancelled'}: "
                f"{successful_matches}/{len(self.peaks)} peaks matched, "
                f"{saturated_peaks} peaks showed detector saturation"
            )
            
            # THIS IS THE CRITICAL FIX - EXPLICIT SIGNAL EMISSION
            self.signals.finished.emit()
            
        except Exception as e:
            # Capture full exception info
            error_msg = f"Error in batch search: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
            self.signals.log_message.emit(f"Error in batch search: {str(e)}")
            
            # Also emit finished signal on error to prevent hanging
            self.signals.finished.emit()
    
    def _on_search_completed(self):
        """Handle MS search completion."""
        # Don't interact with UI objects directly from the worker thread
        self.search_completed = True
        
        # Just emit a signal and let the main thread handle the UI update
        if not self.cancelled:
            self.signals.log_message.emit("MS library search completed")

    def format_casno(casno):
        """Format a CAS number with dashes."""
        if not casno or not isinstance(casno, str):
            return ""
        padded_casno = casno.zfill(9)
        return padded_casno[:-3] + '-' + padded_casno[-3:-1] + '-' + padded_casno[-1:]