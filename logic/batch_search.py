import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QThreadPool
import traceback
import sys
import rainbow as rb

class BatchSearchWorkerSignals(QObject):
    """Signals for the batch search worker."""
    started = Signal(int)  # Total number of peaks
    progress = Signal(int, str, object)  # Peak index, peak name, search results
    finished = Signal()
    error = Signal(str)
    log_message = Signal(str)  # Added signal for log messages

# Add spectrum extraction utilities
def extract_peak_spectrum(data_directory, peak, 
                          subtract_background=True,
                          subtraction_method='min_tic',
                          subtract_weight=0.1,
                          tic_weight=True,
                          extraction_method='apex',  # 'apex', 'average', 'range', or 'midpoint'
                          range_points=5,
                          midpoint_width_percent=20,  # New parameter (as percentage)
                          intensity_threshold=0.01,
                          saturation_threshold=8.0e6,  # Add saturation threshold
                          debug=False):
    """Extract the mass spectrum for a peak.
    
    Args:
        data_directory (str): Path to the GC-MS data directory
        peak: Peak object with start_time and end_time attributes
        subtract_background (bool): Whether to subtract background spectrum
        subtraction_method (str): Method for background subtraction ('left', 'right', 'min_tic', 'average')
        subtract_weight (float): Weight factor for background subtraction
        tic_weight (bool): Whether to weight spectra by TIC intensity
        extraction_method (str): Method for spectrum extraction:
            - 'apex': Use single point at max TIC within peak bounds
            - 'average': Average all points within peak bounds
            - 'range': Average points within range_points of apex
            - 'midpoint': Window centered on the temporal midpoint of the peak
        range_points (int): Number of points on each side of apex for 'range' method
        midpoint_width_percent (int): Width of the midpoint window as a percentage of peak width
        intensity_threshold (float): Minimum intensity threshold for filtering peaks
        saturation_threshold (float): Maximum intensity threshold for detecting saturation
        debug (bool): Whether to print debug information
        
    Returns:
        dict: Dictionary with 'mz', 'intensities', 'rt', 'is_saturated', 'saturation_level', and 'saturation_threshold' keys, or None if extraction fails
    """
    try:
        # Load the MS data
        datadir = rb.read(data_directory)
        ms = datadir.get_file('data.ms')
        tic = np.sum(ms.data, axis=1)
        
        start_time = float(peak.start_time)
        end_time = float(peak.end_time)
        
        if debug:
            print(f"Extracting spectrum for peak from {start_time:.3f} to {end_time:.3f}")
            print(f"Extraction method: {extraction_method}")
            print(f"MS data time range: {ms.xlabels[0]:.3f} to {ms.xlabels[-1]:.3f} min ({len(ms.xlabels)} points)")
        
        # Find indices corresponding to peak bounds
        left_idx = np.argmin(np.abs(np.array(ms.xlabels) - start_time))
        right_idx = np.argmin(np.abs(np.array(ms.xlabels) - end_time))
        
        # Ensure right_idx > left_idx
        if right_idx <= left_idx:
            if debug:
                print(f"Invalid indices: left={left_idx} ({ms.xlabels[left_idx]:.3f}), right={right_idx} ({ms.xlabels[right_idx]:.3f}). Adjusting...")
            right_idx = min(left_idx + 3, len(tic) - 1)
            
        if debug:
            print(f"Peak bounds in MS data: index {left_idx} ({ms.xlabels[left_idx]:.3f}) to {right_idx} ({ms.xlabels[right_idx]:.3f}) - {right_idx-left_idx+1} points")
        
        # Find peak apex (maximum TIC within bounds)
        if right_idx > left_idx:
            tic_slice = tic[left_idx:right_idx+1]
            local_peak = np.argmax(tic_slice)
            tic_peak = local_peak + left_idx
        else:
            tic_peak = left_idx
            
        if debug:
            print(f"Peak apex at index {tic_peak}, RT={ms.xlabels[tic_peak]:.3f}, TIC={tic[tic_peak]}")
        
        # SATURATION CHECK: Check for detector saturation within the peak bounds
        peak_ms_data = ms.data[left_idx:right_idx+1, :]
        max_intensity = np.max(peak_ms_data)
        is_saturated = max_intensity >= saturation_threshold
        
        # If saturation is detected, find the scan index just before saturation
        saturation_adjustment = None
        if is_saturated:
            # Find all saturated data points
            saturated_points = np.where(peak_ms_data >= saturation_threshold)
            
            if debug:
                print(f"DETECTOR SATURATION DETECTED! Max intensity: {max_intensity:.2e}")
                print(f"Found {len(saturated_points[0])} saturated data points")
            
            # Find the earliest scan with saturation
            if len(saturated_points[0]) > 0:
                # Get unique scan indices and sort them
                saturated_scans = np.unique(saturated_points[0])
                first_saturated_scan = saturated_scans[0]
                
                # Convert to global index
                first_saturated_idx = first_saturated_scan + left_idx
                
                # Use the scan right before saturation if possible
                if first_saturated_idx > left_idx:
                    saturation_adjustment = first_saturated_idx - 1
                    
                    if debug:
                        print(f"Using scan at index {saturation_adjustment} (RT={ms.xlabels[saturation_adjustment]:.3f}) to avoid saturation")
                        print(f"This is {first_saturated_idx - saturation_adjustment} scans before first saturation")
        
        # Extract spectrum based on selected method (with saturation adjustment if needed)
        if extraction_method == 'apex':
            # 1. PEAK APEX: Single point at maximum TIC
            if is_saturated and saturation_adjustment is not None:
                # Override with pre-saturation point
                spectrum_left = saturation_adjustment
                spectrum_right = saturation_adjustment + 1
                
                if debug:
                    print(f"SATURATION ADJUSTMENT: Using pre-saturation point at RT={ms.xlabels[saturation_adjustment]:.3f} instead of apex")
            else:
                spectrum_left = tic_peak
                spectrum_right = tic_peak + 1  # +1 because of how slicing works
                
                if debug:
                    print(f"Using single point apex at RT={ms.xlabels[tic_peak]:.3f}")
                
        elif extraction_method == 'average':
            # 2. PEAK AVERAGE: All points between peak bounds
            if is_saturated and saturation_adjustment is not None:
                # Use only points up to saturation
                spectrum_left = left_idx
                spectrum_right = saturation_adjustment + 1
                
                if debug:
                    print(f"SATURATION ADJUSTMENT: Using average of points from {ms.xlabels[left_idx]:.3f} to {ms.xlabels[saturation_adjustment]:.3f}")
            else:
                spectrum_left = left_idx
                spectrum_right = right_idx + 1
                
                if debug:
                    print(f"Using peak average from indices {left_idx}-{right_idx}")
                
        elif extraction_method == 'range':
            # 3. RANGE: Fixed number of points around apex
            if is_saturated and saturation_adjustment is not None:
                # Center the window on the pre-saturation point
                spectrum_left = max(0, saturation_adjustment - range_points)
                spectrum_right = min(len(ms.data), saturation_adjustment + range_points + 1)
                
                if debug:
                    print(f"SATURATION ADJUSTMENT: Using range around pre-saturation point at RT={ms.xlabels[saturation_adjustment]:.3f}")
            else:
                spectrum_left = max(0, tic_peak - range_points)
                spectrum_right = min(len(ms.data), tic_peak + range_points + 1)
                
                if debug:
                    print(f"Using range: {range_points} points on each side of apex")
                
        elif extraction_method == 'midpoint':
            # 4. MIDPOINT: Window centered on the temporal midpoint of the peak
            # Calculate midpoint index
            midpoint_idx = int((left_idx + right_idx) / 2)
            
            if is_saturated and saturation_adjustment is not None:
                # If midpoint is after saturation, use the pre-saturation point instead
                if midpoint_idx >= saturation_adjustment:
                    midpoint_idx = saturation_adjustment
                    
                    if debug:
                        print(f"SATURATION ADJUSTMENT: Using pre-saturation point as midpoint")
            
            midpoint_rt = ms.xlabels[midpoint_idx]
            
            # Calculate window width based on peak width percentage
            peak_width_indices = right_idx - left_idx
            half_window = int((peak_width_indices * midpoint_width_percent / 100) / 2)
            
            # Ensure window is at least 1 point wide
            half_window = max(1, half_window)
            
            # Calculate window boundaries, clamping to integration bounds
            spectrum_left = max(left_idx, midpoint_idx - half_window)
            spectrum_right = min(right_idx + 1, midpoint_idx + half_window + 1)  # +1 for slicing
            
            # If spectrum_right is beyond saturation point, adjust it
            if is_saturated and saturation_adjustment is not None and spectrum_right > saturation_adjustment + 1:
                spectrum_right = saturation_adjustment + 1
                
                if debug:
                    print(f"SATURATION ADJUSTMENT: Limited right window bound to pre-saturation point")
            
            if debug:
                print(f"Using midpoint: index {midpoint_idx} (RT={midpoint_rt:.3f})")
                print(f"Window width: {midpoint_width_percent}% of peak width = {half_window*2} points")
                print(f"Window spans indices {spectrum_left}-{spectrum_right-1}")
        else:
            # Default to apex if method not recognized
            spectrum_left = tic_peak
            spectrum_right = tic_peak + 1
            if debug:
                print(f"Unknown extraction method '{extraction_method}', defaulting to apex")
        
        # Find background subtraction point
        subtract_val = None
        if subtract_background:
            try:
                if subtraction_method == 'left':
                    # Use left bound
                    subtract_val = left_idx
                elif subtraction_method == 'right':
                    # Use right bound
                    subtract_val = right_idx
                elif subtraction_method == 'min_tic':
                    # Find minimum TIC point within bounds
                    subtract_val = np.argmin(tic[left_idx:right_idx+1]) + left_idx
                elif subtraction_method == 'average':
                    # Use average of spectra at boundaries
                    left_spectrum = ms.data[left_idx, :].astype(float)
                    right_spectrum = ms.data[right_idx, :].astype(float)
                    avg_background = (left_spectrum + right_spectrum) / 2
                    subtract_val = 'precomputed'
                else:
                    # Default to min_tic
                    subtract_val = np.argmin(tic[left_idx:right_idx+1]) + left_idx
                    
                if debug and subtract_val != 'precomputed':
                    print(f"Background subtraction point: RT={ms.xlabels[subtract_val]:.3f}")
                elif debug:
                    print("Using average boundary spectrum for background")
            except Exception as e:
                if debug:
                    print(f"Error finding background point: {str(e)}")
                subtract_val = None
        
        # Extract spectrum (single point or average)
        if spectrum_right <= spectrum_left:
            if debug:
                print("Invalid spectrum range, using single-point apex")
            spectrum = ms.data[tic_peak, :].astype(float)
        elif spectrum_right - spectrum_left == 1:
            # Single point
            spectrum = ms.data[spectrum_left, :].astype(float)
            if debug:
                print("Using single-point spectrum")
        else:
            # Average multiple points
            if tic_weight:
                # TIC-weighted average
                weights = tic[spectrum_left:spectrum_right]
                spectrum = np.average(
                    ms.data[spectrum_left:spectrum_right, :].astype(float),
                    axis=0, weights=weights
                )
                if debug:
                    print(f"Using TIC-weighted average from {spectrum_right-spectrum_left} points")
            else:
                # Simple average
                spectrum = np.average(
                    ms.data[spectrum_left:spectrum_right, :].astype(float),
                    axis=0
                )
                if debug:
                    print(f"Using simple average from {spectrum_right-spectrum_left} points")
        
        # Subtract background if requested
        if subtract_background and subtract_val is not None:
            try:
                if subtract_val == 'precomputed':
                    spectrum -= avg_background * subtract_weight
                else:
                    spectrum -= ms.data[subtract_val, :].astype(float) * subtract_weight
                
                if debug:
                    print(f"Subtracted background with weight {subtract_weight}")
            except Exception as e:
                if debug:
                    print(f"Error subtracting background: {str(e)}")
        
        # Filter out low intensity peaks (< intensity_threshold of max)
        max_intensity = max(spectrum) if np.max(spectrum) > 0 else 1.0
        threshold = intensity_threshold * max_intensity
        
        # Create mz array (assuming 1-based indices for m/z values)
        mz_values = np.arange(len(spectrum)) + 1
        
        # Apply threshold and get only positive values
        mask = (spectrum > threshold) & (spectrum > 0)
        
        # Return the filtered spectrum with saturation info
        return {
            'rt': ms.xlabels[tic_peak],  # Always return apex RT for reference
            'mz': mz_values[mask],
            'intensities': spectrum[mask],
            'is_saturated': is_saturated,  # Add saturation flag
            'saturation_level': max_intensity,  # Add maximum intensity value
            'saturation_threshold': saturation_threshold  # Add the threshold used
        }
        
    except Exception as e:
        if debug:
            print(f"Error extracting spectrum: {str(e)}\n{traceback.format_exc()}")
        return None

# Add this helper function to properly format CAS numbers
def format_casno(casno):
    """Format a CAS number with dashes."""
    if not casno or not isinstance(casno, str):
        return ""
    padded_casno = casno.zfill(9)
    return padded_casno[:-3] + '-' + padded_casno[-3:-1] + '-' + padded_casno[-1:]

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
                
                spectrum = extract_peak_spectrum(
                    self.data_directory, 
                    peak,
                    subtract_background=options.get('subtract_enabled', True),
                    subtraction_method=options.get('subtraction_method', 'min_tic'),
                    subtract_weight=options.get('subtract_weight', 0.1),
                    tic_weight=options.get('tic_weight', True),
                    extraction_method=extraction_method,
                    range_points=options.get('range_points', 5),
                    midpoint_width_percent=options.get('midpoint_width_percent', 20),
                    intensity_threshold=options.get('intensity_threshold', 0.01),
                    saturation_threshold=8.0e6,  # Add saturation threshold (~8.38e6 is true limit)
                    debug=debug
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
                        unmatched_method=options.get('unmatched', 'keep_all')
                    )
                else:
                    # Word2Vec search
                    results = self.ms_toolkit.search_w2v(
                        query_spectrum,
                        top_n=options.get('top_n', 5),
                        intensity_power=options.get('intensity_power', 0.6)
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
                                casno = format_casno(compound.casno)
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