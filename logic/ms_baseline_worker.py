import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
import traceback
import sys

class MSBaselineCorrectionSignals(QObject):
    """Signals for MS baseline correction worker."""
    started = Signal(int)  # Total number of ions
    progress = Signal(int, int)  # Current ion, total ions
    finished = Signal(object, object)  # Corrected MS data, MS baselines array
    error = Signal(str)
    log_message = Signal(str)  # For detailed logging

class MSBaselineCorrectionWorker(QRunnable):
    """Worker for full MS baseline correction."""
    
    def __init__(self, ms_data, baseline_params):
        super().__init__()
        self.ms_data = ms_data
        self.baseline_params = baseline_params
        self.signals = MSBaselineCorrectionSignals()
        self.cancelled = False
    
    @Slot()
    def run(self):
        try:
            # Get total number of ions (m/z channels)
            total_ions = self.ms_data.shape[1]
            print(f"MS Baseline Worker starting with {total_ions} m/z channels")
            self.signals.started.emit(total_ions)
            self.signals.log_message.emit(f"Starting baseline correction for {total_ions} m/z channels...")
            
            # Make sure we actually have data to process
            if total_ions == 0:
                self.signals.error.emit("MS data has no m/z channels")
                return
            
            # Create output arrays (make copies to preserve original)
            corrected_ms = np.copy(self.ms_data)
            ms_baselines = np.zeros_like(self.ms_data)  # Store baselines separately
            
            # Get baseline parameters
            method = self.baseline_params.get('method', 'arpls')
            lam = self.baseline_params.get('lambda', 10000)
            p = self.baseline_params.get('asymmetry', 0.001)
            tol = self.baseline_params.get('tol', 0.01)
            
            print(f"Using baseline method: {method} with lambda={lam}, asymmetry={p}")
            
            # Import properly from pybaselines
            try:
                from pybaselines import Baseline
                # Create a new baseline fitter instance
                baseline_fitter = Baseline()
            except ImportError:
                self.signals.error.emit("pybaselines module not found. Please install with: pip install pybaselines")
                return
            
            # Available methods in pybaselines with common algorithms
            methods = {
                "asls": baseline_fitter.asls,         # Asymmetric Least Squares
                "imodpoly": baseline_fitter.imodpoly, # Improved Modified Polynomial
                "modpoly": baseline_fitter.modpoly,   # Modified Polynomial
                "snip": baseline_fitter.snip,         # Statistics-sensitive Non-linear
                "airpls": baseline_fitter.airpls,     # Adaptive Iteratively Reweighted
                "arpls": baseline_fitter.arpls,       # Asymmetrically Reweighted
            }
            
            # Validate method
            if method not in methods:
                print(f"Unknown baseline method: {method}, falling back to arpls")
                method = "arpls"
            
            # Methods that use the lam parameter
            lam_methods = {"asls", "airpls", "arpls"}
            
            # Debug counter for significant changes
            changes_made = 0
            
            # Process each ion channel
            for i in range(total_ions):
                if self.cancelled:
                    self.signals.log_message.emit("MS baseline correction cancelled by user")
                    break
                
                # Extract ion chromatogram for this m/z
                ion_trace = self.ms_data[:, i]
                
                # Skip if all zeros or very low signal
                max_intensity = np.max(ion_trace)
                if max_intensity < 100:
                    # Update progress but don't process
                    self.signals.progress.emit(i + 1, total_ions)
                    continue
                
                # Calculate baseline based on selected method
                try:
                    # Re-initialize baseline fitter to avoid memory issues
                    baseline_fitter = Baseline()
                    
                    if method in lam_methods:
                        if method == 'asls':
                            # asls uses both lambda and p (asymmetry)
                            baseline, params = methods[method](ion_trace, lam=lam, p=p)
                        elif method == 'arpls':
                            # arpls with tolerance parameter
                            baseline, params = methods[method](ion_trace, lam=lam, tol=tol)
                        else:
                            # airpls uses only lambda
                            baseline, params = methods[method](ion_trace, lam=lam)
                    else:
                        # Other methods like modpoly, imodpoly don't use lambda
                        baseline, params = methods[method](ion_trace)
                    
                    # Error check similar to your testing code
                    if np.max(baseline) > np.max(ion_trace) * 10:
                        print(f'Error at m/z={i+1}, setting baseline to 0.')
                        baseline = np.zeros_like(ion_trace)
                    
                    # Store the baseline
                    ms_baselines[:, i] = baseline
                    
                    # Subtract baseline
                    corrected_ion = ion_trace - baseline
                    
                    # Ensure no negative values
                    corrected_ion[corrected_ion < 0] = 0
                    
                    # Store corrected ion trace
                    corrected_ms[:, i] = corrected_ion
                    
                    # Track significant changes - more than 10% reduction in peak height
                    new_max = np.max(corrected_ion)
                    if new_max < 0.9 * max_intensity and max_intensity > 1000:
                        changes_made += 1
                        if changes_made % 10 == 0 or changes_made < 10:
                            print(f"Significant baseline correction for m/z channel {i+1}: {max_intensity:.0f} → {new_max:.0f}")
                
                except Exception as e:
                    # Log error but continue with other ions
                    error_msg = f"Error processing m/z {i+1}: {str(e)}"
                    print(error_msg)
                    self.signals.log_message.emit(error_msg)
                
                # Update progress always
                self.signals.progress.emit(i + 1, total_ions)
                
                # Log periodically
                if (i+1) % 100 == 0 or i == 0 or i == total_ions-1:
                    self.signals.log_message.emit(f"Processed {i+1}/{total_ions} m/z channels")
            
            # Signal completion
            if not self.cancelled:
                self.signals.log_message.emit(f"MS baseline correction completed for {total_ions} m/z channels")
                self.signals.log_message.emit(f"Made significant changes to {changes_made} ion traces")
                print(f"MS Baseline Worker completed - Made significant changes to {changes_made} ion traces")
                
                # Signal with both corrected data and baselines
                self.signals.finished.emit(corrected_ms, ms_baselines)
            
        except Exception as e:
            error_msg = f"Error in MS baseline correction: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.signals.error.emit(error_msg)
            self.signals.log_message.emit(f"Error in MS baseline correction: {str(e)}")