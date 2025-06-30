import numpy as np
from scipy import signal, interpolate
from scipy.signal import medfilt, savgol_filter, find_peaks
from pybaselines import Baseline

class ChromatogramProcessor:
    """Processes chromatogram data with smoothing, baseline subtraction, and peak finding."""
    
    def __init__(self):
        # Initialize the baseline fitter
        self.baseline_fitter = Baseline()
        
        # Default parameters
        self.default_params = {
            'smoothing': {
                'enabled': False,
                'median_filter': {
                    'kernel_size': 5  # Must be odd
                },
                'savgol_filter': {
                    'window_length': 11,  # Must be odd
                    'polyorder': 3
                }
            },
            'baseline': {
                'show_corrected': False,  # Changed from 'enabled' to 'show_corrected'
                'method': 'asls',
                'lambda': 1e6,
                'asymmetry': 0.01
            },
            'peaks': {
                'enabled': False,
                'min_prominence': 0.5,
                'min_height': 0.0,
                'min_width': 0.0
            }
        }
    
    def process(self, x, y, params=None, ms_range=None):
        """Process chromatogram data with given parameters.
        
        Args:
            x: X values (time)
            y: Y values (intensity)
            params: Processing parameters
            ms_range: Optional tuple of (min_time, max_time) for MS data range
        """
        if params is None:
            params = self.default_params
        
        # Make a copy of the input data to avoid modifying the original
        # Note: x and y are already interpolated to standard length
        x_values = np.copy(x)
        y_values = np.copy(y)
        
        # STEP 1: Apply smoothing if enabled
        if params['smoothing']['enabled']:
            smoothed_y = self._apply_smoothing(np.copy(y_values), params['smoothing'])
        else:
            smoothed_y = np.copy(y_values)  # No smoothing
        
        # STEP 2: Always calculate baseline
        baseline_y, baseline_corrected_y = self._apply_baseline_correction(
            x_values, smoothed_y,
            method=params['baseline']['method'],
            lam=params['baseline']['lambda']
        )
        
        # STEP 3: Find peaks if enabled
        peaks_x = np.array([])
        peaks_y = np.array([])
        if params['peaks']['enabled']:
            peaks_x, peaks_y = self._find_peaks(x_values, baseline_corrected_y, params['peaks'], ms_range)
        
        # Return processed data
        return {
            'x': x_values,                  # X values 
            'original_y': y_values,         # Original unprocessed y values
            'smoothed_y': smoothed_y,       # After smoothing (if enabled)
            'baseline_y': baseline_y,       # The calculated baseline
            'corrected_y': baseline_corrected_y,  # After baseline correction
            'peaks_x': peaks_x,             # Peak x positions
            'peaks_y': peaks_y              # Peak y values
        }
    
    def _apply_smoothing(self, y, smoothing_params):
        """Apply smoothing to the data using median and Savitzky-Golay filters."""
        # Get parameters
        med_kernel = smoothing_params['median_filter']['kernel_size']
        sg_window = smoothing_params['savgol_filter']['window_length']
        sg_order = smoothing_params['savgol_filter']['polyorder']
        
        data_length = len(y)
        print(f"Applying smoothing with med_kernel={med_kernel}, sg_window={sg_window}, sg_order={sg_order}")
        
        # Ensure parameters are valid for the data length
        if med_kernel >= data_length:
            med_kernel = min(data_length - 1, 7)
            if med_kernel % 2 == 0:
                med_kernel -= 1
            print(f"Adjusted median kernel to {med_kernel} for data length {data_length}")
        elif med_kernel % 2 == 0:
            med_kernel += 1  # Make sure it's odd
        
        if sg_window >= data_length:
            sg_window = min(data_length - 1, 11)
            if sg_window % 2 == 0:
                sg_window -= 1
            print(f"Adjusted Savitzky-Golay window to {sg_window} for data length {data_length}")
        elif sg_window % 2 == 0:
            sg_window += 1  # Make sure it's odd
        
        # Make sure polynomial order is less than window size
        if sg_order >= sg_window:
            sg_order = sg_window - 1
            print(f"Adjusted polynomial order to {sg_order} for window {sg_window}")
        
        try:
            # Apply median filter - this removes spike noise
            y_med = medfilt(y, kernel_size=med_kernel)
            
            # Apply Savitzky-Golay filter - this smooths the curve
            y_smooth = savgol_filter(y_med, window_length=sg_window, polyorder=sg_order)
            
            # Calculate differences to show the effect
            orig_vs_med_diff = np.max(np.abs(y - y_med))
            med_vs_sg_diff = np.max(np.abs(y_med - y_smooth))
            print(f"Original vs median filter - max diff: {orig_vs_med_diff:.2f}")
            print(f"Median vs S-G filter - max diff: {med_vs_sg_diff:.2f}")
            
            if orig_vs_med_diff < 1.0 and med_vs_sg_diff < 1.0:
                print("Warning: Smoothing effect is very small - might not be visible")
            
            return y_smooth
        except Exception as e:
            print(f"Smoothing failed: {str(e)}")
            return y  # Return original data if smoothing fails
    
    def _apply_baseline_correction(self, x, y, method="asls", lam=1e6):
        """Apply baseline correction using the specified method."""
        # Re-initialize the baseline fitter each time to avoid length mismatch issues
        self.baseline_fitter = Baseline()
        
        # Available methods in pybaselines with common algorithms
        methods = {
            "asls": self.baseline_fitter.asls,           # Asymmetric Least Squares
            "imodpoly": self.baseline_fitter.imodpoly,   # Improved Modified Polynomial
            "modpoly": self.baseline_fitter.modpoly,     # Modified Polynomial
            "snip": self.baseline_fitter.snip,           # Statistics-sensitive Non-linear
            "airpls": self.baseline_fitter.airpls,       # Adaptive Iteratively Reweighted
            "arpls": self.baseline_fitter.arpls,         # Asymmetrically Reweighted
        }
        
        # Validate input data
        if x is None or y is None:
            print("No data for baseline correction")
            return np.zeros_like(y), y
            
        if len(y) == 0:
            print("Empty data for baseline correction")
            return np.zeros_like(y), y
        
        data_length = len(y)
        print(f"Applying baseline correction with method={method}, lambda={lam}, data length={data_length}")
        
        if method not in methods:
            print(f"Unknown baseline method: {method}, falling back to asls")
            # Fallback to asls if method not recognized
            method = "asls"
        
        # Methods that use the lam parameter
        lam_methods = {"asls", "airpls", "arpls"}
        
        try:
            # Apply the selected baseline correction method
            if method in lam_methods:
                print(f"Using lambda parameter with {method}")
                # Use a reasonable lambda based on data length
                scaled_lam = lam
                baseline, params = methods[method](y, lam=scaled_lam)
            else:
                print(f"Method {method} does not use lambda")
                baseline, params = methods[method](y)
                
            corrected = y - baseline
            print(f"Baseline correction successful!")
            
            return baseline, corrected
        except Exception as e:
            # If baseline correction fails, return zeros
            print(f"Baseline correction failed: {str(e)}")
            
            # For debugging, print data info
            print(f"Data type: {type(y)}, shape: {y.shape if hasattr(y, 'shape') else 'unknown'}")
            
            try:
                # Try a simpler approach - just fit a polynomial
                from scipy.signal import savgol_filter
                window_length = min(data_length // 10 * 2 + 1, 501)  # Make sure it's odd and reasonable
                window_length = max(window_length, 5)  # Ensure at least 5
                print(f"Trying fallback with Savitzky-Golay filter, window={window_length}")
                baseline = savgol_filter(y, window_length=window_length, polyorder=3)
                corrected = y - baseline
                return baseline, corrected
            except Exception as e2:
                print(f"Fallback baseline also failed: {str(e2)}")
                return np.zeros_like(y), y
    
    def _find_peaks(self, x, y, peak_params, ms_range=None):
        """Find peaks in the chromatogram.
        
        Args:
            x (np.ndarray): X values array
            y (np.ndarray): Y values array
            peak_params (dict): Peak finding parameters
            ms_range (tuple): Optional (min_time, max_time) for MS data range
            
        Returns:
            tuple: (peaks_x, peaks_y) arrays of peak positions and heights
        """
        try:
            prominence = peak_params['min_prominence']
            height = peak_params.get('min_height', 0)
            width = peak_params.get('min_width', 0)
            
            # Use scipy's find_peaks function
            peak_indices, _ = find_peaks(y, prominence=prominence, height=height, width=width)
            
            if len(peak_indices) > 0:
                peaks_x = x[peak_indices]
                peaks_y = y[peak_indices]
                
                # Filter peaks based on MS data range if provided
                if ms_range is not None:
                    ms_min, ms_max = ms_range
                    # Create mask for peaks within MS range
                    in_range_mask = (peaks_x >= ms_min) & (peaks_x <= ms_max)
                    
                    # Print info about peaks outside the MS range (solvent delay)
                    if not np.all(in_range_mask):
                        filtered_count = np.sum(~in_range_mask)
                        print(f"Filtered out {filtered_count} peaks outside MS range (solvent delay region)")
                    
                    # Only keep peaks within MS range
                    peaks_x = peaks_x[in_range_mask]
                    peaks_y = peaks_y[in_range_mask]
                
                return peaks_x, peaks_y
            
        except Exception as e:
            # Return empty arrays if peak finding fails
            print(f"Peak finding failed: {str(e)}")
            
        return np.array([]), np.array([])
    
    def integrate_peaks(self, processed_data=None, rt_table=None, chemstation_area_factor=0.0784, ms_data=None, quality_options=None):
        """Integrate peaks in a processed chromatogram.
        
        Args:
            processed_data: Dictionary with processed data, or None to use most recent processed data
            rt_table: Optional table mapping retention times to compounds
            chemstation_area_factor: Area scaling factor to match ChemStation
            ms_data: Optional MS data object for peak quality assessment
            quality_options: Options for peak quality assessment
            
        Returns:
            dict: Dictionary containing integration results
        """
        # Import here to avoid circular imports
        from logic.integration import Integrator
        
        # Use provided processed data or attempt to get it from the current state
        if processed_data is None:
            # Fallback to default parameters
            processed_data = self.process(np.array([]), np.array([]), self.default_params)
        
        # Check if we have peak data
        if 'peaks_x' not in processed_data or len(processed_data['peaks_x']) == 0:
            print("No peaks detected for integration. Enable peak detection first.")
            return {
                'peaks': [],
                'x_peaks': [],
                'y_peaks': [],
                'baseline_peaks': [],
                'retention_times': [],
                'integrated_areas': [],
                'integration_bounds': [],
                'peaks_list': []
            }
        
        # Integrate the peaks
        integration_results = Integrator.integrate(
            processed_data,
            rt_table=rt_table,
            chemstation_area_factor=chemstation_area_factor,
            verbose=True,
            ms_data=ms_data,  # Pass MS data for quality assessment
            quality_options=quality_options  # Pass quality options
        )
        
        return integration_results

    def align_tic_to_fid(self, fid_time, fid_signal, tic_time, tic_signal, max_lag_seconds=2.0, num_points=10000, verbose=True):
        """
        Align TIC signal to FID signal using cross-correlation to estimate the time lag.
        
        Args:
            fid_time (np.ndarray): FID time points (minutes)
            fid_signal (np.ndarray): FID signal intensities
            tic_time (np.ndarray): TIC time points (minutes)
            tic_signal (np.ndarray): TIC signal intensities
            max_lag_seconds (float): Maximum lag to consider in seconds
            num_points (int): Number of points for interpolation
            verbose (bool): Whether to print diagnostic information
        
        Returns:
            tuple: (aligned_tic_time, aligned_tic_signal, lag_seconds)
        """
        def log(msg):
            if verbose:
                print(msg)
        
        # Handle empty input data
        if len(fid_time) == 0 or len(tic_time) == 0:
            log("Warning: Empty input data")
            return tic_time, tic_signal, 0.0
            
        # Print input data summary
        log(f"FID time range: {np.min(fid_time):.2f} to {np.max(fid_time):.2f} minutes, {len(fid_time)} points")
        log(f"TIC time range: {np.min(tic_time):.2f} to {np.max(tic_time):.2f} minutes, {len(tic_time)} points")
        
        # Step 1: Identify the valid TIC region (after solvent delay)
        tic_start = np.min(tic_time)  
        tic_end = np.max(tic_time)
        
        # Step 2: Create analysis window (allowing for potential lag in either direction)
        analysis_start = max(tic_start - (max_lag_seconds / 60.0), np.min(fid_time))
        analysis_end = min(tic_end + (max_lag_seconds / 60.0), np.max(fid_time))
        
        if analysis_end <= analysis_start:
            log("Warning: No valid analysis window available")
            return tic_time, tic_signal, 0.0
            
        log(f"Analysis window: {analysis_start:.2f} to {analysis_end:.2f} minutes")
        
        # Step 3: Interpolate to common time base
        common_time = np.linspace(analysis_start, analysis_end, num_points)
        
        # Calculate sampling rate and max lag in points
        dt = (analysis_end - analysis_start) / (num_points - 1) * 60.0  # seconds per point
        max_lag_points = int(max_lag_seconds / dt)
        
        log(f"Common time base: {num_points} points, dt={dt:.6f}s, max lag={max_lag_points} points")
        
        try:
            # Step 4: Interpolate both signals to common time base
            fid_interp = interpolate.interp1d(fid_time, fid_signal, bounds_error=False, fill_value=0)
            tic_interp = interpolate.interp1d(tic_time, tic_signal, bounds_error=False, fill_value=0)
            
            fid_common = fid_interp(common_time)
            tic_common = tic_interp(common_time)
            
            # Step 5: Handle solvent delay by masking
            valid_mask = common_time >= tic_start
            if not np.any(valid_mask):
                log("Warning: No valid data after accounting for solvent delay")
                return tic_time, tic_signal, 0.0
                
            # Focus correlation on regions where both signals have data
            fid_masked = fid_common * valid_mask
            tic_masked = tic_common * valid_mask
            
            # Step 6: Normalize signals for cross-correlation
            nonzero_mask = (fid_masked != 0) & (tic_masked != 0)
            if not np.any(nonzero_mask):
                log("Warning: No non-zero overlapping data points found")
                return tic_time, tic_signal, 0.0
                
            fid_mean = np.mean(fid_masked[nonzero_mask])
            fid_std = np.std(fid_masked[nonzero_mask])
            tic_mean = np.mean(tic_masked[nonzero_mask])
            tic_std = np.std(tic_masked[nonzero_mask])
            
            if fid_std == 0 or tic_std == 0:
                log("Warning: Standard deviation is zero, cannot normalize")
                return tic_time, tic_signal, 0.0
                
            fid_norm = (fid_masked - fid_mean) / fid_std
            tic_norm = (tic_masked - tic_mean) / tic_std
            
            # Replace NaN/inf values
            fid_norm = np.nan_to_num(fid_norm, nan=0.0)
            tic_norm = np.nan_to_num(tic_norm, nan=0.0)
            
            # Step 7: Calculate cross-correlation
            cross_corr = signal.correlate(fid_norm, tic_norm, mode='full')
            
            # Find lag that maximizes correlation
            zero_lag_idx = len(fid_norm) - 1
            correlation_lags = np.arange(len(cross_corr)) - zero_lag_idx
            
            # Step 8: Restrict to reasonable lags
            valid_lags = (correlation_lags >= -max_lag_points) & (correlation_lags <= max_lag_points)
            restricted_corr = cross_corr[valid_lags]
            restricted_lags = correlation_lags[valid_lags]
            
            if len(restricted_corr) == 0:
                log("Warning: No valid lags within specified max_lag_seconds")
                return tic_time, tic_signal, 0.0
                
            # Step 9: Find optimal lag
            best_lag_idx = np.argmax(restricted_corr)
            best_lag_points = restricted_lags[best_lag_idx]
            
            # Convert lag from points to seconds
            lag_seconds = best_lag_points * dt
            log(f"Estimated lag: {lag_seconds:.4f} seconds ({best_lag_points} points)")
            log(f"Max correlation: {restricted_corr[best_lag_idx]:.4f}")
            
            # Step 10: Apply lag to original TIC time
            aligned_tic_time = tic_time - lag_seconds/60.0  # Convert seconds to minutes
            aligned_tic_signal = tic_signal  # Signal values remain unchanged
            
            return aligned_tic_time, aligned_tic_signal, lag_seconds
            
        except Exception as e:
            log(f"Error during signal alignment: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            return tic_time, tic_signal, 0.0