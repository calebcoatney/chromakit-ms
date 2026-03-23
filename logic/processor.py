import numpy as np
from scipy import signal, interpolate
from scipy.signal import medfilt, savgol_filter, find_peaks
from scipy.optimize import curve_fit
from pybaselines import Baseline

class ChromatogramProcessor:
    """Processes chromatogram data with smoothing, baseline subtraction, and peak finding."""

    def _detect_peaks_and_shoulders_derivative(self, x, y, peak_params, shoulder_params, smoothed_y=None):
        """
        Detect peaks and shoulders using derivative analysis.
        Uses noise-adaptive local thresholds for consistent detection
        across heterogeneous peak sizes and varying noise levels.
        
        Args:
            x: Retention time array
            y: Signal intensity array (corrected)
            peak_params: Peak detection parameters from UI
            shoulder_params: Shoulder detection parameters from UI  
            smoothed_y: Pre-smoothed signal (if smoothing was enabled), otherwise None
        """
        from scipy.signal import savgol_filter, find_peaks
        import numpy as np

        # --- Peak detection ---
        peak_prominence = peak_params.get('min_prominence', 0.05)
        peak_width = peak_params.get('min_width', 5)

        signal_for_detection = smoothed_y if smoothed_y is not None else y
        
        signal_range = np.max(signal_for_detection) - np.min(signal_for_detection)
        min_prominence = peak_prominence if peak_prominence > 1 else peak_prominence * signal_range
        
        peak_indices, peak_props = find_peaks(signal_for_detection, prominence=min_prominence, width=peak_width)

        # --- Shoulder detection (noise-adaptive, locally-normalized) ---
        shoulder_indices = []
        shoulder_window = 41
        shoulder_polyorder = 3

        if shoulder_params.get('enabled', False):
            shoulder_window = shoulder_params.get('window_length', 41)
            shoulder_polyorder = shoulder_params.get('polyorder', 3)
            apex_distance = shoulder_params.get('apex_distance', 10)

            # Support both new 'sensitivity' (1-10) and old 'height_factor' (0.01-0.10)
            sensitivity = shoulder_params.get('sensitivity', None)
            if sensitivity is not None:
                # Map sensitivity [1, 10] → SNR multiplier [8.0, 2.5]
                # Higher sensitivity = lower SNR threshold = more detections
                snr_multiplier = 8.0 - (sensitivity - 1) * 5.5 / 9.0
            else:
                height_factor = shoulder_params.get('height_factor', 0.02)
                snr_multiplier = 3.0 + (height_factor - 0.01) * 7.0 / 0.09

            data_length = len(y)
            if shoulder_window >= data_length:
                shoulder_window = min(data_length - 1, 41)
                if shoulder_window % 2 == 0:
                    shoulder_window -= 1
            elif shoulder_window % 2 == 0:
                shoulder_window += 1
            if shoulder_polyorder >= shoulder_window:
                shoulder_polyorder = shoulder_window - 1

            # Use Savgol's built-in derivative computation for better noise rejection.
            # This is mathematically superior to gradient(gradient(smooth(y))) because
            # the polynomial fit provides optimal least-squares derivative estimates.
            dx = np.median(np.diff(x))
            y_shoulder_smooth = savgol_filter(y, shoulder_window, shoulder_polyorder)
            dy = savgol_filter(y, shoulder_window, shoulder_polyorder, deriv=1, delta=dx)
            d2y = savgol_filter(y, shoulder_window, shoulder_polyorder, deriv=2, delta=dx)

            # Noise-adaptive threshold: estimate noise from baseline regions only
            # (peak regions inflate MAD when peaks occupy a large fraction of the signal)
            baseline_mask = y_shoulder_smooth < np.percentile(y_shoulder_smooth, 25)
            if np.sum(baseline_mask) > 20:
                baseline_d2y = d2y[baseline_mask]
                noise_d2y = 1.4826 * np.median(np.abs(baseline_d2y - np.median(baseline_d2y)))
                baseline_dy = dy[baseline_mask]
                noise_dy = 1.4826 * np.median(np.abs(baseline_dy - np.median(baseline_dy)))
            else:
                median_d2y = np.median(d2y)
                noise_d2y = 1.4826 * np.median(np.abs(d2y - median_d2y))
                median_dy = np.median(dy)
                noise_dy = 1.4826 * np.median(np.abs(dy - median_dy))

            noise_threshold = snr_multiplier * noise_d2y

            # Find candidate minima in d2y above noise floor
            shoulder_minima, _ = find_peaks(-d2y, height=max(noise_threshold, 1e-15))

            # Convert apex_distance to time units for sampling-rate independence
            min_time_distance = apex_distance * dx

            # Estimate per-peak exclusion zones from d2y curvature at the apex.
            # For a Gaussian, d2y(apex) = -A/σ², so σ = sqrt(A/|d2y|).
            # Using 1.5σ covers the natural d2y concave region (±σ) with margin.
            peak_exclusion_radii = np.full(len(peak_indices), min_time_distance)
            for pi, p in enumerate(peak_indices):
                apex_val = y_shoulder_smooth[p]
                apex_curv = abs(d2y[p])
                if apex_curv > 1e-15 and apex_val > 0:
                    sigma_est = np.sqrt(apex_val / apex_curv)
                    peak_exclusion_radii[pi] = max(1.5 * sigma_est, min_time_distance)

            # Local context window for signal-relative checks
            local_window_pts = max(len(y) // 20, shoulder_window)
            # Edge margin to avoid Savgol boundary artifacts
            edge_margin = shoulder_window

            for idx in shoulder_minima:
                # Skip candidates near data boundaries
                if idx < edge_margin or idx > len(y) - edge_margin:
                    continue

                # Must be outside every peak's exclusion zone
                in_exclusion = False
                for pi, p in enumerate(peak_indices):
                    if abs(x[idx] - x[p]) < peak_exclusion_radii[pi]:
                        in_exclusion = True
                        break
                if in_exclusion:
                    continue

                # Slope check: a real shoulder sits on the flank of a parent peak,
                # so the first derivative should be non-trivial. Valley minima
                # between isolated peaks have near-zero slope.
                if noise_dy > 0 and abs(dy[idx]) < noise_dy:
                    continue

                # Local signal check: shoulder signal must be meaningful relative
                # to the LOCAL maximum, not the global maximum
                local_start = max(0, idx - local_window_pts)
                local_end = min(len(y), idx + local_window_pts)
                local_max = np.max(y[local_start:local_end])

                if local_max > 0 and y[idx] > 0.1 * local_max:
                    shoulder_indices.append(idx)

            # Deduplicate: when multiple shoulders cluster nearby, keep only the
            # one with the deepest d2y minimum. Use curvature-estimated width of
            # the nearest peak as the dedup radius (shoulders on the same peak
            # should merge).
            if len(shoulder_indices) > 1:
                deduped = []
                used = set()
                for i, si in enumerate(shoulder_indices):
                    if i in used:
                        continue
                    # Find the nearest peak's exclusion radius for dedup distance
                    nearest_radius = min_time_distance
                    for pi, p in enumerate(peak_indices):
                        dist = abs(x[si] - x[p])
                        if dist < 3 * peak_exclusion_radii[pi]:
                            nearest_radius = max(nearest_radius, peak_exclusion_radii[pi])
                    group = [i]
                    for j in range(i + 1, len(shoulder_indices)):
                        if j in used:
                            continue
                        if abs(x[shoulder_indices[j]] - x[si]) < nearest_radius:
                            group.append(j)
                    best = min(group, key=lambda g: d2y[shoulder_indices[g]])
                    deduped.append(shoulder_indices[best])
                    for g in group:
                        used.add(g)
                shoulder_indices = deduped

        # Find shoulder bounds: inner bound from valley detection, outer bound from signal descent
        shoulder_bounds = {}
        if len(shoulder_indices) > 0:
            shoulder_bounds = self._find_shoulder_bounds(
                x, y, shoulder_indices, peak_indices, shoulder_window, shoulder_polyorder
            )

        # Combine peaks and shoulders
        all_peak_indices = np.concatenate([peak_indices, shoulder_indices]) if len(shoulder_indices) > 0 else peak_indices
        all_peak_indices = np.unique(all_peak_indices)
        all_peak_indices = np.sort(all_peak_indices)

        # CRITICAL FIX: Always use original signal for peak intensities
        # Peak positions come from detection signal, but intensities should be from original corrected signal
        peaks_x = x[all_peak_indices] if len(all_peak_indices) > 0 else np.array([])
        peaks_y = y[all_peak_indices] if len(all_peak_indices) > 0 else np.array([])

        peak_metadata = []
        for i, idx in enumerate(all_peak_indices):
            is_shoulder = idx in shoulder_indices
            metadata = {
                'index': int(idx),
                'x': float(peaks_x[i]) if len(peaks_x) > 0 else 0.0,
                'y': float(peaks_y[i]) if len(peaks_y) > 0 else 0.0,
                'is_shoulder': is_shoulder,
                'type': 'shoulder' if is_shoulder else 'peak'
            }
            
            # Add shoulder bounds if this is a shoulder
            if is_shoulder and int(idx) in shoulder_bounds:
                bounds = shoulder_bounds[int(idx)]
                metadata['left_bound'] = bounds['left_bound']
                metadata['right_bound'] = bounds['right_bound']
                
                # Convert bounds to retention time if available
                if bounds['left_bound'] is not None:
                    metadata['left_bound_rt'] = float(x[bounds['left_bound']])
                if bounds['right_bound'] is not None:
                    metadata['right_bound_rt'] = float(x[bounds['right_bound']])
            
            peak_metadata.append(metadata)

        # Return the detection signal used (for debugging/visualization)
        return peaks_x, peaks_y, peak_metadata, signal_for_detection
    
    def __init__(self):
        # Initialize the baseline fitter
        self.baseline_fitter = Baseline()
        
        # Default parameters
        self.default_params = {
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
                'show_corrected': False,
                'method': 'asls',
                'lambda': 1e6,
                'asymmetry': 0.01
            },
            'peaks': {
                'enabled': True,
                'window_length': 41,
                'polyorder': 3,
                'peak_prominence': 0.05,
                'peak_width': 5,
                'shoulder_height_factor': 0.02,
                'apex_shoulder_distance': 10
            }
        }
    
    def process(self, x, y, params=None, ms_range=None):
        """Process chromatogram data with derivative-based peak and shoulder detection.
        
        Args:
            x: X values (time)
            y: Y values (intensity)
            params: Processing parameters
            ms_range: Optional tuple of (min_time, max_time) for MS data range
        """
        import numpy as np
        
        if params is None:
            params = self.default_params
        
        # Make a copy of the input data to avoid modifying the original
        x_values = np.copy(x)
        y_values = np.copy(y)
        
        # STEP 1: Apply smoothing if enabled
        if params['smoothing']['enabled']:
            smoothed_y = self._apply_smoothing(np.copy(y_values), params['smoothing'])
        else:
            smoothed_y = np.copy(y_values)
        
        # STEP 2: Always calculate baseline
        baseline_y, baseline_corrected_y = self._apply_baseline_correction(
            x_values, smoothed_y,
            method=params['baseline']['method'],
            lam=params['baseline']['lambda'],
            fastchrom_params=params['baseline'].get('fastchrom'),
            break_points=params['baseline'].get('break_points', []),
            baseline_offset=params['baseline'].get('baseline_offset', 0.0)
        )
        
        # STEP 3: Find peaks and shoulders using derivative method
        peaks_x = np.array([])
        peaks_y = np.array([])
        peak_metadata = []
        detection_signal = baseline_corrected_y  # Default to corrected signal
        
        if params['peaks']['enabled']:
            # Pass the smoothed signal only if smoothing was enabled
            smoothed_for_detection = smoothed_y if params['smoothing']['enabled'] else None
            
            # Use derivative-based detection, passing both peak and shoulder params
            peaks_x, peaks_y, peak_metadata, detection_signal = self._detect_peaks_and_shoulders_derivative(
                x_values, baseline_corrected_y,
                peak_params=params['peaks'],
                shoulder_params=params.get('shoulders', {'enabled': False}),
                smoothed_y=smoothed_for_detection
            )
            
            # SOLVENT DELAY FILTERING: Filter peaks based on MS data range
            if ms_range is not None and len(peaks_x) > 0:
                ms_min, ms_max = ms_range
                # Create mask for peaks within MS data range (where MS data is available)
                ms_range_mask = (peaks_x >= ms_min) & (peaks_x <= ms_max)
                
                # Count peaks being filtered out
                filtered_count = len(peaks_x) - np.sum(ms_range_mask)
                if filtered_count > 0:
                    print(f"Solvent delay filtering: Removing {filtered_count} peaks outside MS range ({ms_min:.2f}-{ms_max:.2f} min)")
                
                # Apply the filter
                peaks_x = peaks_x[ms_range_mask]
                peaks_y = peaks_y[ms_range_mask]
                
                # Filter peak metadata if it exists
                if peak_metadata and len(peak_metadata) > 0:
                    # Filter metadata using list comprehension since it's a list of dicts
                    filtered_metadata = []
                    for i, keep in enumerate(ms_range_mask):
                        if keep and i < len(peak_metadata):
                            filtered_metadata.append(peak_metadata[i])
                    peak_metadata = filtered_metadata

        # STEP 4: Detect negative peaks if enabled
        if params.get('negative_peaks', {}).get('enabled', False):
            smoothed_for_neg = smoothed_y if params['smoothing']['enabled'] else None
            neg_peaks_x, neg_peaks_y, neg_peak_metadata = self._detect_negative_peaks(
                x_values, baseline_corrected_y,
                neg_peak_params=params['negative_peaks'],
                smoothed_y=smoothed_for_neg
            )

            # Apply same solvent delay filtering as positive peaks
            if ms_range is not None and len(neg_peaks_x) > 0:
                ms_min, ms_max = ms_range
                neg_mask = (neg_peaks_x >= ms_min) & (neg_peaks_x <= ms_max)
                filtered_count = len(neg_peaks_x) - np.sum(neg_mask)
                if filtered_count > 0:
                    print(f"Solvent delay filtering: Removing {filtered_count} negative peaks outside MS range")
                neg_peaks_x = neg_peaks_x[neg_mask]
                neg_peaks_y = neg_peaks_y[neg_mask]
                if neg_peak_metadata:
                    neg_peak_metadata = [m for i, m in enumerate(neg_peak_metadata) if i < len(neg_mask) and neg_mask[i]]

            # Merge negative peaks into the main peak arrays
            if len(neg_peaks_x) > 0:
                if len(peaks_x) > 0:
                    merged_x = np.concatenate([peaks_x, neg_peaks_x])
                    merged_y = np.concatenate([peaks_y, neg_peaks_y])
                    merged_meta = peak_metadata + neg_peak_metadata
                else:
                    merged_x = neg_peaks_x
                    merged_y = neg_peaks_y
                    merged_meta = neg_peak_metadata

                # Sort by retention time
                sort_order = np.argsort(merged_x)
                peaks_x = merged_x[sort_order]
                peaks_y = merged_y[sort_order]
                peak_metadata = [merged_meta[i] for i in sort_order]

        # STEP 5: Range filtering — keep only peaks within user-specified time ranges
        range_filters = params.get('peaks', {}).get('range_filters', [])
        if range_filters and len(peaks_x) > 0:
            range_mask = np.zeros(len(peaks_x), dtype=bool)
            for rf in range_filters:
                if len(rf) == 2:
                    rf_start, rf_end = rf
                    range_mask |= (peaks_x >= rf_start) & (peaks_x <= rf_end)
            
            filtered_count = len(peaks_x) - np.sum(range_mask)
            if filtered_count > 0:
                print(f"Range filtering: Removing {filtered_count} peaks outside specified ranges")
            
            peaks_x = peaks_x[range_mask]
            peaks_y = peaks_y[range_mask]
            if peak_metadata and len(peak_metadata) > 0:
                peak_metadata = [m for i, m in enumerate(peak_metadata) if i < len(range_mask) and range_mask[i]]

        # Return processed data
        return {
            'x': x_values,
            'original_y': y_values,
            'smoothed_y': smoothed_y,
            'baseline_y': baseline_y,
            'corrected_y': baseline_corrected_y,
            'peaks_x': peaks_x,
            'peaks_y': peaks_y,
            'peak_metadata': peak_metadata
        }
    
    def _apply_smoothing(self, y, smoothing_params):
        """Apply smoothing using the selected method.

        Optional median pre-filter removes spike noise before the main smoother.
        Whittaker: asls with p=0.9 so points above the fit (peaks) get high
        weight (preserved) while points below (noise) get low weight.
        Savitzky-Golay: local polynomial fit via scipy.signal.savgol_filter.
        """
        method = smoothing_params.get('method', 'whittaker')
        try:
            # Optional median pre-filter for spike removal
            if smoothing_params.get('median_enabled', False):
                kernel = smoothing_params.get('median_kernel', 5)
                if kernel % 2 == 0:
                    kernel += 1
                if kernel >= len(y):
                    kernel = min(len(y) - 1, 5)
                    if kernel % 2 == 0:
                        kernel -= 1
                y = medfilt(y, kernel_size=kernel)

            if method == 'savgol':
                window = smoothing_params.get('savgol_window', 3)
                polyorder = smoothing_params.get('savgol_polyorder', 1)
                data_length = len(y)
                if window >= data_length:
                    window = min(data_length - 1, 3)
                if window % 2 == 0:
                    window += 1
                if polyorder >= window:
                    polyorder = window - 1
                return savgol_filter(y, window_length=window, polyorder=polyorder)
            else:
                lam = smoothing_params.get('lambda', 1e-1)
                diff_order = smoothing_params.get('diff_order', 1)
                smoothed_y, _ = self.baseline_fitter.asls(
                    data=y, lam=lam, p=0.9, diff_order=diff_order
                )
                return smoothed_y
        except Exception as e:
            print(f"Smoothing failed: {str(e)}")
            return y
    
    def _apply_baseline_correction(self, x, y, method="asls", lam=1e6, fastchrom_params=None, break_points=None, baseline_offset=0.0):
        """Apply baseline correction using the specified method.
        
        If break_points are provided, the signal is split at each break point
        and baselines are fitted independently to each segment.
        """
        # Handle break points: split signal, fit each segment, stitch back
        if break_points and len(break_points) > 0 and x is not None and len(x) > 0:
            # Compute split indices from break point times
            split_indices = []
            for bp in break_points:
                bp_time = bp.get('time', bp) if isinstance(bp, dict) else float(bp)
                bp_tol = bp.get('tolerance', 0.0) if isinstance(bp, dict) else 0.0
                # Find the index closest to the break point time
                idx = int(np.argmin(np.abs(x - bp_time)))
                # Apply tolerance: use the midpoint of the tolerance window
                # The tolerance just gives the user some slack in specifying the exact time
                split_indices.append(idx)
            
            # Sort and deduplicate
            split_indices = sorted(set(split_indices))
            # Filter out boundary indices
            split_indices = [i for i in split_indices if 0 < i < len(x)]
            
            if len(split_indices) > 0:
                print(f"Break point baseline: splitting signal at {len(split_indices)} point(s)")
                # Build segment boundaries: [0, idx1, idx2, ..., len(x)]
                boundaries = [0] + split_indices + [len(x)]
                
                baseline_segments = []
                corrected_segments = []
                
                for seg_i in range(len(boundaries) - 1):
                    seg_start = boundaries[seg_i]
                    seg_end = boundaries[seg_i + 1]
                    seg_y = y[seg_start:seg_end]
                    
                    if len(seg_y) == 0:
                        continue
                    
                    print(f"  Segment {seg_i + 1}: indices [{seg_start}:{seg_end}], length={len(seg_y)}")
                    # Fit baseline to this segment using the same method/params
                    seg_baseline, seg_corrected = self._apply_baseline_correction_single(
                        seg_y, method=method, lam=lam, fastchrom_params=fastchrom_params,
                        baseline_offset=baseline_offset
                    )
                    baseline_segments.append(seg_baseline)
                    corrected_segments.append(seg_corrected)
                
                if len(baseline_segments) > 0:
                    full_baseline = np.concatenate(baseline_segments)
                    full_corrected = np.concatenate(corrected_segments)
                    return full_baseline, full_corrected
                # Fall through to single-segment if something went wrong
        
        return self._apply_baseline_correction_single(y, method=method, lam=lam, fastchrom_params=fastchrom_params, baseline_offset=baseline_offset)
    
    def _apply_baseline_correction_single(self, y, method="asls", lam=1e6, fastchrom_params=None, baseline_offset=0.0):
        """Apply baseline correction to a single signal segment."""
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
            "mixture_model": self.baseline_fitter.mixture_model,  # Mixture Model (spline-based)
            "irsqr": self.baseline_fitter.irsqr,                  # Iterative Reweighted Spline Quantile Regression
            "fastchrom": self.baseline_fitter.fastchrom,           # FastChrom's Baseline Method
        }
        
        # Validate input data
        if y is None:
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
        lam_methods = {"asls", "airpls", "arpls", "mixture_model", "irsqr"}
        
        try:
            # Apply the selected baseline correction method
            if method in lam_methods:
                print(f"Using lambda parameter with {method}")
                # Use a reasonable lambda based on data length
                scaled_lam = lam
                baseline, params = methods[method](y, lam=scaled_lam)
            elif method == "fastchrom":
                fc = fastchrom_params or {}
                kwargs = {}
                if fc.get('half_window') is not None:
                    kwargs['half_window'] = fc['half_window']
                if fc.get('smooth_half_window') is not None:
                    kwargs['smooth_half_window'] = fc['smooth_half_window']
                print(f"Using fastchrom with params: {kwargs}")
                baseline, params = methods[method](y, **kwargs)
            else:
                print(f"Method {method} does not use lambda")
                baseline, params = methods[method](y)
                
            corrected = y - baseline
            
            # Apply baseline offset (positive = baseline lower = areas increase)
            if baseline_offset != 0.0:
                baseline = baseline - baseline_offset
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
    
    def integrate_peaks(self, processed_data=None, rt_table=None, chemstation_area_factor=0.0784, ms_data=None, quality_options=None, peak_groups=None):
        """Integrate peaks in a processed chromatogram.
        
        Args:
            processed_data: Dictionary with processed data, or None to use most recent processed data
            rt_table: Optional table mapping retention times to compounds
            chemstation_area_factor: Area scaling factor to match ChemStation
            ms_data: Optional MS data object for peak quality assessment
            quality_options: Options for peak quality assessment
            peak_groups: Optional list of [start, end] time windows for peak grouping
            
        Returns:
            dict: Dictionary containing integration results
        """
        # Import here to avoid circular imports
        from logic.integration import Integrator
        
        # Use provided processed data or attempt to get it from the current state
        if processed_data is None:
            print("ERROR: No processed data provided to integrate_peaks method")
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
        
        # Check if we have peak data and valid arrays
        if ('peaks_x' not in processed_data or 
            len(processed_data['peaks_x']) == 0 or
            'x' not in processed_data or 
            len(processed_data['x']) == 0):
            print("No peaks detected for integration or invalid data. Enable peak detection first.")
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
            ms_data=ms_data,
            quality_options=quality_options,
            peak_groups=peak_groups
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
    
    def _generate_fitted_curves(self, x_values, peak_fit_data):
        """Generate fitted curves for visualization from peak fit data.
        
        Args:
            x_values (np.ndarray): Original x data array
            peak_fit_data (list): List of fitting data for each peak
            
        Returns:
            list: List of curve dictionaries with 'x' and 'y' arrays
        """
        fitted_curves = []
        
        try:
            for fit_data in peak_fit_data:
                if 'x_window' in fit_data and 'fitted_curve' in fit_data:
                    x_window = fit_data['x_window']
                    fitted_y = fit_data['fitted_curve']
                    
                    if len(x_window) > 0 and len(fitted_y) > 0:
                        fitted_curves.append({
                            'x': x_window,
                            'y': fitted_y,
                            'method': fit_data.get('method', 'unknown'),
                            'quality': fit_data.get('fit_quality', 0.0)
                        })
        except Exception as e:
            print(f"Error generating fitted curves: {e}")
        
        return fitted_curves
    
    def _apply_peak_fitting(self, x, y, peaks_x, peaks_y, fitting_params):
        """Apply peak fitting with derivative-based shoulder detection as primary filter.
        
        Only peaks with clear derivative evidence of convolution get multi-peak fitting.
        Simple asymmetric peaks are left as single peaks with EMG fitting if needed.
        """
        try:
            quality_metric = fitting_params['fit_quality_metric']
            quality_threshold = fitting_params['quality_threshold']
            max_peaks = fitting_params['max_peaks_per_region']
            window_factor = fitting_params['window_factor']
            min_separation = fitting_params['min_peak_separation']
            fit_method = fitting_params.get('fit_method', 'auto')
            
            # Thresholds for determining when to apply multi-peak fitting
            convolution_threshold = fitting_params.get('convolution_threshold', 0.3)
            shoulder_prominence = fitting_params.get('shoulder_prominence', 0.15)
            min_shoulder_height = fitting_params.get('min_shoulder_height', 0.1)
            min_shoulders_for_multipeak = 1  # Need at least 1 clear shoulder

            fitted_peaks_x = []
            fitted_peaks_y = []
            peak_fit_data = []

            print(f"Peak fitting: Analyzing {len(peaks_x)} detected peaks")
            print(f"Convolution threshold: {convolution_threshold}, min shoulders: {min_shoulders_for_multipeak}")

            for i, (peak_x, peak_y) in enumerate(zip(peaks_x, peaks_y)):
                try:
                    print(f"  Processing peak {i+1}: x={peak_x:.3f}, y={peak_y:.1f}")
                    window_size = window_factor * 0.1
                    x_window, y_window, window_indices = self._extract_peak_window(x, y, peak_x, window_size)
                    
                    if len(x_window) < 5:
                        print(f"    Insufficient data, keeping original peak")
                        fitted_peaks_x.append(peak_x)
                        fitted_peaks_y.append(peak_y)
                        peak_fit_data.append({
                            'original_peak': (peak_x, peak_y),
                            'fitted_peaks': [(peak_x, peak_y)],
                            'fit_quality': 1.0,
                            'method': 'insufficient_data',
                            'window_size': len(x_window),
                            'fit_type': 'original'
                        })
                        continue

                    # STEP 1: Derivative analysis to detect convolution
                    shoulders, convolution_score = self._detect_potential_shoulders(
                        x_window, y_window, 
                        prominence_threshold=shoulder_prominence,
                        min_shoulder_height=min_shoulder_height
                    )
                    print(f"    Convolution score: {convolution_score:.3f}, shoulders found: {len(shoulders)}")

                    # STEP 2: Decide fitting strategy based on derivative evidence
                    should_try_multipeak = (convolution_score > convolution_threshold and 
                                          len(shoulders) >= min_shoulders_for_multipeak)
                    
                    if should_try_multipeak:
                        print(f"    Strong convolution evidence - trying multi-peak fitting")
                        
                        # Try multi-peak fitting since we have evidence of convolution
                        multi_fit = self._fit_multi_gaussian(x_window, y_window, max_peaks, min_separation)
                        if multi_fit and len(multi_fit['peaks']) > 1:
                            multi_quality = self._evaluate_fit_quality(y_window, multi_fit['fitted_y'], quality_metric)
                            
                            # Also try single peak for comparison
                            single_fit = self._fit_single_gaussian(x_window, y_window)
                            single_quality = self._evaluate_fit_quality(y_window, single_fit['fitted_y'], quality_metric)
                            
                            # Only use multi-peak if it's significantly better AND we have derivative evidence
                            improvement = multi_quality - single_quality
                            print(f"    Multi-peak quality: {multi_quality:.3f}, single: {single_quality:.3f}, improvement: {improvement:.3f}")
                            
                            if improvement > 0.05:  # Require 5% improvement for multi-peak
                                print(f"    Using multi-peak fit (significant improvement with derivative evidence)")
                                for peak_data in multi_fit['peaks']:
                                    peak_center, peak_height = peak_data[0], peak_data[1]
                                    fitted_peaks_x.append(peak_center)
                                    fitted_peaks_y.append(peak_height)
                                
                                peak_fit_data.append({
                                    'original_peak': (peak_x, peak_y),
                                    'fitted_peaks': [(p[0], p[1]) for p in multi_fit['peaks']],
                                    'detailed_peaks': multi_fit['peaks'],
                                    'fit_quality': multi_quality,
                                    'method': 'multi_gaussian_validated',
                                    'convolution_score': convolution_score,
                                    'shoulders': shoulders,
                                    'window_size': len(x_window),
                                    'x_window': x_window,
                                    'y_window': y_window,
                                    'fitted_curve': multi_fit['fitted_y'],
                                    'fit_type': 'multi_gaussian'
                                })
                                continue
                    
                    # STEP 3: For all other cases, try EMG or single Gaussian
                    print(f"    No strong convolution evidence - using single peak fitting")
                    
                    # Try EMG for asymmetric peaks (good for tailing/fronting)
                    emg_fit = None
                    emg_quality = -1
                    if fit_method in ('emg', 'auto'):
                        popt, emg_y = self._fit_emg(x_window, y_window, baseline=np.min(y_window))
                        if emg_y is not None:
                            emg_quality = self._evaluate_fit_quality(y_window, emg_y, quality_metric)
                    
                    # Always try single Gaussian as fallback
                    single_fit = self._fit_single_gaussian(x_window, y_window)
                    single_quality = self._evaluate_fit_quality(y_window, single_fit['fitted_y'], quality_metric)
                    
                    # Choose best single-peak fit
                    if emg_fit and emg_quality > single_quality + 0.02:  # EMG must be meaningfully better
                        print(f"    Using EMG fit (quality: {emg_quality:.3f} vs Gaussian: {single_quality:.3f})")
                        peak_center = popt[1]  # mu parameter from EMG
                        peak_height = np.interp(peak_center, x_window, y_window)
                        fitted_peaks_x.append(peak_center)
                        fitted_peaks_y.append(peak_height)
                        peak_fit_data.append({
                            'original_peak': (peak_x, peak_y),
                            'fitted_peaks': [(peak_center, peak_height)],
                            'fit_quality': emg_quality,
                            'method': 'emg_asymmetric',
                            'convolution_score': convolution_score,
                            'shoulders': shoulders,
                            'window_size': len(x_window),
                            'x_window': x_window,
                            'y_window': y_window,
                            'fitted_curve': emg_y,
                            'fit_type': 'emg'
                        })
                    else:
                        print(f"    Using Gaussian fit (quality: {single_quality:.3f})")
                        fitted_peaks_x.append(single_fit['peak_center'])
                        fitted_peaks_y.append(single_fit['peak_height'])
                        peak_fit_data.append({
                            'original_peak': (peak_x, peak_y),
                            'fitted_peaks': [(single_fit['peak_center'], single_fit['peak_height'])],
                            'fit_quality': single_quality,
                            'method': 'gaussian_symmetric',
                            'convolution_score': convolution_score,
                            'shoulders': shoulders,
                            'window_size': len(x_window),
                            'x_window': x_window,
                            'y_window': y_window,
                            'fitted_curve': single_fit['fitted_y'],
                            'fit_type': 'gaussian'
                        })

                except Exception as peak_error:
                    print(f"Peak fitting failed for peak {i+1}: {peak_error}")
                    fitted_peaks_x.append(peak_x)
                    fitted_peaks_y.append(peak_y)
                    peak_fit_data.append({
                        'original_peak': (peak_x, peak_y),
                        'fitted_peaks': [(peak_x, peak_y)],
                        'fit_quality': 0.0,
                        'method': 'failed',
                        'error': str(peak_error),
                        'fit_type': 'original'
                    })

            fitted_peaks_x = np.array(fitted_peaks_x)
            fitted_peaks_y = np.array(fitted_peaks_y)
            
            print(f"Peak fitting complete: {len(peaks_x)} → {len(fitted_peaks_x)} peaks")
            multipeak_count = sum(1 for pfd in peak_fit_data if pfd.get('method') == 'multi_gaussian_validated')
            emg_count = sum(1 for pfd in peak_fit_data if pfd.get('method') == 'emg_asymmetric')
            print(f"  Multi-peak fits: {multipeak_count}, EMG fits: {emg_count}")
            
            return fitted_peaks_x, fitted_peaks_y, peak_fit_data

        except Exception as e:
            print(f"Peak fitting failed: {str(e)}")
            return peaks_x, peaks_y, []
    
    def _extract_peak_window(self, x, y, peak_center, window_size):
        """Extract a window of data around a peak for fitting."""
        # Find indices within the window
        window_mask = np.abs(x - peak_center) <= window_size / 2
        window_indices = np.where(window_mask)[0]
        
        if len(window_indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Expand window to include some baseline on both sides
        start_idx = max(0, window_indices[0] - 5)
        end_idx = min(len(x), window_indices[-1] + 6)
        
        return x[start_idx:end_idx], y[start_idx:end_idx], np.arange(start_idx, end_idx)
    
    def _fit_single_gaussian(self, x, y):
        """Fit a single Gaussian to the data."""
        try:
            # Initial parameter estimates
            amplitude = np.max(y)
            center = x[np.argmax(y)]
            sigma = (x[-1] - x[0]) / 6  # Rough estimate
            baseline = np.min(y)
            
            initial_params = [amplitude, center, sigma, baseline]
            
            # Define bounds to keep parameters reasonable
            lower_bounds = [0, x[0], 0.001, 0]
            upper_bounds = [amplitude * 2, x[-1], (x[-1] - x[0]), amplitude]
            
            # Fit the Gaussian
            popt, _ = curve_fit(self._gaussian_with_baseline, x, y, 
                              p0=initial_params, 
                              bounds=(lower_bounds, upper_bounds),
                              maxfev=1000)
            
            fitted_y = self._gaussian_with_baseline(x, *popt)
            
            return {
                'peak_center': popt[1],
                'peak_height': np.interp(popt[1], x, y),  # Interpolate height from fitting data
                'fitted_y': fitted_y,
                'parameters': popt,
                'amplitude': popt[0],  # Store amplitude for quality assessment
                'sigma': popt[2]  # Store sigma for width assessment
            }
            
        except Exception as e:
            # Fallback to simple peak detection
            max_idx = np.argmax(y)
            return {
                'peak_center': x[max_idx],
                'peak_height': y[max_idx],  # Use actual data height
                'fitted_y': np.full_like(y, np.min(y)),
                'parameters': None
            }
    
    def _fit_multi_gaussian(self, x, y, max_peaks, min_separation):
        """Fit multiple Gaussians to the data."""
        try:
            best_fit = None
            best_quality = -1
            
            # Try fitting 2 to max_peaks Gaussians
            for n_peaks in range(2, max_peaks + 1):
                try:
                    fit_result = self._fit_n_gaussians(x, y, n_peaks, min_separation)
                    if fit_result is not None:
                        quality = self._evaluate_fit_quality(y, fit_result['fitted_y'], 'r_squared')
                        if quality > best_quality:
                            best_quality = quality
                            best_fit = fit_result
                except:
                    continue
            
            if best_fit is not None:
                return best_fit
            else:
                # Fallback to single peak
                return self._fit_single_gaussian(x, y)
                
        except Exception as e:
            return self._fit_single_gaussian(x, y)
    
    def _fit_n_gaussians(self, x, y, n_peaks, min_separation):
        """Fit n Gaussians to the data."""
        try:
            # Estimate initial parameters by finding local maxima
            peak_indices, _ = find_peaks(y, distance=max(1, int(min_separation / (x[1] - x[0]))))
            
            if len(peak_indices) < n_peaks:
                # Not enough peaks found, add some at regular intervals
                additional_peaks = []
                for i in range(n_peaks - len(peak_indices)):
                    idx = len(x) // (n_peaks + 1) * (i + 1)
                    additional_peaks.append(idx)
                peak_indices = np.concatenate([peak_indices, additional_peaks])
            
            # Sort and take the highest n_peaks
            peak_heights = y[peak_indices]
            sorted_indices = np.argsort(peak_heights)[::-1]
            peak_indices = peak_indices[sorted_indices[:n_peaks]]
            peak_indices = np.sort(peak_indices)
            
            # Build initial parameters
            initial_params = []
            lower_bounds = []
            upper_bounds = []
            
            baseline = np.min(y)
            
            for idx in peak_indices:
                amplitude = y[idx] - baseline
                center = x[idx]
                sigma = (x[-1] - x[0]) / (n_peaks * 4)  # Narrower for multiple peaks
                
                initial_params.extend([amplitude, center, sigma])
                lower_bounds.extend([0, x[0], 0.001])
                upper_bounds.extend([amplitude * 2, x[-1], (x[-1] - x[0]) / 2])
            
            # Add baseline parameter
            initial_params.append(baseline)
            lower_bounds.append(0)
            upper_bounds.append(np.max(y))
            
            # Fit the multi-Gaussian
            popt, _ = curve_fit(lambda x_data, *params: self._multi_gaussian(x_data, params, n_peaks),
                              x, y, 
                              p0=initial_params,
                              bounds=(lower_bounds, upper_bounds),
                              maxfev=2000)
            
            fitted_y = self._multi_gaussian(x, popt, n_peaks)
            
            # Extract peak information
            peaks = []
            for i in range(n_peaks):
                amplitude = popt[i * 3]
                center = popt[i * 3 + 1]
                sigma = popt[i * 3 + 2]
                
                # Interpolate height from fitting data
                height = np.interp(center, x, y)
                peaks.append((center, height, amplitude, sigma))  # Store amplitude and sigma too
            
            # Filter out peaks that are too close together and outside data bounds
            filtered_peaks = []
            sorted_peaks = sorted(peaks, key=lambda p: p[0])  # Sort by position
            for peak in sorted_peaks:
                center = peak[0]  # Get center from tuple
                # Check if peak is within data bounds
                if x[0] <= center <= x[-1]:
                    # Check minimum separation only if we have other peaks
                    too_close = False
                    for existing_peak in filtered_peaks:
                        if abs(center - existing_peak[0]) < min_separation:
                            # Keep the higher peak
                            if peak[1] > existing_peak[1]:  # Compare heights
                                filtered_peaks.remove(existing_peak)
                            else:
                                too_close = True
                            break
                    
                    if not too_close:
                        filtered_peaks.append(peak)
            
            return {
                'peaks': filtered_peaks,
                'fitted_y': fitted_y,
                'parameters': popt,
                'n_peaks': len(filtered_peaks)
            }
            
        except Exception as e:
            return self._fit_single_gaussian(x, y)
    
    def _gaussian_with_baseline(self, x, amplitude, center, sigma, baseline):
        """Gaussian function with baseline."""
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + baseline
    
    def _multi_gaussian(self, x, params, n_peaks):
        """Sum of multiple Gaussians with shared baseline."""
        result = np.zeros_like(x)
        
        # Add each Gaussian
        for i in range(n_peaks):
            amplitude = params[i * 3]
            center = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            result += amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
        
        # Add baseline
        baseline = params[-1]
        result += baseline
        
        return result
    
    def _evaluate_fit_quality(self, y_true, y_fitted, metric):
        """Evaluate the quality of a fit using the specified metric."""
        try:
            if metric == 'r_squared':
                # R-squared (coefficient of determination)
                ss_res = np.sum((y_true - y_fitted) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                if ss_tot == 0:
                    return 1.0 if ss_res == 0 else 0.0
                return 1 - (ss_res / ss_tot)
                
            elif metric == 'aic':
                # Akaike Information Criterion (lower is better, so return 1-normalized AIC)
                n = len(y_true)
                mse = np.mean((y_true - y_fitted) ** 2)
                if mse <= 0:
                    return 1.0
                aic = n * np.log(mse) + 2 * 4  # Assuming 4 parameters for Gaussian
                # Normalize to 0-1 range (this is approximate)
                max_aic = n * np.log(np.var(y_true)) + 2 * 4
                return max(0, 1 - aic / max_aic)
                
            elif metric == 'residual_rmse':
                # Root Mean Square Error (lower is better, so return 1-normalized RMSE)
                rmse = np.sqrt(np.mean((y_true - y_fitted) ** 2))
                max_rmse = np.std(y_true)
                if max_rmse == 0:
                    return 1.0 if rmse == 0 else 0.0
                return max(0, 1 - rmse / max_rmse)
                
            else:
                # Default to R-squared
                return self._evaluate_fit_quality(y_true, y_fitted, 'r_squared')
                
        except Exception as e:
            print(f"Error evaluating fit quality: {e}")
            return 0.0
    
    def _detect_potential_shoulders(self, x, y, prominence_threshold=0.15, min_shoulder_height=0.1):
        """
        Detect potential shoulders within a local peak window for convolution analysis.
        Used by peak fitting to decide if multi-peak fitting is warranted.
        
        Args:
            x: x-values of the peak window
            y: y-values of the peak window
            prominence_threshold: minimum relative prominence of d2y feature
            min_shoulder_height: minimum signal height as fraction of local peak max
            
        Returns:
            tuple: (list of shoulder dicts, convolution_score 0-1)
        """
        from scipy.signal import savgol_filter, find_peaks
        import numpy as np
        
        if len(x) < 10:
            return [], 0.0
        
        window = min(len(x) // 3, 21)
        if window < 5:
            window = 5
        if window % 2 == 0:
            window += 1
        polyorder = min(3, window - 1)
        
        y_smooth = savgol_filter(y, window, polyorder)
        dx = np.median(np.diff(x))
        d2y = savgol_filter(y, window, polyorder, deriv=2, delta=dx)
        
        # Noise-adaptive threshold from baseline regions
        baseline_mask = y_smooth < np.percentile(y_smooth, 25)
        if np.sum(baseline_mask) > 10:
            baseline_d2y = d2y[baseline_mask]
            noise_d2y = 1.4826 * np.median(np.abs(baseline_d2y - np.median(baseline_d2y)))
        else:
            noise_d2y = 1.4826 * np.median(np.abs(d2y - np.median(d2y)))
        min_height = max(3.0 * noise_d2y, prominence_threshold * np.abs(d2y).max())
        
        shoulder_minima, _ = find_peaks(-d2y, height=min_height)
        
        peak_max = np.max(y)
        shoulders = []
        for idx in shoulder_minima:
            if peak_max > 0 and y[idx] > min_shoulder_height * peak_max:
                shoulders.append({
                    'index': int(idx),
                    'x': float(x[idx]),
                    'y': float(y[idx]),
                    'd2y_depth': float(-d2y[idx])
                })
        
        if len(shoulders) == 0:
            convolution_score = 0.0
        else:
            max_depth = max(s['d2y_depth'] for s in shoulders)
            d2y_range = np.max(np.abs(d2y))
            convolution_score = min(max_depth / d2y_range if d2y_range > 0 else 0.0, 1.0)
        
        return shoulders, convolution_score
    
    def _find_shoulder_bounds(self, x, y, shoulder_indices, peak_indices, shoulder_window=41, shoulder_polyorder=3):
        """
        Find integration bounds for shoulder peaks.
        
        Uses asymmetric bound logic:
        - Inner bound (toward parent peak): d2y inflection point (zero-crossing)
          between shoulder and parent — the natural curvature boundary
        - Outer bound (away from parent): walks outward until signal descends to 
          near-baseline or reaches a local minimum / the next peak's territory.
        
        Args:
            x: Retention time array
            y: Signal intensity array (baseline-corrected, so baseline ≈ 0)
            shoulder_indices: List of shoulder apex indices
            peak_indices: Array of parent peak indices
            shoulder_window: Window for smoothing
            shoulder_polyorder: Polynomial order for smoothing
            
        Returns:
            dict: shoulder_index -> {'left_bound': idx, 'right_bound': idx}
        """
        from scipy.signal import savgol_filter
        import numpy as np
        
        if len(shoulder_indices) == 0:
            return {}
            
        data_length = len(y)
        if shoulder_window >= data_length:
            shoulder_window = min(data_length - 1, 41)
            if shoulder_window % 2 == 0:
                shoulder_window -= 1
        elif shoulder_window % 2 == 0:
            shoulder_window += 1
        if shoulder_polyorder >= shoulder_window:
            shoulder_polyorder = shoulder_window - 1
        
        y_smooth = savgol_filter(y, shoulder_window, shoulder_polyorder)
        dx = np.median(np.diff(x))
        d2y = savgol_filter(y, shoulder_window, shoulder_polyorder, deriv=2, delta=dx)
        
        shoulder_bounds = {}
        
        for shoulder_idx in shoulder_indices:
            shoulder_height = y_smooth[shoulder_idx]
            
            # Determine which side has the parent peak
            if len(peak_indices) > 0:
                distances = np.abs(peak_indices - shoulder_idx)
                nearest_peak_idx = peak_indices[np.argmin(distances)]
            else:
                nearest_peak_idx = 0
            
            parent_is_left = nearest_peak_idx < shoulder_idx
            
            # --- Inner bound (toward parent peak): inflection point ---
            # The d2y zero-crossing between parent and shoulder marks where
            # curvature transitions from one peak's domain to the other.
            # For overlapping peaks: d2y goes negative (parent) → positive
            # (saddle) → negative (shoulder). The zero-crossing closest to
            # the shoulder is where the shoulder's curvature begins.
            if parent_is_left:
                inner_start, inner_end = nearest_peak_idx, shoulder_idx
            else:
                inner_start, inner_end = shoulder_idx, nearest_peak_idx
            
            inner_bound = (inner_start + inner_end) // 2  # fallback: midpoint
            
            if inner_end - inner_start > 2:
                region_d2y = d2y[inner_start:inner_end]
                
                # Find d2y zero-crossings (sign changes) in the inter-peak region
                zero_crossings = []
                for k in range(len(region_d2y) - 1):
                    if region_d2y[k] * region_d2y[k + 1] < 0:
                        zero_crossings.append(inner_start + k)
                
                if zero_crossings:
                    # Use the zero-crossing closest to the shoulder
                    if parent_is_left:
                        inner_bound = zero_crossings[-1]
                    else:
                        inner_bound = zero_crossings[0]
                else:
                    # No zero-crossing: shoulder is very subtle (d2y stays negative).
                    # Use the point of maximum d2y (least negative curvature) as the
                    # best approximation of the inflection-like transition.
                    inner_bound = inner_start + np.argmax(region_d2y)
            
            # --- Outer bound (away from parent peak): walk until signal returns to baseline ---
            descent_threshold = 0.02  # stop at 2% of shoulder height above baseline
            target_level = descent_threshold * shoulder_height
            
            if parent_is_left:
                # Walk right from shoulder apex
                outer_bound = shoulder_idx
                next_peaks = peak_indices[peak_indices > shoulder_idx]
                for j in range(shoulder_idx + 1, data_length):
                    if len(next_peaks) > 0:
                        next_peak = next_peaks[0]
                        if j >= (shoulder_idx + next_peak) // 2:
                            outer_bound = j
                            break
                    if y_smooth[j] <= target_level:
                        outer_bound = j
                        break
                    if j > shoulder_idx + 2 and y_smooth[j] > y_smooth[j - 1] and y_smooth[j - 1] < y_smooth[j - 2]:
                        outer_bound = j - 1
                        break
                    outer_bound = j
            else:
                # Walk left from shoulder apex
                outer_bound = shoulder_idx
                prev_peaks = peak_indices[peak_indices < shoulder_idx]
                for j in range(shoulder_idx - 1, -1, -1):
                    if len(prev_peaks) > 0:
                        prev_peak = prev_peaks[-1]
                        if j <= (prev_peak + shoulder_idx) // 2:
                            outer_bound = j
                            break
                    if y_smooth[j] <= target_level:
                        outer_bound = j
                        break
                    if j < shoulder_idx - 2 and y_smooth[j] > y_smooth[j + 1] and y_smooth[j + 1] < y_smooth[j + 2]:
                        outer_bound = j + 1
                        break
                    outer_bound = j
            
            # Assign to left/right based on parent direction
            if parent_is_left:
                bounds = {'left_bound': int(inner_bound), 'right_bound': int(outer_bound)}
            else:
                bounds = {'left_bound': int(outer_bound), 'right_bound': int(inner_bound)}
            
            shoulder_bounds[int(shoulder_idx)] = bounds
            
        return shoulder_bounds

    def _detect_negative_peaks(self, x, y, neg_peak_params, smoothed_y=None):
        """
        Detect negative peaks by running find_peaks on the negated signal.

        Args:
            x: Retention time array
            y: Signal intensity array (baseline-corrected)
            neg_peak_params: Negative peak detection parameters from UI
            smoothed_y: Pre-smoothed signal (if smoothing was enabled), otherwise None

        Returns:
            tuple: (neg_peaks_x, neg_peaks_y, neg_peak_metadata)
        """
        from scipy.signal import find_peaks
        import numpy as np

        neg_prominence = neg_peak_params.get('min_prominence', 1e5)

        # Use the smoothed signal for detection if available
        signal_for_detection = smoothed_y if smoothed_y is not None else y

        # Negate the signal so that negative peaks become positive peaks
        negated = -signal_for_detection

        # Calculate effective prominence
        signal_range = np.max(negated) - np.min(negated)
        effective_prominence = neg_prominence if neg_prominence > 1 else neg_prominence * signal_range

        # Find peaks on the negated signal
        peak_indices, peak_props = find_peaks(negated, prominence=effective_prominence)

        if len(peak_indices) == 0:
            return np.array([]), np.array([]), []

        neg_peaks_x = x[peak_indices]
        # Use original corrected signal values (not negated) for the peak y-values
        neg_peaks_y = y[peak_indices]

        neg_peak_metadata = []
        for i, idx in enumerate(peak_indices):
            metadata = {
                'index': int(idx),
                'x': float(neg_peaks_x[i]),
                'y': float(neg_peaks_y[i]),
                'is_shoulder': False,
                'is_negative': True,
                'type': 'negative_peak'
            }
            neg_peak_metadata.append(metadata)

        return neg_peaks_x, neg_peaks_y, neg_peak_metadata