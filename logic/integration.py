import numpy as np
from scipy.integrate import simpson
from scipy.stats import skew
import pandas as pd

from logic.feature import Feature

class ChromatographicPeak(Feature):
    """Chromatographic peak. Subclass of Feature with RT, MS, and quantitation fields.

    All original field names are preserved (retention_time, start_time, etc.).
    The base Feature.position property is overridden to return retention_time.
    """
    
    def __init__(self, compound_id, peak_number, retention_time, 
                 integrator, width, area, start_time, end_time,
                 start_index=None, end_index=None):
        """Initialize a ChromatographicPeak object."""
        # Initialize Feature base with generic fields
        super().__init__(
            feature_id=peak_number,
            position=retention_time,
            position_units="min",
            area=area,
            width=width,
            start=start_time,
            end=end_time,
            start_index=start_index or 0,
            end_index=end_index or 0,
        )
        # Chromatography-specific fields — names unchanged from original Peak
        self.compound_id = compound_id
        self.peak_number = peak_number
        self.retention_time = retention_time
        self.integrator = integrator
        self.start_time = start_time
        self.end_time = end_time
        
        # Add MS match properties
        self.compound_name = None
        self.match_score = None
        self.casno = None
        
        # Add quality assessment properties
        self.asymmetry = None
        self.spectral_coherence = None
        self.is_convoluted = False
        self.quality_issues = []
        
        # Add shoulder flag
        self.is_shoulder = False

        # Add negative peak flag
        self.is_negative = False

        # Add saturation properties
        self.is_saturated = False
        self.saturation_level = None

        # Add grouping properties
        self.is_grouped = False
        self.grouped_peak_count = None
        
        # Add quantitation properties (Polyarc + IS method)
        self.mol_C = None  # Moles of carbon
        self.mol_C_percent = None  # Mole percent of carbon
        self.num_carbons = None  # Number of carbons in molecule
        self.mol = None  # Moles of compound
        self.mass_mg = None  # Mass in milligrams
        self.mol_percent = None  # Mole percentage
        self.wt_percent = None  # Weight percentage
    
    @property
    def position(self) -> float:
        """Returns retention_time, satisfying the Feature.position interface."""
        return self.retention_time

    @staticmethod
    def column_headers() -> list:
        return ['Compound ID', 'Peak #', 'Ret Time', 'Integrator', 'Width', 'Area', 'Start Time', 'End Time']

    def as_row(self):
        """Return peak data as a row for display in a table."""
        return [
            self.compound_id, 
            self.peak_number, 
            round(self.retention_time, 3),
            self.integrator, 
            round(self.width, 3), 
            round(self.area, 1), 
            round(self.start_time, 3), 
            round(self.end_time, 3)
        ]
    
    def as_dict(self):
        """Return a dictionary representation of the peak."""
        result = {
            'compound_id': self.compound_id,
            'peak_number': self.peak_number,
            'retention_time': self.retention_time,
            'integrator': self.integrator,
            'width': self.width,
            'area': self.area,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'is_convoluted': self.is_convoluted,
            'asymmetry': self.asymmetry,
            'spectral_coherence': self.spectral_coherence,
            'is_saturated': self.is_saturated,
            'saturation_level': self.saturation_level,
            'is_shoulder': getattr(self, 'is_shoulder', False),
            'is_negative': getattr(self, 'is_negative', False),
            'is_grouped': getattr(self, 'is_grouped', False),
            'grouped_peak_count': getattr(self, 'grouped_peak_count', None)
        }
        
        # Add quality issues
        if hasattr(self, 'quality_issues') and self.quality_issues:
            result['quality_issues'] = self.quality_issues
            
        # Add fields using the exact same names as in the notebook
        if hasattr(self, 'Compound_ID'):
            result['Compound ID'] = self.Compound_ID
            
        if hasattr(self, 'Qual'):
            result['Qual'] = self.Qual
            
        if hasattr(self, 'casno'):
            result['casno'] = self.casno
        
        # Add quantitation fields
        if hasattr(self, 'mol_C') and self.mol_C is not None:
            result['mol_C'] = self.mol_C
            
        if hasattr(self, 'num_carbons') and self.num_carbons is not None:
            result['num_carbons'] = self.num_carbons
            
        if hasattr(self, 'mol') and self.mol is not None:
            result['mol'] = self.mol
            
        if hasattr(self, 'mass_mg') and self.mass_mg is not None:
            result['mass_mg'] = self.mass_mg
            
        if hasattr(self, 'mol_percent') and self.mol_percent is not None:
            result['mol_percent'] = self.mol_percent
            
        if hasattr(self, 'wt_percent') and self.wt_percent is not None:
            result['wt_percent'] = self.wt_percent
            
        return result
    
    @property
    def apex_time(self):
        """Get the retention time at the peak apex."""
        return self.retention_time

Peak = ChromatographicPeak  # backward-compatibility alias — no call sites need to change

# Register ChromatographicPeak as the feature_class for gc/gcms profiles
from logic.signal_profiles import _update_chromatographic_profiles
_update_chromatographic_profiles(ChromatographicPeak)

class Integrator:
    """Provides functionality for integrating chromatographic peaks."""
    
    @staticmethod
    def identify_compound(retention_time, rt_table=None):
        """Identify a compound based on retention time.
        
        Args:
            retention_time: Retention time to match
            rt_table: Dictionary or DataFrame mapping retention times to compounds
            
        Returns:
            str: Identified compound or 'Unknown'
        """
        if rt_table is None:
            return f"Unknown ({retention_time:.3f})"
            
        # If rt_table is a DataFrame
        if isinstance(rt_table, pd.DataFrame):
            # Find the closest retention time within a tolerance
            # This is a simple implementation that can be refined
            tolerance = 0.1  # Tolerance in minutes
            closest_match = None
            min_diff = float('inf')
            
            for _, row in rt_table.iterrows():
                if 'retention_time' in row:
                    diff = abs(row['retention_time'] - retention_time)
                    if diff < min_diff and diff < tolerance:
                        min_diff = diff
                        closest_match = row.get('compound_name', f"Unknown ({retention_time:.3f})")
            
            return closest_match if closest_match else f"Unknown ({retention_time:.3f})"
            
        # If rt_table is a dictionary with retention time ranges
        elif isinstance(rt_table, dict):
            for compound, rt_range in rt_table.items():
                if isinstance(rt_range, tuple) and len(rt_range) == 2:
                    if rt_range[0] <= retention_time <= rt_range[1]:
                        return compound
                elif isinstance(rt_range, (int, float)):
                    # If exact match or within tolerance
                    if abs(rt_range - retention_time) <= 0.1:
                        return compound
            
            return f"Unknown ({retention_time:.3f})"
        
        # If rt_table format is not supported
        return f"Unknown ({retention_time:.3f})"
    
    @staticmethod
    def _calculate_derivatives(x, y, window_length=41, polyorder=3):
        """Calculate first and second derivatives using Savitzky-Golay filter.
        
        Args:
            x: Array of x values
            y: Array of y values
            window_length: Window length for Savitzky-Golay filter
            polyorder: Polynomial order for Savitzky-Golay filter
            
        Returns:
            tuple: (dy, d2y) first and second derivatives
        """
        from scipy.signal import savgol_filter
        
        # Ensure valid smoothing parameters
        data_length = len(y)
        if window_length >= data_length:
            window_length = min(data_length - 1, 41)
            if window_length % 2 == 0:
                window_length -= 1
        elif window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1
            
        # Smooth signal for derivative calculation
        y_smooth = savgol_filter(y, window_length, polyorder)
        
        # Calculate derivatives
        dy = np.gradient(y_smooth, x)
        d2y = np.gradient(dy, x)
        
        return dy, d2y
    
    @staticmethod
    def _find_second_derivative_bounds(x, y, apex_idx, dy=None, d2y=None, window_length=41, polyorder=3):
        """Find integration bounds using second derivative maxima.
        
        Args:
            x: Array of x values
            y: Array of y values
            apex_idx: Index of the peak apex
            dy: First derivative (optional, will calculate if not provided)
            d2y: Second derivative (optional, will calculate if not provided)
            window_length: Window length for Savitzky-Golay filter
            polyorder: Polynomial order for Savitzky-Golay filter
            
        Returns:
            tuple: (left_bound_idx, right_bound_idx)
        """
        from scipy.signal import find_peaks
        
        # Calculate derivatives if not provided
        if dy is None or d2y is None:
            dy, d2y = Integrator._calculate_derivatives(x, y, window_length, polyorder)
        
        # Find local maxima in second derivative (curvature change points)
        maxima_indices, _ = find_peaks(d2y)
        
        # Find left bound: largest maxima index < apex_idx
        left_bound = None
        left_candidates = maxima_indices[maxima_indices < apex_idx]
        if len(left_candidates) > 0:
            left_bound = int(np.max(left_candidates))
        
        # Find right bound: smallest maxima index > apex_idx
        right_bound = None
        right_candidates = maxima_indices[maxima_indices > apex_idx]
        if len(right_candidates) > 0:
            right_bound = int(np.min(right_candidates))
        
        # Fallbacks if bounds not found
        if left_bound is None:
            left_bound = max(0, apex_idx - 20)  # Default to 20 points left
        if right_bound is None:
            right_bound = min(len(y) - 1, apex_idx + 20)  # Default to 20 points right
        
        return left_bound, right_bound
    
    @staticmethod
    def _apply_peak_grouping(peaks_list, peak_groups, x, y, baseline_y,
                              x_peaks, y_peaks, baseline_peaks,
                              ret_times, integrated_areas, integration_bounds, profile=None):
        """Merge peaks within user-specified group windows into composite peaks.
        
        Args:
            peaks_list: List of Peak objects from integration
            peak_groups: List of [start, end] time windows
            x: Full chromatogram time array
            y: Full chromatogram signal array (for finding apex)
            baseline_y: Full baseline array
            x_peaks, y_peaks, baseline_peaks: Per-peak data arrays
            ret_times, integrated_areas, integration_bounds: Per-peak summary lists
            profile: Optional SignalProfile for feature class selection
            
        Returns:
            Tuple of updated (peaks_list, x_peaks, y_peaks, baseline_peaks,
                              ret_times, integrated_areas, integration_bounds)
        """
        feature_cls = profile.feature_class if (profile is not None and profile.feature_class is not None) else Peak

        # Track which peaks get consumed by a group
        consumed_indices = set()
        grouped_peaks = []  # (sort_key, Peak, x_peak, y_peak, baseline_peak)
        
        for group in peak_groups:
            if len(group) != 2:
                continue
            g_start, g_end = float(group[0]), float(group[1])
            
            # Find all peaks whose apex RT falls within this group window
            member_indices = []
            for i, peak in enumerate(peaks_list):
                if g_start <= peak.retention_time <= g_end and i not in consumed_indices:
                    member_indices.append(i)
            
            if len(member_indices) == 0:
                continue
            
            consumed_indices.update(member_indices)
            
            # Compute composite values
            total_area = sum(peaks_list[i].area for i in member_indices)
            
            # Apex = time of max signal within the group window
            window_mask = (x >= g_start) & (x <= g_end)
            if np.any(window_mask):
                window_y = y[window_mask]
                window_x = x[window_mask]
                apex_idx_in_window = np.argmax(window_y)
                apex_rt = float(window_x[apex_idx_in_window])
            else:
                # Fallback: use the peak with the largest area
                best_i = max(member_indices, key=lambda i: peaks_list[i].area)
                apex_rt = peaks_list[best_i].retention_time
            
            # Build start/end indices from the chromatogram arrays
            start_idx = np.searchsorted(x, g_start, side='left')
            end_idx = np.searchsorted(x, g_end, side='right') - 1
            start_idx = max(0, start_idx)
            end_idx = min(len(x) - 1, end_idx)
            
            # Create composite Peak
            width = g_end - g_start
            compound_id = "Group"
            if feature_cls is Peak or issubclass(feature_cls, ChromatographicPeak):
                composite = feature_cls(
                    compound_id=compound_id,
                    peak_number=0,
                    retention_time=apex_rt,
                    integrator='py',
                    width=width,
                    area=total_area,
                    start_time=g_start,
                    end_time=g_end,
                    start_index=start_idx,
                    end_index=end_idx
                )
            else:
                composite = feature_cls(
                    feature_id=0,
                    position=apex_rt,
                    position_units=profile.x_label if profile else "",
                    area=total_area,
                    width=width,
                    start=g_start,
                    end=g_end,
                    start_index=start_idx,
                    end_index=end_idx,
                )
            composite.is_grouped = True
            composite.grouped_peak_count = len(member_indices)
            
            # Build per-peak arrays for the group window
            gx = x[start_idx:end_idx + 1]
            gy = y[start_idx:end_idx + 1]
            gb = baseline_y[start_idx:end_idx + 1]
            
            grouped_peaks.append((g_start, composite, gx, gy, gb))
        
        # Build final lists: ungrouped peaks + grouped peaks, sorted by RT
        new_peaks = []
        new_x_peaks = []
        new_y_peaks = []
        new_baseline_peaks = []
        new_ret_times = []
        new_areas = []
        new_bounds = []
        
        # Add ungrouped peaks
        for i, peak in enumerate(peaks_list):
            if i not in consumed_indices:
                new_peaks.append((peak.retention_time, peak, x_peaks[i], y_peaks[i], baseline_peaks[i]))
        
        # Add grouped peaks
        for sort_key, composite, gx, gy, gb in grouped_peaks:
            new_peaks.append((sort_key, composite, gx, gy, gb))
        
        # Sort by retention time / start time
        new_peaks.sort(key=lambda t: t[0])
        
        # Unpack and re-number
        final_peaks = []
        for idx, (_, peak, xp, yp, bp) in enumerate(new_peaks):
            peak.peak_number = idx + 1
            final_peaks.append(peak)
            new_x_peaks.append(xp)
            new_y_peaks.append(yp)
            new_baseline_peaks.append(bp)
            new_ret_times.append(peak.retention_time)
            new_areas.append(peak.area)
            new_bounds.append((peak.start_time, peak.end_time))
        
        if len(consumed_indices) > 0:
            print(f"Peak grouping: Merged {len(consumed_indices)} peaks into {len(grouped_peaks)} groups")
        
        return final_peaks, new_x_peaks, new_y_peaks, new_baseline_peaks, new_ret_times, new_areas, new_bounds
    
    @staticmethod
    def integrate(processed_data, rt_table=None, chemstation_area_factor=0.0784, verbose=True, ms_data=None, quality_options=None, peak_groups=None, profile=None):
        """Integrate peaks in a chromatogram.
        
        Args:
            processed_data: Dictionary containing the processed chromatogram data
            rt_table: Dictionary or DataFrame mapping retention times to compounds
            chemstation_area_factor: Area scaling factor to match ChemStation
            verbose: Whether to print integration results
            ms_data: Optional MS data object for peak quality assessment
            quality_options: Options for peak quality assessment
            peak_groups: Optional list of [start, end] time windows for peak grouping
            profile: Optional SignalProfile for feature class selection
            
        Returns:
            dict: Dictionary containing integration results
        """
        feature_cls = profile.feature_class if (profile is not None and profile.feature_class is not None) else Peak
        # Extract relevant values from the processed data
        x = processed_data['x']
        y = processed_data.get('smoothed_y', processed_data.get('original_y', processed_data.get('corrected_y')))
        baseline_y = processed_data['baseline_y']
        peaks_x = processed_data['peaks_x']
        peaks_y = processed_data['peaks_y']
        
        # Debug: Check if we have peaks to integrate
        if verbose:
            print(f"Integration starting: Found {len(peaks_x)} peaks to integrate")
            if len(peaks_x) == 0:
                print("ERROR: No peaks found in processed_data!")
                print(f"Available keys in processed_data: {list(processed_data.keys())}")
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
        
        # Check if we have peak metadata with shoulder information
        has_shoulder_info = 'peak_metadata' in processed_data and len(processed_data.get('peak_metadata', [])) > 0
        peak_metadata = processed_data.get('peak_metadata', [])
        
        # Use the original signal for derivative calculation (shoulder detection).
        # Bound-walking threshold and valley-finding use integration_signal (smoothed +
        # baseline-corrected) to prevent raw noise spikes from triggering premature stops.
        bounds_detection_signal = processed_data.get('original_y', y)
        integration_signal = processed_data.get('corrected_y', y)  # Keep for area calculation
        
        # Calculate derivatives for bound detection using the bounds detection signal
        dy, d2y = Integrator._calculate_derivatives(x, bounds_detection_signal)
        
        # Integration criteria
        criteria = ['threshold', 'minimum']
        
        # Initialize lists to store the integration data
        peaks_list = []
        ret_times = []
        integrated_areas = []
        integration_bounds = []
        x_peaks = []
        y_peaks = []
        baseline_peaks = []
        
        # First, identify shoulders and their bounds
        shoulder_bounds = {}  # Index to (left_bound, right_bound)
        shoulder_indices = []  # Indices of shoulders
        
        if has_shoulder_info:
            for i, meta in enumerate(peak_metadata):
                if meta.get('is_shoulder', False):
                    idx = meta.get('index')
                    shoulder_indices.append(idx)
                    
                    # Get pre-calculated bounds if available
                    left_bound = meta.get('left_bound')
                    right_bound = meta.get('right_bound')
                    
                    # If bounds not in metadata, calculate them using second derivative
                    if left_bound is None or right_bound is None:
                        left_bound, right_bound = Integrator._find_second_derivative_bounds(
                            x, bounds_detection_signal, idx, dy, d2y
                        )
                    
                    shoulder_bounds[idx] = (left_bound, right_bound)
        
        # Iterate through each peak for integration
        for i, (apex_x, detected_apex_y) in enumerate(zip(peaks_x, peaks_y)):
            # Find the index of the apex in x and y arrays
            peak_idx = np.where(x == apex_x)[0][0]
            baseline_at_apex = baseline_y[peak_idx]
            
            # IMPORTANT: Use the bounds detection signal value at apex for threshold calculation
            # This ensures consistency between apex height calculation and integration bounds
            apex_y = bounds_detection_signal[peak_idx]
            
            # Define peak number early for use in verbose output
            peak_number = i + 1
            
            # Check if this peak is a shoulder
            is_shoulder = peak_idx in shoulder_indices

            # Check if this peak is a negative peak
            is_negative = False
            if has_shoulder_info and i < len(peak_metadata):
                is_negative = peak_metadata[i].get('is_negative', False)

            # Get bounds based on peak type
            if is_shoulder and peak_idx in shoulder_bounds:
                # If it's a shoulder, use the second derivative bounds
                left_bound, right_bound = shoulder_bounds[peak_idx]
            else:
                # For regular peaks, use standard approach but with shoulder awareness
                # Initialize left and right bounds based on peak apex
                left_bound = peak_idx
                right_bound = peak_idx
                
                # Define the range for min_left and min_right calculations
                if i > 0:  # If not the first peak
                    # Find the index of the previous peak's x value in the x array
                    start_idx = np.where(x == peaks_x[i - 1])[0][0] if np.any(x == peaks_x[i - 1]) else None
                    if start_idx is not None:
                        if is_negative:
                            # For negative peaks, find the point closest to the baseline
                            # between the previous peak and this one (baseline crossing)
                            segment = bounds_detection_signal[start_idx:peak_idx]
                            baseline_segment = baseline_y[start_idx:peak_idx]
                            min_left = start_idx + np.argmin(np.abs(segment - baseline_segment))
                        else:
                            # Use the corrected signal so the valley is found in baseline-corrected
                            # space. On a rising raw baseline argmin would always land at start_idx.
                            min_left = start_idx + np.argmin(integration_signal[start_idx:peak_idx])
                    else:
                        min_left = 0
                else:
                    min_left = 0

                if i < len(peaks_x) - 1:  # If not the last peak
                    # Find the index of the next peak's x value in the x array
                    end_idx = np.where(x == peaks_x[i + 1])[0][0] if np.any(x == peaks_x[i + 1]) else None
                    if end_idx is not None:
                        if is_negative:
                            # For negative peaks, find the point closest to the baseline
                            segment = bounds_detection_signal[peak_idx:end_idx]
                            baseline_segment = baseline_y[peak_idx:end_idx]
                            min_right = peak_idx + np.argmin(np.abs(segment - baseline_segment))
                        else:
                            min_right = peak_idx + np.argmin(integration_signal[peak_idx:end_idx])
                    else:
                        min_right = len(bounds_detection_signal) - 1
                else:
                    min_right = len(bounds_detection_signal) - 1
                
                # Calculate vertical distance between apex and baseline (peak height).
                # Use integration_signal for consistency with diff in bound-walking loops.
                apex_vertical_distance = integration_signal[peak_idx]

                # For negative peaks, distance is negative — use absolute value
                if is_negative or apex_vertical_distance < 0:
                    apex_vertical_distance = abs(apex_vertical_distance)
                
                # The threshold is 0.25% of the apex height
                threshold = apex_vertical_distance * 0.0025
                
                dx = 25  # Step size for slope calculation
                
                # Check for shoulder to the left that would limit this peak's left bound
                # (skip shoulder interactions for negative peaks)
                left_limit_by_shoulder = None
                if not is_negative:
                    for s_idx, s_bounds in shoulder_bounds.items():
                        # If shoulder apex is to the left of this peak and its right bound is after this peak's min_left
                        if s_idx < peak_idx and s_bounds[1] > min_left:
                            left_limit_by_shoulder = s_bounds[1]

                # Calculate the left bound using the specified criteria.
                # Use integration_signal (smoothed + corrected) for diff so that raw noise
                # spikes below baseline don't prematurely fire the threshold criterion.
                previous_slope = None
                left_stop_reason = "no_criteria_met"
                for j in range(peak_idx, 2, -1):
                    diff = integration_signal[j]
                    
                    # If we've hit a shoulder's right bound, stop
                    if left_limit_by_shoulder is not None and j <= left_limit_by_shoulder:
                        left_bound = max(j, left_limit_by_shoulder)
                        left_stop_reason = "shoulder_limit"
                        break
                    
                    # Central difference approximation
                    if j >= dx and j < len(bounds_detection_signal) - dx:
                        slope = (bounds_detection_signal[j] - bounds_detection_signal[j - dx]) / (x[j] - x[j - dx])
                    else:
                        # Edge case fallback to first-order difference
                        slope = (bounds_detection_signal[j] - bounds_detection_signal[j - 1]) / (x[j] - x[j - 1])
                    
                    # Check based on provided criteria
                    if 'threshold' in criteria:
                        # For negative peaks, compare abs(diff) since diff is negative
                        check_val = abs(diff) if is_negative else diff
                        if check_val <= threshold:
                            left_bound = j
                            left_stop_reason = f"threshold (check_val={check_val:.6f} <= {threshold:.6f})"
                            break

                    if 'minimum' in criteria and j == min_left:
                        left_bound = j
                        left_stop_reason = "minimum"
                        break

                    if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                        left_bound = j
                        left_stop_reason = f"slope_change (slope={slope:.6f}, prev_slope={previous_slope:.6f})"
                        break

                    previous_slope = slope

                # Check for shoulder to the right that would limit this peak's right bound
                # (skip shoulder interactions for negative peaks)
                right_limit_by_shoulder = None
                if not is_negative:
                    for s_idx, s_bounds in shoulder_bounds.items():
                        # If shoulder apex is to the right of this peak and its left bound is before this peak's min_right
                        if s_idx > peak_idx and s_bounds[0] < min_right:
                            right_limit_by_shoulder = s_bounds[0]
                
                # Calculate the right bound similarly
                previous_slope = None
                right_stop_reason = "no_criteria_met"
                for j in range(peak_idx, len(bounds_detection_signal) - 3):
                    diff = integration_signal[j]
                    
                    # If we've hit a shoulder's left bound, stop
                    if right_limit_by_shoulder is not None and j >= right_limit_by_shoulder:
                        right_bound = min(j, right_limit_by_shoulder)
                        right_stop_reason = "shoulder_limit"
                        break
                    
                    # Central difference approximation
                    if j >= dx and j < len(bounds_detection_signal) - dx:
                        slope = (bounds_detection_signal[j + dx] - bounds_detection_signal[j]) / (x[j + dx] - x[j])
                    else:
                        # Edge case fallback to first-order difference
                        slope = (bounds_detection_signal[j + 1] - bounds_detection_signal[j]) / (x[j + 1] - x[j])
                    
                    # Check based on provided criteria
                    if 'threshold' in criteria:
                        # For negative peaks, compare abs(diff) since diff is negative
                        check_val = abs(diff) if is_negative else diff
                        if check_val <= threshold:
                            right_bound = j
                            right_stop_reason = f"threshold (check_val={check_val:.6f} <= {threshold:.6f})"
                            break
                    
                    if 'minimum' in criteria and j == min_right:
                        right_bound = j
                        right_stop_reason = "minimum"
                        break
                    
                    if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                        right_bound = j
                        right_stop_reason = f"slope_change (slope={slope:.6f}, prev_slope={previous_slope:.6f})"
                        break
                    
                    previous_slope = slope
            
            # Clamp bounds to prevent overlap with adjacent peaks
            if i > 0:
                prev_peak_idx = np.where(x == peaks_x[i - 1])[0][0] if np.any(x == peaks_x[i - 1]) else None
                if prev_peak_idx is not None and left_bound <= prev_peak_idx:
                    # Don't extend past the previous peak's apex; use midpoint as boundary
                    left_bound = (prev_peak_idx + peak_idx) // 2
            if i < len(peaks_x) - 1:
                next_peak_idx = np.where(x == peaks_x[i + 1])[0][0] if np.any(x == peaks_x[i + 1]) else None
                if next_peak_idx is not None and right_bound >= next_peak_idx:
                    # Don't extend past the next peak's apex; use midpoint as boundary
                    right_bound = (peak_idx + next_peak_idx) // 2

            # Check for problematic case: bounds too close to apex (indicating premature stop)
            if left_bound == peak_idx and right_bound == peak_idx:
                print(f"WARNING: Peak {peak_number} at RT={apex_x:.3f} has zero-width integration!")
                print(f"  Left stop reason: {left_stop_reason}")
                print(f"  Right stop reason: {right_stop_reason}")
                print(f"  Apex signal: {bounds_detection_signal[peak_idx]:.3f}, Baseline: {baseline_y[peak_idx]:.3f}")
                print(f"  Threshold: {threshold:.6f}, Apex height: {apex_vertical_distance:.3f}")
            elif abs(left_bound - peak_idx) <= 2 or abs(right_bound - peak_idx) <= 2:
                print(f"WARNING: Peak {peak_number} at RT={apex_x:.3f} has very narrow integration bounds")
                print(f"  Left bound: {left_bound} (apex: {peak_idx}) - {left_stop_reason}")
                print(f"  Right bound: {right_bound} (apex: {peak_idx}) - {right_stop_reason}")
                print(f"  Integration width: {abs(right_bound - left_bound)} points")
                print(f"  Threshold: {threshold:.6f}, Apex height: {apex_vertical_distance:.3f}")
            
            # Append retention time at the peak (apex)
            ret_times.append(apex_x)
            
            # Extract data for integration using the corrected signal for area calculation
            x_peak = x[left_bound:right_bound + 1]
            y_peak = integration_signal[left_bound:right_bound + 1]  # Use corrected signal for area
            baseline_peak = baseline_y[left_bound:right_bound + 1]
            
            x_peaks.append(x_peak)
            y_peaks.append(y_peak)
            baseline_peaks.append(baseline_peak)
            
            # Check if integration_signal is already baseline-corrected
            # If integration_signal is 'corrected_y', it's already baseline-subtracted
            # If integration_signal is 'original_y', we need to subtract baseline
            if 'corrected_y' in processed_data and np.array_equal(integration_signal, processed_data['corrected_y']):
                # Signal is already baseline-corrected, use directly
                y_peak_corrected = y_peak
            else:
                # Signal is not baseline-corrected, subtract baseline
                y_peak_corrected = y_peak - baseline_peak
            
            # Calculate the integrated area using Simpson's rule
            area = simpson(y_peak_corrected, x=x_peak)
            
            # Apply correction factor
            area *= chemstation_area_factor

            # For negative peaks, report area as positive
            if is_negative:
                area = abs(area)

            # For decreasing x-axes (e.g. FTIR wavenumbers), area may be negative
            # even for non-negative peaks due to integration direction
            area = abs(area)

            # Store integrated area and bounds
            integrated_areas.append(area)
            integration_bounds.append((x[left_bound], x[right_bound]))
            
            # Create a Peak object
            retention_time = apex_x
            compound_id = Integrator.identify_compound(retention_time, rt_table)
            peak_number = i + 1
            integrator = 'py'
            start_time = x[left_bound]
            end_time = x[right_bound]
            width = end_time - start_time
            
            if feature_cls is Peak or issubclass(feature_cls, ChromatographicPeak):
                peak = feature_cls(compound_id, peak_number, retention_time,
                            integrator, width, area, start_time, end_time,
                            start_index=left_bound, end_index=right_bound)
            else:
                # SpectralFeature and other Feature subclasses use the generic signature
                peak = feature_cls(
                    feature_id=peak_number,
                    position=apex_x,
                    position_units=profile.x_label if profile else "",
                    area=area,
                    width=width,
                    start=start_time,
                    end=end_time,
                    start_index=left_bound,
                    end_index=right_bound,
                )
            
            # Add shoulder flag if applicable
            if has_shoulder_info:
                peak.is_shoulder = is_shoulder

            # Add negative peak flag
            if is_negative:
                peak.is_negative = True
            
            # Assess peak quality if MS data is available and quality checks are enabled
            if ms_data is not None and quality_options and quality_options.get('quality_checks_enabled', False):
                quality = assess_peak_quality(peak, x, integration_signal, ms_data, quality_options)
                peak.asymmetry = quality['asymmetry']
                peak.spectral_coherence = quality['spectral_coherence']
                peak.is_convoluted = quality['is_convoluted']
                
                # Add specific quality issues
                peak.quality_issues = []
                if quality['asymmetry'] is not None and quality_options.get('skew_check', False) and abs(quality['asymmetry']) > quality_options.get('skew_threshold', 0.5):
                    issue = "Asymmetric" if quality['asymmetry'] > 0 else "Fronting"
                    peak.quality_issues.append(f"{issue} (skew={quality['asymmetry']:.2f})")
                
                if quality['spectral_coherence'] is not None and quality_options.get('coherence_check', False) and quality['spectral_coherence'] < quality_options.get('coherence_threshold', 0.7):
                    peak.quality_issues.append(f"Low coherence ({quality['spectral_coherence']:.2f})")
            
            peaks_list.append(peak)
        
        # Apply peak grouping if specified
        if peak_groups and len(peak_groups) > 0 and len(peaks_list) > 0:
            peaks_list, x_peaks, y_peaks, baseline_peaks, ret_times, integrated_areas, integration_bounds = \
                Integrator._apply_peak_grouping(
                    peaks_list, peak_groups, x, 
                    integration_signal,
                    baseline_y, x_peaks, y_peaks, baseline_peaks, 
                    ret_times, integrated_areas, integration_bounds,
                    profile=profile
                )

        if verbose:
            try:
                from tabulate import tabulate
                headers = ['Compound ID', 'Peak #', 'Ret Time',
                         'Integrator', 'Width', 'Area', 'Start Time', 'End Time']
                print(tabulate([p.as_row() for p in peaks_list], headers), end='\n\n')
            except ImportError:
                print("Tabulate package not found. Install with 'pip install tabulate' for better formatting.")
                # Print in a basic format
                print("Integration Results:")
                for peak in peaks_list:
                    print(f"Peak {peak.peak_number}: RT={peak.retention_time:.3f}, Area={peak.area:.1f}")
        
        return {
            'peaks': peaks_list,
            'x_peaks': x_peaks,
            'y_peaks': y_peaks,
            'baseline_peaks': baseline_peaks,
            'retention_times': ret_times,
            'integrated_areas': integrated_areas,
            'integration_bounds': integration_bounds,
            'peaks_list': peaks_list
        }

def assess_peak_quality(peak, x, y, ms_data=None, options=None):
    """Assess peak quality to detect possible convolution.
    
    Args:
        peak: Peak object with start_time, end_time, and retention_time
        x: Array of retention times for the chromatogram
        y: Array of intensities for the chromatogram
        ms_data: MS data object with data and xlabels arrays
        options: Dictionary with quality check options
        
    Returns:
        dict: Quality metrics
    """
    # Initialize default options if none provided
    if options is None:
        options = {
            'quality_checks_enabled': False,
            'skew_check': False,
            'coherence_check': False,
            'skew_threshold': 0.5,
            'coherence_threshold': 0.7,
            'high_corr_threshold': 0.5
        }
    
    quality = {
        'asymmetry': None,
        'spectral_coherence': None,
        'is_convoluted': False
    }
    
    # If quality checks are disabled, return early
    if not options.get('quality_checks_enabled', False):
        return quality
    
    # Find indices corresponding to peak bounds
    start_idx = np.searchsorted(x, peak.start_time)
    end_idx = np.searchsorted(x, peak.end_time)
    
    # Ensure we have enough points for analysis
    if end_idx - start_idx < 5:
        print(f"Not enough points for quality assessment of peak {peak.peak_number} (need at least 5)")
        return quality
    
    # Extract peak region
    peak_x = x[start_idx:end_idx+1]
    peak_y = y[start_idx:end_idx+1]
    
    # 1. TEST FOR PEAK SHAPE ASYMMETRY (SKEWNESS) if enabled
    if options.get('skew_check', False):
        try:
            # Normalize intensities to [0,1] for numerical stability
            peak_y_norm = (peak_y - np.min(peak_y)) / (np.max(peak_y) - np.min(peak_y)) if np.max(peak_y) > np.min(peak_y) else peak_y
            
            # Calculate skewness using scipy
            asymmetry = skew(peak_y_norm)
            quality['asymmetry'] = asymmetry
            
            # Get the threshold from options
            skew_threshold = options.get('skew_threshold', 0.5)
            
            # Flag high asymmetry
            if abs(asymmetry) > skew_threshold:
                quality['is_convoluted'] = True
                print(f"Peak {peak.peak_number}: High asymmetry detected ({asymmetry:.2f})")
        except Exception as e:
            print(f"Error calculating peak asymmetry: {str(e)}")
    
    # 2. TEST FOR ION CHROMATOGRAM CORRELATION (SPECTRAL COHERENCE) if enabled
    if options.get('coherence_check', False) and ms_data is not None:
        try:
            # Find corresponding indices in MS data
            ms_start_idx = np.searchsorted(ms_data.xlabels, peak.start_time)
            ms_end_idx = np.searchsorted(ms_data.xlabels, peak.end_time)
            apex_idx = np.searchsorted(ms_data.xlabels, peak.retention_time)
            
            # Ensure we have enough MS scans
            if ms_end_idx - ms_start_idx < 3:
                print(f"Not enough MS scans for spectral coherence test of peak {peak.peak_number}")
                return quality
            
            # Get spectrum at apex to identify top ions
            apex_spectrum = ms_data.data[apex_idx, :].astype(float)
            
            # Find top N ions (by intensity)
            N = min(10, len(apex_spectrum))  # Use top 10 ions
            top_mz_indices = np.argsort(apex_spectrum)[-N:]
            
            # Extract XICs for top ions
            xics = ms_data.data[ms_start_idx:ms_end_idx+1, top_mz_indices]
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(N):
                for j in range(i+1, N):
                    # Skip ions with no variation
                    if np.std(xics[:, i]) > 0 and np.std(xics[:, j]) > 0:
                        r = np.corrcoef(xics[:, i], xics[:, j])[0, 1]
                        correlations.append(r)
            
            if correlations:
                # Calculate mean correlation
                mean_corr = np.mean(correlations)
                quality['spectral_coherence'] = mean_corr
                
                # Calculate percentage of highly correlated pairs (r > 0.9)
                pct_high_corr = sum(1 for r in correlations if r > 0.9) / len(correlations)
                
                # Get thresholds from options
                coherence_threshold = options.get('coherence_threshold', 0.7)
                high_corr_threshold = options.get('high_corr_threshold', 0.5)
                
                # Flag low coherence
                if mean_corr < coherence_threshold or pct_high_corr < high_corr_threshold:
                    quality['is_convoluted'] = True
                    print(f"Peak {peak.peak_number}: Low spectral coherence detected ({mean_corr:.2f}, {pct_high_corr:.2f} high corr)")
        except Exception as e:
            print(f"Error calculating spectral coherence: {str(e)}")
    
    return quality