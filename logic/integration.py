import numpy as np
from scipy.integrate import simpson
from scipy.stats import skew
import pandas as pd

class Peak:
    """Represents a chromatographic peak with its properties."""
    
    def __init__(self, compound_id, peak_number, retention_time, 
                 integrator, width, area, start_time, end_time):
        """Initialize a Peak object."""
        self.compound_id = compound_id
        self.peak_number = peak_number
        self.retention_time = retention_time
        self.integrator = integrator
        self.width = width
        self.area = area
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
        
        # Add saturation properties
        self.is_saturated = False
        self.saturation_level = None
    
    @property
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
    
    @property
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
            'is_convoluted': self.is_convoluted,
            'asymmetry': self.asymmetry,
            'spectral_coherence': self.spectral_coherence,
            'is_saturated': self.is_saturated,
            'saturation_level': self.saturation_level
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
            
        return result
    
    @property
    def apex_time(self):
        """Get the retention time at the peak apex."""
        return self.retention_time

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
    def integrate(processed_data, rt_table=None, chemstation_area_factor=0.0784, verbose=True, ms_data=None, quality_options=None):
        """Integrate peaks in a chromatogram.
        
        Args:
            processed_data: Dictionary containing the processed chromatogram data
            rt_table: Dictionary or DataFrame mapping retention times to compounds
            chemstation_area_factor: Area scaling factor to match ChemStation
            verbose: Whether to print integration results
            ms_data: Optional MS data object for peak quality assessment
            quality_options: Options for peak quality assessment
            
        Returns:
            dict: Dictionary containing integration results
        """
        # Extract relevant values from the processed data
        x = processed_data['x']
        y = processed_data.get('smoothed_y', processed_data.get('original_y'))
        baseline_y = processed_data['baseline_y']
        peaks_x = processed_data['peaks_x']
        peaks_y = processed_data['peaks_y']
        
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
        
        # Iterate through each peak for integration
        for i, (apex_x, apex_y) in enumerate(zip(peaks_x, peaks_y)):
            # Find the index of the apex in x and y arrays
            peak_idx = np.where(x == apex_x)[0][0]
            baseline_at_apex = baseline_y[peak_idx]
            
            # Initialize left and right bounds based on peak apex
            left_bound = peak_idx
            right_bound = peak_idx
            
            # Define the range for min_left and min_right calculations
            if i > 0:  # If not the first peak
                # Find the index of the previous peak's x value in the x array
                start_idx = np.where(x == peaks_x[i - 1])[0][0] if np.any(x == peaks_x[i - 1]) else None
                if start_idx is not None:
                    min_left = start_idx + np.argmin(y[start_idx:peak_idx])
                else:
                    min_left = 0
            else:
                min_left = 0
            
            if i < len(peaks_x) - 1:  # If not the last peak
                # Find the index of the next peak's x value in the x array
                end_idx = np.where(x == peaks_x[i + 1])[0][0] if np.any(x == peaks_x[i + 1]) else None
                if end_idx is not None:
                    min_right = peak_idx + np.argmin(y[peak_idx:end_idx])
                else:
                    min_right = len(y) - 1
            else:
                min_right = len(y) - 1
            
            # Calculate vertical distance between apex and baseline
            vertical_distance = apex_y - baseline_at_apex
            threshold = vertical_distance * 0.0025  # 0.25% threshold for signal proximity to baseline
            
            dx = 25  # Step size for slope calculation
            
            # Calculate the left bound using the specified criteria
            previous_slope = None
            for j in range(peak_idx, 2, -1):
                diff = y[j] - baseline_y[j]
                
                # Central difference approximation
                if j >= dx and j < len(y) - dx:
                    slope = (y[j] - y[j - dx]) / (x[j] - x[j - dx])
                else:
                    # Edge case fallback to first-order difference
                    slope = (y[j] - y[j - 1]) / (x[j] - x[j - 1])
                
                # Check based on provided criteria
                if 'threshold' in criteria and diff <= threshold:
                    left_bound = j
                    break
                
                if 'minimum' in criteria and j == min_left:
                    left_bound = j
                    break
                
                if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                    left_bound = j
                    break
                
                previous_slope = slope
            
            # Calculate the right bound similarly
            previous_slope = None
            for j in range(peak_idx, len(y) - 3):
                diff = y[j] - baseline_y[j]
                
                # Central difference approximation
                if j >= dx and j < len(y) - dx:
                    slope = (y[j + dx] - y[j]) / (x[j + dx] - x[j])
                else:
                    # Edge case fallback to first-order difference
                    slope = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
                
                # Check based on provided criteria
                if 'threshold' in criteria and diff <= threshold:
                    right_bound = j
                    break
                
                if 'minimum' in criteria and j == min_right:
                    right_bound = j
                    break
                
                if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                    right_bound = j
                    break
                
                previous_slope = slope
            
            # Append retention time at the peak (apex)
            ret_times.append(apex_x)
            
            # Extract data for integration
            x_peak = x[left_bound:right_bound + 1]
            y_peak = y[left_bound:right_bound + 1]
            baseline_peak = baseline_y[left_bound:right_bound + 1]
            
            x_peaks.append(x_peak)
            y_peaks.append(y_peak)
            baseline_peaks.append(baseline_peak)
            
            # Correct the signal by subtracting the baseline
            y_peak_corrected = y_peak - baseline_peak
            
            # Calculate the integrated area using Simpson's rule
            area = simpson(y_peak_corrected, x=x_peak)
            
            # Apply correction factor
            area *= chemstation_area_factor
            
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
            
            peak = Peak(compound_id, peak_number, retention_time,
                        integrator, width, area, start_time, end_time)
            
            # Assess peak quality if MS data is available and quality checks are enabled
            if ms_data is not None and quality_options and quality_options.get('quality_checks_enabled', False):
                quality = assess_peak_quality(peak, x, y, ms_data, quality_options)
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
        
        if verbose:
            try:
                from tabulate import tabulate
                headers = ['Compound ID', 'Peak #', 'Ret Time',
                         'Integrator', 'Width', 'Area', 'Start Time', 'End Time']
                print(tabulate([p.as_row for p in peaks_list], headers), end='\n\n')
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