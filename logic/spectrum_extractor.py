import numpy as np
import rainbow as rb
from typing import Dict, Any, Optional, Union, List, Tuple

class SpectrumExtractor:
    """Handles all spectrum extraction logic for ChromaKit.
    
    This class provides a unified interface for extracting mass spectra using
    various methods and parameters. It encapsulates all extraction logic that
    was previously scattered across multiple files.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the SpectrumExtractor.
        
        Args:
            debug: Whether to print debug information
        """
        self.debug = debug
        self.saturation_threshold = 8.0e6  # Default saturation threshold
    
    def extract_at_rt(self, data_directory: str, retention_time: float, 
                     intensity_threshold: float = 0.01) -> Dict[str, Any]:
        """Extract a mass spectrum at a specific retention time.
        
        Args:
            data_directory: Path to the data directory
            retention_time: Retention time (minutes)
            intensity_threshold: Minimum relative intensity threshold
            
        Returns:
            Dictionary with mz and intensities arrays
        """
        if self.debug:
            print(f"Extracting spectrum at RT={retention_time:.4f}")
        
        try:
            # Load the MS data
            datadir = rb.read(data_directory)
            ms = datadir.get_file('data.ms')
            
            # Get TIC data for reference
            tic = np.sum(ms.data, axis=1)
            
            # Find closest RT index
            rt_index = np.argmin(np.abs(np.array(ms.xlabels) - retention_time))
            
            if self.debug:
                print(f"Found closest scan at index {rt_index}, RT={ms.xlabels[rt_index]:.4f}")
            
            # Extract spectrum at that index
            spectrum = ms.data[rt_index, :].astype(float)
            
            # Create m/z values
            mz_values = np.arange(len(spectrum)) + 1  # m/z values start at 1
            
            # Filter low intensity values
            if intensity_threshold > 0:
                max_intensity = np.max(spectrum)
                threshold = intensity_threshold * max_intensity
                mask = spectrum > threshold
                
                if self.debug:
                    print(f"Filtering intensities below {intensity_threshold} of max ({threshold:.1f})")
                    print(f"Keeping {np.sum(mask)}/{len(spectrum)} m/z values")
                
                return {
                    'rt': ms.xlabels[rt_index],
                    'mz': mz_values[mask],
                    'intensities': spectrum[mask]
                }
            else:
                return {
                    'rt': ms.xlabels[rt_index],
                    'mz': mz_values,
                    'intensities': spectrum
                }
                
        except Exception as e:
            if self.debug:
                print(f"Error extracting spectrum at RT={retention_time}: {str(e)}")
            return None

    def extract_for_peak(self, data_directory: str, peak: Any, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract a mass spectrum for a peak using the specified options.
        
        This is the main extraction method that handles all extraction strategies.
        
        Args:
            data_directory: Path to the data directory
            peak: Peak object with start_time and end_time attributes
            options: Dictionary of extraction options
            
        Returns:
            Dictionary with mz and intensities arrays
        """
        # Set default options
        default_options = {
            'extraction_method': 'apex',
            'subtract_background': True,
            'subtraction_method': 'min_tic',
            'subtract_weight': 0.1,
            'tic_weight': True,
            'range_points': 5,
            'midpoint_width_percent': 20,
            'intensity_threshold': 0.01,
            'saturation_threshold': self.saturation_threshold,
            'debug': self.debug
        }
        
        # Merge with provided options
        opts = {**default_options, **(options or {})}
        
        # Call the implementation method with explicit parameters
        return self._extract_peak_spectrum(
            data_directory, 
            peak,
            subtract_background=opts['subtract_background'],
            subtraction_method=opts['subtraction_method'],
            subtract_weight=opts['subtract_weight'],
            tic_weight=opts['tic_weight'],
            extraction_method=opts['extraction_method'],
            range_points=opts['range_points'],
            midpoint_width_percent=opts['midpoint_width_percent'],
            intensity_threshold=opts['intensity_threshold'],
            saturation_threshold=opts['saturation_threshold'],
            debug=opts['debug']
        )
    
    def _extract_peak_spectrum(self, data_directory, peak, 
                              subtract_background=True,
                              subtraction_method='min_tic',
                              subtract_weight=0.1,
                              tic_weight=True,
                              extraction_method='apex',  
                              range_points=5,
                              midpoint_width_percent=20,
                              intensity_threshold=0.01,
                              saturation_threshold=8.0e6,
                              debug=False):
        """Internal implementation of peak spectrum extraction.
        
        This contains the core extraction logic previously in batch_search.py.
        
        Args:
            data_directory: Path to the data directory
            peak: Peak object with start_time and end_time attributes
            subtract_background: Whether to subtract background spectrum
            subtraction_method: Method for background subtraction
            subtract_weight: Weight factor for background subtraction
            tic_weight: Whether to weight spectra by TIC intensity
            extraction_method: Method for spectrum extraction
            range_points: Number of points on each side of apex for range method
            midpoint_width_percent: Width of midpoint window as percentage
            intensity_threshold: Minimum intensity threshold for filtering
            saturation_threshold: Maximum intensity threshold for saturation
            debug: Whether to print debug information
            
        Returns:
            Dictionary with spectrum data
        """
        # Copy the implementation from batch_search.py extract_peak_spectrum
        # This is the existing code with minimal changes
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
                import traceback
                print(f"Error extracting spectrum: {str(e)}\n{traceback.format_exc()}")
            return None