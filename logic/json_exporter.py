"""
JSON export functionality for integration results.

This module provides functions to automatically export integration results
to JSON format, including metadata scraped from the .D directory using
the rainbow API.
"""

import os
import json
import datetime
import rainbow as rb
import numpy as np
from typing import Dict, List, Any, Optional


def scrape_metadata_from_d_directory(d_path: str, detector: str = "FID1A") -> Dict[str, Any]:
    """
    Scrape metadata from an Agilent .D directory using the rainbow API.
    
    Args:
        d_path: Path to the .D directory
        detector: Detector name (usually auto-detected, fallback "FID1A")
        
    Returns:
        Dictionary containing metadata fields
    """
    try:
        # Read the .D directory with rainbow
        data_dir = rb.read(d_path)
        
        # Extract basic metadata
        sample_id = getattr(data_dir, 'name', os.path.basename(d_path))
        
        # Try to get detector-specific metadata
        try:
            # Get the detector file (e.g., FID1A.ch)
            detector_files = [f for f in data_dir.datafiles if f.name.startswith(detector)]
            if detector_files:
                detector_file = detector_files[0]
                detector_metadata = detector_file.metadata
                
                # Extract metadata from detector file
                timestamp = detector_metadata.get('date', data_dir.metadata.get('date', 
                    datetime.datetime.now().strftime("%d %b %y  %I:%M %p")))
                method = detector_metadata.get('method', data_dir.metadata.get('method', 'Unknown'))
                notebook = detector_metadata.get('notebook', data_dir.metadata.get('notebook', sample_id))
                
            else:
                # Fallback to directory metadata
                timestamp = data_dir.metadata.get('date', datetime.datetime.now().strftime("%d %b %y  %I:%M %p"))
                method = data_dir.metadata.get('method', 'Unknown')
                notebook = data_dir.metadata.get('notebook', sample_id)
                
        except Exception as e:
            print(f"Warning: Could not get detector-specific metadata: {e}")
            # Fallback to directory metadata
            timestamp = data_dir.metadata.get('date', datetime.datetime.now().strftime("%d %b %y  %I:%M %p"))
            method = data_dir.metadata.get('method', 'Unknown')
            notebook = data_dir.metadata.get('notebook', sample_id)
        
        # Clean up method name if it contains .M extension
        if method and '.M' in method:
            method = method.split('.M')[0]
        
        # Construct signal path
        signal = f"Signal: {notebook}\\{detector}.ch"
        
        return {
            'sample_id': sample_id,
            'timestamp': timestamp,
            'method': method,
            'detector': detector,
            'signal': signal,
            'notebook': notebook
        }
        
    except Exception as e:
        print(f"Error scraping metadata from {d_path}: {e}")
        # Return fallback metadata
        sample_id = os.path.basename(d_path)
        return {
            'sample_id': sample_id,
            'timestamp': datetime.datetime.now().strftime("%d %b %y  %I:%M %p"),
            'method': 'Unknown',
            'detector': detector,
            'signal': f"Signal: {sample_id}\\{detector}.ch",
            'notebook': sample_id
        }


def _build_processing_metadata(processing_params: Optional[Dict],
                               scaling_factors: Optional[Dict] = None) -> Optional[Dict]:
    """Build a serialisable record of the processing parameters.

    Records every configurable parameter for full reproducibility.
    """
    if not processing_params:
        return None

    meta: Dict[str, Any] = {}

    # --- Smoothing ---
    sm = processing_params.get('smoothing', {})
    sm_info: Dict[str, Any] = {'enabled': sm.get('enabled', False)}
    if sm.get('enabled'):
        sm_info['method'] = sm.get('method', 'whittaker')
        sm_info['median_prefilter'] = sm.get('median_enabled', False)
        if sm.get('median_enabled'):
            sm_info['median_kernel'] = sm.get('median_kernel')
        if sm.get('method') == 'whittaker':
            sm_info['lambda'] = sm.get('lambda')
            sm_info['diff_order'] = sm.get('diff_order')
        elif sm.get('method') == 'savgol':
            sm_info['savgol_window'] = sm.get('savgol_window')
            sm_info['savgol_polyorder'] = sm.get('savgol_polyorder')
    meta['smoothing'] = sm_info

    # --- Baseline ---
    bl = processing_params.get('baseline', {})
    bl_info: Dict[str, Any] = {
        'method': bl.get('method', 'arpls'),
        'lambda': bl.get('lambda'),
        'asymmetry': bl.get('asymmetry'),
        'baseline_offset': bl.get('baseline_offset', 0.0),
    }
    if bl.get('method') == 'fastchrom':
        fc = bl.get('fastchrom', {})
        bl_info['fastchrom_half_window'] = fc.get('half_window')
        bl_info['fastchrom_smooth_half_window'] = fc.get('smooth_half_window')
    break_points = bl.get('break_points', [])
    if break_points:
        bl_info['break_points'] = break_points
    bl_info['align_tic'] = bl.get('align_tic', False)
    meta['baseline'] = bl_info

    # --- Peak detection ---
    pk = processing_params.get('peaks', {})
    pk_info: Dict[str, Any] = {
        'enabled': pk.get('enabled', False),
    }
    if pk.get('enabled'):
        pk_info['mode'] = pk.get('mode', 'classical')
        pk_info['min_prominence'] = pk.get('min_prominence')
        pk_info['min_height'] = pk.get('min_height')
        pk_info['min_width'] = pk.get('min_width')
        range_filters = pk.get('range_filters', [])
        if range_filters:
            pk_info['range_filters'] = range_filters
    meta['peak_detection'] = pk_info

    # --- Deconvolution (when mode == deconvolution) ---
    if pk.get('mode') == 'deconvolution':
        dc = processing_params.get('deconvolution', {})
        dc_info: Dict[str, Any] = {
            'splitting_method': dc.get('splitting_method', 'geometric'),
            'heatmap_threshold': dc.get('heatmap_threshold'),
            'pre_fit_signal_threshold': dc.get('pre_fit_signal_threshold'),
            'min_area_frac': dc.get('min_area_frac'),
            'valley_threshold_frac': dc.get('valley_threshold_frac'),
        }
        windows = dc.get('windows', [])
        if windows:
            dc_info['windows'] = windows
        method = dc.get('splitting_method', 'geometric')
        if method == 'emg':
            dc_info['mu_bound_factor'] = dc.get('mu_bound_factor')
            dc_info['fat_threshold_frac'] = dc.get('fat_threshold_frac')
            dc_info['dedup_sigma_factor'] = dc.get('dedup_sigma_factor')
        else:
            dc_info['dedup_rt_tolerance'] = dc.get('dedup_rt_tolerance')
        meta['deconvolution'] = dc_info

    # --- Negative peaks ---
    np_params = processing_params.get('negative_peaks', {})
    np_info: Dict[str, Any] = {'enabled': np_params.get('enabled', False)}
    if np_params.get('enabled'):
        np_info['min_prominence'] = np_params.get('min_prominence')
    meta['negative_peaks'] = np_info

    # --- Shoulder detection ---
    sh = processing_params.get('shoulders', {})
    sh_info: Dict[str, Any] = {'enabled': sh.get('enabled', False)}
    if sh.get('enabled'):
        sh_info['window_length'] = sh.get('window_length')
        sh_info['polyorder'] = sh.get('polyorder')
        sh_info['sensitivity'] = sh.get('sensitivity')
        sh_info['apex_distance'] = sh.get('apex_distance')
    meta['shoulders'] = sh_info

    # --- Integration / grouping ---
    ig = processing_params.get('integration', {})
    peak_groups = ig.get('peak_groups', [])
    if peak_groups:
        meta['integration'] = {'peak_groups': peak_groups}

    # --- Scaling factors (passed separately from app state) ---
    if scaling_factors:
        meta['scaling'] = scaling_factors

    # Strip None values for cleanliness
    def _strip_none(d):
        return {k: (_strip_none(v) if isinstance(v, dict) else v)
                for k, v in d.items() if v is not None}
    return _strip_none(meta) or None


def export_integration_results_to_json(peaks: List[Any], d_path: str,
                                      detector: str = "FID1A",
                                      quantitation_settings: Optional[Dict] = None,
                                      processing_params: Optional[Dict] = None,
                                      scaling_factors: Optional[Dict] = None) -> bool:
    """
    Export integration results to JSON file with metadata.

    Args:
        peaks: List of Peak objects from integration
        d_path: Path to the .D directory
        detector: Detector name (usually auto-detected, fallback "FID1A")
        quantitation_settings: Optional dict with quantitation settings and results
        processing_params: Optional dict of processing parameters (smoothing, baseline, peaks, deconvolution)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Scrape metadata from the .D directory
        metadata = scrape_metadata_from_d_directory(d_path, detector)

        # Build the result data structure
        result_data = {
            'sample_id': metadata['sample_id'],
            'timestamp': metadata['timestamp'],
            'method': metadata['method'],
            'detector': metadata['detector'],
            'signal': metadata['signal'],
            'notebook': metadata['notebook'],
            'peaks': []
        }

        # Add processing parameters metadata
        proc_meta = _build_processing_metadata(processing_params, scaling_factors)
        if proc_meta:
            result_data['processing_parameters'] = proc_meta

        # Add quantitation settings if provided
        if quantitation_settings:
            result_data['quantitation'] = quantitation_settings
        
        # Add peaks data
        for i, peak in enumerate(peaks):
            peak_data = {
                'Compound ID': getattr(peak, 'compound_id', 'Unknown'),
                'peak_number': getattr(peak, 'peak_number', 0),
                'retention_time': float(getattr(peak, 'retention_time', 0.0)),
                'integrator': getattr(peak, 'integrator', 'py'),
                'width': float(getattr(peak, 'width', 0.0)),
                'area': float(getattr(peak, 'area', 0.0)),
                'start_time': float(getattr(peak, 'start_time', 0.0)),
                'end_time': float(getattr(peak, 'end_time', 0.0))
            }
            
            # DEBUG: Check what fields are available on this peak
            print(f"JSON Export DEBUG - Peak {i}: compound_id={getattr(peak, 'compound_id', 'MISSING')}, "
                  f"Compound_ID={getattr(peak, 'Compound_ID', 'MISSING')}, "
                  f"Qual={getattr(peak, 'Qual', 'MISSING')}")
            
            # Add MS search results if available
            if hasattr(peak, 'Compound_ID') and peak.Compound_ID:
                peak_data['Compound ID'] = peak.Compound_ID
                print(f"  -> Using Compound_ID: {peak.Compound_ID}")
            else:
                print(f"  -> Using default compound_id: {getattr(peak, 'compound_id', 'Unknown')}")
            
            if hasattr(peak, 'Qual') and peak.Qual is not None:
                peak_data['Qual'] = float(peak.Qual)
                print(f"  -> Adding Qual: {peak.Qual}")
            else:
                print(f"  -> No Qual field found")
                
            if hasattr(peak, 'casno') and peak.casno:
                peak_data['casno'] = peak.casno
                print(f"  -> Adding casno: {peak.casno}")
            else:
                print(f"  -> No casno field found")
            
            # Add quality indicators
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                peak_data['is_saturated'] = True
                if hasattr(peak, 'saturation_level'):
                    peak_data['saturation_level'] = float(peak.saturation_level)
            
            if hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                peak_data['is_convoluted'] = True
                
            if hasattr(peak, 'quality_issues') and peak.quality_issues:
                peak_data['quality_issues'] = peak.quality_issues
            
            # Add quantitation results if available
            if hasattr(peak, 'mol_C') and peak.mol_C is not None:
                peak_data['mol_C'] = float(peak.mol_C)
            
            if hasattr(peak, 'mol_C_percent') and peak.mol_C_percent is not None:
                peak_data['mol_C_percent'] = float(peak.mol_C_percent)
            
            if hasattr(peak, 'num_carbons') and peak.num_carbons is not None:
                peak_data['num_carbons'] = int(peak.num_carbons)
            
            if hasattr(peak, 'mol') and peak.mol is not None:
                peak_data['mol'] = float(peak.mol)
            
            if hasattr(peak, 'mass_mg') and peak.mass_mg is not None:
                peak_data['mass_mg'] = float(peak.mass_mg)
            
            if hasattr(peak, 'mol_percent') and peak.mol_percent is not None:
                peak_data['mol_percent'] = float(peak.mol_percent)
            
            if hasattr(peak, 'wt_percent') and peak.wt_percent is not None:
                peak_data['wt_percent'] = float(peak.wt_percent)
            
            result_data['peaks'].append(peak_data)
        
        # Generate filename and save
        result_filename = f"{metadata['notebook']} - {metadata['detector']}.json"
        result_file_path = os.path.join(d_path, result_filename)
        
        # Write JSON file
        with open(result_file_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"Integration results exported to: {result_file_path}")
        return True
        
    except Exception as e:
        print(f"Error exporting integration results to JSON: {e}")
        return False


def update_json_with_ms_search_results(peaks: List[Any], d_path: str,
                                     detector: str = "FID1A",
                                     quantitation_settings: Optional[Dict] = None,
                                     processing_params: Optional[Dict] = None,
                                     scaling_factors: Optional[Dict] = None) -> bool:
    """
    Update existing JSON file with MS search results.

    Args:
        peaks: List of Peak objects with MS search results
        d_path: Path to the .D directory
        detector: Detector name (usually auto-detected, fallback "FID1A")
        quantitation_settings: Optional quantitation settings to include in export
        processing_params: Optional dict of processing parameters

    Returns:
        True if successful, False otherwise
    """
    try:
        # Try to load existing JSON file
        metadata = scrape_metadata_from_d_directory(d_path, detector)
        result_filename = f"{metadata['notebook']} - {metadata['detector']}.json"
        result_file_path = os.path.join(d_path, result_filename)
        
        if os.path.exists(result_file_path):
            # Load existing data
            with open(result_file_path, 'r') as f:
                result_data = json.load(f)
        else:
            # Create new structure if file doesn't exist
            result_data = {
                'sample_id': metadata['sample_id'],
                'timestamp': metadata['timestamp'],
                'method': metadata['method'],
                'detector': metadata['detector'],
                'signal': metadata['signal'],
                'notebook': metadata['notebook'],
                'peaks': []
            }
        
        # Update peaks with MS search results
        updated_peaks = []
        print(f"\n=== JSON UPDATE DEBUG: Processing {len(peaks)} peaks ===")
        for i, peak in enumerate(peaks):
            peak_data = {
                'Compound ID': getattr(peak, 'compound_id', 'Unknown'),
                'peak_number': getattr(peak, 'peak_number', 0),
                'retention_time': float(getattr(peak, 'retention_time', 0.0)),
                'integrator': getattr(peak, 'integrator', 'py'),
                'width': float(getattr(peak, 'width', 0.0)),
                'area': float(getattr(peak, 'area', 0.0)),
                'start_time': float(getattr(peak, 'start_time', 0.0)),
                'end_time': float(getattr(peak, 'end_time', 0.0))
            }
            
            # DEBUG: Check what fields are available on this peak
            print(f"Update Peak {i}: compound_id={getattr(peak, 'compound_id', 'MISSING')}, "
                  f"Compound_ID={getattr(peak, 'Compound_ID', 'MISSING')}, "
                  f"Qual={getattr(peak, 'Qual', 'MISSING')}")
            
            # Add MS search results if available
            if hasattr(peak, 'Compound_ID') and peak.Compound_ID:
                peak_data['Compound ID'] = peak.Compound_ID
                print(f"  -> Using Compound_ID: {peak.Compound_ID}")
            else:
                print(f"  -> Using default compound_id: {getattr(peak, 'compound_id', 'Unknown')}")
            
            if hasattr(peak, 'Qual') and peak.Qual is not None:
                peak_data['Qual'] = float(peak.Qual)
                print(f"  -> Adding Qual: {peak.Qual}")
            else:
                print(f"  -> No Qual field found")
                
            if hasattr(peak, 'casno') and peak.casno:
                peak_data['casno'] = peak.casno
                print(f"  -> Adding casno: {peak.casno}")
            else:
                print(f"  -> No casno field found")
            
            # Add quality indicators
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                peak_data['is_saturated'] = True
                if hasattr(peak, 'saturation_level'):
                    peak_data['saturation_level'] = float(peak.saturation_level)
            
            if hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                peak_data['is_convoluted'] = True
                
            if hasattr(peak, 'quality_issues') and peak.quality_issues:
                peak_data['quality_issues'] = peak.quality_issues
            
            # Add quantitation results if available
            if hasattr(peak, 'mol_C') and peak.mol_C is not None:
                peak_data['mol_C'] = float(peak.mol_C)
            
            if hasattr(peak, 'mol_C_percent') and peak.mol_C_percent is not None:
                peak_data['mol_C_percent'] = float(peak.mol_C_percent)
            
            if hasattr(peak, 'num_carbons') and peak.num_carbons is not None:
                peak_data['num_carbons'] = int(peak.num_carbons)
            
            if hasattr(peak, 'mol') and peak.mol is not None:
                peak_data['mol'] = float(peak.mol)
            
            if hasattr(peak, 'mass_mg') and peak.mass_mg is not None:
                peak_data['mass_mg'] = float(peak.mass_mg)
            
            if hasattr(peak, 'mol_percent') and peak.mol_percent is not None:
                peak_data['mol_percent'] = float(peak.mol_percent)
            
            if hasattr(peak, 'wt_percent') and peak.wt_percent is not None:
                peak_data['wt_percent'] = float(peak.wt_percent)
            
            updated_peaks.append(peak_data)
        
        # Update the peaks data
        result_data['peaks'] = updated_peaks

        # Add or update processing parameters
        proc_meta = _build_processing_metadata(processing_params, scaling_factors)
        if proc_meta:
            result_data['processing_parameters'] = proc_meta

        # Add or update quantitation settings if provided
        if quantitation_settings:
            result_data['quantitation'] = quantitation_settings

        # Write updated JSON file
        with open(result_file_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"JSON file updated with MS search results: {result_file_path}")
        return True
        
    except Exception as e:
        print(f"Error updating JSON file with MS search results: {e}")
        return False
