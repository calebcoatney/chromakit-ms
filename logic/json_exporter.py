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


def export_integration_results_to_json(peaks: List[Any], d_path: str, 
                                      detector: str = "FID1A") -> bool:
    """
    Export integration results to JSON file with metadata.
    
    Args:
        peaks: List of Peak objects from integration
        d_path: Path to the .D directory
        detector: Detector name (usually auto-detected, fallback "FID1A")
        
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
        
        # Add peaks data
        for peak in peaks:
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
            
            # Add MS search results if available
            if hasattr(peak, 'Compound_ID') and peak.Compound_ID:
                peak_data['Compound ID'] = peak.Compound_ID
            
            if hasattr(peak, 'Qual') and peak.Qual is not None:
                peak_data['Qual'] = float(peak.Qual)
                
            if hasattr(peak, 'casno') and peak.casno:
                peak_data['casno'] = peak.casno
            
            # Add quality indicators
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                peak_data['is_saturated'] = True
                if hasattr(peak, 'saturation_level'):
                    peak_data['saturation_level'] = float(peak.saturation_level)
            
            if hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                peak_data['is_convoluted'] = True
                
            if hasattr(peak, 'quality_issues') and peak.quality_issues:
                peak_data['quality_issues'] = peak.quality_issues
            
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
                                     detector: str = "FID1A") -> bool:
    """
    Update existing JSON file with MS search results.
    
    Args:
        peaks: List of Peak objects with MS search results
        d_path: Path to the .D directory
        detector: Detector name (usually auto-detected, fallback "FID1A")
        
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
        for peak in peaks:
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
            
            # Add MS search results if available
            if hasattr(peak, 'Compound_ID') and peak.Compound_ID:
                peak_data['Compound ID'] = peak.Compound_ID
            
            if hasattr(peak, 'Qual') and peak.Qual is not None:
                peak_data['Qual'] = float(peak.Qual)
                
            if hasattr(peak, 'casno') and peak.casno:
                peak_data['casno'] = peak.casno
            
            # Add quality indicators
            if hasattr(peak, 'is_saturated') and peak.is_saturated:
                peak_data['is_saturated'] = True
                if hasattr(peak, 'saturation_level'):
                    peak_data['saturation_level'] = float(peak.saturation_level)
            
            if hasattr(peak, 'is_convoluted') and peak.is_convoluted:
                peak_data['is_convoluted'] = True
                
            if hasattr(peak, 'quality_issues') and peak.quality_issues:
                peak_data['quality_issues'] = peak.quality_issues
            
            updated_peaks.append(peak_data)
        
        # Update the peaks data
        result_data['peaks'] = updated_peaks
        
        # Write updated JSON file
        with open(result_file_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"JSON file updated with MS search results: {result_file_path}")
        return True
        
    except Exception as e:
        print(f"Error updating JSON file with MS search results: {e}")
        return False
