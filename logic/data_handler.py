import os
import numpy as np
import rainbow as rb
from logic.spectrum_extractor import SpectrumExtractor

class DataHandler:
    """Handles loading and navigating through GC-MS data directories."""
    
    def __init__(self):
        self.current_data_dir = None
        self.current_directory_path = None
        self.available_directories = []
        self.current_index = -1
        self.current_detector = 'FID1A'  # Default detector
        self.spectrum_extractor = SpectrumExtractor()
    
    def load_data_directory(self, file_path):
        """Load an Agilent .D directory.
        
        Args:
            file_path (str): Path to the .D directory
            
        Returns:
            dict: Dictionary containing chromatogram and TIC data
        """
        if not file_path.endswith('.D'):
            if os.path.isdir(file_path + '.D'):
                file_path = file_path + '.D'
            else:
                raise ValueError(f"File path must be an Agilent .D directory: {file_path}")
        
        # Update navigation tracking - do this BEFORE loading the data
        self._update_directory_list(file_path)
        
        try:
            # Load the data directory using rainbow
            data_dir = rb.read(file_path)
            self.current_data_dir = data_dir
            self.current_directory_path = file_path
            
            # Get chromatogram data (defaults to FID1A)
            chromatogram_data = self._get_chromatogram_data(data_dir)
            
            # Get TIC data
            tic_data = self._get_tic_data(data_dir)
            
            return {
                'chromatogram': chromatogram_data,
                'tic': tic_data,
                'metadata': {'filename': os.path.basename(file_path)}
            }
            
        except Exception as e:
            raise Exception(f"Error loading data directory {file_path}: {str(e)}")
    
    def get_available_detectors(self, data_dir_path=None):
        """Get all available detector files (*.ch) in the data directory."""
        if data_dir_path is None:
            data_dir_path = self.current_directory_path
        
        if data_dir_path is None or not os.path.exists(data_dir_path):
            return []
        
        try:
            # Load the data directory if needed
            if self.current_data_dir is None or self.current_directory_path != data_dir_path:
                data_dir = rb.read(data_dir_path)
            else:
                data_dir = self.current_data_dir
            
            # Get all files in the data directory - using datafiles attribute instead of files
            files = data_dir.datafiles
            
            # Filter for detector files (ending with .ch)
            detectors = [str(f).replace('.ch', '') for f in files if str(f).endswith('.ch')]
            return detectors
        except Exception as e:
            print(f"Error getting available detectors: {e}")
            return []
    
    def _get_chromatogram_data(self, data_dir, detector=None):
        """Extract chromatogram data from the data directory."""
        if detector is None:
            detector = self.current_detector
        
        try:
            # Try to get the specified detector
            data = data_dir.get_file(detector + '.ch')
        except:
            # If not found, try some common alternatives
            alternatives = ['FID1A', 'TCD2B', 'FID2B', 'TCD1A', 'TCD1B']
            # Remove the failed detector from alternatives if it's in there
            if detector in alternatives:
                alternatives.remove(detector)
            
            for alt_detector in alternatives:
                try:
                    data = data_dir.get_file(alt_detector + '.ch')
                    # Update current_detector if we had to fall back
                    self.current_detector = alt_detector
                    print(f"Fallback to detector: {alt_detector}")
                    break
                except:
                    continue
            else:
                # If no detectors found, raise error
                raise ValueError(f"No valid chromatogram data found in data directory")
        
        # Prepare the data
        x = np.asarray(data.xlabels).flatten()
        y = np.asarray(data.data).flatten() * 7680  # Apply scaling factor
        
        return {'x': x, 'y': y}
    
    def _get_tic_data(self, data_dir):
        """Extract TIC data from the data directory."""
        try:
            ms_data = data_dir.get_file('data.ms')
            x_tic = ms_data.xlabels
            y_tic = np.sum(ms_data.data, axis=1)
            return {'x': x_tic, 'y': y_tic}
        except:
            # If no MS data, return empty arrays
            return {'x': np.array([]), 'y': np.array([])}
    
    def _update_directory_list(self, file_path):
        """Update the list of available .D directories in the parent folder."""
        # Make sure the file path is absolute
        file_path = os.path.abspath(file_path)
        
        # Get the parent directory
        parent_dir = os.path.dirname(file_path)
        print(f"Parent directory: {parent_dir}")
        
        # Find all .D directories in the parent folder
        self.available_directories = []
        if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path) and item.endswith('.D'):
                    self.available_directories.append(item_path)
        
        # Sort the directories for consistent navigation
        self.available_directories.sort()
        print(f"Found {len(self.available_directories)} .D directories")
        
        # Find the index of the current directory
        try:
            self.current_index = self.available_directories.index(file_path)
            print(f"Current directory index: {self.current_index}")
        except ValueError:
            print(f"Warning: Current directory {file_path} not found in directory list")
            # Print all directory paths for debugging
            for i, dir_path in enumerate(self.available_directories):
                print(f"  {i}: {dir_path}")
            self.current_index = -1
    
    def navigate_to_next(self):
        """Navigate to the next .D directory in the parent folder."""
        if not self.available_directories:
            print("No available directories for navigation")
            return None
        
        if self.current_index == -1:
            print("Current index is -1, cannot navigate")
            return None
        
        if self.current_index < len(self.available_directories) - 1:
            self.current_index += 1
            next_path = self.available_directories[self.current_index]
            print(f"Moving to next directory: {next_path}")
            return next_path
        
        print("Already at the last directory")
        return None
    
    def navigate_to_previous(self):
        """Navigate to the previous .D directory in the parent folder."""
        if not self.available_directories:
            print("No available directories for navigation")
            return None
        
        if self.current_index == -1:
            print("Current index is -1, cannot navigate")
            return None
        
        if self.current_index > 0:
            self.current_index -= 1
            prev_path = self.available_directories[self.current_index]
            print(f"Moving to previous directory: {prev_path}")
            return prev_path
        
        print("Already at the first directory")
        return None
    
    def get_processed_files(self):
        """Get a list of all processed files in the current directory."""
        if not self.current_directory_path:
            return []
        
        # Get parent directory
        parent_dir = os.path.dirname(self.current_directory_path)
        
        # Find all .D directories in the parent directory
        processed_files = []
        
        try:
            # Look for directories with integration_results.json
            for root, dirs, files in os.walk(parent_dir):
                for dir_name in dirs:
                    # Check if it's a .D directory
                    if dir_name.endswith('.D'):
                        dir_path = os.path.join(root, dir_name)
                        # Check if it has integration results
                        if os.path.exists(os.path.join(dir_path, 'integration_results.json')):
                            processed_files.append(dir_path)
        except Exception as e:
            print(f"Error scanning for processed files: {e}")
        
        return processed_files
    
    def get_ms_data(self, data_dir=None):
        """Get MS data from the specified directory or current directory."""
        if data_dir is None:
            data_dir = self.current_data_dir
        
        if data_dir is None:
            raise ValueError("No data directory specified or current")
        
        try:
            return data_dir.get_file('data.ms')
        except Exception as e:
            raise ValueError(f"Could not get MS data: {str(e)}")

    def get_detector_metadata(self, detector='FID1A', data_dir=None):
        """Get metadata for a specific detector file."""
        if data_dir is None:
            data_dir = self.current_data_dir
        
        if data_dir is None:
            raise ValueError("No data directory specified or current")
        
        try:
            detector_file = data_dir.get_file(f"{detector}.ch")
            return detector_file.metadata
        except Exception as e:
            raise ValueError(f"Could not get {detector} metadata: {str(e)}")

    def extract_spectrum_at_rt(self, retention_time, aligned_tic_data=None):
        """Extract mass spectrum at given retention time from current data."""
        if not self.current_directory_path:
            return None
        
        return self.spectrum_extractor.extract_at_rt(
            self.current_directory_path, 
            retention_time, 
            intensity_threshold=0.01
        )
    
    def extract_spectrum_for_peak(self, peak, options=None):
        """Extract mass spectrum for a peak using specified options."""
        if not self.current_directory_path:
            return None
        
        return self.spectrum_extractor.extract_for_peak(
            self.current_directory_path,
            peak,
            options
        )