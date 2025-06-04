from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QThreadPool, QMetaObject, Qt, Q_ARG, QTimer, QThread
import os
import time
import traceback
import datetime
import json
from util.thread_helpers import main_thread_dispatcher

class AutomationSignals(QObject):
    """Signals for automation worker in batch processing."""
    file_progress = Signal(str, str, int)  # filename, step, percent
    log_message = Signal(str)  # Log message
    processing_complete = Signal()  # Emitted when directory processing is complete
    error = Signal(str)  # Error message

class AutomationWorkerSignals(QObject):
    """Signals for the automation worker."""
    started = Signal(int)  # Total number of files
    file_started = Signal(str, int, int)  # Filename, file index, total files
    file_progress = Signal(str, str, int)  # Filename, step, percent
    file_completed = Signal(str, bool, str)  # Filename, success, message
    log_message = Signal(str)  # Log message
    finished = Signal()  # Emitted when all processing is complete
    error = Signal(str)  # Error message
    overall_progress = Signal(int, int, int)  # Current file index, total files, overall percentage

class AutomationWorker(QRunnable):
    """Worker for batch processing of files."""
    
    def __init__(self, app, directory_path, signals=None):
        """Initialize the worker.
        
        Args:
            app: The ChromaKitApp instance
            directory_path: Path to directory containing .D files
        """
        super().__init__()
        self.app = app
        self.directory_path = directory_path
        self.signals = AutomationWorkerSignals()
        self.cancelled = False
        self.current_file_index = 0
        self.total_files = 0
    
    @Slot()
    def run(self):
        """Run the automation worker."""
        try:
            # Get list of .D files in the directory
            d_files = []
            for item in os.listdir(self.directory_path):
                item_path = os.path.join(self.directory_path, item)
                if os.path.isdir(item_path) and item.endswith('.D'):
                    d_files.append(item_path)
            
            # Sort files by name
            d_files.sort()
            
            # Store these for progress calculations
            self.total_files = len(d_files)
            self.current_file_index = 0
            
            # Signal the start of processing
            self.signals.started.emit(len(d_files))
            self.signals.log_message.emit(f"Found {len(d_files)} .D files to process")
            
            # Process each file
            for i, file_path in enumerate(d_files):
                self.current_file_index = i  # Update the current file index
                
                if self.cancelled:
                    self.signals.log_message.emit("Processing cancelled by user")
                    break
                
                filename = os.path.basename(file_path)
                self.signals.file_started.emit(filename, i + 1, self.total_files)
                self.signals.log_message.emit(f"Starting file {i+1}/{self.total_files}: {filename}")
                
                try:
                    # Step 1: Load the file - 10% of one file's progress
                    self._update_directory_progress(filename, "Loading", 10)
                    success = self._load_file(file_path)
                    
                    if not success or self.cancelled:
                        self.signals.file_completed.emit(filename, False, "Failed to load file")
                        continue
                    
                    # Step 2: Process and integrate - 40% of one file's progress
                    self._update_directory_progress(filename, "Processing chromatogram", 40)
                    success = self._process_and_integrate()
                    
                    if not success or self.cancelled:
                        self.signals.file_completed.emit(filename, False, "Failed to integrate")
                        continue
                    
                    # Step 3: MS Library Search (if enabled) - 40% of one file's progress
                    # Check if MS search is enabled in batch options
                    ms_search_enabled = True  # Default if not specified
                    if hasattr(self.app, 'batch_options') and 'ms_search' in self.app.batch_options:
                        ms_search_enabled = self.app.batch_options['ms_search']
                    
                    if ms_search_enabled:
                        self._update_directory_progress(filename, "MS Library Search", 60)
                        success = self._run_ms_search()
                        
                        if not success or self.cancelled:
                            self.signals.file_completed.emit(filename, False, "Failed to run MS search")
                            continue
                    else:
                        # Skip MS search if disabled
                        self.signals.log_message.emit("MS search disabled in options - skipping")
                    
                    # File completed successfully
                    self._update_directory_progress(filename, "Completed", 100)
                    self.signals.file_completed.emit(filename, True, "Processing completed")
                    self.signals.log_message.emit(f"Completed processing {filename}")
                    
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.signals.log_message.emit(error_msg)
                    self.signals.file_completed.emit(filename, False, error_msg)
            
            # Signal completion
            self.signals.finished.emit()
            self.signals.log_message.emit("Automation completed")
            
        except Exception as e:
            error_msg = f"Error in automation: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
            self.signals.log_message.emit(error_msg)

    def _update_directory_progress(self, filename, step, file_progress):
        """Update the directory progress based on current file and file progress.
        
        This calculates smooth progress across all files in the directory.
        
        Args:
            filename: Current filename being processed
            step: Current processing step description
            file_progress: Progress percentage for the current file (0-100)
        """
        # Calculate overall directory progress
        # Each file contributes 1/total_files of the total progress
        file_contribution = 100.0 / self.total_files
        
        # Calculate base progress from completed files
        base_progress = self.current_file_index * file_contribution
        
        # Add partial progress from current file
        current_progress = base_progress + (file_progress / 100.0 * file_contribution)
        
        # Ensure progress stays within bounds
        current_progress = min(100, max(0, current_progress))
        
        # Emit the progress signal with calculated percentage
        self.signals.file_progress.emit(filename, step, int(current_progress))

    def _load_file(self, file_path):
        """Load a data file safely without direct UI access."""
        try:
            self.signals.log_message.emit(f"Loading file: {file_path}")
            
            # Set up completion flags
            self.load_completed = False
            self.load_success = False
            
            # Define what to execute on main thread
            def main_thread_load():
                try:
                    # We need to temporarily store the current status message
                    # so we can restore it after the file load
                    status_text = self.app.status_bar.currentMessage()
                    
                    # Load the file (which will update the status bar directly)
                    self.app.on_file_selected(file_path)
                    
                    # Set our success flag
                    self.load_success = True
                    
                    # Update our log via signals (thread-safe)
                    self.signals.log_message.emit(f"Successfully loaded {os.path.basename(file_path)}")
                except Exception as e:
                    self.signals.log_message.emit(f"Error loading file: {str(e)}")
                finally:
                    # Set our completion flag regardless of success/failure
                    self.load_completed = True
            
            # Schedule on main thread and wait for completion
            main_thread_dispatcher.run_on_main_thread(main_thread_load)
            
            # Wait for completion with timeout and cancellation check
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            while not self.load_completed:
                # Check for cancellation
                if self.cancelled:
                    self.signals.log_message.emit("File loading cancelled")
                    return False
                    
                # Check for timeout
                if time.time() - start_time > timeout:
                    self.signals.log_message.emit("Timeout waiting for file load")
                    return False
                    
                # Small sleep to prevent CPU hogging
                time.sleep(0.1)
            
            return self.load_success
            
        except Exception as e:
            self.signals.log_message.emit(f"Error in _load_file: {str(e)}")
            return False
    
    def _process_and_integrate(self):
        """Process and integrate the current file."""
        try:
            # Update progress to show we're starting integration (20% of file progress)
            self._update_directory_progress(
                os.path.basename(self.app.data_handler.current_directory_path),
                "Processing chromatogram", 
                20
            )
            
            # Get current parameters
            params = self.app.parameters_frame.get_parameters()
            
            # Ensure peak detection is enabled
            if not params['peaks']['enabled']:
                self.signals.log_message.emit("Peak detection is not enabled in parameters")
                return False
            
            # First process data with current parameters
            self.signals.log_message.emit("Processing chromatogram...")
            if hasattr(self.app, 'current_x') and hasattr(self.app, 'current_y'):
                # Use a thread-safe version of processing with no UI updates
                self.process_complete = False
                self.process_success = False
                
                def main_thread_process():
                    try:
                        self.app.process_and_display(self.app.current_x, self.app.current_y)
                        self.process_success = True
                    except Exception as e:
                        self.signals.log_message.emit(f"Error processing: {str(e)}")
                    finally:
                        self.process_complete = True
                
                # Run on main thread
                main_thread_dispatcher.run_on_main_thread(main_thread_process)
                
                # Wait for completion
                timeout = 30
                start_time = time.time()
                while not self.process_complete:
                    if self.cancelled or time.time() - start_time > timeout:
                        return False
                    time.sleep(0.1)
                    
                if not self.process_success:
                    return False
            else:
                self.signals.log_message.emit("No data available for processing")
                return False
                
            # Now update progress to show we're integrating (40% of file progress)
            self._update_directory_progress(
                os.path.basename(self.app.data_handler.current_directory_path),
                "Integrating peaks", 
                40
            )
            
            # Now integrate peaks using the non-UI method
            self.signals.log_message.emit("Integrating peaks...")
            
            # Use our integrate_peaks_no_ui method through main thread
            self.integrate_complete = False
            self.integration_results = None
            
            def main_thread_integrate():
                try:
                    self.integration_results = self.app.integrate_peaks_no_ui()
                finally:
                    self.integrate_complete = True
            
            # Run on main thread
            main_thread_dispatcher.run_on_main_thread(main_thread_integrate)
            
            # Wait for completion
            timeout = 30
            start_time = time.time()
            while not self.integrate_complete:
                if self.cancelled or time.time() - start_time > timeout:
                    return False
                time.sleep(0.1)
                
            # Check if integration was successful
            if not self.integration_results or not self.app.integrated_peaks:
                self.signals.log_message.emit("No peaks found during integration")
                return False
                
            # Add this line to apply manual overrides after integration
            self._apply_manual_overrides()
            
            # Update the plot on the main thread (if needed)
            def update_plot():
                try:
                    # Update plot with integration results
                    self.app.plot_frame.shade_integration_areas(self.integration_results)
                except Exception as e:
                    print(f"Error updating plot: {str(e)}")
            
            main_thread_dispatcher.run_on_main_thread(update_plot)
            
            self.signals.log_message.emit(f"Integration complete: {len(self.app.integrated_peaks)} peaks found")
            
            # If we get here, integration succeeded
            self._update_directory_progress(
                os.path.basename(self.app.data_handler.current_directory_path),
                f"Integrated {len(self.app.integrated_peaks)} peaks", 
                50
            )
            
            # Success
            return True
            
        except Exception as e:
            self.signals.log_message.emit(f"Error in processing/integration: {str(e)}")
            return False
    
    def _run_ms_search(self):
        """Run MS library search on integrated peaks."""
        try:
            # Check if cancelled first
            if self.cancelled:
                return False
                
            # Check if we have the required components
            if not hasattr(self.app.ms_frame, 'ms_toolkit') or not self.app.ms_frame.ms_toolkit:
                self.signals.log_message.emit("MS toolkit not available")
                return False
            
            if not self.app.ms_frame.library_loaded:
                self.signals.log_message.emit("MS library not loaded")
                return False
            
            if not hasattr(self.app, 'integrated_peaks') or not self.app.integrated_peaks:
                self.signals.log_message.emit("No integrated peaks available")
                return False
            
            # Create search parameters on the main thread
            self.search_params_ready = False
            self.worker_ready = False
            self.batch_worker = None
            
            def prepare_search_params():
                try:
                    # Create the worker - this must happen on main thread
                    self.batch_worker = self.app._create_batch_search_worker()
                    self.worker_ready = True if self.batch_worker else False
                    
                    # Set flags
                    self.search_params_ready = True
                except Exception as e:
                    self.signals.log_message.emit(f"Error preparing search: {str(e)}")
                    self.search_params_ready = True  # Set to true so we don't hang
                    self.worker_ready = False
            
            # Run on main thread
            main_thread_dispatcher.run_on_main_thread(prepare_search_params)
            
            # Wait for search params
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            while not self.search_params_ready:
                if self.cancelled or time.time() - start_time > timeout:
                    self.signals.log_message.emit("Timeout preparing search parameters")
                    return False
                time.sleep(0.1)
            
            # Check if worker was created successfully
            if not self.worker_ready or not self.batch_worker:
                self.signals.log_message.emit("Failed to create batch search worker")
                return False
            
            # Get options from the batch worker - FIX FOR UNDEFINED 'options'
            options = self.batch_worker.options if hasattr(self.batch_worker, 'options') else {}
            
            # Define debug flag - FIX FOR UNDEFINED 'debug'
            debug = options.get('debug', True)
            
            # Add a detailed log message showing parameters
            param_msg = (
                f"MS search parameters: "
                f"mz_shift={self.app.ms_frame.ms_toolkit.mz_shift if hasattr(self.app.ms_frame, 'ms_toolkit') else 'unknown'}, "
                f"method={options.get('search_method', 'vector')}, "
                f"weighting={options.get('weighting', 'NIST_GC')}"
            )
            self.signals.log_message.emit(param_msg)
            
            # Add peak count to the starting message
            self.signals.log_message.emit(
                f"Starting MS library search on {len(self.app.integrated_peaks)} peaks..."
            )
            
            # Signal we're starting search
            self.signals.log_message.emit(
                "===== Beginning library search for each peak ====="
            )
            
            # Set up completion tracking
            self.search_completed = False
            
            # Connect worker signals - MODIFY THIS PART FOR SMOOTH PROGRESS
            total_peaks = len(self.app.integrated_peaks)
            
            # Update the progress update function for peak processing:
            def update_peak_progress(index, name, results):
                # Calculate MS search progress (starts at 60%, ends at 95%)
                progress_base = 60
                progress_range = 35

                if total_peaks > 0:
                    ms_progress = progress_base + progress_range * ((index + 1) / total_peaks)
                else:
                    ms_progress = progress_base

                # Format a label
                if name and name != "No matches found":
                    label = f"MS Search: Peak {index+1}/{total_peaks} - Found: {name}"
                else:
                    label = f"MS Search: Peak {index+1}/{total_peaks} - No match"

                # Update progress using our directory-aware method
                self._update_directory_progress(
                    os.path.basename(self.app.data_handler.current_directory_path),
                    label,
                    ms_progress
                )

                # Update overall progress
                file_index = self.current_file_index  # Need to track this in the class
                total_files = self.total_files        # Need to track this in the class

                # Each file contributes 1/total_files to the overall progress
                # Each peak contributes a portion of that file's progress
                file_contribution = 1.0 / total_files
                file_base_progress = file_index * file_contribution
                peak_contribution = file_contribution * (ms_progress / 100)

                # Calculate overall progress percentage
                overall_percent = int(100 * (file_base_progress + peak_contribution))

                # Emit updated overall progress
                self.signals.overall_progress.emit(file_index, total_files, overall_percent)

                # Add this to update log with major milestones (avoid spamming)
                if index == 0 or (index+1) % 10 == 0 or index == total_peaks-1:
                    progress_msg = f"Processed {index+1}/{total_peaks} peaks"
                    self.signals.log_message.emit(progress_msg)

                # For all peaks, log the match result (but only show in application log)
                if name and name != "No matches found" and results:
                    match_score = results[0][1]
                    msg = f"Peak {index+1}: {name} ({match_score:.4f})"
                else:
                    msg = f"Peak {index+1}: No match found"

                # Just log to terminal, don't spam the dialog
                if debug:
                    print(msg)
            
            # Connect the signal with our new progress function
            self.batch_worker.signals.progress.connect(update_peak_progress)
            
            self.batch_worker.signals.finished.connect(self._on_search_completed)
            
            # Start worker in a new threadpool to avoid blocking this worker
            search_threadpool = QThreadPool()
            search_threadpool.start(self.batch_worker)
            
            # Wait for completion with timeout and cancellation check
            timeout = 300  # 5 minutes timeout
            start_time = time.time()
            while not self.search_completed:
                if self.cancelled:
                    # Set the worker's cancelled flag too
                    if hasattr(self.batch_worker, 'cancelled'):
                        self.batch_worker.cancelled = True
                    self.signals.log_message.emit("MS search cancelled")
                    return False
                
                if time.time() - start_time > timeout:
                    self.signals.log_message.emit("MS search timed out")
                    return False
                    
                time.sleep(0.1)  # Small sleep to prevent CPU hogging
            
            # Handle success - save results
            if hasattr(self.app.data_handler, 'current_directory_path'):
                self.signals.log_message.emit("Saving MS search results...")
                
                # Save results through main thread
                self.save_completed = False
                self.save_success = False
                
                def save_results():
                    try:
                        # This must run on the main thread
                        current_dir = self.app.data_handler.current_directory_path
                        integration_results = {'peaks': self.app.integrated_peaks}
                        
                        # Save JSON and CSV
                        json_success = self.app._save_integration_json(integration_results)
                        
                        if json_success:
                            csv_filename = os.path.join(current_dir, "RESULTS.CSV")
                            self.app.export_results_csv(csv_filename)
                            self.save_success = True
                    except Exception as e:
                        self.signals.log_message.emit(f"Error saving results: {str(e)}")
                    finally:
                        self.save_completed = True
                
                # Schedule save on main thread
                main_thread_dispatcher.run_on_main_thread(save_results)
                
                # Wait for save completion
                timeout = 30  # 30 seconds timeout
                start_time = time.time()
                while not self.save_completed:
                    if self.cancelled or time.time() - start_time > timeout:
                        self.signals.log_message.emit("Timeout saving results")
                        return False
                    time.sleep(0.1)
                
                if self.save_success:
                    self.signals.log_message.emit("Results saved successfully")
                else:
                    self.signals.log_message.emit("Failed to save results")
            
            return True
        
        except Exception as e:
            self.signals.log_message.emit(f"Error in MS search: {str(e)}")
            return False
    
    def _save_integration_json_no_ui(self, integration_results, data_dir_path):
        """Save integration results to JSON without updating UI."""
        try:
            # Similar to the original method but without StatusBar updates
            # Get the JSON data structure ready
            result_data = {
                'sample_id': os.path.basename(data_dir_path),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'method': "Unknown",
                'detector': "FID1A",
                'signal': f"Signal: {os.path.basename(data_dir_path)}\\FID1A.ch",
                'notebook': os.path.basename(data_dir_path),
                'peaks': []
            }
            
            # Add peaks data
            peaks = integration_results.get('peaks', [])
            for peak in peaks:
                # Access peak attributes safely with getattr to avoid AttributeError
                peak_data = {
                    'compound_id': getattr(peak, 'compound_id', "Unknown"),
                    'peak_number': getattr(peak, 'peak_number', 0),
                    'retention_time': getattr(peak, 'retention_time', 0.0),
                    'integrator': getattr(peak, 'integrator', "py"),
                    'width': getattr(peak, 'width', 0.0),
                    'area': getattr(peak, 'area', 0.0),
                    'start_time': getattr(peak, 'start_time', 0.0),
                    'end_time': getattr(peak, 'end_time', 0.0)
                }
                
                # Add notebook style fields if they exist
                if hasattr(peak, 'Compound_ID'):
                    peak_data['Compound ID'] = peak.Compound_ID
                if hasattr(peak, 'Qual'):
                    peak_data['Qual'] = peak.Qual
                if hasattr(peak, 'casno'):
                    peak_data['casno'] = peak.casno
                    
                result_data['peaks'].append(peak_data)
            
            # Define the file path and save the results as JSON
            result_filename = f"{result_data['notebook']} - {result_data['detector']}.json"
            result_file_path = os.path.join(data_dir_path, result_filename)
            
            with open(result_file_path, 'w') as result_file:
                json.dump(result_data, result_file, indent=4)
            
            return True
            
        except Exception as e:
            self.signals.log_message.emit(f"Error saving integration results: {str(e)}")
            return False

    def _export_csv_no_ui(self, filepath):
        """Export results to CSV without updating UI."""
        try:
            import csv
            
            # Create and write to the CSV file
            with open(filepath, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # Skip the first 9 rows by initializing with empty rows
                for _ in range(9):
                    csv_writer.writerow([])
                
                # Write headers
                headers = ['Library/ID', 'CAS', 'Qual', 'FID R.T.', 'FID Area']
                csv_writer.writerow(headers)
                
                # Write peak data
                for peak in self.app.integrated_peaks:
                    # Use the exact field names from the notebook
                    compound_id = getattr(peak, 'Compound_ID', None) or getattr(peak, 'compound_id', "Unknown")
                    casno = getattr(peak, 'casno', "")
                    qual = getattr(peak, 'Qual', "")
                    
                    # Format qual as a float with 4 decimal places if it's a number
                    if isinstance(qual, (int, float)):
                        qual = f"{qual:.4f}"
                    
                    row = [
                        compound_id,
                        casno,
                        qual,
                        f"{peak.retention_time:.3f}",
                        f"{peak.area:.1f}"
                    ]
                    csv_writer.writerow(row)
            
            return True
            
        except Exception as e:
            self.signals.log_message.emit(f"Error exporting CSV: {str(e)}")
            return False
    
    def _on_search_completed(self):
        """Handle MS search completion."""
        # Just set the flag - no UI interaction
        self.search_completed = True
        self.signals.log_message.emit("MS library search completed")
        
        # Additional log to help with debugging
        print("MS library search completed - search_completed flag set to True")

    def _apply_manual_overrides(self):
        """Apply manual assignment overrides to peaks based on retention time."""
        if not hasattr(self.app, 'integrated_peaks') or not self.app.integrated_peaks:
                return
        
        # Load the override database
        overrides_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'overrides', 'manual_assignments.json')
        
        if not os.path.exists(overrides_file):
            self.signals.log_message.emit("No manual overrides database found")
            return
        
        try:
            import json
            with open(overrides_file, 'r') as f:
                overrides = json.load(f)
                
            if not overrides:
                self.signals.log_message.emit("No manual overrides available")
                return
                
            # Define RT tolerance for matching (0.05 min = 3 sec)
            rt_tolerance = 0.05
            applied_count = 0
            
            # Check each peak against overrides
            for peak in self.app.integrated_peaks:
                for rt_key, override in overrides.items():
                    # Convert string RT key to float if needed
                    override_rt = override.get('retention_time', float(rt_key))
                    
                    # Check if peak RT is within tolerance
                    if abs(peak.retention_time - override_rt) <= rt_tolerance:
                        # We have a retention time match
                        # If we have spectrum data for both, we could check similarity
                        if 'spectrum' in override and hasattr(self.app, 'data_handler') and hasattr(self.app.data_handler, 'current_directory_path'):
                            # Extract spectrum for this peak
                            try:
                                from logic.batch_search import extract_peak_spectrum
                                
                                peak_spectrum = extract_peak_spectrum(
                                    self.app.data_handler.current_directory_path,
                                    peak,
                                    extraction_method='apex',
                                    debug=False
                                )
                                
                                if peak_spectrum and 'mz' in peak_spectrum and 'intensities' in peak_spectrum:
                                    # Get override spectrum
                                    override_spectrum = override['spectrum']
                                    
                                    # Convert to tuples for similarity function
                                    spectrum1 = [(m, i) for m, i in zip(peak_spectrum['mz'], peak_spectrum['intensities'])]
                                    spectrum2 = [(m, i) for m, i in zip(override_spectrum['mz'], override_spectrum['intensities'])]
                                    
                                    try:
                                        # Import similarity function
                                        from ms_toolkit.similarity import dot_product_similarity
                                        
                                        # Calculate similarity
                                        similarity = dot_product_similarity(spectrum1, spectrum2)
                                        
                                        # If below threshold, skip this override
                                        if similarity < 0.7:
                                            continue
                                    except ImportError:
                                        # If ms_toolkit is not available, just proceed with RT match
                                        pass
                            except Exception as e:
                                self.signals.log_message.emit(f"Error comparing spectra for manual override: {str(e)}")
                        
                        # Apply the override assignment
                        peak.compound_id = override['compound_name']
                        if hasattr(peak, 'Compound_ID'):
                            peak.Compound_ID = override['compound_name']
                        
                        # Add this flag to indicate manual assignment
                        peak.manual_assignment = True
                        
                        # Try to get CAS number if available - THIS IS THE PROBLEM SECTION
                        casno = None
                        # FIX: Add proper null checks for ms_toolkit and library
                        ms_toolkit_available = (
                            hasattr(self.app, 'ms_frame') and 
                            hasattr(self.app.ms_frame, 'ms_toolkit') and 
                            self.app.ms_frame.ms_toolkit is not None
                        )
                        
                        if ms_toolkit_available:
                            library_available = (
                                hasattr(self.app.ms_frame.ms_toolkit, 'library') and 
                                self.app.ms_frame.ms_toolkit.library is not None
                            )
                            
                            if library_available and override['compound_name'] in self.app.ms_frame.ms_toolkit.library:
                                compound = self.app.ms_frame.ms_toolkit.library[override['compound_name']]
                                if hasattr(compound, 'casno') and compound.casno is not None:
                                    from logic.batch_search import format_casno
                                    casno = format_casno(compound.casno)
                        
                        # Update CAS number if available
                        if casno:
                            peak.casno = casno
                        
                        # Manual overrides don't have a match score
                        peak.Qual = None
                        
                        applied_count += 1
                        self.signals.log_message.emit(f"Applied manual override: Peak at RT={peak.retention_time:.3f} assigned to {override['compound_name']}")
                        break  # Done with this peak, move to next
            
            if applied_count > 0:
                self.signals.log_message.emit(f"Applied {applied_count} manual overrides to peaks")
            else:
                self.signals.log_message.emit("No matching peaks found for manual overrides")
        
        except Exception as e:
            import traceback
            error_msg = f"Error applying manual overrides: {str(e)}\n{traceback.format_exc()}"
            self.signals.log_message.emit(error_msg)