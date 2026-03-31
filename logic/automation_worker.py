from PySide6.QtCore import QObject, QRunnable, Signal, Slot, QThreadPool, QMetaObject, Qt, Q_ARG, QTimer, QThread
import os
import time
import traceback
import datetime
import json
from logic.c_folder import CFolder
from logic.signal_profiles import SignalProfileRegistry, PipelineStage
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
        # Timing: populated when self.app.batch_options['time_steps'] is True
        self._timings: list[dict] = []
        self._skip_ui_updates: bool = True   # Default: skip renders for speed
        self._time_steps: bool = False        # Default: no timing
        self._current_params = None           # Pre-fetched per-file in _load_file

    @Slot()
    def run(self):
        """Run the automation worker."""
        try:
            batch_opts = getattr(self.app, 'batch_options', {}) or {}
            self._time_steps = batch_opts.get('time_steps', False)
            self._skip_ui_updates = batch_opts.get('skip_ui_updates', True)

            # Get list of .C folders in the directory
            c_files = []
            for item in sorted(os.listdir(self.directory_path)):
                item_path = os.path.join(self.directory_path, item)
                if os.path.isdir(item_path) and item.endswith('.C'):
                    c_files.append(item_path)
            
            # If no .C folders found, emit a message and exit
            if not c_files:
                self.signals.log_message.emit(
                    "No .C folders found in directory. "
                    "Use File > Migrate .D Folders to convert existing Agilent data."
                )
                self.signals.finished.emit()
                return

            # Store these for progress calculations
            self.total_files = len(c_files)
            self.current_file_index = 0
            
            # Signal the start of processing
            self.signals.started.emit(len(c_files))
            self.signals.log_message.emit(f"Found {len(c_files)} .C folders to process")
            
            # Process each file
            for i, file_path in enumerate(c_files):
                self.current_file_index = i  # Update the current file index
                
                if self.cancelled:
                    self.signals.log_message.emit("Processing cancelled by user")
                    break
                
                filename = os.path.basename(file_path)
                self.signals.file_started.emit(filename, i + 1, self.total_files)
                self.signals.log_message.emit(f"Starting file {i+1}/{self.total_files}: {filename}")
                
                # Determine signal type from manifest so we can skip inapplicable steps.
                try:
                    _cf = CFolder.open(file_path)
                    signal_type = _cf.get_manifest().get("signal_type", "gcms")
                    profile_stages = set(SignalProfileRegistry.get(signal_type).pipeline_stages)
                except Exception:
                    signal_type = "gcms"
                    profile_stages = {
                        PipelineStage.SMOOTHING, PipelineStage.BASELINE,
                        PipelineStage.PEAKS, PipelineStage.MS_SEARCH,
                        PipelineStage.QUANTITATION,
                    }
                
                is_chromatography = signal_type in ("gc", "gcms")
                step_label = "Processing chromatogram" if is_chromatography else "Processing signal"
                # Store on self so _process_and_integrate can read them
                self._signal_type = signal_type
                self._is_chromatography = is_chromatography
                
                try:
                    # Step 1: Load the file - 10% of one file's progress
                    t0 = time.perf_counter()
                    self._update_directory_progress(filename, "Loading", 10)
                    success = self._load_file(file_path)
                    t_load = time.perf_counter() - t0

                    if not success or self.cancelled:
                        self.signals.file_completed.emit(filename, False, "Failed to load file")
                        continue

                    # Step 2: Process and integrate - 40% of one file's progress
                    t1 = time.perf_counter()
                    success = self._process_and_integrate()
                    t_process = time.perf_counter() - t1

                    if not success or self.cancelled:
                        self.signals.file_completed.emit(filename, False, "Failed to integrate")
                        continue

                    # Step 2.5: Save integration results to JSON - 50% of one file's progress
                    t2 = time.perf_counter()
                    self._update_directory_progress(filename, "Saving integration results", 50)
                    if hasattr(self.app, 'integrated_peaks') and self.app.integrated_peaks:
                        current_dir = self.app.data_handler.current_directory_path
                        integration_results = {'peaks': self.app.integrated_peaks}
                        json_saved = self._save_integration_json_no_ui(integration_results, current_dir)
                        if json_saved:
                            self.signals.log_message.emit(f"Integration results saved to JSON for {filename}")
                        else:
                            self.signals.log_message.emit(f"Warning: Failed to save JSON for {filename}")
                    t_save_integration = time.perf_counter() - t2

                    # Step 3: MS Library Search (if enabled and applicable)
                    ms_search_enabled = True  # Default if not specified
                    if hasattr(self.app, 'batch_options') and 'ms_search' in self.app.batch_options:
                        ms_search_enabled = self.app.batch_options['ms_search']

                    ms_search_applicable = PipelineStage.MS_SEARCH in profile_stages
                    t_ms = None

                    if ms_search_enabled and ms_search_applicable:
                        self._update_directory_progress(filename, "MS Library Search", 70)
                        t3 = time.perf_counter()
                        success = self._run_ms_search()
                        t_ms = time.perf_counter() - t3

                        if not success or self.cancelled:
                            self.signals.file_completed.emit(filename, False, "Failed to run MS search")
                            continue
                    elif ms_search_enabled:
                        self.signals.log_message.emit(
                            f"Skipping MS search: not applicable for signal type '{signal_type}'"
                        )
                    else:
                        # Skip MS search if disabled
                        self.signals.log_message.emit("MS search disabled in options - skipping")

                    # Record timing for this file
                    if self._time_steps:
                        self._timings.append({
                            'file': filename,
                            'load': t_load,
                            'process': t_process,
                            'save': t_save_integration,
                            'ms_search': t_ms,
                        })

                    # File completed successfully
                    self._update_directory_progress(filename, "Completed", 100)
                    self.signals.file_completed.emit(filename, True, "Processing completed")
                    self.signals.log_message.emit(f"Completed processing {filename}")
                    
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    self.signals.log_message.emit(error_msg)
                    self.signals.file_completed.emit(filename, False, error_msg)
            
            # Emit timing report if enabled
            if self._time_steps and self._timings:
                table = self._format_timing_table(self._timings)
                self.signals.log_message.emit(table)

            # Signal completion
            self.signals.finished.emit()
            self.signals.log_message.emit("Automation completed")
            
        except Exception as e:
            error_msg = f"Error in automation: {str(e)}\n{traceback.format_exc()}"
            self.signals.error.emit(error_msg)
            self.signals.log_message.emit(error_msg)

    @staticmethod
    def _format_timing_table(timings: list[dict]) -> str:
        """Format per-file timing data as a log-friendly table.

        Each entry in timings: {
            'file': str,
            'load': float,        # seconds
            'process': float,     # seconds
            'ms_search': float | None,  # None if not applicable
            'save': float,
        }
        """
        if not timings:
            return "=== Batch Timing Report: no data ==="

        lines = ["", "=== Batch Timing Report ==="]
        header = f"{'File':<35} {'Load':>7} {'Process':>9} {'MS Search':>10} {'Save':>7} {'Total':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        ms_times = []
        for t in timings:
            ms_val = t.get('ms_search')
            ms_str = f"{ms_val:>9.2f}s" if ms_val is not None else f"{'—':>9}"
            total = t['load'] + t['process'] + (ms_val or 0.0) + t['save']
            line = (
                f"{t['file']:<35}"
                f" {t['load']:>6.2f}s"
                f" {t['process']:>8.2f}s"
                f" {ms_str}"
                f" {t['save']:>6.2f}s"
                f" {total:>7.2f}s"
            )
            lines.append(line)
            if ms_val is not None:
                ms_times.append(ms_val)

        # Averages
        lines.append("-" * len(header))
        n = len(timings)
        avg_load = sum(t['load'] for t in timings) / n
        avg_proc = sum(t['process'] for t in timings) / n
        avg_save = sum(t['save'] for t in timings) / n
        avg_ms_str = f"{sum(ms_times)/len(ms_times):>9.2f}s" if ms_times else f"{'—':>9}"
        avg_total = avg_load + avg_proc + (sum(ms_times)/len(ms_times) if ms_times else 0.0) + avg_save
        lines.append(
            f"{'Average':<35}"
            f" {avg_load:>6.2f}s"
            f" {avg_proc:>8.2f}s"
            f" {avg_ms_str}"
            f" {avg_save:>6.2f}s"
            f" {avg_total:>7.2f}s"
        )
        lines.append("")
        return "\n".join(lines)

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
                    # Pass batch_mode=True to suppress error dialogs
                    self.app.on_file_selected(file_path, batch_mode=True)
                    
                    # Set our success flag
                    self.load_success = True

                    # Pre-fetch params while we're on the main thread
                    # so _process_and_integrate can use them off-thread
                    try:
                        self._current_params = self.app.parameters_frame.get_parameters()
                    except Exception:
                        self._current_params = None

                    # Update our log via signals (thread-safe)
                    self.signals.log_message.emit(f"Successfully loaded {os.path.basename(file_path)}")
                except Exception as e:
                    # Debug: Check what type of object we caught
                    print(f"AUTOMATION DEBUG: Exception type: {type(e)}")
                    print(f"AUTOMATION DEBUG: Exception repr: {repr(e)}")
                    
                    if not isinstance(e, BaseException):
                        print(f"AUTOMATION WARNING: Caught non-exception object: {type(e)}")
                        error_msg = f"Error loading file: Unexpected object: {str(e)}"
                    else:
                        error_msg = f"Error loading file: {str(e)}"
                    
                    self.signals.log_message.emit(error_msg)
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
            filename = os.path.basename(
                self.app.data_handler.current_directory_path
            )

            self._update_directory_progress(filename, "Processing chromatogram", 20)

            if not (hasattr(self.app, 'current_x') and hasattr(self.app, 'current_y')
                    and self.app.current_x is not None and self.app.current_y is not None):
                self.signals.log_message.emit("No data available for processing")
                return False

            # Use pre-fetched params when available; fall back to main-thread fetch.
            params = getattr(self, '_current_params', None)
            if params is None:
                # Fetch on main thread (legacy path / params not pre-fetched)
                self._fetch_params_complete = False
                def fetch_params():
                    try:
                        self._current_params = self.app.parameters_frame.get_parameters()
                    finally:
                        self._fetch_params_complete = True
                main_thread_dispatcher.run_on_main_thread(fetch_params)
                timeout, start = 10, time.time()
                while not self._fetch_params_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.05)
                params = self._current_params
                if params is None:
                    self.signals.log_message.emit("Failed to fetch parameters")
                    return False

            if not params['peaks']['enabled']:
                self.signals.log_message.emit("Peak detection is not enabled in parameters")
                return False

            if getattr(self, '_skip_ui_updates', True):
                # ── Fast path: run math directly on this worker thread ──────────
                self.signals.log_message.emit("Processing chromatogram...")
                try:
                    ms_range = None
                    if (hasattr(self.app, 'plot_frame')
                            and hasattr(self.app.plot_frame, 'tic_data')
                            and self.app.plot_frame.tic_data is not None):
                        td = self.app.plot_frame.tic_data
                        if 'x' in td and len(td['x']) > 0:
                            ms_range = (float(td['x'].min()), float(td['x'].max()))

                    profile = getattr(self.app, 'current_profile', None)
                    processed = self.app.processor.process(
                        self.app.current_x,
                        self.app.current_y,
                        params,
                        ms_range,
                        profile=profile,
                    )
                except Exception as e:
                    self.signals.log_message.emit(f"Error processing: {str(e)}")
                    return False

                # Store result so integrate_peaks_no_ui can read it
                self.app.current_processed = processed
                self.app.current_profile = profile

                self._update_directory_progress(filename, "Integrating peaks", 40)
                self.signals.log_message.emit("Integrating peaks...")

                self.integrate_complete = False
                self.integration_results_fast = None

                def main_thread_integrate_fast():
                    try:
                        self.integration_results_fast = self.app.integrate_peaks_no_ui(params=params)
                    except Exception as e:
                        self.signals.log_message.emit(f"Error integrating: {str(e)}")
                    finally:
                        self.integrate_complete = True

                main_thread_dispatcher.run_on_main_thread(main_thread_integrate_fast)
                timeout, start = 30, time.time()
                while not self.integrate_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.1)
                integration_results = self.integration_results_fast

            else:
                # ── Original path: dispatch to main thread ────────────────────
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

                main_thread_dispatcher.run_on_main_thread(main_thread_process)
                timeout, start = 30, time.time()
                while not self.process_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.1)
                if not self.process_success:
                    return False

                self._update_directory_progress(filename, "Integrating peaks", 40)
                self.signals.log_message.emit("Integrating peaks...")

                self.integrate_complete = False
                self.integration_results = None

                def main_thread_integrate():
                    try:
                        self.integration_results = self.app.integrate_peaks_no_ui()
                    finally:
                        self.integrate_complete = True

                main_thread_dispatcher.run_on_main_thread(main_thread_integrate)
                timeout, start = 30, time.time()
                while not self.integrate_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.1)

                integration_results = self.integration_results

            # ── Common completion path ────────────────────────────────────────
            if not integration_results:
                self.signals.log_message.emit("Integration failed")
                return False

            if not self.app.integrated_peaks:
                self.signals.log_message.emit("No peaks found during integration")
                if getattr(self, "_is_chromatography", True):
                    return False
            else:
                if getattr(self, "_is_chromatography", True):
                    self._apply_manual_overrides()

            # Update plot shading (if UI updates are on)
            if not getattr(self, '_skip_ui_updates', True):
                def update_plot():
                    try:
                        self.app.plot_frame.shade_integration_areas(integration_results)
                    except Exception as e:
                        print(f"Error updating plot: {str(e)}")
                main_thread_dispatcher.run_on_main_thread(update_plot)

            self.signals.log_message.emit(
                f"Integration complete: {len(self.app.integrated_peaks)} peaks found"
            )
            self._update_directory_progress(
                filename, f"Integrated {len(self.app.integrated_peaks)} peaks", 50
            )
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
                        
                        # Update JSON with MS search results using the new exporter
                        try:
                            from logic.json_exporter import update_json_with_ms_search_results
                            
                            # Get detector name
                            detector = self.app.data_handler.current_detector if hasattr(self.app.data_handler, 'current_detector') else 'Unknown'
                            
                            # Update JSON with MS search results
                            proc_params = self.app.parameters_frame.get_parameters() if hasattr(self.app, 'parameters_frame') else None
                            json_success = update_json_with_ms_search_results(
                                self.app.integrated_peaks,
                                current_dir,
                                detector,
                                processing_params=proc_params,
                            )
                            
                            if json_success:
                                from logic.csv_exporter import export_results_to_csv
                                csv_filename = os.path.join(current_dir, "RESULTS.CSV")
                                export_results_to_csv(self.app.integrated_peaks, csv_filename)
                                self.save_success = True
                                self.signals.log_message.emit("Results updated with MS search and exported to CSV")
                            else:
                                self.signals.log_message.emit("Failed to update JSON with MS search results")
                                
                        except Exception as e:
                            self.signals.log_message.emit(f"Error with new JSON exporter, using fallback: {str(e)}")
                            
                            # Fallback to old method
                            integration_results = {'peaks': self.app.integrated_peaks}
                            json_success = self.app._save_integration_json(integration_results)
                            
                            if json_success:
                                from logic.csv_exporter import export_results_to_csv
                                csv_filename = os.path.join(current_dir, "RESULTS.CSV")
                                export_results_to_csv(self.app.integrated_peaks, csv_filename)
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
        """Save integration results using export manager."""
        try:
            # Use export manager if available
            if hasattr(self.app, 'export_manager'):
                detector = "Unknown"  # Default fallback
                if hasattr(self.app, 'data_handler') and hasattr(self.app.data_handler, 'current_detector'):
                    detector = self.app.data_handler.current_detector
                
                # Use export manager for batch processing
                export_result = self.app.export_manager.export_during_batch(
                    integration_results.get('peaks', []),
                    data_dir_path,
                    detector
                )
                
                success = export_result.get('json', False) or export_result.get('csv', False)
                if success:
                    messages = [msg for msg in export_result['messages'] if 'successfully' in msg or 'exported' in msg]
                    if messages:
                        self.signals.log_message.emit(f"Integration results exported: {'; '.join(messages)}")
                    else:
                        self.signals.log_message.emit("Integration results saved")
                else:
                    self.signals.log_message.emit("Failed to export integration results")
                
                return success
            else:
                # Fallback to old method
                from logic.json_exporter import export_integration_results_to_json
                
                # Get detector name from app if available
                detector = "Unknown"  # Default fallback
                if hasattr(self.app, 'data_handler') and hasattr(self.app.data_handler, 'current_detector'):
                    detector = self.app.data_handler.current_detector
                
                # Export using the new module
                proc_params = self.app.parameters_frame.get_parameters() if hasattr(self.app, 'parameters_frame') else None
                success = export_integration_results_to_json(
                    integration_results.get('peaks', []),
                    data_dir_path,
                    detector,
                    processing_params=proc_params,
                )
            
            if success:
                self.signals.log_message.emit("Integration results saved to JSON")
            
            return success
            
        except Exception as e:
            self.signals.log_message.emit(f"Error saving integration results: {str(e)}")
            
            # Fallback to original method if new one fails
            try:
                return self._save_integration_json_no_ui_fallback(integration_results, data_dir_path)
            except Exception as e2:
                self.signals.log_message.emit(f"Fallback method also failed: {str(e2)}")
                return False

    def _save_integration_json_no_ui_fallback(self, integration_results, data_dir_path):
        """Fallback method for saving integration results to JSON."""
        try:
            # Similar to the original method but without StatusBar updates
            # Get the JSON data structure ready
            result_data = {
                'sample_id': os.path.basename(data_dir_path),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'method': "Unknown",
                'detector': "Unknown",
                'signal': f"Signal: {os.path.basename(data_dir_path)}\\Unknown.ch",
                'notebook': os.path.basename(data_dir_path),
                'peaks': []
            }
            
            # Add peaks data
            peaks = integration_results.get('peaks', [])
            for peak in peaks:
                from logic.json_exporter import _serialize_peak
                result_data['peaks'].append(_serialize_peak(peak))
            
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
        from logic.csv_exporter import export_results_to_csv
        return export_results_to_csv(self.app.integrated_peaks, filepath)
    
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
                    if abs(peak.position - override_rt) <= rt_tolerance:
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
                        self.signals.log_message.emit(f"Applied manual override: Peak at position={peak.position:.3f} assigned to {override['compound_name']}")
                        break  # Done with this peak, move to next
            
            if applied_count > 0:
                self.signals.log_message.emit(f"Applied {applied_count} manual overrides to peaks")
            else:
                self.signals.log_message.emit("No matching peaks found for manual overrides")
        
        except Exception as e:
            import traceback
            error_msg = f"Error applying manual overrides: {str(e)}\n{traceback.format_exc()}"
            self.signals.log_message.emit(error_msg)