g (export_settings_dialog.py)
**Purpose**: Configure automatic export behavior

**Settings**:
- **Export Triggers** (checkboxes):
  - After Peak Integration
  - After MS Library Search
  - After Manual Peak Assignment
  - During Batch Processing

- **Export Formats**:
  - JSON Format (always enabled)
  - CSV Format (RESULTS.CSV)
  - Filename format presets for each

- **Buttons**: OK, Cancel, Restore Defaults

**Persistence**: Saves to QSettings

---

### 3.3 ParameterVisibilityDialog (parameter_visibility_dialog.py)
**Purpose**: Show/hide parameter sections in ParametersFrame

**Visible Sections**:
- Signal Smoothing
- Baseline Correction
- Advanced Baseline (MS, TIC Align, Break Points)
- Peak Detection
- Negative Peak Detection
- Shoulder Detection
- Peak Range Filters
- Peak Grouping

**Returns**: `get_visibility()` → dict of {section: bool}

---

### 3.4 ScalingFactorsDialog (scaling_factors_dialog.py)
**Purpose**: Adjust signal and area scaling factors for data

**Features**:
- **Spinboxes** for signal factor and area factor (precision: 6 decimals)
- **Preset Manager**:
  - Save current as preset
  - Load saved preset
  - Delete preset
  - Restore defaults (1.0, 1.0)

**Signals Emitted**:
```python
factors_changed = Signal(float, float)  # signal_factor, area_factor
```

**Persistence**: Saves presets and values to QSettings

---

### 3.5 EditAssignmentDialog (edit_assignment_dialog.py)
**Purpose**: Manually assign compound name to a peak

**Features**:
- Peak info display (number, RT)
- Current assignment display
- Compound name search (autocomplete after 5 chars)
- Results list (max 50 items)
- **Cross-File Application** (if multiple files):
  - Checkbox to enable
  - RT tolerance input (default 0.05 min)
  - Spectral similarity threshold (default 0.7)
  - Explanation text

**Returns**:
- `get_selected_compound()` → compound name
- `should_apply_to_files()` → bool
- `get_rt_tolerance()` → float
- `get_similarity_threshold()` → float

**Signal Emitted**:
```python
apply_to_files_requested = Signal(str, float, float, object)
  # compound_name, rt, tolerance, spectrum
```

---

### 3.6 BatchJobDialog (batch_job_dialog.py)
**Purpose**: Set up batch processing job

**Features**:
- Directory list (extended selection)
- Add/Remove directory buttons
- Context menu (right-click)

**Options**:
- Perform peak integration (checkbox)
- Perform MS library search (checkbox)
- Save integration results to JSON (checkbox)
- Export results to CSV (checkbox)
- Overwrite existing result files (checkbox)

**Returns**: Emits `start_batch` signal with directories and options

---

### 3.7 AutomationDialog (automation_dialog.py)
**Purpose**: Show real-time progress during batch automation

**Panels**:
1. **File Progress**
   - Filename label
   - Current step description
   - Progress bar (0-100% per file)

2. **Overall Progress**
   - Label showing current/total files
   - Progress bar (0-100% overall)

3. **Processing Log**
   - Read-only text area
   - Timestamped log messages
   - Auto-scrolls to bottom

**Buttons**:
- Cancel (during processing)
- Close (after completion)

**Methods**:
- `update_file_progress(filename, step, percent)`
- `update_overall_progress(current, total)`
- `add_log_message(message)` (auto-timestamp)
- `mark_completed(success)` / `mark_error(msg)` / `mark_cancelled()`

**Signal Emitted**:
```python
cancelled = Signal()  # Cancel button clicked
```

---

### 3.8 BatchProgressDialog (batch_progress_dialog.py)
**Purpose**: Manage batch queue processing across multiple directories

**Main Components**:
1. **Queue Tree Widget**
   - Columns: Directory, Status, Progress (bar), Details
   - Color-coded status: queued (gray), processing (blue), completed (green), failed (red), skipped (orange)
   - Context menu for queue management

2. **Processing Log**
   - Timestamped messages
   - Separate from per-directory details

3. **Overall Progress Bar**
   - Shows completed/total directories

**Buttons**:
- Add Directory... (add to queue)
- Cancel (stop processing)
- Close (after completion)

**Methods**:
- `update_directory_status(directory, status, progress, details, error)`
- `update_overall_progress(completed, total, percent)`
- `update_file_progress(directory, filename, step, percent)`
- `add_log_message(message)`
- `populate_queue_tree()`
- `show_context_menu(position)` with actions:
  - Show Details
  - Skip (queued items)
  - Retry (failed items)
  - Remove from Queue

**Signals Emitted**:
```python
cancelled = Signal()  # Cancel button clicked
modify_queue = Signal(list)  # Updated directory list
```

---

### 3.9 MSOptionsDialog (ms_options_dialog.py)
**Purpose**: Advanced MS search options (large file, not fully shown)
- Search method selection
- Extraction parameters
- Similarity weighting
- Quality check options
- Peak saturation detection
- Skewness/coherence thresholds

---

---

## 4. WORKER CLASSES (Background Threading)

### 4.1 AutomationWorker (logic/automation_worker.py)
**Type**: `QRunnable` (runs in QThreadPool)

**Purpose**: Batch process all .D files in a directory

**Constructor**:
```python
AutomationWorker(app, directory_path)
```

**Signals** (via `AutomationWorkerSignals`):
```python
started(int)  # Total files to process
file_started(str, int, int)  # filename, file_index, total
file_progress(str, str, int)  # filename, step_description, percent
file_completed(str, bool, str)  # filename, success, message
log_message(str)
overall_progress(int, int, int)  # current, total, percent
finished()
error(str)
```

**Processing Steps Per File**:
1. Load file (10%)
2. Process & integrate (40%)
3. Save integration results (50%)
4. Perform MS search (if enabled)
5. Export results (if enabled)

**State**:
- `self.cancelled = False`: Can be set to stop processing
- `self.current_file_index`: Track progress

---

### 4.2 BatchSearchWorker (logic/batch_search.py)
**Type**: `QRunnable`

**Purpose**: MS library search on all integrated peaks

**Constructor**:
```python
BatchSearchWorker(ms_toolkit, peaks, data_directory, options)
```

**Signals** (via `BatchSearchWorkerSignals`):
```python
started(int)  # Total peaks to search
progress(int, str, object)  # peak_index, peak_name, search_results
finished()
error(str)
log_message(str)
```

**Processing**:
- Iterates through peaks
- **Skips manually assigned peaks** (preserves manual assignments)
- Extracts spectrum using SpectrumExtractor
- Searches MSToolkit
- Emits progress for each peak

**State**:
- `self.cancelled = False`
- `self.search_completed = False`

---

### 4.3 MS Baseline Worker (logic/ms_baseline_worker.py)
**Type**: `QRunnable`

**Purpose**: Perform baseline correction on TIC data

**Constructor**:
```python
MSBaselineWorker(tic_data, baseline_method, lambda_val)
```

**Signals**:
```python
finished(corrected_tic)
error(str)
```

---

## 5. COMPLETE USER WORKFLOW: Load → Integrate → Search → Quantitate → Export

### Phase 1: File Loading
```
1. User clicks "Open Folder" in FileTreeFrame
2. User double-clicks .D directory in tree
3. on_file_selected() called
   ├─ Clear previous peak data
   ├─ Load .D file via DataHandler
   ├─ Interpolate to 10,000 points
   ├─ Auto-detect detector channel (if multiple)
   ├─ Load TIC if MS data available
   ├─ Enable MS tab if MS data present
   └─ Call process_and_display()
4. PlotFrame displays:
   ├─ TIC chromatogram (if MS data)
   └─ Raw signal chromatogram
5. Status bar shows: "Loaded: filename.D (1/25) - with MS data"
6. Export button enabled
```

### Phase 2: Signal Processing
```
1. User adjusts parameters in ParametersFrame:
   ├─ Smoothing method/lambda
   ├─ Baseline method/lambda
   ├─ Peak detection settings
   └─ etc.
2. Each change triggers parameters_changed signal
3. on_parameters_changed() called
   ├─ Calls process_and_display()
   ├─ Re-applies all processing steps:
      ├─ Smoothing (Whittaker/SavGol/Median)
      ├─ Baseline correction (ARPLS/ASLS/etc.)
      ├─ Deconvolution (if enabled)
      └─ Peak detection preview
   └─ Updates PlotFrame with processed signal
4. User sees real-time signal changes in plot
5. Peaks overlay shown if peak detection enabled
```

### Phase 3: Peak Integration
```
1. User clicks "Integrate" button
2. on_integrate() called
   ├─ Validates peak detection enabled
   ├─ Get MS data if available
   ├─ Call integrate_peaks_no_ui()
   │   ├─ Detect peaks (find_peaks)
   │   ├─ Calculate areas (trapezoidal)
   │   ├─ Apply RT matching from RTTable
   │   ├─ Run quality checks (if enabled)
   │   └─ Return Peak objects
   ├─ plot_frame.shade_integration_areas()
   │   └─ Draw filled polygons under peaks
   ├─ Auto-export via export_manager:
   │   ├─ Save JSON with peak data
   │   └─ Save CSV (if enabled)
   ├─ Enable batch MS search button
   └─ Show "View Integration Results" button
3. Status bar: "Integration complete: N peaks found"
4. User can view results by clicking button
```

### Phase 4: MS Library Search
#### Option A: Single Peak Search
```
1. User right-clicks peak on plot
2. Selects "Search Library"
3. on_ms_search_requested(peak_index) called
   ├─ Extract peak spectrum
   ├─ Search MSToolkit
   ├─ Get top N matches
   └─ Display in MSFrame results tree
4. User can view match spectrum and score
5. Can click match to view comparison
```

#### Option B: Batch Search (all peaks)
```
1. User clicks "MS Search All" button
2. run_batch_ms_search() called
   ├─ Validate peaks integrated
   ├─ Create BatchSearchWorker
   ├─ Show BatchProgressDialog
   ├─ Start worker in QThreadPool
   ├─ For each peak:
   │   ├─ Skip if manually assigned
   │   ├─ Extract spectrum
   │   ├─ Search library
   │   ├─ Emit progress signal
   │   │  └─ Dialog updates with match info
   │   └─ Auto-assign top match
   ├─ After all peaks:
   │   ├─ Auto-export with new assignments
   │   ├─ Update plot annotations
   │   └─ Dialog shows completion summary
   └─ Close dialog
3. Status bar: "Batch search complete: X matches found"
```

### Phase 5: Manual Assignment (optional)
```
1. User right-clicks peak on plot
2. Selects "Edit Assignment"
3. EditAssignmentDialog opens
   ├─ Shows current assignment
   ├─ Search field for new compound
   ├─ Results list (autocomplete)
   ├─ User selects compound
   ├─ [Optional] Check "Apply to other files"
   │   ├─ Set RT tolerance (default 0.05 min)
   │   └─ Set spectral similarity threshold (0.7)
   └─ Click Save
4. Assignment updated
5. If cross-file enabled:
   ├─ apply_assignment_to_files() called
   ├─ Searches all processed files in directory
   ├─ Finds peaks with similar RT ± tolerance
   ├─ Matches spectrum similarity
   ├─ Applies assignment to matches
   └─ Auto-export updated results
6. Peak annotations updated on plot
7. JSON results updated
```

### Phase 6: Retention Time (RT) Table
```
1. User clicks "RT Table" tab
2. RTTableFrame shows current ranges
3. User can:
   ├─ Add range: specify start/end time
   ├─ Delete range: select and delete
   ├─ Load from file: import pre-defined ranges
   └─ Save to file: export for reuse
4. During integration:
   ├─ Peaks matched to RT ranges
   ├─ Peak grouped by assigned range
   └─ Results include RT assignment
```

### Phase 7: Quantitation
```
1. User clicks "Quantitation" tab
2. Check "Enable Quantitation"
3. Enter internal standard:
   ├─ Type compound name
   ├─ Click "Search MS Library"
   ├─ System auto-fills: Formula, MW
   ├─ User enters: Density, Volume added (µL)
   └─ Display: mol C of IS
4. Enter sample info:
   ├─ Sample mass (mg)
   ├─ Sample volume (mL)
   └─ Dilution factor
5. Click "Quantitate"
   ├─ _perform_quantitation() called
   ├─ Calculate mol C for each peak
   ├─ Calculate total organic carbon
   ├─ Calculate compound-specific quantities
   └─ Display results
6. Results persist in JSON export
```

### Phase 8: Navigation
```
1. User clicks "Back" or "Next" buttons
2. on_previous_sample() / on_next_sample() called
   ├─ DataHandler.navigate_to_previous/next()
   ├─ Get path to adjacent .D file
   └─ Call on_file_selected(next_path)
3. Workflow repeats from Phase 1
4. Status bar shows: "Loaded: filename2.D (2/25)"
```

### Phase 9: Batch Automation
```
1. User clicks "Batch Process All Files" button
2. start_automation() called
   ├─ Create AutomationWorker for parent directory
   ├─ Show AutomationDialog
   ├─ Start worker in QThreadPool
3. Worker processes each .D file:
   ├─ Load (same as Phase 1)
   ├─ Integrate (same as Phase 3)
   ├─ [If enabled] MS search (same as Phase 4B)
   ├─ [If enabled] Quantitate
   ├─ Export results
   └─ Move to next file
4. Dialog shows real-time progress:
   ├─ Current file name and step
   ├─ Per-file progress bar
   ├─ Overall progress (X/Total)
   ├─ Timestamped log
5. User can:
   ├─ Watch progress
   ├─ Review log messages
   └─ Cancel at any time
6. On completion:
   ├─ Dialog shows summary
   ├─ All results exported (JSON + CSV)
   ├─ Status bar shows completion time
   └─ User can review results
```

### Phase 10: Batch Queue (multiple directories)
```
1. User selects Process → Batch Process Directories...
2. BatchJobDialog opens
3. User:
   ├─ Adds directories via file browser
   ├─ Sets processing options:
   │   ├─ ☑ Perform peak integration
   │   ├─ ☑ Perform MS search
   │   ├─ ☑ Save results to JSON
   │   ├─ ☑ Export to CSV
   │   └─ ☑ Overwrite existing
   └─ Clicks "Start Processing"
4. BatchProgressDialog opens
5. Worker processes directories sequentially:
   ├─ First directory:
   │   ├─ Automation runs (all .D files)
   │   ├─ Status: queued → processing → completed/failed
   │   ├─ Progress bar updated
   │   └─ Details shown in tree
   ├─ Log messages timestamped
   ├─ User can:
   │   ├─ Right-click directory
   │   ├─ Skip (if queued)
   │   ├─ Retry (if failed)
   │   └─ Remove from queue
   └─ Continue to next directory
6. On completion:
   ├─ Summary dialog
   ├─ Show successes/failures/skipped
   └─ All results exported
```

### Phase 11: Export & Data Access
```
Throughout workflow, automatic exports occur:

After Integration (if enabled):
├─ Create JSON: metadata + peak list + areas
└─ Create CSV: RESULTS.CSV format

After MS Search (if enabled):
├─ Update JSON with match_results
└─ Update CSV with compound names

After Manual Assignment (if enabled):
└─ Update JSON with assignment

User can also:
├─ Click "Export" button (manual export)
├─ Settings → Export Settings... (configure)
├─ Tools → JSON ↔ Excel Converter...
│   ├─ Convert JSON to XLSX
│   ├─ Convert XLSX to JSON
│   └─ Batch convert
└─ File browser: results saved in .D directory
```

### Phase 12: Settings & Persistence
```
User can customize:

Settings → Select Detector Channel...
├─ Choose detector if multiple available
└─ Reload file with new channel

Settings → Export Settings...
├─ Toggle export triggers
├─ Choose export formats
└─ Set filename patterns

Settings → Scaling Factors...
├─ Adjust signal/area multipliers
├─ Save/load presets
└─ Match different instrument versions

Settings → Configure Visible Parameters...
├─ Show/hide parameter sections
└─ Customize UI for workflow

Settings → Toggle Dark/Light Mode
├─ Switch theme
└─ All plots update automatically

All settings persisted via QSettings (Windows Registry / macOS plist / Linux ~/.config)
```

---

## 6. KEY DATA STRUCTURES

### Peak Object
```python
class Peak:
    peak_number: int  # 1-indexed
    retention_time: float  # minutes
    start_time: float
    end_time: float
    height: float  # signal intensity
    area: float  # integrated area
    width: float  # peak width
    compound_id: str  # assigned compound name
    match_results: list  # [(compound, score), ...]
    manual_assignment: bool  # User-assigned?
    spectrum: np.ndarray  # MS spectrum data (mz, intensity)
    rt_assignment: str  # Which RT range group
```

### Integration Results
```python
{
    'peaks': [Peak, ...],
    'timestamp': datetime,
    'detector': str,
    'parameters_used': dict,
    'quality_flags': dict,
    'total_area': float,
    'total_height': float
}
```

### Export Data (JSON)
```json
{
    "metadata": {
        "file": "Sample1.D",
        "timestamp": "2024-01-15T10:30:00",
        "detector": "FID",
        "parameters": {...}
    },
    "peaks": [
        {
            "peak_number": 1,
            "retention_time": 2.345,
            "area": 12345.67,
            "height": 1234.5,
            "compound_id": "Decane",
            "match_results": [
                {"compound": "Decane", "score": 0.95},
                {"compound": "Nonane", "score": 0.82}
            ]
        }
    ]
}
```

---

## 7. NOTABLE IMPLEMENTATION DETAILS

### Thread Safety
- AutomationWorker and BatchSearchWorker use QRunnable + QThreadPool
- Signals emitted from workers are queued (thread-safe)
- Main thread updates UI via signal connections
- Uses `Qt.ConnectionType.QueuedConnection` for worker signals

### Data Interpolation
- Raw chromatograms interpolated to 10,000 points at load time
- Ensures consistent processing pipeline
- Original data preserved for export metadata

### Signal Processing Isolation
- Parameters apply live during `process_and_display()`
- Real-time feedback as user adjusts sliders
- No blocking operations in main thread

### Cross-File Assignment
- Dot product similarity matching
- RT tolerance window matching
- Preserves manual assignments during batch MS search

### Export Strategy
- JSON: Comprehensive, all metadata
- CSV: Compatibility with legacy GCMS software
- Auto-triggered by ExportManager based on settings
- Manual export available via buttons

EOF
cat /tmp/chromakit_analysis.md

                EC=$?
                echo "___BEGIN___COMMAND_DONE_MARKER___$EC"
            }
___BEGIN___COMMAND_OUTPUT_MARKER___
# ChromaKit PySide6 Desktop Application - Comprehensive Analysis

## 1. MAIN APPLICATION: ChromaKitApp (ui/app.py - 3600 lines)

### Core Architecture
- **Main Window**: `QMainWindow` (1600x800 default)
- **Central Widget Layout**: `QHBoxLayout` with 3 main sections:
  1. **Left**: FileTreeFrame (file browser)
  2. **Center**: Plot + Button frames (vertically stacked)
  3. **Right**: Tabbed widget with 4 panels

### Instance Variables & State Management
```python
self.processor = ChromatogramProcessor()  # Signal processing
self.data_handler = DataHandler()  # File I/O and data loading
self.export_manager = ExportManager()  # Export orchestration
self.real_data_loaded = bool  # Flag for real vs. sample data

# Scaling factors (persistent via QSettings)
self.signal_factor, self.area_factor = 1.0, 1.0

# Data state
self.current_x, self.current_y  # Interpolated chromatogram (10k points)
self.original_x, self.original_y  # Raw data
self.current_processed  # Post-baseline correction signal
self.integrated_peaks  # List of Peak objects from integration
self.integration_results  # Dict with peaks, areas, metadata
self.current_directory_path  # Active .D file path
```

### Major Features & Workflows

#### A. FILE LOADING & NAVIGATION
**Method**: `on_file_selected(file_path, batch_mode=False, detector=None)`
- **Purpose**: Load a .D chromatography data directory
- **Features**:
  - Auto-detects available detectors (can specify one)
  - Loads chromatogram + MS data (if available)
  - Interpolates raw data to 10,000 points for consistent processing
  - Enables/disables MS tab based on data availability
  - Displays TIC (Total Ion Chromatogram) when MS data present
  - Updates status bar with file info and navigation position (e.g., "1/25 files")

**Navigation Methods**:
- `on_next_sample()` / `on_previous_sample()`: Navigate through .D directories
- Uses `DataHandler.navigate_to_next()` / `navigate_to_previous()`

**Signal Connections**:
```
file_tree.file_selected → on_file_selected()
button_frame.back_clicked → on_previous_sample()
button_frame.next_clicked → on_next_sample()
```

#### B. DATA PROCESSING & VISUALIZATION
**Method**: `process_and_display(x, y, new_file=False)`
- **Purpose**: Apply signal processing and update chromatogram plot
- **Processing Pipeline**:
  1. Get current parameters from ParametersFrame
  2. Apply smoothing (Whittaker/SavGol/median filter)
  3. Apply baseline correction (ARPLS, ASLS, etc.)
  4. Apply deconvolution (if enabled) via `_apply_deconvolution()`
  5. Update PlotFrame with processed signal

**Method**: `_apply_deconvolution(processed, params)`
- Implements EMG (Exponentially Modified Gaussian) or Geometric splitting
- Returns deconvolved peaks with amplitude/position metadata

**Signal Connections**:
```
parameters_frame.parameters_changed → on_parameters_changed()
  ↓ calls process_and_display()
```

#### C. PEAK INTEGRATION
**Method**: `on_integrate()` → `integrate_peaks_no_ui(ms_data, quality_options)`
- **Purpose**: Detect and integrate peaks in the processed chromatogram
- **Workflow**:
  1. Check peak detection enabled (positive or negative)
  2. Get MS data if available (for quality assessment)
  3. Call `integrate_peaks_no_ui()` for core integration
  4. Detect peaks using scipy.signal.find_peaks
  5. Calculate areas using trapezoidal rule
  6. Apply RT matching from RT Table (if configured)
  7. Shade integration areas on plot
  8. Auto-export results (JSON/CSV) via ExportManager
  9. Enable batch MS search button
  10. Show "View Integration Results" button

**Quality Options** (from MS frame):
- Peak skewness check
- Coherence check (if MS data available)
- Saturation detection

**Integration Results Structure**:
```python
{
  'peaks': [Peak objects],  # With area, height, RT, width
  'timestamp': datetime,
  'detector': str,
  'parameters_used': dict,
  'quality_flags': dict
}
```

**Signal Connections**:
```
button_frame.integrate_clicked → on_integrate()
  ↓ calls integrate_peaks_no_ui()
  ↓ calls plot_frame.shade_integration_areas()
  ↓ calls export_manager.export_after_integration()
  ↓ emits on_peaks_integrated()
```

#### D. MASS SPECTROMETRY ANALYSIS
**Frame**: MSFrame (right-side tab, "Mass Spectrometry")
- **Library**: MSToolkit (external)
- **Search Methods**: Vector, Composite, Hybrid
- **Default Search Options**:
  - Extraction method: apex (or apex+flank)
  - Range points: 5, Range time: 0.05 min
  - TIC weighting enabled
  - Subtraction method: min_tic
  - Similarity metric: composite or NIST_GC
  - Top N matches: 5

**MS Workflow**:
1. **Peak Spectrum Extraction**: 
   - `on_peak_spectrum_requested(peak_index)`: Extract spectrum from specific peak
   - `on_ms_spectrum_requested(retention_time)`: Extract full spectrum at RT

2. **MS Search**:
   - `on_ms_search_requested(peak_index)`: Search library for one peak
   - `run_batch_ms_search()`: Search all integrated peaks (threaded)
   
3. **Batch Search Process** (BatchSearchWorker in logic/batch_search.py):
   - Iterates through all peaks
   - Extracts spectrum using SpectrumExtractor
   - Searches MSToolkit library
   - Skips manually assigned peaks (preserves manual assignments)
   - Emits progress signals → updates BatchProgressDialog
   - Auto-exports results on completion

**Signal Connections**:
```
plot_frame.ms_search_requested → on_ms_search_requested()
plot_frame.ms_spectrum_requested → on_ms_spectrum_requested()
plot_frame.peak_spectrum_requested → on_peak_spectrum_requested()
ms_frame.search_completed → on_ms_search_completed()
button_frame.batch_search_clicked → run_batch_ms_search()
```

#### E. PEAK ASSIGNMENT & EDITING
**Method**: `on_edit_assignment_requested(peak_index)` → `EditAssignmentDialog`
- **Purpose**: Manually assign compound name to a peak
- **Features**:
  - Search library compounds (autocomplete)
  - Apply assignment to other files (with RT tolerance + spectral similarity matching)
  - Cross-file matching uses dot product similarity
  
**Method**: `apply_assignment_to_files(compound_name, rt, tolerance, spectrum)`
- Finds matching peaks in all processed files in directory
- Applies same compound assignment
- Auto-exports updated results

**Signal Connections**:
```
plot_frame.edit_assignment_requested → on_edit_assignment_requested()
  ↓ shows EditAssignmentDialog
  ↓ if cross-file: calls apply_assignment_to_files()
```

#### F. RETENTION TIME (RT) TABLE MANAGEMENT
**Frame**: RTTableFrame (right-side tab)
- **Purpose**: Define RT windows for peak grouping
- **Features**:
  - Add RT ranges (start time, end time)
  - Delete ranges
  - Apply grouping to integration

**RT Assignment**: `on_rt_assignment_requested(peak_index)`
- Opens RTAssignmentDialog
- User selects which RT range the peak belongs to
- Updates peak metadata

**RT Matching**: `_apply_rt_matching_to_peaks(peaks)`
- Called during integration
- Groups peaks by their assigned RT ranges
- Affects peak grouping in results

**Signal Connections**:
```
rt_table_frame.rt_table_changed → on_rt_table_changed()
plot_frame.rt_assignment_requested → on_rt_assignment_requested()
plot_frame.add_to_rt_table_requested → on_add_to_rt_table_requested()
```

#### G. QUANTITATION
**Frame**: QuantitationFrame (right-side tab)
- **Method**: Polyarc + Internal Standard
- **Workflow**:
  1. Enter internal standard compound name
  2. Search MS library for formula & molecular weight
  3. User enters: density, volume added
  4. Calculate mol C of internal standard
  5. Enter sample mass/volume
  6. Click "Quantitate" → `_perform_quantitation()`
  7. Calculates total carbon and compound-specific quantities

**Calculations**:
```python
mol_c_is = (volume_ul * density_g_ml * 1e-3) / (mw_g_mol / num_carbons)
total_c_ng = sum([area × area_factor] for integrated peaks)
compound_quantity = (total_c_ng / mol_c_is) × sample_mass_factor
```

**Signal Connections**:
```
quantitation_frame.requantitate_requested → _perform_quantitation()
quantitation_frame.quantitation_changed → on_quantitation_changed()
```

#### H. BATCH PROCESSING & AUTOMATION
**Method**: `start_automation()` → `AutomationWorker` (threaded)
- **Purpose**: Process all .D files in a directory
- **Workflow**:
  1. User clicks "Batch Process All Files" button
  2. AutomationDialog appears with progress bars
  3. AutomationWorker runs in QThreadPool
  4. For each .D file:
     - Load file (10%)
     - Process & integrate (40%)
     - Save results (50%)
     - Perform MS search (if enabled)
     - Export results (if enabled)
  5. Progress signals update dialog in real-time
  6. Shows log of all operations

**Worker Class**: `AutomationWorker` (logic/automation_worker.py)
- `QRunnable` for thread pool execution
- Emits signals: started, file_started, file_progress, log_message, overall_progress, finished, error
- Can be cancelled: `cancel_automation()` sets `worker.cancelled = True`

**Signal Connections**:
```
button_frame.automation_clicked → start_automation()
automation_worker.signals.* → automation_dialog updates
automation_dialog.cancelled → cancel_automation()
automation_worker.signals.finished → on_automation_finished()
```

#### I. BATCH MS SEARCH (all integrated peaks)
**Method**: `run_batch_ms_search()` → `BatchSearchWorker` (threaded)
- **Purpose**: Search MS library for all peaks in current file
- **Workflow**:
  1. Check peaks integrated
  2. Create BatchSearchWorker with peaks list
  3. Show BatchProgressDialog
  4. Worker searches each peak (skips manually assigned)
  5. Updates dialog with match results
  6. Auto-exports with updated assignments
  7. Shows completion summary

**Worker Class**: `BatchSearchWorker` (logic/batch_search.py)
- Uses SpectrumExtractor to get peak spectra
- Iterates through peaks, skips manual assignments
- Emits progress signals for each peak

**Signal Connections**:
```
button_frame.batch_search_clicked → run_batch_ms_search()
batch_search_worker.signals.progress → _update_batch_search_progress()
batch_search_worker.signals.finished → _on_batch_search_finished()
```

#### J. BATCH QUEUE PROCESSING (multiple directories)
**Method**: `show_batch_queue_dialog()` → `BatchJobDialog` + `BatchProgressDialog`
- **Purpose**: Process multiple directories sequentially
- **Features**:
  - Add/remove directories from queue
  - Set processing options per directory
  - Real-time queue management (skip, retry, remove)
  - Overall progress tracking
  - Per-directory status (queued, processing, completed, failed)

**Dialog**: `BatchProgressDialog` (ui/dialogs/batch_progress_dialog.py)
- Tree view showing all directories with status colors
- Progress bars per directory
- Log panel with timestamps
- Context menu for queue management

#### K. EXPORT & PERSISTENCE
**ExportManager** (logic/export_manager.py):
- **Triggers**: After integration, after MS search, after assignment, during batch
- **Formats**: JSON (comprehensive) + CSV (RESULTS.CSV)
- **Automatic**: Controlled by ExportSettingsDialog

**Export Methods**:
- `export_after_integration(peaks, d_path, detector)`
- `export_after_ms_search(peaks, d_path)`
- `export_after_assignment(peak, d_path)`

**Export Settings Dialog**:
- Checkbox for each trigger
- Filename format customization
- Restore defaults button

**JSON Structure**:
```python
{
  'peaks': [
    {
      'peak_number': int,
      'retention_time': float,
      'area': float,
      'height': float,
      'compound_id': str,
      'match_results': [...]
    }
  ],
  'metadata': {
    'timestamp': datetime,
    'detector': str,
    'parameters_used': dict
  }
}
```

#### L. THEMING & UI
**Theme System**: Light/Dark mode toggle
- **Method**: `toggle_theme()` → `apply_stylesheet(theme)`
- **Components**:
  - QSS stylesheet (ui/style.qss)
  - Matplotlib color scheme
  - Widget-specific theme properties
  
**Matplotlib Theme Colors**:
```python
Dark: background='#23272e', axes='#23272e', line1='#4cc2ff', line2='#ff7f50', line3='#7fff7f'
Light: background='#f6f7fa', axes='#f6f7fa', line1='#2B6A99', line2='#8B3C41', line3='#3C6E47'
```

#### M. SETTINGS & PERSISTENCE (QSettings)
**Organization**: "CalebCoatney", "ChromaKit"
- Scaling factors (signal_factor, area_factor)
- Export settings
- Parameter visibility
- RT table configurations
- Detector selection
- Preset scaling factors

### Menu Bar Structure
```
Process
├── Batch Process Directories...  → show_batch_queue_dialog()

Settings
├── Select Detector Channel...     → show_detector_selection_dialog()
├── Export Settings...             → show_export_settings_dialog()
├── Scaling Factors...             → show_scaling_factors_dialog()
├── Configure Visible Parameters... → show_parameter_visibility_dialog()
└── Toggle Dark/Light Mode         → toggle_theme()

Tools
└── JSON ↔ Excel Converter...      → show_json_excel_converter()
```

---

## 2. UI FRAMES (ui/frames/)

### 2.1 PlotFrame (plot.py - 1166 lines)
**Purpose**: Display chromatograms and mass spectra

**Layout**: Matplotlib figure with 2 subplots (TIC on top, chromatogram below)

**Key Methods**:
- `plot_chromatogram(data, show_corrected, new_file)`: Main chromatogram display
- `plot_tic(x, y, show_baseline, new_file)`: TIC display
- `shade_integration_areas(integration_results)`: Overlay integration areas
- `_on_plot_click(event)`: Handle mouse clicks on plot (context menus for peaks)
- `_highlight_selected_peak(peak)`: Highlight selected peak and show info
- `clear_peak_data()`: Clear all peak overlays
- `_show_peak_context_menu(peak_index, event)`: Peak right-click menu

**Signals Emitted**:
```python
point_selected = Signal(float)  # X value clicked
ms_spectrum_requested = Signal(float)  # RT for spectrum extraction
peak_spectrum_requested = Signal(int)  # Peak index for spectrum
ms_search_requested = Signal(int)  # Peak index for library search
edit_assignment_requested = Signal(int)  # Peak index for editing
rt_assignment_requested = Signal(int)  # Peak index for RT assignment
add_to_rt_table_requested = Signal(int)  # Peak index for RT table
```

**Peak Context Menu**:
- View Spectrum
- Search Library
- Edit Assignment
- Add to RT Table
- RT Assignment

**Data Storage**:
- `self.chromatogram_data`: Current displayed chromatogram
- `self.tic_data`: TIC data
- `self.integrated_peaks`: List of Peak objects with visual overlays

---

### 2.2 FileTreeFrame (tree.py - 98 lines)
**Purpose**: File browser for navigating .D directories

**Features**:
- "Open Folder" button to set root directory
- QTreeView showing .D directories (filtered)
- Sorting by name
- Double-click to load

**Signals Emitted**:
```python
file_selected = Signal(str)  # Path to selected .D directory
```

**Key Methods**:
- `open_directory()`: File browser dialog
- `set_root_path(path)`: Update tree root
- `on_item_double_clicked(index)`: Emit file_selected signal

---

### 2.3 ParametersFrame (parameters.py - 1591 lines)
**Purpose**: Control signal processing parameters

**Sections** (scrollable, can be toggled via dialog):
1. **Signal Smoothing**
   - Enable/disable
   - Method: Whittaker, SavGol, Median
   - Lambda, window size, polynomial order

2. **Baseline Correction**
   - Method: ARPLS (default), ASLS, AirPLS, etc.
   - Lambda, asymmetry
   - TIC alignment (experimental)
   - Break points editor (manual baseline anchors)
   - FastChrom settings

3. **Peak Detection**
   - Mode: Classical (find_peaks) or Deconvolution
   - Min prominence, height, width
   - Range filters (time windows to exclude)

4. **Negative Peak Detection**
   - Enable/disable
   - Min prominence

5. **Shoulder Detection**
   - Window length, polyorder, sensitivity, apex distance

6. **Peak Grouping**
   - Define peak groups by RT ranges
   - Table editor

**Signals Emitted**:
```python
parameters_changed = Signal(dict)  # Full parameter dict
ms_baseline_clicked = Signal()  # MS baseline correction button
```

**Key Methods**:
- `get_parameters()`: Return current parameter dict
- `set_parameters(params)`: Restore parameter dict
- `_on_parameter_changed()`: Emit signal on any change
- `set_section_visibility(visibility_dict)`: Show/hide sections

**Data Structure**:
```python
{
  'smoothing': {'enabled', 'method', 'lambda', 'median_kernel', 'savgol_window', 'savgol_polyorder'},
  'baseline': {'method', 'lambda', 'asymmetry', 'align_tic', 'break_points', 'fastchrom'},
  'peaks': {'enabled', 'mode', 'min_prominence', 'min_height', 'min_width', 'range_filters'},
  'negative_peaks': {'enabled', 'min_prominence'},
  'shoulders': {'enabled', 'window_length', 'polyorder', 'sensitivity', 'apex_distance'},
  'integration': {'peak_groups'}
}
```

---

### 2.4 MSFrame (ms.py - 1019 lines)
**Purpose**: MS library search configuration and results display

**Features**:
- MSToolkit library loading
- Search method selection (Vector, Composite, Hybrid)
- Extraction method (apex, apex+flank)
- Spectrum subtraction controls
- Similarity metric (NIST_GC, etc.)
- Top N matches display (tree widget)
- Match scoring visualization
- MS Baseline Correction button (for TIC)

**Search Options**:
```python
{
  'search_method': 'vector',
  'hybrid_method': 'auto',
  'extraction_method': 'apex',
  'range_points': 5,
  'range_time': 0.05,
  'tic_weight': True,
  'subtract_enabled': True,
  'subtraction_method': 'min_tic',
  'subtract_weight': 0.1,
  'similarity': 'composite',
  'weighting': 'NIST_GC',
  'unmatched': 'keep_all',
  'intensity_power': 0.6,
  'top_n': 5,
  'top_k_clusters': 1
}
```

**Signals Emitted**:
```python
search_completed = Signal()  # After batch search completes
```

**Key Methods**:
- `_init_mstoolkit()`: Load MSToolkit library (async)
- `update_ms_results(results)`: Display search matches
- `_create_ms_tools()`: Build UI controls

**Results Display**:
- Tree widget with compound names and match scores
- Double-click compound → view spectrum

---

### 2.5 RTTableFrame (rt_table.py - 1124 lines)
**Purpose**: Define retention time ranges for peak grouping

**Features**:
- Table showing RT ranges (start, end time)
- Add/delete ranges
- Load/save RT table from file
- Auto-apply to integration

**Table Columns**:
- Start Time (min)
- End Time (min)
- Width (calculated)
- Notes (optional)

**Signals Emitted**:
```python
rt_table_changed = Signal(dict)  # Updated RT table settings
```

**Key Methods**:
- `add_rt_range(start, end, notes)`: Add new range
- `delete_rt_range(index)`: Remove range
- `get_rt_table()`: Return current ranges
- `load_from_file(filepath)`: Import RT table
- `save_to_file(filepath)`: Export RT table

---

### 2.6 QuantitationFrame (quantitation.py - 346 lines)
**Purpose**: Quantitative analysis using Polyarc + Internal Standard method

**UI Sections**:
1. **Enable/Overwrite Checkbox**
   - Toggle quantitation on/off
   - Overwrite existing results option

2. **Internal Standard Information**
   - Compound name (with MS library search)
   - Formula (auto-filled)
   - Molecular weight (auto-filled)
   - Density (user input)
   - Volume added (µL) (user input)
   - Calculated mol C (read-only)

3. **Sample Preparation**
   - Sample mass (mg)
   - Sample volume (mL)
   - Dilution factor

4. **Results Display**
   - Total carbon (ng)
   - Compound-specific quantities

**Signals Emitted**:
```python
quantitation_changed = Signal()  # Settings changed
requantitate_requested = Signal()  # Recalculate button clicked
```

---

### 2.7 ButtonFrame (buttons.py - 71 lines)
**Purpose**: Control buttons for main operations

**Buttons**:
```
Row 1:
- Export (disabled until file loaded)
- Back (navigate previous file)
- Next (navigate next file)
- Integrate (detect & integrate peaks)

Row 2:
- Batch Process All Files (run automation)
- MS Search All (batch search integrated peaks)
```

**Signals Emitted**:
```python
export_clicked = Signal()
back_clicked = Signal()
next_clicked = Signal()
integrate_clicked = Signal()
automation_clicked = Signal()
batch_search_clicked = Signal()
```

---

## 3. UI DIALOGS (ui/dialogs/)

### 3.1 DetectorSelectionDialog (detector_dialog.py)
**Purpose**: Choose which detector channel to load

**Controls**:
- Dropdown of available detectors
- Info label explaining detector change requires reload
- OK/Cancel buttons

**Returns**: `get_selected_detector()` → detector name

---

### 3.2 ExportSettingsDialog (export_settings_dialog.py)
**Purpose**: Configure automatic export behavior

**Settings**:
- **Export Triggers** (checkboxes):
  - After Peak Integration
  - After MS Library Search
  - After Manual Peak Assignment
  - During Batch Processing

- **Export Formats**:
  - JSON Format (always enabled)
  - CSV Format (RESULTS.CSV)
  - Filename format presets for each

- **Buttons**: OK, Cancel, Restore Defaults

**Persistence**: Saves to QSettings

---

### 3.3 ParameterVisibilityDialog (parameter_visibility_dialog.py)
**Purpose**: Show/hide parameter sections in ParametersFrame

**Visible Sections**:
- Signal Smoothing
- Baseline Correction
- Advanced Baseline (MS, TIC Align, Break Points)
- Peak Detection
- Negative Peak Detection
- Shoulder Detection
- Peak Range Filters
- Peak Grouping

**Returns**: `get_visibility()` → dict of {section: bool}

---

### 3.4 ScalingFactorsDialog (scaling_factors_dialog.py)
**Purpose**: Adjust signal and area scaling factors for data

**Features**:
- **Spinboxes** for signal factor and area factor (precision: 6 decimals)
- **Preset Manager**:
  - Save current as preset
  - Load saved preset
  - Delete preset
  - Restore defaults (1.0, 1.0)

**Signals Emitted**:
```python
factors_changed = Signal(float, float)  # signal_factor, area_factor
```

**Persistence**: Saves presets and values to QSettings

---

### 3.5 EditAssignmentDialog (edit_assignment_dialog.py)
**Purpose**: Manually assign compound name to a peak

**Features**:
- Peak info display (number, RT)
- Current assignment display
- Compound name search (autocomplete after 5 chars)
- Results list (max 50 items)
- **Cross-File Application** (if multiple files):
  - Checkbox to enable
  - RT tolerance input (default 0.05 min)
  - Spectral similarity threshold (default 0.7)
  - Explanation text

**Returns**:
- `get_selected_compound()` → compound name
- `should_apply_to_files()` → bool
- `get_rt_tolerance()` → float
- `get_similarity_threshold()` → float

**Signal Emitted**:
```python
apply_to_files_requested = Signal(str, float, float, object)
  # compound_name, rt, tolerance, spectrum
```

---

### 3.6 BatchJobDialog (batch_job_dialog.py)
**Purpose**: Set up batch processing job

**Features**:
- Directory list (extended selection)
- Add/Remove directory buttons
- Context menu (right-click)

**Options**:
- Perform peak integration (checkbox)
- Perform MS library search (checkbox)
- Save integration results to JSON (checkbox)
- Export results to CSV (checkbox)
- Overwrite existing result files (checkbox)

**Returns**: Emits `start_batch` signal with directories and options

---

### 3.7 AutomationDialog (automation_dialog.py)
**Purpose**: Show real-time progress during batch automation

**Panels**:
1. **File Progress**
   - Filename label
   - Current step description
   - Progress bar (0-100% per file)

2. **Overall Progress**
   - Label showing current/total files
   - Progress bar (0-100% overall)

3. **Processing Log**
   - Read-only text area
   - Timestamped log messages
   - Auto-scrolls to bottom

**Buttons**:
- Cancel (during processing)
- Close (after completion)

**Methods**:
- `update_file_progress(filename, step, percent)`
- `update_overall_progress(current, total)`
- `add_log_message(message)` (auto-timestamp)
- `mark_completed(success)` / `mark_error(msg)` / `mark_cancelled()`

**Signal Emitted**:
```python
cancelled = Signal()  # Cancel button clicked
```

---

### 3.8 BatchProgressDialog (batch_progress_dialog.py)
**Purpose**: Manage batch queue processing across multiple directories

**Main Components**:
1. **Queue Tree Widget**
   - Columns: Directory, Status, Progress (bar), Details
   - Color-coded status: queued (gray), processing (blue), completed (green), failed (red), skipped (orange)
   - Context menu for queue management

2. **Processing Log**
   - Timestamped messages
   - Separate from per-directory details

3. **Overall Progress Bar**
   - Shows completed/total directories

**Buttons**:
- Add Directory... (add to queue)
- Cancel (stop processing)
- Close (after completion)

**Methods**:
- `update_directory_status(directory, status, progress, details, error)`
- `update_overall_progress(completed, total, percent)`
- `update_file_progress(directory, filename, step, percent)`
- `add_log_message(message)`
- `populate_queue_tree()`
- `show_context_menu(position)` with actions:
  - Show Details
  - Skip (queued items)
  - Retry (failed items)
  - Remove from Queue

**Signals Emitted**:
```python
cancelled = Signal()  # Cancel button clicked
modify_queue = Signal(list)  # Updated directory list
```

---

### 3.9 MSOptionsDialog (ms_options_dialog.py)
**Purpose**: Advanced MS search options (large file, not fully shown)
- Search method selection
- Extraction parameters
- Similarity weighting
- Quality check options
- Peak saturation detection
- Skewness/coherence thresholds

---

---

## 4. WORKER CLASSES (Background Threading)

### 4.1 AutomationWorker (logic/automation_worker.py)
**Type**: `QRunnable` (runs in QThreadPool)

**Purpose**: Batch process all .D files in a directory

**Constructor**:
```python
AutomationWorker(app, directory_path)
```

**Signals** (via `AutomationWorkerSignals`):
```python
started(int)  # Total files to process
file_started(str, int, int)  # filename, file_index, total
file_progress(str, str, int)  # filename, step_description, percent
file_completed(str, bool, str)  # filename, success, message
log_message(str)
overall_progress(int, int, int)  # current, total, percent
finished()
error(str)
```

**Processing Steps Per File**:
1. Load file (10%)
2. Process & integrate (40%)
3. Save integration results (50%)
4. Perform MS search (if enabled)
5. Export results (if enabled)

**State**:
- `self.cancelled = False`: Can be set to stop processing
- `self.current_file_index`: Track progress

---

### 4.2 BatchSearchWorker (logic/batch_search.py)
**Type**: `QRunnable`

**Purpose**: MS library search on all integrated peaks

**Constructor**:
```python
BatchSearchWorker(ms_toolkit, peaks, data_directory, options)
```

**Signals** (via `BatchSearchWorkerSignals`):
```python
started(int)  # Total peaks to search
progress(int, str, object)  # peak_index, peak_name, search_results
finished()
error(str)
log_message(str)
```

**Processing**:
- Iterates through peaks
- **Skips manually assigned peaks** (preserves manual assignments)
- Extracts spectrum using SpectrumExtractor
- Searches MSToolkit
- Emits progress for each peak

**State**:
- `self.cancelled = False`
- `self.search_completed = False`

---

### 4.3 MS Baseline Worker (logic/ms_baseline_worker.py)
**Type**: `QRunnable`

**Purpose**: Perform baseline correction on TIC data

**Constructor**:
```python
MSBaselineWorker(tic_data, baseline_method, lambda_val)
```

**Signals**:
```python
finished(corrected_tic)
error(str)
```

---

## 5. COMPLETE USER WORKFLOW: Load → Integrate → Search → Quantitate → Export

### Phase 1: File Loading
```
1. User clicks "Open Folder" in FileTreeFrame
2. User double-clicks .D directory in tree
3. on_file_selected() called
   ├─ Clear previous peak data
   ├─ Load .D file via DataHandler
   ├─ Interpolate to 10,000 points
   ├─ Auto-detect detector channel (if multiple)
   ├─ Load TIC if MS data available
   ├─ Enable MS tab if MS data present
   └─ Call process_and_display()
4. PlotFrame displays:
   ├─ TIC chromatogram (if MS data)
   └─ Raw signal chromatogram
5. Status bar shows: "Loaded: filename.D (1/25) - with MS data"
6. Export button enabled
```

### Phase 2: Signal Processing
```
1. User adjusts parameters in ParametersFrame:
   ├─ Smoothing method/lambda
   ├─ Baseline method/lambda
   ├─ Peak detection settings
   └─ etc.
2. Each change triggers parameters_changed signal
3. on_parameters_changed() called
   ├─ Calls process_and_display()
   ├─ Re-applies all processing steps:
      ├─ Smoothing (Whittaker/SavGol/Median)
      ├─ Baseline correction (ARPLS/ASLS/etc.)
      ├─ Deconvolution (if enabled)
      └─ Peak detection preview
   └─ Updates PlotFrame with processed signal
4. User sees real-time signal changes in plot
5. Peaks overlay shown if peak detection enabled
```

### Phase 3: Peak Integration
```
1. User clicks "Integrate" button
2. on_integrate() called
   ├─ Validates peak detection enabled
   ├─ Get MS data if available
   ├─ Call integrate_peaks_no_ui()
   │   ├─ Detect peaks (find_peaks)
   │   ├─ Calculate areas (trapezoidal)
   │   ├─ Apply RT matching from RTTable
   │   ├─ Run quality checks (if enabled)
   │   └─ Return Peak objects
   ├─ plot_frame.shade_integration_areas()
   │   └─ Draw filled polygons under peaks
   ├─ Auto-export via export_manager:
   │   ├─ Save JSON with peak data
   │   └─ Save CSV (if enabled)
   ├─ Enable batch MS search button
   └─ Show "View Integration Results" button
3. Status bar: "Integration complete: N peaks found"
4. User can view results by clicking button
```

### Phase 4: MS Library Search
#### Option A: Single Peak Search
```
1. User right-clicks peak on plot
2. Selects "Search Library"
3. on_ms_search_requested(peak_index) called
   ├─ Extract peak spectrum
   ├─ Search MSToolkit
   ├─ Get top N matches
   └─ Display in MSFrame results tree
4. User can view match spectrum and score
5. Can click match to view comparison
```

#### Option B: Batch Search (all peaks)
```
1. User clicks "MS Search All" button
2. run_batch_ms_search() called
   ├─ Validate peaks integrated
   ├─ Create BatchSearchWorker
   ├─ Show BatchProgressDialog
   ├─ Start worker in QThreadPool
   ├─ For each peak:
   │   ├─ Skip if manually assigned
   │   ├─ Extract spectrum
   │   ├─ Search library
   │   ├─ Emit progress signal
   │   │  └─ Dialog updates with match info
   │   └─ Auto-assign top match
   ├─ After all peaks:
   │   ├─ Auto-export with new assignments
   │   ├─ Update plot annotations
   │   └─ Dialog shows completion summary
   └─ Close dialog
3. Status bar: "Batch search complete: X matches found"
```

### Phase 5: Manual Assignment (optional)
```
1. User right-clicks peak on plot
2. Selects "Edit Assignment"
3. EditAssignmentDialog opens
   ├─ Shows current assignment
   ├─ Search field for new compound
   ├─ Results list (autocomplete)
   ├─ User selects compound
   ├─ [Optional] Check "Apply to other files"
   │   ├─ Set RT tolerance (default 0.05 min)
   │   └─ Set spectral similarity threshold (0.7)
   └─ Click Save
4. Assignment updated
5. If cross-file enabled:
   ├─ apply_assignment_to_files() called
   ├─ Searches all processed files in directory
   ├─ Finds peaks with similar RT ± tolerance
   ├─ Matches spectrum similarity
   ├─ Applies assignment to matches
   └─ Auto-export updated results
6. Peak annotations updated on plot
7. JSON results updated
```

### Phase 6: Retention Time (RT) Table
```
1. User clicks "RT Table" tab
2. RTTableFrame shows current ranges
3. User can:
   ├─ Add range: specify start/end time
   ├─ Delete range: select and delete
   ├─ Load from file: import pre-defined ranges
   └─ Save to file: export for reuse
4. During integration:
   ├─ Peaks matched to RT ranges
   ├─ Peak grouped by assigned range
   └─ Results include RT assignment
```

### Phase 7: Quantitation
```
1. User clicks "Quantitation" tab
2. Check "Enable Quantitation"
3. Enter internal standard:
   ├─ Type compound name
   ├─ Click "Search MS Library"
   ├─ System auto-fills: Formula, MW
   ├─ User enters: Density, Volume added (µL)
   └─ Display: mol C of IS
4. Enter sample info:
   ├─ Sample mass (mg)
   ├─ Sample volume (mL)
   └─ Dilution factor
5. Click "Quantitate"
   ├─ _perform_quantitation() called
   ├─ Calculate mol C for each peak
   ├─ Calculate total organic carbon
   ├─ Calculate compound-specific quantities
   └─ Display results
6. Results persist in JSON export
```

### Phase 8: Navigation
```
1. User clicks "Back" or "Next" buttons
2. on_previous_sample() / on_next_sample() called
   ├─ DataHandler.navigate_to_previous/next()
   ├─ Get path to adjacent .D file
   └─ Call on_file_selected(next_path)
3. Workflow repeats from Phase 1
4. Status bar shows: "Loaded: filename2.D (2/25)"
```

### Phase 9: Batch Automation
```
1. User clicks "Batch Process All Files" button
2. start_automation() called
   ├─ Create AutomationWorker for parent directory
   ├─ Show AutomationDialog
   ├─ Start worker in QThreadPool
3. Worker processes each .D file:
   ├─ Load (same as Phase 1)
   ├─ Integrate (same as Phase 3)
   ├─ [If enabled] MS search (same as Phase 4B)
   ├─ [If enabled] Quantitate
   ├─ Export results
   └─ Move to next file
4. Dialog shows real-time progress:
   ├─ Current file name and step
   ├─ Per-file progress bar
   ├─ Overall progress (X/Total)
   ├─ Timestamped log
5. User can:
   ├─ Watch progress
   ├─ Review log messages
   └─ Cancel at any time
6. On completion:
   ├─ Dialog shows summary
   ├─ All results exported (JSON + CSV)
   ├─ Status bar shows completion time
   └─ User can review results
```

### Phase 10: Batch Queue (multiple directories)
```
1. User selects Process → Batch Process Directories...
2. BatchJobDialog opens
3. User:
   ├─ Adds directories via file browser
   ├─ Sets processing options:
   │   ├─ ☑ Perform peak integration
   │   ├─ ☑ Perform MS search
   │   ├─ ☑ Save results to JSON
   │   ├─ ☑ Export to CSV
   │   └─ ☑ Overwrite existing
   └─ Clicks "Start Processing"
4. BatchProgressDialog opens
5. Worker processes directories sequentially:
   ├─ First directory:
   │   ├─ Automation runs (all .D files)
   │   ├─ Status: queued → processing → completed/failed
   │   ├─ Progress bar updated
   │   └─ Details shown in tree
   ├─ Log messages timestamped
   ├─ User can:
   │   ├─ Right-click directory
   │   ├─ Skip (if queued)
   │   ├─ Retry (if failed)
   │   └─ Remove from queue
   └─ Continue to next directory
6. On completion:
   ├─ Summary dialog
   ├─ Show successes/failures/skipped
   └─ All results exported
```

### Phase 11: Export & Data Access
```
Throughout workflow, automatic exports occur:

After Integration (if enabled):
├─ Create JSON: metadata + peak list + areas
└─ Create CSV: RESULTS.CSV format

After MS Search (if enabled):
├─ Update JSON with match_results
└─ Update CSV with compound names

After Manual Assignment (if enabled):
└─ Update JSON with assignment

User can also:
├─ Click "Export" button (manual export)
├─ Settings → Export Settings... (configure)
├─ Tools → JSON ↔ Excel Converter...
│   ├─ Convert JSON to XLSX
│   ├─ Convert XLSX to JSON
│   └─ Batch convert
└─ File browser: results saved in .D directory
```

### Phase 12: Settings & Persistence
```
User can customize:

Settings → Select Detector Channel...
├─ Choose detector if multiple available
└─ Reload file with new channel

Settings → Export Settings...
├─ Toggle export triggers
├─ Choose export formats
└─ Set filename patterns

Settings → Scaling Factors...
├─ Adjust signal/area multipliers
├─ Save/load presets
└─ Match different instrument versions

Settings → Configure Visible Parameters...
├─ Show/hide parameter sections
└─ Customize UI for workflow

Settings → Toggle Dark/Light Mode
├─ Switch theme
└─ All plots update automatically

All settings persisted via QSettings (Windows Registry / macOS plist / Linux ~/.config)
```

---

## 6. KEY DATA STRUCTURES

### Peak Object
```python
class Peak:
    peak_number: int  # 1-indexed
    retention_time: float  # minutes
    start_time: float
    end_time: float
    height: float  # signal intensity
    area: float  # integrated area
    width: float  # peak width
    compound_id: str  # assigned compound name
    match_results: list  # [(compound, score), ...]
    manual_assignment: bool  # User-assigned?
    spectrum: np.ndarray  # MS spectrum data (mz, intensity)
    rt_assignment: str  # Which RT range group
```

### Integration Results
```python
{
    'peaks': [Peak, ...],
    'timestamp': datetime,
    'detector': str,
    'parameters_used': dict,
    'quality_flags': dict,
    'total_area': float,
    'total_height': float
}
```

### Export Data (JSON)
```json
{
    "metadata": {
        "file": "Sample1.D",
        "timestamp": "2024-01-15T10:30:00",
        "detector": "FID",
        "parameters": {...}
    },
    "peaks": [
        {
            "peak_number": 1,
            "retention_time": 2.345,
            "area": 12345.67,
            "height": 1234.5,
            "compound_id": "Decane",
            "match_results": [
                {"compound": "Decane", "score": 0.95},
                {"compound": "Nonane", "score": 0.82}
            ]
        }
    ]
}
```

---

## 7. NOTABLE IMPLEMENTATION DETAILS

### Thread Safety
- AutomationWorker and BatchSearchWorker use QRunnable + QThreadPool
- Signals emitted from workers are queued (thread-safe)
- Main thread updates UI via signal connections
- Uses `Qt.ConnectionType.QueuedConnection` for worker signals

### Data Interpolation
- Raw chromatograms interpolated to 10,000 points at load time
- Ensures consistent processing pipeline
- Original data preserved for export metadata

### Signal Processing Isolation
- Parameters apply live during `process_and_display()`
- Real-time feedback as user adjusts sliders
- No blocking operations in main thread

### Cross-File Assignment
- Dot product similarity matching
- RT tolerance window matching
- Preserves manual assignments during batch MS search

### Export Strategy
- JSON: Comprehensive, all metadata
- CSV: Compatibility with legacy GCMS software
- Auto-triggered by ExportManager based on settings
- Manual export available via buttons

___BEGIN___COMMAND_DONE_MARKER___0
