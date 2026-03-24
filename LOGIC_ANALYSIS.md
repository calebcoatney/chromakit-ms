# ChromaKit Logic Layer - Comprehensive Analysis

## Overview
The `logic/` directory contains the shared backend processing pipeline used by both the GUI and FastAPI. It handles the complete GC-MS data processing workflow from raw data loading through quantitation and export.

---

## 1. COMPLETE FILE INVENTORY & PURPOSE

| File | Size | Purpose |
|------|------|---------|
| **data_handler.py** | 13K | Loads/navigates Agilent .D directories, manages detector selection |
| **processor.py** | 67K | Main chromatogram processing pipeline (smoothing, baseline, peak detection) |
| **integration.py** | 41K | Peak integration, Peak class, Integrator class |
| **spectrum_extractor.py** | 20K | Extracts mass spectra for peaks using various methods |
| **batch_search.py** | 11K | Batch MS library search worker (QRunnable) |
| **quantitation.py** | 9.4K | Polyarc + Internal Standard quantitation calculations |
| **export_manager.py** | 9.0K | Centralized export trigger/settings management |
| **json_exporter.py** | 18K | JSON export with metadata scraping |
| **deconvolution.py** | 64K | Deep learning peak deconvolution (1D U-Net + EMG fitting) |
| **automation_worker.py** | 42K | Batch file automation worker |
| **chunker.py** | 6.9K | Chromatogram windowing for deconvolution |
| **ms_baseline_worker.py** | 7.7K | Per-ion MS baseline correction worker |
| **json_to_xlsx.py** | 6.0K | Standalone JSON→XLSX converter |
| **interpolation.py** | 1.5K | Array interpolation utilities |
| **utils.py** | 0B | (empty, reserved for utilities) |
| **__init__.py** | 219B | Exports main classes: DataHandler, ChromatogramProcessor, Integrator, Peak |

---

## 2. DATA FLOW PIPELINE

### Complete Processing Sequence

```
Input: Agilent .D directory
  ↓
DataHandler.load_data_directory(file_path)
  ├─→ Uses rainbow API to read .D structure
  ├─→ Auto-detects detector (FID1A, TCD2B, etc.)
  ├─→ Returns: {'chromatogram': {x, y}, 'tic': {x, y}, 'metadata': {...}}
  ↓
ChromatogramProcessor.process(x, y, params, ms_range)
  ├─ STEP 1: Apply smoothing (optional)
  │  ├─ Methods: whittaker, median, savgol
  ├─ STEP 2: Calculate baseline (always)
  │  ├─ Methods: asls, arpls, iarpls, snip, ModPoly
  │  ├─ Returns: baseline_y, baseline_corrected_y
  ├─ STEP 3: Detect peaks & shoulders (derivative-based)
  │  ├─ Uses Savgol filtering for derivative calculation
  │  ├─ Noise-adaptive thresholding
  │  ├─ Shoulder detection with exclusion zones
  ├─ STEP 4: Detect negative peaks (optional)
  ├─ STEP 5: Apply solvent delay filtering (if MS data provided)
  │  └─ Removes peaks outside MS time range
  ↓
Returns: {
  'x': time array,
  'y_original': original signal,
  'y_smoothed': smoothed signal,
  'y_baseline': baseline signal,
  'y_corrected': baseline-corrected signal,
  'peaks_x': peak times,
  'peaks_y': peak intensities,
  'peak_metadata': [{index, x, y, is_shoulder, type, bounds...}],
  'detection_signal': signal used for detection,
  'negative_peaks': {...}
}
  ↓
ChromatogramProcessor.integrate_peaks(processed_data, ...)
  └─ Integrator.integrate(processed_data, ...)
    ├─ Fits Gaussians to peaks (single/multi based on quality)
    ├─ Calculates peak areas using Simpson's rule
    ├─ Computes quality metrics (asymmetry, spectral coherence)
    ├─ Optionally applies peak grouping
    ├─ Quality checks (saturation, convolution, etc.)
    ↓
Returns: List[Peak] objects with:
  - compound_id, peak_number, retention_time
  - area, width, start_time, end_time
  - quality_issues, is_saturated, asymmetry, etc.
  ↓
[Optional Deconvolution]
deconvolution.run_deconvolution_pipeline(...)
  ├─ Chunks chromatogram into windows
  ├─ Applies 1D U-Net to predict apex heatmaps
  ├─ Fits EMG components
  ├─ Outputs DeconvolutionResult with EMG components
  ↓
Spectrum Extraction (for MS matching)
DataHandler.extract_spectrum_for_peak(peak, options)
  └─ SpectrumExtractor.extract_for_peak(data_directory, peak, options)
    ├─ Methods: apex, weighted_average, range_average, midpoint, sum
    ├─ Background subtraction (min_tic or fixed)
    ├─ TIC weighting (optional)
    ├─ Intensity filtering
    ├─ Saturation detection (threshold: 8.0e6)
    ↓
Returns: {mz, intensities, rt, is_saturated, saturation_level}
  ↓
MS Library Search
BatchSearchWorker.run()
  ├─ Calls ms_toolkit.search_vector/w2v/hybrid()
  ├─ Methods: vector similarity, Word2Vec, hybrid
  ├─ Handles manual assignments (skips auto-search)
  ├─ Updates Peak.compound_id, Qual (quality/match score), casno
  ├─ Detects detector saturation
  ↓
[Optional: Quantitation]
QuantitationCalculator
  ├─ extract_carbon_count(formula)
  ├─ calculate_mol_C_internal_standard(volume_uL, density, MW, formula)
  ├─ calculate_response_factor(is_area, mol_C_IS)
  ├─ quantitate_peak(area, RF, formula, MW)
  ├─ calculate_composition(peaks_data) → mol_C%, mol%, wt%
  ├─ calculate_carbon_balance(total_mass, sample_mass)
  ↓
Export
ExportManager.export_results(peaks, d_path, trigger_type)
  ├─ Trigger types: 'integration', 'ms_search', 'assignment', 'batch'
  ├─ JSON export: export_integration_results_to_json()
  │  └─ Includes: metadata, processing params, peaks, quantitation
  ├─ CSV export: app.export_results_csv()
  ↓
Output: integration_results.json, RESULTS.CSV, (optional) RESULTS.XLSX
```

---

## 3. KEY CLASSES & PUBLIC INTERFACES

### **DataHandler** (`data_handler.py`)
Primary interface for loading and navigating Agilent .D directories.

**Constructor:**
```python
dh = DataHandler()
```

**Public Methods:**

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `load_data_directory(file_path, detector=None)` | str, str\|None | dict | Load .D directory and return chromatogram + TIC data |
| `get_available_detectors(data_dir_path=None)` | str\|None | list[str] | Get all detector names (FID1A, TCD2B, etc.) |
| `get_ms_data(data_dir=None)` | obj\|None | obj | Get raw MS data object from rainbow API |
| `get_detector_metadata(detector=None, data_dir=None)` | str\|None, obj\|None | dict | Get metadata for specific detector |
| `extract_spectrum_at_rt(retention_time, aligned_tic_data=None)` | float, arr\|None | dict\|None | Extract spectrum at given RT |
| `extract_spectrum_for_peak(peak, options=None)` | Peak, dict\|None | dict\|None | Extract spectrum for a peak (uses SpectrumExtractor) |
| `navigate_to_next()` | - | str\|None | Navigate to next .D file in directory |
| `navigate_to_previous()` | - | str\|None | Navigate to previous .D file |
| `get_processed_files()` | - | list[str] | Get all .D directories with integration_results.json |

**State/Properties:**
```python
dh.current_data_dir        # Loaded data directory object
dh.current_directory_path  # Full path to current .D directory
dh.current_detector        # Currently selected detector name
dh.available_directories   # List of .D directories in parent folder
dh.current_index           # Index in available_directories
dh.signal_factor           # Configurable signal scaling factor
dh.has_ms_data             # Boolean: whether MS data is present
```

---

### **ChromatogramProcessor** (`processor.py`)
Main processing pipeline for chromatogram data.

**Constructor:**
```python
processor = ChromatogramProcessor()
```

**Public Methods:**

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `process(x, y, params=None, ms_range=None)` | arr, arr, dict\|None, tuple\|None | dict | Full processing pipeline: smooth → baseline → detect peaks |
| `integrate_peaks(processed_data=None, rt_table=None, chemstation_area_factor=0.0784, ms_data=None, quality_options=None, peak_groups=None)` | dict\|None, df\|dict\|None, float, arr\|None, dict\|None, list\|None | list[Peak] | Integrate detected peaks |
| `align_tic_to_fid(fid_time, fid_signal, tic_time, tic_signal, max_lag_seconds=2.0, num_points=10000, verbose=True)` | arr, arr, arr, arr, float, int, bool | tuple | Align TIC to FID using cross-correlation |

**State/Properties:**
```python
processor.default_params   # Default processing parameters
processor.baseline_fitter  # pybaselines.Baseline() instance
```

**Processing Parameters Structure:**
```python
params = {
    'smoothing': {
        'enabled': bool,
        'method': 'whittaker'|'median'|'savgol',
        'lambda': float,           # Whittaker lambda
        'diff_order': int,
        'median_kernel': int,
        'savgol_window': int,
        'savgol_polyorder': int
    },
    'baseline': {
        'show_corrected': bool,
        'method': 'asls'|'arpls'|'iarpls'|'snip'|'ModPoly',
        'lambda': float,           # Baseline lambda
        'asymmetry': float,        # For ALS methods
        'break_points': list       # For ALS methods
    },
    'peaks': {
        'enabled': bool,
        'window_length': int,
        'polyorder': int,
        'peak_prominence': float,
        'peak_width': float,
        'shoulder_height_factor': float,
        'apex_shoulder_distance': float
    },
    'shoulders': {
        'enabled': bool,
        'window_length': int,
        'polyorder': int,
        'sensitivity': float,      # 1-10 scale
        'apex_distance': float
    },
    'negative_peaks': {
        'enabled': bool,
        'min_prominence': float,
        'min_width': float
    }
}
```

---

### **Peak** (`integration.py`)
Data class representing an integrated chromatographic peak.

**Constructor:**
```python
peak = Peak(
    compound_id='Unknown',
    peak_number=1,
    retention_time=5.234,
    integrator='Gaussian',
    width=0.045,
    area=12345.6,
    start_time=5.2,
    end_time=5.29,
    start_index=None,
    end_index=None
)
```

**Properties:**
```python
peak.as_row    # [compound_id, peak_number, rt, integrator, width, area, start_time, end_time]
peak.as_dict   # Complete dictionary representation
peak.apex_time # Alias for retention_time
```

**Attributes:**
```python
# Core integration data
peak.compound_id        # Assigned compound name
peak.peak_number        # Sequential peak number
peak.retention_time     # RT in minutes
peak.area               # Integrated area
peak.width              # Peak width in minutes
peak.start_time, peak.end_time  # Integration bounds
peak.start_index, peak.end_index # Array indices

# MS Search Results
peak.compound_id        # From batch search
peak.Compound_ID        # Notebook-style field name
peak.compound_name      # From search
peak.casno              # CAS number (formatted)
peak.Qual              # Match quality score (0-1)
peak.manual_assignment  # Boolean: user overrode auto assignment

# Quality Assessment
peak.asymmetry         # Peak symmetry metric
peak.spectral_coherence # Mass spectrum quality
peak.quality_issues    # List of detected issues
peak.is_convoluted     # Boolean: multi-component peak
peak.is_negative       # Boolean: inverted peak
peak.is_shoulder       # Boolean: detected as shoulder
peak.is_grouped        # Boolean: part of grouped peak
peak.grouped_peak_count # Number of peaks in group

# Saturation Detection
peak.is_saturated      # Boolean: detector overflow
peak.saturation_level  # Max intensity value

# Quantitation (Polyarc + IS)
peak.mol_C             # Moles of carbon
peak.mol_C_percent     # Mole percent of carbon
peak.num_carbons       # Number of carbons in molecule
peak.mol               # Moles of compound
peak.mass_mg           # Mass in milligrams
peak.mol_percent       # Mole percentage
peak.wt_percent        # Weight percentage
```

---

### **Integrator** (`integration.py`)
Static methods for integration calculations.

**Static Methods:**

| Method | Purpose |
|--------|---------|
| `identify_compound(retention_time, rt_table=None)` | Look up compound by RT in table |
| `integrate(processed_data, rt_table=None, chemstation_area_factor=0.0784, verbose=True, ms_data=None, quality_options=None, peak_groups=None)` | Main integration method (returns list[Peak]) |
| `_calculate_derivatives(x, y, window_length=41, polyorder=3)` | Calculate dy, d2y using Savgol |
| `_find_second_derivative_bounds(x, y, apex_idx, dy=None, d2y=None, ...)` | Find integration bounds from d2y |
| `_apply_peak_grouping(peaks_list, peak_groups, x, y, ...)` | Merge peaks within specified windows |

---

### **SpectrumExtractor** (`spectrum_extractor.py`)
Handles all mass spectrum extraction logic.

**Constructor:**
```python
extractor = SpectrumExtractor(debug=False)
extractor.saturation_threshold = 8.0e6  # Configurable
```

**Public Methods:**

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `extract_at_rt(data_directory, retention_time, intensity_threshold=0.01)` | str, float, float | dict\|None | Extract spectrum at specific RT |
| `extract_for_peak(data_directory, peak, options=None)` | str, Peak, dict\|None | dict\|None | Extract spectrum for a peak using options |

**Options Dictionary:**
```python
options = {
    'extraction_method': 'apex'|'weighted_average'|'range_average'|'midpoint'|'sum',
    'subtract_background': bool,
    'subtraction_method': 'min_tic'|'fixed',
    'subtract_weight': float,         # Weight for background subtraction
    'tic_weight': bool,               # Use TIC for weighting
    'range_points': int,              # Points for range methods
    'midpoint_width_percent': float,  # Window width percentage
    'intensity_threshold': float,     # Min relative intensity
    'saturation_threshold': float,    # Default 8.0e6
    'debug': bool
}
```

**Return Format:**
```python
{
    'rt': float,
    'mz': np.array,
    'intensities': np.array,
    'is_saturated': bool,
    'saturation_level': float  # Max intensity if saturated
}
```

---

### **BatchSearchWorker** (`batch_search.py`)
QRunnable for threaded MS library searching.

**Constructor:**
```python
worker = BatchSearchWorker(ms_toolkit, peaks, data_directory, options=None)
```

**Public Methods:**
```python
worker.run()              # @Slot - execute in thread
worker.signals.started.emit(count)
worker.signals.progress.emit(index, name, results)
worker.signals.finished.emit()
worker.signals.error.emit(msg)
worker.signals.log_message.emit(msg)
```

**Signals:**
```python
started = Signal(int)                          # Total peaks
progress = Signal(int, str, list)              # Peak index, name, search results
finished = Signal()
error = Signal(str)
log_message = Signal(str)
```

**Options:**
```python
options = {
    'search_method': 'vector'|'w2v'|'hybrid',
    'extraction_method': 'apex',
    'top_n': int,                    # Number of results to return
    'similarity': 'composite'|'...',
    'weighting': 'NIST_GC'|'...',
    'unmatched': 'keep_all'|'...',
    'intensity_power': float,        # For W2V
    'hybrid_method': 'auto'|'...',
    'mz_shift': float,               # M/Z shift correction
    'debug': bool
}
```

---

### **QuantitationCalculator** (`quantitation.py`)
Polyarc + Internal Standard quantitation.

**Constructor:**
```python
calc = QuantitationCalculator()
```

**Public Methods:**

| Method | Parameters | Returns | Purpose |
|--------|-----------|---------|---------|
| `extract_carbon_count(formula)` | str | int\|None | Parse formula string for carbon count |
| `calculate_mol_C_internal_standard(volume_uL, density_g_mL, molecular_weight, formula)` | float, float, float, str | float\|None | Calculate moles of carbon in IS |
| `calculate_response_factor(is_area, mol_C_IS)` | float, float | float\|None | Calculate universal response factor |
| `calculate_sample_mass(volume_uL, density_g_mL)` | float, float\|None | float\|None | Convert volume to mass (mg) |
| `quantitate_peak(peak_area, response_factor, formula, molecular_weight)` | float, float, str, float | dict\|None | Calculate mol_C, mol, mass_mg |
| `calculate_composition(peaks_data)` | list[dict] | list[dict] | Add mol_C%, mol%, wt% to peaks |
| `calculate_carbon_balance(total_mass_mg, sample_mass_mg)` | float, float\|None | float\|None | Carbon recovery % |
| `validate_inputs(volume_uL, density_g_mL, molecular_weight)` | float, float, float | tuple(bool, str) | Validate input parameters |

**Calculation Example:**
```python
calc = QuantitationCalculator()

# Step 1: Get IS info
mol_C_IS = calc.calculate_mol_C_internal_standard(
    volume_uL=1.0,
    density_g_mL=0.8,
    molecular_weight=128.0,
    formula='C8H16'  # 8 carbons
)

# Step 2: Get response factor from IS peak
RF = calc.calculate_response_factor(
    is_area=50000.0,
    mol_C_IS=mol_C_IS
)

# Step 3: Quantitate unknown peaks
results = calc.quantitate_peak(
    peak_area=25000.0,
    response_factor=RF,
    formula='C10H20',
    molecular_weight=140.0
)
# returns: {'mol_C': x, 'num_carbons': 10, 'mol': y, 'mass_mg': z}

# Step 4: Calculate percentages
peaks_list = [results]  # List of all quantitated peaks
peaks_with_pct = calc.calculate_composition(peaks_list)
```

---

### **ExportManager** (`export_manager.py`)
Centralized export trigger management with QSettings integration.

**Constructor:**
```python
export_mgr = ExportManager(app_instance=None)
```

**Public Methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `should_export_json(trigger_type)` | bool | Check if JSON export enabled for trigger |
| `should_export_csv(trigger_type)` | bool | Check if CSV export enabled for trigger |
| `export_results(peaks, d_path, trigger_type, detector=None, is_update=False, quantitation_settings=None, processing_params=None)` | dict | Export in configured formats |
| `export_after_integration(peaks, d_path, detector=None, quantitation_settings=None)` | dict | Convenience method |
| `export_after_ms_search(peaks, d_path, detector=None, quantitation_settings=None)` | dict | Convenience method |
| `export_after_assignment(peaks, d_path, detector=None, quantitation_settings=None)` | dict | Convenience method |
| `export_during_batch(peaks, d_path, detector=None, quantitation_settings=None)` | dict | Convenience method |
| `get_export_summary()` | str | Human-readable summary of settings |

**Trigger Types:**
- `'integration'` - After peak integration
- `'ms_search'` - After MS library search
- `'assignment'` - After manual peak assignment
- `'batch'` - During batch file processing

**Return Format:**
```python
{
    'json': bool,           # Success status
    'csv': bool,
    'messages': list[str]   # Status messages
}
```

**QSettings Keys:**
```python
# Master enable/disable
export/json_enabled      # Default: True
export/csv_enabled       # Default: True

# Per-trigger settings
export/after_integration # Default JSON: True, CSV: False
export/after_ms_search   # Default JSON: True, CSV: True
export/after_assignment  # Default JSON: True, CSV: False
export/during_batch      # Default JSON: True, CSV: True
```

---

### **JSON Export Functions** (`json_exporter.py`)

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `scrape_metadata_from_d_directory(d_path, detector="FID1A")` | Extract metadata from .D directory using rainbow API |
| `export_integration_results_to_json(peaks, d_path, detector, quantitation_settings=None, processing_params=None)` | Export integration results |
| `update_json_with_ms_search_results(peaks, d_path, detector, quantitation_settings=None, processing_params=None)` | Update JSON with MS search results |

**JSON Structure:**
```python
{
    "sample_id": str,
    "timestamp": str,
    "method": str,
    "detector": str,
    "signal": str,
    "notebook": str,
    "processing": {
        "smoothing": {...},
        "baseline": {...},
        "peak_detection": {...},
        "deconvolution": {...}
    },
    "peaks": [
        {
            "compound_id": str,
            "peak_number": int,
            "retention_time": float,
            "integrator": str,
            "width": float,
            "area": float,
            "start_time": float,
            "end_time": float,
            "is_saturated": bool,
            "saturation_level": float,
            "is_shoulder": bool,
            "Compound ID": str,
            "Qual": float,
            "casno": str,
            "mol_C": float,
            "num_carbons": int,
            "mol": float,
            "mass_mg": float,
            "mol_percent": float,
            "wt_percent": float
        }
    ],
    "quantitation": {
        "method": "polyarc_is",
        "internal_standard": {...},
        "response_factor": float,
        "sample_mass_mg": float,
        "carbon_balance": float
    }
}
```

---

### **Deconvolution Pipeline** (`deconvolution.py`)
Deep learning peak deconvolution (requires PyTorch).

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `GCHeatmapUNet` | 1D U-Net model for apex heatmap prediction |
| `DetectionResult` | Result from heatmap detection phase |
| `EMGComponent` | Single Exponentially Modified Gaussian component |
| `DeconvComponent` | Fitted component with EMG parameters |
| `DeconvolutionResult` | Complete deconvolution output |

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `is_available()` | Check if PyTorch is installed |
| `load_model(weights_path=None)` | Load pre-trained U-Net model |
| `detect_merged_peaks(time, signal, peaks, ...)` | Detect coeluting peaks using U-Net |
| `deconvolve_peak(detection_result, time, signal)` | Fit EMG components to detected apices |
| `run_deconvolution_pipeline(time, corrected_signal, ...)` | Full pipeline: chunking → detection → deconvolution |
| `integrate_emg_components(components, x, corrected_y, baseline_y, area_factor)` | Integrate EMG components |
| `integrate_deconv_components(components, x, corrected_y, baseline_y, area_factor)` | Integrate deconvoloved components |

**Deconvolution Parameters:**
```python
params = {
    'heatmap_threshold': float,           # 0.15 default
    'pre_fit_signal_threshold': float,    # Signal threshold for pre-fit
    'min_area_frac': float,               # Minimum area fraction
    'valley_threshold_frac': float,       # Valley detection threshold
    'splitting_method': 'geometric'|'equal',
    'windows': [                          # Manual windows
        {'start': float, 'end': float}
    ]
}
```

---

## 4. CONFIGURABLE PARAMETERS & SETTINGS

### Processing Parameters (UI Settings)

**Smoothing Options:**
- `method`: whittaker, median, savgol
- `lambda`: Whittaker regularization (1e-1 default)
- `median_kernel`: 3-15 (default 5)
- `savgol_window`: 3-101, odd (default 3)
- `savgol_polyorder`: 1-5 (default 1)

**Baseline Correction:**
- `method`: asls, arpls, iarpls, snip, ModPoly
- `lambda`: 1e3 - 1e8 (default 1e6)
- `asymmetry`: 0.001 - 0.1 (default 0.01)

**Peak Detection:**
- `min_prominence`: 0.01 - 0.5 (fraction or absolute)
- `min_width`: 3 - 100 points
- `shoulder_detection`: enabled/disabled
- `shoulder_sensitivity`: 1-10 scale

**MS Search:**
- `search_method`: vector, w2v, hybrid
- `top_n`: 1-20
- `similarity`: composite or individual
- `weighting`: NIST_GC, NIST_MS, user-defined
- `mz_shift`: -5 to +5 (PPM correction)

**Spectrum Extraction:**
- `extraction_method`: apex, weighted_average, range_average, midpoint, sum
- `subtract_background`: True/False
- `subtraction_method`: min_tic, fixed
- `intensity_threshold`: 0.01 - 0.5

**Quantitation (Polyarc + IS):**
- `internal_standard`: name, formula, volume, density, MW
- `response_factor`: calculated from IS
- `sample_density`: optional for sample mass
- `sample_volume`: optional

**Export Triggers:**
```python
export/after_integration   # Integration results only
export/after_ms_search     # After MS library search
export/after_assignment    # After manual reassignment
export/during_batch        # During batch processing
```

---

## 5. INPUT/OUTPUT DATA FORMATS

### Input Formats

**1. Agilent .D Directory Structure:**
```
Sample_Name.D/
├── acqmethod.txt          # Method file
├── instrument.xml         # Instrument config
├── AcqData/
│   ├── MSData.uv          # Raw MS data
│   ├── FID1A.ch           # FID detector signal
│   ├── TCD2B.ch           # TCD detector signal (if present)
│   └── ...
└── InjLog.txt             # Injection log
```

**2. Chromatogram Data (NumPy Arrays):**
```python
x: np.ndarray shape (n,)     # Time (minutes)
y: np.ndarray shape (n,)     # Signal intensity
```

**3. Peak List (for batch search):**
```python
peaks: list[Peak]
# Each Peak object has:
#   - retention_time, area, width
#   - start_time, end_time
#   - start_index, end_index
```

### Output Formats

**1. JSON Export** (`integration_results.json`):
```json
{
  "sample_id": "Sample1",
  "timestamp": "2024-03-18 14:30:45",
  "method": "Method_Name",
  "detector": "FID1A",
  "processing": {
    "baseline": {"algorithm": "asls", "lam": 1000000},
    "peak_detection": {"min_prominence": 0.05, "min_width": 5}
  },
  "peaks": [
    {
      "peak_number": 1,
      "retention_time": 5.234,
      "area": 12345.6,
      "width": 0.045,
      "Compound ID": "Hexane",
      "Qual": 0.95,
      "casno": "110-54-3",
      "mol_C": 0.00456,
      "mass_mg": 0.654
    }
  ]
}
```

**2. CSV Export** (`RESULTS.CSV`):
```
Peak #,Ret Time,Compound ID,Qual,CAS No,Area,Width,Start Time,End Time,...
1,5.234,Hexane,0.95,110-54-3,12345.6,0.045,5.2,5.29,...
```

**3. XLSX Export** (via `json_to_xlsx.py`):
- Workbook with multiple sheets
- Header info per sample
- Peaks table with calculated values

**4. Deconvolution Output:**
```python
DeconvolutionResult {
    detection_result: DetectionResult,
    components: list[DeconvComponent],  # Fitted EMG components
    time: np.ndarray,
    signal_baseline_corrected: np.ndarray,
    n_components: int,
    areas: list[float],
    x_emg: np.ndarray,
    y_reconstructed: np.ndarray
}
```

**5. Spectrum Data (for MS Search):**
```python
{
    'mz': np.array([73, 74, 75, ...]),
    'intensities': np.array([100.0, 50.2, 25.1, ...]),
    'rt': 5.234,
    'is_saturated': False,
    'saturation_level': None
}
```

---

## 6. KEY PUBLIC API SUMMARY FOR FASTAPI WRAPPING

### Essential Entry Points for API

**1. Data Loading:**
```python
from logic import DataHandler
dh = DataHandler()
data = dh.load_data_directory('/path/to/Sample.D')
# Returns: {chromatogram, tic, metadata}
```

**2. Processing:**
```python
from logic import ChromatogramProcessor
processor = ChromatogramProcessor()
processed = processor.process(x, y, params)
peaks = processor.integrate_peaks(processed, ...)
```

**3. Peak Data:**
```python
from logic import Peak, Integrator
peak = Peak(...)
peaks_list = Integrator.integrate(processed_data, ...)
```

**4. MS Search (Threaded):**
```python
from logic.batch_search import BatchSearchWorker
worker = BatchSearchWorker(ms_toolkit, peaks, data_dir, options)
# Connect signals and run in QThreadPool
```

**5. Quantitation:**
```python
from logic import QuantitationCalculator
calc = QuantitationCalculator()
mol_C = calc.calculate_mol_C_internal_standard(...)
RF = calc.calculate_response_factor(...)
quantitated = calc.quantitate_peak(...)
```

**6. Spectrum Extraction:**
```python
from logic import SpectrumExtractor
extractor = SpectrumExtractor()
spectrum = extractor.extract_for_peak(data_dir, peak, options)
```

**7. Export:**
```python
from logic import ExportManager
export_mgr = ExportManager(app)
result = export_mgr.export_results(peaks, d_path, trigger_type='integration')
```

---

## 7. ARCHITECTURE NOTES

### Design Patterns Used
1. **Data Classes:** `Peak` for immutable-style data containers
2. **Worker Pattern:** `BatchSearchWorker`, `MSBaselineCorrectionWorker`, `AutomationWorker` for threading
3. **Static Methods:** `Integrator` provides functional interface
4. **Manager Pattern:** `ExportManager` centralizes export decisions
5. **Dependency Injection:** `options` dicts allow parameterization

### Dependencies
- **numpy**: Numerical arrays
- **scipy**: Signal processing, optimization, statistics
- **pybaselines**: Baseline correction algorithms
- **rainbow**: Agilent .D file reading
- **PySide6** (GUI only): Qt signals/slots
- **torch** (optional): Deep learning deconvolution
- **openpyxl**: Excel export

### Thread Safety
- Workers are QRunnables (thread-safe signal/slot mechanism)
- No shared state between workers (immutable options passed)
- `cancelled` flag for graceful interruption

### Extensibility Points
1. Add new baseline methods to `pybaselines` config
2. Add new peak fitting models to `_fit_*` methods
3. Add new MS search methods to `BatchSearchWorker.run()`
4. Add new export formats to `ExportManager`
5. Add new extraction methods to `SpectrumExtractor`

---

## 8. CRITICAL GOTCHAS FOR API IMPLEMENTATION

⚠️ **Qt Dependency:** `DataHandler`, `BatchSearchWorker`, and workers use `PySide6.QtCore`. For FastAPI:
- Consider creating thin wrappers that avoid Qt dependencies
- Or run workers in a Qt event loop (complex)
- Recommended: Create parallel non-Qt implementations

⚠️ **Signal/Slot Mechanism:** These are Qt-specific. FastAPI alternatives:
- Use callbacks instead of signals
- Use async/await patterns
- Use message queues (RabbitMQ, Redis)

⚠️ **Rainbow API:** Requires Agilent file reading. Check if this works on server:
- May have licensing restrictions
- May require native libraries on Linux/Docker
- Test thoroughly in production environment

⚠️ **Manual Assignment Flag:** `peak.manual_assignment` must be checked in batch search to avoid overwriting user choices

⚠️ **MS Saturation Detection:** Hardcoded threshold `8.0e6` in `SpectrumExtractor`. May need to be configurable per detector.

⚠️ **Peak Grouping:** `Integrator._apply_peak_grouping()` merges peaks. Must preserve in export to avoid data loss.

