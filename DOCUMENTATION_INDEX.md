# ChromaKit-Qt Documentation Index

## 📚 Available Documentation

### 1. **QUICK_REFERENCE.md** (8 KB) - START HERE
**Best for:** Quick lookup of classes, methods, and parameters
- Core class definitions with examples
- Processing parameters reference
- Data flow diagram
- Input/output format specifications
- Default thresholds

### 2. **LOGIC_ANALYSIS.md** (29 KB) - COMPREHENSIVE BACKEND ANALYSIS
**Best for:** Understanding the complete processing pipeline and API design
- Complete file inventory with purposes
- Detailed data flow pipeline (visual)
- Full public API documentation for all key classes
- Complete method signatures and parameters
- Configuration options and settings
- Input/output format specifications
- Architecture patterns and design decisions
- Critical gotchas for API implementation

### 3. **COMPREHENSIVE_UI_ANALYSIS.md** (56 KB)
**Best for:** Understanding the GUI implementation
- UI component hierarchy
- Signal/slot connections
- Parameter frame structure
- Main widget interactions

### 4. **README.md** (2.8 KB)
**Best for:** Project overview and setup

---

## 🎯 Logic Layer Components

### Entry Points
1. **DataHandler** (`logic/data_handler.py`)
   - Load Agilent .D directories
   - Auto-detect detectors
   - Extract spectra

2. **ChromatogramProcessor** (`logic/processor.py`)
   - Smoothing
   - Baseline correction
   - Peak detection
   - Integration

### Processing Pipeline
3. **Peak** + **Integrator** (`logic/integration.py`)
   - Peak class definition
   - Integration calculations

4. **SpectrumExtractor** (`logic/spectrum_extractor.py`)
   - Multiple extraction methods
   - Background subtraction
   - Saturation detection

### Advanced Features
5. **BatchSearchWorker** (`logic/batch_search.py`)
   - MS library searching
   - Threaded execution (QRunnable)

6. **QuantitationCalculator** (`logic/quantitation.py`)
   - Polyarc + Internal Standard method
   - Response factor calculations
   - Composition calculations

7. **ExportManager** (`logic/export_manager.py`)
   - Centralized export triggers
   - JSON/CSV export coordination

8. **Deconvolution** (`logic/deconvolution.py`)
   - 1D U-Net for peak detection
   - EMG fitting for coeluting peaks

---

## 📊 Key Data Structures

### Input
- **Agilent .D directories** (via `rainbow` API)
- **Chromatogram data**: NumPy arrays (time, intensity)
- **Peak lists**: `Peak` objects

### Output
- **JSON**: Complete analysis results with metadata
- **CSV**: Peak table export
- **Peak objects**: With assignments, quality metrics, quantitation

---

## ⚙️ Configuration Reference

### Processing Parameters
```python
{
    'smoothing': {'enabled', 'method', 'lambda', ...},
    'baseline': {'method', 'lambda', 'asymmetry', ...},
    'peaks': {'enabled', 'min_prominence', 'min_width', ...},
    'shoulders': {'enabled', 'sensitivity', 'apex_distance', ...},
    'negative_peaks': {'enabled', 'min_prominence', 'min_width', ...}
}
```

### Export Triggers
- `integration` - After peak integration
- `ms_search` - After MS library search
- `assignment` - After manual peak assignment
- `batch` - During batch file processing

### Spectrum Extraction Options
- Methods: apex, weighted_average, range_average, midpoint, sum
- Background subtraction (min_tic or fixed)
- TIC weighting
- Intensity thresholding
- Saturation detection (threshold: 8.0e6)

---

## 🚀 For FastAPI Implementation

### Critical Considerations
1. **Qt Dependency**: Some workers use `PySide6.QtCore`
   - May need non-Qt wrappers for server-side use
   - Consider asyncio patterns instead of Qt signals

2. **Rainbow API**: Requires Agilent file access
   - Check compatibility on Linux/Docker
   - May have licensing implications

3. **Processing Overhead**: Large chromatograms
   - Consider parallel processing for batch operations
   - Implement progress tracking/streaming

4. **File System**: Direct .D directory access required
   - Ensure proper path handling
   - Consider mounted storage or file upload mechanism

### Public API Surface to Wrap
```python
# Core processing
DataHandler.load_data_directory()
ChromatogramProcessor.process()
ChromatogramProcessor.integrate_peaks()

# MS search
BatchSearchWorker (convert to async pattern)

# Spectrum extraction
SpectrumExtractor.extract_for_peak()

# Quantitation
QuantitationCalculator.quantitate_peak()

# Export
ExportManager.export_results()
```

---

## 📋 File Manifest

| File | Size | Key Class/Function |
|------|------|-------------------|
| data_handler.py | 13 KB | `DataHandler` |
| processor.py | 67 KB | `ChromatogramProcessor` |
| integration.py | 41 KB | `Peak`, `Integrator` |
| spectrum_extractor.py | 20 KB | `SpectrumExtractor` |
| batch_search.py | 11 KB | `BatchSearchWorker` |
| quantitation.py | 9.4 KB | `QuantitationCalculator` |
| export_manager.py | 9.0 KB | `ExportManager` |
| json_exporter.py | 18 KB | Export functions |
| deconvolution.py | 64 KB | `GCHeatmapUNet`, pipeline functions |
| automation_worker.py | 42 KB | `AutomationWorker` |
| chunker.py | 6.9 KB | `Chunk`, `chunk_chromatogram()` |
| ms_baseline_worker.py | 7.7 KB | `MSBaselineCorrectionWorker` |
| json_to_xlsx.py | 6.0 KB | `process_json_to_excel()` |
| interpolation.py | 1.5 KB | `interpolate_arrays()` |
| utils.py | 0 KB | (reserved) |
| __init__.py | 219 B | Main exports |

**Total Logic Layer: ~316 KB of code**

---

## 🔗 Dependencies

### Core
- `rainbow` - Agilent .D file reading
- `numpy` - Numerical arrays
- `scipy` - Signal processing, statistics
- `pybaselines` - Baseline correction algorithms

### GUI/Threading
- `PySide6` - Qt framework (workers, signals)

### Optional
- `torch` - Deep learning peak deconvolution
- `openpyxl` - Excel export

---

## 💡 Quick Start Examples

### Load and Process Data
```python
from logic import DataHandler, ChromatogramProcessor

dh = DataHandler()
data = dh.load_data_directory('/path/to/Sample.D')

processor = ChromatogramProcessor()
processed = processor.process(
    data['chromatogram']['x'],
    data['chromatogram']['y'],
    params={'smoothing': {'enabled': True, 'method': 'savgol'}, ...}
)

peaks = processor.integrate_peaks(processed)
```

### Extract Spectrum and Search
```python
from logic import SpectrumExtractor
from logic.batch_search import BatchSearchWorker

extractor = SpectrumExtractor()
spectrum = extractor.extract_for_peak(
    '/path/to/Sample.D',
    peaks[0],
    options={'extraction_method': 'apex', ...}
)

worker = BatchSearchWorker(ms_toolkit, peaks, '/path/to/Sample.D')
# Connect signals and run in thread pool
```

### Quantitate Results
```python
from logic import QuantitationCalculator

calc = QuantitationCalculator()
mol_C_IS = calc.calculate_mol_C_internal_standard(
    volume_uL=1.0, density_g_mL=0.8,
    molecular_weight=128, formula='C8H16'
)

RF = calc.calculate_response_factor(50000, mol_C_IS)

for peak in peaks:
    result = calc.quantitate_peak(peak.area, RF, 'C10H20', 140)
    peak.mol_C = result['mol_C']
    peak.mass_mg = result['mass_mg']
```

### Export Results
```python
from logic import ExportManager

export_mgr = ExportManager()
result = export_mgr.export_results(
    peaks, '/path/to/Sample.D',
    trigger_type='integration',
    detector='FID1A'
)
```

---

## 🐛 Known Issues & Gotchas

1. **Manual Assignments**: Check `peak.manual_assignment` before auto-search
2. **Saturation Threshold**: Hardcoded at 8.0e6, needs parameterization
3. **Peak Grouping**: Merges peaks in specified windows, must preserve in export
4. **MS Data Optional**: System gracefully handles missing MS data
5. **Qt Signals**: Not suitable for async server use, need conversion

---

## 📝 For Maintainers

### Adding New Extraction Methods
→ Extend `SpectrumExtractor._extract_peak_spectrum()` or add new method

### Adding New Baseline Methods
→ Configure in `ChromatogramProcessor._apply_baseline_correction()`

### Adding New Export Formats
→ Create function in `json_exporter.py` and register in `ExportManager`

### Adding New Quantitation Methods
→ Create new class in `quantitation.py` following `QuantitationCalculator` pattern

---

**Last Updated**: 2024-03-18
**Logic Layer Version**: 1.0 (as of this analysis)
**Total Documentation**: ~93 KB across 3 files

