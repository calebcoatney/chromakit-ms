# ChromaKit Desktop GUI - Quick Reference

## Application Layout
```
╔═══════════════════════════════════════════════════════════════╗
║                     Menu: Process | Settings | Tools         ║
╠════════╦═════════════════════════════════════════╦═══════════╣
║        ║                                         ║           ║
║ File   ║         Plot Frame                      ║  Right    ║
║ Tree   ║  ┌─────────────────────────────────┐   ║  Tabs:    ║
║        ║  │      TIC (if MS data)           │   ║           ║
║ · .D   ║  ├─────────────────────────────────┤   ║ ┌────────┐║
║ · .D   ║  │    Chromatogram (main plot)     │   ║ │Param   ││
║ · .D   ║  │  (Right-click peaks for menu)   │   ║ │MS      ││
║        ║  │                                 │   ║ │RT Tbl  ││
║        ║  └─────────────────────────────────┘   ║ │Quant   ││
║        ║                                         ║ └────────┘║
║        ║  ┌─────────────────────────────────┐   ║           ║
║        ║  │  Button Frame                   │   ║           ║
║        ║  │  [Export] [Back] [Next] [Integ] │   ║           ║
║        ║  │  [Batch Process] [MS Search All]│   ║           ║
║        ║  └─────────────────────────────────┘   ║           ║
║        ║                                         ║           ║
╠════════╩═════════════════════════════════════════╩═══════════╣
║ Status Bar: "Loaded: Sample1.D (1/25) - with MS data"       ║
╚═════════════════════════════════════════════════════════════╝
```

## Main Workflows (12 Total)

### 1. Load File
```
Click "Open Folder" → Browse → Select directory → Set as root
Double-click .D file → on_file_selected()
├─ Load chromatogram data
├─ Interpolate to 10,000 points
├─ Load TIC (if MS data)
└─ Display on plot
```

### 2. Adjust Parameters (Real-time)
```
Adjust any parameter slider/spinbox in Parameters Tab
└─ parameters_changed signal
    └─ process_and_display()
        ├─ Apply smoothing
        ├─ Apply baseline
        ├─ Apply deconvolution
        └─ Update plot (live preview)
```

### 3. Integrate Peaks
```
Click "Integrate" button
└─ Validate: peak detection enabled?
    └─ integrate_peaks_no_ui()
        ├─ Detect peaks (find_peaks)
        ├─ Calculate areas
        ├─ Apply RT matching
        ├─ Quality checks
        └─ Return Peak objects
            ├─ Shade areas on plot
            ├─ Auto-export JSON/CSV
            ├─ Enable batch search button
            └─ Show "View Results" button
```

### 4. Search Single Peak
```
Right-click peak on plot → "Search Library"
└─ on_ms_search_requested(peak_index)
    ├─ Extract peak spectrum
    ├─ Search MSToolkit
    └─ Show results in MS Tab (top 5 matches)
```

### 5. Batch Search (All Peaks)
```
Click "MS Search All" button
└─ run_batch_ms_search()
    ├─ Create BatchSearchWorker
    ├─ Show BatchProgressDialog
    └─ For each peak:
        ├─ Skip if manually assigned
        ├─ Extract spectrum
        ├─ Search library
        ├─ Auto-assign top match
        └─ Update progress display
            └─ Auto-export updated results
```

### 6. Manual Assignment
```
Right-click peak → "Edit Assignment"
└─ EditAssignmentDialog
    ├─ Type compound name (min 5 chars)
    ├─ Select from autocomplete results
    ├─ [Optional] Check "Apply to other files"
    │   ├─ Set RT tolerance (±min)
    │   └─ Set spectral similarity threshold
    └─ Click Save
        ├─ Update peak assignment
        ├─ [If cross-file] apply_assignment_to_files()
        └─ Auto-export updated results
```

### 7. RT Table
```
Click "RT Table" tab
├─ Add Range: Enter start/end time → Save
├─ Delete Range: Select → Delete
├─ Load from File: Browse → Load
├─ Save to File: Browse → Save
└─ During integration: Peaks grouped by assigned ranges
```

### 8. Quantitation
```
Click "Quantitation" tab
├─ Check "Enable Quantitation"
├─ Enter Internal Standard:
│   ├─ Compound name
│   ├─ [Optional] Click "Search MS Library" for auto-fill
│   └─ Enter: Density, Volume (µL)
├─ Enter Sample Info: Mass (mg), Volume (mL), Dilution
├─ Click "Quantitate"
└─ View Results:
    ├─ mol C of IS
    ├─ Total organic carbon
    └─ Per-compound quantities
```

### 9. Batch Automation (Single Directory)
```
Click "Batch Process All Files" button
└─ start_automation()
    ├─ Create AutomationWorker
    ├─ Show AutomationDialog
    ├─ For each .D file:
    │   ├─ Load (10%)
    │   ├─ Integrate (40%)
    │   ├─ [If enabled] MS Search
    │   ├─ [If enabled] Quantitate
    │   └─ Export results
    ├─ Update dialog:
    │   ├─ Per-file progress bar
    │   ├─ Overall progress (X/Total)
    │   └─ Timestamped log
    └─ Completion: Summary dialog
```

### 10. Batch Queue (Multiple Directories)
```
Menu → Process → "Batch Process Directories..."
├─ BatchJobDialog:
│   ├─ Add directories (multi-select)
│   ├─ Set options: integration, search, export, overwrite
│   └─ Click "Start Processing"
└─ BatchProgressDialog:
    ├─ Queue tree (directory list with status colors)
    ├─ Per-directory progress bars
    ├─ Overall progress
    ├─ Timestamped log
    ├─ Can right-click: Skip, Retry, Remove
    ├─ Can add more directories while running
    └─ Completion: Summary dialog
```

### 11. Navigation
```
Click "Back" or "Next" buttons
└─ Navigate to previous/next .D file
    └─ on_file_selected(next_path)
        └─ Repeat from Workflow 1
```

### 12. Export & Settings
```
Menu → Settings:
├─ "Select Detector Channel..." → Choose detector
├─ "Export Settings..." → Configure auto-export triggers
├─ "Scaling Factors..." → Adjust signal/area multipliers + presets
├─ "Configure Visible Parameters..." → Show/hide parameter sections
└─ "Toggle Dark/Light Mode" → Switch theme

Manual Export: Click "Export" button
File Conversion: Menu → Tools → "JSON ↔ Excel Converter"
```

---

## All UI Components

### Dialogs
| Dialog | Purpose |
|--------|---------|
| DetectorSelectionDialog | Choose detector channel |
| ExportSettingsDialog | Configure auto-export |
| ParameterVisibilityDialog | Show/hide parameters |
| ScalingFactorsDialog | Adjust factors + presets |
| EditAssignmentDialog | Assign compound to peak |
| BatchJobDialog | Setup batch queue |
| AutomationDialog | Show automation progress |
| BatchProgressDialog | Manage batch queue |
| MSOptionsDialog | Advanced MS settings |
| AddToRTTableDialog | Add peak to RT table |
| RTAssignmentDialog | Assign peak to RT range |

### Right-Side Tabs
| Tab | Lines | Purpose |
|-----|-------|---------|
| Parameters | 1591 | Signal processing controls |
| Mass Spectrometry | 1019 | MS search options + results |
| RT Table | 1124 | Define RT ranges |
| Quantitation | 346 | Polyarc + IS calculations |

### Main Area Frames
| Frame | Lines | Purpose |
|-------|-------|---------|
| PlotFrame | 1166 | Chromatogram + TIC display |
| ButtonFrame | 71 | Control buttons |
| FileTreeFrame | 98 | File browser |

### Background Workers
| Worker | Type | Purpose |
|--------|------|---------|
| AutomationWorker | QRunnable | Batch process .D files |
| BatchSearchWorker | QRunnable | MS search on all peaks |
| MSBaselineWorker | QRunnable | TIC baseline correction |

---

## Keyboard & Mouse Interactions

| Action | Result |
|--------|--------|
| Double-click .D file | Load file |
| Left-click peak on plot | Highlight + show info |
| Right-click peak on plot | Context menu: View Spectrum, Search, Assign, etc. |
| Adjust slider/spinbox | Real-time signal update (if enabled) |
| Double-click compound in MS results | View spectrum comparison |

---

## Settings Persistence (QSettings)

All settings saved to system registry/plist/config:

```
scaling/signal_factor
scaling/area_factor
scaling/presets (JSON)

export/after_integration
export/after_ms_search
export/after_assignment
export/during_batch
export/json_enabled
export/csv_enabled
export/json_filename_format
export/csv_filename_format

parameter_visibility/smoothing
parameter_visibility/baseline
parameter_visibility/baseline_advanced
parameter_visibility/peaks
parameter_visibility/negative_peaks
parameter_visibility/shoulders
parameter_visibility/range_filters
parameter_visibility/peak_grouping

theme (light/dark)
```

---

## Menu Structure

```
Process
├── Batch Process Directories...

Settings
├── Select Detector Channel...
├── Export Settings...
├── Scaling Factors...
├── Configure Visible Parameters...
├── Toggle Dark/Light Mode

Tools
└── JSON ↔ Excel Converter...
```

---

## Peak Right-Click Context Menu

```
├── View Spectrum (extract MS spectrum)
├── Search Library (MS library search)
├── Edit Assignment (manual compound assignment)
├── Add to RT Table (define RT range)
└── RT Assignment (assign peak to RT range)
```

---

## Parameter Sections (Scrollable, Toggleable)

1. **Signal Smoothing** - Whittaker, SavGol, Median
2. **Baseline Correction** - ARPLS, ASLS, AirPLS, FastChrom
3. **Advanced Baseline** - TIC alignment, Break points
4. **Peak Detection** - Classical or Deconvolution mode
5. **Negative Peak Detection** - Separate peak finder
6. **Shoulder Detection** - Fine structure peaks
7. **Peak Range Filters** - Exclude peaks in time windows
8. **Peak Grouping** - Group peaks by RT ranges

---

## Data Flow Summary

```
Load .D File
    ↓
[current_x, current_y] = Interpolated (10k points)
[original_x, original_y] = Raw data
    ↓
Adjust Parameters
    ↓
process_and_display()
├─ Smoothing → Baseline → Deconvolution → Update Plot
    ↓
Click "Integrate"
    ↓
integrate_peaks_no_ui()
├─ Detect peaks → Calculate areas → Quality checks
    ↓
[integrated_peaks] = Peak objects list
[integration_results] = Dict with metadata
    ↓
shade_integration_areas()
export_after_integration()
    ↓
Enable "MS Search All" button
    ↓
[User can manually assign peaks or batch search]
    ↓
export_after_ms_search() or export_after_assignment()
    ↓
Results persist in JSON/CSV
```

---

## Critical Features (Must Replicate for Web)

✅ File loading from .D directories
✅ Signal processing with live preview
✅ Peak integration with quality checks
✅ MS library search (single & batch)
✅ Manual peak assignment
✅ RT table for peak grouping
✅ Quantitation (Polyarc + IS)
✅ Batch automation (all files)
✅ Batch MS search
✅ Batch queue (multiple dirs)
✅ Real-time progress with logs
✅ Auto-export (JSON/CSV)
✅ Theme support (dark/light)
✅ Settings persistence
✅ Parameter visibility toggles

