# ChromaKit-MS Web Frontend - User Guide

## Overview

The ChromaKit-MS web frontend provides an interactive interface for GC-MS data analysis with real-time parameter updates and interactive plotting.

## Workflow

### 1. Load Data
- Use the **File Browser** on the left to navigate directories
- Click on a `.D` file to load it
- The chromatogram will appear in the main plot area

### 2. Adjust Processing Parameters (Real-time)
As you change parameters, the chromatogram updates automatically:

#### Signal Smoothing
- **Enable Smoothing**: Turn on/off smoothing
- **Median Filter Size**: Controls noise reduction (3-31, odd values)
- **Savitzky-Golay Window**: Smoothing window size (5-51, odd values)
- **Polynomial Order**: Fit order for Savitzky-Golay (1-5)

#### Baseline Correction
- **Show Corrected Signal**: Toggle between raw+baseline or corrected signal
- **Algorithm**: Choose baseline method
  - `arpls` - Default, works well for most cases
  - `asls` - Classic asymmetric least squares
  - `airpls` - Adaptive iterative reweighting
  - `imodpoly`, `modpoly`, `snip` - Alternative polynomial methods
- **Lambda (λ)**: Smoothness parameter for asls/airpls/arpls (10^2 to 10^12)
- **Align TIC with FID**: Corrects time delays between detectors

#### Peak Detection
- **Enable Peak Detection**: Turn on/off automatic peak finding
- **Min Prominence**: Minimum peak height above baseline
  - Accepts numbers: `100000` or scientific notation: `1e5`
  - Higher values = fewer, more significant peaks
  - Lower values = more peaks detected

### 3. Integrate Peaks
Once peaks are detected:
- Click the **Integrate Peaks** button
- The plot will show **shaded areas** between the baseline and chromatogram
- These shaded regions represent the integrated peak areas

### 4. View Results
The **Peak Integration Results** table shows:
- Peak number
- Retention time (RT)
- Integrated area
- Percent area (relative to total)
- Peak width
- Compound ID (if matched)

### 5. Interactive Plot Controls

The Plotly toolbar provides:
- 🔍 **Zoom**: Click and drag to zoom into a region
- 🖐️ **Pan**: Move the plot around
- 🏠 **Reset**: Return to original view
- 📷 **Download**: Export plot as PNG
- 🔄 **Autoscale**: Fit data to view

## Tips

### Finding the Right Prominence
- Start with `1e5` for typical GC-FID data
- If too many peaks: **increase** (e.g., `5e5`, `1e6`)
- If missing peaks: **decrease** (e.g., `5e4`, `1e4`)
- Watch the plot update in real-time as you type!

### Baseline Methods
- **arpls**: Best general-purpose method
- **asls**: Good for symmetric baselines
- **snip**: Fast, works well for noisy data
- Adjust **Lambda** higher for smoother baselines

### Smoothing
- Use for **noisy** data
- Median filter removes spikes
- Savitzky-Golay preserves peak shapes
- Can combine both filters

## Keyboard Shortcuts (in plot)

When hovering over the plot:
- **Double-click**: Reset zoom
- **Shift + drag**: Pan
- **Scroll**: Zoom in/out (if enabled)

## Example Workflow

1. Load `sample001.D`
2. Enable Peak Detection
3. Set prominence to `1e5`
4. Adjust baseline method to `arpls`
5. Watch chromatogram update showing baseline and peaks
6. Click **Integrate Peaks**
7. View shaded integration areas
8. Export results table or download plot image

## Troubleshooting

### Plot not updating?
- Check API connection status in header
- Make sure backend is running at `http://127.0.0.1:8000`

### Too many/few peaks?
- Adjust **Min Prominence** parameter
- Try different baseline methods
- Enable smoothing to reduce noise

### Integration not working?
- Ensure peaks are detected first (enable Peak Detection)
- Check that prominence is appropriate for your data
- Verify baseline looks reasonable

## Advanced Features

### Scientific Notation
Min Prominence accepts:
- Regular numbers: `100000`
- Scientific: `1e5`, `1.5e6`, `5e-3`
- Very useful for different magnitude ranges

### Export Options
- **Plot Image**: Use Plotly toolbar download button
- **Data**: Copy from results table or export via API

## API Integration

All processing happens server-side via the FastAPI backend:
- File browsing: `GET /api/browse`
- Load file: `POST /api/load`
- Process (real-time): `POST /api/process`
- Integrate: `POST /api/integrate`
