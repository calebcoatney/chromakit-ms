# ChromaKit-MS React Frontend

Modern React frontend for ChromaKit-MS built with Vite.

## Features

- 📁 **File Browser** - Browse server directories and select .D files
- 📊 **Interactive Charts** - Visualize chromatograms with Plotly (zoom, pan, export)
- ⚙️ **Real-time Processing** - See chromatogram update as you adjust parameters
- 🎯 **Baseline Correction** - Multiple algorithms with visual feedback
- 📈 **Peak Detection** - Automatic peak finding with customizable prominence
- � **Peak Integration** - Shaded areas show integrated regions
- 🎨 **Modern UI** - Clean, responsive design with custom styling

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will start at `http://localhost:3000`

### 3. Make Sure API is Running

The backend API must be running at `http://127.0.0.1:8000`:

```bash
cd ../api
python main.py
```

## Project Structure

```
src/
├── components/          # React components
│   ├── Header.jsx           # App header with status
│   ├── FileBrowser.jsx      # Directory/file browser
│   ├── ProcessingControls.jsx  # Parameter controls
│   ├── ChromatogramPlot.jsx    # Plot visualization
│   └── PeakTable.jsx        # Integration results table
├── services/           # API integration
│   └── api.js              # API client functions
├── styles/             # CSS stylesheets
│   └── App.css             # Global styles
├── App.jsx             # Main application component
└── main.jsx            # React entry point
```

## Component Architecture

### App.jsx
- Main orchestrator component
- Manages global state (file data, processed data, integration results)
- Coordinates communication between components

### FileBrowser
- Displays directory contents
- Allows navigation through directories
- Emits file selection events

### ProcessingControls
- Provides parameter inputs
- Validates parameter ranges
- Triggers processing

### ChromatogramPlot
- Uses Plotly.js for interactive visualization
- Built-in toolbar for zoom, pan, reset, export
- Shows chromatogram, baseline, peak markers, and integration areas
- Real-time updates as parameters change

### PeakTable
- Displays integration results
- Shows peak properties (RT, area, width)
- Calculates percent areas

## API Integration

All API calls are centralized in `src/services/api.js`:

```javascript
import { browseDirectory, loadFile, processChromato, integratePeaks } from './services/api';

// Browse directory
const data = await browseDirectory('/path/to/data');

// Load file
const fileData = await loadFile('/path/to/sample.D');

// Process chromatogram
const processed = await processChromato({
  x: fileData.chromatogram.x,
  y: fileData.chromatogram.y,
  params: { smoothing: {...}, baseline: {...}, peaks: {...} }
});

// Integrate peaks
const results = await integratePeaks({
  processed_data: processed,
  chemstation_area_factor: 0.0784
});
```

## Styling

Custom CSS with CSS variables for theming:

```css
:root {
  --primary-color: #667eea;
  --primary-dark: #764ba2;
  --secondary-color: #48bb78;
  --danger-color: #f56565;
  /* ... */
}
```

Components use utility classes and inline styles for flexibility.

## Development

### Hot Module Replacement
Vite provides instant HMR - changes appear immediately without full page reload.

### Proxy Configuration
API requests are proxied to avoid CORS issues:

```javascript
// vite.config.js
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000'
    }
  }
})
```

### Production Build

```bash
npm run build
```

Output will be in `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Customization

### Adding New Parameters

1. Update `ProcessingControls.jsx` state
2. Add UI controls for the parameter
3. Pass to API in correct format

### Adding New Plots

1. Create new component using Recharts
2. Import in `App.jsx`
3. Pass required data as props

### Styling Changes

Modify `src/styles/App.css` - uses CSS variables for easy theming.

## Troubleshooting

### API Connection Issues
- Ensure backend is running at `http://127.0.0.1:8000`
- Check browser console for CORS errors
- Verify proxy configuration in `vite.config.js`

### Build Errors
- Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Update dependencies: `npm update`

### Chart Not Displaying
- Check that data arrays are properly formatted
- Ensure Recharts is installed: `npm install recharts`
- Check browser console for errors

## Dependencies

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Plotly.js** - Interactive charting library with zoom/pan controls
- **react-plotly.js** - React wrapper for Plotly
- **Axios** - HTTP client

## Next Steps

- [ ] Add authentication
- [ ] Implement file upload
- [ ] Add RT table management
- [ ] Export results to CSV/JSON
- [ ] Dark mode toggle
- [ ] Mobile responsive improvements
