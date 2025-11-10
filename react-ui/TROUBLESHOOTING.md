# ChromaKit-MS Frontend/Backend Restart Guide

## The errors you're seeing are due to cached code. Follow these steps:

### 1. Stop Both Servers
- Stop the API backend (Ctrl+C in the terminal running it)
- Stop the Vite frontend (Ctrl+C in the terminal running it)

### 2. Clear Browser Cache
In your browser:
- Press `Ctrl+Shift+Delete`
- Select "Cached images and files"
- Click "Clear data"

OR simply:
- Press `Ctrl+F5` for a hard refresh (clears cache for current page)

### 3. Restart Backend
```bash
cd "c:\Users\ccoatney\OneDrive - NREL\GC-MS code development\chromakit-qt"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Restart Frontend
```bash
cd "c:\Users\ccoatney\OneDrive - NREL\GC-MS code development\chromakit-qt\frontend"
npm run dev
```

### 5. Hard Refresh Browser
- Navigate to `http://localhost:3000`
- Press `Ctrl+Shift+R` (or `Ctrl+F5`) to force reload without cache

## What Was Fixed

### Backend (`logic/integration.py`):
- Added `start_index` and `end_index` to Peak class
- These are needed for shading integration areas

### Frontend (`ChromatogramPlot.jsx`):
- Removed `plotRef` reference (switched to `onRelayout` callback)
- Uses `layoutRef` with `onRelayout` event to preserve zoom

### Frontend (`ProcessingControls.jsx`):
- Removed `handleAlignTicToggle` (no longer needed)
- Removed `align_tic` from state

## Verification Steps

After restarting:

1. ✅ Load a .D file - should work without 500 error
2. ✅ Adjust parameters - plot should stay zoomed (not reset)
3. ✅ Enable peak detection - peaks should appear
4. ✅ Click "Integrate Peaks" - should work without 500 error
5. ✅ Check plot - should show green shaded areas between baseline and signal

## Still Having Issues?

If you still see errors after hard refresh:

### Check Browser Console
Open DevTools (F12) → Console tab
Look for specific error messages

### Check Backend Terminal
Look for Python errors or stack traces

### Nuclear Option: Clear Everything
```bash
# In frontend directory
rm -rf node_modules/.vite
npm run dev
```

This clears Vite's cache completely.
