# ChromaKit-MS FastAPI Backend

A REST API backend for ChromaKit-MS that wraps the existing data processing logic for use with web frontends.

## Features

- 📁 **File Browsing**: Browse server directories for Agilent `.D` files
- 📊 **Data Loading**: Load GC-MS chromatogram and TIC data
- ⚙️ **Processing**: Apply smoothing, baseline correction, and peak detection
- 📈 **Integration**: Integrate detected peaks with optional RT table matching

## Quick Start

### 1. Install Dependencies

```bash
pip install -r api/requirements.txt
```

### 2. Run the Server

```bash
cd api
python main.py
```

The server will start at `http://127.0.0.1:8000`

### 3. View Interactive Documentation

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints

### Browse Files
```http
GET /api/browse?path=/path/to/data
```

Returns list of `.D` files and subdirectories.

**Response:**
```json
{
  "current_path": "/path/to/data",
  "parent_path": "/path/to",
  "entries": [
    {
      "name": "sample1.D",
      "path": "/path/to/data/sample1.D",
      "type": "file",
      "format": "agilent_d"
    }
  ]
}
```

### Load File
```http
POST /api/load
Content-Type: application/json

{
  "file_path": "/path/to/data/sample1.D"
}
```

Returns chromatogram and TIC data.

**Response:**
```json
{
  "chromatogram": {
    "x": [0.0, 0.01, ...],
    "y": [100, 105, ...]
  },
  "tic": {
    "x": [0.5, 0.51, ...],
    "y": [1000, 1050, ...]
  },
  "has_ms": true,
  "metadata": {
    "filename": "sample1.D"
  }
}
```

### Process Chromatogram
```http
POST /api/process
Content-Type: application/json

{
  "x": [0.0, 0.01, 0.02, ...],
  "y": [100, 105, 110, ...],
  "params": {
    "smoothing": {
      "enabled": true,
      "median_filter": {"kernel_size": 5},
      "savgol_filter": {"window_length": 11, "polyorder": 3}
    },
    "baseline": {
      "method": "asls",
      "lambda": 1000000
    },
    "peaks": {
      "enabled": true,
      "peak_prominence": 0.05,
      "peak_width": 5
    }
  },
  "ms_range": [0.5, 10.0]
}
```

Returns processed data with detected peaks.

### Integrate Peaks
```http
POST /api/integrate
Content-Type: application/json

{
  "processed_data": {
    "x": [...],
    "corrected_y": [...],
    "peaks_x": [...],
    "peaks_y": [...],
    "peak_metadata": [...]
  },
  "chemstation_area_factor": 0.0784
}
```

Returns integrated peak areas and metadata.

## Architecture

```
api/
├── main.py           # FastAPI application and endpoints
├── models.py         # Pydantic models for request/response validation
├── utils.py          # Utility functions (JSON encoding, etc.)
└── requirements.txt  # Python dependencies

Uses existing logic/ modules:
├── logic/
│   ├── data_handler.py    # File I/O (via rainbow-api)
│   ├── processor.py       # Chromatogram processing
│   └── integration.py     # Peak integration
```

## Development

### Run with Auto-Reload

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Run in Production

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY logic/ ./logic/
COPY api/ ./api/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## CORS Configuration

By default, CORS is enabled for all origins during development. For production:

```python
# In api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Notes

- **Zero modifications** to existing `logic/` modules - they work as-is!
- Uses existing `rainbow-api` for reading Agilent `.D` files
- All numpy arrays automatically serialized to JSON-compatible lists
- Interactive API documentation generated automatically by FastAPI

## Future Enhancements

- [ ] Session management for multi-step workflows
- [ ] WebSocket support for real-time progress updates
- [ ] Batch processing endpoints
- [ ] Result caching with Redis
- [ ] MS library search integration
- [ ] RT table management endpoints
