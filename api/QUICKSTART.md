# ChromaKit-MS API Quick Start Guide

Get the REST API running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r api/requirements.txt
```

This installs:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- Plus dependencies already in your environment

## Step 2: Start the Server

**From project root (recommended):**
```bash
python api/main.py
```

**Or using uvicorn directly:**
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**Or from api directory:**
```bash
cd api
uvicorn main:app --reload
```

You should see:
```
Starting ChromaKit-MS API server...
API documentation available at: http://127.0.0.1:8000/docs
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Step 3: Test It Works

Open your browser to http://127.0.0.1:8000/docs

You'll see interactive API documentation where you can test all endpoints!


This demonstrates a complete workflow:
1. Browse for .D files
2. Load a file (or use synthetic data)
3. Process the chromatogram
4. Integrate peaks

## API Endpoints Overview

### Browse Files
```bash
curl "http://127.0.0.1:8000/api/browse?path=."
```

### Load a .D File
```bash
curl -X POST "http://127.0.0.1:8000/api/load" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/sample.D"}'
```

### Process Chromatogram
```bash
curl -X POST "http://127.0.0.1:8000/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "x": [0, 0.1, 0.2, ...],
    "y": [100, 105, 110, ...],
    "params": {
      "smoothing": {"enabled": true},
      "baseline": {"method": "asls"},
      "peaks": {"enabled": true}
    }
  }'
```

## Next Steps

1. **Build a web frontend** - Use React, Vue, or any web framework
2. **Integrate with workflows** - Call the API from Python scripts, Jupyter notebooks, etc.
3. **Deploy to production** - Use Docker, cloud services, etc.

## Troubleshooting

### Port already in use?
Change the port in `api/main.py`:
```python
uvicorn.run(app, host="127.0.0.1", port=8001)  # Use 8001 instead
```

### Can't find .D files?
Make sure to use absolute paths or navigate to the correct directory when browsing.

### Import errors?
Make sure you're running from the project root directory so Python can find the `logic/` modules.

## Production Deployment

For production, use multiple workers and bind to all interfaces:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Docker (see `api/README.md` for Dockerfile example).

## Questions?

- Check the full API docs: http://127.0.0.1:8000/docs
- Read the detailed README: [api/README.md](README.md)
- Explore the example client: [api/example_client.py](example_client.py)
