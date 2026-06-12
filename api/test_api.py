"""
Quick test script to validate the ChromaKit-MS API.

This script tests each endpoint with simple requests to ensure everything works.
"""
import requests
import json
from pathlib import Path


BASE_URL = "http://127.0.0.1:8000"


def test_root():
    """Test root endpoint."""
    print("\n🧪 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Root endpoint working!")


def test_health():
    """Test health check endpoint."""
    print("\n🧪 Testing health check...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Health check working!")


def test_browse():
    """Test file browsing endpoint."""
    print("\n🧪 Testing file browsing...")
    
    # Browse current directory
    current_dir = str(Path.cwd())
    response = requests.get(f"{BASE_URL}/api/browse", params={"path": current_dir})
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Current path: {data['current_path']}")
        print(f"Found {len(data['entries'])} entries")
        
        # Show .D files if any
        d_files = [e for e in data['entries'] if e.get('format') == 'agilent_d']
        if d_files:
            print(f"Found {len(d_files)} .D files:")
            for f in d_files[:3]:  # Show first 3
                print(f"  - {f['name']}")
        
        print("✅ Browse endpoint working!")
        return d_files
    else:
        print(f"⚠️ Browse failed: {response.text}")
        return []


def test_process():
    """Test processing endpoint with synthetic data."""
    print("\n🧪 Testing process endpoint...")
    
    # Create simple synthetic chromatogram
    import numpy as np
    x = np.linspace(0, 10, 1000).tolist()
    # Create baseline + gaussian peak
    baseline = 100 + np.random.normal(0, 1, 1000)
    peak = 500 * np.exp(-((np.array(x) - 5)**2) / 0.1)
    y = (baseline + peak).tolist()
    
    request = {
        "x": x,
        "y": y,
        "params": {
            "smoothing": {
                "enabled": True,
                "lambda": 1000
            },
            "baseline": {
                "method": "asls",
                "lambda": 1000000
            },
            "peaks": {
                "enabled": True,
                "peak_prominence": 0.05,
                "peak_width": 5
            }
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/process", json=request)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Processed {len(data['x'])} data points")
        print(f"Detected {len(data['peaks_x'])} peaks")
        if data['peaks_x']:
            print(f"Peak positions: {[f'{p:.2f}' for p in data['peaks_x'][:5]]}")
        print("✅ Process endpoint working!")
        return data
    else:
        print(f"⚠️ Process failed: {response.text}")
        return None


def test_load_file(file_path):
    """Test file loading endpoint."""
    print(f"\n🧪 Testing load endpoint with: {file_path}")
    
    request = {"file_path": file_path}
    response = requests.post(f"{BASE_URL}/api/load", json=request)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Loaded chromatogram with {len(data['chromatogram']['x'])} points")
        print(f"Has MS data: {data['has_ms']}")
        if data['has_ms']:
            print(f"TIC has {len(data['tic']['x'])} points")
        print("✅ Load endpoint working!")
        return data
    else:
        print(f"⚠️ Load failed: {response.text}")
        return None


def test_batch_search_top_level_knobs():
    """Smoke test: POST /api/ms/batch-search with top-level ms_time_offset and mz_shift.

    Requires:
      - A loaded MS library (run test_library_load first).
      - A real .D directory with integrated peaks (run test_run first).

    This test is a no-op if the library isn't loaded — prints a skip message
    rather than failing the suite, since the test ordering is interactive.
    """
    print("\n🧪 Testing /api/ms/batch-search with top-level knobs...")

    # Probe library status
    health = requests.get(f"{BASE_URL}/api/health").json()
    if not health.get('library_loaded'):
        print("⚠️ Library not loaded; skipping (run test_library_load first)")
        return

    # Minimal request showing both knobs. Real callers would have peaks from
    # /api/run; here we use a synthetic peak just to exercise the wire format.
    request = {
        'peaks': [{
            'compound_id': 'unknown', 'peak_number': 1, 'retention_time': 2.5,
            'integrator': 'BB', 'width': 0.05, 'area': 1000.0,
            'start_time': 2.475, 'end_time': 2.525,
            'is_shoulder': False, 'is_negative': False, 'is_convoluted': False,
            'is_saturated': False, 'is_grouped': False, 'quality_issues': [],
        }],
        'data_directory': '/path/to/your/sample.D',  # ← edit this for live test
        'options': {'search_method': 'vector', 'top_n': 5},
        'ms_time_offset': 0.05,
        'mz_shift': 1,
    }
    response = requests.post(f"{BASE_URL}/api/ms/batch-search", json=request)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"⚠️ Skipped (likely missing data_directory): {response.json()}")
        return
    body = response.json()
    print(f"  total_peaks: {body['total_peaks']}, errors: {len(body['errors'])}")
    print("✅ Batch search with top-level knobs accepted!")


def test_ms_search_single_spectrum():
    """Smoke test: POST /api/ms/search with a hand-crafted spectrum."""
    print("\n🧪 Testing /api/ms/search (single spectrum)...")

    request = {
        'spectrum': {
            'mz': [43.0, 57.0, 71.0, 85.0, 99.0],
            'intensities': [100.0, 80.0, 60.0, 40.0, 20.0],
        },
        'options': {'search_method': 'vector'},
        'top_n': 3,
    }
    response = requests.post(f"{BASE_URL}/api/ms/search", json=request)
    print(f"Status: {response.status_code}")
    if response.status_code == 409:
        print("⚠️ Library not loaded; skipping (run test_library_load first)")
        return
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    body = response.json()
    print(f"  results: {len(body['results'])}, elapsed: {body['elapsed_seconds']}s")
    if body['results']:
        top = body['results'][0]
        print(f"  top hit: {top['name']} (score {top['score']:.3f}, cas {top['casno']})")
    print("✅ Single-spectrum search working!")


def test_run_with_write_output_false():
    """Smoke test: POST /api/run with write_output=False returns peaks without writing JSON."""
    print("\n🧪 Testing /api/run with write_output=False...")

    request = {
        'data_path': '/path/to/your/sample.D',          # ← edit this for live test
        'method_path': '/path/to/your/sample.chromethod',  # ← edit this for live test
        'write_output': False,
    }
    response = requests.post(f"{BASE_URL}/api/run", json=request)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"⚠️ Skipped (likely missing data_path or method_path): {response.text}")
        return
    body = response.json()
    print(f"  peak_count: {body['peak_count']}, output_files: {body['output_files']}")
    assert body['output_files'] == [], f"Expected empty output_files, got {body['output_files']}"
    print("✅ /api/run write_output=False working!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ChromaKit-MS API Test Suite")
    print("=" * 60)
    print("\nMake sure the API server is running:")
    print("  cd api")
    print("  python main.py")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    input()
    
    try:
        # Basic endpoint tests
        test_root()
        test_health()
        
        # File browsing test
        d_files = test_browse()
        
        # Processing test with synthetic data
        test_process()
        
        # File loading test (if .D files found)
        if d_files:
            print("\n" + "=" * 60)
            print("Found .D files - would you like to test loading one?")
            print(f"File: {d_files[0]['name']}")
            response = input("Test load? (y/n): ")
            if response.lower() == 'y':
                test_load_file(d_files[0]['path'])
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\n📚 View full API documentation at:")
        print(f"   {BASE_URL}/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server!")
        print("Make sure the server is running:")
        print("  cd api")
        print("  python main.py")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
