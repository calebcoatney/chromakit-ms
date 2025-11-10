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
                "median_filter": {"kernel_size": 5},
                "savgol_filter": {"window_length": 11, "polyorder": 3}
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
