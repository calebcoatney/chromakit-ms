"""
Example client demonstrating how to use the ChromaKit-MS API.

This shows a complete workflow:
1. Browse for .D files
2. Load a file
3. Process the chromatogram
4. Integrate peaks
"""
import requests
import json
from typing import Dict, List, Any


class ChromaKitClient:
    """Simple Python client for ChromaKit-MS API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def browse(self, path: str = ".") -> Dict[str, Any]:
        """Browse directory for .D files."""
        response = requests.get(f"{self.base_url}/api/browse", params={"path": path})
        response.raise_for_status()
        return response.json()
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """Load a .D file."""
        response = requests.post(
            f"{self.base_url}/api/load",
            json={"file_path": file_path}
        )
        response.raise_for_status()
        return response.json()
    
    def process(self, x: List[float], y: List[float], 
                params: Dict[str, Any] = None,
                ms_range: List[float] = None) -> Dict[str, Any]:
        """Process chromatogram data."""
        request = {"x": x, "y": y}
        
        if params:
            request["params"] = params
        if ms_range:
            request["ms_range"] = ms_range
            
        response = requests.post(f"{self.base_url}/api/process", json=request)
        response.raise_for_status()
        return response.json()
    
    def integrate(self, processed_data: Dict[str, Any],
                  rt_table: Dict[str, Any] = None,
                  chemstation_area_factor: float = 0.0784) -> Dict[str, Any]:
        """Integrate detected peaks."""
        request = {
            "processed_data": processed_data,
            "chemstation_area_factor": chemstation_area_factor
        }
        
        if rt_table:
            request["rt_table"] = rt_table
            
        response = requests.post(f"{self.base_url}/api/integrate", json=request)
        response.raise_for_status()
        return response.json()


def example_workflow():
    """Demonstrate a complete analysis workflow."""
    
    print("=" * 70)
    print("ChromaKit-MS API - Example Workflow")
    print("=" * 70)
    
    # Initialize client
    client = ChromaKitClient()
    
    # Step 1: Browse for files
    print("\n📁 Step 1: Browsing for .D files...")
    browse_result = client.browse(".")
    print(f"Current directory: {browse_result['current_path']}")
    
    d_files = [e for e in browse_result['entries'] if e.get('format') == 'agilent_d']
    
    if not d_files:
        print("No .D files found in current directory.")
        print("Creating synthetic data for demonstration instead...")
        use_synthetic = True
    else:
        print(f"Found {len(d_files)} .D file(s):")
        for i, f in enumerate(d_files[:5], 1):
            print(f"  {i}. {f['name']}")
        
        response = input("\nUse first .D file for demo? (y/n): ")
        use_synthetic = response.lower() != 'y'
    
    # Step 2: Load or create data
    if use_synthetic:
        print("\n📊 Step 2: Creating synthetic chromatogram...")
        import numpy as np
        
        x = np.linspace(0, 10, 2000)
        baseline = 1000 + 50 * np.sin(x * 2) + np.random.normal(0, 10, len(x))
        
        # Add multiple peaks
        peak1 = 800 * np.exp(-((x - 3.5)**2) / 0.05)
        peak2 = 1200 * np.exp(-((x - 5.2)**2) / 0.08)
        peak3 = 600 * np.exp(-((x - 7.8)**2) / 0.04)
        
        y = baseline + peak1 + peak2 + peak3
        
        data = {
            "chromatogram": {"x": x.tolist(), "y": y.tolist()},
            "tic": {"x": [], "y": []},
            "has_ms": False
        }
        print(f"Created {len(x)} data points with 3 synthetic peaks")
        
    else:
        print(f"\n📊 Step 2: Loading {d_files[0]['name']}...")
        data = client.load_file(d_files[0]['path'])
        print(f"Loaded {len(data['chromatogram']['x'])} chromatogram points")
        if data['has_ms']:
            print(f"MS data available: {len(data['tic']['x'])} TIC points")
    
    # Step 3: Process the chromatogram
    print("\n⚙️ Step 3: Processing chromatogram...")
    
    processing_params = {
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
    
    ms_range = None
    if data['has_ms'] and len(data['tic']['x']) > 0:
        ms_range = [min(data['tic']['x']), max(data['tic']['x'])]
        print(f"Using MS range: {ms_range[0]:.2f} - {ms_range[1]:.2f} min")
    
    processed = client.process(
        x=data['chromatogram']['x'],
        y=data['chromatogram']['y'],
        params=processing_params,
        ms_range=ms_range
    )
    
    print(f"Processing complete:")
    print(f"  - Data points: {len(processed['x'])}")
    print(f"  - Peaks detected: {len(processed['peaks_x'])}")
    
    if processed['peaks_x']:
        print(f"  - Peak retention times: ", end="")
        print(", ".join([f"{rt:.2f} min" for rt in processed['peaks_x'][:5]]))
        if len(processed['peaks_x']) > 5:
            print(f"    ... and {len(processed['peaks_x']) - 5} more")
    
    # Step 4: Integrate peaks
    if len(processed['peaks_x']) > 0:
        print("\n📈 Step 4: Integrating peaks...")
        
        integration_result = client.integrate(
            processed_data=processed,
            chemstation_area_factor=0.0784
        )
        
        print(f"Integration complete:")
        print(f"  - Total peaks integrated: {integration_result['total_peaks']}")
        print(f"  - Total area: {sum(integration_result['integrated_areas']):.2f}")
        
        print("\nPeak Results:")
        print(f"{'Peak':<6} {'RT (min)':<10} {'Area':<15} {'Width (min)':<12}")
        print("-" * 50)
        
        for i, peak in enumerate(integration_result['peaks'][:10], 1):
            rt = peak.get('retention_time', 0)
            area = peak.get('area', 0)
            width = peak.get('width', 0)
            print(f"{i:<6} {rt:<10.3f} {area:<15.2f} {width:<12.4f}")
        
        if len(integration_result['peaks']) > 10:
            print(f"... and {len(integration_result['peaks']) - 10} more peaks")
    else:
        print("\n⚠️ No peaks detected - skipping integration")
    
    print("\n" + "=" * 70)
    print("✅ Workflow complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("  - View interactive API docs: http://127.0.0.1:8000/docs")
    print("  - Build a web frontend using this API")
    print("  - Integrate with your existing workflows")


if __name__ == "__main__":
    try:
        example_workflow()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server!")
        print("\nMake sure the server is running:")
        print("  cd api")
        print("  python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
