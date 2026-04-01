#!/usr/bin/env python3
"""
watch_and_run.py — Reference file watcher for ChromaKit automated pipelines.

Monitors a directory for new Agilent .D directories. When one appears,
calls POST /api/run on a running ChromaKit API server using a specified
.chromethod file. Results (JSON) are written by the API server alongside
the data file.

Usage:
    python tools/watch_and_run.py \\
        --watch-dir /path/to/gc_data \\
        --method    /path/to/my_method.chromethod \\
        --api-url   http://127.0.0.1:8000

Dependencies (not in setup.py — install manually):
    pip install watchdog requests

This script is a starting point. Adapt it for your own automation loop
(e.g., forward results to a Bayesian optimizer, log to a database, etc.).
"""
import argparse
import sys
import time
from pathlib import Path

try:
    import requests
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    sys.exit(
        "Missing dependencies. Install with:\n"
        "    pip install watchdog requests\n"
    )


class _DDirectoryHandler(FileSystemEventHandler):
    """Fires POST /api/run when a new .D directory is created."""

    def __init__(self, method_path: str, api_url: str):
        self.method_path = method_path
        self.api_url = api_url.rstrip("/")
        self._seen: set = set()

    def on_created(self, event):
        if not event.is_directory:
            return
        path = event.src_path
        if not path.endswith(".D"):
            return
        if path in self._seen:
            return
        self._seen.add(path)
        print(f"[watcher] New .D detected: {path}")
        self._run(path)

    def _run(self, data_path: str):
        payload = {"data_path": data_path, "method_path": self.method_path}
        try:
            resp = requests.post(
                f"{self.api_url}/api/run",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            peak_count = result.get("peak_count", "?")
            output_files = result.get("output_files", [])
            print(
                f"[watcher] {Path(data_path).name}: "
                f"{peak_count} peaks -> {output_files}"
            )
        except requests.exceptions.Timeout:
            print(f"[watcher] ERROR: API timed out processing {data_path}")
        except requests.exceptions.ConnectionError:
            print(f"[watcher] ERROR: Could not connect to API at {self.api_url}")
        except Exception as e:
            print(f"[watcher] ERROR processing {data_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Watch a directory for new .D files and process them via ChromaKit API."
    )
    parser.add_argument(
        "--watch-dir", required=True, help="Directory to watch for new .D files"
    )
    parser.add_argument(
        "--method", required=True, help="Path to .chromethod file"
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="ChromaKit API base URL (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    watch_dir = Path(args.watch_dir)
    if not watch_dir.is_dir():
        sys.exit(f"ERROR: watch-dir does not exist: {watch_dir}")
    if not Path(args.method).is_file():
        sys.exit(f"ERROR: method file does not exist: {args.method}")

    # Verify API is reachable
    try:
        resp = requests.get(f"{args.api_url.rstrip('/')}/api/health", timeout=5)
        resp.raise_for_status()
        print(f"[watcher] API healthy at {args.api_url}")
    except Exception as e:
        sys.exit(f"ERROR: Cannot reach ChromaKit API at {args.api_url}: {e}")

    handler = _DDirectoryHandler(
        method_path=str(Path(args.method).resolve()),
        api_url=args.api_url,
    )
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    print(f"[watcher] Watching {watch_dir} for new .D directories. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
