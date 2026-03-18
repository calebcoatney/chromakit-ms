"""
Entry point for the ChromaKit-MS web application.

Serves the FastAPI backend and the built React frontend on a single port.
Usage:  chromakit-web          (default port 8000)
        chromakit-web --port 9000
        chromakit-web --no-browser
"""
import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="ChromaKit-MS Web Application")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    parser.add_argument("--dev", action="store_true", help="Run API only (use with separate Vite dev server)")
    args = parser.parse_args()

    # Check that the React build exists (unless --dev mode)
    dist_dir = Path(__file__).parent / "react-ui" / "dist"
    if not args.dev and not dist_dir.exists():
        print("⚠️  React build not found at react-ui/dist/")
        print("   Run:  cd react-ui && npm run build")
        print("   Or use --dev to run API-only mode alongside Vite dev server.")
        sys.exit(1)

    # Mount the static frontend on the FastAPI app (unless --dev)
    if not args.dev:
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        from api.main import app

        # Serve static assets (JS, CSS, images) — must be mounted before catch-all
        app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="static-assets")

        # Override the root route to serve the SPA
        for i, route in enumerate(app.routes):
            if hasattr(route, 'path') and route.path == "/":
                app.routes.pop(i)
                break

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(dist_dir / "index.html")

        # Catch-all for SPA client-side routing (must be last)
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            file_path = dist_dir / full_path
            if file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(dist_dir / "index.html")

    # Auto-open browser after a short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    print(f"🧪 ChromaKit-MS Web — http://{args.host}:{args.port}")
    if args.dev:
        print("   Dev mode: API only. Start Vite with: cd react-ui && npm run dev")

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.dev)


if __name__ == "__main__":
    main()
