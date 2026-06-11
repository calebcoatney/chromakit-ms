"""Module-level MSToolkit singleton for the API process.

Lifecycle:
  - At import: _toolkit = None.
  - On POST /api/ms/library/load: load_library() builds (or replaces) the
    toolkit and vectorizes the library.
  - On other endpoints: get_toolkit() returns the singleton or raises
    RuntimeError if not yet loaded.

Concurrency note: FastAPI serializes synchronous endpoints per worker, so no
explicit lock is needed in Phase 1. If Phase 2 introduces background jobs or
multiple workers, wrap the module-level mutations in a threading.RLock.

Loading logic is distilled from ui/frames/ms.py::LibraryLoadThread.run().
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

try:
    from ms_toolkit.api import MSToolkit
    from ms_toolkit.models import Compound
    _HAVE_MS_TOOLKIT = True
except ImportError:
    MSToolkit = None  # type: ignore
    Compound = None  # type: ignore
    _HAVE_MS_TOOLKIT = False


_toolkit: Optional["MSToolkit"] = None
_loaded_paths: Optional[dict] = None


def is_loaded() -> bool:
    """Return True if the singleton has been loaded at least once."""
    return _toolkit is not None


def loaded_paths() -> Optional[dict]:
    """Return the paths used for the most recent load (or None if unloaded)."""
    return _loaded_paths


def get_toolkit() -> "MSToolkit":
    """Return the singleton MSToolkit. Raises RuntimeError if not loaded."""
    if _toolkit is None:
        raise RuntimeError(
            "MS library not loaded. POST /api/ms/library/load first."
        )
    return _toolkit


def load_library(
    library_path: str,
    cache_path: Optional[str],
    preselector_path: Optional[str],
    w2v_path: Optional[str],
) -> dict:
    """Load (or reload) the MS library and models. Synchronous.

    Args:
        library_path: Path to a library file (.txt or .json). Required.
        cache_path: Optional JSON cache path. If provided and exists, loads
            directly from JSON (faster, matches LibraryLoadThread behavior).
        preselector_path: Optional path to a .pkl preselector model.
        w2v_path: Optional path to a Word2Vec .model file.

    Returns:
        Status summary dict: {status, compound_count, library_path,
        preselector_loaded, w2v_loaded, elapsed_seconds}.

    Raises:
        RuntimeError: ms-toolkit-nrel is not installed.
        FileNotFoundError: Neither library_path nor cache_path resolves
            to a usable JSON file.
    """
    global _toolkit, _loaded_paths

    if not _HAVE_MS_TOOLKIT:
        raise RuntimeError(
            "ms-toolkit-nrel is not installed. "
            "pip install ms-toolkit-nrel to enable MS endpoints."
        )

    start_ts = time.time()

    # Determine JSON source (cache_path if it exists, else library_path if .json)
    json_path: Optional[str] = None
    if cache_path and os.path.exists(cache_path):
        json_path = cache_path
    elif library_path.lower().endswith('.json') and os.path.exists(library_path):
        json_path = library_path

    if not json_path:
        raise FileNotFoundError(
            f"No usable JSON file found at library_path={library_path!r} "
            f"or cache_path={cache_path!r}. "
            "Phase 1 requires a pre-built JSON cache."
        )

    # Build toolkit and load library directly from JSON
    toolkit = MSToolkit()
    with open(json_path, 'r') as f:
        compounds_json = json.load(f)
    library = {}
    for name, data in compounds_json.items():
        library[name] = Compound.from_json(data)
    toolkit.library = library

    # Vectorize the library
    toolkit.vectorize_library()

    # Optional preselector
    preselector_loaded = False
    if preselector_path and os.path.exists(preselector_path):
        toolkit.load_preselector(preselector_path)
        preselector_loaded = True

    # Optional Word2Vec model
    w2v_loaded = False
    if w2v_path and os.path.exists(w2v_path):
        toolkit.load_w2v(w2v_path)
        w2v_loaded = True

    # Commit to module-level state
    _toolkit = toolkit
    _loaded_paths = {
        'library_path': library_path,
        'cache_path': cache_path,
        'preselector_path': preselector_path,
        'w2v_path': w2v_path,
    }

    return {
        'status': 'loaded',
        'compound_count': len(library),
        'library_path': library_path,
        'preselector_loaded': preselector_loaded,
        'w2v_loaded': w2v_loaded,
        'elapsed_seconds': round(time.time() - start_ts, 3),
    }
