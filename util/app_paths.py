"""Where ChromaKit writes, in a source checkout and in the frozen Windows build.

A source checkout keeps library/, cache/ and overrides/ beside the code, as it always has. The
frozen build's code folder may be read-only (Program Files) and is replaced on every update, so
those folders go to the user's data folder instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ChromaKit"


def writable_dir(name: str) -> str:
    """A folder ChromaKit may write to, created if missing."""
    path = (user_data_dir() if is_frozen() else _REPO) / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def configure_frozen_runtime() -> Path:
    """For the windowed build: pin matplotlib's cache (else it rebuilds fonts every launch) and send
    stdout/stderr, where ChromaKit prints its errors, to a log file. Returns the log path."""
    data = user_data_dir()
    mpl = data / "mpl"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    logs = data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "chromakit.log"
    stream = open(log, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    return log
