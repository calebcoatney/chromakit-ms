import re
import sys
import types

import chromakit_version
from logic import gcxgc_export as ge


def test_version_is_one_source():
    from pathlib import Path
    assert chromakit_version.__version__ == "0.2.0"
    setup_text = (Path(chromakit_version.__file__).parent / "setup.py").read_text(encoding="utf-8")
    assert "chromakit_version.py" in setup_text and 'version="0.1.0"' not in setup_text
    assert ge._chromakit_version() == "0.2.0"


def test_build_info_commit_wins(monkeypatch):
    monkeypatch.setitem(sys.modules, "_build_info", types.SimpleNamespace(COMMIT="a" * 40))
    assert ge._chromakit_commit() == "a" * 40


def test_without_build_info_falls_back_to_git(monkeypatch):
    monkeypatch.setitem(sys.modules, "_build_info", None)
    assert re.fullmatch(r"[0-9a-f]{40}|", ge._chromakit_commit())
