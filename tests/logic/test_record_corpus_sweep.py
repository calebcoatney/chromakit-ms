# tests/logic/test_record_corpus_sweep.py
"""Alarm for schema drift #10.

Runs against the real result-JSON corpus when it is present and skips cleanly
when it is not, so it is a regression net on Caleb's machine without being a
hard dependency for anyone else.

Point it elsewhere with:  CHROMAKIT_RECORD_CORPUS=/some/path pytest ...
"""
import json
import os
from pathlib import Path

import pytest

from logic.result_record import normalize_peak

DEFAULT_CORPUS = Path.home() / "Library/CloudStorage/OneDrive-NREL"
SEARCH_ROOTS = ("julia", "GC-MS", "GC-MS code development",
                "ChemCatBio - Acid Upgrading")


def _corpus_records():
    base = Path(os.environ.get("CHROMAKIT_RECORD_CORPUS", DEFAULT_CORPUS))
    if not base.is_dir():
        return []
    found = []
    for root in SEARCH_ROOTS:
        start = base / root
        if not start.is_dir():
            continue
        for path in start.rglob("*.json"):
            text = str(path)
            if ".D/" in text or ".C/results/" in text:
                found.append(path)
    return found


CORPUS = _corpus_records()


@pytest.mark.skipif(not CORPUS, reason="real record corpus not present on this machine")
def test_every_corpus_record_normalizes_without_raising():
    checked = 0
    for path in CORPUS:
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue                       # not a ChromaKit record
        if not isinstance(doc, dict):
            continue
        peaks = doc.get("peaks")
        if not isinstance(peaks, list) or not peaks or not isinstance(peaks[0], dict):
            continue
        normalized = [normalize_peak(p) for p in peaks]
        assert any(p.rt_apex is not None for p in normalized), path
        checked += 1
    assert checked > 0, "corpus present but no records were recognised"


@pytest.mark.skipif(not CORPUS, reason="real record corpus not present on this machine")
def test_no_corpus_record_loses_every_compound_name():
    """The silent-blanking failure, swept across the whole corpus."""
    for path in CORPUS:
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        peaks = doc.get("peaks")
        if not isinstance(peaks, list) or not peaks or not isinstance(peaks[0], dict):
            continue
        if not any(p.get("compound_id") or p.get("Compound ID") for p in peaks):
            continue
        assert any(normalize_peak(p).compound for p in peaks), path
