# tests/logic/test_record_corpus_sweep.py
"""Alarm for schema drift #10.

Runs against the real result-JSON corpus when it is present and skips cleanly
when it is not, so it is a regression net on Caleb's machine without being a
hard dependency for anyone else.

Default mode walks SEARCH_ROOTS under DEFAULT_CORPUS (Caleb's OneDrive tree).
Point it at a *different* corpus root entirely with:

    CHROMAKIT_RECORD_CORPUS=/some/path pytest ...

In that mode the given path IS the corpus root and is scanned directly
(recursively) for matching record JSON -- SEARCH_ROOTS is not re-appended.
This is what lets the sweep run against a small fixture tree on the GC box
or in CI without replicating Caleb's exact OneDrive folder names.
"""
import json
import os
from pathlib import Path

import pytest

from logic.result_record import normalize_peak

DEFAULT_CORPUS = Path.home() / "Library/CloudStorage/OneDrive-NREL"
SEARCH_ROOTS = ("julia", "GC-MS", "GC-MS code development",
                "ChemCatBio - Acid Upgrading")

# Below this, treat the sweep as broken rather than "corpus happens to be
# smaller today": the full default corpus is ~453 records, and a regression
# that narrows SEARCH_ROOTS or over-filters discovery can silently drop the
# checked count while `checked > 0` keeps reporting success. 400 stays below
# 453 to tolerate records being added/moved, but well above what an
# accidentally-narrowed root (e.g. only "julia", ~52 records) would produce.
# Only applies to the default-corpus path -- CHROMAKIT_RECORD_CORPUS is
# explicitly meant to point at smaller fixture trees, so it keeps the older,
# permissive `checked > 0` floor.
MIN_DEFAULT_CORPUS_RECORDS = 400

USING_OVERRIDE = "CHROMAKIT_RECORD_CORPUS" in os.environ


def _corpus_records():
    override = os.environ.get("CHROMAKIT_RECORD_CORPUS")
    if override:
        # Override IS the corpus root -- scan it directly, do not re-append
        # SEARCH_ROOTS onto it.
        base = Path(override)
        if not base.is_dir():
            return []
        found = []
        for path in base.rglob("*.json"):
            text = str(path)
            if ".D/" in text or ".C/results/" in text:
                found.append(path)
        return found

    base = DEFAULT_CORPUS
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

    if USING_OVERRIDE:
        assert checked > 0, "corpus present but no records were recognised"
    else:
        assert checked >= MIN_DEFAULT_CORPUS_RECORDS, (
            f"only {checked} of the default corpus's ~453 records were "
            f"recognised (need >= {MIN_DEFAULT_CORPUS_RECORDS}) -- did "
            "SEARCH_ROOTS narrow, or did the .D//.C/results/ discovery "
            "filter start rejecting records it used to accept?"
        )


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
