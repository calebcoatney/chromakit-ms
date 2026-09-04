# tools/make_render_fixtures.py
"""Distil one test fixture per result-JSON schema variant from the real corpus.

Each fixture is a real record trimmed to at most 3 peaks. Source is recorded as
a basename only — absolute OneDrive paths do not belong in the repo.

Usage:
    python tools/make_render_fixtures.py \
        --base "$HOME/Library/CloudStorage/OneDrive-NREL" \
        --out tests/data/records
"""
import argparse
import json
from pathlib import Path

SEARCH_ROOTS = ("julia", "GC-MS", "GC-MS code development",
                "ChemCatBio - Acid Upgrading")


def iter_records(base: Path):
    for root in SEARCH_ROOTS:
        start = base / root
        if not start.is_dir():
            continue
        for path in start.rglob("*.json"):
            text = str(path)
            if ".D/" in text or ".C/results/" in text:
                yield path


# Top-level keys that vary independently of the peak schema. `source_d` records
# share a peak schema with co-located ones, and `processing_parameters` is
# absent from 206 of 453 records — selecting on peak keys alone would miss both.
TOP_LEVEL_MARKERS = ("source_d", "processing_parameters", "quantitation")


def schema_key(doc: dict) -> tuple:
    keys = set()
    for peak in doc["peaks"]:
        keys |= set(peak.keys())
    markers = tuple(m for m in TOP_LEVEL_MARKERS if m in doc)
    return (tuple(sorted(keys)), markers)


def rank(peak: dict) -> tuple:
    """Prefer peaks that are quantitated, then named — they exercise more fields."""
    quantitated = 0 if peak.get("mol_percent") is not None else 1
    named = 0 if (peak.get("compound_id") or peak.get("Compound ID")) else 1
    return (quantitated, named)


def scrub_paths(obj, base: Path):
    """Replace any string under `base` with its basename, recursively.

    Some records carry the source .D directory's absolute path in a field
    (e.g. `source_d`) that survives the peak-trimming above verbatim. Fixture
    provenance must record a basename only — absolute OneDrive paths must not
    enter the repo — so scrub any string value, anywhere in the trimmed doc,
    that looks like a path under the corpus base.
    """
    base_str = str(base)
    if isinstance(obj, dict):
        return {k: scrub_paths(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_paths(v, base) for v in obj]
    if isinstance(obj, str) and base_str in obj:
        return Path(obj).name
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="tests/data/records")
    args = ap.parse_args()

    seen: dict = {}
    for path in iter_records(Path(args.base)):
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        peaks = doc.get("peaks")
        if not isinstance(peaks, list) or not peaks or not isinstance(peaks[0], dict):
            continue
        key = schema_key(doc)
        if key in seen:
            continue
        trimmed = dict(doc)
        trimmed["peaks"] = sorted(peaks, key=rank)[:3]
        trimmed = scrub_paths(trimmed, Path(args.base))
        seen[key] = (path.name, trimmed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (key, (source, doc)) in enumerate(sorted(seen.items()), start=1):
        name = f"schema{index:02d}.json"
        (out / name).write_text(json.dumps(doc, indent=2))
        peak_keys, markers = key
        manifest.append({"fixture": name, "source": source,
                         "peak_keys": list(peak_keys), "markers": list(markers)})
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} fixtures to {out}")


if __name__ == "__main__":
    main()
