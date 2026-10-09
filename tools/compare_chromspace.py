"""Compare a ChromSpace Area-percent export with a ChromaKit GCxGC peaks.json.

Usage: python tools/compare_chromspace.py <chromspace.csv> <peaks.json>
           [--t1-tol 0.1] [--t2-tol 0.1] [--out matches.csv]

ChromSpace rows are read by position: Peak #, Source, Compound, Synonyms, Library RI,
1tR (min), 2tR (min), Area, Area %. A header row is optional; "Group" rows (stencil
totals) are skipped. Tolerances: 1tR in minutes, 2tR in seconds. Rows that cannot be read are
counted in a warning on stderr, as is an export that yields no peaks. The summary reports how many
peaks on each side found no partner; --out writes the matched pairs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys


def read_chromspace(path) -> list[dict]:
    """Peak rows of a ChromSpace export. Warns on stderr about rows it had to skip and about an empty result."""
    rows, skipped = [], 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if not any(c.strip() for c in r):
                continue
            first = r[0].strip().lower()
            if first.startswith("peak") or first.startswith("group"):
                continue
            try:
                if not first:
                    raise ValueError("no peak number")
                rows.append({"peak": r[0].strip(), "compound": r[2].strip(), "t1_min": float(r[5]),
                             "t2_s": float(r[6]) * 60.0, "area": float(r[7])})
            except (ValueError, IndexError):
                skipped += 1
    if skipped:
        print(f"warning: {skipped} ChromSpace data row(s) skipped in {path}: 1tR, 2tR or Area "
              "(columns 6 to 8) missing or not a number", file=sys.stderr)
    if not rows:
        print(f"warning: no peak rows read from {path}; expected a ChromSpace Area-percent CSV with "
              "Peak #, Source, Compound, Synonyms, Library RI, 1tR, 2tR, Area, Area % in that order",
              file=sys.stderr)
    return rows


def read_chromakit(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    return [{"peak": str(p["peak_number"]), "compound": p.get("compound_name") or "",
             "t1_min": float(p["rt1"]), "t2_s": float(p["rt2"]), "area": float(p["volume"])}
            for p in rec["peaks"]]


def match(cs, ck, t1_tol, t2_tol) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, closest pairs first (distance in tolerance units)."""
    cands = []
    for i, a in enumerate(cs):
        for j, b in enumerate(ck):
            d1, d2 = abs(a["t1_min"] - b["t1_min"]) / t1_tol, abs(a["t2_s"] - b["t2_s"]) / t2_tol
            if d1 <= 1 and d2 <= 1:
                cands.append((d1 * d1 + d2 * d2, i, j))
    used_i, used_j, pairs = set(), set(), []
    for _, i, j in sorted(cands):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            pairs.append((i, j))
    return pairs


def _percents(rows):
    total = sum(r["area"] for r in rows)
    return [100.0 * r["area"] / total if total else 0.0 for r in rows]


def summarize(cs, ck, pairs) -> dict:
    ratios = [ck[j]["area"] / cs[i]["area"] for i, j in pairs if cs[i]["area"] > 0]
    pa, pb = _percents(cs), _percents(ck)
    diffs = [pb[j] - pa[i] for i, j in pairs]
    named = [(cs[i]["compound"], ck[j]["compound"]) for i, j in pairs if cs[i]["compound"] and ck[j]["compound"]]
    return {
        "chromspace_peaks": len(cs),
        "chromakit_peaks": len(ck),
        "matched": len(pairs),
        "chromspace_unmatched": len(cs) - len(pairs),
        "chromakit_unmatched": len(ck) - len(pairs),
        "volume_ratio_median": statistics.median(ratios) if ratios else None,
        "volume_ratio_iqr": (lambda q: q[2] - q[0])(statistics.quantiles(ratios, n=4)) if len(ratios) >= 2 else None,
        "area_percent_rmse": (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else None,
        "id_pairs": len(named),
        "id_agree": sum(1 for a, b in named if a.lower() == b.lower()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chromspace_csv")
    ap.add_argument("chromakit_json")
    ap.add_argument("--t1-tol", type=float, default=0.1)
    ap.add_argument("--t2-tol", type=float, default=0.1)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cs, ck = read_chromspace(args.chromspace_csv), read_chromakit(args.chromakit_json)
    pairs = match(cs, ck, args.t1_tol, args.t2_tol)
    print(json.dumps(summarize(cs, ck, pairs), indent=2))
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["chromspace_peak", "chromakit_peak", "cs_1tR_min", "ck_1tR_min", "cs_2tR_s", "ck_2tR_s",
                        "cs_area", "ck_volume", "cs_compound", "ck_compound"])
            for i, j in pairs:
                a, b = cs[i], ck[j]
                w.writerow([a["peak"], b["peak"], a["t1_min"], b["t1_min"], a["t2_s"], b["t2_s"],
                            a["area"], b["area"], a["compound"], b["compound"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
