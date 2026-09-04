# tools/make_envelope_parity_fixture.py
"""Capture rapids-bridge's envelope output so ChromaKit can be tested against it.

Run this with the RAPIDS environment, NOT chromakit-env — the two are
deliberately separate, which is why this is a checked-in fixture and not an
import.

Usage:
    conda activate rapids-env
    python tools/make_envelope_parity_fixture.py \
        --bridge "$HOME/Library/CloudStorage/OneDrive-NREL/RAPIDS/rapids-bridge" \
        --out tests/data/envelope_parity.json
"""
import argparse
import json
import random
import sys

CASES = ((10, 2), (5, 3), (1000, 7), (5000, 49), (3, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--out", default="tests/data/envelope_parity.json")
    args = ap.parse_args()

    sys.path.insert(0, args.bridge)
    from rapids_bridge.chromatogram import _envelope

    random.seed(20260904)
    cases = []
    for n, width in CASES:
        xs = [i * 0.01 for i in range(n)]
        ys = [random.uniform(-100.0, 5000.0) for _ in range(n)]
        out_x, out_y = _envelope(xs, ys, width)
        cases.append({"width": width, "xs": xs, "ys": ys,
                      "out_x": out_x, "out_y": out_y})

    doc = {"source": "rapids_bridge.chromatogram._envelope",
           "regenerate": "conda activate rapids-env && python tools/make_envelope_parity_fixture.py --bridge <rapids-bridge>",
           "cases": cases}
    with open(args.out, "w") as fh:
        json.dump(doc, fh)
    print(f"wrote {len(cases)} parity cases to {args.out}")


if __name__ == "__main__":
    main()
