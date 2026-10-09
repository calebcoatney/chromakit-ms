"""Convert a NIST MSP export (e.g. from lib2nist) into the text layout ms-toolkit parses.

ms-toolkit reads one "m/z intensity" pair per line and a `CASNO:` field. lib2nist's MSP
output can put several "m/z intensity;" pairs on a line and writes `CAS#: ...; NIST#: ...`.

Usage: python tools/nist_text_for_mstoolkit.py <in.msp> <out.txt>
"""
from __future__ import annotations

import re
import sys

_PAIR = re.compile(r"([0-9]+(?:\.[0-9]+)?)[\s,:]+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)")


def convert_lines(lines) -> list[str]:
    out: list[str] = []
    in_peaks = False

    def end_record():
        if out and out[-1] != "":
            out.append("")

    for raw in lines:
        line = raw.strip()
        if not line:
            end_record()
            in_peaks = False
            continue
        key, sep, val = line.partition(":")
        k = key.strip().lower()
        if sep and k == "name":
            end_record()
            out.append(f"Name: {val.strip()}")
            in_peaks = False
        elif sep and k in ("formula", "mw"):
            out.append(f"{key.strip()}: {val.strip()}")
        elif sep and k in ("cas#", "casno", "cas"):
            out.append(f"CASNO: {val.split(';')[0].strip()}")
        elif sep and k == "num peaks":
            out.append(f"Num Peaks: {val.strip()}")
            in_peaks = True
        elif in_peaks:
            out.extend(f"{mz} {inten}" for mz, inten in _PAIR.findall(line))
    end_record()
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[0], encoding="utf-8", errors="replace") as f:
        lines = convert_lines(f.read().splitlines())
    with open(argv[1], "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {sum(1 for x in lines if x.startswith('Name: '))} spectra to {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
