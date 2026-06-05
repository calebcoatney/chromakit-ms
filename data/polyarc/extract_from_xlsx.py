"""
One-off extraction script: reads Kelly's GCMS-Polyarc workbook and writes
data/polyarc/compounds.csv.

Each row of the workbook's "Compounds" sheet describes one library compound.
The Excel column L formula references either `$P$2` (nonane anchor) or
`$P$6` (levoglucosan anchor), which we capture as the `anchor` column.

Usage:
    python data/polyarc/extract_from_xlsx.py \\
        ~/Library/CloudStorage/OneDrive-NREL/GC-MS/examples/GCMS-Polyarc_Alder\\ FPO-11_12_March\\ 2026\\ \\(1\\).xlsx \\
        data/polyarc/compounds.csv

Re-run only when Kelly publishes a new library version.
"""
import argparse
import csv
from pathlib import Path

import openpyxl


def extract(src_xlsx: Path, dst_csv: Path) -> int:
    """Extract Compounds sheet to CSV. Returns row count written."""
    wb_f = openpyxl.load_workbook(src_xlsx, data_only=False)
    wb_d = openpyxl.load_workbook(src_xlsx, data_only=True)
    wsf = wb_f['Compounds']
    wsd = wb_d['Compounds']

    rows = []
    for r in range(2, wsf.max_row + 1):
        name = wsd.cell(r, 1).value
        if name is None:
            continue
        cas = wsd.cell(r, 2).value
        g1 = wsd.cell(r, 3).value
        g2 = wsd.cell(r, 4).value
        g3 = wsd.cell(r, 5).value
        C = wsd.cell(r, 6).value
        H = wsd.cell(r, 7).value
        O = wsd.cell(r, 8).value
        S = wsd.cell(r, 9).value
        N = wsd.cell(r, 10).value
        MW = wsd.cell(r, 11).value

        L_formula = wsf.cell(r, 12).value
        if isinstance(L_formula, str) and '$P$2' in L_formula:
            anchor = 'nonane'
        elif isinstance(L_formula, str) and '$P$6' in L_formula:
            anchor = 'levoglucosan'
        else:
            anchor = ''

        rows.append({
            'compound': str(name).strip(),
            'cas': str(cas).strip() if cas is not None else '',
            'group1': g1 or '',
            'group2': g2 or '',
            'group3': g3 or '',
            'C': int(C) if isinstance(C, (int, float)) else '',
            'H': int(H) if isinstance(H, (int, float)) else '',
            'O': int(O) if isinstance(O, (int, float)) else '',
            'S': int(S) if isinstance(S, (int, float)) else '',
            'N': int(N) if isinstance(N, (int, float)) else '',
            'MW': round(MW, 4) if isinstance(MW, (int, float)) else '',
            'anchor': anchor,
        })

    fieldnames = ['compound', 'cas', 'group1', 'group2', 'group3',
                  'C', 'H', 'O', 'S', 'N', 'MW', 'anchor']
    with open(dst_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('src_xlsx', type=Path, help="Kelly's GCMS-Polyarc workbook")
    ap.add_argument('dst_csv', type=Path, help='Output compounds.csv path')
    args = ap.parse_args()

    args.dst_csv.parent.mkdir(parents=True, exist_ok=True)
    n = extract(args.src_xlsx, args.dst_csv)
    print(f'Wrote {n} rows to {args.dst_csv}')


if __name__ == '__main__':
    main()
