# Polyarc Quantitation Data

Reference data for the library-broadcast Polyarc quantitation method (see
`logic/polyarc_quantitation.py` and `docs/superpowers/specs/2026-06-05-polyarc-quantitator-design.md`).

## Files

### `compounds.csv`
Compound library with chemistry-group taxonomy, atom counts, and molecular
weight. Each row also declares which anchor standard its response factor is
broadcast from (`nonane` for most compounds, `levoglucosan` for sugars and
anhydrosugars).

Schema:

| column   | type   | notes                                          |
|----------|--------|------------------------------------------------|
| compound | string | NIST library name; may include trailing dashes |
| cas      | string | zero-padded form (e.g. `000064-19-7`); empty if unknown |
| group1   | string | e.g. `Oxygenate`, `Aromatic`, `Alkane`         |
| group2   | string | e.g. `Acid`, `Phenols`, `Cycloalkane`          |
| group3   | string | finest classification, e.g. `Acetic Acid`      |
| C        | int    | carbon count; blank parsed as 0                |
| H        | int    | hydrogen count; blank parsed as 0              |
| O        | int    | oxygen count; blank parsed as 0                |
| S        | int    | sulfur count; blank parsed as 0                |
| N        | int    | nitrogen count; blank parsed as 0              |
| MW       | float  | molecular weight (g/mol); informational, recomputed at load time from C/H/O/S/N |
| anchor   | string | `nonane` or `levoglucosan`                     |

Source: Kelly Orton's `GCMS-Polyarc_Alder FPO-11_12_March 2026.xlsx`
(`Compounds` sheet).

### `calibration.csv`
Anchor calibration points (one row per anchor compound). Values are
**chromakit-scaled** — re-run when chromakit's area-scaling profile changes.

Schema:

| column         | type   | notes                                       |
|----------------|--------|---------------------------------------------|
| anchor         | string | matches `compounds.csv:anchor` column       |
| cas            | string | anchor compound's CAS                       |
| C              | int    | anchor's carbon count                       |
| MW             | float  | anchor's molecular weight                   |
| known_wt_pct   | float  | wt% of anchor in cal mix at the chosen point, after dilution |
| area           | float  | measured FID area at that point             |
| run_date       | string | ISO date (YYYY-MM-DD) for traceability      |
| instrument_id  | string | e.g. `Agilent-GCMS-2`                       |
| notes          | string | freeform                                    |

### `extract_from_xlsx.py`
One-off Python script that regenerates `compounds.csv` from Kelly's
workbook. Kept for provenance; re-run only if Kelly publishes an updated
library. Requires `openpyxl`.

## Regeneration

```bash
# compounds.csv (from Kelly's xlsx)
python data/polyarc/extract_from_xlsx.py <path-to-xlsx> data/polyarc/compounds.csv

# calibration.csv (manually maintained until §11.7 of the design spec)
$EDITOR data/polyarc/calibration.csv
```
