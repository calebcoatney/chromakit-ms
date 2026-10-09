# GCxGC-TOFMS/FID in ChromaKit

For SepSolve/Markes INSIGHT + BenchTOF2 + FID runs acquired with TOF-DS/ChromSpace. Install
ChromaKit as described in the [README](../README.md). Library searching also needs an MS library;
see [ms-library-from-nist.md](ms-library-from-nist.md).

## Get a run into ChromaKit

Tools → Batch Convert to .C… → "Add GCxGC Files…", pick the folder holding the run's files, then
Convert. A run is recognised by `<run>.HDR` and `<run>.rsd` together with `<run>.lsc`,
`<run>.dbc.lsc` or `<run>_*.dbc.lsc`. ChromaKit copies `<run>.HDR`, `<run>.rsd`, `<run>.lsc` and any
`.dbc.lsc` files into `<run>.C/data/`, and creates the `<run>.C` folder beside the originals. The
originals, including the large `.DAT`, are not touched, and analysis never needs the `.DAT`. A run
whose `.C` folder already exists is skipped.

## Open it

Open Folder (top of the left panel) → choose the folder that contains the `.C` folder → double-click
the `.C` folder. The top heatmap is the TIC, the bottom the FID. The x axis is first-dimension
retention time in minutes, the y axis second-dimension retention time in seconds. The "TIC Display"
choice under 2D Peak Detection switches the top heatmap between the raw TIC and the background-corrected
(`.dbc.lsc`) TIC when the run has one.

The Parameters tab has the GCxGC controls:

- **2D Layout.** Modulation Period (s) and Phase Shift (s). Leave a field blank to use the run's own
  value; the grey label under it says where the value in force came from (the run's HDR, ChromSpace's
  saved view, your value, or a default). Type a number to override. The period must be a whole number
  of samples at both detector rates; if it is not, ChromaKit refuses it, names the period and the
  rate, and keeps the previous layout. Phase uses ChromSpace's sign convention. If a run's files
  record no modulation period, ChromaKit asks for it when you open the run. **Detection Signal**
  chooses FID, DBC TIC or TIC; a signal the run lacks is greyed out.
- **2D Baseline Correction.** On by default. Lambda and Diff. Order tune the 2D baseline fit.
- **2D Peak Detection.** On by default. Min Prominence (in detector counts; blank for none),
  2D RT Tolerance and Min Modulations control how slices from neighbouring modulations are joined into
  one peak. Spectrum Window (scans) sets how far around a peak's expected position ChromaKit looks for
  its mass-spectral apex. FID→TOF Offset is "Auto (fit)" by default; choose "Fixed" to enter the offset
  in TOF scans yourself.

Layout and detection settings take effect when you click Integrate.

## Find and identify peaks

- Detection Signal: FID (the default; volumes are FID-based, and FID response is roughly carbon
  proportional for hydrocarbons) or DBC TIC (for trace compounds the FID cannot see).
- Integrate (button under the plot): applies the layout, runs the 2D baseline, then finds peaks slice by slice and groups them
  across modulations. Each peak gets a red + marker on both heatmaps, and the status bar reports the
  count. If it reports none, lower Min Prominence. If it reports "FID→TOF offset fit failed", set the
  offset to Fixed under 2D Peak Detection.
- "View Peak Results" (status bar, lower right) lists every peak with its retention times, volume,
  number of modulations, compound and match score.
- Click near a peak's marker on the bottom (FID) heatmap. ChromaKit selects the nearest peak, shows its
  spectrum in the Mass Spectrometry tab and prints its retention times and volume in the status bar.
  The spectrum comes from the TOF scan at the peak's own apex: the FID apex is mapped to TOF time through
  the FID→TOF offset, and the TOF TIC is searched within the Spectrum Window for its maximum. Up to three
  adjacent scans are averaged, leaving out saturated ones.
- Search Library (Mass Spectrometry tab) shows the top hits for the spectrum on display. MS Search All
  (under the plot) searches every peak, writes the best hit into the Compound and Match Score columns
  and records the top hits in the JSON. Both need a loaded library. Use Vector Search (the default) or
  Word2Vec Search, set under Configure…; Hybrid Search is not available with the current ms-toolkit.

## Export

Export (or "Export…" in View Peak Results) writes three files to `<run>.C/results/`:

- `<run> - GCxGC peaks.csv`: ChromSpace's Area-percent layout (2tR in minutes).
- `<run> - GCxGC peaks.json`: every peak with its flags, plus the run settings, the library and the
  search method. The flags `apex_found`, `apex_distance_scans` and `saturated`, with `spectrum_scan` and
  `spectrum_source`, say how much to trust each peak's spectrum.
- `<run> - GCxGC spectra.msp`: one spectrum per peak, in MSP, a plain-text spectrum format that most
  library software can import. Peaks with no spectrum are left out.

By default Integrate and MS Search All also write these files; Settings → Export Settings controls
that. Run Integrate before Export; the files describe the peaks and settings of the last Integrate.
A run with no peaks still gets a header-only CSV, an empty MSP and a JSON with an empty peak list.

Volume (the CSV's Area) is in detector counts × samples and is not calibrated. Compare peaks with Area %.

## Check against ChromSpace

`tools/compare_chromspace.py` (in the repository, not installed by pip) matches the peaks of a
ChromSpace Area-percent CSV export with a ChromaKit peaks JSON and reports how well they agree:

```
python tools/compare_chromspace.py chromspace.csv "<run> - GCxGC peaks.json" --out matches.csv
```

Peaks are paired one-to-one by closest retention time within `--t1-tol` (minutes, default 0.1) and
`--t2-tol` (seconds, default 0.1). It prints peak counts, the number matched, the median and
interquartile range of the volume ratio (ChromaKit over ChromSpace), the RMSE of Area %, and how many
named pairs agree. `--out` saves the matched pairs.

## Not yet

Region/stencil drawing and group-type (PIONA) wt%, deconvolution, retention indices, multi-sample
alignment, and batch processing of GCxGC runs ("Batch Process All Files" says so and does nothing).
