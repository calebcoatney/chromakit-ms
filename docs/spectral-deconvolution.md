# MS spectral deconvolution in ChromaKit

**What it does.** Spectral deconvolution gives each integrated FID peak a cleaner
mass spectrum than the raw scan at its apex, by keeping only the ions that elute
together with the right peak shape.

**Where it runs.** It runs when you click **Deconvolve MS**, and in batch
processing before MS search (on by default). It works inside windows around
groups of FID peaks.

**How it works** (after [mzmine](https://github.com/mzmine/mzmine)'s GC-EI deconvolution):

1. **Ion peaks.** Each m/z trace in the window is split into
   baseline-to-baseline peaks, ignoring points below that ion's noise level.
2. **Groups.** Ion peaks are taken strongest first. Each joins the group whose
   strongest ion is within the RT tolerance and whose peak shape it tracks
   (Pearson r ≥ the minimum); otherwise it starts a group, or is rejected if it
   overlaps a group without matching its shape.
3. **Spectra.** A group's spectrum is each member ion's peak height. Its lead
   ion (the strongest not on the excluded list) sets its retention time, and
   it is matched to the nearest FID peak within the RT match tolerance.

FID peaks with no matched group are searched with their apex spectrum, as
before.

## Settings (MS Options → Spectral Deconvolution, and the Inspector)

| Setting | Default | Raise it when… | Lower it when… |
|---|---|---|---|
| Noise percentile | 0.85 | noisy ions create junk peaks | weak real ions vanish |
| Search range (min) | 0.05 | one compound splits into several peaks | close isomers merge (set near one peak width) |
| Min peak height (counts) | 1000 | noise peaks get in | trace compounds are missed |
| Min peak top/edge ratio | 1.7 | bumps on a high baseline count as peaks | real shouldered peaks are dropped |
| Min points per peak | 4 | spikes appear | very fast peaks are dropped |
| RT tolerance (min) | 0.04 | ions of one compound land in separate groups | neighbouring compounds merge |
| Min shape correlation (r) | 0.80 | spectra pick up co-eluting ions | ions of noisy compounds are rejected |
| Min ions per spectrum | 4 | fragments of noise form spectra | small molecules with few ions disappear |
| Excluded m/z | (empty) | — | — (values here, ±0.5, can't be a group's lead ion; e.g. column-bleed ions) |

The **Spectral Deconvolution Inspector** previews one window at a time.
- **Top plot:** ion peaks coloured by group, ★ = lead ion, hollow = rejected.
- **Bottom plot:** traces with each peak's extent shaded and its correlation r.
- Click a group to library-search its spectrum.

## Known limits

- A minor compound that co-elutes closely with a much larger one (about 5:1 and under
  a peak width apart) usually doesn't get its own spectrum; its ions are
  rejected or absorbed. ChromaKit does not fit overlapping peaks.
- Detector saturation flags are not carried through deconvolved spectra.
