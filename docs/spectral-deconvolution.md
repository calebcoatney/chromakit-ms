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
2. **Groups.** Ion peaks are taken strongest first. An ion peak whose apex is
   within the RT tolerance of a group's strongest ion joins that group if its
   shape tracks that ion's (Pearson r ≥ the minimum; when several groups
   qualify, the best r wins). If it is within the RT tolerance of a stronger
   group but does not track its shape, it is discarded: it does not start a
   group of its own. That is why a close, much smaller co-eluter cannot form its
   own spectrum (see Known limits). An ion with no group within the RT
   tolerance starts a new group, and groups with fewer than the minimum number
   of ions are dropped.
3. **Spectra.** A group's spectrum is each member ion's peak height. Its lead
   ion (the strongest not on the excluded list) sets its retention time, and
   it is matched to the nearest FID peak within the RT match tolerance.

FID peaks with no matched group are searched with their apex spectrum, as
before.

## Settings (MS Options → Spectral Deconvolution, and the Inspector)

| Setting | Default | Raise it when… | Lower it when… |
|---|---|---|---|
| Gap tolerance (min) | Auto (0.5 × median FID peak width) | FID peaks that nearly touch land in separate windows | unrelated peaks are pooled into one window |
| Padding fraction | 0.5 | a compound's ion peaks are cut off at the window edge (an ion peak touching the edge is rejected) | windows reach into neighbouring compounds |
| RT match tolerance (min) | 0.05 | a group's RT sits further from its FID peak than this, so it isn't matched | a group is matched to the wrong neighbouring FID peak |
| Max peaks per window | 10 (0 = Unlimited) | a crowded region is split between peaks that belong together | a crowded region becomes one very wide window |
| Noise percentile | 0.85 | noisy ions create junk peaks | weak real ions vanish |
| Search range (min) | 0.03 | one compound splits into several peaks | close isomers merge (set near one peak width) |
| Min peak height (counts) | 1000 | noise peaks get in | trace compounds are missed |
| Min peak top/edge ratio | 1.7 | bumps on a high baseline count as peaks | real shouldered peaks are dropped |
| Min points per peak | 4 | spikes appear | very fast peaks are dropped |
| RT tolerance (min) | 0.04 | ions of one compound land in separate groups | neighbouring compounds merge |
| Min shape correlation (r) | 0.80 | spectra pick up co-eluting ions | ions of noisy compounds are rejected |
| Min ions per spectrum | 4 | fragments of noise form spectra | small molecules with few ions disappear |
| Excluded m/z | (empty) | — | — (values here, ±0.5, can't lead a group, e.g. column-bleed ions; they stay in its spectrum. If every ion of a group is excluded, its strongest ion leads) |

The **Spectral Deconvolution Inspector** previews one window at a time.
- **Top plot:** ion peaks coloured by group, ★ = lead ion, hollow = rejected.
- **Bottom plot:** traces with each peak's extent shaded and its correlation r.
- Click a group to library-search its spectrum.

## Known limits

- **Close co-eluters usually merge.** Compounds whose apexes are closer than
  about 1.5 peak widths (FWHM) usually come out as one spectrum, assigned to one
  of the FID peaks; the other FID peak is searched with its apex spectrum. In a
  simulation with 1.7 s wide (FWHM) peaks, two equal compounds 2 s apart merged
  and 3 s apart separated at the default settings; changing the search range or
  the RT tolerance did not separate the 2 s pair.
- **A minor compound next to a larger one is usually lost.** A compound about
  4:1 or smaller within about 2 peak widths of a larger one usually gets no
  spectrum: an ion whose apex is within the RT tolerance of a stronger group but
  does not match its shape is discarded, as in mzmine. ChromaKit does not fit
  overlapping peaks.
- **Earlier versions separated some of these pairs.** The method used before
  this release split some close co-eluters, so a pair that used to get two
  spectra can now get one.
- Detector saturation flags are not carried through deconvolved spectra.
