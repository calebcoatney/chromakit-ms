"""Hover help text for the Integration Parameters panel.

``PARAMETER_HELP`` maps a section key (the same keys used in
``ParametersFrame.section_groups``) to a dict of ``{label text: html}``.
The special key ``SECTION_KEY`` holds the "About this section" text.
Label keys must match the visible QLabel / QCheckBox text exactly.
"""

SECTION_KEY = "__section__"

PARAMETER_HELP = {
    "smoothing": {
        SECTION_KEY: (
            "<b>Signal Smoothing</b><br>"
            "Reduces high-frequency noise <i>before</i> baseline correction and peak "
            "detection. Smoothing changes the signal used to find and integrate peaks.<br><br>"
            "• <b>More</b> smoothing gives fewer noise peaks and smoother peak edges, "
            "but lowers and widens sharp peaks and can merge close peaks.<br>"
            "• <b>Less</b> smoothing keeps peak shape and resolution, but noise may be "
            "detected as small peaks and integration bounds may jitter."
        ),
        "Enable Smoothing": (
            "<b>Enable Smoothing</b><br>"
            "Turns smoothing on or off. When off, the raw detector signal is used for "
            "baseline fitting, peak detection and integration."
        ),
        "Method:": (
            "<b>Smoothing Method</b><br>"
            "• <b>Whittaker</b>: penalized least-squares smoother controlled by one "
            "value (λ). It handles uneven noise well and is a good default.<br>"
            "• <b>Savitzky-Golay</b>: fits a local polynomial in a moving window. "
            "It keeps peak height and width well when the window is small compared "
            "with the peak width, and is a common choice in chromatography."
        ),
        "Median Pre-filter": (
            "<b>Median Pre-filter</b><br>"
            "Runs a median filter before the main smoother. It removes single-point "
            "spikes (electrical noise, MS scan glitches) without spreading them out "
            "like an averaging filter would.<br><br>"
            "Use it when isolated spikes are detected as peaks. It has little effect "
            "on broad, real peaks."
        ),
        "Median kernel:": (
            "<b>Median Kernel Size</b> (odd number of points)<br>"
            "Number of points in the median window.<br><br>"
            "• <b>Higher</b> removes wider spikes but can flatten the tops of very "
            "narrow peaks and lower their area.<br>"
            "• <b>Lower</b> is gentler and only removes 1–2 point spikes."
        ),
        "Smoothing \u03bb:": (
            "<b>Whittaker λ</b> (log10 scale)<br>"
            "Sets the trade-off between following the data and being smooth.<br><br>"
            "• <b>Higher λ</b> gives a smoother signal: less noise, but peaks become "
            "shorter and wider and close peaks may merge into one.<br>"
            "• <b>Lower λ</b> follows the raw data closely: sharper peaks, but more "
            "noise and possibly more false small peaks."
        ),
        "Penalty order:": (
            "<b>Whittaker Penalty Order</b><br>"
            "• <b>d=1 (slope)</b>: penalizes changes in slope. Smoothing is stronger "
            "and can flatten peak apexes slightly.<br>"
            "• <b>d=2 (curvature)</b>: penalizes bending. Peak shapes are kept better "
            "at the same λ, but noise is removed less aggressively, so a higher λ is "
            "usually needed."
        ),
        "Window length:": (
            "<b>Savitzky-Golay Window Length</b> (odd number of points)<br>"
            "Number of points used in each local polynomial fit.<br><br>"
            "• <b>Larger</b> windows smooth more but shorten and widen narrow peaks. "
            "Keep the window smaller than the narrowest peak width (in points).<br>"
            "• <b>Smaller</b> windows keep peak shape but remove less noise."
        ),
        "Polynomial order:": (
            "<b>Savitzky-Golay Polynomial Order</b><br>"
            "Degree of the local polynomial.<br><br>"
            "• <b>Higher</b> order follows sharp peak tops more closely (keeps "
            "height and area) but removes less noise.<br>"
            "• <b>Lower</b> order (1–2) smooths more but flattens sharp peaks.<br>"
            "Must be less than the window length."
        ),
    },
    "baseline": {
        SECTION_KEY: (
            "<b>Baseline Correction</b><br>"
            "Estimates the slowly varying background (column bleed, drift, solvent "
            "tail) under the peaks. Peak areas are integrated <i>above this "
            "baseline</i>, so the baseline directly controls how large each area is.<br><br>"
            "• A baseline that runs too <b>high</b> (cutting into peaks) gives "
            "areas that are too small.<br>"
            "• A baseline that runs too <b>low</b> (below the true background) "
            "gives areas that are too large and can make noise look like peaks."
        ),
        "Enable Baseline Correction": (
            "<b>Enable Baseline Correction</b><br>"
            "Turns baseline fitting on or off. When off, no baseline is subtracted: "
            "peaks are detected and integrated on the (smoothed) signal measured from "
            "zero, so areas include any background offset. The algorithm, λ, offset "
            "and break-point settings are ignored."
        ),
        "Show Corrected Signal": (
            "<b>Show Corrected Signal</b><br>"
            "Display only. Unchecked shows the raw signal with the fitted baseline "
            "drawn on top. Checked shows the signal minus the baseline (baseline at "
            "zero). This does not change integration results."
        ),
        "Algorithm:": (
            "<b>Baseline Algorithm</b> (from pybaselines)<br>"
            "• <b>arpls</b> (default): reweighted least squares that ignores points "
            "above the baseline. Works well for most GC data.<br>"
            "• <b>asls</b>: classic asymmetric least squares. Simple, but tends to "
            "sit slightly high under large peaks.<br>"
            "• <b>airpls</b>: adaptive reweighting. Good for strongly curved "
            "baselines; can dip below noisy regions.<br>"
            "• <b>imodpoly / modpoly</b>: iterative polynomial fits. Good for smooth, "
            "simple drift; cannot follow sharp baseline changes. λ is not used.<br>"
            "• <b>snip</b>: clipping filter. Robust for dense peak clusters; λ is not "
            "used.<br>"
            "• <b>mixture_model</b>: statistical baseline/peak separation. Slower, "
            "good with heavy noise.<br>"
            "• <b>irsqr</b>: spline quantile regression. Flexible, follows wavy "
            "baselines.<br>"
            "• <b>fastchrom</b>: finds flat, low-variance regions and interpolates "
            "between them. Similar to commercial software; uses its own window "
            "settings instead of λ."
        ),
        "Lambda (\u03bb):": (
            "<b>Baseline λ (stiffness)</b> (log10 scale)<br>"
            "Sets how stiff the fitted baseline is.<br><br>"
            "• <b>Higher λ</b> gives a stiffer, flatter baseline that will not rise "
            "into broad peaks. It may not follow real drift, which can give "
            "too-large or too-small areas where the background changes quickly.<br>"
            "• <b>Lower λ</b> gives a flexible baseline that follows drift closely, "
            "but it can rise into the base of wide peaks and lower their area."
        ),
        "Baseline Offset:": (
            "<b>Baseline Offset</b> (signal units)<br>"
            "A constant subtracted from the fitted baseline.<br><br>"
            "• <b>Positive</b> values lower the baseline, which <b>increases</b> "
            "every peak's area and height.<br>"
            "• <b>Negative</b> values raise it, which <b>decreases</b> areas.<br>"
            "Use it to match another integration engine's baseline placement. "
            "Accepts scientific notation (e.g. 1.5e-2)."
        ),
        "Half Window:": (
            "<b>FastChrom Half Window</b> (points, 0 = Auto)<br>"
            "Half-width of the rolling window used to find flat baseline regions. "
            "Set it to about the FWHM of a typical peak.<br><br>"
            "• <b>Larger</b> values ignore wider peaks when finding baseline, but may "
            "miss short flat regions between close peaks.<br>"
            "• <b>Smaller</b> values find more baseline points, but parts of broad "
            "peaks may be treated as baseline (which lowers their area)."
        ),
        "Smooth Half Window:": (
            "<b>FastChrom Smooth Half Window</b> (points, 0 = Auto)<br>"
            "Smoothing applied to the interpolated FastChrom baseline. "
            "<b>Larger</b> values give a smoother baseline; <b>smaller</b> values "
            "follow the baseline points more closely."
        ),
        "Signal Break Points": (
            "<b>Signal Break Points</b><br>"
            "Splits the chromatogram into separate segments at the given times. Each "
            "segment gets its own baseline fit. Use these at valve switches or "
            "other sudden step changes, where a single smooth baseline would cut "
            "through the step and distort nearby peak areas.<br><br>"
            "<b>Tolerance</b>: window (min) around the break time."
        ),
    },
    "peaks": {
        SECTION_KEY: (
            "<b>Peak Detection</b><br>"
            "Finds peak apexes in the baseline-corrected signal and sets the "
            "start/end points used for integration. These settings decide <i>which</i> "
            "peaks get integrated and how merged peaks are split."
        ),
        "Enable Peak Detection": (
            "<b>Enable Peak Detection</b><br>"
            "Turns peak finding and integration on or off. When off, only smoothing "
            "and baseline correction are shown."
        ),
        "Detection Mode:": (
            "<b>Detection Mode</b><br>"
            "• <b>Classical</b>: scipy <i>find_peaks</i> on the corrected signal, using "
            "prominence and width thresholds. Fast and predictable. Overlapping "
            "peaks are split at the valley minimum (drop-line).<br>"
            "• <b>Peak Splitting (U-Net)</b>: a trained neural network finds apexes, "
            "including hidden ones in merged peaks. Each peak is then separated "
            "geometrically or by EMG curve fitting. Gives better areas for co-eluting "
            "peaks; slower, and needs PyTorch. If time windows are set, splitting "
            "runs only inside them and classical detection is used elsewhere."
        ),
        "Min Prominence:": (
            "<b>Minimum Prominence</b> (signal units)<br>"
            "How far a peak must rise above its surrounding valleys to count as a "
            "peak.<br><br>"
            "• <b>Higher</b> values ignore small peaks and noise; only larger peaks "
            "are integrated.<br>"
            "• <b>Lower</b> values find trace peaks, but noise and baseline wiggles may "
            "also be integrated.<br>"
            "In Peak Splitting mode this applies to classical detection outside the "
            "splitting windows."
        ),
        "Min Width (samples):": (
            "<b>Minimum Width</b> (data points)<br>"
            "Peaks narrower than this are discarded. Noise spikes are usually "
            "1–3 points wide; real GC peaks are wider.<br><br>"
            "• <b>Higher</b> removes more spikes, but can drop very sharp real "
            "peaks.<br>"
            "• <b>0</b> disables the width filter."
        ),
        "Splitting Method:": (
            "<b>Splitting Method</b><br>"
            "How merged peaks are divided after the U-Net finds their apexes.<br>"
            "• <b>Geometric</b>: splits at valleys between apexes. Fast, with "
            "accurate retention times and areas for large peaks.<br>"
            "• <b>EMG Curve Fitting</b>: fits exponentially modified Gaussian shapes. "
            "Better at recovering small shoulders under large peaks and gives peak-"
            "shape parameters (σ, τ). Slower and more sensitive to settings."
        ),
        "U-Net Confidence:": (
            "<b>U-Net Confidence</b><br>"
            "Minimum network confidence needed to accept an apex.<br><br>"
            "• <b>Lower</b> finds more (smaller, hidden) peaks, with more false "
            "positives.<br>"
            "• <b>Higher</b> keeps only confident apexes; merged peaks may stay unsplit."
        ),
        "Pre-fit Signal Gate:": (
            "<b>Pre-fit Signal Gate</b> (fraction of max signal)<br>"
            "Ignores U-Net detections where the signal is below this fraction of the "
            "largest peak, so baseline noise is not fitted.<br><br>"
            "• <b>Higher</b> skips more low-level detections.<br>"
            "• <b>Lower</b> (or 0) attempts to fit even tiny features."
        ),
        "Min Area Fraction:": (
            "<b>Minimum Area Fraction</b><br>"
            "Removes split components whose area is below this fraction of the "
            "median component area (\"phantom\" peaks).<br><br>"
            "• <b>Higher</b> filters more aggressively and can remove real minor "
            "peaks.<br>"
            "• <b>Lower</b> keeps more small components."
        ),
        "Valley Depth:": (
            "<b>Valley Depth</b><br>"
            "How deep the valley between two apexes must be (relative to the peaks) "
            "before they are split.<br><br>"
            "• <b>Lower</b> splits at shallow valleys, so more peaks are separated.<br>"
            "• <b>Higher</b> needs a deep valley, so close peaks are more often "
            "integrated together."
        ),
        "Position Bounds (x sigma):": (
            "<b>EMG Position Bounds</b> (× σ)<br>"
            "How far each fitted peak center may move from the U-Net apex.<br><br>"
            "• <b>Smaller</b> keeps fitted retention times close to detected apexes "
            "(more stable).<br>"
            "• <b>Larger</b> allows more movement, which can improve fits but lets "
            "components drift or swap."
        ),
        "Width Threshold:": (
            "<b>EMG Width Threshold</b><br>"
            "Components wider (FWHM) than this fraction of the fit window are treated "
            "as too wide and trigger a refit.<br><br>"
            "• <b>Lower</b> refits more often, preventing one wide component from "
            "absorbing nearby peaks.<br>"
            "• <b>Higher</b> allows broader components."
        ),
        "Dedup Distance (sigma):": (
            "<b>EMG Dedup Distance</b> (× σ)<br>"
            "Fitted components closer than this are merged into one peak.<br><br>"
            "• <b>Higher</b> merges more, giving fewer duplicate peaks.<br>"
            "• <b>0</b> disables merging."
        ),
        "Dedup Distance (min):": (
            "<b>Geometric Dedup Distance</b> (minutes)<br>"
            "Split components with retention times closer than this are merged.<br><br>"
            "• <b>Higher</b> merges more, giving fewer duplicates but possibly "
            "combining real close peaks.<br>"
            "• <b>0</b> disables merging."
        ),
    },
    "negative_peaks": {
        SECTION_KEY: (
            "<b>Negative Peak Detection</b><br>"
            "Finds peaks that point <i>below</i> the baseline, such as H₂ on a "
            "helium-carrier TCD. They are integrated as positive areas between the "
            "baseline and the dip."
        ),
        "Enable Negative Peak Detection": (
            "<b>Enable Negative Peak Detection</b><br>"
            "Also search the inverted signal for downward peaks. Leave off unless "
            "your detector produces negative responses, so baseline dips are not "
            "integrated as peaks."
        ),
        "Min Prominence:": (
            "<b>Negative Peak Minimum Prominence</b> (signal units)<br>"
            "How deep a dip must be to count as a negative peak.<br><br>"
            "• <b>Higher</b> detects only deep negative peaks.<br>"
            "• <b>Lower</b> detects shallow dips, but baseline noise may be "
            "integrated."
        ),
    },
    "shoulders": {
        SECTION_KEY: (
            "<b>Shoulder Detection</b><br>"
            "Finds peaks hidden in larger ones: <b>shoulders</b> (merged on a flank, "
            "no valley) and <b>riders</b> (a small peak on a long tail). Curvature "
            "proposes each candidate; it is kept only if a peak-shape fit with it is "
            "significantly better than without it. Detected peaks are split from the "
            "main peak, so the main peak's area becomes smaller and the shoulder gets "
            "its own area."
        ),
        "Enable Shoulder Detection": (
            "<b>Enable Shoulder Detection</b><br>"
            "Turns shoulder search on or off. When off, shoulders stay part of the "
            "main peak's area."
        ),
        "Sensitivity:": (
            "<b>Shoulder Sensitivity</b> (1–10)<br>"
            "How strong a curvature bump must be to be tested as a shoulder.<br><br>"
            "• <b>Higher</b> finds weaker and closer shoulders; the shape fit still "
            "rejects most noise, but more peaks get split.<br>"
            "• <b>Lower</b> splits only obvious shoulders.<br>"
            "5 = default; 1 = very strict; 10 = most sensitive."
        ),
        "Detection:": (
            "<b>Shoulder Detection Settings</b><br>"
            "How sensitive the shoulder search is."
        ),
    },
    "range_filters": {
        SECTION_KEY: (
            "<b>Peak Range Filters</b><br>"
            "Keeps only peaks whose apex is inside one of these time ranges; all other "
            "peaks are removed from the results. Use it to exclude solvent fronts, "
            "column bleed at the end of the run, or uninteresting regions. "
            "Leave the table empty to keep all peaks."
        ),
    },
    "peak_grouping": {
        SECTION_KEY: (
            "<b>Peak Grouping</b><br>"
            "All peaks with apexes inside a window are combined into <b>one</b> "
            "reported peak. Its area is the sum of their areas, and its retention "
            "time is the highest point in the window.<br><br>"
            "Use it for compounds that always split into several peaks (isomers, "
            "fronting), or to report a lump area for an unresolved region. Unlike "
            "Range Filters, no peaks are removed; they are combined."
        ),
    },
    "gcxgc_layout": {
        SECTION_KEY: (
            "<b>2D Layout</b><br>"
            "Defines how the 1D detector signal is cut into modulations to build the "
            "GC×GC 2D map. Wrong values smear or split peaks across modulations, "
            "which affects detection and volumes."
        ),
        "Modulation Period (s):": (
            "<b>Modulation Period</b> (seconds)<br>"
            "Length of each second-dimension slice. It must match the modulator "
            "setting. Leave blank to use the value from the run files."
        ),
        "Phase Shift (s):": (
            "<b>Phase Shift</b> (seconds)<br>"
            "Moves where each modulation starts, so peaks are not split across the "
            "top and bottom of the 2D map. Uses ChromSpace's sign convention; leave "
            "blank to use ChromSpace's saved view."
        ),
        "Detection Signal:": (
            "<b>Detection Signal</b><br>"
            "Signal used for 2D peak detection and volumes.<br>"
            "• <b>FID</b>: best for quantitation (response is roughly proportional to "
            "carbon).<br>"
            "• <b>DBC TIC</b>: background-corrected MS total ion current; cleaner "
            "baseline.<br>"
            "• <b>TIC</b>: raw MS total ion current."
        ),
    },
    "gcxgc_baseline": {
        SECTION_KEY: (
            "<b>2D Baseline Correction</b><br>"
            "Removes background across the 2D map before peak detection. As in 1D, "
            "a baseline that is too high lowers peak volumes and one that is too low "
            "raises them."
        ),
        "Apply 2D Baseline Correction": (
            "<b>Apply 2D Baseline Correction</b><br>"
            "Turns 2D background removal on or off."
        ),
        "Lambda (\u03bb):": (
            "<b>2D Baseline λ</b> (log10 scale)<br>"
            "• <b>Higher</b> gives a stiffer, flatter baseline that will not rise into "
            "peaks, but may not follow real drift.<br>"
            "• <b>Lower</b> follows drift more closely but can remove part of broad "
            "peaks."
        ),
        "Diff. Order:": (
            "<b>2D Baseline Difference Order</b><br>"
            "Order of the smoothness penalty. <b>Higher</b> orders allow smooth "
            "curves (curvature) in the baseline; <b>lower</b> orders keep it closer "
            "to flat or straight-line."
        ),
    },
    "gcxgc_peaks": {
        SECTION_KEY: (
            "<b>2D Peak Detection</b><br>"
            "Finds peak slices in each modulation, then joins slices from adjacent "
            "modulations into one 2D peak (blob) whose volume is integrated."
        ),
        "Enable Peak Detection": (
            "<b>Enable 2D Peak Detection</b><br>"
            "Turns 2D peak finding and volume integration on or off."
        ),
        "Min Prominence:": (
            "<b>2D Minimum Prominence</b> (signal units)<br>"
            "• <b>Higher</b> keeps only larger 2D peaks.<br>"
            "• <b>Lower</b> finds trace peaks but may add noise blobs.<br>"
            "Blank = no threshold."
        ),
        "2D RT Tolerance (scans):": (
            "<b>2D RT Tolerance</b> (scans)<br>"
            "Largest second-dimension shift allowed between slices in adjacent "
            "modulations for them to be joined into the same peak.<br><br>"
            "• <b>Higher</b> joins slices more easily; close compounds may merge.<br>"
            "• <b>Lower</b> is stricter; one compound may be split into several."
        ),
        "Min Modulations:": (
            "<b>Minimum Modulations</b><br>"
            "Number of modulations a peak must cover to be kept.<br>"
            "• <b>Higher</b> removes single-slice noise but can drop narrow real "
            "peaks.<br>"
            "• <b>Lower</b> keeps more peaks."
        ),
        "Use DBC spectra (.dbc.lsc) for MS search": (
            "<b>Use DBC Spectra</b><br>"
            "Extract mass spectra from the background-corrected (.dbc.lsc) file for "
            "cleaner library matches. Unchecked uses the raw .lsc. Affects "
            "identification only, not peak volumes."
        ),
        "Spectrum Window (scans):": (
            "<b>Spectrum Window</b> (TOF scans)<br>"
            "Searches ± this many scans around each peak's mapped apex for its "
            "spectrum. <b>Larger</b> is more forgiving of alignment error, but "
            "may pick up a neighbor's spectrum."
        ),
        "FID\u2192TOF Offset:": (
            "<b>FID→TOF Offset Mode</b><br>"
            "Time offset between the FID and the MS (TOF) detector.<br>"
            "• <b>Auto (fit)</b>: estimated from the data for each run.<br>"
            "• <b>Fixed</b>: uses the value below; use this if the automatic fit is "
            "unreliable."
        ),
        "Offset (TOF scans):": (
            "<b>Fixed FID→TOF Offset</b> (scans)<br>"
            "Offset applied in Fixed mode. Wrong values match FID peaks to the wrong "
            "spectra."
        ),
        "TIC Display:": (
            "<b>TIC Display</b><br>"
            "Display only: shows the raw or background-corrected TIC heatmap. It does "
            "not change detection or volumes."
        ),
    },
}
