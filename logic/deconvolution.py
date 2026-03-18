"""
Deep learning peak deconvolution pipeline for merged/coeluting GC peaks.

Uses a 1D U-Net to predict apex heatmaps from chromatogram windows,
then fits Exponentially Modified Gaussians (EMGs) to deconvolve
the individual peaks and recover their true areas.

Torch is an optional dependency — this module degrades gracefully
when it is not installed.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
from scipy.stats import exponnorm
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
#  Optional torch import
# --------------------------------------------------------------------------- #
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None


# =========================================================================== #
#  1D U-Net model (mirrors unet-training/unet_model.py)
# =========================================================================== #

def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep-learning peak deconvolution. "
            "Install it with: pip install torch"
        )


class _ConvBlock1D(nn.Module):
    """Conv -> BatchNorm -> ReLU  x2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GCHeatmapUNet(nn.Module):
    """1D U-Net that maps a normalised chromatogram window → apex heatmap."""

    def __init__(self, in_channels=1, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [16, 32, 64, 128]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder
        in_c = in_channels
        for f in features:
            self.encoder.append(_ConvBlock1D(in_c, f))
            in_c = f

        # Bottleneck
        self.bottleneck = _ConvBlock1D(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(f * 2, f, kernel_size=2, stride=2)
            )
            self.decoder.append(_ConvBlock1D(f * 2, f))

        self.final_conv = nn.Sequential(
            nn.Conv1d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)          # upsample
            x = torch.cat((skips[i // 2], x), dim=1)
            x = self.decoder[i + 1](x)      # conv block

        return self.final_conv(x)


# =========================================================================== #
#  Lazy model loader (singleton)
# =========================================================================== #

_model_cache = {}  # keyed by (path, out_channels)

# Default path: cache/gc_heatmap_unet.pth relative to the project root
_DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache", "gc_heatmap_unet.pth"
)

# For backward compatibility with evaluation code that sets _model_instance
_model_instance = None


def load_model(weights_path=None):
    """Load the pre-trained U-Net (singleton per path, CPU-only).

    Auto-detects out_channels from the state dict so both 1-channel
    and 3-channel models are loaded correctly.
    """
    global _model_instance, _model_cache
    _require_torch()

    path = weights_path or _DEFAULT_WEIGHTS_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"U-Net weights not found at {path}. "
            "Copy gc_heatmap_unet.pth into the cache/ directory."
        )

    # Check cache
    if path in _model_cache:
        return _model_cache[path]

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    out_channels = state_dict['final_conv.0.weight'].shape[0]

    model = GCHeatmapUNet(out_channels=out_channels)
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache[path] = model
    _model_instance = model
    return model


# =========================================================================== #
#  Window extraction & interpolation
# =========================================================================== #

WINDOW_LENGTH = 256  # model's fixed input size


def _extract_peak_trace(time, signal, start_idx, end_idx):
    """
    Extract a peak trace between integration bounds and interpolate to WINDOW_LENGTH.

    Returns
    -------
    t_win : 1-D array  – original time values in the window
    s_win : 1-D array  – original signal values in the window
    s_norm : 1-D array (WINDOW_LENGTH,) – normalised, interpolated signal
    t_interp : 1-D array (WINDOW_LENGTH,) – interpolated time grid
    scale : float – (s_max - s_min) used for de-normalisation
    """
    t_win = time[start_idx:end_idx + 1]
    s_win = signal[start_idx:end_idx + 1]

    if len(t_win) < 2:
        return t_win, s_win, np.zeros(WINDOW_LENGTH), np.zeros(WINDOW_LENGTH), 1.0

    # Interpolate to WINDOW_LENGTH points
    t_interp = np.linspace(t_win[0], t_win[-1], WINDOW_LENGTH)
    s_interp = np.interp(t_interp, t_win, s_win)

    # Min-max normalise to [0, 1]
    s_min, s_max = s_interp.min(), s_interp.max()
    scale = s_max - s_min
    if scale > 1e-12:
        s_norm = (s_interp - s_min) / scale
    else:
        s_norm = np.zeros(WINDOW_LENGTH)
        scale = 1.0

    return t_win, s_win, s_norm, t_interp, scale


# =========================================================================== #
#  Inference: heatmap → apex indices
# =========================================================================== #

def _predict_heatmap(model, signal_norm):
    """
    Run a single normalised 256-pt signal through the U-Net.

    Parameters
    ----------
    signal_norm : 1-D array (WINDOW_LENGTH,)

    Returns
    -------
    heatmap : 1-D array (WINDOW_LENGTH,) in [0, 1]
        For multi-channel models, returns only the apex channel.
    """
    _require_torch()
    x = torch.tensor(signal_norm, dtype=torch.float32).reshape(1, 1, -1)
    with torch.no_grad():
        pred = model(x)
    # pred shape: (1, out_channels, WINDOW_LENGTH)
    if pred.shape[1] == 1:
        return pred.squeeze().cpu().numpy()
    else:
        return pred[0, 0, :].cpu().numpy()


def _extract_apices(heatmap, height=0.15, distance=10):
    """Find apex positions in the predicted heatmap."""
    indices, props = find_peaks(heatmap, height=height, distance=distance)
    return indices, props


# =========================================================================== #
#  EMG deconvolution
# =========================================================================== #

def _single_emg(t, amp, mu, sigma, tau):
    """Single Exponentially Modified Gaussian."""
    sigma = max(sigma, 1e-6)
    K = tau / sigma if sigma > 0 else 1e-6
    return amp * exponnorm.pdf(t, K, loc=mu, scale=sigma)


def _multi_emg(t, *params):
    """Sum of N EMG peaks.  params = [amp1, mu1, sig1, tau1, ...]"""
    n = len(params) // 4
    y = np.zeros_like(t)
    for i in range(n):
        y += _single_emg(t, *params[i * 4:(i + 1) * 4])
    return y


def _deconvolve(time_array, signal_array, apex_indices):
    """
    Fit a multi-EMG model seeded by the CNN apex positions.

    Returns a list of dicts with keys:
        retention_time, area, sigma, tau, amplitude
    or an empty list on failure.
    """
    n_peaks = len(apex_indices)
    if n_peaks == 0:
        return []

    # Scale initial guesses relative to the actual peak window
    window_width = time_array[-1] - time_array[0] if len(time_array) > 1 else 1.0
    sig_guess = window_width / (n_peaks * 4)  # ~1/4 of per-peak share
    tau_guess = sig_guess * 0.3
    sig_max = window_width / 2
    tau_max = window_width / 2

    p0, lo, hi = [], [], []
    for idx in apex_indices:
        idx = min(idx, len(time_array) - 1)
        mu_guess = time_array[idx]
        amp_guess = max(signal_array[idx], 1e-6) * sig_guess * 5.0
        p0.extend([amp_guess, mu_guess, sig_guess, tau_guess])
        lo.extend([0, time_array[0], 1e-4, 1e-4])
        hi.extend([np.inf, time_array[-1], sig_max, tau_max])

    try:
        popt, _ = curve_fit(
            _multi_emg, time_array, signal_array,
            p0=p0, bounds=(lo, hi), maxfev=5000
        )
    except RuntimeError:
        return []

    results = []
    for i in range(n_peaks):
        amp, mu, sigma, tau = popt[i * 4:(i + 1) * 4]
        results.append({
            'retention_time': mu,
            'area': amp,       # for EMG via exponnorm, amp ≡ total area
            'sigma': sigma,
            'tau': tau,
            'amplitude': amp,
        })
    return results


# =========================================================================== #
#  Public API
# =========================================================================== #

def is_available():
    """Return True if torch is installed and model weights exist."""
    return _TORCH_AVAILABLE and os.path.isfile(_DEFAULT_WEIGHTS_PATH)


def detect_merged_peaks(time, signal, peaks, *, heatmap_threshold=0.15,
                        min_peak_distance=10, weights_path=None):
    """
    Scan integrated peaks for hidden coeluting components.

    For each Peak object, extracts the trace between its integration bounds,
    interpolates to 256 points, runs the U-Net, and checks whether the
    predicted heatmap contains more than one apex.

    Parameters
    ----------
    time : 1-D array
        Full retention-time axis.
    signal : 1-D array
        Full baseline-corrected signal.
    peaks : list of Peak
        Integrated Peak objects (must have start_index / end_index set).
    heatmap_threshold : float
        Minimum heatmap probability to accept an apex (0-1).
    min_peak_distance : int
        Minimum separation (in interpolated points) between detected apices.

    Returns
    -------
    list of DetectionResult (one per input peak, same order)
    """
    _require_torch()
    model = load_model(weights_path)

    results = []
    for peak in peaks:
        si = peak.start_index
        ei = peak.end_index

        # Skip peaks that are too narrow or have negligible signal
        if si is None or ei is None or ei - si < 10:
            results.append(DetectionResult(peak=peak, is_merged=False,
                                           heatmap=None, apex_indices=np.array([])))
            continue

        _tw, _sw, s_norm, t_interp, scale = _extract_peak_trace(time, signal, si, ei)

        # Skip peaks where the signal range is essentially noise
        if scale < 1e-6:
            results.append(DetectionResult(peak=peak, is_merged=False,
                                           heatmap=None, apex_indices=np.array([])))
            continue

        heatmap = _predict_heatmap(model, s_norm)
        apex_idx, _ = _extract_apices(heatmap, height=heatmap_threshold,
                                       distance=min_peak_distance)

        results.append(DetectionResult(
            peak=peak,
            is_merged=len(apex_idx) > 1,
            heatmap=heatmap,
            apex_indices=apex_idx,
            t_interp=t_interp,
            s_norm=s_norm,
            scale=scale,
        ))

    return results


def deconvolve_peak(detection_result, time, signal):
    """
    Deconvolve a single merged peak using its stored heatmap apex indices.

    Uses the apex positions from the U-Net heatmap as initial guesses for
    multi-EMG fitting on the original signal.

    Parameters
    ----------
    detection_result : DetectionResult
        Must have is_merged=True and valid apex_indices.
    time : 1-D array
        Full retention-time axis.
    signal : 1-D array
        Full baseline-corrected signal.

    Returns
    -------
    list of dicts with keys: retention_time, area, sigma, tau, amplitude
    """
    dr = detection_result
    peak = dr.peak
    if not dr.is_merged or len(dr.apex_indices) < 2:
        return []

    si, ei = peak.start_index, peak.end_index
    t_win = time[si:ei + 1]
    s_win = signal[si:ei + 1]

    # Map heatmap apex indices (0-255) back to the real time grid
    t_interp = dr.t_interp
    apex_rts = t_interp[dr.apex_indices]

    # Find nearest indices in the real signal for initial guesses
    real_apex_indices = []
    for art in apex_rts:
        idx = np.argmin(np.abs(t_win - art))
        real_apex_indices.append(idx)

    # Fit multi-EMG on the real (non-normalised) signal
    emg_results = _deconvolve(t_win, s_win, real_apex_indices)

    # Filter noise artifacts
    if emg_results:
        total_area = sum(abs(e['area']) for e in emg_results)
        if total_area > 0:
            emg_results = [e for e in emg_results
                           if abs(e['area']) / total_area >= 0.02]

    return emg_results


class DetectionResult:
    """Result of U-Net merged-peak detection for a single peak."""

    __slots__ = ('peak', 'is_merged', 'heatmap', 'apex_indices',
                 't_interp', 's_norm', 'scale')

    def __init__(self, *, peak, is_merged, heatmap, apex_indices,
                 t_interp=None, s_norm=None, scale=1.0):
        self.peak = peak
        self.is_merged = is_merged
        self.heatmap = heatmap
        self.apex_indices = apex_indices
        self.t_interp = t_interp
        self.s_norm = s_norm
        self.scale = scale


# =========================================================================== #
#  Full deconvolution pipeline (chunk → U-Net → EMG fit)
# =========================================================================== #

@dataclass
class EMGComponent:
    """A single deconvolved EMG peak component."""
    retention_time: float   # mu from fit
    area: float             # proportionally rescaled integrated area
    sigma: float
    tau: float
    peak_height: float      # max of evaluated EMG curve
    chunk_index: int
    t_curve: np.ndarray = field(repr=False)  # evaluated curve time grid
    y_curve: np.ndarray = field(repr=False)  # evaluated curve signal


@dataclass
class DeconvComponent:
    """A single deconvolved peak component from geometric splitting."""
    retention_time: float      # apex RT (snapped or U-Net predicted)
    area: float                # np.trapz of signal segment
    peak_height: float         # signal value at apex position
    chunk_index: int
    start_time: float          # left boundary RT
    end_time: float            # right boundary RT
    start_index: int           # left boundary in full time array
    end_index: int             # right boundary in full time array
    apex_index: int            # apex in full time array
    split_type_left: str       # 'chunk' | 'valley' | 'inflection' | 'midpoint'
    split_type_right: str      # 'chunk' | 'valley' | 'inflection' | 'midpoint'


@dataclass
class DeconvolutionResult:
    """Result of the full deconvolution pipeline."""
    components: list  # list of EMGComponent or DeconvComponent
    chunks: list      # list of Chunk objects


_MAX_PEAKS_PER_FIT = 4  # curve_fit scales terribly beyond this


def _apply_smoothing_for_unet(signal, smoothing_params):
    """
    Apply production-style smoothing to a signal before U-Net inference.

    Mirrors ChromatogramProcessor._apply_smoothing():
      optional medfilt → whittaker (asls, p=0.9) or savgol

    Only called when ``smoothing_params`` is not None and has 'enabled': True.
    """
    from scipy.signal import medfilt, savgol_filter

    y = signal.copy()

    # Optional median pre-filter (spike removal)
    if smoothing_params.get('median_enabled', False):
        kernel = smoothing_params.get('median_kernel', 5)
        if kernel % 2 == 0:
            kernel += 1
        if kernel >= len(y):
            kernel = max(3, len(y) - 1)
            if kernel % 2 == 0:
                kernel -= 1
        y = medfilt(y, kernel_size=kernel)

    method = smoothing_params.get('method', 'whittaker')

    if method == 'savgol':
        window = smoothing_params.get('savgol_window', 3)
        polyorder = smoothing_params.get('savgol_polyorder', 1)
        if window % 2 == 0:
            window += 1
        if window >= len(y):
            window = max(3, len(y) - 1)
            if window % 2 == 0:
                window -= 1
        if polyorder >= window:
            polyorder = window - 1
        y = savgol_filter(y, window_length=window, polyorder=polyorder)
    else:
        try:
            from pybaselines import Baseline
            lam = smoothing_params.get('lambda', 1e-1)
            diff_order = smoothing_params.get('diff_order', 1)
            y, _ = Baseline().asls(y, lam=lam, p=0.9, diff_order=diff_order)
        except ImportError:
            pass  # pybaselines not available — leave signal as-is

    return y


def run_deconvolution_pipeline(time, corrected_signal, *,
                                splitting_method='emg',
                                heatmap_threshold=0.44,
                                heatmap_distance=10,
                                weights_path=None,
                                min_prominence=0.07,
                                smoothing_params=None,
                                pre_fit_signal_threshold=0.057,
                                dedup_sigma_factor=0.83,
                                dedup_rt_tolerance=0.005,
                                min_area_frac=0.12,
                                mu_bound_factor=1.5,
                                fat_threshold_frac=0.5,
                                valley_threshold_frac=0.5):
    """
    Full deconvolution pipeline: chunk → U-Net inference → EMG fit or geometric split.

    Parameters
    ----------
    time : 1-D array
        Retention time axis.
    corrected_signal : 1-D array
        Baseline-corrected signal.
    splitting_method : str
        'emg' for EMG curve fitting (default), 'geometric' for geometric
        area splitting with boundary detection.
    heatmap_threshold : float
        Minimum heatmap probability to accept an apex.
    heatmap_distance : int
        Minimum pixel separation between detected apexes.
    weights_path : str or None
        Path to U-Net weights (None = default).
    min_prominence : float
        Minimum EMG component peak height to keep.  If < 1, treated as a
        fraction of the full signal range (e.g. 0.01 = 1% of range).
        0 = no filtering (default, backwards-compatible).
    smoothing_params : dict or None
        Production smoothing settings (same structure as
        ``params['smoothing']`` from the UI).  When provided AND
        ``smoothing_params['enabled']`` is True, a smoothed copy of the
        signal is used *only* for U-Net normalization; chunking and EMG
        fitting always use the original ``corrected_signal``.
    pre_fit_signal_threshold : float
        Minimum signal value at a U-Net-detected apex (as a fraction of
        the global signal range) to attempt EMG fitting.  Apexes where
        the corrected signal is below this fraction are discarded before
        fitting, saving computation.  0 = no filtering.
    dedup_sigma_factor : float
        Post-fit deduplication (EMG mode): merge components whose RTs are
        within this many sigma of each other, keeping the one with larger
        area.  0 = no deduplication.
    dedup_rt_tolerance : float
        Post-fit deduplication (geometric mode): merge components whose
        RTs are within this many minutes, keeping the one with larger area.
        0 = no deduplication.
    min_area_frac : float
        Post-fit area filter: drop components with area less than this
        fraction of the median component area.  0 = no filtering.

    Returns
    -------
    DeconvolutionResult
    """
    import time as _time
    from logic.chunker import chunk_chromatogram, interpolate_chunk

    _require_torch()
    model = load_model(weights_path)

    time_arr = np.asarray(time, dtype=float)
    corrected_signal = np.asarray(corrected_signal, dtype=float)
    global_signal_range = float(np.max(corrected_signal) - np.min(corrected_signal))

    # Optionally smooth signal for U-Net — chunking/EMG use the original
    if smoothing_params and smoothing_params.get('enabled', False):
        signal_for_unet = _apply_smoothing_for_unet(corrected_signal, smoothing_params)
    else:
        signal_for_unet = corrected_signal

    # Step 1: Chunk the chromatogram (always on original signal)
    t0 = _time.perf_counter()
    chunks = chunk_chromatogram(time_arr, corrected_signal)
    if not chunks:
        return DeconvolutionResult(components=[], chunks=[])
    print(f"  Chunking: {len(chunks)} chunks in {_time.perf_counter()-t0:.3f}s")

    # Step 2: Batch U-Net inference — all chunks in one forward pass
    # Interpolation uses signal_for_unet so the model sees smoothed data
    t0 = _time.perf_counter()
    interp_data = []
    batch_norms = []
    for chunk in chunks:
        t_interp, s_interp, s_norm, scale, offset = interpolate_chunk(
            time_arr, signal_for_unet, chunk
        )
        interp_data.append((t_interp, s_interp, s_norm, scale, offset))
        batch_norms.append(s_norm)

    batch_tensor = torch.tensor(
        np.array(batch_norms), dtype=torch.float32
    ).unsqueeze(1)  # (N, 1, 256)
    with torch.no_grad():
        raw_output = model(batch_tensor).cpu().numpy()  # (N, C, 256)
    # Channel 0 is always the apex heatmap
    all_heatmaps = raw_output[:, 0, :]  # (N, 256)
    # Check for multi-channel model (3-ch: apex, sigma, tau)
    has_param_channels = raw_output.shape[1] >= 3
    if has_param_channels:
        all_sigma_maps = raw_output[:, 1, :]  # (N, 256)
        all_tau_maps = raw_output[:, 2, :]    # (N, 256)
    else:
        all_sigma_maps = None
        all_tau_maps = None
    print(f"  U-Net inference: {len(chunks)} chunks ({raw_output.shape[1]}-ch) "
          f"in {_time.perf_counter()-t0:.3f}s")

    # Step 3: Extract apexes per chunk (shared by both splitting methods)
    t0 = _time.perf_counter()
    per_chunk_apexes = []  # list of (apex_pixels, apex_rts, predicted_params)

    for ci, chunk in enumerate(chunks):
        t_interp = interp_data[ci][0]
        heatmap = all_heatmaps[ci]

        # Scale heatmap_distance so minimum apex separation is ≥ 0.03 min.
        chunk_width = t_interp[-1] - t_interp[0]
        min_sep_time = 0.03  # min
        min_distance_pixels = max(
            heatmap_distance,
            int(np.ceil(min_sep_time / chunk_width * WINDOW_LENGTH))
        )

        apex_pixels, _ = _extract_apices(
            heatmap, height=heatmap_threshold, distance=min_distance_pixels
        )
        if len(apex_pixels) == 0:
            per_chunk_apexes.append(None)
            continue

        apex_rts = [float(t_interp[int(p)]) for p in apex_pixels]

        # Pre-fit signal filter: discard apexes in noise regions
        if pre_fit_signal_threshold > 0 and global_signal_range > 0:
            si_chunk, ei_chunk = chunk.start_index, chunk.end_index + 1
            s_chunk_raw = corrected_signal[si_chunk:ei_chunk]
            keep = []
            for ai, p in enumerate(apex_pixels):
                frac = int(p) / (WINDOW_LENGTH - 1)
                orig_idx = int(round(frac * (len(s_chunk_raw) - 1)))
                orig_idx = max(0, min(orig_idx, len(s_chunk_raw) - 1))
                signal_at_apex = s_chunk_raw[orig_idx]
                if signal_at_apex >= pre_fit_signal_threshold * global_signal_range:
                    keep.append(ai)
            if len(keep) == 0:
                per_chunk_apexes.append(None)
                continue
            if len(keep) < len(apex_pixels):
                apex_pixels = [apex_pixels[i] for i in keep]
                apex_rts = [apex_rts[i] for i in keep]

        # Extract predicted sigma/tau from multi-channel model
        predicted_params = None
        if has_param_channels and len(apex_pixels) > 0:
            predicted_params = []
            for p in apex_pixels:
                p_int = int(p)
                pred_sigma = float(all_sigma_maps[ci][p_int]) * chunk_width
                pred_tau = float(all_tau_maps[ci][p_int]) * chunk_width
                predicted_params.append({
                    'sigma': max(pred_sigma, 1e-4),
                    'tau': max(pred_tau, 1e-4),
                })

        per_chunk_apexes.append((apex_pixels, apex_rts, predicted_params))

    print(f"  Apex extraction: {sum(1 for a in per_chunk_apexes if a is not None)} "
          f"chunks with apexes in {_time.perf_counter()-t0:.3f}s")

    # Step 4: Split into components (EMG fitting or geometric splitting)
    t0 = _time.perf_counter()
    all_components = []

    if splitting_method == 'geometric':
        # ── Geometric splitting path ──
        for ci, chunk in enumerate(chunks):
            if per_chunk_apexes[ci] is None:
                continue
            _, apex_rts, _ = per_chunk_apexes[ci]
            si, ei = chunk.start_index, chunk.end_index + 1
            t_chunk = time_arr[si:ei]
            s_chunk = corrected_signal[si:ei]
            if len(t_chunk) < 4:
                continue

            geo_comps = _geometric_split_chunk(
                t_chunk, s_chunk, apex_rts, chunk_start_index=si,
                chunk_index=ci, time_full=time_arr,
                valley_threshold_frac=valley_threshold_frac,
            )
            all_components.extend(geo_comps)

        print(f"  Geometric splitting: {len(all_components)} components in "
              f"{_time.perf_counter()-t0:.3f}s")

    else:
        # ── EMG fitting path (default) ──
        _PIPELINE_BUDGET = 10.0
        skipped_chunks = 0

        for ci, chunk in enumerate(chunks):
            if _time.perf_counter() - t0 > _PIPELINE_BUDGET:
                skipped_chunks = len(chunks) - ci
                print(f"  EMG fitting: budget exceeded, skipping {skipped_chunks} remaining chunks")
                break

            if per_chunk_apexes[ci] is None:
                continue
            apex_pixels, apex_rts, predicted_params = per_chunk_apexes[ci]

            si, ei = chunk.start_index, chunk.end_index + 1
            t_chunk = time_arr[si:ei]
            s_chunk = corrected_signal[si:ei]
            if len(t_chunk) < 4:
                continue

            fit_groups = _group_nearby_apexes(apex_rts, t_chunk, s_chunk,
                                                valley_threshold_frac=valley_threshold_frac)

            for sub_rts, sub_si, sub_ei in fit_groups:
                sub_t = t_chunk[sub_si:sub_ei + 1]
                sub_s = s_chunk[sub_si:sub_ei + 1]
                if len(sub_t) < 4:
                    continue

                sub_params = None
                if predicted_params is not None:
                    sub_params = []
                    for srt in sub_rts:
                        best_idx = int(np.argmin([abs(srt - art) for art in apex_rts]))
                        sub_params.append(predicted_params[best_idx])

                emg_fits, _ = _deconvolve_chunk(sub_t, sub_s, sub_rts,
                                                predicted_params=sub_params,
                                                mu_bound_factor=mu_bound_factor,
                                                fat_threshold_frac=fat_threshold_frac)
                if not emg_fits:
                    continue

                for fit in emg_fits:
                    y_curve = _single_emg(
                        t_chunk, fit['amplitude'], fit['retention_time'],
                        fit['sigma'], fit['tau']
                    )
                    peak_height = float(np.max(y_curve))
                    if fit['area'] < 1e-10 or peak_height < 1e-10:
                        continue
                    all_components.append(EMGComponent(
                        retention_time=fit['retention_time'],
                        area=fit['area'],
                        sigma=fit['sigma'],
                        tau=fit['tau'],
                        peak_height=peak_height,
                        chunk_index=ci,
                        t_curve=t_chunk.copy(),
                        y_curve=y_curve,
                    ))

        print(f"  EMG fitting: {len(all_components)} components in "
              f"{_time.perf_counter()-t0:.3f}s")

    # Step 5: Post-fit boundary resolution (EMG only — geometric has exact bounds)
    if splitting_method == 'emg':
        t0 = _time.perf_counter()
        all_components = _resolve_boundary_truncations(
            all_components, chunks, time_arr, corrected_signal,
            interp_data, all_heatmaps, heatmap_threshold, heatmap_distance,
            mu_bound_factor=mu_bound_factor,
            fat_threshold_frac=fat_threshold_frac,
            valley_threshold_frac=valley_threshold_frac,
        )
        print(f"  Boundary resolution: {len(all_components)} components in "
              f"{_time.perf_counter()-t0:.3f}s")

    # Step 6: Filter spurious low-height components
    if min_prominence > 0 and all_components:
        signal_range = float(np.max(corrected_signal) - np.min(corrected_signal))
        abs_threshold = (min_prominence * signal_range
                         if min_prominence < 1 else min_prominence)
        n_before = len(all_components)
        all_components = [c for c in all_components
                          if c.peak_height >= abs_threshold]
        n_removed = n_before - len(all_components)
        if n_removed:
            print(f"  Prominence filter: removed {n_removed} spurious components "
                  f"(threshold={abs_threshold:.1f})")

    # Step 7: Merge nearby components into single peaks
    # When two predicted apexes are too close, they're likely the same peak
    # split by noise — merge their boundaries and re-integrate as one.
    if len(all_components) > 1:
        if splitting_method == 'geometric' and dedup_rt_tolerance > 0:
            all_components = sorted(all_components, key=lambda c: c.retention_time)
            kept = [all_components[0]]
            for c in all_components[1:]:
                prev = kept[-1]
                if abs(c.retention_time - prev.retention_time) < dedup_rt_tolerance:
                    # Merge: extend boundaries, pick apex with higher signal,
                    # re-integrate the combined region
                    si = min(prev.start_index, c.start_index)
                    ei = max(prev.end_index, c.end_index)
                    seg_s = np.maximum(corrected_signal[si:ei + 1], 0)
                    seg_t = time_arr[si:ei + 1]
                    merged_area = float(np.trapz(seg_s, seg_t)) if len(seg_t) >= 2 else prev.area + c.area
                    # Keep apex with higher peak_height
                    if c.peak_height > prev.peak_height:
                        apex = c
                    else:
                        apex = prev
                    kept[-1] = DeconvComponent(
                        retention_time=apex.retention_time,
                        area=merged_area,
                        peak_height=apex.peak_height,
                        chunk_index=apex.chunk_index,
                        start_time=float(time_arr[si]),
                        end_time=float(time_arr[ei]),
                        start_index=si,
                        end_index=ei,
                        apex_index=apex.apex_index,
                        split_type_left=prev.split_type_left,
                        split_type_right=c.split_type_right,
                    )
                else:
                    kept.append(c)
            n_deduped = len(all_components) - len(kept)
            if n_deduped:
                print(f"  Deduplication: merged {n_deduped} components "
                      f"(within {dedup_rt_tolerance:.3f} min)")
            all_components = kept
        elif splitting_method == 'emg' and dedup_sigma_factor > 0:
            all_components = sorted(all_components, key=lambda c: c.retention_time)
            kept = [all_components[0]]
            for c in all_components[1:]:
                prev = kept[-1]
                separation = abs(c.retention_time - prev.retention_time)
                merge_thresh = dedup_sigma_factor * max(c.sigma, prev.sigma)
                if separation < merge_thresh:
                    # Merge: keep dominant peak's shape, combine areas,
                    # extend curve to cover both
                    if c.peak_height > prev.peak_height:
                        dominant, minor = c, prev
                    else:
                        dominant, minor = prev, c
                    merged_area = dominant.area + minor.area
                    # Rebuild curve: sum both EMG curves
                    merged_y = dominant.y_curve + _single_emg(
                        dominant.t_curve, minor.area, minor.retention_time,
                        minor.sigma, minor.tau,
                    )
                    kept[-1] = EMGComponent(
                        retention_time=dominant.retention_time,
                        area=merged_area,
                        sigma=dominant.sigma,
                        tau=dominant.tau,
                        peak_height=float(np.max(merged_y)),
                        chunk_index=dominant.chunk_index,
                        t_curve=dominant.t_curve,
                        y_curve=merged_y,
                    )
                else:
                    kept.append(c)
            n_deduped = len(all_components) - len(kept)
            if n_deduped:
                print(f"  Deduplication: merged {n_deduped} components "
                      f"(within {dedup_sigma_factor}σ)")
            all_components = kept

    # Step 8: Drop components with area < X% of median area
    if min_area_frac > 0 and all_components:
        areas = [c.area for c in all_components]
        median_area = float(np.median(areas))
        area_threshold = min_area_frac * median_area
        n_before = len(all_components)
        all_components = [c for c in all_components if c.area >= area_threshold]
        n_removed = n_before - len(all_components)
        if n_removed:
            print(f"  Area filter: removed {n_removed} components "
                  f"(threshold={area_threshold:.2f}, {min_area_frac:.0%} of median)")

    # Step 9: Proportional area rescaling (EMG only — geometric areas are already correct)
    if splitting_method == 'emg' and len(all_components) > 1:
        t0 = _time.perf_counter()
        curves = np.zeros((len(all_components), len(time_arr)))
        for i, comp in enumerate(all_components):
            curves[i] = _single_emg(
                time_arr, comp.area, comp.retention_time,
                comp.sigma, comp.tau,
            )
        total_fit = curves.sum(axis=0)
        signal_pos = np.maximum(corrected_signal, 0)

        for i, comp in enumerate(all_components):
            mask = total_fit > 1e-12
            share = np.zeros_like(time_arr)
            share[mask] = curves[i][mask] / total_fit[mask] * signal_pos[mask]
            comp.area = float(np.trapz(share, time_arr))

        print(f"  Area rescaling: {len(all_components)} components in "
              f"{_time.perf_counter()-t0:.3f}s")

    return DeconvolutionResult(components=all_components, chunks=chunks)


# =========================================================================== #
#  Geometric splitting helpers
# =========================================================================== #

def _resolve_apex_position(s_chunk, predicted_idx, window=5):
    """
    Snap a predicted apex to the nearest local maximum, or keep as shoulder.

    Parameters
    ----------
    s_chunk : 1-D array — signal within the chunk
    predicted_idx : int — predicted apex index in s_chunk
    window : int — search radius (±window points)

    Returns
    -------
    (snapped_idx, is_shoulder) : (int, bool)
    """
    lo = max(0, predicted_idx - window)
    hi = min(len(s_chunk), predicted_idx + window + 1)
    region = s_chunk[lo:hi]
    local_max_idx = lo + int(np.argmax(region))

    # Check if the local max is a true local maximum (higher than both neighbors)
    if 0 < local_max_idx < len(s_chunk) - 1:
        if (s_chunk[local_max_idx] >= s_chunk[local_max_idx - 1] and
                s_chunk[local_max_idx] >= s_chunk[local_max_idx + 1]):
            return local_max_idx, False

    # Edge case: at array boundary, still accept if it's the region max
    if s_chunk[local_max_idx] > s_chunk[predicted_idx] * 1.01:
        return local_max_idx, False

    # No clear local max — this is a shoulder, keep the predicted position
    return predicted_idx, True


def _find_split_boundary(t_chunk, s_chunk, idx_a, idx_b,
                         valley_threshold_frac=0.5):
    """
    Find the best split point between two predicted apexes.

    Three-tier logic:
    1. Valley: signal minimum < 50% of shorter peak height
    2. Inflection: d2y zero-crossing with savgol smoothing
    3. Midpoint: (rt_a + rt_b) / 2

    Parameters
    ----------
    t_chunk, s_chunk : 1-D arrays — chunk time and signal
    idx_a, idx_b : int — indices of two adjacent apexes (idx_a < idx_b)

    Returns
    -------
    (boundary_idx, split_type) : (int, str)
    """
    from scipy.signal import savgol_filter

    if idx_a >= idx_b:
        idx_a, idx_b = idx_b, idx_a

    # Region between the two apexes
    region_s = s_chunk[idx_a:idx_b + 1]
    if len(region_s) < 3:
        mid = (idx_a + idx_b) // 2
        return mid, 'midpoint'

    # Tier 1: Valley detection
    valley_local_idx = int(np.argmin(region_s))
    valley_idx = idx_a + valley_local_idx
    valley_val = region_s[valley_local_idx]
    shorter_peak = min(s_chunk[idx_a], s_chunk[idx_b])

    if shorter_peak > 1e-12 and valley_val < valley_threshold_frac * shorter_peak:
        return valley_idx, 'valley'

    # Tier 2: Inflection point via d2y zero-crossing
    # Use heavy savgol smoothing on the inter-peak region to find curvature changes
    region_len = idx_b - idx_a + 1
    if region_len > 5:
        # Savgol window: use about half the region, odd number
        sg_window = min(region_len, max(5, region_len // 2))
        if sg_window % 2 == 0:
            sg_window += 1
        sg_window = min(sg_window, region_len)
        if sg_window % 2 == 0:
            sg_window -= 1

        if sg_window >= 5:
            smoothed = savgol_filter(s_chunk[idx_a:idx_b + 1],
                                     window_length=sg_window, polyorder=3)
            d2y = np.gradient(np.gradient(smoothed))

            # Find zero-crossings
            zero_crossings = []
            for k in range(len(d2y) - 1):
                if d2y[k] * d2y[k + 1] < 0:
                    zero_crossings.append(idx_a + k)

            if zero_crossings:
                # Use the zero-crossing closest to the midpoint between apexes
                midpoint = (idx_a + idx_b) / 2
                best_zc = min(zero_crossings, key=lambda z: abs(z - midpoint))
                return best_zc, 'inflection'

    # Tier 3: Midpoint fallback
    mid = (idx_a + idx_b) // 2
    return mid, 'midpoint'


def _geometric_split_chunk(t_chunk, s_chunk, apex_rts, chunk_start_index,
                            chunk_index, time_full,
                            valley_threshold_frac=0.5):
    """
    Split a chunk into components using geometric boundaries.

    Parameters
    ----------
    t_chunk, s_chunk : 1-D arrays — chunk time and signal
    apex_rts : list of float — predicted apex retention times
    chunk_start_index : int — index of t_chunk[0] in the full time array
    chunk_index : int — chunk index for labeling
    time_full : 1-D array — full time axis (for index mapping)

    Returns
    -------
    list of DeconvComponent
    """
    if len(apex_rts) == 0:
        return []

    sorted_rts = sorted(apex_rts)

    # Resolve apex positions: snap to local max or keep as shoulder
    apex_indices = []
    for rt in sorted_rts:
        predicted_idx = int(np.argmin(np.abs(t_chunk - rt)))
        snapped_idx, _ = _resolve_apex_position(s_chunk, predicted_idx, window=5)
        apex_indices.append(snapped_idx)

    # Find boundaries between adjacent apexes
    n_apexes = len(apex_indices)
    boundaries = []  # list of (boundary_idx, split_type)

    for i in range(n_apexes - 1):
        b_idx, b_type = _find_split_boundary(
            t_chunk, s_chunk, apex_indices[i], apex_indices[i + 1],
            valley_threshold_frac=valley_threshold_frac,
        )
        boundaries.append((b_idx, b_type))

    # Build component segments
    components = []
    for i in range(n_apexes):
        # Left boundary
        if i == 0:
            left_idx = 0
            left_type = 'chunk'
        else:
            left_idx = boundaries[i - 1][0]
            left_type = boundaries[i - 1][1]

        # Right boundary
        if i == n_apexes - 1:
            right_idx = len(t_chunk) - 1
            right_type = 'chunk'
        else:
            right_idx = boundaries[i][0]
            right_type = boundaries[i][1]

        # Ensure valid range
        if right_idx <= left_idx:
            continue

        # Integrate the segment
        seg_s = np.maximum(s_chunk[left_idx:right_idx + 1], 0)
        seg_t = t_chunk[left_idx:right_idx + 1]
        if len(seg_t) < 2:
            continue

        area = float(np.trapz(seg_s, seg_t))
        if area < 1e-10:
            continue

        apex_idx_chunk = apex_indices[i]
        peak_height = float(s_chunk[apex_idx_chunk])

        # Map chunk-local indices to full time array indices
        full_start = chunk_start_index + left_idx
        full_end = chunk_start_index + right_idx
        full_apex = chunk_start_index + apex_idx_chunk

        components.append(DeconvComponent(
            retention_time=float(t_chunk[apex_idx_chunk]),
            area=area,
            peak_height=peak_height,
            chunk_index=chunk_index,
            start_time=float(t_chunk[left_idx]),
            end_time=float(t_chunk[right_idx]),
            start_index=full_start,
            end_index=full_end,
            apex_index=full_apex,
            split_type_left=left_type,
            split_type_right=right_type,
        ))

    return components


def _group_nearby_apexes(apex_rts, t_chunk, s_chunk,
                          max_per_group=_MAX_PEAKS_PER_FIT,
                          valley_threshold_frac=0.5):
    """
    Group apex RTs by proximity for joint fitting.

    Peaks whose valleys don't reach baseline (< 50% of the shorter peak)
    are fit together; well-separated peaks are fit independently.
    Groups exceeding max_per_group are split at the deepest valley.

    Returns list of (sub_rts, start_index, end_index) in t_chunk coords.
    """
    sorted_rts = sorted(apex_rts)
    if len(sorted_rts) <= 1:
        return [(sorted_rts, 0, len(t_chunk) - 1)]

    # Determine which consecutive apexes need joint fitting
    # by checking if the valley between them is shallow (overlapping peaks)
    merge_flags = []  # True = merge apex[i] with apex[i+1]
    for i in range(len(sorted_rts) - 1):
        rt_a, rt_b = sorted_rts[i], sorted_rts[i + 1]
        idx_a = int(np.argmin(np.abs(t_chunk - rt_a)))
        idx_b = int(np.argmin(np.abs(t_chunk - rt_b)))
        valley_min = float(np.min(s_chunk[idx_a:idx_b + 1]))
        shorter_peak = min(s_chunk[idx_a], s_chunk[idx_b])
        # If valley doesn't drop below threshold of the shorter peak, they overlap
        merge_flags.append(valley_min > shorter_peak * valley_threshold_frac)

    # Build groups of connected peaks
    groups = [[sorted_rts[0]]]
    for i, merge in enumerate(merge_flags):
        if merge:
            groups[-1].append(sorted_rts[i + 1])
        else:
            groups.append([sorted_rts[i + 1]])

    # Split any group that exceeds max_per_group
    final_groups = []
    for group in groups:
        while len(group) > max_per_group:
            final_groups.append(group[:max_per_group])
            group = group[max_per_group:]
        final_groups.append(group)

    # Convert to (sub_rts, si, ei) with time sub-regions + padding
    result = []
    for i, group in enumerate(final_groups):
        # Time boundaries: midpoint to neighboring groups
        if i == 0:
            t_start = t_chunk[0]
        else:
            t_start = (final_groups[i - 1][-1] + group[0]) / 2

        if i == len(final_groups) - 1:
            t_end = t_chunk[-1]
        else:
            t_end = (group[-1] + final_groups[i + 1][0]) / 2

        si = max(0, int(np.searchsorted(t_chunk, t_start, side='left')))
        ei = min(len(t_chunk) - 1,
                 int(np.searchsorted(t_chunk, t_end, side='right') - 1))

        # Pad for EMG tails
        pad = max(1, (ei - si) // 6)
        si = max(0, si - pad)
        ei = min(len(t_chunk) - 1, ei + pad)

        result.append((group, si, ei))

    return result


_FIT_MAX_POINTS = 60  # downsample data fed to curve_fit for speed


def _deconvolve_chunk(t_chunk, s_chunk, apex_rts, _max_retries=2,
                      predicted_params=None,
                      mu_bound_factor=1.5,
                      fat_threshold_frac=0.5):
    """
    Fit a multi-EMG model to a chunk with tight mu bounds (±1.5σ of seed).

    On convergence failure, drops the lowest-signal seed and retries
    (up to ``_max_retries`` times).

    After a successful fit, any component whose FWHM exceeds a threshold
    (half the fit window) is flagged as "fat".  When fat components exist,
    we refit without the fattest one — if the refit residual is comparable
    (within 2×), we accept the leaner model.  This catches the case where
    the U-Net predicted too many (or too few) peaks and the optimizer
    compensated by inflating widths.

    Data is downsampled to _FIT_MAX_POINTS before fitting for speed;
    the returned parameters are valid on the original time grid.

    Parameters
    ----------
    predicted_params : list of dict or None
        If provided (from 3-channel model), each dict has 'sigma' and 'tau'
        keys with denormalized predicted values. Used as informed initial
        guesses with tighter bounds instead of heuristics.

    Returns
    -------
    list of dicts, popt (or [], None on failure)
    """
    import time as _time

    if len(apex_rts) == 0:
        return [], None

    # Downsample for faster curve_fit — EMG is smooth, 60 pts is plenty
    if len(t_chunk) > _FIT_MAX_POINTS:
        indices = np.linspace(0, len(t_chunk) - 1, _FIT_MAX_POINTS, dtype=int)
        t_fit = t_chunk[indices]
        s_fit = s_chunk[indices]
    else:
        t_fit = t_chunk
        s_fit = s_chunk

    current_rts = sorted(apex_rts)
    # Sort predicted_params to match sorted RTs if provided
    if predicted_params is not None and len(predicted_params) == len(apex_rts):
        # Re-order predicted_params to match sorted order
        sorted_indices = sorted(range(len(apex_rts)),
                                key=lambda k: apex_rts[k])
        current_params = [predicted_params[k] for k in sorted_indices]
    else:
        current_params = predicted_params

    for retry in range(_max_retries + 1):
        n_peaks = len(current_rts)
        if n_peaks == 0:
            return [], None

        window_width = t_fit[-1] - t_fit[0]

        # Use predicted params if available, else fall back to heuristics
        use_predictions = (current_params is not None
                           and len(current_params) == n_peaks)

        if use_predictions:
            # Informed initial guesses from 3-channel model
            sig_max = window_width / 2
            tau_max = window_width / 2
        else:
            sig_guess = window_width / (n_peaks * 4)
            tau_guess = sig_guess * 0.3
            sig_max = window_width / 2
            tau_max = window_width / 2

        p0, lo, hi = [], [], []
        for i, rt in enumerate(current_rts):
            idx = int(np.argmin(np.abs(t_fit - rt)))

            if use_predictions:
                # Heuristic fallback values
                heuristic_sig = window_width / (n_peaks * 4)
                heuristic_tau = heuristic_sig * 0.3

                sig_guess_i = current_params[i]['sigma']
                tau_guess_i = current_params[i]['tau']

                # If predicted value is unreasonably small, fall back to heuristic
                min_reasonable = window_width * 0.001  # 0.1% of window
                if sig_guess_i < min_reasonable:
                    sig_guess_i = heuristic_sig
                if tau_guess_i < min_reasonable:
                    tau_guess_i = heuristic_tau

                # Tighter bounds: [0.3x, 3x] of predicted value
                sig_lo = max(1e-4, 0.3 * sig_guess_i)
                sig_hi = min(sig_max, 3.0 * sig_guess_i)
                tau_lo = max(1e-4, 0.3 * tau_guess_i)
                tau_hi = min(tau_max, 3.0 * tau_guess_i)

                # Safety: ensure lo < hi (can fail if pred is tiny)
                if sig_lo >= sig_hi:
                    sig_lo = 1e-4
                    sig_hi = sig_max
                    sig_guess_i = heuristic_sig
                if tau_lo >= tau_hi:
                    tau_lo = 1e-4
                    tau_hi = tau_max
                    tau_guess_i = heuristic_tau
            else:
                sig_guess_i = sig_guess
                tau_guess_i = tau_guess
                sig_lo = 1e-4
                sig_hi = sig_max
                tau_lo = 1e-4
                tau_hi = tau_max

            amp_guess = max(s_fit[idx], 1e-6) * sig_guess_i * 5.0

            # Tight mu bounds: ±N × σ_guess of the U-Net seed RT,
            # clamped to the fit window
            mu_lo = max(t_fit[0], rt - mu_bound_factor * sig_guess_i)
            mu_hi = min(t_fit[-1], rt + mu_bound_factor * sig_guess_i)

            # Clamp initial guesses to be strictly within bounds
            sig_guess_i = float(np.clip(sig_guess_i, sig_lo * 1.01, sig_hi * 0.99))
            tau_guess_i = float(np.clip(tau_guess_i, tau_lo * 1.01, tau_hi * 0.99))
            rt_clamped = float(np.clip(rt, mu_lo, mu_hi))

            p0.extend([amp_guess, rt_clamped, sig_guess_i, tau_guess_i])
            lo.extend([0, mu_lo, sig_lo, tau_lo])
            hi.extend([np.inf, mu_hi, sig_hi, tau_hi])

        # Wall-clock abort: wrap _multi_emg to bail if too slow
        _fit_start = _time.perf_counter()
        _call_count = [0]
        _FIT_TIMEOUT = 0.5  # seconds per fit call

        def _timed_multi_emg(t, *params):
            _call_count[0] += 1
            if _call_count[0] % 50 == 0:
                if _time.perf_counter() - _fit_start > _FIT_TIMEOUT:
                    raise _FitTimeout()
            return _multi_emg(t, *params)

        try:
            popt, _ = curve_fit(
                _timed_multi_emg, t_fit, s_fit,
                p0=p0, bounds=(lo, hi), maxfev=10000
            )
            # Success — build results and check for fat components
            results = []
            for i in range(n_peaks):
                amp, mu, sigma, tau = popt[i * 4:(i + 1) * 4]
                results.append({
                    'retention_time': mu,
                    'area': amp,
                    'sigma': sigma,
                    'tau': tau,
                    'amplitude': amp,
                })

            # Post-fit fat-component check: if any component's FWHM
            # exceeds half the window, try refitting without it
            if n_peaks > 1:
                fwhms = [2.355 * r['sigma'] + r['tau'] for r in results]
                fat_threshold = window_width * fat_threshold_frac
                fattest_idx = int(np.argmax(fwhms))

                if fwhms[fattest_idx] > fat_threshold:
                    # Compute residual of current fit
                    rss_n = float(np.sum((s_fit - _multi_emg(t_fit, *popt)) ** 2))

                    # Try without the fattest component
                    reduced_rts = current_rts[:fattest_idx] + current_rts[fattest_idx + 1:]
                    reduced_results, reduced_popt = _deconvolve_chunk(
                        t_chunk, s_chunk, reduced_rts, _max_retries=0,
                        mu_bound_factor=mu_bound_factor,
                        fat_threshold_frac=fat_threshold_frac,
                    )
                    if reduced_results and reduced_popt is not None:
                        rss_reduced = float(np.sum(
                            (s_fit - _multi_emg(t_fit, *reduced_popt)) ** 2
                        ))
                        # Accept if residual doesn't more than double
                        if rss_reduced < rss_n * 2.0:
                            return reduced_results, reduced_popt

            return results, popt

        except (RuntimeError, ValueError, _FitTimeout):
            if retry >= _max_retries or n_peaks <= 1:
                return [], None
            # Drop the seed with the lowest signal value and retry
            signal_vals = []
            for rt in current_rts:
                idx = int(np.argmin(np.abs(t_fit - rt)))
                signal_vals.append(s_fit[idx])
            weakest = int(np.argmin(signal_vals))
            current_rts = current_rts[:weakest] + current_rts[weakest + 1:]
            if current_params is not None and len(current_params) > weakest:
                current_params = current_params[:weakest] + current_params[weakest + 1:]

    return [], None


class _FitTimeout(Exception):
    """Raised to abort a curve_fit call that exceeds the wall-clock budget."""
    pass


def _resolve_boundary_truncations(components, chunks, time_arr, corrected_signal,
                                   interp_data, all_heatmaps,
                                   heatmap_threshold, heatmap_distance,
                                   mu_bound_factor=1.5,
                                   fat_threshold_frac=0.5,
                                   valley_threshold_frac=0.5):
    """
    Detect EMG components truncated at chunk boundaries and re-fit them
    using merged super-chunks.

    A component is "truncated" if its mass boundary (RT ± (2σ + τ)) extends
    past its chunk boundary by > 20% of its total EMG width.

    One round of merging only (no cascading).
    """
    if not components or len(chunks) < 2:
        return components

    # Build a set of chunk-pair indices that need merging
    merge_pairs = set()  # (ci_left, ci_right) with ci_left < ci_right

    for comp in components:
        ci = comp.chunk_index
        if ci < 0 or ci >= len(chunks):
            continue
        chunk = chunks[ci]
        chunk_t_start = time_arr[chunk.start_index]
        chunk_t_end = time_arr[min(chunk.end_index, len(time_arr) - 1)]

        mass_radius = 2 * comp.sigma + comp.tau
        total_width = 2 * mass_radius
        if total_width < 1e-12:
            continue

        left_overflow = chunk_t_start - (comp.retention_time - mass_radius)
        right_overflow = (comp.retention_time + mass_radius) - chunk_t_end

        # Check left boundary
        if left_overflow > 0.20 * total_width and ci > 0:
            merge_pairs.add((ci - 1, ci))
        # Check right boundary
        if right_overflow > 0.20 * total_width and ci < len(chunks) - 1:
            merge_pairs.add((ci, ci + 1))

    if not merge_pairs:
        return components

    # For each merge pair, combine chunks and re-fit
    # Track which chunk indices are involved in merges
    merged_chunk_indices = set()
    for left, right in merge_pairs:
        merged_chunk_indices.add(left)
        merged_chunk_indices.add(right)

    # Collect all apex seeds from both chunks in each merge pair
    new_components = []
    refitted_chunk_indices = set()

    for ci_left, ci_right in sorted(merge_pairs):
        # Skip if already refitted in a prior merge
        if ci_left in refitted_chunk_indices and ci_right in refitted_chunk_indices:
            continue

        chunk_l = chunks[ci_left]
        chunk_r = chunks[ci_right]

        # Merged time/signal region
        si = chunk_l.start_index
        ei = chunk_r.end_index + 1
        t_merged = time_arr[si:ei]
        s_merged = corrected_signal[si:ei]
        if len(t_merged) < 4:
            continue

        # Collect apex RTs from both chunks' heatmaps
        apex_rts = []
        for ci in (ci_left, ci_right):
            t_interp_ci = interp_data[ci][0]
            heatmap_ci = all_heatmaps[ci]
            chunk_width = t_interp_ci[-1] - t_interp_ci[0]
            min_sep_time = 0.03
            min_dist_px = max(
                heatmap_distance,
                int(np.ceil(min_sep_time / chunk_width * WINDOW_LENGTH))
            )
            apex_px, _ = _extract_apices(
                heatmap_ci, height=heatmap_threshold, distance=min_dist_px
            )
            for p in apex_px:
                apex_rts.append(float(t_interp_ci[int(p)]))

        if not apex_rts:
            continue

        # Deduplicate apex RTs that are very close (within 0.02 min)
        apex_rts.sort()
        deduped = [apex_rts[0]]
        for rt in apex_rts[1:]:
            if rt - deduped[-1] > 0.02:
                deduped.append(rt)
        apex_rts = deduped

        # Group and fit
        fit_groups = _group_nearby_apexes(apex_rts, t_merged, s_merged,
                                          valley_threshold_frac=valley_threshold_frac)
        for sub_rts, sub_si, sub_ei in fit_groups:
            sub_t = t_merged[sub_si:sub_ei + 1]
            sub_s = s_merged[sub_si:sub_ei + 1]
            if len(sub_t) < 4:
                continue
            emg_fits, _ = _deconvolve_chunk(sub_t, sub_s, sub_rts,
                                            mu_bound_factor=mu_bound_factor,
                                            fat_threshold_frac=fat_threshold_frac)
            if not emg_fits:
                continue
            for fit in emg_fits:
                y_curve = _single_emg(
                    t_merged, fit['amplitude'], fit['retention_time'],
                    fit['sigma'], fit['tau']
                )
                peak_height = float(np.max(y_curve))
                if fit['area'] < 1e-10 or peak_height < 1e-10:
                    continue
                new_components.append(EMGComponent(
                    retention_time=fit['retention_time'],
                    area=fit['area'],
                    sigma=fit['sigma'],
                    tau=fit['tau'],
                    peak_height=peak_height,
                    chunk_index=ci_left,  # assign to left chunk
                    t_curve=t_merged.copy(),
                    y_curve=y_curve,
                ))

        refitted_chunk_indices.add(ci_left)
        refitted_chunk_indices.add(ci_right)

    # Replace: keep components from non-merged chunks, use new fits for merged
    kept = [c for c in components if c.chunk_index not in refitted_chunk_indices]
    result = kept + new_components
    result.sort(key=lambda c: c.retention_time)
    return result


def integrate_emg_components(components, x, corrected_y, baseline_y, area_factor):
    """
    Convert filtered EMGComponent list → integration result dict
    compatible with Integrator.integrate() output.

    Parameters
    ----------
    components : list of EMGComponent
    x : 1-D array — full time axis
    corrected_y : 1-D array — baseline-corrected signal
    baseline_y : 1-D array — baseline
    area_factor : float — ChemStation area scaling factor

    Returns
    -------
    dict with keys: peaks, x_peaks, y_peaks, baseline_peaks,
                    retention_times, integrated_areas, integration_bounds, peaks_list
    """
    from logic.integration import Peak, Integrator

    peaks_list = []
    x_peaks = []
    y_peaks = []
    baseline_peaks = []
    ret_times = []
    integrated_areas = []
    integration_bounds = []

    for i, comp in enumerate(components):
        # Find integration bounds: where EMG > 0.1% of peak height
        threshold = comp.peak_height * 0.001
        above = comp.y_curve > threshold
        if not np.any(above):
            continue
        indices = np.where(above)[0]
        left_idx_in_curve = indices[0]
        right_idx_in_curve = indices[-1]

        t_start = comp.t_curve[left_idx_in_curve]
        t_end = comp.t_curve[right_idx_in_curve]

        # Map to full chromatogram indices
        start_index = int(np.searchsorted(x, t_start, side='left'))
        end_index = int(np.searchsorted(x, t_end, side='right') - 1)
        start_index = max(0, start_index)
        end_index = min(len(x) - 1, end_index)

        area = comp.area * area_factor

        width = t_end - t_start
        compound_id = Integrator.identify_compound(comp.retention_time)

        peak = Peak(
            compound_id=compound_id,
            peak_number=i + 1,
            retention_time=comp.retention_time,
            integrator='py',
            width=width,
            area=area,
            start_time=t_start,
            end_time=t_end,
            start_index=start_index,
            end_index=end_index,
        )

        # Use EMG curve values for the peak region (for shading)
        x_peak = comp.t_curve[left_idx_in_curve:right_idx_in_curve + 1]
        y_peak = comp.y_curve[left_idx_in_curve:right_idx_in_curve + 1]
        baseline_peak = np.interp(x_peak, x, baseline_y)

        peaks_list.append(peak)
        x_peaks.append(x_peak)
        y_peaks.append(y_peak)
        baseline_peaks.append(baseline_peak)
        ret_times.append(comp.retention_time)
        integrated_areas.append(area)
        integration_bounds.append((t_start, t_end))

    return {
        'peaks': peaks_list,
        'x_peaks': x_peaks,
        'y_peaks': y_peaks,
        'baseline_peaks': baseline_peaks,
        'retention_times': ret_times,
        'integrated_areas': integrated_areas,
        'integration_bounds': integration_bounds,
        'peaks_list': peaks_list,
    }


def integrate_deconv_components(components, x, corrected_y, baseline_y, area_factor):
    """
    Convert DeconvComponent list → integration result dict
    compatible with Integrator.integrate() output.

    Parameters
    ----------
    components : list of DeconvComponent
    x : 1-D array — full time axis
    corrected_y : 1-D array — baseline-corrected signal
    baseline_y : 1-D array — baseline
    area_factor : float — ChemStation area scaling factor

    Returns
    -------
    dict with keys: peaks, x_peaks, y_peaks, baseline_peaks,
                    retention_times, integrated_areas, integration_bounds, peaks_list
    """
    from logic.integration import Peak, Integrator

    peaks_list = []
    x_peaks = []
    y_peaks = []
    baseline_peaks = []
    ret_times = []
    integrated_areas = []
    integration_bounds = []

    for i, comp in enumerate(components):
        si = max(0, min(comp.start_index, len(x) - 1))
        ei = max(0, min(comp.end_index, len(x) - 1))
        if ei <= si:
            continue

        area = comp.area * area_factor
        width = comp.end_time - comp.start_time
        compound_id = Integrator.identify_compound(comp.retention_time)

        peak = Peak(
            compound_id=compound_id,
            peak_number=i + 1,
            retention_time=comp.retention_time,
            integrator='py',
            width=width,
            area=area,
            start_time=comp.start_time,
            end_time=comp.end_time,
            start_index=si,
            end_index=ei,
        )

        x_peak = x[si:ei + 1]
        y_peak = corrected_y[si:ei + 1]
        baseline_peak = baseline_y[si:ei + 1]

        peaks_list.append(peak)
        x_peaks.append(x_peak)
        y_peaks.append(y_peak)
        baseline_peaks.append(baseline_peak)
        ret_times.append(comp.retention_time)
        integrated_areas.append(area)
        integration_bounds.append((comp.start_time, comp.end_time))

    return {
        'peaks': peaks_list,
        'x_peaks': x_peaks,
        'y_peaks': y_peaks,
        'baseline_peaks': baseline_peaks,
        'retention_times': ret_times,
        'integrated_areas': integrated_areas,
        'integration_bounds': integration_bounds,
        'peaks_list': peaks_list,
    }
