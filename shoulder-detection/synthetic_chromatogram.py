"""
Synthetic GC chromatogram generator for testing the peak detection,
clustering, U-Net inference, and EMG deconvolution pipeline.

Produces full chromatograms with ground-truth peak metadata, compatible
with the processor output format used by logic/integration.py.

Usage:
    gen = SyntheticChromatogramGenerator(seed=42)
    result = gen.generate(num_peaks=12, merge_probability=0.4)

    # Processor-compatible arrays
    time = result['x']
    signal = result['original_y']
    baseline = result['baseline_y']
    corrected = result['corrected_y']

    # Ground truth
    for peak in result['true_peaks']:
        print(peak['rt'], peak['is_merged'], peak['cluster_id'])
"""

import numpy as np
from scipy.stats import exponnorm


class SyntheticChromatogramGenerator:
    """Generate realistic synthetic GC-FID/TCD chromatograms."""

    def __init__(self, seed=None):
        """
        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    # ----- EMG primitive --------------------------------------------------- #

    def _emg(self, t, amp, mu, sigma, tau):
        """Single Exponentially Modified Gaussian on time array *t*."""
        sigma = max(sigma, 1e-6)
        K = tau / sigma
        curve = amp * exponnorm.pdf(t, K, loc=mu, scale=sigma)
        return np.nan_to_num(curve)

    def _emg_apex(self, t, amp, mu, sigma, tau):
        """Return (apex_index, apex_rt) for an EMG curve."""
        curve = self._emg(t, amp, mu, sigma, tau)
        idx = np.argmax(curve)
        return idx, t[idx]

    # ----- Baseline generation --------------------------------------------- #

    def _make_baseline(self, t, baseline_type, amplitude):
        """Generate a baseline curve.

        Parameters
        ----------
        t : 1-D array
        baseline_type : str
            'flat', 'linear', 'quadratic', 'sine', or 'random'.
            'random' picks one at random.
        amplitude : float
            Rough magnitude of baseline drift relative to peak signals.
        """
        if baseline_type == 'random':
            baseline_type = self.rng.choice(['flat', 'linear', 'quadratic', 'sine'])

        t_norm = (t - t[0]) / (t[-1] - t[0])  # normalise to [0, 1]

        if baseline_type == 'flat':
            return np.full_like(t, amplitude * self.rng.uniform(0, 0.5))

        if baseline_type == 'linear':
            slope = self.rng.uniform(-1, 1)
            return amplitude * (slope * t_norm + self.rng.uniform(0, 0.3))

        if baseline_type == 'quadratic':
            a = self.rng.uniform(-2, 2)
            b = self.rng.uniform(-1, 1)
            return amplitude * (a * t_norm**2 + b * t_norm)

        if baseline_type == 'sine':
            freq = self.rng.uniform(0.5, 3.0)
            phase = self.rng.uniform(0, 2 * np.pi)
            return amplitude * 0.5 * np.sin(2 * np.pi * freq * t_norm + phase)

        return np.zeros_like(t)

    # ----- Noise generation ------------------------------------------------ #

    def _make_noise(self, t, signal_max, snr, noise_mode='realistic'):
        """
        Generate noise for a synthetic chromatogram.

        Parameters
        ----------
        t : 1-D array
        signal_max : float
        snr : float
        noise_mode : str
            'simple' — white noise + random-walk wander (original).
            'realistic' — adds colored (1/f) noise and chromatographic bumps.
        """
        noise_level = signal_max / snr
        white = self.rng.normal(0, noise_level, len(t))
        # baseline wander: cumulative random walk, scaled down
        wander = np.cumsum(self.rng.normal(0, noise_level * 0.03, len(t)))
        wander -= wander[0]

        if noise_mode == 'simple':
            return white + wander

        # --- Colored (1/f^α) noise ---
        n = len(t)
        white_for_color = self.rng.normal(0, noise_level, n)
        fft_vals = np.fft.rfft(white_for_color)
        freqs = np.fft.rfftfreq(n, d=(t[1] - t[0]) if n > 1 else 1.0)
        # Avoid division by zero at DC
        freqs[0] = 1.0
        alpha = self.rng.uniform(0.3, 1.0)
        fft_vals *= 1.0 / (freqs ** alpha)
        colored = np.fft.irfft(fft_vals, n=n)
        # Normalize to same RMS as white noise
        rms = np.sqrt(np.mean(colored ** 2))
        if rms > 1e-12:
            colored *= noise_level / rms

        # --- Medium-frequency sinusoidal ripple (flow/temperature fluctuations) ---
        # Period longer than white noise but shorter than baseline wander:
        # 3-8 sinusoids with periods ~50-500x the sample spacing
        dt = t[1] - t[0] if n > 1 else 1.0
        ripple = np.zeros(n)
        n_waves = self.rng.integers(3, 9)
        for _ in range(n_waves):
            period = self.rng.uniform(50, 500) * dt
            phase = self.rng.uniform(0, 2 * np.pi)
            amp = self.rng.uniform(0.05, 0.3) * noise_level
            ripple += amp * np.sin(2 * np.pi * t / period + phase)

        return white + wander + colored + ripple

    # ----- Peak placement -------------------------------------------------- #

    def _place_peaks(self, t, num_peaks, merge_probability,
                     apex_separation_sigma, amplitude_range,
                     sigma_range, tau_range):
        """
        Place peaks across the time axis, some with merged neighbours.

        Returns
        -------
        peak_params : list of dicts
            Each dict has keys: amp, mu, sigma, tau, cluster_id, is_merged.
        """
        t_start, t_end = t[0], t[-1]
        span = t_end - t_start
        # leave margins so peaks don't get clipped at edges
        margin = span * 0.05
        available_start = t_start + margin
        available_end = t_end - margin

        # distribute primary peak centres roughly evenly with jitter
        if num_peaks <= 1:
            centres = np.array([available_start + (available_end - available_start) / 2])
        else:
            centres = np.linspace(available_start, available_end, num_peaks)
            # add jitter (up to half the spacing)
            spacing = centres[1] - centres[0] if num_peaks > 1 else span
            jitter = self.rng.uniform(-spacing * 0.3, spacing * 0.3, num_peaks)
            centres = centres + jitter
            centres = np.clip(centres, available_start, available_end)
            centres.sort()

        peak_params = []
        cluster_id = 0

        i = 0
        while i < len(centres):
            mu = centres[i]
            sigma = self.rng.uniform(*sigma_range)
            tau = self.rng.uniform(*tau_range)
            amp = self.rng.uniform(*amplitude_range)

            peak_params.append({
                'amp': amp, 'mu': mu, 'sigma': sigma, 'tau': tau,
                'cluster_id': cluster_id, 'is_merged': False,
            })

            # decide whether to add merged neighbours
            if self.rng.random() < merge_probability:
                # 1-2 additional coeluting peaks
                n_extra = self.rng.integers(1, 3)
                primary_idx = len(peak_params) - 1
                primary_curve = self._emg(t, amp, mu, sigma, tau)
                cluster_apex_rts = [t[np.argmax(primary_curve)]]
                cluster_curves = [primary_curve]

                for _ in range(n_extra):
                    sep_sigmas = self.rng.uniform(*apex_separation_sigma)
                    direction = self.rng.choice([-1, 1])
                    mu2 = mu + direction * sep_sigmas * sigma
                    mu2 = np.clip(mu2, t_start + margin * 0.5, t_end - margin * 0.5)
                    sigma2 = sigma * self.rng.uniform(0.5, 1.5)
                    tau2 = tau * self.rng.uniform(0.4, 2.0)
                    amp2 = amp * self.rng.uniform(0.2, 1.2)

                    curve2 = self._emg(t, amp2, mu2, sigma2, tau2)
                    apex2_rt = t[np.argmax(curve2)]
                    avg_sigma = (sigma + sigma2) / 2

                    # Must be separated enough from ALL existing cluster members
                    too_close = any(abs(apex2_rt - art) < avg_sigma * 0.5
                                    for art in cluster_apex_rts)
                    if too_close:
                        continue

                    # Reject only if the shoulder is completely invisible:
                    # check that the composite's second derivative shows a
                    # sign change (inflection point) between the two apexes.
                    # This accepts both valleys and shoulders, rejecting only
                    # featureless humps where the minor peak is fully absorbed.
                    composite = sum(cluster_curves) + curve2
                    nearest_rt = min(cluster_apex_rts,
                                     key=lambda r: abs(r - apex2_rt))
                    idx_a = np.argmin(np.abs(t - nearest_rt))
                    idx_b = np.argmin(np.abs(t - apex2_rt))
                    lo_idx, hi_idx = min(idx_a, idx_b), max(idx_a, idx_b)
                    if hi_idx - lo_idx < 3:
                        continue
                    d2 = np.diff(composite[lo_idx:hi_idx + 1], n=2)
                    has_inflection = np.any(d2[:-1] * d2[1:] < 0)
                    if not has_inflection:
                        continue

                    cluster_apex_rts.append(apex2_rt)
                    cluster_curves.append(curve2)
                    peak_params.append({
                        'amp': amp2, 'mu': mu2, 'sigma': sigma2, 'tau': tau2,
                        'cluster_id': cluster_id, 'is_merged': True,
                    })

                if len(cluster_apex_rts) > 1:
                    peak_params[primary_idx]['is_merged'] = True

            cluster_id += 1
            i += 1

        return peak_params

    # ----- Main generation ------------------------------------------------- #

    def generate(self, *,
                 time_range=(0.0, 60.0),
                 num_points=10000,
                 num_peaks=15,
                 merge_probability=0.4,
                 apex_separation_sigma=(0.8, 3.5),
                 amplitude_range=(50.0, 500.0),
                 sigma_range=(0.05, 0.3),
                 tau_range=(0.01, 0.5),
                 baseline_type='random',
                 baseline_amplitude=None,
                 snr=None,
                 noise_mode='realistic'):
        """
        Generate a single synthetic chromatogram.

        Parameters
        ----------
        time_range : (float, float)
            Start and end retention time in minutes.
        num_points : int
            Number of data points (sampling density).
        num_peaks : int or (int, int)
            Exact peak count, or (min, max) to sample uniformly.
        merge_probability : float
            Probability [0, 1] that a primary peak gets coeluting neighbours.
        apex_separation_sigma : (float, float)
            Range of apex separations in multiples of sigma for merged peaks.
            Lower values = tighter merges (harder to resolve).
        amplitude_range : (float, float)
            Peak amplitude range.
        sigma_range : (float, float)
            Peak width (sigma) range in minutes.
        tau_range : (float, float)
            EMG tailing/fronting parameter range in minutes.
        baseline_type : str
            'flat', 'linear', 'quadratic', 'sine', or 'random'.
        baseline_amplitude : float or None
            Drift magnitude. None = auto (fraction of mean peak amplitude).
        snr : float or None
            Signal-to-noise ratio. None = random between 30 and 500.
        noise_mode : str
            'simple' (white + wander) or 'realistic' (adds colored noise
            and chromatographic bumps). Default 'realistic'.

        Returns
        -------
        dict with keys:
            x : 1-D array — retention times
            original_y : 1-D array — raw signal (peaks + baseline + noise)
            baseline_y : 1-D array — true baseline (for validation)
            corrected_y : 1-D array — signal with baseline subtracted
            pure_signal : 1-D array — peaks only, no noise or baseline
            true_peaks : list of dicts — ground truth peak info
            peak_params : list of dicts — raw EMG parameters per peak
        """
        t = np.linspace(time_range[0], time_range[1], num_points)

        # resolve num_peaks if given as range
        if isinstance(num_peaks, (tuple, list)):
            num_peaks = int(self.rng.integers(num_peaks[0], num_peaks[1] + 1))

        # place peaks
        peak_params = self._place_peaks(
            t, num_peaks, merge_probability,
            apex_separation_sigma, amplitude_range,
            sigma_range, tau_range,
        )

        # build pure signal (sum of all EMGs)
        pure_signal = np.zeros_like(t)
        true_peaks = []

        for pp in peak_params:
            curve = self._emg(t, pp['amp'], pp['mu'], pp['sigma'], pp['tau'])
            pure_signal += curve

            apex_idx = np.argmax(curve)
            apex_rt = t[apex_idx]

            # compute area by numerical integration of the individual curve
            dt = t[1] - t[0]
            area = np.trapz(curve, dx=dt)

            true_peaks.append({
                'rt': apex_rt,
                'apex_index': int(apex_idx),
                'area': float(area),
                'amplitude': float(curve[apex_idx]),
                'sigma': pp['sigma'],
                'tau': pp['tau'],
                'mu': pp['mu'],
                'cluster_id': pp['cluster_id'],
                'is_merged': pp['is_merged'],
            })

        # baseline
        if baseline_amplitude is None:
            mean_amp = np.mean([pp['amp'] for pp in peak_params]) if peak_params else 100
            baseline_amplitude = mean_amp * self.rng.uniform(0.01, 0.15)

        baseline = self._make_baseline(t, baseline_type, baseline_amplitude)

        # noise
        if snr is None:
            snr = self.rng.uniform(30, 500)

        sig_max = pure_signal.max() if pure_signal.max() > 0 else 1.0
        noise = self._make_noise(t, sig_max, snr, noise_mode=noise_mode)

        # compose
        original_y = pure_signal + baseline + noise
        corrected_y = original_y - baseline  # ideal correction

        return {
            'x': t,
            'original_y': original_y,
            'baseline_y': baseline,
            'corrected_y': corrected_y,
            'pure_signal': pure_signal,
            'true_peaks': true_peaks,
            'peak_params': peak_params,
            'snr': snr,
        }

    def generate_batch(self, n, **kwargs):
        """Generate *n* chromatograms with the same parameters.

        Returns a list of result dicts.
        """
        return [self.generate(**kwargs) for _ in range(n)]


# --------------------------------------------------------------------------- #
#  Convenience: quick visual sanity check
# --------------------------------------------------------------------------- #

def plot_synthetic(result, show_ground_truth=True, engine='plotly', ax=None):
    """
    Plot a synthetic chromatogram result dict.

    Parameters
    ----------
    result : dict from SyntheticChromatogramGenerator.generate()
    show_ground_truth : bool
        If True, overlay individual EMG curves and mark apexes.
    engine : str
        'plotly' (default) for interactive plots, 'matplotlib' for static.
    ax : matplotlib Axes or None
        Only used when engine='matplotlib'. If None, creates a new figure.

    Returns
    -------
    plotly Figure (engine='plotly') or matplotlib Axes (engine='matplotlib')
    """
    if engine == 'plotly':
        return _plot_synthetic_plotly(result, show_ground_truth)
    else:
        return _plot_synthetic_matplotlib(result, show_ground_truth, ax)


def _plot_synthetic_plotly(result, show_ground_truth=True):
    """Interactive plotly version of plot_synthetic."""
    import plotly.graph_objects as go
    from scipy.stats import exponnorm as _exponnorm

    t = result['x']
    title = (f'Synthetic Chromatogram — {len(result["true_peaks"])} peaks '
             f'(SNR={result["snr"]:.0f})')

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=t, y=result['original_y'],
        mode='lines', name='Raw signal',
        line=dict(color='#999999', width=0.5),
        opacity=0.5, hoverinfo='skip',
    ))
    fig.add_trace(go.Scattergl(
        x=t, y=result['corrected_y'],
        mode='lines', name='Baseline-corrected',
        line=dict(color='#1f77b4', width=1),
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scattergl(
        x=t, y=result['baseline_y'],
        mode='lines', name='True baseline',
        line=dict(color='orange', width=1, dash='dash'),
        opacity=0.7, hoverinfo='skip',
    ))

    if show_ground_truth:
        # plotly tab10-equivalent
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # overlay individual EMG curves, grouped by cluster
        clusters_seen = set()
        for pp in result['peak_params']:
            K = pp['tau'] / max(pp['sigma'], 1e-6)
            curve = pp['amp'] * _exponnorm.pdf(t, K, loc=pp['mu'],
                                                scale=max(pp['sigma'], 1e-6))
            color = colors[pp['cluster_id'] % len(colors)]
            show_legend = pp['cluster_id'] not in clusters_seen
            clusters_seen.add(pp['cluster_id'])
            fig.add_trace(go.Scattergl(
                x=t, y=curve,
                mode='lines',
                name=f'Cluster {pp["cluster_id"]}',
                legendgroup=f'cluster_{pp["cluster_id"]}',
                showlegend=show_legend,
                line=dict(color=color, width=0.8, dash='dash'),
                opacity=0.5, hoverinfo='skip',
            ))

        # apex markers
        merged = [p for p in result['true_peaks'] if p['is_merged']]
        singles = [p for p in result['true_peaks'] if not p['is_merged']]

        if singles:
            fig.add_trace(go.Scattergl(
                x=[p['rt'] for p in singles],
                y=[result['corrected_y'][p['apex_index']] for p in singles],
                mode='markers', name='Single peak',
                marker=dict(symbol='circle', size=7, color=[
                    colors[p['cluster_id'] % len(colors)] for p in singles
                ], line=dict(color='black', width=0.5)),
                text=[f"RT={p['rt']:.2f}<br>Area={p['area']:.1f}<br>"
                      f"Cluster {p['cluster_id']}" for p in singles],
                hoverinfo='text',
            ))
        if merged:
            fig.add_trace(go.Scattergl(
                x=[p['rt'] for p in merged],
                y=[result['corrected_y'][p['apex_index']] for p in merged],
                mode='markers', name='Merged peak',
                marker=dict(symbol='diamond', size=8, color=[
                    colors[p['cluster_id'] % len(colors)] for p in merged
                ], line=dict(color='black', width=0.5)),
                text=[f"RT={p['rt']:.2f}<br>Area={p['area']:.1f}<br>"
                      f"Cluster {p['cluster_id']}" for p in merged],
                hoverinfo='text',
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Retention Time (min)',
        yaxis_title='Signal',
        template='plotly_white',
        height=400,
        legend=dict(font=dict(size=10)),
        hovermode='closest',
    )

    return fig


def _plot_synthetic_matplotlib(result, show_ground_truth=True, ax=None):
    """Static matplotlib version of plot_synthetic."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    t = result['x']
    ax.plot(t, result['original_y'], color='#333333', linewidth=0.5,
            alpha=0.6, label='Raw signal')
    ax.plot(t, result['corrected_y'], color='#1f77b4', linewidth=0.8,
            label='Baseline-corrected')
    ax.plot(t, result['baseline_y'], color='orange', linewidth=0.8,
            linestyle='--', alpha=0.7, label='True baseline')

    if show_ground_truth:
        cmap = plt.cm.get_cmap('tab10')

        for peak in result['true_peaks']:
            color = cmap(peak['cluster_id'] % 10)
            marker = 'D' if peak['is_merged'] else 'o'
            ax.plot(peak['rt'], result['corrected_y'][peak['apex_index']],
                    marker=marker, color=color, markersize=6,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)

        from scipy.stats import exponnorm as _exponnorm
        for pp in result['peak_params']:
            K = pp['tau'] / max(pp['sigma'], 1e-6)
            curve = pp['amp'] * _exponnorm.pdf(t, K, loc=pp['mu'],
                                                scale=max(pp['sigma'], 1e-6))
            color = cmap(pp['cluster_id'] % 10)
            ax.plot(t, curve, color=color, linewidth=0.7, alpha=0.5,
                    linestyle='--')

    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Signal')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'Synthetic Chromatogram — {len(result["true_peaks"])} peaks '
                 f'(SNR={result["snr"]:.0f})')

    return ax


if __name__ == '__main__':
    gen = SyntheticChromatogramGenerator(seed=42)

    # Example 1: sparse, mostly clean peaks
    r1 = gen.generate(num_peaks=8, merge_probability=0.2, snr=200,
                      baseline_type='flat')
    fig1 = plot_synthetic(r1)
    fig1.update_layout(title='Sparse, clean (8 peaks, SNR=200, flat baseline)')
    fig1.show()

    # Example 2: dense with merges
    r2 = gen.generate(num_peaks=20, merge_probability=0.5, snr=80,
                      baseline_type='quadratic')
    fig2 = plot_synthetic(r2)
    fig2.update_layout(title='Dense, merged (20 peaks, SNR=80, quadratic baseline)')
    fig2.show()

    # Example 3: tight merges, low SNR
    r3 = gen.generate(num_peaks=12, merge_probability=0.6,
                      apex_separation_sigma=(0.5, 1.5), snr=40,
                      baseline_type='sine')
    fig3 = plot_synthetic(r3)
    fig3.update_layout(title='Tight merges, noisy (12 peaks, SNR=40, sine baseline)')
    fig3.show()
