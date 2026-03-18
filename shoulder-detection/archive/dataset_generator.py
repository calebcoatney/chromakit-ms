import numpy as np
from scipy.stats import exponnorm

class ApexHeatmapGenerator:
    def __init__(self, window_length=256, time_span=100, heatmap_sigma=2.0):
        """
        generates additive gc peaks and apex heatmaps for keypoint detection.
        
        args:
            window_length: discrete points in the output array
            time_span: arbitrary time units for the window
            heatmap_sigma: the width (in pixels) of the target gaussian blob 
                           placed at the true apex.
        """
        self.window_length = window_length
        self.time_span = time_span
        self.t = np.linspace(0, time_span, window_length)
        self.heatmap_sigma = heatmap_sigma

    def _emg(self, amp, mu, sigma, tau):
        """generates a stable exponentially modified gaussian."""
        K = tau / sigma if sigma > 0 else 1e-6
        peak = amp * exponnorm.pdf(self.t, K, loc=mu, scale=sigma)
        return np.nan_to_num(peak)

    def _create_heatmap_target(self, true_indices):
        """places a gaussian blob at each true apex index."""
        target = np.zeros(self.window_length)
        x_grid = np.arange(self.window_length)
        
        for idx in true_indices:
            # add a gaussian centered exactly at the index
            blob = np.exp(-0.5 * ((x_grid - idx) / self.heatmap_sigma)**2)
            target = np.maximum(target, blob) # max() prevents overlapping blobs from exceeding 1.0
            
        return target

    def generate_batch(self, num_samples=1000):
        """
        returns:
            X: (num_samples, window_length, 1) - noisy additive signal [0, 1]
            Y: (num_samples, window_length, 1) - apex heatmap target [0, 1]
        """
        X = np.zeros((num_samples, self.window_length, 1))
        Y = np.zeros((num_samples, self.window_length, 1))
        
        for i in range(num_samples):
            true_apexes = []
            
            # --- peak 1 (always present) ---
            mu1 = np.random.uniform(30, 70)
            sig1 = np.random.uniform(2, 6)
            tau1 = np.random.uniform(0.1, 4.0)
            amp1 = np.random.uniform(10, 100)
            
            curve1 = self._emg(amp1, mu1, sig1, tau1)
            
            # find the *exact* pixel index of the pure curve's maximum
            # (mu is not the apex due to the tailing factor tau)
            idx1 = np.argmax(curve1)
            true_apexes.append(idx1)
            
            # --- peak 2 (50% chance of co-elution) ---
            is_multi = np.random.choice([True, False])
            if is_multi:
                # offset: 1.0 to 3.0 sigma away (covers tight splits to hidden shoulders)
                offset = np.random.uniform(1.0, 3.0) * sig1 * np.random.choice([-1, 1])
                mu2 = mu1 + offset
                sig2 = sig1 * np.random.uniform(0.6, 1.2)
                tau2 = tau1 * np.random.uniform(0.5, 1.5)
                amp2 = amp1 * np.random.uniform(0.1, 1.5)
                
                curve2 = self._emg(amp2, mu2, sig2, tau2)
                
                idx2 = np.argmax(curve2)
                # only append if it's within the window
                if 0 <= idx2 < self.window_length:
                    true_apexes.append(idx2)
            else:
                curve2 = np.zeros_like(curve1)

            # strictly additive physics
            pure_signal = curve1 + curve2
            
            # --- realistic noise stack ---
            snr_target = np.random.uniform(20, 1000)
            noise_level = np.max(pure_signal) / snr_target
            
            white_noise = np.random.normal(0, noise_level, self.window_length)
            wander = np.cumsum(np.random.normal(0, noise_level * 0.05, self.window_length))
            wander -= wander[0]
            
            noisy_signal = pure_signal + white_noise + wander
            
            # min-max normalization for the neural net
            sig_min = np.min(noisy_signal)
            sig_max = np.max(noisy_signal)
            if sig_max > sig_min:
                X[i, :, 0] = (noisy_signal - sig_min) / (sig_max - sig_min)
                
            # build the heatmap target
            Y[i, :, 0] = self._create_heatmap_target(true_apexes)
            
        return X, Y