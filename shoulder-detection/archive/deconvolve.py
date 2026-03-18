import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import exponnorm

def single_emg(t, amp, mu, sigma, tau):
    """Single Exponentially Modified Gaussian for the optimizer to use."""
    # Add a tiny epsilon to sigma to prevent division by zero during fitting
    sigma = max(sigma, 1e-6) 
    K = tau / sigma if sigma > 0 else 1e-6
    return amp * exponnorm.pdf(t, K, loc=mu, scale=sigma)

def multi_emg(t, *params):
    """
    Additive function for N peaks. 
    params is a flat list: [amp1, mu1, sig1, tau1, amp2, mu2, sig2, tau2, ...]
    """
    n_peaks = len(params) // 4
    y_total = np.zeros_like(t)
    for i in range(n_peaks):
        amp, mu, sigma, tau = params[i*4 : (i+1)*4]
        y_total += single_emg(t, amp, mu, sigma, tau)
    return y_total

def deconvolve_peaks(time_array, signal_array, inferred_indices):
    """
    Takes the CNN's output indices and solves for the true underlying areas.
    """
    n_peaks = len(inferred_indices)
    if n_peaks == 0:
        return []

    # 1. Build Initial Guesses (p0)
    # The optimizer needs a starting point. We use the CNN's indices for mu,
    # and rough estimates for amplitude, sigma, and tau.
    p0 = []
    bounds_lower = []
    bounds_upper = []
    
    for idx in inferred_indices:
        # Initial guesses
        guess_mu = time_array[idx]
        guess_amp = signal_array[idx] * 5.0 # Area is roughly height * width
        guess_sigma = 3.0
        guess_tau = 1.0
        
        p0.extend([guess_amp, guess_mu, guess_sigma, guess_tau])
        
        # Bounding the optimizer prevents it from fitting impossible physics
        # e.g., a peak centered completely outside the retention window
        bounds_lower.extend([0, time_array[0], 0.1, 0.01])
        bounds_upper.extend([np.inf, time_array[-1], 20.0, 20.0])

    # 2. Run the Non-Linear Least Squares Optimizer
    try:
        popt, pcov = curve_fit(
            multi_emg, 
            time_array, 
            signal_array, 
            p0=p0, 
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000 # Allow extra iterations for complex doublet merges
        )
    except RuntimeError:
        print("Optimizer failed to converge.")
        return []

    # 3. Extract the Analytical Areas
    # For an EMG defined via scipy.stats.exponnorm, the 'amp' parameter 
    # we fitted is mathematically equivalent to the total area of the curve.
    areas = []
    for i in range(n_peaks):
        area = popt[i*4] 
        mu = popt[i*4 + 1]
        areas.append({'retention_time': mu, 'area': area})

    return areas