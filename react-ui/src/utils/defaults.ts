/**
 * Default processing parameters — mirrors ParametersFrame.current_params
 * from the desktop app (ui/frames/parameters.py).
 */
import type { ProcessingParams } from '../types';

export const DEFAULT_PROCESSING_PARAMS: ProcessingParams = {
  smoothing: {
    enabled: false,
    method: 'whittaker',
    median_enabled: false,
    median_kernel: 5,
    lambda: 1e-1,
    diff_order: 1,
    savgol_window: 3,
    savgol_polyorder: 1,
  },
  baseline: {
    show_corrected: false,
    method: 'arpls',
    lambda: 1e4,
    asymmetry: 0.01,
    align_tic: false,
    break_points: [],
    fastchrom: {
      half_window: null,
      smooth_half_window: null,
    },
  },
  peaks: {
    enabled: false,
    mode: 'classical',
    min_prominence: 1e5,
    min_height: 0.0,
    min_width: 0.0,
    range_filters: [],
  },
  deconvolution: {
    splitting_method: 'geometric',
    windows: [],
    heatmap_threshold: 0.36,
    pre_fit_signal_threshold: 0.001,
    min_area_frac: 0.15,
    valley_threshold_frac: 0.48,
    mu_bound_factor: 0.68,
    fat_threshold_frac: 0.44,
    dedup_sigma_factor: 1.32,
    dedup_rt_tolerance: 0.005,
  },
  negative_peaks: {
    enabled: false,
    min_prominence: 1e5,
  },
  shoulders: {
    enabled: false,
    window_length: 41,
    polyorder: 3,
    sensitivity: 8,
    apex_distance: 10,
  },
  integration: {
    peak_groups: [],
  },
};
