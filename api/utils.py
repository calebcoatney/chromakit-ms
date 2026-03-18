"""Utility functions for the API."""
import json
import numpy as np
from typing import Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def serialize_numpy(data: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {k: serialize_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [serialize_numpy(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data


def convert_params_for_processor(params: dict) -> dict:
    """Convert API parameters to the format expected by ChromatogramProcessor.

    Handles:
    - Pydantic alias 'lambda_' → processor key 'lambda'
    - Shoulder params extraction from peaks → separate 'shoulders' dict
    - Negative peak params passthrough
    - Deconvolution params passthrough
    - Break points in baseline
    - Fastchrom params in baseline
    """
    # ── Smoothing lambda ──
    if 'smoothing' in params:
        sm = params['smoothing']
        if 'lambda_' in sm:
            sm['lambda'] = sm.pop('lambda_')
        # Pydantic alias may also serialize as 'lambda' already — no-op then

    # ── Baseline lambda + break points ──
    if 'baseline' in params:
        bl = params['baseline']
        if 'lambda_' in bl:
            bl['lambda'] = bl.pop('lambda_')

        # Ensure break_points is a list (may be None from Pydantic)
        if bl.get('break_points') is None:
            bl['break_points'] = []

        # Ensure fastchrom is a dict
        if bl.get('fastchrom') is None:
            bl['fastchrom'] = {'half_window': None, 'smooth_half_window': None}

    # ── Peaks → separate out shoulder params for processor ──
    if 'peaks' in params:
        pk = params['peaks']
        if pk.get('min_prominence') is not None:
            pk['peak_prominence'] = pk['min_prominence']
        if pk.get('min_width') is not None:
            pk['peak_width'] = pk['min_width']

        # Range filters: ensure list
        if pk.get('range_filters') is None:
            pk['range_filters'] = []

    # ── Shoulders ──
    if 'shoulders' in params:
        sh = params['shoulders']
        # Processor expects: { enabled, window_length, polyorder, sensitivity, apex_distance }
        # That matches our model already, so just pass through
    else:
        params['shoulders'] = {'enabled': False}

    # ── Negative peaks ──
    if 'negative_peaks' not in params:
        params['negative_peaks'] = {'enabled': False}

    # ── Deconvolution ──
    if 'deconvolution' not in params:
        params['deconvolution'] = {'splitting_method': 'geometric', 'windows': []}

    return params
