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
    elif isinstance(data, list):
        return [serialize_numpy(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data


def convert_params_for_processor(params: dict) -> dict:
    """Convert API parameters to processor format."""
    # Handle lambda parameter name conversion
    if 'baseline' in params and 'lambda_' in params['baseline']:
        params['baseline']['lambda'] = params['baseline'].pop('lambda_')
    
    # Convert peak params if min_prominence/min_width are provided
    if 'peaks' in params:
        if params['peaks'].get('min_prominence') is not None:
            params['peaks']['peak_prominence'] = params['peaks']['min_prominence']
        if params['peaks'].get('min_width') is not None:
            params['peaks']['peak_width'] = params['peaks']['min_width']
    
    return params
