import numpy as np

def interpolate_arrays(x_data, y_data, target_length=None):
    """Interpolate x and y arrays to a common length.
    
    Args:
        x_data (np.ndarray): X values array
        y_data (np.ndarray): Y values array
        target_length (int, optional): Target length for interpolation. If None,
                                      uses the minimum of 10000 or the shorter array length.
    
    Returns:
        tuple: (interpolated_x, interpolated_y) arrays of the same length
    """
    # Ensure inputs are numpy arrays
    x_data = np.asarray(x_data).flatten()
    y_data = np.asarray(y_data).flatten()
    
    # Determine target length if not specified
    if target_length is None:
        target_length = min(10000, len(x_data), len(y_data))
    
    # Check if interpolation is needed
    if len(x_data) == len(y_data) == target_length:
        return x_data, y_data
    
    # Interpolate x values if needed
    if len(x_data) != target_length:
        interpolated_x = np.interp(
            np.linspace(0, 1, target_length),
            np.linspace(0, 1, len(x_data)),
            x_data
        )
    else:
        interpolated_x = x_data
    
    # Interpolate y values if needed
    if len(y_data) != target_length:
        interpolated_y = np.interp(
            np.linspace(0, 1, target_length),
            np.linspace(0, 1, len(y_data)),
            y_data
        )
    else:
        interpolated_y = y_data
    
    return interpolated_x, interpolated_y