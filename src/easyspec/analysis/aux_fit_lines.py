"""
In this file we collect all secondary functions used to support the main function fit_lines().

"""


import numpy as np


def aux_validate_line_names(line_names):
    """Validate that line names are unique if provided."""
    if line_names is not None and len(line_names) != len(set(line_names)):
        raise ValueError("Duplicate line names detected. Each line must have a unique name.")

def aux_ensure_numpy_array(data, param_name):
    """
    Convert input to consistent numpy array format.
    
    Handles astropy Quantity, scalar values, lists, and ensures proper array shape.
    """
    # Extract value if it's an astropy Quantity
    if hasattr(data, 'value'):
        data = data.value
    
    # Handle scalar inputs by converting to 1D array
    if np.isscalar(data):
        data = np.array([data])
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Parameter '{param_name}' must be a scalar, list, or numpy array")
    
    # Ensure 1D array
    if data.ndim == 0:
        data = data.reshape(1)
    elif data.ndim > 1:
        raise ValueError(f"Parameter '{param_name}' must be 1-dimensional")
    
    return data

def aux_validate_models(which_models, n_lines):
    """Validate and expand model specifications."""
    
    if isinstance(which_models, str):
        # Apply same model to all lines
        which_models = [which_models] * n_lines
    elif isinstance(which_models, list):
        # Validate list length matches number of lines
        if len(which_models) != n_lines:
            raise ValueError(
                f"Number of models ({len(which_models)}) must match number of lines ({n_lines})"
            )
    else:
        raise TypeError("'which_models' must be a string or list of strings")
    
    # Validate each model type
    valid_models = {"Gaussian","gaussian","Gauss","gauss","Lorentzian", "lorentzian", "Lorentz", "lorentz", "Voigt", "voigt",
                    "skewed_gaussian","Skewed_Gaussian","Skewed_gaussian","skewed_Gaussian","Skewedgaussian","SkewedGaussian","skewedgaussian",
                    "skewedGaussian", "skewed_lorentzian","Skewed_Lorentzian","Skewed_lorentzian","skewed_Lorentzian","Skewedlorentzian",
                    "SkewedLorentzian", "skewedlorentzian", "skewedLorentzian"}
    for i, model in enumerate(which_models):
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}' at index {i}. Must be one of: {valid_models}")
        elif model=="gaussian" or model=="Gauss" or model=="gauss":
            which_models[i] = "Gaussian"
        elif model=="Lorentzian" or model=="lorentzian" or model=="lorentz":
            which_models[i] = "Lorentz"
        elif model=="voigt":
            which_models[i] = "Voigt"
        elif model=="skewed_gaussian" or model=="Skewed_gaussian" or model=="skewed_Gaussian" or model=="Skewed_Gaussian" or model=="SkewedGaussian" or model=="skewedgaussian" or model=="skewedGaussian":
            which_models[i] = "Skewedgaussian"
        elif model=="skewed_lorentzian" or model=="Skewed_lorentzian" or model=="skewed_Lorentzian" or model=="Skewed_Lorentzian" or model=="SkewedLorentzian" or model=="skewedlorentzian" or model=="skewedLorentzian":
            which_models[i] = "Skewedlorentzian"
    return which_models

def aux_generate_line_names(line_names, length):
    """Generate default line names if not provided."""
    if line_names is None:
        return [f"line_{i}" for i in range(length)]
    return line_names

def aux_calculate_line_region(peak_positions, number, total_lines):
    """Calculate minimum and maximum wavelength bounds for a line region."""
    # Default 100 Angstrom window around the peak
    line_region_min = peak_positions[number] - 100
    line_region_max = peak_positions[number] + 100
    
    # Adjust lower bound based on previous peak
    if number > 0:
        midpoint_prev = (peak_positions[number-1] + peak_positions[number]) / 2
        line_region_min = max(line_region_min, midpoint_prev)
    
    # Adjust upper bound based on next peak
    if number < total_lines - 1:
        midpoint_next = (peak_positions[number] + peak_positions[number+1]) / 2
        line_region_max = min(line_region_max, midpoint_next)
    
    return line_region_min, line_region_max

def aux_parse_custom_priors(priors, peak_position, peak_height):
    """Parse user-provided priors for line fitting."""
    # Handle different prior formats
    user_priors = np.asarray(priors, dtype="object")
    
    # Create initial values from priors
    if len(user_priors) == 3:
        initial = np.array([peak_position, peak_height, np.mean(user_priors[2])])
    else:  # Voigt profile case
        initial = np.array([peak_position, peak_height, np.mean(user_priors[2]), np.mean(user_priors[3])])
    
    line_region_min, line_region_max = user_priors[0]
    
    # Return only the values we can determine
    return initial, user_priors, line_region_min, line_region_max

def aux_resetting_wavelength_windows(number,rest_frame_line_wavelengths,priors,blended_line,wavelength_peak_positions,line_region_min_cache,line_region_min, line_region_max):
    """Reseting wavelength windows for blended lines"""
    if number < (len(rest_frame_line_wavelengths)-1) and priors[number] is None:
        counter = 0
        for i in blended_line[number+1:]:
            if i is False:
                break
            else:
                counter = counter + 1
                
        line_region_max = wavelength_peak_positions[number+counter] + 100
        if (number+counter) < (len(rest_frame_line_wavelengths)-1):
            mean_point = (wavelength_peak_positions[number+counter] + wavelength_peak_positions[number+counter+1])/2
            if mean_point < line_region_max:
                line_region_max = mean_point

    if blended_line[number] is True and priors[number] is None:
        counter = 1
        for i in np.flip(blended_line[:number]):
            if bool(i) is True:
                counter = counter + 1
            else:
                break
        line_region_min = line_region_min_cache[-counter-1]
    return line_region_min, line_region_max

