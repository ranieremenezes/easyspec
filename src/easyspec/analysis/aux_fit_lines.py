"""
In this file we collect the secondary functions used by fit_lines(): validation of the inputs, line windows and priors, and the whole MCMC
machinery (likelihood, priors, maximum a posteriori fit, initialization of the walkers, automatic choice of the number of walkers, of the
burn-in and of the chain length from the autocorrelation time, and the summary of the posterior distributions).
"""

import re
import time
import warnings

import numpy as np
import emcee
from scipy import optimize

# Number of free parameters of each line model (see line_models.py):
MODEL_PARAMETERS = {"Gaussian": ["Location", "Amplitude", "fwhm_Gauss"],
                    "Lorentz": ["Location", "Amplitude", "fwhm_Lorentz"],
                    "Voigt": ["Location", "Amplitude", "fwhm_Lorentz", "fwhm_Gauss"],
                    "Skewedgaussian": ["Location", "Amplitude", "fwhm_Gauss", "skewness"],
                    "Skewedlorentzian": ["Location", "Amplitude", "fwhm_Lorentz", "skewness"]}

SPEED_OF_LIGHT = 299792.458   # km/s


# ======================================================================================================================================
# Input validation
# ======================================================================================================================================

def aux_validate_line_names(line_names):
    """Validate that line names are unique if provided."""
    if line_names is not None and len(line_names) != len(set(line_names)):
        raise ValueError("Duplicate line names detected. Each line must have a unique name.")


def aux_ensure_numpy_array(data, param_name):
    """
    Convert input to consistent numpy array format.

    Handles astropy Quantity, scalar values, lists, tuples, and ensures proper array shape.
    """
    if hasattr(data, 'value'):
        data = data.value
    if np.isscalar(data):
        data = np.array([data])
    elif isinstance(data, (list, tuple)):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Parameter '{param_name}' must be a scalar, list, tuple, or numpy array")
    if data.ndim == 0:
        data = data.reshape(1)
    elif data.ndim > 1:
        raise ValueError(f"Parameter '{param_name}' must be 1-dimensional")
    return data.astype(float)


def aux_validate_models(which_models, n_lines):
    """
    Validate and expand the model specification (a string for all lines, or one string per line). Upper/lower case, "_" and the suffixes
    "ian"/"" are ignored: e.g. "gauss", "Gaussian", "skewed_gaussian", "SkewedGaussian", "lorentzian" and "Lorentz" are all accepted.
    """
    if isinstance(which_models, str):
        which_models = [which_models]*n_lines
    elif isinstance(which_models, (list, tuple)):
        which_models = list(which_models)
        if len(which_models) != n_lines:
            raise ValueError(f"Number of models ({len(which_models)}) must match number of lines ({n_lines})")
    else:
        raise TypeError("'which_models' must be a string or list of strings")
    canonical = {"gauss": "Gaussian", "gaussian": "Gaussian", "lorentz": "Lorentz", "lorentzian": "Lorentz", "voigt": "Voigt",
                 "skewedgauss": "Skewedgaussian", "skewedgaussian": "Skewedgaussian", "skewedlorentz": "Skewedlorentzian",
                 "skewedlorentzian": "Skewedlorentzian"}
    for i, model in enumerate(which_models):
        key = re.sub(r"[^a-z]", "", str(model).lower())
        if key not in canonical:
            raise ValueError(f"Invalid model '{model}' at index {i}. Options are 'Gaussian', 'Lorentz', 'Voigt', 'skewed_gaussian' and "
                             "'skewed_lorentzian'.")
        which_models[i] = canonical[key]
    return which_models


def aux_generate_line_names(line_names, length):
    """Generate default line names if not provided."""
    if line_names is None:
        return [f"line_{i}" for i in range(length)]
    if isinstance(line_names, str):
        line_names = [line_names]
    if len(line_names) != length:
        raise ValueError(f"'line_names' has {len(line_names)} names, but there are {length} lines.")
    return list(line_names)


# ======================================================================================================================================
# Line windows and priors
# ======================================================================================================================================

def aux_estimate_fwhm(x, y, peak_index):
    """
    Rough FWHM of a line, from the half-maximum crossings of the continuum-subtracted profile y around peak_index (used for the initial
    values, the priors and the fit window). Returns np.nan if it cannot be measured.
    """
    peak = y[peak_index]
    if not np.isfinite(peak) or peak == 0:
        return np.nan
    half = np.abs(peak)/2
    left = peak_index
    while left > 0 and np.sign(y[left]) == np.sign(peak) and np.abs(y[left]) > half:
        left -= 1
    right = peak_index
    while right < len(y) - 1 and np.sign(y[right]) == np.sign(peak) and np.abs(y[right]) > half:
        right += 1
    if left == 0 or right == len(y) - 1:
        return np.nan
    return float(x[right] - x[left])


def aux_calculate_line_region(peak_positions, number, total_lines, half_width=100.0):
    """
    Wavelength window around a line: +-half_width around the peak (100 Angstroms by default), limited by the midpoints to the neighbouring
    lines.
    """
    line_region_min = peak_positions[number] - half_width
    line_region_max = peak_positions[number] + half_width
    if number > 0:
        line_region_min = max(line_region_min, (peak_positions[number-1] + peak_positions[number])/2)
    if number < total_lines - 1:
        line_region_max = min(line_region_max, (peak_positions[number] + peak_positions[number+1])/2)
    return line_region_min, line_region_max


def aux_group_blended_lines(peak_positions, min_separation):
    """Groups of consecutive lines (indices) closer than min_separation, which are fitted together."""
    groups = [[0]]
    for i in range(1, len(peak_positions)):
        if peak_positions[i] - peak_positions[i-1] < min_separation:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def aux_automatic_priors(model, location, height, fwhm, location_range, width_range):
    """
    Initial values, priors (uniform) and labels of one line component.

    Parameters
    ----------
    model: string
        "Gaussian", "Lorentz", "Voigt", "Skewedgaussian" or "Skewedlorentzian".
    location, height, fwhm: floats
        Initial peak position, (normalized) peak height above the continuum, and FWHM.
    location_range, width_range: lists
        Allowed ranges of the peak position and of the FWHM(s).
    """
    if height > 0:
        amplitude_range = [0.1*height, 10*height]
    elif height < 0:
        amplitude_range = [10*height, 0.1*height]
    else:
        raise ValueError("A line with zero height above the continuum cannot be fitted.")
    fwhm = float(np.clip(fwhm, *width_range))
    location = float(np.clip(location, *location_range))
    labels = MODEL_PARAMETERS[model]
    if model in ("Gaussian", "Lorentz"):
        initial = [location, height, fwhm]
        priors = [location_range, amplitude_range, width_range]
    elif model == "Voigt":
        initial = [location, height, fwhm/2, fwhm/2]
        priors = [location_range, amplitude_range, width_range, width_range]
    else:   # skewed models
        initial = [location, height, fwhm, 0.0]
        priors = [location_range, amplitude_range, width_range, [-50.0, 50.0]]
    return np.array(initial, dtype=float), np.array(priors, dtype=float), list(labels)


def aux_parse_custom_priors(priors, model, peak_position, peak_height):
    """
    User-provided priors of one line: a list with one [min, max] range per parameter of the model, in the order of MODEL_PARAMETERS[model]
    (the amplitude range is given in units of the normalization of the spectrum, as the automatic priors). The initial values are the
    detected peak position and height (moved inside the priors if needed) and the centres of the width ranges.
    """
    user_priors = np.array(priors, dtype=float)
    n_parameters = len(MODEL_PARAMETERS[model])
    if user_priors.shape != (n_parameters, 2):
        raise ValueError(f"The priors of a {model} line must be a list of {n_parameters} ranges [min, max], for "
                         f"{MODEL_PARAMETERS[model]}; got {priors}.")
    user_priors = np.sort(user_priors, axis=1)
    initial = np.array([peak_position, peak_height] + [np.mean(p) for p in user_priors[2:]], dtype=float)
    if model.startswith("Skewed"):
        initial[-1] = np.clip(0.0, *user_priors[-1])
    for i, (low, high) in enumerate(user_priors):
        if not low < initial[i] < high:
            initial[i] = 0.5*(low + high)
    return initial, user_priors


# ======================================================================================================================================
# Probability
# ======================================================================================================================================

def ln_prior(theta, priors):
    """Uniform priors: 0 inside the ranges, -inf outside."""
    if np.all((priors[:, 0] < theta) & (theta < priors[:, 1])):
        return 0.0
    return -np.inf


def ln_likelihood(theta, x, y, yerr, model):
    """Gaussian likelihood of the data given the model."""
    return -0.5*np.sum(((y - model(theta, x))/yerr)**2)


def ln_probability(theta, priors, x, y, yerr, model):
    """Log-posterior (module-level function, so that it can be pickled for parallel processing)."""
    lp = ln_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    value = lp + ln_likelihood(theta, x, y, yerr, model)
    return value if np.isfinite(value) else -np.inf


def aux_find_map(initial, priors, x, y, yerr, model):
    """
    Maximum a posteriori parameters (bounded optimization from the initial values, with two methods; the best result is kept). This gives
    the starting point of the walkers, which shortens the burn-in and avoids walkers trapped in secondary maxima.
    """
    start = np.clip(np.asarray(initial, dtype=float), priors[:, 0] + 1e-9*np.ptp(priors, axis=1), priors[:, 1] - 1e-9*np.ptp(priors, axis=1))
    scale = np.ptp(priors, axis=1)
    def negative(u):   # Parameters scaled to their prior ranges (better conditioned)
        return -ln_probability(priors[:, 0] + u*scale, priors, x, y, yerr, model)
    u0 = (start - priors[:, 0])/scale
    best_u, best_value = u0, negative(u0)
    bounds = [(1e-9, 1 - 1e-9)]*len(u0)
    for method in ("Powell", "L-BFGS-B"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = optimize.minimize(negative, best_u, method=method, bounds=bounds)
            except (ValueError, RuntimeError):
                continue
        if np.isfinite(result.fun) and result.fun < best_value:
            best_u, best_value = result.x, result.fun
    return priors[:, 0] + best_u*scale


class MCMCResult:

    """
    Result of aux_run_mcmc(), with the same attributes used by the plotting functions as an emcee sampler: flatchain (posterior samples after
    the burn-in, thinned), flatlnprobability, and get_chain() (the full chain, for the time series plots). The diagnostics (number of walkers
    and steps, burn-in, thinning, autocorrelation times, acceptance fraction, reduced chi2, error scale) are in 'info'.
    """

    def __init__(self, sampler, burn_in, thin, info, theta_map):
        self.sampler = sampler
        self.flatchain = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        self.flatlnprobability = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
        self.info = info
        self.theta_map = theta_map

    def get_chain(self, **kwargs):
        return self.sampler.get_chain(**kwargs)

    @property
    def acceptance_fraction(self):
        return self.sampler.acceptance_fraction


def aux_run_mcmc(initial, priors, x, y, yerr, model, walkers="auto", iterations="auto", burn_in="auto", max_steps=20000, check_interval=250,
                 rescale_errors=True, show_progress=True, label="", rng=None):

    """
    Sample the posterior distribution of the line parameters with emcee.

    Steps:
    1. Maximum a posteriori (MAP) fit, from the initial values.
    2. If rescale_errors and the reduced chi2 of the MAP fit is larger than 1 (the scatter of the data around the model is larger than the
       uncertainties, e.g. because of weak features or of an imperfect continuum), the uncertainties are scaled by sqrt(chi2_red), so that the
       errors of the parameters are not underestimated.
    3. The walkers start in a small ball around the MAP parameters.
    4. Chain length: with iterations="auto", the chain is run in blocks of check_interval steps until it is longer than 50 autocorrelation
       times and the autocorrelation time has changed by less than 5% between checks (the convergence criterion of the emcee documentation),
       up to max_steps. With an integer, exactly that number of steps is run after the burn-in.
    5. Burn-in: with "auto", 3 times the longest autocorrelation time is discarded; with an integer, that number of steps. The chain is also
       thinned by half of the shortest autocorrelation time.

    Parameters
    ----------
    walkers: "auto" or int
        Number of walkers. "auto": max(32, 8 x number of parameters).
    iterations: "auto" or int
        Number of steps after the burn-in (see above).
    burn_in: "auto" or int
        Number of initial steps discarded (see above).

    Returns
    -------
    result: MCMCResult
    """

    rng = np.random.default_rng() if rng is None else rng
    ndim = len(initial)
    yerr = np.asarray(yerr, dtype=float).copy()

    theta_map = aux_find_map(initial, priors, x, y, yerr, model)
    dof = max(1, len(x) - ndim)
    chi2_red = float(np.sum(((y - model(theta_map, x))/yerr)**2)/dof)
    error_scale = 1.0
    if rescale_errors and chi2_red > 1:
        error_scale = np.sqrt(chi2_red)
        yerr = yerr*error_scale

    n_walkers = max(32, 8*ndim) if walkers == "auto" else int(walkers)
    if n_walkers < 2*ndim:
        raise ValueError(f"The number of walkers ({n_walkers}) must be at least twice the number of parameters ({ndim}).")
    n_walkers += n_walkers % 2

    # Small ball around the MAP (1e-3 of the prior ranges), kept inside the priors:
    width = 1e-3*np.ptp(priors, axis=1)
    p0 = theta_map + width*rng.standard_normal((n_walkers, ndim))
    p0 = np.clip(p0, priors[:, 0] + 1e-6*np.ptp(priors, axis=1), priors[:, 1] - 1e-6*np.ptp(priors, axis=1))

    sampler = emcee.EnsembleSampler(n_walkers, ndim, ln_probability, args=(priors, x, y, yerr, model))
    start = time.time()
    converged = None
    if iterations == "auto":
        state, old_tau = p0, np.inf
        while sampler.iteration < max_steps:
            state = sampler.run_mcmc(state, check_interval, progress=show_progress, skip_initial_state_check=True)
            tau = sampler.get_autocorr_time(tol=0)
            converged = bool(np.all(tau*50 < sampler.iteration) and np.all(np.abs(old_tau - tau)/tau < 0.05))
            if converged and sampler.iteration >= 1000:
                break
            old_tau = tau
        n_burn = int(np.ceil(3*np.max(tau))) if burn_in == "auto" else int(burn_in)
    else:
        n_burn_fixed = 0 if burn_in == "auto" else int(burn_in)
        sampler.run_mcmc(p0, n_burn_fixed + int(iterations), progress=show_progress, skip_initial_state_check=True)
        tau = sampler.get_autocorr_time(tol=0)
        converged = bool(np.all(tau*50 < sampler.iteration))
        n_burn = n_burn_fixed if burn_in != "auto" else int(min(np.ceil(3*np.max(tau)), sampler.iteration//2))
    n_burn = int(min(n_burn, sampler.iteration - 10))
    thin = int(max(1, 0.5*np.min(tau)))
    acceptance = float(np.mean(sampler.acceptance_fraction))
    info = {"walkers": n_walkers, "steps": int(sampler.iteration), "burn_in": n_burn, "thin": thin, "autocorrelation_time": tau,
            "converged": converged, "acceptance_fraction": acceptance, "chi2_red": chi2_red, "error_scale": error_scale,
            "time": time.time() - start}
    result = MCMCResult(sampler, n_burn, thin, info, theta_map)
    info["n_samples"] = len(result.flatchain)

    print(f"{label}: {n_walkers} walkers x {sampler.iteration} steps (burn-in {n_burn}, thin {thin}, autocorrelation time "
          f"{np.max(tau):.0f}), {info['n_samples']} samples, acceptance {acceptance:.2f}, reduced chi2 of the best fit {chi2_red:.2f}"
          + (f" (uncertainties scaled by {error_scale:.2f})" if error_scale > 1 else "") + f" [{info['time']:.1f} s]")
    if not converged:
        warnings.warn(f"{label}: the MCMC chain may not have converged (it is shorter than 50 autocorrelation times). Increase "
                      "'MCMC_iterations' (or use 'auto') and check the time series of the parameters (plot_MCMC=True).", stacklevel=3)
    if not 0.15 < acceptance < 0.75:
        warnings.warn(f"{label}: unusual acceptance fraction ({acceptance:.2f}). Check the priors and the plots of the MCMC.", stacklevel=3)
    return result


def aux_summarize_posterior(samples, labels, rest_wavelengths, normalization, quantiles=(0.16, 0.5, 0.84), observed_to_vacuum=None,
                            rest_to_vacuum=None):

    """
    Median and asymmetric 1-sigma errors (16% and 84% quantiles) of each parameter, plus the redshift of each line.

    The redshift is computed for each posterior sample as z = lambda_obs/lambda_rest - 1, with both wavelengths in vacuum (the observed
    wavelengths are in air; the rest wavelengths are converted if needed, see fit_lines()), which avoids the bias of mixing air and vacuum
    wavelengths (~80 km/s for UV lines observed in the optical).

    Parameters
    ----------
    samples: array (n_samples, n_parameters)
    labels: list of lists
        Parameter names of each line of the group.
    rest_wavelengths: list
        Rest wavelength of each line (np.nan for none).
    normalization: float
        The amplitudes are multiplied by this factor.
    observed_to_vacuum, rest_to_vacuum: callables
        Conversions of the observed and rest wavelengths to vacuum (identity if None).

    Returns
    -------
    values, errors, names: lists, one sublist per line: [z, Location, Amplitude, ...], [[down, up], ...], ["redshift", "Location", ...].
    """

    to_vacuum_obs = observed_to_vacuum if observed_to_vacuum is not None else (lambda w: w)
    to_vacuum_rest = rest_to_vacuum if rest_to_vacuum is not None else (lambda w: w)
    values, errors, names = [], [], []
    column = 0
    for line_labels, rest in zip(labels, rest_wavelengths):
        line_values, line_errors, line_names = [], [], []
        for label in line_labels:
            chain = samples[:, column]
            if label == "Location" and rest is not None and np.isfinite(rest):
                z = to_vacuum_obs(chain)/to_vacuum_rest(rest) - 1
                q16, q50, q84 = np.quantile(z, quantiles)
                line_values.append(float(q50))
                line_errors.append([float(q50 - q16), float(q84 - q50)])
                line_names.append("redshift")
            factor = normalization if label == "Amplitude" else 1.0
            q16, q50, q84 = np.quantile(chain*factor, quantiles)
            if label == "Amplitude" and normalization < 0:
                q16, q84 = q84, q16
            line_values.append(float(q50))
            line_errors.append([float(abs(q50 - q16)), float(abs(q84 - q50))])
            line_names.append(label)
            column += 1
        values.append(line_values)
        errors.append(line_errors)
        names.append(line_names)
    return values, errors, names
